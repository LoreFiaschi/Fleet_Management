"""
Rainflow cells, version 2:  TWO interchangeable MILP encodings of the same model.

``rainflow.py`` encodes every logical ("if this action is taken, the state
follows that recursion") constraint with Gurobi *indicator* constraints.  That is
readable and numerically safe, but it has one property that hurts: an indicator
constraint contributes **nothing to the LP relaxation** until its binary is
fixed.  The root relaxation therefore sees a model in which the states are
essentially free, the bound is weak, and the branch-and-bound tree has to do all
the work.

This module keeps that encoding (``formulation='indicator'``, the default, and
mathematically identical to ``rainflow.py``) and adds a second one
(``formulation='bigm'``) that writes the same logic as plain linear rows.  Select
it per solve::

    rainflow_v2.solve(cfg, formulation="bigm")          # or via cfg options:
    #   formulation: bigm
    #   bigM: 1.1

Why the big-M version is expected to be TIGHTER as well as sparser
------------------------------------------------------------------
1. **The big-M rows are in the LP.**  Every recursion contributes two linear
   halfspaces to the root relaxation instead of nothing.
2. **The M's are tight.**  Each state already has a valid, problem-derived upper
   bound from ``_tighten_bounds`` (``mu <= tau`` or ``eps*tau``, ``v``, ``R``,
   ``K``).  Those UBs -- not a generic 1e6 -- are the big-M values; ``bigM``
   (default 1.1, the range the states live in) is only a fallback for a state
   with no finite bound.
3. **Several rows need no M at all.**  Because a repair is a convex combination
   of the pre-repair state and its latch (weights ``1-rho`` / ``rho``, and
   ``(1-rho)^2`` / ``1-(1-rho)^2``), and because the latch never exceeds the
   state, every branch of the recursion satisfies

       state_k <= state_{k-1} + increment_k        ("states never overshoot")
       latch_k <= state_k                          ("the latch never overshoots")

   unconditionally.  Those go in as bare inequalities, valid for all three
   branches simultaneously; they are exactly the cuts the indicator model cannot
   express.
4. **Aggregated segment selection.**  The piecewise-tangent reliability encoding
   picks its segment with ``mu >= sum_s lo_s z_s`` / ``mu <= sum_s hi_s z_s``
   (two rows, both in the LP) instead of ``2K`` indicator constraints (none in
   the LP).

Variables that are substituted out (sparsity)
---------------------------------------------
``nb``  ``nb = 1 - m - r`` is implied by eq. 3, so the big-M formulation never
        creates it: **F*L*T fewer binaries**.  Readers go through
        ``ctx.nb_of(i, l, k)`` / ``ctx.act_of(i, l, k)``, and the three gating
        rows collapse into the single stronger row ``m + r <= x_0``.
``z``   the removed-damage variable is kept (the objective and the result dict
        read it) but its three indicator constraints collapse to the single row
        ``z_k >= mu_{k-1} - mu_k``: with ``C_R > 0`` the minimisation drives
        ``z_k`` to ``max(0, mu_{k-1} - mu_k)``, which *is* eq. 6, because the
        difference is <= 0 on a no-intervention step and >= 0 on a repair.  When
        ``C_R <= 0`` (or ``z_exact=True``) two upper rows are added to pin it.
``x``   NOT substituted.  ``x[i,0,k] = 1 - sum_j x[i,j,k]`` looks tempting but is
        wrong: it would force every idle vehicle into the depot and make it pay
        ``C_M``.  The assignment stays ``sum_j x[i,j,k] <= 1``.

Everything else -- the reliability-bound registry, the state semantics, the
result dict, the config schema -- is unchanged, so ``test.py`` /
``run_studies.py`` compare the two encodings on exactly the same instances.

Where things live
-----------------
    solve                          entry point
    prepare                        auxiliary variables, per-cell arrays, bounds
    _add_rainflow_cell             gating + state + reliability, per formulation
      _add_rainflow_state_ind        indicator recursions (verbatim from v1)
      _add_rainflow_state_bigm       big-M recursions (this module's point)
        _state_bigm                    one state's four/six rows
        _latch_bigm                    one ARD1 latch's four rows
        _z_bigm                        eq. 6 without indicators
    RELIABILITY_BOUNDS             bound -> impl -> builder (unchanged registry)

Author: Johann Tschan  (v2: big-M / substituted formulation)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from fleet_management.degradation_model.base import (
    FleetModel as _RFModel,          # shared context (alias keeps the local name)
    add_base_constraints,
    assembly_of,
    encoding_of,
    add_maintenance_gating,
    build_fleet,
    build_objective,
    cell_max as _cell_max,
    extract_solution as _extract_solution,
    make_accessor,
    pick as _pick,
    register_cell_builder,
    resolve_costs as _resolve_costs,
    resolve_run_options,
)


# Reliability bounds and the extra data each needs (mirrors config.py).
_NEEDS_SUPPORT = ("hoeffding", "bernstein")
_NEEDS_CGF = ("chernoff",)
_TRACK_V = ("cantelli", "bernstein")                    # bounds that use the variance state
# extra per-cell accumulator a bound needs beyond (mu, v): 'R' Hoeffding, 'K' Chernoff
_BOUND_DESCRIPTORS = {"hoeffding": "R", "chernoff": "K"}
_REPAIR_MODELS = ("ard1", "ardinf")

_ARD1_UNSUPPORTED = ("chernoff",)

# A 2x2 grid of (encoding, assembly) flattened into one string; see
# base.FORMULATIONS / encoding_of / assembly_of.  'sparse' is the 'indicator'
# program and 'bigm_sparse' the 'bigm' program, both assembled through the
# matrix API (rainflow_sparse).  prepare() is shared by all four.
FORMULATIONS = ("indicator", "bigm", "sparse", "bigm_sparse")


# ===========================================================================
# Entry point
# ===========================================================================
def solve(cfg, *, allow_replacement=None, depot_capacity=None,
          verbose=None, mip_gap=None, time_limit=None, fast=None,
          gurobi_params=None,
          reliability_impl=None, pwl_points=None, tangent_ref=None,
          formulation=None, bigM=None, z_exact=None) -> dict:
    """Solve a fleet from a normalized ``FleetConfig``.

    Identical to ``rainflow.solve`` plus three knobs:

    formulation : {'indicator', 'bigm'}
        How the logical constraints are encoded.  ``'indicator'`` (default)
        reproduces ``rainflow.py`` exactly; ``'bigm'`` uses linear big-M rows,
        substitutes ``nb`` out and adds the unconditional monotonicity cuts.
    bigM : float
        Fallback big-M for a state without a finite upper bound (default 1.1 --
        the range the states actually live in).  States that *do* have a bound
        (all of them, after ``_tighten_bounds``) use that bound instead.
    z_exact : bool or None
        Pin ``z`` with its two upper rows.  ``None`` (default) decides
        automatically: they are needed only when ``C_R <= 0``.
    """
    opts = resolve_run_options(
        cfg,
        allow_replacement=allow_replacement, depot_capacity=depot_capacity,
        verbose=verbose, mip_gap=mip_gap, time_limit=time_limit, fast=fast,
        gurobi_params=gurobi_params, reliability_impl=reliability_impl,
        pwl_points=pwl_points, tangent_ref=tangent_ref,
        formulation=formulation, bigM=bigM, z_exact=z_exact,
    )
    # ``base._load_builders`` deterministically installs THIS builder for the
    # "rainflow" name, so no registry juggling is needed here.
    ctx = build_fleet(cfg, opts, model_name="fleet_management_rainflow_v2")
    ctx.model.optimize()
    out = _extract_solution(ctx, cfg, ctx.model)
    out["formulation"] = str(ctx.formulation)
    return out


# ===========================================================================
# Preparation hook: rainflow-specific variables, per-cell arrays, solver flags
# ===========================================================================
def prepare(ctx: _RFModel, cfg, cells, opts: dict) -> None:
    """Create everything the rainflow cells need on top of the shared skeleton.

    Called once by ``base.build_fleet`` with the list of rainflow ``cells``.
    Fills the model-specific slots of the shared context: per-cell selector /
    parameter arrays, the auxiliary state variables (variance, ARD1 latch, the
    Hoeffding ``R`` and Chernoff ``K`` accumulators), the resolved reliability
    implementation per cell, and the tightened variable bounds.  The tightening
    runs LAST on purpose: the big-M values are read straight off those bounds.
    """
    F, L, T = ctx.F, ctx.L, ctx.T
    md = ctx.model
    cells = list(cells)
    if not cells:
        return

    # ---- encoding ------------------------------------------------------
    ctx.formulation = str(opts.get("formulation", ctx.formulation)).lower()
    if ctx.formulation not in FORMULATIONS:
        raise ValueError(f"unknown formulation {ctx.formulation!r}; "
                         f"pick from {FORMULATIONS}.")
    ctx.bigM = float(opts.get("bigM", ctx.bigM))
    if not (ctx.bigM > 0.0):
        raise ValueError(f"bigM must be positive (got {ctx.bigM}).")
    if encoding_of(ctx.formulation) == "bigm" and ctx.nb is not None:
        # base only drops nb when it was told the formulation up front
        raise ValueError(f"formulation={ctx.formulation!r} needs the context "
                         f"built with the same option; pass it through "
                         f"resolve_run_options.")

    # z is pinned by minimisation whenever repairing costs something
    costs = _resolve_costs(cfg, cfg.tau)
    z_exact = opts.get("z_exact", ctx.z_exact)
    ctx.z_exact = (float(costs["C_R"]) <= 0.0) if z_exact is None else bool(z_exact)

    # ---- per-cell parameter arrays -------------------------------------
    ctx.bound_of = cfg.bound_method
    ctx.repair_of = cfg.repair_model
    ctx.v_0 = cfg.v_0 if cfg.v_0 is not None else np.zeros((F, L))
    ctx.v_new = cfg.replacement_v if cfg.replacement_v is not None else np.zeros((F, L))

    if bool(opts.get("replacement_as_new", True)):
        ctx.mu_new = np.array(ctx.mu_new, dtype=float, copy=True)
        ctx.v_new = np.array(ctx.v_new, dtype=float, copy=True)
        for (i, l) in cells:
            ctx.mu_new[i, l] = 0.0
            ctx.v_new[i, l] = 0.0
    ctx.s_chernoff = (cfg.s_chernoff if cfg.s_chernoff is not None
                      else np.full((F, L), np.nan))

    track_v_of = np.zeros((F, L), dtype=bool)
    latch_of = np.zeros((F, L), dtype=bool)
    support_max_of = np.zeros((F, L))
    Le = np.zeros((F, L))
    ln_eps = np.zeros((F, L))
    for (i, l) in cells:
        b = str(ctx.bound_of[i, l])
        rep = str(ctx.repair_of[i, l])

        if b in _ARD1_UNSUPPORTED and rep == "ard1":
            raise ValueError(
                f"cell (i={i}, l={l}): bound {b!r} has no closed ARD1 recursion "
                f"(its descriptor is not homogeneous under a pathwise "
                f"contraction); use repair_model='ardinf' or bound_method="
                f"'bernstein'.")
        track_v_of[i, l] = b in _TRACK_V
        latch_of[i, l] = rep == "ard1"
        support_max_of[i, l] = _cell_max(cfg.support, cfg.support_trans, i, l)
        Le[i, l] = math.log(1.0 / float(ctx.eps[i, l]))
        ln_eps[i, l] = math.log(float(ctx.eps[i, l]))
    ctx.track_v_of, ctx.latch_of = track_v_of, latch_of
    ctx.support_max_of, ctx.Le, ctx.ln_eps = support_max_of, Le, ln_eps

    # ---- resolve the reliability implementation per cell ---------------
    nonconvex = False
    for (i, l) in cells:
        name, spec = _resolve_impl(str(ctx.bound_of[i, l]), opts["reliability_impl"])
        ctx.impl_of[(i, l)] = name
        nonconvex = nonconvex or spec.quadratic
    if nonconvex:
        md.Params.NonConvex = 2

    # ---- profile accessors this model needs ----------------------------
    H1, H2 = ctx.H1, ctx.H2
    ctx.v_inc = make_accessor(cfg.v, cfg.v_trans, H1, H2)
    ctx.w2_inc = make_accessor(cfg.support, cfg.support_trans, H1, H2,
                               transform=lambda a: a * a)
    ctx.cgf_inc = make_accessor(cfg.cgf, cfg.cgf_trans, H1, H2)

    # ---- auxiliary variables (created only if some cell needs them) ----
    rf_bounds = {str(ctx.bound_of[i, l]) for (i, l) in cells}
    rf_repairs = {str(ctx.repair_of[i, l]) for (i, l) in cells}
    need_v = bool(track_v_of.any())
    need_latch = "ard1" in rf_repairs
    # gR is needed only where BOTH conditions hold in the same cell
    need_gR = any(str(ctx.bound_of[i, l]) == "hoeffding" and latch_of[i, l]
                  for (i, l) in cells)
    if need_v:
        ctx.v_var = md.addVars(F, L, T, lb=0.0, name="v")
    if need_latch:
        ctx.gmu = md.addVars(F, L, T, lb=0.0, name="gmu")
        if need_v:
            ctx.gv = md.addVars(F, L, T, lb=0.0, name="gv")
    if "hoeffding" in rf_bounds:
        ctx.R_var = md.addVars(F, L, T, lb=0.0, name="R")
    if need_gR:
        ctx.gR = md.addVars(F, L, T, lb=0.0, name="gR")
    if "chernoff" in rf_bounds:
        ctx.K_var = md.addVars(F, L, T, lb=0.0, name="K")

    # big-M values are read off these bounds, so they must be set first
    if assembly_of(ctx.formulation) == "sparse":
        _tighten_bounds_vec(ctx, cfg, cells)
    else:
        _tighten_bounds(ctx, cfg, cells)
    md.update()                       # make the fresh UBs readable


def extract(ctx: _RFModel, cfg, out: dict) -> None:
    """Model-specific result fields (the shared arrays are handled by base)."""
    out.setdefault("formulation", str(ctx.formulation))
    return None


# ===========================================================================
# Rainflow cell = gating + state recursion + reliability
# ===========================================================================
def _add_rainflow_cell(ctx: _RFModel, i: int, l: int) -> None:
    if assembly_of(ctx.formulation) == "sparse":
        raise RuntimeError(
            f"formulation={ctx.formulation!r} assembles the whole fleet at "
            f"once; the per-cell dispatch loop must not be reached. "
            f"base.build_fleet should have routed to "
            f"rainflow_sparse.build_fleet_sparse.")
    add_maintenance_gating(ctx, i, l)        # shared (base); nb-aware
    if encoding_of(ctx.formulation) == "bigm":
        _add_rainflow_state_bigm(ctx, i, l)
    else:
        _add_rainflow_state_ind(ctx, i, l)
    _add_reliability(ctx, i, l)


# ---------------------------------------------------------------------------
# Big-M helpers
# ---------------------------------------------------------------------------
def _ub(ctx: _RFModel, var, i: int, l: int, k: int) -> float:
    """A valid, TIGHT big-M for one state variable: its own upper bound.

    ``_tighten_bounds`` has already put a problem-derived bound on every state
    (``tau`` / ``eps*tau`` for mu, the reachable variance for v, ``T*b^2`` for R,
    ``T*max cgf`` for K).  Falling back on ``ctx.bigM`` only happens for a
    variable left free, which in practice means a cell whose bound does not use
    that accumulator."""
    try:
        b = float(var[i, l, k].UB)
    except (AttributeError, gp.GurobiError):
        b = float("inf")
    if not math.isfinite(b) or b >= GRB.INFINITY:
        return float(ctx.bigM)
    return b


def _state_bigm(ctx: _RFModel, i: int, l: int, *, var, s0, inc, a, latch,
                new, name) -> None:
    """Big-M rows for ONE state with the shared three-branch recursion

        no intervention   s_k = s_{k-1} + inc_k
        repair            s_k = a*s_{k-1} + (1-a)*g_{k-1}    (a = 1-rho or (1-rho)^2;
                                                              g omitted for ARD-inf)
        replacement       s_k = new

    ``a in (0, 1]`` and the latch never exceeds the state, so the repair branch
    is a convex combination lying in ``[g_{k-1}, s_{k-1}]``.  That is what makes
    the M-free row (U1) below valid for *all three* branches at once.
    """
    md, T = ctx.model, ctx.T
    m_rep, r_rep = ctx.m_rep, ctx.r_rep
    allow_rep = ctx.allow_replacement
    new = float(new)

    for k in range(T):
        prev = s0 if k == 0 else var[i, l, k - 1]
        cur = var[i, l, k]
        act = ctx.act_of(i, l, k)                    # m + r  ( = 1 - nb )
        m_k = m_rep[i, l, k]
        r_k = r_rep[i, l, k] if allow_rep else None
        M = _ub(ctx, var, i, l, k)
        inc_k = inc(k)

        # --- (U1) no branch ever overshoots the carry value.  No big-M: this is
        # the cut the indicator model cannot state, and it is what tightens the
        # relaxation the most (it caps every state by its own history).
        md.addConstr(cur <= prev + inc_k + (new * r_k if (allow_rep and new > 0.0)
                                            else 0.0),
                     name=f"{name}_carry_ub_{i}_{l}_{k}")

        # --- (L1) carry branch, active when nb = 1.  M = UB is tight because a
        # depot day (m + r = 1) forces inc_k = 0 through the gating.
        md.addConstr(cur >= prev + inc_k - M * act,
                     name=f"{name}_carry_lb_{i}_{l}_{k}")

        # --- (L2, U2) repair branch, active when m = 1
        if latch is not None:
            g_prev = 0.0 if k == 0 else latch[i, l, k - 1]
            rep_expr = a * prev + (1.0 - a) * g_prev
        else:
            rep_expr = a * prev
        md.addConstr(cur >= rep_expr - M * (1.0 - m_k),
                     name=f"{name}_rep_lb_{i}_{l}_{k}")
        md.addConstr(cur <= rep_expr + M * (1.0 - m_k),
                     name=f"{name}_rep_ub_{i}_{l}_{k}")

        # --- (L3, U3) replacement branch, active when r = 1
        if allow_rep:
            md.addConstr(cur <= new + M * (1.0 - r_k),
                         name=f"{name}_repl_ub_{i}_{l}_{k}")
            if new > 0.0:            # otherwise implied by cur >= 0
                md.addConstr(cur >= new - M * (1.0 - r_k),
                             name=f"{name}_repl_lb_{i}_{l}_{k}")


def _latch_bigm(ctx: _RFModel, i: int, l: int, *, latch, var, name) -> None:
    """Big-M rows for one ARD1 latch  ``g_k = g_{k-1} if nb else state_k``.

    Two of the four rows carry no big-M:
      * ``g_k <= s_k``   -- the latch is a past state and the state only grows
        between interventions, and on an intervention step the latch IS the
        state;
      * ``g_k >= g_{k-1}`` (relaxed by ``r`` only) -- a repair moves the state
        into ``[g_{k-1}, s_{k-1}]``, so the latch never decreases unless the
        component is replaced.
    """
    md, T = ctx.model, ctx.T
    allow_rep = ctx.allow_replacement

    for k in range(T):
        g_prev = 0.0 if k == 0 else latch[i, l, k - 1]
        g_k = latch[i, l, k]
        s_k = var[i, l, k]
        act = ctx.act_of(i, l, k)
        Mg = _ub(ctx, latch, i, l, k)

        md.addConstr(g_k <= s_k, name=f"{name}_le_state_{i}_{l}_{k}")
        md.addConstr(g_k >= s_k - Mg * (1.0 - act), name=f"{name}_set_{i}_{l}_{k}")
        md.addConstr(g_k <= g_prev + Mg * act, name=f"{name}_hold_ub_{i}_{l}_{k}")
        if allow_rep:
            md.addConstr(g_k >= g_prev - Mg * ctx.r_rep[i, l, k],
                         name=f"{name}_hold_lb_{i}_{l}_{k}")
        else:
            md.addConstr(g_k >= g_prev, name=f"{name}_hold_lb_{i}_{l}_{k}")


def _z_bigm(ctx: _RFModel, i: int, l: int) -> None:
    """Removed expected damage (eq. 6) without indicators.

    Eq. 6 says ``z_k = 0`` on a no-intervention step and ``z_k = mu_{k-1} - mu_k``
    on a repair / replacement step.  On a no-intervention step the difference is
    ``-inc_k <= 0`` and on an intervention step it is ``>= 0``, so eq. 6 is
    exactly ``z_k = max(0, mu_{k-1} - mu_k)`` -- and with ``C_R > 0`` the single
    row ``z_k >= mu_{k-1} - mu_k`` (plus ``z_k >= 0``) is driven to it by the
    objective.  The two upper rows are added only when nothing pushes z down.
    """
    md, T = ctx.model, ctx.T
    mu_var, z_var = ctx.mu_var, ctx.z_var
    for k in range(T):
        mu_prev = ctx.mu_0[i, l] if k == 0 else mu_var[i, l, k - 1]
        z_k = z_var[i, l, k]
        md.addConstr(z_k >= mu_prev - mu_var[i, l, k], name=f"z_lb_{i}_{l}_{k}")
        if ctx.z_exact:
            Mz = _ub(ctx, mu_var, i, l, k)
            act = ctx.act_of(i, l, k)
            md.addConstr(z_k <= mu_prev - mu_var[i, l, k] + Mz * (1.0 - act),
                         name=f"z_ub_{i}_{l}_{k}")
            md.addConstr(z_k <= Mz * act, name=f"z_zero_{i}_{l}_{k}")


def _add_rainflow_state_bigm(ctx: _RFModel, i: int, l: int) -> None:
    """Per-cell state recursion, big-M encoding.

    Same four state families as the indicator version -- mean, variance, ARD1
    latches, and the bound-specific accumulators R (Hoeffding) / K (Chernoff) --
    but every ``addGenConstrIndicator`` is replaced by linear rows and ``nb`` is
    gone.  A new bound with a new accumulator plugs in the same way: call
    ``_state_bigm`` (and ``_latch_bigm`` if it latches under ARD1) with that
    accumulator's contraction factor.
    """
    M = ctx.M
    x = ctx.x
    bound = str(ctx.bound_of[i, l])
    track_v = bool(ctx.track_v_of[i, l])
    use_latch = bool(ctx.latch_of[i, l])

    r_il = float(ctx.rho[i, l])
    k1 = 1.0 - r_il                      # (1 - rho)
    k2 = k1 * k1                         # (1 - rho)^2

    def _inc(accessor):
        return lambda k: gp.quicksum(x[i, j, k] * accessor(i, j - 1, l, k)
                                     for j in range(1, M + 1))

    # ----- mean -----
    _state_bigm(ctx, i, l, var=ctx.mu_var, s0=float(ctx.mu_0[i, l]),
                inc=_inc(ctx.mu_inc), a=k1,
                latch=ctx.gmu if use_latch else None,
                new=float(ctx.mu_new[i, l]), name="mu")
    if use_latch:
        _latch_bigm(ctx, i, l, latch=ctx.gmu, var=ctx.mu_var, name="gmu")

    # ----- variance (only bounds that use it) -----
    if track_v:
        _state_bigm(ctx, i, l, var=ctx.v_var, s0=float(ctx.v_0[i, l]),
                    inc=_inc(ctx.v_inc), a=k2,
                    latch=ctx.gv if use_latch else None,
                    new=float(ctx.v_new[i, l]), name="v")
        if use_latch:
            _latch_bigm(ctx, i, l, latch=ctx.gv, var=ctx.v_var, name="gv")

    # ----- removed expected damage z (eq. 6) -----
    _z_bigm(ctx, i, l)

    # ----- extra descriptor recursions -----
    if bound == "hoeffding":
        # R = sum_j n_j b_j^2 is degree-2 homogeneous under a pathwise
        # contraction, hence the same (1-rho)^2 factor as the variance.
        _state_bigm(ctx, i, l, var=ctx.R_var, s0=0.0,
                    inc=_inc(ctx.w2_inc), a=k2,
                    latch=ctx.gR if use_latch else None,
                    new=0.0, name="R")
        if use_latch:
            _latch_bigm(ctx, i, l, latch=ctx.gR, var=ctx.R_var, name="gR")

    if bound == "chernoff":
        # ARD-inf only (guarded in prepare / config): (1-rho)*K over-estimates
        # the true post-repair CGF psi((1-rho)s), so the bound stays valid.
        _state_bigm(ctx, i, l, var=ctx.K_var, s0=0.0,
                    inc=_inc(ctx.cgf_inc), a=k1, latch=None,
                    new=0.0, name="K")


# ---------------------------------------------------------------------------
# Indicator encoding -- the original v1 math, kept verbatim for comparison
# ---------------------------------------------------------------------------
def _add_rainflow_state_ind(ctx: _RFModel, i: int, l: int) -> None:
    """Per-cell state recursion: mean, variance (if the bound uses it), the ARD1
    latch (if repair_model == ard1), removed-damage ``z`` (eq. 6), and the
    bound-specific accumulators R (Hoeffding) / K (Chernoff).

    NOTE: if you add a bound needing a new accumulator, add its recursion here
    (mirroring the R / K blocks) and register the descriptor in
    ``_BOUND_DESCRIPTORS``."""
    md, T, M = ctx.model, ctx.T, ctx.M
    x, nb, m_rep, r_rep = ctx.x, ctx.nb, ctx.m_rep, ctx.r_rep
    mu_var, v_var, gmu, gv, z_var = ctx.mu_var, ctx.v_var, ctx.gmu, ctx.gv, ctx.z_var
    R_var, K_var, gR = ctx.R_var, ctx.K_var, ctx.gR
    bound = str(ctx.bound_of[i, l])
    track_v = bool(ctx.track_v_of[i, l])
    use_latch = bool(ctx.latch_of[i, l])
    allow_rep = ctx.allow_replacement

    r_il = float(ctx.rho[i, l])
    k1 = 1.0 - r_il                      # (1 - rho)
    k2 = k1 * k1                         # (1 - rho)^2

    for k in range(T):
        mu_prev = ctx.mu_0[i, l] if k == 0 else mu_var[i, l, k - 1]
        mean_inc = gp.quicksum(x[i, j, k] * ctx.mu_inc(i, j - 1, l, k)
                               for j in range(1, M + 1))

        # ----- mean recursion -----
        md.addGenConstrIndicator(nb[i, l, k], True,
                                 mu_var[i, l, k] == mu_prev + mean_inc,
                                 name=f"mu_carry_{i}_{l}_{k}")
        if use_latch:
            gmu_prev = 0.0 if k == 0 else gmu[i, l, k - 1]
            md.addGenConstrIndicator(m_rep[i, l, k], True,
                                     mu_var[i, l, k] == k1 * mu_prev + r_il * gmu_prev,
                                     name=f"mu_ard1_{i}_{l}_{k}")
        else:
            md.addGenConstrIndicator(m_rep[i, l, k], True,
                                     mu_var[i, l, k] == k1 * mu_prev,
                                     name=f"mu_ardinf_{i}_{l}_{k}")
        if allow_rep:
            md.addGenConstrIndicator(r_rep[i, l, k], True,
                                     mu_var[i, l, k] == float(ctx.mu_new[i, l]),
                                     name=f"mu_repl_{i}_{l}_{k}")

        # ----- variance recursion (only bounds that use variance) -----
        if track_v:
            v_prev = ctx.v_0[i, l] if k == 0 else v_var[i, l, k - 1]
            var_inc = gp.quicksum(x[i, j, k] * ctx.v_inc(i, j - 1, l, k)
                                  for j in range(1, M + 1))
            md.addGenConstrIndicator(nb[i, l, k], True,
                                     v_var[i, l, k] == v_prev + var_inc,
                                     name=f"v_carry_{i}_{l}_{k}")
            if use_latch:
                gv_prev = 0.0 if k == 0 else gv[i, l, k - 1]
                md.addGenConstrIndicator(m_rep[i, l, k], True,
                                         v_var[i, l, k] == k2 * v_prev + (1.0 - k2) * gv_prev,
                                         name=f"v_ard1_{i}_{l}_{k}")
            else:
                md.addGenConstrIndicator(m_rep[i, l, k], True,
                                         v_var[i, l, k] == k2 * v_prev,
                                         name=f"v_ardinf_{i}_{l}_{k}")
            if allow_rep:
                md.addGenConstrIndicator(r_rep[i, l, k], True,
                                         v_var[i, l, k] == float(ctx.v_new[i, l]),
                                         name=f"v_repl_{i}_{l}_{k}")

        # ----- ARD1 latch (only when present) -----
        if use_latch:
            gmu_prev = 0.0 if k == 0 else gmu[i, l, k - 1]
            md.addGenConstrIndicator(nb[i, l, k], True, gmu[i, l, k] == gmu_prev,
                                     name=f"gmu_hold_{i}_{l}_{k}")
            md.addGenConstrIndicator(m_rep[i, l, k], True, gmu[i, l, k] == mu_var[i, l, k],
                                     name=f"gmu_setm_{i}_{l}_{k}")
            if allow_rep:
                md.addGenConstrIndicator(r_rep[i, l, k], True, gmu[i, l, k] == mu_var[i, l, k],
                                         name=f"gmu_setr_{i}_{l}_{k}")
            if track_v:
                gv_prev = 0.0 if k == 0 else gv[i, l, k - 1]
                md.addGenConstrIndicator(nb[i, l, k], True, gv[i, l, k] == gv_prev,
                                         name=f"gv_hold_{i}_{l}_{k}")
                md.addGenConstrIndicator(m_rep[i, l, k], True, gv[i, l, k] == v_var[i, l, k],
                                         name=f"gv_setm_{i}_{l}_{k}")
                if allow_rep:
                    md.addGenConstrIndicator(r_rep[i, l, k], True, gv[i, l, k] == v_var[i, l, k],
                                             name=f"gv_setr_{i}_{l}_{k}")

        # ----- removed expected damage z (reference eq. 6) -----
        md.addGenConstrIndicator(nb[i, l, k], True, z_var[i, l, k] == 0.0,
                                 name=f"z_zero_{i}_{l}_{k}")
        md.addGenConstrIndicator(m_rep[i, l, k], True,
                                 z_var[i, l, k] == mu_prev - mu_var[i, l, k],
                                 name=f"z_m_{i}_{l}_{k}")
        if allow_rep:
            md.addGenConstrIndicator(r_rep[i, l, k], True,
                                     z_var[i, l, k] == mu_prev - mu_var[i, l, k],
                                     name=f"z_r_{i}_{l}_{k}")

        # ----- extra descriptor recursions (Hoeffding R / Chernoff K) -----
        if bound == "hoeffding":
            R_prev = 0.0 if k == 0 else R_var[i, l, k - 1]
            w2_expr = gp.quicksum(x[i, j, k] * ctx.w2_inc(i, j - 1, l, k)
                                  for j in range(1, M + 1))
            md.addGenConstrIndicator(nb[i, l, k], True, R_var[i, l, k] == R_prev + w2_expr,
                                     name=f"R_carry_{i}_{l}_{k}")
            if use_latch:
                # ARD1: contract only the range budget accrued since the last
                # action; gR is the unrepairable floor. The latch weight
                # (1 - k2) = rho*(2 - rho) is the same as the variance latch,
                # because R = sum_j n_j b_j^2 is degree-2 homogeneous under a
                # pathwise contraction D -> (1-rho)D (widths scale by (1-rho)).
                #   R+ = gR + (1-rho)^2 (R- - gR)
                #      = (1-rho)^2 R- + (1 - (1-rho)^2) gR
                gR_prev = 0.0 if k == 0 else gR[i, l, k - 1]
                md.addGenConstrIndicator(m_rep[i, l, k], True,
                                         R_var[i, l, k] == k2 * R_prev
                                         + (1.0 - k2) * gR_prev,
                                         name=f"R_ard1_{i}_{l}_{k}")
            else:
                md.addGenConstrIndicator(m_rep[i, l, k], True, R_var[i, l, k] == k2 * R_prev,
                                         name=f"R_ardinf_{i}_{l}_{k}")
            if allow_rep:
                md.addGenConstrIndicator(r_rep[i, l, k], True, R_var[i, l, k] == 0.0,
                                         name=f"R_repl_{i}_{l}_{k}")

            # ----- ARD1 range-budget latch (mirrors the gmu / gv registers) -----
            if use_latch:
                md.addGenConstrIndicator(nb[i, l, k], True, gR[i, l, k] == gR_prev,
                                         name=f"gR_hold_{i}_{l}_{k}")
                md.addGenConstrIndicator(m_rep[i, l, k], True,
                                         gR[i, l, k] == R_var[i, l, k],
                                         name=f"gR_setm_{i}_{l}_{k}")
                if allow_rep:
                    md.addGenConstrIndicator(r_rep[i, l, k], True,
                                             gR[i, l, k] == R_var[i, l, k],
                                             name=f"gR_setr_{i}_{l}_{k}")
        if bound == "chernoff":
            K_prev = 0.0 if k == 0 else K_var[i, l, k - 1]
            cgf_expr = gp.quicksum(x[i, j, k] * ctx.cgf_inc(i, j - 1, l, k)
                                   for j in range(1, M + 1))
            md.addGenConstrIndicator(nb[i, l, k], True, K_var[i, l, k] == K_prev + cgf_expr,
                                     name=f"K_carry_{i}_{l}_{k}")
            # ARD-inf only (guarded in prepare / config): (1-rho)*K over-estimates
            # the true post-repair CGF psi((1-rho)s), so the bound stays valid.
            md.addGenConstrIndicator(m_rep[i, l, k], True, K_var[i, l, k] == k1 * K_prev,
                                     name=f"K_ardinf_{i}_{l}_{k}")
            if allow_rep:
                md.addGenConstrIndicator(r_rep[i, l, k], True, K_var[i, l, k] == 0.0,
                                         name=f"K_repl_{i}_{l}_{k}")


def _repeatability(ctx: _RFModel, i: int, l: int, k_ref: int, k_end: int) -> None:
    """Rainflow's share of eq. (loop_impl) for one cell.

    ``base.add_repeatability_constraints`` has already imposed the mean row
    ``mu[k_end] <= mu[k_ref]`` (the "all bounds" line).  What is left is one row
    per EXTRA descriptor this cell's bound carries, because a loop is only
    closed when every state the reliability constraint reads is no worse at the
    end of the horizon than at the end of the transitory phase:

        v[k_end] <= v[k_ref]    Cantelli, Bernstein   (``_TRACK_V``)
        R[k_end] <= R[k_ref]    Hoeffding
        K[k_end] <= K[k_ref]    Chernoff

    Markov reads only the mean, so it contributes nothing here and is fully
    covered by the shared row.

    The K row extends the three lines of the reference: Chernoff's cumulant
    accumulator is the whole state of such a cell, so leaving it out would close
    the loop on a quantity (mu) that the Chernoff reliability row never reads,
    and the cycle would not in fact be repeatable.  It is imposed for the same
    reason as R, and drops out for every other bound.

    The ARD1 latch registers (gmu / gv / gR) are deliberately NOT constrained.
    They are memory of the last intervention rather than degradation state: they
    only ever enter as the floor a further repair cannot go below, so a row on
    them would restrict the schedule without being required by the loop.

    These rows are purely linear and read only the state variables, so they are
    identical under all four formulations; ``rainflow_sparse._repeatability_rows``
    emits the same set through the matrix API.  If you add a descriptor here,
    add it there too -- ``test_sparse_version.py`` will fail loudly if you don't.
    """
    md = ctx.model
    bound = str(ctx.bound_of[i, l])
    if bool(ctx.track_v_of[i, l]) and ctx.v_var is not None:
        md.addConstr(ctx.v_var[i, l, k_end] <= ctx.v_var[i, l, k_ref],
                     name=f"rep_v_{i}_{l}")
    if bound == "hoeffding" and ctx.R_var is not None:
        md.addConstr(ctx.R_var[i, l, k_end] <= ctx.R_var[i, l, k_ref],
                     name=f"rep_R_{i}_{l}")
    if bound == "chernoff" and ctx.K_var is not None:
        md.addConstr(ctx.K_var[i, l, k_end] <= ctx.K_var[i, l, k_ref],
                     name=f"rep_K_{i}_{l}")


# ===========================================================================
# ############  RELIABILITY CONSTRAINTS  -- bounds x implementations  ########
# ===========================================================================
# A *bound* (markov/cantelli/hoeffding/bernstein/chernoff) is the probabilistic
# inequality; an *implementation* is how that inequality is encoded in the MILP.
# The same bound can have several implementations that trade accuracy for solver
# difficulty. The registry is a two-level map  bound -> impl_name -> _Impl.
#
# Each builder `f(ctx, i, l)` adds the per-step  P(D > tau) <= eps  constraints
# for ONE rainflow cell, reading state (mu_var / v_var / R_var / K_var) and the
# cell parameters (tau, eps, Le, ln_eps, support_max, s_chernoff) from ctx.
#
# The three moment/accumulator bounds share one shape:  Q <= c2*d^2 + c1*d  with
# d = tau - mu (convex in mu). Encodings:
#   * "exact"   : the nonconvex quadratic as-is (NonConvex=2). Accurate, slowest.
#   * "tangent" : ONE supporting tangent of the convex RHS -> a single linear
#                 halfspace. Safe INNER approximation; no binaries at all, so it
#                 is identical under both formulations.
#   * "pwl"     : piecewise tangent over `pwl_points` segments of mu in [0, tau].
#                 THIS is where the formulation matters: the segment is selected
#                 with indicator constraints ('indicator') or with two aggregated
#                 linear rows plus tight per-segment big-Ms ('bigm').
# Markov and Chernoff (fixed s) are already linear -> only "exact".
#
# ADD A NEW RELAXATION FAMILY (e.g. a secant OUTER bound, McCormick, SOC, an
# iterative-cut scheme) by writing a builder and registering it as another impl:
#     RELIABILITY_BOUNDS["cantelli"]["secant"] = _Impl(_rel_cantelli_secant,
#                                                       quadratic=False,
#                                                       validity="outer")
# `quadratic` drives the NonConvex=2 gate; `validity` documents the relationship
# to the exact feasible set ("exact" | "inner"=safe | "outer"=unsafe/diagnostic).
# A bound needing a NEW accumulator must also add its recursion in BOTH
# `_add_rainflow_state_ind` and `_add_rainflow_state_bigm`, and declare it in
# `_BOUND_DESCRIPTORS`.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Impl:
    fn: Callable[["_RFModel", int, int], None]
    quadratic: bool          # contributes a nonconvex quadratic term -> NonConvex=2
    validity: str            # "exact" | "inner" (safe) | "outer" (unsafe, diagnostic)
    note: str = ""


# ---- shared coefficients: the convex cap  Q(mu) <= c2*(tau-mu)^2 + c1*(tau-mu) --
def _cap_coeffs(ctx: _RFModel, i: int, l: int, bound: str):
    """Return (c2, c1, Q_var) so the bound reads  Q <= c2*d^2 + c1*d, d = tau-mu.
    Only defined for the quadratic-family bounds."""
    eps = float(ctx.eps[i, l])
    if bound == "cantelli":
        return eps / (1.0 - eps), 0.0, ctx.v_var
    if bound == "hoeffding":
        return 2.0 / float(ctx.Le[i, l]), 0.0, ctx.R_var
    if bound == "bernstein":
        b = float(ctx.support_max_of[i, l])
        return 1.0 / (2.0 * float(ctx.Le[i, l])), -(b / 3.0), ctx.v_var
    raise ValueError(f"no convex-cap coefficients for bound {bound!r}")


def _add_tangent_cap(ctx, i, l, c2, c1, Qvar, mu_p, name):
    """Q <= tangent of g(mu)=c2*(tau-mu)^2 + c1*(tau-mu) taken at mu = mu_p.
    Since g is convex, the tangent lies below g, so this is a SAFE (inner) linear
    cap: Q <= tangent(mu) <= g(mu). Valid for any mu (single halfspace)."""
    md, T, tau = ctx.model, ctx.T, float(ctx.tau[i, l])
    d_p = tau - mu_p
    g_p = c2 * d_p * d_p + c1 * d_p                 # g(mu_p)
    gp_prime = -2.0 * c2 * d_p - c1                 # g'(mu_p)  (chain rule, d=tau-mu)
    for k in range(T):
        mu = ctx.mu_var[i, l, k]
        md.addConstr(Qvar[i, l, k] <= g_p + gp_prime * (mu - mu_p),
                     name=f"{name}_{i}_{l}_{k}")


def _quadratic_family(bound):
    return bound in ("cantelli", "hoeffding", "bernstein")


# ---------------------------------------------------------------------------
# EXACT encodings (the original, verbatim math)
# ---------------------------------------------------------------------------
def _rel_markov(ctx: _RFModel, i: int, l: int) -> None:
    md, T = ctx.model, ctx.T
    tau, eps = float(ctx.tau[i, l]), float(ctx.eps[i, l])
    for k in range(T):
        md.addConstr(ctx.mu_var[i, l, k] <= eps * tau, name=f"rel_{i}_{l}_{k}")


def _rel_cantelli_exact(ctx: _RFModel, i: int, l: int) -> None:
    md, T = ctx.model, ctx.T
    tau, eps = float(ctx.tau[i, l]), float(ctx.eps[i, l])
    for k in range(T):
        mu = ctx.mu_var[i, l, k]
        md.addConstr(mu <= tau, name=f"rel_{i}_{l}_{k}_gap")
        md.addQConstr((1.0 - eps) * ctx.v_var[i, l, k] <= eps * (tau - mu) * (tau - mu),
                      name=f"rel_{i}_{l}_{k}")


def _rel_hoeffding_exact(ctx: _RFModel, i: int, l: int) -> None:
    md, T = ctx.model, ctx.T
    tau, Le = float(ctx.tau[i, l]), float(ctx.Le[i, l])
    for k in range(T):
        mu = ctx.mu_var[i, l, k]
        md.addConstr(mu <= tau, name=f"rel_{i}_{l}_{k}_gap")
        md.addQConstr((tau - mu) * (tau - mu) >= 0.5 * Le * ctx.R_var[i, l, k],
                      name=f"rel_{i}_{l}_{k}")


def _rel_bernstein_exact(ctx: _RFModel, i: int, l: int) -> None:
    md, T = ctx.model, ctx.T
    tau, Le = float(ctx.tau[i, l]), float(ctx.Le[i, l])
    b = float(ctx.support_max_of[i, l])
    for k in range(T):
        mu = ctx.mu_var[i, l, k]
        t = tau - mu
        md.addConstr(mu <= tau, name=f"rel_{i}_{l}_{k}_gap")
        md.addQConstr(0.5 * t * t - (Le * b / 3.0) * t - Le * ctx.v_var[i, l, k] >= 0,
                      name=f"rel_{i}_{l}_{k}")


def _rel_chernoff(ctx: _RFModel, i: int, l: int) -> None:
    md, T = ctx.model, ctx.T
    tau = float(ctx.tau[i, l])
    s, ln_eps = float(ctx.s_chernoff[i, l]), float(ctx.ln_eps[i, l])
    for k in range(T):
        md.addConstr(ctx.K_var[i, l, k] - s * tau <= ln_eps, name=f"rel_{i}_{l}_{k}")


# ---------------------------------------------------------------------------
# SINGLE-TANGENT encodings (safe inner, linear, no extra binaries)
# ---------------------------------------------------------------------------
def _rel_quadratic_tangent(ctx: _RFModel, i: int, l: int) -> None:
    """One supporting tangent at mu_p = tangent_ref * tau, plus the mu<=tau gap.
    Formulation-independent: there is no logical constraint to encode."""
    md, T = ctx.model, ctx.T
    bound = str(ctx.bound_of[i, l])
    tau = float(ctx.tau[i, l])
    c2, c1, Qvar = _cap_coeffs(ctx, i, l, bound)
    for k in range(T):
        md.addConstr(ctx.mu_var[i, l, k] <= tau, name=f"rel_{i}_{l}_{k}_gap")
    mu_p = float(np.clip(ctx.tangent_ref, 0.0, 1.0)) * tau
    _add_tangent_cap(ctx, i, l, c2, c1, Qvar, mu_p, name="rel")


# ---------------------------------------------------------------------------
# PIECEWISE-TANGENT encodings (safe inner, linear per piece, segment binaries)
# ---------------------------------------------------------------------------
def _rel_quadratic_pwl(ctx: _RFModel, i: int, l: int) -> None:
    """Partition mu in [0, tau] into `pwl_points` segments; in each segment the
    tangent at the segment midpoint caps Q. A per-step binary selects the active
    segment (mu in [m_{s-1}, m_s]) and activates that segment's tangent. Each
    tangent is <= g on its segment, so the encoding is a SAFE inner approximation
    whose error shrinks as pwl_points grows.

    pwl_points == 1 degenerates to a single midpoint tangent with no binaries.

    Segment selection is where the two formulations differ; both give the same
    integer feasible set, but the big-M one puts the whole selection into the LP.
    """
    md, T = ctx.model, ctx.T
    bound = str(ctx.bound_of[i, l])
    tau = float(ctx.tau[i, l])
    c2, c1, Qvar = _cap_coeffs(ctx, i, l, bound)
    K = max(1, int(ctx.pwl_points))
    edges = np.linspace(0.0, tau, K + 1)          # breakpoints in mu
    bigm = encoding_of(ctx.formulation) == "bigm"

    for k in range(T):
        md.addConstr(ctx.mu_var[i, l, k] <= tau, name=f"rel_{i}_{l}_{k}_gap")

    if K == 1:                                     # single tangent, no binaries
        _add_tangent_cap(ctx, i, l, c2, c1, Qvar, 0.5 * tau, name="rel")
        return

    # per-segment tangent data, computed once for the whole cell
    seg = []
    for s in range(K):
        lo, hi = float(edges[s]), float(edges[s + 1])
        mid = 0.5 * (lo + hi)
        d_p = tau - mid
        g_p = c2 * d_p * d_p + c1 * d_p
        gp_prime = -2.0 * c2 * d_p - c1
        seg.append((lo, hi, mid, g_p, gp_prime))

    for k in range(T):
        mu = ctx.mu_var[i, l, k]
        Q = Qvar[i, l, k]
        zs = md.addVars(K, vtype=GRB.BINARY, name=f"relseg_{i}_{l}_{k}")
        md.addConstr(gp.quicksum(zs[s] for s in range(K)) == 1,
                     name=f"relseg1_{i}_{l}_{k}")

        if bigm:
            # Aggregated membership: with sum_s z_s = 1 these two rows say
            # exactly "mu lies in the selected segment", and unlike 2K indicator
            # constraints they are present in the relaxation.
            md.addConstr(mu >= gp.quicksum(seg[s][0] * zs[s] for s in range(K)),
                         name=f"relslo_{i}_{l}_{k}")
            md.addConstr(mu <= gp.quicksum(seg[s][1] * zs[s] for s in range(K)),
                         name=f"relshi_{i}_{l}_{k}")
            Qub = _ub(ctx, Qvar, i, l, k)
            for s in range(K):
                lo, hi, mid, g_p, gp_prime = seg[s]
                # smallest M that leaves the row slack when z_s = 0: the tangent
                # is affine in mu, so its minimum over [0, tau] is at an endpoint
                lo_val = min(g_p + gp_prime * (0.0 - mid),
                             g_p + gp_prime * (tau - mid))
                Ms = max(0.0, Qub - lo_val)
                md.addConstr(Q <= g_p + gp_prime * (mu - mid) + Ms * (1.0 - zs[s]),
                             name=f"relcap_{i}_{l}_{k}_{s}")
        else:
            for s in range(K):
                lo, hi, mid, g_p, gp_prime = seg[s]
                # segment membership (exact, via indicators)
                md.addGenConstrIndicator(zs[s], True, mu >= lo,
                                         name=f"relslo_{i}_{l}_{k}_{s}")
                md.addGenConstrIndicator(zs[s], True, mu <= hi,
                                         name=f"relshi_{i}_{l}_{k}_{s}")
                # segment tangent cap (only when this segment is active)
                md.addGenConstrIndicator(zs[s], True,
                                         Q <= g_p + gp_prime * (mu - mid),
                                         name=f"relcap_{i}_{l}_{k}_{s}")


# ===========================================================================
# Registry:  bound -> impl_name -> _Impl   (default impl per bound is "exact")
# ===========================================================================
RELIABILITY_BOUNDS: Dict[str, Dict[str, _Impl]] = {
    "markov": {
        "exact": _Impl(_rel_markov, quadratic=False, validity="exact"),
    },
    "cantelli": {
        "exact":   _Impl(_rel_cantelli_exact,     quadratic=True,  validity="exact"),
        "tangent": _Impl(_rel_quadratic_tangent,  quadratic=False, validity="inner",
                         note="single supporting tangent"),
        "pwl":     _Impl(_rel_quadratic_pwl,      quadratic=False, validity="inner",
                         note="piecewise tangent, pwl_points segments"),
    },
    "hoeffding": {
        "exact":   _Impl(_rel_hoeffding_exact,    quadratic=True,  validity="exact"),
        "tangent": _Impl(_rel_quadratic_tangent,  quadratic=False, validity="inner",
                         note="single supporting tangent"),
        "pwl":     _Impl(_rel_quadratic_pwl,      quadratic=False, validity="inner",
                         note="piecewise tangent, pwl_points segments"),
    },
    "bernstein": {
        "exact":   _Impl(_rel_bernstein_exact,    quadratic=True,  validity="exact"),
        "tangent": _Impl(_rel_quadratic_tangent,  quadratic=False, validity="inner",
                         note="single supporting tangent"),
        "pwl":     _Impl(_rel_quadratic_pwl,      quadratic=False, validity="inner",
                         note="piecewise tangent, pwl_points segments"),
    },
    "chernoff": {
        "exact": _Impl(_rel_chernoff, quadratic=False, validity="exact"),
    },
}


def _resolve_impl(bound: str, requested: str):
    """Return (impl_name, _Impl) for a bound and a requested implementation.
    Falls back to the bound's 'exact' encoding when the requested impl doesn't
    exist for it (e.g. asking for 'pwl' on markov/chernoff, already linear)."""
    if bound not in RELIABILITY_BOUNDS:
        raise ValueError(f"unknown bound_method {bound!r}; "
                         f"registered: {tuple(RELIABILITY_BOUNDS)}.")
    impls = RELIABILITY_BOUNDS[bound]
    name = requested if requested in impls else "exact"
    return name, impls[name]


def _add_reliability(ctx: _RFModel, i: int, l: int) -> None:
    """Dispatch one rainflow cell to its resolved (bound, impl) builder."""
    bound = str(ctx.bound_of[i, l])
    name = ctx.impl_of.get((i, l))
    if name is None:                                   # defensive: resolve if missing
        name, _ = _resolve_impl(bound, "exact")
    RELIABILITY_BOUNDS[bound][name].fn(ctx, i, l)


# ===========================================================================
# Bound tightening
# ===========================================================================
def _tighten_bounds_vec(ctx: _RFModel, cfg, cells) -> None:
    """Vectorised twin of ``_tighten_bounds``: identical numbers, one setAttr
    per variable block instead of one assignment per variable.

    Used only by ``formulation='sparse'``.  The scalar version below is the
    reference and is left untouched; ``test_sparse_version.py`` compares the
    two models variable by variable, which is what makes replacing a
    ``Theta(F*L*T)`` loop safe to do.
    """
    md, F, L, T = ctx.model, ctx.F, ctx.L, ctx.T
    inf = float(GRB.INFINITY)
    cell_mask = np.zeros((F, L), dtype=bool)
    for i, l in cells:
        cell_mask[i, l] = True

    tau = np.asarray(ctx.tau, dtype=float)
    eps = np.asarray(ctx.eps, dtype=float)
    is_markov = np.asarray([[str(ctx.bound_of[i, l]) == "markov"
                             for l in range(L)] for i in range(F)])
    latch = np.asarray(ctx.latch_of, dtype=bool)
    track_v = np.asarray(ctx.track_v_of, dtype=bool)

    def _apply(block, ub_fl):
        """Broadcast an (F, L) bound over the horizon and set it in one call."""
        if block is None:
            return
        md.setAttr(GRB.Attr.UB, list(block.values()),
                   np.repeat(ub_fl.ravel(), T).tolist())

    # mu / z: build_context already set tau everywhere; only rainflow cells move
    mu_ub = np.where(cell_mask, np.where(is_markov, eps * tau, tau), tau)
    _apply(ctx.mu_var, mu_ub)
    _apply(ctx.z_var, np.where(cell_mask, tau, tau))

    if ctx.gmu is not None:
        _apply(ctx.gmu, np.where(cell_mask & latch, mu_ub, inf))

    if ctx.v_var is not None:
        v_ub = np.full((F, L), inf)
        for i, l in cells:
            if not track_v[i, l]:
                continue
            vmax = _cell_max(cfg.v, cfg.v_trans, i, l)
            v_reach = float(ctx.v_0[i, l]) + T * vmax
            v_ub[i, l] = (min(v_reach, eps[i, l] / (1.0 - eps[i, l])
                              * tau[i, l] * tau[i, l])
                          if str(ctx.bound_of[i, l]) == "cantelli" else v_reach)
        _apply(ctx.v_var, v_ub)
        if ctx.gv is not None:
            _apply(ctx.gv, np.where(cell_mask & latch & track_v, v_ub, inf))

    if ctx.R_var is not None:
        R_ub = np.full((F, L), inf)
        for i, l in cells:
            if str(ctx.bound_of[i, l]) == "hoeffding":
                R_ub[i, l] = float(T * (_cell_max(cfg.support, cfg.support_trans,
                                                  i, l) ** 2))
        _apply(ctx.R_var, R_ub)
        if ctx.gR is not None:
            _apply(ctx.gR, np.where(np.isfinite(R_ub) & latch, R_ub, inf))

    if ctx.K_var is not None:
        K_ub = np.full((F, L), inf)
        for i, l in cells:
            if str(ctx.bound_of[i, l]) == "chernoff":
                K_ub[i, l] = float(T * _cell_max(cfg.cgf, cfg.cgf_trans, i, l))
        _apply(ctx.K_var, K_ub)


def _tighten_bounds(ctx: _RFModel, cfg, cells) -> None:
    """Set valid, tight upper bounds per rainflow cell (strengthens the
    relaxation without changing the optimum). Unused-cell entries of a shared
    auxiliary variable are left free (they appear in no constraint/objective).

    In the big-M formulation these bounds do double duty: they ARE the big-M
    constants, so anything tightened here tightens the relaxation twice."""
    T = ctx.T
    for i, l in cells:
        bound = str(ctx.bound_of[i, l])
        tau = float(ctx.tau[i, l]); eps = float(ctx.eps[i, l])
        mu_ub = eps * tau if bound == "markov" else tau
        for k in range(T):
            ctx.mu_var[i, l, k].UB = mu_ub
            ctx.z_var[i, l, k].UB = tau
            if ctx.gmu is not None and ctx.latch_of[i, l]:
                ctx.gmu[i, l, k].UB = mu_ub
        if ctx.track_v_of[i, l]:
            vmax = _cell_max(cfg.v, cfg.v_trans, i, l)
            v_reach = float(ctx.v_0[i, l]) + T * vmax
            if bound == "cantelli":
                v_ub = min(v_reach, eps / (1.0 - eps) * tau * tau)
            else:
                v_ub = v_reach
            for k in range(T):
                ctx.v_var[i, l, k].UB = v_ub
                if ctx.gv is not None and ctx.latch_of[i, l]:
                    ctx.gv[i, l, k].UB = v_ub
        if bound == "hoeffding":
            R_ub = float(T * (_cell_max(cfg.support, cfg.support_trans, i, l) ** 2))
            for k in range(T):
                ctx.R_var[i, l, k].UB = R_ub
                if ctx.gR is not None and ctx.latch_of[i, l]:
                    ctx.gR[i, l, k].UB = R_ub
        if bound == "chernoff":
            K_ub = float(T * _cell_max(cfg.cgf, cfg.cgf_trans, i, l))
            for k in range(T):
                ctx.K_var[i, l, k].UB = K_ub


# ===========================================================================
# Registration: plug this model into base's cell-builder registry
# ===========================================================================
class RainflowCellBuilder:
    """Rainflow implementation of the ``base.CellBuilder`` interface (v2)."""

    name = "rainflow"

    prepare = staticmethod(prepare)
    add_cell = staticmethod(_add_rainflow_cell)
    extract = staticmethod(extract)
    # Without this hook base.add_repeatability_constraints closes the loop on the
    # MEAN ONLY, so a cantelli / hoeffding / bernstein / chernoff cell could end
    # the horizon with a larger variance / range / cumulant budget than it
    # started the operating phase with -- i.e. a "repeatable" cycle that is not
    # repeatable in the quantity its reliability row actually reads.
    repeatability = staticmethod(_repeatability)


register_cell_builder("rainflow", RainflowCellBuilder())


# ===========================================================================
# Runnable demo: the same instance under both encodings
# ===========================================================================
if __name__ == "__main__":
    from fleet_management.config import load_config

    print("Fleet management (rainflow v2: indicator vs big-M) demo")
    base_input = {
        "F": 3, "M": 1, "model": "rainflow", "bound_method": "cantelli",
        "repair_model": "ard1", "tau": 0.30, "epsilon": 0.10,
        "rho": 0.6, "mu_0": 0.02, "v_0": 4e-4, "mu": 0.06, "v": 0.0015,
        "C_M": 1.0, "C_R": 0.5, "C_D": 2.0,
    }
    for form in FORMULATIONS:
        res = solve(load_config({**base_input, "H": 4}), verbose=0, time_limit=30,
                    formulation=form)
        print(f"  {form:9s} H1=H2=4 -> status={res['status']} T={res['T']} "
              f"obj={res['objective']}")
