"""
Fleet management with a rainflow / remaining-life (Palmgren-Miner) degradation
model, solved with Gurobi -- MODULAR, per-cell version (Step 2).

This module consumes a normalized ``FleetConfig`` (see ``config.py``) and builds
one Gurobi program for the whole fleet.  The unit of modelling is a **cell**
``(i, l)`` = (vehicle ``i``, component ``l``); every cell carries its own model
(``rainflow`` / ``gamma`` / ...), reliability bound, repair model, threshold,
etc.  The shared skeleton (assignment ``x``, depot capacity, aggregate-damage
cap, safety ``u`` and the objective cost terms) is built once; each cell then
adds its own degradation / maintenance / reliability / repeatability block.

Where things live (the "clear locations")
-----------------------------------------
    solve                         entry point: build skeleton, then dispatch cells
    _build_objective              shared objective
    _add_base_constraints         shared: assignment, depot cap, aggregate cap, u
    _dispatch_cell                per-cell model switch  (rainflow / gamma / ...)
      _add_rainflow_cell            one rainflow cell = gating + state + reliability + repeat
        _add_maintenance_gating       eq. 3 gating (m, r, nb)
        _add_rainflow_state           mean / variance / ARD latch / z / R / K recursion
        _add_reliability              -> RELIABILITY_BOUNDS[bound](ctx, i, l)   <== bounds
        _add_repeatability            loop the moments (+ descriptors)
      _add_gamma_cell               PLACEHOLDER (work in progress)

Reliability bounds  ***edit / add formulations here***
------------------------------------------------------
Each entry of ``RELIABILITY_BOUNDS`` is a function ``f(ctx, i, l)`` that adds the
per-step  ``P(D > tau) <= eps``  constraints for one rainflow cell, reading the
cell's state variables and parameters from ``ctx``.  To try a different
formulation of an existing bound, edit its function; to add a new bound, write a
``_rel_<name>`` function and register it (and, if it needs a new accumulator such
as Hoeffding's ``R`` or Chernoff's ``K``, add that accumulator's recursion in
``_add_rainflow_state`` alongside the existing ones and declare the descriptor in
``_BOUND_DESCRIPTORS``).

Two horizons
------------
Time axis has a transitory phase ``H1`` (steps 0..H1-1, run-up from ``mu_0``) and
an operating phase ``H2`` (steps H1..H1+H2-1, the repeatable cycle); ``T = H1 +
H2``.  Repeatability loops the operating horizon: ``state(T-1) <= state(H1-1)``.
A single-int ``H`` gives ``H1 = H2 = H`` and ``T = 2H``.  Operating-phase
profiles come from ``cfg`` as ``(F, L, M, H2)``; optional transitory profiles are
``(F, L, M, H1)`` and reused from the operating profile when absent.

Author: Johann Tschan  (revised; modular Step-2 rewrite)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import gurobipy as gp
from gurobipy import GRB


# Reliability bounds and the extra data each needs (mirrors config.py).
_QUADRATIC = ("cantelli", "hoeffding", "bernstein")     # need Gurobi NonConvex = 2
_NEEDS_SUPPORT = ("hoeffding", "bernstein")
_NEEDS_CGF = ("chernoff",)
_TRACK_V = ("cantelli", "bernstein")                    # bounds that use the variance state
# extra per-cell accumulator a bound needs beyond (mu, v): 'R' Hoeffding, 'K' Chernoff
_BOUND_DESCRIPTORS = {"hoeffding": "R", "chernoff": "K"}
_REPAIR_MODELS = ("ard1", "ardinf")


# ---------------------------------------------------------------------------
# Shared context: model, variables, per-cell parameters, increment accessors.
# ---------------------------------------------------------------------------
@dataclass
class _RFModel:
    """Everything the per-cell builders need, built once and passed around.

    Scalar problem sizes plus **per-cell** parameter arrays of shape (F, L)
    (strings for the selectors, floats otherwise) and the shared Gurobi
    variables.  Auxiliary variables (``v_var``, ``gmu``, ``gv``, ``R_var``,
    ``K_var``) exist only when *some* cell needs them and are referenced only by
    the cells whose bound / repair model uses them.
    """
    model: gp.Model
    F: int; H1: int; H2: int; M: int; L: int; T: int
    # shared decision variables / states
    x: gp.tupledict
    m_rep: gp.tupledict
    r_rep: Optional[gp.tupledict]
    nb: gp.tupledict
    mu_var: gp.tupledict
    v_var: Optional[gp.tupledict]
    gmu: Optional[gp.tupledict]
    gv: Optional[gp.tupledict]
    z_var: gp.tupledict
    u_var: gp.tupledict
    R_var: Optional[gp.tupledict]
    K_var: Optional[gp.tupledict]
    # per-cell (F, L) parameters
    model_of: np.ndarray
    bound_of: np.ndarray
    repair_of: np.ndarray
    tau: np.ndarray
    eps: np.ndarray
    rho: np.ndarray
    mu_0: np.ndarray
    v_0: np.ndarray
    mu_new: np.ndarray
    v_new: np.ndarray
    s_chernoff: np.ndarray
    support_max_of: np.ndarray
    Le: np.ndarray
    ln_eps: np.ndarray
    track_v_of: np.ndarray                 # bool (F, L)
    latch_of: np.ndarray                   # bool (F, L)  (ARD1)
    # reliability-implementation selection
    impl_of: dict                          # (i, l) -> resolved impl name (rainflow cells)
    pwl_points: int                        # segments for the piecewise-linear surrogate
    tangent_ref: float                     # linearization point (fraction of tau) for single tangent
    # build flags
    allow_replacement: bool
    # per-mission increment accessors (i, j0, l, k) with phase-aware local time
    mu_inc: Callable[[int, int, int, int], float]
    v_inc: Callable[[int, int, int, int], float]
    w2_inc: Callable[[int, int, int, int], float]
    cgf_inc: Callable[[int, int, int, int], float]

    def rainflow_cells(self):
        return [(i, l) for i in range(self.F) for l in range(self.L)
                if self.model_of[i, l] == "rainflow"]


# ===========================================================================
# Entry point
# ===========================================================================
def solve(cfg, *, allow_replacement=None, depot_capacity=None,
          verbose=None, mip_gap=None, time_limit=None, fast=None,
          gurobi_params=None,
          reliability_impl=None, pwl_points=None, tangent_ref=None) -> dict:
    """Solve a (possibly mixed) fleet from a normalized ``FleetConfig``.

    Rainflow cells are fully supported; a ``gamma`` (or other) cell hits its
    placeholder block and raises ``NotImplementedError`` (work in progress).
    Run-time options default to the values in ``cfg.options`` when not passed
    explicitly.
    """
    F, L, M = cfg.F, cfg.L, cfg.M
    H1, H2, T = cfg.H1, cfg.H2, cfg.T
    opt = cfg.options

    allow_replacement = _pick(allow_replacement, opt.get("allow_replacement"), True)
    depot_capacity = _pick(depot_capacity, opt.get("depot_capacity"), F - M)
    verbose = _pick(verbose, opt.get("verbose"), 1)
    mip_gap = _pick(mip_gap, opt.get("mip_gap"), 0.12)
    time_limit = _pick(time_limit, opt.get("time_limit"), None)
    fast = _pick(fast, opt.get("fast"), False)
    gurobi_params = _pick(gurobi_params, opt.get("gurobi_params"), None)

    # reliability-constraint IMPLEMENTATION (exact nonconvex vs a linear surrogate);
    # see the RELIABILITY_BOUNDS registry. Global for now, resolved per cell below.
    reliability_impl = _pick(reliability_impl, opt.get("reliability_impl"), "exact")
    pwl_points = int(_pick(pwl_points, opt.get("pwl_points"), 8))
    tangent_ref = float(_pick(tangent_ref, opt.get("tangent_ref"), 0.5))

    # ---- per-cell parameter arrays -------------------------------------
    model_of = cfg.model
    bound_of = cfg.bound_method
    repair_of = cfg.repair_model
    tau, eps = cfg.tau, cfg.epsilon
    rho = cfg.rho
    mu_0 = cfg.mu_0
    v_0 = cfg.v_0 if cfg.v_0 is not None else np.zeros((F, L))
    mu_new = cfg.replacement_mu if cfg.replacement_mu is not None else np.zeros((F, L))
    v_new = cfg.replacement_v if cfg.replacement_v is not None else np.zeros((F, L))
    s_chernoff = cfg.s_chernoff if cfg.s_chernoff is not None else np.full((F, L), np.nan)

    is_rf = (model_of == "rainflow")
    track_v_of = np.zeros((F, L), dtype=bool)
    latch_of = np.zeros((F, L), dtype=bool)
    support_max_of = np.zeros((F, L))
    Le = np.zeros((F, L))
    ln_eps = np.zeros((F, L))
    for i in range(F):
        for l in range(L):
            if not is_rf[i, l]:
                continue
            b = str(bound_of[i, l])
            track_v_of[i, l] = b in _TRACK_V
            latch_of[i, l] = str(repair_of[i, l]) == "ard1"
            support_max_of[i, l] = _cell_max(cfg.support, cfg.support_trans, i, l)
            Le[i, l] = math.log(1.0 / float(eps[i, l]))
            ln_eps[i, l] = math.log(float(eps[i, l]))

    # ---- which auxiliary variables / solver flags the fleet needs ------
    rf_bounds = {str(bound_of[i, l]) for i in range(F) for l in range(L) if is_rf[i, l]}
    rf_repairs = {str(repair_of[i, l]) for i in range(F) for l in range(L) if is_rf[i, l]}
    need_v = bool(track_v_of.any())
    need_latch = "ard1" in rf_repairs
    need_gv = need_latch and need_v
    need_R = "hoeffding" in rf_bounds
    need_K = "chernoff" in rf_bounds

    # Resolve the reliability implementation per rainflow cell. NonConvex=2 is
    # needed only if some cell keeps an *exact* (nonconvex quadratic) encoding;
    # if every quadratic-bound cell uses a linear surrogate the model is a MILP.
    impl_of = {}
    nonconvex = False
    for i in range(F):
        for l in range(L):
            if not is_rf[i, l]:
                continue
            name, spec = _resolve_impl(str(bound_of[i, l]), reliability_impl)
            impl_of[(i, l)] = name
            nonconvex = nonconvex or spec.quadratic

    # ---- phase-aware increment accessors (read cfg's (F, L, M, H) arrays) --
    def _acc(op, tr, transform=lambda a: a):
        def f(i, j0, l, k):
            if k < H1:
                src = tr if tr is not None else op
                h = k % (H1 if tr is not None else H2)
                return float(transform(src[i, l, j0, h]))
            return float(transform(op[i, l, j0, (k - H1) % H2]))
        return f
    mu_inc = _acc(cfg.mu, cfg.mu_trans)
    v_inc = _acc(cfg.v, cfg.v_trans) if cfg.v is not None else (lambda *a: 0.0)
    w2_inc = _acc(cfg.support, cfg.support_trans, transform=lambda a: a * a) if cfg.support is not None else (lambda *a: 0.0)
    cgf_inc = _acc(cfg.cgf, cfg.cgf_trans) if cfg.cgf is not None else (lambda *a: 0.0)

    # ---- model + variables ---------------------------------------------
    md = gp.Model("fleet_management_rainflow_modular")
    md.Params.OutputFlag = int(verbose)
    if nonconvex:
        md.Params.NonConvex = 2
    _apply_performance_params(md, time_limit, mip_gap, fast, gurobi_params)

    x = md.addVars(F, M + 1, T, vtype=GRB.BINARY, name="x")
    m_rep = md.addVars(F, L, T, vtype=GRB.BINARY, name="m")
    r_rep = md.addVars(F, L, T, vtype=GRB.BINARY, name="r") if allow_replacement else None
    nb = md.addVars(F, L, T, vtype=GRB.BINARY, name="nb")

    mu_var = md.addVars(F, L, T, lb=0.0, name="mu")
    v_var = md.addVars(F, L, T, lb=0.0, name="v") if need_v else None
    gmu = md.addVars(F, L, T, lb=0.0, name="gmu") if need_latch else None
    gv = md.addVars(F, L, T, lb=0.0, name="gv") if need_gv else None
    z_var = md.addVars(F, L, T, lb=0.0, name="z")
    u_var = md.addVars(T, lb=0.0, name="u")
    R_var = md.addVars(F, L, T, lb=0.0, name="R") if need_R else None
    K_var = md.addVars(F, L, T, lb=0.0, name="K") if need_K else None

    ctx = _RFModel(
        model=md, F=F, H1=H1, H2=H2, M=M, L=L, T=T,
        x=x, m_rep=m_rep, r_rep=r_rep, nb=nb,
        mu_var=mu_var, v_var=v_var, gmu=gmu, gv=gv, z_var=z_var, u_var=u_var,
        R_var=R_var, K_var=K_var,
        model_of=model_of, bound_of=bound_of, repair_of=repair_of,
        tau=tau, eps=eps, rho=rho, mu_0=mu_0, v_0=v_0, mu_new=mu_new, v_new=v_new,
        s_chernoff=s_chernoff, support_max_of=support_max_of, Le=Le, ln_eps=ln_eps,
        track_v_of=track_v_of, latch_of=latch_of,
        impl_of=impl_of, pwl_points=pwl_points, tangent_ref=tangent_ref,
        allow_replacement=allow_replacement,
        mu_inc=mu_inc, v_inc=v_inc, w2_inc=w2_inc, cgf_inc=cgf_inc,
    )

    # ---- tighten variable bounds per rainflow cell (relaxation strength) ---
    _tighten_bounds(ctx, cfg)

    # ---- assemble: shared skeleton, then one block per cell ------------
    costs = _resolve_costs(cfg, tau, allow_replacement)
    _build_objective(ctx, costs)
    _add_base_constraints(ctx, int(depot_capacity))
    for i in range(F):
        for l in range(L):
            _dispatch_cell(ctx, i, l)

    # ---- solve & extract ------------------------------------------------
    md.optimize()
    return _extract_solution(ctx, cfg, md)


# ===========================================================================
# Shared skeleton
# ===========================================================================
def _build_objective(ctx: _RFModel, costs: dict) -> None:
    """C_M per depot-day, C_R per unit removed by repair, C_rep per replacement,
    C_S on worst aggregate damage, C_P periodicity slack (operating loop)."""
    md, F, L, T, H1 = ctx.model, ctx.F, ctx.L, ctx.T, ctx.H1
    C_M, C_R, C_S, C_P, C_rep = costs["C_M"], costs["C_R"], costs["C_S"], costs["C_P"], costs["C_rep"]
    obj = gp.LinExpr()
    for k in range(T):
        obj += C_S * ctx.u_var[k]
        for i in range(F):
            obj += C_M * ctx.x[i, 0, k]
            for l in range(L):
                obj += C_R * ctx.z_var[i, l, k]
                if ctx.allow_replacement:
                    obj += C_rep * ctx.r_rep[i, l, k]
    for i in range(F):
        for l in range(L):
            obj += C_P * (ctx.mu_var[i, l, H1 - 1] - ctx.mu_var[i, l, T - 1])
            if ctx.track_v_of[i, l]:
                obj += C_P * (ctx.v_var[i, l, H1 - 1] - ctx.v_var[i, l, T - 1])
    md.setObjective(obj, GRB.MINIMIZE)


def _add_base_constraints(ctx: _RFModel, depot_capacity: int) -> None:
    """Assignment, depot capacity, aggregate-damage cap, safety ``u``.

    These couple *all* cells, so every (i, l) must have a defined ``mu_var``
    recursion -- which holds for an all-rainflow fleet.  (A gamma cell would
    raise in its placeholder before the model is solved.)"""
    md, F, M, L, T = ctx.model, ctx.F, ctx.M, ctx.L, ctx.T
    x, mu_var, u_var = ctx.x, ctx.mu_var, ctx.u_var

    for i in range(F):
        for k in range(T):
            md.addConstr(gp.quicksum(x[i, j, k] for j in range(M + 1)) <= 1,
                         name=f"assign_{i}_{k}")
    for j in range(1, M + 1):
        for k in range(T):
            md.addConstr(gp.quicksum(x[i, j, k] for i in range(F)) == 1,
                         name=f"demand_{j}_{k}")
    for k in range(T):
        md.addConstr(gp.quicksum(x[i, 0, k] for i in range(F)) <= depot_capacity,
                     name=f"depot_cap_{k}")
    for k in range(T):
        md.addConstr(gp.quicksum(mu_var[i, l, k] for i in range(F) for l in range(L)) <= F - M,
                     name=f"capacity_{k}")
        for i in range(F):
            md.addConstr(u_var[k] >= gp.quicksum(mu_var[i, l, k] for l in range(L)),
                         name=f"u_{i}_{k}")


# ===========================================================================
# Per-cell dispatch  (model switch)
# ===========================================================================
def _dispatch_cell(ctx: _RFModel, i: int, l: int) -> None:
    """Route one (vehicle, component) cell to its degradation-model block."""
    model = str(ctx.model_of[i, l])
    if model == "rainflow":
        _add_rainflow_cell(ctx, i, l)
    elif model == "gamma":
        _add_gamma_cell(ctx, i, l)
    else:
        raise NotImplementedError(
            f"cell (i={i}, l={l}): degradation model {model!r} has no builder yet."
        )


# --------------------------------------------------------------------------
# GAMMA PLACEHOLDER  (work in progress)
# --------------------------------------------------------------------------
def _add_gamma_cell(ctx: _RFModel, i: int, l: int) -> None:
    """PLACEHOLDER for the modular gamma block.

    The gamma degradation dynamics, maintenance, and reliability constraints for
    a single cell will be built here, using ``ctx`` (shared variables) and the
    gamma parameters carried on the FleetConfig (``gamma_beta``, ``rho``,
    ``tau``, ``eps``, ``mu_0``, mean profile via ``ctx.mu_inc``).  A gamma cell
    still contributes its mean state ``ctx.mu_var[i, l, k]`` to the shared
    aggregate-damage cap / safety ``u`` / objective, so this block must define
    that recursion.

    Until it is implemented, a gamma cell is rejected explicitly.
    """
    raise NotImplementedError(
        f"cell (i={i}, l={l}) model='gamma': the modular gamma block is not "
        "implemented yet (Step-2 work in progress). Use a gamma-only input with "
        "the existing gamma backend for now."
    )


# ===========================================================================
# Rainflow cell = gating + state recursion + reliability + repeatability
# ===========================================================================
def _add_rainflow_cell(ctx: _RFModel, i: int, l: int) -> None:
    _add_maintenance_gating(ctx, i, l)
    _add_rainflow_state(ctx, i, l)
    _add_reliability(ctx, i, l)
    _add_repeatability(ctx, i, l)


def _add_maintenance_gating(ctx: _RFModel, i: int, l: int) -> None:
    """Maintenance gating (reference eq. 3): m, r each require a depot day;
    m + r <= 1; nb is the no-intervention indicator."""
    md, T = ctx.model, ctx.T
    x, m_rep, r_rep, nb = ctx.x, ctx.m_rep, ctx.r_rep, ctx.nb
    for k in range(T):
        md.addConstr(m_rep[i, l, k] <= x[i, 0, k], name=f"m_gate_{i}_{l}_{k}")
        if ctx.allow_replacement:
            md.addConstr(r_rep[i, l, k] <= x[i, 0, k], name=f"r_gate_{i}_{l}_{k}")
            md.addConstr(nb[i, l, k] == 1 - m_rep[i, l, k] - r_rep[i, l, k],
                         name=f"nb_def_{i}_{l}_{k}")
        else:
            md.addConstr(nb[i, l, k] == 1 - m_rep[i, l, k], name=f"nb_def_{i}_{l}_{k}")


def _add_rainflow_state(ctx: _RFModel, i: int, l: int) -> None:
    """Per-cell state recursion: mean, variance (if the bound uses it), the ARD1
    latch (if repair_model == ard1), removed-damage ``z`` (eq. 6), and the
    bound-specific accumulators R (Hoeffding) / K (Chernoff).

    NOTE: if you add a bound needing a new accumulator, add its recursion here
    (mirroring the R / K blocks) and register the descriptor in
    ``_BOUND_DESCRIPTORS``."""
    md, T, M = ctx.model, ctx.T, ctx.M
    x, nb, m_rep, r_rep = ctx.x, ctx.nb, ctx.m_rep, ctx.r_rep
    mu_var, v_var, gmu, gv, z_var = ctx.mu_var, ctx.v_var, ctx.gmu, ctx.gv, ctx.z_var
    R_var, K_var = ctx.R_var, ctx.K_var
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
            md.addGenConstrIndicator(m_rep[i, l, k], True, R_var[i, l, k] == k2 * R_prev,
                                     name=f"R_rep_{i}_{l}_{k}")
            if allow_rep:
                md.addGenConstrIndicator(r_rep[i, l, k], True, R_var[i, l, k] == 0.0,
                                         name=f"R_repl_{i}_{l}_{k}")
        if bound == "chernoff":
            K_prev = 0.0 if k == 0 else K_var[i, l, k - 1]
            cgf_expr = gp.quicksum(x[i, j, k] * ctx.cgf_inc(i, j - 1, l, k)
                                   for j in range(1, M + 1))
            md.addGenConstrIndicator(nb[i, l, k], True, K_var[i, l, k] == K_prev + cgf_expr,
                                     name=f"K_carry_{i}_{l}_{k}")
            md.addGenConstrIndicator(m_rep[i, l, k], True, K_var[i, l, k] == k1 * K_prev,
                                     name=f"K_rep_{i}_{l}_{k}")
            if allow_rep:
                md.addGenConstrIndicator(r_rep[i, l, k], True, K_var[i, l, k] == 0.0,
                                         name=f"K_repl_{i}_{l}_{k}")


def _add_repeatability(ctx: _RFModel, i: int, l: int) -> None:
    """Loop the operating horizon: end-of-operating (T-1) <= end-of-transitory
    (H1-1), for the moments and any bound-specific accumulator."""
    md, H1, T = ctx.model, ctx.H1, ctx.T
    bound = str(ctx.bound_of[i, l])
    md.addConstr(ctx.mu_var[i, l, T - 1] <= ctx.mu_var[i, l, H1 - 1], name=f"repeat_mu_{i}_{l}")
    if ctx.track_v_of[i, l]:
        md.addConstr(ctx.v_var[i, l, T - 1] <= ctx.v_var[i, l, H1 - 1], name=f"repeat_v_{i}_{l}")
    if bound == "hoeffding":
        md.addConstr(ctx.R_var[i, l, T - 1] <= ctx.R_var[i, l, H1 - 1], name=f"repeat_R_{i}_{l}")
    if bound == "chernoff":
        md.addConstr(ctx.K_var[i, l, T - 1] <= ctx.K_var[i, l, H1 - 1], name=f"repeat_K_{i}_{l}")


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
#                 halfspace. Safe INNER approximation (never accepts an unsafe
#                 (mu, Q)); conservative; no extra binaries. Linearization point
#                 is tangent_ref (fraction of tau).
#   * "pwl"     : piecewise tangent over `pwl_points` segments of mu in [0, tau],
#                 one tangent active per segment (segment chosen by binaries).
#                 Safe INNER approximation, accuracy grows with pwl_points, at the
#                 cost of extra binaries.
# Markov and Chernoff (fixed s) are already linear -> only "exact".
#
# ADD A NEW RELAXATION FAMILY (e.g. a secant OUTER bound, McCormick, SOC, an
# iterative-cut scheme) by writing a builder and registering it as another impl:
#     RELIABILITY_BOUNDS["cantelli"]["secant"] = _Impl(_rel_cantelli_secant,
#                                                       quadratic=False,
#                                                       validity="outer")
# `quadratic` drives the NonConvex=2 gate; `validity` documents the relationship
# to the exact feasible set ("exact" | "inner"=safe | "outer"=unsafe/diagnostic).
# A bound needing a NEW accumulator must also add its recursion in
# `_add_rainflow_state` and declare it in `_BOUND_DESCRIPTORS`.
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
    """One supporting tangent at mu_p = tangent_ref * tau, plus the mu<=tau gap."""
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
    """
    md, T = ctx.model, ctx.T
    bound = str(ctx.bound_of[i, l])
    tau = float(ctx.tau[i, l])
    c2, c1, Qvar = _cap_coeffs(ctx, i, l, bound)
    K = max(1, int(ctx.pwl_points))
    edges = np.linspace(0.0, tau, K + 1)          # breakpoints in mu

    for k in range(T):
        md.addConstr(ctx.mu_var[i, l, k] <= tau, name=f"rel_{i}_{l}_{k}_gap")

    if K == 1:                                     # single tangent, no binaries
        _add_tangent_cap(ctx, i, l, c2, c1, Qvar, 0.5 * tau, name="rel")
        return

    for k in range(T):
        mu = ctx.mu_var[i, l, k]
        Q = Qvar[i, l, k]
        zs = md.addVars(K, vtype=GRB.BINARY, name=f"relseg_{i}_{l}_{k}")
        md.addConstr(gp.quicksum(zs[s] for s in range(K)) == 1, name=f"relseg1_{i}_{l}_{k}")
        for s in range(K):
            lo, hi = float(edges[s]), float(edges[s + 1])
            mid = 0.5 * (lo + hi)
            d_p = tau - mid
            g_p = c2 * d_p * d_p + c1 * d_p
            gp_prime = -2.0 * c2 * d_p - c1
            # segment membership (exact, via indicators)
            md.addGenConstrIndicator(zs[s], True, mu >= lo, name=f"relslo_{i}_{l}_{k}_{s}")
            md.addGenConstrIndicator(zs[s], True, mu <= hi, name=f"relshi_{i}_{l}_{k}_{s}")
            # segment tangent cap (only when this segment is active)
            md.addGenConstrIndicator(zs[s], True, Q <= g_p + gp_prime * (mu - mid),
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
# Helpers
# ===========================================================================
def _pick(explicit, from_options, default):
    """First non-None of (explicit kwarg, cfg.options value, default)."""
    if explicit is not None:
        return explicit
    if from_options is not None:
        return from_options
    return default


def _cell_max(op, tr, i, l) -> float:
    """Max of a profile over cell (i, l), across operating (op) and optional
    transitory (tr) arrays, each shaped (F, L, M, H)."""
    m = float(op[i, l].max()) if op is not None else 0.0
    if tr is not None:
        m = max(m, float(tr[i, l].max()))
    return m


def _resolve_costs(cfg, tau, allow_replacement) -> dict:
    c = dict(cfg.costs)
    for key in ("C_M", "C_R", "C_S", "C_P"):
        if key not in c:
            raise KeyError(f"missing required cost coefficient '{key}'.")
    if "C_rep" not in c or c["C_rep"] is None:
        c["C_rep"] = float(c["C_R"]) * float(np.max(tau))     # default C_R * max(tau)
    return c


def _tighten_bounds(ctx: _RFModel, cfg) -> None:
    """Set valid, tight upper bounds per rainflow cell (strengthens the
    relaxation without changing the optimum). Unused-cell entries of a shared
    auxiliary variable are left free (they appear in no constraint/objective)."""
    T = ctx.T
    for i, l in ctx.rainflow_cells():
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
        if bound == "chernoff":
            K_ub = float(T * _cell_max(cfg.cgf, cfg.cgf_trans, i, l))
            for k in range(T):
                ctx.K_var[i, l, k].UB = K_ub


def _apply_performance_params(model, time_limit, mip_gap, fast, extra) -> None:
    """Presolve / heuristics tuning aimed at 'good feasible fast' on this hard
    nonconvex MIQCP. Everything here is overridable via ``gurobi_params``."""
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)
    if fast:
        model.Params.MIPFocus = 1
        model.Params.Heuristics = 0.5
        model.Params.ImproveStartGap = 0.5
        if time_limit is not None:
            model.Params.NoRelHeurTime = max(2.0, 0.15 * float(time_limit))
    if extra:
        for key, val in extra.items():
            model.setParam(key, val)


def _collapse_dict(impl_of, F, L, model_of):
    """Per-cell impl names -> a single string when all rainflow cells agree,
    else a nested (F, L) list ('' for non-rainflow cells)."""
    flat = set(impl_of.values())
    if not flat:
        return ""
    if len(flat) == 1:
        return next(iter(flat))
    grid = [["" for _ in range(L)] for _ in range(F)]
    for (i, l), name in impl_of.items():
        grid[i][l] = name
    return grid


def _collapse(arr):
    """A per-cell (F, L) selector array -> a single string when every cell
    agrees (the common uniform case), else a plain nested list of strings.
    Keeps the result dict free of NumPy objects so it serializes cleanly."""
    flat = {str(v) for v in np.asarray(arr).ravel()}
    if len(flat) == 1:
        return next(iter(flat))
    return np.asarray(arr).astype(str).tolist()


def _status_string(code: int) -> str:
    return {
        GRB.OPTIMAL: "optimal",
        GRB.TIME_LIMIT: "time_limit",
        GRB.SUBOPTIMAL: "suboptimal",
        GRB.INTERRUPTED: "interrupted",
        GRB.SOLUTION_LIMIT: "solution_limit",
        GRB.NODE_LIMIT: "node_limit",
        GRB.ITERATION_LIMIT: "iteration_limit",
        GRB.INFEASIBLE: "infeasible",
        GRB.INF_OR_UNBD: "inf_or_unbounded",
        GRB.UNBOUNDED: "unbounded",
    }.get(code, f"gurobi_status_{code}")


def _extract_solution(ctx: _RFModel, cfg, model) -> dict:
    F, L, M, T = ctx.F, ctx.L, ctx.M, ctx.T
    status = _status_string(model.status)
    meta = {
        "status": status,
        "method": _collapse(cfg.bound_method),
        "bound_method": _collapse(cfg.bound_method),
        "repair_model": _collapse(cfg.repair_model),
        "reliability_impl": _collapse_dict(ctx.impl_of, F, L, ctx.model_of),
        "models": cfg.models,
        "F": F, "H": cfg.H, "H1": ctx.H1, "H2": ctx.H2, "T": T, "M": M, "L": L,
        "tau": cfg.tau, "mu_0": cfg.mu_0, "v_0": cfg.v_0, "model": model,
    }
    if model.SolCount == 0:
        meta.update({"objective": None, "mip_gap": None, "bound": None,
                     "x": None, "mu": None, "v": None, "z": None,
                     "m": None, "r": None, "u": None})
        return meta

    try:
        gap = float(model.MIPGap)
    except (AttributeError, gp.GurobiError):
        gap = None
    try:
        objbnd = float(model.ObjBound)
    except (AttributeError, gp.GurobiError):
        objbnd = None

    x_sol = np.zeros((F, M + 1, T)); mu_sol = np.zeros((F, L, T))
    v_sol = np.zeros((F, L, T)); z_sol = np.zeros((F, L, T))
    m_sol = np.zeros((F, L, T)); r_sol = np.zeros((F, L, T)); u_sol = np.zeros(T)
    for k in range(T):
        u_sol[k] = ctx.u_var[k].X
        for i in range(F):
            for j in range(M + 1):
                x_sol[i, j, k] = ctx.x[i, j, k].X
            for l in range(L):
                mu_sol[i, l, k] = ctx.mu_var[i, l, k].X
                if ctx.v_var is not None and ctx.track_v_of[i, l]:
                    v_sol[i, l, k] = ctx.v_var[i, l, k].X
                z_sol[i, l, k] = ctx.z_var[i, l, k].X
                m_sol[i, l, k] = ctx.m_rep[i, l, k].X
                if ctx.allow_replacement:
                    r_sol[i, l, k] = ctx.r_rep[i, l, k].X
    meta.update({"objective": model.ObjVal, "mip_gap": gap, "bound": objbnd,
                 "x": x_sol, "mu": mu_sol, "v": v_sol, "z": z_sol,
                 "m": m_sol, "r": r_sol, "u": u_sol})
    return meta


# ===========================================================================
# Runnable demo
# ===========================================================================
if __name__ == "__main__":
    from fleet_management.config import load_config

    print("Fleet management (rainflow, modular) demo -- two horizons")
    base = {
        "F": 3, "M": 1, "model": "rainflow", "bound_method": "cantelli",
        "repair_model": "ard1", "tau": 0.30, "epsilon": 0.10,
        "rho": 0.6, "mu_0": 0.02, "v_0": 4e-4, "mu": 0.06, "v": 0.0015,
        "C_M": 1.0, "C_R": 0.5, "C_S": 2.0, "C_P": 1.0,
    }
    cfg = load_config({**base, "H": 4})
    res = solve(cfg, verbose=0, time_limit=30)
    print(f"equal   H1=H2=4  -> status={res['status']}, T={res['T']}, obj={res['objective']}")

    cfg2 = load_config({**base, "H": [6, 4]})
    res2 = solve(cfg2, verbose=0, time_limit=30)
    print(f"unequal H1=6,H2=4 -> status={res2['status']}, T={res2['T']}, obj={res2['objective']}")