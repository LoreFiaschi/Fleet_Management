"""Remaining-life (rainflow/Palmgren--Miner) component formulation.

The shared fleet skeleton, model registry and Gamma component builder live in
``degradation_model.base``. This module contains only the remaining-life cell
state, maintenance, reliability and repeatability constraints, then registers
``RainflowCellBuilder`` under the model name ``"rainflow"``.

To add a reliability bound, implement its builder here, register it in
``RELIABILITY_BOUNDS`` and declare any additional state descriptor beside the
existing descriptor definitions. Operating profiles cover ``H2`` and optional
initialization profiles cover ``H1``; the shared accessor handles their time
indexing.
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


# ===========================================================================
# Entry point
# ===========================================================================
def solve(cfg, *, allow_replacement=None, depot_capacity=None,
          verbose=None, mip_gap=None, time_limit=None, fast=None,
          gurobi_params=None,
          reliability_impl=None, pwl_points=None, tangent_ref=None) -> dict:
    """Solve a fleet from a normalized ``FleetConfig``.

    The shared skeleton (variables, general constraints, objective) comes from
    ``base``; this entry point simply drives it. Run-time options default to the
    values in ``cfg.options`` when not passed explicitly. The public solver uses
    this wrapper only for uniform remaining-life fleets; mixed fleets are driven
    directly by ``base.solve_mixed``.
    """
    opts = resolve_run_options(
        cfg,
        allow_replacement=allow_replacement, depot_capacity=depot_capacity,
        verbose=verbose, mip_gap=mip_gap, time_limit=time_limit, fast=fast,
        gurobi_params=gurobi_params, reliability_impl=reliability_impl,
        pwl_points=pwl_points, tangent_ref=tangent_ref,
    )
    ctx = build_fleet(cfg, opts, model_name="fleet_management_rainflow_modular")
    ctx.model.optimize()
    return _extract_solution(ctx, cfg, ctx.model)


# ===========================================================================
# Preparation hook: rainflow-specific variables, per-cell arrays, solver flags
# ===========================================================================
def prepare(ctx: _RFModel, cfg, cells, opts: dict) -> None:
    """Create everything the rainflow cells need on top of the shared skeleton.

    Called once by ``base.build_fleet`` with the list of rainflow ``cells``.
    Fills the model-specific slots of the shared context: per-cell selector /
    parameter arrays, the auxiliary state variables (variance, ARD1 latch, the
    Hoeffding ``R`` and Chernoff ``K`` accumulators), the resolved reliability
    implementation per cell, and the tightened variable bounds. Sets
    ``NonConvex=2`` only if some cell keeps an exact quadratic encoding.
    """
    F, L, T = ctx.F, ctx.L, ctx.T
    md = ctx.model
    cells = list(cells)
    if not cells:
        return

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

    _tighten_bounds(ctx, cfg, cells)


def extract(ctx: _RFModel, cfg, out: dict) -> None:
    """Model-specific result fields (the shared arrays are handled by base)."""
    return None


# ===========================================================================
# Rainflow cell = gating + state recursion + reliability
# ===========================================================================
def _add_rainflow_cell(ctx: _RFModel, i: int, l: int) -> None:
    add_maintenance_gating(ctx, i, l)        # shared (base)
    _add_rainflow_state(ctx, i, l)
    _add_reliability(ctx, i, l)
    _add_repeatability(ctx, i, l)


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
# ############  REPEATABILITY CONSTRAINT  -- the loop condition  #############
# ===========================================================================
# Reliability keeps P(D > tau) <= eps *inside* the window; it does not stop the
# schedule from leaving the fleet more damaged than it found it. Repeatability
# closes that gap: the operating block must be executable indefinitely.
#
# The condition is imposed on the DESCRIPTOR, not on the certificate U. U is
# many-to-one in the descriptor, so U_T <= U_H1 can hold at two different
# descriptor values that one further block maps to different certificates --
# the certified risk could then creep up over repetitions. The descriptor
# version is both sufficient for U and closed under the dynamics, so it chains:
# every branch (no-intervention, ARD1, ARDinf, replacement) is componentwise
# non-decreasing, hence theta_{H1 + m*H2} is non-increasing in m.
#
# Which components are compared depends on what the cell actually tracks; see
# _BOUND_DESCRIPTORS / _TRACK_V. Since _TRACK_V and _BOUND_DESCRIPTORS are
# disjoint, v and R never co-occur, and chernoff+ard1 is rejected upstream
# (_ARD1_UNSUPPORTED), so the worst case is {mu, v, gmu, gv} or
# {mu, R, gmu, gR} -- four rows, matching the 2*n_theta*F*L of the write-up.


def _add_repeatability(ctx: _RFModel, i: int, l: int) -> None:
    """Loop condition: the descriptor at the end of the operating phase must not
    exceed its value at the end of the transitory phase, componentwise.

    Adds one to four linear rows for cell ``(i, l)``. No new variables and no
    new binaries -- every quantity compared is an existing continuous state
    variable -- so the model class is unchanged.

    Index mapping: the write-up is 1-based over ``k = 1..H`` with the transitory
    phase ``k = 1..H1``; this module is 0-based, so ``theta_H1`` is stored at
    index ``H1 - 1`` and ``theta_H`` at index ``T - 1``. Getting this off by one
    yields a feasible but wrong model, so it is spelled out here.
    """
    md, kS, kE = ctx.model, ctx.H1 - 1, ctx.T - 1   # end transitory / end operating
    # Defensive only: config.py already enforces H1 > 0 and H2 > 0, so
    # kE - kS == H2 >= 1 and this never fires.
    if kE <= kS:
        return
    # Moment / accumulator part of the descriptor. Each guard mirrors exactly
    # the creation guard in prepare(), so the variable is always present.
    b, pairs = str(ctx.bound_of[i, l]), [("mu", ctx.mu_var)]
    if ctx.track_v_of[i, l]:                         # cantelli, bernstein
        pairs.append(("v", ctx.v_var))
    if b == "hoeffding":
        pairs.append(("R", ctx.R_var))
    if b == "chernoff":
        pairs.append(("K", ctx.K_var))
    # ARD1 epoch state: under ARD1 the moments alone do not determine the next
    # block's trajectory, so the latched state must be carried too. Absent
    # under ARDinf, where latch_of is False and gmu/gv/gR are never created.
    if ctx.latch_of[i, l]:
        pairs.append(("gmu", ctx.gmu))
        if ctx.track_v_of[i, l]:
            pairs.append(("gv", ctx.gv))
        if b == "hoeffding":
            pairs.append(("gR", ctx.gR))
    for nm, var in pairs:
        md.addConstr(var[i, l, kE] <= var[i, l, kS], name=f"loop_{nm}_{i}_{l}")

# ===========================================================================
# Bound tightening
# ===========================================================================
def _tighten_bounds(ctx: _RFModel, cfg, cells) -> None:
    """Set valid, tight upper bounds per rainflow cell (strengthens the
    relaxation without changing the optimum). Unused-cell entries of a shared
    auxiliary variable are left free (they appear in no constraint/objective)."""
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
    """Rainflow implementation of the ``base.CellBuilder`` interface."""

    name = "rainflow"

    prepare = staticmethod(prepare)
    add_cell = staticmethod(_add_rainflow_cell)
    extract = staticmethod(extract)


register_cell_builder("rainflow", RainflowCellBuilder())


# ===========================================================================
# Runnable demo
# ===========================================================================
if __name__ == "__main__":
    from fleet_management.config import load_config

    print("Fleet management (rainflow, modular on base.py) demo")
    base_input = {
        "F": 3, "M": 1, "model": "rainflow", "bound_method": "cantelli",
        "repair_model": "ard1", "tau": 0.30, "epsilon": 0.10,
        "rho": 0.6, "mu_0": 0.02, "v_0": 4e-4, "mu": 0.06, "v": 0.0015,
        "C_M": 1.0, "C_R": 0.5, "C_D": 2.0,
    }
    res = solve(load_config({**base_input, "H": 4}), verbose=0, time_limit=30)
    print(f"  equal   H1=H2=4   -> status={res['status']} T={res['T']} obj={res['objective']}")
    res2 = solve(load_config({**base_input, "H": [6, 4]}), verbose=0, time_limit=30)
    print(f"  unequal H1=6,H2=4 -> status={res2['status']} T={res2['T']} obj={res2['objective']}")
