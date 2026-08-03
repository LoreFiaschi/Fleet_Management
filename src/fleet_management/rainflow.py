"""
Fleet management with a rainflow / remaining-life (Palmgren-Miner) degradation
model, solved with Gurobi.

This is the rainflow / accumulated-damage counterpart of the Gaussian solver.
The accumulated Palmgren-Miner damage D of every component is tracked through its
first two moments (mean ``mu`` and variance ``v``); some reliability bounds also
carry an extra additive descriptor (Hoeffding's squared-support sum, Chernoff's
CGF).  The reliability requirement ``P(D > tau) <= eps`` is enforced with a
distribution-free concentration bound chosen via ``method`` (reference doc, Sec.
2.1.4 / slide 35); repeatability loops the *moments*, not the bound (Sec. 2.1.5 /
slide 36).

Structure
---------
``solve_fleet_management`` wires together the problem: it validates inputs, builds
the variables, and then delegates each block of the formulation to a dedicated
helper that receives a shared ``_RFModel`` context:

    _build_objective            objective (maintenance / repair / replace / safety / loop)
    _add_base_constraints       assignment, depot capacity, aggregate cap, safety u
    _add_maintenance_constraints gating (eq. 3) + state recursion / ARD1 / replace / z
    _add_reliability_constraints per-step  P(D > tau) <= eps  (method-dependent)
    _add_repeatability_constraints  loop the moments (+ descriptors)  H vs 2H

Maintenance formulation (reference doc, Sec. 1.1-1.2, 2.3):

  * Maintenance is decided **per component**.  On a depot day (``x[i,0,k] = 1``)
    each component ``l`` independently chooses imperfect repair ``m[i,l,k]``, full
    replacement ``r[i,l,k]``, or no intervention, with ``m <= x[i,0,k]``,
    ``r <= x[i,0,k]``, ``m + r <= 1`` (eq. 3).

  * The repair operator is **ARD1** by default (eq. 127): only the damage
    accumulated *since the previous maintenance epoch* is partially reversed.
    ``ARD-inf`` (``D+ = (1 - rho) D-``) is available via ``repair_model``.

  * ``z`` is the removed expected damage ``E[D-] - E[D+]`` on a maintenance action
    (eq. 6); the variance keeps a ``(1 - rho)^2`` fraction under repair.

  * Case logic uses Gurobi **indicator constraints**, so the dynamics are exact.

Author: Johann Tschan  (revised)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import gurobipy as gp
from gurobipy import GRB


# Reliability methods and the extra data each one needs.
_METHODS = ("markov", "cantelli", "hoeffding", "bernstein", "chernoff")
_QUADRATIC = ("cantelli", "hoeffding", "bernstein")   # need NonConvex = 2
_NEEDS_SUPPORT = ("hoeffding", "bernstein")
_NEEDS_CGF = ("chernoff",)
_REPAIR_MODELS = ("ard1", "ardinf")


# ---------------------------------------------------------------------------
# Shared context: model, variables, parameters and increment accessors.
# ---------------------------------------------------------------------------
@dataclass
class _RFModel:
    """Everything the constraint helpers need, built once and passed around."""
    model: gp.Model
    F: int; H: int; M: int; L: int; T: int
    # decision variables / states
    x: gp.tupledict
    m_rep: gp.tupledict
    r_rep: Optional[gp.tupledict]          # None when replacement disabled
    nb: gp.tupledict
    mu_var: gp.tupledict
    v_var: Optional[gp.tupledict]          # None when the method ignores variance
    gmu: Optional[gp.tupledict]            # None for ardinf (no latch needed)
    gv: Optional[gp.tupledict]
    z_var: gp.tupledict
    u_var: gp.tupledict
    R_var: Optional[gp.tupledict]
    K_var: Optional[gp.tupledict]
    # parameters
    mu_0: np.ndarray
    v_0: np.ndarray
    rho: np.ndarray
    mu_new: np.ndarray
    v_new: np.ndarray
    tau: float
    epsilon: float
    method: str
    repair_model: str
    s_chernoff: Optional[float]
    support_param: Optional[np.ndarray]
    Le: float
    ln_eps: float
    # build-configuration flags
    track_v: bool                          # variance participates in the model
    use_latch: bool                        # ARD1 latch is present
    allow_replacement: bool
    # per-mission increment accessors (k wraps with period H)
    mu_inc: Callable[[int, int, int, int], float]
    v_inc: Callable[[int, int, int, int], float]
    w2_inc: Callable[[int, int, int, int], float]
    cgf_inc: Callable[[int, int, int, int], float]


def validate_inputs(
    F, H, M, L, mu_param, v_param, tau, epsilon, xi,
    C_M, C_R, C_S, C_P, mu_0, v_0, method,
    support_param=None, cgf_param=None, s_chernoff=None,
    repair_model="ard1", C_rep=None, mu_new=None, v_new=None,
) -> None:
    if F <= 0 or H <= 0 or M <= 0 or L <= 0:
        raise ValueError("F, H, M, L must be positive integers.")
    if tau <= 0:
        raise ValueError("tau (damage threshold) must be positive.")
    if not (0.0 < epsilon < 1.0):
        raise ValueError(f"epsilon must be in (0, 1) (got {epsilon}).")
    if C_M <= 0 or C_R <= 0 or C_S <= 0 or C_P <= 0:
        raise ValueError("All cost coefficients must be positive.")
    if C_rep is not None and C_rep < 0:
        raise ValueError("C_rep must be non-negative.")
    if F <= M:
        raise ValueError(f"F must be greater than M (got F={F}, M={M}).")

    if mu_param.shape != (F, M, L, H):
        raise ValueError(f"mu_param shape must be {(F, M, L, H)}, got {mu_param.shape}.")
    if v_param.shape != (F, M, L, H):
        raise ValueError(f"v_param shape must be {(F, M, L, H)}, got {v_param.shape}.")

    if xi.shape != (F, L):
        raise ValueError(f"xi must have shape {(F, L)}.")
    if not np.all(xi > 0) or not np.all(xi <= 1):
        raise ValueError("xi (repair efficiency rho) must be in (0, 1] element-wise.")

    if mu_0.shape != (F, L) or v_0.shape != (F, L):
        raise ValueError(f"mu_0 and v_0 must have shape {(F, L)}.")

    if not np.all(mu_param > 0):
        raise ValueError("All entries of mu_param must be positive.")
    if not np.all(v_param > 0):
        raise ValueError("All entries of v_param must be positive.")
    if not np.all(mu_0 >= 0) or not np.all(v_0 >= 0):
        raise ValueError("mu_0 and v_0 must be >= 0 element-wise.")

    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS} (got '{method}').")
    if repair_model not in _REPAIR_MODELS:
        raise ValueError(f"repair_model must be one of {_REPAIR_MODELS}.")

    if mu_new is not None and (np.asarray(mu_new).shape != (F, L)):
        raise ValueError(f"mu_new must have shape {(F, L)}.")
    if v_new is not None and (np.asarray(v_new).shape != (F, L)):
        raise ValueError(f"v_new must have shape {(F, L)}.")

    if method in _NEEDS_SUPPORT:
        if support_param is None:
            raise ValueError(f"method='{method}' requires support_param.")
        if support_param.shape != (F, M, L, H):
            raise ValueError(f"support_param shape must be {(F, M, L, H)}.")
        if not np.all(support_param > 0):
            raise ValueError("support_param must be positive element-wise.")

    if method in _NEEDS_CGF:
        if cgf_param is None or s_chernoff is None:
            raise ValueError("method='chernoff' requires cgf_param and s_chernoff > 0.")
        if cgf_param.shape != (F, M, L, H):
            raise ValueError(f"cgf_param shape must be {(F, M, L, H)}.")
        if not np.all(cgf_param > 0):
            raise ValueError("cgf_param must be positive element-wise.")
        if s_chernoff <= 0:
            raise ValueError("s_chernoff must be positive.")


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
def _build_objective(ctx: _RFModel, C_M, C_R, C_S, C_P, C_rep) -> None:
    """C_M per depot-day, C_R per unit removed by repair, C_rep per replacement,
    C_S on worst aggregate damage, C_P periodicity slack."""
    md, F, L, T, H = ctx.model, ctx.F, ctx.L, ctx.T, ctx.H
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
            obj += C_P * (ctx.mu_var[i, l, H - 1] - ctx.mu_var[i, l, T - 1])
            if ctx.track_v:
                obj += C_P * (ctx.v_var[i, l, H - 1] - ctx.v_var[i, l, T - 1])
    md.setObjective(obj, GRB.MINIMIZE)


# ---------------------------------------------------------------------------
# Base constraints: assignment, depot capacity, aggregate cap, safety u.
# ---------------------------------------------------------------------------
def _add_base_constraints(ctx: _RFModel, depot_capacity: int) -> None:
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
        md.addConstr(
            gp.quicksum(mu_var[i, l, k] for i in range(F) for l in range(L)) <= F - M,
            name=f"capacity_{k}")
    for k in range(T):
        for i in range(F):
            md.addConstr(u_var[k] >= gp.quicksum(mu_var[i, l, k] for l in range(L)),
                         name=f"u_{i}_{k}")


# ---------------------------------------------------------------------------
# Maintenance constraints: gating + per-component state recursion.
# ---------------------------------------------------------------------------
def _add_maintenance_constraints(ctx: _RFModel) -> None:
    md, F, L, T = ctx.model, ctx.F, ctx.L, ctx.T
    x, m_rep, r_rep, nb = ctx.x, ctx.m_rep, ctx.r_rep, ctx.nb
    mu_var, v_var, gmu, gv, z_var = ctx.mu_var, ctx.v_var, ctx.gmu, ctx.gv, ctx.z_var
    R_var, K_var = ctx.R_var, ctx.K_var
    method, repair_model = ctx.method, ctx.repair_model
    track_v, use_latch, allow_rep = ctx.track_v, ctx.use_latch, ctx.allow_replacement

    # ---- maintenance gating (reference eq. 3) ---------------------------
    for i in range(F):
        for l in range(L):
            for k in range(T):
                md.addConstr(m_rep[i, l, k] <= x[i, 0, k], name=f"m_gate_{i}_{l}_{k}")
                if allow_rep:
                    md.addConstr(r_rep[i, l, k] <= x[i, 0, k], name=f"r_gate_{i}_{l}_{k}")
                    md.addConstr(nb[i, l, k] == 1 - m_rep[i, l, k] - r_rep[i, l, k],
                                 name=f"nb_def_{i}_{l}_{k}")
                else:
                    md.addConstr(nb[i, l, k] == 1 - m_rep[i, l, k], name=f"nb_def_{i}_{l}_{k}")

    one_minus_rho = 1.0 - ctx.rho
    var_keep = one_minus_rho ** 2

    for i in range(F):
        for l in range(L):
            r_il = float(ctx.rho[i, l])
            k1 = float(one_minus_rho[i, l])       # (1 - rho)
            k2 = float(var_keep[i, l])            # (1 - rho)^2
            for k in range(T):
                mu_prev = ctx.mu_0[i, l] if k == 0 else mu_var[i, l, k - 1]
                mean_inc = gp.quicksum(x[i, j, k] * ctx.mu_inc(i, j - 1, l, k)
                                       for j in range(1, ctx.M + 1))

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

                # ----- variance recursion (only if the method uses variance) -----
                if track_v:
                    v_prev = ctx.v_0[i, l] if k == 0 else v_var[i, l, k - 1]
                    var_inc = gp.quicksum(x[i, j, k] * ctx.v_inc(i, j - 1, l, k)
                                          for j in range(1, ctx.M + 1))
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
                if method == "hoeffding":
                    R_prev = 0.0 if k == 0 else R_var[i, l, k - 1]
                    w2_expr = gp.quicksum(x[i, j, k] * ctx.w2_inc(i, j - 1, l, k)
                                          for j in range(1, ctx.M + 1))
                    md.addGenConstrIndicator(nb[i, l, k], True, R_var[i, l, k] == R_prev + w2_expr,
                                             name=f"R_carry_{i}_{l}_{k}")
                    md.addGenConstrIndicator(m_rep[i, l, k], True, R_var[i, l, k] == k2 * R_prev,
                                             name=f"R_rep_{i}_{l}_{k}")
                    if allow_rep:
                        md.addGenConstrIndicator(r_rep[i, l, k], True, R_var[i, l, k] == 0.0,
                                                 name=f"R_repl_{i}_{l}_{k}")
                if method == "chernoff":
                    K_prev = 0.0 if k == 0 else K_var[i, l, k - 1]
                    cgf_expr = gp.quicksum(x[i, j, k] * ctx.cgf_inc(i, j - 1, l, k)
                                           for j in range(1, ctx.M + 1))
                    md.addGenConstrIndicator(nb[i, l, k], True, K_var[i, l, k] == K_prev + cgf_expr,
                                             name=f"K_carry_{i}_{l}_{k}")
                    md.addGenConstrIndicator(m_rep[i, l, k], True, K_var[i, l, k] == k1 * K_prev,
                                             name=f"K_rep_{i}_{l}_{k}")
                    if allow_rep:
                        md.addGenConstrIndicator(r_rep[i, l, k], True, K_var[i, l, k] == 0.0,
                                                 name=f"K_repl_{i}_{l}_{k}")


# ---------------------------------------------------------------------------
# Reliability constraints: per-step  P(D > tau) <= eps.
# ---------------------------------------------------------------------------
def _add_reliability_constraints(ctx: _RFModel) -> None:
    md, F, L, T = ctx.model, ctx.F, ctx.L, ctx.T
    mu_var, v_var, R_var, K_var = ctx.mu_var, ctx.v_var, ctx.R_var, ctx.K_var
    tau, eps, method = ctx.tau, ctx.epsilon, ctx.method
    Le, ln_eps = ctx.Le, ctx.ln_eps

    for i in range(F):
        for l in range(L):
            for k in range(T):
                mu_ik = mu_var[i, l, k]
                rname = f"rel_{i}_{l}_{k}"
                if method == "markov":
                    md.addConstr(mu_ik <= eps * tau, name=rname)
                elif method == "cantelli":
                    md.addConstr(mu_ik <= tau, name=f"{rname}_gap")
                    md.addQConstr((1.0 - eps) * v_var[i, l, k] <= eps * (tau - mu_ik) * (tau - mu_ik),
                                  name=rname)
                elif method == "hoeffding":
                    md.addConstr(mu_ik <= tau, name=f"{rname}_gap")
                    md.addQConstr((tau - mu_ik) * (tau - mu_ik) >= 0.5 * Le * R_var[i, l, k],
                                  name=rname)
                elif method == "bernstein":
                    b = float(ctx.support_param.max())
                    t = tau - mu_ik
                    md.addConstr(mu_ik <= tau, name=f"{rname}_gap")
                    md.addQConstr(0.5 * t * t - (Le * b / 3.0) * t - Le * v_var[i, l, k] >= 0, name=rname)
                elif method == "chernoff":
                    md.addConstr(K_var[i, l, k] - ctx.s_chernoff * tau <= ln_eps, name=rname)


# ---------------------------------------------------------------------------
# Repeatability constraints: loop the moments (+ descriptors), H vs 2H.
# ---------------------------------------------------------------------------
def _add_repeatability_constraints(ctx: _RFModel) -> None:
    md, F, L, H, T = ctx.model, ctx.F, ctx.L, ctx.H, ctx.T
    mu_var, v_var, R_var, K_var = ctx.mu_var, ctx.v_var, ctx.R_var, ctx.K_var
    method = ctx.method

    for i in range(F):
        for l in range(L):
            md.addConstr(mu_var[i, l, T - 1] <= mu_var[i, l, H - 1], name=f"repeat_mu_{i}_{l}")
            if ctx.track_v:
                md.addConstr(v_var[i, l, T - 1] <= v_var[i, l, H - 1], name=f"repeat_v_{i}_{l}")
            if method == "hoeffding":
                md.addConstr(R_var[i, l, T - 1] <= R_var[i, l, H - 1], name=f"repeat_R_{i}_{l}")
            if method == "chernoff":
                md.addConstr(K_var[i, l, T - 1] <= K_var[i, l, H - 1], name=f"repeat_K_{i}_{l}")


# ---------------------------------------------------------------------------
# Performance parameters
# ---------------------------------------------------------------------------
def _apply_performance_params(model, time_limit, mip_gap, fast, extra) -> None:
    """Presolve / heuristics tuning aimed at 'good feasible fast' on this hard
    nonconvex MIQCP.  Everything here is overridable via `gurobi_params`."""
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)
    if fast:
        model.Params.MIPFocus = 1          # prioritise finding good incumbents
        model.Params.Heuristics = 0.5      # spend more effort in heuristics
        model.Params.ImproveStartGap = 0.5 # switch to incumbent-improving early
        if time_limit is not None:
            # run the no-relaxation heuristic first; excellent on hard MIPs.
            model.Params.NoRelHeurTime = max(2.0, 0.15 * float(time_limit))
    if extra:
        for key, val in extra.items():
            model.setParam(key, val)


def _status_string(code: int) -> str:
    """Human-readable Gurobi optimisation status."""
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


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
def solve_fleet_management(
    F, H, M, L,
    mu_param, v_param, tau, epsilon, xi,
    C_M, C_R, C_S, C_P, mu_0, v_0,
    method="cantelli",
    support_param=None, cgf_param=None, s_chernoff=None,
    repair_model="ard1",
    C_rep=None,
    mu_new=None, v_new=None,
    depot_capacity=None,
    allow_replacement=True,
    fast=True,
    gurobi_params=None,
    verbose=1, mip_gap=0.12, time_limit=None,
) -> dict:
    """
    Solve the rainflow (accumulated-damage) fleet-management problem.

    Performance-related parameters
    ------------------------------
    allow_replacement : bool, default True
        If False, replacement is dropped: the ``r`` binaries and every associated
        indicator constraint disappear (~5 fewer indicators per component-step),
        leaving imperfect repair as the only maintenance action.  Use False when
        components are never replaced -- it makes the model noticeably smaller.
    fast : bool, default True
        Apply a Gurobi tuning preset (MIPFocus=1, more heuristics, and the
        no-relaxation heuristic when a time limit is set) aimed at finding good
        feasible schedules quickly.  Set False for Gurobi defaults.
    gurobi_params : dict, optional
        Extra Gurobi parameters, applied last so they override the preset, e.g.
        ``{"Threads": 8, "MIPGap": 0.05, "Symmetry": 2}``.
    time_limit : float, optional
        Wall-clock limit (seconds).  A feasible incumbent found before the limit
        is returned with ``status="time_limit"`` and a non-``None`` ``mip_gap``.

    The variance state and the ARD1 latch are only built when the chosen
    ``method`` / ``repair_model`` actually use them, and all continuous states are
    given tight upper bounds -- both shrink the search and strengthen the
    relaxation without changing the optimum.

    ``xi`` is the per-component repair efficiency ``rho`` in (0, 1].  Returns a
    dict; solution arrays are populated whenever a feasible incumbent exists.
    """
    validate_inputs(F, H, M, L, mu_param, v_param, tau, epsilon, xi,
                    C_M, C_R, C_S, C_P, mu_0, v_0, method,
                    support_param, cgf_param, s_chernoff,
                    repair_model, C_rep, mu_new, v_new)

    # ---- defaults -------------------------------------------------------
    rho = xi
    if C_rep is None:
        C_rep = C_R * tau
    if mu_new is None:
        mu_new = np.zeros((F, L))
    if v_new is None:
        v_new = np.zeros((F, L))
    if depot_capacity is None:
        depot_capacity = F - M

    Le = math.log(1.0 / epsilon)
    ln_eps = math.log(epsilon)
    T = 2 * H

    # ---- build configuration -------------------------------------------
    track_v = method in ("cantelli", "bernstein")   # only these use variance
    use_latch = (repair_model == "ard1")            # ARD-inf needs no D_last latch

    # ---- tight, valid variable bounds (strengthen the relaxation) -------
    mu_ub = float(epsilon * tau) if method == "markov" else float(tau)
    z_ub = float(tau)
    v_reach = float(v_0.max() + T * v_param.max())
    if method == "cantelli":
        v_ub = min(v_reach, float(epsilon / (1.0 - epsilon) * tau * tau))
    else:
        v_ub = v_reach
    R_ub = float(T * (support_param.max() ** 2)) if method == "hoeffding" else None
    K_ub = float(T * cgf_param.max()) if method == "chernoff" else None

    def mu_inc(i, j, l, k):  return float(mu_param[i, j, l, k % H])
    def v_inc(i, j, l, k):   return float(v_param[i, j, l, k % H])
    def w2_inc(i, j, l, k):  return float(support_param[i, j, l, k % H] ** 2)
    def cgf_inc(i, j, l, k): return float(cgf_param[i, j, l, k % H])

    # ---- model + variables ----------------------------------------------
    model = gp.Model("fleet_management_rainflow")
    model.Params.OutputFlag = int(verbose)
    if method in _QUADRATIC:
        model.Params.NonConvex = 2
    _apply_performance_params(model, time_limit, mip_gap, fast, gurobi_params)

    x = model.addVars(F, M + 1, T, vtype=GRB.BINARY, name="x")
    m_rep = model.addVars(F, L, T, vtype=GRB.BINARY, name="m")
    r_rep = model.addVars(F, L, T, vtype=GRB.BINARY, name="r") if allow_replacement else None
    nb = model.addVars(F, L, T, vtype=GRB.BINARY, name="nb")

    mu_var = model.addVars(F, L, T, lb=0.0, ub=mu_ub, name="mu")
    v_var = model.addVars(F, L, T, lb=0.0, ub=v_ub, name="v") if track_v else None
    gmu = model.addVars(F, L, T, lb=0.0, ub=mu_ub, name="gmu") if use_latch else None
    gv = model.addVars(F, L, T, lb=0.0, ub=v_ub, name="gv") if (use_latch and track_v) else None
    z_var = model.addVars(F, L, T, lb=0.0, ub=z_ub, name="z")
    u_var = model.addVars(T, lb=0.0, name="u")

    R_var = model.addVars(F, L, T, lb=0.0, ub=R_ub, name="R") if method == "hoeffding" else None
    K_var = model.addVars(F, L, T, lb=0.0, ub=K_ub, name="K") if method == "chernoff" else None

    ctx = _RFModel(
        model=model, F=F, H=H, M=M, L=L, T=T,
        x=x, m_rep=m_rep, r_rep=r_rep, nb=nb,
        mu_var=mu_var, v_var=v_var, gmu=gmu, gv=gv, z_var=z_var, u_var=u_var,
        R_var=R_var, K_var=K_var,
        mu_0=mu_0, v_0=v_0, rho=rho, mu_new=mu_new, v_new=v_new,
        tau=tau, epsilon=epsilon, method=method, repair_model=repair_model,
        s_chernoff=s_chernoff, support_param=support_param,
        Le=Le, ln_eps=ln_eps,
        track_v=track_v, use_latch=use_latch, allow_replacement=allow_replacement,
        mu_inc=mu_inc, v_inc=v_inc, w2_inc=w2_inc, cgf_inc=cgf_inc,
    )

    # ---- assemble the program from its blocks ---------------------------
    _build_objective(ctx, C_M, C_R, C_S, C_P, C_rep)
    _add_base_constraints(ctx, depot_capacity)
    _add_maintenance_constraints(ctx)
    _add_reliability_constraints(ctx)
    _add_repeatability_constraints(ctx)

    # ---- solve ----------------------------------------------------------
    model.optimize()
    status_str = _status_string(model.status)

    if model.SolCount > 0:
        try:
            gap = float(model.MIPGap)
        except (AttributeError, gp.GurobiError):
            gap = None
        try:
            bound = float(model.ObjBound)
        except (AttributeError, gp.GurobiError):
            bound = None

        x_sol = np.zeros((F, M + 1, T))
        mu_sol = np.zeros((F, L, T))
        v_sol = np.zeros((F, L, T))
        z_sol = np.zeros((F, L, T))
        m_sol = np.zeros((F, L, T))
        r_sol = np.zeros((F, L, T))
        u_sol = np.zeros(T)
        for k in range(T):
            u_sol[k] = u_var[k].X
            for i in range(F):
                for j in range(M + 1):
                    x_sol[i, j, k] = x[i, j, k].X
                for l in range(L):
                    mu_sol[i, l, k] = mu_var[i, l, k].X
                    if track_v:
                        v_sol[i, l, k] = v_var[i, l, k].X
                    z_sol[i, l, k] = z_var[i, l, k].X
                    m_sol[i, l, k] = m_rep[i, l, k].X
                    if allow_replacement:
                        r_sol[i, l, k] = r_rep[i, l, k].X
        return {
            "status": status_str, "objective": model.ObjVal,
            "mip_gap": gap, "bound": bound,
            "method": method, "repair_model": repair_model,
            "F": F, "H": H, "M": M, "L": L, "tau": tau,
            "x": x_sol, "mu": mu_sol, "v": v_sol, "z": z_sol,
            "m": m_sol, "r": r_sol, "u": u_sol, "model": model,
        }

    return {
        "status": status_str, "objective": None,
        "mip_gap": None, "bound": None,
        "method": method, "repair_model": repair_model,
        "F": F, "H": H, "M": M, "L": L, "tau": tau,
        "x": None, "mu": None, "v": None, "z": None,
        "m": None, "r": None, "u": None, "model": model,
    }


# ---------------------------------------------------------------------------
# Runnable demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Fleet management (rainflow, ARD1 maintenance) demo")
    F, H, M, L = 3, 4, 1, 1
    tau, epsilon = 0.30, 0.10
    mu_param = np.full((F, M, L, H), 0.06)
    v_param = np.full((F, M, L, H), 0.0015)
    xi = np.full((F, L), 0.6)
    mu_0 = np.full((F, L), 0.02)
    v_0 = np.full((F, L), 4e-4)
    C_M, C_R, C_S, C_P = 1.0, 0.5, 2.0, 1.0

    res = solve_fleet_management(
        F, H, M, L, mu_param, v_param, tau, epsilon, xi,
        C_M, C_R, C_S, C_P, mu_0, v_0,
        method="cantelli", repair_model="ard1", verbose=0, time_limit=30)
    print("status   :", res["status"])
    print("objective:", res["objective"])
    print("mip_gap  :", res["mip_gap"])