"""Gurobi backend for common-rate Gamma degradation.

This backend is intentionally isolated from the Gaussian implementation.
It stores the Gamma shape parameter ``A`` as the optimization state and uses
the shape-rate convention

    D ~ Gamma(A, beta)
    E[D] = A / beta.

For each component, ``beta`` is constant over the complete horizon. Mission
increments therefore accumulate exactly by adding their shape parameters.

Supported decisions
-------------------
Idle
    The shape remains unchanged.
Mission
    ``A_k = A_{k-1} + beta * mu_increment``.
Imperfect repair
    ``A_k = (1-rho) * A_{k-1}``, a mean-matched common-beta approximation.
Replacement
    ``A_k = beta * replacement_mu``.

``x[i, 0, k]`` grants maintenance access.  Separate component-level binary
variables ``m[i,l,k]`` and ``r[i,l,k]`` select imperfect repair and full
replacement.  The exact scaled-repair distribution is deliberately left to
the independent offline validator.
"""

# After Midterm 28.07.2026, required for degradation/gamma.py

from __future__ import annotations

import gurobipy as gp
import numpy as np
import time                             # performance measurement
from gurobipy import GRB
from scipy.stats import gamma

from fleet_management.degradation_model.gamma import maximum_reliable_shape


def validate_inputs(
    F: int,
    H: int,
    M: int,
    L: int,
    mu_param: np.ndarray,
    tau: np.ndarray,
    epsilon: float,
    gamma_beta: np.ndarray,
    C_M: float,
    C_R: float,
    C_rep: float,
    C_S: float,
    C_P: float,
    mu_0: np.ndarray,
    replacement_mu: np.ndarray,
    repair_rho: np.ndarray,
) -> None:
    """Validate the common-rate Gamma solver contract."""

    if F <= 0 or H <= 0 or M <= 0 or L <= 0:
        raise ValueError("F, H, M and L must be positive integers.")
    if F <= M:
        raise ValueError(f"F must be greater than M (got F={F}, M={M}).")
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must lie strictly between zero and one.")
    if any(cost <= 0.0 for cost in (C_M, C_R, C_rep, C_S, C_P)):
        raise ValueError("All cost coefficients must be positive.")

    expected_4d = (F, M, L, H)
    if mu_param.shape != expected_4d:
        raise ValueError(
            f"mu_param must have shape {expected_4d}, got {mu_param.shape}."
        )
    expected_2d = (F, L)
    if mu_0.shape != expected_2d:
        raise ValueError(f"mu_0 must have shape {expected_2d}.")
    if replacement_mu.shape != expected_2d:
        raise ValueError(f"replacement_mu must have shape {expected_2d}.")
    if gamma_beta.shape != (L,):
        raise ValueError(f"gamma_beta must have shape ({L},).")
    if tau.shape != (L,):
        raise ValueError(f"tau must have shape ({L},).")
    if repair_rho.shape != (L,):
        raise ValueError(f"repair_rho must have shape ({L},).")

    named_arrays = {
        "mu_param": mu_param,
        "mu_0": mu_0,
        "replacement_mu": replacement_mu,
        "gamma_beta": gamma_beta,
        "tau": tau,
        "repair_rho": repair_rho,
    }
    for name, value in named_arrays.items():
        if np.any(~np.isfinite(value)):
            raise ValueError(f"{name} must contain only finite values.")

    if np.any(mu_param < 0.0):
        raise ValueError("mu_param cannot contain negative damage increments.")
    if np.any(mu_0 < 0.0):
        raise ValueError("mu_0 cannot contain negative damage.")
    if np.any(replacement_mu < 0.0):
        raise ValueError("replacement_mu cannot contain negative damage.")
    if np.any(gamma_beta <= 0.0):
        raise ValueError("Every Gamma rate must be positive.")
    if np.any(tau <= 0.0):
        raise ValueError("Every failure threshold must be positive.")
    if np.any((repair_rho < 0.0) | (repair_rho > 1.0)):
        raise ValueError("Every repair effectiveness must lie in [0, 1].")


def solve_fleet_management(
    F: int,
    H: int,
    M: int,
    L: int,
    mu_param: np.ndarray,
    tau: np.ndarray,
    epsilon: float,
    gamma_beta: np.ndarray,
    C_M: float,
    C_R: float,
    C_rep: float,
    C_S: float,
    C_P: float,
    mu_0: np.ndarray,
    replacement_mu: np.ndarray,
    repair_rho: np.ndarray,
    verbose: int = 1,
    mip_gap: float | None = None,
    time_limit: float | None = None,
    gurobi_params: dict | None = None,
) -> dict:
    """Solve the fleet problem with common-rate Gamma degradation."""

    backend_start = time.perf_counter()         # performance measurement

    validate_inputs(
        F=F,
        H=H,
        M=M,
        L=L,
        mu_param=mu_param,
        tau=tau,
        epsilon=epsilon,
        gamma_beta=gamma_beta,
        C_M=C_M,
        C_R=C_R,
        C_rep=C_rep,
        C_S=C_S,
        C_P=C_P,
        mu_0=mu_0,
        replacement_mu=replacement_mu,
        repair_rho=repair_rho,
    )

    beta = np.asarray(gamma_beta, dtype=float)
    initial_shape = mu_0 * beta[np.newaxis, :]
    replacement_shape = replacement_mu * beta[np.newaxis, :]
    increment_shape = mu_param * beta[np.newaxis, np.newaxis, :, np.newaxis]
    maximum_shape = np.array(
        [
            maximum_reliable_shape(
                beta=float(beta[l]),
                threshold=float(tau[l]),
                epsilon=epsilon,
            )
            for l in range(L)
        ],
        dtype=float,
    )

    def mission_shape(i: int, j: int, l: int, k: int) -> float:
        return float(increment_shape[i, j, l, k % H])

    model = gp.Model("fleet_management_gamma_degradation")
    model.Params.OutputFlag = int(verbose)
    if mip_gap is not None:
        model.Params.MIPGap = float(mip_gap)
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)
    if gurobi_params:
        for name, value in gurobi_params.items():
            model.setParam(name, value)

    # j=0 grants maintenance access; j=1,...,M are missions.
    x = model.addVars(F, M + 1, 2 * H, vtype=GRB.BINARY, name="x")
    m = model.addVars(F, L, 2 * H, vtype=GRB.BINARY, name="m")
    r = model.addVars(F, L, 2 * H, vtype=GRB.BINARY, name="r")
    # q means maintenance access without an action on this component.
    q = model.addVars(F, L, 2 * H, vtype=GRB.BINARY, name="q")
    shape_var = model.addVars(
        F, L, 2 * H, vtype=GRB.CONTINUOUS, lb=0.0, name="A"
    )
    u_var = model.addVars(
        2 * H, vtype=GRB.CONTINUOUS, lb=0.0, name="u"
    )
    z_var = model.addVars(F, L, 2 * H, vtype=GRB.CONTINUOUS, lb=0.0, name="z")

    objective = gp.LinExpr()
    for k in range(2 * H):
        objective += C_S * u_var[k]
        for i in range(F):
            objective += C_M * x[i, 0, k]
            for l in range(L):
                objective += C_R * z_var[i, l, k]
                objective += C_rep * r[i, l, k]
    for i in range(F):
        for l in range(L):
            objective += (C_P / beta[l]) * (
                shape_var[i, l, H - 1]
                - shape_var[i, l, 2 * H - 1]
            )
    model.setObjective(objective, GRB.MINIMIZE)

    # Preserve the normalized aggregate-capacity constraint used by the
    # existing fleet formulation, expressed here in expected-damage units.
    for k in range(2 * H):
        model.addConstr(
            gp.quicksum(
                shape_var[i, l, k] / beta[l]
                for i in range(F)
                for l in range(L)
            )
            <= F - M,
            name=f"capacity_{k}",
        )

    for i in range(F):
        for l in range(L):
            for k in range(2 * H):
                previous_shape = (
                    float(initial_shape[i, l])
                    if k == 0
                    else shape_var[i, l, k - 1]
                )
                assigned_increment = gp.quicksum(
                    x[i, j, k] * mission_shape(i, j - 1, l, k)
                    for j in range(1, M + 1)
                )

                # With no maintenance access, accumulate the assigned mission
                # increment (or preserve the state when idle).
                model.addGenConstrIndicator(
                    x[i, 0, k],
                    False,
                    shape_var[i, l, k] == previous_shape + assigned_increment,
                    name=f"accumulate_or_idle_{i}_{l}_{k}",
                )
                # During maintenance access, every component is unchanged,
                # repaired, or replaced.
                model.addConstr(
                    q[i, l, k] + m[i, l, k] + r[i, l, k] == x[i, 0, k],
                    name=f"maintenance_choice_{i}_{l}_{k}",
                )
                model.addGenConstrIndicator(
                    q[i, l, k],
                    True,
                    shape_var[i, l, k] == previous_shape,
                    name=f"maintenance_idle_{i}_{l}_{k}",
                )
                model.addGenConstrIndicator(
                    m[i, l, k],
                    True,
                    shape_var[i, l, k]
                    == (1.0 - float(repair_rho[l])) * previous_shape,
                    name=f"imperfect_repair_{i}_{l}_{k}",
                )
                model.addGenConstrIndicator(
                    r[i, l, k],
                    True,
                    shape_var[i, l, k] == float(replacement_shape[i, l]),
                    name=f"replacement_{i}_{l}_{k}",
                )

                # For fixed beta and tau, the Gamma tail is monotone in A.
                model.addConstr(
                    shape_var[i, l, k] <= maximum_shape[l],
                    name=f"reliability_{i}_{l}_{k}",
                )
                model.addConstr(
                    u_var[k] >= shape_var[i, l, k] / beta[l],
                    name=f"u_bound_{i}_{l}_{k}",
                )

                # z is expected degradation removed by imperfect repair only.
                model.addGenConstrIndicator(
                    m[i, l, k],
                    False,
                    z_var[i, l, k] == 0.0,
                    name=f"no_repair_cost_{i}_{l}_{k}",
                )
                model.addGenConstrIndicator(
                    m[i, l, k],
                    True,
                    z_var[i, l, k]
                    == float(repair_rho[l]) * previous_shape / beta[l],
                    name=f"repair_cost_{i}_{l}_{k}",
                )

    # Repeatability under a common beta is equivalent to comparing shapes.
    for i in range(F):
        for l in range(L):
            model.addConstr(
                shape_var[i, l, 2 * H - 1]
                <= shape_var[i, l, H - 1],
                name=f"shape_periodic_{i}_{l}",
            )

    # Each member receives at most one action in a planning step.
    for i in range(F):
        for k in range(2 * H):
            model.addConstr(
                gp.quicksum(x[i, j, k] for j in range(M + 1)) <= 1,
                name=f"assignment_{i}_{k}",
            )

    # Each mission and one fleet-level maintenance opportunity are assigned
    # exactly once.  Component-level m/r variables decide how that opportunity
    # is used.
    for j in range(M + 1):
        for k in range(2 * H):
            model.addConstr(
                gp.quicksum(x[i, j, k] for i in range(F)) == 1,
                name=f"demand_{j}_{k}",
            )

    model.update()                                                  # begin performance measurement
    construction_seconds = time.perf_counter() - backend_start

    optimizer_start = time.perf_counter()
    model.optimize()
    optimizer_call_seconds = time.perf_counter() - optimizer_start

    performance = _collect_model_performance(
        model=model,
        construction_seconds=construction_seconds,
        optimizer_call_seconds=optimizer_call_seconds,
    )                                                               # end performance measurement

    extraction_start = time.perf_counter()
    # A time/node/solution-limited MIP may still have a valid incumbent. Keep
    # that schedule instead of returning an empty result merely because
    # optimality was not proven.
    if int(model.SolCount) > 0:
        x_solution = np.zeros((F, M + 1, 2 * H))
        m_solution = np.zeros((F, L, 2 * H))
        r_solution = np.zeros((F, L, 2 * H))
        shape_solution = np.zeros((F, L, 2 * H))
        expected_solution = np.zeros((F, L, 2 * H))
        tail_solution = np.zeros((F, L, 2 * H))
        u_solution = np.zeros(2 * H)
        z_solution = np.zeros((F, L, 2 * H))

        for k in range(2 * H):
            u_solution[k] = u_var[k].X
            for i in range(F):
                for j in range(M + 1):
                    x_solution[i, j, k] = x[i, j, k].X
                for l in range(L):
                    m_solution[i, l, k] = m[i, l, k].X
                    r_solution[i, l, k] = r[i, l, k].X
                    z_solution[i, l, k] = z_var[i, l, k].X
                    shape = shape_var[i, l, k].X
                    shape_solution[i, l, k] = shape
                    expected_solution[i, l, k] = shape / beta[l]
                    if shape > 0.0:
                        tail_solution[i, l, k] = gamma.sf(
                            tau[l],
                            a=shape,
                            scale=1.0 / beta[l],
                        )

        performance["solution_extraction_seconds"] = (          # performance measurement
            time.perf_counter() - extraction_start
        )
        performance["backend_wall_seconds"] = (                 # performance measurement
            time.perf_counter() - backend_start
        )

        status = {
            GRB.OPTIMAL: "optimal",
            GRB.TIME_LIMIT: "time_limit",
            GRB.SOLUTION_LIMIT: "solution_limit",
            GRB.NODE_LIMIT: "node_limit",
            GRB.ITERATION_LIMIT: "iteration_limit",
            GRB.INTERRUPTED: "interrupted",
            GRB.SUBOPTIMAL: "suboptimal",
        }.get(int(model.status), f"gurobi_status_{int(model.status)}")

        return {
            "status": status,
            "objective": model.ObjVal,
            "bound": model.ObjBound,
            "mip_gap": model.MIPGap,
            "F": F,
            "H": H,
            "M": M,
            "L": L,
            "tau": tau,
            "gamma_beta": beta,
            "replacement_mu": replacement_mu,
            "repair_rho": repair_rho,
            "maximum_shape": maximum_shape,
            "x": x_solution,
            "m": m_solution,
            "r": r_solution,
            "A": shape_solution,
            "mu": expected_solution,
            "tail_probability": tail_solution,
            "u": u_solution,
            "z": z_solution,
            "model": model,
            "performance": performance,                 # performance measurement
        }

    performance["backend_wall_seconds"] = (             # performance measurement
        time.perf_counter() - backend_start
    )

    status = {
        GRB.INFEASIBLE: "infeasible",
        GRB.INF_OR_UNBD: "inf_or_unbounded",
        GRB.UNBOUNDED: "unbounded",
        GRB.TIME_LIMIT: "time_limit_no_incumbent",
        GRB.INTERRUPTED: "interrupted_no_incumbent",
    }.get(int(model.status), f"gurobi_status_{int(model.status)}")
    return {
        "status": status,
        "objective": None,
        "bound": None,
        "mip_gap": None,
        "F": F,
        "H": H,
        "M": M,
        "L": L,
        "tau": tau,
        "gamma_beta": beta,
        "replacement_mu": replacement_mu,
        "repair_rho": repair_rho,
        "maximum_shape": maximum_shape,
        "x": None,
        "m": None,
        "r": None,
        "A": None,
        "mu": None,
        "tail_probability": None,
        "u": None,
        "z": None,
        "model": model,
        "performance": performance,                     # performance measurement
    }


def _collect_model_performance(                         # performance measurement
    *,
    model: gp.Model,
    construction_seconds: float,
    optimizer_call_seconds: float,
) -> dict[str, object]:
    """Collect solver-size and runtime diagnostics after optimization."""

    solution_exists = int(model.SolCount) > 0

    objective_value = None
    objective_bound = None
    relative_mip_gap = None

    if solution_exists:
        objective_value = float(model.ObjVal)
        objective_bound = float(model.ObjBound)

        # MIPGap is meaningful for MIP models with an incumbent solution.
        if int(model.NumIntVars) > 0:
            relative_mip_gap = float(model.MIPGap)

    return {
        "gurobi_version": ".".join(
            str(part) for part in gp.gurobi.version()
        ),
        "status_code": int(model.Status),
        "solutions_found": int(model.SolCount),

        # Model scale
        "variables": int(model.NumVars),
        "continuous_variables": int(model.NumVars - model.NumIntVars),
        "integer_variables": int(model.NumIntVars),
        "binary_variables": int(model.NumBinVars),
        "linear_constraints": int(model.NumConstrs),
        "general_constraints": int(model.NumGenConstrs),
        "nonzeros": int(model.NumNZs),

        # Timing
        "model_construction_seconds": float(construction_seconds),
        "optimizer_call_seconds": float(optimizer_call_seconds),
        "gurobi_runtime_seconds": float(model.Runtime),
        "solution_extraction_seconds": 0.0,
        "backend_wall_seconds": 0.0,

        # Optimization work
        "branch_and_bound_nodes": float(model.NodeCount),
        "simplex_iterations": float(model.IterCount),
        "barrier_iterations": float(model.BarIterCount),
        "work_units": float(model.Work),

        # Solution quality
        "objective_value": objective_value,
        "objective_bound": objective_bound,
        "relative_mip_gap": relative_mip_gap,
    }
