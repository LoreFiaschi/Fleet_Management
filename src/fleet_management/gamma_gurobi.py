"""Gurobi backend for exact constant-rate Gamma degradation.

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
Replacement
    ``A_k = beta * replacement_mu``.

The decision ``x[i, 0, k]`` is interpreted as replacement in this backend.
Imperfect repair is not implemented because scaling a Gamma variable changes
its rate and breaks the exact common-beta closure used by the MILP.
"""

# After Midterm 28.07.2026, required for degradation/gamma.py

from __future__ import annotations

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from scipy.stats import gamma

from fleet_management.degradation.gamma import maximum_reliable_shape


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
    C_S: float,
    C_P: float,
    mu_0: np.ndarray,
    replacement_mu: np.ndarray,
) -> None:
    """Validate the exact constant-rate Gamma solver contract."""

    if F <= 0 or H <= 0 or M <= 0 or L <= 0:
        raise ValueError("F, H, M and L must be positive integers.")
    if F <= M:
        raise ValueError(f"F must be greater than M (got F={F}, M={M}).")
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must lie strictly between zero and one.")
    if any(cost <= 0.0 for cost in (C_M, C_R, C_S, C_P)):
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

    named_arrays = {
        "mu_param": mu_param,
        "mu_0": mu_0,
        "replacement_mu": replacement_mu,
        "gamma_beta": gamma_beta,
        "tau": tau,
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
    C_S: float,
    C_P: float,
    mu_0: np.ndarray,
    replacement_mu: np.ndarray,
    verbose: int = 1,
    mip_gap: float | None = None,
) -> dict:
    """Solve the fleet problem with exact constant-rate Gamma degradation."""

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
        C_S=C_S,
        C_P=C_P,
        mu_0=mu_0,
        replacement_mu=replacement_mu,
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

    # j=0 is full replacement; j=1,...,M are missions.
    x = model.addVars(F, M + 1, 2 * H, vtype=GRB.BINARY, name="x")
    shape_var = model.addVars(
        F, L, 2 * H, vtype=GRB.CONTINUOUS, lb=0.0, name="A"
    )
    u_var = model.addVars(
        2 * H, vtype=GRB.CONTINUOUS, lb=0.0, name="u"
    )
    z_var = model.addVars(
        F, 2 * H, vtype=GRB.CONTINUOUS, lb=0.0, name="z"
    )

    objective = gp.LinExpr()
    for k in range(2 * H):
        objective += C_S * u_var[k]
        for i in range(F):
            objective += C_M * x[i, 0, k]
            objective += C_R * z_var[i, k]
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

                # Exact transition if no replacement occurs. With no assigned
                # mission this also represents the idle transition.
                model.addGenConstrIndicator(
                    x[i, 0, k],
                    False,
                    shape_var[i, l, k] == previous_shape + assigned_increment,
                    name=f"accumulate_or_idle_{i}_{l}_{k}",
                )
                # Exact full-replacement transition.
                model.addGenConstrIndicator(
                    x[i, 0, k],
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

        for k in range(2 * H):
            previous_expected_damage = gp.quicksum(
                (
                    float(initial_shape[i, l])
                    if k == 0
                    else shape_var[i, l, k - 1]
                )
                / beta[l]
                for l in range(L)
            )
            replacement_expected_damage = float(np.sum(replacement_mu[i]))

            model.addGenConstrIndicator(
                x[i, 0, k],
                False,
                z_var[i, k] == 0.0,
                name=f"no_replacement_cost_{i}_{k}",
            )
            model.addGenConstrIndicator(
                x[i, 0, k],
                True,
                z_var[i, k]
                >= previous_expected_damage - replacement_expected_damage,
                name=f"replacement_cost_{i}_{k}",
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

    # Each mission and the replacement slot are assigned exactly once.
    for j in range(M + 1):
        for k in range(2 * H):
            model.addConstr(
                gp.quicksum(x[i, j, k] for i in range(F)) == 1,
                name=f"demand_{j}_{k}",
            )

    model.optimize()

    if model.status == GRB.OPTIMAL:
        x_solution = np.zeros((F, M + 1, 2 * H))
        shape_solution = np.zeros((F, L, 2 * H))
        expected_solution = np.zeros((F, L, 2 * H))
        tail_solution = np.zeros((F, L, 2 * H))
        u_solution = np.zeros(2 * H)
        z_solution = np.zeros((F, 2 * H))

        for k in range(2 * H):
            u_solution[k] = u_var[k].X
            for i in range(F):
                z_solution[i, k] = z_var[i, k].X
                for j in range(M + 1):
                    x_solution[i, j, k] = x[i, j, k].X
                for l in range(L):
                    shape = shape_var[i, l, k].X
                    shape_solution[i, l, k] = shape
                    expected_solution[i, l, k] = shape / beta[l]
                    if shape > 0.0:
                        tail_solution[i, l, k] = gamma.sf(
                            tau[l],
                            a=shape,
                            scale=1.0 / beta[l],
                        )

        return {
            "status": "optimal",
            "objective": model.ObjVal,
            "F": F,
            "H": H,
            "M": M,
            "L": L,
            "tau": tau,
            "gamma_beta": beta,
            "replacement_mu": replacement_mu,
            "maximum_shape": maximum_shape,
            "x": x_solution,
            "A": shape_solution,
            "mu": expected_solution,
            "tail_probability": tail_solution,
            "u": u_solution,
            "z": z_solution,
            "model": model,
        }

    return {
        "status": model.status,
        "objective": None,
        "F": F,
        "H": H,
        "M": M,
        "L": L,
        "tau": tau,
        "gamma_beta": beta,
        "replacement_mu": replacement_mu,
        "maximum_shape": maximum_shape,
        "x": None,
        "A": None,
        "mu": None,
        "tail_probability": None,
        "u": None,
        "z": None,
        "model": model,
    }