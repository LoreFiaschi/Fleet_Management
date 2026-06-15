# Tiny, deterministic MILP for testing

from __future__ import annotations

import numpy as np
import gurobipy as gp
from gurobipy import GRB


def solve_ev_deterministic(
    F: int,
    M: int,
    L: int,
    H: int,
    damage_increment: np.ndarray,
    initial_damage: np.ndarray,
    repair_fraction: np.ndarray,
    alpha: float,
    C_M: float,
    C_D: float,
    C_R: np.ndarray,
    verbose: int = 1,
    mip_gap: float | None = None,
) -> dict:
    """
    Deterministic EV fleet-management MILP

    x[i, 0, k] = 1 means vehicle i is in maintenance at time k,
    x[i, j, k] = 1 for j > 0 means vehicle i serves mission j-1,
    D[i, l, k] is component damage,
    z[i, l, k] is repaired damage amount proxy,
    u[k] is worst-case damage at time k.
    """

    if damage_increment.shape != (F, M, L, H):
        raise ValueError(
            f"damage_increment must have shape {(F, M, L, H)}, "
            f"got {damage_increment.shape}."
        )
    
    if initial_damage.shape != (F, L):
        raise ValueError(
            f"initial_damage must have shape {(F, L)}, got {initial_damage.shape}."
        )
    
    if repair_fraction.shape != (F, L):
        raise ValueError(
            f"repair_fraction must have shape {(F, L)}, got {repair_fraction.shape}."
        )
    
    if C_R.shape != (F, L):
        raise ValueError(f"C_R must have shape {(F, L)}, got {C_R.shape}.")
    
    model = gp.Model("ev_deterministic_fleet_management")
    model.Params.OutputFlag = int(verbose)

    if mip_gap is not None:
        model.Params.MIPGap = mip_gap

    x = model.addVars(F, M + 1, 2 * H, vtype=GRB.BINARY, name="x")
    D = model.addVars(F, L, 2 * H, vtype=GRB.CONTINUOUS, lb=0.0, name="D")
    z = model.addVars(F, L, 2 * H, vtype=GRB.CONTINUOUS, lb=0.0, name="z")
    u = model.addVars(2 * H, vtype=GRB.CONTINUOUS, lb=0.0, name="u")

    # Objective:
    # maintenance cost + repair cost + damage regularisation
    obj = gp.LinExpr()

    for k in range(2 * H):
        obj += C_D * u[k]

        for i in range(F):
            obj += C_M * x[i, 0, k]

            for ell in range(L):
                obj += float(C_R[i, ell]) * z[i, ell, k]
    
    model.setObjective(obj, GRB.MINIMIZE)

    def delta(i: int, j: int, ell: int, k: int) -> float:
        return float(damage_increment[i, j, ell, k % H])
    
    # At most one action per vehicle per day (assignment constraint)
    for i in range(F):
        for k in range(2 * H):
            model.addConstr(
                gp.quicksum(x[i, j, k] for j in range(M + 1)) <= 1,
                name=f"one_action_{i}_{k}",
            )
    
    # Each mission is served once per day.
    # Maintenance is capacity-limited separately; it is NOT forced every day.
    for j in range(1, M + 1):
        for k in range(2 * H):
            model.addConstr(
                gp.quicksum(x[i, j, k] for i in range(F)) == 1,
                name=f"mission_demand_{j}_{k}",
            )
    
    # Damage dynamics and repair proxy
    for i in range(F):
        for ell in range(L):
            for k in range(2 * H):
                if k == 0:
                    D_prev = float(initial_damage[i, ell])
                else:
                    D_prev = D[i, ell, k - 1]

                mission_damage = gp.quicksum(
                    x[i, j, k] * delta(i, j - 1, ell, k)
                    for j in range(1, M + 1)
                )

                # If assigned to a mission, accumulate damage.
                # If in maintenance, allow a large reset term through -alpha*x.
                model.addConstr(
                    D[i, ell, k]
                    >= D_prev + mission_damage - alpha * x[i, 0, k],
                    name=f"damage_update_{i}_{ell}_{k}",
                )

                # Imperfect repair lower bound during maintenance
                model.addConstr(
                    D[i, ell, k]
                    >= (1.0 - float(repair_fraction[i, ell])) * D_prev,
                    name=f"repair_lower_bound_{i}_{ell}_{k}",
                )

                # repaired amount proxy:
                # z becomes positive only if maintenance happens and damage is removable
                model.addConstr(
                    z[i, ell, k]
                    >= repair_fraction[i, ell] * D_prev
                    - alpha * (1.0 - x[i, 0, k]),
                    name=f"repair_amount_proxy_{i}_{ell}_{k}",
                )

                # keep damage below threshold
                model.addConstr(
                    D[i, ell, k] <= alpha,
                    name=f"damage_threshold_{i}_{ell}_{k}",
                )

                model.addConstr(
                    u[k] >= D[i, ell, k],
                    name=f"u_bound_{i}_{ell}_{k}",
                )
    
    # Periodic stabilization, same style as current code
    for i in range(F):
        for ell in range(L):
            model.addConstr(
                D[i, ell, 2 * H - 1] <= D[i, ell, H - 1],
                name=f"damage_periodic_{i}_{ell}",
            )

    model.optimize()

    if model.status == GRB.OPTIMAL:
        x_sol = np.zeros((F, M + 1, 2 * H))
        D_sol = np.zeros((F, L, 2 * H))
        z_sol = np.zeros((F, L, 2 * H))
        u_sol = np.zeros(2 * H)

        for k in range(2 * H):
            u_sol[k] = u[k].X

            for i in range(F):
                for j in range(M + 1):
                    x_sol[i, j, k] = x[i, j, k].X

                for ell in range(L):
                    D_sol[i, ell, k] = D[i, ell, k].X
                    z_sol[i, ell, k] = z[i, ell, k].X

        return {
            "status": "optimal",
            "objective": model.ObjVal,
            "degradation": "ev_deterministic",
            "F": F,
            "M": M,
            "L": L,
            "H": H,
            "alpha": alpha,
            "x": x_sol,
            "D": D_sol,
            "z": z_sol,
            "u": u_sol,
            "model": model,
        }

    return {
        "status": model.status,
        "objective": None,
        "degradation": "ev_deterministic",
        "F": F,
        "M": M,
        "L": L,
        "H": H,
        "alpha": alpha,
        "x": None,
        "D": None,
        "z": None,
        "u": None,
        "model": model,
    }
