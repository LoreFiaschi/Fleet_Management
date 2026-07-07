"""Shared building blocks used by every degradation-model module.

These functions add constraints that are identical in form across models:
assignment/coverage/maintenance-day compatibility, the replacement big-M reset,
the repair-cost McCormick envelope, the loop (sustainability) constraint, and
the damage-regularisation variable ``u``. Model-specific dynamics live in the
individual ``models/*.py`` modules; the ARD1 accumulate/repair pattern shared
by every ARD1 model lives in ``fleet_management.maintenance.ard1``.

Reference: spec/spec.tex, "Constraints" and "Objective Function" sections.
"""

import gurobipy as gp
from scipy.stats import norm


def phi_inv_sq(epsilon: float) -> float:
    """(Phi^{-1}(1 - epsilon))^2 -- the Gaussian/Wiener reliability constant."""
    return float(norm.ppf(1 - epsilon) ** 2)


def mission_delta(x, inc, i, l, k, M):
    """sum_{j=1}^{M} x[i,j,k] * inc[i, j-1, l, k] (0-based mission index).

    ``inc`` is any (F, M, L, 2H) increment tensor (mu_inc or v_inc); the
    increment values are constants (already periodically wrapped), so this
    returns a plain linear expression in the assignment variables ``x``.
    """
    return gp.quicksum(
        x[i, j, k] * float(inc[i, j - 1, l, k]) for j in range(1, M + 1)
    )


def add_assignment_constraints(model, x, x_m, x_r, F, L, M, two_h):
    """Assignment, mission-coverage, and maintenance-day compatibility rows.

    These are independent of every component's degradation model, so they are
    added exactly once per solve rather than per (i, l).
    """
    for i in range(F):
        for k in range(two_h):
            model.addConstr(
                gp.quicksum(x[i, j, k] for j in range(M + 1)) <= 1,
                name=f"one_activity_{i}_{k}",
            )
    for j in range(1, M + 1):
        for k in range(two_h):
            model.addConstr(
                gp.quicksum(x[i, j, k] for i in range(F)) == 1,
                name=f"coverage_{j}_{k}",
            )
    for i in range(F):
        for l in range(L):
            for k in range(two_h):
                model.addConstr(
                    x_m[i, l, k] <= x[i, 0, k], name=f"maint_day_m_{i}_{l}_{k}"
                )
                model.addConstr(
                    x_r[i, l, k] <= x[i, 0, k], name=f"maint_day_r_{i}_{l}_{k}"
                )
                model.addConstr(
                    x_m[i, l, k] + x_r[i, l, k] <= 1,
                    name=f"no_repair_and_replace_{i}_{l}_{k}",
                )


def add_replacement_linear(model, var, new_val, x_r, big_m, name):
    """Two-sided big-M linearisation of ``var == new_val`` when ``x_r == 1``."""
    model.addConstr(var <= new_val + big_m * (1 - x_r), name=f"{name}_rep_ub")
    model.addConstr(var >= new_val - big_m * (1 - x_r), name=f"{name}_rep_lb")


def add_repair_cost_mccormick(model, z, prev_expr, rho, x_m, tau_big_m, name):
    """McCormick envelope for z = rho * prev_expr * x_m (ARD1 repair-cost form)."""
    model.addConstr(z <= rho * prev_expr, name=f"{name}_z1")
    model.addConstr(z <= rho * tau_big_m * x_m, name=f"{name}_z2")
    model.addConstr(
        z >= rho * prev_expr - rho * tau_big_m * (1 - x_m), name=f"{name}_z3"
    )


def add_loop_constraint(model, var, i, l, H, name):
    """var[i, l, 2H-1] <= var[i, l, H-1] (0-based indices).

    The hard sustainability constraint shared by every model: it bounds the
    tracked state at the end of the second half-horizon by its value at the
    end of the first, which is what makes every future repetition of the
    second half certified (see spec/spec.tex Sections 5.3-5.8, "Loop
    Constraint").
    """
    model.addConstr(var[i, l, 2 * H - 1] <= var[i, l, H - 1], name=f"{name}_loop_{i}_{l}")


def add_damage_regularization(model, u, mu, F, L, two_h, penalty_type):
    """u >= sum_l mu_ilk (inf_norm) or u >= sum_l mu_ilk^2 (quadratic).

    Must be called after every component's ``build_component`` has populated
    ``mu`` for all (i, l, k), since it needs the full per-train sum at each
    step.
    """
    for i in range(F):
        for k in range(two_h):
            terms = [mu[i, l, k] for l in range(L)]
            if penalty_type == "inf_norm":
                model.addConstr(u >= gp.quicksum(terms), name=f"u_bound_{i}_{k}")
            else:  # "quadratic": convex rotated-SOC-equivalent row, no NonConvex needed
                model.addConstr(
                    u >= gp.quicksum(t * t for t in terms), name=f"u_bound_{i}_{k}"
                )
