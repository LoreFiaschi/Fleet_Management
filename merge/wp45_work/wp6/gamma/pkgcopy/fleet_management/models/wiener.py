"""Wiener degradation model (ARD1 and ARA1).

Both the mean mu and the variance v are tracked, but imperfect repair acts on
the mean only -- variance accumulates deterministically from missions
(``sigma^2`` per mission, independent of which one) and can only be reset by
replacement. Reference: spec/spec.tex Section 5.4 (state dynamics) and its
"Reliability Constraint (Wiener)" / "Loop Constraint (Wiener, hard)"
subsections.
"""

import gurobipy as gp
import numpy as np

from fleet_management.maintenance import ara1, ard1
from fleet_management.models import base

MODEL_NAME = "wiener"


def validate_inputs(mask, mu_0, v_0, mu_inc, tau, rho, sigma, maintenance_type):
    """Consistency checks restricted to the (i, l) entries where mask is True.

    Unlike Gaussian/IG/Rainflow, both ARD1 and ARA1 are valid maintenance
    types here, so there is no maintenance_type restriction to enforce.
    """
    for i, l in np.argwhere(mask):
        i, l = int(i), int(l)
        if maintenance_type[i][l] not in ("ARD1", "ARA1"):
            raise ValueError(
                f"Wiener component ({i}, {l}): maintenance_type must be 'ARD1' or "
                f"'ARA1' (got '{maintenance_type[i][l]}')."
            )
        if not (0 < rho[i, l] <= 1):
            raise ValueError(f"Wiener component ({i}, {l}): rho must be in (0, 1].")
        if sigma[i, l] <= 0:
            raise ValueError(f"Wiener component ({i}, {l}): sigma must be positive.")
        if not (v_0[i, l] > 0):
            raise ValueError(f"Wiener component ({i}, {l}): v_0 must be positive.")
        if not (mu_0[i, l] < tau[i, l]):
            raise ValueError(f"Wiener component ({i}, {l}): mu_0 must be < tau.")
        if not np.all(mu_inc[i, :, l, :] < tau[i, l]):
            raise ValueError(f"Wiener component ({i}, {l}): mu_inc must be < tau.")


def build_component(ctx, i, l):
    """Add all Wiener variables/constraints for component (i, l) to ctx.model."""
    model = ctx.model
    two_h, M = ctx.two_h, ctx.M
    tau_il = float(ctx.tau[i, l])
    rho_il = float(ctx.rho[i, l])
    sigma_il = float(ctx.sigma[i, l])
    v_max = tau_il ** 2 / ctx.phi_inv_sq
    mu_new_il = float(ctx.mu_new[i, l])
    v_new_il = float(ctx.v_new[i, l])
    is_ara1 = ctx.maintenance_type[i][l] == "ARA1"

    mu = model.addVars(two_h, lb=0.0, name=f"mu_w_{i}_{l}")
    v = model.addVars(two_h, lb=0.0, name=f"v_w_{i}_{l}")
    mu_last = model.addVars(two_h, lb=0.0, name=f"mu_last_w_{i}_{l}") if is_ara1 else None

    v_max_user_il = None
    if ctx.v_max_user is not None:
        raw = ctx.v_max_user[i, l]
        if not np.isnan(raw):
            v_max_user_il = float(raw)

    for k in range(two_h):
        mu_prev = float(ctx.mu_0[i, l]) if k == 0 else mu[k - 1]
        v_prev = float(ctx.v_0[i, l]) if k == 0 else v[k - 1]
        x_m_ilk, x_r_ilk = ctx.x_m[i, l, k], ctx.x_r[i, l, k]
        delta_mu = base.mission_delta(ctx.x, ctx.mu_inc, i, l, k, M)
        # Mission-only, mission-independent variance increment (spec remark):
        # the same sigma applies regardless of which mission is performed.
        delta_v = sigma_il ** 2 * gp.quicksum(ctx.x[i, j, k] for j in range(1, M + 1))

        if is_ara1:
            mu_last_prev = float(ctx.mu_0[i, l]) if k == 0 else mu_last[k - 1]
            ara1.update_anchor(
                model, mu_last[k], mu_prev, mu_last_prev, mu_new_il, x_m_ilk, x_r_ilk,
                tau_il, name=f"w_anchor_{i}_{l}_{k}",
            )
            ara1.accumulate_and_repair(
                model, mu[k], mu_prev, delta_mu, mu_last_prev, rho_il,
                x_m_ilk, x_r_ilk, tau_il, name=f"w_mu_{i}_{l}_{k}",
            )
        else:
            ard1.accumulate_and_repair(
                model, mu[k], mu_prev, delta_mu, 1 - rho_il,
                x_m_ilk, x_r_ilk, tau_il, name=f"w_mu_{i}_{l}_{k}",
            )
        base.add_replacement_linear(model, mu[k], mu_new_il, x_r_ilk, tau_il, name=f"w_mu_{i}_{l}_{k}")

        # Variance: mission-only accumulation, no repair term, replace-only reset.
        model.addConstr(
            v[k] >= v_prev + delta_v - v_max * x_r_ilk, name=f"w_v_accum_{i}_{l}_{k}",
        )
        base.add_replacement_linear(model, v[k], v_new_il, x_r_ilk, v_max, name=f"w_v_{i}_{l}_{k}")

        if ctx.formulation == "exact":
            model.addConstr(mu[k] <= tau_il, name=f"w_rel_mu_{i}_{l}_{k}")
            model.addConstr(
                ctx.phi_inv_sq * v[k] <= (tau_il - mu[k]) * (tau_il - mu[k]),
                name=f"w_rel_{i}_{l}_{k}",
            )
        else:  # "lp"
            model.addConstr(
                ctx.phi_inv_sq * v[k] + 2 * tau_il * mu[k] <= tau_il ** 2,
                name=f"w_rel_lp_{i}_{l}_{k}",
            )

        if v_max_user_il is not None:
            model.addConstr(v[k] <= v_max_user_il, name=f"w_vmax_user_{i}_{l}_{k}")

        anchor_for_z = mu_last_prev if is_ara1 else 0.0
        base.add_repair_cost_mccormick(
            model, ctx.z[i, l, k], mu_prev, rho_il, x_m_ilk, tau_il,
            name=f"w_z_{i}_{l}_{k}", anchor_prev_expr=anchor_for_z,
        )

        ctx.mu[i, l, k] = mu[k]
        ctx.v[i, l, k] = v[k]
        if is_ara1:
            ctx.mu_last[i, l, k] = mu_last[k]

    base.add_loop_constraint(model, ctx.mu, i, l, ctx.H, name="w_mu")
    base.add_loop_constraint(model, ctx.v, i, l, ctx.H, name="w_v")
    if is_ara1:
        base.add_loop_constraint(model, ctx.mu_last, i, l, ctx.H, name="w_mu_last")
