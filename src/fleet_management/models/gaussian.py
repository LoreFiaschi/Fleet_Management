"""Gaussian degradation model (ARD1 only).

Both the mean mu and the variance v are tracked as independent decision
variables. Reference: spec/spec.tex Section 5.3 (state dynamics) and its
"Reliability Constraint (Gaussian)" subsection.
"""

import numpy as np

from fleet_management.maintenance.ard1 import accumulate_and_repair
from fleet_management.models import base

MODEL_NAME = "gaussian"


def validate_inputs(mask, mu_0, v_0, mu_inc, v_inc, tau, rho, maintenance_type):
    """Consistency checks restricted to the (i, l) entries where mask is True.

    ``mu_inc``/``v_inc`` are the raw (F, M, L, H) arrays (pre period-doubling);
    checking them here is equivalent to checking the doubled (F, M, L, 2H)
    version since the doubling is a periodic repeat.
    """
    for i, l in np.argwhere(mask):
        i, l = int(i), int(l)
        if maintenance_type[i][l] != "ARD1":
            raise ValueError(
                f"Gaussian component ({i}, {l}): only 'ARD1' maintenance is "
                f"supported for the Gaussian model (got "
                f"'{maintenance_type[i][l]}')."
            )
        if not (0 < rho[i, l] <= 1):
            raise ValueError(f"Gaussian component ({i}, {l}): rho must be in (0, 1].")
        if not (mu_0[i, l] < tau[i, l]):
            raise ValueError(f"Gaussian component ({i}, {l}): mu_0 must be < tau.")
        if not (mu_0[i, l] >= 3 * np.sqrt(v_0[i, l])):
            raise ValueError(
                f"Gaussian component ({i}, {l}): mu_0 must be >= 3*sqrt(v_0)."
            )
        if not np.all(mu_inc[i, :, l, :] < tau[i, l]):
            raise ValueError(f"Gaussian component ({i}, {l}): mu_inc must be < tau.")
        if not np.all(mu_inc[i, :, l, :] >= 3 * np.sqrt(v_inc[i, :, l, :])):
            raise ValueError(
                f"Gaussian component ({i}, {l}): mu_inc must be >= 3*sqrt(v_inc)."
            )


def build_component(ctx, i, l):
    """Add all Gaussian variables/constraints for component (i, l) to ctx.model."""
    model = ctx.model
    two_h, M = ctx.two_h, ctx.M
    tau_il = float(ctx.tau[i, l])
    rho_il = float(ctx.rho[i, l])
    v_max = tau_il ** 2 / ctx.phi_inv_sq
    mu_new_il = float(ctx.mu_new[i, l])
    v_new_il = float(ctx.v_new[i, l])

    mu = model.addVars(two_h, lb=0.0, name=f"mu_g_{i}_{l}")
    v = model.addVars(two_h, lb=0.0, name=f"v_g_{i}_{l}")

    v_max_user_il = None
    if ctx.v_max_user is not None:
        raw = ctx.v_max_user[i, l]
        if not np.isnan(raw):
            v_max_user_il = float(raw)

    for k in range(two_h):
        mu_prev = float(ctx.mu_0[i, l]) if k == 0 else mu[k - 1]
        v_prev = float(ctx.v_0[i, l]) if k == 0 else v[k - 1]
        delta_mu = base.mission_delta(ctx.x, ctx.mu_inc, i, l, k, M)
        delta_v = base.mission_delta(ctx.x, ctx.v_inc, i, l, k, M)
        x_m_ilk, x_r_ilk = ctx.x_m[i, l, k], ctx.x_r[i, l, k]

        accumulate_and_repair(
            model, mu[k], mu_prev, delta_mu, 1 - rho_il,
            x_m_ilk, x_r_ilk, tau_il, name=f"g_mu_{i}_{l}_{k}",
        )
        accumulate_and_repair(
            model, v[k], v_prev, delta_v, (1 - rho_il) ** 2,
            x_m_ilk, x_r_ilk, v_max, name=f"g_v_{i}_{l}_{k}",
        )
        base.add_replacement_linear(model, mu[k], mu_new_il, x_r_ilk, tau_il, name=f"g_mu_{i}_{l}_{k}")
        base.add_replacement_linear(model, v[k], v_new_il, x_r_ilk, v_max, name=f"g_v_{i}_{l}_{k}")

        if ctx.formulation == "exact":
            model.addConstr(mu[k] <= tau_il, name=f"g_rel_mu_{i}_{l}_{k}")
            model.addConstr(
                ctx.phi_inv_sq * v[k] <= (tau_il - mu[k]) * (tau_il - mu[k]),
                name=f"g_rel_{i}_{l}_{k}",
            )
        else:  # "lp": conservative inner approximation, error = mu_ilk^2
            model.addConstr(
                ctx.phi_inv_sq * v[k] + 2 * tau_il * mu[k] <= tau_il ** 2,
                name=f"g_rel_lp_{i}_{l}_{k}",
            )

        if v_max_user_il is not None:
            model.addConstr(v[k] <= v_max_user_il, name=f"g_vmax_user_{i}_{l}_{k}")

        base.add_repair_cost_mccormick(
            model, ctx.z[i, l, k], mu_prev, rho_il, x_m_ilk, tau_il, name=f"g_z_{i}_{l}_{k}",
        )

        ctx.mu[i, l, k] = mu[k]
        ctx.v[i, l, k] = v[k]

    base.add_loop_constraint(model, ctx.mu, i, l, ctx.H, name="g_mu")
    base.add_loop_constraint(model, ctx.v, i, l, ctx.H, name="g_v")
