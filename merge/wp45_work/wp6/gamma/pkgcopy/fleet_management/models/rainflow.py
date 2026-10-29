"""Rainflow degradation model (ARD1 only).

Distribution-free: the per-mission damage-increment law is measured
empirically (rainflow counting) and enters the model only through its mean,
variance, and -- for the Bernstein certificate -- support upper bound. Both
mu and v are tracked; the dynamics, replacement, and loop-constraint rows are
structurally identical to the Gaussian ARD1 model (same big-M accumulate/
repair pattern), differing only in the variance big-M (V_max^RF instead of
V_max) and the reliability constraint, which is certified by a
distribution-free tail bound (Cantelli or Bernstein) selected per component
via ``tail_bound`` rather than an exact parametric CDF. Reference:
spec/spec.tex Section 5.8 (state dynamics) and its "Reliability Constraint
(Rainflow)" subsection.
"""

import math

import numpy as np

from fleet_management.maintenance.ard1 import accumulate_and_repair
from fleet_management.models import base

MODEL_NAME = "rainflow"


def _support_constant(b_0_il, b_new_il, b_inc_il):
    """b_il = max(b_0, b_new, max_{j,k} b_inc_ijlk) -- Eq. (rf_bsupport)."""
    return max(float(b_0_il), float(b_new_il), float(np.max(b_inc_il)))


def validate_inputs(mask, mu_0, v_0, mu_new, v_new, mu_inc, v_inc, tau, rho,
                     maintenance_type, tail_bound, epsilon, b_inc=None, b_0=None,
                     b_new=None):
    """Consistency checks restricted to the (i, l) entries where mask is True.

    Only ARD1 is supported (same reason as Gaussian: no established basis for
    an ARA1 variance extension). The Bernstein-only Bhatia-Davis checks and
    the tau > 2*kappa*b/3 precondition are applied only where
    tail_bound == "bernstein".
    """
    kappa = math.log(1.0 / epsilon)
    for i, l in np.argwhere(mask):
        i, l = int(i), int(l)
        if maintenance_type[i][l] != "ARD1":
            raise ValueError(
                f"Rainflow component ({i}, {l}): only 'ARD1' maintenance is "
                f"supported for the rainflow model (got "
                f"'{maintenance_type[i][l]}')."
            )
        if not (0 < rho[i, l] <= 1):
            raise ValueError(f"Rainflow component ({i}, {l}): rho must be in (0, 1].")
        if not (mu_0[i, l] < tau[i, l]):
            raise ValueError(f"Rainflow component ({i}, {l}): mu_0 must be < tau.")
        if not np.all(mu_inc[i, :, l, :] < tau[i, l]):
            raise ValueError(f"Rainflow component ({i}, {l}): mu_inc must be < tau.")

        tb = tail_bound[i][l]
        if tb not in ("cantelli", "bernstein"):
            raise ValueError(
                f"Rainflow component ({i}, {l}): tail_bound must be 'cantelli' or "
                f"'bernstein' (got '{tb}')."
            )
        if tb != "bernstein":
            continue

        mu_inc_il = mu_inc[i, :, l, :]
        v_inc_il = v_inc[i, :, l, :]
        b_inc_il = b_inc[i, :, l, :]
        b_0_il, b_new_il = float(b_0[i, l]), float(b_new[i, l])

        if not np.all(b_inc_il >= mu_inc_il):
            raise ValueError(
                f"Rainflow component ({i}, {l}): b_inc must be >= mu_inc element-wise."
            )
        if not np.all(v_inc_il <= mu_inc_il * (b_inc_il - mu_inc_il)):
            raise ValueError(
                f"Rainflow component ({i}, {l}): v_inc must be <= mu_inc*(b_inc-mu_inc) "
                "element-wise (Bhatia-Davis: measured moments and support are "
                "mutually inconsistent)."
            )
        if not (b_0_il >= mu_0[i, l]):
            raise ValueError(f"Rainflow component ({i}, {l}): b_0 must be >= mu_0.")
        if not (b_new_il >= mu_new[i, l]):
            raise ValueError(f"Rainflow component ({i}, {l}): b_new must be >= mu_new.")
        if not (v_0[i, l] <= mu_0[i, l] * (b_0_il - mu_0[i, l])):
            raise ValueError(
                f"Rainflow component ({i}, {l}): v_0 must be <= mu_0*(b_0-mu_0) "
                "(Bhatia-Davis)."
            )
        if not (v_new[i, l] <= mu_new[i, l] * (b_new_il - mu_new[i, l])):
            raise ValueError(
                f"Rainflow component ({i}, {l}): v_new must be <= "
                "mu_new*(b_new-mu_new) (Bhatia-Davis)."
            )

        b_il = _support_constant(b_0_il, b_new_il, b_inc_il)
        if not (tau[i, l] > 2 * kappa * b_il / 3):
            raise ValueError(
                f"Rainflow component ({i}, {l}): tau must be > 2*kappa*b/3 "
                f"(kappa={kappa:.4f}, b={b_il}) for the Bernstein certificate to "
                "admit positive variance."
            )


def build_component(ctx, i, l):
    """Add all Rainflow variables/constraints for component (i, l) to ctx.model."""
    model = ctx.model
    two_h, M = ctx.two_h, ctx.M
    tau_il = float(ctx.tau[i, l])
    rho_il = float(ctx.rho[i, l])
    mu_new_il = float(ctx.mu_new[i, l])
    v_new_il = float(ctx.v_new[i, l])
    epsilon = ctx.epsilon
    tb = ctx.tail_bound[i][l]

    kappa_c = math.sqrt((1 - epsilon) / epsilon)
    kappa = math.log(1.0 / epsilon)
    if tb == "bernstein":
        b_il = _support_constant(
            ctx.b_0[i, l], ctx.b_new[i, l], ctx.b_inc[i, :, l, :],
        )
        c_il = kappa * b_il / 3
        v_max_rf = tau_il * (tau_il - 2 * c_il) / (2 * kappa)
    else:
        c_il = None
        v_max_rf = tau_il ** 2 * epsilon / (1 - epsilon)

    mu = model.addVars(two_h, lb=0.0, name=f"mu_rf_{i}_{l}")
    v = model.addVars(two_h, lb=0.0, name=f"v_rf_{i}_{l}")

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
            x_m_ilk, x_r_ilk, tau_il, name=f"rf_mu_{i}_{l}_{k}",
        )
        accumulate_and_repair(
            model, v[k], v_prev, delta_v, (1 - rho_il) ** 2,
            x_m_ilk, x_r_ilk, v_max_rf, name=f"rf_v_{i}_{l}_{k}",
        )
        base.add_replacement_linear(model, mu[k], mu_new_il, x_r_ilk, tau_il, name=f"rf_mu_{i}_{l}_{k}")
        base.add_replacement_linear(model, v[k], v_new_il, x_r_ilk, v_max_rf, name=f"rf_v_{i}_{l}_{k}")

        if tb == "cantelli":
            if ctx.formulation == "exact":
                model.addConstr(mu[k] <= tau_il, name=f"rf_rel_mu_{i}_{l}_{k}")
                model.addConstr(
                    kappa_c ** 2 * v[k] <= (tau_il - mu[k]) * (tau_il - mu[k]),
                    name=f"rf_rel_{i}_{l}_{k}",
                )
            else:  # "lp"
                model.addConstr(
                    kappa_c ** 2 * v[k] + 2 * tau_il * mu[k] <= tau_il ** 2,
                    name=f"rf_rel_lp_{i}_{l}_{k}",
                )
        else:  # "bernstein"
            if ctx.formulation == "exact":
                model.addConstr(mu[k] <= tau_il - c_il, name=f"rf_rel_mu_{i}_{l}_{k}")
                model.addConstr(
                    (tau_il - c_il - mu[k]) * (tau_il - c_il - mu[k])
                    >= c_il ** 2 + 2 * kappa * v[k],
                    name=f"rf_rel_{i}_{l}_{k}",
                )
            else:  # "lp"
                model.addConstr(
                    2 * (tau_il - c_il) * mu[k] + 2 * kappa * v[k]
                    <= tau_il * (tau_il - 2 * c_il),
                    name=f"rf_rel_lp_{i}_{l}_{k}",
                )

        if v_max_user_il is not None:
            model.addConstr(v[k] <= v_max_user_il, name=f"rf_vmax_user_{i}_{l}_{k}")

        base.add_repair_cost_mccormick(
            model, ctx.z[i, l, k], mu_prev, rho_il, x_m_ilk, tau_il, name=f"rf_z_{i}_{l}_{k}",
        )

        ctx.mu[i, l, k] = mu[k]
        ctx.v[i, l, k] = v[k]

    base.add_loop_constraint(model, ctx.mu, i, l, ctx.H, name="rf_mu")
    base.add_loop_constraint(model, ctx.v, i, l, ctx.H, name="rf_v")
