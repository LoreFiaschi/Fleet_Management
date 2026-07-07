"""Inverse Gaussian degradation model (ARD1 only).

Only the mean mu is tracked; the reliability constraint reduces, offline (before
the Gurobi model is built), to a precomputed linear cap on mu. Reference:
spec/spec.tex Section 5.4 (state dynamics) and its "Reliability Constraint (IG)"
subsection, Eq. (ig_mubar).
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import norm

from fleet_management.maintenance.ard1 import accumulate_and_repair
from fleet_management.models import base

MODEL_NAME = "inverse_gaussian"


def validate_inputs(mask, mu_0, mu_inc, tau, rho, eta, maintenance_type):
    for i, l in np.argwhere(mask):
        i, l = int(i), int(l)
        if maintenance_type[i][l] != "ARD1":
            raise ValueError(
                f"Inverse Gaussian component ({i}, {l}): only 'ARD1' maintenance "
                f"is supported (got '{maintenance_type[i][l]}')."
            )
        if not (0 < rho[i, l] <= 1):
            raise ValueError(f"IG component ({i}, {l}): rho must be in (0, 1].")
        if eta[i, l] <= 0:
            raise ValueError(f"IG component ({i}, {l}): eta must be positive.")
        if not (mu_0[i, l] < tau[i, l]):
            raise ValueError(f"IG component ({i}, {l}): mu_0 must be < tau.")
        if not np.all(mu_inc[i, :, l, :] < tau[i, l]):
            raise ValueError(f"IG component ({i}, {l}): mu_inc must be < tau.")


def _reliability_gap(mu_bar: float, eta: float, tau: float, epsilon: float) -> float:
    """LHS - (1 - epsilon) of Eq. (ig_mubar); its root in mu_bar is the cap.

    Computed in log-space (per spec's numerical-implementation note) to avoid
    overflow in exp(2/eta) for small eta.
    """
    ratio = np.sqrt(mu_bar / (eta * tau))
    a = ratio * (tau / mu_bar - 1)
    b = ratio * (tau / mu_bar + 1)
    log_term1 = norm.logcdf(a)
    log_term2 = 2.0 / eta + norm.logcdf(-b)
    lhs = np.exp(logsumexp([log_term1, log_term2]))
    return lhs - (1 - epsilon)


def solve_mu_bar(eta: float, tau: float, epsilon: float) -> float:
    """Offline precomputation of the IG reliability cap mu_bar (Eq. ig_mubar).

    Pr(D > tau) is monotone increasing in mu (spec remark), so
    ``_reliability_gap(mu) = epsilon - Pr(D > tau; mu)`` is monotone
    *decreasing* in mu, and the cap is its unique root in (0, tau).
    """
    lo, hi = 1e-9 * tau, tau * (1 - 1e-9)
    g_lo = _reliability_gap(lo, eta, tau, epsilon)
    g_hi = _reliability_gap(hi, eta, tau, epsilon)
    if g_lo <= 0:
        # Even at mu_bar -> 0 the failure probability already exceeds epsilon:
        # the reliability target is unachievable for this (eta, tau, epsilon).
        raise ValueError(
            f"Inverse Gaussian reliability target unachievable for eta={eta}, "
            f"tau={tau}, epsilon={epsilon}: no mu in (0, tau) satisfies "
            "Pr(D > tau) <= epsilon."
        )
    if g_hi >= 0:
        # The reliability constraint is still satisfied even at mu_bar = tau:
        # the cap is non-binding, so it can be taken as tau itself.
        return hi
    return brentq(_reliability_gap, lo, hi, args=(eta, tau, epsilon))


def build_component(ctx, i, l):
    """Add all IG variables/constraints for component (i, l) to ctx.model."""
    model = ctx.model
    two_h, M = ctx.two_h, ctx.M
    tau_il = float(ctx.tau[i, l])
    rho_il = float(ctx.rho[i, l])
    eta_il = float(ctx.eta[i, l])
    mu_new_il = float(ctx.mu_new[i, l])
    mu_bar_il = solve_mu_bar(eta_il, tau_il, ctx.epsilon)

    mu = model.addVars(two_h, lb=0.0, name=f"mu_ig_{i}_{l}")

    for k in range(two_h):
        mu_prev = float(ctx.mu_0[i, l]) if k == 0 else mu[k - 1]
        delta_mu = base.mission_delta(ctx.x, ctx.mu_inc, i, l, k, M)
        x_m_ilk, x_r_ilk = ctx.x_m[i, l, k], ctx.x_r[i, l, k]

        accumulate_and_repair(
            model, mu[k], mu_prev, delta_mu, 1 - rho_il,
            x_m_ilk, x_r_ilk, tau_il, name=f"ig_mu_{i}_{l}_{k}",
        )
        base.add_replacement_linear(model, mu[k], mu_new_il, x_r_ilk, tau_il, name=f"ig_mu_{i}_{l}_{k}")
        model.addConstr(mu[k] <= mu_bar_il, name=f"ig_rel_{i}_{l}_{k}")
        base.add_repair_cost_mccormick(
            model, ctx.z[i, l, k], mu_prev, rho_il, x_m_ilk, tau_il, name=f"ig_z_{i}_{l}_{k}",
        )

        ctx.mu[i, l, k] = mu[k]

    base.add_loop_constraint(model, ctx.mu, i, l, ctx.H, name="ig_mu")
