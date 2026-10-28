"""Gamma degradation model (ARD1 and ARA1).

Tracks the accumulated shape parameter alpha (not the mean directly):
alpha_0 = mu_0/beta, alpha_new = mu_new/beta, and the mean is recovered as
mu = alpha*beta. Reference: spec/spec.tex Section 5.5 (state dynamics) and its
"Reliability Constraint (Gamma, precomputed linear bound)" /
"Loop Constraint (Gamma, hard)" subsections.
"""

import warnings

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaincc

from fleet_management.maintenance import ara1, ard1
from fleet_management.models import base

MODEL_NAME = "gamma"


def validate_inputs(mask, mu_0, alpha_inc, tau, rho, beta, maintenance_type):
    """Consistency checks restricted to the (i, l) entries where mask is True.

    Both ARD1 and ARA1 are valid maintenance types here, so there is no
    maintenance_type restriction to enforce (unlike Gaussian/IG/Rainflow).
    """
    for i, l in np.argwhere(mask):
        i, l = int(i), int(l)
        if maintenance_type[i][l] not in ("ARD1", "ARA1"):
            raise ValueError(
                f"Gamma component ({i}, {l}): maintenance_type must be 'ARD1' or "
                f"'ARA1' (got '{maintenance_type[i][l]}')."
            )
        if not (0 < rho[i, l] <= 1):
            raise ValueError(f"Gamma component ({i}, {l}): rho must be in (0, 1].")
        if beta[i, l] <= 0:
            raise ValueError(f"Gamma component ({i}, {l}): beta must be positive.")
        if not (mu_0[i, l] < tau[i, l]):
            raise ValueError(f"Gamma component ({i}, {l}): mu_0 must be < tau.")
        if not np.all(alpha_inc[i, :, l, :] > 0):
            raise ValueError(f"Gamma component ({i}, {l}): alpha_inc must be positive.")


def _reliability_gap(alpha_hat: float, beta: float, tau: float, epsilon: float) -> float:
    """Q(alpha_hat, tau/beta) - epsilon; its root in alpha_hat is the cap."""
    return gammaincc(alpha_hat, tau / beta) - epsilon


def solve_alpha_hat(beta: float, tau: float, epsilon: float) -> float:
    """Offline precomputation of the Gamma reliability cap alpha_hat.

    Pr(D > tau) = Q(alpha, tau/beta) is monotone increasing in alpha (spec
    remark), so _reliability_gap is also monotone increasing, and the cap is
    its unique root in (0, tau/beta] -- using scipy.optimize.brentq per the
    spec's implementation note (gammaincc solves for the wrong argument to
    invert directly).
    """
    alpha_bar = tau / beta
    lo, hi = 1e-9 * alpha_bar, alpha_bar * (1 - 1e-9)
    g_lo = _reliability_gap(lo, beta, tau, epsilon)
    g_hi = _reliability_gap(hi, beta, tau, epsilon)
    if g_lo >= 0:
        # By Markov's inequality Pr(D > tau) <= E[D]/tau = alpha*beta/tau -> 0
        # as alpha -> 0, for any beta > 0, so this should be unreachable; kept
        # as a numerical safety net (mirrors inverse_gaussian.solve_mu_bar).
        raise ValueError(
            f"Gamma reliability target unachievable for beta={beta}, tau={tau}, "
            f"epsilon={epsilon}: no alpha in (0, tau/beta] satisfies "
            "Pr(D > tau) <= epsilon."
        )
    if g_hi <= 0:
        # Per spec: if the root would exceed tau/beta, the reliability target
        # is unachievable-in-the-binding-sense (non-binding up to the big-M
        # cap itself); raise a diagnostic warning and cap there.
        warnings.warn(
            f"Gamma reliability cap is non-binding for beta={beta}, tau={tau}, "
            f"epsilon={epsilon}: alpha_hat capped at tau/beta={alpha_bar}.",
            UserWarning,
        )
        return hi
    return brentq(_reliability_gap, lo, hi, args=(beta, tau, epsilon))


def build_component(ctx, i, l):
    """Add all Gamma variables/constraints for component (i, l) to ctx.model."""
    model = ctx.model
    two_h, M = ctx.two_h, ctx.M
    tau_il = float(ctx.tau[i, l])
    rho_il = float(ctx.rho[i, l])
    beta_il = float(ctx.beta[i, l])
    alpha_bar = tau_il / beta_il
    alpha_0_il = float(ctx.mu_0[i, l]) / beta_il
    alpha_new_il = float(ctx.mu_new[i, l]) / beta_il
    alpha_hat_il = solve_alpha_hat(beta_il, tau_il, ctx.epsilon)
    is_ara1 = ctx.maintenance_type[i][l] == "ARA1"

    alpha = model.addVars(two_h, lb=0.0, name=f"alpha_{i}_{l}")
    alpha_last = model.addVars(two_h, lb=0.0, name=f"alpha_last_{i}_{l}") if is_ara1 else None

    for k in range(two_h):
        alpha_prev = alpha_0_il if k == 0 else alpha[k - 1]
        x_m_ilk, x_r_ilk = ctx.x_m[i, l, k], ctx.x_r[i, l, k]
        delta_alpha = base.mission_delta(ctx.x, ctx.alpha_inc, i, l, k, M)

        if is_ara1:
            alpha_last_prev = alpha_0_il if k == 0 else alpha_last[k - 1]
            ara1.update_anchor(
                model, alpha_last[k], alpha_prev, alpha_last_prev, alpha_new_il,
                x_m_ilk, x_r_ilk, alpha_bar, name=f"gamma_anchor_{i}_{l}_{k}",
            )
            ara1.accumulate_and_repair(
                model, alpha[k], alpha_prev, delta_alpha, alpha_last_prev, rho_il,
                x_m_ilk, x_r_ilk, alpha_bar, name=f"gamma_alpha_{i}_{l}_{k}",
            )
        else:
            ard1.accumulate_and_repair(
                model, alpha[k], alpha_prev, delta_alpha, 1 - rho_il,
                x_m_ilk, x_r_ilk, alpha_bar, name=f"gamma_alpha_{i}_{l}_{k}",
            )
        base.add_replacement_linear(
            model, alpha[k], alpha_new_il, x_r_ilk, alpha_bar, name=f"gamma_alpha_{i}_{l}_{k}",
        )

        model.addConstr(alpha[k] <= alpha_hat_il, name=f"gamma_rel_{i}_{l}_{k}")

        anchor_for_z = alpha_last_prev if is_ara1 else 0.0
        base.add_repair_cost_mccormick(
            model, ctx.z[i, l, k], alpha_prev * beta_il, rho_il, x_m_ilk, tau_il,
            name=f"gamma_z_{i}_{l}_{k}", anchor_prev_expr=anchor_for_z * beta_il,
        )

        # mu = alpha*beta recovered here (spec post-processing substitution);
        # storing the linear expression (not a raw Var) into the shared ctx.mu
        # lets damage-regularisation and the mu-loop-constraint helpers work
        # unchanged, since comparisons/sums scale consistently by beta > 0.
        ctx.mu[i, l, k] = alpha[k] * beta_il
        if is_ara1:
            ctx.alpha_last[i, l, k] = alpha_last[k]

    base.add_loop_constraint(model, ctx.mu, i, l, ctx.H, name="gamma_mu")
    if is_ara1:
        base.add_loop_constraint(model, ctx.alpha_last, i, l, ctx.H, name="gamma_alpha_last")
