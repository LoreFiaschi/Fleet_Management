"""Prototype of the inverse Gaussian mean cap (spec, "Inverse Gaussian Model", Eq. ig_row)
and its numeric checks. Usage: python ig_cap.py

The functions ig_unit_quantile and ig_mean_cap below are the code that the plan puts in
config.py (section 2 of merge/wp6_plan_giw.md), copied verbatim.
"""
import math

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import log_ndtr
from scipy.stats import invgauss


# ---------------------------------------------------------------- code for config.py
def _ig_unit_log_cdf(x: float, eta: float) -> float:
    """ln F(x) of Z ~ IG(1, 1/eta), x > 0 (spec, "Inverse Gaussian Model")."""
    r = 1.0 / math.sqrt(eta * x)
    return float(np.logaddexp(log_ndtr(r * (x - 1.0)), 2.0 / eta + log_ndtr(-r * (x + 1.0))))


def _ig_unit_log_sf(x: float, eta: float) -> float:
    """ln (1 - F(x)) of Z ~ IG(1, 1/eta), x > 0, without the cancellation of 1 - F."""
    r = 1.0 / math.sqrt(eta * x)
    first = float(log_ndtr(-r * (x - 1.0)))
    second = 2.0 / eta + float(log_ndtr(-r * (x + 1.0)))
    return first + math.log1p(-math.exp(second - first))


def ig_unit_quantile(eta: float, epsilon: float) -> float:
    """The (1 - epsilon)-quantile q of Z ~ IG(1, 1/eta): Pr(Z > q) = epsilon.

    Root of ln Pr(Z > e^u) = ln epsilon in u = ln q, by brentq on [-50, ln b], with
    b = 1 + sqrt(eta (2/epsilon - 1)): Var Z = eta, so Pr(Z > b) <= epsilon/2 (Cantelli).
    """
    target = math.log(epsilon)

    def gap(u):
        return _ig_unit_log_sf(math.exp(u), eta) - target

    lo, hi = -50.0, math.log(1.0 + math.sqrt(eta * (2.0 / epsilon - 1.0)))
    if not (gap(lo) > 0.0 > gap(hi)):
        raise ValueError(f"no quantile bracket for eta={eta}, epsilon={epsilon}")
    return math.exp(brentq(gap, lo, hi, xtol=1e-14, rtol=1e-15, maxiter=200))


def ig_mean_cap(eta: float, tau: float, epsilon: float) -> float:
    """Mean cap mu_bar = tau / E[Z | Z > q] of an inverse Gaussian cell (Eq. ig_row).

    E[Z; Z > q] = F(1/q), so mu_bar = tau * epsilon / F(1/q), computed in log space.
    """
    q = ig_unit_quantile(eta, epsilon)
    return tau * math.exp(math.log(epsilon) - _ig_unit_log_cdf(1.0 / q, eta))
# ---------------------------------------------------------------- end of code for config.py


def pdf(z, eta):
    return (2.0 * math.pi * eta * z ** 3) ** -0.5 * math.exp(-(z - 1.0) ** 2 / (2.0 * eta * z))


def tail_mean_quadrature(q, eta):
    # E[Z; Z > q] = int_q^inf z f(z) dz, on u = ln z: int z^2 f(e^u) du
    value, _ = quad(lambda u: math.exp(2 * u) * pdf(math.exp(u), eta), math.log(q), math.log(q) + 60,
                    limit=500, epsabs=0, epsrel=1e-12)
    return value


def lore_cap(eta, tau, eps):
    return tau / ig_unit_quantile(eta, eps)


if __name__ == "__main__":
    worst_q = worst_tail = 0.0
    grid_eta = [1e-3, 1e-2, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 1e3]
    grid_eps = [1e-6, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.3, 0.5, 0.9]
    for eta in grid_eta:
        for eps in grid_eps:
            q = ig_unit_quantile(eta, eps)
            q_sp = float(invgauss.isf(eps, mu=eta, scale=1.0 / eta))
            worst_q = max(worst_q, abs(q - q_sp) / q_sp)
            closed = math.exp(_ig_unit_log_cdf(1.0 / q, eta))
            numeric = tail_mean_quadrature(q, eta)
            worst_tail = max(worst_tail, abs(closed - numeric) / numeric)
            cap = ig_mean_cap(eta, 1.0, eps)
            assert 0.0 < cap < 1.0, (eta, eps, cap)
            assert cap <= 1.0 / q, (eta, eps)        # E[Z | Z > q] >= q
    print(f"grid {len(grid_eta)} x {len(grid_eps)}: max rel. diff of q vs invgauss.isf = {worst_q:.2e}")
    print(f"max rel. diff of the tail mean F(1/q) vs quadrature = {worst_tail:.2e}")

    print("\ncost of the CVaR cap against lore's cap tau/q, eps = 0.01 (decisions.md: 9.8, 18.2, 22.5 %)")
    for eta in (0.1, 0.5, 1.0):
        print(f"  eta={eta}: {100 * (1 - ig_mean_cap(eta, 1.0, 0.01) / lore_cap(eta, 1.0, 0.01)):.1f} %")

    rng = np.random.default_rng(1)
    print("\nMonte Carlo check of E[Z | Z > q] (2e6 draws)")
    for eta, eps in ((0.5, 0.1), (2.0, 0.05)):
        z = invgauss.rvs(eta, scale=1.0 / eta, size=2_000_000, random_state=rng)
        q = ig_unit_quantile(eta, eps)
        print(f"  eta={eta} eps={eps}: tau/cap = {1.0 / ig_mean_cap(eta, 1.0, eps):.5f}, "
              f"sample mean of Z > q = {z[z > q].mean():.5f}, sample Pr(Z > q) = {(z > q).mean():.5f}")

    print("\nreference values (tau, eta, epsilon) -> q, mu_bar, lore's tau/q")
    for tau, eta, eps in ((1.0, 0.5, 0.1), (1.0, 0.1, 0.01), (2.0, 2.0, 0.05), (1.0, 1.0, 0.2)):
        q = ig_unit_quantile(eta, eps)
        print(f"  ({tau}, {eta}, {eps}): q = {q:.10f}, mu_bar = {ig_mean_cap(eta, tau, eps):.10f}, "
              f"tau/q = {tau / q:.10f}")
