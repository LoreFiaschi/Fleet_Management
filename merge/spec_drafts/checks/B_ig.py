"""Writer B, spec v1.0, inverse Gaussian section: spot checks.

Standard library only. Run: python -I B_ig.py > B_ig_output.txt
Z ~ IG(1, lam), lam = 1/eta. tau = 1.

Checks:
 (1) Closed form of the tail mean: E[Z; Z > q] = F(1/q), against quadrature of z f(z) on (q, inf).
 (2) Cap table: q, C = E[Z | Z > q] = F(1/q)/eps, cap 1/C, and the ratio q/C to the quantile cap 1/q;
     against inverse_gaussian.md S4.1 (eps = 0.01: 0.93, 0.86, 0.78, 0.67, 0.59 at eta = 0.05, 0.25, 1, 4, 10).
 (3) Stop-loss bound at the cap: mu E(Z - q)_+ / (tau - mu q) = eps at mu = 1/C.
 (4) Boundary case: eta -> 0 gives C -> 1 (cap -> tau).
 (5) Convex order spot check: X1 = a1 Z1, X2 = a2 Z2 independent; E(X1 + X2 - d)_+ <= E((a1 + a2) Z - d)_+
     (Monte Carlo against the closed form); comonotone case Z1 = Z2: equality.
 (6) The quantile cap tau/q is not safe for sums: eta = 10, eps = 0.1, three equal independent increments
     with total mean tau/q (inverse_gaussian.md: true tail 0.1180).
 (7) Density of IG(mu, mu/eta) integrates to 1 (the spec's form), checked at mu = 0.25 and 4.
"""
import math
import random

random.seed(26102026)


def Phi(z):
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def log_Phi(z):
    """ln Phi(z); asymptotic series for z < -20, where erfc underflows."""
    if z > -20.0:
        return math.log(Phi(z))
    y = -z
    series = 1.0 - 1.0 / y ** 2 + 3.0 / y ** 4 - 15.0 / y ** 6 + 105.0 / y ** 8 - 945.0 / y ** 10
    return -0.5 * y * y - math.log(y) - 0.5 * math.log(2.0 * math.pi) + math.log(series)


def second_term(x, lam):
    """exp(2 lam) Phi(-sqrt(lam/x)(x+1)), computed in log space."""
    r = math.sqrt(lam / x)
    return math.exp(2.0 * lam + log_Phi(-r * (x + 1.0)))


def F(x, lam):
    """CDF of IG(1, lam)."""
    if x <= 0.0:
        return 0.0
    r = math.sqrt(lam / x)
    return Phi(r * (x - 1.0)) + second_term(x, lam)


def S(x, lam):
    """Survival of IG(1, lam), computed without 1 - F where possible."""
    r = math.sqrt(lam / x)
    return Phi(-r * (x - 1.0)) - second_term(x, lam)


def dens(x, mu, lam):
    return math.sqrt(lam / (2.0 * math.pi * x ** 3)) * math.exp(-lam * (x - mu) ** 2 / (2.0 * mu * mu * x))


def quantile(eps, lam):
    """q with S(q) = eps (bisection in log x)."""
    lo, hi = math.log(1e-8), math.log(1e8)
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if S(math.exp(mid), lam) > eps:
            lo = mid
        else:
            hi = mid
    return math.exp(0.5 * (lo + hi))


def tail_mean_quad(q, lam, n=400000):
    """int_q^inf z f(z) dz by the trapezoid rule in y = ln z, up to z = q e^25."""
    y0, y1 = math.log(q), math.log(q) + 25.0
    h = (y1 - y0) / n
    s = 0.0
    for k in range(n + 1):
        z = math.exp(y0 + k * h)
        g = z * dens(z, 1.0, lam) * z
        s += g * (0.5 if k in (0, n) else 1.0)
    return s * h


def sample_ig(mu, lam):
    """Michael, Schucany and Haas sampler."""
    nu = random.gauss(0.0, 1.0)
    y = nu * nu
    x = mu + mu * mu * y / (2.0 * lam) - mu / (2.0 * lam) * math.sqrt(4.0 * mu * lam * y + mu * mu * y * y)
    if random.random() <= mu / (mu + x):
        return x
    return mu * mu / x


def stop_loss_model(mu, d, lam):
    """E(mu Z - d)_+ = mu [F(1/q') - q' S(q')], q' = d/mu (closed form of section (1))."""
    qq = d / mu
    return mu * (F(1.0 / qq, lam) - qq * S(qq, lam))


if __name__ == "__main__":
    print("(1) closed form F(1/q) against quadrature of z f(z) on (q, inf)")
    for eta in (0.05, 0.25, 1.0, 10.0):
        lam = 1.0 / eta
        for eps in (0.001, 0.01, 0.1):
            q = quantile(eps, lam)
            cf, qu = F(1.0 / q, lam), tail_mean_quad(q, lam)
            print("  eta=%g eps=%g: q=%.6f  F(1/q)=%.10f  quadrature=%.10f  rel.diff=%.1e" % (
                eta, eps, q, cf, qu, abs(cf - qu) / qu))
    print("(2) cap table (tau = 1): cap = 1/C, quantile cap = 1/q, ratio q/C")
    for eps in (0.001, 0.01, 0.1):
        row = []
        for eta in (0.05, 0.25, 1.0, 4.0, 10.0):
            lam = 1.0 / eta
            q = quantile(eps, lam)
            C = F(1.0 / q, lam) / eps
            row.append("eta=%g: cap=%.4f q-cap=%.4f ratio=%.3f" % (eta, 1.0 / C, 1.0 / q, q / C))
        print("  eps=%g: " % eps + "; ".join(row))
    print("(3) stop-loss bound at the cap equals eps")
    for eta, eps in ((0.25, 0.01), (1.0, 0.1), (10.0, 0.001)):
        lam = 1.0 / eta
        q = quantile(eps, lam)
        C = F(1.0 / q, lam) / eps
        mu = 1.0 / C
        EZq = F(1.0 / q, lam) - q * S(q, lam)
        print("  eta=%g eps=%g: bound/eps = %.10f" % (eta, eps, mu * EZq / (1.0 - mu * q) / eps))
    print("(4) boundary case eta -> 0: C -> 1")
    for eta in (0.1, 0.01, 0.001, 0.0001):
        lam = 1.0 / eta
        q = quantile(0.01, lam)
        print("  eta=%g: q=%.6f  C=%.6f" % (eta, q, F(1.0 / q, lam) / 0.01))
    print("(5) convex order, Monte Carlo (10^6 draws): E(a1 Z1 + a2 Z2 - d)_+ against E((a1+a2) Z - d)_+")
    for eta, a1, a2 in ((0.25, 0.3, 0.2), (4.0, 0.1, 0.4)):
        lam = 1.0 / eta
        N = 1000000
        draws_ind = [(sample_ig(1.0, lam), sample_ig(1.0, lam)) for _ in range(N)]
        for d in (0.3, 0.6, 1.0):
            ind = sum(max(a1 * z1 + a2 * z2 - d, 0.0) for z1, z2 in draws_ind) / N
            com = sum(max((a1 + a2) * z1 - d, 0.0) for z1, _ in draws_ind) / N
            exact = stop_loss_model(a1 + a2, d, lam)
            print("  eta=%g a=(%g,%g) d=%g: independent %.6f <= comonotone MC %.6f, closed form %.6f" % (
                eta, a1, a2, d, ind, com, exact))
    print("(6) quantile cap 1/q with three equal independent increments: eta = 10, eps = 0.1")
    lam = 1.0 / 10.0
    q = quantile(0.1, lam)
    mubar = 1.0 / q
    a = mubar / 3.0
    N = 2000000
    hits = 0
    for _ in range(N):
        s = sum(a * sample_ig(1.0, lam) for _ in range(3))
        if s > 1.0:
            hits += 1
    p = hits / N
    print("  1/q = %.4f; true tail (MC, 2e6 draws) = %.4f +- %.4f; model tail = 0.1" % (
        mubar, p, math.sqrt(p * (1 - p) / N)))
    print("(7) integral of the density of IG(mu, mu/eta), eta = 0.5")
    for mu in (0.25, 4.0):
        lam = mu / 0.5
        n = 400000
        y0, y1 = math.log(mu) - 15.0, math.log(mu) + 15.0
        h = (y1 - y0) / n
        s = sum(dens(math.exp(y0 + k * h), mu, lam) * math.exp(y0 + k * h) for k in range(n + 1)) * h
        print("  mu=%g: %.8f" % (mu, s))
    print("(8) ARD1 weights: D = G + (1-rho) Delta, G = mu0 Z0 split by the anchor g0, Delta = a Z1 (Monte Carlo)")
    print("    against E(m Z - d)_+, m = E D; tail at the CVaR cap; the cap rescaled so that m = cap")
    for eta, eps, rho in ((1.0, 0.01, 0.5), (10.0, 0.1, 0.3)):
        lam = 1.0 / eta
        q = quantile(eps, lam)
        cap = eps / F(1.0 / q, lam)
        mu0, g0, a = 0.4, 0.15, 0.5
        m_raw = g0 + (1 - rho) * (mu0 - g0) + (1 - rho) * a
        scale = cap / m_raw                      # put the true mean at the cap (tau = 1)
        N = 1000000
        ds = []
        for _ in range(N):
            z0, z1 = sample_ig(1.0, lam), sample_ig(1.0, lam)
            ds.append(scale * (g0 * z0 + (1 - rho) * (mu0 - g0) * z0 + (1 - rho) * a * z1))
        for d in (0.3 * cap, cap, 2.0 * cap):
            lhs = sum(max(x - d, 0.0) for x in ds) / N
            print("  eta=%g eps=%g rho=%g d=%.4f: E(D-d)_+ = %.6f <= E(cap Z - d)_+ = %.6f" % (
                eta, eps, rho, d, lhs, stop_loss_model(cap, d, lam)))
        p = sum(1 for x in ds if x > 1.0) / N
        print("  eta=%g eps=%g: P(D > tau) at mean = cap: %.5f +- %.5f <= eps" % (eta, eps, p, math.sqrt(p * (1 - p) / N)))
