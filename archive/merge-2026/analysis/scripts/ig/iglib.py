"""Standard-library tools for inverse Gaussian (IG) laws.

IG(mu, lam): density sqrt(lam/(2 pi x^3)) exp(-lam (x-mu)^2 / (2 mu^2 x)), x > 0.
Mean mu, variance mu^3/lam.
"""
import math
import random

SQ2 = math.sqrt(2.0)
LOG_SQ2PI = 0.5 * math.log(2.0 * math.pi)


def npdf(t):
    return math.exp(-0.5 * t * t - LOG_SQ2PI)


def ncdf(t):
    return 0.5 * math.erfc(-t / SQ2)


def mills(t):
    """R(t) = Phi(-t) / phi(t), for t >= 0 (continued fraction for large t)."""
    if t < 20.0:
        return 0.5 * math.erfc(t / SQ2) / npdf(t)
    # Laplace continued fraction R(t) = 1/(t+1/(t+2/(t+3/(t+...))))
    f = t
    for k in range(80, 0, -1):
        f = t + k / f
    return 1.0 / f


def ig_ab(x, mu, lam):
    r = math.sqrt(lam / x)
    return r * (x / mu - 1.0), r * (x / mu + 1.0)


def ig_sf(x, mu, lam):
    """P(X > x), computed without cancellation: S = phi(a) (R(a) - R(b))."""
    if x <= 0.0:
        return 1.0
    a, b = ig_ab(x, mu, lam)
    if a > 0.0:
        return npdf(a) * (mills(a) - mills(b))
    return ncdf(-a) - npdf(a) * mills(b)


def ig_cdf(x, mu, lam):
    """P(X <= x) = Phi(a) + phi(a) R(b)."""
    if x <= 0.0:
        return 0.0
    a, b = ig_ab(x, mu, lam)
    if a < 0.0:
        return npdf(a) * (mills(-a) + mills(b))
    return 1.0 - ig_sf(x, mu, lam)


def ig_pdf(x, mu, lam):
    if x <= 0.0:
        return 0.0
    e = lam * (x - mu) ** 2 / (2.0 * mu * mu * x)
    if e > 700.0:
        return 0.0
    return math.sqrt(lam / (2.0 * math.pi * x ** 3)) * math.exp(-e)


def ig_sample(mu, lam, rng):
    """Michael-Schucany-Haas sampler, in a cancellation-free form."""
    nu = rng.gauss(0.0, 1.0)
    w = mu * nu * nu / (2.0 * lam)
    x = mu / (1.0 + w + math.sqrt(w * w + 2.0 * w))
    if rng.random() <= mu / (mu + x):
        return x
    return mu * mu / x


def bisect(f, lo, hi, it=200):
    """Root of an increasing function f on [lo, hi]."""
    flo = f(lo)
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if (f(mid) > 0.0) == (flo > 0.0):
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-15 * max(1.0, abs(hi)):
            break
    return 0.5 * (lo + hi)


def mu_bar_eta(eta, tau, eps):
    """lore's cap: largest mu with P(IG(mu, mu/eta) > tau) <= eps, capped below tau."""
    g = lambda m: ig_sf(tau, m, m / eta) - eps
    hi = tau * (1.0 - 1e-9)
    if g(hi) <= 0.0:
        return hi
    return bisect(g, 1e-9 * tau, hi)


def simpson(vals, h):
    n = len(vals) - 1
    assert n % 2 == 0
    s = vals[0] + vals[-1] + 4.0 * sum(vals[1:n:2]) + 2.0 * sum(vals[2:n:2])
    return s * h / 3.0


def sum_tail(terms, tau, n=2000):
    """P(sum of independent IG terms > tau). terms = [(mu, lam), ...].

    Recursion T_k = T_{k-1} + int_0^tau g_{k-1}(y) S_k(tau - y) dy, with g the
    density of the partial sum on a grid of [0, tau] (trapezoid convolution of
    smooth densities that vanish with all derivatives at 0), Simpson outer rule.
    """
    h = tau / n
    xs = [i * h for i in range(n + 1)]
    mu1, lam1 = terms[0]
    T = ig_sf(tau, mu1, lam1)
    g = [ig_pdf(x, mu1, lam1) for x in xs]
    for k, (mu, lam) in enumerate(terms[1:], start=1):
        sk = [ig_sf(tau - y, mu, lam) for y in xs]
        T += simpson([g[i] * sk[i] for i in range(n + 1)], h)
        if k < len(terms) - 1:
            f = [ig_pdf(x, mu, lam) for x in xs]
            g = [h * sum(g[j] * f[i - j] for j in range(i + 1)) for i in range(n + 1)]
    return T


def mc_tail(terms, tau, n, seed=1):
    rng = random.Random(seed)
    c = 0
    for _ in range(n):
        s = 0.0
        for mu, lam in terms:
            s += ig_sample(mu, lam, rng)
        if s > tau:
            c += 1
    p = c / n
    return p, math.sqrt(p * (1 - p) / n)
