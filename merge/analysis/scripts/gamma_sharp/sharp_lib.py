"""Tools for the sharp Gamma cap over the class C(m) (standard library only).

Units: rate b = 1 (scale), so t stands for b*t and m for b*m = shape of the single Gamma.
A member of C(m): S = delta + sum_i c_i * Y_i, Y_i ~ Gamma(m_i / c_i, 1), c_i in (0, 1],
sum_i m_i + delta = m (delta >= 0 is the drift, the limit c -> 0 at fixed mean).
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gamma_lib import (gammaQ, shape_for_tail, sum_tail_fast, stoploss_bound,  # noqa: E402
                       chernoff_gamma, cap_for_bound)

EULER = 0.5772156649015329


def E1(x):
    """Exponential integral E1(x) = int_x^inf e^-u / u du, x > 0."""
    if x <= 0.0:
        return float("inf")
    if x <= 1.0:
        s, term, k = 0.0, 1.0, 0
        while True:
            k += 1
            term *= -x / k
            add = -term / k
            s += add
            if abs(add) < 1e-17 * abs(s) + 1e-300:
                break
        return -EULER - math.log(x) + s
    # continued fraction (modified Lentz), valid for x > 1
    b = x + 1.0
    c = 1.0 / 1e-300
    d = 1.0 / b
    h = d
    for i in range(1, 100000):
        an = -i * i
        b += 2.0
        d = 1.0 / (an * d + b)
        c = b + an / c
        de = c * d
        h *= de
        if abs(de - 1.0) < 1e-16:
            break
    return h * math.exp(-x)


def tanh_sinh(g, a, b, level=7, kmax=None):
    """int_a^b g(x, dist_a, dist_b) dx by tanh-sinh; g receives the distances to both ends
    (computed without cancellation), so that end singularities can be evaluated accurately."""
    h = 2.0 ** (-level)
    half = 0.5 * (b - a)
    s = 0.0
    k = 0
    if kmax is None:
        kmax = int(6.0 / h)
    while k <= kmax:
        for sgn in ((1,) if k == 0 else (1, -1)):
            tk = sgn * k * h
            u = 0.5 * math.pi * math.sinh(tk)
            cu = math.cosh(u)
            w = h * 0.5 * math.pi * math.cosh(tk) / (cu * cu)
            # 1 -/+ tanh(u) without cancellation: 1 - tanh(u) = 2 / (1 + e^{2u})
            if u >= 0:
                dist_b = half * 2.0 / (1.0 + math.exp(2.0 * u))   # distance to b
                dist_a = 2.0 * half - dist_b
            else:
                dist_a = half * 2.0 / (1.0 + math.exp(-2.0 * u))
                dist_b = 2.0 * half - dist_a
            if dist_a <= 0.0 or dist_b <= 0.0:
                continue
            x = a + dist_a
            s += half * w * g(x, dist_a, dist_b)
        k += 1
        if w < 1e-20:
            break
    return s


def gamma_density(m, z):
    if z <= 0.0:
        return 0.0
    return math.exp((m - 1.0) * math.log(z) - z - math.lgamma(m))


def phi(c, m, t, level=8):
    """phi(c) = (1/c) E[E1((t - Z)/c); Z < t], Z ~ Gamma(m, 1); phi(0) = density of Z at t.
    D(c) = phi(c) - phi(1) is the derivative of P(S > t) when mean mass moves from the
    single Gamma(m, 1) to a term c*Gamma(eta/c, 1) (notes, T1)."""
    if c <= 0.0:
        return gamma_density(m, t)
    if m < 1.0:
        # z = t u^(1/m); f(z) dz = t^m / Gamma(m+1) e^{-z} du
        pref = math.exp(m * math.log(t) - math.lgamma(m + 1.0))

        def g(u, da, db):
            # u in [0, 1]; db = 1 - u
            lu = math.log1p(-db) if db < 0.5 else math.log(u)
            z = t * math.exp(lu / m)
            tz = -t * math.expm1(lu / m)      # t - z
            if tz <= 0.0:
                return 0.0
            return math.exp(-z) * E1(tz / c)

        return pref * tanh_sinh(g, 0.0, 1.0, level) / c
    # m >= 1: integrate in z on pieces of [0, t]
    sd = math.sqrt(m)
    npieces = max(4, int(t / max(0.25 * sd, 1e-3)) + 1)
    npieces = min(npieces, 400)
    edges = [t * i / npieces for i in range(npieces + 1)]
    s = 0.0
    for i in range(npieces):
        a, b = edges[i], edges[i + 1]
        last = (i == npieces - 1)

        def g(z, da, db, b=b, last=last):
            tz = db if last else (t - z)
            return gamma_density(m, z) * E1(tz / c)

        s += tanh_sinh(g, a, b, level=6)
    return s / c


def A_hat(t, eps):
    return shape_for_tail(eps, 1.0, t)


def A_icx(t, eps):
    """S5 cap of gamma.md: B(A) = min(stop-loss bound, Chernoff bound) <= eps."""
    return max(cap_for_bound(stoploss_bound, 1.0, t, eps), cap_for_bound(chernoff_gamma, 1.0, t, eps))


def member_tail(terms, delta, t, mass_tol=1e-15):
    """P(delta + sum c_i Gamma(m_i/c_i, 1) > t) for terms [(c_i, m_i)]; returns (lower, upper)."""
    if delta >= t:
        return 1.0, 1.0
    comps = [(mi / ci, 1.0 / ci) for ci, mi in terms if mi > 0.0]
    if not comps:
        return 0.0, 0.0
    return sum_tail_fast(comps, t - delta, mass_tol=mass_tol)


def nelder_mead(f, x0, step=0.3, iters=400, tol=1e-12):
    """Minimise f by Nelder-Mead (unconstrained)."""
    n = len(x0)
    pts = [list(x0)]
    for i in range(n):
        p = list(x0)
        p[i] += step
        pts.append(p)
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = sorted(range(n + 1), key=lambda i: vals[i])
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) <= tol * (abs(vals[0]) + 1e-300):
            break
        cen = [sum(p[j] for p in pts[:-1]) / n for j in range(n)]
        xr = [cen[j] + (cen[j] - pts[-1][j]) for j in range(n)]
        fr = f(xr)
        if fr < vals[0]:
            xe = [cen[j] + 2.0 * (cen[j] - pts[-1][j]) for j in range(n)]
            fe = f(xe)
            if fe < fr:
                pts[-1], vals[-1] = xe, fe
            else:
                pts[-1], vals[-1] = xr, fr
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = [cen[j] + 0.5 * (pts[-1][j] - cen[j]) for j in range(n)]
            fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    pts[i] = [pts[0][j] + 0.5 * (pts[i][j] - pts[0][j]) for j in range(n)]
                    vals[i] = f(pts[i])
    i = min(range(n + 1), key=lambda i: vals[i])
    return pts[i], vals[i]
