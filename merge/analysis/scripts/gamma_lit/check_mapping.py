"""Sanity checks for the interpretation of Roosta-Khorasani and Szekely (2015), Theorem 1.

Rate b = 1 everywhere. Tail of c1*G(a1) + c2*G(a2) by quantile substitution.
"""
import numpy as np
from scipy import special, integrate, optimize


def tail_single(a, t):
    return special.gammaincc(a, t)


def cdf_two(c1, a1, c2, a2, t):
    # P(c1 G1 + c2 G2 <= t) = int_0^{F1(t/c1)} F2((t - c1 Q1(y)) / c2) dy
    ymax = special.gammainc(a1, t / c1)

    def g(y):
        x = c1 * special.gammaincinv(a1, y)
        r = max(t - x, 0.0)
        return special.gammainc(a2, r / c2)

    pts = np.linspace(0, ymax, 41)
    val = 0.0
    for lo, hi in zip(pts[:-1], pts[1:]):
        v, _ = integrate.quad(g, lo, hi, limit=200, epsabs=1e-14, epsrel=1e-10)
        val += v
    return val


def tail_two(c1, a1, c2, a2, t):
    return 1.0 - cdf_two(c1, a1, c2, a2, t)


def sign_changes(f, ts):
    vals = [f(t) for t in ts]
    out = []
    for (t0, v0), (t1, v1) in zip(zip(ts, vals), zip(ts[1:], vals[1:])):
        if v0 == 0 or np.sign(v0) != np.sign(v1):
            out.append(optimize.brentq(f, t0, t1))
    return out, vals


# (a) exponential units: alpha = 1, single Gamma(3) vs Gamma(2) + 0.5 Gamma(2) (theta = 1,1,.5,.5)
f_a = lambda t: tail_single(3.0, t) - tail_two(1.0, 2.0, 0.5, 2.0, t)
ts = np.linspace(0.05, 15, 300)
roots, _ = sign_changes(f_a, ts)
print("(a) alpha=1, m=3: crossings of single - member:", [round(r, 4) for r in roots],
      " RS guarantee t >", 3 * 1.5)

# (b) counterexample family, A = 0.0025: single Gamma(A) vs 0.5 Gamma(A) + Gamma(A/2)
A = 0.0025
f_b = lambda t: tail_single(A, t) - tail_two(0.5, A, 1.0, A / 2, t)
ts = np.linspace(0.002, 6, 400)
roots, vals = sign_changes(f_b, ts)
print("(b) A=0.0025: crossings:", [round(r, 4) for r in roots], " RS guarantee b*t >", 1 + A)
print("    at t=0.01: single", tail_single(A, 0.01), " member", tail_two(0.5, A, 1.0, A / 2, 0.01))

# (c) small-shape limit: root of E1(u) = exp(-u)
u_star = optimize.brentq(lambda u: special.exp1(u) - np.exp(-u), 0.1, 2.0)
print("(c) root of E1(u) = exp(-u):", u_star)
# crossing for the pair in the Levy-measure limit: E1(t) = 2 E1(2t)
t_pair = optimize.brentq(lambda t: special.exp1(t) - 2 * special.exp1(2 * t), 0.05, 2.0)
print("    limit crossing of the pair (E1(t) = 2 E1(2t)):", t_pair)
