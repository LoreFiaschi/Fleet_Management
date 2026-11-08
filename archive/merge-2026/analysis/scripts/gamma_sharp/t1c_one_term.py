"""T1c: the one-term family Y_c = c*Gamma(A/c, 1), c in (0, 1], at the cap A = A_hat(t, eps).

(a) threshold t1(eps): root in t of g(t) = t f_A(t) - A dQ/da(A, t), A = A_hat(t, eps);
    g = d/dc Q(A/c, t/c) at c = 1 (g > 0: the single Gamma is above the family near c = 1).
(b) dense check above t1: max over c in (0, 0.999] of Q(A/c, t/c) / eps on a log grid of t.
(c) witnesses below t1: c* and the exact tail of c* Gamma(A_hat/c*, 1); Monte Carlo cross-check.
(d) eps -> 0: t1(eps) tends to y0 = 0.4348 (root of e^y E1(y) = 1).
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sharp_lib import A_hat, gammaQ, gamma_density  # noqa: E402

CGRID = [10 ** (-4 + 4 * i / 400) for i in range(400)] + [0.999]


def one_term(A, t, c):
    return gammaQ(A / c, t / c)


def best_c(A, t, cmax=0.999):
    """max over c in [1e-4, cmax] of Q(A/c, t/c): log grid, then golden refinement."""
    grid = [c for c in CGRID if c <= cmax] + [cmax]
    vals = [one_term(A, t, c) for c in grid]
    i = max(range(len(grid)), key=lambda k: vals[k])
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    gr = (math.sqrt(5) - 1) / 2
    x1, x2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
    f1, f2 = one_term(A, t, x1), one_term(A, t, x2)
    for _ in range(80):
        if f1 > f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - gr * (hi - lo)
            f1 = one_term(A, t, x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + gr * (hi - lo)
            f2 = one_term(A, t, x2)
    cands = [(vals[i], grid[i]), (f1, x1), (f2, x2)]
    return max(cands)


def g(t, eps):
    A = A_hat(t, eps)
    h = 1e-6 * A
    dQda = (gammaQ(A + h, t) - gammaQ(A - h, t)) / (2 * h)
    return t * gamma_density(A, t) - A * dQda


def t1(eps, lo=0.3, hi=1.0):
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if g(mid, eps) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


print("(a) threshold t1(eps) = root of t f_A(t) = A dQ/da(A,t), A = A_hat(t, eps)")
T1 = {}
for eps in [0.001, 0.01, 0.1]:
    T1[eps] = t1(eps)
    print(f"  eps={eps}: t1={T1[eps]:.5f}  A_hat(t1)={A_hat(T1[eps], eps):.5g}  "
          f"g(t1-0.01)={g(T1[eps] - 0.01, eps):+.2e}  g(t1+0.01)={g(T1[eps] + 0.01, eps):+.2e}")

print("(b) above t1: largest Q(A_hat/c, t/c)/eps over c in [1e-4, 0.999], t on a log grid t1..1000")
for eps in [0.001, 0.01, 0.1]:
    worst, worst_t, worst_c, gmin = -1.0, None, None, 1e300
    n = 300
    for i in range(n + 1):
        t = T1[eps] * (1000.0 / T1[eps]) ** (i / n)
        if i == 0:
            t = T1[eps] * (1 + 1e-6)
        A = A_hat(t, eps)
        v, c = best_c(A, t)
        if v / eps > worst:
            worst, worst_t, worst_c = v / eps, t, c
        gmin = min(gmin, g(t, eps) / (A * eps))
    print(f"  eps={eps}: max ratio {worst:.10f} at t={worst_t:.4g}, c={worst_c:.4g}; "
          f"min of g/(A eps) on the grid = {gmin:+.3e} (> 0: slope at c=1 has the safe sign)")

print("(c) below t1: worst one-term member at the cap (exact) and Monte Carlo check")
rng = random.Random(5)
for eps in [0.001, 0.01, 0.1]:
    for t in [0.01, 0.1, 0.3, 0.4, 0.43, 0.6]:
        if t >= T1[eps]:
            continue
        A = A_hat(t, eps)
        v, c = best_c(A, t)
        drift = " (A_hat > t: drift alone gives tail 1)" if A > t else ""
        line = f"  eps={eps:<6} t={t:<5} A_hat={A:<10.4g} c*={c:<8.4g} tail/eps={v / eps:.6f}{drift}"
        if t in (0.3, 0.1) and eps == 0.01:
            nmc, hits = 400000, 0
            for _ in range(nmc):
                hits += c * rng.gammavariate(A / c, 1.0) > t
            p = hits / nmc
            line += f"  MC {p / eps:.4f} +- {math.sqrt(p * (1 - p) / nmc) / eps:.4f}"
        print(line)

print("(d) eps -> 0: t1(eps) (Q loses relative precision below eps = 1e-4, so no smaller eps)")
for eps in [1e-4]:
    print(f"  eps={eps:g}: t1={t1(eps, 0.3, 1.0):.5f}")

print("(e) cross-check: g/A against phi'(1) (central difference of the quadrature phi of T1a)")
from sharp_lib import phi  # noqa: E402
for eps in [0.01, 0.1]:
    for t in [T1[eps], 1.0, 3.0]:
        A = A_hat(t, eps)
        dphi = (phi(1.0005, A, t) - phi(0.9995, A, t)) / 0.001
        print(f"  eps={eps} t={t:.5f}: g/A={g(t, eps) / A:+.6e}  phi'(1)={dphi:+.6e}")
