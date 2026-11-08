"""T2: the one-term cap A_1(t, eps) = sup{A < t : max_{c in (0,1]} Q(A/c, t/c) <= eps}, against A_hat and A_icx.

Under Assumption U (notes, T1 Statement 2) A_sharp = A_1. Without it, A_1 is an upper bound on A_sharp:
at A = A_1 (1 + 1e-6) the script exhibits a one-term member with exact tail > eps (a witness).
A_icx (S5 of gamma.md) is a proven lower bound on A_sharp.
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sharp_lib import A_hat, A_icx, gammaQ  # noqa: E402

CGRID = [10 ** (-6 + 6 * i / 600) for i in range(600)] + [1.0]


def worst_one_term(A, t):
    vals = [(gammaQ(A / c, t / c), c) for c in CGRID]
    v, c = max(vals)
    i = CGRID.index(c)
    lo, hi = CGRID[max(i - 1, 0)], CGRID[min(i + 1, len(CGRID) - 1)]
    gr = (math.sqrt(5) - 1) / 2
    f = lambda x: gammaQ(A / x, t / x)
    x1, x2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
    f1, f2 = f(x1), f(x2)
    for _ in range(80):
        if f1 > f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - gr * (hi - lo)
            f1 = f(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + gr * (hi - lo)
            f2 = f(x2)
    return max([(v, c), (f1, x1), (f2, x2)])


def A_one(t, eps):
    lo, hi = 0.0, min(A_hat(t, eps), t)
    if worst_one_term(hi, t)[0] <= eps * (1 + 1e-12):
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if worst_one_term(mid, t)[0] <= eps:
            lo = mid
        else:
            hi = mid
    return lo


rng = random.Random(2027)
print("t     eps     A_hat       A_icx/A_hat  A_1/A_hat   witness at A_1(1+1e-6): c*, tail/eps")
for eps in [0.001, 0.01, 0.1]:
    for t in [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0]:
        Ah = A_hat(t, eps)
        Ai = A_icx(t, eps)
        A1 = A_one(t, eps)
        if A1 < Ah * (1 - 1e-9):
            v, c = worst_one_term(A1 * (1 + 1e-6), t)
            wit = f"c*={c:.4g} tail/eps={v / eps:.8f}"
        else:
            wit = "A_1 = A_hat (c* = 1)"
        print(f"{t:<6}{eps:<8}{Ah:<12.5g}{Ai / Ah:<13.4f}{A1 / Ah:<12.4f}{wit}", flush=True)

# Monte Carlo check of one witness: t = 0.1, eps = 0.01
t, eps = 0.1, 0.01
A1 = A_one(t, eps)
v, c = worst_one_term(A1 * 1.05, t)
n, hits = 3000000, 0
for _ in range(n):
    hits += c * rng.gammavariate(A1 * 1.05 / c, 1.0) > t
p = hits / n
print(f"MC check t=0.1 eps=0.01, A=1.05 A_1, c={c:.4g}: exact {v:.5f}, MC {p:.5f} +- {math.sqrt(p * (1 - p) / n):.5f}")
