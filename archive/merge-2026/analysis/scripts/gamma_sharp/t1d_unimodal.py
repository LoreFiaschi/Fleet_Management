"""T1d: spot check of Assumption U (unimodal density) on random members of C(m).

Member: S = sum_i Gamma(a_i, r_i), rates r_i = 1/c_i >= 1. Density by the mixture formula of gamma.md:
f_S(x) = sum_k w_k R f_{A+k}(R x), R = max r_i, A = sum a_i, w = pmf of K = sum_i NegBin(a_i, r_i/R).
The density is evaluated on a grid of 600 points (geometric near 0, then linear up to the
0.9999 quantile region); a member fails if its density goes down and then up by more than 1e-9
(relative).
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gamma_lib import negbin_pmf, convolve  # noqa: E402


def mixture(comps):
    R = max(r for _, r in comps)
    A = sum(a for a, _ in comps)
    pmf = [1.0]
    for a, r in comps:
        if r < R:
            w, _ = negbin_pmf(a, r / R, 1e-14)
            pmf = convolve(pmf, w, 20000)
    return R, A, pmf


def density(R, A, pmf, x):
    y = R * x
    f = math.exp((A - 1.0) * math.log(y) - y - math.lgamma(A))   # f_A(y)
    s = 0.0
    a = A
    for wk in pmf:
        s += wk * f
        f *= y / a
        a += 1.0
        if f == 0.0:
            break
    return R * s


rng = random.Random(3)
fails, n = 0, 600
worst = 0.0
for trial in range(n):
    K = rng.choice([2, 3, 4])
    comps = []
    for _ in range(K):
        c = math.exp(rng.uniform(math.log(0.05), 0.0))
        a = math.exp(rng.uniform(math.log(1e-3), math.log(10.0)))
        comps.append((a, 1.0 / c))
    mean = sum(a / r for a, r in comps)
    sd = math.sqrt(sum(a / r ** 2 for a, r in comps))
    R, A, pmf = mixture(comps)
    xmax = mean + 8 * sd + 5.0
    xs = [xmax * 1e-6 * (1e4 ** (i / 200)) for i in range(200)] + \
         [xmax * 1e-2 + (xmax - xmax * 1e-2) * i / 400 for i in range(1, 401)]
    fs = [density(R, A, pmf, x) for x in xs]
    went_down, bad = False, 0.0
    peak = max(fs)
    for i in range(1, len(fs)):
        d = fs[i] - fs[i - 1]
        if d < -1e-9 * peak:
            went_down = True
        elif went_down and d > 1e-9 * peak:
            bad = max(bad, d / peak)
    if bad > 0:
        fails += 1
        print("  not unimodal on the grid:", [(round(a, 4), round(1 / r, 4)) for a, r in comps], f"rise {bad:.2e}")
    worst = max(worst, bad)
print(f"members checked: {n}; with a rise after a fall (> 1e-9 of the peak): {fails}; largest rise {worst:.2e}")
