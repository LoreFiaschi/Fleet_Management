"""S1: sum of Gamma(a_i, b_i) versus Gamma(sum a_i, b*) with b* <= min b_i."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gamma_lib import sum_tail, tail, shape_for_tail  # noqa: E402

# 1) dominance check on random cases: P(sum > t) <= P(Gamma(sum a, b*) > t) for all t on a grid
rng = random.Random(7)
worst = -1.0
n_checks = 0
for _ in range(300):
    n = rng.randint(2, 3)
    comps = [(rng.uniform(0.05, 5.0), rng.uniform(0.5, 20.0)) for _ in range(n)]
    bstar = min(b for _, b in comps) * rng.uniform(0.5, 1.0)
    A = sum(a for a, _ in comps)
    mean_true = sum(a / b for a, b in comps)
    for f in [0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        t = f * mean_true
        lo, up = sum_tail(comps, t)
        s = tail(A, bstar, t)
        worst = max(worst, up - s)
        n_checks += 1
print(f"dominance: {n_checks} (case, t) pairs; max of [true tail (upper) - bound tail] = {worst:.3e}")

# 2) conservatism on one example: tau = 1, eps = 0.01
tau, eps = 1.0, 0.01
t1 = (1.0, 10.0)    # shape 1, rate 10: mean 0.1
t2 = (10.0, 100.0)  # shape 10, rate 100: mean 0.1
bstar = 10.0
Amax = shape_for_tail(eps, bstar, tau)
print(f"A_max at b*={bstar}: {Amax:.4f}")


def exact_ok(n1, n2):
    comps = [(n1 * t1[0], t1[1]), (n2 * t2[0], t2[1])]
    return sum_tail(comps, tau)[1] <= eps


def s1_ok(n1, n2):
    return n1 * t1[0] + n2 * t2[0] <= Amax


T = 15
ex = [(n1, n2) for n1 in range(T + 1) for n2 in range(T + 1 - n1) if exact_ok(n1, n2)]
s1 = [(n1, n2) for n1 in range(T + 1) for n2 in range(T + 1 - n1) if s1_ok(n1, n2)]
print(f"pairs (n1, n2), n1+n2 <= {T}: exactly safe {len(ex)}; accepted by the S1 bound {len(s1)}")
print("largest n2 for each n1, exact:", {n1: max(n2 for a, n2 in ex if a == n1) for n1 in sorted({a for a, _ in ex})})
print("largest n2 for each n1, S1   :", {n1: max(n2 for a, n2 in s1 if a == n1) for n1 in sorted({a for a, _ in s1})})
print("S1 accepts only exactly safe pairs:", set(s1) <= set(ex))
