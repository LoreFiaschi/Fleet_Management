"""S5: the increasing-convex-order surrogate with a stop-loss / Chernoff cap.

Tracked A' = b* * mu (mu: tracked mean, linear dynamics for both repair laws).
Theorem (notes): D_k <=icx Gamma(A'_k, b*). Hence P(D_k > tau) <= B(A'_k),
B(A) = min(stop-loss bound, Chernoff bound) of Gamma(A, b*) at tau. Cap: A' <= A_icx, B(A_icx) = eps.
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gamma_lib import (tail, shape_for_tail, stoploss_bound, chernoff_gamma, cap_for_bound,  # noqa: E402
                       sum_tail_fast, stoploss_gamma)
from main_calibration import calibrate_cell  # noqa: E402

tau = 1.0


def B(A, b, t=tau):
    return min(stoploss_bound(A, b, t), chernoff_gamma(A, b, t))


def cap_icx(b, eps):
    return max(cap_for_bound(stoploss_bound, b, tau, eps), cap_for_bound(chernoff_gamma, b, tau, eps))


def cap_cantelli(b, eps):
    k = (1 - eps) / eps
    x = b * tau
    return ((2 * x + k) - math.sqrt((2 * x + k) ** 2 - 4 * x * x)) / 2


# (b) price of the cap, one rate: allowed shape (= allowed mean times b) relative to the exact Gamma cap
print("caps, one rate: A_hat (exact Gamma), A_icx (this proposal), A_Cantelli; ratios to A_hat")
for eps in [0.01, 0.1]:
    for btau in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        Ah = shape_for_tail(eps, btau, tau)
        Ai = cap_icx(btau, eps)
        Ac = cap_cantelli(btau, eps)
        print(f"  eps={eps:<5} b*tau={btau:<7} A_hat={Ah:<10.4g} A_icx={Ai:<10.4g} ({Ai/Ah:.3f}) "
              f"A_Cantelli={Ac:<10.4g} ({Ac/Ah:.3f}); exact tail at A_icx = {tail(Ai, btau, tau):.2e}")

# (c) the counterexamples of S2 and S3 are rejected: B at the tracked A' exceeds eps (and the true tail)
eps = 0.01
b = 0.01
Ah = shape_for_tail(eps, b, tau)
lo, _ = sum_tail_fast([(Ah, b / 0.5), (0.5 * Ah, b)], tau)
print(f"S3 example: tracked A'={Ah:.5g}, true tail {lo:.5f}, bound B(A')={B(Ah, b):.4f}, A_icx={cap_icx(b, eps):.4g}")
for label, means, rates, counts in [("S2 example A (3,2)", [0.1, 0.1], [10.0, 200.0], [3, 2]),
                                    ("S2 example B (0,2)", [0.1, 0.4], [10.0, 50.0], [0, 2])]:
    bstar = min(rates)
    Ap = bstar * sum(m * n for m, n in zip(means, counts))
    lo, _ = sum_tail_fast([(n * m * r, r) for m, r, n in zip(means, rates, counts)], tau)
    print(f"{label}: A'=b*mu={Ap:.4f}, true tail {lo:.5f}, bound B(A')={B(Ap, bstar):.4f}, A_icx={cap_icx(bstar, eps):.4f}")

# (d) conservatism on the S1 example: type 1 Gamma(1,10), type 2 Gamma(10,100), eps 0.01, n1+n2<=15
Ai = cap_icx(10.0, eps)
acc = []
for n1 in range(16):
    for n2 in range(16 - n1):
        if 10.0 * (0.1 * n1 + 0.1 * n2) <= Ai:
            acc.append((n1, n2))
print(f"S1 example: A_icx={Ai:.4f}; accepted pairs {len(acc)}; largest n2 per n1:",
      {n1: max(n2 for a, n2 in acc if a == n1) for n1 in sorted({a for a, _ in acc})})

# (a) Monte Carlo sanity check of the theorem on random schedules (both repair laws, replacement)
rng = random.Random(2026)
worst_tail, worst_sl, nchk = -1.0, -1.0, 0
for trial in range(12):
    law = "ard1" if trial % 2 else "ardinf"
    types = [(rng.choice([0.05, 0.1, 0.2]), rng.choice([2.0, 10.0, 50.0])) for _ in range(2)]
    bstar = min(r for _, r in types)
    rho = rng.choice([0.3, 0.6, 0.9])
    mu0, b0 = 0.1, rng.choice([5.0, 50.0])
    bstar = min(bstar, b0)
    T = 12
    sched = []
    for _ in range(T):
        u = rng.random()
        sched.append(("m", rng.randrange(2)) if u < 0.65 else ("r", None) if u < 0.92 else ("R", None))
    # tracked exact recursion on A' = b* mu, anchor g'
    A = bstar * mu0
    g = A
    trackedA = []
    for kind, q in sched:
        if kind == "m":
            A += bstar * types[q][0]
        elif kind == "r":
            A = (1 - rho) * A if law == "ardinf" else (1 - rho) * A + rho * g
            g = A
        else:
            A = bstar * mu0
            g = A
        trackedA.append(A)
    N = 20000
    hits = [0] * T
    excess = [[0.0, 0.0] for _ in range(T)]
    excess2 = [[0.0, 0.0] for _ in range(T)]
    for _ in range(N):
        D = rng.gammavariate(mu0 * b0, 1 / b0)
        G = D
        for k, (kind, q) in enumerate(sched):
            if kind == "m":
                m, r = types[q]
                D += rng.gammavariate(m * r, 1 / r)
            elif kind == "r":
                D = (1 - rho) * D if law == "ardinf" else G + (1 - rho) * (D - G)
                G = D
            else:
                D = rng.gammavariate(mu0 * b0, 1 / b0)
                G = D
            if D > tau:
                hits[k] += 1
            for j, x in enumerate([0.5, 0.8]):
                e = max(0.0, D - x * tau)
                excess[k][j] += e
                excess2[k][j] += e * e
    for k in range(T):
        p = hits[k] / N
        se = math.sqrt(max(p * (1 - p), 1.0 / N) / N)
        worst_tail = max(worst_tail, p - 3 * se - B(trackedA[k], bstar))
        for j, x in enumerate([0.5, 0.8]):
            m1 = excess[k][j] / N
            sd = math.sqrt(max(excess2[k][j] / N - m1 * m1, 0.0) / N)
            worst_sl = max(worst_sl, m1 - 3 * sd - stoploss_gamma(trackedA[k], bstar, x))
        nchk += 1
print(f"Monte Carlo, 12 random schedules x 12 days ({nchk} checks, 20000 paths each):")
print(f"  max of (MC tail - 3 s.e. - bound B(A')) = {worst_tail:.4f} (<= 0 expected)")
print(f"  max of (MC stop-loss - 3 s.e. - Gamma(A', b*) stop-loss) at x = 0.5, 0.8 = {worst_sl:.4f} (<= 0 expected)")
