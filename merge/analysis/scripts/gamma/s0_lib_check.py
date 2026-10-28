"""Self-check of gamma_lib against closed forms and Monte Carlo."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gamma_lib import gammaQ, sum_tail, mc_tail, shape_for_tail, tail  # noqa: E402

worst = 0.0
for n in range(1, 40):
    for x in [0.01, 0.5, 1, 3, 10, 30, 60, 120]:
        exact = math.exp(-x) * sum(x ** k / math.factorial(k) for k in range(n))
        q = gammaQ(n, x)
        if exact > 1e-250:
            worst = max(worst, abs(q - exact) / exact)
for x in [1e-6, 0.01, 0.3, 2, 9, 25]:
    exact = math.erfc(math.sqrt(x))
    worst = max(worst, abs(gammaQ(0.5, x) - exact) / exact)
print(f"max relative error of Q against closed forms: {worst:.2e}")

# inverse
A = shape_for_tail(0.01, 10.0, 1.0)
print(f"shape with P(Gamma(A,10)>1)=0.01: A={A:.10f}, check tail={tail(A, 10.0, 1.0):.12f}")

# sum of Gammas with different rates: series vs Monte Carlo
cases = [
    [(1.0, 10.0), (10.0, 100.0)],
    [(0.3, 2.0), (2.0, 7.0), (0.7, 3.0)],
    [(0.0025, 0.02), (0.00124, 0.01)],
]
for comps, t in zip(cases, [1.0, 1.2, 1.0]):
    lo, up = sum_tail(comps, t)

    def sampler(rng, comps=comps):
        return sum(rng.gammavariate(a, 1.0 / b) for a, b in comps)

    p, se = mc_tail(sampler, t, 400000)
    print(f"{comps} t={t}: series [{lo:.6f}, {up:.6f}]  MC {p:.6f} +- {1.96 * se:.6f}")
