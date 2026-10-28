"""S4: ARD1 repair in parameter space (shape -> shape - rho*(shape - anchor)) versus the
pathwise law G + (1-rho)*Delta.

Part initially in state D_0 ~ Gamma(g, b) (g = 0: new part), anchor shape g (the merge default:
the anchor starts at the initial state; main starts it at 0, which equals this case when g = 0).
Day 1: mission of shape d with g + d = A_hat (the cap). Day 2: repair. Day 3: mission of shape rho*d.
Tracked: g + d = A_hat; g + (1-rho) d; g + d = A_hat. Every row holds.
True on day 3: D_0 + (1-rho)*Delta + Z = Gamma(g + rho*d, b) + Gamma(d, b/(1-rho)).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gamma_lib import sum_tail_fast, mc_tail, shape_for_tail  # noqa: E402

tau = 1.0
print("ratio P(D_3 > tau)/eps; columns rho = 0.1, 0.3, 0.5, 0.8, 0.95")
for eps in [0.01, 0.1]:
    for btau in [0.01, 0.1, 0.3, 1.0, 10.0]:
        b = btau / tau
        A = shape_for_tail(eps, b, tau)
        for gfrac in [0.0, 0.25, 0.5]:
            g = gfrac * A
            d = A - g
            row = []
            for rho in [0.1, 0.3, 0.5, 0.8, 0.95]:
                lo, up = sum_tail_fast([(g + rho * d, b), (d, b / (1 - rho))], tau)
                row.append(f"{lo/eps:.3f}")
            print(f"  eps={eps:<5} b*tau={btau:<5} anchor g = {gfrac:.2f}*A_hat: {' '.join(row)}")

# worked case with a nonzero anchor, with a Monte Carlo cross-check
eps, rho, b, gfrac = 0.01, 0.5, 0.01, 0.25
A = shape_for_tail(eps, b, tau)
g, d = gfrac * A, (1 - gfrac) * A
lo, up = sum_tail_fast([(g + rho * d, b), (d, b / (1 - rho))], tau)


def sampler(rng):
    G = rng.gammavariate(g, 1 / b)
    Delta = rng.gammavariate(d, 1 / b)
    Z = rng.gammavariate(rho * d, 1 / b)
    return G + (1 - rho) * Delta + Z


p, se = mc_tail(sampler, tau, 1000000, seed=11)
print(f"worked case: eps={eps} rho={rho} b={b} g={g:.6g} d={d:.6g}: tracked day 3 = {g + d:.6g} = A_hat; "
      f"true tail [{lo:.6f}, {up:.6f}] = {lo/eps:.3f} eps; MC {p:.5f} +- {1.96*se:.5f}")
