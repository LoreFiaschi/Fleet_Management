"""S1b: counterexamples with Monte Carlo, and the eps threshold above which lore's cap fails."""
import math, sys
sys.path.insert(0, sys.argv[1])
from iglib import *

TAU = 1.0
NS = [2, 3, 4, 5, 7, 10, 20, 30, 100, 1000, 10 ** 4, 10 ** 6]

print("== counterexamples: n equal independent increments, total mean = mu_bar ==")
for eta, eps, n in [(10.0, 0.1, 3), (10.0, 0.1, 2), (100.0, 0.01, 2), (4.0, 0.2, 10)]:
    mb = mu_bar_eta(eta, TAU, eps)
    a = mb / n
    exact = ig_sf(TAU, mb, n * mb / eta)
    p_mc, se = mc_tail([(a, a / eta)] * n, TAU, 1000000, seed=11)
    model = ig_sf(TAU, mb, mb / eta)
    print(f"eta={eta} eps={eps}: mu_bar={mb:.4f}, {n} x IG({a:.4f}, {a/eta:.5f}): "
          f"true tail {exact:.5f} (MC {p_mc:.5f} +- {se:.5f}), model tail {model:.5f}")


def worst_ratio(eta, eps):
    mb = mu_bar_eta(eta, TAU, eps)
    return max(ig_sf(TAU, mb, n * mb / eta) for n in NS) / eps


print("\n== threshold eps*(eta): smallest eps with a failing n-equal structure (n in NS) ==")
for eta in (0.01, 0.1, 0.25, 1.0, 2.0, 4.0, 10.0, 30.0, 100.0):
    grid = [10 ** (-4 + 0.02 * i) for i in range(0, 186)]  # 1e-4 .. ~0.48
    fail = [e for e in grid if worst_ratio(eta, e) > 1.0]
    if not fail:
        print(f"eta={eta:<6} CV={math.sqrt(eta):.2f}: no failure for eps in [1e-4, 0.48]")
        continue
    lo = max([e for e in grid if e < fail[0]], default=1e-4)
    es = bisect(lambda e: worst_ratio(eta, e) - 1.0, lo, fail[0], it=40)
    print(f"eta={eta:<6} CV={math.sqrt(eta):.2f}: fails for eps above about {es:.4f}")
