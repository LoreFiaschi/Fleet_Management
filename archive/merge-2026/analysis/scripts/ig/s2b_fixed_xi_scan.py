"""S2b: fixed-xi family with parameter-space repair; scan dispersed cases (small xi) for failures."""
import math, sys
sys.path.insert(0, sys.argv[1])
from iglib import *

TAU = 1.0


def mu_bar_xi(xi, tau, eps):
    g = lambda m: ig_sf(tau, m, xi * m * m) - eps
    hi = tau * (1 - 1e-9)
    return hi if g(hi) <= 0 else bisect(g, 1e-9 * tau, hi)


print("one repair at the cap (factor c), then fresh increments back to the cap; and two repairs")
for xi in (0.2, 2.0, 20.0):
    for eps in (0.01, 0.05, 0.1, 0.2):
        mb = mu_bar_xi(xi, TAU, eps)
        row = f"xi={xi:<5} eps={eps:<5} mu_bar_xi={mb:.4f} eta_eff=1/(xi mu_bar)={1/(xi*mb):.3g} |"
        worst = 0.0
        for c in (0.9, 0.8, 0.5, 0.2, 0.05):
            one = [(c * mb, c * xi * mb * mb), (mb - c * mb, xi * (mb - c * mb) ** 2)]
            two = [(c * c * mb, c * c * xi * mb * mb),
                   (c * (1 - c) * mb, c * xi * ((1 - c) * mb) ** 2),
                   (mb - c * mb, xi * (mb - c * mb) ** 2)]
            r1 = sum_tail(one, TAU, 1500) / eps
            r2 = sum_tail(two, TAU, 1500) / eps
            worst = max(worst, r1, r2)
            row += f" c={c}: {r1:.3f}/{r2:.3f}"
        print(row + f" | worst {worst:.3f}")
