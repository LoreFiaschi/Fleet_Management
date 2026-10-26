"""S1c: the eta above which lore's cap fails, for eps = 0.01 and eps = 0.1 (n-equal structures)."""
import math, sys
sys.path.insert(0, sys.argv[1])
from iglib import *

TAU = 1.0
NS = [2, 3, 4, 5, 7, 10, 20, 30, 100, 1000, 10 ** 4, 10 ** 6]


def worst_ratio(eta, eps):
    mb = mu_bar_eta(eta, TAU, eps)
    return max(ig_sf(TAU, mb, n * mb / eta) for n in NS) / eps


for eps in (0.001, 0.01, 0.1):
    grid = [10 ** (-1 + 0.05 * i) for i in range(0, 81)]  # eta 0.1 .. 1e3
    fail = [e for e in grid if worst_ratio(e, eps) > 1.0]
    lo = max(e for e in grid if e < fail[0])
    et = bisect(lambda e: worst_ratio(e, eps) - 1.0, lo, fail[0], it=40)
    print(f"eps={eps}: lore's cap fails for eta above about {et:.2f} (CV above {math.sqrt(et):.2f})")
