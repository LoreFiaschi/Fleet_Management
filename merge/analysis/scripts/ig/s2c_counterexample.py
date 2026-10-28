"""S2c: Monte Carlo check of the fixed-xi counterexample (pathwise repair, then increments)."""
import math, random, sys
sys.path.insert(0, sys.argv[1])
from iglib import *

TAU, XI, EPS, C, M = 1.0, 0.2, 0.1, 0.5, 20  # M fresh missions after the repair
g = lambda m: ig_sf(TAU, m, XI * m * m) - EPS
mb = bisect(g, 1e-9, 1 - 1e-9)
print(f"xi={XI} eps={EPS}: mu_bar_xi={mb:.4f}; parameter-space tail at the cap = {ig_sf(TAU, mb, XI*mb*mb):.4f}")
# simulate: pre-repair state Q(mu_bar) as a sum of M missions of size mu_bar/M (exact by the semigroup),
# repair multiplies the path by C, then M missions of size (1-C) mu_bar / M
rng = random.Random(21)
N, hit = 400000, 0
a1, a2 = mb / M, (1 - C) * mb / M
for _ in range(N):
    pre = sum(ig_sample(a1, XI * a1 * a1, rng) for _ in range(M))
    post = C * pre + sum(ig_sample(a2, XI * a2 * a2, rng) for _ in range(M))
    hit += post > TAU
p = hit / N
print(f"MC (each mission sampled, {N} paths): true tail {p:.4f} +- {math.sqrt(p*(1-p)/N):.4f}")
terms = [(C * mb, C * XI * mb * mb), ((1 - C) * mb, XI * ((1 - C) * mb) ** 2)]
print(f"convolution: true tail {sum_tail(terms, TAU, 2000):.4f}")
