"""Self-test of iglib: closed-form SF vs quadrature, sampler moments, Mills ratio."""
import math, random, sys
sys.path.insert(0, sys.argv[1])
from iglib import *

# Mills ratio: erfc branch vs continued fraction near the switch
for t in (5.0, 10.0, 19.9):
    f = t
    for k in range(80, 0, -1):
        f = t + k / f
    print(f"mills t={t}: erfc {mills(t):.15e}  CF {1/f:.15e}")

# SF vs 1 - quadrature of pdf, several (mu, lam)
for mu, lam, x in [(1, 1, 3), (1, 0.2, 5), (1, 20, 1.5), (0.6, 0.6 / 0.05, 1.0), (1, 100, 1.3)]:
    n = 200000
    h = x / n
    q = simpson([ig_pdf(i * h, mu, lam) for i in range(n + 1)], h)
    print(f"IG({mu},{lam}) x={x}: sf={ig_sf(x, mu, lam):.12e}  1-int pdf={1 - q:.12e}  cdf+sf-1={ig_cdf(x, mu, lam) + ig_sf(x, mu, lam) - 1:.1e}")

# sampler moments
rng = random.Random(7)
for mu, lam in [(1, 1), (2, 0.5), (0.3, 5)]:
    xs = [ig_sample(mu, lam, rng) for _ in range(400000)]
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    p = sum(1 for x in xs if x > 1.5 * mu) / len(xs)
    print(f"sampler IG({mu},{lam}): mean {m:.4f} (exact {mu}), var {v:.4f} (exact {mu**3/lam:.4f}), "
          f"P(>1.5mu) {p:.4f} (exact {ig_sf(1.5*mu, mu, lam):.4f})")
