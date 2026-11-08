"""Self-check of sharp_lib: E1 values, phi against finite differences of exact tails,
member_tail against Monte Carlo."""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sharp_lib import E1, phi, A_hat, member_tail, gammaQ  # noqa: E402

ref = {0.01: 4.037929576538114, 0.5: 0.5597735947761608, 1.0: 0.21938393439552029,
       2.0: 0.04890051070806112, 10.0: 4.156968929685324e-06}
print("E1 relative errors:", {x: f"{abs(E1(x) - v) / v:.1e}" for x, v in ref.items()})

# y0: root of e^y E1(y) = 1
lo, hi = 0.1, 1.0
for _ in range(100):
    mid = 0.5 * (lo + hi)
    if math.exp(mid) * E1(mid) > 1.0:
        lo = mid
    else:
        hi = mid
print(f"y0 (e^y E1(y) = 1) = {lo:.6f}")

# phi(c) - phi(1) against the forward difference of the exact tail
print("D(c) = phi(c) - phi(1) by quadrature vs finite difference of exact tails")
for t, eps in [(0.3, 0.01), (1.0, 0.01), (1.0, 0.1), (3.0, 0.01), (10.0, 0.01), (100.0, 0.1)]:
    m = A_hat(t, eps)
    p1 = phi(1.0, m, t)
    base = gammaQ(m, t)
    for c in [0.0, 0.2, 0.5, 0.9]:
        dq = phi(c, m, t) - p1
        eta = 1e-6 * m
        if c == 0.0:
            T = gammaQ(m - eta, t - eta)
        else:
            T, _ = member_tail([(1.0, m - eta), (c, eta)], 0.0, t)
        dfd = (T - base) / eta
        print(f"  t={t:<6} eps={eps:<5} m={m:<9.4g} c={c:<4} quad {dq:+.6e}  fd {dfd:+.6e}  "
              f"rel.diff {abs(dq - dfd) / max(abs(dfd), 1e-300):.1e}")

# member_tail against Monte Carlo
rng = random.Random(7)
print("member_tail vs Monte Carlo (400,000 draws)")
for terms, delta, t in [([(1.0, 0.004), (0.3, 0.006)], 0.0, 0.1),
                        ([(0.5, 0.2), (0.1, 0.1)], 0.05, 1.0),
                        ([(1.0, 2.0), (0.4, 1.0), (0.2, 0.5)], 0.3, 6.0)]:
    lo_, up_ = member_tail(terms, delta, t)
    n, hits = 400000, 0
    for _ in range(n):
        s = delta + sum(c * rng.gammavariate(mi / c, 1.0) for c, mi in terms)
        hits += s > t
    p = hits / n
    se = math.sqrt(p * (1 - p) / n)
    print(f"  {terms} delta={delta} t={t}: exact [{lo_:.6f}, {up_:.6f}], MC {p:.6f} +- {se:.6f}")
