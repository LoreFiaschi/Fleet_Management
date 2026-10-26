"""T1a: first-order (local) optimality of the single Gamma(m, 1) at the cap m = A_hat(t).

D(c) = phi(c) - phi(1) = derivative of P(S > t) when mean mass eta moves from Gamma(m, 1)
to c*Gamma(eta/c, 1) (c = 0: a deterministic shift). The single Gamma is a local maximum
only if D(c) <= 0 for every c in [0, 1]. The threshold t1(eps) is the smallest t on the grid
above which max_c D(c) <= 0.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sharp_lib import phi, A_hat  # noqa: E402

CS = [0.0] + [0.02 * i for i in range(1, 50)] + [0.99, 0.995, 0.999]


def scan(t, eps):
    m = A_hat(t, eps)
    p1 = phi(1.0, m, t)
    best_c, best = None, -1e300
    for c in CS:
        d = phi(c, m, t) - p1
        if d > best:
            best, best_c = d, c
    slope1 = (p1 - phi(0.999, m, t)) / 0.001     # phi'(1) estimate
    return m, p1, best_c, best, slope1


print("t, m=A_hat, phi(1), argmax_c D(c), max_c D(c) / phi(1), phi'(1)")
for eps in [0.001, 0.01, 0.1]:
    for t in [0.01, 0.1, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 1.0, 3.0, 10.0, 100.0]:
        m, p1, bc, bd, s1 = scan(t, eps)
        print(f"eps={eps:<6} t={t:<6} m={m:<10.4g} phi1={p1:<10.4g} argmax c={bc:<6} "
              f"max D/phi1={bd / p1:+.4e}  phi'(1)={s1:+.4e}")

# threshold by bisection on t of max_c D(c) (sign change), per eps
print("first-order threshold t1(eps): max_c D(c) changes sign")
for eps in [0.001, 0.01, 0.1]:
    lo, hi = 0.3, 1.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        _, p1, bc, bd, _ = scan(mid, eps)
        if bd > 1e-12 * p1:
            lo = mid
        else:
            hi = mid
    m, p1, bc, bd, s1 = scan(hi, eps)
    print(f"  eps={eps}: t1 in [{lo:.5f}, {hi:.5f}], A_hat(t1)={m:.5g}, last argmax c={bc}, phi'(1)={s1:+.3e}")
