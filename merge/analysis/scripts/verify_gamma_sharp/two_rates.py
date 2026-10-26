"""Two mission rates in one cell: how much a safe per-rate weighting could gain over the sharp cap.

Claude, 26 Oct 2026. Illustration on the example S2 of gamma.md: tau = 1, eps = 0.01; slow missions with
mean 0.1 and rate 10 (b* = 10); fast missions with mean 0.1 and rate 200. Units of 1/b*: t = b*·tau = 10;
one mission of either kind has mean mass A = b*·0.1 = 1.
Worst case searched over one term per group: slow c_l in (0, 1], fast c_h in (0, b*/200] (repairs only lower c).
"""
import math
from check import Q, A_hat

def tail2(a1, c1, a2, c2, t, n=2000):
    """P(c1·G1 + c2·G2 > t), G_i ~ Gamma(a_i, 1) independent; substitution x = t·w^(1/a1), Simpson."""
    pre = math.exp(a1 * math.log(t / c1) - math.lgamma(a1 + 1.0))
    def g(w):
        if w <= 0.0:
            return pre * Q(a2, t / c2)
        x = t * w ** (1.0 / a1)
        return pre * math.exp(-x / c1) * Q(a2, (t - x) / c2)
    h = 1.0 / n
    s = g(0.0) + g(1.0)
    for k in range(1, n):
        s += (4.0 if k % 2 else 2.0) * g(k * h)
    return Q(a1, t / c1) + s * h / 3.0

def worst(Al, Ah, t, ch_max):
    """Largest tail over a small grid of (c_l, c_h)."""
    best = 0.0
    for cl in [1.0, 0.7, 0.4]:
        for f in [1.0, 0.5, 0.1]:
            ch = ch_max * f
            if Al <= 1e-12:
                v = Q(Ah / ch, t / ch)
            elif Ah <= 1e-12:
                v = Q(Al / cl, t / cl)
            else:
                v = tail2(Al / cl, cl, Ah / ch, ch, t)
            best = max(best, v)
    return best

if __name__ == "__main__":
    t, eps, ch_max = 10.0, 0.01, 10.0 / 200.0
    L = A_hat(t, eps)                      # slow-only cap, = sharp cap on the total mean
    H_fast = ch_max * A_hat(t / ch_max, eps)  # fast-only cap (exact for fast missions alone)
    print("sharp cap on total mass: %.4f missions of any kind" % L)
    print("fast missions alone, exact cap: %.4f missions" % H_fast)
    S = [k / 10.0 for k in range(11)]
    def line_ok(H):
        return max(worst(L * (1.0 - s), H * s, t, ch_max) for s in S)
    print("straight line from (%.3f slow, 0) to (0, %.3f fast): worst tail/eps = %.4f" % (L, H_fast, line_ok(H_fast) / eps))
    lo, hi = L, H_fast
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        if line_ok(mid) <= eps * (1.0 + 1e-7):  # the pivot itself sits at eps, up to rounding
            lo = mid
        else:
            hi = mid
    print("largest safe fast intercept H of a line through (%.3f, 0): %.4f missions (worst tail/eps %.5f)" % (L, lo, line_ok(lo) / eps))
    for s in [0.25, 0.5, 0.75]:
        print("  on that line at s=%.2f: slow %.3f, fast %.3f, tail/eps %.4f" % (s, L * (1 - s), lo * s, worst(L * (1 - s), lo * s, t, ch_max) / eps))
