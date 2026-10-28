"""Claude's independent check of gamma_sharp.md (26 Oct 2026).

Standard library only; no code shared with scripts/gamma_sharp/. Rate 1 (t stands for b*tau).
Checks: A_hat, the threshold t1(eps) (sign of d/dc Q(A/c, t/c) at c = 1, by finite differences),
the worst one-term member at the cap below t1, the one-term cap A_1, the grid above t1, and A_icx.
"""
import math

def Q(a, x):
    """Regularized upper incomplete gamma Q(a, x) = P(Gamma(a, 1) > x)."""
    if x <= 0.0:
        return 1.0
    lg = a * math.log(x) - x - math.lgamma(a)
    if x < a + 1.0:
        # series for P(a, x)
        term = 1.0 / a
        s = term
        n = 0
        while True:
            n += 1
            term *= x / (a + n)
            s += term
            if abs(term) < abs(s) * 1e-17 or n > 100000:
                break
        return 1.0 - math.exp(lg) * s
    # modified Lentz continued fraction for Q(a, x)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    i = 0
    while True:
        i += 1
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-16 or i > 100000:
            break
    return math.exp(lg) * h

def bisect(f, lo, hi, n=200):
    """Root of an increasing f on [lo, hi] (f(lo) < 0 < f(hi))."""
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def A_hat(t, eps):
    """Q(A, t) = eps; Q increases in A. Bisection in log A."""
    f = lambda la: Q(math.exp(la), t) - eps
    return math.exp(bisect(f, math.log(1e-14), math.log(10.0 * t + 50.0)))

def one_term(A, t, c):
    return Q(A / c, t / c)

def worst_c(A, t, cmin=1e-4):
    """max over c in [cmin, 1] of Q(A/c, t/c): log grid of 400 points, then golden section."""
    best_c, best = 1.0, one_term(A, t, 1.0)
    n = 400
    grid = [math.exp(math.log(cmin) * (1.0 - k / n)) for k in range(n + 1)]
    for k, c in enumerate(grid):
        v = one_term(A, t, c)
        if v > best:
            best, best_c, kb = v, c, k
    if best_c == 1.0:
        return best_c, best
    lo = grid[max(kb - 1, 0)]
    hi = grid[min(kb + 1, n)]
    g = (math.sqrt(5.0) - 1.0) / 2.0
    x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
    f1, f2 = one_term(A, t, x1), one_term(A, t, x2)
    for _ in range(80):
        if f1 < f2:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + g * (hi - lo)
            f2 = one_term(A, t, x2)
        else:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - g * (hi - lo)
            f1 = one_term(A, t, x1)
    c = 0.5 * (lo + hi)
    v = one_term(A, t, c)
    if v > best:
        best, best_c = v, c
    return best_c, best

def slope_c1(A, t, h=1e-5):
    """d/dc Q(A/c, t/c) at c = 1, central difference."""
    return (one_term(A, t, 1.0 + h) - one_term(A, t, 1.0 - h)) / (2.0 * h)

def t1(eps):
    """Sign change of the slope at c = 1, with A = A_hat(t, eps): negative below, positive above."""
    f = lambda t: slope_c1(A_hat(t, eps), t)
    lo, hi = 0.2, 1.5
    assert f(lo) < 0.0 < f(hi)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def A_one(t, eps):
    """Largest A <= A_hat with max_c Q(A/c, t/c) <= eps (bisection in log A)."""
    Ah = A_hat(t, eps)
    f = lambda la: worst_c(math.exp(la), t)[1] - eps
    if f(math.log(Ah)) <= 0.0:
        return Ah
    return math.exp(bisect(f, math.log(Ah * 1e-4), math.log(Ah), n=60))

def A_icx(t, eps):
    """S5 cap: largest A with min(stop-loss bound, Chernoff bound) <= eps."""
    def stop_loss(A):
        # inf over x < t of E(G - x)_+ / (t - x), E(G - x)_+ = A Q(A+1, x) - x Q(A, x)
        best = 1.0
        n = 2000
        for k in range(n):
            x = t * k / n
            v = (A * Q(A + 1.0, x) - x * Q(A, x)) / (t - x)
            best = min(best, v)
        return best
    def chernoff(A):
        return math.exp(-(t - A)) * (t / A) ** A if A < t else 1.0
    f = lambda la: min(stop_loss(math.exp(la)), chernoff(math.exp(la))) - eps
    return math.exp(bisect(f, math.log(1e-14), math.log(t), n=60))

if __name__ == "__main__":
    EPS = [0.001, 0.01, 0.1]
    print("(1) A_hat(t, eps): Q(A_hat, t) = eps")
    for eps in EPS:
        print("  eps=%g: " % eps + "  ".join("t=%g: %.5g" % (t, A_hat(t, eps)) for t in [0.01, 0.1, 0.3, 1, 3, 10, 100]))
    print("(2) threshold t1(eps): sign change of d/dc Q(A_hat/c, t/c) at c = 1 (finite differences)")
    T1 = {}
    for eps in EPS + [1e-4]:
        T1[eps] = t1(eps)
        print("  eps=%g: t1=%.5f  slope(t1-0.01)=%+.3e  slope(t1+0.01)=%+.3e" % (
            eps, T1[eps], slope_c1(A_hat(T1[eps] - 0.01, eps), T1[eps] - 0.01),
            slope_c1(A_hat(T1[eps] + 0.01, eps), T1[eps] + 0.01)))
    print("(3) below t1: worst one-term member c Gamma(A_hat/c, 1) at the cap, tail/eps")
    for eps in EPS:
        for t in [0.01, 0.1, 0.3, 0.4, 0.43, 0.6]:
            if t >= T1[eps]:
                continue
            Ah = A_hat(t, eps)
            if Ah >= t:
                print("  eps=%g t=%g: A_hat=%.4g >= t, the drift alone gives tail 1" % (eps, t, Ah))
                continue
            c, v = worst_c(Ah, t)
            print("  eps=%g t=%g: A_hat=%.4g  c*=%.4g  tail/eps=%.6f" % (eps, t, Ah, c, v / eps))
    print("(4) one-term cap A_1/A_hat and S5 cap A_icx/A_hat")
    for eps in EPS:
        for t in [0.01, 0.1, 0.3, 1, 3, 10, 100]:
            Ah = A_hat(t, eps)
            print("  eps=%g t=%g: A_1/A_hat=%.4f  A_icx/A_hat=%.4f" % (eps, t, A_one(t, eps) / Ah, A_icx(t, eps) / Ah))
    print("(5) above t1: max over c in [1e-4, 1] of Q(A_hat/c, t/c)/eps, log grid of 120 values of t in [t1, 1000]")
    for eps in EPS:
        worst, where = 0.0, None
        min_slope = float("inf")
        n = 120
        for k in range(n + 1):
            t = T1[eps] * (1000.0 / T1[eps]) ** (k / n) * (1.0 + 1e-3)
            Ah = A_hat(t, eps)
            c, v = worst_c(Ah, t)
            if c < 1.0 and v / eps > worst:
                worst, where = v / eps, (t, c)
            min_slope = min(min_slope, slope_c1(Ah, t) / eps)
        print("  eps=%g: largest ratio at c < 1: %s; min slope/eps at c = 1: %+.3e" % (
            eps, "none above Q(A_hat, t) (c* = 1 at every t)" if where is None else "%.7f at t=%.4g c=%.4g" % (worst, where[0], where[1]), min_slope))
