"""Writer B, spec v1.0, Gamma section: spot checks of the numbers and remarks.

Standard library only. Run: python -I B_gamma.py > B_gamma_output.txt
Rate-1 units: t = beta* tau, A = beta* mu. Q(a, x) = P(Gamma(a, 1) > x).

Checks:
 (1) A_hat(t, eps): Q(A_hat, t) = eps, against the T2 table of merge/analysis/gamma_sharp.md.
 (2) A_1(t, eps) by the computation the spec prescribes: bisection on A in [eps t, min(A_hat, t)),
     inner maximum over c in (0, 1] of Q(A/c, t/c) (log grid, then golden section).
     A_1/A_hat against the T2 table (eps = 0.01).
 (3) Below t1(eps): A_1(t, eps)/t is one number r*(eps) (the mean cap is r* tau).
 (4) Remark: A_1(t)/t is nondecreasing in t (so the smallest rate is the best safe beta*).
 (5) Boundary case: the Markov value A = eps t satisfies max_c Q(A/c, t/c) <= eps.
 (6) Single peak of c -> Q(A/c, t/c) on a fine grid (justifies grid plus golden section).
 (7) Witness: eps = 0.01, t = 0.1, a new part with one mission of mean A_hat and scale c*:
     tail / eps, and the rate ratio 1/c*.
"""
import math


def Q(a, x):
    """Regularised upper incomplete gamma function Q(a, x) = P(Gamma(a, 1) > x)."""
    if x <= 0.0:
        return 1.0
    lg = a * math.log(x) - x - math.lgamma(a)
    if x < a + 1.0:
        term = 1.0 / a
        s = term
        n = 0
        while True:
            n += 1
            term *= x / (a + n)
            s += term
            if abs(term) < abs(s) * 1e-17 or n > 200000:
                break
        return max(0.0, 1.0 - math.exp(lg) * s)
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
        if abs(de - 1.0) < 1e-16 or i > 200000:
            break
    return math.exp(lg) * h


def A_hat(t, eps):
    lo, hi = math.log(1e-14), math.log(10.0 * t + 50.0)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if Q(math.exp(mid), t) < eps:
            lo = mid
        else:
            hi = mid
    return math.exp(0.5 * (lo + hi))


def tail(A, t, c):
    return Q(A / c, t / c)


def w(A, t, n=400):
    """max over c in (0, 1] of Q(A/c, t/c): log grid on [min(1e-4, t/100), 1], golden section."""
    cmin = min(1e-4, t / 100.0)
    grid = [math.exp(math.log(cmin) * (1.0 - k / n)) for k in range(n + 1)]
    vals = [tail(A, t, c) for c in grid]
    kb = max(range(n + 1), key=lambda k: vals[k])
    best, best_c = vals[kb], grid[kb]
    if kb == n:
        return best, best_c
    lo, hi = grid[max(kb - 1, 0)], grid[min(kb + 1, n)]
    g = (math.sqrt(5.0) - 1.0) / 2.0
    x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
    f1, f2 = tail(A, t, x1), tail(A, t, x2)
    for _ in range(100):
        if f1 < f2:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + g * (hi - lo)
            f2 = tail(A, t, x2)
        else:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - g * (hi - lo)
            f1 = tail(A, t, x1)
    c = 0.5 * (lo + hi)
    v = tail(A, t, c)
    if v > best:
        best, best_c = v, c
    return best, best_c


def A_one(t, eps):
    """The spec's computation: bisection on A in [eps t, min(A_hat, t)); returns the lower end."""
    Ah = A_hat(t, eps)
    hi = min(Ah, t)
    if hi == Ah and w(Ah, t)[0] <= eps:
        return Ah
    lo = eps * t
    assert w(lo, t)[0] <= eps
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if w(mid, t)[0] <= eps:
            lo = mid
        else:
            hi = mid
    return lo


if __name__ == "__main__":
    EPS = [0.001, 0.01, 0.1]
    TS = [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0]
    print("(1) A_hat(t, eps); T2 of gamma_sharp.md: eps=0.01 row 0.0024877 0.0055012 0.011021 0.043653 0.39997 3.9717 77.997")
    for eps in EPS:
        print("  eps=%g: " % eps + "  ".join("%.5g" % A_hat(t, eps) for t in TS))
    print("(2) A_1/A_hat, eps = 0.01; T2: 0.142 0.642 0.961 1 1 1 1")
    print("  " + "  ".join("t=%g: %.4f" % (t, A_one(t, 0.01) / A_hat(t, 0.01)) for t in TS))
    print("(3) below t1 (0.436, 0.448, 0.614): A_1(t, eps)/t; decisions.md: 0.00355, 0.0353, 0.331")
    for eps in EPS:
        print("  eps=%g: " % eps + "  ".join("t=%g: %.6f" % (t, A_one(t, eps) / t) for t in [0.001, 0.01, 0.1, 0.3, 0.4]))
    print("(4) A_1(t)/t nondecreasing in t (eps = 0.01), t on a log grid 0.001 .. 100")
    prev, ok = 0.0, True
    for k in range(26):
        t = 0.001 * 10 ** (5.0 * k / 25)
        r = A_one(t, 0.01) / t
        if r < prev * (1 - 1e-9):
            ok = False
            print("  decrease at t=%g: %.6g < %.6g" % (t, r, prev))
        prev = r
    print("  nondecreasing on the grid:", ok)
    print("(5) boundary case, Markov value A = eps t: max_c Q(A/c, t/c)/eps <= 1")
    for eps in EPS:
        print("  eps=%g: " % eps + "  ".join("t=%g: %.4f" % (t, w(eps * t, t)[0] / eps) for t in TS))
    print("(6) single peak of c -> Q(A/c, t/c): count of local maxima on a 4000-point log grid")
    worst = 0
    for t in [0.001, 0.01, 0.1, 0.3, 0.5, 1.0, 3.0, 10.0, 100.0]:
        for frac in [0.05, 0.2, 0.5, 0.8, 0.95, 0.999]:
            A = frac * t
            n = 4000
            cmin = min(1e-5, t / 1000.0)
            grid = [math.exp(math.log(cmin) * (1.0 - k / n)) for k in range(n + 1)]
            vals = [tail(A, t, c) for c in grid]
            peaks = sum(1 for k in range(1, n) if vals[k] > vals[k - 1] and vals[k] > vals[k + 1])
            if vals[n] > vals[n - 1]:
                peaks += 1
            worst = max(worst, peaks)
            if peaks != 1:
                print("  t=%g A/t=%g: %d local maxima" % (t, frac, peaks))
    print("  largest number of local maxima over 54 cases:", worst)
    print("(7) witness, eps = 0.01, t = 0.1: one mission of mean A_hat at scale c*")
    Ah = A_hat(0.1, 0.01)
    v, c = w(Ah, 0.1)
    print("  A_hat=%.6g  c*=%.4f  rate ratio 1/c*=%.3f  tail/eps=%.4f" % (Ah, c, 1.0 / c, v / 0.01))
    A1 = A_one(0.1, 0.01)
    print("  A_1=%.6g  A_1/A_hat=%.4f  w(A_1)/eps=%.6f" % (A1, A1 / Ah, w(A1, 0.1)[0] / 0.01))
    print("(8) w(A) < 1/2 for every A < t (so for eps >= 1/2 every A < t qualifies), and w -> 1/2 as A -> t")
    worst = 0.0
    for t in [0.01, 0.1, 1.0, 10.0, 100.0]:
        for r in [0.5, 0.9, 0.99, 0.999]:
            worst = max(worst, w(r * t, t, n=800)[0])
    print("  largest w(A) over t in {0.01, ..., 100}, A/t up to 0.999: %.6f (< 0.5)" % worst)
