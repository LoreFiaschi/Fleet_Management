"""Standard-library tools for Gamma tails (no numpy/scipy).

Conventions: Gamma(a, b) has shape a and RATE b (mean a/b).
"""
import math
import random

EPS_ = 1e-16
FPMIN = 1e-300


def _gser(a, x):
    """Regularised lower incomplete gamma P(a, x) by its series (x < a + 1)."""
    if x <= 0.0:
        return 0.0
    ap = a
    term = 1.0 / a
    s = term
    for _ in range(100000):
        ap += 1.0
        term *= x / ap
        s += term
        if abs(term) < abs(s) * EPS_:
            break
    return s * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gcf(a, x):
    """Regularised upper incomplete gamma Q(a, x) by a continued fraction (Lentz)."""
    b = x + 1.0 - a
    c = 1.0 / FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, 100000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < FPMIN:
            d = FPMIN
        c = b + an / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS_:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gammaP(a, x):
    if x <= 0.0:
        return 0.0
    if x < a + 1.0:
        return _gser(a, x)
    return 1.0 - _gcf(a, x)


def gammaQ(a, x):
    """P(Gamma(a, 1) > x)."""
    if x <= 0.0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _gser(a, x)
    return _gcf(a, x)


def tail(a, b, t):
    """P(Gamma(shape a, rate b) > t)."""
    if a <= 0.0:
        return 0.0
    return gammaQ(a, b * t)


def shape_for_tail(p, b, t, lo=1e-300, hi=None):
    """Smallest A with P(Gamma(A, b) > t) = p (tail increases in A). Bisection."""
    if hi is None:
        hi = max(1.0, b * t)
        while tail(hi, b, t) < p:
            hi *= 2.0
    lo = 0.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if tail(mid, b, t) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-14 * hi:
            break
    return 0.5 * (lo + hi)


def negbin_pmf(a, p, mass_tol=1e-15, kmax=200000):
    """pmf of K ~ NegBin(a, p): P(K=k) = G(a+k)/(G(a) k!) p^a (1-p)^k.

    Returns (list of weights, leftover mass)."""
    if p >= 1.0:
        return [1.0], 0.0
    q = 1.0 - p
    logw = a * math.log(p)
    out = [math.exp(logw)]
    k = 0
    mean = a * q / p
    left = 1.0
    while k < kmax:
        ratio = (a + k) / (k + 1) * q      # w_{k+1} / w_k
        logw += math.log(ratio)
        k += 1
        w = math.exp(logw)
        out.append(w)
        if k > mean:
            # bound on the mass after k: ratios decrease to q if a >= 1, increase to q if a < 1
            nxt = (a + k) / (k + 1) * q
            r = nxt if a >= 1.0 else q
            if r < 1.0:
                left = w * r / (1.0 - r)
                if left < mass_tol:
                    break
    return out, left


def convolve(u, v, cut=None):
    n = len(u) + len(v) - 1
    if cut is not None:
        n = min(n, cut)
    out = [0.0] * n
    for i, ui in enumerate(u):
        if ui == 0.0 or i >= n:
            continue
        for j, vj in enumerate(v):
            if i + j >= n:
                break
            out[i + j] += ui * vj
    return out


def sum_tail(components, t, mass_tol=1e-15, cut=20000):
    """P(sum of independent Gamma(a_i, b_i) > t), exact up to truncation.

    Derivation (own): with c = max b_i, Gamma(a, b) for b <= c is the mixture of
    Gamma(a + K, c) with K ~ NegBin(a, b/c) (equal Laplace transforms). Hence the
    sum is Gamma(sum a_i + K, c) with K the sum of the independent K_i.
    Returns (lower, upper): lower drops the truncated mixture mass, upper adds it.
    """
    comps = [(a, b) for a, b in components if a > 0.0]
    if not comps:
        return 0.0, 0.0
    c = max(b for _, b in comps)
    base = sum(a for a, _ in comps)
    pmf = [1.0]
    lost = 0.0
    for a, b in comps:
        if b >= c:
            continue
        w, left = negbin_pmf(a, b / c, mass_tol)
        lost += left
        pmf = convolve(pmf, w, cut)
    kept = sum(pmf)
    lost = max(lost, 1.0 - kept)
    s = 0.0
    for k, wk in enumerate(pmf):
        if wk > 0.0:
            s += wk * gammaQ(base + k, c * t)
    return s, s + lost


def sum_tail_fast(components, t, mass_tol=1e-15, cut=200000):
    """As sum_tail, but evaluates Q(base + k, c t) by the upward recurrence
    Q(a + 1, x) = Q(a, x) + x^a e^{-x} / Gamma(a + 1) (all terms positive)."""
    comps = [(a, b) for a, b in components if a > 0.0]
    if not comps:
        return 0.0, 0.0
    c = max(b for _, b in comps)
    base = sum(a for a, _ in comps)
    pmf = [1.0]
    lost = 0.0
    for a, b in comps:
        if b >= c:
            continue
        w, left = negbin_pmf(a, b / c, mass_tol)
        lost += left
        pmf = convolve(pmf, w, cut)
    lost = max(lost, 1.0 - sum(pmf))
    x = c * t
    q = gammaQ(base, x)
    # log of x^a e^{-x} / Gamma(a + 1), kept in log space so that it cannot underflow
    # to a permanent zero when it is tiny at the start of the recursion
    log_term = base * math.log(x) - x - math.lgamma(base + 1.0)
    lx = math.log(x)
    s = 0.0
    a = base
    for k, wk in enumerate(pmf):
        s += wk * q
        q = min(1.0, q + math.exp(log_term))
        a += 1.0
        log_term += lx - math.log(a)
        if k % 500 == 499:
            q = gammaQ(a, x)  # periodic exact refresh against drift
    return s, s + lost


def mc_tail(sampler, t, n, seed=12345):
    rng = random.Random(seed)
    hits = 0
    for _ in range(n):
        if sampler(rng) > t:
            hits += 1
    p = hits / n
    se = math.sqrt(max(p * (1 - p), 1e-300) / n)
    return p, se


def chernoff_gamma(A, b, t):
    """Chernoff bound inf_s E exp(s(G - t)) for G ~ Gamma(A, b); 1 if t <= mean."""
    if A <= 0.0:
        return 0.0
    u = b * t / A
    if u <= 1.0:
        return 1.0
    return math.exp(-A * (u - 1.0 - math.log(u)))


def stoploss_gamma(A, b, x):
    """E[(G - x)^+] for G ~ Gamma(A, b)."""
    if x <= 0.0:
        return A / b - x
    return (A / b) * gammaQ(A + 1.0, b * x) - x * gammaQ(A, b * x)


def stoploss_bound(A, b, t, grid=400):
    """inf_{x < t} E[(G - x)^+] / (t - x): golden-section search on x in [0, t)."""
    if A <= 0.0:
        return 0.0
    f = lambda x: stoploss_gamma(A, b, x) / (t - x)
    lo, hi = 0.0, t * (1 - 1e-12)
    # coarse scan then golden section around the best point
    best_x, best = 0.0, f(0.0)
    for i in range(1, grid):
        x = t * i / grid
        v = f(x)
        if v < best:
            best, best_x = v, x
    lo = max(0.0, best_x - t / grid)
    hi = min(t * (1 - 1e-12), best_x + t / grid)
    g = (math.sqrt(5) - 1) / 2
    x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
    f1, f2 = f(x1), f(x2)
    for _ in range(100):
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - g * (hi - lo)
            f1 = f(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + g * (hi - lo)
            f2 = f(x2)
    return min(best, f1, f2)


def cap_for_bound(bound, b, t, eps):
    """Largest A with bound(A, b, t) <= eps (bound increasing in A). Bisection."""
    lo, hi = 0.0, max(1.0, b * t)
    while bound(hi, b, t) <= eps:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if bound(mid, b, t) <= eps:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-13 * hi:
            break
    return lo
