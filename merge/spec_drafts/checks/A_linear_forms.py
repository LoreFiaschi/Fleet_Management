"""Writer A: spot checks for sec:linear_forms (standard library only).

1. Proposition prop:a_tangent: tangent = f - c2 (mu - mu_r)^2 <= f, for the four
   rows of Table tab:a_quad, random points (spot check) and mu_r = 0 (boundary).
2. pwl: loss <= c2 tau^2 / (4 P^2); the segment tangent equals max over tangents.
3. Worked example (Cantelli, eps = 0.1, tau = 1, v = 0.004).
4. Boundary: tangent_ref = 1 with c1 = 0 gives Q <= 0.
"""
import math
import random

random.seed(7)


def norm_ppf(p):
    # bisection on the standard normal CDF (erf based)
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def rows(eps, tau, b):
    kappa = math.log(1 / eps)
    z = norm_ppf(1 - eps)
    c = kappa * b / 3
    return {
        "gaussian": (1 / z ** 2, 0.0, tau),
        "cantelli": (eps / (1 - eps), 0.0, tau),
        "hoeffding": (2 / kappa, 0.0, tau),
        "bernstein": (1 / (2 * kappa), -b / 3, tau - c),
    }


def f(mu, c2, c1, tau):
    d = tau - mu
    return c2 * d * d + c1 * d


def fp(mu, c2, c1, tau):
    return -2 * c2 * (tau - mu) - c1


def check1(n=20000):
    worst_identity = 0.0
    worst_below = -1.0
    for _ in range(n):
        eps = random.uniform(0.001, 0.4)
        tau = random.uniform(0.2, 5)
        b = random.uniform(0, 0.2 * tau)
        for name, (c2, c1, side) in rows(eps, tau, b).items():
            for mu_r in (0.0, random.uniform(0, 0.999) * tau):
                mu = random.uniform(-tau, 2 * tau)
                t = f(mu_r, c2, c1, tau) + fp(mu_r, c2, c1, tau) * (mu - mu_r)
                worst_identity = max(worst_identity,
                                     abs(t - (f(mu, c2, c1, tau) - c2 * (mu - mu_r) ** 2)))
                worst_below = max(worst_below, t - f(mu, c2, c1, tau))
    return worst_identity, worst_below


def check2(n=20000):
    worst_ratio = 0.0
    max_mismatch = 0.0
    for _ in range(n):
        tau = random.uniform(0.2, 5)
        c2 = random.uniform(0.01, 10)
        c1 = -random.uniform(0, 1)
        P = random.randint(1, 16)
        mu = random.uniform(0, tau)
        p = min(int(mu / (tau / P)), P - 1) + 1
        mid = (p - 0.5) * tau / P
        seg_t = f(mid, c2, c1, tau) + fp(mid, c2, c1, tau) * (mu - mid)
        loss = f(mu, c2, c1, tau) - seg_t
        worst_ratio = max(worst_ratio, loss / (c2 * tau ** 2 / (4 * P ** 2)))
        best = max(f((q - 0.5) * tau / P, c2, c1, tau)
                   + fp((q - 0.5) * tau / P, c2, c1, tau) * (mu - (q - 0.5) * tau / P)
                   for q in range(1, P + 1))
        max_mismatch = max(max_mismatch, best - seg_t)
    return worst_ratio, max_mismatch


def largest_mu(cap, tau, v, lo=0.0):
    """Largest mu in [0, tau] with v <= cap(mu) (cap decreasing in mu here)."""
    hi = tau
    if v > cap(lo):
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if v <= cap(mid):
            lo = mid
        else:
            hi = mid
    return lo


def check3():
    eps, tau, v = 0.1, 1.0, 0.004
    c2, c1 = eps / (1 - eps), 0.0
    exact = largest_mu(lambda m: f(m, c2, c1, tau), tau, v)
    tan0 = largest_mu(lambda m: f(0, c2, c1, tau) + fp(0, c2, c1, tau) * m, tau, v)
    tan5 = largest_mu(lambda m: f(0.5, c2, c1, tau) + fp(0.5, c2, c1, tau) * (m - 0.5), tau, v)
    P = 8

    def pwl_cap(m):
        p = min(int(m / (tau / P)), P - 1) + 1
        mid = (p - 0.5) * tau / P
        return f(mid, c2, c1, tau) + fp(mid, c2, c1, tau) * (m - mid)
    # pwl: scan (cap is not monotone across segment ends in general)
    best = 0.0
    N = 200000
    for i in range(N + 1):
        m = tau * i / N
        if v <= pwl_cap(m):
            best = m
    tan0_v0 = largest_mu(lambda m: f(0, c2, c1, tau) + fp(0, c2, c1, tau) * m, tau, 0.0)
    return exact, tan0, tan5, best, tan0_v0


def check4():
    tau = 1.0
    c2, c1 = 9.0, 0.0
    return f(tau, c2, c1, tau), fp(tau, c2, c1, tau)


def check5(n=400, grid=2000):
    """Bernstein: the tangent row at mu_r admits a point with v > 0, mu in
    [0, tau - c], iff mu_r < sqrt(tau (tau - 2c)); the pwl form iff
    tau/(2P) < sqrt(tau (tau - 2c)). Grid search against the rule."""
    mism_t = mism_p = 0
    cases = 0
    for _ in range(n):
        tau = random.uniform(0.2, 5)
        eps = random.uniform(0.001, 0.3)
        kappa = math.log(1 / eps)
        c = random.uniform(0.0, 0.499) * tau
        b = 3 * c / kappa
        c2, c1 = 1 / (2 * kappa), -b / 3
        side = tau - c
        thr = math.sqrt(tau * (tau - 2 * c))
        mus = [side * i / grid for i in range(grid + 1)]
        for mu_r in (random.uniform(0, tau), thr * random.uniform(0.98, 1.02)):
            if abs(mu_r - thr) < 1e-9:
                continue
            cases += 1
            feas = any(f(mu_r, c2, c1, tau) + fp(mu_r, c2, c1, tau) * (m - mu_r) > 1e-12
                       for m in mus)
            if feas != (mu_r < thr):
                mism_t += 1
        for P in (1, 2, 3, 8):
            mid1 = tau / (2 * P)
            if abs(mid1 - thr) < 1e-9:
                continue

            def cap(m):
                p = min(int(m / (tau / P)), P - 1) + 1
                mid = (p - 0.5) * tau / P
                return f(mid, c2, c1, tau) + fp(mid, c2, c1, tau) * (m - mid)
            feas = any(cap(m) > 1e-12 for m in mus)
            if feas != (mid1 < thr):
                mism_p += 1
    return cases, mism_t, mism_p


def check6():
    """Reviewer's example: tau = 1, eps = 0.01, c = 0.1: sqrt(0.8) = 0.894 < 0.9;
    tangent_ref = 0.897 gives T(0) < 0."""
    tau, eps, c = 1.0, 0.01, 0.1
    kappa = math.log(1 / eps)
    T0 = (tau * (tau - 2 * c) - 0.897 ** 2) / (2 * kappa)
    return round(math.sqrt(0.8), 4), round(T0, 6)


if __name__ == "__main__":
    print("5. Bernstein tangent/pwl condition: cases, tangent mismatches, pwl mismatches =", check5())
    print("6. reviewer example: sqrt(tau(tau-2c)), T(0) at mu_r = 0.897 =", check6())
    print("1. max |tangent - (f - c2 (mu-mu_r)^2)| =", check1()[0],
          "; max (tangent - f) =", check1()[1], "(must be <= 0)")
    print("2. pwl: max loss / (c2 tau^2/(4P^2)) =", check2()[0],
          "(must be <= 1); max (best tangent - segment tangent) =", check2()[1], "(must be ~0)")
    e, t0, t5, pw, t0v0 = check3()
    print("3. Cantelli example: exact %.4f, tangent@0 %.4f, tangent@0.5 %.4f, pwl(8) %.4f; tangent@0 at v=0: %.4f"
          % (e, t0, t5, pw, t0v0))
    print("4. tangent at tau with c1 = 0: f(tau), f'(tau) =", check4())
