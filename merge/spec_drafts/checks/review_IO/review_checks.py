"""Independent checks of the reviewer (spec v1.0, inputs/outputs/tools). Standard library only.

Run: python -I review_checks.py
"""
import math
import random
from statistics import NormalDist

N = NormalDist()
out = []


def say(msg):
    print(msg)
    out.append(msg)


# ---------------------------------------------------------------------------
# 1. Bernstein tangent row: when is it empty?
#    f(mu) = ((tau-mu)^2 - 2c(tau-mu)) / (2 kappa), side row mu <= tau - c, v >= 0, mu >= 0.
#    Tangent at mu_r: T(mu) = f(mu_r) + f'(mu_r)(mu - mu_r).
# ---------------------------------------------------------------------------
def f_b(mu, tau, c, kap):
    d = tau - mu
    return (d * d - 2 * c * d) / (2 * kap)


def fp_b(mu, tau, c, kap):
    d = tau - mu
    return -(2 * d - 2 * c) / (2 * kap)


def tangent_nonempty(mu_r, tau, c, kap, n=20001):
    """Grid search: is there mu in [0, tau - c] with T(mu) >= 0 (so some v >= 0 fits)?"""
    best = -1e300
    for s in range(n):
        mu = (tau - c) * s / (n - 1)
        t = f_b(mu_r, tau, c, kap) + fp_b(mu_r, tau, c, kap) * (mu - mu_r)
        best = max(best, t)
    return best >= 0.0, best


random.seed(1)
gap_cases = 0
spec_rule_wrong = 0
proposed_rule_wrong = 0
for _ in range(400):
    tau = random.uniform(0.2, 3.0)
    eps = random.uniform(0.001, 0.3)
    kap = math.log(1 / eps)
    # support constant b with tau > 2c
    cmax = tau / 2 * 0.999
    c = random.uniform(0.001, cmax)
    lo = math.sqrt(tau * (tau - 2 * c))
    hi = tau - c
    for mu_r in (lo * 0.999, (lo + hi) / 2, hi * 0.999):
        ne, best = tangent_nonempty(mu_r, tau, c, kap)
        spec_ok = mu_r < tau - c                      # spec, line 667
        prop_ok = mu_r < lo                           # proposed: mu_r < sqrt(tau (tau - 2c))
        if spec_ok and not ne:
            spec_rule_wrong += 1
        if prop_ok != ne:
            proposed_rule_wrong += 1
    gap_cases += 1
say(f"[1] Bernstein tangent: {gap_cases} random (tau, eps, c); cases accepted by the spec rule "
    f"mu_r < tau - c but empty: {spec_rule_wrong}; mismatches of the rule mu_r < sqrt(tau(tau-2c)): "
    f"{proposed_rule_wrong}")

# worked example for the review
tau, eps = 1.0, 0.1
kap = math.log(10)
b = 0.3
c = kap * b / 3
lo = math.sqrt(tau * (tau - 2 * c))
say(f"[1] example tau=1, eps=0.1, b=0.3: c={c:.4f}, tau-c={tau-c:.4f}, sqrt(tau(tau-2c))={lo:.4f}")
for tr in (0.70, 0.75):
    ne, best = tangent_nonempty(tr * tau, tau, c, kap)
    say(f"    tangent_ref={tr}: spec check passes={tr*tau < tau - c}; row nonempty={ne}; "
        f"max over mu of tangent RHS={best:.5f}; T(0)=f(0)-c2*mu_r^2={f_b(0,tau,c,kap) - (tr*tau)**2/(2*kap):.5f}")

# pwl with P segments: first midpoint tau/(2P) must satisfy the same rule
for P in (1, 2):
    tau, eps = 1.0, 0.1
    kap = math.log(10)
    b = 3 * 0.4 / kap        # c = 0.4 (> 3 tau / 8 = 0.375), tau > 2c holds
    c = kap * b / 3
    mid = tau / (2 * P)
    # pwl: segment p selects mu in [(p-1) tau/P, p tau/P]; cap by tangent at midpoint
    feasible = False
    for p in range(1, P + 1):
        a0, a1 = (p - 1) * tau / P, min(p * tau / P, tau - c)
        if a0 > a1:
            continue
        m = (p - 0.5) * tau / P
        for s in range(2001):
            mu = a0 + (a1 - a0) * s / 2000
            if f_b(m, tau, c, kap) + fp_b(m, tau, c, kap) * (mu - m) >= 0:
                feasible = True
                break
    say(f"[1] pwl P={P}, tau=1, c=0.4: nonempty={feasible} (rule tau/(2P) <= sqrt(tau(tau-2c))="
        f"{math.sqrt(tau*(tau-2*c)):.4f}: {mid <= math.sqrt(tau*(tau-2*c))})")

# ---------------------------------------------------------------------------
# 2. Popoviciu with the mean (non-negative variable): sharp maximum variance
# ---------------------------------------------------------------------------
w, mu = 0.2, 0.01
sharp = min(mu, w / 2) * (w - min(mu, w / 2))
say(f"[2] non-negative X, width {w}, mean {mu}: largest variance {sharp:.5f}; Popoviciu cap {w*w/4:.5f}")

# ---------------------------------------------------------------------------
# 3. YAML example, recomputed independently
# ---------------------------------------------------------------------------
def q_upper(a, x):
    """Regularised upper incomplete gamma Q(a, x) (series / continued fraction)."""
    if x <= 0:
        return 1.0
    if x < a + 1:
        s, term, n = 1.0 / a, 1.0 / a, 0
        while True:
            n += 1
            term *= x / (a + n)
            s += term
            if abs(term) < abs(s) * 1e-15 or n > 100000:
                break
        p = s * math.exp(-x + a * math.log(x) - math.lgamma(a))
        return 1.0 - p
    # Lentz
    tiny = 1e-300
    b0 = x + 1 - a
    c0 = 1 / tiny
    d = 1 / b0
    h = d
    for i in range(1, 100000):
        an = -i * (i - a)
        b0 += 2
        d = an * d + b0
        if abs(d) < tiny:
            d = tiny
        c0 = b0 + an / c0
        if abs(c0) < tiny:
            c0 = tiny
        d = 1 / d
        de = d * c0
        h *= de
        if abs(de - 1) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def bisect(fun, lo, hi, it=200):
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if fun(mid):
            lo = mid
        else:
            hi = mid
    return lo


# Gamma sharp cap A_1(t, eps) = largest A with max_{c in (0,1]} Q(A/c, t/c) <= eps
def worst_tail(A, t):
    best = 0.0
    for s in range(1, 401):
        cc = s / 400
        best = max(best, q_upper(A / cc, t / cc))
    return best


t, eps = 20.0, 0.05
A_exact = bisect(lambda A: q_upper(A, t) <= eps, 1e-6, t)
A_1 = bisect(lambda A: worst_tail(A, t) <= eps, 1e-6, A_exact * 1.01, it=60)
say(f"[3] Gamma cell: one-rate cap A={A_exact:.4f} (mean cap {A_exact/20:.4f}); sharp cap A_1 ~ {A_1:.4f} "
    f"(mean cap {A_1/20:.4f}); replacement_mu 0.02 below: {0.02 <= A_1/20}")

# IG cap: Z ~ IG(1, lam), lam = 1/eta; mean cap = tau / E[Z | Z > q]
eta = 0.15
lam = 1 / eta


def ig_cdf(z):
    a = math.sqrt(lam / z)
    return N.cdf(a * (z - 1)) + math.exp(2 * lam) * N.cdf(-a * (z + 1))


def ig_pdf(z):
    return math.sqrt(lam / (2 * math.pi * z ** 3)) * math.exp(-lam * (z - 1) ** 2 / (2 * z))


q = bisect(lambda z: ig_cdf(z) <= 1 - eps, 1e-9, 50.0)
# E[Z ; Z > q] by Simpson on [q, 40]
n = 200000
hi = 40.0
hstep = (hi - q) / n
acc = 0.0
for k in range(n + 1):
    z = q + k * hstep
    wgt = 1 if k in (0, n) else (4 if k % 2 else 2)
    acc += wgt * z * ig_pdf(z)
tail_mean = acc * hstep / 3 / eps
say(f"[3] IG cell: q={q:.4f}, E[Z|Z>q]={tail_mean:.4f}, mean cap={1/tail_mean:.4f}")

kG = N.inv_cdf(0.95)
say(f"[3] V_max(eps=0.05)={1/kG**2:.4f}; Bernstein (eps 0.1, b 0.05): c={math.log(10)*0.05/3:.5f}, "
    f"V^B={(1-2*math.log(10)*0.05/3)/(2*math.log(10)):.4f}; Chernoff K range={5-math.log(10):.4f}")
# Gaussian (1,1): tangent at 0.4, replacement state (0.02, 4e-5)
f0 = lambda m: (1 - m) ** 2 / kG ** 2
T = f0(0.4) - 2 * (1 - 0.4) / kG ** 2 * (0.02 - 0.4)
say(f"[3] Gaussian tangent row at the replacement state: RHS {T:.4f} >= v_new 4e-5: {T >= 4e-5}")
say(f"[3] three-sigma: mu_0 0.12 >= {3*math.sqrt(0.0004):.3f}; mu_inc >= 0.05 >= {3*math.sqrt(0.0001):.3f}")
say(f"[3] Popoviciu Be cell: v_0 4e-4 <= {0.05**2/4}; v_new 4e-5 <= {0.03**2/4}; v_inc 1e-4 <= {0.04**2/4}")
say(f"[3] Jensen Ch cell: cgf 0.22 >= {5*0.04:.2f}, 0.38 >= {5*0.07:.2f}; anchor_K_0 0.50 >= s*anchor_mu_0 = {5*0.10:.2f}")
d = 1 - 0.02
cB = math.log(10) * 0.05 / 3
say(f"[3] Bernstein exact row at replacement: d^2-2cd-2kv = {d*d-2*cB*d-2*math.log(10)*4e-5:.4f} >= 0; side row {0.02 <= 1 - cB}")

with open(__file__.replace(".py", "_output.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(out) + "\n")
