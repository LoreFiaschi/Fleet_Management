# Reviewer's independent check of the inverse Gaussian section (standard library only).
import math

def Phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2))

def F(x, lam):
    # CDF of IG(1, lam)
    r = math.sqrt(lam / x)
    second = 0.0
    z = -r * (x + 1)
    p = Phi(z)
    if p > 0:
        second = math.exp(2 * lam + math.log(p))
    return Phi(r * (x - 1)) + second

def quantile(eps, lam):
    lo, hi = -30.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 1 - F(math.exp(mid), lam) > eps:
            lo = mid
        else:
            hi = mid
    return math.exp(lo)

def tailmean_quad(q, lam, n=200000):
    # integral_q^inf z f(z) dz by substitution z = q + u/(1-u)
    def f(z):
        return math.sqrt(lam / (2 * math.pi * z ** 3)) * math.exp(-lam * (z - 1) ** 2 / (2 * z))
    s = 0.0
    h = 1.0 / n
    for k in range(1, n):
        u = k * h
        z = q + u / (1 - u)
        jac = 1 / (1 - u) ** 2
        wgt = 4 if k % 2 else 2
        s += wgt * z * f(z) * jac
    return s * h / 3

print("mu_bar/tau = eps/F(1/q); spec table rows eps = 0.001, 0.01, 0.1; columns eta = 0.05 0.25 1 4 10")
for eps in [0.001, 0.01, 0.1]:
    row = []
    for eta in [0.05, 0.25, 1, 4, 10]:
        lam = 1 / eta
        q = quantile(eps, lam)
        C = F(1 / q, lam) / eps
        row.append("%.4g" % (1 / C))
    print("  eps=%g: %s" % (eps, "  ".join(row)))

print("closed form vs quadrature of E[Z; Z>q] (eps=0.01)")
for eta in [0.05, 1, 10]:
    lam = 1 / eta
    q = quantile(0.01, lam)
    print("  eta=%g  F(1/q)=%.6g  quad=%.6g" % (eta, F(1 / q, lam), tailmean_quad(q, lam)))

print("ratio mu_bar/(tau/q) at eps=0.01 (spec: 0.93 to 0.59 for eta 0.05..10)")
for eta in [0.05, 0.25, 1, 4, 10]:
    lam = 1 / eta
    q = quantile(0.01, lam)
    C = F(1 / q, lam) / 0.01
    print("  eta=%g  q=%.5g  ratio=%.4f" % (eta, q, q / C))

print("eta=10, eps=0.1: tau/q (spec 0.473 tau); 3 iid missions with total mean tau/q: tail (spec 0.118)")
lam = 0.1
q = quantile(0.1, lam)
print("  tau/q = %.4f tau" % (1 / q))
# sum of 3 iid IG(a, a/eta): IG(3a, 9a/eta) = 3a * IG(1, 3/eta); total mean tau/q, so tail = P(IG(1, 3 lam) > q)
print("  tail = %.4f" % (1 - F(q, 3 * lam)))
