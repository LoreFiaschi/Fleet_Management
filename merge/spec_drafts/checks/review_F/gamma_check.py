# Reviewer's independent check of the Gamma section of spec_assembled.tex (standard library only).
import math, random, sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gq import Q

def bisect(f, lo, hi, n=200):
    # f(lo) <= 0 < f(hi) style; returns lower end
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        if f(mid) <= 0:
            lo = mid
        else:
            hi = mid
    return lo

def A_hat(t, eps):
    hi = 1.0
    while Q(hi, t) < eps:
        hi *= 2
    return bisect(lambda A: Q(A, t) - eps, 0.0, hi)

def w(A, t, ngrid=2000):
    # max over c in (0,1] of Q(A/c, t/c); log grid down to 1e-7 plus golden refinement
    lo = math.log(1e-7)
    best, bi = -1, 0
    cs = [math.exp(lo + (0 - lo) * i / (ngrid - 1)) for i in range(ngrid)]
    vals = [Q(A / c, t / c) for c in cs]
    bi = max(range(ngrid), key=lambda i: vals[i])
    a = cs[max(bi - 1, 0)]; b = cs[min(bi + 1, ngrid - 1)]
    g = (math.sqrt(5) - 1) / 2
    f = lambda c: -Q(A / c, t / c)
    x1 = b - g * (b - a); x2 = a + g * (b - a)
    f1, f2 = f(x1), f(x2)
    for _ in range(80):
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - g * (b - a); f1 = f(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + g * (b - a); f2 = f(x2)
    return max(vals[bi], -f1, -f2), cs[bi]

def A_1(t, eps):
    Ah = A_hat(t, eps)
    if Ah < t and w(Ah, t)[0] <= eps:
        return Ah
    return bisect(lambda A: w(A, t)[0] - eps, eps * t, min(Ah, t), n=60)

print("(1) A_1/A_hat at eps = 0.01; spec table: 0.142 0.642 0.961 1 1")
for t in [0.01, 0.1, 0.3, 1, 10]:
    Ah = A_hat(t, 0.01); A1 = A_1(t, 0.01)
    print("  t=%g  A_hat=%.7g  A_1=%.7g  ratio=%.4f" % (t, Ah, A1, A1 / Ah))

print("(2) witness: eps=0.01, t=0.1, A_hat=0.0055012 (spec); mission rate 4.56 beta*, tail 1.565 eps")
t, eps = 0.1, 0.01
Ah = A_hat(t, eps)
c = 1 / 4.56
print("  A_hat=%.7f  tail at c=1/4.56: %.5f eps" % (Ah, Q(Ah / c, t / c) / eps))

print("(3) r*: max_s Q(r s, s) = eps; spec: 0.00355, 0.0353, 0.331")
def wfree(r):
    # max over s in (0, 1e4] of Q(r s, s)
    best = 0
    for i in range(3000):
        s = math.exp(math.log(1e-5) + (math.log(1e4) - math.log(1e-5)) * i / 2999)
        best = max(best, Q(r * s, s))
    return best
for eps in [0.001, 0.01, 0.1]:
    r = bisect(lambda r: wfree(r) - eps, 0.0, 1.0, n=40)
    print("  eps=%g  r*=%.5f" % (eps, r))

print("(4) t1: sign change of d/dc Q(A_hat/c, t/c) at c=1; spec: 0.436, 0.448, 0.614")
def dQdc(t, eps):
    Ah = A_hat(t, eps); h = 1e-5
    return (Q(Ah / (1 + h), t / (1 + h)) - Q(Ah / (1 - h), t / (1 - h))) / (2 * h)
for eps in [0.001, 0.01, 0.1]:
    lo, hi = 0.05, 3.0
    slo = dQdc(lo, eps)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if (dQdc(mid, eps) > 0) == (slo > 0):
            lo = mid
        else:
            hi = mid
    print("  eps=%g  t1=%.4f  (sign of dQ/dc below t1: %s)" % (eps, lo, "+" if slo > 0 else "-"))
