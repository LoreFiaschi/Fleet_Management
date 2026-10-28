"""Gamma sharp cap: fast recomputation with scipy (R2 review).
Q(a, x) = gammaincc(a, x). Recomputes A_hat, t1, r*, A_1/A_hat, the 4.56-rate example,
the parameter-space repair example (0.0133), and checks the single-peak claim
for small A/t (below the stated range 0.05)."""
import numpy as np
from scipy.special import gammaincc
from scipy.optimize import brentq, minimize_scalar

Q = gammaincc

def A_hat(t, eps):
    return brentq(lambda A: Q(A, t) - eps, 1e-300, 10 * t + 100, xtol=1e-300, rtol=1e-15)

def peak_s(r, smin=1e-12, smax=1e8, n=4000):
    s = np.logspace(np.log10(smin), np.log10(smax), n)
    v = Q(r * s, s)
    return s, v

def w(A, t):
    r = A / t
    s = np.logspace(np.log10(t), np.log10(t) + 9, 3000)
    v = Q(r * s, s)
    i = int(np.argmax(v))
    lo, hi = s[max(i - 1, 0)], s[min(i + 1, len(s) - 1)]
    res = minimize_scalar(lambda ls: -Q(r * np.exp(ls), np.exp(ls)), bounds=(np.log(lo), np.log(hi)), method='bounded', options={'xatol': 1e-12})
    return max(v[i], -res.fun, Q(A, t))

def A1(t, eps):
    Ah = A_hat(t, eps)
    if Ah < t and w(Ah, t) <= eps:
        return Ah, Ah
    lo, hi = eps * t, min(Ah, t)
    while (hi - lo) > 1e-12 * hi:
        mid = (lo + hi) / 2
        if w(mid, t) <= eps:
            lo = mid
        else:
            hi = mid
    return lo, Ah

if __name__ == '__main__':
    print('A_1/A_hat at eps = 0.01 (scoping table: 0.142 0.642 0.961 1 1)')
    for t in (0.01, 0.1, 0.3, 1, 10):
        a1, ah = A1(t, 0.01)
        print(f'  t={t}: A_hat={ah:.8g} A_1={a1:.8g} A_1/A_hat={a1/ah:.6f}  A_1/t={a1/t:.6f}')
    print('r* for eps = 0.001, 0.01, 0.1 (scoping: 0.00355 0.0353 0.331)')
    for eps in (0.001, 0.01, 0.1):
        a1, ah = A1(0.01, eps)
        print(f'  eps={eps}: r*={a1/0.01:.6g}')
    # example: eps 0.01, t = 0.1, one mission mean A_hat/beta*, rate b = beta*/c
    t, eps = 0.1, 0.01
    Ah = A_hat(t, eps)
    cs = np.linspace(0.05, 1, 200000)
    tails = Q(Ah / cs, t / cs)
    i = int(np.argmax(tails))
    print(f'example t=0.1 eps=0.01: A_hat={Ah:.8g}, worst c*={cs[i]:.5f}, rate={1/cs[i]:.4f} beta*, tail/eps={tails[i]/eps:.5f}')
    print(f'  with rate exactly 4.56 beta*: tail/eps={Q(Ah*4.56, t*4.56)/eps:.5f}')
    # single peak of s -> Q(r s, s) on (0, inf) for small r
    print('single peak check of s -> Q(r s, s) on s in [1e-12, 1e8]:')
    bad = 0
    for r in np.logspace(-5, np.log10(0.999), 400):
        s, v = peak_s(r)
        d = np.diff(v)
        sg = np.sign(d[np.abs(d) > 1e-15 * np.max(v)])
        changes = int(np.sum(sg[1:] != sg[:-1]))
        if changes > 1:
            bad += 1
            print('  r', r, 'sign changes', changes)
    print('  r values with more than one sign change of the slope:', bad)
    for r in (1e-4, 0.00355, 0.0353, 0.05, 0.331):
        s, v = peak_s(r)
        print(f'  r={r}: peak at s={s[np.argmax(v)]:.5g}, value {np.max(v):.5g}')
