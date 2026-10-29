"""Prototype of the Gamma cap functions of the WP6 plan (section 2), with reference values.

Usage: python cap_proto.py
Compares with merge/analysis/scripts/verify_gamma_sharp/ (single_peak_check_output.txt,
check_output.txt) and with an independent brute-force maximum over c.
"""
import math
import time

import numpy as np
from scipy.special import gammaincc
from scipy.optimize import brentq, minimize_scalar

_GRID_POINTS = 401
_GRID_EXTENSIONS = 10
_GOLDEN_STEPS = 80
_SHAPE_RTOL = 1e-12
_CAP_RTOL = 1e-9
_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


def one_rate_cap(t, eps):
    lo, hi = 0.0, max(1.0, 2.0 * t)
    while gammaincc(hi, t) <= eps:
        lo, hi = hi, 2.0 * hi
    while hi - lo > _SHAPE_RTOL * hi:
        mid = 0.5 * (lo + hi)
        if gammaincc(mid, t) <= eps:
            lo = mid
        else:
            hi = mid
    return lo


def worst_one_term_tail(A, t):
    grid = np.geomspace(min(1e-4, t / 100.0), 1.0, _GRID_POINTS)
    values = gammaincc(A / grid, t / grid)
    best = int(np.argmax(values))
    extensions = 0
    while best == 0:
        if extensions == _GRID_EXTENSIONS:
            raise RuntimeError(f"max below c = {grid[0]:.3g}")
        grid = np.geomspace(grid[0] / 100.0, grid[0], _GRID_POINTS)
        values = gammaincc(A / grid, t / grid)
        best = int(np.argmax(values))
        extensions += 1

    def tail(u):
        c = math.exp(u)
        return float(gammaincc(A / c, t / c))

    lo = math.log(grid[max(best - 1, 0)])
    hi = math.log(grid[min(best + 1, _GRID_POINTS - 1)])
    u1, u2 = hi - _GOLDEN * (hi - lo), lo + _GOLDEN * (hi - lo)
    f1, f2 = tail(u1), tail(u2)
    for _ in range(_GOLDEN_STEPS):
        if f1 < f2:
            lo, u1, f1 = u1, u2, f2
            u2 = lo + _GOLDEN * (hi - lo)
            f2 = tail(u2)
        else:
            hi, u2, f2 = u2, u1, f1
            u1 = hi - _GOLDEN * (hi - lo)
            f1 = tail(u1)
    return max(float(values[best]), f1, f2)


def _below_level(A, t, eps):
    return eps >= 0.5 or worst_one_term_tail(A, t) <= eps


def sharp_cap(t, eps):
    a_hat = one_rate_cap(t, eps)
    if a_hat < t and _below_level(a_hat, t, eps):
        return a_hat
    lo, hi = eps * t, min(a_hat, t)
    while hi - lo > _CAP_RTOL * hi:
        mid = 0.5 * (lo + hi)
        if _below_level(mid, t, eps):
            lo = mid
        else:
            hi = mid
    return lo


# --- independent references -------------------------------------------------
def ref_a_hat(t, eps):
    return brentq(lambda A: gammaincc(A, t) - eps, 1e-300, 10 * t + 100, xtol=1e-300, rtol=1e-15)


def ref_w(A, t):
    """Brute force: dense log grid of s = t/c in [t, t*1e12], then bounded refinement."""
    s = np.geomspace(t, t * 1e12, 20001)
    v = gammaincc(A / t * s, s)
    i = int(np.argmax(v))
    lo, hi = s[max(i - 1, 0)], s[min(i + 1, len(s) - 1)]
    r = A / t
    res = minimize_scalar(lambda ls: -gammaincc(r * math.exp(ls), math.exp(ls)),
                          bounds=(math.log(lo), math.log(hi)), method="bounded",
                          options={"xatol": 1e-13})
    return max(float(v[i]), -res.fun)


def ref_a1(t, eps):
    ah = ref_a_hat(t, eps)
    if ah < t and ref_w(ah, t) <= eps:
        return ah
    lo, hi = eps * t, min(ah, t)
    while hi - lo > 1e-13 * hi:
        mid = 0.5 * (lo + hi)
        if ref_w(mid, t) <= eps:
            lo = mid
        else:
            hi = mid
    return lo


if __name__ == "__main__":
    print("A_hat vs brentq (rel diff)")
    worst = 0.0
    for eps in (1e-6, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.99):
        for t in (1e-6, 0.01, 0.1, 0.3, 1, 3, 10, 100, 1e4):
            a, b = one_rate_cap(t, eps), ref_a_hat(t, eps)
            worst = max(worst, abs(a - b) / b)
    print(f"  largest relative difference: {worst:.2e}")

    print("A_1 at eps = 0.01 (verify: 0.00035288876 0.0035288876 0.010586663 0.043652729 3.971716)")
    for t in (0.01, 0.1, 0.3, 1, 10):
        print(f"  t={t}: A_hat={one_rate_cap(t, 0.01):.10g}  A_1={sharp_cap(t, 0.01):.10g}")
    print("r* = A_1(0.01, eps)/0.01 (verify: 0.00355013 0.0352889 0.330617)")
    for eps in (0.001, 0.01, 0.1):
        print(f"  eps={eps}: r*={sharp_cap(0.01, eps) / 0.01:.9g}")
    print("A_1/A_hat table (check_output.txt, 4 digits)")
    for eps in (0.001, 0.01, 0.1):
        row = []
        for t in (0.01, 0.1, 0.3, 1, 3, 10, 100):
            row.append(f"{sharp_cap(t, eps) / one_rate_cap(t, eps):.4f}")
        print(f"  eps={eps}: " + " ".join(row))
    print("A_1 vs brute-force reference (rel diff)")
    worst = 0.0
    for eps in (0.001, 0.01, 0.1, 0.3):
        for t in (1e-4, 0.01, 0.1, 0.4, 0.6, 1, 10, 1000):
            a, b = sharp_cap(t, eps), ref_a1(t, eps)
            d = abs(a - b) / b
            worst = max(worst, d)
            if d > 1e-7:
                print(f"  eps={eps} t={t}: {a:.12g} vs {b:.12g} ({d:.2e})")
    print(f"  largest relative difference: {worst:.2e}")
    print("eps >= 1/2: A_1/t")
    for eps in (0.5, 0.7):
        for t in (0.01, 1.0, 100.0):
            print(f"  eps={eps} t={t}: A_1/t = {sharp_cap(t, eps) / t:.12f}")
    print("w(A_1) <= eps and w(A_1 (1 + 1e-6)) > eps where A_1 < A_hat")
    for eps in (0.001, 0.01, 0.1):
        for t in (0.01, 0.1, 0.3):
            a1 = sharp_cap(t, eps)
            print(f"  eps={eps} t={t}: w(A_1)/eps={worst_one_term_tail(a1, t) / eps:.9f}  "
                  f"w(1.000001 A_1)/eps={worst_one_term_tail(a1 * 1.000001, t) / eps:.9f}")
    print("timing")
    t0 = time.perf_counter()
    n = 0
    for eps in (0.001, 0.01, 0.1):
        for t in (0.01, 0.1, 0.3, 1, 10, 100):
            sharp_cap(t, eps)
            n += 1
    print(f"  {1000 * (time.perf_counter() - t0) / n:.1f} ms per cap ({n} caps)")
