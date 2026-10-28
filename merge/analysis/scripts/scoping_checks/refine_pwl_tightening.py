"""Piecewise-linear tightening of the tangent surrogates: checks of scoping.tex.

Checks the paragraph "Piecewise-linear tightening of the tangent surrogates." of the
subsubsection "Two refinements of the Route-I constraints ..." (label sec:routeI_refinements),
Appendix drv:pwl_recovery, and the sentence "Neither refinement changes the loop argument".
One numbered check per claim; each prints PASS or FAIL.

Notation: kC2 = kappa_C^2 = (1 - eps)/eps, f(mu) = (tau - mu)^2, tangent at mu_r:
T_r(mu) = (tau - mu_r)(tau + mu_r - 2 mu). Regions in the (mu, v) plane, v >= 0:
exact kC2 v <= f(mu); single kC2 v <= T_0(mu); union kC2 v <= max_r T_r(mu);
intersection kC2 v <= min_r T_r(mu).

Deterministic (fixed seed). Run: python refine_pwl_tightening.py > refine_pwl_tightening_output.txt
"""
import math

import numpy as np
import sympy as sp

RNG = np.random.default_rng(20261028)


def report(tag, ok, text):
    print("[%s] %s: %s" % ("PASS" if ok else "FAIL", tag, text))


mu, mur, tau, v, eps, kap = sp.symbols("mu mu_r tau v epsilon kappa", real=True)
d = sp.symbols("d", positive=True)


def symbolic():
    print("=== Identities (sympy) ===")
    # Cantelli <= eps  <=>  mu + kC sqrt(v) <= tau, for d = tau - mu > 0, v >= 0, 0 < eps < 1
    vv = sp.symbols("vv", nonnegative=True)
    e = sp.symbols("e", positive=True)
    lhs = vv / (vv + d ** 2) - e                     # <= 0
    rhs = (1 - e) / e * vv - d ** 2                  # <= 0  (kC2 v <= d^2)
    report("I1", sp.simplify(lhs * (vv + d ** 2) / e - rhs) == 0,
           "(v/(v+d^2) - eps) (v+d^2)/eps = kC2 v - d^2, so Cantelli <= eps <=> kC2 v <= d^2 <=> mu + kC sqrt(v) <= tau")
    f = (tau - mu) ** 2
    T = f.subs(mu, mur) + sp.diff(f, mu).subs(mu, mur) * (mu - mur)
    report("I2", sp.expand(T - (tau - mur) * (tau + mur - 2 * mu)) == 0,
           "tangent of (tau - mu)^2 at mu_r = (tau - mu_r)(tau + mu_r - 2 mu)")
    report("I3", sp.expand(f - T - (mu - mur) ** 2) == 0, "f - T_r = (mu - mu_r)^2 (Appendix drv:pwl_recovery)")
    report("I4", sp.expand(f - (tau ** 2 - 2 * tau * mu) - mu ** 2) == 0,
           "f - (tau^2 - 2 tau mu) = mu^2: the tangent at 0 loses mu^2/kC2 of variance")
    # tangent of sqrt at v_r, half-plane identity
    kC = sp.symbols("k_C", positive=True)
    sr = (tau - mur) / kC                               # sqrt(v_r) for tau > mu_r
    tr = sr + (v - sr ** 2) / (2 * sr)                  # tangent to sqrt at v_r = sr^2
    expr = 2 * (tau - mur) * (tau - mu - kC * tr)       # (tau - mu - kC t_r(v)) * 2 (tau - mu_r) > 0 multiple
    report("I5", sp.simplify(sp.cancel(expr - ((tau - mur) * (tau + mur - 2 * mu) - kC ** 2 * v))) == 0,
           "2 (tau - mu_r)(tau - mu - kC t_r(v)) = T_r(mu) - kC2 v: the tangent to sqrt at v_r and the tangent at mu_r"
           " bound the same half-plane")
    a, b = sp.symbols("a b", positive=True)
    ta = sp.sqrt(a) + (v - a) / (2 * sp.sqrt(a))
    tb = sp.sqrt(b) + (v - b) / (2 * sp.sqrt(b))
    vx = sp.solve(sp.Eq(ta, tb), v)
    val = sp.simplify(ta.subs(v, vx[0]))
    report("I6", len(vx) == 1 and sp.simplify(vx[0] - sp.sqrt(a * b)) == 0
           and sp.simplify(val - (sp.sqrt(a) + sp.sqrt(b)) / 2) == 0,
           "consecutive tangents of sqrt at a and b meet at v = sqrt(a b) with value (sqrt a + sqrt b)/2")
    p, q = sp.symbols("p q", positive=True)            # p = a^(1/4), q = b^(1/4)
    above = sp.expand((p ** 2 + q ** 2) / 2 - p * q - (p - q) ** 2 / 2) == 0
    concave = sp.simplify(sp.diff(sp.sqrt(v), v, 2) + 1 / (4 * v ** sp.Rational(3, 2))) == 0
    report("I7", above and concave,
           "(sqrt a + sqrt b)/2 - (ab)^(1/4) = (a^(1/4) - b^(1/4))^2/2 >= 0: the vertex lies above the curve"
           " (strictly for a != b); sqrt'' = -1/(4 v^(3/2)) < 0, so chords between points on the curve lie below it")


def regions():
    print()
    print("=== Single tangent, union and intersection (tau = 1, eps = 0.01, kC2 = 99) ===")
    t, e = 1.0, 0.01
    kc2 = (1 - e) / e
    pts = [0.0, 0.25, 0.5, 0.75]
    g = np.linspace(0, 1, 2000001)
    ex = (t - g) ** 2
    single = t * t - 2 * t * g
    Ts = np.array([(t - r) * (t + r - 2 * g) for r in pts])
    union, inter = Ts.max(axis=0), Ts.min(axis=0)
    # single tangent claims
    lo = g <= 0.5
    report("R1", np.allclose(ex[lo] - single[lo], g[lo] ** 2) and abs(single[g == 0.5][0]) < 1e-12
           and (single[g > 0.5] < 0).all() and (ex[g < 1] > 0).all(),
           "single tangent: loss mu^2 on [0, tau/2], zero at tau/2, negative (so v >= 0 is infeasible) for mu > tau/2;"
           " exact admits v > 0 up to mu = tau")
    # inclusions intersection <= single <= union <= exact (as caps on kC2 v)
    report("R2", (inter <= single + 1e-15).all() and (single <= union + 1e-15).all() and (union <= ex + 1e-15).all(),
           "intersection <= each single tangent <= union <= exact, pointwise on [0, 1]")
    for r, Tr in zip(pts, Ts):
        report("R3", (inter <= Tr + 1e-15).all() and (inter < Tr - 1e-9).any(),
               "intersection strictly inside the region of the tangent at mu_r = %.2f" % r)
    # areas (in units of kC2 v, clipped at v >= 0)
    area = lambda cap: np.trapezoid(np.maximum(cap, 0), g)
    A_ex, A_si, A_un, A_in = area(ex), area(single), area(union), area(inter)
    print("  areas of the regions (mu in [0,1], units of kC2 v): exact %.6f, single %.6f, union %.6f, intersection %.6f"
          % (A_ex, A_si, A_un, A_in))
    rec = 1 - (A_ex - A_un) / (A_ex - A_si)
    # exact value with sympy: lost by single = 1/12, lost by union = 1/192
    x = sp.symbols("x")
    lost_single = sp.integrate(x ** 2, (x, 0, sp.Rational(1, 2))) + sp.integrate((1 - x) ** 2, (x, sp.Rational(1, 2), 1))
    D = sp.Rational(1, 4)
    lost_union = 3 * 2 * sp.integrate(x ** 2, (x, 0, D / 2)) \
        + sp.integrate((x - sp.Rational(3, 4)) ** 2, (x, sp.Rational(3, 4), sp.Rational(7, 8))) \
        + sp.integrate((1 - x) ** 2, (x, sp.Rational(7, 8), 1))
    exact_rec = 1 - lost_union / lost_single
    report("R4", abs(rec - float(exact_rec)) < 1e-9 and round(100 * float(exact_rec)) == 94,
           "union recovers %.6f of the area lost by the single tangent; exact %s = %.4f (text 94%%)"
           % (rec, exact_rec, float(exact_rec)))

    # table of the text (v x 1e3)
    print("  table (v x 1e3): mu, exact, single, union")
    text = {0.0: (10.10, 10.10, 10.10), 0.2: (6.46, 6.06, 6.44), 0.4: (3.64, 2.02, 3.54),
            0.5: (2.53, 0.0, 2.53), 0.6: (1.62, None, 1.52)}
    ok = True
    for m, (te, ts, tu) in text.items():
        e_ = (t - m) ** 2 / kc2 * 1e3
        s_ = (t * t - 2 * t * m) / kc2 * 1e3
        u_ = max((t - r) * (t + r - 2 * m) for r in pts) / kc2 * 1e3
        s_str = "infeasible" if s_ < -1e-12 else "%.2f" % s_
        print("   %.1f  %.2f  %s  %.2f" % (m, e_, s_str, u_))
        ok &= round(e_, 2) == te and round(u_, 2) == tu
        ok &= (ts is None and s_ < 0) or (ts is not None and round(abs(s_), 2) == ts)
    report("R5", ok, "the table of the text is reproduced")


def shortcuts():
    print()
    print("=== Encodings and the shortcuts that fail ===")
    t, e = 1.0, 0.01
    kc2 = (1 - e) / e
    kc = math.sqrt(kc2)
    pts = [0.0, 0.25, 0.5, 0.75]
    vr = [(t - r) ** 2 / kc2 for r in pts]
    tang = lambda vv, p: math.sqrt(p) + (vv - p) / (2 * math.sqrt(p))
    bad_union = bad_inter = 0
    for _ in range(200000):
        m, vv = RNG.uniform(0, 1), RNG.uniform(0, 0.012)
        union = kc2 * vv <= max((t - r) * (t + r - 2 * m) for r in pts) + 1e-12
        inter = kc2 * vv <= min((t - r) * (t + r - 2 * m) for r in pts) + 1e-12
        sos2 = m + kc * min(tang(vv, p) for p in vr) <= t + 1e-12   # y = min of tangents (encoding ii)
        ymax = m + kc * max(tang(vv, p) for p in vr) <= t + 1e-12   # y >= every tangent
        bad_union += sos2 != union
        bad_inter += ymax != inter
    report("S1", bad_union == 0, "encoding (ii), y = min of the tangents to sqrt, equals the union (selector encoding (i)):"
           " %d mismatches in 200000 points" % bad_union)
    report("S2", bad_inter == 0, "y >= every tangent gives the intersection: %d mismatches in 200000 points" % bad_inter)
    # y <= every tangent: y is free below, e.g. y = 0, so any mu <= tau is accepted
    m, vv = 0.9, 0.05
    cant = vv / (vv + (t - m) ** 2)
    report("S3", m + kc * 0.0 <= t and cant > e,
           "y <= every tangent admits y = 0: the point (mu, v) = (0.9, 0.05) is accepted, but its Cantelli bound is %.3f > eps" % cant)


def recovery():
    print()
    print("=== Remaining gap of the union (Appendix drv:pwl_recovery) ===")
    t = 1.0
    g = np.linspace(0, 1, 2000001)
    ex = (t - g) ** 2
    prev = None
    for P in (4, 8, 16):
        Dl = t / P
        pts = [i * Dl for i in range(P)]                       # 0, Delta, ..., tau - Delta
        un = np.array([(t - r) * (t + r - 2 * g) for r in pts]).max(axis=0)
        gap_clip = (ex - np.maximum(un, 0)).max()
        gap_formula = np.min([(g - r) ** 2 for r in pts], axis=0).max()
        mids = [(i + 0.5) * Dl for i in range(P)]              # tangent points of the pwl form
        un_m = np.array([(t - r) * (t + r - 2 * g) for r in mids]).max(axis=0)
        gap_mid = (ex - un_m).max()
        print("  P=%2d, Delta=%.4f: worst gap (clipped at v >= 0) %.6f; worst of min_r (mu - mu_r)^2 on [0, tau] %.6f;"
              " midpoints: %.6f; (Delta/2)^2 = %.6f" % (P, Dl, gap_clip, gap_formula, gap_mid, (Dl / 2) ** 2))
        ok = abs(gap_clip - (Dl / 2) ** 2) < 1e-9 and abs(gap_mid - (Dl / 2) ** 2) < 1e-9
        report("G%d" % P, ok, "worst case (Delta/2)^2 holds for the clipped gap and for midpoint tangents")
        if prev is not None:
            report("G%d/4" % P, abs(prev / gap_clip - 4) < 1e-6, "doubling the tangents divides the worst gap by %.4f" % (prev / gap_clip))
        prev = gap_clip
    print("  note: with tangent points 0, Delta, ..., tau - Delta, the formula min_r (mu - mu_r)^2 reaches Delta^2 at"
          " mu = tau; the admissible variance is 0 there, so the loss measured in v >= 0 is still at most (Delta/2)^2")


def bernstein_and_loop():
    print()
    print("=== Bernstein and the loop argument ===")
    c = sp.symbols("c", nonnegative=True)
    k = sp.symbols("kappa", positive=True)
    fB = (tau - c - mu) ** 2 - c ** 2                    # 2 kappa v <= fB on the side mu <= tau - c
    TB = fB.subs(mu, mur) + sp.diff(fB, mu).subs(mu, mur) * (mu - mur)
    report("B1", sp.expand(fB - TB - (mu - mur) ** 2) == 0, "Bernstein: gap of a tangent = (mu - mu_r)^2, the same law")
    s = sp.sqrt(c ** 2 + 2 * k * v)
    report("B2", sp.simplify(sp.diff(s, v, 2) + k ** 2 / (c ** 2 + 2 * k * v) ** sp.Rational(3, 2)) == 0,
           "second derivative of sqrt(c^2 + 2 kappa v) in v = -kappa^2/(c^2 + 2 kappa v)^(3/2) < 0: concave,"
           " so its tangents lie above it")
    # loop: the union region and the conditional Bernstein rows are down-sets in (mu, v)
    t, e = 1.0, 0.01
    kc2 = (1 - e) / e
    pts = [0.0, 0.25, 0.5, 0.75]
    inside = lambda m, vv: kc2 * vv <= max((t - r) * (t + r - 2 * m) for r in pts) + 1e-12
    bad = 0
    for _ in range(100000):
        m, vv = RNG.uniform(0, 1), RNG.uniform(0, 0.012)
        if inside(m, vv):
            m2, v2 = m * RNG.uniform(0, 1), vv * RNG.uniform(0, 1)
            bad += not inside(m2, v2)
    report("L1", bad == 0, "the union region is a down-set in (mu, v): %d violations; so componentwise smaller moments"
           " (the moment loop) keep every refined row satisfied" % bad)


if __name__ == "__main__":
    symbolic()
    regions()
    shortcuts()
    recovery()
    bernstein_and_loop()
