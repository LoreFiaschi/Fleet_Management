"""Schedule-dependent Bernstein support: checks of scoping.tex.

Checks the paragraph "Schedule-dependent Bernstein support." of the subsubsection
"Two refinements of the Route-I constraints ..." (label sec:routeI_refinements) and
Appendix drv:sched_support. One numbered check per claim; each prints PASS or FAIL.

Notation: kappa = ln(1/eps), c = kappa b / 3. The Bernstein row at the constant c is
    cert(mu, v; c) = mu + c + sqrt(c^2 + 2 kappa v) <= tau,
equivalently (tau - c - mu)^2 >= c^2 + 2 kappa v with the side row mu <= tau - c.

Deterministic (fixed seed). Run: python refine_schedule_support.py > refine_schedule_support_output.txt
"""
import math

import numpy as np
import sympy as sp

RNG = np.random.default_rng(20261028)
TOL = 1e-12


def report(tag, ok, text):
    print("[%s] %s: %s" % ("PASS" if ok else "FAIL", tag, text))


def cert(mu, v, c, kappa):
    return mu + c + math.sqrt(c * c + 2 * kappa * v)


def row_ok(mu, v, c, kappa, tau):
    """Bernstein row and its side row at the constant c."""
    return mu <= tau - c + TOL and cert(mu, v, c, kappa) <= tau + TOL


# ---------------------------------------------------------------- worked example
def worked_example():
    print("=== Worked example (eps = 0.01, tau = 1) ===")
    eps, tau = 0.01, 1.0
    kappa = math.log(1 / eps)
    mu1, v1 = 0.01, 4e-6

    def nmax(c):
        n = 0
        while cert((n + 1) * mu1, (n + 1) * v1, c, kappa) <= tau:
            n += 1
        return n

    c_big = kappa * max(0.02, 0.02, 0.02, 0.30) / 3
    c_small = kappa * max(0.02, 0.02, 0.02) / 3
    n_big, n_small = nmax(c_big), nmax(c_small)
    report("E1", round(kappa, 3) == 4.605, "kappa = ln 100 = %.6f (text 4.605)" % kappa)
    report("E2", round(c_big, 4) == 0.4605, "kappa b_l/3 with b_l = 0.30: %.6f (text 0.4605)" % c_big)
    report("E3", round(tau - 2 * c_big, 3) == 0.079, "mean budget tau - 2c = %.6f (text 0.079)" % (tau - 2 * c_big))
    report("E4", n_big == 7, "largest n_i1 = %d (text 7); cert(7) = %.6f, cert(8) = %.6f"
           % (n_big, cert(7 * mu1, 7 * v1, c_big, kappa), cert(8 * mu1, 8 * v1, c_big, kappa)))
    report("E5", round(c_small, 4) == 0.0307, "kappa b/3 with the supports present (0.02): %.6f (text 0.0307)" % c_small)
    report("E6", round(tau - 2 * c_small, 3) == 0.939, "mean budget = %.6f (text 0.939)" % (tau - 2 * c_small))
    report("E7", n_small == 90, "largest n_i1 = %d (text 90); cert(90) = %.6f, cert(91) = %.6f"
           % (n_small, cert(90 * mu1, 90 * v1, c_small, kappa), cert(91 * mu1, 91 * v1, c_small, kappa)))
    report("E8", round(n_small / n_big, 1) == 12.9, "factor 90/7 = %.4f (text 12.9); ratio of the mean budgets = %.4f"
           % (n_small / n_big, (tau - 2 * c_small) / (tau - 2 * c_big)))
    val = 2 * kappa * 0.35 / 3
    report("E9", round(val, 3) == 1.075 and val > tau,
           "b_l2 = 0.35: 2 kappa b_l/3 = %.6f (text 1.075) > tau, so even mu = v = 0 fails: cert(0, 0) = %.6f"
           % (val, cert(0, 0, kappa * 0.35 / 3, kappa)))


# ---------------------------------------------------------------- monotonicity and equivalence
def monotonicity():
    print()
    print("=== Monotonicity in the support constant ===")
    mu, v, c, kappa = sp.symbols("mu v c kappa", nonnegative=True)
    expr = mu + c + sp.sqrt(c ** 2 + 2 * kappa * v)
    d = sp.diff(expr, c)
    report("M1", sp.simplify(d - (1 + c / sp.sqrt(c ** 2 + 2 * kappa * v))) == 0,
           "d/dc [mu + c + sqrt(c^2 + 2 kappa v)] = %s" % d)
    # 0 <= c / sqrt(c^2 + 2 kappa v) <= 1 for c >= 0, v >= 0 (not both 0)
    cs = RNG.uniform(0, 5, 200000)
    vs = RNG.uniform(0, 5, 200000) * (RNG.uniform(size=200000) > 0.1)
    ks = RNG.uniform(0.01, 30, 200000)
    good = (cs > 0) | (vs > 0)
    dd = 1 + cs[good] / np.sqrt(cs[good] ** 2 + 2 * ks[good] * vs[good])
    report("M2", dd.min() >= 1 - 1e-15 and dd.max() <= 2 + 1e-15,
           "derivative over 200000 random (c, v, kappa) lies in [%.6f, %.6f] (text [1, 2])" % (dd.min(), dd.max()))
    # a row at a larger constant implies the row at a smaller constant (random points)
    bad = 0
    for _ in range(100000):
        tau = 1.0
        kap = RNG.uniform(0.1, 20)
        c1, c2 = sorted(RNG.uniform(0, 0.6, 2))
        m, vv = RNG.uniform(0, 1), RNG.uniform(0, 0.05)
        if row_ok(m, vv, c2, kap, tau) and not row_ok(m, vv, c1, kap, tau):
            bad += 1
    report("M3", bad == 0, "row at the larger constant implies the row at the smaller one: %d violations in 100000" % bad)
    # largest admissible variance at mu = 0: tau (tau - 2c) / (2 kappa), decreasing in c
    tau_s = sp.symbols("tau", positive=True)
    vmax = sp.solve(sp.Eq((tau_s - c) ** 2, c ** 2 + 2 * kappa * v), v)[0]
    report("M4", sp.simplify(vmax - tau_s * (tau_s - 2 * c) / (2 * kappa)) == 0 and sp.diff(vmax, c) == -tau_s / kappa,
           "largest admissible variance (at mu = 0) = %s, derivative in c = %s < 0: a big-M taken from b_l cuts the region of c_0"
           % (sp.factor(vmax), sp.diff(vmax, c)))


def equivalence():
    print()
    print("=== Equivalence: conditional rows at c_j plus the row at c_0 == one row at kappa b_l(zeta)/3 ===")
    bad, active_cases = 0, 0
    for _ in range(50000):
        tau = 1.0
        kap = RNG.uniform(0.1, 12)
        M = int(RNG.integers(1, 6))
        bj = RNG.uniform(0, 0.5, M)
        bstart, bnew = RNG.uniform(0, 0.2, 2)
        zeta = RNG.integers(0, 2, M)
        cj = kap * np.maximum(bj, max(bstart, bnew)) / 3
        c0 = kap * max(bstart, bnew) / 3
        b_zeta = max([bstart, bnew] + [bj[j] for j in range(M) if zeta[j]])
        c_single = kap * b_zeta / 3
        m, vv = RNG.uniform(0, 1), RNG.uniform(0, 0.03)
        lhs = row_ok(m, vv, c0, kap, tau) and all(row_ok(m, vv, cj[j], kap, tau) for j in range(M) if zeta[j])
        rhs = row_ok(m, vv, c_single, kap, tau)
        bad += lhs != rhs
        active_cases += rhs
    report("Q1", bad == 0, "%d mismatches in 50000 random cases (%d feasible)" % (bad, active_cases))
    print("  validity of b_l(zeta) (Appendix drv:sched_support): by derivation. Between resets the state is a")
    print("  weighted sum (weights in [0, 1]) of one reset draw and of increments of served missions; zeta_ij >= x_ijt")
    print("  makes every served mission enter b_l(zeta). A zeta_ij = 1 on an unserved mission only adds a row with a")
    print("  larger constant, which is more conservative.")


# ---------------------------------------------------------------- precondition, side effects
def precondition():
    print()
    print("=== Empty schedule and the local precondition ===")
    mu, v = 0.0, 0.0
    ok = all(abs(cert(mu, v, c, k) - 2 * c) < 1e-15 for c in (0.1, 0.3, 0.7) for k in (1.0, 4.6))
    report("P1", ok, "cert(0, 0; c) = 2c, so the empty schedule satisfies the row only if 2 kappa b_l/3 <= tau")
    # a mission with 2 c_j > tau: no (mu, v) >= 0 satisfies its row
    tau, kap = 1.0, 4.605
    grid_mu = np.linspace(0, 1, 401)
    grid_v = np.linspace(0, 0.05, 401)
    counts = {}
    for cj in (0.5, 0.5 + 1e-9, 0.6):
        feas = [(float(m), float(vv)) for m in grid_mu for vv in grid_v if row_ok(m, vv, cj, kap, tau)]
        counts[cj] = feas
        print("  c_j = %.9f (2 c_j - tau = %+.1e): feasible grid points %d%s"
              % (cj, 2 * cj - tau, len(feas), (", only " + str(feas)) if 0 < len(feas) <= 3 else ""))
    ok = counts[0.5] == [(0.0, 0.0)] and not counts[0.5 + 1e-9] and not counts[0.6]
    report("P2", ok, "2 c_j > tau: the row admits no point, so zeta_ij = 1 is infeasible and zeta_ij = 0 is forced;"
           " at 2 c_j = tau the row admits only mu = v = 0")


# ---------------------------------------------------------------- big-M constant (not in the footnote)
def big_m():
    print()
    print("=== Big-M constant 2 tau (tau - c_0) of the conditional tangent rows (extra check) ===")
    # Row of mission j at tangent point mu_r: 2 kappa v <= T_j(mu) = h_j(mu_r) + h_j'(mu_r)(mu - mu_r),
    # h_c(mu) = (tau - mu)^2 - 2 c (tau - mu). With zeta_ij = 0 the point satisfies the row at c_0:
    # mu in [0, tau - 2 c_0], 0 <= 2 kappa v <= h_{c0}(mu). The violation 2 kappa v - T_j(mu) is largest
    # at an end point; the closed form below is checked against a grid.
    worst = 0.0
    for _ in range(20000):
        tau = 1.0
        c0 = RNG.uniform(0, 0.4999)
        cj = RNG.uniform(c0, 0.4999)          # 2 c_j < tau
        mur = RNG.uniform(0, 1.0) * tau       # tangent point in [0, tau)
        mus = np.linspace(0, tau - 2 * c0, 801)
        h0 = (tau - mus) ** 2 - 2 * c0 * (tau - mus)
        Tj = ((tau - mur) ** 2 - 2 * cj * (tau - mur)) + (-2 * (tau - mur) + 2 * cj) * (mus - mur)
        viol = max((h0 - Tj).max(), (-Tj).max())
        worst = max(worst, viol / (2 * tau * (tau - c0)))
    report("B1", worst <= 1 + 1e-12,
           "largest violation / (2 tau (tau - c_0)) over 20000 random cases with 2 c_j < tau, mu_r in [0, tau): %.6f" % worst)


if __name__ == "__main__":
    worked_example()
    monotonicity()
    equivalence()
    precondition()
    big_m()
