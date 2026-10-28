"""Loop constraint under the Route-I bounds: the worked instance and the margin claim.

Checks scoping.tex, subsubsection "The loop constraint under the Route-I bounds"
(label sec:loop_routeI):
  - the instance "A concrete instance has tau = 10, b_l = 3, rho = 0.5 ..." and its three
    bullets (Cantelli, Bernstein, Loop constraint);
  - the sentence "A fixed margin delta > 0, U(1) <= U(0) - delta, does not fix this.";
and Appendix drv:loopI_no_chain (the Cantelli instance with sum_j n_ij v_lj = 4).

Dynamics. The operating phase is one repair day followed by mission days. The text does not
name the repair law; the numbers of the text are reproduced with ARD_inf:
    E -> (1 - rho) E,  Var -> (1 - rho)^2 Var   (repair), then E += 3, Var += sum v (missions).
U^(n) is the bound at the end of the n-th repetition; U^(0) is the bound at day H_1.

Margin claim (Cantelli, ARD_inf, no replacement in the phase). One repetition maps
d = tau - E and V affinely: d -> d* + A (d - d*), V -> V* + A^2 (V - V*), A in (0, 1).
With r = V / d^2 (U = r / (1 + r)), x = A^n, Delta = d_0 - d*, W = V_0 - V*:
    r_n = (V* + W x^2) / (d* + Delta x)^2.
A sign analysis of the derivative in x shows: if some U^(n) > U^(0) (n >= 1) and U^(1) < U^(0),
then Delta > 0, W > 0 and r* = V*/d*^2 > r_0, and r_1 > r_0 / 2. Hence
    U^(0) - U^(1) < U0 (1 - U0) / (2 - U0)  =: delta_max(U0),
and instances approach this value. The script checks the bound by a seeded random search and
shows instances that approach it.

Deterministic (fixed seed). Run: python loop_routeI_instance.py > loop_routeI_instance_output.txt
"""
import math
import random
from fractions import Fraction as Fr

TAU, B, RHO = 10, 3, Fr(1, 2)
START = (Fr(4), Fr(9))
DM = Fr(3)


def cantelli(m, v, tau=TAU):
    return v / (v + (tau - m) ** 2)


def bernstein(m, v, tau=TAU, b=B):
    a = tau - m
    return math.exp(-(a * a / 2) / (v + b * a / 3))


def phase(state, dv, rho=RHO, dm=DM):
    m, v = state
    m, v = (1 - rho) * m, (1 - rho) ** 2 * v   # repair day, ARD_inf
    return m + dm, v + dv                     # mission days


def trajectory(U, dv, n=6):
    s = START
    out = [(s, U(*s))]
    for _ in range(n):
        s = phase(s, dv)
        out.append((s, U(*s)))
    return out


def fixed_point(dv, rho=RHO, dm=DM):
    return dm / rho, dv / (1 - (1 - rho) ** 2)


def show(label, U, dv):
    print("%s, sum v = %s" % (label, dv))
    for n, (s, u) in enumerate(trajectory(U, dv, 4)):
        print("  n=%d: (E, Var) = (%s, %s)  U = %.6f" % (n, float(s[0]), float(s[1]), float(u)))
    ms, vs = fixed_point(dv)
    print("  limit: (E, Var) = (%s, %s)  U = %.6f" % (float(ms), float(vs), float(U(ms, vs))))


def main():
    print("=== Instance of sec:loop_routeI (ARD_inf repair, then the mission days) ===")
    show("Cantelli", cantelli, Fr("3.99"))
    print("  text: U0 = 0.2000, U1 = 0.1997, U2 = 0.2151, limit 0.2495")
    show("Bernstein (b = 3)", bernstein, Fr("3.15"))
    print("  text: 'also fails at the second repetition'")
    show("Cantelli, Appendix drv:loopI_no_chain", cantelli, Fr(4))
    print("  text: U1 = U0 = 0.2000 and U2 = 0.2155")

    print()
    print("=== Loop constraint (eq:loop_moments) on the instance ===")
    for dv in (Fr("3.99"), Fr("3.15")):
        mT, vT = phase(START, dv)
        print("  sum v = %s: E[D_T] = %s vs E[D_H1] = %s -> %s;  Var[D_T] = %s vs 9"
              % (dv, float(mT), float(START[0]), "rejected" if mT > START[0] else "accepted", float(vT)))

    print()
    print("=== Realisability with terms whose range has width <= 3 (b_l = 3) ===")
    # D_H1: 6 i.i.d. two-point terms on {0, c} with probability p.
    c = Fr(2, 3) + Fr(9, 4)
    p = Fr(2, 3) / c
    print("  D_H1 = sum of 6 two-point terms on {0, %.4f}, p = %.4f: mean %s, var %s"
          % (float(c), float(p), float(6 * c * p), float(6 * c * c * p * (1 - p))))
    for dv in (Fr("3.99"), Fr("3.15")):
        d = math.sqrt(float(dv) / 2)
        print("  missions (sum v = %s): 2 days, each two-point on {1.5 - d, 1.5 + d}, d = %.4f, "
              "values {%.4f, %.4f}, width %.4f: mean %.4f, var %.4f"
              % (dv, d, 1.5 - d, 1.5 + d, 2 * d, 2 * 1.5, 2 * d * d))

    print()
    print("=== Margin claim: U(1) <= U(0) - delta ===")
    c1 = trajectory(cantelli, Fr("3.99"), 2)
    print("  instance: U0 - U1 = %.6f, so it satisfies every margin delta <= %.6f and fails at n = 2"
          % (float(c1[0][1] - c1[1][1]), float(c1[0][1] - c1[1][1])))
    ms, vs = fixed_point(Fr("3.99"))
    u_ss = cantelli(ms, vs)
    print("  steady state (E, Var) = (%s, %s): U1 = U0 = %.6f; the moment loop accepts it, "
          "every delta > 0 rejects it" % (float(ms), float(vs), float(u_ss)))

    def delta_max(u0):
        return u0 * (1 - u0) / (2 - u0)

    print("  derived bound delta_max(U0) = U0 (1 - U0) / (2 - U0); delta_max(0.2) = %.6f" % delta_max(0.2))

    # Near-extremal instances with U0 = 0.2 (normalised d* = 1, tau = 2 + Delta, E_0 = 1).
    print("  near-extremal instances with U0 = 0.2:")
    r0 = 0.25
    for Delta, eta in ((5.0, 0.05), (50.0, 0.01), (1000.0, 1e-3)):
        w = (Delta ** 2 + 2 * Delta) * (1 - eta)
        A = Delta / w
        g1 = (1 + w) / (1 + Delta) ** 2
        vstar = r0 / g1
        tau = 2.0 + Delta
        mstar = tau - 1.0
        m, v = 1.0, vstar * (1 + w)
        Bm, Bv = mstar * (1 - A), vstar * (1 - A * A)
        U = [cantelli(m, v, tau)]
        for n in range(1, 400):
            m, v = A * m + Bm, A * A * v + Bv
            U.append(cantelli(m, v, tau))
        first = next((n for n in range(1, len(U)) if U[n] > U[0]), None)
        print("    Delta=%7.1f rho=%.4f: U0=%.6f U1=%.6f delta=U0-U1=%.6f (%.1f%% of delta_max); "
              "first n with U_n > U0: %s; limit %.6f"
              % (Delta, 1 - A, U[0], U[1], U[0] - U[1], 100 * (U[0] - U[1]) / delta_max(U[0]), first, U[-1]))

    # Seeded random search: no failing instance may exceed delta_max.
    random.seed(20261028)
    n_fail, worst = 0, 0.0
    for _ in range(200000):
        tau = 10.0
        A = random.uniform(0.01, 0.99)
        m0, v0 = random.uniform(0, 9.9), random.uniform(0, 50)
        Bm, Bv = random.uniform(0, 9.9) * (1 - A), random.uniform(0, 50)
        if Bm / (1 - A) >= tau:
            continue
        m, v = m0, v0
        U0 = cantelli(m, v, tau)
        U = []
        ok = True
        for _ in range(200):
            m, v = A * m + Bm, A * A * v + Bv
            if m >= tau:
                ok = False
                break
            U.append(cantelli(m, v, tau))
        if not ok or U[0] >= U0:
            continue
        if max(U[1:]) > U0:
            n_fail += 1
            worst = max(worst, (U0 - U[0]) / delta_max(U0))
    print("  random search (200000 samples, seed 20261028): %d failing instances with U1 < U0; "
          "largest (U0 - U1) / delta_max(U0) = %.4f (must be < 1)" % (n_fail, worst))


if __name__ == "__main__":
    main()
