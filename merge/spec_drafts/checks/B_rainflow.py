"""Writer B, spec v1.0, rainflow section: spot checks of the new propositions.

Standard library only. Run: python -I B_rainflow.py > B_rainflow_output.txt
Laws are finite discrete laws on [0, inf), so every statistic is computed exactly.

Checks:
 (1) Chernoff, ARDinf: K_{(1-rho)D}(s) <= (1-rho) K_D(s).
 (2) Chernoff, ARD1: K_{G + (1-rho) Delta}(s) <= (1-rho) K_{G+Delta}(s) + rho K_G(s), G, Delta independent.
     Boundary cases: rho = 0 and rho = 1 (equality); Delta a constant (equality).
 (3) Jensen: K_D(s) >= s E D, equality for a constant.
 (4) Variance and R under ARD1: Var(G + (1-rho) Delta) = (1-rho)^2 Var(D) + (1-(1-rho)^2) Var(G);
     the same identity for the Hoeffding budget R = sum of squared weighted widths, on a random schedule
     with term bookkeeping against the recursion.
 (5) Hoeffding row: exp(-2 d^2 / R) <= eps  iff  d^2 >= (kappa/2) R (boundary d^2 = (kappa/2) R).
 (6) Bernstein row: exp(-d^2 / (2 (v + b d/3))) <= eps  iff  d^2 - 2 c d - 2 kappa v >= 0 (d > 0);
     with d >= c and v >= 0 the row forces d >= 2c; natural range of v = tau (tau - 2c)/(2 kappa).
 (7) Bernstein tangent with tangent point mu_r >= tau - c: no point (mu, v), v >= 0, mu <= tau - c, satisfies it.
 (8) Natural ranges: V_max / tau^2 = 1/z^2 at eps = 0.01, 0.1, 0.3, 0.49; Cantelli eps/(1-eps); Hoeffding 2/kappa.
 (9) Chernoff counterexample of decisions.md (Defect B): K_0 = s mu_0 is needed.
"""
import math
import random

random.seed(20261026)


def law(n, scale):
    """Random discrete law on [0, scale]: list of (value, prob)."""
    xs = [random.random() * scale for _ in range(n)]
    ps = [random.random() + 0.05 for _ in range(n)]
    z = sum(ps)
    return [(x, p / z) for x, p in zip(xs, ps)]


def cgf(L, s):
    return math.log(sum(p * math.exp(s * x) for x, p in L))


def mean(L):
    return sum(p * x for x, p in L)


def var(L):
    m = mean(L)
    return sum(p * (x - m) ** 2 for x, p in L)


def conv(A, B, a=1.0, b=1.0):
    """Law of a X + b Y, X ~ A, Y ~ B independent."""
    return [(a * x + b * y, p * q) for x, p in A for y, q in B]


def Phi_inv(p):
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * math.erfc(-mid / math.sqrt(2.0)) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    print("(1) ARDinf: max of K_{(1-rho)D}(s) - (1-rho) K_D(s) over 2000 random cases (must be <= 0)")
    worst = -1e9
    for _ in range(2000):
        D = law(random.randint(1, 6), random.uniform(0.1, 3.0))
        s, rho = random.uniform(0.01, 20.0), random.uniform(0.0, 1.0)
        worst = max(worst, cgf([((1 - rho) * x, p) for x, p in D], s) - (1 - rho) * cgf(D, s))
    print("  %.3e" % worst)

    print("(2) ARD1: max of K_{G+(1-rho)Delta}(s) - [(1-rho) K_{G+Delta}(s) + rho K_G(s)] over 2000 cases")
    worst = -1e9
    for _ in range(2000):
        G = law(random.randint(1, 5), random.uniform(0.1, 2.0))
        Dl = law(random.randint(1, 5), random.uniform(0.1, 2.0))
        s, rho = random.uniform(0.01, 20.0), random.uniform(0.0, 1.0)
        lhs = cgf(conv(G, Dl, 1.0, 1.0 - rho), s)
        rhs = (1 - rho) * cgf(conv(G, Dl), s) + rho * cgf(G, s)
        worst = max(worst, lhs - rhs)
    print("  %.3e" % worst)
    G = law(4, 1.0)
    Dl = law(4, 1.0)
    s = 3.0
    for rho in (0.0, 1.0):
        lhs = cgf(conv(G, Dl, 1.0, 1.0 - rho), s)
        rhs = (1 - rho) * cgf(conv(G, Dl), s) + rho * cgf(G, s)
        print("  boundary rho=%g: lhs - rhs = %.3e (equality expected)" % (rho, lhs - rhs))
    const = [(0.7, 1.0)]
    rho = 0.4
    lhs = cgf(conv(G, const, 1.0, 1.0 - rho), s)
    rhs = (1 - rho) * cgf(conv(G, const), s) + rho * cgf(G, s)
    print("  boundary Delta = 0.7 (constant), rho = 0.4: lhs - rhs = %.3e (equality expected)" % (lhs - rhs))

    print("(3) Jensen: min of K_D(s) - s E D over 2000 cases (must be >= 0); constant: difference")
    worst = 1e9
    for _ in range(2000):
        D = law(random.randint(2, 6), random.uniform(0.1, 3.0))
        s = random.uniform(0.01, 20.0)
        worst = min(worst, cgf(D, s) - s * mean(D))
    print("  random laws: %.3e;  constant 0.6 at s = 5: %.3e" % (worst, cgf([(0.6, 1.0)], 5.0) - 5.0 * 0.6))

    print("(4) ARD1 maps of v and R")
    worst_v = 0.0
    for _ in range(500):
        G = law(random.randint(1, 4), random.uniform(0.1, 2.0))
        Dl = law(random.randint(1, 4), random.uniform(0.1, 2.0))
        rho = random.uniform(0.0, 1.0)
        a = (1 - rho) ** 2
        lhs = var(conv(G, Dl, 1.0, 1.0 - rho))
        rhs = a * var(conv(G, Dl)) + (1 - a) * var(G)
        worst_v = max(worst_v, abs(lhs - rhs))
    print("  variance identity, largest error over 500 cases: %.3e" % worst_v)
    # R: term bookkeeping against the recursion on a random schedule (both laws)
    worst_R = 0.0
    for law_name in ("ardinf", "ard1"):
        for _ in range(200):
            rho = random.uniform(0.05, 1.0)
            terms = [[1.0, random.uniform(0.0, 0.5) ** 2, True]]   # weight, squared width, in G?
            R = terms[0][1]
            gR = R                                                # starting anchor: the initial value
            for day in range(40):
                u = random.random()
                if u < 0.6:                                       # mission
                    w2 = random.uniform(0.0, 0.3) ** 2
                    terms.append([1.0, w2, False])
                    R = R + w2
                elif u < 0.9:                                     # repair
                    a = (1 - rho) ** 2
                    for tm in terms:
                        if law_name == "ardinf" or not tm[2]:
                            tm[0] *= (1 - rho)
                    if law_name == "ardinf":
                        R = a * R
                    else:
                        R = a * R + (1 - a) * gR
                    for tm in terms:
                        tm[2] = True
                    gR = R
                else:                                             # replacement
                    w2 = random.uniform(0.0, 0.2) ** 2
                    terms = [[1.0, w2, True]]
                    R = w2
                    gR = R
                Rt = sum(tm[0] ** 2 * tm[1] for tm in terms)
                worst_R = max(worst_R, abs(Rt - R))
    print("  R recursion against sum of squared weighted widths, largest error: %.3e" % worst_R)

    print("(5) Hoeffding: d^2 = (kappa/2) R gives exp(-2 d^2/R) = eps")
    for eps in (0.001, 0.01, 0.1):
        kap = math.log(1 / eps)
        R = 0.37
        d = math.sqrt(kap / 2 * R)
        print("  eps=%g: exp(-2 d^2/R)/eps = %.12f" % (eps, math.exp(-2 * d * d / R) / eps))

    print("(6) Bernstein: equivalence on a random grid, and the forced region d >= 2c")
    bad = 0
    for _ in range(20000):
        eps = random.uniform(0.001, 0.3)
        kap = math.log(1 / eps)
        b = random.uniform(0.0, 0.3)
        c = kap * b / 3
        d = random.uniform(1e-6, 3.0)
        v = random.uniform(0.0, 0.5)
        lhs = math.exp(-d * d / (2 * (v + b * d / 3))) <= eps if (v + b * d) > 0 else True
        rhs = d * d - 2 * c * d - 2 * kap * v >= 0
        if lhs != rhs and abs(d * d - 2 * c * d - 2 * kap * v) > 1e-12:
            bad += 1
    print("  mismatches: %d of 20000" % bad)
    viol = 0
    for _ in range(20000):
        c = random.uniform(0.0, 0.5)
        d = random.uniform(c, 3.0)
        v = random.uniform(0.0, 0.5)
        kap = random.uniform(0.1, 7.0)
        if d * d - 2 * c * d - 2 * kap * v >= 0 and d < 2 * c - 1e-12:
            viol += 1
    print("  points with d >= c, v >= 0, row true and d < 2c: %d" % viol)
    tau, eps, b = 1.0, 0.01, 0.1
    kap = math.log(1 / eps)
    c = kap * b / 3
    vmax = max((((tau - mu) ** 2 - 2 * c * (tau - mu)) / (2 * kap)) for mu in [k * (tau - 2 * c) / 1000 for k in range(1001)])
    print("  tau=1, eps=0.01, b=0.1: max of f_B over [0, tau-2c] = %.10f; tau(tau-2c)/(2 kappa) = %.10f" % (
        vmax, tau * (tau - 2 * c) / (2 * kap)))

    print("(7) Bernstein tangent at mu_r >= tau - c: largest tangent value on mu in [0, tau - c] (must be < 0)")
    worst = -1e9
    for _ in range(5000):
        tau = 1.0
        eps = random.uniform(0.001, 0.3)
        kap = math.log(1 / eps)
        b = random.uniform(0.01, 0.3)
        c = kap * b / 3
        if 2 * c >= tau:
            continue
        mu_r = random.uniform(tau - c, tau * 0.999999) if tau - c < tau * 0.999999 else tau - c
        f = lambda mu: ((tau - mu) ** 2 - 2 * c * (tau - mu)) / (2 * kap)
        fp = lambda mu: (-2 * (tau - mu) + 2 * c) / (2 * kap)
        for mu in [k * (tau - c) / 200 for k in range(201)]:
            worst = max(worst, f(mu_r) + fp(mu_r) * (mu - mu_r))
    print("  %.3e" % worst)

    print("(8) natural ranges divided by tau^2")
    for eps in (0.01, 0.1, 0.3, 0.49):
        z = Phi_inv(1 - eps)
        print("  eps=%g: z=%.5f  V_max/tau^2=1/z^2=%.4g  Cantelli eps/(1-eps)=%.4g  Hoeffding 2/kappa=%.4g" % (
            eps, z, 1 / z ** 2, eps / (1 - eps), 2 / math.log(1 / eps)))

    print("(9) Chernoff with a known initial damage: tau=1, eps=0.1, s=5, mu_0=0.6, X in {0, 0.5} w.p. 1/2")
    s, eps, tau, mu0 = 5.0, 0.1, 1.0, 0.6
    KX = cgf([(0.0, 0.5), (0.5, 0.5)], s)
    print("  K_X(5) = %.4f;  row with K_0 = 0: %.3f <= %.3f ? %s" % (KX, KX - s * tau, math.log(eps), KX - s * tau <= math.log(eps)))
    print("  row with K_0 = s mu_0: %.3f <= %.3f ? %s;  true P(D > tau) = 0.5" % (
        s * mu0 + KX - s * tau, math.log(eps), s * mu0 + KX - s * tau <= math.log(eps)))
