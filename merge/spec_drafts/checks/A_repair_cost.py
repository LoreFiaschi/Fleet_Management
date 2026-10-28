"""Writer A: spot check of Proposition prop:a_repair_cost (standard library only).

Random schedules for one cell (mean and its anchor), random one-sided slack, both
laws. On every repair day, the smallest feasible z, rho*(mu_{k-1} - g_{k-1}), must be
at least the reference removal rho*(zeta_mu_{k-1} - zeta_g_{k-1}).
Boundary case: zero slack gives equality.
Also: the worked example of the anchor rule (0.6, 0.2, rho 0.5 -> 0.4, 0.4) and the
starting-anchor example (mu_0 = 0.5, rho = 0.5: 0.25 with anchor 0, 0.5 with mu_0).
"""
import random

random.seed(2026)


def run(law, rho, mu0, g0, sched, slack):
    """Return the list of (z_min, reference removal) on repair days."""
    mu, g = mu0, (g0 if law == "ard1" else 0.0)
    zm, zg = mu0, (g0 if law == "ard1" else 0.0)
    out = []
    for kind, inc, sl in zip(sched, [random.uniform(0, 0.1) for _ in sched], slack):
        if kind == "repair":
            out.append((rho * (mu - g), rho * (zm - zg)))
            if law == "ard1":
                mu = (1 - rho) * mu + rho * g + sl
                g = mu
                zm = (1 - rho) * zm + rho * zg
                zg = zm
            else:
                mu = (1 - rho) * mu + sl
                zm = (1 - rho) * zm
        elif kind == "replace":
            mu = zm = 0.02
            if law == "ard1":
                g = zg = 0.02
        else:
            mu = mu + inc + sl
            zm = zm + inc
    return out


def check(n=5000):
    worst = 0.0       # max over (reference - z_min), must be <= 0
    worst_eq = 0.0    # zero slack: max |z_min - reference|
    count = 0
    for _ in range(n):
        law = random.choice(["ardinf", "ard1"])
        rho = random.uniform(0.05, 1.0)
        mu0 = random.uniform(0, 0.4)
        g0 = random.uniform(0, mu0)
        T = random.randint(2, 20)
        sched = [random.choice(["none", "none", "repair", "replace"]) for _ in range(T)]
        slack = [random.uniform(0, 0.05) if random.random() < 0.5 else 0.0 for _ in range(T)]
        state = random.getstate()
        for zmin, ref in run(law, rho, mu0, g0, sched, slack):
            worst = max(worst, ref - zmin)
            count += 1
        random.setstate(state)
        for zmin, ref in run(law, rho, mu0, g0, sched, [0.0] * T):
            worst_eq = max(worst_eq, abs(zmin - ref))
    return count, worst, worst_eq


def examples():
    rho = 0.5
    s, g = 0.6, 0.2
    s_new = s - rho * (s - g)
    ex1 = (s_new, s_new)
    mu0 = 0.5
    with_zero = (1 - rho) * mu0 + rho * 0.0
    with_mu0 = (1 - rho) * mu0 + rho * mu0
    return ex1, with_zero, with_mu0


if __name__ == "__main__":
    c, w, we = check()
    print("repair days checked:", c, "; max (reference - z_min) =", w, "(must be <= 0)")
    print("zero slack: max |z_min - reference| =", we, "(must be ~0)")
    print("anchor example (s, g) after repair; start-anchor example (anchor 0, anchor mu_0):", examples())
