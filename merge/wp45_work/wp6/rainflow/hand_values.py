"""Expected values of the rainflow hand case (WP6 plan, section 3).

Usage: python hand_values.py
Exact day maps of the fixed schedule (train 0: fly, repair, fly, replace;
train 1: idle, fly, idle, fly with zero increments), then the margin of every
reliability row on every day, in the forms exact, tangent (0.5) and pwl (P = 4),
for the feasible and the infeasible epsilon of each bound.
"""
import math

TAU, RHO, MU0, MU_NEW, MU_INC = 1.0, 0.5, 0.2, 0.1, 0.3
# day kinds of train 0 (days 1..4); train 1 never changes its state
KINDS = ["fly", "repair", "fly", "replace"]

# second tracked quantity per bound: (start, anchor start, increment, replacement, factor)
SECOND = {
    "cantelli": (0.004, 0.004, 0.01, 0.002, (1 - RHO) ** 2),
    "hoeffding": (0.01, 0.01, 0.25 ** 2 * 0 + 0.04, 0.005, (1 - RHO) ** 2),
    "bernstein": (0.004, 0.004, 0.01, 0.002, (1 - RHO) ** 2),
    "chernoff": (1.0, 1.0, 1.5, 0.5, 1 - RHO),
}
EPS = {  # (feasible, infeasible under ard1 only)
    "markov": (0.7, 0.6),
    "cantelli": (0.2, 0.11),
    "hoeffding": (0.05, 0.01),
    "bernstein": (0.4, 0.3),
    "chernoff": (0.5, 0.4),
}
B_BERNSTEIN = 0.3            # support constant b = max(support_0 0.3, support_new 0.1, support 0.25)
S_CHERNOFF = 4.0


def trajectory(start, g_start, inc, new, a, law):
    s, g, out, gout = start, g_start, [], []
    for kind in KINDS:
        if kind == "fly":
            s = s + inc
        elif kind == "repair":
            s = a * s + (1 - a) * g if law == "ard1" else a * s
            g = s
        else:
            s = new
            g = s
        out.append(s)
        gout.append(g)
    return out, gout


def z_values(mu, gmu, law):
    z, prev, gprev = [], MU0, MU0
    for k, kind in enumerate(KINDS):
        if kind == "repair":
            z.append(RHO * (prev - (gprev if law == "ard1" else 0.0)))
        else:
            z.append(0.0)
        prev, gprev = mu[k], gmu[k]
    return z


def quad(bound, eps):
    kappa = math.log(1 / eps)
    if bound == "cantelli":
        return eps / (1 - eps), 0.0, TAU
    if bound == "hoeffding":
        return 2 / kappa, 0.0, TAU
    c = kappa * B_BERNSTEIN / 3
    return 1 / (2 * kappa), -B_BERNSTEIN / 3, TAU - c


def caps(c2, c1, mu):
    def f(m):
        return c2 * (TAU - m) ** 2 + c1 * (TAU - m)

    def df(m):
        return -2 * c2 * (TAU - m) - c1

    tangent = f(0.5) + df(0.5) * (mu - 0.5)
    P = 4
    pwl = -math.inf
    for p in range(P):
        lo, hi, mid = p * TAU / P, (p + 1) * TAU / P, (p + 0.5) * TAU / P
        if lo <= mu <= hi:
            pwl = max(pwl, f(mid) + df(mid) * (mu - mid))
    return {"exact": f(mu), "tangent": tangent, "pwl": pwl}


def main():
    for law in ("ard1", "ardinf"):
        mu, gmu = trajectory(MU0, MU0, MU_INC, MU_NEW, 1 - RHO, law)
        print(f"[{law}] mu {fmt(mu)} g_mu {fmt(gmu)} z {fmt(z_values(mu, gmu, law))}")
        for bound, (start, gstart, inc, new, a) in SECOND.items():
            s, g = trajectory(start, gstart, inc, new, a, law)
            print(f"  {bound:9s} state {fmt(s)} anchor {fmt(g)}  (train 1: {start})")
    print()
    for bound, eps_pair in EPS.items():
        for eps in eps_pair:
            kappa = math.log(1 / eps)
            for law in ("ard1", "ardinf"):
                mu, _ = trajectory(MU0, MU0, MU_INC, MU_NEW, 1 - RHO, law)
                points = list(zip(mu, [None] * 4))
                if bound in SECOND:
                    start, gstart, inc, new, a = SECOND[bound]
                    s, _ = trajectory(start, gstart, inc, new, a, law)
                    points = list(zip(mu, s)) + [(MU0, start)]
                if bound == "markov":
                    margin = min(eps * TAU - m for m, _ in points + [(MU0, None)])
                    print(f"{bound:9s} eps {eps:<5} {law:6s} cap {eps*TAU:.6f} margin {margin:+.6f}")
                    continue
                if bound == "chernoff":
                    cap = S_CHERNOFF * TAU + math.log(eps)
                    margin = min(cap - k for _, k in points)
                    print(f"{bound:9s} eps {eps:<5} {law:6s} cap {cap:.6f} range_K {S_CHERNOFF*TAU-kappa:.6f} margin {margin:+.6f}")
                    continue
                c2, c1, side = quad(bound, eps)
                rng = c2 * TAU ** 2 + c1 * TAU
                line = f"{bound:9s} eps {eps:<5} {law:6s} c2 {c2:.6f} c1 {c1:+.6f} side {side:.6f} range {rng:.6f}"
                for form in ("exact", "tangent", "pwl"):
                    margin = min(caps(c2, c1, m)[form] - x for m, x in points)
                    side_ok = all(m <= side for m, _ in points)
                    line += f" | {form} {margin:+.6f}{'' if side_ok else ' SIDE'}"
                print(line)


def fmt(values):
    return "[" + ", ".join(f"{v:.6g}" for v in values) + "]"


if __name__ == "__main__":
    main()
