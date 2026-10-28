"""S1: is IG(sum a_i, sum a_i/eta) a safe tail model for a sum of independent IG(a_i, a_i/eta)?"""
import math, sys, time
sys.path.insert(0, sys.argv[1])
from iglib import *

TAU = 1.0

print("== A. lead: closure and cumulant test ==")
eta = 0.25
# density of IG(1,1/eta)+IG(1,1/eta) by convolution vs IG(2,4/eta) and model IG(2,2/eta)
n, X = 4000, 8.0
h = X / n
f = [ig_pdf(i * h, 1.0, 1.0 / eta) for i in range(n + 1)]
worst_true = worst_model = 0.0
for i in range(0, n + 1, 50):
    g = h * sum(f[j] * f[i - j] for j in range(i + 1))
    worst_true = max(worst_true, abs(g - ig_pdf(i * h, 2.0, 4.0 / eta)))
    worst_model = max(worst_model, abs(g - ig_pdf(i * h, 2.0, 2.0 / eta)))
print(f"eta={eta}: max|conv - pdf IG(2,4/eta)| = {worst_true:.2e};  max|conv - pdf IG(2,2/eta)| = {worst_model:.3f}")
for ms in [(1, 1), (1, 2), (1, 1, 5), (0.1,) * 10]:
    s1, s2, s3 = sum(ms), sum(m * m for m in ms), sum(m ** 3 for m in ms)
    print(f"means {ms}: k1*k3/k2^2 = {3 * s1 * s3 / s2 ** 2:.4f} (IG law: 3);  var true/model = {s2 / s1 ** 2:.3f}")


def groups_to_terms(groups, eta):
    """groups = [(n, a)]: n independent terms IG(a, a/eta); their sum is IG(n a, n^2 a/eta)."""
    return [(k * a, k * k * a / eta) for k, a in groups]


print("\n== B. realistic examples: true tail at the cap mu_bar (model tail = eps) ==")
cases = [(0.05, 0.01), (0.25, 0.01), (1.0, 0.01), (0.25, 0.001), (0.25, 0.1), (1.0, 0.1)]
structs = {
    "D0 0.2 + 10x0.05 + 5x0.06": [(1, 0.2), (10, 0.05), (5, 0.06)],
    "repaired 0.55 + 3x0.15": [(1, 0.55), (3, 0.15)],
    "2 equal halves": [(2, 0.5)],
    "big 0.9 + 10x0.01": [(1, 0.9), (10, 0.01)],
}
for eta, eps in cases:
    mb = mu_bar_eta(eta, TAU, eps)
    line = f"eta={eta:<5} eps={eps:<6} mu_bar={mb:.4f} |"
    for name, gr in structs.items():
        terms = groups_to_terms([(k, a * mb) for k, a in gr], eta)
        t1 = sum_tail(terms, TAU, 1000)
        t2 = sum_tail(terms, TAU, 2000)
        line += f" {t2 / eps:.3g} (err {abs(t2 - t1) / eps:.0e}) |"
    print(line)
print("(entries: true tail / eps for each structure, in the order", list(structs), ")")

print("\n== B'. Monte Carlo cross-check (10^6 draws) ==")
eta, eps = 1.0, 0.1
mb = mu_bar_eta(eta, TAU, eps)
for name in ("D0 0.2 + 10x0.05 + 5x0.06", "big 0.9 + 10x0.01"):
    gr = structs[name]
    terms_groups = groups_to_terms([(k, a * mb) for k, a in gr], eta)
    terms_single = []
    for k, a in gr:
        terms_single += [(a * mb, a * mb / eta)] * k
    p_num = sum_tail(terms_groups, TAU, 2000)
    p_mc, se = mc_tail(terms_single, TAU, 1000000, seed=3)
    print(f"eta={eta} eps={eps} {name}: convolution {p_num:.5f}, MC (each mission sampled) {p_mc:.5f} +- {se:.5f}")

print("\n== C. scan for unsafe cases: n equal terms (true law IG(mu, n mu/eta)), mu = mu_bar ==")
ns = [2, 3, 5, 10, 30, 100, 1000, 10 ** 4, 10 ** 6]
for eps in (0.001, 0.01, 0.05, 0.1, 0.2):
    row = f"eps={eps:<6}"
    for eta in (0.01, 0.05, 0.25, 1.0, 4.0, 10.0, 30.0, 100.0):
        mb = mu_bar_eta(eta, TAU, eps)
        r = [(ig_sf(TAU, mb, n * mb / eta) / eps, n) for n in ns]
        best = max(r)
        row += f" | eta={eta}: {best[0]:.3g}@n={best[1]}"
    print(row)

print("\n== C'. two-group structures (big term b + n small terms), eta, eps where C is closest to 1 ==")
for eta, eps in [(1.0, 0.1), (4.0, 0.1), (10.0, 0.1), (10.0, 0.01), (30.0, 0.01)]:
    mb = mu_bar_eta(eta, TAU, eps)
    best = (0, None)
    for b in (0.0, 0.2, 0.4, 0.6, 0.8, 0.9):
        for n in (1, 3, 10, 100):
            gr = [(n, (1 - b) / n)] + ([(1, b)] if b > 0 else [])
            terms = groups_to_terms([(k, a * mb) for k, a in gr], eta)
            p = sum_tail(terms, TAU, 1000) if len(terms) > 1 else ig_sf(TAU, *terms[0])
            if p / eps > best[0]:
                best = (p / eps, (b, n))
    print(f"eta={eta} eps={eps} mu_bar={mb:.4f}: max true/eps = {best[0]:.3f} at (b, n) = {best[1]}")

print("\n== D. crossing points of the true and model survival functions (mean 1) ==")
for eta in (0.25, 1.0, 10.0):
    for n in (2, 10):
        d = lambda t: ig_sf(t, 1.0, n / eta) - ig_sf(t, 1.0, 1.0 / eta)
        ts = [0.01 * i for i in range(1, 1001)]
        cross = [round(ts[i], 2) for i in range(len(ts) - 1) if (d(ts[i]) > 0) != (d(ts[i + 1]) > 0)]
        print(f"eta={eta} n={n}: sign changes of S_true - S_model on (0,10]: {cross}; "
              f"model tail there = {[round(ig_sf(c, 1.0, 1.0 / eta), 4) for c in cross]}")
