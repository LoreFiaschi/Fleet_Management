"""S2: fixed xi = lam/mu^2 family Q(m) = IG(m, xi m^2); pathwise vs parameter-space ARD repair."""
import math, sys
sys.path.insert(0, sys.argv[1])
from iglib import *

TAU = 1.0


def Q(m, xi):
    return (m, xi * m * m)


def mu_bar_xi(xi, tau, eps):
    g = lambda m: ig_sf(tau, *Q(m, xi)) - eps
    hi = tau * (1 - 1e-9)
    return hi if g(hi) <= 0 else bisect(g, 1e-9 * tau, hi)


def stoploss(terms, d, n=4000):
    """E[(S - d)_+] for a sum of independent IG terms, via int_d^inf P(S > x) dx."""
    mu = sum(m for m, _ in terms)
    var = sum(m ** 3 / l for m, l in terms)
    top = d + mu + 60 * math.sqrt(var) + 60 * max(m ** 2 / l for m, l in terms)
    h = (top - d) / n
    if len(terms) == 1:
        vals = [ig_sf(d + i * h, *terms[0]) for i in range(n + 1)]
    else:
        vals = [sum_tail(terms, d + i * h, 400) for i in range(0, n + 1)]
    return simpson(vals, h)


print("== A. semigroup check: Q(0.3)+Q(0.5) vs Q(0.8), xi=200 (tails at several t) ==")
xi = 200.0
for t in (0.9, 1.0, 1.1):
    print(f"t={t}: convolution {sum_tail([Q(0.3, xi), Q(0.5, xi)], t, 2000):.10f}  closed form {ig_sf(t, *Q(0.8, xi)):.10f}")

print("\n== B. repaired law: pathwise c*Q(mu) = IG(c mu, c xi mu^2) vs parameter-space Q(c mu) ==")
for xi, mu, c in [(200.0, 0.8, 0.5), (200.0, 0.8, 0.2), (20.0, 0.8, 0.5)]:
    path = (c * mu, c * xi * mu * mu)
    par = Q(c * mu, xi)
    ts = [0.005 * i for i in range(1, 600)]
    d = [ig_sf(t, *path) - ig_sf(t, *par) for t in ts]
    cross = [round(ts[i], 3) for i in range(len(ts) - 1) if (d[i] > 0) != (d[i + 1] > 0)]
    print(f"xi={xi} mu={mu} c={c}: mean {c*mu:.3f}; S_path - S_param changes sign at t = {cross}; "
          f"S_param there = {[round(ig_sf(x, *par), 4) for x in cross]}")
    for dd in (0.5 * c * mu, c * mu, 1.5 * c * mu):
        print(f"   stop-loss at d={dd:.3f}: path {stoploss([path], dd):.6f} <= param {stoploss([par], dd):.6f}")

print("\n== C. repair at the cap, then increments up to the cap again: true tail / eps ==")
print("(true law = c*Q(mu_bar) + Q(mu_bar - c mu_bar), param-space law = Q(mu_bar), tail at tau = 1)")
for xi in (20.0, 200.0, 2000.0):
    for eps in (0.001, 0.01, 0.1, 0.2):
        mb = mu_bar_xi(xi, TAU, eps)
        row = f"xi={xi:<7} eps={eps:<6} mu_bar_xi={mb:.4f} |"
        for c in (0.8, 0.5, 0.2, 0.05):
            s = c * mb
            terms = [(s, c * xi * mb * mb), Q(mb - s, xi)]
            p = sum_tail(terms, TAU, 2000)
            row += f" c={c}: {p / eps:.3g}"
        print(row)

print("\n== D. does a tail comparison survive further increments? xi=200, c=0.2, eps=0.2 ==")
xi, c, eps = 200.0, 0.2, 0.2
mb = mu_bar_xi(xi, TAU, eps)
s = c * mb
for extra in (0.0, 0.1, 0.3, 0.5, mb - s):
    mean = s + extra
    terms = [(s, c * xi * mb * mb)] + ([Q(extra, xi)] if extra > 0 else [])
    # tau' = (1-eps) quantile of the parameter-space law Q(mean)
    tq = bisect(lambda t: eps - ig_sf(t, *Q(mean, xi)), 1e-6, 10.0)
    p = sum_tail(terms, tq, 2000) if len(terms) > 1 else ig_sf(tq, *terms[0])
    print(f"added mean {extra:.3f}: total mean {mean:.3f}, tau'={tq:.4f}: true tail {p:.4f} vs param tail {eps}")

print("\n== E. same scan with MC cross-check for one cell ==")
xi, eps, c = 200.0, 0.2, 0.2
mb = mu_bar_xi(xi, TAU, eps)
s = c * mb
terms = [(s, c * xi * mb * mb), Q(mb - s, xi)]
p_num = sum_tail(terms, TAU, 2000)
p_mc, se = mc_tail(terms, TAU, 1000000, seed=5)
print(f"convolution {p_num:.5f}, MC {p_mc:.5f} +- {se:.5f}")
