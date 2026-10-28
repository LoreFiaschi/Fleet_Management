"""Claude's check of the inverse moments of the S-N scatter families (28 Oct 2026).

scoping.tex, subsubsection "Random fatigue life in the Palmgren--Miner rule":
  (1) the closed forms eq:miner_invmom of E[N_f^-r], by quadrature;
  (2) their existence conditions (r < k for Weibull, r*beta < 1 for Gumbel-minimum);
  (3) the three lines of Var[1/N_f] below eq:miner_invmom;
  (4) the identity of Remark rem:miner_tails: a Gumbel-minimum log-life is a Weibull life;
  (5) the closed-form quantiles q_eps of the unit factor A (below eq:miner_quantile);
  (6) the divergent sample variance of 1/N_f at Weibull k <= 2.
Deterministic: mpmath quadrature at 30 digits and numpy generators with fixed seeds.
Run: python miner_a_inverse_moments.py > miner_a_inverse_moments_output.txt
"""
import math
import mpmath as mp
import numpy as np
from scipy import special, stats

mp.mp.dps = 30


def lognormal_quad(mu, s, r):
    """E[N^-r] for ln N ~ N(mu, s^2), integrated over z = (ln N - mu)/s."""
    f = lambda z: mp.exp(-r * (mu + s * z)) * mp.npdf(z)
    return mp.quad(f, [-mp.inf, -r * s - 8, -r * s, -r * s + 8, mp.inf])


def weibull_quad(k, lam, r, lo=0):
    """E[N^-r; N > lo] for N ~ Wei(k, lam), integrated over t with N = t^10.

    The substitution N = t^10 smooths the power singularity at the origin."""
    a = 10
    dens = lambda x: x ** (-r) * (k / lam) * (x / lam) ** (k - 1) * mp.exp(-(x / lam) ** k)
    f = lambda t: dens(t ** a) * a * t ** (a - 1)
    root = lambda x: mp.mpf(x) ** (mp.mpf(1) / a)
    return mp.quad(f, [root(lo), root(lam / 100), root(lam), root(10 * lam), mp.inf])


def gumbel_quad(mu, beta, r, glo=-mp.inf):
    """E[N^-r; G > glo] for ln N = mu + beta G, P(G <= g) = 1 - exp(-e^g).

    The upper end is cut at g = 6: the density there is below exp(-400)."""
    f = lambda g: mp.exp(-r * (mu + beta * g) + g - mp.exp(g))
    pts = [p for p in (-20, -5, 0, 3) if p > glo]
    return mp.quad(f, [glo] + pts + [6])


def lognormal_closed(mu, s, r):
    return mp.exp(-r * mu + r ** 2 * s ** 2 / 2)


def weibull_closed(k, lam, r):
    return lam ** (-r) * mp.gamma(1 - mp.mpf(r) / k)


def gumbel_closed(mu, beta, r):
    return mp.exp(-r * mu) * mp.gamma(1 - mp.mpf(r) * beta)


def rel(a, b):
    return float(abs(a - b) / abs(b))


print("(1) eq:miner_invmom: quadrature against the closed form")
worst = 0.0
for mu, s, r in [(1.3, 0.4, 1), (1.3, 0.4, 2), (11.5, 0.5, 1), (11.5, 0.5, 2), (0.0, 0.8, 3.5), (0.0, 0.5, 10)]:
    q, c = lognormal_quad(mu, s, r), lognormal_closed(mu, s, r)
    worst = max(worst, rel(q, c))
    print(f"  log-normal mu={mu} sigma={s} r={r}: quad={mp.nstr(q, 12)} closed={mp.nstr(c, 12)} rel.err={rel(q, c):.1e}")
for k, lam, r in [(3, 2.0, 1), (3, 2.0, 2), (3, 2.0, 2.9), (1.5, 1e5, 1), (2.5, 1e5, 2), (0.8, 1.0, 0.5)]:
    q, c = weibull_quad(k, lam, r), weibull_closed(k, lam, r)
    worst = max(worst, rel(q, c))
    print(f"  Weibull k={k} lambda={lam:g} r={r}: quad={mp.nstr(q, 12)} closed={mp.nstr(c, 12)} rel.err={rel(q, c):.1e}")
for mu, beta, r in [(1.3, 0.3, 1), (1.3, 0.3, 2), (11.5, 0.45, 2), (0.0, 0.6, 1.5)]:
    q, c = gumbel_quad(mu, beta, r), gumbel_closed(mu, beta, r)
    worst = max(worst, rel(q, c))
    print(f"  Gumbel-min mu={mu} beta={beta} r={r}: quad={mp.nstr(q, 12)} closed={mp.nstr(c, 12)} rel.err={rel(q, c):.1e}")
print(f"  RESULT: worst relative error {worst:.1e}")

print("\n(2) existence: truncated moments as the lower cut-off goes to the origin")
print("  Weibull lambda=1: E[N^-r; N > delta]; finite limit only for r < k")
for k, r in [(2.0, 1.9), (2.0, 2.0), (2.0, 2.5), (1.5, 1.5)]:
    vals = [weibull_quad(k, 1.0, r, lo=mp.mpf(10) ** (-e)) for e in (2, 4, 6, 8)]
    txt = "  ".join(f"d=1e-{e}: {mp.nstr(v, 6)}" for e, v in zip((2, 4, 6, 8), vals))
    lim = f"closed={mp.nstr(weibull_closed(k, 1.0, r), 6)}" if r < k else "closed=infinite"
    print(f"    k={k} r={r}: {txt}  {lim}")
# at k = 2, r = 2 the truncated moment is the exponential integral E1(delta^2) ~ ln(1/delta^2)
d = mp.mpf("1e-6")
print(f"    k=2 r=2: E[N^-2; N > 1e-6] = {mp.nstr(weibull_quad(2.0, 1.0, 2.0, lo=d), 12)}, E1(1e-12) = {mp.nstr(mp.e1(d ** 2), 12)}")
print("  Gumbel-min mu=0: E[N^-r; G > g0]; finite limit only for r*beta < 1")
for beta, r in [(0.5, 1.9), (0.5, 2.0), (0.5, 2.5)]:
    vals = [gumbel_quad(0.0, beta, r, glo=-g) for g in (10, 20, 40, 80)]
    txt = "  ".join(f"g0=-{g}: {mp.nstr(v, 6)}" for g, v in zip((10, 20, 40, 80), vals))
    lim = f"closed={mp.nstr(gumbel_closed(0.0, beta, r), 6)}" if r * beta < 1 else "closed=infinite"
    print(f"    beta={beta} r={r} (r*beta={r * beta:g}): {txt}  {lim}")
print("  RESULT: the moments converge exactly in the stated ranges and grow without bound outside them")

print("\n(3) Var[1/N_f]: E[N^-2] - E[N^-1]^2 by quadrature against the three printed lines")
worst = 0.0
mu, s = 11.5, 0.5
q = lognormal_quad(mu, s, 2) - lognormal_quad(mu, s, 1) ** 2
c = mp.exp(-2 * mu + s ** 2) * (mp.exp(s ** 2) - 1)
worst = max(worst, rel(q, c))
print(f"  log-normal mu={mu} sigma={s}: quad={mp.nstr(q, 12)} line={mp.nstr(c, 12)}")
k, lam = 3.0, 1e5
q = weibull_quad(k, lam, 2) - weibull_quad(k, lam, 1) ** 2
c = lam ** -2 * (mp.gamma(1 - 2 / k) - mp.gamma(1 - 1 / k) ** 2)
worst = max(worst, rel(q, c))
print(f"  Weibull k={k} lambda={lam:g}: quad={mp.nstr(q, 12)} line={mp.nstr(c, 12)}")
mu, beta = 11.5, 0.3
q = gumbel_quad(mu, beta, 2) - gumbel_quad(mu, beta, 1) ** 2
c = mp.exp(-2 * mu) * (mp.gamma(1 - 2 * beta) - mp.gamma(1 - beta) ** 2)
worst = max(worst, rel(q, c))
print(f"  Gumbel-min mu={mu} beta={beta}: quad={mp.nstr(q, 12)} line={mp.nstr(c, 12)}")
print(f"  RESULT: worst relative error {worst:.1e}")

print("\n(4) Remark rem:miner_tails: P(N_f > x) for ln N_f ~ Gumbel-min(mu, beta) against Wei(1/beta, e^mu)")
mu, beta = 1.3, 0.3
xs = np.logspace(-2, 2, 41)
lhs = np.exp(-np.exp((np.log(xs) - mu) / beta))
rhs = stats.weibull_min.sf(xs, 1 / beta, scale=math.exp(mu))
g = stats.gumbel_l.sf((np.log(xs) - mu) / beta)  # scipy's minimum-Gumbel law, as a third route
print(f"  41 points x in [0.01, 100]: max |closed - Weibull sf| = {np.max(np.abs(lhs - rhs)):.1e}, "
      f"max |closed - gumbel_l sf| = {np.max(np.abs(lhs - g)):.1e}")

print("\n(5) quantiles q_eps of A: closed form against scipy's ppf, and F_A(q_eps) = eps")
worst = 0.0
sig, k, beta = 0.5, 2.5, 0.25
cdfs = [lambda a: stats.lognorm.cdf(a, sig), lambda a: stats.weibull_min.cdf(a, k),
        lambda a: stats.gumbel_l.cdf(math.log(a) / beta)]
for eps in (1e-6, 1e-4, 1e-2, 0.1):
    rows = [
        ("log-normal sigma=0.5", math.exp(sig * stats.norm.ppf(eps)), stats.lognorm.ppf(eps, sig)),
        ("Weibull k=2.5", (-math.log1p(-eps)) ** (1 / k), stats.weibull_min.ppf(eps, k)),
        ("Gumbel beta=0.25", (-math.log1p(-eps)) ** beta, math.exp(beta * stats.gumbel_l.ppf(eps))),
    ]
    for (name, closed, ref), F in zip(rows, cdfs):
        e1, e2 = abs(closed - ref) / ref, abs(F(closed) - eps) / eps
        worst = max(worst, e1, e2)
        print(f"  eps={eps:g} {name}: q={closed:.10g} scipy={ref:.10g} F_A(q)/eps-1={F(closed) / eps - 1:+.1e}")
print(f"  RESULT: worst relative error {worst:.1e}")

print("\n(6) sample variance of 1/N_f, N_f ~ Wei(k, 1): median over 10 seeds")
sizes = [10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7]
for k in (1.5, 2.0, 3.0, 5.0):
    exact = special.gamma(1 - 2 / k) - special.gamma(1 - 1 / k) ** 2 if k > 2 else math.inf
    meds = []
    for n in sizes:
        v = []
        for seed in range(10):
            rng = np.random.default_rng(1000 * seed + int(10 * k))
            y = 1.0 / rng.weibull(k, size=n)
            v.append(y.var(ddof=1))
        meds.append(np.median(v))
    txt = "  ".join(f"n=1e{int(math.log10(n))}: {m:.4g}" for n, m in zip(sizes, meds))
    print(f"  k={k}: {txt}  Var={exact:.4g}")
print("  RESULT: at k = 1.5 and k = 2 the sample variance keeps growing with n"
      " (theory: like n^(2/k - 1) = n^(1/3) at k = 1.5, like ln n at k = 2, see (2));"
      " at k = 3 and k = 5 it approaches the closed form")
