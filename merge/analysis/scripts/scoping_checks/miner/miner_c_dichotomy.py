"""Claude's check of Remark rem:miner_dichotomy, interpretations (I) and (II) (28 Oct 2026).

scoping.tex, Remark rem:miner_dichotomy:
  (1) the means coincide and Var_II / Var_I = (sum m_t)^2 / sum m_t^2 <= number of days
      (Cauchy-Schwarz), with equality for equal increments;
  (2) the "representative experiment": log-normal factor, sigma = 0.5, fifty days, median Miner
      sum 0.6. Reading used: equal daily increments m_t = 0.6 / 50; the deterministic Miner sum
      M = 0.6 is the median of D = M / A under (II); under (I) each day draws its own factor
      with the same law, D = sum_t m_t / A_t.
      P(II) = Phi(ln 0.6 / sigma) exactly. P(I) by a discretised convolution whose floor and
      ceiling versions bound P(I) from below and above, and by an independent conditional
      Monte Carlo estimator with a 95 % confidence interval;
  (3) other readings of the experiment, and which of them give a value near 1e-6;
  (4) the reversal "at high accumulated damage": the Miner sum at which P(I) = P(II).
Deterministic: numpy generators with fixed seeds.
Run: python miner_c_dichotomy.py > miner_c_dichotomy_output.txt
"""
import math
import numpy as np
from scipy import optimize, special, stats

SIG, NDAYS, MSUM = 0.5, 50, 0.6


# ---------- discretised convolution on [0, 1), cell width h = 1/N ----------
def cell_probs(m, sig, N):
    """P(X in [i/N, (i+1)/N)) for i = 0..N-1, ln X ~ N(ln m, sig^2); X = m / A with ln A ~ N(0, sig^2)."""
    edges = np.arange(N + 1) / N
    with np.errstate(divide="ignore"):
        z = (np.log(edges) - math.log(m)) / sig
    zl, zh = z[:-1], z[1:]
    return np.where(zl > 0, special.ndtr(-zl) - special.ndtr(-zh), special.ndtr(zh) - special.ndtr(zl))


def conv_trunc(a, b, N):
    """Linear convolution of two sub-probability arrays, kept on cells 0..N-1."""
    L = 1 << int(math.ceil(math.log2(2 * N)))
    return np.fft.irfft(np.fft.rfft(a, L) * np.fft.rfft(b, L), L)[:N]


def power_trunc(p, n, N):
    res, base = None, p.copy()
    while n:
        if n & 1:
            res = base.copy() if res is None else conv_trunc(res, base, N)
        n >>= 1
        if n:
            base = conv_trunc(base, base, N)
    return res


def law_of_sum(ms, sig, N):
    """Cell masses of sum_t floor(N X_t) on 0..N-1, X_t = m_t / A_t independent."""
    vals, counts = np.unique(np.asarray(ms, dtype=float), return_counts=True)
    res = None
    for v, c in zip(vals, counts):
        part = power_trunc(cell_probs(v, sig, N), int(c), N)
        res = part if res is None else conv_trunc(res, part, N)
    return res


def p_fail_I(ms, sig, N):
    """(lower bound, midpoint estimate, upper bound) of P(sum_t m_t / A_t >= 1).

    X_t lies in [i h, (i+1) h). The floor sum S_lo = h sum i_t and the ceiling sum S_lo + n h
    bracket S, so P(S_lo >= 1) <= P(S >= 1) <= P(S_lo + n h >= 1)."""
    n = len(ms)
    cum = np.cumsum(law_of_sum(ms, sig, N))
    lo = 1.0 - cum[N - 1]
    hi = 1.0 - cum[N - 1 - n]
    mid = 1.0 - cum[int(math.ceil(N - n / 2)) - 1]
    return lo, mid, hi


def p_fail_II(M, sig):
    return stats.norm.cdf(math.log(M) / sig)


def equal(M, n=NDAYS):
    return np.full(n, M / n)


# ---------- (1) means and variances ----------
print("(1) means and the variance ratio, (I) against (II)")
laws = {
    "log-normal sigma=0.5": (lambda rng, s: np.exp(SIG * rng.standard_normal(s)),
                             math.exp(SIG ** 2 / 2), math.exp(SIG ** 2) * (math.exp(SIG ** 2) - 1)),
    "Weibull k=5": (lambda rng, s: rng.weibull(5.0, s),
                    special.gamma(0.8), special.gamma(0.6) - special.gamma(0.8) ** 2),
}
rng = np.random.default_rng(1)
profiles = {
    "equal": np.full(NDAYS, 0.012),
    "linear m_t ~ t": np.arange(1, NDAYS + 1) * 0.6 / (NDAYS * (NDAYS + 1) / 2),
    "Dirichlet(1)": 0.6 * rng.dirichlet(np.ones(NDAYS)),
    "ten zero days": np.concatenate([np.full(40, 0.015), np.zeros(10)]),
}
for pname, m in profiles.items():
    ratio = m.sum() ** 2 / (m ** 2).sum()
    print(f"  profile {pname}: (sum m)^2 / sum m^2 = {ratio:.4f}, days = {len(m)}, days with m_t > 0 = {np.count_nonzero(m)}")
    for lname, (sample, e1, v1) in laws.items():
        R, chunk = 400_000, 50_000
        dI, dII = [], []
        for _ in range(R // chunk):
            dI.append((m / sample(rng, (chunk, len(m)))).sum(axis=1))
            dII.append(m.sum() / sample(rng, chunk))
        dI, dII = np.concatenate(dI), np.concatenate(dII)
        print(f"    {lname}: mean I {dI.mean():.5f} II {dII.mean():.5f} formula {m.sum() * e1:.5f}; "
              f"Var I {dI.var():.4e} formula {(m ** 2).sum() * v1:.4e}; Var II {dII.var():.4e} formula {m.sum() ** 2 * v1:.4e}; "
              f"MC ratio {dII.var() / dI.var():.3f}")
W = rng.dirichlet(np.ones(NDAYS) * 0.3, size=100_000)
r = W.sum(axis=1) ** 2 / (W ** 2).sum(axis=1)
print(f"  100000 random profiles: max ratio / days = {r.max() / NDAYS:.4f} (bound 1); equal profile ratio = "
      f"{profiles['equal'].sum() ** 2 / (profiles['equal'] ** 2).sum():.10f}")

# ---------- (2) the experiment ----------
print("\n(2) the experiment: sigma = 0.5, 50 days, equal increments m_t = 0.012, M = 0.6")
pII = p_fail_II(MSUM, SIG)
print(f"  P(II) = Phi(ln 0.6 / 0.5) = {pII:.6f}")
print("  P(I) by convolution: N cells on [0, 1); lower bound, midpoint estimate, upper bound")
for e in range(13, 22, 2):
    N = 1 << e
    lo, mid, hi = p_fail_I(equal(MSUM), SIG, N)
    print(f"    N=2^{e}: lower {lo:.6e}  mid {mid:.6e}  upper {hi:.6e}")
LO, MID, HI = lo, mid, hi
# floating-point error of the FFT: same arrays by direct convolution
N = 1 << 12
p = cell_probs(MSUM / NDAYS, SIG, N)
direct = p.copy()
for _ in range(NDAYS - 1):
    direct = np.convolve(direct, p)[:N]
fft = power_trunc(p, NDAYS, N)
fp_err = np.max(np.abs(np.cumsum(direct) - np.cumsum(fft)))
print(f"  floating-point check at N=2^12: max |cumulative(FFT) - cumulative(direct)| = {fp_err:.1e}")

# independent check: conditional Monte Carlo (the largest factor integrated exactly) with a mean shift
print("  P(I) by conditional Monte Carlo: P(S >= c) = n E[ P(Y_n > max(max_{t<n} Y_t, c - S_{n-1})) ],")
print("  Y_t = exp(sigma Z_t), c = n / M, the first n-1 normals shifted by theta and reweighted")
c = NDAYS / MSUM
rng = np.random.default_rng(20261028)
for theta in (0.4, 0.5, 0.6):
    vals = []
    for _ in range(40):
        Z = rng.standard_normal((100_000, NDAYS - 1)) + theta
        Y = np.exp(SIG * Z)
        thr = np.maximum(Y.max(axis=1), c - Y.sum(axis=1))
        lw = -theta * Z.sum(axis=1) + (NDAYS - 1) * theta ** 2 / 2
        vals.append(NDAYS * special.ndtr(-np.log(thr) / SIG) * np.exp(lw))
    vals = np.concatenate(vals)
    est, se = vals.mean(), vals.std(ddof=1) / math.sqrt(len(vals))
    inside = est - 1.96 * se <= HI and est + 1.96 * se >= LO
    print(f"    theta={theta}, 4e6 samples: {est:.4e}, 95% CI [{est - 1.96 * se:.4e}, {est + 1.96 * se:.4e}], "
          f"overlaps the convolution bracket: {inside}; relative standard error {se / est:.2%}")

# the estimator with one uniform mean shift of all fifty normals (the shift that centres D on 1)
print("  importance sampling with one uniform mean shift of all fifty normals:")
th_u = (math.log(1 / MSUM) - SIG ** 2 / 2) / SIG
vals = []
for _ in range(20):
    Z = rng.standard_normal((100_000, NDAYS)) + th_u
    hit = np.exp(SIG * Z).sum(axis=1) >= c
    vals.append(np.where(hit, np.exp(-th_u * Z.sum(axis=1) + NDAYS * th_u ** 2 / 2), 0.0))
vals = np.concatenate(vals)
est, se = vals.mean(), vals.std(ddof=1) / math.sqrt(len(vals))
print(f"    theta={th_u:.4f}, 2e6 samples: {est:.4e}, 95% CI [{est - 1.96 * se:.4e}, {est + 1.96 * se:.4e}], "
      f"overlaps the convolution bracket: {est - 1.96 * se <= HI and est + 1.96 * se >= LO}; "
      f"relative standard error {se / est:.2%}")
th_s = math.log(c / NDAYS) / SIG
rate_sym = NDAYS * th_s ** 2 / 2
f = lambda r: (math.log(c - (NDAYS - 1) * math.exp(SIG * r)) / SIG) ** 2 / 2 + (NDAYS - 1) * r ** 2 / 2
res = optimize.minimize_scalar(f, bounds=(-3, th_s), method="bounded")
b = math.log(c - (NDAYS - 1) * math.exp(SIG * res.x)) / SIG
print(f"    reason: the cheapest way to reach the event (min |z|^2 / 2) is not the symmetric point "
      f"(all z_t = {th_s:.3f}, cost {rate_sym:.2f}) but one large factor (z = {b:.2f}, the others {res.x:.2f}, "
      f"cost {res.fun:.2f}); a uniform shift rarely samples it")
print(f"  RESULT: P(II) = {pII:.4f}; P(I) in [{LO:.4e}, {HI:.4e}], midpoint {MID:.4e}; ratio P(II)/P(I) = {pII / MID:.3g}")

# ---------- (3) other readings ----------
print("\n(3) other readings of the experiment (convolution at N=2^18, midpoint; lower and upper bounds in brackets)")
N = 1 << 18


def fmt(t):
    return f"{t[1]:.3e} [{t[0]:.3e}, {t[2]:.3e}]"


print(f"  R1 the reading of (2): P(II) = {pII:.4f}, P(I) = {fmt(p_fail_I(equal(MSUM), SIG, N))}")
# R2: 0.6 is the median of D under (I)
g = lambda Mp: p_fail_I(equal(Mp), SIG, N)[1] - 0.5
Mp = optimize.brentq(g, 0.4, 1.2, xtol=1e-10)
M_I = MSUM * Mp  # median of (M_I/n) sum 1/A_t is 0.6
print(f"  R2 median of D under (I) equal to 0.6: M = {M_I:.5f}; P(I) = {fmt(p_fail_I(equal(M_I), SIG, N))}; "
      f"P(II) at M = 0.6 stays {pII:.4f}, at M = {M_I:.4f} it is {p_fail_II(M_I, SIG):.4f}")
# R3: 0.6 is the mean of D (equal under both)
M_mean = MSUM * math.exp(-SIG ** 2 / 2)
print(f"  R3 mean of D equal to 0.6: M = {M_mean:.5f}; P(II) = {p_fail_II(M_mean, SIG):.4f} (not 0.15); "
      f"P(I) = {fmt(p_fail_I(equal(M_mean), SIG, N))}")
# R4: equal increments on fewer days
print("  R4 equal increments on n days, M = 0.6:")
pn = {}
for n in (1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50):
    pn[n] = p_fail_I(equal(MSUM, n), SIG, N)
    print(f"    n={n}: P(I) = {fmt(pn[n])}")
ns = sorted(pn)
for a, b2 in zip(ns, ns[1:]):
    if pn[a][1] >= 1e-6 > pn[b2][1]:
        print(f"    P(I) = 1e-6 lies between n = {a} and n = {b2}")
# R5: unequal increments over 50 days, m_t ~ t^a
print("  R5 unequal increments over 50 days, m_t proportional to t^a, M = 0.6:")
t = np.arange(1, NDAYS + 1, dtype=float)
prof = lambda a: MSUM * t ** a / (t ** a).sum()
neff = lambda m: m.sum() ** 2 / (m ** 2).sum()
for a in (0.0, 1.0, 2.0, 3.0, 4.0):
    m = prof(a)
    print(f"    a={a}: effective days (sum m)^2/sum m^2 = {neff(m):.2f}, P(I) = {fmt(p_fail_I(m, SIG, N))}")
a6 = optimize.brentq(lambda a: math.log(p_fail_I(prof(a), SIG, N)[1]) - math.log(1e-6), 0.0, 4.0, xtol=1e-6)
print(f"    P(I) = 1e-6 at a = {a6:.3f}, effective days {neff(prof(a6)):.2f}, last-day share {prof(a6)[-1] / MSUM:.3f}")
rng = np.random.default_rng(5)
pr, ne = [], []
for _ in range(20):
    m = MSUM * rng.dirichlet(np.ones(NDAYS))
    pr.append(p_fail_I(m, SIG, N)[1])
    ne.append(neff(m))
print(f"    20 random profiles (uniform on the simplex): effective days {min(ne):.1f}..{max(ne):.1f}, "
      f"P(I) min {min(pr):.2e}, median {np.median(pr):.2e}, max {max(pr):.2e}")
# R6: plain Monte Carlo with 1e6 samples
lam = 1e6 * MID
print(f"  R6 plain Monte Carlo with 1e6 samples: expected failures {lam:.3f}; P(0 failures) = {math.exp(-lam):.3f}, "
      f"P(1 failure, printed as 1e-6) = {lam * math.exp(-lam):.3f}, P(2 or more) = {1 - math.exp(-lam) * (1 + lam):.3f}")
# R7: sigma as the scatter of log10 N_f
print(f"  R7 sigma = 0.5 read on log10 N_f: P(II) = Phi(ln 0.6 / (0.5 ln 10)) = {p_fail_II(MSUM, SIG * math.log(10)):.4f} (not 0.15)")
# R8: approximations under (I)
mean_I = MSUM * math.exp(SIG ** 2 / 2)
var_I = NDAYS * (MSUM / NDAYS) ** 2 * math.exp(SIG ** 2) * (math.exp(SIG ** 2) - 1)
print(f"  R8 (I) by a normal approximation: {stats.norm.sf((1 - mean_I) / math.sqrt(var_I)):.2e}; "
      f"by the Cantelli bound: {var_I / (var_I + (1 - mean_I) ** 2):.4f}")

# ---------- (4) the reversal ----------
print("\n(4) the reversal: equal increments, 50 days, sigma = 0.5, P(I) and P(II) as functions of M")
N = 1 << 18
h = lambda M: p_fail_I(equal(M), SIG, N)[1] - p_fail_II(M, SIG)
Mx = optimize.brentq(h, 0.7, 1.0, xtol=1e-9)
N = 1 << 20
for M in (Mx - 0.001, Mx, Mx + 0.001):
    lo, mid, hi = p_fail_I(equal(M), SIG, N)
    print(f"  M={M:.5f}: P(I) in [{lo:.6f}, {hi:.6f}], P(II) = {p_fail_II(M, SIG):.6f}")
print(f"  RESULT: P(I) = P(II) = {p_fail_II(Mx, SIG):.4f} at M = {Mx:.4f} "
      f"(median damage under (II)); mean damage there {Mx * math.exp(SIG ** 2 / 2):.4f}")
# sign on a grid, with monotonicity in M: on [Ma, Mb], P(I)(M) <= upper(Mb) and P(II)(M) >= P(II)(Ma)
N = 1 << 16
tol = 1e-12
grid = np.geomspace(0.1, 3.0, 301)
P = [p_fail_I(equal(M), SIG, N) for M in grid]
below = above = undecided = 0
for i in range(len(grid) - 1):
    Ma, Mb = grid[i], grid[i + 1]
    if P[i + 1][2] + tol < p_fail_II(Ma, SIG):
        below += 1  # P(I) < P(II) on the whole interval
    elif P[i][0] - tol > p_fail_II(Mb, SIG):
        above += 1  # P(I) > P(II) on the whole interval
    else:
        undecided += 1
        print(f"  undecided interval [{Ma:.4f}, {Mb:.4f}]")
print(f"  grid of 300 intervals on M in [0.1, 3]: P(I) < P(II) on {below}, P(I) > P(II) on {above}, undecided {undecided}")
print(f"  P(II) at the ends of the grid: {p_fail_II(0.1, SIG):.2e} at M = 0.1, {p_fail_II(3.0, SIG):.4f} at M = 3")
