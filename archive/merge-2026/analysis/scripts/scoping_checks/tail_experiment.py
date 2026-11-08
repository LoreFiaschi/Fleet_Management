"""Tail experiment for the Route-I bounds.

Checks the claims of scoping.tex, subsubsection "Conservative surrogates ...", paragraph
"The choice among the Route~I bounds depends on how far into the tail the constraint lies"
(the bullets Chernoff, Cantelli, Bernstein and their footnote).

Setting
  D = X_1 + ... + X_k, with X_t independent and identically distributed on [0, 1], k in {30, 100}.
  The threshold tau is chosen so that the exact tail Pr(D > tau) is close to a target eps.
  Main table: eps = 1e-4. Depth sweep: eps in {1e-2, 1e-4, 1e-6, 1e-8}.

Exact tail
  - Uniform(0, 1): Irwin-Hall formula, evaluated in mpmath with 120 digits; tau solves
    Pr(D > tau) = eps.
  - Beta laws: lattice bracket. X_lo = h*floor(X/h) has an exact lattice law (cell masses from
    the regularised incomplete Beta function). S_lo, the sum of k copies, is computed by FFT
    convolution. Since S_lo <= D < S_lo + k*h almost surely,
        Pr(S_lo > tau) <= Pr(D > tau) <= Pr(S_lo > tau - k*h).
    Both ends are printed; the ratios use their geometric mean, and the printed "+-" is the
    largest relative error that the bracket allows. The uniform law is run through the same
    lattice code as a check of the method (the bracket must contain the Irwin-Hall value).
  - Two-point law on {0, 1} (Bernoulli(p)): D is binomial and takes integer values. tau is the
    integer whose tail Pr(D >= tau) is closest to eps (in log scale), and every bound is compared
    with Pr(D >= tau). All five bounds also bound Pr(D >= tau). For tau' in [tau - 1, tau) the
    tail Pr(D > tau') is the same number and every bound is larger, so these ratios are the
    smallest over the thresholds that give this tail.

Bounds (formulas of scoping.tex), with E = k*mean, V = k*var, a = tau - E
  Markov     E / tau
  Cantelli   V / (V + a^2)
  Hoeffding  exp(-2 a^2 / R),  R = k (sum of the squared widths, width 1 per term)
  Bernstein  exp(-(a^2 / 2) / (V + b a / 3)),  b = 1 (the width of the range)
  Chernoff   inf_{s > 0} exp(-s tau + k psi(s)),  psi the exact cumulant generating function,
             minimised over s (s* solves k psi'(s*) = tau)
  Extra column "Chern-ms": Chernoff with the mean-and-support bound
             psi(s) <= ln(1 + mean (e^s - 1)) of scoping.tex, the bound that needs only the mean
             and the width; it equals the exact Chernoff bound for the two-point law on {0, 1}.

Deterministic: no random numbers. Run: python tail_experiment.py > tail_experiment_output.txt
"""
import math

import mpmath as mp
import numpy as np
from scipy import special, stats
from scipy.signal import fftconvolve

mp.mp.dps = 120

K_LIST = (30, 100)
EPS_MAIN = 1e-4
EPS_SWEEP = (1e-2, 1e-4, 1e-6, 1e-8)
H_STEP = 2e-5  # lattice step of the bracket

# name, kind, parameters
LAWS = [
    ("Uniform(0,1)", "uniform", (1.0, 1.0)),
    ("Beta(2,2)", "beta", (2.0, 2.0)),
    ("Beta(0.5,5)", "beta", (0.5, 5.0)),
    ("Two-point p=0.5", "bern", 0.5),
    ("Two-point p=0.05", "bern", 0.05),
]


# ---------------------------------------------------------------- moments and CGF
def moments(kind, par):
    if kind in ("uniform", "beta"):
        a, b = par
        m = a / (a + b)
        v = a * b / ((a + b) ** 2 * (a + b + 1))
        return m, v
    p = par
    return p, p * (1 - p)


def chernoff_exact(kind, par, k, tau):
    """inf_s exp(-s tau + k psi(s)) with the exact CGF; returns (bound, s*)."""
    with mp.workdps(40):
        return _chernoff_exact(kind, par, k, tau)


def _chernoff_exact(kind, par, k, tau):
    q = mp.mpf(tau) / k
    if kind == "bern":
        p = mp.mpf(par)
        if q >= 1:
            return float(p ** k), float("inf")
        s = mp.log(q * (1 - p) / (p * (1 - q)))
        kl = q * mp.log(q / p) + (1 - q) * mp.log((1 - q) / (1 - p))
        return float(mp.e ** (-k * kl)), float(s)
    a, b = (mp.mpf(x) for x in par)

    def dpsi(s):
        return (a / (a + b)) * mp.hyp1f1(a + 1, a + b + 1, s) / mp.hyp1f1(a, a + b, s)

    lo, hi = mp.mpf(0), mp.mpf(1)
    while dpsi(hi) < q:
        hi *= 2
    for _ in range(120):  # bisection on k psi'(s) = tau
        mid = (lo + hi) / 2
        if dpsi(mid) < q:
            lo = mid
        else:
            hi = mid
    s = (lo + hi) / 2
    val = mp.e ** (-s * tau + k * mp.log(mp.hyp1f1(a, a + b, s)))
    return float(val), float(s)


def chernoff_mean_support(mean, k, tau):
    """Chernoff bound with psi(s) <= ln(1 + mean (e^s - 1)) (two-point law on {0, 1})."""
    q, p = mp.mpf(tau) / k, mp.mpf(mean)
    if q >= 1:
        return float(p ** k)
    kl = q * mp.log(q / p) + (1 - q) * mp.log((1 - q) / (1 - p))
    return float(mp.e ** (-k * kl))


def all_bounds(kind, par, k, tau):
    m, v = moments(kind, par)
    E, V = k * m, k * v
    a = tau - E
    out = {
        "Markov": E / tau,
        "Cantelli": V / (V + a * a),
        "Hoeffding": math.exp(-2 * a * a / k),
        "Bernstein": math.exp(-(a * a / 2) / (V + a / 3)),
    }
    out["Chernoff"], s_star = chernoff_exact(kind, par, k, tau)
    out["Chern-ms"] = chernoff_mean_support(m, k, tau)
    z = a / math.sqrt(V)
    return out, s_star, z


# ---------------------------------------------------------------- exact tails
def irwin_hall_sf(k, tau):
    """Pr(U_1 + ... + U_k > tau) = Pr(sum < k - tau), exact alternating sum."""
    x = mp.mpf(k) - mp.mpf(tau)
    if x <= 0:
        return mp.mpf(0)
    tot = mp.mpf(0)
    for j in range(int(mp.floor(x)) + 1):
        tot += (-1) ** j * mp.binomial(k, j) * (x - j) ** k
    return tot / mp.factorial(k)


def irwin_hall_tau(k, eps):
    m, v = k / 2, k / 12
    z = stats.norm.isf(eps)
    lo, hi = mp.mpf(m), mp.mpf(k) - mp.mpf("1e-30")
    for _ in range(200):  # bisection: sf is decreasing in tau
        mid = (lo + hi) / 2
        if irwin_hall_sf(k, mid) > eps:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2), z * math.sqrt(v) + m


def lattice_pmf(a, b, h):
    """mass[i] = Pr(i h <= X < (i+1) h) for X ~ Beta(a, b)."""
    n = int(round(1 / h))
    x = np.arange(n + 1) * h
    x[-1] = 1.0
    cdf = special.betainc(a, b, x)
    sf = special.betainc(b, a, 1.0 - x)
    upper = x[1:] > 0.5 + 1e-12
    mass = np.where(upper, sf[:-1] - sf[1:], cdf[1:] - cdf[:-1])
    return mass


def conv_power(p, k):
    result, base = None, p.copy()
    while k:
        if k & 1:
            result = base.copy() if result is None else fftconvolve(result, base)
        k >>= 1
        if k:
            base = fftconvolve(base, base)
    return result


class LatticeTail:
    """Bracket of Pr(D > tau) from the lattice sum S_lo."""

    def __init__(self, a, b, k, h):
        self.k, self.h = k, h
        p = conv_power(lattice_pmf(a, b, h), k)
        self.min_entry = float(p.min())
        self.mass = float(p.sum())
        c = np.cumsum(p[::-1])[::-1]          # c[i] = Pr(S_lo >= i h)
        self.T = np.append(c[1:], 0.0)        # T[i] = Pr(S_lo > i h)

    def bracket_at_index(self, i):
        """tau = (i + 0.5) h: (Pr(S_lo > tau), Pr(S_lo > tau - k h))."""
        return self.T[i], self.T[i - self.k]

    def tau_for(self, eps):
        prod = np.clip(self.T[self.k:] * self.T[:-self.k], 0.0, None)  # round-off can give -1e-21
        g = np.sqrt(prod)  # geometric mean of the bracket at tau = (j + k + 0.5) h
        j = int(np.argmax(g <= eps))
        if j > 0 and abs(math.log(g[j - 1] / eps)) < abs(math.log(g[j] / eps)):
            j -= 1
        i = j + self.k
        return (i + 0.5) * self.h, i

    def bracket(self, tau):
        i = int(math.floor(tau / self.h))
        return self.T[i], self.T[i - self.k]


def binom_sf_ge(k, p, t):
    """Pr(Bin(k, p) >= t), exact in mpmath."""
    p = mp.mpf(p)
    return sum(mp.binomial(k, j) * p ** j * (1 - p) ** (k - j) for j in range(t, k + 1))


def binom_tau(k, p, eps):
    best = None
    for t in range(1, k + 1):
        tail = binom_sf_ge(k, p, t)
        if tail <= 0:
            continue
        d = abs(mp.log(tail / eps))
        if best is None or d < best[0]:
            best = (d, t, tail)
    return best[1], float(best[2])


# ---------------------------------------------------------------- experiment
COLS = ["Markov", "Cantelli", "Hoeffding", "Bernstein", "Chernoff", "Chern-ms"]


def run_case(name, kind, par, k, eps, lat_cache):
    """Return a dict with tau, exact tail (lo, hi) and the bounds."""
    if kind == "uniform":
        tau, _ = irwin_hall_tau(k, eps)
        exact = float(irwin_hall_sf(k, tau))
        lo = hi = exact
        event = ">"
        lat = lat_cache.get((name, k))
        check = None
        if lat is not None:
            blo, bhi = lat.bracket(tau)
            check = (blo, exact, bhi, blo <= exact <= bhi)
    elif kind == "beta":
        lat = lat_cache[(name, k)]
        tau, i = lat.tau_for(eps)
        lo, hi = lat.bracket_at_index(i)
        event = ">"
        check = None
    else:
        t, tail = binom_tau(k, par, eps)
        tau = float(t)
        lo = hi = tail
        event = ">="
        sp_tail = stats.binom.sf(t - 1, k, par)
        check = (sp_tail, tail, abs(sp_tail / tail - 1))
    b, s_star, z = all_bounds(kind, par, k, tau)
    mid = math.sqrt(lo * hi)
    return dict(name=name, k=k, eps=eps, tau=tau, lo=lo, hi=hi, mid=mid, event=event,
                bounds=b, s=s_star, z=z, check=check)


def fmt_row(r):
    relerr = math.sqrt(r["hi"] / r["lo"]) - 1
    ratios = "  ".join("%9.3g" % (r["bounds"][c] / r["mid"]) for c in COLS)
    return ("%-17s %4d %9.4f %2s %9.3e +-%4.1f%% %6.2f %6.3g | %s"
            % (r["name"], r["k"], r["tau"], r["event"], r["mid"], 100 * relerr, r["z"], r["s"], ratios))


HEADER = ("%-17s %4s %9s %2s %9s %7s %6s %6s | %s"
          % ("law", "k", "tau", "ev", "exact", "", "z", "s*", "  ".join("%9s" % c for c in COLS)))


def main():
    print("Tail experiment for the Route-I bounds (deterministic).")
    print("Lattice step h = %g; mpmath digits = %d" % (H_STEP, mp.mp.dps))
    print()
    lat_cache = {}
    for name, kind, par in LAWS:
        if kind in ("uniform", "beta"):
            for k in K_LIST:
                lat = LatticeTail(par[0], par[1], k, H_STEP)
                lat_cache[(name, k)] = lat
                print("lattice %-14s k=%3d: total mass - 1 = %.2e, smallest entry = %.2e"
                      % (name, k, lat.mass - 1, lat.min_entry))
    print()

    results = []
    for eps in EPS_SWEEP:
        for k in K_LIST:
            for name, kind, par in LAWS:
                results.append(run_case(name, kind, par, k, eps, lat_cache))

    print("Columns: bound / exact tail. 'exact' is Pr(D ev tau); '+-' is the relative half-width")
    print("of the rigorous bracket of the exact tail (0 for the Irwin-Hall and binomial values).")
    print("z = (tau - E[D]) / sd(D); s* = optimal Chernoff point.")
    for eps in EPS_SWEEP:
        print()
        print("=== target eps = %g ===" % eps)
        print(HEADER)
        for r in results:
            if r["eps"] == eps:
                print(fmt_row(r))

    print()
    print("=== Checks of the method ===")
    for r in results:
        c = r["check"]
        if c is None:
            continue
        if len(c) == 4:
            print("Uniform k=%3d eps=%g: lattice bracket [%.6e, %.6e] contains Irwin-Hall %.6e: %s"
                  % (r["k"], r["eps"], c[0], c[2], c[1], c[3]))
        else:
            print("Binomial %s k=%3d eps=%g: scipy %.10e vs mpmath %.10e, rel. diff %.1e"
                  % (r["name"], r["k"], r["eps"], c[0], c[1], c[2]))

    print()
    print("=== Summary over the cases ===")
    for eps in EPS_SWEEP:
        rs = [r for r in results if r["eps"] == eps]
        ch = [r["bounds"]["Chernoff"] / r["mid"] for r in rs]
        ca = [r["bounds"]["Cantelli"] / r["mid"] for r in rs]
        be = [r["bounds"]["Bernstein"] / r["mid"] for r in rs]
        ho = [r["bounds"]["Hoeffding"] / r["mid"] for r in rs]
        better = [r["name"] + " k=%d" % r["k"] for r in rs
                  if r["bounds"]["Bernstein"] >= r["bounds"]["Cantelli"]]
        print("eps=%g: Chernoff/exact in [%.3g, %.3g]; Cantelli/exact in [%.3g, %.3g]; "
              "Bernstein/exact in [%.3g, %.3g]; Hoeffding/exact in [%.3g, %.3g]"
              % (eps, min(ch), max(ch), min(ca), max(ca), min(be), max(be), min(ho), max(ho)))
        print("          cases with Bernstein >= Cantelli: %s" % (better if better else "none"))
        order = [r["name"] + " k=%d" % r["k"] for r in rs
                 if not (r["bounds"]["Chernoff"] <= r["bounds"]["Bernstein"] <= r["bounds"]["Cantelli"])]
        print("          cases where Chernoff <= Bernstein <= Cantelli fails: %s" % (order if order else "none"))


if __name__ == "__main__":
    main()
