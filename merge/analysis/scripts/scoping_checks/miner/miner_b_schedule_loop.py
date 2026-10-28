"""Claude's check of interpretation (II), unit-specific scatter, along a schedule (28 Oct 2026).

scoping.tex, subsubsection "Random fatigue life in the Palmgren--Miner rule", interpretation (II):
  (1) eq:miner_quantile is exact along a schedule with missions, ARD repair and replacement:
      Monte Carlo of the physical damage D_k = sum of n_j / N_f,j (with N_f,j = A * Nbar_j, a new A
      at each replacement) against F_A(M_k), M_k the deterministic Miner state;
  (2) F_A(M_k) <= eps if and only if M_k <= q_eps, on every day;
  (3) the rewriting sum_j n_ij n_j / (q_eps Nbar_j) <= 1 holds without repair since the last
      replacement, and fails after a repair; q_eps Nbar_j is the eps-quantile of N_f,j;
  (4) eq:loop_prob is equivalent to eq:miner_loop_det, M_T <= M_H1;
  (5) eq:miner_loop_det chains: random operating phases, 200 repetitions each;
  (6) the Monte Carlo of (1) continued over four repetitions of the operating phase.
Three laws of the unit factor A: log-normal (sigma = 0.5), Weibull (k = 2.5), Gumbel-minimum
(beta = 0.25). Deterministic: numpy generators with fixed seeds.
Run: python miner_b_schedule_loop.py > miner_b_schedule_loop_output.txt
"""
import math
import numpy as np
from scipy import stats

# unit factor A: sampler, F_A and closed-form quantile q_eps
FAMILIES = {
    "log-normal sigma=0.5": (
        lambda rng, size: np.exp(0.5 * rng.standard_normal(size)),
        lambda a: stats.lognorm.cdf(a, 0.5),
        lambda eps: math.exp(0.5 * stats.norm.ppf(eps)),
    ),
    "Weibull k=2.5": (
        lambda rng, size: rng.weibull(2.5, size),
        lambda a: stats.weibull_min.cdf(a, 2.5),
        lambda eps: (-math.log1p(-eps)) ** (1 / 2.5),
    ),
    # A = exp(beta G), G a minimum-Gumbel variable; numpy's gumbel is the maximum law, so G = -gumbel
    "Gumbel beta=0.25": (
        lambda rng, size: np.exp(-0.25 * rng.gumbel(size=size)),
        lambda a: stats.gumbel_l.cdf(np.log(np.maximum(a, 1e-300)) / 0.25) * (np.asarray(a) > 0),
        lambda eps: (-math.log1p(-eps)) ** 0.25,
    ),
}

# missions: one stress level each; Basquin Nbar = C s^-m
C_B, M_B = 1e12, 3.0
STRESS = {"m1": 80.0, "m2": 100.0, "m3": 120.0}
CYCLES = {"m1": 8e4, "m2": 6e4, "m3": 4e4}
NBAR = {j: C_B * s ** (-M_B) for j, s in STRESS.items()}
INC = {j: CYCLES[j] / NBAR[j] for j in STRESS}
RHO = 0.5
TRANSITORY = "m1 m2 m3 m1 m2 idle m3 m1 rep m2 m3 m1".split()
OPERATING = {
    "S1 (repair and replacement)": "m2 m3 m1 m2 rep m3 m1 m2 new m1 m2 m3".split(),
    "S2 (repair only)": "m2 m3 m1 m2 rep m3 m1 m2 rep m1 m2 idle".split(),
}
REPS = 4  # repetitions of the operating phase after the horizon in the Monte Carlo
R = 1_000_000


def run_M(ops, M0=0.0, inc=INC, rho=RHO):
    """Deterministic Miner state after each day."""
    M, out = M0, []
    for op in ops:
        if op in inc:
            M = M + inc[op]
        elif op == "rep":
            M = (1 - rho) * M
        elif op == "new":
            M = 0.0
        out.append(M)
    return np.array(out)


def run_physical(ops, sample, rng):
    """Fraction of R units with D_k >= 1 on each day; D in units of the life fraction."""
    D, A = np.zeros(R), sample(rng, R)
    freq = []
    for op in ops:
        if op in INC:
            D += CYCLES[op] / (A * NBAR[op])  # n_j cycles, each 1 / N_f,j with N_f,j = A Nbar_j
        elif op == "rep":
            D *= 1 - RHO
        elif op == "new":
            D[:] = 0.0
            A = sample(rng, R)  # a new component carries a new factor
        freq.append(np.count_nonzero(D >= 1.0) / R)
    return np.array(freq)


print("increments n_j / Nbar_j:", {j: round(v, 6) for j, v in INC.items()}, " rho =", RHO)
print(f"(1) and (6) Monte Carlo, {R} units per law and schedule: max |z| of (frequency - F_A(M_k))")
zmax_all, ntest = 0.0, 0
for sname, oper in OPERATING.items():
    horizon = TRANSITORY + oper
    full = horizon + oper * REPS
    M = run_M(full)
    H1, T = len(TRANSITORY), len(horizon)
    print(f"  {sname}: H1={H1} T={T}, M_H1={M[H1 - 1]:.5f} M_T={M[T - 1]:.5f}, "
          f"max M on the horizon={M[:T].max():.5f}, days simulated={len(full)}")
    for i, (fname, (sample, F, q)) in enumerate(FAMILIES.items()):
        rng = np.random.default_rng(100 + 10 * i + len(sname))
        freq = run_physical(full, sample, rng)
        p = np.asarray(F(M), dtype=float)
        zero = p == 0
        z = np.zeros_like(p)
        z[~zero] = (freq[~zero] - p[~zero]) / np.sqrt(p[~zero] * (1 - p[~zero]) / R)
        bad0 = np.count_nonzero(freq[zero] > 0)
        zmax = np.max(np.abs(z))
        zmax_all, ntest = max(zmax_all, zmax), ntest + len(full)
        # (6): the smallest eps that the horizon meets, and the days of the repetitions
        eps_h = p[:T].max()
        worst_rep = p[T:].max()
        print(f"    {fname}: F_A(M_k) in [{p[~zero].min():.4f}, {p.max():.4f}], max |z| = {zmax:.2f}, "
              f"days with M=0 and a failure = {bad0}; eps met on the horizon = {eps_h:.4f}, "
              f"max F_A on the repetitions = {worst_rep:.4f}, max frequency on the repetitions = {freq[T:].max():.4f}")
p_exceed = 1 - (1 - 2 * stats.norm.sf(4.0)) ** ntest
print(f"  RESULT: max |z| = {zmax_all:.2f} over {ntest} day-tests; if the formula is exact, "
      f"max |z| > 4 has probability about {p_exceed:.3f}")

print("\n(2) F_A(M_k) <= eps  <=>  M_k <= q_eps, every day of both schedules")
for fname, (sample, F, q) in FAMILIES.items():
    for eps in (0.01, 0.05, 0.1):
        qe = q(eps)
        mism, below = 0, 0
        for oper in OPERATING.values():
            M = run_M(TRANSITORY + oper + oper * REPS)
            lhs = np.asarray(F(M)) <= eps
            rhs = M <= qe
            mism += np.count_nonzero(lhs != rhs)
            below += np.count_nonzero(rhs)
        print(f"  {fname} eps={eps}: q_eps={qe:.5f}, days with M_k <= q_eps: {below} of "
              f"{2 * (len(TRANSITORY) + 12 * (REPS + 1))}, mismatches = {mism}")

print("\n(3) the rewriting with percentile lives")
for sname, oper in OPERATING.items():
    full = TRANSITORY + oper + oper * REPS
    M = run_M(full)
    counts = {j: 0 for j in INC}
    repaired = False
    same, differ, worst = 0, 0, 0.0
    for k, op in enumerate(full):
        if op == "new":
            counts = {j: 0 for j in INC}
            repaired = False
        elif op == "rep":
            repaired = True
        elif op in INC:
            counts[op] += 1
        count_sum = sum(counts[j] * CYCLES[j] / NBAR[j] for j in INC)  # sum_j n_ij n_j / Nbar_j
        if not repaired:
            same += 1
            worst = max(worst, abs(count_sum - M[k]) / max(M[k], 1e-300))
        elif count_sum > M[k] * (1 + 1e-12):
            differ += 1
    print(f"  {sname}: days without repair since the last replacement: {same}, "
          f"max |count sum - M_k| / M_k = {worst:.1e}; days after a repair with count sum > M_k: {differ}")
rng = np.random.default_rng(7)
for fname, (sample, F, q) in FAMILIES.items():
    eps = 0.05
    A = sample(rng, R)
    for j in INC:
        life = A * NBAR[j]
        emp = np.quantile(life, eps)
        print(f"  {fname} {j}: q_eps Nbar_j = {q(eps) * NBAR[j]:.6g}, empirical eps-quantile of N_f,j = {emp:.6g}, "
              f"fraction of N_f,j <= q_eps Nbar_j = {np.mean(life <= q(eps) * NBAR[j]):.5f} (eps = {eps})")

print("\n(4) eq:loop_prob  <=>  M_T <= M_H1: random pairs in [0, 2], with ties and zeros")
rng = np.random.default_rng(11)
MT = np.concatenate([rng.uniform(0, 2, 20000), np.zeros(10), rng.uniform(0, 2, 10)])
MH = np.concatenate([rng.uniform(0, 2, 20000), rng.uniform(0, 2, 10), np.zeros(10)])
MH[:100] = MT[:100]  # ties
for fname, (sample, F, q) in FAMILIES.items():
    mism = np.count_nonzero((np.asarray(F(MT)) <= np.asarray(F(MH))) != (MT <= MH))
    print(f"  {fname}: mismatches = {mism} of {len(MT)}")

print("\n(5) chaining of eq:miner_loop_det: random operating phases, 200 repetitions each")
rng = np.random.default_rng(2026)
q = math.exp(0.5 * stats.norm.ppf(0.01))
valid = viol_mono = viol_end = viol_q = 0
open_cases = open_broken = 0
while valid < 5000:
    H2 = int(rng.integers(2, 16))
    kinds = rng.choice(["mis", "idle", "rep", "new"], size=H2, p=[0.55, 0.15, 0.22, 0.08])
    ops = [f"x{t}" if kd == "mis" else kd for t, kd in enumerate(kinds)]
    inc = {f"x{t}": rng.uniform(0, 0.1) for t in range(H2)}
    rho = rng.uniform(0.05, 0.95)
    MH1 = rng.uniform(0, q)
    traj0 = run_M(ops, MH1, inc, rho)
    if traj0.max() > q:
        continue  # the horizon itself violates eq:miner_quantile
    if traj0[-1] <= MH1:
        valid += 1
        start, prev_end = traj0[-1], traj0[-1]
        for _ in range(200):
            tr = run_M(ops, start, inc, rho)
            viol_mono += bool(np.any(tr > traj0))
            viol_end += bool(tr[-1] > prev_end)
            viol_q += bool(tr.max() > q)
            prev_end = start = tr[-1]
    else:
        open_cases += 1
        start = traj0[-1]
        for _ in range(200):
            tr = run_M(ops, start, inc, rho)
            if tr.max() > q:
                open_broken += 1
                break
            start = tr[-1]
print(f"  q = q_0.01 (log-normal) = {q:.5f}; cases with M_T <= M_H1 and M_k <= q on the horizon: {valid}")
print(f"  violations over 200 repetitions: day state above the horizon state = {viol_mono}, "
      f"end state above the previous end state = {viol_end}, state above q = {viol_q}")
print(f"  control, cases with M_k <= q on the horizon but M_T > M_H1: {open_cases}; "
      f"of these, {open_broken} exceed q in a later repetition")
