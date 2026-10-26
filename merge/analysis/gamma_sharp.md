# Gamma: the sharp safe cap over the class of weighted Gamma sums

Status: done: all computations for T1 to T4 (scripts run, outputs saved). In progress: the write-up;
T1 is written below in full draft form, T2 to T4 are written as drafts with all numbers. Next: re-read
the draft against the outputs in `scripts/gamma_sharp/`, tighten it to about 220 lines, set Status to done.
Paused on 26 Oct 2026 at the coordinator's request.

All derivations, proofs, counterexamples and numbers in this file are Claude's own (26 Oct 2026).
No literature was read for it. Symbols as in `gamma.md`.

## Symbols

- Scaling: with rate `b`, `b·S` is a member of the rate-1 class. So set `b = 1`; `t` stands for `b·t`
  (or `b·tau`) and `A` for `b·m` (the shape of the single Gamma with mean `m`).
- `Q(a, x) = P(Gamma(a, 1) > x)`; `f_a(x) = x^(a-1) e^(-x) / Γ(a)`, the density of `Gamma(a, 1)`.
- Members: `S = δ + Σ_i c_i·Y_i`, `Y_i ~ Gamma(m_i/c_i, 1)` independent, `c_i ∈ (0, 1]`, `δ >= 0`,
  `δ + Σ m_i = m`. The drift `δ` is the limit `c -> 0` at fixed mean `c·a`; it lies in the closure of
  `C(m)`. A member is a finite measure `ν = δ·δ_0 + Σ m_i·δ_{c_i}` on `[0, 1]` of mass `m` (mean mass at scale `c`).
- One-term member: `c·Gamma(A/c, 1)`; its tail at `t` is `Q(A/c, t/c)`. `c = 1` is the single Gamma.
- `E1(y) = ∫_y^∞ e^(-u)/u du`. `y0 = 0.434818`: the root of `e^y·E1(y) = 1` (`t0_lib_check.py`).
- `W(m, t)`, `A_sharp(t, eps)`: as in the brief. `A_hat`: `Q(A_hat, t) = eps`. `A_icx`: S5 of `gamma.md`.
- **One-term cap** `A_1(t, eps) = sup{A < t : max over c ∈ (0, 1] of Q(A/c, t/c) <= eps}`.
- **Threshold** `t1(eps)`: the root in `t` of `g(t) = t·f_A(t) - A·∂Q/∂a(A, t)`, with `A = A_hat(t, eps)`.
  `g = d/dc Q(A/c, t/c)` at `c = 1`.
- **Assumption U** (labelled; not proven here): every finite sum of independent Gamma variables (any
  shapes, any rates) has a unimodal density on `(0, ∞)`. See T4.

## T1. Is the single Gamma the worst case at the cap?

**Verdict.** No below `t1(eps)`; yes above it, under Assumption U.

| `eps` | 0.001 | 0.01 | 0.1 | limit `eps -> 0` |
|---|---|---|---|---|
| `t1(eps)` | 0.43614 | 0.44839 | 0.61422 | `y0 = 0.43482` (`t1(1e-4) = 0.43495`) |

- `b·t >= 1` is therefore safe for every `eps <= 0.1` (under U). The S3 conjecture of `gamma.md` holds,
  for every schedule with one rate `b`, under U, and already for `b·tau >= t1(eps)`.

### Statement 1 (proven, no assumption). Below `t1(eps)` the single Gamma is not the worst case.

- **(a) First-order rate.** Let `S` be a member with a density continuous at `t`. Add mean mass `η` as a
  term `c·Gamma(η/c, 1)`. Then `P(S > t)` grows at rate
  `φ_S(c) = (1/c)·E[E1((t - S)/c); S < t]`, and `φ_S(0) = f_S(t)` (drift).
  - Proof. The Lévy density of `Gamma(η/c, 1/c)` is `(η/c)·e^(-y/c)/y`. By dominated convergence the
    rate is `∫ P(t - y < S <= t)·e^(-y/c)/(c·y) dy`. Fubini over `S` gives the `E1` form.
  - Removing mass from a term `c_i·Y_i` gives rate `-φ_S(c_i)`; moving its scale `c_i` at fixed mass
    `m_i` gives rate `m_i·∂φ_S/∂c (c_i)` (same Lévy-density computation).
  - Check (`t0_lib_check.py`): quadrature of `φ_S(c) - φ_S(1)` against finite differences of exact
    tails, single Gamma, 24 cases: relative difference at most 1.1e-5.
- **(b) Slope at `c = 1`.** `d/dc Q(A/c, t/c)` at `c = 1` equals `g(t) = t·f_A(t) - A·∂Q/∂a(A, t) = A·φ'(1)`.
  Check (`t1c_one_term.py`, part e): `g/A` and `φ'(1)` agree to 7 digits at `t = 1, 3`.
  - If `g < 0`, a one-term member with `c` slightly below 1 has a larger tail than `Gamma(A)`.
  - Small-shape limit: `t·f_A(t)/A -> e^(-t)` and `∂Q/∂a -> E1(t)`, so `g/A -> e^(-t) - E1(t)`, whose
    root is `y0`. Hence `t1(eps) -> y0` as `eps -> 0`.
- **(c) Single crossing (proven).** For `c ∈ (0, 1)` the survival functions of `Gamma(A, 1)` and
  `c·Gamma(A/c, 1)` cross exactly once, at `t_x(A, c)`; `Gamma(A)` has the larger tail for `t > t_x`.
  - Proof. `ln(f_X/f_Y)(x) = const + A(1 - 1/c)·ln x + (1/c - 1)·x` is strictly convex, so the densities
    cross at most twice. Equal mass and equal mean force exactly two crossings, with `f_X > f_Y` near 0
    and near ∞. Integrate `f_X - f_Y` from `t` to ∞: the difference of tails is 0 at both ends,
    first negative, then positive, so it changes sign once.
  - For a fixed `A`, the single Gamma beats every one-term member exactly on `t >= sup_c t_x(A, c)`.
- **(d) Witnesses** (`t1c_one_term.py`, part c). Worst one-term member at the cap `A = A_hat`: `c*` and
  exact tail / `eps`.

| `t` | `eps = 0.001` | `eps = 0.01` | `eps = 0.1` |
|---|---|---|---|
| 0.01 | `c* = 0.0225`: 7.007 | `c* = 0.0180`: 7.375 | `A_hat > t`: the drift alone gives tail 1 |
| 0.1 | 0.229: 1.546 | 0.219: 1.565 (Monte Carlo 1.555 ± 0.020) | 0.1115: 1.836 |
| 0.3 | 0.688: 1.0365 | 0.668: 1.0413 (Monte Carlo 1.028 ± 0.016) | 0.4675: 1.109 |
| 0.4 | 0.917: 1.0021 | 0.892: 1.0035 | 0.641: 1.0385 |
| 0.43 | 0.986: 1.000056 | 0.959: 1.00048 | 0.693: 1.0266 |
| 0.6 | above `t1` | above `t1` | 0.977: 1.000114 |

  - Each witness is a member with tail `> eps` at `A_hat`; by continuity in `A`, `A_sharp < A_hat` there.
  - For comparison: at `b·tau = 0.01`, `eps = 0.01`, the S3 schedules reach 1.41·`eps`; the class reaches 7.4·`eps`.
  - Local scan (`t1a_first_order.py`): `max over c of φ(c) - φ(1)` is positive at every grid `t < t1`
    and negative at every grid `t > t1` (grid: 14 values of `t` from 0.01 to 100, 52 values of `c`).

### Statement 2 (theorem, under Assumption U). The worst member is a one-term member.

- **Statement.** Let `t > m`. Under U, `W(m, t) = max over c ∈ (0, 1] of Q(m/c, t/c)`.
  Hence `A_sharp(t, eps) = A_1(t, eps)` for every `t` and `eps`.
- **Proof** (Claude's own; `φ*` is `φ_S` of the maximiser `S*`; `λ = 1/c`).
  - P1. For `K >= 3`, the members with at most `K` atoms (`c_i ∈ [0, 1]`, `c = 0` the drift) form a
    compact set. `P(S > t)` is continuous on it: only the pure drift `m` has an atom, and `m ≠ t`.
    So a maximiser `S*` exists for each `K`, and `W = sup over K` of these maxima.
  - P2 (first-order conditions, Statement 1a). (i) `φ*` takes one value `κ` on the atoms with positive
    mass. (ii) `∂φ*/∂λ = 0` at each such atom with `c ∈ (0, 1)`. (iii) An atom of zero mass can sit at
    any `c`; then `φ*(c) <= κ` on all of `[0, 1]`.
  - P3. `h(u) = P(t - u < S* <= t)/u` is the running average of `v -> f*(t - v)` (zero for `v > t`).
    Under U this function is unimodal. A running average of a unimodal function is unimodal: after the
    mode, `D(u) = u·g(u) - ∫_0^u g` is nonincreasing, so `g - h` changes sign once.
  - P4. `φ*(λ) - κ = λ·∫ (h(u) - κ)·e^(-λu) du`. If `h - κ` changes sign at most twice, then `φ* - κ`
    has at most two zeros in `λ > 0`, counted with multiplicity (induction with Rolle's theorem:
    multiply by `e^(λ·u1)` at a sign change `u1` and differentiate in `λ`). Also `φ*(λ) -> 0` as `λ -> 0+`.
    So every superlevel set of `φ*` is an interval: `φ*` is quasi-concave. It is analytic and not
    constant, so its maximum over `λ ∈ [1, ∞]` is reached at one point only.
  - P5. By (i), (ii), each atom with `c ∈ (0, 1)` is a zero of `φ* - κ` of multiplicity at least 2, an
    atom at `c = 1` a zero of multiplicity at least 1, the drift no zero. With P4, at most two atoms
    carry mass. For `K >= 3` an atom of zero mass remains, so by (iii) `κ = max φ*`. By P4 the maximum
    is at one point, so `S*` has one atom. The drift alone has tail 0 `< Q(m, t)`. So
    `S* = c*·Gamma(m/c*, 1)` with `c* ∈ (0, 1]`. ∎
- **Consequence.** Combined with the dense one-dimensional check (`t1c_one_term.py`, part b):
  for `t` on a log grid of 301 points in `[t1(eps), 1000]`, the largest `Q(A_hat/c, t/c)/eps` over
  `c ∈ [1e-4, 0.999]` is at most 0.9999997, and `g > 0` at every grid point. So under U,
  `A_sharp = A_hat` on that range (checked on the grid, not for every real `t`).

### Statement 3 (numerical). Checks of Statement 2 that do not use U.

- **Global search** (`t1b_global_search.py`, outputs `t1b_output_<eps>.txt`). Nelder-Mead with random
  restarts over members with `K = 1, 2, 3` terms plus a drift; 16 values of `t` from 0.01 to 100; three
  `eps`. The best `K = 2` and `K = 3` members never beat the best one-term member (equal to 6 digits).
  Above `t1` every search returns the single Gamma (ratio 1.000000).
- **Unimodality** (`t1d_unimodal.py`). 600 random members (2 to 4 terms, `c ∈ [0.05, 1]`, shapes
  `1e-3` to 10), density on a 600-point grid: no density rises after a fall.

## T2. The sharp cap: table

Source: `t2_table.py` (output `t2_output.txt`).

| `b·t` | `A_hat`, `eps` = 0.001 / 0.01 / 0.1 | `A_icx/A_hat`, 0.001 / 0.01 / 0.1 | `A_1/A_hat`, 0.001 / 0.01 / 0.1 |
|---|---|---|---|
| 0.01 | 0.00024776 / 0.0024877 / 0.025959 | 0.040 / 0.040 / 0.039 | 0.143 / 0.142 / 0.127 |
| 0.1 | 0.00054872 / 0.0055012 / 0.056549 | 0.182 / 0.182 / 0.177 | 0.647 / 0.642 / 0.585 |
| 0.3 | 0.0011039 / 0.011021 / 0.10906 | 0.278 / 0.278 / 0.280 | 0.965 / 0.961 / 0.909 |
| 1 | 0.0045372 / 0.043653 / 0.34564 | 0.340 / 0.349 / 0.396 | 1 / 1 / 1 |
| 3 | 0.067651 / 0.39997 / 1.4207 | 0.389 / 0.491 / 0.599 | 1 / 1 / 1 |
| 10 | 2.3715 / 3.9717 / 6.5748 | 0.781 / 0.807 / 0.805 | 1 / 1 / 1 |
| 100 | 71.077 / 77.997 / 87.798 | 0.965 / 0.960 / 0.948 | 1 / 1 / 1 |

- **Status of each value.**
  - `A_icx`: proven safe (S5 of `gamma.md`), so `A_icx <= A_sharp` without any assumption.
  - `A_1` for `b·t < t1`: an upper bound on `A_sharp`, proven by a witness (a one-term member with exact
    tail `> eps` at `A_1·(1 + 1e-6)`; for example `c* = 0.223`, tail `1.000001·eps`, at `b·t = 0.1`,
    `eps = 0.01`; Monte Carlo at `1.05·A_1`: exact 0.01050, 3·10^6 draws 0.01051 ± 0.00006).
    Under U it equals `A_sharp` (Statement 2).
  - `A_1 = A_hat` for `b·t >= 1`: `A_hat` is a trivial upper bound (the single Gamma is a member);
    equality `A_sharp = A_hat` holds under U and the grid check.
- Gain over S5 below the threshold: about 3.5 times `A_icx` at `b·t = 0.01` and `0.1`; at `b·t = 1`
  the gain is `1/0.35 = 2.9` (`eps = 0.01`).

## T3. Threshold and cap rule (draft)

- **Threshold as a function of `eps`.** `t1(eps)` solves `t·f_A(t) = A·∂Q/∂a(A, t)` with
  `A = A_hat(t, eps)`. Values: 0.43614, 0.44839, 0.61422 for `eps` = 0.001, 0.01, 0.1; it tends to
  `y0 = 0.43482` as `eps -> 0`.
- **Cap rule (Claude's own proposal).** For each cell (train `i`, component `l`) with rate `b`
  (or `b*`, the smallest rate, when `A' = b*·mu` is tracked as in S5):
  - offline constant `A_cap = A_1(b·tau, eps)`: bisection on `A`, inner one-dimensional maximum over `c`;
  - `A_cap = A_hat(b·tau, eps)` when `b·tau >= t1(eps)`; `A_1 < A_hat` below it (T2 table);
  - row on every day `k`: `A_k <= A_cap`. One row per cell and day, as `lore`'s row `alpha <= alpha_hat`.
    No new variables, no binaries; `F·L·2H` rows in total, the same count as `lore`.
  - Without Assumption U: `A_cap = A_icx` everywhere (S5), same rows.
- **Caveat.** The class `C(b, m)` is larger than the set of states a schedule can reach: a reachable
  state also satisfied the cap on every earlier day. So `A_1` is safe, but may be conservative below
  `t1`. Not studied.

## T4. Theorems relied on (draft)

- **Assumption U** (not read, labelled). It would follow from two results in the literature:
  every self-decomposable law is unimodal (Yamazato, 1978); and a law on `[0, ∞)` whose Lévy density is
  `k(y)/y` with `k` nonincreasing is self-decomposable. Each member has `k(y) = Σ a_i e^(-y/c_i)`,
  nonincreasing. Neither source was read; another agent reads the literature.
- **Standard facts used without proof:** compactness of a closed bounded parameter set; the Lévy density
  `a·e^(-ry)/y` of `Gamma(a, r)`; dominated convergence; Rolle's theorem; analyticity of a Laplace transform.
- **Proven here:** the first-order rate (Statement 1a), the single crossing (1c), the running-average
  lemma (P3), the zero count for Laplace transforms (P4), the reduction (Statement 2).
- **Not used:** Diaconis-Perlman 1990, Székely-Bakirov 2003, Roosta-Khorasani-Székely-Ascher 2015.

## Resume here

- T1 to T4: all numbers are computed and recorded above. Nothing needs to be recomputed.
- Remaining work: editing only.
  1. Re-read each statement against its output file in `scripts/gamma_sharp/`.
  2. Shorten to about 220 lines (merge the witness table and the T2 table where they overlap).
  3. Keep the labels: Statement 2 and the cap `A_1` are conditional on Assumption U; the range above
     `t1` is a grid check up to `b·t = 1000`, not a proof for every `t`.
  4. Set `Status:` to done and send the final reply (at most 35 lines: verdicts, T2 table, notes path).
- Optional, if time allows: a numerical check of the scale-derivative rate `m_i·∂φ_S/∂c` on a two-term
  member (only the single-Gamma case is checked, in `t1c_one_term.py`, part e).

## Scripts

Folder `C:\Users\utente\Documents\GitHub\Fleet_Management\merge\analysis\scripts\gamma_sharp\` (copied from
the session scratchpad). Each script runs with `python -I`, standard library only; its output is in the
`*_output*.txt` file of the same name.

| Script | Content |
|---|---|
| `gamma_lib.py` | copy of the library of `gamma.md` (incomplete gamma, sum of Gammas, bounds) |
| `sharp_lib.py` | `E1`, tanh-sinh quadrature, `φ`, `A_hat`, `A_icx`, member tails, Nelder-Mead |
| `t0_lib_check.py` | `E1` values, `y0`, `φ` against finite differences, member tails against Monte Carlo |
| `t1a_first_order.py` | local scan of `φ(c) - φ(1)` at the cap |
| `t1b_global_search.py` | global search, `K` = 1 to 3 terms plus drift (one output per `eps`) |
| `t1c_one_term.py` | `t1(eps)`, dense check above `t1`, witnesses, cross-check `g/A = φ'(1)` |
| `t1d_unimodal.py` | unimodality spot check |
| `t2_table.py` | T2 table and witness check |
