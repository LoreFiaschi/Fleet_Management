# Gamma degradation: are the `lore` and `main` formulations safe?

Status: done: S1 to S5, each checked by its script. In progress: nothing. Next: the user's decision on
the item "Gamma formulation" in `../decisions.md`.

All derivations, proofs, counterexamples and proposals in this file are Claude's own (26 Oct 2026).
No literature was read for it.

## Symbols

- `Gamma(a, b)`: Gamma law with shape `a > 0` and **rate** `b > 0`; mean `a/b`, variance `a/b²`.
  `lore`'s scale `beta` is `1/b`. `main`'s `gamma_beta` is a rate.
- `Q(a, x) = P(Gamma(a, 1) > x)`, so `P(Gamma(a, b) > t) = Q(a, b·t)`. It increases in `a`.
- `tau`: failure threshold; `eps`: target probability; `rho`: repair efficiency in (0, 1].
- Repair laws: ARD∞ repairs the whole state, `D+ = (1-rho)·D-`; ARD1 repairs the damage since the last
  intervention, `D+ = D- - rho·(D- - G)`, with `G` the state just after that intervention.
- MILP: mixed-integer linear program. MC: Monte Carlo. s.e.: standard error.
- Tracked shape `A`; reliability row `A <= A_hat`, with `Q(A_hat, b·tau) = eps` (`lore`) or
  `A <= A_max`, with `Q(A_max, b*·tau) = eps` (`main`, common rate `b*`).
- Orders between random variables `X`, `Y` (standard definitions):
  - usual stochastic order `X <=st Y`: `P(X > t) <= P(Y > t)` for every `t`;
  - convex order `X <=cx Y`: `E f(X) <= E f(Y)` for every convex `f`;
  - increasing convex order `X <=icx Y`: the same for every increasing convex `f`.
- Tools check (`s0_lib_check.py`): `Q` matches closed forms to `2e-14` (relative); the sum formula
  below matches MC (400,000 draws) within the 95 % band in three cases.
- **Sum of Gammas with different rates, exact formula** (own derivation, used by every script). Let
  `c = max b_i`. For `b <= c`, `Gamma(a, b)` is the mixture of `Gamma(a + K, c)` over
  `K ~ NegBin(a, p = b/c)`, `P(K = k) = Γ(a+k)/(Γ(a)·k!)·p^a·(1-p)^k`. Proof: both Laplace transforms
  equal `(b/(b+s))^a`. So `Σ X_i` is `Gamma(Σ a_i + K, c)` with `K = Σ K_i`, and
  `P(Σ X_i > t) = Σ_k P(K = k)·Q(Σ a_i + k, c·t)`. Cutting the series loses at most the mass left out;
  the scripts report both ends.

## S1. Sum of Gammas with different rates

- **Statement.** Let `X_i ~ Gamma(a_i, b_i)` be independent and `0 < b* <= min b_i`. Then
  `Σ X_i <=st Gamma(Σ a_i, b*)`.
- **Proof.** Let `Y_i ~ Gamma(a_i, b*)` be independent and set `X_i := (b*/b_i)·Y_i`; this has the law
  `Gamma(a_i, b_i)`. Since `b*/b_i <= 1` and `Y_i >= 0`, `X_i <= Y_i` on every path. So
  `Σ X_i <= Σ Y_i ~ Gamma(Σ a_i, b*)` on every path, which gives the order.
- **Consequence.** Keeping each mission's own shape `a_q` at the common rate `b*` gives a safe linear
  formulation: the row `Σ a_q <= A_max` is never optimistic.
- **Check** (`s1_common_rate.py`): 300 random cases with two or three terms, seven thresholds each.
  The largest value of (true tail minus bound tail) is `-5.5e-11`: the bound is never below the truth.
- **Conservatism, one example.** `tau = 1`, `eps = 0.01`. Type 1: `Gamma(1, 10)` (mean 0.1). Type 2:
  `Gamma(10, 100)` (mean 0.1, much less spread). `b* = 10`, so `A_max = 3.9717`.

| Missions of type 1 | Largest safe count of type 2 (exact) | Largest count of type 2 (S1 bound) |
|---|---|---|
| 0 | 7 | 0 |
| 1 | 5 | 0 |
| 2 | 3 | 0 |
| 3 | 1 | 0 |

  - At rate `b*` the type-2 shape 10 has mean 1.0 = `tau`, so the bound forbids type 2 altogether.
  - With at most 15 missions, 20 pairs `(n1, n2)` are safe and the bound accepts 4 of them.
  - The cost of the bound grows with the spread of the rates. With one rate the bound is exact.
    S5 is less conservative when the rates differ widely (10 pairs here), and it also allows repair.

## S2. `main`: the `repeated_increment` calibration with two increment types

- **The method, as coded.** It is the default (main:src/fleet_management/config.py:79, 374-375). For a
  Gamma-only fleet it runs when `gamma_beta_bound` is given (main:src/fleet_management/solver.py:107).
  - Types: distinct pairs (mean, rate); each may repeat up to `min(T, occurrences)` times
    (main:src/fleet_management/degradation_model/gamma_utils/gamma_repeated_calibration.py:369-386, 429).
  - `b*` = the given `gamma_beta_bound`, or the smallest rate (same file, 411-415).
  - `m*` = the largest count `m` with `P(Gamma(m·a_q, b_q) > tau) <= eps` (same file, 196-232).
  - `a'_q = max over m = 1..m* of A_m/m`, where `A_m` is the common-rate shape with the same tail
    at `tau` as `m` copies (same file, 275-287, 498-528).
  - The model adds the `a'_q` of the chosen missions to `A'` and imposes `A' <= A_max`,
    `Q(A_max, b*·tau) = eps` (main:src/fleet_management/degradation_model/base.py:606-610, 899-906, 1026-1046).
- **Statement.** The surrogate is not safe. A schedule can satisfy `A' <= A_max` on every day while
  the true `P(D > tau)` is far above `eps`. This holds for mixes of two types, even when each count
  stays within its own `m*`, and for one type repeated more than `m*` times.
- **Cause.** `a'_q` matches the tail of `m` copies at `tau` only. A type with a high rate has a tiny
  tail at `tau` after one copy, so `a'_q` is tiny. Example: a mission of mean 0.4 and rate 50 gets
  `a' = 0.089` at `b* = 10`, which stands for a mean of 0.009. Nothing in the model bounds the count
  of a type by its `m*`. In a mix, each type only has to exceed `tau` minus the damage of the other
  types, a lower level where its calibrated tail is too small.
- **Counterexample A: a mix within `m*`.** `tau = 1`, `eps = 0.01`, new part, no repair. Type 1: mean
  0.1, rate 10 (shape 1). Type 2: mean 0.1, rate 200 (shape 20). Then `b* = 10`, `a' = (1.0, 0.3287)`,
  `m* = (3, 8)`, `A_max = 3.9717`.

| Missions (type 1, type 2) | `A'` | Accepted | True `P(D > tau)` |
|---|---|---|---|
| (3, 0) | 3.0000 | yes | 0.0028 |
| (0, 8) | 2.6292 | yes | 0.0016 |
| (3, 2) | 3.6573 | yes | **0.0142** (Monte Carlo, 500,000 draws: 0.0141 ± 0.0003) |
| (2, 4) | 3.3146 | yes | **0.0187** |
| (1, 7) | 3.3006 | yes | **0.0597** |

- **Counterexample B: one type beyond `m*`.** Same `tau`, `eps` and type 1; type 2 now has mean 0.4
  and rate 50 (shape 20). Then `a'_2 = 0.0891` and `m*_2 = 1`. Three type-2 missions give
  `A' = 0.267 <= 3.97`: accepted, with true mean 1.2 and true tail 0.908. Five give tail 1.000.
- **Search** (`s2_main_calibration.py`, `s2b_mix_within_mstar.py`). `tau = 1`, `T = 20`,
  `eps` in {0.01, 0.1}, `b*` in {2, 10, 50}, rate ratio `b2/b1` in {2, 5, 20}, means in
  {0.02, 0.05, 0.1, 0.2, 0.4}, `b* = b1`.
  - Any counts with `n1 + n2 <= 20`: an accepted unsafe schedule exists in 360 of 441 sets.
  - Counts within `m*`: in 300 of 450 sets. Worst true tail 0.9986 at `eps = 0.01`.
  - One type with a user-chosen `b* < b`: `m* + 1` copies are accepted in 27 of 90 sets, with true
    tails from 0.0103 to 0.49.
- `GAMMA_DIAGNOSTICS.md` states that the guarantee covers repetitions of one type only
  (main:src/fleet_management/degradation_model/gamma_utils/GAMMA_DIAGNOSTICS.md:59-63). The model does
  not enforce even that scope, because `n_q <= m*_q` is not a row.

## S3. ARD-infinity repair in parameter space

- **Code.** Both branches map the shape `A` to `(1-rho)·A` and keep the rate
  (lore:src/fleet_management/models/gamma.py:116-125;
  main:src/fleet_management/degradation_model/base.py:843-851, 898-906). With one rate `b`, the
  surrogate after a repair is `S = Gamma((1-rho)·A, b)`. The pathwise law is
  `T = (1-rho)·X = Gamma(A, b/(1-rho))`, with `X ~ Gamma(A, b)` the state before the repair.
- **Statement 1 (orders).** `E T = E S` and `T <=cx S`. Neither `T <=st S` nor `S <=st T` holds.
  The convex order survives the addition of independent increments, scaling and further repairs.
  - Proof of `T <=cx S`. Take independent `U ~ Gamma((1-rho)A, b)` and `V ~ Gamma(rho·A, b)`, and
    `X = U + V`. The change of variables `(u, v) -> (x = u + v, w = u/x)` splits the joint density into
    a function of `x` times a function of `w`. So `W = U/X ~ Beta((1-rho)A, rho·A)` is independent of
    `X`, and `E[U | X] = (1-rho)·X = T`. Jensen's inequality for conditional expectations gives
    `E f(T) <= E f(U)` for every convex `f`, and `U` has the law of `S`.
  - No usual order: `X <=st Y` with `E X = E Y` forces equal laws. Here the variances differ:
    `(1-rho)²A/b²` against `(1-rho)A/b²`.
  - Survival: for convex `f` and an independent `Z`, `f(· + z)` is convex for each `z`; integrate over `z`.
- **Statement 2 (repair day).** On the repair day the true tail is at most the tail checked the day
  before, because `(1-rho)·X <= X` on every path. So the repair day itself is safe.
- **Statement 3 (later days).** The surrogate can understate a later tail, and both branches can accept
  an unsafe schedule. Counterexample (`s3_ardinf.py`): `tau = 1`, `eps = 0.01`, `rho = 0.5`, one rate
  `b = 0.01` (`lore`: `beta = 100`). Then `A_hat = 0.0024877`; the mean at the cap is `0.249·tau`, so
  `lore`'s cap `alpha <= tau/beta` holds.
  - Schedule on a new part: day 1, a mission of shape `A_hat`; day 2, a repair; day 3, a mission of
    shape `0.5·A_hat`. Tracked shapes: `A_hat`, `0.5·A_hat`, `A_hat`. Every row holds.
  - True tails: day 1 0.0100; day 2 0.0083; day 3 **0.01329** (Monte Carlo, 10⁶ draws: 0.0135 ± 0.0002).
  - `main` accepts the same schedule: with one rate its calibration is exact, `a' = a` and
    `A_max = A_hat` (`s3c_main_same_example.py`).
- **Where it happens** (`s3_ardinf.py`, `s3b_periodic.py`). Ratio of the true day-3 tail to `eps`,
  same schedule, `eps = 0.01`. With `eps = 0.1` (larger `A_hat`) each largest ratio moves by at most 0.08.

| `b·tau` | 0.001 | 0.01 | 0.1 | 0.3 | 1 | 10 | 100 |
|---|---|---|---|---|---|---|---|
| `A_hat` | 0.0016 | 0.0025 | 0.0055 | 0.011 | 0.044 | 3.97 | 78 |
| largest ratio over `rho` in {0.1, 0.3, 0.5, 0.8, 0.95} | 1.54 | 1.41 | 1.17 | 1.03 | 0.95 | 0.92 | 0.88 |

  - Periodic schedule (repair, then a mission of shape `rho·A_hat`, repeated): the excess grows with
    the cycles. At `b·tau = 0.01`, `rho = 0.5`: 1.33, 1.58, 1.75, 1.84 times `eps` after 1 to 4 cycles.
  - The unsafe cases have `A_hat <= 0.11`. Such a Gamma law has most mass near 0 and rare large
    values (coefficient of variation `1/sqrt(A) >= 3`).
  - **Conjecture** (not proven): with `b·tau >= 1` the parameter-space ARD∞ repair never accepts an
    unsafe schedule. Evidence: no ratio above 0.98 for `b·tau` in {1, 3, 10, 30, 100}, both `eps`, in
    both schedule families. These two families do not cover all schedules.

## S4. ARD1 repair in parameter space

- **Code.** The shape becomes `A - rho·(A - g)`, with `g` the anchor shape
  (main:src/fleet_management/degradation_model/base.py:915-946, 1026-1042; `lore`'s `ARA1` at
  lore:src/fleet_management/models/gamma.py:106-115 has the anchor defect that the merge has already fixed).
  The pathwise law is `G + (1-rho)·Δ`: `G` is the state just after the last intervention, `Δ` the
  damage since then, independent of `G`.
- **Statement 1 (orders).** With one rate and `G ~ Gamma(g, b)`, the true post-repair state is `<=cx`
  the surrogate `Gamma(g + (1-rho)(A - g), b)`. Proof: S3, Statement 1, applied to `Δ ~ Gamma(A - g, b)`,
  then add the independent `G` to both sides. As in S3, no usual order holds, and the convex order
  survives later increments and repairs.
- **Statement 2 (repair day).** Safe: `G + (1-rho)·Δ <= G + Δ` on every path.
- **Statement 3 (later days).** Not safe. With `g = 0` (a new part) both laws coincide on the first
  repair, so the S3 counterexample applies unchanged. With a nonzero anchor (`s4_ard1.py`):
  - `tau = 1`, `eps = 0.01`, `rho = 0.5`, `b = 0.01`. Initial state `Gamma(g, b)` with
    `g = 0.25·A_hat = 0.000622`; the anchor starts at `g`, the merge default.
  - Day 1, a mission of shape `d = 0.75·A_hat`; day 2, a repair; day 3, a mission of shape `0.5·d`.
    Tracked: `A_hat`, `g + 0.5·d`, `A_hat`. Every row holds.
  - True day-3 tail **0.01247** = 1.25·`eps` (Monte Carlo, 10⁶ draws: 0.0124 ± 0.0002).
- **Where.** A larger anchor shrinks the excess. At `b·tau = 0.01`, `eps = 0.01`, the largest ratio
  over `rho` is 1.41, 1.31 and 1.21 for `g/A_hat` = 0, 0.25 and 0.5. Unsafe cases again need
  `b·tau <= 0.3`; for `b·tau` in {1, 10} the largest ratio is 0.99.

## S5. A safe Gamma formulation for a MILP

- **Proposal (Claude's own).** Track `A' = b*·mu`, where `mu` is the tracked mean (linear dynamics, both
  repair laws, one-sided rows) and `b*` is the smallest rate of all increments, the initial state and the
  replacement state. Row: `A' <= A_icx`, a constant computed offline from `B(A_icx) = eps`, where for
  `Γ_A ~ Gamma(A, b*)`:
  - `B(A) = min( inf over x < tau of E[(Γ_A - x)^+]/(tau - x) , exp(-A·(u - 1 - ln u)) )`, `u = b*·tau/A`;
  - the first term is a stop-loss bound, the second the Chernoff bound (used when `u > 1`);
  - `lore` (one rate): the dynamics stay as they are, and only the cap `A_hat` becomes `A_icx`;
  - `main`: the calibration and the bounding shape go; `A' = b*·mu` replaces them.
- **Theorem.** For every schedule that the rows accept, on every day of every repetition,
  `D_k <=icx Gamma(A'_k, b*)`, so `P(D_k > tau) <= B(A'_k) <= eps`.
- **Proof.**
  - L1 (S3, Statement 1): `c·X <=cx Gamma(c·a, b)` for `X ~ Gamma(a, b)` and `c` in (0, 1].
  - L2: an increment `Gamma(a_q, b_q)` equals `(b*/b_q)·Gamma(a_q, b*)`, so by L1 it is
    `<=cx Gamma(b*·m_q, b*)`, with `m_q = a_q/b_q` its mean. The same holds for the initial and replacement states.
  - L3: `<=icx` is kept by sums of independent terms, by scaling with `c >= 0`, and by a larger shape
    (`Gamma(A, b) <=st Gamma(A2, b)` when `A <= A2`).
  - Induction on the days of the exact tracked recursion `(A*, g*)`. Claim: `D_k = G_k + Δ_k`, independent,
    with `G_k <=icx Gamma(g*_k, b*)` and `Δ_k <=icx Gamma(A*_k - g*_k, b*)`. A mission: L2 and L3. ARD∞:
    `(1-rho)·D <=icx Gamma((1-rho)·A*, b*)` by L3 and L1. ARD1: `G + (1-rho)·Δ <=icx
    Gamma(g* + (1-rho)(A* - g*), b*)` by L1 on `Δ` and L3. Replacement: L2.
  - One-sided rows: the tracked maps increase in `(A', g')`, so `A'_k >= A*_k`, and L3 applies. The loop row
    extends this to every repetition, by the argument of the decided encoding.
  - Tail: take `f(y) = (y - x)^+`, then `(tau - x)·P(D > tau) <= E(D - x)^+ <= E(Γ - x)^+`; take
    `f(y) = e^{s·y}` for Chernoff. `B` increases in `A`, so `A' <= A_icx` gives `P(D > tau) <= eps`.
- **Checks** (`s5_icx_proposal.py`; a check of the proof, not a proof):
  - Monte Carlo on 12 random schedules (both laws, repairs, replacements, two types with rates 2 to 50),
    12 days, 20,000 paths: the true tail never exceeds `B(A')` (largest MC − 3 s.e. − `B`: −0.0002);
    the true stop-loss never exceeds the surrogate's (largest MC − 3 s.e. − surrogate: negative, `> -5e-5`).
  - The row rejects every counterexample above: S3, `A' = 0.0025 > A_icx = 0.0001`; S2 A,
    `A' = 5.0 > 3.20`; S2 B, `A' = 8.0 > 3.20`.
- **Price** (`s5_icx_proposal.py`). Allowed shape relative to the exact cap, one rate, no repair:

| `b*·tau` | 0.01 | 0.1 | 1 | 10 | 100 | 1000 |
|---|---|---|---|---|---|---|
| `A_icx/A_hat`, `eps = 0.01` | 0.04 | 0.18 | 0.35 | 0.81 | 0.96 | 0.99 |
| `A_icx/A_hat`, `eps = 0.1` | 0.04 | 0.18 | 0.40 | 0.80 | 0.95 | 0.98 |
| Cantelli on mean and variance, `eps = 0.01` | 0.00 | 0.02 | 0.23 | 0.21 | 0.49 | 0.79 |

  - For `b*·tau >= 10`, the exact Gamma tail at `A_icx` is about `0.38·eps`: there the stop-loss bound
    is about 2.6 times the Gamma tail.
  - For `b*·tau <= 0.1` the bound reduces to Markov's, `mu <= eps·tau`. The price is high exactly where
    the parameter-space repair fails (S3).
  - Several rates add the cost of mean matching. In the S1 example the row accepts 10 of the 20 safe
    pairs (largest `n2` for `n1` = 0..3: 3, 2, 1, 0, against 7, 5, 3, 1); the S1 bound accepts 4.
- **Alternatives**, with trade-offs:
  - No parameter-space repair (replacement only): exact with one rate (`lore`'s cap `A_hat`), safe by S1
    with several rates. Imperfect repair is lost.
  - The parameter-space repair with the exact cap, limited by an input check to `b·tau >= 1`: rests on
    the S3 conjecture only.
  - Two tracked shapes (Claude's own): the un-repaired shape sum `A_u` (by S1, `D <=st Gamma(A_u, b*)`,
    reset only by replacement) and `A'`. Row: `A_u <= A_max` or `A' <= A_icx`, with one binary per day.
    Safe and tighter, but the disjunction adds binaries, and a day that uses `A_u` needs the loop row on
    `A_u`, so a replacement in every period (as the Wiener variance on `lore`).
  - Chernoff at one fixed `s`, tracking the exact cumulant `Σ -a_q·ln(1 - s/b_q)` as `main`'s rainflow
    Chernoff does (ARD∞ `K+ = (1-rho)·K`, ARD1 `K+ = (1-rho)·K + rho·gK`): safe, no mean matching, but no
    minimum over `s` and no stop-loss term.
  - Distribution-free moments, as rainflow's Cantelli: safe, but the most conservative (table).

## Scripts

Folder `C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad\gamma\`
(session scratchpad: it is deleted when the session ends unless copied). Each script runs with
`python -I`, standard library only; its output is in the `*_output.txt` file of the same name.

| Script | Content |
|---|---|
| `gamma_lib.py`, `s0_lib_check.py` | incomplete gamma, the sum formula, bounds; self-check |
| `main_calibration.py` | replica of `main`'s `repeated_increment` calibration |
| `s1_common_rate.py` | S1 |
| `s2_main_calibration.py`, `s2b_mix_within_mstar.py` | S2 |
| `s3_ardinf.py`, `s3b_periodic.py`, `s3c_main_same_example.py` | S3 |
| `s4_ard1.py` | S4 |
| `s5_icx_proposal.py` | S5 |
