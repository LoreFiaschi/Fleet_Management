Status: done (30 Sep 2026). All seven questions answered. Next: none; open checks are listed under "Evidence basis".

# B. Degradation models: `lore` versus `main`

Scope: degradation-model modules only. Merge base: 462ac1d (25 Jun 2026).
Evidence format: `branch:path:line`. "Read in part" and UNVERIFIED mark claims not read in full.

## Q1. Models that exist and are reachable from the public `solve()`

| Model | `lore` | `main` |
|---|---|---|
| `gaussian` | reachable, new module (lore:src/fleet_management/solver.py:36-45) | accepted by the schema (main:src/fleet_management/config.py:34), but no cell builder: `NotImplementedError` (main:src/fleet_management/degradation_model/base.py:149-156). The file is the unchanged merge-base code. |
| `inverse_gaussian` | reachable, new module | as `gaussian` on `main` |
| `wiener` | reachable, new module | **absent**: `git grep -i wiener main -- src README.md` returns nothing |
| `gamma` | reachable, one formulation | reachable, two routes: modular (base.py block, registered at main:src/fleet_management/degradation_model/base.py:1187) and legacy constant-rate (main:src/fleet_management/solver.py:101-128) |
| `rainflow` | reachable (Cantelli, Bernstein) | reachable (five bounds), registered at main:src/fleet_management/degradation_model/rainflow.py:695 |

- `main` routing: Gamma-only fleet goes to the modular route if `gamma_beta_bound` is given, else to the legacy route; every other fleet goes to `base.solve_mixed` (main:src/fleet_management/solver.py:70-90).
- The README claim that Gaussian and IG are "Reserved" and raise `NotImplementedError` is correct (main:README.md:16).
- Change since the merge base 462ac1d: only `gaussian.py` and `inverse_gaussian.py` existed there, at package level (`git ls-tree 462ac1d -- src`). `lore` rewrote both and added Wiener, Gamma and rainflow under `models/`. `main` moved the two files unchanged and added Gamma and rainflow under `degradation_model/`. So every difference in this report comes from independent additions, except the Gaussian and IG rewrite, which is a `lore` change.

## Q2. State variables, dynamics, maintenance types, replacement

Acronyms. ARD: Arithmetic Reduction of Degradation. ARA: Arithmetic Reduction of Age. The suffix is the memory order (1 or infinite, "inf").

**The two branches use the same label for different repair operators.**

| Operator (`s` = state before repair) | `lore` label | `main` label |
|---|---|---|
| `s+ = (1-rho)*s` (whole state reduced) | `ARD1` | `ardinf` |
| `s+ = s - rho*(s - a)`, `a` = state at the last intervention | `ARA1` (`a` = pre-repair state, initial `a = mu_0`) | `ard1` (`a` = post-intervention latch, initial `a = 0`) |

Evidence: lore:src/fleet_management/maintenance/ard1.py:39-42; lore:src/fleet_management/maintenance/ara1.py:39-44, 92-95; main:src/fleet_management/degradation_model/base.py:776-778, 1017-1042; main:src/fleet_management/degradation_model/rainflow.py (see Q5).
Interpretation: a merge must map `lore ARD1 -> main ardinf`, and cannot map `lore ARA1 -> main ard1` without changing the anchor rule.

### 2a. `lore` (all five modules read in full)

Common facts on `lore`:
- ARD1 rows: `s[k] >= s[k-1] + delta - M*x_m - M*x_r` and `s[k] >= f*s[k-1] - M*(1-x_m)`, one-sided (lore:src/fleet_management/maintenance/ard1.py:35-42).
- ARA1 rows: the same accumulate row, and `s[k] >= s[k-1] - rho*(s[k-1] - anchor[k-1]) - M*(1-x_m)` (lore:src/fleet_management/maintenance/ara1.py:88-95).
- The ARA1 anchor snaps to the pre-repair state on a repair day and to the reset value on a replacement day (lore:src/fleet_management/maintenance/ara1.py:28-56).
- Replacement: two-sided big-M reset `s[k] = s_new` when `x_r = 1` (lore:src/fleet_management/models/base.py:67-70).
- Loop constraint on every tracked state: `s[2H-1] <= s[H-1]` (lore:src/fleet_management/models/base.py:91-100).
- Repair cost `z = rho*(s_prev - anchor_prev)*x_m` by a McCormick envelope, anchor 0 for ARD1 (lore:src/fleet_management/models/base.py:73-88).

| Model | Tracked states | Dynamics | Maintenance | Replacement | Evidence |
|---|---|---|---|---|---|
| `gaussian` | `mu`, `v` | ARD1 on both; `v` repair factor `(1-rho)^2`; increments `mu_inc`, `v_inc` per mission | ARD1 only (validation error otherwise) | resets `mu_new`, `v_new` | lore:src/fleet_management/models/gaussian.py:25-30, 73-82 |
| `inverse_gaussian` | `mu` | ARD1 | ARD1 only | resets `mu_new` | lore:src/fleet_management/models/inverse_gaussian.py:23-27, 95-99 |
| `wiener` | `mu`, `v`, anchor `mu_last` if ARA1 | `mu`: ARD1 or ARA1. `v`: `v[k] >= v[k-1] + sigma^2 * (number of missions on day k) - V_max*x_r`; no repair term | ARD1, ARA1 | resets `mu_new`, `v_new` (the only way `v` falls) | lore:src/fleet_management/models/wiener.py:74-97 |
| `gamma` | shape `alpha`, anchor `alpha_last` if ARA1 | `alpha`: ARD1 or ARA1 with increments `alpha_inc`; `alpha_0 = mu_0/beta`; `ctx.mu = alpha*beta` (an expression) | ARD1, ARA1 | resets `alpha_new = mu_new/beta` | lore:src/fleet_management/models/gamma.py:92-137 |
| `rainflow` | `mu`, `v` | as Gaussian; variance big-M is `V_max^RF` | ARD1 only | resets `mu_new`, `v_new` | lore:src/fleet_management/models/rainflow.py:44-49, 144-153 |

- No ARDinf on `lore`: `git grep -i ardinf lore -- src` returns nothing.
- Maintenance type input is `maintenance_type`, values `ARD1` or `ARA1` (lore:src/fleet_management/solver.py:181-189).
- **Possible defect in `lore` ARA1 (interpretation, derived from the code, not tested with Gurobi).**
  - The repair-cost row `z <= rho*(s[k-1] - a[k-1])` has no `x_m` term, and `z >= 0` (lore:src/fleet_management/models/base.py:85-86; lore:src/fleet_management/solver.py:523). So every feasible point has `s[k-1] >= a[k-1]` for all `k`.
  - After a repair on day `k1`, the anchor is `a[k1] = s[k1-1]`, the pre-repair state (lore:src/fleet_management/maintenance/ara1.py:39-44).
  - At `k = k1+1` this forces `s[k1] >= s[k1-1]`. The repair on day `k1` then lowers nothing, but it still costs `C_R*z`. The only exception is a repair on the last day `2H-1`, where no later row applies.
  - Example: `s[k1-1] = 0.6`, `a = 0.2`, `rho = 0.5`. The repair row allows `s[k1] = 0.4`, but the next day's `z` row needs `s[k1] >= 0.6`.
  - The `lore` spec prose defines the anchor as the state "immediately after" the last maintenance (lore:spec/spec.tex:135-136). The `lore` spec equations and the code use the pre-repair state (lore:spec/spec.tex:1018-1019). With the post-repair anchor, as on `main`, the gap is 0 right after a repair and the problem disappears.
  - Scope: Wiener ARA1 and Gamma ARA1 on `lore`.

### 2b. `main` (public path read in full; `base.py` Gamma block lines 164-1187 read in full)

Common facts on `main`:
- `main` changed neither `gaussian.py` nor `inverse_gaussian.py` since the merge base. Both are byte-identical to 462ac1d versions, only moved into `degradation_model/` (`git diff 462ac1d:src/fleet_management/gaussian.py main:src/fleet_management/degradation_model/gaussian.py` is empty; same for IG).
- Maintenance selector is `repair_model`, values `ard1` (default) or `ardinf` (main:src/fleet_management/config.py:36, 354). There is no ARA1 on `main`.
- Depot gating: `idle + m + r = x[i,0,k]` per cell (main:src/fleet_management/degradation_model/base.py:1697-1725).
- Horizon: `T = H1 + H2` (transitory, then operating); the loop row is `state[T-1] <= state[H1-1]` (main:src/fleet_management/degradation_model/base.py:1055-1063).

| Model | Tracked states | Dynamics | Maintenance | Replacement | Evidence |
|---|---|---|---|---|---|
| `gaussian` (old code, not reachable) | `mu`, `v` | `mu[k] >= mu[k-1] + sum x*mu_inc - alpha*x0`, `mu[k] >= (1-xi)*mu[k-1]`; same for `v` with factor `(1-xi)` | whole-state reduction by `xi` (ARD1 in `lore` terms) | none | main:src/fleet_management/degradation_model/gaussian.py:273-313 |
| `inverse_gaussian` (old code, not reachable) | `mu` | as old Gaussian, `mu` only | as above | none | main:src/fleet_management/degradation_model/inverse_gaussian.py:210-238 |
| `gamma`, modular | physical mean `mu`, bounding shape `A'`; for `ard1` also latches `g_mu`, `g_A` | exact equality balances with binary-product hulls, no big-M: `ardinf`: `mu = mu_prev + inc - rho*mu_prev*m`, same for `A'`. `ard1`: removes `rho*(state - latch)*m`; latch becomes the post-repair state | `ard1`, `ardinf` | replaces state and latch by seed values `mu_new`, `A'_new` (products with `r`) | main:src/fleet_management/degradation_model/base.py:812-1047 |
| `gamma`, legacy constant-rate | shape `A` | see Q6 | see Q6 | see Q6 | main:src/fleet_management/degradation_model/legacy/gamma_gurobi.py |
| `rainflow` | see Q5 | see Q5 | `ard1`, `ardinf` | see Q5 | main:src/fleet_management/degradation_model/rainflow.py |

- `main` Gamma uses the shape-rate convention: `E[D] = A/beta`, `Var[D] = A/beta^2` (main:src/fleet_management/degradation_model/gamma.py:3-12). `lore` uses shape-scale: `mu = alpha*beta` (lore:src/fleet_management/models/gamma.py:3-5). **The same input name `beta` means a rate on `main` and a scale on `lore`.**
- `main` Gamma tracks `mu` and `A'` separately, because `A'/beta_bar` need not equal the physical mean (main:src/fleet_management/degradation_model/base.py:1053-1054). `lore` tracks only `alpha` and derives `mu`.
- `main` Gamma variable bounds come from an offline reachability recursion, `_gamma_reachable_upper_bounds`, not from big-M constants (main:src/fleet_management/degradation_model/base.py:235-341, 612-632).
- The `main` repair approximation keeps the rate and scales the shape by `(1-rho)`. This keeps the mean but not the variance of the exact repaired variable (main:src/fleet_management/degradation_model/gamma.py:27-35, 222-246).

## Q3. Reliability constraint per model

### 3a. `lore`

Notation: `z_eps = Phi^{-1}(1-eps)`, where `Phi` is the standard normal cumulative distribution function (CDF).
`formulation` selects `exact` (default) or `lp` (lore:src/fleet_management/solver.py:147).

| Model | `exact` form | `lp` form | Evidence |
|---|---|---|---|
| `gaussian` | `mu <= tau` and `z_eps^2 * v <= (tau - mu)^2` (nonconvex quadratic) | tangent line `z_eps^2 * v + 2*tau*mu <= tau^2` | lore:src/fleet_management/models/gaussian.py:84-94 |
| `inverse_gaussian` | linear cap `mu <= mu_bar` in both forms | same | lore:src/fleet_management/models/inverse_gaussian.py:100 |
| `wiener` | as Gaussian | as Gaussian | lore:src/fleet_management/models/wiener.py:99-109 |
| `gamma` | linear cap `alpha <= alpha_hat` in both forms | same | lore:src/fleet_management/models/gamma.py:125 |
| `rainflow`, Cantelli | `mu <= tau`, `((1-eps)/eps) * v <= (tau - mu)^2` | `((1-eps)/eps) * v + 2*tau*mu <= tau^2` | lore:src/fleet_management/models/rainflow.py:155-166 |
| `rainflow`, Bernstein | `mu <= tau - c`, `(tau - c - mu)^2 >= c^2 + 2*kappa*v` | `2*(tau-c)*mu + 2*kappa*v <= tau*(tau - 2c)` | lore:src/fleet_management/models/rainflow.py:167-180 |

Precomputed caps and constants on `lore`:
- IG cap `mu_bar`: root in `(0, tau)` of `Phi(a) + exp(2/eta)*Phi(-b) = 1 - eps`, with `a = r*(tau/mu - 1)`, `b = r*(tau/mu + 1)`, `r = sqrt(mu/(eta*tau))`. Solved by `brentq` in log space. The cap is `tau` if non-binding; an error is raised if unachievable (lore:src/fleet_management/models/inverse_gaussian.py:38-75).
- Gamma cap `alpha_hat`: root of `Q(alpha, tau/beta) = eps`, where `Q` is the regularised upper incomplete gamma function (`gammaincc`). This is `Pr(D > tau)` for `D ~ Gamma(shape alpha, scale beta)`. Cap at `tau/beta` with a warning if non-binding (lore:src/fleet_management/models/gamma.py:45-82).
- Rainflow constants: `kappa = log(1/eps)`; `b = max(b_0, b_new, max b_inc)`; `c = kappa*b/3` (lore:src/fleet_management/models/rainflow.py:26-28, 116-122).
- Rainflow variance big-M: Cantelli `tau^2*eps/(1-eps)`; Bernstein `tau*(tau-2c)/(2*kappa)` (lore:src/fleet_management/models/rainflow.py:123-126).
- Optional user cap `v[k] <= v_max_user` for Gaussian, Wiener and rainflow (lore:src/fleet_management/models/gaussian.py:96-97; wiener.py:111-112; rainflow.py:182-183).
- `NonConvex = 2` only if `formulation == "exact"` and some component is Gaussian, Wiener or rainflow (lore:src/fleet_management/solver.py:515-518).

### 3b. `main`, Gaussian and IG (old merge-base code, not reachable)

| Model | Reliability row | Evidence |
|---|---|---|
| `gaussian` | `v[k-1]*(z_eps^2 - 9) + 2*mu[k-1] - 3*alpha*x0 <= alpha + sum_j x_j*W_j`, with `W = mu_inc^2 + 2*mu_inc - v_inc*z_eps^2`; `alpha` is the threshold | main:src/fleet_management/degradation_model/gaussian.py:94-117, 260-271 |
| `inverse_gaussian` | `mu <= alpha - sqrt(alpha/c)*z_eps` (a normal approximation, not the IG CDF) | main:src/fleet_management/degradation_model/inverse_gaussian.py:138-144, 198-200 |

- Interpretation: the `lore` IG cap uses the exact IG CDF; the old `main` cap does not. The two caps differ numerically.

### 3c. `main`, Gamma

Modular route (used when `gamma_beta_bound` is given, or in any fleet that mixes Gamma with rainflow; main:src/fleet_management/solver.py:81-110):
- Row: `A'[k] <= A_max`, where `A_max` solves `P(Gamma(A_max, rate beta*) > tau) = eps` (main:src/fleet_management/degradation_model/base.py:606-610, 903-906, 1043-1046; main:src/fleet_management/degradation_model/gamma_utils/gamma_repeated_calibration.py:498-528).
- The mission shape increments of `A'` are calibrated offline. The default calibration is `repeated_increment` (main:src/fleet_management/config.py:79, 374-376; main:src/fleet_management/degradation_model/base.py:580-585).
- Repeated-increment calibration, per distinct pair (mean `mu_q`, rate `beta_q`), exact shape `alpha_q = mu_q*beta_q` (main:src/fleet_management/degradation_model/gamma_utils/gamma_repeated_calibration.py:196-330, 369-435):
  - common rate `beta* = min` of all exact rates, or `gamma_beta_bound` if given (must be `<=` that minimum);
  - safe count `m_q* = max{m <= n_q : P(Gamma(m*alpha_q, beta_q) > tau) <= eps}`, with `n_q` the number of occurrences of the pair in the profile;
  - bounding shape `alpha_q* = max over m = 1..max(1, m_q*)` of `A_req(m)/m`, where `P(Gamma(A_req(m), beta*) > tau) = P(Gamma(m*alpha_q, beta_q) > tau)`.
  - Initial and replacement states are calibrated as single seeds (count 1).
- Scope of the guarantee, as stated by `main`: repetitions of one increment type only, one threshold, finite horizon. It is not full stochastic dominance (main:src/fleet_management/degradation_model/gamma_utils/GAMMA_DIAGNOSTICS.md:59-63; main:src/fleet_management/degradation_model/base.py:1118-1121).
- Interpretation: a schedule that mixes different increment types is not covered by these checks. Tail dominance for such sums is not proven by this code.
- Alternative calibration `finite_count` (`gamma_calibration_method: finite_count`): a linear program (LP) that minimises `w @ A'` subject to `c @ A' >= A_req(c)` for every enumerated count vector `c`, with `0 <= A' <= A`. Exact tails of mixed sums use the Moschopoulos (1985) series with a truncation upper bound (main:src/fleet_management/degradation_model/gamma_utils/gamma_tail_bound.py:754-925, 928-1007). The module calls itself legacy (main:src/fleet_management/degradation_model/gamma_utils/gamma_tail_bound.py:1-4).

Legacy constant-rate route (Gamma-only fleet without `gamma_beta_bound`; main:src/fleet_management/solver.py:112-128):
- One rate per component; row `A <= A_max(beta_l, tau_l, eps)` from `maximum_reliable_shape` (main:src/fleet_management/degradation_model/legacy/gamma_gurobi.py:158-172, 277-281; main:src/fleet_management/degradation_model/gamma.py:392-488).
- This is the same form as the `lore` Gamma cap (`alpha <= alpha_hat`), but with a rate instead of a scale.

Comparison with `lore`: `lore` has one rate (scale) per component and no calibration step, so its cap is exact for that model (lore:src/fleet_management/models/gamma.py:45-82). It has no counterpart to `gamma_beta_bound`, time-varying rates, or seed calibration.

## Q4. Model-specific inputs and their names

### 4a. `lore` (input keys read in `_parse_and_validate`)

| Input key | Used by | Shape | Evidence |
|---|---|---|---|
| `model` | all; values `gaussian`, `inverse_gaussian`, `wiener`, `gamma`, `rainflow` | `(F, L)` | lore:src/fleet_management/solver.py:36, 172-179 |
| `maintenance_type` | all; `ARD1` or `ARA1` | `(F, L)` | lore:src/fleet_management/solver.py:181-189 |
| `mu` (mean increment) | all except Gamma | `(F, M, L, H)` | lore:src/fleet_management/solver.py:217-225 |
| `mu_0`, `mu_new` | all (Gamma divides by `beta`) | `(F, L)` | lore:src/fleet_management/solver.py:201-202 |
| `v_0`, `v_new`, `v_max_user` | Gaussian, Wiener, rainflow | `(F, L)` | lore:src/fleet_management/solver.py:244-251 |
| `v` (variance increment) | Gaussian, rainflow | `(F, M, L, H)` | lore:src/fleet_management/solver.py:265-272 |
| `eta` | IG | `(F, L)` | lore:src/fleet_management/solver.py:275-279 |
| `sigma` | Wiener | `(F, L)` | lore:src/fleet_management/solver.py:285-287 |
| `beta`, `alpha_inc` | Gamma | `(F, L)`, `(F, M, L, H)` | lore:src/fleet_management/solver.py:293-301 |
| `tail_bound` | rainflow; `cantelli` (default) or `bernstein` | `(F, L)` | lore:src/fleet_management/solver.py:305 |
| `b_inc`, `b_0`, `b_new` | rainflow with Bernstein | `(F, M, L, H)`, `(F, L)` | lore:src/fleet_management/solver.py:308-321 |
| `epsilon`, `tau`, `rho`, `formulation` | all | scalar, `(F, L)` | lore:src/fleet_management/solver.py:134, 147, 194, 197 |

### 4b. `main` (keys read by `load_config`; main:src/fleet_management/config.py:349-406)

| `main` key | Used by | `lore` counterpart | Evidence |
|---|---|---|---|
| `model` (`rainflow`, `gamma`, `gaussian`, `inverse_gaussian`) | all | `model` (adds `wiener`) | main:src/fleet_management/config.py:34, 324-330 |
| `repair_model` (`ard1` default, `ardinf`) | rainflow, Gamma | `maintenance_type` (`ARD1`, `ARA1`); labels do not map one to one (see Q2) | main:src/fleet_management/config.py:354 |
| `bound_method` (alias `method`; default `cantelli`) | rainflow | `tail_bound` (default `cantelli`) | main:src/fleet_management/config.py:352-353 |
| `rho` (aliases `xi`, `repair_rho`) | all | `rho` | main:src/fleet_management/config.py:357 |
| `epsilon`, per cell `(F, L)`, in `(0, 1)` | all | `epsilon`, one scalar, in `(0, 0.01]` | main:src/fleet_management/config.py:356, 470-471; lore:src/fleet_management/solver.py:134-136 |
| `mu` (required, `> 0`), `mu_trans` | all, including Gamma | `mu` (not for Gamma) | main:src/fleet_management/config.py:377, 381, 474-475 |
| `replacement_mu`, `replacement_v` (aliases `mu_new`, `v_new`) | all | `mu_new`, `v_new` | main:src/fleet_management/config.py:360-363 |
| `v_0` (default 0), `v`, `v_trans` | rainflow | `v_0`, `v` | main:src/fleet_management/config.py:359, 378, 382 |
| `support`, `support_trans` | Hoeffding, Bernstein | `b_inc`, `b_0`, `b_new` (Bernstein) | main:src/fleet_management/config.py:379, 383 |
| `cgf`, `cgf_trans`, `s_chernoff` | Chernoff | none | main:src/fleet_management/config.py:364, 380, 384 |
| `gamma_beta` (a **rate**, profile `(F, L, M, H)`), `gamma_beta_trans` | Gamma | `beta` (a **scale**, `(F, L)`) | main:src/fleet_management/config.py:71-72, 365-368 |
| `gamma_beta_0`, `gamma_beta_new` (required if `mu_0 > 0` or `replacement_mu > 0`) | Gamma | none (`lore` uses `mu_0/beta`, `mu_new/beta`) | main:src/fleet_management/config.py:371-373, 540-547 |
| `gamma_beta_bound` (common rate; selects the modular route; must be `<=` smallest exact rate) | Gamma | none | main:src/fleet_management/config.py:369-370, 548-565 |
| `gamma_calibration_method` (`repeated_increment` default, `finite_count`) | Gamma | none | main:src/fleet_management/config.py:37, 374-376 |
| options `reliability_impl`, `pwl_points`, `tangent_ref`, `replacement_as_new`, `allow_replacement` | rainflow (the last also Gamma) | `formulation` | main:src/fleet_management/config.py:397-406 |
| none | — | `sigma` (Wiener), `eta` (IG), `alpha_inc` (Gamma shape increments), `v_max_user` | lore:src/fleet_management/solver.py:244-301 |

- Old `main` Gaussian and IG code expects different names: `alpha` (threshold), `xi` (repair fraction), `c` (IG shape), `C_S`, `C_P` (main:src/fleet_management/degradation_model/gaussian.py:120-135; main:src/fleet_management/degradation_model/inverse_gaussian.py:71-88).

## Q5. Rainflow on `main` versus rainflow on `lore`

"Remaining-life builder" on `main` means the rainflow (Palmgren-Miner cumulative damage) cell builder in `rainflow.py`. The module calls itself the "Remaining-life (rainflow/Palmgren--Miner) component formulation" (main:src/fleet_management/degradation_model/rainflow.py:1-6). It registers as `"rainflow"` (main:src/fleet_management/degradation_model/rainflow.py:685-695).

| Aspect | `lore` | `main` |
|---|---|---|
| Descriptors tracked | `mu`, `v` | `mu` always; `v` for Cantelli, Bernstein; `R = sum of squared support widths` for Hoeffding; `K = sum of cumulant generating function (CGF) values at tilt s` for Chernoff; latches `gmu`, `gv`, `gR` for `ard1` (main:src/fleet_management/degradation_model/rainflow.py:44-46, 152-171) |
| Encoding of dynamics | one-sided big-M rows, `>=` (lore:src/fleet_management/models/rainflow.py:144-153) | equalities through Gurobi indicator constraints on `nb`, `m`, `r` (main:src/fleet_management/degradation_model/rainflow.py:212-338) |
| Repair | ARD1 only (`lore` meaning: whole state) (lore:src/fleet_management/models/rainflow.py:44-49) | `ardinf` (whole state) or `ard1` (since latch): `mu+ = (1-rho)*mu + rho*gmu`, `v+ = (1-rho)^2*v + (1-(1-rho)^2)*gv`; Chernoff with `ard1` is rejected (main:src/fleet_management/degradation_model/rainflow.py:221-251, 122-127) |
| Replacement value | `mu_new`, `v_new` from input (lore:src/fleet_management/models/rainflow.py:152-153) | `mu_new = v_new = 0` by default: option `replacement_as_new` defaults to `True` and overwrites the input for rainflow cells (main:src/fleet_management/degradation_model/rainflow.py:104-109; main:src/fleet_management/degradation_model/base.py:1220-1221) |
| Repair cost `z` | only on repair days (McCormick with `x_m`) (lore:src/fleet_management/models/base.py:73-88) | `z = mu_prev - mu` on repair **and** on replacement days (main:src/fleet_management/degradation_model/rainflow.py:277-286). The `main` Gamma block excludes replacement from `z` (main:src/fleet_management/degradation_model/base.py:313-315). |
| Bounds | `cantelli`, `bernstein` (`tail_bound`) | `markov`, `cantelli`, `hoeffding`, `bernstein`, `chernoff` (`bound_method`) |
| Linear forms | `formulation: lp`, tangent at `mu = 0` | `reliability_impl`: `exact`, `tangent` at `mu = tangent_ref*tau` (default 0.5), `pwl` with `pwl_points` segments (default 8) and one binary per segment (main:src/fleet_management/degradation_model/rainflow.py:399-524; main:src/fleet_management/degradation_model/base.py:1215-1219) |
| Loop constraint | `mu`, `v` at `2H-1 <= H-1` | every tracked descriptor and latch at `T-1 <= H1-1` (main:src/fleet_management/degradation_model/rainflow.py:604-641) |
| Variable bounds | big-M `tau`, `V_max^RF` | tightened upper bounds, for example `v <= min(v_0 + T*max v, tau^2*eps/(1-eps))` for Cantelli (main:src/fleet_management/degradation_model/rainflow.py:646-680) |
| Input checks | Bhatia-Davis checks and `tau > 2*kappa*b/3` for Bernstein (lore:src/fleet_management/models/rainflow.py:66-102) | positive `v`, `support`, `cgf`; `s_chernoff > ln(1/eps)/tau` (main:src/fleet_management/config.py:477-518) |

Reliability rows on `main` (`d = tau - mu`, `Le = ln(1/eps)`, `b` = largest `support` value):

| Bound | `main` exact row | Same as `lore`? | Evidence |
|---|---|---|---|
| Markov | `mu <= eps*tau` | no counterpart | main:src/fleet_management/degradation_model/rainflow.py:420-424 |
| Cantelli | `(1-eps)*v <= eps*d^2`, `mu <= tau` | yes, identical to `lore` Cantelli | main:src/fleet_management/degradation_model/rainflow.py:427-434 |
| Hoeffding | `d^2 >= 0.5*Le*R`, `mu <= tau` | no counterpart | main:src/fleet_management/degradation_model/rainflow.py:437-444 |
| Bernstein | `0.5*d^2 - (Le*b/3)*d - Le*v >= 0`, `mu <= tau` | algebraically equal to `lore` `(d - c)^2 >= c^2 + 2*kappa*v`, `c = kappa*b/3` (expand both to `d^2 >= 2*Le*v + (2/3)*Le*b*d`). The `lore` side row is `mu <= tau - c`, which is stricter than `mu <= tau`. The definition of `b` differs: support width on `main`, `max(b_0, b_new, max b_inc)` on `lore`. | main:src/fleet_management/degradation_model/rainflow.py:447-456 |
| Chernoff | `K - s*tau <= ln(eps)` | no counterpart | main:src/fleet_management/degradation_model/rainflow.py:459-464 |

- `NonConvex = 2` is set only if some rainflow cell uses a quadratic implementation (main:src/fleet_management/degradation_model/rainflow.py:136-143).
- Interpretation: the `lore` `lp` form is the `main` `tangent` form with `tangent_ref = 0`. Both are inner (safe) approximations of a convex right-hand side.
- Interpretation: the `main` exact Bernstein set also contains the isolated point `d = 0, v = 0` (`mu = tau`), the second root branch of the quadratic. The `lore` side row `mu <= tau - c` excludes it. The point is probably unreachable, because `main` requires positive `v` increments.

## Q6. Status of the secondary files on `main`

| File | Purpose | Used by public `solve()`? | Status | Read |
|---|---|---|---|---|
| `degradation_model/gaussian.py`, `inverse_gaussian.py` | old monolithic solvers (merge-base code) | no: no cell builder is registered, so `get_cell_builder` raises `NotImplementedError` (main:src/fleet_management/degradation_model/base.py:149-156) | dead code on the public path | full |
| `degradation_model/gamma.py` | `GammaModel` helper class; `maximum_reliable_shape` | indirectly: the legacy Gamma route imports `maximum_reliable_shape` (main:src/fleet_management/degradation_model/legacy/gamma_gurobi.py:38) | live helper | full |
| `gamma_utils/gamma_repeated_calibration.py` | default offline calibration | **yes**, default modular route (main:src/fleet_management/degradation_model/base.py:358-361, 580-585) | live code (listed as secondary in the brief, but it is on the default path) | full |
| `gamma_utils/gamma_tail_bound.py` | finite-count LP calibration, Moschopoulos series | only with `gamma_calibration_method: finite_count` (main:src/fleet_management/degradation_model/base.py:586-594) | self-described legacy, still reachable | full (error branches skimmed) |
| `gamma_utils/gamma_process.py` | shape-rate helper functions | no; imported only by `validation/validator.py` | validation helper | in part |
| `gamma_utils/gamma_gurobi.py` | 12-line re-export of the legacy backend | no | compatibility shim | full |
| `legacy/gamma_gurobi.py` | constant-rate Gamma MILP with indicator constraints | **yes**, for a Gamma-only fleet without `gamma_beta_bound` (main:src/fleet_management/solver.py:112-128); README calls it "legacy fallback/regression oracle" (main:README.md:311) | regression oracle, still live | in part (lines 1-60, 156-310) |
| `gamma_utils/GAMMA_DIAGNOSTICS.md` | documentation of calibration, row counts, replay | not code | documentation | full |
| `rainflow_v2.py` | study of indicator versus big-M encodings | no; imports only the archived `legacy.rainflow_formulation_base` (main:src/fleet_management/degradation_model/rainflow_v2.py:1-8, 97) | historical study | in part (docstring, imports) |
| `rainflow_sparse.py` | study of sparse matrix assembly | no (main:src/fleet_management/degradation_model/rainflow_sparse.py:1-8, 134) | historical study | in part |
| `legacy/rainflow_formulation_base.py`, `legacy/rainflow_formulation_config.py` | archived copies of an older `base.py` (Gamma still a placeholder that raises) and `config.py` | no; used only by the two studies (main:src/fleet_management/degradation_model/legacy/rainflow_formulation_base.py:291-316, 752, 781) | archive | in part |
| `gamma_utils/gamma_diagnostics.py` | formulation-size estimate and Gurobi statistics | **yes**, for the `performance` block after every modular solve (main:src/fleet_management/degradation_model/base.py:2058-2080) | live reporting, adds no constraints | in part |
| `gamma_validator.py`, `gamma_tail_validator.py`, `gamma_replay_validator.py`, `gamma_stochastic_validator.py` | post-solve checks: constant-rate replay, exact Moschopoulos replay, bookkeeping replay, Monte Carlo | not called by `solve()`; three are exported by the package (main:src/fleet_management/__init__.py) | live validation tools | in part (docstrings) |

Facts on the legacy Gamma route:
- It ignores `repair_model`: `_cfg_to_gamma_kwargs` passes no repair selector (main:src/fleet_management/solver.py:184-212), and repair is always `A = (1-rho)*A_prev` (main:src/fleet_management/degradation_model/legacy/gamma_gurobi.py:263-269).
- It uses only `H1` as the horizon and a `2H` layout (main:src/fleet_management/solver.py:188-197).
- It keeps the old capacity row `sum mu <= F - M` and the `C_P` penalty (main:src/fleet_management/degradation_model/legacy/gamma_gurobi.py:209-228).

## Q7. Items with no counterpart in the other branch

Only on `lore`:
- **Wiener model**: variance grows by `sigma^2` per mission, falls only by replacement; supports ARD1 and ARA1 (lore:src/fleet_management/models/wiener.py).
- **ARA1 with a pre-repair anchor** for Wiener and Gamma (lore:src/fleet_management/maintenance/ara1.py). See the possible defect in Q2.
- **A working Gaussian model** with `Phi^{-1}` reliability, variance repair `(1-rho)^2` and replacement (lore:src/fleet_management/models/gaussian.py).
- **Exact IG cap** from the IG CDF, solved in log space (lore:src/fleet_management/models/inverse_gaussian.py:38-75).
- **User variance cap** `v_max_user` (Gaussian, Wiener, rainflow).
- **Bhatia-Davis consistency checks** and the precondition `tau > 2*kappa*b/3` for Bernstein (lore:src/fleet_management/models/rainflow.py:66-102).
- **Gamma shape increments given directly** as `alpha_inc`, shape-scale convention (lore:src/fleet_management/models/gamma.py:22-42, 104).

Only on `main`:
- **Five rainflow bounds** (adds Markov, Hoeffding, Chernoff) and the `pwl` implementation with `pwl_points` (main:src/fleet_management/degradation_model/rainflow.py:530-558).
- **Latches `gmu`, `gv`, `gR`** for the since-last-intervention repair on means, variances and Hoeffding budgets (main:src/fleet_management/degradation_model/rainflow.py:257-325).
- **Common-rate Gamma surrogate**: separate physical mean and bounding shape, offline calibration, seed states, mission- and time-varying rates, product-hull dynamics without big-M, reachable-set variable bounds (main:src/fleet_management/degradation_model/base.py:164-1187).
- **Legacy constant-rate Gamma backend** with indicator constraints (main:src/fleet_management/degradation_model/legacy/gamma_gurobi.py).
- **Transitory and operating phases** `H1`, `H2` with `*_trans` profiles (main:src/fleet_management/config.py:320, 381-384).
- **Options `allow_replacement` and `replacement_as_new`** (main:src/fleet_management/degradation_model/base.py:1206-1221).
- **Per-cell `epsilon`** (main:src/fleet_management/config.py:356).
- **Post-solve Gamma validators and diagnostics** (Q6).

Stale text on `main` (fact): the `solve_mixed` docstring still says Gamma raises `NotImplementedError` (main:src/fleet_management/degradation_model/base.py:2011-2012). That describes the archived placeholder (main:src/fleet_management/degradation_model/legacy/rainflow_formulation_base.py:300-309), not the current Gamma block.

## Evidence basis

- `lore`, read in full: `models/base.py`, `gaussian.py`, `inverse_gaussian.py`, `wiener.py`, `gamma.py`, `rainflow.py`, `maintenance/ard1.py`, `maintenance/ara1.py`; `solver.py` lines 30-50, 128-150, 180-330, 505-560; `spec/spec.tex` lines 128-140, 1010-1052.
- `main`, read in full: `degradation_model/gaussian.py`, `gamma.py`, `rainflow.py`, `gamma_utils/gamma_repeated_calibration.py`, `gamma_tail_bound.py` (error branches skimmed), `gamma_gurobi.py`, `GAMMA_DIAGNOSTICS.md`; `base.py` lines 1-160, 164-1187, 1202-1250, 1697-1734, 1978-2080.
- `main`, read in part: `inverse_gaussian.py` (validation and solve body; output block skimmed); `solver.py` lines 16-240; `config.py` lines 20-80, 300-420, 460-567; `README.md` lines 1-110; `legacy/gamma_gurobi.py` lines 1-60, 156-310; `gamma_process.py`; `rainflow_v2.py`, `rainflow_sparse.py`, `legacy/rainflow_formulation_*.py` (docstrings, imports); the four validators and `gamma_diagnostics.py` (docstrings, call sites).
- Not run: no solve was executed (no Gurobi). The ARA1 finding in Q2 and the Bernstein point in Q5 are derived from the code, not tested.
