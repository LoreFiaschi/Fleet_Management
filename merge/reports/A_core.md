Status: done. All ten questions answered. Read in full: every scope file on both branches, and the merge-base `solver.py` and `gaussian.py`. Next: none.

# A. Core pipeline and optimisation formulation: `lore` vs `main`

Merge base: 462ac1d. Each section first states the merge-base form. A branch whose entry differs from that form changed the item since the merge base. Both branches rewrote `solver.py` completely (`git diff --stat 462ac1d <branch>`: lore +772/−247, main +470/−172 with `__init__.py`). Citations `base:` refer to the merge base.

## 1. Public API

Merge base: `solve(input_path, degradation, results_path=None)` and `plot_management` (base:solver.py:18, base:__init__.py). Both branches changed `solve`.

| Item | lore | main |
|---|---|---|
| `solve` signature | `solve(input_path, results_path="output.yaml") -> dict` (lore:solver.py:59) | `solve(input_path, results_path=None, *, warm_start=None) -> dict` (main:solver.py:16-21) |
| `degradation` argument | removed; model comes from the input key `model` | removed; model comes from the input key `model` (main:solver.py:25-27) |
| `results_path=None` | skips writing the file (lore:solver.py:68-69, 95) | writes `output.yaml` (main:solver.py:48, 305-306) |
| Warm start | input key `warm_start: bool`, used only in the horizon loop (lore:solver.py:161) | keyword argument `warm_start: dict` (main:solver.py:20); forbidden on the legacy Gamma path (main:solver.py:112-116) |
| Return value | result dict, or a dict keyed by `H` for `[H_min, H_max]` (lore:solver.py:90-93) | one result dict (main:solver.py:57-64) |
| Other exports | `plot_management` only (lore:__init__.py:1-4, unchanged from base) | 15 names (main:__init__.py:27-43): `solve`, `plot_management`, `plot_mixed_management`, `plot_horizon_sweep`, `sweep_operating_horizons`, `sweep_horizon_grid`, `sweep_formulation_dimensions`, `validate`, `validate_baseline_assignment_feasibility`, 6 Gamma validators |
| `degradation_model` package exports | none (`models/__init__.py` is empty) | `GammaAction`, `GammaModel` (main:degradation_model/__init__.py:1-6) |

Signatures of main-only exports (read only at the `def` line; bodies out of scope, UNVERIFIED beyond the signature): `sweep_operating_horizons(input_path, operating_horizons, *, output_path=None, stop_on_gradient=False, gradient_tolerance=1e-3, flat_gradients_required=2, minimum_cases=3, ...)` (main:horizon_sweep.py:335); `sweep_horizon_grid(input_path, transitory_horizons, operating_horizons, *, output_path=None, warm_start=True, evaluation_horizon=None, **sweep_options)` (main:horizon_sweep.py:558); `sweep_formulation_dimensions(input_path, *, candidates=None, output_path=None)` (main:formulation_size_sweep.py:199).

main's `validate(input_path, degradation, results_path, validation_path="validation.yaml", tol=1e-6)` still takes a `degradation` argument, unlike `solve` (main:validation/validator.py:28).

## 2. Input schema and validation

Both branches rewrote parsing. lore keeps it in `solver._parse_and_validate`; main moved it to `config.load_config` and a dataclass `FleetConfig` (main:config.py:48-95, 293-418).

| Item | Merge base | lore | main |
|---|---|---|---|
| Model selector | `degradation` argument, whole fleet (base:solver.py:18) | key `model`, shape (F, L) only (lore:solver.py:172-176) | key `model`: one string, length-L list, or (F, L) (main:config.py:313-330) |
| Supported models | gaussian, inverse_gaussian | gaussian, inverse_gaussian, wiener, gamma, rainflow (lore:solver.py:36) | rainflow, gamma, gaussian, inverse_gaussian (main:config.py:34); no wiener |
| Repair selector | none | `maintenance_type`, (F, L), values `ARD1`/`ARA1` (lore:solver.py:181-188) | `repair_model`, string / (L,) / (F, L), values `ard1`/`ardinf`, default `ard1` (main:config.py:36, 354) |
| Tail-bound selector | none | `tail_bound`, string or (F, L), `cantelli`/`bernstein`, default `cantelli` (lore:solver.py:449-485) | `bound_method` (alias `method`), `markov`/`cantelli`/`hoeffding`/`bernstein`/`chernoff`, default `cantelli` (main:config.py:35, 352-353) |
| Threshold | `alpha`, scalar | `tau`, scalar or (F, L), > 0 (lore:solver.py:194-196) | `tau`, scalar / (L,) / (F, L), > 0 per cell (main:config.py:355, 468-469) |
| Repair efficiency | `xi`, (F, L) | `rho` only (lore:solver.py:197); range (0, 1] checked in each model module (for example lore:models/gaussian.py:31-32) | `rho`, aliases `xi`, `repair_rho` (main:config.py:357); (0, 1] (main:config.py:472-473) |
| Replacement state | none | `mu_new`, `v_new`, default 0 (lore:solver.py:202, 251) | `replacement_mu` (alias `mu_new`), `replacement_v` (alias `v_new`), default 0 (main:config.py:360-363) |
| `epsilon` | scalar in (0, 0.5) (base:gaussian.py:49-51) | scalar in (0, 0.01] (lore:solver.py:134-136) | per cell (F, L), in (0, 1) (main:config.py:356, 470-471) |
| Increment axis order | (F, M, L, H) | (F, M, L, H) (lore:solver.py:378-399) | (F, L, M, H) (main:config.py:241-261) |
| Increment shapes accepted | (F,M,L,H), (F,M,L), and two L=1 forms (base:solver.py:224-246) | (F,M,L) or (F,M,L,H); `null` entries allowed, become NaN (lore:solver.py:430-446) | scalar, (M,), (L,M), (L,M,H), (F,L,M), (F,L,M,H) (main:config.py:241-261) |
| Per-cell scalars accepted | (F,) if L=1, or (F, L) | scalar or (F, L); `null` allowed for model-specific fields (lore:solver.py:356-363, 413-427) | scalar, (L,) tiled over F, or (F, L) (main:config.py:166-179) |
| Costs `C_R`, `C_rep` | scalar | scalar, (F,), or (F, L): per train and component (lore:solver.py:366-375) | scalar or (L,): per component only, finite and >= 0 (main:config.py:220-238) |
| `C_M` | scalar | scalar (lore:solver.py:138) | scalar or (L,) (main:config.py:387-390) |
| `C_D` | absent (`C_S`, `C_P` instead) | required, > 0 (lore:solver.py:139-141) | required, alias `C_S` (main:base.py:1259-1263); no sign check |
| `mu` required | always | only if some component is not gamma (lore:solver.py:217-227) | always, > 0 at every cell (main:config.py:377, 474-475) |
| `F > M` check | yes (base:gaussian.py:36-37) | yes (lore:solver.py:131-132) | none found in core (grep of config.py, solver.py, base.py) |
| `mu_0 < tau` check | `mu_0 < alpha` | yes (lore:solver.py:203-204) | none found; `mu_var.UB = tau` instead (main:base.py:1654-1659) |
| Gamma keys | — | `alpha_inc` (shape increments), `beta` with `mu = alpha·beta` (lore:solver.py:292-301; lore:solver.py:594-599) | `mu` (expected damage), `gamma_beta` (a rate: tail uses scale `1/rate`, main:base.py:1106-1107), `gamma_beta_bound`, `_0`, `_new`, `_trans`, `gamma_calibration_method` (main:config.py:365-376) |
| Rainflow bound keys | — | `b_inc`, `b_0`, `b_new` for bernstein (lore:solver.py:303-322) | `support`, `cgf`, `s_chernoff` (main:config.py:364, 379-380, 499-518) |
| Variance cap | — | `v_max_user` (lore:solver.py:253-261) | none |
| Run options | `verbose`, `mip_gap` | `penalty_type`, `formulation`, `n_workers`, `warm_start`, `verbose`, `mip_gap`, `time_limit` (lore:solver.py:143-169) | 15 keys copied to `options`: `verbose, mip_gap, time_limit, fast, allow_replacement, gurobi_params, reliability_impl, pwl_points, tangent_ref, replacement_as_new, objective_mode, evaluation_horizon, progress_interval_seconds, relaxation_warm_start, relaxation_time_limit` (main:config.py:397-405) |
| Component names | none | none | `component_names`, default "Component n" (main:config.py:198-217) |
| Removed keys | — | `alpha`, `xi`, `C_S`, `C_P` not read | `depot_capacity` raises an error (main:config.py:303-308) |

Consistency-check placement:

- lore: global checks in `solver.py`; per-model checks in each module's `validate_inputs(mask, ...)` (lore:solver.py:324-331). ARA1 is rejected for gaussian, inverse_gaussian and rainflow inside those modules (lore:solver.py:189-192).
- main: all checks in `_validate_cells` (main:config.py:438-567). Only rainflow and gamma cells have model checks. The comment at main:config.py:567 says gaussian and inverse_gaussian are "reserved (handled by their builders)".
- main checks `repair_model` only at rainflow cells (main:config.py:481-483) and gamma cells (main:base.py:521-525). `chernoff` with `ard1` is rejected (main:config.py:484-494).
- main rejects a non-zero `replacement_mu`/`replacement_v` at rainflow cells while `replacement_as_new` is true (default) (main:config.py:444-458).
- HDF5 input on main reads a fixed key list (main:solver.py:239-277). The list has no `model`, `rho`, `C_D`, `bound_method` or `repair_model`. Interpretation: an HDF5 input on main cannot pass `load_config`, which requires `model` (main:config.py:313). lore reads every attribute and dataset generically (lore:solver.py:717-725).

## 3. Horizon structure

Merge base: `2H` steps; increments of length `H` wrap with `k % H` (base:gaussian.py:111-117).

| Item | lore | main |
|---|---|---|
| Scalar `H` | `2H` steps (lore:solver.py:494) | `H1 = H2 = H`, `T = 2H` (main:config.py:157-160) |
| Two-element `H` | `[H_min, H_max]`: one solve per integer `H` in the range (lore:solver.py:656-687); needs `H_max > H_min >= 1` (lore:solver.py:115-123) | `[H1, H2]`: one solve with `T = H1 + H2` steps (main:config.py:150-156); `H1` is a transitory phase, `H2` an operating phase; needs `H1, H2 > 0` |
| Increment length | `H` (scalar) or `H_max` (range); each solve uses the first `H` entries twice (lore:solver.py:402-405) | `H_prof = H2` if any rainflow cell or a modular gamma cell, else `H1` (main:config.py:336-341) |
| Transitory profiles | none | optional `mu_trans`, `v_trans`, `support_trans`, `cgf_trans`, `gamma_beta_trans`, length `H1` (main:config.py:367-368, 381-384) |
| Step-to-profile map | `k < H`: entry `k`; `k >= H`: entry `k - H` | `k < H1`: transitory entry `k`, or operating entry `k % H2` if no transitory profile; `k >= H1`: operating entry `(k - H1) % H2` (main:base.py:1291-1304) |
| Horizon loop in core | yes, sequential or `ProcessPoolExecutor` with `n_workers` (lore:solver.py:662-685) | no; sweeps live in `horizon_sweep.py`, outside the core (main:__init__.py:2; body not read) |
| Legacy gamma path | — | uses `H1` only when `H = [H1, H2]` (main:solver.py:188-189, 197) |

Interpretation: with `H1 = H2 = H` and no transitory profile, main's time structure equals lore's. main's `[H1, H2]` means a two-phase horizon. lore's `[H_min, H_max]` means a sweep of periods. The same input syntax has two different meanings.

## 4. Model assembly

Merge base: one monolithic solver per degradation model, one model for the whole fleet (base:solver.py:62-65).

| Item | lore | main |
|---|---|---|
| Top-level routing | always one shared model per horizon (lore:solver.py:492-567) | by fleet composition: gamma-only without `gamma_beta_bound` goes to the legacy constant-rate backend; all other fleets go to `base.solve_mixed` (main:solver.py:70-128) |
| Per-cell dispatch | `_DISPATCH[model](ctx, i, l)`, a dict of five functions (lore:solver.py:38-44, 545-547) | registry `CELL_BUILDERS` with `prepare` once per model, then `add_cell` per cell, optional `extract` (main:base.py:127-161, 1985-2004, 2052-2055) |
| Context object | `SimpleNamespace` (lore:solver.py:526-541) | dataclass `FleetModel` (main:base.py:54-121) |
| Assignment `x` | binary (F, M+1, 2H) (lore:solver.py:520) | binary (F, M+1, T) (main:base.py:1620) |
| Repair binary | `x_m` (F, L, 2H) (lore:solver.py:521) | `m` (F, L, T) (main:base.py:1621) |
| Replacement binary | `x_r` (F, L, 2H), always (lore:solver.py:522) | `r` (F, L, T), only if `allow_replacement` (main:base.py:1622) |
| Depot-idle binary | none | `idle` (F, L, T) (main:base.py:1627) |
| No-intervention binary | none | `nb` (F, L, T) for non-gamma cells (main:base.py:1628-1635) |
| Mean state | created by each model module (lore:solver.py:549; for example lore:models/wiener.py:57) | shared `mu` (F, L, T), `lb=0`, `ub=tau` (main:base.py:1636, 1654-1659) |
| Repair cost `z` | (F, L, 2H), `lb=0` (lore:solver.py:523) | (F, L, T), `lb=0`, `ub=tau` (main:base.py:1637, 1659) |
| Damage variable `u` | scalar (lore:solver.py:524) | scalar (main:base.py:1638) |
| One activity per train and step | `sum_j x[i,j,k] <= 1` (lore:models/base.py:40-45) | same (main:base.py:1676-1679) |
| Mission coverage | `sum_i x[i,j,k] = 1`, j = 1..M (lore:models/base.py:46-51) | same (main:base.py:1681-1684) |
| Depot gating | `x_m <= x[i,0,k]`, `x_r <= x[i,0,k]`, `x_m + x_r <= 1` (lore:models/base.py:52-64) | `idle + m + r = x[i,0,k]` (main:base.py:1711-1721); added inside each cell builder (for example main:base.py:746) |
| Carry selector | none | `nb + m + r = 1` for non-gamma cells (main:base.py:1722-1732) |
| Fleet capacity `sum mu <= F - M` | removed (present at base:gaussian.py:241-247) | removed, with a comment (main:base.py:1685-1688) |

Mixed-model support:

- lore: any mix of the five models in one model (lore:solver.py:545-547). NonConvex is set once for the whole model (lore:solver.py:515-518).
- main: builders are registered only for `gamma` (main:base.py:1187) and `rainflow` (main:rainflow.py:695; `_load_builders` imports only `rainflow`, main:base.py:1978-1982).
- main's `gaussian.py` and `inverse_gaussian.py` are byte-identical to the merge-base monolithic solvers (empty `git diff 462ac1d:src/fleet_management/gaussian.py main:src/fleet_management/degradation_model/gaussian.py`, same for inverse_gaussian). They register no builder, and `solver.py` does not import them.
- Consequence (read from code, not run): a main input with a `gaussian` or `inverse_gaussian` cell passes `load_config` and then raises `NotImplementedError` in `get_cell_builder` (main:base.py:149-156, 1992-1994). So main solves mixes of rainflow and gamma only.

## 5. Objective function

Merge base: `C_M·Σ x[i,0,k] + C_R·Σ z[i,k] + C_S·Σ_k u_k + C_P·Σ (mu[H-1] - mu[2H-1] + v[H-1] - v[2H-1])`, with `u_k >= mu[i,l,k]` (base:gaussian.py:225-237, 331-338).

| Term | lore (lore:solver.py:551-561) | main (main:base.py:1749-1796) |
|---|---|---|
| Maintenance | `C_M · Σ_{i,k} x[i,0,k]`: paid per train per depot day, even if no component is repaired | `Σ C_M[l] · m[i,l,k]`: paid per component repair; a depot day with all components idle costs 0 |
| Repair | `Σ C_R[i,l] · z[i,l,k]` | `Σ C_R[l] · z[i,l,k]` |
| Replacement | `Σ C_rep[i,l] · x_r[i,l,k]` | `Σ C_rep[l] · r[i,l,k]`, only if `allow_replacement`; default `C_rep[l] = C_R[l] · max_i tau[i,l]` (main:base.py:1272-1273) |
| Damage | `C_D · u` | `C_D · u` (alias `C_S`) |
| `u` definition | `u >= Σ_l mu[i,l,k]` (`inf_norm`, default) or `u >= Σ_l mu[i,l,k]²` (`quadratic`) (lore:models/base.py:103-118) | `u >= Σ_l mu[i,l,k]` only (main:base.py:1689-1694) |
| Periodicity penalty `C_P` | removed | removed from the modular path; still passed to the legacy gamma backend (main:solver.py:206) |
| Objective modes | one | `objective_mode`: `total` (default), `operating_average` (additive costs of the `H2` phase divided by `H2`), `evaluation_total` (initial phase plus `(evaluation_horizon - H1)/H2` times the operating phase) (main:base.py:1768-1794) |

- Both branches replaced the per-step `u_k` with one scalar `u` and dropped `C_P` in the modular formulation.
- Only main makes the costs component vectors. Only lore allows per-train costs (F,) or (F, L).
- Interpretation: the `C_M` term has a different meaning. Example with L = 3 components repaired on one depot day: lore charges `C_M` once; main charges `C_M[0] + C_M[1] + C_M[2]`. On both branches a train with no activity (all `x = 0`) costs nothing.

## 6. Loop / periodicity / sustainability constraint

Merge base: hard rows `mu[2H-1] <= mu[H-1]` and `v[2H-1] <= v[H-1]` (base:gaussian.py:315-329), plus the soft `C_P` term.

| Item | lore | main |
|---|---|---|
| Form | `state[2H-1] <= state[H-1]` (lore:models/base.py:91-100) | `state[T-1] <= state[H1-1]` (main:rainflow.py:617, 641; main:base.py:1055-1063) |
| Where | shared helper `base.add_loop_constraint`, called by each model module (for example lore:models/wiener.py:125) | written inside each model: rainflow `_add_repeatability` (main:rainflow.py:604-641), gamma `add_cell` (main:base.py:1053-1063); no shared helper in core |
| States covered | `mu` for all models; `v` for gaussian, wiener, rainflow; ARA1 anchors `mu_last`, `alpha_last` (lore:models/gaussian.py:106-107, inverse_gaussian.py:107, wiener.py:125-128, gamma.py:141-143, rainflow.py:192-193) | rainflow: `mu`, `v` (cantelli, bernstein), `R` (hoeffding), `K` (chernoff), and the ARD1 latches `gmu`, `gv`, `gR` (main:rainflow.py:624-639); gamma: bounding shape `A` and mean `mu`, but not the ARD1 latches (main:base.py:1056-1063) |
| Soft penalty | none | none in the modular path; `C_P` goes only to the legacy gamma backend (main:solver.py:206) |

- With `H1 = H2 = H` the two forms are the same row.
- Interpretation: main's gamma ARD1 cells carry latches `gmu`, `gA` (main:base.py:494-502) that the loop rows do not cover. main's own rainflow comment says the latch must be carried under ARD1 (main:rainflow.py:631-633). The gamma block may therefore miss the latch in its periodicity argument. The gamma agent should confirm.

## 7. Maintenance and replacement

**The names of the repair models collide.** Facts from the code:

| Repair law on the tracked state `s` | lore name | main name |
|---|---|---|
| `s ← (1-rho)·s` (whole state; `(1-rho)²` for a variance) | `ARD1` (lore:maintenance/ard1.py:3-6, 39-42) | `ardinf` (main:base.py:179-180, 818-819) |
| `s ← s - rho·(s - anchor)`, anchor = state at an earlier intervention | `ARA1` (lore:maintenance/ara1.py:3-7, 92-95) | `ard1` (main:base.py:180-184, 1017-1036) |

- The two "since an anchor" laws differ in the anchor. lore sets the anchor to the **pre-repair** state on a repair day (lore:maintenance/ara1.py:32-33, 39-44; the caller passes `mu_prev`, lore:models/wiener.py:78-80). main sets the latch to the **post-repair** state: `latch = latch_prev + (1-rho)·(mu_prev - latch_prev)` (main:base.py:1031-1036).
- Both reset the anchor or latch to the replacement value on a replacement day (lore:maintenance/ara1.py:45-50; main:base.py:1011-1016).
- lore allows ARA1 only for wiener and gamma (lore:solver.py:189-192). main's gamma accepts `ard1` and `ardinf` (main:base.py:521-525); rainflow accepts both except `chernoff` with `ard1` (main:config.py:484-494).
- Interpretation (my derivation, not run): lore's repair-cost row `z <= rho·(prev - anchor_prev)` is not gated by `x_m` (lore:models/base.py:86), and `z >= 0` (lore:solver.py:523). So every step forces `mu[k-1] >= anchor[k-1]`. The day after a repair, the anchor equals the pre-repair state. The row then forces the post-repair mean back up to the pre-repair mean. If this is right, an ARA1 repair on lore cannot lower the tracked mean. main's post-repair latch keeps `mu - latch >= 0` without this effect. This needs a check with a Gurobi run.

Dynamics and linearisation:

| Item | lore | main (gamma block in base.py; rainflow is out of scope) |
|---|---|---|
| Row type | one-sided big-M inequalities `>=`: accumulate row and repair row (lore:maintenance/ard1.py:35-42; ara1.py:88-95) | equality state balances plus exact binary-product hulls, "no conditional Big-M row" (main:base.py:185, 893-902, 1017-1030) |
| Big-M / bound constants | `tau` for means, `V_max` for variance, `alpha_bar` for gamma shape (lore:maintenance/ard1.py:31; lore CLAUDE.md) | time-dependent reachable upper bounds per cell (main:base.py:235-341, 612-680); `mu`, `z` capped at `tau` generically (main:base.py:1654-1659) |
| Product linearisation | McCormick for `z = rho·(prev - anchor)·x_m`, coefficient `rho·tau` (lore:models/base.py:73-88) | hull `product = scale·state·active`, coefficient `scale·upper` (main:base.py:187-218) |
| Repair cost `z` | ARD1: `rho·mu_prev·x_m`; ARA1: `rho·(mu_prev - anchor_prev)·x_m` | `ardinf`: `z = rho·mu_prev·m` (main:base.py:834-842); `ard1`: `z = rho·(mu_prev - latch_prev)·m` (main:base.py:1017-1020) |
| Replacement | always allowed; two-sided big-M reset to `mu_new`, `v_new` (lore:models/base.py:67-70) | option `allow_replacement`, default true (main:base.py:1206-1207); reset through products `qR = mu_prev·r` (main:base.py:852-892) |
| Replacement of a rainflow cell | resets to `mu_new`, `v_new` | `replacement_as_new` (default true) forces 0 (main:rainflow.py:104-109) |

## 8. Reliability-constraint plumbing and solver parameters

Merge base: one linear reliability row (base:gaussian.py:260-271); parameters `OutputFlag`, `MIPGap` (base:gaussian.py:209-211).

| Item | lore | main |
|---|---|---|
| Option name | `formulation`: `exact` (default) or `lp`; `socp` is a deprecated alias for `exact`; other values raise (lore:solver.py:147-154) | `reliability_impl`: `exact` (default), `tangent`, `pwl` (main:base.py:1215-1216; main:rainflow.py:355-363) |
| Scope | global; affects gaussian, wiener, rainflow (lore:solver.py:513-518) | global option, resolved per rainflow cell by bound; markov and chernoff have only `exact` (main:rainflow.py:136-141, 364) |
| Unknown value | `ValueError` | silent fallback to `exact` (main:rainflow.py:561-570) |
| Linear form | `lp`: `Φ⁻¹(1-ε)²·v + 2·tau·mu <= tau²`, the tangent of `(tau - mu)²` at `mu = 0` (lore:models/wiener.py:105-109) | `tangent` at `tangent_ref·tau` (default 0.5); `pwl` with `pwl_points` segments (default 8) and extra binaries (main:base.py:1217-1219; main:rainflow.py:356-363) |
| Gamma reliability | linear cap `alpha <= alpha_hat` (lore:models/gamma.py:125) | linear cap `A <= maximum_shape` (main:base.py:903-906, 1043-1046); no option |
| Coupling to damage penalty | `penalty_type='quadratic'` requires `formulation='exact'` (lore:solver.py:155-156) | none |
| `NonConvex = 2` | set in `solver.py` if `exact` and some cell is gaussian, wiener or rainflow (lore:solver.py:515-518) | set in rainflow `prepare` if some cell resolves to a quadratic implementation (main:rainflow.py:137-143) |

Gurobi parameters:

| Parameter | lore | main |
|---|---|---|
| `OutputFlag` | `verbose`, default 1 (lore:solver.py:165, 506) | `verbose`, default 1 (main:base.py:1208, 1616) |
| `MIPGap` | `mip_gap`, default `None` = Gurobi default (lore:solver.py:166-167, 507-508) | `mip_gap`, default **0.12** (main:base.py:1209, 1309-1310) |
| `TimeLimit` | `time_limit`, default **3600 s** (lore:solver.py:168-169, 509-510) | `time_limit`, default `None` (main:base.py:1210, 1311-1312) |
| Speed preset | none | `fast`: `MIPFocus=1`, `Heuristics=0.5`, `ImproveStartGap=0.5`, `NoRelHeurTime` (main:base.py:1313-1318) |
| Any other parameter | none | `gurobi_params` dict passed to `setParam` (main:base.py:1319-1321) |
| Progress log | none | MIP callback every `progress_interval_seconds` (default 5.0) (main:base.py:1232-1236, 1324-1367) |

Warm start:

| Item | lore | main |
|---|---|---|
| Trigger | input key `warm_start: true`, only in the sequential `[H_min, H_max]` loop; forbidden with `n_workers > 1` (lore:solver.py:161-163, 662-673) | keyword `solve(..., warm_start=<previous result dict>)` (main:solver.py:20, 57) |
| Mechanism | `VarHintVal` on `x`, `x_m`, `x_r` for the first `min(2H_prev, 2H)` steps (lore:solver.py:570-587) | `Start` (MIP start) on `x`, `m`, `r`, `idle`, `nb`; operating steps repeat the old `H2` schedule cyclically (main:base.py:1403-1464) |
| Completion | none | optional: if the dict has `complete_binary_start`, fix the binaries, solve, copy all values (time limit `completion_time_limit`, default 60 s), IIS names on failure (main:base.py:1467-1554) |
| Relaxation start | none | `relaxation_warm_start`: `feasRelaxS` phase-I solve with `MIPGap=0.05`, copies binaries (main:base.py:1557-1603, 2032-2036) |

## 9. Output

Merge base keys: `status, objective, F, H, M, L, alpha, x, mu, v, u, z, model` (Gurobi object), plus `degradation, mu_0, v_0` added in `solve` (base:gaussian.py:387-417; base:solver.py:67-70).

| Item | lore (lore:solver.py:602-649) | main (main:base.py:1855-1972; main:solver.py:57-64, 93-98) |
|---|---|---|
| Solution arrays | `x` (F, M+1, 2H), `x_m`, `x_r`, `mu`, `v`, `z` (F, L, 2H) | `x` (F, M+1, T), `m`, `r`, `idle`, `mu`, `v`, `z` (F, L, T) |
| `v` where not tracked | NaN (lore:solver.py:625) | 0 (main:base.py:1888) |
| `u` | value of the variable `u` (lore:solver.py:647) | recomputed as `max_{i,k} Σ_l mu[i,l,k]` (main:base.py:1908-1912) |
| Key `model` | model assignment, nested list (F, L) (lore:solver.py:606) | the disposed Gurobi model object (main:base.py:1868, 2157); the assignment is in `model_assignment` |
| Echoed inputs | `tau`, `tail_bound`, `F, M, L, H` | `tau, mu_0, v_0, H` (as written), `H1, H2, T`, `method`, `bound_method`, `repair_model`, `reliability_impl`, `models`, `component_names`, `backend`, `degradation` |
| Solver quality | none | `mip_gap`, `bound` (main:base.py:1870-1886) |
| Cost breakdown | none | `J_initialization, J_op, J_op_average, peak_damage, damage_penalty, operating_objective, J_total, projected_evaluation_cost, evaluation_objective, step_costs, component_costs` (main:base.py:1916-1971) |
| Diagnostics | none | `optimization_progress`, `warm_start`, `relaxation_warm_start`, `performance` (sizes, timings, threads, seed), gamma calibration and formulation reports (main:base.py:2049-2154) |

Status handling:

| Item | lore | main |
|---|---|---|
| When arrays are returned | only if status is `OPTIMAL`; otherwise all arrays `None` (lore:solver.py:614-619) | whenever `SolCount > 0`, so a `time_limit` incumbent is kept (main:base.py:1875-1881) |
| Status names | `optimal`, `infeasible`, `infeasible_or_unbounded`, `unbounded`, `time_limit`, `suboptimal`, else `status_code_N` (lore:solver.py:46-52, 590-591) | adds `interrupted`, `solution_limit`, `node_limit`, `iteration_limit`; uses `inf_or_unbounded`; else `gurobi_status_N` (main:base.py:1816-1828) |
| Per-horizon failure | an exception becomes status `error: ...`, the loop continues (lore:solver.py:665-671, 681-685) | no loop in core |

Serialisation:

| Format | lore | main |
|---|---|---|
| Path | `results_path=None` writes nothing; missing extension gets `.yaml`; directory checked after the solve (lore:solver.py:95-98, 728-742) | `None` means `output.yaml`; directory checked before the solve (main:solver.py:48-53, 303-310) |
| YAML | `yaml.dump`; multi-horizon file keyed by integer `H` (lore:solver.py:775-782) | `yaml.safe_dump` of a whitelisted key set (main:solver.py:328-449, 471-477) |
| JSON | multi-horizon keys are strings (lore:solver.py:785-792) | same whitelist (main:solver.py:480-483) |
| HDF5 | groups `metadata`, `solution` (`int8` binaries), `parameters` (`epsilon, tau, rho, model`, rainflow `tail_bound, b_inc, b_0, b_new`); multi-horizon groups `H<h>` (lore:solver.py:795-845) | flat root: attributes for scalars and JSON strings, datasets for arrays (main:solver.py:486-579) |

## 10. Features present on one branch only

In main's core, with no counterpart in lore:

- **Formulation**: two-phase horizon `[H1, H2]` (Q3); objective modes and cost breakdown (main:base.py:1735-1811, 1916-1971); per-component costs with a default `C_rep` (main:base.py:1250-1279); `allow_replacement` (main:base.py:1206-1207); `idle`, `nb` binaries (main:base.py:1627-1635, 1697-1732); per-cell `epsilon` (main:config.py:356).
- **Modular gamma block** inside `base.py`: offline common-rate calibration, bounding shape `A`, reachable upper bounds, product hulls, no big-M (main:base.py:164-1187). The gamma agent covers the mathematics. A **legacy gamma backend** serves gamma-only fleets without `gamma_beta_bound` (main:solver.py:101-212).
- **Rainflow options**: bounds markov, hoeffding, chernoff; `tangent`/`pwl` implementations; `replacement_as_new` (main:config.py:35, 441-458; main:rainflow.py:104-109, 355-364).
- **Solver tooling**: `fast`, `gurobi_params`, progress callback, MIP start with completion, relaxation warm start, `performance` report (Q8, Q9).
- **Schema conveniences**: aliases `xi`, `repair_rho`, `method`, `mu_new`, `v_new`, `C_S` (main:config.py:352-363; main:base.py:1259-1263); (L,) arrays tiled over F (main:config.py:166-195); `component_names` (main:config.py:198-217).
- **Deprecated registry** `utils/model_registry.py`, marked `# DEPRECATED` (main:utils/model_registry.py:8). It reads a `degradation` key and the merge-base keys `alpha`, `C_S`, `C_P` (main:utils/model_registry.py:17-66, 311-346). Only `validation/validator.py:14` and `validation/dashboard_rainflow.py:469` import it; `solve` does not.
- **Protocol** `AccumulatedDegradationModel` (main:base.py:2164-2180): defined, not used in `src/` outside `legacy/`.

In lore's core, with no counterpart in main:

- **Models**: wiener (lore:solver.py:36, 284-290); working gaussian and inverse_gaussian cells in the shared model, which main lists but cannot build (Q4).
- **Horizon sweep inside `solve`**: `H = [H_min, H_max]`, `n_workers`, per-horizon error capture (lore:solver.py:656-687).
- **Formulation options**: quadratic damage penalty (lore:models/base.py:115-118); `formulation='lp'` for Gaussian-type reliability (lore:solver.py:147-156); variance cap `v_max_user` (lore:solver.py:253-261).
- **Schema**: per-train costs (F,) and (F, L) (lore:solver.py:366-375); `null` entries mapped to NaN (lore:solver.py:408-446).
- **Checks**: `F > M`, `mu_0 < tau`, `epsilon <= 0.01`, `C_D > 0` (lore:solver.py:131-141, 203-204).
- **Output**: skip writing with `results_path=None` (lore:solver.py:95); grouped HDF5 with a `parameters` group (lore:solver.py:804-845).

Main risks found in this area (interpretation, from reading only):

1. main's `solve` cannot build gaussian or inverse_gaussian cells (Q4).
2. The names `ard1`/`ARD1` mean different repair laws on the two branches (Q7).
3. lore's ARA1 repair may be neutralised by the ungated `z` row (Q7).
4. main reads HDF5 input with a fixed key list that lacks `model` (Q2).
5. `H: [a, b]` means a two-phase horizon on main and a sweep on lore (Q3).
6. main's default `mip_gap = 0.12` accepts solutions up to 12 % from the bound (Q8).
