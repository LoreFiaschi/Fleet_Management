# Differences between the branches `lore` and `main`

| | `lore` | `main` |
|---|---|---|
| Commit compared | 651aa71 (30 Jul 2026) | 978ea9a (28 Sep 2026), equal to `origin/main` after a fetch on 30 Sep 2026 |
| Commits since the common ancestor | 6 | 116, merged into `main` through pull request 1 (`final-main-integration-20260928`) |
| Authors since the common ancestor | Lorenzo Fiaschi | forzinkahd (73 commits), jotschan and jtschan (36), Codex (6), Christoph Langenauer (1) |

- **Common ancestor** (merge base): commit 462ac1d, "rainflow", 25 Jun 2026. At that commit the package had two
  monolithic solvers, Gaussian and inverse Gaussian.
- **Evidence format**: `branch:path:line`. Code paths are relative to `src/fleet_management/`.
- **Nothing was run.** Gurobi is not installed on this computer. Statements about behaviour come from reading the
  code. Statements marked *derived* were worked out by hand from the constraints.

## 1. Summary

- **Two independent designs.** Since 25 Jun 2026 each branch rewrote the solver separately. Neither branch contains
  the other's work, so a merge means reconciling two designs.
- **Direction of `lore`**: more degradation models, a formal specification (`spec/spec.tex` version 0.5), and the
  ARA1 repair model.
- **Direction of `main`**: rainflow and Gamma in depth. It adds:
  - a two-phase horizon, horizon sweeps and warm starts;
  - validators;
  - cluster studies on synthetic cases modelled on VBZ (Verkehrsbetriebe Zürich, the Zurich public transport operator).
- **Which models run from `solve()`:**

| Model | `lore` | `main` |
|---|---|---|
| Gaussian | yes | accepted by the input check, then `NotImplementedError` |
| Inverse Gaussian (IG) | yes | accepted by the input check, then `NotImplementedError` |
| Wiener | yes | absent |
| Gamma | yes, one formulation | yes, two routes |
| Rainflow | yes, 2 tail bounds | yes, 5 tail bounds |

- **The same name has different meanings:**

| Name | `lore` | `main` |
|---|---|---|
| Input `H: [a, b]` | a range of horizons: one solve per `H` from `a` to `b`, each with `2H` steps | one solve: transitory phase `H1 = a`, then operating phase `H2 = b`, `T = a + b` steps |
| Repair model `ARD1` / `ard1` | the whole state is multiplied by `1-rho` | removes `rho` times the damage accumulated since the last intervention |
| Gamma `beta` / `gamma_beta` | a scale: mean `= alpha·beta` | a rate: mean `= A/beta` |
| Output key `model` | the model name of each component, shape (F, L) | the Gurobi model object; the names are in `model_assignment` |
| Cost `C_M` | once per train per depot day | per repaired component, `C_M[l]` |
| `solve(..., results_path=None)` | writes no file | writes `output.yaml` |
| Output `v` where no variance is tracked | NaN | 0 |

- **Specification.** Only `lore` updated `spec/spec.tex`, to version 0.5. `main` carries the older version
  "0.2 (revised)". Its README states that the code is authoritative.
- **Problems.** Section 12 ranks the problems found on each branch. The most serious ones:
  - `lore`: an ARA1 repair cannot lower the tracked state (derived).
  - `main`: the README quick-start input does not load.
  - `main`: the Gaussian and IG models are listed as supported but do not build.

## 2. Public API

| Item | `lore` | `main` |
|---|---|---|
| `solve` | `solve(input_path, results_path="output.yaml")` (lore:solver.py:59) | `solve(input_path, results_path=None, *, warm_start=None)` (main:solver.py:16-21) |
| Return value | one result dict, or a dict keyed by `H` for a range | one result dict |
| Exported names | `solve`, `plot_management` | 15 names: `solve`, `plot_management`, `plot_mixed_management`, `plot_horizon_sweep`, `sweep_operating_horizons`, `sweep_horizon_grid`, `sweep_formulation_dimensions`, `validate`, `validate_baseline_assignment_feasibility`, 6 Gamma validators (main:\_\_init\_\_.py:27-43) |
| Model selection | input key `model` | input key `model` |
| Command-line entry point | none | none in the package; many scripts at the top level and in `examples/`, `experiments/` |

- `main`'s `validate(input_path, degradation, results_path, ...)` still takes the `degradation` argument that `solve`
  dropped (main:validation/validator.py:28).

## 3. Input schema

| Concept | `lore` key | `main` key |
|---|---|---|
| Parser | `solver._parse_and_validate` | `config.load_config`, which returns a `FleetConfig` dataclass (main:config.py:48-95, 293-418) |
| Degradation model | `model`, shape (F, L) only | `model`: one string, an (L,) list, or (F, L) |
| Repair model | `maintenance_type`: `ARD1`, `ARA1` | `repair_model`: `ard1` (default), `ardinf` |
| Tail bound | `tail_bound`: `cantelli` (default), `bernstein` | `bound_method` (alias `method`): `markov`, `cantelli` (default), `hoeffding`, `bernstein`, `chernoff` |
| Linear or exact form | `formulation`: `exact` (default), `lp` | `reliability_impl`: `exact` (default), `tangent`, `pwl` |
| Reliability level `epsilon` | one scalar in (0, 0.01] | per cell (F, L), in (0, 1) |
| Threshold `tau` | scalar or (F, L) | scalar, (L,) or (F, L) |
| Repair efficiency | `rho` | `rho`, aliases `xi`, `repair_rho` |
| Replacement state | `mu_new`, `v_new` | `replacement_mu`, `replacement_v` (aliases `mu_new`, `v_new`) |
| Increment tensors | axis order (F, M, L, H); shapes (F, M, L) or (F, M, L, H); `null` allowed, read as NaN | axis order **(F, L, M, H)**; shapes scalar, (M,), (L, M), (L, M, H), (F, L, M), (F, L, M, H) |
| Transitory-phase profiles | none | `mu_trans`, `v_trans`, `support_trans`, `cgf_trans`, `gamma_beta_trans` |
| Costs `C_R`, `C_rep` | scalar, (F,) or (F, L) | scalar or (L,) |
| Cost `C_M` | scalar | scalar or (L,) |
| Cost `C_D` | required, > 0 | required, alias `C_S`, no sign check |
| Mean increment `mu` | required unless every component is Gamma | always required, > 0 |
| Variance cap | `v_max_user` | none |
| Component labels | none | `component_names` |
| Run options | `penalty_type`, `formulation`, `n_workers`, `warm_start`, `verbose`, `mip_gap`, `time_limit` | 15 options, among them `fast`, `gurobi_params`, `allow_replacement`, `replacement_as_new`, `objective_mode`, `evaluation_horizon`, `relaxation_warm_start` (main:config.py:397-405) |
| Removed keys | `alpha`, `xi`, `C_S`, `C_P` are not read | `depot_capacity` raises an error (main:config.py:303-308) |

Consistency checks:

- `lore` checks `F > M`, `mu_0 < tau`, `epsilon <= 0.01` and `C_D > 0` (lore:solver.py:131-141, 203-204). `main` makes
  none of these checks. It caps `mu` at `tau` through the variable bound instead (main:degradation_model/base.py:1654-1659).
- `lore` checks each model in its own module, for example Bhatia-Davis moment checks and `tau > 2·kappa·b/3` for
  Bernstein. `main` checks only rainflow and Gamma cells, in `config._validate_cells` (main:config.py:438-567).
- Model-specific keys are listed in section 7.

## 4. Horizon structure

| Item | `lore` | `main` |
|---|---|---|
| Scalar `H` | `2H` steps; the loop constraint compares step `2H` with step `H` | `H1 = H2 = H`, `T = 2H`: the same structure |
| Pair `[a, b]` | range `[H_min, H_max]`: one solve per integer `H` (lore:solver.py:656-687) | `[H1, H2]`: one solve with `T = H1 + H2` steps (main:config.py:150-156) |
| Mapping of increments to steps | the `H` entries are repeated once for the second half | step `k < H1`: transitory entry `k`, or operating entry `k mod H2` if no transitory profile; step `k >= H1`: operating entry `(k - H1) mod H2` (main:degradation_model/base.py:1291-1304) |
| Horizon loop inside `solve()` | yes: sequential or parallel (`n_workers`); a failed horizon is recorded and the loop continues | no; sweeps are separate functions (section 10.2) |

An input with `H: [4, 6]` therefore means three solves on `lore` and one solve of 10 steps on `main`.

## 5. Formulation

### 5.1 Variables and constraints

| Item | `lore` | `main` |
|---|---|---|
| Model assembly | one shared Gurobi model per horizon; each cell is built by `_DISPATCH[model](ctx, i, l)` | registry `CELL_BUILDERS`: `prepare` once per model, `add_cell` per cell (main:degradation_model/base.py:127-161) |
| Routing | always the shared model | a Gamma-only fleet without `gamma_beta_bound` goes to a legacy backend; every other fleet goes to `base.solve_mixed` (main:solver.py:70-128) |
| Assignment `x` | binary (F, M+1, 2H); `j = 0` is the depot | binary (F, M+1, T); same meaning |
| Repair and replacement binaries | `x_m`, `x_r` (F, L, 2H); replacement always allowed | `m`, `r` (F, L, T); `r` only if `allow_replacement` (default true) |
| Other binaries | none | `idle` (F, L, T); `nb` ("no intervention") for rainflow cells |
| Depot rule | `x_m <= x[i,0,k]`, `x_r <= x[i,0,k]`, `x_m + x_r <= 1` | `idle + m + r = x[i,0,k]`; `nb + m + r = 1` for rainflow cells |
| One activity per train and step; every mission covered | yes | yes |
| Dynamics rows | one-sided big-M inequalities (`>=`): the states are upper bounds | equalities: exact binary-product hulls for Gamma (no big-M), indicator constraints for rainflow |
| Variable bounds | big-M constants: `tau`, `V_max`, `tau/beta` | tighter bounds computed per cell and step (main:degradation_model/base.py:235-341; main:degradation_model/rainflow.py:646-680) |

### 5.2 Objective

| Term | `lore` | `main` |
|---|---|---|
| Maintenance | `C_M · Σ x[i,0,k]`: per train per depot day, even with no repair | `Σ C_M[l] · m[i,l,k]`: per repaired component; a depot day with no repair costs nothing |
| Repair | `Σ C_R[i,l] · z[i,l,k]` | `Σ C_R[l] · z[i,l,k]` |
| Replacement | `Σ C_rep[i,l] · x_r[i,l,k]` | `Σ C_rep[l] · r[i,l,k]`; default `C_rep[l] = C_R[l] · max_i tau[i,l]` |
| Damage | `C_D · u`, with `u >= Σ_l mu` (`inf_norm`) or `u >= Σ_l mu²` (`quadratic`) | `C_D · u`, with `u >= Σ_l mu` only |
| Loop penalty `C_P` | none | none in the modular path; still used by the legacy Gamma backend |
| Objective modes | one | `objective_mode`: `total` (default), `operating_average` (operating-phase costs divided by `H2`), `evaluation_total` (main:degradation_model/base.py:1768-1794) |

Example for `C_M`: three components repaired on one depot day cost `C_M` on `lore` and `C_M[0] + C_M[1] + C_M[2]` on `main`.

### 5.3 Loop (sustainability) constraint

| Item | `lore` | `main` |
|---|---|---|
| Form | `state[2H-1] <= state[H-1]` (0-based) | `state[T-1] <= state[H1-1]` |
| Where | shared helper `base.add_loop_constraint` | written inside each model |
| States covered | `mu` always; `v` for Gaussian, Wiener, rainflow; the ARA1 anchors | rainflow: every tracked quantity and its ARD1 latch. Gamma: bounding shape `A` and mean `mu`, **not** its ARD1 latches `gmu`, `gA` (main:degradation_model/base.py:1053-1063) |

## 6. Maintenance and replacement

ARD stands for Arithmetic Reduction of Degradation, ARA for Arithmetic Reduction of Age. The suffix is the memory
order: 1 or infinite ("inf").

| Repair law on the state `s` before repair | `lore` name | `main` name |
|---|---|---|
| `s+ = (1-rho)·s`, or `(1-rho)²·s` for a variance | `ARD1` | `ardinf` |
| `s+ = s - rho·(s - a)`, with `a` an anchor from the last intervention | `ARA1` | `ard1` |

- **The anchor differs.** On `lore` the anchor is the state *before* the last repair (lore:maintenance/ara1.py:39-44).
  On `main` the anchor ("latch") is the state *after* the last intervention (main:degradation_model/base.py:1031-1036).
- Both branches reset the anchor to the replacement value on a replacement day.
- **Which models accept which law.**
  - `lore`: ARA1 only for Wiener and Gamma; ARD1 for all five models.
  - `main`: `ard1` and `ardinf` for Gamma and rainflow, except rainflow with Chernoff and `ard1`.
  - A merge maps `lore ARD1` to `main ardinf`. `lore ARA1` maps to `main ard1` only if the anchor rule changes.
- **Replacement.**
  - `lore`: resets the state to `mu_new`, `v_new` from the input.
  - `main`, rainflow: the option `replacement_as_new` (default true) sets the replacement state to 0 and overrides the
    input (main:degradation_model/rainflow.py:104-109).
- **Repair cost `z`.**
  - `lore`: only on repair days.
  - `main`, rainflow: on repair days and on replacement days (main:degradation_model/rainflow.py:277-286).
  - `main`, Gamma: only on repair days.

## 7. Degradation models and reliability constraints

### 7.1 Per model

| Model | `lore` | `main` |
|---|---|---|
| Gaussian | states `mu`, `v`; ARD1; exact form `Φ⁻¹(1-ε)²·v <= (tau-mu)²` or a tangent line | `degradation_model/gaussian.py` is byte-identical to the file at the common ancestor; no cell builder, so `NotImplementedError` |
| IG | state `mu`; ARD1; cap `mu <= mu_bar` from the exact IG distribution function, solved in log space | old file, byte-identical to the common ancestor; its cap is a normal approximation; not reachable |
| Wiener | states `mu`, `v`; `v` grows by `sigma²` per mission and falls only by replacement; ARD1 or ARA1 | absent (`git grep -i wiener main` finds nothing) |
| Gamma | state: shape `alpha` (scale convention), mean `= alpha·beta`; ARD1 or ARA1; cap `alpha <= alpha_hat`, root of `Q(alpha, tau/beta) = ε` | two routes (section 7.3) |
| Rainflow | states `mu`, `v`; ARD1; Cantelli or Bernstein | five bounds, latches for `ard1` (section 7.2) |

Model-specific input keys:

| `lore` | `main` |
|---|---|
| `eta` (IG), `sigma` (Wiener) | none |
| `beta` (Gamma scale), `alpha_inc` (Gamma shape increments) | `gamma_beta` (rate profile), `gamma_beta_0`, `gamma_beta_new`, `gamma_beta_bound`, `gamma_calibration_method` |
| `b_inc`, `b_0`, `b_new` (Bernstein support) | `support`, `support_trans` (Hoeffding, Bernstein) |
| none | `cgf`, `cgf_trans`, `s_chernoff` (Chernoff) |

### 7.2 Rainflow

| Aspect | `lore` | `main` |
|---|---|---|
| Tracked quantities | `mu`, `v` | `mu`; `v` for Cantelli and Bernstein; `R` (sum of squared support widths) for Hoeffding; `K` (sum of cumulant generating function values) for Chernoff; latches `gmu`, `gv`, `gR` for `ard1` |
| Linear form | `formulation: lp`: tangent at `mu = 0` | `tangent` at `mu = tangent_ref·tau` (default 0.5), or `pwl` with `pwl_points` segments (default 8) and one binary per segment |
| Input checks | Bhatia-Davis; `tau > 2·kappa·b/3` for Bernstein | positive `v`, `support`, `cgf`; `s_chernoff > ln(1/ε)/tau` |

Exact reliability rows (`d = tau - mu`, `Le = ln(1/ε)`):

| Bound | `main` row | Relation to `lore` |
|---|---|---|
| Markov | `mu <= ε·tau` | not on `lore` |
| Cantelli | `(1-ε)·v <= ε·d²`, `mu <= tau` | identical to `lore` |
| Hoeffding | `d² >= 0.5·Le·R`, `mu <= tau` | not on `lore` |
| Bernstein | `0.5·d² - (Le·b/3)·d - Le·v >= 0`, `mu <= tau` | the same quadratic as `lore`'s `(d - c)² >= c² + 2·kappa·v`, `c = kappa·b/3`. `lore` adds the side row `mu <= tau - c`. The constant `b` differs: largest `support` value on `main`, `max(b_0, b_new, max b_inc)` on `lore`. |
| Chernoff | `K - s·tau <= ln ε` | not on `lore` |

### 7.3 Gamma

| Aspect | `lore` | `main`, modular route | `main`, legacy route |
|---|---|---|---|
| When used | always | `gamma_beta_bound` given, or Gamma mixed with rainflow | Gamma-only fleet without `gamma_beta_bound` |
| Parameter convention | shape-scale | shape-rate | shape-rate |
| States | `alpha` (mean derived) | physical mean `mu` and bounding shape `A'`, tracked separately | shape `A` |
| Reliability row | `alpha <= alpha_hat`, exact for one scale per component | `A' <= A_max` after an offline calibration to one common rate (default method `repeated_increment`) | `A <= A_max`, one rate per component |
| Repair laws | ARD1, ARA1 | `ard1`, `ardinf` | always `A = (1-rho)·A_prev`; `repair_model` is ignored |
| Horizon | `2H` | `T = H1 + H2` | `H1` only, `2H` layout; keeps `C_P` and the old capacity row `Σ mu <= F - M` |

- `main`'s documentation says the calibration certifies repetitions of one increment type only
  (main:degradation_model/gamma_utils/GAMMA_DIAGNOSTICS.md:59-63).
- Interpretation: a schedule that mixes increment types is not certified by the calibration.

### 7.4 Status of other model files on `main`

| File | Status |
|---|---|
| `rainflow_v2.py`, `rainflow_sparse.py`, `legacy/rainflow_formulation_*.py` | historical encoding studies; not used by `solve()` |
| `gamma_utils/gamma_repeated_calibration.py` | live: the default Gamma calibration |
| `gamma_utils/gamma_tail_bound.py` | reachable only with `gamma_calibration_method: finite_count`; describes itself as legacy |
| `gamma_utils/gamma_diagnostics.py` | runs after every modular solve, for the `performance` report only |
| `gamma_utils/gamma_*validator.py` (4 files) | standalone post-solve checks; not called by `solve()` |
| `legacy/gamma_gurobi.py` | the live legacy Gamma route |

## 8. Solver settings and warm start

| Item | `lore` | `main` |
|---|---|---|
| `MIPGap` default | Gurobi default | **0.12** (main:degradation_model/base.py:1209) |
| `TimeLimit` default | 3600 s | none |
| Other parameters | none | `fast` preset (`MIPFocus=1` and others); any Gurobi parameter through `gurobi_params`; progress callback |
| `NonConvex = 2` | if `exact` and some cell is Gaussian, Wiener or rainflow | if some rainflow cell uses a quadratic form |
| Unknown linear-form value | `ValueError` | silent fallback to `exact` (main:degradation_model/rainflow.py:561-570) |
| Warm start | `VarHintVal` hints on `x`, `x_m`, `x_r`, only inside the sequential range loop, only from an optimal previous horizon | `Start` values from a previous result (`solve(..., warm_start=result)`), mapped cyclically onto the operating phase; optional completion of the continuous states; optional relaxation start (`feasRelaxS`); a greedy heuristic start (`greedy_warm_start.py`) |

- The greedy heuristic never schedules a replacement: its `r` array stays zero (main:greedy_warm_start.py:448).

## 9. Output

| Item | `lore` | `main` |
|---|---|---|
| Arrays | `x` (F, M+1, 2H); `x_m`, `x_r`, `mu`, `v`, `z` (F, L, 2H) | `x` (F, M+1, T); `m`, `r`, `idle`, `mu`, `v`, `z` (F, L, T) |
| `u` | the solved variable | recomputed as `max_{i,k} Σ_l mu[i,l,k]` |
| Non-optimal status | every array is `None` | arrays are kept whenever a solution exists, for example after a time limit |
| Extra content | `tau`, `tail_bound` | `mip_gap`, `bound`, a cost breakdown (`J_initialization`, `J_op`, `J_total` and others), `performance`, warm-start and calibration reports |
| YAML and JSON | all keys; a range result keyed by `H` | a fixed list of keys, `yaml.safe_dump` |
| HDF5 | groups `metadata`, `solution`, `parameters`, one group `H<h>` per horizon | flat root |

## 10. Tooling

### 10.1 Plotting

| Aspect | `lore` `plot_management` | `main` `plot_management` | `main` `plot_mixed_management` |
|---|---|---|---|
| Rows | one per train, split into `L` strips | same | one per (vehicle, component) |
| Columns | `2H` | `T+1`; column 0 is `mu_0` | `T+1`; first column "initial" |
| Model shown by | border style, with legend | not shown | row label ("Gamma" or "Remaining-life") |
| Action marks | mission number, gear (repair), "R" (replacement), "zzz" (idle) | mission number, gear (any depot day), "zzz"; no replacement mark | `M_j`, `I` (idle), `R` (repair), `P` (replacement) per component |
| Extra | title with sizes and objective | none | run-summary panel (status, gap, sizes, runtime, counts); reads a sibling Gurobi log |
| Non-optimal result | raises `ValueError` | no check | plotted |

- The branches write different output schemas, so neither plotter reads the other branch's output.
- `main` also has `plot_horizon_sweep`: `J_op/H2` against `H2`, the MIP gap, and model sizes.

### 10.2 Horizon search

| Aspect | `lore` | `main` `sweep_operating_horizons` | `main` `sweep_horizon_grid` |
|---|---|---|---|
| What varies | `H` from `H_min` to `H_max` | a list of `H2` values, `H1` fixed | an `H1 × H2` grid |
| Parallel | yes (`n_workers`) | no | no |
| A case fails | recorded as `status: "error: ..."`; the loop continues | the exception stops the sweep; a checkpoint keeps finished cases | same |
| Best horizon | not selected | best proven-optimal and best feasible `H2`, ranked by `J_op/H2` | by `J_op/H2` or by the projected cost over an evaluation horizon |
| Early stop | no | optional, on the gradient of `J_op/H2` | through the same options |

- `main`'s `sweep_formulation_dimensions` counts variables and rows analytically. It builds no Gurobi model.

### 10.3 Validation

- `lore` has no result validator. It checks inputs in `_parse_and_validate`, and the plotter checks the status.
- `main` has a legacy `validate()`, a Streamlit dashboard (`validator_dashboard.py`) and two model dashboards.
  Streamlit is a Python library for web dashboards.
- The legacy tools cannot read `main`'s current output, as `main`'s README warns. Examples:
  - they require the key `alpha`;
  - they read `H` as one integer;
  - they expect `u` with shape (2H,) and `z` with shape (F, 2H).
- The four Gamma validators are current.

### 10.4 Tests

| Layer | `lore` | `main` |
|---|---|---|
| pytest | `test/` exists only outside git (`test/*` is ignored); content unknown | `tests/`, 4 files: Gamma tail cap, Gamma transitions, two legacy validator tests |
| Scenario runs | `remote_smoke_test.py`: the pytest suite plus 8 solve scenarios; needs the untracked `input/` files | 30 `check_*.py` regression scripts; each prints `PASS` and exits with a nonzero code on failure; 4 are missing from `RegressionREADME.md` |
| How to run | `pip install -e ".[dev]"`, then `cd test && pytest -v` | `pip install -e ".[dev,dashboard]"`, then `pytest tests/` from the repository root |

- On `main`, a bare `pytest` also collects `test_rainflow.py` and `test_sparse_version.py`. These are experiment
  harnesses (interpretation: pytest would fail on their positional arguments).

### 10.5 Cluster jobs and experiments (only on `main`)

- **Euler** (the ETH Zürich cluster) jobs use Slurm, a job scheduler. They form two families:
  - **Family A**: `setup_euler.sh`, `submit*.sh` and their arrays. It calls `test.py`, which commit 1f31386
    (25 Sep 2026) renamed to `test_rainflow.py`, so family A fails on a fresh clone.
  - **Family B**: study jobs that call `examples/regression/run_*.py` and the experiment runners.
- **`experiments/`**: 9 folders.
  - The cases are synthetic, modelled on VBZ lines 161 and 162 and the Hardau garage.
  - Sizes go up to `F = 65`, `M = 13`, `L = 4`, `H = [4, 48]`.
  - The studies cover the Gamma horizon (`H1`, `H2` grids), feasibility transfer, and the greedy warm start.
- **Results in git**: only two Gamma-only horizon sweeps (`F = 4`, `M = 1`, `L = 1`, `H1 = 4`).
  - They are proven optimal up to `H2 = 16`, where `J_op/H2 = 0.6175`.
  - From `H2 = 20` every case stops at the time limit, with gaps of 22-63 %.

### 10.6 Data files

| Kind | `lore` | `main` |
|---|---|---|
| `input/` | none (`input/*` ignored) | 5 files; 3 use the old Gaussian format without `model:`, 1 Gamma input, 1 year input that sets the removed key `depot_capacity` |
| Other scenario inputs | none | 11 in `examples/regression/`, 12 in `experiments/` |
| `results/` | `output.yaml`, `schedule.png` | the same two files, plus 28 others (old Gaussian runs, validator fixtures and reports, convergence sweeps) |

- `results/output.yaml` and `results/schedule.png` are byte-identical on both branches and at the common ancestor.
  They use the old Gaussian format. The current plotter of either branch cannot read them.

## 11. Documents

- **`main` changed no `.tex` or `.bib` file after the common ancestor.** Every difference in the LaTeX files comes
  from `lore`.

| File | `lore` | `main` |
|---|---|---|
| `spec/spec.tex` | version 0.5, 2782 lines | "Version 0.2 (revised)", 1867 lines; version 0.3 in content |
| `spec/main.tex` | deleted (7 Jul 2026) | present: a literature review from May 2026 on Gamma, Wiener and IG; not a specification |
| `spec/review_report.tex` | deleted (7 Jul 2026) | present: an automated five-role panel review of that literature review (13 May 2026), verdict "major revision" |
| `scoping.tex`, `scoping.pdf` | revised (see below) | as at the common ancestor |
| `ASSESSMENT.md` | present (gap analysis, 7 Jul 2026) | absent |
| `Things to change.txt` | present (planned changes, 30 Jul 2026) | absent |
| `lore_check.txt` | deleted (30 Jul 2026) | present (research notes from May 2026, partly in Italian); its content is unrelated to `Things to change.txt` |
| `pyproject.toml` | Python >= 3.9 | Python >= 3.10; adds `pandas` and the extra `dashboard = ["streamlit"]` |
| `.gitignore` | 10 lines | 45 lines; adds `tests/*`, `results/*`, `docs/*`, `mixed_plotter.py` and others |
| `.vscode/settings.json` | absent | present (editor defaults) |

What `lore`'s spec version 0.5 changes relative to the version on `main`:

- It adds the rainflow model, which assumes no probability distribution. Its reliability is certified by a Cantelli
  or Bernstein bound (`tail_bound`), each in an exact and a tangent form.
- It renames the Gaussian and Wiener exact form from "rotated SOC" (second-order cone) to "nonconvex quadratic". The
  input token `socp` becomes `exact`, and `socp` stays as a deprecated alias.
- It removes the soft loop penalty `C_P`. Only the hard loop constraint remains, with a proof by domination of the
  moments. Version 0.2 used first-order stochastic dominance (FSD) instead.
- It adds loop rows for the ARA1 anchors, states that an idle day is a row of zeros, and removes `alpha_target`.
- It adds six open items, among them schedule-dependent Bernstein support and multi-tangent tightening.
- The horizon structure is the same in both versions: two halves of length `H`.

What `lore` changes in `scoping.tex`:

- It adds a maintenance-day variable `y_ik`, so that `C_M` becomes a real decision cost.
- It adds two subsections: refinements of the Bernstein and tangent rows, and random fatigue life in the
  Palmgren-Miner rule.
- It corrects several formulas, for example the IG closure exponent and the ordering table for compound Poisson
  processes.
- The scoping document and the spec on `lore` use two different idle-day rules. The scoping has an equality
  constraint plus `y_ik`. The spec has an inequality, where an idle day is a row of zeros. Both make `C_M` a real
  decision cost.

READMEs:

| Topic | `lore` README | `main` README |
|---|---|---|
| Source of truth | `spec/spec.tex` version 0.5 | the code (`base.py`, `rainflow.py`, `solver.py`) |
| Broken references | `test/TEST_DOCUMENTATION.md`, `input/data_example*.yaml` (ignored by git) | `docs/PROJECT_LAYOUT.md`, `euler/README.md` (never committed); "bound equations under `spec/`" (no file there contains them) |
| Internal mismatch | none found | the quick start calls `plot_management`, but the API section documents `plot_mixed_management`; the plot description promises a `D` (depot) mark that no code draws; the YAML example sets `C_P`, which the modular path ignores |

## 12. Problems found during the comparison

Ranked by consequence. **Checked** means verified in the repository. **Derived** means worked out by hand from the
constraints. **Read** means read in the code but not re-checked.

1. **`lore`: an ARA1 repair cannot lower the tracked state (derived).** The repair-cost row
   `z <= rho·(s[k-1] - a[k-1])` has no repair-day condition, and `z >= 0` (lore:models/base.py:85-86;
   lore:solver.py:523). So `s[k-1] >= a[k-1]` must hold at every step. After a repair on day `k`, the anchor is
   `a[k] = s[k-1]`, the state before the repair. The row on day `k+1` then forces `s[k] >= s[k-1]`.
   - Consequence: the repair removes nothing, but it is still charged. The only exception is a repair on the last step.
   - Scope: Wiener and Gamma with ARA1.
   - `lore`'s own spec lists the pre-repair anchor convention as an open item. `main`'s post-repair latch avoids the
     problem.
2. **`main`: Gaussian and IG are accepted, then fail (checked).** `config.py` lists them as supported. Only `gamma`
   and `rainflow` register a cell builder, so the build raises `NotImplementedError` (main:degradation_model/base.py:149-156, 1187).
3. **`main`: the README quick-start input does not load (checked).** `input/data.yaml` has no `model:` key, which
   `load_config` requires. The year input `vbz_man12e_year.yaml` sets `depot_capacity`, which `load_config` rejects.
4. **`main`: the Gamma loop constraint omits the ARD1 latches (checked).** Only `A` and `mu` are compared
   (main:degradation_model/base.py:1053-1063). The rainflow code carries its latches (main:degradation_model/rainflow.py:624-639).
   Interpretation: the repeatability argument may not cover the full Gamma state.
5. **`main`: the Gamma calibration certifies repetitions of one increment type only (read, stated by `main` itself).**
   Schedules that mix increment types are not covered (interpretation).
6. **`main`: HDF5 input cannot load (checked).** The reader takes a fixed list of keys without `model`, `rho`, `C_D`,
   `bound_method` or `repair_model` (main:solver.py:239-277).
7. **`main`: cluster family A calls the deleted `test.py` (checked).**
8. **`main`: the legacy validator and dashboards cannot read the current output (read; `main`'s README says so).**
9. **`main`: silent defaults (checked).** An unknown `reliability_impl` value becomes `exact`. The default MIP gap is 12 % and there is no time limit.
10. **`main`: repository hygiene (checked).**
    - 42 tracked files match `main`'s own `.gitignore`, among them the current plotter `utils/mixed_plotter.py`, all of `tests/` and all of `results/`.
    - The plot of `utils/plotter.py` labels its y axis "Flight i".
    - `egg-info/SOURCES.txt` lists files that do not exist on `main`.
11. **`lore`: the test suite and the inputs are outside git (checked).** The README examples and
    `remote_smoke_test.py` cannot run from a clone.
12. **Both branches: stale files (checked).**
    - `src/fleet_management.egg-info/` is tracked.
    - `pyproject.toml` still says "Gaussian degradation via MILP".
    - `results/output.yaml` uses a format that neither current plotter reads.

## 13. Items of `Things to change.txt` that `main` already covers

`Things to change.txt` exists only on `lore`. The table checks each item against `main`.

| Item | On `main`? |
|---|---|
| Targets are electric vehicles, not trains | partly: inputs use vehicle components (for example tyres, motor insulation, traction battery); the README still says "Vehicle/train fleet" |
| Split the horizon into a transitory and an operating horizon | yes: `H: [H1, H2]`, commit 70d57f0 (3 Aug 2026), and the `H1 × H2` sweep functions |
| Plot: a label at the start of each row instead of border styles | yes: `plot_mixed_management` labels each row |
| Tighten the linear surrogates beyond one tangent | yes: `reliability_impl: pwl` |
| Schedule-dependent Bernstein support | no: the support is one constant per cell. `lore`'s scoping document and spec describe the method. |

## 14. Evidence basis

- **Read in full, both branches**: the core pipeline (`solver.py`, `config.py`, `__init__.py`,
  `degradation_model/base.py` on `main`; `solver.py`, `models/`, `maintenance/` on `lore`); the public model files;
  the plotters; `horizon_sweep.py`, `greedy_warm_start.py`, `formulation_size_sweep.py`, `tests/`,
  `RegressionREADME.md`; both spec diffs from the common ancestor; `spec/main.tex`; `spec/review_report.tex`; both
  scoping diffs; both READMEs; the project files.
- **Read in part**: on `main`, `validation/`, the dashboards, the historical rainflow studies, `legacy/`, the Gamma
  validators, `test_rainflow.py`, `test_sparse_version.py`, most `euler/` and `examples/regression/` scripts.
- **Not available**: `lore`'s `test/` and `input/` folders, which are not in git.
- **Not run**: no solve and no test. Gurobi is not installed on this computer.
