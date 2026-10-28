Status: done. All eight questions answered; length checked (223 lines). Next: none.

# C. Tooling, plotting, validation, tests, experiments and data files: `lore` vs `main`

Merge base: 462ac1d (25 Jun 2026). Evidence format: `branch:path:line`. Nothing was executed; all findings come from reading code and data with `git show`.

## Main findings

- **Same key, two meanings.** `H: [a, b]` is an interval of horizon lengths on lore and a transitory/operating split on main (section 2).
- **main has much more tooling**: a horizon-sweep study tool, three MIP-start mechanisms, a result-overview plot, 30 regression checks, Slurm jobs, 9 experiment folders and 3 further study jobs. lore has one plotter, a `VarHintVal` hint and a smoke test.
- **main's legacy validator and dashboards cannot read main's current output** (section 4b). The README warning is correct.
- **main's Slurm family A is broken on a clean clone**: `test.py` was renamed to `test_rainflow.py` on 25 Sep 2026 without updating the scripts (section 6a).
- **43 tracked files on main match main's own `.gitignore`**, including the current plotter `utils/mixed_plotter.py`. Two files cited by main's README were never committed (section 8).
- **The only tracked numerical results** on main are two Gamma-only horizon sweeps (section 6b). lore tracks one old Gaussian result, identical to main's.

## 1. Plotting

Evidence: read in full: lore `plotter.py` (306 lines), main `utils/plotter.py` (290), main `utils/mixed_plotter.py` (882).
lore exports `plot_management` (lore:src/fleet_management/plotter.py:36). main exports `plot_management`, `plot_mixed_management`, `plot_horizon_sweep` (main:src/fleet_management/__init__.py:4-8).

| Aspect | lore `plot_management` | main `plot_management` (`utils/plotter.py`) | main `plot_mixed_management` (`utils/mixed_plotter.py`) |
|---|---|---|---|
| Rows | one per train, split into `L` strips (lore:plotter.py:197-209) | same (main:utils/plotter.py:122-134) | one per (vehicle, component), `F·L` rows (main:utils/mixed_plotter.py:528-536) |
| Columns | `2H` days (lore:plotter.py:186,199) | `T+1`; column 0 is `mu_0` (main:utils/plotter.py:111-114) | `T+1`; first column "initial" (main:utils/mixed_plotter.py:494-495,553) |
| Colour | `mu/τ`, green to red, clipped (lore:plotter.py:194,202) | `mu/threshold`; threshold is `alpha`, else `tau` (main:utils/plotter.py:65-67) | `mu/τ`, green-yellow-red (main:utils/mixed_plotter.py:524-527) |
| Model encoding | border style per model, with legend (lore:plotter.py:27-33,263-273) | none | row label, only "Gamma" or "Remaining-life" (main:utils/mixed_plotter.py:708-723) |
| Action marks | mission number; gear = repair; "R" disc = replacement; "zzz" = idle (lore:plotter.py:224-245) | mission number; gear = any depot day; "zzz"; no replacement mark (main:utils/plotter.py:149-171) | per component `M_j`, `I`, `R` (repair), `P` (replacement) (main:utils/mixed_plotter.py:602-613) |
| Horizon divider | dashed line at `k = H` (lore:plotter.py:221) | line at `H1`, "transitory"/"operating" (main:utils/plotter.py:189-194) | line at `H1`, "initialization"/"operating" (main:utils/mixed_plotter.py:571-582) |
| Side information | title with `F, M, H, L`, objective (lore:plotter.py:275-277) | none | "Run summary" panel: status, objective, horizons, models, MIP gap, presolved sizes, runtime, nodes, repair and replacement counts (main:utils/mixed_plotter.py:616-677) |
| Non-optimal result | raises `ValueError` (lore:plotter.py:179-183) | no status check | plots any result with `mu_0`, `mu`, `x` (main:utils/mixed_plotter.py:419-426) |
| Multi-horizon file | one image per `H`, suffix `_H<h>` (lore:plotter.py:61-66) | not supported | not supported |
| Keys read | `status, F, H, M, L, tau, mu, x, x_m, x_r, model` (lore:plotter.py:185-192) | `F, M, L, alpha/tau, mu_0, mu, x, H1, H2` (main:utils/plotter.py:61-106) | adds `m, r, model_assignment, component_names, performance`; parses a sibling Gurobi log `runner.out` (main:utils/mixed_plotter.py:454-511,779-839) |
| HDF5 layout | groups `metadata`, `solution`, `parameters`, `H<h>` (lore:plotter.py:133-171) | flat (main:utils/plotter.py:234-267) | flat (main:utils/mixed_plotter.py:842-864) |

Defects on main (facts):
- main `plot_management` labels the y axis "Flight i" (main:utils/plotter.py:179).
- The README promises a `D` (depot) mark (main:README.md:180-181). No code path returns `D`; the colours `depot` and `mixed_intervention` are unused (main:utils/mixed_plotter.py:18-22,602-613).
- The README Quick start calls `plot_management` (main:README.md:44-50) but documents only `plot_mixed_management` (main:README.md:169-183), "current result visualisation" (main:README.md:313).

Horizon-sweep plot, main only: `plot_horizon_sweep` merges compatible sweep reports, one case per `H2` (main:utils/mixed_plotter.py:57-154). Panels: `J_op/H2` against `H2` with proven-optimal, time-limited and infeasible marks; MIP gap with the 5 % stopping line; variable and constraint counts (main:utils/mixed_plotter.py:256-379).

Interpretation: the branches plot different output schemas. Neither plotter reads the other branch's output without changes.

## 2. Horizon search

Evidence: lore `solver.py:59-170,560-687` read; main `horizon_sweep.py` (716 lines) and `formulation_size_sweep.py` (286 lines) read in full.

- lore: `H: [H_min, H_max]` means one solve per `H`, each with `2H` days (lore:src/fleet_management/solver.py:114-124).
- main: `H: [H1, H2]` means one solve with transitory `H1` and operating cycle `H2`, `T = H1 + H2` (main:README.md:67).
- An input written for one branch is read with a different meaning on the other.

| Aspect | lore: loop inside `solve()` | main: `sweep_operating_horizons` | main: `sweep_horizon_grid` |
|---|---|---|---|
| Entry point | `solve()` when `H` is a list (lore:solver.py:90-91) | exported function (main:src/fleet_management/__init__.py:2) | exported function (same line) |
| What varies | `H = H_min..H_max`, step 1 (lore:solver.py:664) | any list of `H2`; `H1` fixed (main:horizon_sweep.py:368-380) | Cartesian `H1 × H2` grid (main:horizon_sweep.py:582-587) |
| Case run | same parsed data, first `H` increments (lore:solver.py:382-383) | temporary YAML, then `solve()` (main:horizon_sweep.py:405-422) | one operating sweep per `H1` (main:horizon_sweep.py:599-617) |
| Parallel | `ProcessPoolExecutor` if `n_workers > 1` (lore:solver.py:675-685) | sequential | sequential |
| One case fails | caught, `status: "error: ..."`, loop continues (lore:solver.py:667-671) | no `try` around `solve()` (main:horizon_sweep.py:418-422) | same |
| Best horizon | none; dict keyed by `H` (lore:solver.py:660-687) | `best_proven_H2` (optimal, gap ≤ 1e-8) and `best_feasible_H2` by `J_op/H2` (main:horizon_sweep.py:99-129) | by `J_op/H2`, or `projected_evaluation_cost` (main:horizon_sweep.py:650-666) |
| Early stop | none | optional: positive relative gradient, or 2 flat gradients; both cases need gap ≤ 5 % (main:horizon_sweep.py:146-231,340-344) | passed through `**sweep_options` (main:horizon_sweep.py:566,616) |
| Output | result dicts in one file (lore:solver.py:95-98) | report YAML with cases, growth, timings; checkpoint after each case (main:horizon_sweep.py:234-328,522-537) | grid report; checkpoint after each `H1` (main:horizon_sweep.py:621-627,687-716) |
| Objective | plain objective per `H` | `objective_mode = operating_average`, `J_op/H2` (main:horizon_sweep.py:346,410) | with `evaluation_horizon E`: `J_init + (E−H1)·J_op/H2 + C_D·u` (main:horizon_sweep.py:610-614,690-694) |
| Input formats | YAML, JSON, HDF5 | YAML only (main:horizon_sweep.py:362-363) | YAML (main:horizon_sweep.py:578) |

- main's `sweep_formulation_dimensions` counts variables and rows analytically, varying one of `F, M, L, T`; it builds no Gurobi model and counts only the shared and Gamma parts (main:src/fleet_management/formulation_size_sweep.py:1-6,18-20,67-79,214-219).
- Interpretation: lore gives no ranking, and raw objectives grow with `H`, so they are not comparable directly. main normalises by `H2` and separates proven from time-limited cases. main stops the whole sweep on one solver exception; the checkpoint keeps finished cases.

## 3. Warm start

Evidence: lore `solver.py:560-687` read; main `greedy_warm_start.py` read in full (550 lines); main `degradation_model/base.py:1403-1603,1870-1882,2010-2055` read.

| Aspect | lore: `VarHintVal` hint | main: previous-case start | main: greedy start | main: relaxation start |
|---|---|---|---|---|
| Function | `_apply_warm_hint` (lore:solver.py:570-587) | `apply_binary_warm_start` (main:base.py:1403-1464) | `build_greedy_warm_start` (main:greedy_warm_start.py:421-550) | `apply_relaxation_warm_start` (main:base.py:1557-1603) |
| Source | previous horizon's optimal solution (lore:solver.py:666,673) | any result passed as `solve(..., warm_start=result)` (main:solver.py:16-21) | heuristic from the configuration alone | Gurobi `feasRelaxS` on a model copy (main:base.py:1578-1585) |
| Attribute | `VarHintVal`, a branching hint (lore:solver.py:584-587) | `Start`, a candidate incumbent (main:base.py:1433-1455) | `Start`, through the same function (main:greedy_warm_start.py:525-538) | `Start` (main:base.py:1600) |
| Variables | `x`, `x_m`, `x_r` on the first `min(2H_prev, 2H)` days | `x`, `m_rep`, `r_rep`, `idle`, `nb`; operating days repeated cyclically from the old `H2` (main:base.py:1421-1457) | `x`, `m`, `r`, `idle`, all `T` days | all binaries and integers |
| Continuous states | not set | Gurobi reconstructs them (main:base.py:1406-1409) | completed: fixed-binary copy solved (60 s), full solution copied back (main:base.py:1467-1529) | not set |
| Kept after a time limit? | no: a non-optimal result has `x = None` (lore:solver.py:614-619,576-577) | yes, if an incumbent exists (main:base.py:1875-1880; main:horizon_sweep.py:423-424) | n/a | used only if no other start (main:base.py:2032) |
| Restriction | forbids `n_workers > 1` (lore:solver.py:161-163) | not for legacy Gamma without `gamma_beta_bound` (main:solver.py:112-116) | none stated | option `relaxation_warm_start` |

Greedy heuristic (main:greedy_warm_start.py:445-523): each day, the hardest mission goes to the free vehicle with the smallest worst-component ratio; other vehicles go to the depot and repair above `0.70·τ`. A second pass adds operating-phase repairs until the replayed mean, variance and Gamma shape close the loop and meet the reliability cap (main:greedy_warm_start.py:342-418). If `F < M`, missing assignments are reported as "artificial" (main:greedy_warm_start.py:436-440).

- The greedy start never schedules a replacement: `r` stays zero (main:greedy_warm_start.py:448,391,408), so the report field `replacements` is always 0.
- It is not exported (main:src/fleet_management/__init__.py:1-24). Callers: two checks and two experiments (main:examples/regression/check_greedy_warm_start.py:8; main:experiments/vbz_greedy_warm_start/run_comparison.py:9; main:experiments/garage_hardau_strict_greedy/run_case.py:11).
- Interpretation: lore's hint is weaker; a hint guides search, a completed `Start` is a feasible point. lore copies day `k` to day `k`, so after `H` grows the old second half no longer lines up with the new one; main's cyclic mapping respects `[H1, H2]`.

## 4. Validation of results

Evidence: main `validation/validator.py` read in part (lines 1-300, 427-450, 584-600, 875-900, 1084-1322 of 1322); dashboards, `utils/model_registry.py` and Gamma validators read in part (headers, parsing, entry points). lore has no validator and no dashboard.

| Tool on main | Status | What it checks | Evidence |
|---|---|---|---|
| `validate(input, degradation, results)` | legacy | status; `x` binary; ≤ 1 activity per vehicle-day; `Σ_i x[i,j,k] = 1` for every `j` including depot `j = 0`; `u ≥ mu`; `mu[2H−1] ≤ mu[H−1]`; Gaussian: `v` loop rule and objective recomputation | main:validation/validator.py:28-246 |
| `validate_baseline_assignment_feasibility`, `validate_gamma_synthetic_diagnostic` | legacy | per-assignment threshold check against `alpha`; Gamma diagnostic on a synthetic schedule | main:validator.py:249-262,875-890 |
| `validator_dashboard.py` (Streamlit, a Python web-dashboard library) | legacy | modes: baseline Gaussian, "Gamma synthetic diagnostic", "Rainflow Inspector" | main:validator_dashboard.py:1-31,40-56 |
| `validation/dashboard_rainflow.py` | partly current | recomputes structural checks, reliability margins and the operating loop rule | main:validation/dashboard_rainflow.py:1-21,216-239 |
| Gamma replay, stochastic and exact validators | current (degradation-model area) | replay of saved states; Monte Carlo on a fixed schedule; exact reconstruction | main:RegressionREADME.md:76-87; main:gamma_utils/gamma_replay_validator.py:1-4 |

README claim "predates the new schema" (main:README.md:212-214,230): confirmed by code reading.

| Current output (main:README.md:191-206; main:base.py:1867-1868) | What `validate()` requires | Consequence |
|---|---|---|
| `tau` of shape `(F, L)`; `alpha` only if present (main:solver.py:350-351) | key `alpha` as one float (main:validator.py:1102-1116,1135) | `KeyError` |
| `degradation` = `rainflow`, `gamma` or `mixed` | `gaussian`, `inverse_gaussian`, `rainflow` (main:validator.py:24,76-80) | `gamma`, `mixed` rejected |
| input with `model:` (main:config.py:313-318) | parsed by the registry marked `# DEPRECATED`, which knows only Gaussian, inverse Gaussian, Gamma (main:utils/model_registry.py:8,17-21,339-343) | `rainflow` rejected |
| `H` may be `[H1, H2]` | `int(data["H"])` (main:validator.py:1124) | error |
| `u` scalar; `z` `(F, L, T)` | `u` `(2H,)`; `z` `(F, 2H)` (main:validator.py:1193-1199) | shape error |

- The Rainflow Inspector reads `tau` as one float and `H` as one integer (main:dashboard_rainflow.py:58,66,75), so per-cell `tau` or `H: [H1, H2]` also fails there.
- The `validate()` docstring lists a "capacity constraint" (main:validator.py:45) that the code lacks. The tracked `results/validation.yaml` still has `capacity_sum_mu_le_F_minus_M` (main:results/validation.yaml:25-28), so an older validator wrote it.
- Interpretation: the demand check includes the depot `j = 0` (main:validator.py:174-183). With no depot limit in main's current model (main:RegressionREADME.md:111), a valid schedule with 0 or 2 depot vehicles on a day fails it. The replay validator sums missions only (main:gamma_replay_validator.py:95-96).
- lore checks inputs in `_parse_and_validate` and the status in the plotter (lore:solver.py:107-170; lore:plotter.py:179-183). Its smoke test asserts status, models seen and the NaN pattern of `v` (lore:remote_smoke_test.py:257-273).

## 5. Tests and regression checks

Evidence: read in full: lore `remote_smoke_test.py` (323 lines), main `tests/*.py` (4 files, 454 lines), `RegressionREADME.md` (250 lines). Read in part: `examples/regression/*.py` (sizes, solver use; two checks in full), `test_rainflow.py` (4005 lines) and `test_sparse_version.py` (1123 lines) (docstrings, imports, functions).

| Layer | lore | main |
|---|---|---|
| pytest suite | untracked `test/`, git-ignored and absent here (lore:.gitignore:8). UNVERIFIED: content unknown | `tests/`: Gamma tail cap against `scipy.stats.gamma`, Gamma transitions (no licence needed); two legacy validator tests on Gaussian files |
| Solver scenarios | `remote_smoke_test.py`: `pytest -v test/`, then 8 scenarios (five-model fleet, ARA1 plus Bernstein, `lp`, quadratic penalty, forced replacement, sequential and parallel horizon loop, input-check failures) (lore:remote_smoke_test.py:62-72,109-309) | 30 `check_*.py`; each prints `PASS` and exits nonzero on failure (main:RegressionREADME.md:242) |
| Inputs | `input/data_example.yaml`, `input/data_example_loop.yaml` (lore:remote_smoke_test.py:113,186), ignored and absent | built inline or tracked in `examples/regression/` |
| Result | `smoke_test_output/`, `summary.json`; exit code always 0 (lore:remote_smoke_test.py:21-30) | stdout; nonzero exit on failure |
| Harnesses | none | `test_rainflow.py`: bound comparison, hypotheses H1-H3 (main:test_rainflow.py:1-40); `test_sparse_version.py`: sparse and loop assembly give the same model (main:test_sparse_version.py:1-45). Both use archived code in `legacy/` and `rainflow_v2` (main:test_rainflow.py:204-230) |

`check_*.py` groups (type from main:RegressionREADME.md:42-118,205-209): public interface (5, Gurobi); Gamma calibration and tail numbers (5, no Gurobi); Gamma dynamics and truth tables (6, Gurobi); validators (4, mixed); horizon, objective and formulation size (5, mixed); legacy backend (1). Four checks are not in the RegressionREADME: `check_greedy_warm_start`, `check_greedy_warm_start_acceptance` (Gurobi), `check_horizon_sweep_visualization`, `check_result_visualization`. Also unlisted: `report_gamma_validator_slide8.py`, `report_indicator_presolve.py`, `run_progressive_vbz_feasibility.py`, `gamma_replacement_truth_table_common.py`, the six `vbz_progressive/*.yaml`.

How to run on main:
- `pip install -e ".[dev,dashboard]"` (main:README.md:35). `gurobipy` is a hard dependency. Every test imports `fleet_management`, whose `__init__` pulls in `gurobipy` (main:src/fleet_management/__init__.py:3; main:solver.py:9; main:base.py:47). A licence is needed only for solves.
- `pytest tests/` from the repository root; the validator tests use root-relative paths and write reports into `results/` (main:tests/test_validator_positive_case.py:5-9; main:tests/test_validator_negative_cases.py:8-9,31).
- Regression checks: the PowerShell loop in main:RegressionREADME.md:27-35.
- Interpretation: a bare `pytest` in the root also collects `test_rainflow.py` and `test_sparse_version.py` (no pytest configuration exists). Their `test_*` functions take positional arguments (main:test_rainflow.py:1952; main:test_sparse_version.py:587), which pytest reports as missing fixtures.
- Stale: `bad_capacity.yaml` expects the failure `u_ge_mu` (main:tests/test_validator_negative_cases.py:18). `tests/` has no test for rainflow, mixed fleets, the sweep or the plotters; only `check_*.py` scripts cover them.

## 6. Cluster workflow and experiments

Evidence: `euler/setup_euler.sh` and `euler/submit.sh` read in full; 13 other `euler/*` files read in part; all 9 `experiments/*/README.md` read in full; experiment inputs and the two tracked sweep reports parsed. lore has no cluster scripts and no experiments.

Slurm is the job scheduler of Euler, the ETH Zürich cluster.

| Family | Files | Environment | Calls | Status |
|---|---|---|---|---|
| A. Harnesses | `setup_euler.sh`, `submit.sh`, `run_array`, `merge`, `submit_studies.sh`, `run_studies_array`, `merge_studies`, `submit_sparse.sh`, `run_sparse`, `submit_year.sh`, `run_year_array`, `merge_year` | `.venv-euler`, Gurobi 12.0.1 (main:euler/setup_euler.sh:15-19) | `test.py`, `run_studies.py`, `test_sparse_version.py`, `run_year.py` | broken on a clean clone (below) |
| B. Study jobs | `gamma_horizon_convergence`, `vbz_progressive_feasibility`, `zurich_mixed_formulation_counts`, all `experiments/*/*.sbatch` | `.venv`, Gurobi 13.0.0 (main:euler/zurich_mixed_formulation_counts.sbatch:18-22) | `examples/regression/run_*.py`, `report_mixed_formulation_size.py`, experiment runners | runnable if inputs exist; the Zurich job needs two never-committed inputs (main:euler/zurich_mixed_formulation_counts.sbatch:13-14) |

- `setup_euler.sh` and `submit.sh` stop if `test.py` is missing (main:euler/setup_euler.sh:26-29; main:euler/submit.sh:101-110). `run_year.py` and `run_studies.py` load `test.py` by path at import (main:run_year.py:61-80; main:run_studies.py:221-227).
- `test.py` was deleted by commit 1f31386 (25 Sep 2026, "file changed to test_rainflow.py"). `test_rainflow.py` begins "test.py -- experiment harness" (main:test_rainflow.py:2). The scripts were not updated. `submit_year.sh` calls `feas_oracle.py` only if present (main:euler/submit_year.sh:78-83); it was never committed.
- `submit.sh` runs a harness test as a Slurm array (default 20 shards, 10 at once) plus a merge job; it checks the environment, flags, CRLF endings and a dirty tree first (main:euler/submit.sh:69,96-154,191-231).

VBZ is the Zurich public transport operator (Verkehrsbetriebe Zürich). The READMEs state that sizes, increments and costs are synthetic (main:experiments/vbz_lines161_162_5pct/README.md:10-11; main:experiments/garage_hardau_strict_greedy/README.md:3-4,17-19).

| Study | Question | Case | Results in git? |
|---|---|---|---|
| `euler/gamma_horizon_convergence.sbatch` | `J_op/H2` against `H2`, Gamma only | `F=4, M=1, L=1, H1=4`; `H2 = 4..40` (1800 s each), then `44..64` (6000 s) | yes, 2 reports plus PNG: proven optimal up to `H2 = 16` (`J_op/H2 = 0.6175`); from `H2 = 20` every case hits the time limit, gaps 22-63 %; best feasible `H2 = 44` (0.5549) |
| `experiments/gamma_horizon_stabilization` | every `H2 ∈ {4,…,64}`, 55 min each, no early stop | `F=4, M=1, L=1`, Gamma | no |
| `experiments/gamma_initial_damage_horizon` | does a longer `H1` lower `J_op/H2`? (main:experiments/gamma_initial_damage_horizon/README.md:5-10) | `F=4, M=2, L=2`, Gamma plus rainflow | no |
| `experiments/gamma_h1_h2_daily_5400s` | grid `H1 ∈ {2,6,10}`, `H2 ∈ {8,…,20}`, 5400 s each (its `.sbatch`) | `F=4, M=2, L=2` | no; its README is byte-identical to the previous row's |
| `experiments/gamma_horizon_grid_final` | grid ranked by `J_init + (52−H1)·J_op/H2` (main:experiments/gamma_horizon_grid_final/README.md:6-23) | `F=4, M=2, L=2`, tangent Cantelli | no |
| `experiments/vbz_lines161_162_5pct`, `..._optimality_1h` | two-line case, 5 % gap; 1 h bound-focused run | `F=3` proven infeasible; `F=4, M=2, L=4/2, H=[4,12]` (main:experiments/vbz_lines161_162_5pct/README.md:15-31) | no; archive names in main:.gitignore:42-45 |
| `experiments/vbz_greedy_warm_start` | greedy start against none, 1 h each | `F=4, M=2, L=2` | no |
| `experiments/vbz_F8_feasibility_transfer` | does the feasible `F=6` schedule transfer to `F=8`? | `M=1, L=4, H=[4,48]` | the `F=6` start only |
| `experiments/garage_hardau_strict_greedy` | incumbent from the greedy start at scale | `F=65, M=13, L=4, H=[4,48]` | no |
| `euler/vbz_progressive_feasibility.sbatch` | feasibility ladder | `F=4, M=1, L=2` up to `F=16, M=5, L=4` | no |
| `euler/*year*` (family A) | one VBZ year under five solver strategies | `input/vbz_man12e_year.yaml`, `F=22, M=16` | no |

## 7. Data files

Evidence: `git ls-tree -r` of both branches; top-level keys of every tracked input and result; inputs parsed.

| Kind | lore | main |
|---|---|---|
| `input/` | 0 (`input/*` ignored, lore:.gitignore:4) | 5 |
| Other scenario inputs | 0 | 11 in `examples/regression/`, 12 in `experiments/`, 1 start result |
| `results/` | 2: `output.yaml`, `schedule.png` | 30: the same 2; 4 old Gaussian runs and validator reports; 5 convergence files; 10 corrupted fixtures and 9 reports in `validator_tests/` |

- `results/output.yaml` and `schedule.png` are byte-identical on lore, main and the merge base (both `git diff` runs are empty). The YAML is old Gaussian: `degradation: gaussian`, `alpha`; no `L`, `mu_0`, `tau`, `model`. By code reading, lore's plotter cannot read it (needs `tau`, `model`, `x_m`, `x_r`; lore:plotter.py:185-192), nor can main's `plot_management` (needs `mu_0`; main:utils/plotter.py:94).
- `results/validator_tests/validation_bad_*.yaml` are outputs that the negative test writes (main:tests/test_validator_negative_cases.py:31), committed to git.

| main input | Format | Used by | Runs with current `solve()`? |
|---|---|---|---|
| `data.yaml` | old Gaussian, no `model:` | README Quick start (main:README.md:47) | no: `model:` required (main:config.py:313-318) |
| `data_test_baseline.yaml` | old Gaussian | validator tests, dashboard default (main:validator_dashboard.py:158) | no; validator input only |
| `tiny.yaml` | old Gaussian | no reference found | no |
| `gamma_horizon_euler_convergence.yaml` | current, Gamma | `euler/gamma_horizon_convergence.sbatch:13` | yes (UNVERIFIED: not run) |
| `vbz_man12e_year.yaml` | current, rainflow | explicitly un-ignored (main:.gitignore:5); scripts use `vbz_man12e_year_solve` (main:euler/submit_year.sh:14) | no: `depot_capacity: 6` (main:input/vbz_man12e_year.yaml:34) is rejected (main:config.py:303-308) |

- The convergence input snapshots also set `depot_capacity: 1` (main:results/gamma_horizon_convergence_12483826_input.yaml:33); they predate commit 6c3b228 (9 Sep 2026, "Remove depot limit ...") and cannot be re-run as they are.

## 8. Repository hygiene

Evidence: main's `.gitignore` (45 lines) read in full; tracked files tested with `git check-ignore --no-index` in an empty scratch repository holding only main's `.gitignore`.

Tracked on main although main's `.gitignore` matches them: 43 files.

| Rule (main:.gitignore:line) | Tracked files |
|---|---|
| `input/*` (4); only `vbz_man12e_year.yaml` un-ignored (5) | 4 inputs: `data`, `data_test_baseline`, `gamma_horizon_euler_convergence`, `tiny` |
| `results/*` (6, repeated at 31) | all 30 files in `results/` |
| `tests/*` (11) | all 4 files in `tests/` |
| `mixed_plotter.py` (38) | `src/fleet_management/utils/mixed_plotter.py`: no slash, so it matches at any depth; this is the plotter the README calls current |
| `csv_to_plotter.py`, `fix_root_gap.py`, `run_year.py` (39-41) | the three top-level scripts |

Interpretation: the rules do not affect files already tracked. They silently exclude new tests, results or inputs from `git add .`.

Cited but absent from git (`git log --all -- <path>` gives no commit unless stated):

| Path | Cited at |
|---|---|
| `docs/PROJECT_LAYOUT.md` (and `docs/*` is ignored, main:.gitignore:28) | main:README.md:282-283,318 |
| `euler/README.md` | main:README.md:293-294 |
| `input/zurich_*_mixed.yaml` | main:README.md:288; Zurich job (section 6a) |
| `test.py` (deleted 25 Sep 2026, commit 1f31386) | family-A scripts, `run_year.py`, `run_studies.py` |
| `euler.md` | main:euler/setup_euler.sh:27 |
| `input/gamma_horizon_local.yaml`, `horizon_grid.yaml`, `gamma_horizon_euler.yaml` | main:examples/regression/RegressionREADME.md:124,148,196 |
| `input/gamma_horizon_euler_overnight.yaml` | key `input` of main:results/gamma_horizon_convergence_12533716.yaml |

Other stale items on main:
- `src/fleet_management.egg-info/SOURCES.txt` lists removed files (`src/fleet_management/test.py`, `tests/test_new_gamma.py`) and omits `mixed_plotter.py` and `horizon_sweep.py` (main:src/fleet_management.egg-info/SOURCES.txt:6,30). lore's lists the old flat layout (lore:src/fleet_management.egg-info/SOURCES.txt:4-5).
- `utils/benchmark.py` documents `python -m fleet_management.benchmark`; the module is `fleet_management.utils.benchmark` (main:utils/benchmark.py:3-8).
- `validator_dashboard.py` documents `streamlit run scripts/validator_dashboard.py` (main:validator_dashboard.py:4-6); there is no `scripts/` folder. The README path is correct (main:README.md:218).
- `lore_check.txt` (the user's research notes, partly in Italian) is still tracked on main, unchanged since commit 61be6c3 (26 May 2026). lore deleted it in commit 651aa71.
- `.vscode/settings.json` holds only editor defaults (`venv`, `pip`).
