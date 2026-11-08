# WP8, part `legacy`: inventory of earlier tests and scripts (task L1)

Status (7 Nov 2026): done (task L1, Sonnet). Groups 1, 2 and 3 are filled. Next: Claude fills the column "decision"
in every table; then tasks L2 (ports and rewrites) and L3 (smoke test).

Proposed actions (counts of rows):

| group | rows | proposed |
|---|---|---|
| 1 `lore` tests | 129 (125 test functions, `conftest.py`, the documentation, 2 scripts) | port 6, drop 123 |
| 2 `main` scripts and inputs | 103 rows (104 files) | keep 38, rewrite 38, remove 27 |
| 3 checks | 28 YAML inputs, 51 Python scripts | 0 of 28 inputs load as they are; 15 load after dropping the rejected keys |

The six proposed ports: the monotone-in-`eta` test of the IG cap; a free solve of a Gamma-only fleet; the YAML, JSON
and HDF5 solve-then-plot tests; the five-model solve, validate and plot test. Four items need the user, not only
Claude: the scalar `C_M` of five experiment inputs (the cost meaning changes), the fate of `starts/...result.yaml`
(a result of the old formulation), the fate of `input/vbz_man12e_year.yaml` (data of the dropped family A), and the
notes on the formulation in the READMEs of experiments.

## Decisions (Claude, 7 Nov 2026)

Every proposed action of the tables below is accepted, with these exceptions and details. Task L2 carries them out.

1. **Ports go to one file**, `tests/test_ported.py` (not `test_ported_lore.py`), inputs in `tests/data/ported/`: the six
   `lore` ports of group 1, and the two cases of `check_gamma_ard1_integration.py` (complete repair `rho = 1`;
   two consecutive repairs), each a Gurobi test with a hand-derived expected trajectory.
2. **`input/vbz_man12e_year.yaml`**: removed with family A (decision "Cluster jobs and experiments").
3. **Scalar `C_M`**: `vbz_greedy_warm_start/input.yaml` gets `C_M: 1.0` (both entries equal). The four inputs with
   different entries (`horizon_initial_damage_daily_5400s.yaml`, `horizon_grid.yaml`, `horizon_initial_damage.yaml`,
   `garage_hardau_strict_greedy/input.yaml`): the user chose the largest entry (7 Nov 2026): `C_M: 1.2` in the first three, `1.3` in the last (commit 47178b8).
4. **`replacement_as_new` with non-zero replacement inputs** (`matched_mixed_ard1.yaml`, `matched_mixed_ardinf.yaml`,
   `mixed_gamma_ard1_public.yaml`): keep `main`'s meaning. Read how `main` used the flag (`git show main:<path>` in the
   worktree; the solver and config files of `main`). If the flag overrode the values, drop the replacement inputs and
   keep the flag; if `main` used the values, set the flag to `false`. Record which in the progress file.
5. **READMEs**: `gamma_h1_h2_daily_5400s/README.md` is rewritten to describe its own grid (from its job script).
   `gamma_horizon_grid_final/README.md` and `vbz_lines161_162_5pct/README.md` get one dated sentence where they state
   results: "Obtained with `main`'s formulation before the merge (November 2026)." No other README changes.
6. **`run_f8_feasibility_transfer.py`**: the smaller rewrite (remove the estimator block, fix the three calls); keep its
   hand-built solve.
7. **`starts/vbz_F6_M1_L4_H2_48_result.yaml`**: kept as is (decision "Cluster jobs and experiments").
8. **`euler/*.sbatch`** (not in the inventory): L2 checks the two experiment jobs for copies of removed files and
   for removed keys, as `gamma_horizon_stabilization.sbatch`, and fixes them.
9. **`RegressionREADME.md`**: rewritten last, listing only the surviving scripts and inputs.
10. **Removals**: L2 lists every file to remove in `wp8_plan_legacy.md` section 4; Claude removes them at the commit.

- **Evidence basis.** Read in full: `remote_smoke_test.py`, `test_solver.py`, `test_gaussian.py`,
  `test_end_to_end.py`, `conftest.py`, `TEST_DOCUMENTATION.md`, the two `unit_test_*.py`, the first 144 lines of
  `test_plotter.py` (all of it), and the passages of the regression scripts quoted in the tables. Read in part
  (docstring, imports, function list, every call of the package and targeted passages): the other regression
  scripts, the experiment scripts, the job scripts (grep of their commands and diff against each other), the first
  30 lines of the READMEs, and the bodies of the dynamics tests in `test_inverse_gaussian.py`, `test_wiener.py`,
  `test_gamma.py` and `test_rainflow.py` (their names, docstrings and fixtures were read). The result file
  `experiments/vbz_F8_feasibility_transfer/starts/vbz_F6_M1_L4_H2_48_result.yaml` (11331 lines): top-level keys only.
  Group 3 was run locally on 7 Nov 2026 with `fleet_management.config.load_config` and the stub of
  `merge/wp45_work/wp5/stubs`; no Gurobi solve was run. "Smoke scenario N" means scenario N of the list in
  `wp8_plan_legacy.md` section 3.
- **Worktree.** `C:\Users\utente\Documents\GitHub\Fleet_Management-merge` (commit `fb6097a`, WP7 done). `lore`'s tests
  are in `C:\Users\utente\Documents\GitHub\Fleet_Management\test\`; `lore`'s `input/` folder is not in that checkout.
- **Merged suite already present** in the worktree `tests/` (14 files, 264 test functions). Its files are cited as
  "covered by `tests/...`" below. Other WP8 parts (`tests/test_oracle_unit.py`, `tests/test_spec_inputs.py`) are
  untracked and were not used as evidence.
- **Abbreviations.** G, IG, W, Ga, RF: the five models. "rejected key": a key of spec Rejected Keys (lines 500-525).
  "Removed" below means: not in the merged package (checked in `src/`).

## Group 2: `main`'s scripts and inputs (worktree)

### 2a `examples/regression/`

Columns: path; what it does; removed or renamed items it uses; still meaningful; proposed action; size of the
rewrite in lines (touched or new); decision. "Gurobi" = needs gurobipy and a licence.

| path | what it does | removed or renamed items it uses | still meaningful? | proposed action | rewrite (lines) | decision |
|---|---|---|---|---|---|---|
| `RegressionREADME.md` (250) | Index of the checks, run commands, maintenance rules | Names 7 files that do not exist (`check_gamma_repeated_calibration`, `check_gamma_randomized_properties`, `check_gamma_repair_legacy`, `check_gamma_incumbent_extraction`, `check_formulation_sweep`, `run_formulation_size_sweep`, `report_mixed_formulation_size`); the legacy Gamma backend and calibration; inputs `input/horizon_grid.yaml`, `gamma_horizon_local.yaml`, `gamma_horizon_euler.yaml` (absent) | Partly: run commands and maintenance rules | Rewrite last: list only the surviving scripts and inputs | ~120 | |
| `check_gamma_ard1_integration.py` (301) | Gurobi: forces repair and replacement on a Gamma ARD1 cell, checks mean, shape and latch trajectories, repeated and complete repair | Input `gamma_beta_bound`; result keys `gamma_beta_bound`, `gamma_shape_bound`, `gamma_shape_latch`, `gamma_mean_latch`; `extract_solution(ctx, cfg, ctx.model)` (now 2 arguments); builder hook `.extract` | No: the shape and latch states are removed. Mean, anchor and repair cost are covered by `tests/test_gamma.py::test_dynamics_and_cap` | Remove; port the complete repair (`rho = 1`) and consecutive repairs as 2 cases into `tests/` if not covered | ~40 (port) | |
| `check_gamma_ard1_replacement_truth_table.py` (100) | Gurobi: truth table of replacement-enabled Gamma ARD1 transitions, plus names and counts of the product-hull rows | Imports `gamma_diagnostics` (file removed); `context.extras["gamma"]`; hull row names (`qmu_gamma_...`, `gA_gamma_...`) | No: it audits the removed exact-hull rows | Remove | 0 | |
| `check_gamma_ard1_truth_table.py` (272) | Gurobi: truth table of no-replacement Gamma ARD1 (mission, idle, repair), hull row inventory | Same as above; `gamma_beta_bound`; result keys `gamma_shape_latch`, `gamma_mean_latch`; "initial latch 0" convention (now `anchor_mu_0`, default `mu_0`) | No | Remove | 0 | |
| `check_gamma_ardinf_replacement_truth_table.py` (102) | Gurobi: truth table of replacement-enabled Gamma ARD-infinity, hull inventory | Same (`gamma_diagnostics`, `extras["gamma"]["dynamics_formulation"]`) | No | Remove | 0 | |
| `check_gamma_big_m_formulation.py` (362) | Gurobi: product-hull rows of Gamma ARD1 and ARD-infinity, with and without replacement | `gamma_diagnostics`; inputs `gamma_beta_bound`, `gamma_calibration_method`, `tail_bound`; `extras["gamma"]`; reads `gamma_tail_bound_public.yaml` | No: hull rows are replaced by one-sided rows and the sharp cap | Remove | 0 | |
| `check_gamma_cell_integration.py` (122) | Gurobi: `solve_mixed` on a mixed Gamma/RF fleet with varying rates, checks rate shapes and result keys | Input `gamma_beta_bound`; result keys `gamma_calibration`, `gamma_tail_bound`, `gamma_shape_bound`, `gamma_beta_bound` | Partly: shape asserts (`cfg.gamma_beta.shape`) stay; covered by `tests/test_config.py::test_gamma_beta_shapes`; the solve by smoke scenario 1 | Remove | 0 | |
| `check_gamma_complexity_diagnostics.py` (141) | Gurobi: checks the fields of `report_gamma_complexity.build_complexity_report` (calibration seconds, LP counts) | Fields `calibration_seconds`, `calibration_lp_variables`; input key of the report's cases | No: no calibration step | Remove with `report_gamma_complexity.py` | 0 | |
| `check_gamma_convolution_quadrature.py` (70) | Numerical, no Gurobi: Moschopoulos tail against independent quadrature (`gamma_quadrature_reference.py`) | None; ran locally and finished | Yes; also in `tests/test_gamma_validators.py::test_series_matches_the_quadrature` | Keep as is | 0 | |
| `check_gamma_exact_validator.py` (170) | Gurobi: exact post-solve Gamma validator on a public solve and on a forced ARD-infinity repair | Inputs `gamma_beta_bound`, `gamma_calibration_method` (writes a `finite_count` input), `tail_bound`; result keys `gamma_shape_bound`, `gamma_tail_bound`; `extract_solution` 3 arguments; report keys `backend`, `minimum_conservativeness_margin` (wp7 plan) | Partly: exact validator stays (internal, spec); covered by `tests/test_gamma_validators.py::test_exact_*` on hand results | Remove | 0 | |
| `check_gamma_public_interface.py` (154) | Gurobi: public `solve` of `gamma_tail_bound_public.yaml`, checks result keys of the Gamma routing | Result keys `gamma_calibration`, `gamma_big_m_bound_strategy`, `gamma_dynamics_formulation`, `gamma_tail_bound`, `gamma_beta_bound`, `gamma_calibration_method` | No: it tests the routing away from a legacy backend, which is gone | Remove (public Gamma solve: smoke scenario 1) | 0 | |
| `check_gamma_repair_integration.py` (119) | Gurobi: forces one ARD-infinity repair on a Gamma cell, checks shape scaling | Input `gamma_beta_bound`; `ctx.extras["gamma"]` | No: covered by `tests/test_gamma.py::test_dynamics_and_cap` | Remove | 0 | |
| `check_gamma_replay_validator.py` (74) | Gurobi: solve `mixed_gamma_ard1_public.yaml`, replay-validate, corrupt `mu`, check rejection and `raise_on_failure` | Report key `maximum_errors["physical_mean"]` (now `tracked_mean`); the input has `gamma_beta_bound` | Yes: the only end-to-end solve-then-replay check | Rewrite: fix the input; rename the error key | ~4 (+ input) | |
| `check_gamma_stochastic_validator.py` (84) | Numerical, no Gurobi: fixed-schedule Monte Carlo, fixed seed, Clopper-Pearson bound | Input `gamma_beta_bound`; hand result has no `model` key | Partly: covered by `tests/test_gamma_validators.py::test_monte_carlo_*`, `test_clopper_pearson` | Remove | 0 | |
| `check_gamma_tail_bound.py` (145) | Numerical: `finite_count` calibrators and `moschopoulos_tail_probability` (wp1 item U4) | `calculate_tail_bound_parameters`, `calculate_profile_...`, `calculate_fleet_...`, `shapes_from_expected_damage` (removed in WP6); ran locally: `AttributeError` at line 39 | Only the Moschopoulos part (lines 56-89) | Rewrite: cut to the Moschopoulos part (decision U4) | ~35 kept, ~110 deleted | |
| `check_gamma_validator_edge_cases.py` (204) | Gurobi: exact validator edge cases (zero seed near the threshold, repeated repairs, complete repair) | Input `gamma_beta_bound`; `extract_solution` 3 arguments; report keys | Partly: covered by `tests/test_gamma_validators.py::test_exact_full_repair`, `test_exact_tail_edge_cases` | Remove | 0 | |
| `check_greedy_warm_start.py` (60) | Structure of `build_greedy_warm_start`; fallback case `F = 1`, `M = 2` | Imports `_simulate_cell` (removed); fields `idle`, `feasible_assignment`, `artificial_assignments`; the fallback case is now refused (`F > M`); input `gamma_beta_bound` | No: covered by `tests/test_warm_start.py::test_greedy_cases`, `test_source_and_report_keys` | Remove | 0 | |
| `check_greedy_warm_start_acceptance.py` (102) | Gurobi: Gurobi accepts the greedy start; `result["warm_start"]["applied"]` and completion | Inputs `gamma_beta_bound`, `gamma_calibration_method`; `report` fields of the old heuristic | Yes, as an idea; becomes smoke scenario 10 (`greedy_start`) | Remove; the scenario replaces it | 0 | |
| `check_horizon_sweep.py` (243) | Gurobi: operating-horizon sweep and grid: bounds, proven or feasible ranking, gradient stop | Imports `_annotate_latest_gradient` (removed); keyword `maximum_mip_gap=`; fields `best_H2`, `row["calibration"]`; inputs `gamma_beta_bound`, `gamma_calibration_method` | Partly: ranking and stop rules are in `tests/test_horizon_sweep.py`; a real solve is smoke scenario 9 | Remove | 0 | |
| `check_horizon_sweep_visualization.py` (114) | `plot_horizon_sweep` on hand reports; image not blank; extension reports | Line 16 checks `__module__ == "fleet_management.utils.mixed_plotter"` (file removed in WP7) | Partly: `tests/test_plotter.py::test_sweep_*` cover the plot | Rewrite: delete the module check (lines 16-18) | ~3 | |
| `check_matched_ard_inputs.py` (81) | Gurobi: solves the two matched inputs (ARD-infinity, ARD1), validates both ways | Both inputs: `gamma_beta_bound`, `gamma_calibration_method`, `C_P`; then `replacement_mu` must be 0 with `replacement_as_new: true` | Yes: numerical tables of the documents | Rewrite the two inputs; script as is | ~8 per input | |
| `check_mixed_gamma_ard1_public.py` (110) | Gurobi: public mixed Gamma/RF ARD1 solve, serialization, replay validation | Result keys `gamma_calibration`, `gamma_formulation`, `gamma_mean_latch`; input as above | Partly: duplicates `check_gamma_replay_validator.py` (README rule: prefer extending one check) | Remove | 0 | |
| `check_mixed_public_interface.py` (116) | Gurobi: public mixed Gamma/RF ARD-infinity solve and replay | Result keys `gamma_shape_bound`, `gamma_tail_bound`, `model_assignment`; input as above | Partly: same duplicate | Remove | 0 | |
| `check_operating_average_objective.py` (106) | Gurobi: `objective_mode` `operating_average`, checks `J_op / H2 + C_D u` from the result | Reads `gamma_tail_bound_public.yaml` (rejected keys); `costs["C_M"]` is now a scalar | Yes, but covered by smoke scenario 7 and `tests/test_base.py::test_objective_equals_the_cost_report` | Remove | 0 | |
| `check_result_visualization.py` (94) | Plot of a hand-built mixed result; image not blank | Imports `plot_mixed_management` (removed, WP7); `__module__` check; no key `model` in the hand result | No: covered by `tests/test_plotter.py::test_schedule_layout` | Remove | 0 | |
| `check_unrestricted_maintenance.py` (82) | Gurobi: no fleet-level depot capacity; `depot_capacity` input refused; two trains in the depot on one day | Inputs `gamma_beta_bound`, `gamma_calibration_method` (the refusal test sees these first); message test expects "removed", the message says "Remove it" | Yes (spec: depot days) | Rewrite: drop 2 keys; match "Remove it" | ~8 | |
| `gamma_quadrature_reference.py` (94) | Helper: independent quadrature of the tail of a sum of Gammas | None | Yes: cited by `tests/test_gamma_validators.py` | Keep as is | 0 | |
| `gamma_replacement_truth_table_common.py` (169) | Helper of the three truth-table checks and of `report_gamma_validator_slide8.py` | Imports `gamma_diagnostics`; `gamma_beta_bound`; `extras`; result keys `gamma_shape_bound` | No | Remove with its users | 0 | |
| `plot_optimization_progress.py` (115) | Plots incumbent, bound and gap from a result file or a Gurobi log | None: row keys equal those of `optimization_progress` in `base.py` | Yes | Keep as is | 0 | |
| `report_gamma_complexity.py` (122) | Gurobi: solves 3 inputs, writes a complexity and timing report | Result keys `gamma_calibration`, `gamma_formulation`; the 3 inputs are rejected | No: estimates of the removed formulation | Remove | 0 | |
| `report_gamma_validator_slide8.py` (309) | Gurobi: slide evidence: forced truth-table schedules, replay, Monte Carlo, corruption | Imports the truth-table helper; replay field `physical_mean`; input keys | No: one-off evidence for the old formulation | Remove | 0 | |
| `report_indicator_presolve.py` (424) | Gurobi: builds models, compares presolve of indicator rows with 3 `PreSOS1BigM` settings | `resolve_run_options(cfg, verbose=0, relaxation_warm_start=False)` (now `resolve_run_options(cfg)` only); needs real `GRB.GENCONSTR_*` | Yes: the flag `encoding: indicator` exists | Rewrite: set the options in `cfg.options` | ~10 | |
| `run_gamma_schedule_validation.py` (94) | CLI of `validate_gamma_schedule_files` (modes, repetitions, seed, confidence) | None: report keys read (`mode`, `deterministic`, `stochastic`, `schedule`) exist in `gamma_stochastic_validator.py` | Yes | Keep as is | 0 | |
| `run_horizon_sweep.py` (139) | CLI of `sweep_operating_horizons` and `sweep_horizon_grid`; table print | Print loop formats `None` of an error row with `:>11.3f` (`TypeError`, wp7 plan); everything else matches the sweep report keys | Yes: called by 5 job scripts under `experiments/` and by `euler/gamma_horizon_convergence.sbatch` | Rewrite: print `None` safely (lines 112-127) | ~8 | |
| `run_progressive_vbz_feasibility.py` (128) | Solves a ladder of inputs, validates each, writes a summary | None (reads result `objective`; calls `validate_gamma_schedule_files`) | Yes; its inputs are `vbz_progressive/*.yaml` | Keep as is | 0 | |
| `gamma_tail_bound_public.yaml` (59) | Input: uniform Gamma, varying rates, ARD-infinity, `H = [2, 3]` | `gamma_beta_bound`, `gamma_calibration_method`, `C_P`; loads after dropping them | Yes (used by 4 scripts; 2 of them are removed) | Rewrite or remove with its last user | ~3 | |
| `matched_mixed_ard1.yaml` (82) | Input: Gamma and RF Ca cells, ARD1, `replacement_as_new: true` | Same 3 keys; then `replacement_mu` must be 0 with `replacement_as_new: true` | Yes | Rewrite | ~8 | |
| `matched_mixed_ardinf.yaml` (82) | Same with ARD-infinity (differs in `repair_model` and 2 comment lines) | Same | Yes | Rewrite (same edits) | ~8 | |
| `mixed_gamma_ard1_public.yaml` (78) | Input: Gamma and RF, ARD1, `replacement_as_new: true` | Same as `matched_mixed_ard1.yaml` | Yes (used by `check_gamma_replay_validator.py`) | Rewrite | ~8 | |
| `mixed_gamma_rainflow_public.yaml` (73) | Input: Gamma and RF Ca, ARD-infinity | Same | Only for removed scripts | Remove with its users | 0 | |
| `vbz_progressive/vbz_00_reference.yaml` (37) | Input: F4 M1 L2, `H = [4, 12]`, Gamma and RF | `gamma_beta_bound`, `gamma_calibration_method`; loads after dropping | Yes (VBZ ladder rung 0) | Rewrite | 2 | |
| `vbz_progressive/vbz_01_full_horizon.yaml` (37) | Rung 1: `H = [4, 48]` | Same 2 keys | Yes | Rewrite | 2 | |
| `vbz_progressive/vbz_02_four_components.yaml` (37) | Rung 2: L4 | Same 2 keys | Yes | Rewrite | 2 | |
| `vbz_progressive/vbz_03_regional_fleet.yaml` (37) | Rung 3: F16 | Same 2 keys | Yes | Rewrite | 2 | |
| `vbz_progressive/vbz_04_three_missions.yaml` (37) | Rung 4: M3 | Same 2 keys | Yes | Rewrite | 2 | |
| `vbz_progressive/vbz_05_oberes_glattal.yaml` (37) | Rung 5: M5 | Same 2 keys | Yes | Rewrite | 2 | |

### 2b `experiments/`

Decision "Cluster jobs and experiments": experiment definitions stay; results computed before the merge are removed
(none is tracked inside these folders except the F6 start file). "Rejected keys" below always means
`gamma_beta_bound` and `gamma_calibration_method` unless another key is named. A per-component `C_M` list is an
error since the spec makes `C_M` a scalar (Costs and Objective, line 475): the choice of the scalar changes the
experiment and needs the user (marked "C_M choice"). Every job script copies only files that exist in the merged
`src/` unless noted.

| path | what it does | removed or renamed items it uses | still meaningful? | proposed action | rewrite (lines) | decision |
|---|---|---|---|---|---|---|
| `gamma_h1_h2_daily_5400s/README.md` (34) | Text of the 90-minute joint sweep | Text equals `gamma_initial_damage_horizon/README.md` (describes `H1 = {2..12}`, not this grid `H1 = {2, 6, 10}`, `H2 = {8, 12, 16, 20}`, 5400 s) | Text is wrong for this folder | Rewrite | ~20 | |
| `gamma_h1_h2_daily_5400s/gamma_h1_h2_daily_5400s.sbatch` (107) | Euler job: `run_horizon_sweep.py --h1 2 6 10 --h2 8 12 16 20 --warm-start`, summary, plot | None | Yes | Keep as is | 0 | |
| `gamma_h1_h2_daily_5400s/input/horizon_initial_damage_daily_5400s.yaml` (96) | Input: F4 M2 L2, Gamma and RF Ca, 5400 s, relaxation start | Rejected keys; `C_M: [1.0, 1.2]` (not a scalar) | Yes | Rewrite (C_M choice) | ~3 | |
| `gamma_h1_h2_daily_5400s/plot_results.py` (177) | Plots H1 sensitivity or the H1/H2 grid from a sweep report | None: case-row keys exist in the spec | Yes | Keep as is | 0 | |
| `gamma_h1_h2_daily_5400s/preflight_small_grid.py` (50) | Preflight: `load_config`, asserts on `mu_0`, replacement flag, time limit | None (needs the input fixed) | Yes | Keep as is | 0 | |
| `gamma_h1_h2_daily_5400s/summarize.py` (86) | CSV and YAML summary of a sweep report | None | Yes | Keep as is | 0 | |
| `gamma_horizon_grid_final/README.md` (96) | Text: grid `H1 = {2, 4, 6}`, `H2 = {8, 12, 16, 24}`, 52-period evaluation | Words "explicit-idle formulation", "tangent Cantelli" (describe `main`'s formulation) | Yes | Keep; add the dated note on the formulation if the decision asks | ~3 | |
| `gamma_horizon_grid_final/gamma_horizon_grid_final.sbatch` (176) | Euler job: preflight (asserts `cfg.models`, `tangent`, limits), grid with warm start and `--evaluation-horizon 52` | None | Yes | Keep as is | 0 | |
| `gamma_horizon_grid_final/input/horizon_grid.yaml` (83) | Input: F4 M2 L2, tangent Cantelli, `epsilon = 1e-4`, 32 threads | Rejected keys; `C_M: [1.0, 1.2]` | Yes | Rewrite (C_M choice) | ~3 | |
| `gamma_horizon_grid_final/plot_grid.py` (88) | Heat maps of projected cost and gap | None | Yes | Keep as is | 0 | |
| `gamma_horizon_grid_final/summarize_grid.py` (96) | CSV and YAML summary of a grid report | None | Yes | Keep as is | 0 | |
| `gamma_horizon_stabilization/README.md` (61) | Text: `H2 = 4, 8, ..., 64`, 55 min per case, 16 h | None | Yes | Keep as is | 0 | |
| `gamma_horizon_stabilization/gamma_horizon_stabilization.sbatch` (129) | Euler job: `run_horizon_sweep.py --h2-range 4 64 --h2-step 4 --maximum-stopping-gap 0.05`, plot | Lines 67-70 copy `gamma_diagnostics.py` and `utils/mixed_plotter.py` (both removed) under `set -euo pipefail`: the job stops | Yes | Rewrite: delete 4 lines | 4 | |
| `gamma_horizon_stabilization/input/gamma_horizon_stabilization.yaml` (42) | Input: F4 M1 L1, Gamma, ARD-infinity | Rejected keys (both); loads after dropping | Yes | Rewrite | 2 | |
| `gamma_horizon_stabilization/plot_result.py` (20) | `plot_horizon_sweep` on a report | Help text offers SVG; the plotter refuses SVG (wp7 plan) | Yes | Rewrite: fix 1 help line | 1 | |
| `gamma_horizon_stabilization/summarize_result.py` (76) | CSV and YAML summary of a sweep report | None | Yes | Keep as is | 0 | |
| `gamma_horizon_stabilization/runs/.gitignore`, `.gitkeep` | Keep the empty `runs/` folder, ignore outputs | None | Yes | Keep as is | 0 | |
| `gamma_initial_damage_horizon/README.md` (34) | Text: `H2 = 24` with `H1 = {2..12}`; grid with `H2 = {12..40}` | None | Yes | Keep as is | 0 | |
| `gamma_initial_damage_horizon/gamma_h1_diagnostic.sbatch` (108) | Euler job: `H1` sweep at `H2 = 24` | None | Yes | Keep as is | 0 | |
| `gamma_initial_damage_horizon/gamma_h1_h2_grid.sbatch` (108) | Euler job: `H1` by `H2` grid | None | Yes | Keep as is | 0 | |
| `gamma_initial_damage_horizon/input/horizon_initial_damage.yaml` (79) | Input: F4 M2 L2, `allow_replacement: false`, `evaluation_horizon: 52` | Rejected keys; `C_M: [1.0, 1.2]` | Yes | Rewrite (C_M choice) | ~3 | |
| `gamma_initial_damage_horizon/plot_results.py` (177) | Plots (differs in a few lines from the `gamma_h1_h2_daily_5400s` copy) | None | Yes | Keep as is | 0 | |
| `gamma_initial_damage_horizon/preflight.py` (50) | Preflight (differs in 3 asserts from `preflight_small_grid.py`) | None | Yes | Keep as is | 0 | |
| `gamma_initial_damage_horizon/summarize.py` (86) | Summary (differs slightly from the copy) | None | Yes | Keep as is | 0 | |
| `garage_hardau_strict_greedy/README.md` (19) | Text: F65 M13 L4, `H = [4, 48]`, greedy start with fallback | Mentions "feasibility-relaxation start generator" (still `relaxation_warm_start`) | Yes | Keep as is | 0 | |
| `garage_hardau_strict_greedy/garage_hardau_strict_greedy.sbatch` (99) | Euler job: preflight, `run_case.py`; copies `greedy_warm_start.py` | None | Yes | Keep as is | 0 | |
| `garage_hardau_strict_greedy/input.yaml` (74) | Input: F65 M13 L4, tangent, `tangent_ref: 0.32` | Rejected keys; `C_M` list of 4 | Yes | Rewrite (C_M choice) | ~3 | |
| `garage_hardau_strict_greedy/run_case.py` (166) | One-step tangent screen, greedy start, `solve` with the start, summary | `report.feasible_assignment` (removed; line 103); `cfg.options.get("tangent_ref")` (now the cell value `cfg.tangent_ref[i, l]`, so the default 0.5 is used silently); the screen copies the Cantelli tangent formula (spec: Linear Forms) | Yes | Rewrite: use `repeatability_feasible`; read `cfg.tangent_ref`; or replace the screen by the greedy report | ~25 | |
| `vbz_F8_feasibility_transfer/README.md` (85) | Text: F6 to F8 schedule transfer, 3 cases | None | Yes | Keep as is | 0 | |
| `vbz_F8_feasibility_transfer/inputs/vbz_F6_M1_L4_H2_48.yaml` (37) | Input F6 | Rejected keys; loads after dropping | Yes | Rewrite | 2 | |
| `vbz_F8_feasibility_transfer/inputs/vbz_F8_M1_L4_H2_48.yaml` (37) | Input F8 (matched) | Rejected keys | Yes | Rewrite | 2 | |
| `vbz_F8_feasibility_transfer/run_f8_feasibility_transfer.py` (432) | Builds F8 models, fixes or hints the F6 schedule, 3 cases, validates | Imports `gamma_diagnostics` (removed); `extract_solution(ctx, cfg, ctx.model)` (now 2 arguments); `ctx.extras["gamma"]["calibration_seconds"]`; `estimate_gamma_formulation`; result keys `gamma_formulation`, `model_assignment`, `backend`; builds the solve by hand instead of the warm-start API | Yes | Rewrite: remove the estimator block (~30 lines), fix 3 calls, or rebuild the hint and start cases on `solve_mapping(..., warm_start=...)` | ~100 | |
| `vbz_F8_feasibility_transfer/vbz_F8_feasibility_transfer.sbatch` (47) | Euler job | None | Yes | Keep as is | 0 | |
| `vbz_F8_feasibility_transfer/starts/vbz_F6_M1_L4_H2_48_result.yaml` (11331) | Old F6 result; the runner reads `F`, `objective`, `x`, `m`, `r` only | Result of the old formulation (keys `gamma_shape_bound`, `gamma_tail_bound`, `method`, `degradation`, `backend`); not an input of `load_config` | Kept by decision (regenerated when the experiment is re-run) | Keep; Claude decides whether to regenerate | 0 | |
| `vbz_F8_feasibility_transfer/runs/.gitkeep` | Keeps `runs/` | None | Yes | Keep as is | 0 | |
| `vbz_greedy_warm_start/README.md` (16) | Text: baseline against greedy start | None | Yes | Keep as is | 0 | |
| `vbz_greedy_warm_start/input.yaml` (54) | Input: F4 M2 L2, `tangent_ref: 0.32` | Rejected keys; `C_M: [1.0, 1.0]` | Yes | Rewrite (C_M choice: both values equal, so the scalar is 1.0) | ~3 | |
| `vbz_greedy_warm_start/run_comparison.py` (122) | Solves the input without and with the greedy start; summary | `greedy_report.feasible_assignment` (removed; line 75; the old report counted missions on artificial vehicles) | Yes | Rewrite: drop that check (the new report has `repeatability_feasible` only) | ~6 | |
| `vbz_greedy_warm_start/vbz_greedy_warm_start.sbatch` (79) | Euler job; preflight builds the greedy start | None | Yes | Keep as is | 0 | |
| `vbz_lines161_162_5pct/README.md` (57) | Text: cases F3 and F4, 5 % target, F3 proven infeasible | Statement "F = 3 proven infeasible" was found with the old formulation | Yes | Keep; add the dated note if the decision asks | ~3 | |
| `vbz_lines161_162_5pct/input/vbz_line162_H2_12_5pct.yaml` (68) | Input: line 162 control, F2 | Rejected keys; loads after dropping | Yes | Rewrite | 2 | |
| `vbz_lines161_162_5pct/input/vbz_lines161_162_F4_H2_12_5pct.yaml` (71) | Input: F4 retry | Same | Yes | Rewrite | 2 | |
| `vbz_lines161_162_5pct/input/vbz_lines161_162_H2_12_5pct.yaml` (71) | Input: F3 probe | Same | Yes | Rewrite | 2 | |
| `vbz_lines161_162_5pct/run_case.py` (58) | `solve` one case, JSON summary | None (all result keys exist) | Yes | Keep as is | 0 | |
| `vbz_lines161_162_5pct/summarize_cases.py` (94) | Combines the case summaries | None | Yes | Keep as is | 0 | |
| `vbz_lines161_162_5pct/vbz_line162_5pct_fallback.sbatch` (118) | Euler job: fallback case | None | Yes | Keep as is | 0 | |
| `vbz_lines161_162_5pct/vbz_lines161_162_5pct.sbatch` (125) | Euler job: primary case | None | Yes | Keep as is | 0 | |
| `vbz_lines161_162_optimality_1h/README.md` (32) | Text: tangent Cantelli, 32 cores, 1 h | None | Yes | Keep as is | 0 | |
| `vbz_lines161_162_optimality_1h/input/vbz_lines161_162_F4_L2_tangent_boundfocus_1h_32c.yaml` (65) | Input: F4 M2 L2, `tangent_ref: 0.32` | Rejected keys; loads after dropping | Yes | Rewrite | 2 | |
| `vbz_lines161_162_optimality_1h/run_case.py` (66) | `solve`, YAML summary | None | Yes | Keep as is | 0 | |
| `vbz_lines161_162_optimality_1h/vbz_tangent_optimality_1h_32c.sbatch` (145) | Euler job | None | Yes | Keep as is | 0 | |

### 2c `input/` and `remote_smoke_test.py`

| path | what it does | removed or renamed items it uses | still meaningful? | proposed action | rewrite (lines) | decision |
|---|---|---|---|---|---|---|
| `input/data.yaml` (51) | README quick start: F6 M3 L2, Gaussian, 10-day horizon | Old format: `alpha`, `xi`, `C_S`, `C_P` rejected; no `model`, `rho`, `C_rep`, `C_D`, `tau` | No: the merged reader needs another file | Remove; the README quick start uses a merged-format example (spec lines 2768-2827, planned file `examples/smoke/mixed_example.yaml`) | 0 (new file in L3) | |
| `input/data_test_baseline.yaml` (773) | Input of the two legacy validator tests (`tests/test_validator_*`, replaced by `tests/test_validator.py`) | Same old keys (`alpha`, `xi`, `C_S`, `C_P`) | No | Remove | 0 | |
| `input/gamma_horizon_euler_convergence.yaml` (51) | Euler horizon-convergence input: F4 M1 L1, Gamma, `H = [4, 4]`, `operating_average` | `gamma_calibration_method`, `gamma_beta_bound`; loads after dropping; used by `euler/gamma_horizon_convergence.sbatch` and the regression README | Yes | Rewrite: delete 2 keys and the comment on calibration | ~4 | |
| `input/tiny.yaml` (31) | Tiny Gaussian input, no reference found in the repository | Old format as `data.yaml` | No | Remove | 0 | |
| `input/vbz_man12e_year.yaml` (42) | Input of the family-A year jobs: F22 M16, RF Bernstein, `H = 26`, 2 components | Rejected `C_S`, `C_P`, `depot_capacity`; unknown keys `damage_capacity`, `_missions`; `C_D` missing | Belongs to family A, dropped by decision; the data itself is a realistic Bernstein input | Remove with family A (alternative: keep, 8-line fix) | 0 (8 if kept) | |
| `remote_smoke_test.py` (323) | Runs `pytest -v test/`, then 8 scenarios on `input/data_example.yaml` and `data_example_loop.yaml` (both absent) | Input keys `formulation`, `maintenance_type`, `n_workers`, `warm_start`; result keys `x_r`, `tail_bound`; multi-horizon results keyed by `H`; `solve(...)` with `H: [4, 6]` meant a horizon loop | Yes: the decision orders a port, extended to the merge's changes | Rewrite in full (task L3: 12 scenarios, inputs in `examples/smoke/`, job `euler/run_smoke.sh`) | ~350 new | |

## Group 1: `lore`'s tests (`Fleet_Management\test\`)

The folder has 8 test files (125 test functions, 7 of them parametrized), `conftest.py`, `TEST_DOCUMENTATION.md` and 2
scripts. All files target `lore`'s old API (`solver._parse_and_validate`, `solver._solve_single_horizon`,
`models.*.validate_inputs`, `solver._save_results(result, path, shared)`, results keyed by `H`). Inputs: "memory" =
a dict built by `conftest.tiny_problem_data` (old keys); "hand result" = a dict from `conftest.fake_result`;
`data_example.yaml` and `data_example_loop.yaml` are not in the checkout (`input/` of `lore` is absent). "Gurobi"
means a real solve. "Covered by" names a test of the worktree `tests/` that checks the same behaviour on the merged
keys. Columns: file; test function; what it checks; spec section; does the merged package still have that
behaviour; inputs; proposed action and reason; decision.

### 1a `test_solver.py` (43 functions; none needs Gurobi)

| file | test function | what it checks | spec section | merged package | inputs | action and reason | decision |
|---|---|---|---|---|---|---|---|
| solver | `test_broadcast_FL_scalar` | `_broadcast_FL(1.5, 3, 2)` gives shape `(3, 2)` | Input Specification, cell values | Removed helper; `config._cell_value` | none | Drop: `test_config.py::test_cell_value_shapes` | |
| solver | `test_broadcast_FL_exact_shape` | An `(F, L)` list passes unchanged | same | same | none | Drop: same test | |
| solver | `test_broadcast_FL_bad_shape_raises` | A list of 3 for `(2, 2)` raises | same | Yes (error names the key) | none | Drop: same test, error part | |
| solver | `test_broadcast_cost_FL_accepts_F_vector` | A cost of shape `(F,)` is broadcast to `(F, L)` | Costs and Objective | Changed on purpose: costs are scalar, `(L,)` or `(F, L)` (decision "Cost shapes and objective terms") | none | Drop: `test_config.py::test_cost_shapes` | |
| solver | `test_broadcast_increment_raw_3d_repeats_over_H` | Increments `(F, M, L)` repeat over `H` | Increment Profiles and the Phase Lookup | Changed on purpose: profiles `(L, M, H2)` and `*_trans` (decision "Input key names, shapes and axis order") | none | Drop: `test_config.py::test_profile_shapes` | |
| solver | `test_broadcast_increment_raw_4d_passthrough` | A 4-D increment array passes unchanged | same | same | none | Drop: same test | |
| solver | `test_broadcast_FL_nullable_handles_none` | `None` becomes `NaN` | Input Specification (null rules) | Yes | none | Drop: `test_null_where_key_applies`, `test_value_where_key_does_not_apply_warns` | |
| solver | `test_wrap_increment_periodic` | Increments repeat over the second half | Phase Lookup | Changed on purpose: transitory days then operating days (decision "Meaning of `H: [a, b]`") | none | Drop: `test_base.py::test_increment_follows_the_phase_lookup` | |
| solver | `test_wrap_increment_slices_period_for_smaller_H` | Interval `H`: slice the period, then double | same | Removed: `H` interval no longer means a horizon loop; sweeps do that | none | Drop: `tests/test_horizon_sweep.py` | |
| solver | `test_parse_scalar_H_example` | Parses `data_example.yaml`; shapes; `v_0` is `NaN` at IG | Input File Schema | Yes (`load_config`) | `data_example.yaml` (absent) | Drop: `test_config.py::test_example_loads` parses the spec example | |
| solver | `test_parse_interval_H_example` | `H = [4, 6]` gives period 6 and `warm_start` true | Sizes and Horizon | Changed on purpose: `H = [H1, H2]`, `warm_start` rejected | `data_example_loop.yaml` (absent) | Drop: `test_example_loads` (`H = [2, 3]`) | |
| solver | `test_epsilon_out_of_range_raises` | `epsilon` 0.5 and 0.0 raise | Consistency Checks | Changed on purpose: `epsilon` in (0, 1), below 0.5 at G and W cells (decision "Reliability level `epsilon`") | data_example | Drop: `test_tau_epsilon_rho` | |
| solver | `test_F_must_exceed_M` | `M = 10` with `F = 4` raises | Sizes and Horizon | Yes | data_example | Drop: `test_sizes` | |
| solver | `test_unknown_model_string_raises` | A model name `bogus` raises | Selectors | Yes | data_example | Drop: `test_selector_values` | |
| solver | `test_bad_maintenance_type_string_raises` | A bad `maintenance_type` value raises | Selectors; Rejected Keys | Changed on purpose: key rejected, `repair_model` in `ardinf`, `ard1` (decision "Names and set of repair laws") | data_example | Drop: `test_rejected_keys`, `test_selector_values` | |
| solver | `test_ara1_rejected_for_ard1_only_models` (3 params) | ARA1 on G, IG, RF raises | Repair Laws | Changed on purpose: both laws for all five models | memory | Drop: behaviour reversed; per-model tests run both laws | |
| solver | `test_ara1_accepted_for_wiener_and_gamma` (2 params) | ARA1 on W and Ga is accepted | Repair Laws | Yes (as `ard1`) | memory | Drop: `test_wiener.py::test_dynamics`, `test_gamma.py::test_dynamics_and_cap` | |
| solver | `test_lp_and_quadratic_incompatible` | `formulation = lp` with `penalty_type = quadratic` raises | Damage Penalty | Changed on purpose: `tangent` or `pwl` with the quadratic penalty and `C_D > 0` raises (decision "Linear forms of the reliability rows") | data_example | Drop: `test_quadratic_penalty` | |
| solver | `test_bad_formulation_raises` | `formulation = bogus` raises | Rejected Keys | Changed on purpose: key `formulation` rejected | data_example | Drop: `test_rejected_keys`, `test_selector_values` | |
| solver | `test_bad_penalty_type_raises` | `penalty_type = bogus` raises | Costs and Objective | Yes | data_example | Drop: `test_selector_values` | |
| solver | `test_warm_start_requires_single_worker` | `warm_start` with `n_workers = 2` raises | Rejected Keys | Removed: both keys rejected, they are sweep options (decision "Horizon sweep") | data_example | Drop: `test_rejected_keys`; `tests/test_horizon_sweep.py` | |
| solver | `test_mu_0_must_be_below_tau` | `mu_0 = 1.5` raises | Consistency Checks | Yes | data_example | Drop: `test_mu_0_range` | |
| solver | `test_socp_alias_warns_and_maps_to_exact` | `formulation = socp` warns and maps to `exact` | Rejected Keys | Removed (key rejected) | data_example | Drop | |
| solver | `test_H_interval_validation` | `H = [6, 4]`, `[0, 4]`, `[4, 5, 6]` raise | Sizes and Horizon | Changed on purpose: `[6, 4]` is valid (`H1 = 6`, `H2 = 4`) | data_example_loop | Drop: `test_sizes` | |
| solver | `test_mu_not_required_for_all_gamma_fleet` | `mu` may be missing in a Gamma-only fleet | Input Specification | Changed on purpose: `mu` required at every cell (Gamma uses `mu` and `gamma_beta`; decision "Gamma formulation") | memory | Drop: `test_missing_required_keys` | |
| solver | `test_mu_required_when_non_gamma_present` | `mu` missing with a Gaussian cell raises | same | Yes | memory | Drop: same test | |
| solver | `test_v_inc_not_required_for_wiener_only` | `v` not required in a Wiener-only fleet | Keys by Cell Type | Yes (W has `sigma`, no `v`) | memory | Drop: `test_cell_type_required_keys` | |
| solver | `test_wiener_requires_sigma` | `sigma` missing raises | Keys by Cell Type | Yes | memory | Drop: same test | |
| solver | `test_wiener_sigma_must_be_positive` | `sigma < 0` raises | Consistency Checks | Yes | memory | Drop: `test_wiener_inverse_gaussian_constants` | |
| solver | `test_wiener_v0_must_be_strictly_positive` | `v_0 = 0` raises for W | Consistency Checks | Changed on purpose: `v_0 >= 0` (decision "Input checks") | memory | Drop: same test accepts `v_0 = 0` | |
| solver | `test_gamma_requires_beta_and_alpha_inc` | `beta` and `alpha_inc` missing raise | Keys by Cell Type; Rejected Keys | Changed on purpose: `gamma_beta` (a rate) required, `beta` and `alpha_inc` rejected | memory | Drop: `test_cell_type_required_keys`, `test_rejected_keys` | |
| solver | `test_gamma_beta_must_be_positive` | `beta < 0` raises | Consistency Checks | Changed on purpose: `gamma_beta > 0` | memory | Drop: `test_gamma_rates` | |
| solver | `test_gamma_alpha_inc_must_be_positive` | `alpha_inc < 0` raises | Consistency Checks | Removed (`alpha_inc` rejected) | memory | Drop | |
| solver | `test_rainflow_tail_bound_defaults_to_cantelli` | Missing `tail_bound` becomes Cantelli | Selectors | Yes (`bound_method`) | memory | Drop: `test_selector_defaults` | |
| solver | `test_rainflow_bad_tail_bound_string_raises` | `tail_bound = bogus` raises | Selectors | Yes | memory | Drop: `test_selector_values` | |
| solver | `test_tail_bound_at_non_rainflow_position_warns` | A `tail_bound` at a G cell warns | Input Specification | Yes (warning for a value where the key does not apply) | memory | Drop: `test_value_where_key_does_not_apply_warns` | |
| solver | `test_rainflow_bernstein_requires_b_fields` | Bernstein without `b_inc` raises | Keys by Cell Type | Yes (`support`) | memory | Drop: `test_cell_type_required_keys` | |
| solver | `test_rainflow_bernstein_bhatia_davis_v_inc_violation` | `v` above `mu (b - mu)` raises | Consistency Checks | Changed on purpose: Popoviciu `v <= support^2/4` (decision "Rainflow tail bounds") | memory | Drop: `test_bernstein_popoviciu_increments` | |
| solver | `test_rainflow_bernstein_tau_precondition` | `tau <= 2 kappa b / 3` raises | Consistency Checks | Yes | memory | Drop: `test_bernstein_states_and_tau` | |
| solver | `test_v_max_user_applies_to_wiener_and_rainflow` | `v_max_user` is kept at W and RF cells | Input Specification | Yes | memory | Drop: `test_v_max_user` | |
| solver | `test_save_and_read_roundtrip_single_horizon` (3 params) | Result round trip in YAML, JSON, HDF5 | Result Files | Changed on purpose: `_save_results(result, path)`, keys `m`, `r`, `idle`, reader `read_result` (decision "Output schema and file formats") | hand result | Drop: `tests/test_io.py::test_yaml_round_trip`, `test_json_round_trip`, `test_hdf5_round_trip` | |
| solver | `test_save_and_read_roundtrip_multi_horizon` (3 params) | Result keyed by `H`, three formats | Result Files | Removed: one result per solve; sweeps have their own report | hand result | Drop | |
| solver | `test_non_optimal_status_serializes_with_null_arrays` | An infeasible result is saved with null arrays | Result Files | Yes | hand result | Drop: `test_io.py::test_no_solution_text_files`, `test_no_solution_hdf5` | |

### 1b `test_gaussian.py` (12 functions), `test_inverse_gaussian.py` (10), `test_wiener.py` (13), `test_gamma.py` (13)

The five `validate_inputs` functions (per-model checks of `lore`) no longer exist: the checks are in
`config._check_state`, `_check_increments`, `_check_models` and `_check_ranges`. The dynamics tests of `lore` solve a
free schedule and check that the tracked state is not below a recomputed recursion. The merged tests fix the schedule
and minimise the states, so they check equality on a hand case, for both repair laws and both encodings.

| file | test function | what it checks | spec section | merged package | inputs | action and reason | decision |
|---|---|---|---|---|---|---|---|
| gaussian | `test_validate_inputs_happy_path` | `validate_inputs` accepts valid arrays | Consistency Checks | Removed function | arrays | Drop: `test_config.py::test_example_loads` | |
| gaussian | `test_validate_inputs_rejects_ara1` | ARA1 on G raises | Repair Laws | Changed on purpose: both laws for all models | arrays | Drop: behaviour reversed | |
| gaussian | `test_validate_inputs_rejects_rho_out_of_range` | `rho = 1.5` raises | Consistency Checks | Yes (`rho` in (0, 1]) | arrays | Drop: `test_tau_epsilon_rho` | |
| gaussian | `test_validate_inputs_rejects_3sigma_violation_mu0` | `mu_0 < 3 sqrt(v_0)` raises | Consistency Checks | Yes | arrays | Drop: `test_gaussian_three_sigma_states` | |
| gaussian | `test_validate_inputs_rejects_mu_ge_tau` | `mu_0 >= tau` raises | Consistency Checks | Yes | arrays | Drop: `test_mu_0_range` | |
| gaussian | `test_validate_inputs_skips_entries_outside_mask` | A bad value outside the mask is ignored | Input Specification | Yes (keys read only where they apply) | arrays | Drop: `test_value_where_key_does_not_apply_warns` | |
| gaussian | `test_gaussian_mean_dynamics_dominate_recursion` | Gurobi: solved `mu` is not below the ARD1 recursion | Gaussian Model; Repair Laws | Changed on purpose: laws `ardinf` and `ard1` with anchors (decision "Anchor of the repair law") | memory | Drop: `test_gaussian.py::test_dynamics` (hand case, equality) | |
| gaussian | `test_gaussian_loop_constraint_holds` | Gurobi: `mu`, `v` at `2H-1` not above `H-1` | Loop Constraint | Changed on purpose: days `T-1` and `H1-1`, also the anchors (decision "States covered by the loop constraint") | memory | Drop: `test_gaussian.py::test_free_schedule` | |
| gaussian | `test_gaussian_reliability_constraint_holds_exact` | Gurobi: `mu + Phi^-1 sqrt(v) <= tau` on the solution | Gaussian Model, reliability row | Yes | memory | Drop: `test_free_schedule` (same inequality squared) | |
| gaussian | `test_gaussian_lp_formulation_solves` | Gurobi: `formulation = lp` solves; tangent row holds | Linear Forms of the Reliability Rows | Changed on purpose: `reliability_impl` `tangent` with `tangent_ref`, also `pwl` | memory | Drop: `test_gaussian.py::test_reliability_forms`, `test_base.py::test_gaussian_reliability_forms` | |
| gaussian | `test_quadratic_penalty_solves` | Gurobi: quadratic penalty, `u >= mu^2` | Damage Penalty | Changed on purpose: quadratic penalty needs `reliability_impl` `exact` | memory | Drop: `test_base.py::test_objective_equals_the_cost_report` (`u` equals the recomputed value) | |
| gaussian | `test_replacement_reset_is_enforced` | Gurobi, hand model: forced `x_r = 1` sets `mu`, `v` to `mu_new`, `v_new` | Replacement | Changed on purpose: `replacement_mu`, `replacement_v`; builds the model by hand with removed internals (`models.base`, `ctx.mu_inc`) | hand model | Drop: the replacement day is in the hand schedule of `test_gaussian.py::test_dynamics` | |
| inverse_gaussian | `test_solve_mu_bar_matches_scipy_invgauss` (5 params) | Tail of the IG at the cap equals `epsilon` (scipy) | Inverse Gaussian Model | Changed on purpose: the cap is `tau / E[Z given Z > q]`, so the tail is below `epsilon` (decision "Inverse Gaussian reliability cap"); function `ig_mean_cap` | none | Drop: `test_inverse_gaussian.py::test_inverse_gaussian_cap` | |
| inverse_gaussian | `test_solve_mu_bar_non_binding_at_extreme_eta` | At `eta = 1e5` the cap is about `tau` | same | Changed on purpose: the new cap at `eta = 1e5` is 0.0100 (checked locally) | none | Drop: the old statement is false for the new cap | |
| inverse_gaussian | `test_solve_mu_bar_never_unachievable_for_valid_epsilon` | A finite cap exists for `eta` from 1e-6 to 1e6 | same | Yes (checked locally: caps from 0.9973 to 0.0100) | none | Drop: `test_inverse_gaussian_cap` (range part) | |
| inverse_gaussian | `test_solve_mu_bar_is_below_tau` | `0 < cap < tau` | same | Yes | none | Drop: same | |
| inverse_gaussian | `test_solve_mu_bar_monotone_in_eta` | The cap falls when `eta` grows | same | Yes (checked locally: 0.570, 0.399, 0.229, 0.155 for `eta` 0.05, 0.15, 0.5, 1.0) | none | Port: rewrite for `ig_mean_cap`; not in `tests/` | |
| inverse_gaussian | `test_validate_inputs_happy_path` | Valid IG arrays pass | Consistency Checks | Removed function | arrays | Drop: `test_example_loads` | |
| inverse_gaussian | `test_validate_inputs_rejects_ara1` | ARA1 on IG raises | Repair Laws | Changed on purpose | arrays | Drop: behaviour reversed | |
| inverse_gaussian | `test_validate_inputs_rejects_nonpositive_eta` | `eta <= 0` raises | Consistency Checks | Yes | arrays | Drop: `test_wiener_inverse_gaussian_constants` | |
| inverse_gaussian | `test_validate_inputs_rejects_mu0_ge_tau` | `mu_0 >= tau` raises | Consistency Checks | Yes | arrays | Drop: `test_mu_0_range` | |
| inverse_gaussian | `test_ig_mean_dynamics_and_reliability_cap` | Gurobi: `mu` not below recursion, below the cap, loop holds, `v` is `NaN` | IG Model; Loop Constraint | Changed on purpose (as above; the cap is the CVaR cap) | memory | Drop: `test_inverse_gaussian.py::test_dynamics`, `test_cap_rows`, `test_free_schedule` | |
| wiener | `test_validate_inputs_happy_path_ard1` | Valid W arrays pass (ARD1) | Consistency Checks | Removed function | arrays | Drop: `test_example_loads` | |
| wiener | `test_validate_inputs_happy_path_ara1` | Valid W arrays pass (ARA1) | Repair Laws | Yes (as `ard1`) | arrays | Drop: `test_wiener.py::test_dynamics` | |
| wiener | `test_validate_inputs_rejects_bad_maintenance_type` | A bad `maintenance_type` raises | Selectors | Changed on purpose: key rejected | arrays | Drop: `test_rejected_keys` | |
| wiener | `test_validate_inputs_rejects_nonpositive_sigma` | `sigma <= 0` raises | Consistency Checks | Yes | arrays | Drop: `test_wiener_inverse_gaussian_constants` | |
| wiener | `test_validate_inputs_rejects_nonpositive_v0` | `v_0 <= 0` raises | Consistency Checks | Changed on purpose: `v_0 >= 0` (decision "Input checks") | arrays | Drop: same test accepts 0 | |
| wiener | `test_validate_inputs_rejects_mu0_ge_tau` | `mu_0 >= tau` raises | Consistency Checks | Yes | arrays | Drop: `test_mu_0_range` | |
| wiener | `test_wiener_ard1_mean_dominates_recursion` | Gurobi: `mu` not below ARD1 recursion | Wiener Model | Changed on purpose (laws) | memory | Drop: `test_wiener.py::test_dynamics` | |
| wiener | `test_wiener_variance_dominates_mission_only_recursion` | Gurobi: `v` grows by `sigma^2` per mission, only replacement resets it | Wiener Model | Yes | memory | Drop: `test_wiener.py::test_dynamics` (`v` of the hand case), `test_loop_needs_replacement` | |
| wiener | `test_wiener_ara1_mean_dominates_anchor_aware_recursion` | Gurobi: ARA1 mean with anchor, reset at replacement | Repair Laws | Changed on purpose: law `ard1`, anchor after the last intervention (decision "Anchor of the repair law") | memory | Drop: `test_wiener.py::test_dynamics` (law `ard1`, `g_mu`) | |
| wiener | `test_wiener_loop_constraint_holds` | Gurobi: loop rows incl. the ARA1 anchor | Loop Constraint | Changed on purpose (as above) | memory | Drop: `test_wiener.py::test_free_schedule` | |
| wiener | `test_wiener_reliability_constraint_holds_exact` | Gurobi: exact reliability on the solution | Wiener Model | Yes | memory | Drop: `test_free_schedule` | |
| wiener | `test_wiener_lp_formulation_solves` | Gurobi: `formulation = lp` | Linear Forms | Changed on purpose (`tangent`, `pwl`) | memory | Drop: `test_wiener.py::test_exact_form_is_quadratic`; G and W build their rows with one function, `base.add_quadratic_reliability_rows`, tested in `test_base.py::test_gaussian_reliability_forms`; no solve of a W cell with `tangent` or `pwl` exists in `tests/` (smoke scenario 3 does not include W either) | |
| wiener | `test_wiener_replacement_reset_is_enforced` | Gurobi, hand model: forced replacement resets `mu`, `v` | Replacement | Changed on purpose (as for G) | hand model | Drop: replacement days `(0, 0, 3)` and `(1, 0, 2)` in the hand schedule of `test_wiener.py` | |
| gamma | `test_solve_alpha_hat_matches_scipy_gamma` (5 params) | Gamma tail at the cap equals `epsilon` for scale `beta` | Gamma Model | Changed on purpose: rate `gamma_beta`, sharp cap `sharp_cap`, `one_rate_cap` (decision "Gamma formulation") | none | Drop: `test_gamma.py::test_one_rate_cap_reaches_the_level`, `test_sharp_cap_is_safe_and_sharp` | |
| gamma | `test_solve_alpha_hat_non_binding_warns_and_caps` | A warning when the cap does not bind | Gamma Model | Removed (no such warning) | none | Drop | |
| gamma | `test_solve_alpha_hat_is_below_alpha_bar` | `alpha_hat < tau / beta` | Gamma Model | Changed on purpose (shape cap replaced by the mean cap `mean_cap`) | none | Drop: `test_gamma.py::test_mean_cap` | |
| gamma | `test_validate_inputs_happy_path_ard1` | Valid Ga arrays pass | Consistency Checks | Removed function | arrays | Drop: `test_example_loads` | |
| gamma | `test_validate_inputs_happy_path_ara1` | Valid Ga arrays pass (ARA1) | Repair Laws | Yes (as `ard1`) | arrays | Drop: `test_dynamics_and_cap` | |
| gamma | `test_validate_inputs_rejects_bad_maintenance_type` | Bad `maintenance_type` raises | Selectors | Changed on purpose | arrays | Drop: `test_rejected_keys` | |
| gamma | `test_validate_inputs_rejects_nonpositive_beta` | `beta <= 0` raises | Consistency Checks | Changed on purpose (`gamma_beta`) | arrays | Drop: `test_gamma_rates` | |
| gamma | `test_validate_inputs_rejects_nonpositive_alpha_inc` | `alpha_inc <= 0` raises | Consistency Checks | Removed (`alpha_inc` rejected) | arrays | Drop | |
| gamma | `test_validate_inputs_rejects_mu0_ge_tau` | `mu_0 >= tau` raises | Consistency Checks | Yes | arrays | Drop: `test_mu_0_range` | |
| gamma | `test_gamma_mu_dominates_alpha_beta_recursion` | Gurobi: `mu = alpha beta` not below the shape recursion | Gamma Model | Changed on purpose: no shape state; `mu` is tracked | memory | Drop: `test_gamma.py::test_dynamics_and_cap` | |
| gamma | `test_gamma_reliability_cap_and_loop_constraint_hold` | Gurobi: shape below the cap, loop holds, `v` is `NaN` | Gamma Model; Loop Constraint | Changed on purpose (cap on `mu`) | memory | Port: one free solve of a Gamma-only fleet, `mu` not above `mean_cap`, loop holds (no free solve of a Ga cell in `tests/`) | |
| gamma | `test_gamma_lp_formulation_solves` | Gurobi: `formulation = lp` | Gamma Model | Changed on purpose: the Gamma row is one cap row, no `reliability_impl` | memory | Drop: `test_cap_binds` | |
| gamma | `test_gamma_replacement_reset_is_enforced` | Gurobi, hand model: forced replacement resets the state | Replacement | Changed on purpose (`gamma_beta_new`) | hand model | Drop: replacement day `(0, 0, 3)` in the hand schedule of `test_gamma.py` | |

### 1c `test_rainflow.py` (18 functions), `test_plotter.py` (9), `test_end_to_end.py` (7), other files

The merged `tests/test_rainflow.py` covers the five bounds (Markov, Cantelli, Hoeffding, Bernstein, Chernoff); `lore`
had two (Cantelli, Bernstein).

| file | test function | what it checks | spec section | merged package | inputs | action and reason | decision |
|---|---|---|---|---|---|---|---|
| rainflow | `test_support_constant_takes_max_of_all_three` | `_support_constant` is the maximum of `b_0`, `b_new` and `max(b_inc)` | Rainflow Model | Changed on purpose: keys `support`, `support_0`, `support_new`; the constant is derived in `config._derive` (decision "Rainflow tail bounds") | none | Drop: `test_config.py::test_derived_fields` | |
| rainflow | `test_validate_inputs_happy_path_cantelli` | Valid RF Cantelli arrays pass | Consistency Checks | Removed function | arrays | Drop: `test_example_loads` | |
| rainflow | `test_validate_inputs_happy_path_bernstein` | Valid RF Bernstein arrays pass | Consistency Checks | Removed function | arrays | Drop: same | |
| rainflow | `test_validate_inputs_rejects_ara1` | ARA1 on RF raises | Repair Laws | Changed on purpose: both laws | arrays | Drop: behaviour reversed | |
| rainflow | `test_validate_inputs_rejects_bad_tail_bound` | A bad bound name raises | Selectors | Yes | arrays | Drop: `test_selector_values` | |
| rainflow | `test_validate_inputs_bernstein_rejects_b_inc_below_mu_inc` | `b_inc >= mu_inc` | Consistency Checks | Removed: `config._check_increments` has no `support >= mu` rule | arrays | Drop | |
| rainflow | `test_validate_inputs_bernstein_rejects_bhatia_davis_v_inc` | `v_inc <= mu (b - mu)` | Consistency Checks | Changed on purpose: Popoviciu `v <= support^2/4` | arrays | Drop: `test_bernstein_popoviciu_increments` | |
| rainflow | `test_validate_inputs_bernstein_rejects_b0_below_mu0` | `b_0 >= mu_0` | Consistency Checks | Removed (as above) | arrays | Drop | |
| rainflow | `test_validate_inputs_bernstein_rejects_bhatia_davis_v0` | `v_0 <= mu_0 (b_0 - mu_0)` | Consistency Checks | Changed on purpose: rule of `test_bernstein_states_and_tau` | arrays | Drop: that test | |
| rainflow | `test_validate_inputs_bernstein_tau_precondition` | `tau > 2 kappa b / 3` | Consistency Checks | Yes | arrays | Drop: `test_bernstein_states_and_tau` | |
| rainflow | `test_validate_inputs_cantelli_ignores_missing_b_fields` | Cantelli needs no `b_*` | Keys by Cell Type | Yes (`support` only for Ho, Be) | arrays | Drop: `test_cell_type_required_keys` | |
| rainflow | `test_rainflow_cantelli_mean_and_variance_dominate_recursion` | Gurobi: `mu`, `v` not below the ARD1 recursion | Rainflow Model | Changed on purpose (laws, anchors) | memory | Drop: `test_rainflow.py::test_trajectories` (5 bounds, both laws, both encodings) | |
| rainflow | `test_rainflow_cantelli_reliability_and_loop_constraint` | Gurobi: reliability and loop on the solution | Rainflow Model; Loop Constraint | Yes (days `T-1`, `H1-1`) | memory | Drop: `test_rainflow.py::test_reliability_row_binds`, `test_base.py::test_loop_rows_cover_states_and_anchors` | |
| rainflow | `test_rainflow_bernstein_reliability_holds` | Gurobi: `mu + c + sqrt(c^2 + 2 kappa v) <= tau` | Rainflow Model, Bernstein row | Yes | memory | Drop: `test_bernstein_side_row_is_tau_minus_c`, `test_reliability_rows_carry_the_constants` | |
| rainflow | `test_rainflow_v_max_rf_matches_derivation_cantelli` | `kappa_c^2 V_max = tau^2` | Rainflow Model | Calls no package code (algebra in the test) | none | Drop: no code under test | |
| rainflow | `test_rainflow_v_max_rf_matches_derivation_bernstein` | `c^2 + 2 kappa V_max = (tau - c)^2` | Rainflow Model | Calls no package code | none | Drop: same reason | |
| rainflow | `test_rainflow_lp_formulation_solves_both_tail_bounds` | Gurobi: `formulation = lp` for Ca and Be | Linear Forms | Changed on purpose (`tangent`, `pwl`) | memory | Drop: `test_rainflow.py::test_reliability_row_binds`, `test_segment_binaries_and_nonconvex` | |
| rainflow | `test_rainflow_replacement_reset_is_enforced` | Gurobi, hand model: forced replacement resets `mu`, `v` | Replacement | Changed on purpose | hand model | Drop: replacement day in the hand schedule of `test_rainflow.py::test_trajectories` | |
| plotter | `test_plot_management_produces_file` (3 params) | PNG from a YAML, JSON or HDF5 result | Plotting API and Layout | Yes | hand result | Drop: `test_plotter.py::test_schedule_layout` (3 extensions) | |
| plotter | `test_plot_management_default_output_name` | Default file `output.png` | Plotting API | Yes | hand result | Drop: `test_schedule_defaults` | |
| plotter | `test_plot_management_pdf_extension` | PDF output | Plotting API | Yes | hand result | Drop: `test_plot_management_writes_png_and_pdf` | |
| plotter | `test_plot_management_rejects_unsupported_output_extension` | `.svg` raises | Plotting API | Yes | hand result | Drop: `test_plot_management_path_errors` | |
| plotter | `test_plot_management_missing_input_raises` | Missing input raises `FileNotFoundError` | Plotting API | Yes | none | Drop: same test | |
| plotter | `test_plot_management_rejects_non_optimal` | An infeasible result raises `ValueError` | Plotting API | Yes (message "missing keys: mu, x") | hand result | Drop: `test_plot_management_without_solution` | |
| plotter | `test_plot_management_multi_horizon_writes_one_file_per_H` | One file per `H` (`schedule_H2.png`) | Plotting API | Removed: one result per solve | hand result | Drop | |
| plotter | `test_plot_annotations_all_paths_render_without_error` | Marks: mission number, gear, wrench, both, idle, blank | Schedule Plot | Changed on purpose: marks `P`, `R`, `M_j`, `D`, `I` from keys `m`, `r`, `idle` (decision "Plotting") | hand result | Drop: `test_schedule_layout`, `test_schedule_summary` | |
| plotter | `test_border_styles_cover_all_five_models` | `_BORDER_STYLES` has the five models | Schedule Plot | Removed: the model name is in the row label, not in a border | none | Drop | |
| end_to_end | `test_yaml_end_to_end_mixed` | Gurobi: `solve` of a YAML mixed fleet, then `plot_management` | Data Flow of One Solve | Yes | memory, written to YAML | Port: rewrite on the spec example with a short horizon (public `solve` and `plot_management`, real solve) | |
| end_to_end | `test_json_end_to_end_gaussian_only` | Gurobi: JSON input and JSON output | Input File Schema | Yes | memory, written to JSON | Port: rewrite (real solve from JSON; reading is tested only without Gurobi in `test_io.py`) | |
| end_to_end | `test_hdf5_end_to_end_gaussian_only` | Gurobi: HDF5 input and output, plot | Input File Schema, HDF5 Layout | Changed on purpose: flat attributes and datasets (the lore writer helper is obsolete) | memory, written to HDF5 | Port: rewrite with an HDF5 input written by the new layout | |
| end_to_end | `test_solve_with_results_path_none_still_returns_dict` | `results_path=None` writes no file | Data Flow of One Solve | Yes | memory | Drop: `test_io.py::test_solve_writes_or_not` (stub; the real solve is in the tests above) | |
| end_to_end | `test_horizon_loop_sequential_warm_start` | Gurobi: `H = [2, 3]` solves two horizons with warm start; one plot per `H` | Meaning of `H`; Horizon Sweeps | Removed: `H = [2, 3]` is `H1 = 2`, `H2 = 3`; sweeps are `sweep_operating_horizons` | memory | Drop: `tests/test_horizon_sweep.py`; smoke scenario 9 does a real sweep | |
| end_to_end | `test_horizon_loop_parallel` | Gurobi: `n_workers = 2` | Horizon Sweeps | Removed: `n_workers` is a sweep option | memory | Drop: same | |
| end_to_end | `test_all_five_models_mixed_fleet_with_ara1_and_bernstein` | Gurobi: 6 trains with all five models, ARA1 on W and Ga, Bernstein RF; `v` is `NaN` at IG and Ga; plot | Data Flow; Result Dictionary | Changed on purpose (repair law names; `v` is `NaN` also at RF Mk, Ho, Ch) | memory | Port: rewrite as a solve, then `validate`, then `plot_management` on a fleet with the five models (G, IG, W, Ga, RF) and both repair laws | |
| conftest | `conftest.py` (171): `fake_result`, `fake_shared`, `tiny_problem_data` | Fixtures: hand result, `shared` dict, in-memory input | none | Obsolete: `tiny_problem_data` writes the old keys (`formulation`, `n_workers`, `warm_start`, `maintenance_type`, `mu_new`, `v_new`, `tail_bound`, `b_inc`, `b_0`, `b_new`, `beta`, `alpha_inc`); `shared` has no counterpart | none | Drop; the ported tests use a builder of merged keys (about 40 lines inside `tests/test_ported_lore.py`) | |
| doc | `TEST_DOCUMENTATION.md` (188) | Describes the suite (API of spec v0.5, "dominates, not equals") | none | Obsolete: names the removed modules and keys | none | Drop; the ported file carries its own docstring | |
| script | `unit_test_0.py` (5) | Script: `solve("../input/data_example.yaml", ...)`; no assertion | Data Flow | Input absent | `data_example.yaml` (absent) | Drop: not a test; smoke scenarios replace it | |
| script | `unit_test_01.py` (5) | Script: `plot_management("../results/output.yaml", plot_file_path=...)`; no assertion | Plotting API | Argument names still valid | `results/output.yaml` (an old Gaussian result) | Drop: not a test | |

## Group 3: local checks

Method: `fleet_management.config.read_input` and `load_config` imported from the worktree `src/` (the package imports
without Gurobi; no stub was needed for the YAML runs). "After dropping" means: delete every rejected or unknown key
of the file, then call `load_config` again. This is a diagnosis of the next error, not a proposed edit of the values.
No file loaded with a warning. Scripts: `ast.parse` for syntax, then every `from fleet_management... import` name
checked against `src/`, then (where no arguments were needed) a local run with the stub of `wp5/stubs` on the path.

### 3a YAML inputs (28 files: all YAML of group 2 except the result file in `starts/`)

| path | first error of `load_config` | after dropping the rejected keys |
|---|---|---|
| `examples/regression/gamma_tail_bound_public.yaml` | `gamma_beta_bound` not accepted | Loads (dropped: `gamma_beta_bound`, `gamma_calibration_method`, `C_P`) |
| `examples/regression/matched_mixed_ard1.yaml` | `gamma_beta_bound` not accepted | Error: `replacement_mu` must be 0 when `replacement_as_new` is true (Gamma cell (0, 0)) |
| `examples/regression/matched_mixed_ardinf.yaml` | `gamma_beta_bound` not accepted | Same error |
| `examples/regression/mixed_gamma_ard1_public.yaml` | `gamma_beta_bound` not accepted | Same error |
| `examples/regression/mixed_gamma_rainflow_public.yaml` | `gamma_beta_bound` not accepted | Same error |
| `examples/regression/vbz_progressive/vbz_00_reference.yaml` to `vbz_05_oberes_glattal.yaml` (6 files) | `gamma_beta_bound` not accepted | All 6 load (dropped: `gamma_beta_bound`, `gamma_calibration_method`) |
| `experiments/gamma_h1_h2_daily_5400s/input/horizon_initial_damage_daily_5400s.yaml` | `gamma_beta_bound` not accepted | Error: `C_M` must be a number (got `[1.0, 1.2]`) |
| `experiments/gamma_horizon_grid_final/input/horizon_grid.yaml` | `gamma_beta_bound` not accepted | Same error (`[1.0, 1.2]`) |
| `experiments/gamma_horizon_stabilization/input/gamma_horizon_stabilization.yaml` | `gamma_calibration_method` not accepted | Loads |
| `experiments/gamma_initial_damage_horizon/input/horizon_initial_damage.yaml` | `gamma_beta_bound` not accepted | Error: `C_M` must be a number (`[1.0, 1.2]`) |
| `experiments/garage_hardau_strict_greedy/input.yaml` | `gamma_beta_bound` not accepted | Error: `C_M` must be a number (`[1.0, 1.2, 1.1, 1.3]`) |
| `experiments/vbz_F8_feasibility_transfer/inputs/vbz_F6_M1_L4_H2_48.yaml` | `gamma_beta_bound` not accepted | Loads |
| `experiments/vbz_F8_feasibility_transfer/inputs/vbz_F8_M1_L4_H2_48.yaml` | `gamma_beta_bound` not accepted | Loads |
| `experiments/vbz_greedy_warm_start/input.yaml` | `gamma_beta_bound` not accepted | Error: `C_M` must be a number (`[1.0, 1.0]`) |
| `experiments/vbz_lines161_162_5pct/input/vbz_line162_H2_12_5pct.yaml` | `gamma_beta_bound` not accepted | Loads |
| `experiments/vbz_lines161_162_5pct/input/vbz_lines161_162_F4_H2_12_5pct.yaml` | `gamma_beta_bound` not accepted | Loads |
| `experiments/vbz_lines161_162_5pct/input/vbz_lines161_162_H2_12_5pct.yaml` | `gamma_beta_bound` not accepted | Loads |
| `experiments/vbz_lines161_162_optimality_1h/input/vbz_lines161_162_F4_L2_tangent_boundfocus_1h_32c.yaml` | `gamma_beta_bound` not accepted | Loads |
| `input/data.yaml` | `alpha` not accepted (use `tau`) | Error: missing required keys `model`, `tau`, `rho`, `C_rep`, `C_D` (dropped: `alpha`, `xi`, `C_S`, `C_P`) |
| `input/data_test_baseline.yaml` | `alpha` not accepted | Same missing keys |
| `input/gamma_horizon_euler_convergence.yaml` | `gamma_calibration_method` not accepted | Loads |
| `input/tiny.yaml` | `alpha` not accepted | Same missing keys |
| `input/vbz_man12e_year.yaml` | `C_S` not accepted (use `C_D`) | Error: missing required key `C_D` (dropped: `C_S`, `C_P`, `depot_capacity`, unknown `damage_capacity`, `_missions`) |

Counts: 0 of 28 load as they are; 15 load after dropping the rejected keys (6 of them are the `vbz_progressive` files);
5 need a scalar `C_M`; 4 need `replacement_as_new` or `replacement_mu` changed; 4 are old-format files with missing
keys (`data.yaml`, `data_test_baseline.yaml`, `tiny.yaml`, and `vbz_man12e_year.yaml`, which lacks only `C_D`).
The result file `experiments/vbz_F8_feasibility_transfer/starts/vbz_F6_M1_L4_H2_48_result.yaml` is not an input: its
first key rejected by `load_config` is `method`; the runner reads only `F`, `objective`, `x`, `m`, `r` from it.

### 3b Python scripts (51 files: 34 in `examples/regression/`, 16 in `experiments/`, `remote_smoke_test.py`)

All 51 files pass the syntax check. Import problems and the first local failure:

| script | syntax | import check against `src/` | first local failure |
|---|---|---|---|
| `check_gamma_ard1_replacement_truth_table.py`, `check_gamma_ard1_truth_table.py`, `check_gamma_ardinf_replacement_truth_table.py`, `check_gamma_big_m_formulation.py`, `gamma_replacement_truth_table_common.py`, `experiments/vbz_F8_feasibility_transfer/run_f8_feasibility_transfer.py` | ok | Missing module `gamma_utils.gamma_diagnostics` | `ModuleNotFoundError` at import (the 4 check scripts and `report_gamma_validator_slide8.py`, through the common helper; F8 not run) |
| `check_greedy_warm_start.py` | ok | Missing name `_simulate_cell` | `ImportError` |
| `check_horizon_sweep.py` | ok | Missing name `_annotate_latest_gradient` | `ImportError` |
| `check_result_visualization.py` | ok | Missing name `plot_mixed_management` | `ImportError` |
| `check_horizon_sweep_visualization.py` | ok | none | `AssertionError`: routes to `mixed_plotter.py` (line 16) |
| `check_gamma_tail_bound.py` | ok | none (loads the module by path) | `AttributeError` at line 39: `calculate_tail_bound_parameters` |
| `check_gamma_ard1_integration.py`, `check_gamma_cell_integration.py`, `check_gamma_complexity_diagnostics.py`, `check_gamma_exact_validator.py`, `check_gamma_public_interface.py`, `check_gamma_repair_integration.py`, `check_gamma_replay_validator.py`, `check_gamma_stochastic_validator.py`, `check_gamma_validator_edge_cases.py`, `check_greedy_warm_start_acceptance.py`, `check_matched_ard_inputs.py`, `check_mixed_gamma_ard1_public.py`, `check_mixed_public_interface.py`, `check_operating_average_objective.py`, `report_gamma_complexity.py` | ok | none | `ValueError`: key `gamma_beta_bound` not accepted (in `load_config`, directly or inside `solve`) |
| `check_unrestricted_maintenance.py` | ok | none | `AssertionError`: unclear removed-option error (the first key seen is `gamma_calibration_method`; the check also expects the word "removed") |
| `check_gamma_convolution_quadrature.py` | ok | none | none: runs and finishes |
| `report_indicator_presolve.py` | ok | none | Not run with the stub (needs real `GRB.GENCONSTR_*`); by reading, `resolve_run_options(cfg, verbose=0, relaxation_warm_start=False)` raises `TypeError` |
| `experiments/garage_hardau_strict_greedy/run_case.py`, `experiments/vbz_greedy_warm_start/run_comparison.py` | ok | none | Not run (arguments, Gurobi); by reading, `report.feasible_assignment` raises `AttributeError` |
| `examples/regression/run_horizon_sweep.py` | ok | none | Not run (arguments, Gurobi); by reading, an error row prints `None` with a number format (`TypeError`) |
| `gamma_quadrature_reference.py`, `plot_optimization_progress.py`, `run_gamma_schedule_validation.py`, `run_progressive_vbz_feasibility.py`, `report_gamma_validator_slide8.py` | ok | none directly (the last one imports the truth-table helper) | Not run (helpers or arguments); the last one fails through the helper |
| `experiments/gamma_h1_h2_daily_5400s/{plot_results,preflight_small_grid,summarize}.py`, `experiments/gamma_horizon_grid_final/{plot_grid,summarize_grid}.py`, `experiments/gamma_horizon_stabilization/{plot_result,summarize_result}.py`, `experiments/gamma_initial_damage_horizon/{plot_results,preflight,summarize}.py`, `experiments/vbz_lines161_162_5pct/{run_case,summarize_cases}.py`, `experiments/vbz_lines161_162_optimality_1h/run_case.py` | ok | none | Not run (arguments, Gurobi, or the YAML of the first table); the preflight scripts fail on their input while it holds the rejected keys |
| `remote_smoke_test.py` | ok | none (imports inside the functions: `solve`, `plot_management` exist) | Not run (it would write `smoke_test_output/` in the worktree); by reading, `input/data_example.yaml` and the folder `test/` do not exist |

Shell scripts (`*.sbatch`) were read, not run. The only fault found is in `gamma_horizon_stabilization.sbatch`
(lines 67-70 copy two removed files). Their preflight blocks call `load_config` on the experiment inputs, so they fail
on the rejected keys until the inputs are rewritten.
