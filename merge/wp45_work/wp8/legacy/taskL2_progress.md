# Task L2 progress

Status: done (7 Nov 2026). Ports, rewrites and the removal list are complete. Open: the four `C_M` inputs (user), the Gurobi tests (to run on Euler).

## Part 1: ports (done)
- `tests/test_ported.py`: 8 test functions, 15 test items (IG cap monotone in `eta`, 6 parametrized items; Gamma free solve; YAML, JSON and HDF5 end to end; five models solve + validate + plot; Gamma complete repair and consecutive repairs, each in `bigm` and `indicator`).
- Inputs in `tests/data/ported/`: `five_models.yaml`, `gamma_only.yaml`, `e2e_mixed.yaml`, `e2e_gaussian.yaml` (source of the JSON and HDF5 inputs, written in `tmp_path` by the test, HDF5 by the layout of spec "HDF5 Layout").
- Local check: `run_tests_gamma.py` gives 6 passed (IG cap), 9 skipped (Gurobi), 0 failed. The hand schedule of each input has no violated row in `tests/reference_model.py` (`row_violations` empty). The two Gamma trajectories equal the hand values in the test comments. The HDF5 helper was run: `read_input` + `load_config` equal the YAML configuration.

## Part 2: rewrites (done)
YAML inputs (CRLF kept; `gamma_beta_bound`, `gamma_calibration_method`, `C_P` dropped):
- `examples/regression/`: `matched_mixed_ard1.yaml`, `matched_mixed_ardinf.yaml`, `mixed_gamma_ard1_public.yaml` (also `replacement_as_new: false`, see below), `vbz_progressive/vbz_00..05`.
- `experiments/`: the 9 other inputs of the inventory; `vbz_greedy_warm_start/input.yaml` also `C_M: 1.0`.
- `input/gamma_horizon_euler_convergence.yaml`.
- Local check (`load_config`): all load, except the four waiting `C_M` inputs, which fail only on `C_M` (`horizon_initial_damage_daily_5400s.yaml`, `horizon_grid.yaml`, `horizon_initial_damage.yaml`, `garage_hardau_strict_greedy/input.yaml`). Warnings only: values at cells where a key does not apply (inert placeholders in the matched files, `v_0` at Gamma cells in three experiment inputs).

Decision 4, `replacement_as_new`: in `main`, the flag (default `true`) is applied only to rainflow cells (`rainflow.py` line 104, `config.py` lines 441-457: it zeroes `mu_new` and `v_new` of rainflow cells and rejects non-zero values there). The Gamma module of `main` does not read it; Gamma cells used `replacement_mu` (0.001 in the three files). So `main` used the values: flag set to `false` in the three files; the rainflow entries are already 0. Meaning kept.

Python scripts (ast.parse ok; `fleet_management` imports exist):
- `check_gamma_replay_validator.py`: error key `tracked_mean`; corruption now downward (saved states are upper bounds, only a lower saved mean is an error).
- `check_gamma_tail_bound.py`: cut to the Moschopoulos part; run locally, PASS.
- `check_horizon_sweep_visualization.py`: module check deleted; run locally, PASS.
- `check_unrestricted_maintenance.py`: two keys dropped, message "Remove it"; scenario loads and the message matches (checked without Gurobi).
- `report_indicator_presolve.py`: options set on the dict from `resolve_run_options(cfg)`.
- `run_horizon_sweep.py`: error rows print `-` (new `_format_integer`).
- `check_matched_ard_inputs.py`: unchanged (the two inputs still differ only in `repair_model`).
- `experiments/garage_hardau_strict_greedy/run_case.py`: screen reads `cfg.tangent_ref[i, l]` at cells with form `tangent`; start used if `repeatability_feasible`.
- `experiments/vbz_greedy_warm_start/run_comparison.py`: check on `feasible_assignment` dropped.
- `experiments/vbz_F8_feasibility_transfer/run_f8_feasibility_transfer.py` (decision 6): estimator block and `gamma_diagnostics` removed, `extract_solution(ctx, cfg)`, statistics from `base._model_statistics`, selector `n` set only where it exists (indicator cells), result keys `backend`, `degradation`, `models`, `model_assignment`, `gamma_formulation` dropped.
- `experiments/gamma_horizon_stabilization/plot_result.py`: help line.

Job scripts (decision 8): the two `euler/*.sbatch` need no change (they copy only kept inputs; no removed key). Found beyond the inventory: `experiments/vbz_greedy_warm_start/vbz_greedy_warm_start.sbatch` preflight asserted `report.feasible_assignment` and `report.artificial_assignments` (removed); line deleted. `gamma_horizon_stabilization.sbatch`: 4 lines deleted. Both converted to LF (`.gitattributes` says `*.sbatch text eol=lf`; the working copies were CRLF); `bash -n` ok.

READMEs (decision 5): `gamma_h1_h2_daily_5400s/README.md` rewritten from its job script; one dated sentence added to `gamma_horizon_grid_final/README.md` (it states the formulation, no results: "This is `main`'s formulation before the merge (November 2026).") and to `vbz_lines161_162_5pct/README.md` (after the F=3 result: "Obtained with `main`'s formulation before the merge (November 2026).").
`RegressionREADME.md` rewritten last (decision 9): the surviving scripts and inputs only.

## Part 3: removals
Listed in `merge/wp8_plan_legacy.md` section 4: 28 files (24 in `examples/regression/`, 4 in `input/`). Nothing deleted.

## Line endings
Edited files keep their endings (CRLF in the working tree for tracked files, LF for new files and the two `.sbatch`).
