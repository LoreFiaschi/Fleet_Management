Status: done. 10 files read in full; 1 edit (1 file).

# Edit made

`experiments/vbz_greedy_warm_start/README.md` (CRLF kept: 16 before, 16 after)
- Statement: "reports any mission that would require an artificial vehicle ... nonzero count is the trigger for a separate penalized-artificial-vehicle model". `GreedyWarmStartReport` (`greedy_warm_start.py`) has no such count; it has `repeatability_feasible`, and `run_comparison.py` raises if it is false.
- New text: "reports whether its schedule meets the loop rows (`repeatability_feasible`) ... The run stops with an error if they fail; ...".

# Checked, no change
- `examples/regression/RegressionREADME.md`: every script, input file, command-line option (`--h1`, `--h2`, `--h2-range`, `--stop-on-gradient`, `--warm-start`, `--evaluation-horizon`, `--mode`, `--repetitions`, ...), key (`repair_model`, `replacement_mu`, `replacement_as_new`, `relaxation_warm_start`, `relaxation_time_limit`, `encoding`, `depot_capacity`) and API name matches the worktree and the spec. No removed key or file is named.
- `gamma_h1_h2_daily_5400s`, `gamma_horizon_grid_final` (dated sentence present), `vbz_lines161_162_5pct` (dated sentence present), `gamma_horizon_stabilization`, `gamma_initial_damage_horizon`, `garage_hardau_strict_greedy`, `vbz_F8_feasibility_transfer`, `vbz_lines161_162_optimality_1h`: no removed key, function or file; the scripts, sbatch files and inputs they name exist; the grids and flags match the job scripts.

# Left alone, for the user
- `vbz_F8_feasibility_transfer`: `starts/vbz_F6_M1_L4_H2_48_result.yaml` holds old keys (`gamma_calibration_method`, `gamma_beta_bound`, `method`); the runner replays it with `validate_gamma_schedule_files`. Kept by WP8 decision 7; whether the validator reads that file was not tested (no Gurobi run).
- `vbz_F8_feasibility_transfer` ("previously obtained feasible F6 result") and `vbz_lines161_162_optimality_1h` ("archived tangent run") mention results of `main`'s formulation without the dated sentence. WP8 decision 5 allows the sentence in two READMEs only.
- `gamma_horizon_grid_final` shows `git push origin gamma-tail-integration`, a branch name of `main`'s history; not changed (decision 5: no other change).
- `GAMMA_DIAGNOSTICS.md` is not in the worktree.
