Status: done. B1, B2, B3 done; 18 tests pass (see Steps).

# Task B progress (solver.py, tests/test_io.py)

Worktree: C:\Users\utente\Documents\GitHub\Fleet_Management-merge. Runner: scratchpad\wp4\run_tests.py.
Command: `cd <scratchpad>\wp4 && PYTHONIOENCODING=utf-8 python run_tests.py <worktree>\tests\test_io.py [filter]`
Old solver.py kept as scratchpad\wp4\solver_old.py (reference only).

## Dependency on Task A
`config.read_input` and `INPUT_EXTENSIONS` (step A1) exist in config.py since B1 (checked 29 Oct 2026).

## Steps
- B1 done: solve path (solve, solve_mapping, _check_results_path, constants, `_read_input` alias). Old dispatch, readers removed.
  Check: tests B1, 9 passed (with the two file-writing tests passing after B2).
- B2 done: `_save_results`, `_to_plain`, `_text_output`, `_save_yaml`, `_save_json`. Check: 14 passed, 0 failed (B1 + B2 tests).
  Choice not in the plan: `_result_format(path)` (yaml/json/hdf5, or the ValueError with the message of the results file type)
  is shared by `_check_results_path` and `_save_results`, so the message exists once.
  `_check_results_path` does not skip the checks when the parent is the current folder (the plan kept s:50-53 `!= Path("")`; spec: directory must exist and be writable).
- Resume check (29 Oct 2026): B1 and B2 re-run, 14 passed, 0 failed. `_save_hdf5` was already rewritten in solver.py (new layout, as the plan's B3); B3 tests are not yet in tests/test_io.py.
- B3 done: `_save_hdf5` as in the plan (17 float64 datasets; numbers, strings, `u`, `H` (int64) as attributes; the 7 JSON keys and a non-`str` selector as JSON text; `None` values skipped, so no `objective` attribute without a solution; NaN and inf kept; keys outside RESULT_KEYS not written).
  Tests B3 added to tests/test_io.py: `test_hdf5_round_trip`, `test_no_solution_hdf5`, plus two small extras (`test_hdf5_keeps_non_finite_numbers`, `test_hdf5_selector_that_is_one_string_is_an_attribute`).
  Check: 18 passed, 0 failed (B1 + B2 + B3). Scratch cross-check (`b3_scratch_readback.py`, not a test): an HDF5 result file read with `config.read_input` gives the keys back (H `[1, 2]`, F an `int`, selector lists, NaN kept).
  Greps on solver.py: none of the deleted helpers (`_solve_mixed`, `_identify_result`, `_solve_gamma_fleet`, `_require_uniform`, `_uniform_over_vehicles`, `_legacy_gamma_beta`, `_cfg_to_gamma_kwargs`, `_read_hdf5`, `_resolve_results_path`) and no `model_assignment`/`backend`/`legacy`; `base` imported only inside `solve_mapping`.

Status: done. B1, B2, B3 complete; 18 tests pass.

## Problems
- None blocking. Notes for the planner:
  - `_result_format(path)` (shared by `_check_results_path` and `_save_results`) is an addition not named in the plan, so the "Unsupported results file type" message exists once.
  - `test_solve_mapping_checks_input_without_gurobi` passes with the current `config.py` (Task A, A2 done at the time of the run).
  - `base.solve_mixed` in the worktree's `base.py` (line 2004, checked 29 Oct 2026) has the signature `solve_mixed(cfg, **overrides)`, not `solve_mixed(cfg, *, warm_start=None, warm_start_mode="start")` as `solve_mapping` calls it. WP5 must give it the new signature; the tests stub it, so they do not depend on it.
