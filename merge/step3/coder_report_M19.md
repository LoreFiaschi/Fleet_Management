Status: done. Rule M19 implemented in the worktree `Fleet_Management-merge`; all local checks pass (counts below). Nothing committed.

## Changes by file

| File | Change |
|---|---|
| `src/fleet_management/validation/tolerance.py` (new) | `gurobi_tolerances(cfg)` (case-insensitive names, Gurobi defaults 1e-6 and 1e-5), `block_tolerance`, `resolve_tolerance(cfg, argument)` (steps 1 and 2), `report_tolerance`, `binary_tolerance`, `cost_tolerance`, `state_tolerance(cfg, q, single, i, l)` (`FeasibilityTol + IntFeasTol * range`; `g_*` and `y_*` use the range of their quantity; where the range is NaN: `FeasibilityTol`). |
| `config.py` | `_run_options`: `validation.tolerance`, when not null, must be a finite number > 0, else `ValueError` naming `validation.tolerance`. Nothing else changed by me. |
| `validation/validator.py` | `check_result(cfg, result, tol=None)`, `validate(..., tol=None)`. Binary checks: `IntFeasTol`; `objective`: `FeasibilityTol`; `loop_*`: per-cell tolerance array (new `_loop_cell_violations`; `_loop_violation` kept). `add` passes when no cell exceeds its tolerance; reports the largest violation and the tolerance at that cell. Report `tolerance`: number (steps 1, 2) or `{"FeasibilityTol", "IntFeasTol"}`. |
| `gamma_replay_validator.py` | `_check_tolerance` accepts `None`; replay: `tolerance=None`; binaries `IntFeasTol`; all five fields use the per-cell `range_mu` tolerance (`z` and cap included). New report keys `tolerance` and `maximum_error_tolerances` (tolerance at the cell of each field's largest error). |
| `gamma_tail_validator.py` | `tolerance=None`; binaries `IntFeasTol`; `reliability` `FeasibilityTol`; `tracked_mean`, `mean_anchor` per-cell state tolerance; same two new report keys. |
| `gamma_stochastic_validator.py` | Monte Carlo: `tolerance=None`, used only for the binaries (`IntFeasTol`); report `tolerance` as above. Dispatcher: `tolerance` joins `_SETTING_DEFAULTS` (so the block key is allowed and the None rule reads it); `validate_gamma_schedule(tolerance=None)`. |
| Dashboards | Not touched (they pass the sidebar tolerance). |

## Tests

New `tests/test_step3_tolerance.py`: 30 tests (resolution order, `gurobi_params` 1e-7/1e-6 and case-insensitive names, state gap 0.5x passes and 2x fails, per-cell tolerance with tau 1 and 2, binary gap 5e-6 passes and 2e-5 fails (structural, replay, exact, Monte Carlo), block and argument for every validator, dispatcher reads `validation.tolerance` in all modes, `validation.tolerance` in {-1, 0, nan, inf, "x", True} raises).

Existing tests changed (assertions of the old default only):
1. `tests/test_validator.py::test_valid_result_passes`: `report["tolerance"] == 1e-6` becomes the mapping `{"FeasibilityTol": 1e-6, "IntFeasTol": 1e-5}`.
2. `tests/test_spec_validation.py::test_validate_valid_pair_report`: same mapping; per check `c["tolerance"] == 1e-6` becomes `>= 1e-6`.
3. `tests/test_spec_validation.py::test_check_result_returns_report_without_writing`: same mapping.
No test asserted the Gamma default 1e-8.

## Counts (passed / failed), before then after

| File | Before | After |
|---|---|---|
| test_step3_tolerance | (new) | 30 / 0 |
| test_validator | 38 / 0 | 38 / 0 (one temporary failure fixed by change 1) |
| test_spec_validation | 131 / 2 | 131 / 2 (the two known runner limits) |
| test_gamma_validators | 60 / 0 | 60 / 0 |
| test_validator_dashboards | 14 / 0 | 14 / 0 |
| test_step3_tools | 11 / 0 | 11 / 0 |
| test_step3_start_tools | 15 / 0 | 15 / 0 |
| test_step3_series | 20 / 0 | 20 / 0 |
| test_spec_checks | 350 / 0 | 350 / 0 |
| test_spec_inputs | 83 / 0 | 83 / 0 |

Line endings: validator files and tests are LF before and after (0 CR). `config.py` changed between CRLF and LF during my session (another coder edits it concurrently; HEAD is LF); my lines follow whatever the file had.

## Open points
- The spec text for the validators (tolerance default, report keys `tolerance` mapping, `maximum_error_tolerances`, dispatcher key) still needs the matching update; I did not edit the spec.
- Global violations of the Gamma validators are exact comparisons on the already rounded binaries (the rounding uses `IntFeasTol`); no separate tolerance is needed there.
- The dashboards still default their sidebar to 1e-6 (structural) and 1e-8 (Gamma); left as instructed.
- `check_result` does not validate an explicit `tol` (as before); the Gamma validators still refuse `tolerance <= 0`.
