# Y3 report (tools, start of the repetition)

Status: done. The three tools read `y_q` when the result has it; the old rule holds without it. Local checks pass.

## Changes by file (worktree `Fleet_Management-merge`)

| File | Function | Change |
|---|---|---|
| `src/fleet_management/validation/validator.py` | `start_values` (new) | returns `y_<key>` as an `F x L` float array, or `None` if absent or without a number |
| | `_loop_violation` | new argument `y`; with `y`: `max{0, max(q_T - y), max(q_H1 - y)}`; without: old `max{0, max(q_T - q_H1)}`; `NaN` skipped |
| | `check_result` | passes `start_values(result, key)`; names and order of the checks unchanged |
| `src/fleet_management/degradation_model/gamma_utils/gamma_replay_validator.py` | `validate_gamma_replay_schedule` | reads `y_mu`, `y_g_mu` (shape `F x L`, else `ValueError`); `repeatability` with `y` finite at the cell: `expected = y`, `error = max{0, mu_T - y, mu_H1 - y}` (same for the anchor under `ard1`); else old rule. Record stays on day `T` with its event. Docstring updated |
| `src/fleet_management/validation/dashboard_inspector.py` | `loop_table` | columns `train, component, quantity, value_at_H1, y, value_at_T, margin_T, margin_H1, margin, holds`. With `y`: `margin_T = y - q_T`, `margin_H1 = y - q_H1`, `margin` = their minimum. Without `y`: `y`, `margin_H1` NaN, `margin = margin_T = q_H1 - q_T`. Row order as before. `holds`: `margin >= -tol` |
| | `render_result_inspector` | text of the tab Repeatability |

The greedy heuristic is untouched.

## Tests

- New `tests/test_step3_start_tools.py`: 15 tests (no Gurobi). For each tool: pass (`q_T` above `q_H1`, below `y`), fail on `q_T > y`, fail on `q_H1 > y`, and the old rule without `y`. Extra: the anchor under `ard1`, the file function `validate`, check names unchanged, `NaN` handling.
- Existing tests: none changed. The old `margin`, `value_at_H1`, `value_at_T` columns keep their meaning without `y`.

## Local counts (runner `run_t2.py`)

| File | After |
|---|---|
| `test_step3_start_tools.py` | 15 passed |
| `test_spec_validation.py` | 131 passed, 2 failed (the known runner limit) |
| `test_validator.py` | 38 passed |
| `test_gamma_validators.py` | 60 passed |
| `test_validator_dashboards.py` | 14 passed |
| `test_step3_tools.py` | 11 passed |
| `test_step3_plots.py` | 4 passed |

The changes are inactive for results without `y`, so the counts before are the same for the six existing files (not run separately). Line endings: 0 CR before and after in all edited files.

## Open points

- A cell whose `y_q` is `NaN` while `q` is finite, in a result where `y_q` has numbers elsewhere, is skipped by the validator and the loop table (as `NaN` entries are). The old rule is not used there.
- `check_sizes` does not check the shape of `y_*`; only the replay validator raises on a wrong shape.
- The spec (Structural Validator, Replay validator, Dashboards) still needs the text for the new rules.
