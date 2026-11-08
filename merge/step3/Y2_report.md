Status: code and unit tests done (local runs below); tightness rerun in progress (8 processes, scratchpad `step3\Y2\tight\`). Next: tightness table, append to `tightness_check.md`.

# Y2: reference model for the start of the repetition

## Changes by function (`tests/reference_model.py`, worktree `Fleet_Management-merge`)
- `_y_value` (new): reads `sol["y_..."][i, l]`; an absent or `None` key gives the value on day `H1` (the old rule).
- `row_violations`: new rows `<q>_start` and `g_<q>_start` (`s_{H1} <= y`, record day `H1`, tolerance `tol`); the rows of day `H1 + 1` read `y` (dynamics: carry, repair, replacement is unchanged; anchor keep rows; repair-cost row `z_row`, with `y^g = 0` under `ardinf`); loop rows `<q>_loop` and `g_<q>_loop` are `s_T <= y`. The docstring describes it. Without `y` keys every row is the old one (the start rows hold trivially).
- `_least_quantity`: the arrays have length `T + 2`; index `T + 1` is `y`. The rows of day `H1 + 1` read index `T + 1` (dynamics, anchor keep rows both ways, the backward pass over carry days). The lower bounds `y >= s_{H1}` and `y >= s_T` replace `s_{H1} >= s_T`. Start value of `y`: the exact value on day `H1`.
- `_least_cell`: returns `y` and `gy` per quantity; the domain check includes `y` and `y^g` in `[0, M_s]`; `z` on day `H1 + 1` reads `y`.
- `least_solution`: returns `y_mu`, `y_v`, `y_R`, `y_K`, `y_g_mu`, `y_g_v`, `y_g_R`, `y_g_K` (`F x L`, NaN where not tracked) and passes them to `row_violations`. The comment on why the least solution decides feasibility is updated (start of the repetition, the monotone maps applied to `y`).
- `best_onesided` follows with no code change. `reference_trajectory`, `best_reference`, `repeated_safety` unchanged (`best_reference` calls `row_violations` without `y` keys: the old rule on equalities, which is the same since `y` = the value on day `H1`).

## Tests
- Added (`tests/test_oracle_unit.py`): `test_start_of_repetition_hand_case_is_accepted_and_the_old_rule_refuses_it` (train X replaced on day `H1 = 1`, mission, repair; `y_mu = 0.2` the fixed point of `y -> 0.5 (y + 0.2)`; old rule shown infeasible and new rule minimum shown by two linear programs; the same trajectory fails the old loop rows by 0.2 and 0.1 and passes the new rows; every number derived in the comment) and `test_row_violations_start_rows_day_h1_plus_one_and_loop_rows_read_y` (start row, day `H1 + 1` carry, repair and repair-cost rows reading `y`, loop row to `y`, NaN, no `y` keys).
- Changed: `test_least_solution_rescues_a_schedule_that_breaks_the_loop_row` (day 1 is no longer pushed up by the loop row: `mu(X) = 0.04, 0.09, 0.05, 0.10`, `mu(Y) = 0.04, 0.04, 0.10, 0.05`; asserts `y_mu`, the NaN pattern; the "larger solution" part now gives `y_mu`), `test_least_solution_with_anchors` (`s(X) = 0.08, 0.08, 0.15, 0.10`, `g(X) = 0.08, 0.08, 0.10, 0.10`; asserts `y_mu`, `y_g_mu`; derivation rewritten), comment of `test_best_onesided_tiny_input_by_hand` (cost 3.0 unchanged).
- Not changed (they run `row_violations` on solutions without `y` keys, so they assert the old rule, which is still valid in that mode): `test_row_violations_hand_solution_values_and_no_violation`, `test_row_violations_each_broken_row_is_reported_alone` (`_edit_loop`), `test_row_violations_nan_counts_as_a_failure`, `test_row_violations_without_replacement_array_and_loop_rows`, `test_row_violations_reference_solution_of_every_kind_has_no_violation`, the `repeated_safety` tests, `best_reference` tests, `test_least_solution_reports_no_solution_when_the_iteration_diverges`.
- `tests/test_oracle_bruteforce.py` and `tests/test_oracle_replay.py`: not edited (no assertion depends on the old least solution; replay tests call `row_violations` without `y` keys).

## Local counts (runner copy `step3\Y2\run_t2.py`; its `parametrize` stub now accepts `ids=`)
- `test_oracle_unit.py`: 146 passed, 0 skipped, 0 failed (was 138 before this task; 144 with the changed tests).
- `test_oracle_bruteforce.py`: 85 passed, 42 skipped (Gurobi), 0 failed.
- `test_oracle_replay.py`: 23 passed, 22 skipped (Gurobi), 0 failed.

## Tightness (all 42 inputs, 176912 schedules; full table in `tightness_check.md`)
| | before | after |
|---|---|---|
| A and B | 7836 | 9348 |
| A and not B | 0 | 0 |
| B and not A | 1512 | 0 |

- No safe schedule is refused any more: the 1512 schedules refused for a replacement on day `H1` are accepted, all safe on 300 repetitions. No case has `B and not A`, so there is nothing to diagnose.
- `best_onesided` is identical to the old value in all 42 cases (and `infeasible_G` stays infeasible).
- `G_mode_opavg`: the 96 schedules whose reference cost (1.3 to 1.5) was below the optimum 1.6 cost 1.6, 1.625 or 1.65 under the least solution (larger `y`), so the optimum stays 1.6.
- Scripts and raw results: `scratchpad/step3/Y2/tight/` (`run.py`, `agg.py`, `diag2.py`, `all.json`).
