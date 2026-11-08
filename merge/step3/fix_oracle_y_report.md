Status: done (local checks only; the Gurobi path was not run, no Euler).

# Fix: the oracle tests pass the start of the repetition `y`

Plan: `merge/step3/start_of_repetition_plan.md`. Files edited: `tests/test_oracle_replay.py`, `tests/test_oracle_bruteforce.py`. `tests/reference_model.py` is unchanged (no defect found).

## Changes
- `test_oracle_replay.py`: new tuple `START_ARRAYS` (the eight `y` keys). Check B passes them (float arrays) into `row_violations`. `_check_domains` adds `0 <= y <= M_s` for `y_<q>` and `y_g_<q>` (NaN skipped). Check D adds `y >= reference value on day H1`.
- `test_oracle_bruteforce.py`: `_solution` copies the eight `y` keys. `_check_upper_bounds` (check D) adds `y >= reference on day H1` and `0 <= y <= M_s`.
- Check D "reported arrays at least the reference trajectory" is kept as is: the solver starts day H1+1 from `y >= q_H1`, so the reference trajectory stays a lower bound.
- Line endings unchanged (0 CR in both files before and after).

## Runs
- `run_t2.py` (copy in scratchpad `step3\fixy\`): `test_oracle_replay.py` 23 passed, 22 skipped, 0 failed; `test_oracle_bruteforce.py` 85 passed, 42 skipped, 0 failed.
- Fake result (`step3\fixy\fake.py`): `least_solution` arrays (day 0 dropped) plus `y`, with the schedule of `best_onesided`, through `_check_rows_and_safety` and `_check_upper_bounds`: all 41 feasible brute-force cases pass. For replay, the schedule of `_schedule_in_turn` (with `depot_repair` True and False) through check B, `_check_domains` and the NaN pattern: every case where `least_solution` is feasible passes; the others are infeasible by construction (for example "mu: day 4: lower bound above the range"), so they were not checked. Replay checks D and E were not run on a fake result.
