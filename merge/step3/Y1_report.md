# Y1 report: start of the repetition, code core

Status (8 Nov 2026): done. Code and tests written; local checks pass; the Gurobi tests have not run on real Gurobi
(Claude submits the Euler job). Nothing committed.

## Changes in the code

| File, function | Change |
|---|---|
| `degradation_model/base.py`, `FleetModel` | New fields `start_vars` (`q -> {(i, l): Var}`) and `start_anchor_vars` (the same for `y^{g,s}`), the registry for the extraction. `build_context` initialises both with one empty table per quantity. |
| `base.add_state_vars` | Per cell and quantity one continuous `y^s` in `[0, M_s]`, named `y{q}_{i}_{l}` (`ymu_0_0`); at an anchored `ard1` cell also `y^{g,s}`, named `yg{q}_{i}_{l}`. None for the Wiener variance anchor (`anchored=False`). |
| `base.add_quantity_rows` | Start rows `start_{q}_{i}_{l}` (`y >= s_{H1}`) and `start_g{q}_{i}_{l}` (`y^g >= g_{H1}`). On day index `H1` (day `H1 + 1`) `s_prev` and `g_prev` are `y` and `y^g`: this covers carry, repair (`psi`), replacement-free rows, both encodings, the Wiener variance row (`w_vrow`) and the anchor keep rows. Loop rows now `S[T-1] <= y` (`loop_{q}_{i}_{l}`) and `G[T-1] <= y^g` (`loop_g{q}_{i}_{l}`); names unchanged. Raises `ValueError` if `y` is missing (call `add_state_vars` in `prepare`). |
| `base.add_repair_cost_rows` | The row of day index `H1` reads `y^mu` and `y^{g,mu}` (`g = 0` under `ardinf`). |
| `base._SOLUTION_KEYS`, `base.extract_solution` | The eight keys `y_mu ... y_g_K` (`F x L`, `NaN` where not tracked); set to `None` when there is no solution. |
| `solver.py`, `RESULT_KEYS` | The eight keys after `g_K`, before `u` (59 keys). |
| `solver.py`, `_HDF5_DATASET_KEYS` | The eight keys as `float64` datasets (25 datasets). `read_result` converts them to float arrays through this tuple; the YAML and JSON writers follow `RESULT_KEYS`. |

The five cell builders did not change: all build their previous-day rows through `base`. Row names are the old ones plus
`start_*`; no existing row was renamed.

## Tests

New file `tests/test_step3_start.py` (17 cases):

- (a) `test_replacement_on_day_h1_then_rising_operating_phase` (bigm, indicator): inverse Gaussian cell, `ardinf`, `rho = 0.5`,
  `tau = 2`, mission increment 0.2, replacement value 0, flights fixed, repair and replacement free. The derivation by hand is in
  the comment above `_HAND`: optimum replace on day 2 (`H1`), repair on day 4, objective 9, `y_mu = 0.2` (the fixed point of
  `y -> 0.5 (y + 0.2)`), `mu = [>= 0.7, 0, 0.4, 0.2]`. `test_old_loop_row_refuses_the_schedule_and_costs_more` adds the old row
  `s_T <= s_{H1}`: the schedule is refused and the optimum is 10 (replace on days 2 and 4).
- (b) rows by name and coefficient with Gurobi's `getRow`, `Sense`, `RHS`, `getGenConstrIndicator`: start, loop, day `H1 + 1`
  carry and repair (mean, variance), anchor keep, repair cost, for Gaussian `ard1` bigm, `ardinf`, Wiener (variance row,
  no variance anchor) and the indicator encoding; the set of rows that read each `y` is checked to be exactly the listed one
  (no reliability row, no row of another day); variables and registry (bounds, type, which cells).
- (c) `test_result_keys_and_hdf5_datasets_hold_the_start_keys`, round trip through the YAML, JSON and HDF5 writers and
  `read_result` (NaN pattern kept, float64), HDF5 datasets, a result without the keys, and one free Gaussian solve
  (Gurobi) checking the NaN pattern and `s_{H1} <= y`, `s_T <= y`.

Existing tests changed (minimal):

| File | Change |
|---|---|
| `test_base.py` | `_SOLUTION_KEYS` gets the eight keys (so `_RESULT_KEYS` too); `_min_states` minimises the `y` and `y^g` variables as well; `_assert_untracked_keys` checks the four `y_*` keys of `R`, `K`. |
| `test_gaussian.py`, `test_wiener.py`, `test_inverse_gaussian.py`, `test_gamma.py`, `test_ported.py`, `test_rainflow.py` | The objective of the fixed-schedule solves adds every `y` and `y^g`. Without it an anchor after day `H1` equals a free `y^g`, and the hand trajectories are no longer unique. No expected value changed. |
| `test_ported.py` | `test_gamma_free_solve_cap_and_loop_rows` asserts `s_{H1} <= y`, `s_T <= y` (and the anchors, `NaN` at the `ardinf` component) instead of `s_T <= s_{H1}`. |
| `test_io.py` | `_result` has the eight `y_*` arrays (59 keys); the no-solution list; HDF5 datasets 17 to 25, `y_*` shapes and `NaN`. |
| `test_spec_outputs.py` | `RESULT_KEYS`, `SOLUTION_KEYS`, `HDF5_DATASETS` get the keys (`Y_KEYS`); shape `(F, L)`; the `NaN` pattern of the `y_*` keys follows `_tracked`; the HDF5 layout test. |

The tests that assert only row names (`loop_mu_0_0`, `loop_gmu_0_0`, ... in `test_base.py`, `test_gaussian.py`, `test_gamma.py`,
`test_rainflow.py`) did not change: the names are the same.

Line endings kept (CRLF files `base.py`, `test_base.py`, `test_rainflow.py`, `test_gamma.py`; LF files the others; checked by `tr`).

## Local checks

Real Gurobi is not here. Runner `scratchpad/wp8/finish/run_t2.py` copied to `scratchpad/step3/Y1/`.

| Run | Result |
|---|---|
| `test_step3_start.py`, `test_io.py`, `test_spec_outputs.py` (stub, no Gurobi) | 6 passed (11 skipped) / 18 passed / 15 passed (128 skipped) |
| `test_step3_base.py` (stub) | 35 passed, 8 skipped |
| `test_base.py` (module skips without `gurobipy`) | not loadable with the plain runner |
| With a local fake `gurobipy` (my own file `scratchpad/step3/Y1/fake/gurobipy/__init__.py`: builds rows like Gurobi, solves linear models with scipy HiGHS, indicators by big-M 100, no quadratic rows) | `test_step3_start.py` 17 passed (the hand case solves to the numbers above, both encodings; the old-row solve gives 10); `test_gaussian` 11/14, `test_wiener` 9/11, `test_inverse_gaussian` 11/11, `test_gamma` 36/36, `test_base` 39/44, `test_rainflow` 107/124. Every failure is a quadratic row (`addQConstr`, which the fake lacks) or `test_ported`'s package import (the runner has no `__init__`). |

The fake is a structural check only: the row coefficients of the real Gurobi (`getRow`, `getGenConstrIndicator`) are assumed to
match its behaviour (Var-Var rows as `a - b >= 0`, constants moved to the right-hand side).

## What needs real Gurobi (Euler)

Everything in `test_step3_start.py` marked Gurobi (the 11 skipped cases of the stub run), `test_base.py`, `test_gaussian.py`,
`test_wiener.py`, `test_inverse_gaussian.py`, `test_gamma.py`, `test_rainflow.py`, `test_ported.py`; above all the quadratic
(`exact`, `pwl`) cases that the fake could not run, and `test_ported.py::test_gamma_free_solve_cap_and_loop_rows`.

## Open points

- The indicator-row check of (b) relies on `getGenConstrIndicator` returning the left-hand side as a `LinExpr` with the sense in
  `{<, >, =}` (Gurobi 13 documentation, "Model.getGenConstrIndicator"); `_normalise` handles all three senses.
- Greedy warm start, validators and dashboards are not touched (tasks Y2, Y3). `validation/validator.py` still checks
  `loop_q` against day `H1`; on a real solve with `y > q_{H1}` that check is stricter than the rows until Y3 is done.
- The module docstring of `base.py` still says "loop rows" without naming `y`; the spec text follows after the Euler run.
