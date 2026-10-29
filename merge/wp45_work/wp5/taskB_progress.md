Status: done. B1 to B4 written and checked locally; the file is CRLF (as the other files in tests/). Next: nothing for Task B; the planner reviews; WP8 runs the file on the licensed machine.

# Task B progress (tests/test_base.py of the merge worktree)

Test file: `C:\Users\utente\Documents\GitHub\Fleet_Management-merge\tests\test_base.py` (772 lines, 20 test functions, 44 test cases after parametrization).
Local checks (this folder): `check_test_base.py` (ast parse, line length, names_check on the test file, stubbed load of the test module, key counts, fixture values, expected arrays against `hand_case.py`, T2 against the real `FleetModel.increment`, the feasibility flags of T8 against a direct evaluation of the rows). Last run: ALL PASS (51 checks), names_check 0 undefined names, no line over 88 characters.

## Steps

| Step | Result | Check |
|---|---|---|
| B1 | Done (365 lines; the plan said about 150, the key tuples and the fixture take the space). Docstring, imports, `_cfg`, `_SCHEDULE`, `_SOLUTION_KEYS`, `_COST_KEYS`, `_RESULT_KEYS` (51), `_PERFORMANCE_KEYS` (24), `_INFEASIBLE`, `_NO_DUAL_REDUCTIONS`, helpers `_register`, `_build`, `_fix_schedule`, `_min_states`, `_solve_fixed(cfg, min_states=True)`, `_rows`, the three builders. Status checks use `gp.GRB` constants, not `base.status_string`. | ast OK; names_check 0; check_test_base.py ALL PASS |
| B2 | Done. T1 `test_unknown_model_has_no_cell_builder`; T2 `test_increment_follows_the_phase_lookup` (2 cases) and the extra `test_increment_of_a_missing_profile_raises`; T3 as six tests (penalty 2 cases, repair-cost keys 2, replacement and depot row 2, selector 2, pwl 3, anchors 2); T6 `test_loop_rows_cover_states_and_anchors` (2 cases). | ast OK; names_check 0; ALL PASS |
| B3 | Done. T4 `test_dynamics_reproduce_the_hand_case` (4 cases), T5 `test_wiener_variance_is_kept_by_a_repair` (4 cases), T7 `test_repair_cost_on_transitory_days_is_computed_after_the_solve`; constants `_TRAIN_0`, `_V`; helpers `_both_trains`, `_assert_mean_trajectory`, `_assert_untracked_keys`. T4 and T5 sit before T6 in the file. | expected arrays agree with `hand_case.py` for ard1 and ardinf, Gaussian-like and Wiener-like |
| B4 | Done. T8 `test_gaussian_reliability_forms` (5 cases) and `test_bernstein_reliability_row` (3 cases); T9 `test_objective_equals_the_cost_report` (4 cases); T10 `test_solve_mixed_returns_the_documented_keys`; T11 `test_infeasible_solve_reports_no_solution`; T12 `test_unknown_warm_start_mode_is_rejected`; T13 `test_warm_start_reports_skip_the_relaxation`. Last check: every `base.<name>` in the file is a name of section 2 (the registry names `CELL_BUILDERS` and `get_cell_builder` are used by the plan's own T1 and `_register`). | T8 constants agree with the plan (0.2378416, 0.2354632, 0.2283280, c2 0.2171472, c1 -0.1, side 0.7697415, range_v 0.1171472, cap 0.0364021) and the flags of all 8 cases match a direct evaluation of the rows |

## Problems

None blocks the file. Points for the planner and for the licensed run:

1. **Not run with Gurobi.** Tests T3 to T13 need gurobipy; only the pure parts were checked (fixture, expected values, T2 on the real `increment`).
2. **Extras beyond the plan** (small, each tied to a name or rule of Task A): the `KeyError` test for a missing profile (A1); `sel_*` row count in the selector test (A4 names `sel_{i}_{l}_{k}`); `anchor_vars["v"]` count; the size of row `depot_0_0_0` (3 terms with replacement, 2 without; Eq. `a_depot`); rows `loop_mu_0_0`, `loop_v_0_0` present and `loop_gv_0_0` absent under `ardinf`; a third pwl case (`tangent` creates no `w`); `set(out) == _RESULT_KEYS` in T11.
3. **Gurobi behaviour that the tests assume**: `getConstrByName` returns `None` for a missing name; `Model.getRow(constr).size()`; `float * LinExpr` with the float 0.0 (the Wiener-like train 1 has `sigma = 0`); the builders set `NonConvex = 2` in `prepare` when a cell has `impl_of == "exact"` (spec 247), also the `_RowOnlyBuilder`.
4. **T9, case `total`/`quadratic`.** The plan says "(form exact)"; read as `reliability_impl="exact"` for that case, so the model has `NonConvex = 2` and the convex quadratic penalty row.
5. **T11 infeasibility.** With `mu` increments 0.9, `mu_0 = 0.2` and `tau = 1`, whichever train flies on day 1 exceeds `tau`; a train cannot repair and fly on the same day, so the model is infeasible whatever the encoding.
6. **Warm-start modes.** The plan has no test for `"hint"` and `"both"` (they raise `NotImplementedError` until WP7); T12 tests only an invalid mode. Not added.
7. **Import chain.** The file imports `fleet_management.degradation_model.base`, which runs `fleet_management/__init__.py` (imports `solver`, `horizon_sweep`, the validators and the plotters), as the other test files do. If one of them fails to import on the merge branch, the whole file errors at collection; this is outside Task B.
8. **Line endings.** The Write tool wrote LF; the file was converted to CRLF to match the other files in `tests/` (`core.autocrlf` is true).
