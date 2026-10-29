Status: done (B1 to B6); awaiting the planner's review.

# Task B progress (greedy_warm_start.py, base.py warm starts, tests/test_warm_start.py)

## Steps
- B1 done: header, `GreedyWarmStartReport`, `_profiles`, `_construction_pass`. Check `check_b1.py`: case A activities `[1,-1,1,-1]`, `[-1,1,-1,1]`; no gurobipy loaded.
- B2 done: `_cell_quantities`, `_cell_rows`, `_row_cap`. Check `check_b2.py`: 44 cells of the greedy_ref cases agree with the reference (quantities, rows, caps, `_row_cap` on a grid); case G caps as in the plan.
- B3 done: `_reference_trajectory`, `_cell_excess`, `_best_trial`, `_gains`, `_loop_pass`, `build_greedy_warm_start` (Wiener rule of decision 2). Check `check_b3.py` (gurobipy blocked): the 10 cases of greedy_ref equal the reference with `joint_models=("wiener",)` (x, m, r, report); the 7 rows of the table of 3.2 match; source keys, shapes and the ValueError of repair_trigger (-0.1, 1.1, nan) OK. `verify_greedy_lp.py --wiener`: agreement True. Timing: F=8, M=4, L=5, H=[10, 30] Gaussian: 0.11 s.
- B4 done: `warm_start_day`, `apply_binary_warm_start(ctx, source, mode)` in base.py. Checked by `test_warm_start_day`, `test_apply_modes`, `test_apply_partial_sources` with `--fake`.
- B5 done: `complete_binary_warm_start(model, source, report, mode)` (the copy runs only in start/both with an applied value), `apply_relaxation_warm_start` disposes the copy at both returns, `solve_mixed` calls the new signatures (the NotImplementedError and `has_source` removed). `grep NotImplementedError base.py`: only `get_cell_builder` (line 169). `names_check_try.py`: 0 names for base.py and greedy_warm_start.py. `import fleet_management` loads no gurobipy. The diff of base.py has hunks only in the three warm-start functions, the new `warm_start_day` and `solve_mixed`.
- B6 done: `tests/test_warm_start.py` (CRLF). `run_tests_wp7.py`: 12 passed, 13 skipped (no gurobipy), 0 failed. With `--fake`: 23 passed, 2 skipped (`test_solve_mixed_modes[hint]` and `test_relaxation_start`, the two "Gurobi only" tests), 0 failed.
- Line endings: greedy_warm_start.py, base.py and test_warm_start.py are CRLF (0 lines with LF only).
- Other test files with `--fake` (not edited): test_gamma 34 passed, 2 skipped; test_inverse_gaussian, test_gaussian, test_wiener, test_io pass. test_rainflow: 5 failures from the emulator (`getConstrByName` missing), none touches warm starts. test_base.py does not load in the runner (`pytest.param` missing).

## Problems
1. `apply_binary_warm_start` validates `H1`, `H2` and the shape of `x` (plan B4, Q4). It does not validate the shape of `m` and `r`: an `m` or `r` with a last axis shorter than `H1' + H2'` raises `IndexError`. The spec (2114) says they have shape `F' x L' x T'`; the plan names no error. Not added.
2. `apply_binary_warm_start` called directly with a mode outside start/hint/both sets nothing and still reports `values`. `solve_mixed` rejects such a mode first (its `ValueError` stays); the plan names no check in the function.
3. `H1` and `H2` of a source go through `int(...)` as the plan says, so `2.7` is read as `2` and `True` as `1`.
4. The Wiener sentence of plan 7.1 is not yet in the spec (Claude adds it); the module and the tests follow decision 2.
5. `_cell_excess` counts `v <= v_max_user` (Q3) as the plan says; spec 2234 does not list it yet (spec sentence for the reviewer).
6. The tests `test_solve_mixed_modes` and `test_relaxation_start` were not run against a real Gurobi (none on this computer). The other model tests ran only with the linear emulator.
