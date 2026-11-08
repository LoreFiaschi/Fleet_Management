Status: done. G1 to G3 and W1 to W3 finished; nothing is left for tasks G and W. The tests that need Gurobi (3 + 1) run on the licensed machine (WP8).

# Task G and W progress

## G1 builder (done)
- Created `src/fleet_management/degradation_model/gaussian.py` as a copy of `proto/degradation_model/gaussian.py`; it equals the code block of plan G1 byte for byte (checked by comparing with the plan text).
- Checks: ast parse ok; `names_check.py`: 0 undefined names (outside and inside the Gamma block); no line over 100 characters; LF endings only; importing `fleet_management` does not load `fleet_management.degradation_model.gaussian` (prints False).
- Note: the plan names `python ..\..\wp5\names_check.py`; from the `wp6\giw\` folder the working path is `..\..\wp5\names_check.py` (it is `scratchpad\wp5\`).

## G2 tests (done)
- Created `tests/test_gaussian.py` as a copy of `proto/tests/test_gaussian.py` (byte-identical, checked with `cmp`); read against plan section 5.1: every listed case is there (registered, dynamics 4 cases, rows_and_variables, exact_form_is_quadratic, pwl_form, v_max_user 2 cases, reliability_forms 3 cases, free_schedule).
- Checks: ast ok; names_check 0; no line over 100 characters; LF only.
- `python run_giw_tests.py <wt>\tests\test_gaussian.py`: `11 passed, 0 failed, 3 need Gurobi`. The three that need Gurobi: `test_free_schedule`, `test_reliability_forms['exact', True]`, `test_reliability_forms['pwl', True]` (as the plan expects).

## G3 report
- Deviations from the plan: none.

## W1 builder (done)
- Created `src/fleet_management/degradation_model/wiener.py` as a copy of `proto/degradation_model/wiener.py` (byte-identical, `cmp`); read against the bullets of plan W1 (docstring, class and `name`, `prepare` with `anchored=False` and the two extras, `sigma` read after `kappa`, the comment lines, the `"v"` call with `sigma ** 2 * base.mission_count` and `repair_keeps_state=True`, the registration line): all present. The spec labels in the comments (`w_mu_carry`, `w_anchor_keep`, `w_map_v`, `w_vrow`, `w_v_repl`, `g_row`) exist in `spec/spec.tex` of the worktree.
- Checks: ast ok; names_check 0 and 0; no line over 100 characters; LF only; importing `fleet_management` does not load the `wiener` module (prints False).
- `base.py` has the helper signatures the builder calls (`add_state_vars(..., anchored=)`, `add_quantity_rows(..., repair_keeps_state=)`, `mission_count`, `add_upper_cap_rows(..., name=)`), checked in the worktree's `base.py` while the other agents edit it.

## W2 tests (done)
- Created `tests/test_wiener.py` as a copy of `proto/tests/test_wiener.py` (byte-identical, `cmp`); read against plan section 5.3: all listed cases are there (registered, dynamics 4 cases, rows_and_variables, exact_form_is_quadratic, loop_needs_replacement, v_max_user 2 cases, free_schedule).
- Checks: ast ok; names_check 0; no line over 100 characters; LF only.
- `python run_giw_tests.py <wt>\tests\test_wiener.py`: `10 passed, 0 failed, 1 need Gurobi` (`test_free_schedule`).
- Both test files were run a second time at the end, after the other agents had changed `base.py`: same counts (11/0/3 and 10/0/1).

## W3 report
- Deviations from the plan: none.

## Problems
- None in the builders or tests.
- The emulator (`fakegp`) cannot check these: `test_free_schedule` of both files, and `test_reliability_forms['exact', True]` and `['pwl', True]` of the Gaussian file. They need the licensed machine. In particular, the last assertion of the Wiener `test_free_schedule` (the output arrays `x` of shape `(F, M+1, T)` and `r` of shape `(F, L, T)` indexed as `x[i, 1:, H1:]`, `r[i, 0, H1:]`) matches the shapes in `base.extract_solution` by reading, but was not run.
- The plan's path `python ..\..\wp5\names_check.py` is correct only from `wp6\giw\` (the check was run that way).
- `run_giw_tests.py` prints "note: config.ig_mean_cap injected from the scratchpad prototype ig_cap.py" because Task C has not run yet; this concerns IG, not G or W.
