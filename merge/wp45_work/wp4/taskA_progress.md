Status: done (29 Oct 2026). Steps A1 to A10 finished; tests/test_config.py: 55 passed, 0 failed. Next: the planner's review.

Files: C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src\fleet_management\config.py and tests\test_config.py
Runner: PYTHONIOENCODING=utf-8 python <scratchpad>\wp4\run_tests.py <test file> [filter]
Note: config.py has CRLF line endings; the Edit tool preserves them. tests/test_config.py is a new file (LF).
Resumed 29 Oct 2026 by a second executor: found A2 and A3 already in the code but not recorded; checked them (tests: 10 passed; check_a3.py in scratchpad output matches the plan's messages).

## Steps
- A1 done (29 Oct 2026): read_input, _read_hdf5, _hdf5_text, _hdf5_value, INPUT_EXTENSIONS at the end of config.py; imports json, warnings, Path, ndtri added. Tests: 5 passed (read_input_*).
- A2 done: key tables (SUPPORTED_MODELS ... CANONICAL_KEYS 63, REJECTED_KEYS 21 via _REJECTED_ROWS, _check_keys) and the first line of load_config. Tests: test_rejected_keys, test_unknown_key, test_missing_required_keys, test_key_check_runs_before_arrays, test_input_must_be_a_mapping pass.
- A3 done: _integer, _number, _flag, _parse_horizon, _infer_L, _component_names, _run_options (all in config.py). Check: scratchpad check_a3.py prints the plan's messages.
- A4 done: section "Cells, selectors and profiles" after _parse_horizon: _first_cell (extra private helper), _float_array, _warn_not_applicable, _selector, _masks, _cell_value, _profile. Check: scratchpad check_a4.py reproduces the shape tables (F=3, L=2, M=2, Hs=3); 10 tests still pass. Old helpers (_fl_scalar ...) stay until A5.
- A5 done: new FleetConfig (plan field list; models property kept), new load_config; deleted _fl_scalar, _fl_str, _component_cost, _flmh_prof, _gamma_rate_prof, _validate_cells. Layout of config.py now: Keys and options; Normalized representation; Scalars; Sizes and names (_parse_horizon, _infer_L, _component_names); Run options (_run_options); Cells, selectors and profiles; Public loader; Input reader. A6 to A9 add a "Consistency checks" section between "Cells, selectors and profiles" and "Public loader". Surgery script: a5_surgery.py (run once; backup tmp_config_before_a5.py.bak). Tests: 29 passed (A1, A2, A5 tests). test_cost_values lacks the "C_R negative at one cell" case until A6 (add it then).
- A6 done: section "Consistency checks": _TOL, _le (relative to both sides, as the user's answer says), _require, _check_state, _derive. Tests added (test_tau_epsilon_rho ... test_derived_fields, and the C_R case of test_cost_values). Derived values match the plan to 1e-12.
- A7 done: _lookup, _check_increments (single-key rows first, then the looked-up pair rows). Tests added, incl. test_tolerance.
- A8 done: _check_models (all rows of the plan table; Be tau row before the two sqrt rows). Tests added.
- A9 done: _quad_coefficients, _cap, _check_ranges. Cap values checked by scratchpad check_a9.py against the plan's numbers (0.0060887, -0.14613, 0.00084566, 0.0011111, 0.0086859, side 0.96162). Tests added. IG and Gamma replacement caps are deferred to WP6 (plan section 4).
- A10 done: module, FleetConfig and load_config docstrings rewritten; _derive moved before _check_models (definition order = call order); forbidden names are gone (grep: gamma_beta_bound and gamma_calibration_method occur only in _REJECTED_ROWS). Extra scratch checks: check_a10.py (tolerance mutation: without _le the Chernoff case 5*0.14 vs 0.7 fails, with _le it passes).

## Final state
- config.py: 966 lines, CRLF kept. Order: Keys and options; Normalized representation; Scalars; Sizes and names; Run options; Cells, selectors and profiles; Consistency checks; Public loader; Input reader.
- load_config order of errors: keys; F, M; H; L; selectors; cell values; profiles; pwl_points; flags; C_M, C_D; penalty_type; component_names; run options; then _check_state, _check_increments, _derive, _check_models, _check_ranges.
- Imports at module level: json, math, warnings, dataclasses, pathlib, typing, numpy, scipy.special.ndtri (no yaml, h5py, gurobipy).
- tests/test_config.py: 55 tests, plain functions, only the fixture tmp_path.

## Problems
- Observation (not a deviation): `_float_array` follows the plan (`np.asarray(raw, dtype=float)`), so a numeric string such as "0.5" and a bool (`true`) are accepted as numbers in cell values and profiles. `_number` (C_M, C_D) refuses bools. The reviewer may want `_float_array` to refuse str and bool entries.
- Choice: `_le` is used for every comparison of two quantities (anchor <= initial value, 3*sqrt(v) <= mu, cgf >= s*mu, row caps, ...). Comparisons with a literal zero (`>= 0`, `== 0`) and strict inequalities stay exact, so `-1e-13` is still refused as negative.
- Choice: `_selector` and `_cell_value`/`_profile` do not check values at cells where the key does not apply, except the allowed-values check of selectors (spec: a selector entry there is still checked).
- Note: the `replacement_as_new` check (replacement inputs must be 0) runs also when `allow_replacement` is false, as the plan's table and the spec say (condition: the flag only).
- Extra private helper not in the plan: `_first_cell(mask)`.
- Test numbers in the plan (section 3) all reproduced; no test needed a changed value.
