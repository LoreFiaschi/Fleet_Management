Status: done. `tests/test_gamma_validators.py` written (36 test functions, 59 runs). Before Task 2: 15 passed, 44 failed (as expected). Against a scratch prototype of the Task 2 modules (built from the plan): 59 passed, 0 failed. Next: Task 2 (another agent), then the planner's review.

# Task 1 progress (tests/test_gamma_validators.py)

## Steps
- [x] Read plan sections 0-4, 7; wp7_interface.md; spec lines 2461-2658; existing validators.
- [x] 3.1 helpers (`_data`, `_cfg`, `_result`, `_mc_data`, `_mc_result`, `_assert_record`, `_steps`, `_assert_plain`)
- [x] 3.2 replay tests (6 functions, 11 runs)
- [x] 3.3 exact tests (7 functions, 9 runs)
- [x] 3.4 Monte Carlo tests (8 functions, 12 runs)
- [x] 3.5 dispatcher tests (5 functions, 7 runs)
- [x] 3.6 series tests (4 functions, 5 runs)
- [x] 3.7 files and errors (6 functions, 15 runs)
- [x] checks without Task 2 (`task1_checks.py` in this folder)
  - file parses; imports only names that exist today (new names such as `_exact_tail` are read
    inside the test, so the module is collected before Task 2); names check: 0 undefined;
  - every input of the tests (17) loads in `load_config` with warnings as errors;
  - the hand results of A, B, C round-trip through `_save_results` (.yaml, .json, .h5) and
    `solver.read_result` with equal keys, float64 dtype and values (NaN equal); NaN in `z` survives
    .h5 and .yaml; the Monte Carlo results too;
  - expected values agree with `expected_values_output.txt` and `plan_prototype_output.txt`.
- [x] runner before Task 2 (`run_tests_gamma.py`): `15 passed, 0 skipped, 44 failed`. Passing: the
  series tests (4 of 5; `test_tail_bound_keeps_only_the_series` fails as the plan says), the
  Clopper-Pearson test and four Monte Carlo tests that the old code already satisfies. Failing for
  the gaps the plan lists (old code reads removed result keys such as `gamma_shape_bound`, refuses
  NaN in `z`, no known-constant rate, no anchor split, no `_integer` check, no `gamma_cells`).
- [x] run against a scratch prototype of Task 2 (no package file edited):
  `proto/*.py` (C1, C2, C3, C5, C6 written from the plan; file versions read with
  `solver.read_result`), `make_overlay.py` builds `proto_src/` (copy of src plus the prototype and a
  reduced `gamma_tail_bound.py`), run with `FM_SRC=<proto_src>`: `59 passed, 0 skipped, 0 failed`.
- [x] mutation check, `mutations.py` (19 one-line mutations of the prototype, each in its own copy
  under `mutations/`): 18 are caught by at least one test (two-sided replay, anchor starting at 0,
  cap = tau, no depot-day violation, no horizon check, no anchor split in exact and in Monte Carlo,
  repaired rates kept, rho = 1 keeps terms, reliability check off, rate 1 for a constant, `int()` on
  repetitions, mode lower-cased, unknown key ignored, mode "both" raising early, report path checked
  late, raise before write, folder not created). The 19th (z NaN not skipped) cannot be seen from
  outside, see Problems.

## Notes on the form
- `_mc_result(flights, repairs, model)`: `flights` are entries `(i, j, k)` of `x`, `repairs` are
  cells `(i, l, k)` of `m`; default: train 0 flies both days.
- `test_errors` of the plan is two tests: `test_errors` (replay, exact, Monte Carlo) and
  `test_errors_of_the_state_arrays` (replay, exact only).
- The unknown key of the block `validation` is its own test
  (`test_dispatcher_rejects_unknown_validation_keys`); the series tests are four functions.
- File-version tests write each report to `tmp_path / <name> / "out" / report<suffix>`, so a
  validator that does not write cannot pass on the file of another.
- Extra assertions beyond the plan, all from the spec: the reports dump to JSON with
  `allow_nan=False` (plain types), the events of the steps, `saved_mean`, `saved_anchor`,
  `reliability_slack`, `maximum_errors` keys, `excluded_cells` keys `component` and `reason`.
- The result dictionaries have every key of `RESULT_KEYS` (`None` where absent), float arrays and
  NaN for `null`, as `solver.read_result` returns them (checked by `task1_checks.py`).

## Problems
1. **Plan C1 against decision 1.** C1's `_read_pair` reads the result with `read_input`; the
   interface file says the file versions use `solver.read_result`. The tests are written for
   `read_result` (the prototype uses it). Task 2 should follow decision 1.
2. **NaN skip of `removed_mean` is not observable.** With `max(0.0, removed - z)`, a NaN `z` gives
   error 0.0 in Python, so `test_replay_skips_nan_repair_cost` (plan: "valid") passes with or without
   the skip. I added `maximum_errors["removed_mean"] == 0.0` and the JSON-dump check, which fail if
   a NaN error is stored. A stricter test is not possible from outside.
3. **Counts the exact test relies on** (not stated in the plan): `exact_tail_evaluations == 3` for
   case A (days 0 to 2 of train 0; an equal-rate tail counts as one series term), and
   `maximum_series_terms_used >= 1`. If C3 counts differently, change that line.
4. **Names the tests read** that the plan fixes only in prose: `ExactGammaTerm(shape, rate, source,
   step)` (existing dataclass), `_exact_tail(terms, constant, tau, convolution_tolerance,
   max_series_terms)` called positionally and its keys `estimate`, `upper_bound`, `remaining_mass`,
   `series_terms`; step keys `terms[*]["part"]`, `shape`, `rate`; `maximum_errors` keys of the exact
   report `reliability`, `tracked_mean`, `mean_anchor`.
5. The module uses LF line ends (`tests/test_gamma.py` is CRLF in this checkout; the git index of
   both is LF).
