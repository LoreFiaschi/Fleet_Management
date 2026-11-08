Status: done. `tests/test_spec_warm_starts.py` written (35 tests); local run 17 passed, 18 skipped (Gurobi), 0 failed. Next: Euler run by Claude for the 18 Gurobi tests.

# T3 progress

- [x] Read interface, plan, spec Warm Starts
- [x] Read signatures in src (names only)
- [x] Greedy scripts in scratchpad (`C:\Users\utente\AppData\Local\Temp\claude\97949f6cddfa4b53\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad\wp8\T3\spec_greedy.py`): scenarios A, B, C, D, E, L1, L2, F, W agree with the hand derivations
- [x] Inputs in `tests/data/tools/warm_starts/` (zero_increments.yaml, greedy_A..E, L1, L2, F, W)
- [x] tests/test_spec_warm_starts.py
- [x] Local run: runner copy `...\scratchpad\wp8\T3\run_t3.py` (adds `monkeypatch`, `approx`): 17 passed, 18 skipped, 0 failed

## Tests

No Gurobi (17): day-map formula, 4 sweep tests with the fake `solve_mapping` (sources skip error and no-solution cases, no source without `warm_start`, grid first case per H1, grid error case), 12 greedy tests (A with trigger 0.45, default trigger, B ties, C key, D repair factor, strict threshold, E components, trigger errors and bounds, L1 later day, L2 replacement, F failure, W Wiener rule).

Gurobi (18, skip here): invalid mode, invalid horizons, invalid axes, integer-valued horizons, source without x, no-source report, two day-map tests, value counts, indicator count, modes report, hint never completes, completion start/both, greedy source completed, completion without values, relaxation disabled / skipped / runs.

## Code defects

(none found locally)

## Spec findings

- Spec 2260: "tie to the repair, then to the later day" is read as: among equal excess prefer a repair, then the later day. The tests avoid a case where the reading matters (the replacement ties are between replacements only; the repair-replacement tie has one free day).
- Spec 2270: "largest excess term before the division by the natural ranges" is tested only on cases where the largest term is a loop term of the mean or of the variance.
- Spec 2271: whether `maximum_proxy_ratio` counts the initial state is not stated; the tests assert it only where a later state is larger (greedy_E has no proxy assertion).
- The report key `assignment_order` is read as 1-based mission numbers j (the spec numbers missions 1..M); 2272 does not say it. The tests pass, so the code agrees.
- Spec 1838: the type of `source_horizon` (list, tuple or mapping) is not fixed; the tests accept all three.
- Spec 1847 and 2176: `fixed_integer_variables` is asserted equal to `values` (every mapped binary is fixed); `values_copied > 0` after a completed completion. Both are readings of the spec.
- Spec 2155 and 1836: the selector n of the indicator form is counted in `values` (50 in the indicator test). If the code counts only x, m, r, this is a spec question.
- Day-map tests: Gurobi may return a solution other than the start; then only `values` is checked (stated in the test).

## Open points for the Euler run

- The base input has SolutionLimit 1 and all costs 0; the status of the result may be `solution_limit`; the tests do not read it.
- `test_invalid_mode_raises` passes `None` as a mode; the spec says "another value raises ValueError".
