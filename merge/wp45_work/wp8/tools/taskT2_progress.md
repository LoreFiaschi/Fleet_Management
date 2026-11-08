# Task T2 progress (sweeps)

Status: done locally. `tests/test_spec_sweeps.py` written; local run (runner copy with `monkeypatch` and `caplog`): 84 passed, 4 skipped (the four real-solve tests, Gurobi), 0 failed. Next: Euler run for the four Gurobi tests.

## Files
- `tests/test_spec_sweeps.py` (worktree `Fleet_Management-merge`).
- `tests/data/tools/sweeps/`: `gaussian_small.yaml` (fake tests, H = [2, 3]), `gaussian_scalar_H.yaml`, `gaussian_h_three.yaml`, `not_a_mapping.yaml`, `gaussian_real.yaml` (H = [1, 2]), `gaussian_profile_len2.yaml`.

## Code defects found
None (every local test passes).

## Spec questions (the spec is silent or ambiguous; the tests assert only what is fixed)
- `cost_gradient_per_H2` (spec 2039): the absolute gradient is not defined in the spec; not asserted. The relative one is (Eq. `d_gradient`).
- Gradient keys of a row that is `unavailable`, or of a case-error row (null or annotated?): spec 1984 says every row is annotated, spec 2045 says null in every entry of an error row. Tests exclude the four gradient keys from the null check of the error row.
- `formulation_growth` with fewer than two rows with counts (spec 2019): the parenthesis may mean null entries or a null mapping; the test accepts both.
- `optimizer_seconds` (spec 2036) has no stated source key in `performance`; only presence is asserted.
- Value of `warm_start.mode` in the report when warm start is disabled (spec 2015): not fixed; only the key is asserted.
- Shape of `dimensions` and `fixed_dimensions`: tests assume a mapping with keys F, M, L (and H1, H2, T). `H1_sweeps` (spec 2103): list assumed; a mapping of sweeps is also tolerated.
- `n_workers` bounds (0, negative) and the exact message of errors are not in the spec; not tested.

## Open points
- The four real-solve tests assume the small Gaussian fleet (F = 3, M = 1, L = 1, mu 0.05, 8-day horizons) is feasible and solves quickly with the loop rows; to be confirmed on Euler.
- `test_real_parallel_sweep` uses processes; the `monkeypatch` wrapper is used only in the serial real tests.
