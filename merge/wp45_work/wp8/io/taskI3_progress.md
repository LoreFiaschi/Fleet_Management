# Task I3 progress

Status: done. `tests/test_spec_outputs.py` written; local run 13 passed, 128 skipped (Gurobi), 1 failed (code defect or spec question, below).

- Files (worktree `Fleet_Management-merge`): `tests/test_spec_outputs.py`, `tests/data/io/infeasible_wiener.yaml`, `tests/data/io/results/partial.yaml`, `tests/data/io/results/nulls.yaml`. Loaded locally with `load_config`: the infeasible file passes the checks.
- No Gurobi (run locally): data flow (7 tests: missing input, `.txt` input, result extension and folder checked before the input is read, bad result extension), `read_result` on hand-written YAML, JSON and HDF5 files and `.txt`, public API in a subprocess with `sys.modules['gurobipy'] = None` (exports and non-exports).
- Gurobi (skip locally, run on Euler): 13 solve cases (`CASES`: the 7 library files, `C_D: 0`, quadratic penalty, `C_R: [0, 1.5]`, the three objective modes, scalar `H`), each checked for keys and shapes, selectors, NaN rules, `idle`, `u`, cost report, objective against the key of the mode; Eq. `c_zpost`; infeasible Wiener input; `performance` and `optimization_progress`; the five extensions, default `output.yaml`, no extension, `None`; YAML/JSON plain data and `null`; `objective` null and None keys absent; HDF5 layout (solved G fleet, IG fleet, mixed example, infeasible); round trip of `mixed_example` in three formats.
- Locally the helpers were exercised on every library file (sizes, per-cell models, bounds) and on the comparison function.

## Failing locally

- `test_bad_result_extension_with_missing_input_raises_value_error` (plan 3.3 reading). Input: a missing input file with result path `out.txt`. Observed: `FileNotFoundError` (the code checks that the input exists first). Spec 221 lists the input check first; spec 1880 says the extension and directory are "checked before the input file is read". The two lines disagree for a missing file. This is a spec question for the user. The two tests that follow 1880 unambiguously (the input exists but cannot be parsed) pass.

## Spec findings

- Plan 3.3 says records of `optimization_progress` have five keys; the table (spec 1818-1820) gives six: `runtime_seconds`, `incumbent`, `best_bound`, `relative_gap`, `nodes`, `solutions`. The test uses six.
- Spec 1739 gives `g_v` at G and RF (Ca, Be) only; a Wiener ARD1 cell has `v` but `g_v` NaN by the literal reading. The test follows it (no library file has Wiener with `ard1`).
- Spec 1727 does not say what `reliability_impl` is at cells where the form does not apply (IG, Ga); the test checks it only at cells where the input gives a value, and at G and W cells (default `exact`).
- Spec 1738 reads R at Ho cells and K at Ch cells only; the test asserts NaN of R at Ch cells and of K at Ho cells.

## Open points

- The Gurobi tests are unverified until the Euler run. Possible test errors there: the library files must be feasible and optimal in time (`status == "optimal"` is asserted for every case, including the quadratic penalty and `C_R: [0, 1.5]` variations); `optimization_progress` must have at least one record; the `performance` keys are checked as a subset.
- `mixed_example` is solved with `mip_gap` 0.05 and `time_limit` 60; the objective is then only checked as `>=` the key of the mode.
