Status: done. Steps read_result (solver.py), A1 to A6 finished; 36 test runs pass with run_tests_validator.py. Next: the planner reviews.

# Task A progress (validator, tests, fixtures)

## Steps
- (Decision 1) `read_result` in `solver.py`, below `_save_hdf5`, section "Result files: reading": done. Check: `check_A2.py` (scratchpad): reads the three files `result.{yaml,json,h5}`, equal float64 arrays, `R` all NaN, `H == [2, 3]`, `.txt` raises the ValueError. Passed.
- A1 fixtures: done. `fixture_values.py` output identical to `fixture_values_out.txt`; the regenerated `result.yaml` is byte-identical to the one the plan names. Copied to `results/validator_tests/input.yaml` (47 keys) and `result.yaml` (51 keys, objective 533.8820000000001, u 0.56, status optimal).
- A2 head and reader: done. Check: `check_A2.py` passed (module imports without gurobipy; `validator.read_result is solver.read_result`).
- A3 checks: done. Check: `check_A3.py` compares `check_result` with `reference_checks.py` on the fixture, the 15 corruptions, `allow_replacement=False` and the four objective modes; violations equal exactly.
- A4 validate: done. Check: `check_A4.py`: 14 check names, passed True, the YAML file equals the returned dictionary, the folder is created, `.json` works, nothing printed.
- A5 tests 1-9: done, the first 150 lines of `tests/test_validator.py`.
- A6 tests 10-17: done. The file has 249 lines. `python run_tests_validator.py tests/test_validator.py`: 36 passed, 0 failed (parametrized runs counted). Each failing-check test asserts that no other check fails (`_failed(...) == {name: approx}`).
- Extra checks: the names check (`names_check_try.py`) reports nothing for `validator.py`, `solver.py` and the tests; no unused import; `import fleet_management` works with the real package and `fleet_management.validate is validate`. Five mutations of `validator.py` (mission coverage with the depot, objective violation with abs, depot gating removed, NaN not skipped in the loop rows, replacement cost removed) each make at least one test fail (copies in `scratchpad\wp7\validator\mut\`).

## Problems
1. The plan's step A2 lists the constants `ARRAY_KEYS` and the import of `RESULT_KEYS` and `_HDF5_DATASET_KEYS` for `read_result`. With `read_result` in `solver.py` (Decision 1) none of the three is used in `validator.py`, so I did not define `ARRAY_KEYS` and did not import `RESULT_KEYS` or `_HDF5_DATASET_KEYS`. `validator.read_result` stays importable (Task B imports it from there).
2. I kept the three header comment lines of the old file (author and date of creation) above the new module docstring. They are attribution, not legacy code.
3. Test 12 has one more case than the plan: a result without the key `K` still gives the 14 checks and passes (Question Q4, the key `None` branch of `_loop_violation`).
4. Test 2 compares the whole `checks` lists of the two reports (names, passed, violation, tolerance), not only names, passed and violations.
5. Two docstrings of `validator.py` name the spec section as "Validator" instead of "Structural Validator", to keep lines at 99 characters or less (the longest line of `solver.py`).
6. The plan's A1 check says "`config.read_input` of the copy has 51 keys": 51 is the key count of the result copy; the input copy has 47 keys.
7. A helper function of my mutation check contained a stray `rm -rf /dev/null`. The system refused it ("Operation not permitted"). No file was deleted.
8. The Gamma-validator and dashboard names that Tasks B and C import from `validator.py` (`LOOP_KEYS`, `check_result`, `check_sizes`, `read_result`) all exist.
