Status: in progress. B1 and B2 done. Next: B3 (HDF5 writer in solver.py, tests B3 in tests/test_io.py).

# Task B progress (solver.py, tests/test_io.py)

Worktree: C:\Users\utente\Documents\GitHub\Fleet_Management-merge. Runner: scratchpad\wp4\run_tests.py.
Command: `cd <scratchpad>\wp4 && PYTHONIOENCODING=utf-8 python run_tests.py <worktree>\tests\test_io.py [filter]`
Old solver.py kept as scratchpad\wp4\solver_old.py (reference only).

## Dependency on Task A
`config.read_input` and `INPUT_EXTENSIONS` (step A1) exist in config.py since B1 (checked 29 Oct 2026).

## Steps
- B1 done: solve path (solve, solve_mapping, _check_results_path, constants, `_read_input` alias). Old dispatch, readers removed.
  Check: tests B1, 9 passed (with the two file-writing tests passing after B2).
- B2 done: `_save_results`, `_to_plain`, `_text_output`, `_save_yaml`, `_save_json`. Check: 14 passed, 0 failed (B1 + B2 tests).
  Choice not in the plan: `_result_format(path)` (yaml/json/hdf5, or the ValueError with the message of the results file type)
  is shared by `_check_results_path` and `_save_results`, so the message exists once.
  `_check_results_path` does not skip the checks when the parent is the current folder (the plan kept s:50-53 `!= Path("")`; spec: directory must exist and be writable).
- B3: in `_save_hdf5` the old writer is still in solver.py until B3 is done.

## Problems
(none yet)
