# Task G progress (Gamma cell builder)

Status: done (G1 to G5). Next: the planner reviews; Claude removes the two old test files at the commit.

## Steps
- G1 done. `degradation_model/__init__.py`: 6 lines, CRLF, one docstring expression, no import statement.
- G2 done. `remove_gamma_block.py` output: "removed docstring lines 26-29 and block lines 184-1209 (1026 lines); 2358 -> 1328 lines".
  Checks: `load_base.py` loads; `names_check.py` 0 undefined names outside and inside the old block;
  `local_checks.py` `failures: 0`; `git diff --stat` base.py: 1030 deletions, no insertion;
  `grep -n -i gamma base.py`: one line, 525 (the `"gamma_beta"` profile key in `build_context`).
- G2b done. `tests/test_base.py`: one line `base._load_builders(["gamma"])` inserted as first line of
  `test_unknown_model_has_no_cell_builder` (CRLF kept: 777 CRLF = 777 LF; `ast.parse` ok; diff: 1 insertion).
  Extra check (`t1_check.py`, gurobipy stub, since `test_base.py` needs Gurobi): registry empty before the load,
  `('gamma',)` after it, message "... available: ('gamma',)." matches "gamma": T1 body passes.
- G3 done. `python copy_crlf.py gamma_planned.py <dm>\gamma.py` (202 lines, CRLF, longest line 99). Read against
  sections 1 and 2: guard, six constants, five functions, class, row name `rel`, registration all present.
  Checks: `model_cases.py <dm>\gamma.py`: 88 `ok` lines, `failures: 0`. `names_check.py <dm>\gamma.py`: 0 outside the
  block; inside: only `base` in `GammaCellBuilder.prepare`, `add_cell` and its list comprehension (expected).
  With `gurobipy` blocked and PYTHONPATH on the worktree `src`: prints `None 0.744517325`; `base` is not in
  `sys.modules`.
- G4 done. `python copy_crlf.py test_gamma_planned.py <wt>\tests\test_gamma.py` (256 lines, CRLF, longest line 99).
  `python run_tests_gamma.py <wt>\tests\test_gamma.py`: `21 passed, 15 skipped, 0 failed` (the 15 model cases need
  gurobipy).
- G5 done. `git status --short` for my files: modified `__init__.py`, `base.py`, `gamma.py`, `tests/test_base.py`;
  new `tests/test_gamma.py`. (Other entries in the status belong to the other executors: `rainflow.py`,
  `gaussian.py`, `inverse_gaussian.py`, `wiener.py` and their tests.)

## Problems
- None that change the result. Notes:
  - `wp4\run_tests.py` cannot run `tests/test_gamma.py` (its pytest stub has no `mark` and no `importorskip`);
    `run_tests_gamma.py`, which the plan names, does. Not a defect of the files.
  - `tests/test_gamma_reliability_bound.py` and `tests/test_gamma_transitions.py` are untouched, as ordered. Pytest
    cannot collect them (they import `GammaModel`, `GammaAction`, `maximum_reliable_shape`) until Claude removes them.
  - The 15 Gurobi model tests of `tests/test_gamma.py` and `tests/test_base.py` were not run (no Gurobi here);
    the emulator check `model_cases.py` reproduces the same expected values (88 checks).
