Status: job 1 done in code (spawn + new test); two existing tests in tests/test_horizon_sweep.py now fail (see below); job 2 (recycle 28 files) next.

## Job 1
- src/fleet_management/horizon_sweep.py: `import multiprocessing`; `ProcessPoolExecutor(max_workers=..., mp_context=multiprocessing.get_context("spawn"))`. CRLF kept.
- tests/test_spec_sweeps.py: new test `test_parallel_workers_start_with_spawn`. Whole file: 85 passed, 4 skipped (Gurobi), 0 failed.
- tests/test_horizon_sweep.py: 30 passed, 5 failed.
  - 3 fail only because the local runner has no `pytest.approx` (test_grid{evaluation_horizon}, test_rows_ranking_and_errors, test_stop_cost_increase).
  - 2 fail because of the change: `test_parallel_cases` (line 420) and `test_grid_with_parallel_cases` replace `ProcessPoolExecutor` by `ThreadPoolExecutor`, which rejects `mp_context`. Fix (not allowed in my scope): replace by `lambda max_workers, mp_context=None: ThreadPoolExecutor(max_workers=max_workers)`.
- Picklability: `_solve_case` is module level; its arguments are plain data (dict from YAML, warm-start dict or None, str, bool). Under spawn the worker imports fleet_management.horizon_sweep itself. No problem found.

## Job 2
Status: done. Script `scratchpad\wp8\finish\recycle28.ps1`: dry run 28 listed, 0 skipped; real run 28 recycled, 0 skipped.
`git status --short`: exactly 28 ` D` lines (24 in examples/regression, 4 in input), no other deletion.
Remaining references to removed names (not edited): README.md:47 (`input/data.yaml` quick start); tests/test_ported.py:13 (docstring, historical);
lore-vs-main.md:401-402; merge/reports/C_tooling.md, merge/wp1_removals.md, merge/spec_drafts/status_D.md (records).
Overall Status: done, except the two test_horizon_sweep.py tests that need the mp_context fix.

## Follow-up
test_horizon_sweep.py: two fake executors now accept mp_context (CRLF kept); runner has pytest.approx. See final counts in the reply.
