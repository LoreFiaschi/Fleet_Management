Status: done.

# M5b: non-finite values in `optimization_progress` records

- `base.py` (CRLF kept: 1527 `\r` before and after): `_make_progress_callback` and `_append_final_progress` pass `incumbent`, `best_bound` and `relative_gap` through `finite_or_none`. Existing rules kept: Gurobi infinity (>= 0.5e100) gives `None`; the timing rule of records and the two gap formulas are unchanged.
- New `tests/test_step3_progress.py`: 11 tests with fake models, no Gurobi (inf, -inf, NaN, each entry apart, finite values kept, callback gap 0.2).
- Runs (runner with gurobipy stub): test_step3_progress 11 passed; test_step3_base 35 passed, 8 skipped; test_io 18 passed; test_spec_outputs 15 passed, 128 failed with `RuntimeError: gurobipy stub: no model can be built` (tests need a real Gurobi; not caused by this change, they build models).
