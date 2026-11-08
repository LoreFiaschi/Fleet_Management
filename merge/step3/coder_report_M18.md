Status: done (not committed).

# M18: binaries rounded in the result

- `src/fleet_management/degradation_model/base.py`: new helper `_read_binary_values` (`np.rint`, clip to [0, 1], NaN kept). `extract_solution` reads `x`, `m`, `r` through it, so `idle` and `cost_report` (`step_costs`) use the rounded arrays. Other quantities unchanged. CRLF kept (1519 CR before, 1527 after, one per line).
- New `tests/test_step3_binaries.py`: helper on fake variables (0.99999901, 1e-7, 1.0000002, -3e-9, absent key), cost report from rounded arrays, and a Gurobi test (solve `gaussian.yaml`; `x`, `m`, `r` exactly 0 or 1; `idle` equals `x[:,0,:][:,None,:] - m - r`). The Gurobi test skips here.
- Runs (stub runner, no Gurobi), after: binaries 2 passed/1 skipped; test_step3_base 35/8 skipped/0 failed; test_step3_start 6/11 skipped/0 failed; test_io 18/0/0; test_spec_outputs 15 passed/128 failed (each failure is the stub's "no model can be built" RuntimeError, not an assertion; the file does not skip without Gurobi). Before: new file absent; other files untouched by the change, and the failures come from the stub.
- Not run: the real solve (needs Gurobi).
