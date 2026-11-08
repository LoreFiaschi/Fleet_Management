# Task I1 progress

Status: done. Library ready; `tests/test_spec_inputs.py` written; local run 83 passed, 0 skipped, 0 failed.

- Library: 8 files in `Fleet_Management-merge/tests/data/io/valid/` (gamma, gaussian, gaussian_ard1_forms, inverse_gaussian,
  minimal, mixed_example, rainflow_bounds, wiener). All pass `load_config` locally.
  `mixed_example.yaml` = spec lines 2769-2827 verbatim plus one leading comment line.
  `minimal.yaml` has no run options (so the `verbose` default 1 can be tested); F=3, M=1, L=2 via `model: [gaussian, gaussian]`.
- Tests: `Fleet_Management-merge/tests/test_spec_inputs.py`, 83 cases (parametrised): library, defaults, sizes, cell values and
  selectors, profiles, gamma rates, integers, null rules, case sensitivity, formats (YAML .yml/.YAML, JSON, HDF5 .h5/.hdf5 built with h5py).
- Local run: runner copy in the scratchpad (`run_tests_gamma.py`), 83 passed, 0 skipped, 0 failed. Under real pytest the module imports
  `fleet_management` (package `__init__`), which needs gurobipy (Euler has it); the local runner bypasses `__init__`.
- Code defects found: none.
- Open points:
  - Plan 3.1 asks for a test of the ambiguous profile shape (F = L, array fitting both (L,M,H*) and (F,L,M)). The shapes coincide
    only if F = L = M, which F > M (spec 369) excludes, so the case cannot occur. The test
    `test_profile_ambiguous_shape_cannot_occur` records this and checks that the two near-identical shapes (2,2,3) and (3,2,2) are read
    as their own shapes. Spec 435 may be intended for another reading ("fits" by broadcasting); the user may want to clarify.
  - I read `config.py` beyond names and field lists (the broadcasting helpers) before writing the tests, while looking up
    signatures. Expected values were still derived from the spec; no expected value was taken from a function body.
  - The tests import `read_input` and `load_config` from `fleet_management.config`; `mip_gap`-style options are read from `cfg.options`.
  - `cfg.mu_trans is None` when absent, and `cfg.gurobi_params`-style defaults (`None`) are asserted through `cfg.options`; the spec says "none".
