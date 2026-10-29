Status: done. All steps finished; 57 passed, 0 failed; import check without gurobipy ok.

# Task C progress (config.py and tests/test_config.py)

- Step 0, baseline: `run_tests.py tests\test_config.py` gives 55 passed, 0 failed. `inverse_gaussian.ig_mean_cap` and
  `gamma.mean_cap`, `gamma.smallest_rate` exist; both modules guard `base` and load without gurobipy.
- Step 1, GIW part: in `config.py`, `_check_ranges`, after the Markov check, the block of C3 with the first line
  `from fleet_management.degradation_model.inverse_gaussian import ig_mean_cap` (no C1, no C2). In
  `tests/test_config.py`, after `test_replacement_rows_off`: the banner comment and `test_inverse_gaussian_replacement_cap`
  (no `test_inverse_gaussian_cap`). Result: 56 passed, 0 failed.
- Step 2, Gamma part: the Gamma check (plan section 5, C1) directly after the IG block; the module docstring (C2) as the
  plan words it; `test_gamma_replacement_cap` (C3) after the IG test. Result: 57 passed, 0 failed.
- Step 3, import check: `taskC_import_check.py` in this folder. A `sys.meta_path` finder refuses `gurobipy`;
  `import fleet_management` (the real `__init__.py`) and `fleet_management.config` import; `load_config` accepts
  IG `replacement_mu` 0.54 and Gamma 0.7437, rejects IG 0.56 and Gamma 0.7453 with the expected messages;
  `gurobipy` is never in `sys.modules`; `gamma.base` and `inverse_gaussian.base` are `None`. Output: "import check: ok".
- Line length: no added line is longer than 99 characters (three lines of 100 characters are older).
- Line endings: `config.py` is all CRLF and `tests/test_config.py` all LF, as before; the Edit tool kept them.

## Problems
1. The module docstring of `config.py` (C2 of the Gamma plan, applied as written) names only the Gamma cap as imported
   inside its check. The inverse Gaussian check does the same now. Suggested wording if Claude agrees: "(``yaml`` and
   ``h5py`` are imported inside the reader, and the caps of ``degradation_model.inverse_gaussian`` and
   ``degradation_model.gamma`` inside their checks), so it runs without Gurobi." Not applied: the brief says to follow
   the plans exactly.
2. The IG plan's C3 text does not say which of the two checks comes first; the IG block is placed first, the Gamma
   block after it, both after the Markov check. The order does not change any result.
