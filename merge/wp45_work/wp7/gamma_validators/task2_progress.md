Status: done. C1 to C6 written in the four modules; `tests/test_gamma_validators.py`: 59 passed, 0 skipped, 0 failed (`run_tests_gamma.py`); `import_check.py` prints `ok`; names check 0 undefined in each module. Next: the planner's review.

# Task 2 progress (the four modules of gamma_utils)

Files edited (worktree `src/fleet_management/degradation_model/gamma_utils/`): `gamma_replay_validator.py`,
`gamma_tail_validator.py`, `gamma_tail_bound.py`, `gamma_stochastic_validator.py`. Nothing else in the worktree.
`gamma_diagnostics.py`, `gamma_process.py` and `GAMMA_DIAGNOSTICS.md` are untouched. The four files are written
with LF line ends (the git index has LF; two of them were CRLF in this checkout; `core.autocrlf` is true).

## Steps
- [x] C1 gamma_replay_validator.py: shared helpers (names check 0 undefined; the module imports alone;
  `import_check.py` could not print `ok` before C5, because the old stochastic module still imported the removed
  `_phase` and `_read`)
- [x] C2 gamma_replay_validator.py: the replay validator and its file version (the test module imports all four
  modules, so the check ran against a scratch overlay `overlays/c2` = worktree + the Task 1 prototype for the
  other modules, built by `task2_overlay.py`: replay tests 11 passed, 0 failed; whole module 59 passed)
- [x] C3 gamma_tail_validator.py: the exact validator (overlay `overlays/c3`: exact tests 10 passed; whole module
  58 passed, 1 failed = `test_tail_bound_keeps_only_the_series`, as the plan says before C4)
- [x] C4 gamma_tail_bound.py: the series only, vectorised (overlay `overlays/c4`: whole module 59 passed;
  `task2_quad_check.py` = `examples/regression/check_gamma_convolution_quadrature.py` with the worktree src: PASS,
  worst error 1.6e-12, 6 cases, max 554 series terms; `task2_tb_compare.py`: new code against the git HEAD code,
  7 cases, same number of series terms, worst difference of the upper bounds 1.7e-15; one term repaired 8 times =
  12,416 series terms in 0.54 s)
- [x] C5 gamma_stochastic_validator.py: the Monte Carlo validator (worktree, all four modules: `monte_carlo`
  tests 11 passed, 0 failed)
- [x] C6 gamma_stochastic_validator.py: the dispatcher and its file version (whole test module: 59 passed,
  0 skipped, 0 failed; 3.4 s)
- [x] extra: `plan_prototype.py` against the new modules: the numbers agree with `plan_prototype_output.txt` to the
  last digit of the series (1e-18); `fixture_check.py` runs.

## What differs from the plan, or goes beyond it
1. **Decision 1 applied.** `_read_pair` reads the result with `solver.read_result` and the input with
   `config.read_input` (the plan's C1 said `read_input` for both).
2. **Docstring of `gamma_tail_bound.py`.** The plan says "lines 37-46 only". Those lines call the series user
   "the optimizer" and name `partial_tail`; the calibrators are gone, so the text now says "the exact validator" and
   `estimate`. I added a title line and a pointer to the spec section.
3. **Equal-rate shortcut** keeps `scipy.stats.gamma.sf`; the series uses `scipy.special.gammaincc` (the two agree to
   roundoff). The series now sums in float64; the old code used `np.longdouble`, which is float64 on Windows (checked:
   same terms and mass at tolerances 1e-12 to 1e-16 here). On a Linux machine with 80-bit longdouble the old code
   could reach a tolerance below about 1e-16 where the new code cannot.
4. **Annotations.** The three file versions carry `str | Path` annotations; the plan listed `Path` among the imports
   of the exact validator without saying where it is used.
5. **`excluded_cells`** uses a set of the Gamma cells (a list would be quadratic in the fleet size).

## Problems
1. **`examples/regression/check_gamma_stochastic_validator.py`** (plan section 5 says it is not broken): its result
   has no `model` key, so the new `gamma_cells` raises "result is missing 'model'" (spec, "Common rules"). The script
   also builds its input with `gamma_beta_bound`, which `load_config` already rejects, so it was broken before.
   WP8 should add `model` to its result.
2. **`validation/dashboard_gamma.py:66-73`** (validator plan) calls `validate_gamma_tail_bound_schedule(...,
   include_steps=True)`; the step records now carry the keys of C3 (`i`, `l`, `k`, `exact_tail_upper_bound`,
   `terms`, ...), not the old `history` and `bound_tail`. The validator plan must read the new keys.
3. **Task 1 problem 2 stays**: with `max(0.0, removed - z)`, a `NaN` `z` gives error 0.0 even without the skip, so the
   skip of `removed_mean` is not observable from outside (the guard `math.isnan(z)` is there as the spec says).
4. The `import_check.py` script cannot print `ok` while the old stochastic module is in place (see C1). It prints
   `ok` now.
