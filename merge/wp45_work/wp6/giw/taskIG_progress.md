# Task IG progress

Status: done (IG1, IG2, IG3). Next: Task C (another agent: `config.py` imports `ig_mean_cap` from
`inverse_gaussian` inside the check; `test_inverse_gaussian_replacement_cap` into `tests/test_config.py`),
then the planner's review.

## Steps

- **IG1 done** (builder and cap): created
  `Fleet_Management-merge\src\fleet_management\degradation_model\inverse_gaussian.py` (LF endings, as the
  Gaussian executor's file; no non-ASCII character) from the proto, changed by Claude's decisions 1 and 2: the four
  cap functions (`_ig_unit_log_cdf`, `_ig_unit_log_sf`, `_ig_unit_quantile`, `ig_mean_cap`, the code of plan
  section 4, C2; the three helpers are verbatim, `ig_mean_cap` has an adapted docstring) sit in the module below the
  guarded import of `base`; imports `math`, `numpy`, `brentq`, `log_ndtr`; no import from `config`; cap row
  `name="rel"`; registration under `if base is not None:`; the module docstring says that the cap lives here and that
  `config.py` imports it inside the check.
  Checks:
  - `ast.parse`: ok.
  - `names_check.py`: 3 reports, all `base` (`prepare`, `add_cell`, its list comprehension). Expected: the guarded
    import binds `base` in a `try` block and the tool does not see it (the Gamma plan says the same). No other name.
  - Lines over 100 characters: 0.
  - `ig_nogurobi_check.py` (fresh interpreter, a `sys.meta_path` finder refuses `gurobipy`): the module imports,
    `base is None`, `gurobipy` not in `sys.modules`; `ig_mean_cap` and `_ig_unit_quantile` give the three reference
    values (mu_bar 0.3823093163, 0.4627586840, 0.3469170694; q 1.8859953361, 1.9488253601, 3.5525960458) within 1e-9
    relative.
  - `import fleet_management` does not load `fleet_management.degradation_model.inverse_gaussian`: prints False.
- **IG2 done** (tests): created `Fleet_Management-merge\tests\test_inverse_gaussian.py` from the proto test file with
  three changes: the docstring says that the cap is tested here; the cap rows are `rel_{i}_0_{k}` (decision 2, not
  `rel_ig_...`); the new `test_inverse_gaussian_cap` (plan C4, calling `inverse_gaussian.ig_mean_cap` and
  `inverse_gaussian._ig_unit_quantile`). Nothing added to `tests/test_config.py`; `config.py` not edited.
  Checks:
  - `python run_giw_tests.py <wt>\tests\test_inverse_gaussian.py` (worktree module, not `--proto`):
    `9 passed, 0 failed, 1 need Gurobi`. The plan's `8 passed ... 1 need Gurobi` plus the new cap test. The one that
    needs Gurobi is `test_free_schedule`. The runner still prints "note: config.ig_mean_cap injected": the runner
    injects when `config` has no `ig_mean_cap`; the builder no longer uses it, so the note is harmless until Task C.
  - Mutations of the builder module (scratchpad `mut_ig\`, loaded with `GIW_PROTO_DIR` and `--proto`): `ig_mean_cap`
    returning `tau`, and returning `tau / q` (the cap of `lore`): each gives `6 passed, 3 failed`
    (`test_cap_boundary[1.01, False]`, `test_cap_rows`, `test_inverse_gaussian_cap`).
  - `ast.parse` ok; 0 lines over 100 characters.
- **IG3 done** (report): no change of the plan's text beyond the changes listed (decisions 1 and 2 of
  `wp6_interface.md`). See the final reply.

## Problems

- **Cap test skipped without Gurobi.** `test_inverse_gaussian_cap` needs no Gurobi, but the file starts with the
  module-level `gp = pytest.importorskip("gurobipy")` (the plan's pattern, as `tests/test_base.py`), so pytest skips
  the whole file, the cap test included, on a machine without gurobipy. I kept the plan's pattern. The Gurobi-free
  check of the cap is the scratchpad script `ig_nogurobi_check.py`, and Task C's
  `test_inverse_gaussian_replacement_cap` in `tests/test_config.py` runs without Gurobi once Task C has applied it.
  If Claude wants the cap test to run without Gurobi too: replace the module-level `importorskip` by a per-test
  skip, and read `base` through `inverse_gaussian.base`. Not done, because it departs from the plan's pattern.
- **`names_check.py` reports `base`** (3 lines) at the new module: expected, see IG1.
- **Line endings.** The new files have LF, as the plan allows (git has `core.autocrlf = true`). The existing worktree
  files (`base.py`, `test_base.py`) have CRLF in the working tree; the Gaussian executor's `gaussian.py` also has LF.
  The Gamma plan asks for CRLF for its own file; this plan does not.
