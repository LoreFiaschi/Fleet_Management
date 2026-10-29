# WP6 (cell builders): split, file ownership and shared rules

Status (29 Oct 2026): fixed by Claude before the planners start; the three plans follow it. A change needs Claude's
agreement, because the other plans depend on it.

- **Purpose.** WP6 builds the five cell builders on the shared model of WP5. Three Opus planners plan it in parallel;
  this file fixes which files each plan may change and what the plans share.
- **Where.** Code: the worktree `C:\Users\utente\Documents\GitHub\Fleet_Management-merge`, branch
  `merge/lore-into-main`, at commit 5c8de19 (WP4 and WP5). Records (plans, reviews): the `lore` checkout,
  `C:\Users\utente\Documents\GitHub\Fleet_Management\merge\`. The merge branch never edits its own `merge/` folder.
- **Authority.** `spec/spec.tex` of the worktree; the model sections are lines 914-1542. Where the worktree's spec and
  the `lore` copy (`C:\Users\utente\Documents\GitHub\Fleet_Management\spec\spec.tex`) differ, the `lore` copy wins: it
  holds the user's answers of 29 Oct 2026 (null at optional keys, check tolerance, result files, `mip_gap`).
- **Contract with WP5.** The helper API of `base.py` is section 2 of `merge/wp5_plan.md`, as corrected by its review
  (section 7). A builder uses these helpers for every generic row. If a builder needs a helper that does not exist,
  the plan says so under "Questions"; it does not change `base.py` (except the Gamma plan, below).

## Workflow (user, 28 Oct 2026, confirmed for WP6 on 29 Oct 2026)

1. An Opus agent plans each part: `merge/wp6_plan_giw.md`, `merge/wp6_plan_gamma.md`, `merge/wp6_plan_rainflow.md`.
2. Sonnet agents carry out the plans, one per builder file, at most four at once.
3. The same Opus agent reviews the result against its plan and the spec, and corrects it.
4. Claude reviews the diff and commits on `merge/lore-into-main`.

## File ownership

| Plan | Builder files | Other files it may change | Tests (new files) |
|---|---|---|---|
| GIW | new `degradation_model/gaussian.py`, `inverse_gaussian.py`, `wiener.py` | -- | `tests/test_gaussian.py`, `tests/test_inverse_gaussian.py`, `tests/test_wiener.py` |
| Gamma | `degradation_model/gamma.py` (rewritten) | `degradation_model/base.py`: only the removal of the old Gamma block (`_add_binary_scaled_product`, `_gamma_reachable_upper_bounds`, `GammaCellBuilder`, its registration); `degradation_model/__init__.py`; the old tests `tests/test_gamma_reliability_bound.py` and `tests/test_gamma_transitions.py` (rewrite or replace) | `tests/test_gamma.py` |
| Rainflow | `degradation_model/rainflow.py` (rewritten) | -- | `tests/test_rainflow.py` |

- **`config.py` is shared.** WP4 deferred two checks to WP6: the replacement state of an inverse Gaussian cell must
  satisfy the mean cap, and that of a Gamma cell the Gamma cap (spec, "Consistency Checks", "Natural ranges and
  replacement states"). The GIW and Gamma plans each write a section "config.py change" with the exact code. One
  Sonnet executor applies both sections after the builders are done. No other executor edits `config.py`.
- **`config.py` imports no Gurobi.** It must stay importable without `gurobipy`, and so must every function it calls
  for these checks. Each builder module imports `base`, which imports `gurobipy` at module level. So a cap that
  `config.py` needs is computed by a function that imports without Gurobi: either in the builder module, imported by
  `config.py` inside the check function, or in a module under `gamma_utils/` or beside it. Choose a placement that
  fits the spec's "Package Layout" (spec line 169-203); if the layout must change, ask under "Questions".
- **Nobody edits** `solver.py`, `horizon_sweep.py`, `greedy_warm_start.py`, `utils/`, `validation/`,
  `validator_dashboard.py`, `fleet_management/__init__.py`, `gamma_utils/` validators (WP7), or the spec.
- **`lore`'s builders** (`src/fleet_management/models/`, `src/fleet_management/maintenance/`) are sources for the GIW
  and Rainflow plans. They are removed after WP6, by Claude.

## Rules for every builder

- A builder module defines its builder class with `name` and the hooks of the spec's "Cell-Builder Interface"
  (`prepare`, `add_cell`, optional `extract`) and registers it with `register_cell_builder`. `base._load_builders`
  imports `fleet_management.degradation_model.<model name>` lazily, so no registry edit is needed.
- `import fleet_management` must still work after WP6, also without Gurobi (WP4 and WP5 checked that it works today).
- Every variable and row name is unique (the relaxation start maps variables by name).
- Code style: as `base.py` after WP5 (section 0 of `merge/wp5_plan.md`).

## Local tools (this computer has numpy, scipy, pyyaml, h5py; no pytest, no gurobipy; nothing may be installed)

Scratchpad: `C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad\`
(copies in `merge/wp45_work/` of the `lore` checkout).
- `wp4\run_tests.py <test module> [name]`: runs pytest-style tests without pytest.
- `wp5\stubs\gurobipy\`: a stub that lets modules importing `gurobipy` load.
- `wp5\fakegp\`: a linear emulator of `gurobipy`; it solves the LP once the binaries are fixed or given by a complete
  MIP start. It cannot handle quadratic rows, indicator rows with free binaries, or a search over the binaries.
- `wp5\load_base.py`, `wp5\names_check.py`: module load and undefined-name checks.

## Claude's decisions on the plans' questions (29 Oct 2026)

These override the three plans where they differ. Executors and reviewers follow them.

1. **Placement of the caps: one rule for both.** Each cap lives in its builder module, which imports `base` under the
   guard of `wp6_plan_gamma.md`, section 2 (`try: from fleet_management.degradation_model import base` /
   `except ModuleNotFoundError as error: if error.name != "gurobipy": raise` / `base = None`), and registers its
   builder only if `base` loaded. The spec's package layout holds as written. For the inverse Gaussian this changes
   `wp6_plan_giw.md`:
   - Task IG: `inverse_gaussian.py` holds `_ig_unit_log_cdf`, `_ig_unit_log_sf`, `_ig_unit_quantile` and
     `ig_mean_cap`, with the code of section 4 (C2), below the guarded import. Its imports become `math`, `numpy as
     np`, `from scipy.optimize import brentq`, `from scipy.special import log_ndtr`, and the guarded `base`. The last
     line is `if base is not None:` followed by the registration. The docstring says that the cap lives here and that
     `config.py` imports it inside the check. No import from `config`.
   - Task IG: the test `test_inverse_gaussian_cap` (section 4, C4) goes into `tests/test_inverse_gaussian.py`, calling
     `inverse_gaussian.ig_mean_cap` and `inverse_gaussian._ig_unit_quantile`.
   - Task C: no C1 and no C2. In C3, the first line of the inserted block is
     `from fleet_management.degradation_model.inverse_gaussian import ig_mean_cap`. C4 adds only
     `test_inverse_gaussian_replacement_cap` to `tests/test_config.py`.
2. **Row names.** Every cap row is named `rel` (IG: `name="rel"`, not `"rel_ig"`). The optional variance-cap row is
   named `v_max_user` at every model.
3. **Test T1 of `tests/test_base.py`** (Gamma plan, step G2b): agreed. The Gamma executor adds the one line
   `base._load_builders(["gamma"])`; it is the only edit allowed in that file.
4. **The two old Gamma test files** are removed by Claude at the commit.
5. **Numerical choices of the caps** (brackets, tolerances, grid sizes, golden-section steps): accepted. In their
   reviews, the GIW and Gamma planners write the exact sentence that the spec needs for each cap; Claude adds it to the
   spec.
6. **Rainflow:** no tightening of variable bounds; the redundant rows are built, as the spec lists them.

## Executors (29 Oct 2026)

Four Sonnet agents at once: (1) Gamma, task G of `wp6_plan_gamma.md`; (2) rainflow, steps R1-R7 of
`wp6_plan_rainflow.md`; (3) Gaussian then Wiener, tasks G and W of `wp6_plan_giw.md`; (4) inverse Gaussian, task IG.
After all four: one Sonnet agent for Task C (`config.py`: GIW section 4 as changed above, Gamma section 5).
