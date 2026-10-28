# WP4 and WP5: interface and file ownership

Status (28 Oct 2026): fixed by Claude before the planners start; both plans follow it. A change needs Claude's
agreement, because the other package depends on it.

- **Purpose.** WP4 (input and output layer) and WP5 (shared model) run in parallel. This file fixes what one package
  gives the other, and which files each package may edit.
- **Where.** Code: the worktree `C:\Users\utente\Documents\GitHub\Fleet_Management-merge`, branch
  `merge/lore-into-main`. Records (plans, reviews): `C:\Users\utente\Documents\GitHub\Fleet_Management\merge\` (the
  `lore` checkout). The merge branch never edits its own `merge/` folder, so later merges of `lore` stay clean.
- **Authority.** `spec/spec.tex` of the worktree. Section names below are the spec's.

## Workflow (user, 28 Oct 2026)

1. An Opus agent plans each package: `merge/wp4_plan.md`, `merge/wp5_plan.md`.
2. Sonnet agents carry out the plan.
3. The same Opus agent reviews the result against the plan and the spec, and corrects it.
4. Claude reviews the diff and commits on `merge/lore-into-main`.

## File ownership

| Package | May edit | Must not edit |
|---|---|---|
| WP4 | `src/fleet_management/config.py`, `src/fleet_management/solver.py`, new `tests/test_config.py`, new `tests/test_io.py` | everything else |
| WP5 | `src/fleet_management/degradation_model/base.py`, new `tests/test_base.py` | everything else |

Inside `base.py`, WP5 leaves two parts alone:
- the Gamma block (`_gamma_reachable_upper_bounds`, `GammaCellBuilder`): WP6 replaces it with the sharp cap;
- the warm-start functions (`apply_binary_warm_start`, `complete_binary_warm_start`, `apply_relaxation_warm_start`):
  WP7 rewrites them. A mechanical rename that a changed shared variable forces is allowed.

No package edits `rainflow.py`, `gamma.py`, `gamma_utils/`, `models/`, `maintenance/` (`lore`'s builders, sources for
WP6), `horizon_sweep.py`, `greedy_warm_start.py`, `utils/`, `validation/`, `validator_dashboard.py`, `__init__.py`,
`spec/`, or the scoping document.

## `FleetConfig` (WP4 produces it, WP5 reads it)

`load_config(data) -> FleetConfig`. One attribute per input key of the spec's tables (sections "Sizes and Horizon" to
"Run Options"), with the key's name, in its canonical shape:

| Kind | Attributes | Type and shape | Where the key does not apply |
|---|---|---|---|
| Sizes | `F`, `M`, `L`, `H1`, `H2`, `T`; `H` as written | int | -- |
| Names | `component_names` | list of `L` str | -- |
| Selectors | `model`, `repair_model`, `bound_method`, `reliability_impl`, `encoding` | `(F, L)` array of str | `''`; `reliability_impl` holds the form that is built, so `'exact'` at a cell with a linear row (IG, Ga, Mk, Ch) |
| Cell values | every cell-value key of the tables, `C_R` and `C_rep` included | `(F, L)` float64, table defaults filled | `NaN`; also `NaN` for an optional key without default that is absent (`v_max_user`, `gamma_beta_0`, `gamma_beta_new`) |
| Other inputs | `pwl_points` int; `allow_replacement`, `replacement_as_new` bool; `C_M`, `C_D` float; `penalty_type` str | scalar | -- |
| Profiles | `mu`, `v`, `support`, `cgf`, `gamma_beta` | `(F, L, M, H2)` float64, after eager broadcasting | `NaN`; `None` if the key is absent |
| Transitory profiles | `mu_trans`, `v_trans`, `support_trans`, `cgf_trans`, `gamma_beta_trans` | `(F, L, M, H1)` float64 | `NaN`; `None` if absent |
| Run options | `options` | dict with every key of "Run Options", defaults filled (`None` where the default is "none") | -- |
| Derived | `support_const` | `(F, L)` float64: the Bernstein support constant `b` | `NaN` at cells that are not RF (Be) |
| Derived | `range_mu`, `range_v`, `range_R`, `range_K` | `(F, L)` float64: the natural ranges `M_s` of "Consistency Checks" and "Encoding of the Dynamics" | `NaN` where the state is not tracked |

- The property `models` (sorted names of the models present) stays.
- Fields that the spec drops leave `FleetConfig` (for example `gamma_beta_bound`, `gamma_calibration_method`, the
  `costs` dict).
- `config.py` imports only the standard library, numpy and scipy, so it runs without Gurobi.
- The phase lookup of "Increment Profiles and the Phase Lookup" (Eq. `c_phase`) is applied in `base.py` (WP5), by
  the increment accessor of the context. `config.py` checks the profiles entry by entry.
- A check that needs a constant computed by a builder module (the IG cap, the Gamma cap `A_1`) is added to `config.py`
  in WP6. The WP4 plan lists these checks.

## Solve path

- `solver.py` (WP4): `solve(input_path, results_path="output.yaml", *, warm_start=None, warm_start_mode="start")` and
  `solve_mapping(data, results_path=None, *, warm_start=None, warm_start_mode="start")`, as in "Data Flow of One
  Solve". `solve_mapping` calls `load_config(data)`, then
  `base.solve_mixed(cfg, warm_start=warm_start, warm_start_mode=warm_start_mode)`, then writes the result file.
  `solver.py` imports `base` inside `solve_mapping`, so the readers and writers import without Gurobi.
- `base.solve_mixed(cfg, *, warm_start=None, warm_start_mode="start") -> dict` (WP5): build, warm start, optimise,
  extract. It returns the whole in-memory result: every key of "Result Dictionary", with "Rules for the Values" and
  "Reports". Arrays are numpy arrays; an entry that does not apply is `NaN`; a key that the spec marks `null` is
  `None`.
- The writers (WP4) follow "Result Files": `NaN` becomes `null` in YAML and JSON, a key whose value is `None` is not
  written (except `objective`), the flat HDF5 layout. A file is written whenever `results_path` is not `None`, also
  for a solve that found no solution.
- The input reader lives in `config.py` (spec, "Package Layout"); it imports `yaml` and `h5py` inside the function.
  `solver._read_input` stays as an alias for the callers of WP6 and WP7.
- Eq. `c_phase` exists twice: `config._lookup` (vectorised, for the checks) and `base.phase_day` (one day, for the
  rows). The WP5 review adds a local check that both give the same day map.

## State between the packages

- After WP4 and WP5 the package solves nothing end to end: the G, IG and W builders arrive in WP6, and WP6 rewrites the
  Gamma and rainflow builders.
- Tests that need Gurobi are written for the licensed machine. On this computer: numpy, scipy, pyyaml, h5py, pandas
  and matplotlib exist; pytest and gurobipy do not, and nothing may be installed.

## User's answers to the planners' questions (29 Oct 2026)

Recorded in `decisions.md` (item "Follow-up decisions from writing the spec") and written into the spec on `lore`; the
worktree's `spec/spec.tex` gets them at the next merge of `lore`. They match the WP4 plan's recommendations, except
the tolerance: relative, `1e-12·max(1, |a|, |b|)`, not absolute `1e-12`. The WP5 plan's three questions follow its
recommendations.
