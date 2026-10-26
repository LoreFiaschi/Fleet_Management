Status: done (26 Oct 2026). Phase 1 (the list) and phase 2 (the removals and the hygiene files) are finished on the
branch `merge/lore-into-main`. Nothing is committed: Claude reviews the diff and commits.
Next: none in this work package. WP3 (merge of `lore`) and the later packages follow `code_plan.md`.

# WP1 and WP2: removals and repository hygiene

## Phase 2: what was changed (26 Oct 2026)

Claude approved the list with these answers: all 54 files of section 1; U3, U5 and U9 as removals; U1, U2, U4, U6, U7
and U8 kept; `plot_management` kept as an alias; the four function-local imports left in place; section 5 as proposed.

- **Removed with `git rm` (65 files)**: the 54 files of section 1, `euler/zurich_mixed_formulation_counts.sbatch` and
  `examples/regression/report_mixed_formulation_size.py` (U3), and the 9 files `results/validator_tests/validation_bad_*.yaml`
  (U5). The tracked files went from 219 to 149.
- **Untracked with `git rm -r --cached` (5 files)**: `src/fleet_management.egg-info/`. The files stay on disk and are
  ignored.
- **Edited (5 files)**:
  - `.gitignore`: replaced by the text of section 5.1.
  - `pyproject.toml`: the description of section 5.2.
  - `src/fleet_management/__init__.py`: removed the imports and exports of `sweep_formulation_dimensions`,
    `validate_gamma_result` and `validate_baseline_assignment_feasibility`; the import of `plot_management` now reads
    `from fleet_management.utils.mixed_plotter import plot_mixed_management as plot_management`; `plot_mixed_management`
    stays exported.
  - `src/fleet_management/validation/validator.py`: removed line 14 (import of `model_registry`). The legacy `validate()`
    fails when called, until WP7 rewrites it.
  - `tests/test_gamma_transitions.py`: removed the import of the three helpers of `gamma_validator` and the three tests that
    used them. 13 tests remain.
- **Checks**:
  - `py_compile` of the three changed `.py` files: all compile. (`__init__.py`, `validator.py`, `test_gamma_transitions.py`;
    the output went to the scratchpad.)
  - `git ls-files -ci --exclude-standard`: empty (0 tracked files match a rule of the new `.gitignore`).
  - Module-level imports of a removed module (AST scan of every `.py` file of the tree, `scratchpad\wp1\check_imports.py`):
    0. Every name that a file imports from `fleet_management` is still exported.
  - Function-local imports of a removed module: exactly the four of section 4.2 (`base.py:358`, `greedy_warm_start.py:76`,
    `solver.py:119`, `validation/dashboard_rainflow.py:469`), left in place as decided.
  - Other text that still names a removed file: `gamma_tail_bound.py:4` (docstring), the documents of section 4.3, and
    `examples/regression/RegressionREADME.md`. They are edited with the documents.
  - `git status --short`: 75 lines (70 deletions, 5 modifications). No untracked file.
- **Left for later**: sections 3 and 4.2; the unsure items U1, U2, U4, U6, U7, U8; the lines of the documents.

- **Worktree**: `C:\Users\utente\Documents\GitHub\Fleet_Management-merge`, branch `merge/lore-into-main`, created from `main`
  at 978ea9a. 219 tracked files. `git status` was clean.
- **Evidence basis.**
  - Read in full: the items of `decisions.md` named in the brief; `code_plan.md`; `reports/C_tooling.md`;
    `spec_drafts/fragments_D.tex` ("File & Folder Structure", "Validation Tools").
  - Read in part: each candidate file (its first 25 to 40 lines, its imports, its function list). The large files
    (`test_rainflow.py`, `run_studies.py`, `run_year.py`, `rainflow_v2.py`, `rainflow_sparse.py`, the legacy files) were
    not read in full. What each one is comes from its header and from `C_tooling.md`.
  - The references of section 4 come from a scan of every tracked file with the names of the candidates, then a check of
    each hit. The scan script is `scratchpad\wp1\scan_refs.py`.
- **Not on this branch** (checked): `ASSESSMENT.md`, `Things to change.txt`, `lore-vs-main.md`, `merge/`,
  `remote_smoke_test.py`, `lore`'s plotter (`src/fleet_management/plotter.py`). `ASSESSMENT.md` exists only on `lore`
  (`git cat-file -e 978ea9a:ASSESSMENT.md` fails). It arrives with the merge of `lore` (WP3) and must be removed after it.
- **Counts**: section 1, 54 files; section 2, 9 unsure items; section 3, 20 partial removals; section 4,
  6 import sites to edit in phase 2, 4 function-local sites left in place, and the text mentions; section 5, one
  `.gitignore`, one `pyproject.toml` change, 5 egg-info files to untrack.

## 1. Files to remove (54)

A decision names a file, or the file only works with a named file (marked "dependent"). Dependent files are listed so
that Claude can veto them one by one.

### 1.1 Package `src/fleet_management/` (16)

| Path | Decision item | What it is |
|---|---|---|
| `degradation_model/gaussian.py` | "Set of degradation models" (`main`'s old `gaussian.py` leaves the package) | Stand-alone Gaussian solver of the common ancestor; old keys `alpha`, `xi`, `C_S`, `C_P`; not a cell builder (417 lines). |
| `degradation_model/inverse_gaussian.py` | "Set of degradation models" | Same, for the inverse Gaussian model (320 lines). |
| `degradation_model/rainflow_v2.py` | "Follow-up decisions from writing the spec" ("the archived rainflow variants") | Historical study of the indicator and big-M encodings; uses the archived context; `solve()` never calls it (1344 lines). |
| `degradation_model/rainflow_sparse.py` | "Follow-up decisions from writing the spec" | Historical sparse-assembly study of the rainflow model (1405 lines). |
| `degradation_model/legacy/__init__.py` | "Follow-up decisions from writing the spec" (`legacy/`) | One-line docstring of the archive folder. |
| `degradation_model/legacy/gamma_gurobi.py` | "Gamma formulation" (the legacy one-rate route leaves the solve path); "Follow-up decisions" (`legacy/`) | Stand-alone constant-rate Gamma backend; importers: `solver.py:119`, the shim `gamma_utils/gamma_gurobi.py`, one check (510 lines). |
| `degradation_model/legacy/rainflow_formulation_base.py` | "Follow-up decisions from writing the spec" (`legacy/`) | Archived shared model layer used by `rainflow_v2.py`, `rainflow_sparse.py`, `test_rainflow.py`, `test_sparse_version.py`, `run_studies.py` (849 lines). |
| `degradation_model/legacy/rainflow_formulation_config.py` | "Follow-up decisions from writing the spec" (`legacy/`) | Archived input loader of the same study code (380 lines). |
| `degradation_model/gamma_utils/gamma_gurobi.py` | "Gamma formulation" (dependent) | 12-line shim that re-exports `legacy/gamma_gurobi.py`. |
| `degradation_model/gamma_utils/gamma_repeated_calibration.py` | "Gamma formulation" (the `repeated_increment` calibration leaves the solve path) | Offline common-rate calibration of the Gamma shapes (528 lines). |
| `degradation_model/gamma_utils/gamma_validator.py` | "Follow-up decisions from writing the spec" (`validate_gamma_result`) | Validator of results of the old common-beta Gamma solver; the whole file is that validator and its helpers (675 lines). |
| `formulation_size_sweep.py` | "Follow-up decisions from writing the spec" (formulation-count tools) | Analytical counts of variables and rows, varying `F`, `M`, `L`, `T` (286 lines). |
| `report_gamma_formulation_counts.py` | "Follow-up decisions from writing the spec" (formulation-count tools) | Script that solves four small Gamma models and writes Gurobi counts (170 lines). |
| `utils/benchmark.py` | "Follow-up decisions from writing the spec" (formulation-count tools) | Benchmarks of the common-beta Gamma backend; calls `solve` and `gamma_validator` (396 lines). |
| `utils/model_registry.py` | "Follow-up decisions from writing the spec" (`utils/model_registry.py`) | Parser registry for Gaussian, inverse Gaussian and Gamma inputs; marked `# DEPRECATED` (346 lines). |
| `utils/plotter.py` | "Plotting" (`main`'s older plotter is removed) | One row per train, label "Flight i", no replacement mark (290 lines). |

### 1.2 Top-level scripts (6)

| Path | Decision item | What it is |
|---|---|---|
| `csv_to_plotter.py` | "Follow-up decisions from writing the spec" | Converts a CSV schedule into the input of the older plotter (225 lines). |
| `fix_root_gap.py` | "Follow-up decisions from writing the spec" | Repairs `root_gap` in results of the archived studies of `run_studies.py` (138 lines). |
| `test_rainflow.py` | "Follow-up decisions from writing the spec" | Bound-comparison experiment: one case solved under the five bounds (4005 lines). |
| `test_sparse_version.py` | Dependent. "Cluster jobs and experiments", family A | Sparse-versus-loop assembly harness; imports `legacy/`; run by `euler/submit_sparse.sh` (1123 lines). |
| `run_studies.py` | Dependent. "Cluster jobs and experiments", family A | Scaling studies; loads the harness by path; imports `legacy/`; run by `euler/submit_studies.sh` (2417 lines). |
| `run_year.py` | Dependent. "Cluster jobs and experiments", family A | One VBZ year solved with warm starts; loads the harness by path; run by `euler/submit_year.sh` (527 lines). |

The three dependent scripts are named in the family-A row of `C_tooling.md` section 6, not in the decisions themselves.

### 1.3 `euler/`, family A (12)

All 12 run the harness `test.py` (now `test_rainflow.py`) or the three dependent scripts above. Decision item: "Cluster jobs
and experiments" ("Family A: dropped, with `test_rainflow.py` and the archived rainflow variants") and "Follow-up
decisions from writing the spec" (the family-A cluster jobs).

| Path | What it is |
|---|---|
| `euler/setup_euler.sh` | Builds `.venv-euler`; stops if `test.py` is missing. |
| `euler/submit.sh`, `euler/run_array.sbatch`, `euler/merge.sbatch` | Submit, array job and merge job of the bound-test harness. |
| `euler/submit_studies.sh`, `euler/run_studies_array.sbatch`, `euler/merge_studies.sbatch` | The same for `run_studies.py`. |
| `euler/submit_sparse.sh`, `euler/run_sparse.sbatch` | The same for `test_sparse_version.py`. |
| `euler/submit_year.sh`, `euler/run_year_array.sbatch`, `euler/merge_year.sbatch` | The same for `run_year.py`. |

### 1.4 `examples/regression/` (6)

| Path | Decision item | What it is |
|---|---|---|
| `check_formulation_sweep.py` | "Follow-up decisions from writing the spec" (the regression scripts that call the formulation-count tools) | Imports only `formulation_size_sweep`. |
| `run_formulation_size_sweep.py` | Same | Command-line runner; imports only `formulation_size_sweep`. |
| `check_gamma_repeated_calibration.py` | Dependent. "Gamma formulation" | Imports only `gamma_repeated_calibration`. |
| `check_gamma_incumbent_extraction.py` | Dependent. "Gamma formulation" (legacy route) | Imports only `legacy/gamma_gurobi`. |
| `check_gamma_repair_legacy.py` | Dependent. "Gamma formulation" (the `finite_count` calibration) | Calls `calculate_seeded_profile_tail_bound_parameters`; marked `# DEPRECATED`. |
| `check_gamma_randomized_properties.py` | Dependent. "Gamma formulation" (the `finite_count` calibration) | Calls only `calculate_tail_bound_parameters`, the `finite_count` calibrator. |

The maintenance rule of `RegressionREADME.md` says a check goes together with the functionality it certifies.

### 1.5 Documents (3)

| Path | Decision item | What it is |
|---|---|---|
| `lore_check.txt` | "Documents" ("stays deleted, and its notes are dropped") | The user's five research notes. `lore` deleted it too. |
| `spec/main.tex` | "Documents" ("stay deleted") | Literature review of May 2026. `lore` deleted it too. |
| `spec/review_report.tex` | "Documents" ("stay deleted") | Automated review of that literature review. `lore` deleted it too. |

`ASSESSMENT.md`: not on this branch (see above). The merge of `lore` will bring it.

### 1.6 `results/` (11)

Decision item: "Cluster jobs and experiments" (results computed before the merge are removed; scope to confirm file by
file).

| Path | What it is | Confirmed |
|---|---|---|
| `results/gamma_horizon_convergence_12483826.yaml` | Report of the Gamma sweep, `H2 = 4..40`; input `gamma_horizon_euler_convergence.yaml`. | Yes: computed with `repeated_increment` and `gamma_beta_bound: 10`. |
| `results/gamma_horizon_convergence_12483826_input.yaml` | Input snapshot of that sweep (`gamma_calibration_method: repeated_increment`, `depot_capacity: 1`). | Yes. |
| `results/gamma_horizon_convergence_12533716.yaml` | Report of the Gamma sweep, `H2 = 44..64`. | Yes. |
| `results/gamma_horizon_convergence_12533716_input.yaml` | Input snapshot of that sweep. | Yes. |
| `results/gamma_horizon_convergence_combined_corrected.png` | Plot of the two sweeps. | Yes. |
| `results/output.yaml` | Old Gaussian solve result (`degradation: gaussian`, `alpha`, no `L`). | Yes. |
| `results/schedule.png` | Plot of `output.yaml`. | Yes. |
| `results/output_baseline.yaml` | Old Gaussian solve result, `F=6, M=3, H=10, L=2`. | Yes, but see U5: a test reads it. |
| `results/schedule_baseline.png` | Plot of `output_baseline.yaml`. | Yes. |
| `results/validation.yaml` | Report of the legacy validator on `output.yaml`. | Yes. |
| `results/validation_baseline.yaml` | Report of the legacy validator on `output_baseline.yaml`. | Yes. |

Not in scope, as the decision says, and kept: `results/validator_tests/bad_*.yaml` (10 fixtures) and
`experiments/vbz_F8_feasibility_transfer/starts/vbz_F6_M1_L4_H2_48_result.yaml`. Every experiment folder holds
definitions only; none holds a solve result.

## 2. Unsure (9 items, 49 files)

Each item gives the facts and a recommendation. No decision names these files.

| Item | Files | Reason | Recommendation |
|---|---|---|---|
| **U1 `gamma_diagnostics.py`** | `degradation_model/gamma_utils/gamma_diagnostics.py` (292 lines) | One function stays: `collect_gurobi_model_statistics` (line 232), called by `base.py:2058` (performance counters). The other two (`estimate_gamma_formulation`, line 18; `compare_estimate_with_actual`, line 268) count rows of the exact-hull Gamma formulation, which the merge replaces. The file is not in the package layout of the spec. Six scripts of `examples/regression/` and `experiments/vbz_F8_feasibility_transfer/run_f8_feasibility_transfer.py:44` import it. | Keep the file now. WP5 moves `collect_gurobi_model_statistics` into `base.py`; WP6 or WP8 drops the two estimators. |
| **U2 `gamma_process.py`** | `degradation_model/gamma_utils/gamma_process.py` (480 lines) | Helper functions for the shape-rate Gamma state (`mean_to_shape`, `reliability_passed`, ...). Its only importer is `validation/validator.py:15`, the legacy Gamma diagnostic. Not in the package layout of the spec. | Remove it together with the legacy diagnostic in WP7. Keep it in WP1. |
| **U3 formulation-count job and report** | `euler/zurich_mixed_formulation_counts.sbatch`, `examples/regression/report_mixed_formulation_size.py` | The job runs the report; the report calls `estimate_gamma_formulation`. The spec layout lists only two Slurm scripts in `euler/`, and this is not one of them. Its two inputs (`input/zurich_*_mixed.yaml`) were never committed. The decision lists the formulation-count tools by name, and this report is not on the list. | Remove both now. |
| **U4 `check_gamma_tail_bound.py`** | `examples/regression/check_gamma_tail_bound.py` | Mixed: it checks the `finite_count` calibrators (removed in WP6) and `moschopoulos_tail_probability` (stays). | Keep; WP8 cuts it down to the Moschopoulos part. |
| **U5 test fixtures and reports** | `results/validator_tests/validation_bad_*.yaml` (9 files); `results/output_baseline.yaml` (listed in 1.6) | The 9 files are reports that `tests/test_validator_negative_cases.py:31` writes on each run. They are outputs, not fixtures. `output_baseline.yaml` is read by `tests/test_validator_positive_case.py`; the `bad_*.yaml` fixtures are copies of it in the old Gaussian format, so the rewritten validator cannot read them either. | Remove the 9 reports and ignore `results/validator_tests/validation_*` (the proposed `.gitignore` does). Remove `output_baseline.yaml` as the decision says. WP8 makes new fixtures. |
| **U6 old-format inputs** | `input/data.yaml`, `input/data_test_baseline.yaml`, `input/tiny.yaml`, `input/vbz_man12e_year.yaml` | The first three are in the old Gaussian format without `model:`; `solve()` refuses them. `data_test_baseline.yaml` is the input of the two validator tests; `data.yaml` is the README quick start; `tiny.yaml` has no reference. `vbz_man12e_year.yaml` belongs to the family-A year jobs and has `depot_capacity: 6`, which `config.py` rejects. | Keep all four in WP1. WP8 and the README rewrite replace them. |
| **U7 Gamma shape model and its tests** | `degradation_model/gamma.py` (`GammaModel`, lines 61-381; `maximum_reliable_shape`, 392-490; `maximum_reliable_expected_damage`, 491-504); `tests/test_gamma_reliability_bound.py`; `tests/test_gamma_transitions.py` (13 of 16 tests) | The sharp cap tracks the mean, not the shape, so `GammaModel` has no user after WP6. `maximum_reliable_shape` is the exact one-rate cap, which the sharp cap `A_1` equals above the threshold `t1(eps)` (decision "Gamma formulation"). | Keep in WP1; WP6 decides. |
| **U8 check scripts and inputs with removed Gamma keys** | 23 scripts of `examples/regression/` (list below); 25 YAML inputs outside `results/` use `gamma_beta_bound` or `gamma_calibration_method` | The scripts name those keys, read the YAML inputs that carry them, or import the estimators of U1. They test the Gamma solve or the validators, not a removed file. The keys become errors in the merged code, and the hull rows become one-sided. | Keep; WP8 rewrites or drops each one. |
| **U9 the three dependent scripts of 1.2** | `test_sparse_version.py`, `run_studies.py`, `run_year.py` | Listed in section 1 by inference (family-A table of `C_tooling.md`; they import archived code). No decision item names them. | Confirm or veto. |

U8 scripts (23): `check_gamma_ard1_integration.py`, `check_gamma_ard1_truth_table.py`, `check_gamma_ard1_replacement_truth_table.py`,
`check_gamma_ardinf_replacement_truth_table.py`, `check_gamma_big_m_formulation.py`, `check_gamma_cell_integration.py`,
`check_gamma_complexity_diagnostics.py`, `check_gamma_exact_validator.py`, `check_gamma_public_interface.py`,
`check_gamma_repair_integration.py`, `check_gamma_replay_validator.py`, `check_gamma_stochastic_validator.py`,
`check_gamma_validator_edge_cases.py`, `check_greedy_warm_start.py`, `check_greedy_warm_start_acceptance.py`,
`check_horizon_sweep.py`, `check_matched_ard_inputs.py`, `check_mixed_gamma_ard1_public.py`,
`check_mixed_public_interface.py`, `check_operating_average_objective.py`, `check_unrestricted_maintenance.py`,
`gamma_replacement_truth_table_common.py`, `report_gamma_complexity.py`.

## 3. Partial removals (code inside files that stay; later work packages)

Line numbers are those of the branch at 978ea9a.

| File | Lines | Content | Work package |
|---|---|---|---|
| `solver.py` | 70-91 | `_solve_mixed`: Gamma-only branch (lines 82-83) and its docstring (72-75). | WP4 |
| `solver.py` | 101-128 | `_solve_gamma_fleet`: the old constant-rate route (line 125 sets `backend = "legacy_gamma"`). | WP4 |
| `solver.py` | 131-213 | Helpers of that route: `_require_uniform`, `_uniform_over_vehicles`, `_legacy_gamma_beta`, `_cfg_to_gamma_kwargs`. | WP4 |
| `solver.py` | 380, 391, 513, 538 | Result keys `gamma_calibration_method` and `gamma_beta_bound` in the YAML, JSON and HDF5 writers. | WP4 |
| `config.py` | 37, 73-79, 116-119, 333-345, 369-375, 410-414, 548-565 | `GAMMA_CALIBRATION_METHODS`; the fields `gamma_calibration_method` and `gamma_beta_bound`; routing by `gamma_beta_bound`; the check `gamma_beta_bound <= smallest rate`. | WP4 |
| `degradation_model/base.py` | 187-343 | Gamma hull helpers `_add_binary_scaled_product`, `_gamma_reachable_upper_bounds` (exact rows; the merge uses one-sided rows). | WP5 and WP6 |
| `degradation_model/base.py` | 344-1185 | `GammaCellBuilder`: calibration at 358-361, 515-518, 568-607; calibration entries of `extract` at 1097-1121 and 1174-1176; replaced as a whole by the sharp-cap builder. | WP6 |
| `degradation_model/gamma_utils/gamma_tail_bound.py` | 76-329 | Result classes of the calibrators (`TailConstraint`, `GammaTailBoundResult`, `GammaProfileTailBoundResult`, `GammaFleetTailBoundResult`, `GammaSeededProfileTailBoundResult`). | WP6 |
| `degradation_model/gamma_utils/gamma_tail_bound.py` | 330-927 | The four `finite_count` calibrators: `calculate_profile_tail_bound_parameters`, `calculate_seeded_profile_tail_bound_parameters`, `calculate_fleet_tail_bound_parameters`, `calculate_tail_bound_parameters`. | WP6 |
| `degradation_model/gamma_utils/gamma_tail_bound.py` | 1010-1150, 1171-1224 | `required_shape_for_tail`, `enumerate_count_vectors`, `shapes_from_expected_damage`, `_resolve_count_vectors`, `_integer_vector`, `_broadcast_fleet_cell`, `_broadcast_fleet_profile`. `required_shape_for_tail` may serve the `A_1` computation: WP6 decides. | WP6 |
| `degradation_model/gamma_utils/gamma_tail_bound.py` | 1-4 | Docstring that refers to `gamma_repeated_calibration`. | WP6 |
| Stays in that file | 66-74, 928-1009, 1151-1169 | `ConvolutionTail`, `moschopoulos_tail_probability`, `_validate_parameters`. The exact validator imports them. | -- |
| `degradation_model/gamma_utils/gamma_diagnostics.py` | 18-231, 268-292 | `estimate_gamma_formulation`, `compare_estimate_with_actual` (see U1). | WP6 or WP8 |
| `greedy_warm_start.py` | 66-170 | `_gamma_shape_data`: reproduces the calibration; its call sites in `_simulate_cell` and later. | WP7 |
| `validation/validator.py` | 28-248 | Legacy `validate(input, degradation, results)`. | WP7 (rewritten) |
| `validation/validator.py` | 249-426 | `validate_baseline_assignment_feasibility` (dropped by decision). | WP7 |
| `validation/validator.py` | 427-583, 584-874 | `build_assignment_feasibility_dataframe`, `build_gamma_diagnostic_dataframe`: they feed the two dropped dashboards. | WP7 |
| `validation/validator.py` | 875-1083 | `validate_gamma_synthetic_diagnostic` (dropped by decision). | WP7 |
| `validator_dashboard.py` | 1-1795 | Mode "Baseline Gaussian validator" is the whole body; the mode selector is at 40-56. Rewritten on the new keys. | WP7 |
| `validation/dashboard_gamma.py` | 1-704 | Dashboard of the "Gamma synthetic diagnostic"; rewritten as mode "Gamma diagnostic". | WP7 |
| `degradation_model/__init__.py` | 1-5 | Exports `GammaAction`, `GammaModel` (see U7). | WP6 |

## 4. References to the files of section 1

### 4.1 Lines edited in phase 2 (minimal edits)

These are module-level imports. After the removals, `import fleet_management` or the test collection would fail on them.

| File and line | Line | Proposed edit |
|---|---|---|
| `src/fleet_management/__init__.py:1` and `:32` | `from ...formulation_size_sweep import sweep_formulation_dimensions`; `"sweep_formulation_dimensions"` in `__all__` | Delete both lines. |
| `src/fleet_management/__init__.py:8` and `:29` | `from ...utils.plotter import plot_management`; `"plot_management"` in `__all__` | Decided: import `plot_mixed_management as plot_management` from `utils/mixed_plotter.py`; keep both names exported. WP7 renames the plotter. |
| `src/fleet_management/__init__.py:22-24` and `:42` | Import of `validate_gamma_result` from `gamma_validator`; `"validate_gamma_result"` in `__all__` | Delete both. |
| `src/fleet_management/__init__.py:9-12` (name on line 11) and `:36` | Import and export of `validate_baseline_assignment_feasibility` (dropped by decision; its code goes in WP7) | Delete the name from the import and from `__all__`. |
| `src/fleet_management/validation/validator.py:14` | `from ...utils.model_registry import extract_degradation_parameters` | Delete the line. The legacy `validate()` calls the function at 110, 451, 639, 907 and fails when called, until WP7 rewrites it. |
| `tests/test_gamma_transitions.py:5-9` | Imports `_exact_repair_diagnostic`, `_reconstruct_shapes`, `_repair_cost_violation` from `gamma_validator` | Delete the import and the three tests that use the helpers (lines 194-255). The 13 other tests stay (U7). |

### 4.2 Function-local references, left in place

Each sits in code that a later work package removes (section 3). Until then, the code path raises `ImportError` when run.

| File and line | Line | Removed by |
|---|---|---|
| `degradation_model/base.py:358-361` (calls at 581 and 606) | Local import of `gamma_repeated_calibration` in `GammaCellBuilder.prepare`. Every Gamma solve breaks until WP6. | WP6 |
| `greedy_warm_start.py:76-79` (calls at 142 and 164) | Local import of `gamma_repeated_calibration` in `_gamma_shape_data`. | WP7 |
| `solver.py:119-121` (call at 123) | Local import of `legacy.gamma_gurobi` in `_solve_gamma_fleet`. | WP4 |
| `validation/dashboard_rainflow.py:469` (call at 471) | Local import of `model_registry` inside `try ... except Exception`; the code falls back to a raw read. Harmless. | WP7 |

Decided (26 Oct 2026): all four stay in place. The first three are Gamma-builder code that WP4, WP6 and WP7 replace.

### 4.3 Text mentions (documents, comments; edited with the documents)

- `README.md` lines 16, 44, 50, 64 (models, `plot_management`), 297-325 (file tree and notes on the archived study code).
- `examples/regression/RegressionREADME.md` lines 58, 61, 74, 115, 116, 195, 209, 212.
- `src/fleet_management/degradation_model/gamma_utils/GAMMA_DIAGNOSTICS.md:204` (command of `run_formulation_size_sweep.py`).
- Comments: `config.py:335` (`gamma_gurobi.py`), `gamma_tail_bound.py:4` (`gamma_repeated_calibration`).

Nothing in `experiments/` imports or calls a file of section 1. `experiments/vbz_F8_feasibility_transfer/` imports
`gamma_diagnostics` (U1) and the private `solver._save_results`.

### 4.4 Notes for the merge of `lore` (WP3)

- Of the 54 files, `lore` changed only `lore_check.txt`, `spec/main.tex` and `spec/review_report.tex` since the common
  ancestor, and it deleted them. So the removals merge without conflict.
- `lore` did not change `.gitignore`, `pyproject.toml` or `src/fleet_management.egg-info/` since the ancestor, so the
  files of section 5 win in the merge.
- `ASSESSMENT.md` and `Things to change.txt` arrive from `lore`. `ASSESSMENT.md` is removed after the merge.

## 5. WP2 proposal

### 5.1 New `.gitignore`

The same text is in `scratchpad\wp1\gitignore_proposed.txt`.

```
# Build artefacts
*.egg-info/
__pycache__/
*.py[cod]
build/
dist/
.pytest_cache/

# Virtual environments
.venv/
.venv-*/
venv/

# Generated outputs
# results/ holds solver output. The corrupted results that tests read stay tracked.
results/*
!results/validator_tests/
results/validator_tests/validation_*
logs/
results-euler/
test_results/
smoke_test_output/
# Run folders of the experiments; only the placeholder files stay tracked.
experiments/**/runs/*
!experiments/**/runs/.gitignore
!experiments/**/runs/.gitkeep
# Archives and PDF builds one folder below the root (for example spec/spec.pdf)
*.zip
*.tar.gz
*/*.pdf

# Local folders
literature/
agentic/
.claude/
bkp/
```

- **Dry run** (a scratch repository with only this file and the nested `runs/.gitignore`; `git check-ignore --no-index`):
  - Of the 160 tracked files that stay after phase 2, none matches a rule, except the 9 reports of U5. They go with U5.
  - The 11 files of 1.6 and the 5 egg-info files match, as intended.
  - 22 sample untracked paths (`__pycache__`, `build/`, `dist/`, `.venv-euler/`, `results/new.yaml`, an experiment run
    folder, `.zip`, `.tar.gz`, `spec/spec.pdf`, `literature/`, `agentic/`, `.claude/`, `bkp/`, `smoke_test_output/`)
    are ignored.
  - 11 sample paths that must stay visible are not ignored: `CLAUDE.md`, `input/new.yaml`, `tests/test_new.py`,
    `test/a.py`, `src/fleet_management/utils/plotter.py`, `results/validator_tests/bad_new.yaml`, `scoping.pdf`, and others.
  - Phase 2 repeated the check with `git ls-files -ci --exclude-standard` on the real tree: empty.
- **Rules of `main`'s file that are dropped** (list for Claude's veto):
  - `CLAUDE*` (the decision: `CLAUDE.md` is tracked). I did not add a rule for `CLAUDE.local.md`; the decision does not ask for it.
  - `input/*` with its exception, `tests/*` and `test/*` (tracked folders; the decision "Tests and test inputs").
  - `mixed_plotter.py`, `csv_to_plotter.py`, `fix_root_gap.py`, `run_year.py` (they match tracked files).
  - Personal scratch names: `ds.py`, `run_audit.py`, `fleet_matrix_audit.py`, `src/fleet_management/test.py*`,
    `helper_inspector.py*`, `tempCodeRunnerFile.py*`, `opt_analysis.py*`, `analysis.py*`, `PlottingScripts/*`, `docs/*`.
  - The four named archives (`vbz_*_14*.tar.gz`) and `results.tar.gz`; `*.tar.gz` covers them.
- **Added**: `dist/`, `.pytest_cache/`, `venv/`, `smoke_test_output/` (the output folder of the ported smoke test).
- **Kept as on both branches**: `*.zip`, `*/*.pdf`.
- The new file has LF line endings. The old file had CRLF in the working tree; the index holds LF.

### 5.2 `pyproject.toml`

`main`'s file already has Python >= 3.10, `pandas` and the extra `dashboard` (`streamlit`). Only the description changes.

```
-description = "Fleet management optimization with Gaussian degradation via MILP"
+description = "Maintenance scheduling of a train fleet over a time horizon under five stochastic degradation models (Gaussian, inverse Gaussian, Wiener, Gamma and rainflow), solved as a mixed-integer program with Gurobi"
```

Not in this change: `testpaths = ["tests"]` (item "Tests and test inputs", WP8) and the version number.

### 5.3 Untracking `src/fleet_management.egg-info/` (5 files)

`PKG-INFO`, `SOURCES.txt`, `dependency_links.txt`, `requires.txt`, `top_level.txt`. Phase 2 runs
`git rm -r --cached src/fleet_management.egg-info`. The files stay on disk, untracked and ignored.

### 5.4 Not part of this work package

`CLAUDE.md` is added to git from the `lore` checkout by Claude (it is not on this branch).
