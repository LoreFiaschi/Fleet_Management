# Code of the merged package: plan

Status: proposal (26 Oct 2026), waiting for the user's agreement. No code is changed yet.

- **Purpose.** How the code of the merged package is built from `main`: the work packages, their order, what each
  needs before it starts, and who does it.
- **Rule.** `spec/spec.tex` is authoritative (item "Source of truth"). Each work package implements named sections of
  the spec. A package that depends on spec text starts only when those sections have passed review (the must-fix
  items of the two reviews applied).
- **Out of scope here**: the documents (`documents_plan.md`) and the to-do items outside the merge.

## Where the work happens

- **A separate working copy.** A git worktree (a second checkout of the same repository in its own folder) at
  `C:\Users\utente\Documents\GitHub\Fleet_Management-merge`, on a new branch `merge/lore-into-main` created from
  `main`. The current checkout stays on `lore`, where the spec and the scoping document are being finished. Nothing
  touches `main` itself.
- **Commits.** Claude commits only when the user asks. Each work package ends at a commit point; the user commits or
  tells Claude to commit on the branch `merge/lore-into-main`.
- **Running the code on this computer.** Today there is no numpy, no pytest and no Gurobi here, so no part of the code
  can run locally. Installing the package's dependencies from PyPI (Python Package Index) into a virtual environment
  inside the worktree would allow it: numpy, scipy, pyyaml, h5py, matplotlib, pandas, pytest, and `gurobipy`, whose
  pip package carries a free licence limited to small models (2000 variables and 2000 constraints). That is enough for
  unit tests on fleets of a few trains and days. Large solves stay on the licensed machine, through the smoke test.
  This needs the user's permission (downloads).

## Work packages

| Package | Content | Spec sections | Can start |
|---|---|---|---|
| WP0 Branch | the worktree and the branch from `main` | -- | now |
| WP1 Removals | delete from the branch every file the decisions drop (list below) | -- | now |
| WP2 Repository hygiene | one coherent `.gitignore`; `src/fleet_management.egg-info/` untracked; `pyproject.toml` with `main`'s requirements and a description of the merged package | -- | now |
| WP3 Merge of `lore` | `git merge lore` into the branch, after the user commits `lore`'s finished spec, records and scoping document. The source conflicts are resolved toward `main`'s code; `lore`'s builders arrive as sources for WP6 and leave after the port | -- | after step 1 |
| WP4 Input and output layer | `config.py` (`load_config` with the key table, the errors for unknown and old keys, `null` cells, broadcasting, every consistency check); the input reader for YAML, JSON and HDF5 (every key); the result writers (YAML, JSON, flat HDF5) and the result dictionary | Input Specification, Output Specification, Input File Schema | after the review of the inputs and outputs |
| WP5 Shared model | `degradation_model/base.py`: fleet rows, the depot row, the generic one-sided rows in both encodings, anchors, loop rows, repair cost `z`, damage penalty `u` (none where the coefficient is 0), the three objective modes, the `tangent` and `pwl` forms, result extraction | Decision Variables, Objective Function, Constraints (common subsections), Full Optimization Problem | after the review of the formulation |
| WP6 Cell builders | `gaussian.py`, `inverse_gaussian.py`, `wiener.py` (ported from `lore`), `gamma.py` (the sharp cap and the offline computation of `A_1`), `rainflow.py` (five bounds, Bernstein and Chernoff fixes, `R` and `K` start and reset values, both laws) | the five model sections | after WP5 |
| WP7 Tools | horizon sweeps; warm starts and the greedy start; `plot_management` and `plot_horizon_sweep`; `validate`, the dashboards, the Gamma validators | Horizon Sweeps, Warm Starts, File & Folder Structure, Validation Tools | after WP4 to WP6 |
| WP8 Tests | one pytest suite; the smoke test extended to the merged package | every section | after WP7; needs `lore`'s `test/` and `input/` folders from the user |

- **Files dropped in WP1** (decisions "Set of degradation models", "Plotting", "Validation tools", "Gamma formulation",
  "Cluster jobs and experiments", "Documents", "Follow-up decisions from writing the spec"): `main`'s old
  `gaussian.py` and `inverse_gaussian.py`; the Gamma calibrations and the old Gamma solve route; `main`'s older
  plotter; `utils/model_registry.py`; `validate_gamma_result` and the two validators of the old Gaussian format; the
  formulation-count tools and the regression scripts that call them; `csv_to_plotter.py`; `fix_root_gap.py`;
  `test_rainflow.py`; `rainflow_v2.py`, `rainflow_sparse.py` and `legacy/`; the family-A cluster jobs; the results
  computed before the merge (`results/`, scope confirmed file by file). WP1 lists every file before deleting it, and
  Claude checks each against the decisions.
- **Order and parallel work.** WP1 and WP2 run now, one agent. WP4 and WP5 can run in parallel once their reviews are
  done, because they touch different files; WP4 first fixes the fields of `FleetConfig`, which WP5 reads. WP6 runs one
  agent per builder file, in parallel. WP7 runs one agent per tool, in parallel.
- **Models of the agents.** Opus for WP4 to WP7 (formulation and tools); Sonnet for WP1 and WP2 (narrow tasks with a
  precise list).
- **Checks of each package.** The agent writes unit tests with the package; Claude reviews the diff against the spec
  sections before the commit point. Step 3 (the `spec-alignment` skill) then checks the whole code and the README.
