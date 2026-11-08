Status: done. `CLAUDE.md` written in `C:\Users\utente\Documents\GitHub\Fleet_Management-merge\CLAUDE.md` (not committed).

## Items of the old `CLAUDE.md`

| Item of the old file | Action | Reason |
|---|---|---|
| Spec and scoping hold different things (user, 26 Oct 2026), with its two sub-bullets | kept verbatim, quotation and date included | standing instruction |
| Style model for the spec: `merge/spec_drafts/style_reference_spec_v03.tex` | dropped | merge file, to be archived |
| "No version is trusted by default" (user, 25 Oct 2026) | kept, shortened: the merge-specific part (port `lore` into `main`) dropped; the rest is in the section "standing instructions" as "Judge before choosing" | the principle outlives the merge |
| Pointer to `EULER.md` and the Euler workflow (7 Nov 2026) | kept as a pointer, with the line on Sonnet agents; `EULER.md` is not in the branch, so the path is the working copy | standing instruction |
| Agents rule | moved to a pointer to `~/.claude/rules/subagents.md` | the rule lives there now |
| State of this checkout (branch states, `input/` and `test/` absent, stale tracked files, `.gitignore` of `CLAUDE*`, divergence of `lore` and `main`) | dropped | merge in progress; `CLAUDE*` is no longer ignored in the worktree |
| No Gurobi or pytest on the workstation | kept in "Commands", reworded (tests run on Euler) | still true |
| Known defect of `lore` (ARA1 anchor) | dropped | merge decision, now in code and spec |
| "Known differences between code and spec" | dropped | all four concerned `lore`'s code |
| `lore-vs-main.md`, `merge/` | one line: archived at the clean-up | as the task asked |
| `ASSESSMENT.md` | kept in "Other files" as superseded | file still exists |
| `Things to change.txt` | kept (to-do list) | |
| Old architecture (models/, `_DISPATCH`, `formulation: lp`, `ctx` namespace, ARA1) | replaced by the architecture of the merged code | `lore` code is obsolete |
| Recycle Bin (from memory note, not the old file) | added one line: guarded script, global rule | memory note exists |

## Checks (against the worktree, commit 6a22bd3)

- Paths exist: `euler/` (7 files, all named in the table), `examples/smoke`, `examples/regression`, `experiments/`, `tests/data`, `tests/reference_model.py`, `remote_smoke_test.py`, `spec/spec.tex` and `references.bib`, `scoping.*`, `ASSESSMENT.md`, `Things to change.txt`, `lore-vs-main.md`, `merge/`, `input/`, `results/validator_tests`.
- Smoke-test options (`--no-pytest`, `--dry-run`, `--out-dir`), outputs and 12 scenarios: from its docstring.
- Names in `base.py`: `FleetModel`, `register_cell_builder`, `get_cell_builder`, `dispatch_cell`, `solve_mixed`, `build_objective`, `add_*_rows`, `add_state_vars`, `add_selector_vars`, `add_pwl_vars`, `repair_factor`, `apply_binary_warm_start`, `complete_binary_warm_start`, `apply_relaxation_warm_start`, `_load_builders`. In `config.py`: `FleetConfig`, `load_config`, `read_input`, `REJECTED_KEYS`, `SUPPORTED_MODELS`. In `solver.py`: `solve_mapping`, `RESULT_KEYS`, `read_result`.
- Spec sections named in the file exist in the worktree spec (version 1.0): Architecture, Cell-Builder Interface, Repair Laws, Encoding of the Dynamics, Linear Forms of the Reliability Rows, Loop Constraint, Result Files, HDF5 Layout, Input File Schema.
- Models table and repair factors: from the builder docstrings and spec "Repair Laws".
- The sweep workers use `spawn` (`horizon_sweep.py` line 766).

## Open points

- The worktree spec has no section "Tolerance of the checks" (the `lore` spec has it). `CLAUDE.md` states the rule from `validation/tolerance.py` and says so. The section should be ported to the spec.
- `EULER.md` is not in the branch; `CLAUDE.md` points to the working copy. Decide whether to track it.
- I did not run the code. The parts of the architecture on `extract`, warm starts and the greedy heuristic come from the spec and docstrings.
