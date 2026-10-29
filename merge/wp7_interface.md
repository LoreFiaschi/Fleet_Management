# WP7 (tools): split, file ownership and shared rules

Status (29 Oct 2026): fixed by Claude before the planners start; the four plans follow it. A change needs Claude's
agreement, because the other plans depend on it.

- **Purpose.** WP7 builds the tools of the merged package on WP4 to WP6: horizon sweeps, warm starts and the greedy
  start, the plots, the structural validator and the dashboards, the Gamma validators, and the public API. Four Opus
  planners plan it in parallel; this file fixes which files each plan may change and what the plans share.
- **Where.** Code: the worktree `C:\Users\utente\Documents\GitHub\Fleet_Management-merge`, branch
  `merge/lore-into-main`, at commit 0421e80 (WP6). Records (plans, reviews): the `lore` checkout,
  `C:\Users\utente\Documents\GitHub\Fleet_Management\merge\`. The merge branch never edits its own `merge/` folder.
- **Authority.** `spec/spec.tex` of the worktree: Architecture (165-261), Output Specification (1694-1870, above all
  "Reports"), Horizon Sweeps (1871-2092), Warm Starts (2093-2255), Plotting API and Layout (2256-2355), Validation
  Tools (2356-2658). Where the worktree's spec and the `lore` copy
  (`C:\Users\utente\Documents\GitHub\Fleet_Management\spec\spec.tex`) differ, the `lore` copy wins; it adds sentences
  to the input, check, result-file and model sections (29 Oct 2026), hardly any to the tool sections.
- **Contracts already in the code.** `config.load_config`, `config.read_input` and `FleetConfig`
  (`merge/wp4_wp5_interface.md`); `solver.solve`, `solver.solve_mapping`; `base.solve_mixed` and the result dictionary
  (spec, "Result Dictionary"); the cell builders of WP6. A plan that needs a change to one of them asks under
  "Questions"; it does not change them, except as the table below allows.
- **Deferred to WP7 by earlier packages** (read these sections): `merge/wp4_plan.md` section 4 and section 7;
  `merge/wp5_plan.md` sections 5 and 7 (warm-start modes `hint` and `both`, the three warm-start functions);
  `merge/wp6_plan_gamma.md` section 8, point 4 (WP7 modules that read data the removed Gamma code supplied:
  `gamma_replay_validator.py:75-83`, `gamma_tail_validator.py:187-216`, `greedy_warm_start.py:76` and `:120-152`,
  `horizon_sweep.py`, `mixed_plotter.py:460`).
- **Decisions** (`merge/decisions.md`): "Horizon sweep", "Warm starts", "Plotting", "Validation tools", "Output schema
  and file formats", "Solver defaults", "Gamma formulation" (for the validators), "General rules for the merge".

## Workflow (user, 28 Oct 2026, confirmed for WP7 on 29 Oct 2026)

1. An Opus agent plans each part: `merge/wp7_plan_<part>.md`.
2. Sonnet agents carry out the plans, at most four at once.
3. The same Opus agent reviews the result against its plan and the spec, and corrects it.
4. Claude reviews the diff and commits on `merge/lore-into-main`.

## File ownership

| Plan (part) | Files it may change or create | Spec |
|---|---|---|
| `sweeps` | `horizon_sweep.py`, `greedy_warm_start.py`; in `degradation_model/base.py` only the warm-start functions (`apply_binary_warm_start`, `complete_binary_warm_start`, `apply_relaxation_warm_start`) and the warm-start part of `solve_mixed` (the modes `hint` and `both`); tests `tests/test_horizon_sweep.py`, `tests/test_warm_start.py` | Horizon Sweeps, Warm Starts |
| `plots` | `utils/plotter.py` (from `utils/mixed_plotter.py`: the spec names `utils/plotter.py`), `fleet_management/__init__.py` (the public API of spec "Public API", for every part); test `tests/test_plotter.py` | Plotting API and Layout, Public API |
| `validator` | `validation/validator.py`, `validation/dashboard_inspector.py` (the spec's name; from `validation/dashboard_rainflow.py` as the decision "Validation tools" says), `validation/dashboard_gamma.py`, `validator_dashboard.py`; tests `tests/test_validator*.py` (the two existing files rewritten or replaced) | Structural Validator, Dashboards |
| `gamma_validators` | `degradation_model/gamma_utils/*.py` (the four modules the spec names; `gamma_diagnostics.py` and `gamma_process.py` only if the decisions keep them); test `tests/test_gamma_validators.py` | Validators of the Gamma Model |

- **The public API** (`fleet_management/__init__.py`) is owned by `plots`. It exports exactly the names of spec
  "Public API" (lines 222-233). The other plans keep those names and signatures; each plan states, in its section
  "Exports", the names its files define for `__init__.py`.
- **`import fleet_management` must work without Gurobi**, as it does today. A tool module imports `base` or
  `gurobipy` only inside the function that solves.
- **Nobody edits** `config.py`, `solver.py`, the cell builders, the parts of `base.py` that `sweeps` does not own,
  the spec, `examples/`, `experiments/`, `euler/`, `remote_smoke_test.py`. These scripts are WP8's; each plan lists, in
  its section "Callers outside src", the scripts in `examples/`, `experiments/` and `euler/` that its changes break.
- **Files the spec drops** (for example `utils/mixed_plotter.py` after the move, `dashboard_rainflow.py` after the
  rename): the plan names them in a section "Removals"; Claude removes them at the commit. Executors do not delete
  files.

## Rules for every plan

- Every result file the tools read follows the spec's "Result Files" and "HDF5 Layout"; read it with the writer's
  counterpart, not with a private reader.
- Code style: as the existing file being changed, or as `base.py` after WP5 for a new file.
- Tests: pytest style; a test that needs Gurobi is skipped per test, not per module (the pattern of
  `tests/test_gamma.py`).

## Local tools (this computer has numpy, scipy, pyyaml, h5py, pandas, matplotlib; no pytest, no gurobipy, no
streamlit; nothing may be installed)

Scratchpad: `C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad\`
(copies in `merge/wp45_work/` of the `lore` checkout).
- `wp4\run_tests.py <test module> [name]`: runs pytest-style tests without pytest; `wp6\gamma\run_tests_gamma.py` and
  `wp6\rainflow\review_run.py` add `pytest.mark`, `skipif` and `importorskip`.
- `wp5\stubs\gurobipy\`: a stub that lets modules importing `gurobipy` load.
- `wp5\fakegp\` (extended in `wp6\giw\`): a linear emulator of `gurobipy`; it solves the LP once the binaries are
  fixed or given by a complete MIP start. It cannot handle quadratic rows, indicator rows with free binaries, or a
  search over the binaries.
- `wp6\rainflow\names_check_try.py`: the undefined-name check, aware of the guarded import of `base`.
- Plots: matplotlib with the `Agg` backend; render to PNG and look at the image.

## Decisions on the plans' questions (29 Oct 2026)

These override the four plans where they differ. Executors and reviewers follow them.

1. **One reader of result files: `solver.read_result(path) -> dict`.** It is the validator plan's `read_result`
   (`wp7_plan_validator.md`, step A2: suffix check, `read_input`, the keys of `RESULT_KEYS` with `None` for a
   missing key, every key of `_HDF5_DATASET_KEYS` as a float array with `null` as NaN), placed in `solver.py` below
   the writers instead of `validator.py`. The validator plan's Task A writes it there as its first step; this is the
   only edit of `solver.py` in WP7. `validator.py` imports it from `solver`. The plots plan (schedule plot) and the
   Gamma validators' file versions read result files with it, not with `config.read_input`. Inputs and sweep reports
   stay with `config.read_input`.
2. **Greedy heuristic (user, 29 Oct 2026):** keep the spec's order (repair first, replacement only when no repair
   helps) for every model except Wiener. At a Wiener cell, on each free day, try a repair and a replacement and keep
   the one that lowers the excess most. The spec's "Greedy Heuristic" changes for Wiener only; the sweeps planner
   amends its step B3 and its expected values, and writes the spec sentence.
3. **To-do (user, 29 Oct 2026):** a cheaper certified tail bound for the exact Gamma validator when a component is
   repaired many times goes to `Things to change.txt`. WP7 keeps the spec's procedure.
4. **A result whose `model` differs from the input** raises `ValueError`, in the dashboards and the Gamma validators.
5. **Spec additions, accepted:** the row of the plot error tables (plots Q3), the clause on result errors the spec
   does not list (validator Q3), the definition of the greedy's reference trajectory (sweeps Q6), the error on an
   unknown key of the `validation` block (Gamma validators Q9). Each reviewer writes the exact sentence; Claude adds
   it to the spec.
6. **Validator:** the fixtures `results/validator_tests/input.yaml` and `result.yaml` may be written by its Task A;
   `validate` checks the loop rows of every tracked state, as the spec says.
7. **Removals, by Claude at the commit:** `utils/mixed_plotter.py`, `validation/dashboard_rainflow.py`,
   `tests/test_validator_negative_cases.py`, `tests/test_validator_positive_case.py`, the old
   `results/validator_tests/bad_*.yaml`, `gamma_utils/gamma_diagnostics.py`, `gamma_utils/gamma_process.py`,
   `gamma_utils/GAMMA_DIAGNOSTICS.md`.
8. Every other question follows the plan's recommendation.

## Execution order

Batch 1: validator Task A (with `read_result` first), sweeps Task A, Gamma validators Task 1, and the sweeps planner's
amendment of step B3. Batch 2: plots Task A, Gamma validators Task 2, sweeps Task B, validator Tasks B and C (at most
four at once). Batch 3: plots Task B (public API), last.
