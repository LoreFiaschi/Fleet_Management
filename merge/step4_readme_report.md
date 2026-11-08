Status: done. `README.md` of the worktree `Fleet_Management-merge` is rewritten (452 lines, CRLF kept: 327 CR before, 452 CR and 452 lines after). Not committed.

## Sections

Purpose and status; Prerequisites; Installation; Quick start; API reference (Solve, Horizon sweeps, Plots, Validation); Input file; Output; Formulation in brief (models and reliability rows, rainflow bounds, repair laws, encodings, forms, loop rows); Dashboards; Validation (with the tolerance rule); Horizon sweeps and warm starts; Project structure; Running the tests; Further documentation.

## Checks run

| Check | Result |
|---|---|
| Python snippets parse (`ast.parse`) | quick-start block parses; the 11 signatures parse as function definitions |
| Signatures against the code (names, parameter order, defaults, keyword-only marker) | all 11 equal |
| API names equal `__all__` of `src/fleet_management/__init__.py` | equal (11 names) |
| Paths named in the README exist in the worktree | all exist, except two files created at run time (`logs/`, `mixed_result.yaml`) |
| `load_config(read_input("examples/smoke/mixed_example.yaml"))` | loads (F=3, M=2, L=2, H1=2, H2=3); no solve (no Gurobi) |
| Defaults of run options against `config.py` | `verbose` 1, `mip_gap` 0.1, `time_limit` 3600, `objective_mode` total, `progress_interval_seconds` 5.0, `relaxation_time_limit` 60.0, selector defaults: equal |
| Tolerance rule against `validation/tolerance.py` and the spec section "Tolerance of the checks" | stated as in both; Gurobi defaults 1e-6 and 1e-5 |
| Smoke-test options against `remote_smoke_test.py` | `--no-pytest`, `--dry-run`, `--out-dir` (default `./smoke_test_output`), 12 scenarios |
| No reference outside the repository | `docs/PROJECT_LAYOUT.md`, `euler/README.md`, `EULER.md` not named |

## Disagreements and open points

- No disagreement between code and spec found in what the README states (signatures, defaults, keys).
- Not checked by a solve: the quick start calls `solve` on `mixed_example.yaml`; it needs Gurobi, absent here. The input loads and the scenario is part of the smoke test (scenario 12).
- The free `gurobipy` licence: the README says "limited to small models", as the task asked, with no size number (not checked here).
- The `euler/*.sh` scripts contain the ETH cluster modules and `VENV=/cluster/home/...`. The README says so and tells the reader to edit these lines on another cluster. It names the folder `euler/` and the "ETH Euler cluster" only as a fact about the scripts.
- `examples/regression/RegressionREADME.md` is named under Further documentation; the plan updates that file in a later step of step 4. If it is removed, delete that bullet.
- The README names the exact Gamma validator and the greedy heuristic as "not exported", as the spec does.
- `git status` of the worktree also shows `experiments/vbz_greedy_warm_start/README.md` modified and `CLAUDE.md` untracked: not from this task.
