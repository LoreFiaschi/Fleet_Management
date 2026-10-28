Status: in progress. Done: A1, A2. Next: A3.

# Task A progress (base.py of the merge worktree)

Baseline (before A1): load_base OK; names_check 0 undefined; local_checks failures 0 (MISSING: the pure functions of later steps).

Tools in this folder: `run_checks.py` (ast + CRLF count + load_base + names_check + local_checks in one go), `splice.py START END NEWFILE` (replace a line range of base.py, keeps CRLF), `check_a1.py` (T2 increment case). base.py uses CRLF line endings; the Edit tool keeps them (checked).

## Steps

| Step | Result | Check |
|---|---|---|
| A2 Options, build_context | Done. `resolve_run_options(cfg)` returns `dict(cfg.options)`; `pick`, `resolve_costs`, `cell_max`, `make_accessor` removed; `cost_days` added; `build_context` rewritten as in the plan. Section banner renamed "Options, costs, accessors" -> "Run options and solver settings" (costs and accessors are gone from it). | run_checks OK; `cost_days` 3 PASS; names_check: only `build_fleet: resolve_costs` (A7 rewrites `build_fleet`) |
| A1 Context | Done. `import importlib` added at the top (needed by A7). `TRACKED_QUANTITIES`, `phase_day`, new `FleetModel` (fields in the plan's order, `increment`), `rainflow_cells` removed. | run_checks OK; local_checks `phase_day` 3 PASS; `check_a1.py` gives `[1,2,1,2,3]` and `[10,20,1,2,3]`, depot 0.0, KeyError message as planned |

## Problems

(none yet)
