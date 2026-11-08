Status (8 Nov 2026): done, all five items. File: `merge/spec_drafts/fragments_D.tex`, first part only (Architecture to Greedy Heuristic). Line endings: LF before and after (0 CR). Not built. Brace, `itemize`, `enumerate`, `lstlisting` and `\tabstart`/`\tabend` counts balanced in the edited part.

Code read: `solver.py`, `config.py`, `degradation_model/base.py` and the five builders, `horizon_sweep.py`, `greedy_warm_start.py` (worktree `Fleet_Management-merge`).

# Changes (section: old -> new; code)

## Item 1 (group 2)
- **M25, Early Stop, "Eligible to stop"**: "cases n-1 and n qualify" -> also "it is available (cases n-1 and n are feasible)". `_annotate_gradients` sets eligibility only after the feasibility test.
- **M26, Package Layout**: `pyproject.toml` line gets the extras `dev` (pytest) and `dashboard` (streamlit); new lines `scoping.tex`, `scoping_bib.bib` (`scoping.pdf` named as the build) and `input/` (the input of the Euler convergence job; used by `euler/gamma_horizon_convergence.sbatch`); `config.py` line now also lists the derived constants (support constant, natural ranges, `_derive`) and the checks of the replacement states (`_check_ranges`). `git ls-files` and `pyproject.toml`.

## Item 2 (group 3 B, Cell-Builder Interface)
- `prepare` bullet: also creates `y^s`, `y^{g,s}`, selectors `n`, segment binaries `w` (`add_state_vars`, `add_selector_vars`, `add_pwl_vars`).
- `add_cell` bullet: the start rows and the loop rows are built with the dynamics.
- Old paragraph on `ctx` replaced by two tables: members of `ctx` (`FleetModel` fields) and entries of `ctx.extras` (gaussian `v_max_user`; wiener `sigma`, `v_max_user`; inverse_gaussian `mu_bar`; gamma `beta_star`, `cap`; rainflow `rows`).
- New block "Helper functions of `base.py`": what `prepare` calls, the order in `add_cell` (selector rows; `add_quantity_rows` per quantity, `mu` first, with start rows, day rows, loop rows; reliability rows and the optional `v_max_user` row; repair-cost rows); reading helpers; fleet rows.
- New table "Errors of the builders": `ValueError` for a non-finite value read (`_cell_value`, `day_increment`, rainflow `prepare`), missing variable, wrong number of increments, unknown form/bound/quantity; `NotImplementedError` for a model without a builder.
- Building order: new step 3, `gurobi_params` applied again after `prepare` (M1; `build_fleet`); `NonConvex` of the user wins.

## Item 3 (group 3 A, Data Flow)
- Step 2: `TypeError` for file data that are not a mapping (`read_input`).
- Closing paragraph: `solve_mapping` checks `warm_start_mode` first, then `results_path`; so an unreadable file is reported before a wrong mode; `load_config` raises `TypeError` for non-mapping data (`_check_keys`); Gurobi errors are not caught, except those of the completion and the relaxation start (M4).

## Item 4 (group 3 C)
Horizon Sweeps:
- "Feasible case": from "result holds a solution" to "row has an `objective` not null and status not in the three"; error row not feasible (`_is_feasible`).
- `input_path` row: how `H1` is read (integer rule, `>= 1`); steps and Errors: `H1 < 1` and a missing or non-integer `H` raise `ValueError` before the first case (M14; `_horizon_H1`, `sweep_operating_horizons`).
- `objective_mode` row: the three values, ranking by J_op/H2 in all three; new table "Texts of `objective`" (`SWEEP_OBJECTIVES`), label `sec:sweep_texts`.
- Early Stop: rule on the last candidate gives the reason of the rule, `stopped_early` false; same in the `stopping_rule` report row.
- `fixed_dimensions` (sweep and grid) and case-error row: all three of F, M, L are null if one cannot be read (`_input_sizes`).
- Parallel Cases: worker pool (one pool, `spawn`, all cases submitted at once, a free worker takes the next); a dead worker makes every unfinished case a case error (`future.result()` raises).
- Grid: new step 1 (read input, must be a mapping, `H` not read, all checks for every `H1` before the first case); Errors table now `TypeError` (non-mapping, unknown keyword in `sweep_options`) and `ValueError` (empty or non-positive-integer lists, `objective_mode` in `sweep_options`, `evaluation_horizon`, any operating-sweep argument error for any `H1`, `output_path` extension); `objective` text of the grid report and of `H1_sweeps`.
Warm Starts:
- Interface: `warm_start_mode` checked in `solve_mapping` and by the sweeps; `completion_time_limit` must be a finite number `>= 0` (null and flag refused), order of the source checks (M15; `_check_completion_time_limit`, applies only to a source with `x`); new bullets on the time of `model_construction_seconds` (covers build and all warm-start phases) and a table "Step that did not run" (no source, completion not run, relaxation disabled, skipped).
- Day Mapping: rounding is Python `round` (ties to even), no clipping.
- Completion: time-limit rule; copy fixes binary and integer variables (rounded); copy parameters (OutputFlag 0, LogToConsole 0, LogFile empty, TimeLimit; others inherited); `runtime_seconds` is Gurobi's `Runtime` of the copy; solution means `SolCount > 0`; Gurobi error gives status `"error"`, `values_copied` 0, empty `iis_constraints`; copy always disposed (M4).
- Relaxation: `feasRelaxS(0, False, False, True)` described: objective is the sum of violations only, original objective dropped, quadratic rows, bounds, general constraints kept; copy keeps other parameters incl. logging; sets `Start` of binary and integer variables; error gives status `"error"` (M4).
- Warm Starts in Sweeps: test for a source (result has `x`; source holds `H1`, `H2`, `x`, `m`, `r`, no completion request) and its difference from the feasibility test (`_source_of`).
- Greedy: `"pwl"` segment: first segment for a negative mean (clip to `[0, tau]`, tangent taken at the true mean); one sentence that the loop excess compares day `T` with day `H1` and is stricter than the loop rows to `y` (item 5; `_cell_excess`).

## Item 5
Done in the Greedy Heuristic (see above).

# Disagreements between the code and the items or the old spec
- `prepare` "may tighten variable bounds": no builder does this; the phrase was left as written ("may"). Remove it if you prefer.
- The grid reads no `H` of the input; its checks do not include `H1 < 1` because the candidates are positive integers.

# Open points for other fragments (not edited)
- **`fragments_C.tex` line 414** (keys of `completion`, `status`): add `"error"` to the list.
- **`fragments_C.tex` line 428** (`relaxation_warm_start`, "status is a Gurobi status"): add a row "`status` is `"error"`: Gurobi raised an error; `relaxation_objective` null, `values_copied` 0".
- The `performance` table (fragments_C): `model_construction_seconds` includes the warm-start phases (stated here in Warm Starts, "Time"); the table there should agree.
