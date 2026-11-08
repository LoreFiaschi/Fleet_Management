# WP7 plan, part `sweeps`: horizon sweeps, warm starts, greedy heuristic

Status (29 Oct 2026): done. Plan amended after the decisions of `merge/wp7_interface.md` (Wiener rule in B3 and
3.2); Tasks A and B executed and reviewed (section 8: corrections, decisions, spec sentences, test counts). Next:
Claude adds the spec sentences of section 8 and commits; WP8 runs the Gurobi-only tests.

Read in full: `merge/wp7_interface.md`; the worktree spec 165-261 and 1694-2255 (same text on `lore`; line numbers
are the worktree's); `horizon_sweep.py`, `greedy_warm_start.py`, `solver.py`, `config.py`, `base.py`, the five
builders; `wp4_plan.md` 4, 7; `wp5_plan.md` 5, 7; `wp6_plan_gamma.md` 8; the four decisions named in the brief and
"Follow-up decisions"; scoping 387 (reference trajectory).

**Scratchpad** `SP = C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad\wp7\sweeps\`
(the planner's tools; executors write their own files only there):
- `run_tests_wp7.py <test file> [name] [--fake]`: pytest-style tests without pytest, fixtures `tmp_path`,
  `monkeypatch`, `caplog`; `--fake` loads `fakegp`, the WP6 linear emulator plus `Start`/`VarHintVal` defaults
  `1e101`, `NonConvex = -1`, `Model.copy`, `computeIIS` (marks every row), `feasRelaxS` (SKIP). Checked on
  `tests/test_gamma.py`: 21 passed and 15 skipped without `--fake`, 34 passed and 2 skipped with it.
  `probe_model_inputs.py`: the model inputs of section 3.2 build, and case G completes (120 values).
- `greedy_ref.py` (`greedy_ref_out.txt`): the planner's own implementation of spec 2216-2253. `verify_greedy_lp.py`
  fixes each greedy schedule in the real WP6 rows and solves the LP: every schedule without failures is feasible.
- `sweep_expected.py` (`sweep_expected_out.txt`): gradients, classes, stops and rankings of the test tables.

## 1. Gaps

Code in `src/fleet_management/`; "spec N" is a line of the worktree spec. Items of other packages are marked.

**Horizon sweeps** (`horizon_sweep.py`)
- S1 `:13, 405-422`: temporary YAML files solved with `solve`; spec 1933: `solve_mapping(..., None)`, no file.
- S2 `:362-366, 577-580`: YAML input only; spec 1930, 2056: the input reader `config.read_input`.
- S3 `:335-348`: no `warm_start_mode`, no `n_workers`; `maximum_mip_gap_for_stopping=0.05`; spec 1901-1909: `None`.
- S4 `:376, 582-583`: `int(value)` turns 2.5 into 2 and accepts `"3"`; spec 1914, 2057-2058: positive integers.
- S5 `:387-393`: the stop gap must be a number; spec 1920: `None` or a finite number >= 0.
- S6 no check of `warm_start_mode`, `n_workers`, `objective_mode`, `evaluation_horizon`; spec 1924-1925, 1931, 1941.
- S7 `:418-422`: one exception stops the sweep; spec 1942, 2028-2030: ERROR log, row `"error: <message>"`, go on.
- S8 `:453-455, 262-265, 705-707`: `data["L"]` raises `KeyError` when the input omits `L` (optional); defect.
- S9 no parallel cases; spec 2033-2040. S10 `:421`: warm starts always in the mode `start`; spec 2183.
- S11 `:29-64, 426-439, 486, 491-493`: `calibration` and Gamma entries read keys of the removed Gamma code (WP6
  point 4); spec 2021-2022.
- S12 `:448-509`: row key `time_limit_seconds` (spec 2019: `requested_time_limit_seconds`); key order not spec's.
- S13 `:110-117, 659-664`: proven means optimal with gap <= 1e-8; spec 1889, decision "Solver defaults": `"optimal"`.
- S14 `:140-143, 197-198`: one gap threshold; spec 1966: per case, the argument or its `requested_mip_gap`.
- S15 `:146-200`: only the last row is annotated; spec 2039: every rewrite recomputes all annotations.
- S16 `:523-537`: the checkpoint after a stop has `complete = true`; spec 2027: `false` and `"in_progress"`.
- S17 `:261-327`: `fixed_dimensions` holds `H1`; `warm_start` has `method`, no `mode`; no `n_workers`; extra keys
  `interpretation`, `selection_interpretation`, `best_H2`, `best_J_op_average`; spec 1989-2006.
- S18 `:67-96`: an error row or a `null` count makes the growth raise `TypeError`; spec 2003: rows with counts.
- S19 `:119-128, 665-666`: the tie rule holds only through the list order; spec 1893 (parallel rows).
- S20 `:259`: the objective text ignores the mode `evaluation_total` of the grid; spec 1994.
- S21 `:597-617`: the grid writes a temporary YAML file per `H1`; spec 2067. S22 `:621-627`: the grid report is
  rewritten after each `H1`; spec 2069: after every case. S23: `objective_mode` in `sweep_options`; spec 2062.

**Greedy heuristic** (`greedy_warm_start.py`)
- G1 `:16-41, 452-476`: fields `feasible_assignment`, `artificial_assignments`; spec 2243-2253 (seven keys; `F > M`
  is checked by `config.py:857`). G2 `:504-516`: every free train goes to the depot; spec 2223: idle unless some
  `s_hat > r tau`. G3 `:449, 523, 531`: the source carries `idle`; spec 2209-2214.
- G4 `:62-68, 177-269`: only rainflow Cantelli and Bernstein variances; spec 2230-2234: every tracked quantity and
  anchor of every model (Gaussian `v`; Wiener `v`, `sigma^2` per mission, no repair effect; `R`, `K`).
- G5 `:71-169`: imports `gamma_repeated_calibration` (removed in WP1), reads removed fields (WP6 point 4).
- G6 `:195-197`: anchors start at 0; spec 2230: the starting anchors.
- G7 `:282-339`: excesses divided by `tau`, `tau^2`, not by the natural ranges; no anchor excess; reliability only at
  Cantelli cells, the form read from `cfg.options` keys that do not exist; no `pwl`; spec 2231-2236.
- G8 `:383-388`: trial days are depot days only; spec 2237: every free day. G9 `:400`: the tie key holds the raw
  excess; spec 2237. G10: no replacement trial; spec 2238. G11 `:368-376`: the proxy ratio takes every intermediate
  trajectory; spec 2251 (Q7).

**Warm starts** (`degradation_model/base.py`)
- W1 `:1261-1265`: `NotImplementedError` for `"hint"`, `"both"` (WP5); spec 2143-2147. W2 `:292-348`: `Start` only;
  spec 2144-2146. W3 `:304-305`: `H1`, `H2` default to the model's; spec 2114 (Q4).
- W4 `:346, 1276`: `source_horizon` is a mapping, `mode` follows `completion`; spec 1820-1824: a pair, key order.
- W5 `:324-330`: no `x^m` value without a source `m`; spec 2135: `0`. W6 `:331-341`: `n = 1 - m` also when `x^r`
  has no value; spec 2138. W7 `:351-379`: the completion runs in every mode; spec 2145: not with `"hint"`.
- W8 `:441-487`: the relaxed copy is never disposed (defect); the docstring says "hints"; spec 2175: `Start`.

**Other packages** (not planned here): `__init__.py` exports (`plots`); `utils/mixed_plotter.py` reads sweep
reports (`plots`); the spec has no definition of the reference trajectory (Q6, spec text).

## 2. Executor tasks

Rules for both tasks: worktree `C:\Users\utente\Documents\GitHub\Fleet_Management-merge` only; keep CRLF line endings
(both files and `base.py` are CRLF; write new test files with CRLF too); style of `base.py` after WP5 (docstrings that
cite the spec, `# Eq.` comments); no file outside the task's list; no deletion; progress notes in
`SP\taskA_progress.md` or `SP\taskB_progress.md` (a `Status:` line first), one line per finished step. Do not spawn
sub-agents. Never put the user's name or email address in any request. Check after every step:
`python -c "import sys; sys.path.insert(0, r'C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src'); import fleet_management, sys as s; assert 'gurobipy' not in s.modules"`.

### Task A: `horizon_sweep.py` and `tests/test_horizon_sweep.py` (8 steps)

The module is rewritten in place, top to bottom; functions keep their names where they survive
(`_select_horizon_cases`, `_gradient_stop_reason`, `_formulation_growth`, `_assemble_sweep_report`,
`_write_sweep_report`, `_assemble_grid_report`).

**A1. Header, imports, constants, small helpers** (spec 1871-1894, 2009-2024).
- Docstring: "Horizon sweeps (spec, "Horizon Sweeps"): `sweep_operating_horizons` and `sweep_horizon_grid`. Each case is
  solved with `solver.solve_mapping`, without a file; `base` and Gurobi load only inside it."
- Imports: `logging`; `from concurrent.futures import ProcessPoolExecutor, as_completed`; `deepcopy`; `Path`;
  `Iterable`; `numpy as np`; `yaml`; `from fleet_management.config import OBJECTIVE_MODES, _infer_L, _integer,
  read_input`; `from fleet_management.solver import WARM_START_MODES, _to_plain, solve_mapping` (module-level names,
  so that tests patch `horizon_sweep.solve_mapping` and `horizon_sweep.ProcessPoolExecutor`).
- `logger = logging.getLogger(__name__)`. Delete `_gamma_calibration_summary`.
- Constants: `FORMULATION_KEYS` (the nine keys of today); `TIMING_KEYS = ("model_construction_seconds",
  "optimizer_call_seconds", "solution_extraction_seconds", "backend_wall_seconds", "branch_and_bound_nodes",
  "simplex_iterations", "work_units")`; `COST_KEYS = ("J_initialization", "J_op", "J_op_average", "peak_damage",
  "damage_penalty", "operating_objective", "J_total", "projected_evaluation_cost", "evaluation_objective")`;
  `GRADIENT_KEYS = ("cost_gradient_per_H2", "relative_cost_gradient_per_H2", "gradient_classification",
  "gradient_stopping_eligible")`; `ROW_KEYS = ("H1", "H2", "T", "dimensions", "status", "objective_mode",
  "evaluation_horizon", "objective") + COST_KEYS + ("warm_start", "objective_bound", "mip_gap", "absolute_mip_gap",
  "requested_mip_gap", "requested_time_limit_seconds", "optimizer_seconds", "solutions_found", "formulation",
  "timing") + GRADIENT_KEYS`; `NO_SOLUTION_STATUSES = ("infeasible", "unbounded", "inf_or_unbounded")`;
  `SWEEP_OBJECTIVES = {"operating_average": "minimize J_op/H2 + C_D*u; rank by J_op/H2", "evaluation_total":
  "minimize J_initialization + (evaluation_horizon - H1)*J_op/H2 + C_D*u; rank by J_op/H2", "total": "minimize
  J_initialization + J_op + C_D*u; rank by J_op/H2"}`.
- `_is_integer(value) -> bool`: `isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))`.
- `_positive_integers(values, name) -> list[int]`: `values = list(values)`; empty: `ValueError(f"{name} must contain
  at least one value.")`; an entry that is not `_is_integer` or is `<= 0`: `ValueError(f"{name} must contain positive
  integers (got {value!r}).")`; returns `[int(v) for v in values]`.
- `_horizon_H1(data) -> int`: `H = data.get("H")`; a list or tuple of length other than 2: `ValueError("'H' must be an
  integer or a pair [H1, H2].")`; returns `_integer(H[0], "H1")` for a pair, else `_integer(H, "H")` (spec 1913).
- `_input_sizes(data) -> dict`: `{"F": F, "M": M, "L": L}` with `F = _integer(data["F"], "F")`, `M` alike,
  `L = _infer_L(data, F)`; on `KeyError`, `TypeError`, `ValueError` all three are `None` (the cases then fail with
  the real message). Fixes S8.
- Check: the module imports; `ROW_KEYS` has 31 entries.

**A2. Rows of a case** (spec 2009-2030; fixes S1, S7, S11, S12).
- `_case_row(result) -> dict`: the keys of `ROW_KEYS`, in that order: `H1`, `H2` (ints from the result), `T = H1 + H2`;
  `dimensions = {"F", "M", "L", "H1", "H2", "T"}` from the result; `status`, `objective_mode`, `evaluation_horizon`,
  `objective` and the `COST_KEYS` from the result; `warm_start` = `result["warm_start"]`; `objective_bound` =
  `result["bound"]`; `mip_gap`; `absolute_mip_gap = max(0.0, objective - bound)`, `None` if either is `None`;
  `requested_mip_gap`, `requested_time_limit_seconds` from `performance`; `optimizer_seconds` =
  `performance["optimizer_call_seconds"]`; `solutions_found`; `formulation = {k: performance.get(k) for k in
  FORMULATION_KEYS}`; `timing = {k: performance.get(k) for k in TIMING_KEYS}`; the four `GRADIENT_KEYS` as `None`.
  Return `_to_plain(row)` (numpy to plain types; NaN and infinity to `None`).
- `_error_row(H1, H2, sizes, error) -> dict`: every key of `ROW_KEYS` `None`, then `H1`, `H2`, `T`,
  `dimensions = {**sizes, "H1": H1, "H2": H2, "T": H1 + H2}` and `status = f"error: {error}"`.
- `_source_of(result) -> dict | None`: `None` if `result.get("x") is None`, else `{k: result.get(k) for k in ("H1",
  "H2", "x", "m", "r")}` (the row keeps no schedule; spec 1933).
- `_solve_case(case, warm_start, warm_start_mode, keep_source) -> tuple`: `result = solve_mapping(case, None,
  warm_start=warm_start, warm_start_mode=warm_start_mode)`; returns `(_case_row(result), _source_of(result) if
  keep_source else None)`. Module level, so a worker process can run it. It catches nothing.
- Check: `list(_case_row(r)) == list(ROW_KEYS)` for a stub result `r`; `_error_row(2, 5, {"F": 3, "M": 1, "L": 1},
  RuntimeError("boom"))["status"] == "error: boom"`.

**A3. Feasibility, ranking, gradients, stop rule** (spec 1878-1894, 1945-1982; fixes S13, S14, S15, S19).
- `_is_feasible(row)`: `row.get("objective") is not None and row.get("status") not in NO_SOLUTION_STATUSES` (an
  error row has no objective). `_is_proven(row)`: `_is_feasible(row) and row.get("status") == "optimal"`.
- `_select_horizon_cases(rows) -> (best_proven, best_feasible)`: `min` over the feasible (proven) rows with the key
  `(row["J_op_average"], row["H2"])`; `None` if there is none.
- `_select_grid_cases(rows, metric) -> (best_proven, best_feasible)`: the same with the key `(row[metric], row["H1"],
  row["H2"])`.
- `_qualifies(row, maximum_mip_gap_for_stopping) -> bool`: `threshold = row.get("requested_mip_gap") if
  maximum_mip_gap_for_stopping is None else maximum_mip_gap_for_stopping`; true iff `gap = row.get("mip_gap")` and
  `threshold` are not `None`, `gap` is finite and `float(gap) <= float(threshold)`.
- `_annotate_gradients(rows, *, gradient_tolerance, maximum_mip_gap_for_stopping) -> None` (replaces
  `_annotate_latest_gradient`): `rows` is ordered by `H2`; for every index `n`, set the four keys to `None`, `None`,
  `"unavailable"`, `False`; if `n > 0` and rows `n-1` and `n` are feasible: `gradient = (J_n - J_{n-1}) / (H2_n -
  H2_{n-1})` with `J = J_op_average`; `relative = gradient / max(abs(J_{n-1}), np.finfo(float).eps)` (Eq.
  d_gradient); class `"increase"` if `relative > t`, `"decrease"` if `relative < -t`, else `"flat"`; eligible iff
  both rows `_qualifies`.
- `_gradient_stop_reason(rows, *, minimum_cases, flat_gradients_required)`: unchanged.
- Check: on the table T2 of `sweep_expected.py` the classes and the stop equal `sweep_expected_out.txt`.

**A4. Reports** (spec 1984-2007, 2077-2089; fixes S16, S17, S18, S20).
- `_formulation_growth(rows)`: `counted = [row for row in rows if row["formulation"] is not None]`; `None` if
  `len(counted) < 2`; else `from_H2`, `to_H2`, `added_time_steps = last["T"] - first["T"]` of the first and last
  counted rows, and for each key `q` of `FORMULATION_KEYS`: `added_q = stop - start` and `q_per_added_time_step =
  added_q / added_time_steps`, both `None` if either count is `None`.
- `_assemble_sweep_report(*, source, sizes, H1, settings, rows, last_completed_H2, stopping_reason, complete)`; keys in
  this order: `input = str(source)`, `complete`, `last_completed_H2`, `objective =
  SWEEP_OBJECTIVES[settings["objective_mode"]]`, `varied_parameter = "H2"`, `fixed_dimensions = dict(sizes)`, `H1`,
  `planned_operating_horizons = list(settings["candidates"])`, `evaluated_operating_horizons = [row["H2"] for row in
  rows]`, `warm_start = {"enabled": bool(warm_start), "mode": warm_start_mode if warm_start else None}`, `n_workers`,
  `stopping_rule = {"enabled", "relative_gradient_tolerance_per_H2", "flat_gradients_required", "minimum_cases",
  "maximum_mip_gap_for_stopping", "reason", "stopped_early"}`, `cases = rows`, `formulation_growth`,
  `best_proven_H2`, `best_feasible_H2`, `best_proven_J_op_average`, `best_feasible_J_op_average`,
  `best_feasible_status`. `reason = (stopping_reason or "candidate_range_exhausted") if complete else "in_progress"`;
  `stopped_early = complete and len(rows) < len(candidates)`.
- `_assemble_grid_report(*, source, sizes, h1_values, h2_values, subreports, warm_start, evaluation_horizon,
  complete)`: `cases` = the rows of the subreports in order (by `H1`, then `H2`); `metric =
  "projected_evaluation_cost"` if `evaluation_horizon` is given, else `"J_op_average"`; keys in this order: `input`,
  `complete`, `objective` (`"minimize J_initialization + (evaluation_horizon - H1)*J_op/H2 + C_D*u; rank by
  projected_evaluation_cost"` with `E`, else `SWEEP_OBJECTIVES["operating_average"]`), `comparison_metric = metric`,
  `mip_certificate_metric = "objective"`, `evaluation_horizon`, `varied_parameters = ["H1", "H2"]`,
  `fixed_dimensions = dict(sizes)`, `planned_transitory_horizons`, `planned_operating_horizons`,
  `warm_start_enabled`, `H1_sweeps = subreports`, `cases`, `best_proven`, `best_feasible`. A best case is `None` or
  `{"H1", "H2", "T", "status", "mip_gap", "objective"}` plus the `COST_KEYS` except `J_op` and `J_total`, in this order.
- `_write_sweep_report`: unchanged.

**A5. Argument checks** (spec 1911-1926, 1941; fixes S3 to S6).
`_check_sweep_arguments(H1, operating_horizons, *, output_path, stop_on_gradient, gradient_tolerance,
flat_gradients_required, minimum_cases, maximum_mip_gap_for_stopping, warm_start, warm_start_mode, n_workers,
objective_mode, evaluation_horizon) -> dict`. In this order, each a `ValueError` with this message:
1. `candidates = sorted(_positive_integers(operating_horizons, "operating_horizons"))`; a repeated value:
   `"operating_horizons must not contain duplicates."`
2. `gradient_tolerance` not a finite number >= 0 (a bool fails): `"gradient_tolerance must be a finite number >= 0."`
3. Not `_is_integer` or too small: `"flat_gradients_required must be an integer >= 1."`, `"minimum_cases must be an
   integer >= 2."`
4. `"maximum_mip_gap_for_stopping must be None or a finite number >= 0."`
5. `f"warm_start_mode must be one of {WARM_START_MODES} (got {warm_start_mode!r})."`
6. `"n_workers must be an integer >= 1."`; with `n_workers > 1`: `"warm_start=True needs n_workers=1: a warm start
   uses the preceding case."` and `"stop_on_gradient=True needs n_workers=1: the early stop needs the cases in order."`
7. `f"objective_mode must be one of {OBJECTIVE_MODES} (got {objective_mode!r})."`
8. `evaluation_horizon` given and not (`_is_integer` and `> H1`): `f"evaluation_horizon must be an integer > H1 =
   {H1} (got {evaluation_horizon!r})."`; mode `"evaluation_total"` without it: `"objective_mode 'evaluation_total'
   needs evaluation_horizon."`
9. A suffix other than `.yaml`/`.yml` (lower case): `"output_path must end in .yaml or .yml."`; then create the
   parent folders (`mkdir(parents=True, exist_ok=True)`).
Returns `{"candidates", "target" (Path or None), "stop_on_gradient": bool, "gradient_tolerance": float,
"flat_gradients_required": int, "minimum_cases": int, "maximum_mip_gap_for_stopping": float or None, "warm_start":
bool, "warm_start_mode", "n_workers": int, "objective_mode", "evaluation_horizon": int or None}`.
- Check: each message once, by a scratch call.

**A6. The case loop** `_sweep(data, source, sizes, H1, settings, on_checkpoint=None) -> dict` (spec 1928-1936,
2026-2040, 2179-2186; fixes S1, S7, S9, S10, S16).
- `case_input(H2)`: `deepcopy(data)` with `H = [H1, H2]`, `objective_mode = settings["objective_mode"]`, and
  `evaluation_horizon` if it is not `None`.
- After each case (`record`): append the row, sort `rows` by `H2`, set `last_completed_H2 = H2`, call
  `_annotate_gradients`; then the checkpoint: `report = _assemble_sweep_report(..., complete=False)`, written with
  `_write_sweep_report` if `settings["target"]` is set, and passed to `on_checkpoint(report)` if given.
- `n_workers == 1`: for `H2` in `candidates`: `warm = previous if settings["warm_start"] else None`; `try: row, found
  = _solve_case(case_input(H2), warm, settings["warm_start_mode"], settings["warm_start"])`; `except Exception as
  error: logger.error("horizon sweep case H1=%d, H2=%d failed: %s", H1, H2, error)`; `row, found =
  _error_row(H1, H2, sizes, error), None`. `if found is not None: previous = found`. Record; if
  `stop_on_gradient`, `stopping_reason = _gradient_stop_reason(...)`; checkpoint; stop the loop if a reason is set.
- `n_workers > 1`: `with ProcessPoolExecutor(max_workers=n_workers) as pool:` submit `_solve_case(case_input(H2),
  None, warm_start_mode, False)` for every `H2`; `for future in as_completed(futures)`: `row = future.result()[0]`,
  or on `Exception` the same log line and `_error_row`; record; checkpoint.
- End: `report = _assemble_sweep_report(..., complete=True)`; write it if `target` is set; return it.

**A7. Public functions** (spec 1901-1943, 2042-2075; fixes S2, S21, S22, S23).
- `sweep_operating_horizons` with the signature of spec 1902-1908 (defaults as there). Body: `source =
  Path(input_path)`; `data = read_input(source)` (it raises `TypeError` for data that are not a mapping);
  `H1 = _horizon_H1(data)`; `settings = _check_sweep_arguments(H1, operating_horizons, ...)`; `return _sweep(data,
  source, _input_sizes(data), H1, settings)`. Docstring: the six steps of spec 1929-1936 in one line each.
- `sweep_horizon_grid(input_path, transitory_horizons, operating_horizons, *, output_path=None, warm_start=True,
  evaluation_horizon=None, **sweep_options)`: read as above; `h1_values = sorted(set(_positive_integers(
  transitory_horizons, "transitory_horizons")))`, `h2_values` alike; `"objective_mode" in sweep_options`:
  `ValueError("sweep_horizon_grid sets objective_mode itself; remove it from sweep_options.")`; `evaluation_horizon`
  given and not (`_is_integer` and `> max(h1_values)`): `ValueError("evaluation_horizon must be an integer that
  exceeds every H1 candidate.")`; the `output_path` check of A5 item 9; `objective_mode = "evaluation_total"` if
  `evaluation_horizon` is given, else `"operating_average"`; `settings = {h1: _check_sweep_arguments(h1, h2_values,
  output_path=None, warm_start=warm_start, objective_mode=objective_mode, evaluation_horizon=evaluation_horizon,
  **sweep_options) for h1 in h1_values}` (all checks pass before the first case; an unknown option is the
  `TypeError` of Python). Then for each `h1`: `subreport = _sweep(data, source, sizes, h1, settings[h1],
  on_checkpoint=lambda partial: write(done + [partial], False))`; `done.append(subreport)`; `write(subreports,
  complete)` assembles the grid report and writes it if `output_path` is given. Final: `complete=True`, written,
  returned. No temporary file.
- Check: `grep -n "TemporaryDirectory\|safe_load\|gamma" horizon_sweep.py` finds nothing.

**A8. Tests**: write `tests/test_horizon_sweep.py` as section 3.1; run `python SP\run_tests_wp7.py
tests\test_horizon_sweep.py`: every test passes (none needs Gurobi).

### Task B: `greedy_warm_start.py`, the warm starts of `base.py`, `tests/test_warm_start.py` (6 steps)

`greedy_warm_start.py` is rewritten. The logic of each function equals the function of `SP\greedy_ref.py` named in
brackets (the planner's reference, checked by `verify_greedy_lp.py`); port it in the style of `base.py`, with the
names, docstrings and order below. In `base.py`, touch only `apply_binary_warm_start`, `complete_binary_warm_start`,
`apply_relaxation_warm_start`, the new `warm_start_day` above them, and `solve_mixed` lines 1257-1291.

**B1. Greedy: header, report, construction pass** (spec 2188-2225, 2243-2253; fixes G1, G2, G3).
- Docstring: the heuristic builds a source from the input (spec "Greedy Heuristic"); a construction pass assigns the
  missions and the repairs with a proxy state; a loop pass adds repairs or replacements in the operating phase until
  the reference trajectory of each cell (the rows with every inequality replaced by an equality) meets its loop rows
  and its reliability row; numpy and scipy only, the caps imported inside the functions.
- Imports: `math`, `dataclass`, `numpy as np`, `from scipy.special import ndtri`, `from fleet_management.config import
  _lookup`. Constants `EXCESS_TOLERANCE = 1e-10`, `COMPLETION_TIME_LIMIT = 60.0`.
- `GreedyWarmStartReport` (frozen dataclass), fields in this order: `repairs: int`, `replacements: int`,
  `repeatability_feasible: bool`, `repeatability_failures: int`, `maximum_repeatability_excess: float`,
  `maximum_proxy_ratio: float`, `assignment_order: list[list[int]]`; `as_dict()` returns them in this order as plain
  `int`, `bool`, `float` and lists of `int`.
- `_profiles(cfg) -> dict`: for `key` in `("mu", "v", "support", "cgf", "gamma_beta")`: `_lookup(getattr(cfg, key),
  getattr(cfg, key + "_trans"), cfg.H1, cfg.H2)` (shape `(F, L, M, T)`), or `None` if `getattr(cfg, key)` is `None`.
- `_construction_pass(cfg, mu_inc, repair_trigger) -> (x, m, r, assignment_order, maximum_ratio)` [the first loop
  of `greedy`]: `x` `(F, M+1, T)`, `m`, `r` `(F, L, T)` float zeros; `s_hat = cfg.mu_0` copy. Day `k`: missions by
  decreasing `max_{i,l} mu_inc[i, l, j, k] / tau[i, l]`, tie to the larger `j` (`sorted(..., reverse=True)` on
  `(value, j)`); append `[j + 1 ...]` to the order; each mission to the free train with the smallest `(max_l ratio,
  sum_l ratio, i)`, `ratio = (s_hat[i] + mu_inc[i, :, j, k]) / tau[i]`; `x[i, j + 1, k] = 1`, `s_hat[i] +=
  mu_inc[i, :, j, k]`. Then each free train `i` in increasing order: `due = s_hat[i] > repair_trigger * tau[i]`; if any,
  `x[i, 0, k] = 1`, `m[i, due, k] = 1`, `s_hat[i, due] *= 1 - rho[i, due]`; otherwise the train stays idle (all
  `x[i, :, k] = 0`). At the end of the day `maximum_ratio = max(maximum_ratio, max(s_hat / tau))`.
- Check: case A of section 3.2 gives the activities of the construction pass `[1, -1, 1, -1]` and `[-1, 1, -1, 1]`.

**B2. Greedy: quantities and rows of a cell** (spec 2230-2236 and the model sections; fixes G4, G5, G7).
- `_cell_quantities(cfg, i, l, profiles) -> list[dict]` [`quantities`]; one dict per tracked quantity, keys `q`,
  `start`, `new`, `anchored`, `anchor`, `factor`, `increment` (array `(M, T)`), `range`; `ard1 = repair_model ==
  "ard1"`, `rho = cfg.rho[i, l]`:

| Cell | `q` | `start` | `new` | `anchored` | `anchor` | `factor` | `increment` | `range` |
|---|---|---|---|---|---|---|---|---|
| every | `mu` | `mu_0` | `replacement_mu` | `ard1` | `anchor_mu_0` | `1 - rho` | `profiles["mu"][i, l]` | `range_mu` |
| G; RF Cantelli, Bernstein | `v` | `v_0` | `replacement_v` | `ard1` | `anchor_v_0` | `(1 - rho)**2` | `profiles["v"][i, l]` | `range_v` |
| W | `v` | `v_0` | `replacement_v` | `False` | NaN | `1.0` | `sigma[i, l]**2` everywhere | `range_v` |
| RF Hoeffding | `R` | `R_0` | `R_new` | `ard1` | `anchor_R_0` | `(1 - rho)**2` | `profiles["support"][i, l]**2` | `range_R` |
| RF Chernoff | `K` | `cgf_0` | `cgf_new` | `ard1` | `anchor_K_0` | `1 - rho` | `profiles["cgf"][i, l]` | `range_K` |

  (All values are the `cfg` fields at `[i, l]`, as `float`.)
- `_cell_rows(cfg, i, l, profiles) -> dict` [`rows`]: Gaussian, Wiener: `{"kind": "quadratic", "q": "v", "c2": 1 /
  ndtri(1 - eps)**2, "c1": 0.0, "side": tau}`; inverse Gaussian: `{"kind": "cap", "q": "mu", "cap":
  ig_mean_cap(eta, tau, eps)}`; Gamma: `{"kind": "cap", "q": "mu", "cap": mean_cap(smallest_rate(profiles["mu"][i,
  l], profiles["gamma_beta"][i, l], mu_0, gamma_beta_0, replacement_mu, gamma_beta_new, bool(cfg.allow_replacement)),
  tau, eps)}` (as the builder, `gamma.py:182-188`); rainflow: `c = row_constants(bound, tau, eps,
  cfg.support_const[i, l], cfg.s_chernoff[i, l])`, then `{"kind": "cap", "q": c["q"], "cap": c["cap"]}` if `"cap" in
  c`, else `{"kind": "quadratic", "q": c["q"], "c2", "c1", "side": c["mu_side"]}`. Every dict also gets `"form":
  cfg.reliability_impl[i, l]`, `"tangent_ref": cfg.tangent_ref[i, l]` and `"v_max_user": cfg.v_max_user[i, l]` if
  `v` is tracked, else NaN (Q3). Imports inside the function: `inverse_gaussian.ig_mean_cap`, `gamma.mean_cap`,
  `gamma.smallest_rate`, `rainflow.row_constants` (all load without gurobipy).
- `_row_cap(mu, tau, c2, c1, form, tangent_ref, P) -> np.ndarray` [`cap_of`], over the days: `f(m) = c2 (tau - m)^2
  + c1 (tau - m)`, `f'(m) = -2 c2 (tau - m) - c1`; `"exact"`: `f(mu)`; `"tangent"`: `f(mr) + f'(mr) (mu - mr)`,
  `mr = tangent_ref * tau`; `"pwl"`: with `mc = clip(mu, 0, tau)`, the largest `f(h_p) + f'(h_p) (mu - h_p)`,
  `h_p = (p - 0.5) tau / P`, over the `p = 1..P` with `(p - 1) tau / P <= mc <= p tau / P` (P = 1 is the tangent at
  `tau / 2`, as `base.py:856-857`). The clip keeps the cap finite above `tau`; the side row then counts the excess.
- Check: case G's rows hold the caps of `ig_mean_cap(0.25, 1, 0.5)`, `mean_cap(8.0, 1, 0.5)`, `0.5` (Markov) and
  `4 + ln 0.5` (Chernoff).

**B3. Greedy: trajectory, excess, loop pass, builder** (spec 2226-2241; fixes G6, G8, G9, G10, G11).
- `_reference_trajectory(cfg, x, m, r, i, l, quantities) -> (states, anchors)` [`trajectory`]: dicts `q -> array
  (T,)`, anchors only for anchored `q`; state and anchor start at `start` and `anchor`. Day `k`: a replacement
  (`r > 0.5`) sets `s = new` and `g = s`; else a repair (`m > 0.5`) sets `s = factor*s + (1 - factor)*g` if anchored,
  `factor*s` otherwise, and `g = s`; else `s += increment[j - 1, k]` for the mission `j >= 1` flown that day (0 if
  none).
- `_cell_excess(cfg, x, m, r, i, l, quantities, rows) -> (e, raw, mean_ratio)` [`excess`]: terms `(value, range)`:
  `max(0, s[T-1] - s[H1-1])` for every state and every anchor (range of its quantity); a cap row: `max(0, max_k(s_q -
  cap))` with the range of `q`; a quadratic row: `max(0, max_k(chi_k - _row_cap(mu_k)))` with the range of `chi`, and
  the side row `max(0, max_k(mu_k - side))` with `range_mu`; `v_max_user` if finite: `max(0, max_k(v_k -
  v_max_user))` with `range_v`. `e = max(value / range)`, `raw = max(value)` (both 0.0 without terms);
  `mean_ratio = max_k mu_k / tau`.
- `_best_trial(cfg, x, m, r, i, l, quantities, rows, targets) -> (e, k, target) | None`: over the free days `k` in
  `range(H1, T)` (no mission of train `i`, `m[i, l, k] == r[i, l, k] == 0`) and the arrays in `targets`: set
  `target[i, l, k] = 1` and `x[i, 0, k] = 1`, compute `e`, restore both; keep the smallest key `(e, position of target
  in targets, -k)`.
- `_loop_pass(cfg, x, m, r, profiles) -> (failures, maximum_excess, maximum_ratio)` [second loop of `greedy` with
  `joint_models=("wiener",)`]: cells `i` outer, `l` inner; a trial "gains" if it is not `None` and its `e` is below
  `e - EXCESS_TOLERANCE`. While `e > EXCESS_TOLERANCE`:
  - Wiener cell (`str(cfg.model[i, l]) == "wiener"`; user's rule, decision 2 of `merge/wp7_interface.md`): `trial =
    _best_trial(..., (m, r) if cfg.allow_replacement else (m,))`, one trial over repairs and replacements on every
    free day (smallest `e`, tie to the repair, then to the later day).
  - Every other cell (spec order): `trial = _best_trial(..., (m,))`; if it does not gain and
    `cfg.allow_replacement`: `trial = _best_trial(..., (r,))`.
  - If the trial gains, apply it (`target[i, l, k] = 1`, `x[i, 0, k] = 1`, `e = trial e`); else `failures += 1` and go
    to the next cell.
  After the cell, `raw` and `mean_ratio` of its final trajectory update the two maxima. Docstring: "At a Wiener cell
  repairs and replacements are tried together (its variance falls only at a replacement)."
- `build_greedy_warm_start(cfg, *, repair_trigger=0.70)`: `not 0.0 <= repair_trigger <= 1.0`:
  `ValueError("repair_trigger must lie in [0, 1]")`; `profiles`, construction pass, loop pass; source `{"H1": cfg.H1,
  "H2": cfg.H2, "x": x, "m": m, "r": r, "complete_binary_start": True, "completion_time_limit":
  COMPLETION_TIME_LIMIT}`; report with `repairs = int(m.sum())`, `replacements = int(r.sum())`,
  `maximum_proxy_ratio = max(construction, loop)`.
- Check: the seven cases of section 3.2 give the expected values; the module runs with gurobipy blocked.

**B4. `base.py`: day mapping and mapped values** (spec 2106-2147; fixes W2 to W6).
- New `warm_start_day(k, H1, source_H1, source_H2) -> int` (Eq. d_daymap, 0-based): `k` if `k < min(H1,
  source_H1)`, else `source_H1 + max(0, k - H1) % source_H2`.
- `apply_binary_warm_start(ctx, source, mode="start") -> dict`. Without a source or with `x` `None`: return
  `{"applied": False, "values": 0, "mode": None, "source_horizon": None}`. `H1s = int(source["H1"])`, `H2s` alike;
  on `KeyError`, `TypeError`, `ValueError`, or a value below 1: `ValueError("a warm-start source must carry the
  integers 'H1' and 'H2' (both >= 1).")`. `x_src = np.asarray(source["x"], float)`; if `x_src.ndim != 3` or
  `x_src.shape[2] != H1s + H2s`: `ValueError(f"'x' of a warm-start source must have the shape (F', M'+1, {H1s +
  H2s}); got {x_src.shape}.")`. `set_value(var, value)` sets `var.Start` if `mode in ("start", "both")` and
  `var.VarHintVal` if `mode in ("hint", "both")`. For every `k`, `sk = warm_start_day(...)`, every `i <
  min(F, F')`: every `j < min(M + 1, M' + 1)`: `x`; every `l < L`: `mv` = `0.0` if the source has no `m`, the rounded
  `m_src[i, l, sk]` if `i, l` are inside `m_src`, else `None`; `rv` = `0.0` if `ctx.r_rep is None`, the rounded
  `r_src[i, l, sk]` if the source has `r` and `i, l` are inside it, else `None`; set `mv` on `ctx.m_rep` and `rv` on
  `ctx.r_rep` where they are values and the variable exists; selector `(i, l, k) in ctx.nb`: `max(0.0, 1 - mv - rv)`
  if both are values, `0.0` if one of them is `1.0`, else no value. `values` counts the variables that got a value
  (once, also in `"both"`; Q5). Return `{"applied": values > 0, "values": values, "mode": mode, "source_horizon":
  [H1s, H2s]}` (key order of spec 1820-1824; `completion` follows).
- Check: the values of section 3.2, test `apply_modes`, with `run_tests_wp7.py --fake`.

**B5. `base.py`: completion, relaxation, `solve_mixed`** (spec 2140-2177; fixes W1, W7, W8).
- `complete_binary_warm_start(model, source, report, mode="start")`: unchanged, except that the copy runs only if
  `requested and mode in ("start", "both") and report["applied"]`; `requested` stays the source's flag, and the status
  stays `"not_requested"` when the copy does not run (Q2). Docstring cites spec "Start, Hints and Completion".
- `apply_relaxation_warm_start`: `relaxed.dispose()` before each return; docstring: "its integer values become
  `Start` values".
- `solve_mixed`: delete lines 1261-1265 (`has_source` and the `NotImplementedError`) and line 1276; call
  `apply_binary_warm_start(ctx, warm_start, warm_start_mode)` and `complete_binary_warm_start(ctx.model, warm_start,
  warm_start_diagnostics, warm_start_mode)`. The `ValueError` for an unknown mode stays.
- Check: `grep -n "NotImplementedError" base.py` finds only `get_cell_builder`; the undefined-name tool
  `python SP\..\..\wp6\rainflow\names_check_try.py <file>` reports 0 names for `base.py`,
  `greedy_warm_start.py` (and, in Task A, `horizon_sweep.py`).

**B6. Tests**: write `tests/test_warm_start.py` as section 3.2; run it twice, `python SP\run_tests_wp7.py
tests\test_warm_start.py` (greedy tests pass, model tests skip) and with `--fake` (model tests pass except those
marked "Gurobi only").

## 3. Tests

Pytest style; a test that needs Gurobi calls `_base()` (`pytest.importorskip("gurobipy")`, then imports `base`), as
`tests/test_gamma.py:171-174`. Expected values come from `SP\sweep_expected_out.txt` and `SP\greedy_ref_out.txt`.

### 3.1 `tests/test_horizon_sweep.py` (no Gurobi; Task A)

**Helpers.**
- `_write_input(tmp_path, suffix=".yaml", **changes) -> Path`: `{"F": 3, "M": 1, "H": [2, 9], "model": "gaussian",
  "tau": 1.0, "epsilon": 0.1, "rho": 0.5, "mu_0": 0.1, "mu": 0.2, "v": 0.001, "C_M": 1.0, "C_R": 1.0, "C_rep": 5.0,
  "C_D": 0.0}` updated by `changes`, written as YAML or JSON. No `L` key (checks S8).
- `_Stub(table, j_init=None, delay=None, report_path=None)` replaces `solve_mapping`
  (`monkeypatch.setattr(horizon_sweep, "solve_mapping", stub)`). A call `(data, results_path=None, *, warm_start,
  warm_start_mode)` asserts `results_path is None`; appends to `self.calls` `H1`, `H2` (from `data["H"]`), `source_H2`
  (`warm_start["H2"]` or `None`), `mode`, `objective_mode`, `evaluation_horizon` and, with `report_path`, the file's
  `(complete, stopping_rule.reason, evaluated_operating_horizons)` or `None`; sleeps `delay.get(H2, 0)`; the entry
  `"error"` raises `RuntimeError(f"boom {H2}")`. An entry `(status, J, gap, requested)` (key `H2`, or `(H1, H2)` in
  the grid) gives, with `T = H1 + H2`, `Ji = j_init.get(H1, 2.0)`, `E = data.get("evaluation_horizon")`:
  `J_initialization = Ji`, `J_op = J*H2`, `J_op_average = operating_objective = J`, `peak_damage = 0.5`,
  `damage_penalty = 0.0`, `J_total = Ji + J*H2`, `projected_evaluation_cost = evaluation_objective = Ji + (E - H1)*J`
  (`None` without `E`); `objective` = the value of the mode, `bound = objective*(1 - gap)`; `x = zeros((3, 2, T))`,
  `m = r = zeros((3, 1, T))`; all these `None` if `J is None`. Also `F, M, L = 3, 1, 1`, `H1`, `H2`, `T`, `status`,
  `mip_gap = gap`, `objective_mode`, `evaluation_horizon = E`, `warm_start = {"applied": warm_start is not None}`,
  and `performance`: `variables 10T, continuous_variables 4T, integer_variables 6T, binary_variables 6T,
  linear_constraints 20T, general_, indicator_, quadratic_constraints 0, nonzeros 50T, requested_mip_gap,
  requested_time_limit_seconds 60.0, optimizer_call_seconds 0.5, solutions_found 1 (0 without J), every other timing
  key of `TIMING_KEYS` 0.1`.
- Tables (`SP\sweep_expected.py`): T1 = `{2: ("optimal", 1.0, 0.0, 0.1), 3: ("optimal", 0.9, 0.0, 0.1), 4:
  ("time_limit", 0.8, 0.05, 0.1), 5: "error", 6: ("infeasible", None, None, 0.1), 7: ("optimal", 0.9, 0.0, 0.1)}`;
  T2 = `{2: (opt, 1.0, 0, .1), 3: (opt, 0.9, 0, .1), 5: (opt, 0.85, 0, .1), 6: (opt, 0.95, 0.08, .1), 8: (opt, 0.7, 0,
  .1)}`; T3 = `{2: 1.0, 3: 1.0005, 4: 1.0004, 5: 1.0, 6: 0.5}`, all `("optimal", J, 0.0, 0.1)`.

**Tests** (expected values in brackets).
1. `test_rows_ranking_and_errors` (T1, `operating_horizons=[7, 2, 3, 4, 5, 6]`, `caplog`): evaluated `[2..7]`;
   classes `[unavailable, decrease, decrease, unavailable x3]`; relative gradients `-0.1`, `-0.11111111111111108`;
   eligible `[F, T, T, F, F, F]`; best proven `H2 3` (tie with 7), `0.9`; best feasible `H2 4`, `0.8`,
   `"time_limit"`; row 4 `absolute_mip_gap == approx(0.04)`; row 5: `"error: boom 5"`, `objective`, `formulation`,
   `timing`, `warm_start` `None`, `dimensions {F 3, M 1, L 1, H1 2, H2 5, T 7}`; one ERROR record, with `H1=2` and
   `H2=5`; reason `"candidate_range_exhausted"`; `list(row) == list(ROW_KEYS)`; the 19 report keys in the order of
   A4; growth `from_H2 2, to_H2 7, added_time_steps 5, added_variables 50, variables_per_added_time_step 10.0`.
2. `test_warm_start_sources` (T1, `warm_start=True`, `warm_start_mode="both"`): `source_H2` per call `[None, 2, 3, 4,
   4, 4]`; every `mode == "both"`; report `warm_start == {"enabled": True, "mode": "both"}`.
3. `test_stop_cost_increase` (T2, `stop_on_gradient=True`): evaluated `[2, 3, 5, 6]`; no call with `H2 = 8`; reason
   `"cost_increase"`; `stopped_early` true; relative gradients `-0.1`, `-0.0277777777777778`, `0.11764705882352938`.
4. `test_stop_needs_qualified_gaps` (T2, `stop_on_gradient=True, maximum_mip_gap_for_stopping=0.05`): row 6 not
   eligible; evaluated `[2, 3, 5, 6, 8]`; reason `"candidate_range_exhausted"`.
5. `test_stop_flat_gradient`, parametrized `(minimum_cases, flat_gradients_required, evaluated)` in `(3, 2, [2, 3,
   4])`, `(4, 2, [2, 3, 4, 5])`, `(3, 3, [2, 3, 4, 5])` (T3): reason `"flat_gradient"`.
6. `test_checkpoints` (T1, `output_path = tmp_path / "out" / "sweep.yaml"`, a missing folder): first call
   `checkpoint is None`; call `n >= 2`: `(False, "in_progress", candidates[:n-1])`; the final file loads with
   `yaml.safe_load` to the returned report, `complete` true.
7. `test_argument_errors`, parametrized `(kwargs, message of A5)`, stub never called: `operating_horizons` `[]`,
   `[2, 2]`, `[0]`, `[2.5]`, `[True]`; `gradient_tolerance` `-1`, `inf`; `flat_gradients_required=0`;
   `minimum_cases=1`; `maximum_mip_gap_for_stopping=-0.1`; `warm_start_mode="hints"`; `n_workers=0`; `n_workers=2`
   with `warm_start=True` and with `stop_on_gradient=True`; `objective_mode` `"average"` and `"evaluation_total"` (no
   horizon); `evaluation_horizon=2`; `output_path="r.json"`. Plus `H: [1, 2, 3]` (`ValueError`, "a pair") and a YAML
   list (`TypeError`).
8. `test_case_input_and_formats` (JSON input, `objective_mode="evaluation_total", evaluation_horizon=20`): each call
   sees `H == [2, H2]`, the mode and `20`; `fixed_dimensions == {"F": 3, "M": 1, "L": 1}`.
9. `test_parallel_cases` (T1 restricted to `[2, 3, 4]`, `n_workers=3`, `delay={2: 0.3}`, `monkeypatch.setattr(
   horizon_sweep, "ProcessPoolExecutor", ThreadPoolExecutor)`, and `_write_sweep_report` wrapped to count writes,
   `output_path` set): rows ordered `[2, 3, 4]`; annotations equal to test 1 for these rows; `last_completed_H2 ==
   2`; `n_workers == 3`; 4 writes (3 checkpoints, 1 final).
10. `test_grid`, parametrized `E` in `(None, 10)`: `transitory_horizons=[3, 2, 2]`, `operating_horizons=[4, 2]`,
    table `{(2, 2): (opt, 1.0, 0, .1), (2, 4): (opt, 0.9, 0, .1), (3, 2): ("time_limit", 0.85, 0.05, .1), (3, 4):
    (opt, 0.9, 0, .1)}`, `j_init={2: 5.0, 3: 4.0}`: planned `[2, 3]`, `[2, 4]`; cases `(2,2), (2,4), (3,2), (3,4)`;
    `source_H2` `[None, 2, None, 2]`; mode `"evaluation_total"` with `E`, else `"operating_average"`; `E = None`:
    metric `"J_op_average"`, best proven `(2, 4)` (tie with (3, 4)), best feasible `(3, 2)`; `E = 10`:
    `"projected_evaluation_cost"`, `(3, 4)` (10.3), `(3, 2)` (9.95); keys of a best case as A4; `H1_sweeps[k]["input"]
    == str(input_path)`.
11. `test_grid_checkpoints_and_errors`: with `output_path`, 5 writes (4 cases, 1 final) and every checkpoint has
    `complete` false; `ValueError` for `objective_mode` in the options, for `evaluation_horizon=3` with
    `transitory_horizons=[2, 3]`, and for `n_workers=2` with the default `warm_start=True`.

### 3.2 `tests/test_warm_start.py` (Task B)

**Greedy heuristic (no Gurobi).** `_CASES`: the dicts `A_gaussian`, `F_gaussian`, `G_linear`, `B_wiener`,
`H_wiener_gaussian`, `C_wiener_norepl`, `D_mixed_ard1` of `SP\greedy_ref.py`, copied literally, built with
`load_config(dict(case, verbose=0))`. `_activities(x)`: per train and day, the mission `j`, `0` the depot, `-1` idle.
Expected values: `greedy(cfg, joint_models=("wiener",))`, the rule of B3 (output lines "rule" of
`SP\greedy_ref_out.txt`).

| Case | Activities per train | Interventions (others zero) | Report: repairs, replacements, failures, max excess, max ratio |
|---|---|---|---|
| A | `[1,-1,1,0]`, `[-1,1,0,1]` | m(0,0) day 4; m(1,0) day 3 | 2, 0, 1, 0.0625, 0.625 |
| F | `[1,-1,0,1]`, `[-1,1,-1,-1]`, `[-1,-1,1,0]` | m(0,0) day 3; m(2,0) day 4 | 2, 0, 0, 0.0, 0.75 |
| G | as F | m(0,l) day 3 and m(2,l) day 4, l = 0..3 | 8, 0, 0, 0.0, 0.3125 |
| B (Wiener) | as A | r(0,0) day 4; r(1,0) day 3 | 0, 2, 0, 0.0, 0.625 |
| H (Wiener, Gaussian) | as A | r(0,0), m(0,1) day 4; r(1,0), m(1,1) day 3 | 2, 2, 1, 0.0625, 0.625 |
| C (Wiener, no replacement) | as A | m(0,0) day 4; m(1,0) day 3; `r` all zero | 2, 0, 2, 0.0625, 0.625 |
| D | `[2,-1,2,0,2]`, `[1,1,0,2,1]`, `[-1,2,1,1,0]` | m(0,l) day 4, l = 0..2; m(1,l) day 3; m(2,2) day 5; r(2,0), r(2,1) day 5 | 7, 2, 8, 0.78125, 0.8125 |

Days are 1-based. `assignment_order` is `[[1]] * T`, and `[[2, 1]] * 5` for D. B and H are the cases that the rule
changes: with the spec order at every cell, B would have the repairs of A and 2 failures, H the repairs of A at both
components and 3 failures. In H the Gaussian cell (1,1) keeps the spec order and fails as in A. The schedules without
failures (F, G, B) are feasible for the real rows (`verify_greedy_lp.py --wiener`).
1. `test_greedy_cases`, parametrized over the seven cases: activities, `m`, `r` and the report of the table
   (`approx` for floats).
2. `test_repair_trigger_range`: `-0.1`, `1.1`, `nan` raise `ValueError` ("must lie in").
3. `test_source_and_report_keys` (case F): `list(source) == ["H1", "H2", "x", "m", "r", "complete_binary_start",
   "completion_time_limit"]`, `completion_time_limit == 60.0`, shapes `(3, 2, 4)`, `(3, 1, 4)`;
   `list(report.as_dict())` is the seven keys of B1; `yaml.safe_dump(report.as_dict())` works.
4. `test_greedy_needs_no_gurobi`: as `tests/test_gamma.py:125-141` (a subprocess with `sys.modules['gurobipy'] =
   None` and `PYTHONPATH` from the package path): build case G with `build_greedy_warm_start`; return code 0.

**Warm starts (`_base()` per test; run locally with `--fake`).** `_model_input(**changes)`: `{"F": 3, "M": 1, "H":
[2, 3], "model": "inverse_gaussian", "encoding": "indicator", "tau": 1.0, "epsilon": 0.5, "rho": 1.0, "mu_0": 0.0625,
"eta": 0.25, "mu": 0.125, "C_M": 1.0, "C_R": 1.0, "C_rep": 5.0, "C_D": 0.0, "verbose": 0}`. `_SOURCE`: `H1 = 1`, `H2 =
3`; `x` `(2, 2, 4)` with `x[0, 1] = [1,0,1,0]`, `x[0, 0] = [0,1,0,0]`, `x[1, 1] = [0,1,0,1]`, `x[1, 0] = [0,0,1,0]`;
`m` `(2, 1, 4)` with `m[0, 0] = [0,1,0,0]`, `m[1, 0] = [0,0,1,0]`; `r` `(2, 1, 4)` with `r[0, 0] = [0,0,0,1]`.
5. `test_warm_start_day`: `[warm_start_day(k, 2, 1, 3) for k in range(5)] == [0, 1, 1, 2, 3]`; `[warm_start_day(k, 1,
   3, 2) for k in range(4)] == [0, 3, 4, 3]`.
6. `test_apply_modes`, parametrized `mode`: after `build_fleet`, `update`, `apply_binary_warm_start(ctx, _SOURCE,
   mode)`, `update`: expected `x[0,0] = [0,1,1,0,0]`, `x[0,1] = [1,0,0,1,0]`, `x[1,0] = [0,0,0,1,0]`, `x[1,1] =
   [0,1,1,0,1]`, `m[0,0] = [0,1,1,0,0]`, `m[1,0] = [0,0,0,1,0]`, `r[0,0] = [0,0,0,0,1]`, `r[1,0] = 0`, `n[0,0] =
   [1,0,0,1,0]`, `n[1,0] = [1,1,1,0,1]`; train 2 has no value (`>= 1e100`); `Start` holds them for `start`, `both`,
   `VarHintVal` for `hint`, `both`, the other attribute none; report `{"applied": True, "values": 50, "mode": mode,
   "source_horizon": [1, 3]}`.
7. `test_apply_partial_sources`: no `m`, no `r`: `values == 30`, `m` 0 for trains 0 and 1, no `r`, no `n`;
   `allow_replacement=False` and no `r`: `values == 40`, `n[0,0] = [1,0,0,1,1]`; `x` `None`: the empty report of B4;
   no `H1`: `ValueError` ("must carry"); `x` of shape `(2, 2, 5)`: `ValueError` ("must have the shape").
8. `test_completion`, parametrized `mode` in `start`, `both` (case G, greedy source, `solve_mixed`): completion
   `requested` true, `"completed"`, `fixed_integer_variables == 120`, `values_copied == performance["variables"]`;
   `warm_start["values"] == 120`, `source_horizon == [2, 2]`; status `"optimal"`.
9. `test_hint_skips_completion` (case G; `apply` and `complete` called directly with `"hint"`): completion
   `requested` true, status `"not_requested"`, `fixed_integer_variables == 0`.
10. `test_failed_completion` (case C with `gurobi_params={"DualReductions": 0}`; `apply` and `complete` directly):
    status `"infeasible"`; `0 < len(iis_constraints) <= 50`, all strings.
11. Gurobi only (the emulator cannot search over binaries): `test_solve_mixed_modes` (case G, each mode: `mode`
    reported, `relaxation_warm_start` `"skipped"` with `relaxation_warm_start=True`); `test_relaxation_start` (case
    G, no source, `relaxation_warm_start=True`: `enabled` true, status not `"disabled"` or `"skipped"`,
    `runtime_seconds` a float).

## 4. Exports

- From `fleet_management.horizon_sweep`: `sweep_operating_horizons` and `sweep_horizon_grid`, with the signatures of
  spec 1902-1908 and 2048-2051. Their import is light: `horizon_sweep` loads `config` and `solver` only; `base` and
  Gurobi load inside `solve_mapping`.
- Nothing else. `build_greedy_warm_start` and `GreedyWarmStartReport` stay internal (spec 233); the warm-start
  functions of `base.py` are not part of the API.

## 5. Callers outside src

Broken by these changes (WP8 fixes them; `remote_smoke_test.py` uses none):
- `examples/regression/check_horizon_sweep.py`: imports `_annotate_latest_gradient` (12, removed); passes
  `maximum_mip_gap=` (57, 75); reads `best_H2` (143, 163, 167), `row["calibration"]` (185-188); input key
  `gamma_calibration_method` (99), rejected since WP4.
- `examples/regression/run_horizon_sweep.py:112-127`: formats `None` values of an error row with `:>11.3f`
  (`TypeError`); `euler/gamma_horizon_convergence.sbatch` and the `experiments/gamma_*` jobs run it.
- `examples/regression/check_greedy_warm_start.py`: imports `_simulate_cell` (9, removed); reads `start["idle"]` (30),
  `feasible_assignment`, `artificial_assignments` (32-33, 49-56); expects a fallback for `F = 1, M = 2` (45-50; now
  refused by `F > M`); input key `gamma_beta_bound` (22), rejected.
- `examples/regression/check_greedy_warm_start_acceptance.py:51` (`feasible_assignment`; input key
  `gamma_calibration_method`, 37); `experiments/garage_hardau_strict_greedy/run_case.py:103`;
  `experiments/vbz_greedy_warm_start/run_comparison.py:76` and `vbz_greedy_warm_start.sbatch:70`.
- Part `plots`, not this one: `check_horizon_sweep_visualization.py:16`, `gamma_horizon_stabilization/plot_result.py`.

## 6. Removals

- No file. Functions: `horizon_sweep._gamma_calibration_summary`; `_is_feasible_case`,
  `_gap_qualifies_for_stopping`, `_annotate_latest_gradient` (renamed, A3); in `greedy_warm_start.py`
  `_mean_increment`, `_variance_increment`, `_uses_variance_repeatability`, `_gamma_shape_data`, `_mission_index`,
  `_simulate_cell`, `_repeatability_score`, `_enforce_repeatability` (replaced, B1-B3). Keys: `feasible_assignment`,
  `artificial_assignments`, `idle`; `calibration`, `time_limit_seconds`, `interpretation`, `selection_interpretation`,
  `best_H2`, `best_J_op_average`, `method`.

## 7. Questions

All questions are decided (`merge/wp7_interface.md`, "Decisions on the plans' questions", 29 Oct 2026): Q1 by the
user's rule (decision 2), Q6 accepted (decision 5), the others as recommended (decision 8). The steps follow them.
- **Q1. Order of the trials in the loop pass** (spec 2237-2239; step B3). The spec tries repairs first and a
  replacement only when no repair lowers `e`. A repair then takes the last free day, and a cell that needs a
  replacement fails. Example, case B (Wiener, `H = [2, 2]`): a repair on day 4 removes the mean excess and uses the
  only free day; the variance excess stays, 1 failure per train. A replacement on that day would remove both.
  Options were (a) the spec as written and (b) one trial over repairs and replacements at every cell. **Decided (user,
  29 Oct 2026): the spec order at every cell except Wiener; at a Wiener cell, one trial over repairs and
  replacements** (step B3; spec sentence in 7.1).
- **Q2. Completion report in the mode `"hint"`** (spec 1831-1832, 2145; B5). The completion never runs, but the
  greedy source requests it, and no listed status fits. Options: (a) `requested` = the source's flag, status
  `"not_requested"` whenever the copy does not run (as today for a source that sets no value); (b) `requested`
  false; (c) a new status `"skipped"`. **Recommendation: (a)**, and the spec sentence "the status is `not_requested`
  whenever the completion does not run".
- **Q3. `v_max_user` in the excess** (spec 2234; B2). The spec counts the reliability row only; a start that breaks
  `v <= v_max_user` can never be completed. **Recommendation: count it, divided by `range_v`**, and say so in spec 2234.
- **Q4. An invalid source** (no `H1` or `H2`; `x` whose last axis is not `H1' + H2'`; B4). The spec gives the keys
  (2114), not the error; today the code takes the model's values. **Recommendation: `ValueError`** (messages of B4),
  and one spec sentence.
- **Q5. `values` in the mode `"both"`** (spec 1821). **Recommendation: the number of variables that got a value.**
- **Q6. The reference trajectory** (spec 2230 cites the section "Encoding of the Dynamics", which does not define
  it; scoping 387 does). **Recommendation:** add "(the rows with every inequality replaced by an equality)". For the
  redesign item of `Things to change.txt`: this test is stricter than the one-sided rows; case A fails it, yet its
  schedule is feasible for the real rows (`verify_greedy_lp.py`).
- **Q7. `maximum_proxy_ratio`** (spec 2251; B3): every trajectory computed, or the final one? **Recommendation: the
  final trajectory of each cell** (the returned schedule).
- **Q8. `last_completed_H2` with parallel cases** (spec 1993; A6). **Recommendation: the `H2` of the case that
  finished last.**
- **Q9. Private helpers** (A1, B1): `config._lookup`, `_integer`, `_infer_L`, `solver._to_plain`, so that each rule
  has one implementation (the contracts forbid changing `config.py`, `solver.py`). **Recommendation: import them as
  they are**; making them public is a later clean-up.

### 7.1 Spec sentence for the Wiener exception (decision 2)

Worktree spec, section "Greedy Heuristic", block "Steps of the loop pass": insert this paragraph between
`\end{enumerate}` (line 2240) and "A start with failures is returned." (line 2241):

```latex
At a Wiener cell, steps 3 and 4 form one trial: on every free day, put a repair and, if replacement is allowed, a replacement, and recompute $e$ after each; keep the intervention with the smallest $e$, a tie to the repair, then to the later day.
```

Reason, for the scoping document (not the spec): the Wiener variance falls only at a replacement (spec, "Wiener
Model"), so a repair placed first can take the last free day of a cell that needs a replacement. The other accepted
spec sentences (Q2 to Q8, decisions 5 and 8) come with the review of Task B.

## 8. Review (29 Oct 2026)

Read in full against the amended plan and the spec (Horizon Sweeps 1871-2092, Warm Starts 2093-2255): the diffs of
`horizon_sweep.py` (920 lines), `greedy_warm_start.py` (424), `base.py` (hunks only in the warm-start functions, the new
helpers above them and `solve_mixed`; checked with `git diff`), and both test files. Both executors followed the plan;
every step is done once. Rankings, tie rules, Eq. d_gradient, classes, eligibility, stop rule, checkpoints, report
and row keys, day mapping, selector values, completion and relaxation logic, Wiener rule, rows and caps: no defect.

**Independent checks** (scratchpad, `SP`):
- `review_random_greedy.py`: 300 random fleets (five models, both laws, three forms, `pwl_points` 1-4, with and
  without replacement and `v_max_user`): `build_greedy_warm_start` equals `greedy_ref.py` in `x`, `m`, `r` and report.
- `review_lp_random.py`: 400 random fleets in the tangent form; all 86 starts without failures are feasible for the
  real WP6 rows (emulator LP with the binaries fixed). `verify_greedy_lp.py --wiener`: agreement.
- `review_process_pool.py`: a real `ProcessPoolExecutor` sweep on Windows (`n_workers=2`, no Gurobi): every case
  error comes back from its worker, is logged and recorded; rows ordered by `H2`; report written.

**Defects corrected**
| File:line | What was wrong | Change |
|---|---|---|
| `base.py:369-374` | `m` or `r` with a short last axis raised `IndexError` (executor B1; plan gap) | `ValueError` "'m' (or 'r') of a warm-start source must have the shape (F', L', T')" |
| `base.py:301-321` | `int()` read `H1 = 2.7` as 2 and `True` as 1 (B3) | `_source_horizon`: an integer >= 1, or a number with zero fractional part; else `ValueError` "must carry the integers ..." |
| `base.py:292-298, 356, 449, 1343` | an unknown mode in a direct call set nothing and reported `values` (B2) | `_check_warm_start_mode` (`ValueError`, the message of `solve_mixed`) in `apply_binary_warm_start`, `complete_binary_warm_start` and `solve_mixed` |
| `tests/test_warm_start.py:270-298` | no test of the three checks | `test_invalid_sources` (6 cases, and `3.0` accepted), `test_invalid_mode` |
| `tests/test_warm_start.py:328-345` | `test_failed_completion` failed on Gurobi if the copy reports INF_OR_UNBD | accepts both statuses; the IIS is checked when the status is `"infeasible"` |
| `base.py`, `greedy_warm_start.py`, `horizon_sweep.py` | 13 lines over 100 characters (`base.py` had none) | split |

**Executor problems: decisions**
- A1 `_sweep_defaults` via `inspect.signature`: kept (one source of the defaults; the plan's call lacked them).
- A2 output folder made after the per-`H1` checks: kept (every check still precedes the first case).
- A3 deep copies of the grid rows: kept (no YAML anchors in the file). A4 `absolute_mip_gap` `None` when not
  finite: kept (spec sentence 9). A5 extra tests: kept. A6 process pool: run here (above).
- B4, B5: spec sentences 1 and 3. B6: `test_solve_mixed_modes[hint]` and `test_relaxation_start` need Gurobi (WP8).
- Tool limits only, not code defects: with `--fake`, `test_rainflow.py` failed for the missing `getConstrByName`
  (added to `patch_fakegp.py`: 88 passed, 36 skipped, 0 failed; the WP6 runner: 95 passed); `test_base.py` needed
  `pytest.param` (added to `run_tests_wp7.py`) and fails 2 tests on the missing `Model.getRow` (27 passed).

**Spec sentences** (worktree spec; Claude adds them):
1. Wiener rule: insert between line 2240 (`\end{enumerate}`) and 2241: `At a Wiener cell, steps 3 and 4 form one
   trial: on every free day, put a repair and, if replacement is allowed, a replacement, and recompute $e$ after
   each; keep the intervention with the smallest $e$, a tie to the repair, then to the later day.`
2. Reference trajectory (Q6): new item after line 818: `\item \textbf{Reference trajectory} $\zeta$ of a schedule:
   the values of the rows of this section with every inequality replaced by an equality, from the initial values,
   the starting anchors and the replacement values.` (Lines 2230 and 2499 then cite a definition.)
3. Excess (Q3), new item after line 2234: `\item the violations of the side row $\mu \le \bar\mu_{\mathrm{side}}$
   and, where \texttt{v\_max\_user} is given, of $v \le V^{\mathrm{user}}_{\max}$, divided by $\tau$ and by the
   natural range of $v$.` Line 2236, after "contains the mean of the day": ` (the larger cap where two segments
   contain it, the last segment for a mean above $\tau$)`.
4. Completion (Q2), new item after line 2153: `\item \textbf{Report}: \texttt{requested} is the flag of the source;
   while the completion does not run, its status is \texttt{"not\_requested"}.`
5. Invalid source (Q4, B1-B3), new item after line 2115: `\item \textbf{Invalid source}: \texttt{solve()} raises
   \texttt{ValueError} for a source with \texttt{x} whose \texttt{H1} or \texttt{H2} is not an integer $\ge 1$ (a
   number with zero fractional part is one, a flag is not), or whose \texttt{x}, \texttt{m} or \texttt{r} does not
   have three axes with $T'$ days on the last.`
6. Line 1821 (Q5): `{Number of variables that received a value; each counts once, also in the mode \texttt{"both"}}`.
7. Line 2251 (Q7): `{the largest $\hat s_{i\ell}/\tau_{i\ell}$ of the construction pass and the largest reference
   mean over $\tau_{i\ell}$ of the schedule returned}`; line 2250: `{the largest excess term of a cell before the
   division by the natural ranges, over all cells, after the loop pass}`.
8. Line 1993 (Q8): `{the $H_2$ of the case that finished last}`.
9. Line 2017 (A4): append `; \texttt{null} if the objective or the bound is \texttt{null} or not finite`.
10. Line 1925 (plan A5, item 8): `{Written into every case input; if given, an integer $E > H_1$;
    \texttt{"evaluation\_total"} needs it}`.
11. Line 2029: after "except the sizes" add ` (\texttt{H1}, \texttt{H2}, \texttt{T}, \texttt{dimensions}; $F$, $M$,
    $L$ are read from the input data, \texttt{null} if they cannot be read)`.

**Tests (29 Oct 2026, `run_tests_wp7.py`)**: `test_horizon_sweep.py` 35 passed; `test_warm_start.py` 12 passed and
20 skipped without gurobipy, 30 passed and 2 skipped (Gurobi only) with `--fake`; `test_gamma.py` with `--fake` 34
passed, 2 skipped. All files CRLF; `import fleet_management` loads no gurobipy; 0 undefined names in the three
modules. Not run here: the 2 Gurobi-only tests and every real solve (WP8, licensed machine).
