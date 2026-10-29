# WP7 plan, part `plots`: the plotter and the public API

Status (29 Oct 2026): done. Plan (sections 1 to 7); Task A done and reviewed (section 8); Task B done and reviewed
(section 9): `tests/test_plotter.py` 20 passed. Next: Claude removes `utils/mixed_plotter.py` and commits.

## 0. Scope, places and the reference prototype

- **In scope**: `src/fleet_management/utils/plotter.py` (new, from `utils/mixed_plotter.py`),
  `src/fleet_management/__init__.py`, `tests/test_plotter.py` (new). Spec: Plotting API and Layout (2256-2355),
  Public API (222-233). Line numbers below are those of the worktree spec; the `lore` copy adds nothing to these
  sections. It adds two sentences on result files (1877-1878 there): NaN and ±infinity are `null` in YAML and JSON,
  and HDF5 stores no `objective` when it is `None`. The plotter already treats a missing key as `null`.
- **Out of scope**: the sweep report writer (`sweeps`), the validators and dashboards (`validator`,
  `gamma_validators`), the scripts outside `src` (WP8), the README (documents plan).
- **W** = `C:\Users\utente\Documents\GitHub\Fleet_Management-merge` (branch `merge/lore-into-main`, commit 0421e80).
- **P** = `C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad\wp7\plots`.
  It holds the planner's verified prototype:
  - `P\proto_plotter.py`: the target content of `W\src\fleet_management\utils\plotter.py` (CRLF, 773 lines).
  - `P\proto_test_plotter.py`: the target content of `W\tests\test_plotter.py` (LF, 438 lines).
  - `P\run_tests_plots.py`: the pytest-style runner of `wp6\gamma\run_tests_gamma.py`, plus the option
    `--plotter <path>` (used only for the prototype; executors run it without that option).
  - `P\render.py`: renders the schedule plot and two sweep plots of the test fixtures to PNG files.
  - `P\render_out\*.png`: the prototype's images, checked by the planner.
  - Claude: copy `P` into `merge/wp45_work/wp7/plots/` with the other local tools.
- **Prototype checks (29 Oct 2026, this computer)**: `run_tests_plots.py --plotter proto_plotter.py
  proto_test_plotter.py`: 18 passed, 1 failed (`test_public_api_without_gurobi`: the prototype lies outside
  `src`, so its subprocess finds no package; in the worktree after Task A it fails because the package still exports
  `plot_mixed_management`, and it passes after Task B). The subprocess code of that test, run alone with
  `PYTHONPATH=W\src`, imports the package and lists today's twelve names. Undefined-name check (`wp6\rainflow\names_check_try.py`): 0 in both
  files. Every top-level function of the prototype is used.

## 1. Gaps

Code lines are those of `W\src\fleet_management\utils\mixed_plotter.py` and `W\src\fleet_management\__init__.py`.

**Schedule plot (`plot_management`)**
- G1. File `utils/mixed_plotter.py` → `utils/plotter.py` (spec 199, 2260).
- G2. Function `plot_mixed_management` (`:25`) → `plot_management` (spec 2267; decision "Plotting").
- G3. Private reader `_read_input` and `_read_hdf5` (`:768-776`, `:842-864`), used at `:96` and `:786` →
  `config.read_input` (spec 2664 "One reader serves every function that takes an input path"; interface rule;
  `wp4_plan.md` G47).
- G4. Keys outside the spec read for the counts: `presolved_performance`, `performance.presolved_performance`,
  `performance.presolved_model` (`:463-469`, `:787-793`) → counts from `performance`, or from `runner.out`
  (spec 2320).
- G5. `gamma_formulation.actual_gurobi_model` (`:460-462`): key of the removed Gamma code (`wp6_plan_gamma.md`
  section 8, point 4) → removed.
- G6. Required keys: the code requires `F`, `M`, `mu_0`, `mu`, `x` (`:420-426`); `L` defaults to 1 (`:428`); `tau`
  defaults to 1.0 and `alpha` wins over it (`:444`); `H1` comes from the old `H` or `T // 2` (`:449`, `:757-760`);
  `H2` defaults to `T - H1` (`:450`) → required `F`, `M`, `L`, `H1`, `H2`, `tau`, `mu_0`, `mu`, `x`, and the
  message names the missing keys (spec 2288).
- G7. Lenient shapes: `mu_0` of one axis and `mu` of two axes are reshaped (`:432-435`); `tau` is broadcast from a
  scalar, `(L,)` or `(F,)` (`:445`, `:735-745`) → a shape other than spec "Result Dictionary" raises `ValueError`
  (spec 2288; shapes 1709-1720).
- G8. `T` is taken from `mu` (`:441`) and never compared with the key `T` (spec 1707, 2288).
- G9. Model source: `model_assignment`, then `reliability_impl`, then `models` or `degradation` (`:680-694`) → the
  key `model` (spec 2281; decisions "Output schema" and "Plotting").
- G10. Model label: `_model_abbreviation` prints "Gamma" or "Remaining-life" (`:714-723`) → the model name
  (spec 2294).
- G11. Row label "Vehicle i" (`:711`), axis label "Vehicle and component model" (`:561`) → "Train i" (spec 2294).
- G12. Day labels `0..T-1` (`:553`), axis label "Time step" (`:560`) → days `1..T` (spec 2295).
- G13. Phase label "initialization" (`:574`) → "transitory" (spec 2295).
- G14. "physical mean / failure threshold" (`:586`), title "physical expected damage" (`:562`), note on a Gamma
  "bounding-shape state" (`:672-677`) → colour bar "tracked mean / failure threshold" (spec 2296); `mu` is the
  tracked upper bound (spec 1720, 1746), and the merged Gamma model has no separate shape state.
- G15. No mark `D`: a depot day without intervention gets `I` (`:610-613`); the legend has four marks (`:588-592`)
  → five marks `P`, `R`, `M_j`, `D`, `I` (spec 2297-2306; decision "Plotting").
- G16. Summary: `_compact(repair_model)` prints nested lists (`:627`, `:871-874`) → the repair laws present
  (spec 2313).
- G17. Summary: status colours only for `optimal`, `time_limit`, `infeasible` (`:619-620`) → coloured by outcome
  (spec 2311).
- G18. Summary: the counts say "Presolved" or "Original" but not where they come from (`:634-651`) → the panel
  says that the counts come from `runner.out` (spec 2320).
- G19. `runner.out` with several solves: the first "Presolved:" line wins (`:809-813`); the warm-start completion
  and the relaxed copy are solved before the main model. Spec silent (question Q1).
- G20. `_draw_solution` saves and closes its figure (`:598-599`); the tests need the figure. Not a spec item.

**Sweep plot (`plot_horizon_sweep`)**
- G21. `maximum_mip_gap_for_stopping`: default 0.05 when absent, and `float(None)` fails when it is `null` (`:163`),
  which is the spec's default (spec 2001) → a line if it is a number, otherwise a mark at the requested MIP gap of
  each case (spec 2351).
- G22. Panel (b): the reference line is drawn only when some gap is finite (`:310-320`); the grey gap line is drawn
  twice (`:322-332`).
- G23. `is_proven` is `status == "optimal"` without the feasible test (`:198-200`) → a proven case is feasible
  (spec 1889).
- G24. `F`, `M`, `L`, `H1` are in a separate text line, not in the title (`:381-387`) → the title names them
  (spec 2345).
- G25. Labels "proven optimal", "feasible", "time-limited feasible", "no feasible solution found",
  "no feasible solution; gap undefined" (`:271-357`) → one name per mark of spec 2350, the same in panels (a)
  and (b).
- G26. `_draw_horizon_sweep` saves and closes its figure (`:398-399`); as G20.

**Public API**
- G27. `__init__.py:3-7, 23` imports from `utils.mixed_plotter` and exports `plot_mixed_management` → exactly the
  eleven names of spec 225-232, the plots from `utils.plotter`.
- G28. `__init__.py` has no docstring, and `__all__` is not in the spec's order. Minor.

**Checked, no gap**: plot paths and their errors (`:43-52`, `:83-94`, `:763-765`; spec 2273, 2286-2287, 2333);
any status with a solution is plotted (spec 2278); missing `m`, `r` read as zero (`:454-455`; spec 2279); default
component names and the length check (`:697-705`; spec 2280, 2288); heavier line between trains (`:565-568`); dashed
line after day `H1` (`:571-572`); no mark in the column "initial" (`:541`); merge of reports and its errors
(`:101-154`; spec 2332, 2339); selection for equal `H2` (`:141-154`; spec 2344); panel (c) (`:365-379`; spec 2352);
`H1` of a report read from `H1` or `fixed_dimensions` (`:106-107`), so the spec's report (`H1` at the top level)
and old reports both work.

**Other packages, not planned here**: `validation/validator.py:13` imports `solver._read_input` (validator);
`horizon_sweep.py:29-62, 426-492` reads `gamma_calibration*` and `gamma_formulation` (sweeps); the Gamma validators
read removed keys (gamma_validators).

## 2. Executor tasks

Two tasks with disjoint files. Task A: `utils/plotter.py` and `tests/test_plotter.py`. Task B: `__init__.py`.
Task A can start at once. Task B runs last in WP7, after Task A and after the executors of the other three parts,
because `__init__.py` imports their modules.

Rules for both tasks: do not spawn sub-agents; do not delete any file (`utils/mixed_plotter.py` stays until Claude
removes it at the commit); write intermediate outputs only to `P`; never put the user's name or email address in
any request; no installs, no commits; run Python as `python`, with `PYTHONIOENCODING=utf-8`; write scripts with the
Write tool, not with Bash heredocs. Before each edit, re-read the region from disk.

### Task A: the plotter and its tests (one Sonnet executor)

Each step names the functions to change. The prototype `P\proto_plotter.py` holds the exact target text of each
function; copy it from there. The words below say what changes, so that a reader of the plan knows it without the
prototype. Keep CRLF line endings in `plotter.py` (the Edit tool keeps them).

**A1. Copy.** Copy `W\src\fleet_management\utils\mixed_plotter.py` to `W\src\fleet_management\utils\plotter.py`
byte for byte (`cp`). Check: `cmp` of the two files prints nothing.

**A2. Header and the two public functions** (spec 2260, 2267, 2664; G1-G3, G20, G26).
- Module docstring: the seven lines at the top of the prototype.
- Imports: remove `json` and `yaml`; add `from fleet_management.config import read_input` after the matplotlib and
  numpy imports.
- `ACTION_COLOURS`: remove the entry `"mixed_intervention"`. Add the constant
  `REQUIRED_RESULT_KEYS = ("F", "M", "L", "H1", "H2", "tau", "mu_0", "mu", "x")`.
- Rename `plot_mixed_management` to `plot_management`; new docstring. Its checks stay. Its last line becomes four:
  `view = _normalise_solution(read_input(input_file), _read_presolve_log(input_file))`,
  `figure = _draw_solution(view)`, `figure.savefig(plot_path, dpi=170, bbox_inches="tight")`, `plt.close(figure)`.
- `plot_horizon_sweep`: `reports = [read_input(report_path) for report_path in report_paths]`; then
  `figure = _draw_horizon_sweep(_normalise_horizon_sweep(merged_report))`, save and close as above.
- Check: the text from line 1 to the end of `plot_horizon_sweep` equals the prototype
  (`diff --strip-trailing-cr`, restricted to that range, or compare by eye).

**A3. Readers** (spec 2320, 2664; G3, G4, G19).
- Delete `_read_input`, `_read_solution_input` and `_read_hdf5`.
- Add `_read_presolve_log(input_file: Path) -> dict[str, int]` after `_resolve_plot_path`: `{}` if
  `input_file.parent / "runner.out"` is not a file; otherwise `_parse_gurobi_presolve` of its text (UTF-8,
  `errors="replace"`).
- `_parse_gurobi_presolve`: take the **last** "Presolved:" line (`list(re.finditer(...))[-1]`), then the first
  "Variable types:" line after it, as before. Docstring as in the prototype. (Q1; a different answer changes only
  this line.)
- Check: `grep -n "presolved_performance\|presolved_model\|h5py\|json\.\|yaml\." plotter.py` prints nothing.

**A4. `_normalise_solution(data, presolved)`** (spec 2279, 2288; G5-G8). Replace the whole function by the
prototype's. In words:
- Missing keys: `[key for key in REQUIRED_RESULT_KEYS if data.get(key) is None]`; if any, `ValueError("The result
  has no plottable solution; missing keys: " + ", ".join(missing))`.
- `F, M, L, H1, H2` as `int`; `tau`, `mu_0`, `mu`, `x` as float arrays. Errors, in this order: `tau` or `mu_0` not
  `(F, L)` → `"{name} has shape {shape}; expected {(F, L)}"`; `mu` not of three axes with `(F, L)` first →
  `"mu has shape ...; expected (F, L, T)"`; `T = mu.shape[2]`; key `T` present and different →
  `"mu has {T} days; the key T is {data['T']}"`; `H1 + H2 != T` → `"H1 + H2 must equal T; got ..."`; `x` not
  `(F, M + 1, T)` → `"x has shape ..."`; a `tau <= 0` → as before.
- `m`, `r` through `_optional_array` (unchanged).
- Counts: `presolved` if it is not empty, with `counts_source = "runner.out (presolved)"`; otherwise
  `data.get("performance") or {}` with `counts_source = "performance"`. View keys `variables`,
  `continuous_variables`, `integer_variables`, `linear_rows` (from `linear_constraints`), `counts_source`.
- View keys `repairs` and `replacements` (counts of `m > 0.5`, `r > 0.5`) replace `repaired_components` and
  `replaced_components`; `formulation_scope` goes.
- Delete `_cell_grid` and `_legacy_h1`.
- Check: `grep -n "alpha\|_legacy_h1\|_cell_grid\|gamma_formulation\|model_assignment" plotter.py` prints only the
  `"alpha": 0.88` of the mark boxes.

**A5. Drawing of the schedule** (spec 2294-2306; G11-G15, G20).
- `_draw_solution(view)`: no `plot_path` argument; docstring "Return the figure of the schedule plot: the grid and
  the summary panel."; it ends with `return fig` instead of `savefig` and `close`.
- Texts: x tick labels `["initial", *range(1, T + 1)]`; x label "Day"; y label "Train and component"; axis title
  "Schedule and tracked mean degradation"; phase label "transitory" (and "operating", unchanged); colour bar label
  "tracked mean / failure threshold".
- Legend: five handles in this order and with these labels: "P replacement", "R repair", "M$_j$ mission $j$",
  "D depot, no intervention", "I idle train"; colours `replacement`, `repair`, `mission`, `depot`, `idle`.
- `_component_action_label`: after the mission test, `if view["x"][i, 0, k] > 0.5: return "D", "depot"`; then
  `return "I", "idle"`. New docstring "Return the mark of cell (i, l) on day index k: the first rule that applies."
- Check: compare `_draw_solution` and `_component_action_label` with the prototype.

**A6. Summary panel and the helpers of the rows** (spec 2281, 2294, 2309-2320; G9, G10, G16-G18).
- `_draw_statistics`, lines in this order (label, value): "Status" (`status` with `_` as space; colour
  `#15803d` if `optimal`, `#b45309` for any other status, `#374151` and the text "unknown" if the key is missing);
  "Objective"; "Dimensions"; "Horizon"; "Models" (`", ".join(_unique_values(view["models"].tolist()))`);
  "Repair laws" (`", ".join(_unique_values(data.get("repair_model"))) or "-"`); "Objective mode" and "J_op / H2"
  (only if `objective_mode` is present); "MIP gap" (if present); "Counts from" (`view["counts_source"]`);
  "Variables", "Continuous variables", "Integer variables", "Linear rows" (each if not `None`); "Optimizer";
  "B&B nodes"; "Repairs"; "Replacements". Remove the grey note box at the end.
- `_model_grid(data, F, L)`: `data.get("model")`; `None` → an `(F, L)` object array of `"unknown"` (Q2); otherwise
  `np.asarray(value, dtype=object)`, and a shape other than `(F, L)` → `ValueError("model has shape {shape};
  expected {(F, L)}")`.
- `_component_row_label`: `f"Train {i + 1} — {component}\n{view['models'][i, l]}"` (em dash, as now).
- New `_unique_values(value) -> list[str]`: the distinct strings of a string or a nested list, in order of first
  appearance; `None` and `None` entries are skipped.
- Delete `_model_abbreviation` and `_unique_models`.
- Check: compare these functions with the prototype; `grep -n "Vehicle\|physical\|initialization\|Remaining"
  plotter.py` prints nothing.

**A7. Sweep plot** (spec 2343-2352, 1889, 2001; G21-G26).
- `_normalise_horizon_sweep`: `maximum_stopping_gap = _optional_float(stopping_rule.get(
  "maximum_mip_gap_for_stopping"))` (no default); compute `has_feasible` once before the row; row key
  `"requested_mip_gap": _optional_float(case.get("requested_mip_gap"))`; `"is_proven": has_feasible and status ==
  "optimal"`.
- `_draw_horizon_sweep(view)`: no `plot_path`; docstring "Return the figure of the sweep plot: panels (a), (b),
  (c) of the spec."; ends with `return figure`.
- Labels in panels (a) and (b): "proven (optimal)", "feasible, not proven", "no feasible solution"; panel (a) keeps
  "best feasible".
- Panel (b) reference, drawn before the gap marks and whether or not a gap is finite: if
  `view["maximum_mip_gap_for_stopping"]` is a number, `axhline` at 100 times it, `color="#15803d"`,
  `linestyle="--"`, label `f"{value:g}% stopping gap"`; otherwise one `scatter` of the points
  `(H2, 100 * requested_mip_gap)` of the rows with a requested gap, `marker="_"`, `s=160`, `color="#15803d"`, label
  "requested MIP gap" (nothing if no row has one). Remove the first of the two grey `gap_ax.plot` calls.
- Title: `figure.suptitle(f"Operating-horizon sweep ({dimensions})", ...)`, with `dimensions` as now
  (`"F=4, M=1, L=1, H1=4"`); remove the separate `figure.text(0.02, 0.925, dimensions, ...)`.
- Check: `diff --strip-trailing-cr P\proto_plotter.py W\src\fleet_management\utils\plotter.py` prints nothing;
  the file has CRLF only (`python -c` count of `\r\n` equals the count of `\n`); the undefined-name check
  `python P\..\..\wp6\rainflow\names_check_try.py W\src\fleet_management\utils\plotter.py` reports 0.

**A8. Tests, part 1: helpers and schedule layout.** Create `W\tests\test_plotter.py` (LF line endings, as
`tests/test_io.py`) with lines 1-215 of `P\proto_test_plotter.py`: docstring, imports (`matplotlib.use("Agg")`
before `pyplot`), the helpers `_result`, `EXPECTED_MARKS`, `_write`, `_schedule_figure`, `_axes`, `_marks`,
`_summary`, `_inside`, and the tests `test_schedule_layout`, `test_schedule_summary`,
`test_schedule_counts_from_runner_log`, `test_schedule_defaults`. Check:
`python P\run_tests_plots.py W\tests\test_plotter.py test_schedule` → 6 passed.

**A9. Tests, part 2: files and errors of the schedule plot.** Append lines 216-290 of the prototype:
`test_plot_management_writes_png_and_pdf`, `test_plot_management_without_solution`,
`test_plot_management_empty_result`, `test_plot_management_path_errors`, `test_plot_management_shape_errors`,
`test_parse_gurobi_presolve`. Check: `... test_plot_management` → 7 passed; `... test_parse` → 1 passed.

**A10. Tests, part 3: sweep plot and public API.** Append lines 291-438 of the prototype: `_case`, `_report`,
`SWEEP_CASES`, `_sweep_figure`, `test_sweep_layout`, `test_sweep_requested_gap_marks`,
`test_sweep_selection_of_equal_H2`, `test_sweep_errors`, `PUBLIC_API`, `test_public_api_without_gurobi`. Check:
`diff P\proto_test_plotter.py W\tests\test_plotter.py` prints nothing; the whole module → 18 passed, 1 failed
(`test_public_api_without_gurobi`, until Task B).

**A11. Rendered check.** Run `python P\render.py W\src\fleet_management\utils\plotter.py W\tests\test_plotter.py
P\render_task_a`, then open the three PNG files with the Read tool. What must be seen is in section 3, "Rendered
check". Compare with `P\render_out\*.png`.

**A12. Report** (at most 30 lines): the test counts, the result of each check, anything that did not match.

### Task B: the public API (one Sonnet executor, after everything else in WP7)

**B1. Names and modules.** Read the section "4. Exports" of `merge\wp7_plan_sweeps.md`,
`merge\wp7_plan_validator.md` and `merge\wp7_plan_gamma_validators.md`. For each of the eleven names of section 4
below, run `grep -rn "^def <name>" W\src\fleet_management`. If a name is defined in another module than section 4
says, use that module in B2 and in `PUBLIC_API` of `tests/test_plotter.py`, and report it. If a name is not defined
anywhere, stop and report.

**B2. Write `W\src\fleet_management\__init__.py`** with the content of section 4, CRLF line endings (as the
current file).

**B3. Checks.**
- `python -c` with `PYTHONPATH=W\src`: `import sys; sys.modules["gurobipy"] = None; import fleet_management;
  print(fleet_management.__all__)` → the eleven names in the order of section 4, no error.
- `python P\run_tests_plots.py W\tests\test_plotter.py` → 19 passed.
- `grep -rn "mixed_plotter\|plot_mixed_management" W\src W\tests` → no line outside `utils/mixed_plotter.py`.

**B4. Report** (at most 20 lines): the modules used, any difference from section 4, the check results.

## 3. Tests

`tests/test_plotter.py`, pytest style; no test needs Gurobi, so none is skipped. Result files are written by
`solver._save_results` (the real writers) from a hand-built result; they are read back by the plotter through
`config.read_input`. Sweep reports are written as the sweep writes them (`yaml.safe_dump`), or as JSON.

**Fixture.** `_result()`: `F = 2`, `M = 1`, `L = 2`, `H = [2, 2]`, `T = 4`; models `gaussian` and `rainflow`;
components "Wheelset" and "Brake pad"; status `time_limit`. `x`, `m`, `r` are chosen so that every mark occurs:

| Row (train, component) | Day 1 | Day 2 | Day 3 | Day 4 |
|---|---|---|---|---|
| 1, Wheelset | M1 | I | M1 | R |
| 1, Brake pad | M1 | I | M1 | P |
| 2, Wheelset | R | M1 | D | M1 |
| 2, Brake pad | D | M1 | P | M1 |

`_result(solution=False)` sets the solution keys to `None` and the status to `infeasible`, as `solve` does without
a solution. The figure is built by `_draw_solution(_normalise_solution(read_input(path), _read_presolve_log(path)))`
and inspected as objects.

| Test | What it checks (spec) |
|---|---|
| `test_schedule_layout` (YAML, JSON, HDF5) | four row labels "Train i — name" + model name; columns "initial", 1-4; the 16 marks of the table; one dashed line at x = 2.5; "transitory", "operating"; legend of the five marks in order; colour bar label (2294-2306) |
| `test_schedule_summary` | each line of the panel, for example "Status" = "time limit", "Repair laws" = "ard1, ardinf", "Counts from" = "performance", "Repairs" = "2" (2309-2320) |
| `test_schedule_counts_from_runner_log` | with a `runner.out` beside the result: "Counts from" = "runner.out (presolved)", the four counts of the log (2320) |
| `test_schedule_defaults` | without `m`, `r`, `component_names`, `model`: depot days marked `D`, names "Component 1", "Component 2", model "unknown" (2279-2281; Q2) |
| `test_plot_management_writes_png_and_pdf` | a path without extension gets `.png`; `.pdf` works; the default `output.png`; the PNG is not blank (2273) |
| `test_plot_management_without_solution` (three formats) | `ValueError` "missing keys: mu, x" (2278, 2288) |
| `test_plot_management_empty_result` | an empty mapping: `ValueError` naming all nine keys (2288) |
| `test_plot_management_path_errors` | `FileNotFoundError` (result file, image folder), `ValueError` (both extensions), `PermissionError` (via `mock.patch.object(plotter.os, "access", return_value=False)`) (2286-2288) |
| `test_plot_management_shape_errors` | `tau`, `mu_0`, `x`, `m`, `model` of a wrong shape; `H1 + H2 != T`; key `T` against `mu`; `component_names` of wrong length (2288) |
| `test_parse_gurobi_presolve` | counts with thousands separators; no match → `{}`; the last of two solves (Q1) |
| `test_sweep_layout` | title "Operating-horizon sweep (F=4, M=1, L=1, H1=4)"; legend of panel (a); values of the proven cases; the 5 % stopping line; the three lines of panel (c) (2345-2352) |
| `test_sweep_requested_gap_marks` | `maximum_mip_gap_for_stopping: null`: no line, one mark per case with a requested gap, at 1 % (2351) |
| `test_sweep_selection_of_equal_H2` | two reports, one YAML and one JSON, with the same `H2 = 12`: the case with the smaller `J_op/H2` is kept; the PNG is written; the default `horizon_sweep.png` (2333, 2344) |
| `test_sweep_errors` | missing file; `.h5`; empty list; different `F`; different `H1`; no `cases` (2338-2339) |
| `test_public_api_without_gurobi` | a subprocess with `sys.modules["gurobipy"] = None` imports the package; `__all__` is the eleven names in the spec's order, each from its module (222-233) |

**How to run here.** `python P\run_tests_plots.py W\tests\test_plotter.py [name]` (no pytest on this computer;
the runner loads the package without `__init__.py`, so the API test starts its own subprocess with
`PYTHONPATH=W\src`). Expected: 18 passed and 1 failed after Task A; 19 passed after Task B. With real pytest
(WP8): 19 passed.

**Rendered check** (step A11). `P\render.py` writes `schedule.png`, `sweep.png` and `sweep_requested.png`. Open
each with the Read tool. What must be seen:
- `schedule.png`: four rows with the labels of the table above and the model name on the second line; a heavier
  line between train 1 and train 2; columns "initial", 1, 2, 3, 4 and the axis label "Day"; a blue dashed line
  between days 2 and 3, with "transitory" and "operating" above the grid; the marks of the table; green-to-red
  colours that grow along each row; colour bar "tracked mean / failure threshold"; a legend of five marks under the
  grid; the panel "Run summary" with 18 lines, the status "time limit" in amber.
- `sweep.png`: title with `F=4, M=1, L=1, H1=4`; panel "Operating objective" with two blue proven points labelled
  0.911 and 0.798, the orange plus sign "best feasible" at `H2 = 12` (labelled 0.618), a red x at `H2 = 16`; panel "MIP gap"
  with a green dashed line at 5 %; panel "Formulation growth" with three lines.
- `sweep_requested.png`: as `sweep.png`, but in panel "MIP gap" three short green marks at 1 % instead of the line.

## 4. Public API

The exact content of `W\src\fleet_management\__init__.py` (CRLF). The modules are those of the spec's Package
Layout (spec 181-202) and of the code today; the other plans have not yet filled their section "Exports" (checked
29 Oct 2026). The spec does not name the module of the dispatcher `validate_gamma_schedule` and its file version;
today it is `gamma_stochastic_validator.py:386` and `:462`. Step B1 confirms each module.

```python
"""Public API of the package fleet_management (spec, "Public API").

``import fleet_management`` works without Gurobi: the modules below import ``gurobipy``,
or ``degradation_model.base``, only inside the functions that solve.
"""

from fleet_management.solver import solve
from fleet_management.horizon_sweep import sweep_horizon_grid, sweep_operating_horizons
from fleet_management.utils.plotter import plot_horizon_sweep, plot_management
from fleet_management.validation.validator import validate
from fleet_management.degradation_model.gamma_utils.gamma_replay_validator import (
    validate_gamma_replay_files,
    validate_gamma_replay_schedule,
)
from fleet_management.degradation_model.gamma_utils.gamma_stochastic_validator import (
    validate_gamma_schedule,
    validate_gamma_schedule_files,
    validate_gamma_stochastic_schedule,
)


__all__ = [
    "solve",
    "sweep_operating_horizons",
    "sweep_horizon_grid",
    "plot_management",
    "plot_horizon_sweep",
    "validate",
    "validate_gamma_replay_schedule",
    "validate_gamma_replay_files",
    "validate_gamma_stochastic_schedule",
    "validate_gamma_schedule",
    "validate_gamma_schedule_files",
]
```

- Not exported (spec 233): `solve_mapping`, `load_config`, `build_greedy_warm_start`,
  `validate_gamma_tail_bound_schedule` and `validate_gamma_tail_bound_files`.
- `plot_mixed_management` leaves the API. No alias stays: the spec lists exactly these names.
- Import without Gurobi, checked today with `sys.modules["gurobipy"] = None`: the package imports, and neither
  `gurobipy` nor `degradation_model.base` is loaded. The other three plans must keep this; a module that imports
  `base` at module level breaks every test module of `tests/` (risk noted in `wp4_plan.md`, section 7).

## 5. Callers outside src

Scripts that the changes of this plan break (WP8 fixes them; grep of `examples`, `experiments`, `euler`,
`remote_smoke_test.py`, 29 Oct 2026):
- `examples/regression/check_result_visualization.py:12, 16-21, 76`: imports `plot_mixed_management` (an
  `ImportError` after Task B) and checks `__module__ == "fleet_management.utils.mixed_plotter"`. Its result has no
  key `model` (it writes `model_assignment`), so its rows would show "unknown"; its printed text names the removed
  labels "Gamma/Remaining-life" and "physical mean".
- `examples/regression/check_horizon_sweep_visualization.py:16-17`: checks `__module__` against `mixed_plotter`.
  The rest still runs (its report keeps `H1` in `fixed_dimensions` and has no `requested_mip_gap`).
- `experiments/gamma_horizon_stabilization/gamma_horizon_stabilization.sbatch:69-70`: copies
  `src/fleet_management/utils/mixed_plotter.py` under `set -euo pipefail`; the job stops once the file is removed.

Not broken: `remote_smoke_test.py:110-275` calls `plot_management` from the package (the name stays);
`experiments/gamma_horizon_stabilization/plot_result.py:7, 15` calls `plot_horizon_sweep` (its help text offers SVG,
which the plotter refuses today already); `euler/` has no caller. Documents, not scripts: `README.md:169, 297, 313`
name `plot_mixed_management` and `mixed_plotter.py` (documents plan).

## 6. Removals

- `src/fleet_management/utils/mixed_plotter.py`, after Task B. Condition: the last check of B3 finds no import of
  it in `src` or `tests`. Claude removes it at the commit; git should record a rename to `utils/plotter.py` if the
  two files are similar enough.
- Nothing else leaves. `ACTION_COLOURS["mixed_intervention"]` and the functions named in steps A3, A4 and A6 leave
  inside the new file.

## 7. Questions

No question blocks a step: each step follows the recommendation, and a different answer changes only the line named.

- **Q1. `runner.out` with several solves.** The spec says the panel shows the counts of the lines "Presolved: ..."
  and "Variable types: ...", but a log can hold several solves: the warm-start completion and the relaxed copy are
  solved before the main model, in the same log. The code takes the first "Presolved:" line, which can be that of
  the relaxed copy. Options: the **last** "Presolved:" line and the first "Variable types:" line after it
  (recommended: the main solve is the last one); the first, as now. Step A3, one line; `test_parse_gurobi_presolve`.
- **Q2. A result without the key `model`.** The spec reads the model of each row from `model` (2281), but `model`
  is not among the keys whose absence raises (2288). `solve` always writes it; old files lack it. Options: the label
  "unknown" (recommended: old files still plot, and the error list of the spec stays exact); a `ValueError`, by
  adding `model` to the required keys. Step A6; `test_schedule_defaults`.
- **Q3. A result file that holds no mapping** (an empty YAML file, a JSON list). `config.read_input` raises
  `TypeError`; the plot's error table (2284-2289) has no such row. Options: let the reader's `TypeError` pass
  (recommended: one reader for every input path, spec 2664; the sweep's table already has this row, 1940), and add
  the row to the spec's two plot tables; catch it and raise `ValueError`. No step depends on it; no test.
- **Q4. A sweep report with an empty list `cases`.** The spec raises only for a report without the list (2339). The
  code also raises `ValueError` for an empty list. Options: keep the error (recommended: there is nothing to plot,
  and a sweep writes its first report after its first case, 1934); draw empty panels. No step changes.
- **Q5. "The title names the sweep" (2345).** Options: the fixed name "Operating-horizon sweep" with `F`, `M`, `L`,
  `H1` (recommended, step A7); also the input file of the report (its key `input`).

## 8. Review of Task A (29 Oct 2026)

Read in full: `utils/plotter.py`, `tests/test_plotter.py`, `solver.read_result`, the executor's progress file,
interface decision 1, spec 1741-1770 and 2256-2355. A2 to A12 were done as planned. The schedule plot reads its file
with `solver.read_result` (decision 1), and the sweep plot reads with `config.read_input`.

**Defects corrected** (lines of the corrected files)
- `plotter.py:468-475`: a `null` entry of `tau`, `mu_0`, `mu`, `x`, `m` or `r` becomes NaN in `read_result`. The plot
  drew it as a cell without colour, or as a false mark `I` (NaN > 0.5 is false). A result with a solution has finite
  numbers there (`base.py:595-598` builds `x`, `m`, `r`, `mu` on every index). Now `ValueError` "<key> has an entry
  that is not a finite number". This covers the executor's problem 3 (NaN in `tau`).
- `plotter.py:677-678`: a string `component_names` was taken as one name, so it passed the length check when
  `L = 1`. Now `ValueError`, because the spec's type is a list.
- `plotter.py:138-142`: with several reports, a case without `H2` raised `KeyError` or `TypeError`. Now
  `ValueError`, as with a single report.
- `plotter.py:364-372`: the gap labels overlapped the reference marks (the executor's problem 4). They now have a
  white box (alpha 0.85) above the marks; the re-rendered `sweep_requested.png` shows "0.0%" readable at H2 = 4 and 8.
- Tests: no test checked the colours, the NaN keys of the result rules, or the new errors. Added: the colour of a
  cell is clipped at 1, and the column "initial" shows `mu_0` (`test_schedule_layout`); NaN `z`, `R`, `K`, `g_*` in
  the fixture, through every writer and reader (the plot does not read them); four cases in
  `test_plot_management_shape_errors`; `test_plot_management_not_a_mapping`; a `TypeError` and a case without `H2`
  in `test_sweep_errors`.

**NaN rule checked.** The plot reads only `tau`, `mu_0`, `mu`, `x`, `m`, `r`, `model`, `component_names`, the scalars
and `performance`. None of these is NaN in a spec result. `v`, `z`, `R`, `K`, `g_*` are not read.

**Checks.** `run_tests_plots.py tests/test_plotter.py`: 19 passed, 1 failed (`test_public_api_without_gurobi`;
`import fleet_management` works again, and only the old `__init__.py` remains, which Task B replaces). Undefined
names: 0 in both files. Line endings: `plotter.py` CRLF only (785 lines); `test_plotter.py` LF only (482 lines). The
images in `P\render_review\` were checked by eye.

**Spec sentences** (Plotting API and Layout; worktree lines, the `lore` copy is 11 lines lower). The first two are
Q3 (decision 5). The others state choices of the code that the spec leaves open.
```latex
% schedule plot, Errors: new row after PermissionError; the ValueError row ends with the last clause
\rejrow{\texttt{TypeError}}{the data of the result file are not a mapping}
... \texttt{component\_names} is not a list of $L$ entries; an entry of \texttt{tau}, \texttt{mu\_0}, \texttt{mu},
\texttt{x}, \texttt{m} or \texttt{r} is not a finite number, or an entry of \texttt{tau} is not positive}
% sweep plot, Errors: new row; the ValueError row ends with the last two clauses
\rejrow{\texttt{TypeError}}{the data of a report file are not a mapping}
... a report has no list \texttt{cases}; the reports hold no case; a case has no \texttt{H2}}
% schedule plot, Reading rules: two new items first; the item on the model becomes the third
\item The result file is read with \texttt{solver.read\_result}: a missing key is \texttt{null}, and a \texttt{null} entry of an array is \texttt{NaN}.
\item $T$ is the length of the last axis of \texttt{mu}; a key \texttt{T} must equal it.
\item The model of each row is read from the key \texttt{model}; without that key, every row shows the model ``unknown''.
% Summary panel: first and last items
\item status, in green if \texttt{optimal}, otherwise in amber; ``unknown'' in grey without the key;
\item numbers of repairs and replacements, counted over all cells and days.
% Source of the counts: sentence appended
If the log holds several solves, the panel uses its last line ``Presolved: ...'' and the first line ``Variable types: ...'' after it; a log without such a pair is not used.
% Sweep plot, Content: Selection replaced, Sizes and Title new
\item \textbf{Selection.} For equal $H_2$ the plot keeps the case with a solution (\texttt{J\_op\_average} is not \texttt{null}), then the smaller $J_{\mathrm{op}}/H_2$, then the larger objective bound. Two values of $J_{\mathrm{op}}/H_2$ are equal if \texttt{numpy.isclose} holds with its default tolerances; a case without a bound loses against one with a bound; a remaining tie keeps the case of the earlier report in the list.
\item \textbf{Sizes.} $F$, $M$, $L$ are the entries of \texttt{fixed\_dimensions}; $H_1$ is its entry \texttt{H1} if present, otherwise the key \texttt{H1}.
\item \textbf{Title.} ``Operating-horizon sweep (F=$F$, M=$M$, L=$L$, H1=$H_1$)''; a size that the reports do not give is left out.
% panel (a): last sentence replaced
A case is feasible if its \texttt{J\_op\_average} is not \texttt{null} and its status is not \texttt{"infeasible"}, \texttt{"unbounded"} or \texttt{"inf\_or\_unbounded"}. The values of the proven cases and of the best feasible case are shown; a case without a feasible solution is marked at the bottom of the panel.
% panel (b): replaced
Relative MIP gap in percent of the feasible cases, each value written beside its point; a case without a feasible solution is marked at the top of the panel. A horizontal line at \texttt{maximum\_mip\_gap\_for\_stopping} of the \texttt{stopping\_rule} of the first report if it is a number; otherwise a mark at the requested MIP gap of each case that has one.
```
The first reading rule names `solver.read_result`. The validator plan's spec sentence must define that reader.

## 9. Review of Task B (29 Oct 2026)

Read: `__init__.py` (34 lines) against section 4, spec 222-233 (Public API) and the signatures of the tool
sections; the file it replaced (`git show HEAD:src/fleet_management/__init__.py`); the progress file.

- **Content.** The file equals section 4 word for word: docstring, six import statements, `__all__` with the
  eleven names in the spec's order. CRLF only (34 and 34), as the old file. The old file had no docstring; the
  other style points (import blocks in parentheses, two blank lines before `__all__`) are kept.
- **Signatures.** `P\review_b_signatures.py` imports the package with `gurobipy` and `streamlit` refused. It compares
  each exported signature, with annotations removed, against the `lore` spec (214-215, 1913-1919, 2059-2062, 2278,
  2338, 2384-2385, 2494-2497, 2597-2600, 2646-2651). Result: 11 of 11 equal, and every return annotation is `dict`
  or `None`, as in the spec.
- **Import without Gurobi.** It works. `degradation_model.base` and `utils.mixed_plotter` stay unloaded. The
  docstring's claim holds: among the exported modules, only `solver.solve_mapping:121` imports `base`, inside the
  function. `gamma_utils/gamma_diagnostics.py:14` imports `gurobipy` at module level, but nothing imports that module,
  and decision 7 removes it.
- **Checks.** `run_tests_plots.py tests/test_plotter.py`: 20 passed, 0 failed. `grep -rln
  "mixed_plotter|plot_mixed_management" src tests` finds only `utils/mixed_plotter.py`, which Claude removes at the
  commit (section 6).
- **Defects:** none. No change made.
