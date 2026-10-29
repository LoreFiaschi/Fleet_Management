Status: done. Steps A1 to A12 finished; 18 tests pass, 1 fails as the plan expects (`test_public_api_without_gurobi`, until Task B). Next: Task B (`__init__.py`), by another agent.

# Task A progress

Change from the plan (interface decision 1): the schedule plot reads its result file with `solver.read_result`, not with `config.read_input`. The sweep plot keeps `config.read_input`.

Files written: `src/fleet_management/utils/plotter.py` (CRLF, 774 lines) and `tests/test_plotter.py` (LF, 437 lines plus the final newline; 438 line breaks). `utils/mixed_plotter.py` and `__init__.py` are untouched.

Method: `transplant.py` (this folder) copies whole top-level blocks of `proto_plotter.py` into `plotter.py` verbatim, CRLF kept. `build_tests.py` (this folder) copies line ranges of `proto_test_plotter.py`. The reader change was then made with Edit.

| Step | Check | Result |
|---|---|---|
| A1 copy | cmp | identical (CRLF kept) |
| A2 header, `plot_management`, `plot_horizon_sweep` | diff of lines 1-112 against the prototype | only the three reader differences (below) |
| A3 readers | grep `presolved_performance\|presolved_model\|h5py\|json\.\|yaml\.` | empty once A4 was done (at the end of A3 three hits remained in the old `_normalise_solution`, which A4 replaces) |
| A4 `_normalise_solution` | grep `alpha\|_legacy_h1\|_cell_grid\|gamma_formulation\|model_assignment` | only `"alpha": 0.88` of the mark boxes (the `model_assignment` hit was in the old `_model_grid`, gone after A6) |
| A5 `_draw_solution`, `_component_action_label` | block text from the prototype | done; covered by the whole-file diff at A7 |
| A6 summary and row helpers | grep `Vehicle\|physical\|initialization\|Remaining` | empty |
| A7 sweep plot, whole file | `diff --strip-trailing-cr` against `proto_plotter.py` | 7 differing lines, all from the reader change; CRLF count 774 = LF count 774; undefined-name check: 0 outside and 0 inside the Gamma block |
| A8 tests part 1 | `run_tests_plots.py tests/test_plotter.py test_schedule` | 6 passed |
| A9 tests part 2 | `... test_plot_management` / `... test_parse` | 7 passed / 1 passed |
| A10 tests part 3 | `diff proto_test_plotter.py tests/test_plotter.py`; whole module | two differing places (below); 18 passed, 1 failed (`test_public_api_without_gurobi`) |
| A11 render | `render.py` into `render_task_a`, looked at the three PNG files; `compare_png.py` against `render_out` | all three images are what section 3 of the plan requires (below); pixel-identical to the planner's images (0 differing pixels in each) |

## Differences from the prototype (all caused by decision 1)

`plotter.py`:
1. Module docstring, lines 5-6: "A result file is read with ``solver.read_result`` and a sweep report with ``config.read_input``." (the prototype names `config.read_input` for both).
2. New import line 23: `from fleet_management.solver import read_result`. `read_input` stays, for the sweep reports.
3. `plot_management`, line 61: `read_result(input_file)` instead of `read_input(input_file)`.

`tests/test_plotter.py`:
1. Line 26: `from fleet_management import solver` (the prototype also imports `config`, which is unused once the reader changes).
2. Line 100, helper `_schedule_figure`: `solver.read_result(path)` instead of `config.read_input(path)`.

## Extra check of decision 1 (`probe_nan.py`, this folder)

A result with a `null` in `mu` and in `x`, written as YAML, JSON and HDF5, is read back by `read_result` (float arrays, NaN) and plotted by `plot_management` without error; a PNG is written for each format. The fixture's `v` and `v_0` already carry NaN through the real writers and readers in every test. No code change was needed: `_normalise_solution` converts with `np.asarray(..., dtype=float)`, and a NaN cell gets the colour map's "bad" colour (transparent) and no mark of its own.

## What the images show (`render_task_a\*.png`)

- `schedule.png`: four rows "Train 1 — Wheelset / gaussian", "Train 1 — Brake pad / rainflow", "Train 2 — Wheelset / gaussian", "Train 2 — Brake pad / rainflow"; a heavier line between train 1 and train 2; columns "initial", 1, 2, 3, 4 and axis label "Day"; a blue dashed line between days 2 and 3 with "transitory" and "operating" above the grid; the 16 marks M1 I M1 R / M1 I M1 P / R M1 D M1 / D M1 P M1 as in the table; colours go from green to red and grow along each row; colour bar "tracked mean / failure threshold"; legend of five marks under the grid (P replacement, R repair, M_j mission j, D depot no intervention, I idle train); panel "Run summary" with 18 lines (Status to Replacements) and "time limit" in amber. Matches the plan.
- `sweep.png`: title "Operating-horizon sweep (F=4, M=1, L=1, H1=4)"; panel "Operating objective" with blue proven points 0.911 (H2=4) and 0.798 (H2=8), an orange plus "best feasible" at H2=12 labelled 0.618, a red x at H2=16 on the lower axis; panel "MIP gap" with a green dashed line "5% stopping gap" and a 10.9% point at H2=12; panel "Formulation growth" with three lines. Matches the plan.
- `sweep_requested.png`: as `sweep.png`, but panel "MIP gap" has three short green marks at 1 % (H2=4, 8, 12) labelled "requested MIP gap", and no dashed line. Matches the plan. The marks at H2=4 and H2=8 overlap the small "0.0%" labels of the proven points (cosmetic; the planner's prototype is identical).

## Problems

1. `test_public_api_without_gurobi` fails, as the plan expects until Task B: `__init__.py` still exports `plot_mixed_management` and the old names. Today it fails earlier for a second reason that is not Task A's: another agent is editing `degradation_model/gamma_utils/gamma_replay_validator.py`, and `import fleet_management` stops with an `ImportError` (first `validate_gamma_replay_files`, in a later run `_phase`). Task B must run after that agent has finished.
2. The plan's check for A3 (grep for the removed key names) cannot be empty at the end of A3, because the old `_normalise_solution` still holds them until A4. It is empty after A4.
3. Not covered by the plan: `_normalise_solution` does not reject a NaN in `tau` (`np.any(tau <= 0)` is false for NaN), so such a result plots without colours instead of raising. The spec lists no such error; no change made.
