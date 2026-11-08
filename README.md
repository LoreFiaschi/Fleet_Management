# Fleet Management

`fleet_management` is a Python package that schedules a fleet of trains over a time horizon.
Each train has several components. Each component degrades at random. On a depot day, a component
can be repaired (imperfect repair) or replaced. The package builds a mixed-integer program and
solves it with Gurobi. The schedule keeps the failure probability of every component below a
given level on every day, also on every later repetition of the operating phase.

## Purpose and status

- **Source of truth.** `spec/spec.tex` is the specification of the formulation and of the tools.
  Every change is written there first, and the code follows it.
- **Reasons.** `scoping.tex` (with `scoping_bib.bib` and the build `scoping.pdf`) is the project
  scoping document. It gives the aim of the project, the background, the literature and the
  reasons for each choice.
- **This file** is a user guide. It summarises the spec. Where this file and the spec differ, the
  spec is right.

What the package does:

| Item | Content |
|---|---|
| Fleet | `F` trains, `M` missions per day, `L` components per train |
| Horizon | `T = H1 + H2` days: a transitory phase of `H1` days, then the operating phase of `H2` days, which is repeated |
| Degradation models | Gaussian, inverse Gaussian (IG), Wiener, Gamma, rainflow; one model per cell (a cell is one component of one train) |
| Repair laws | ARD∞ (`ardinf`, the default) and ARD1 (`ard1`) |
| Goal | minimise the cost of depot days, repairs and replacements, plus a penalty on the largest train damage |
| Problem class | a mixed-integer linear program (MILP) when every reliability row is linear and the damage penalty is `inf_norm` or absent; otherwise a mixed-integer quadratically constrained program (MIQCP), which is nonconvex when a quadratic reliability row has the form `exact` |
| Input and output files | YAML, JSON or HDF5 |

## Prerequisites

- Python 3.10 or newer.
- [Gurobi](https://www.gurobi.com/) with a licence. The licence that comes with the `gurobipy`
  pip package is free but limited to small models. Most tests and
  all real solves need a licence that fits the model size.

## Installation

From the root of the repository:

```bash
pip install -e ".[dev]"
```

| Extra | Adds | Needed for |
|---|---|---|
| `dev` | `pytest` | the test suite |
| `dashboard` | `streamlit` | the dashboards |

For both extras: `pip install -e ".[dev,dashboard]"`.

`import fleet_management` works without Gurobi. Only the functions that solve need `gurobipy`.

## Quick start

The input `examples/smoke/mixed_example.yaml` is the example of the spec. It has three trains,
two missions and two components. The six cells use the models Gaussian, inverse Gaussian, Wiener,
Gamma, rainflow with the Bernstein bound and rainflow with the Chernoff bound.

```python
from fleet_management import plot_management, solve, validate

result = solve("examples/smoke/mixed_example.yaml", "mixed_result.yaml")
print(result["status"], result["objective"])

plot_management("mixed_result.yaml", "mixed_result.png")
report = validate("examples/smoke/mixed_example.yaml", "mixed_result.yaml", "mixed_validation.yaml")
print(report["passed"])
```

- `solve` returns the result dictionary and writes it to `mixed_result.yaml`.
- `plot_management` reads the result file and writes the schedule plot.
- `validate` checks the saved result against the input and writes a report.

## API reference

All names below are exported by `fleet_management` (`src/fleet_management/__init__.py`).
Each section of the spec named in brackets gives the full rules, arguments and errors.

### Solve

```python
solve(input_path: str, results_path: str | None = 'output.yaml', *,
      warm_start: dict | None = None, warm_start_mode: str = 'start') -> dict
```

Reads the input file, checks it, builds the model, solves it, writes the result file and returns
the result dictionary. `results_path=None` writes no file. The directory of `results_path` must
exist. [Data Flow of One Solve]

### Horizon sweeps

```python
sweep_operating_horizons(input_path, operating_horizons, *, output_path=None,
                         stop_on_gradient=False, gradient_tolerance=0.001,
                         flat_gradients_required=2, minimum_cases=3,
                         maximum_mip_gap_for_stopping=None, warm_start=False,
                         warm_start_mode='start', n_workers=1,
                         objective_mode='operating_average', evaluation_horizon=None) -> dict

sweep_horizon_grid(input_path, transitory_horizons, operating_horizons, *,
                   output_path=None, warm_start=True, evaluation_horizon=None,
                   **sweep_options) -> dict
```

- `sweep_operating_horizons` solves one input for several values of `H2` with `H1` fixed, and
  ranks the cases. [Horizon Sweeps]
- `sweep_horizon_grid` does the same for a grid of `H1` and `H2`. [Horizon Sweeps]

### Plots

```python
plot_management(input_file_path: str, plot_file_path: str | None = None) -> None
plot_horizon_sweep(sweep_report_path, plot_file_path: str | None = None) -> None
```

- `plot_management` draws the schedule of a result file: one row per cell, one column per day, the
  tracked mean as colour, the activity as a mark, and a summary panel. The default output is
  `output.png`; the format is `.png` or `.pdf`. [Plotting API and Layout]
- `plot_horizon_sweep` draws the cases of one or more sweep reports over `H2`. The default output
  is `horizon_sweep.png`. [Plotting API and Layout]

### Validation

```python
validate(input_path: str, results_path: str, validation_path: str = 'validation.yaml',
         tol: float | None = None) -> dict

validate_gamma_replay_schedule(cfg, result, *, tolerance=None, raise_on_failure=False) -> dict
validate_gamma_replay_files(input_path, result_path, report_path=None, **kwargs) -> dict

validate_gamma_stochastic_schedule(cfg, result, *, repetitions=100000, random_seed=20260908,
                                   maximum_schedule_failure_rate=None, confidence_level=0.95,
                                   batch_size=20000, tolerance=None,
                                   raise_on_failure=False) -> dict

validate_gamma_schedule(cfg, result, *, mode=None, repetitions=None, random_seed=None,
                        maximum_schedule_failure_rate=None, confidence_level=None,
                        batch_size=None, tolerance=None, raise_on_failure=False) -> dict
validate_gamma_schedule_files(input_path, result_path, report_path=None, **kwargs) -> dict
```

- `validate` is the structural validator. [Validation Tools]
- `validate_gamma_replay_schedule` and `validate_gamma_replay_files` are the replay validator of
  the Gamma model (a schedule given as `cfg` and `result`, or as files).
- `validate_gamma_stochastic_schedule` is the Monte Carlo validator of the Gamma model.
- `validate_gamma_schedule` runs the validators that `mode` selects (`"deterministic"`,
  `"stochastic"` or `"both"`). `validate_gamma_schedule_files` is its file version.
- `cfg` is the normalised input: `fleet_management.config.load_config(fleet_management.config.read_input(path))`.

Not exported: `solve_mapping` and `load_config` (internal entry points), the greedy warm-start
heuristic `build_greedy_warm_start`, and the exact Gamma validator
`validate_gamma_tail_bound_schedule`.

## Input file

An input is one mapping from key names to values. The formats are YAML (`.yaml`, `.yml`), JSON
(`.json`) and HDF5 (`.h5`, `.hdf5`); the extension chooses the format.

- A **cell value** is a scalar, a list of `L` numbers (one per component) or an `F x L` array.
  A **selector** has the same shapes with strings.
- A **profile** (increments per mission and day) has the shape `(F, L, M, H2)` after broadcasting
  (`H1` days for a transitory profile); shorter shapes are expanded.
- A key outside the tables of the spec raises an error.

| Group | Keys (defaults in brackets) |
|---|---|
| Sizes and horizon | `F`, `M`, `H` (an integer, or `[H1, H2]`), `L` (inferred from `model`), `component_names` |
| Selectors | `model` (required), `repair_model` (`ardinf`), `bound_method` (`cantelli`, rainflow cells only), `reliability_impl` (`exact`), `encoding` (`bigm`) |
| Reliability | `tau`, `epsilon`, `tangent_ref` (0.5), `pwl_points` (8) |
| Repair | `rho`, `anchor_mu_0`, `anchor_v_0`, `anchor_R_0`, `anchor_K_0` (`ard1` cells) |
| Initial state | `mu_0`, `v_0`, `support_0`, `R_0`, `cgf_0` |
| Replacement | `replacement_mu`, `replacement_v`, `support_new`, `R_new`, `cgf_new`, `replacement_as_new` (false), `allow_replacement` (true) |
| Increment profiles | `mu`, `v`, `support`, `cgf`, `gamma_beta`, and the transitory versions `mu_trans`, `v_trans`, `support_trans`, `cgf_trans`, `gamma_beta_trans` |
| Model constants | `sigma` (Wiener), `eta` (inverse Gaussian), `s_chernoff` (Chernoff), `v_max_user`, `gamma_beta_0`, `gamma_beta_new` |
| Costs | `C_M`, `C_R`, `C_rep`, `C_D`, `penalty_type` (`inf_norm` or `quadratic`) |
| Run options | `verbose` (1), `mip_gap` (0.1), `time_limit` (3600 s), `fast` (false), `gurobi_params`, `objective_mode` (`total`), `evaluation_horizon`, `progress_interval_seconds` (5.0), `relaxation_warm_start` (false), `relaxation_time_limit` (60.0), `validation` |

Which keys a cell needs depends on its model and bound. The full rules are in the spec:

- **Input Specification**: every key, its shape, its default and the cells it applies to.
- **Consistency Checks**: the checks that the solver runs before it builds the model. A violation
  raises a `ValueError` that names the key and the cell.
- **Input File Schema**: the file formats, the layout of an HDF5 file, the keys by cell type, and a
  YAML example (also in `examples/smoke/mixed_example.yaml`).

Old key names (for example `maintenance_type`, `tail_bound`, `formulation`, `beta`, `C_S`, `alpha`)
are rejected with a message that names the new key. The table is in the section **Rejected Keys**
of the spec.

## Output

`solve` returns one result dictionary. Arrays are `numpy` arrays. If Gurobi found no solution, the
solution arrays and the cost report are `null`.

| Group | Keys |
|---|---|
| Status and quality | `status`, `objective`, `mip_gap`, `bound` |
| Problem | `F`, `M`, `L`, `H1`, `H2`, `T`, `H`, `model`, `component_names`, `tau`, `mu_0`, `v_0`, `bound_method`, `repair_model`, `reliability_impl`, `encoding`, `objective_mode`, `evaluation_horizon` |
| Schedule | `x` (assignment, shape `F x (M+1) x T`; index 0 is the depot), `m` (repair), `r` (replacement), `idle` |
| Tracked values | `mu`, `v`, `R`, `K`, `z`, the anchors `g_mu`, `g_v`, `g_R`, `g_K`, the starts of the repetition `y_mu`, `y_v`, `y_R`, `y_K`, `y_g_mu`, `y_g_v`, `y_g_R`, `y_g_K`, and `u` |
| Cost report | `step_costs`, `J_initialization`, `J_op`, `J_op_average`, `peak_damage`, `damage_penalty`, `operating_objective`, `J_total`, `projected_evaluation_cost`, `evaluation_objective`, `component_costs` |
| Reports | `performance`, `optimization_progress`, `warm_start`, `relaxation_warm_start` |

Facts about the values:

- **Tracked values are upper bounds.** `mu`, `v`, `R`, `K`, the anchors, the starts `y_*` and `z`
  hold the tracked variables of the model. The dynamics rows are one-sided, so these values are
  upper bounds on the true moments, not the moments themselves.
- **`x`, `m` and `r` are rounded** to the nearest integer, so each entry is exactly 0 or 1.
  Gurobi returns binary values only within its integrality tolerance. `idle` and the cost report
  are computed from the rounded values.
- A key that does not apply to a cell holds `NaN` there (for example `v` at an inverse Gaussian
  cell).

**Result files.** `results_path` selects the format by its extension: `.yaml` or `.yml` (YAML),
`.json` (JSON), `.h5` or `.hdf5` (HDF5). A path without extension gets `.yaml`. In YAML and JSON,
`NaN` and infinite values are written as `null`. In HDF5, arrays are `float64` datasets that keep
`NaN`; lists, mappings and records are JSON text in attributes. `read_result(path)` in
`fleet_management.solver` reads a result file of any format. The spec sections **Output
Specification** and **Result Files** give every key and its shape.

## Formulation in brief

The spec has the equations. This section gives the structure and the pointers.

**Model, per cell.** The input `model` selects the degradation model of each cell:

| Model | Tracked values | Reliability row, on every day `k` | Spec section |
|---|---|---|---|
| `gaussian` | mean `mu`, variance `v` | `kG² v <= (tau - mu)²` and `mu <= tau`, with `kG = Phi^-1(1 - epsilon)` and `epsilon < 0.5` (nonconvex quadratic in the form `exact`) | Gaussian Model |
| `inverse_gaussian` | `mu` | `mu <= tau / qbar`, a cap computed once per cell from the inverse Gaussian law (`qbar` is the mean of the law above its `(1 - epsilon)`-quantile); linear | Inverse Gaussian Model |
| `wiener` | `mu`, `v` (grows by `sigma²` per mission; falls only at a replacement) | the Gaussian row | Wiener Model |
| `gamma` | `mu` | `mu <= A1(beta* tau, epsilon) / beta*`, a cap computed once per cell (`beta*` is the smallest rate of the cell, `A1` the sharp cap of the Gamma tail); linear | Gamma Model |
| `rainflow` | `mu`, plus at most one quantity that the bound needs | one of the five bounds below | Rainflow Model |

**Rainflow tail bounds** (`bound_method`, per cell; `kappa = ln(1/epsilon)`). The increment law
enters only through summary statistics. All rows are in the spec section Rainflow Model, subsection
Reliability rows.

| `bound_method` | Row | Needs | Row type |
|---|---|---|---|
| `markov` | `mu <= epsilon tau` | mean | linear |
| `cantelli` (default) | `kC² v <= (tau - mu)²`, `mu <= tau`, with `kC² = (1 - epsilon)/epsilon` | mean, variance `v` | quadratic |
| `hoeffding` | `(tau - mu)² >= (kappa/2) R`, `mu <= tau` (`R` is a budget of squared widths) | mean, widths `support` | quadratic |
| `bernstein` | `(tau - c - mu)² >= c² + 2 kappa v`, `mu <= tau - c`, with `c = kappa b / 3` and `b` the largest width | mean, variance, widths | quadratic |
| `chernoff` | `K - s tau <= ln(epsilon)` (`K` is the tracked cumulant generating function at `s_chernoff`) | cumulant generating function | linear |

**Repair laws** (`repair_model`; spec section Repair Laws). With `D-` the degradation before a
repair and `D+` after it, and `rho` the repair efficiency:

| Law | Name in the input | Rule |
|---|---|---|
| ARD∞ (arithmetic reduction of degradation, infinite memory) | `ardinf` | `D+ = (1 - rho) D-` |
| ARD1 (arithmetic reduction of degradation, memory 1) | `ard1` | `D+ = D- - rho (D- - D_last)`, with `D_last` the level just after the last intervention |

Every model accepts both laws. Under `ard1`, each tracked quantity has an anchor (the value after
the last intervention).

**Encodings** (`encoding`; spec section Encoding of the Dynamics). Both encodings have the same
feasible set at binary points.

| Encoding | Meaning |
|---|---|
| `bigm` (default) | one-sided big-M rows; the constant of each quantity is its natural range |
| `indicator` | one extra binary per cell and day, and Gurobi indicator constraints |

**Forms of the quadratic reliability rows** (`reliability_impl`; spec section Linear Forms of the
Reliability Rows). They apply to the Gaussian, Wiener, Cantelli, Hoeffding and Bernstein rows. A
linear row is always built as it is, and the value is ignored.

| `reliability_impl` | Row | Class |
|---|---|---|
| `exact` (default) | the quadratic row as written | nonconvex MIQCP (`NonConvex = 2`) |
| `tangent` | one tangent line at `tangent_ref * tau` | MILP; its feasible set lies inside the exact one |
| `pwl` | `pwl_points` tangent lines on equal segments of `[0, tau]`, one binary per segment | MILP; inside the exact set; closer to it as `pwl_points` grows |

**Loop rows** (spec section Loop Constraint). The variable `y` is the value of a tracked quantity
at the start of every repetition of the operating phase. For each cell and each tracked
quantity `s` (and each anchor, under `ard1`) there are two hard rows:

- the start row `y >= s` at day `H1`;
- the loop row `s <= y` at the last day `T`.

Day `H1 + 1` reads `y` instead of the value of day `H1`. With these rows every day of every
repetition satisfies the reliability requirement. As a consequence, a Wiener cell, and an `ard1`
cell with `rho < 1` that gains damage in the operating phase, need at least one replacement there.

## Dashboards

The dashboards are Streamlit pages. They need the extra `dashboard`.

```bash
python -m streamlit run src/fleet_management/validator_dashboard.py
```

- In the sidebar, choose the mode, the input file, the result file and the tolerance. Press
  "Load". The mode and the tolerance are read when "Load" is pressed.
- **Result inspector** (every model): overview, reliability margins, repeatability (the loop rows),
  train timeline, heatmap, component comparison, the checks of `validate`, and the raw data with a
  CSV download.
- **Gamma diagnostic** (Gamma cells only): the replay validator, the tracked mean against its cap,
  and the exact tail probability of the true damage against `epsilon`.

The spec section **Dashboards** has the details.

## Validation

The validators check a saved result against its input. They do not call the optimiser.

| Tool | What it checks |
|---|---|
| `validate` | `x`, `m`, `r` are binary; at most one activity per train and day; every mission is covered once a day; repairs and replacements only on depot days; `idle` is consistent; the loop rows hold for each saved tracked quantity; the saved objective is not below the objective recomputed from the schedule |
| `validate_gamma_replay_schedule` | the global schedule rules; the tracked mean, the anchor, the removed mean `z`, the cap and the repeatability of every Gamma cell, replayed from the saved schedule |
| `validate_gamma_stochastic_schedule` | a Monte Carlo run of the fixed schedule on the Gamma cells: schedule failure rate with a one-sided Clopper-Pearson upper bound |
| `validate_gamma_schedule` | the dispatcher: mode `"deterministic"` (replay, the default), `"stochastic"` or `"both"` |

The exact Gamma validator (`validate_gamma_tail_bound_schedule`, not exported) computes the exact
tail probability of the true damage by the Moschopoulos series and compares it with `epsilon`.

**Tolerance of a check.** The tolerance of each check is resolved in this order:

1. the argument (`tol` or `tolerance`), if it is not `None`: one value for every check;
2. else the key `validation.tolerance` of the input: one value for every check;
3. else a default from the Gurobi tolerances of the input (`FeasibilityTol` and `IntFeasTol` in
   `gurobi_params`, matched without regard to case; if absent, Gurobi's defaults 1e-6 and 1e-5):

| Check on | Default tolerance |
|---|---|
| a tracked quantity, an anchor, a start value, `z`, the cap of a cell | `FeasibilityTol + IntFeasTol x M_s`, with `M_s` the natural range of the quantity at the cell |
| a binary | `IntFeasTol` |
| a probability or a cost | `FeasibilityTol` |

The reason is this: a binary that Gurobi accepts within `IntFeasTol` inside a big-M row can move a
tracked state by `IntFeasTol x M_s`. A check over several cells passes when no cell exceeds its own
tolerance. The dashboards pass the tolerance of their sidebar as the argument. The spec section
**Tolerance of the checks** has the details.

## Horizon sweeps and warm starts

**Horizon sweeps.** `sweep_operating_horizons` solves one input for a list of `H2` values with `H1`
fixed; `sweep_horizon_grid` runs it for each `H1` of a list. Each case is solved with the input
data and the horizons of the case. The cases are ranked by the cost per operating day (or by the
projected cost over an evaluation horizon `E`, in the grid). With `stop_on_gradient=True` a
sweep stops early when the cost per operating day rises, or stays flat, over consecutive cases.
With `n_workers > 1` the cases run in parallel processes, each with its own Gurobi licence token;
this is not allowed together with `warm_start` or `stop_on_gradient`. An error inside a case is
recorded in the row of that case, and the sweep goes on. The report is YAML.

**Warm starts.** A warm start gives Gurobi values for the binary variables. `solve` takes a
`warm_start` source (a result dictionary, or the dictionary of the greedy heuristic) and a
`warm_start_mode`: `"start"` (the attribute `Start`, the default), `"hint"` (the attribute
`VarHintVal`) or `"both"`. The source may have other dimensions than the problem; its days are
mapped to the days of the problem. The run option `relaxation_warm_start` computes a start from a
relaxed copy of the model when no other warm start applies. In a sweep, `warm_start=True` uses the
nearest earlier case with the same `H1` that has a solution as the source. The greedy heuristic
`build_greedy_warm_start` (module `fleet_management.greedy_warm_start`) builds a source from the
input. The result keys `warm_start` and `relaxation_warm_start` report what each mechanism did.
The spec section **Warm Starts** has the rules.

## Project structure

```
Fleet_Management/
  pyproject.toml            Python >= 3.10; extras "dev" (pytest) and "dashboard" (streamlit)
  README.md
  scoping.tex               the project scoping document, with
  scoping_bib.bib           its bibliography (scoping.pdf: its build)
  input/                    input file of the Euler convergence job
  spec/                     the specification: spec.tex, references.bib
  remote_smoke_test.py      smoke test: real solves of the scenarios in examples/smoke/
  examples/                 example inputs, the smoke-test inputs and regression scripts
  tests/                    pytest suite
  experiments/              experiments: inputs, runners, job scripts
  results/validator_tests/  fixtures of the validator tests (test data)
  euler/                    Slurm jobs: env_install.sh, env_uninstall.sh, run_tests.sh,
                            run_smoke.sh, run_regression.sh, and two experiment jobs
  src/fleet_management/
    __init__.py             public API
    config.py               reader, checks, broadcasting, derived constants, load_config
    solver.py               solve(): checks, dispatch, result files
    horizon_sweep.py        sweep_operating_horizons, sweep_horizon_grid
    greedy_warm_start.py    build_greedy_warm_start
    validator_dashboard.py  Streamlit entry point of the dashboards
    degradation_model/
      base.py               shared model, registry, objective, warm starts, result extraction
      gaussian.py, inverse_gaussian.py, wiener.py, gamma.py, rainflow.py
                            cell builders (and the caps of the IG and Gamma models)
      gamma_utils/
        gamma_replay_validator.py      replay validator
        gamma_stochastic_validator.py  Monte Carlo validator
        gamma_tail_validator.py        exact validator
        gamma_tail_bound.py            moschopoulos_tail_probability
    utils/plotter.py        plot_management, plot_horizon_sweep
    validation/
      validator.py          validate
      tolerance.py          the tolerance of the checks
      dashboard_inspector.py, dashboard_gamma.py
```

The spec section **Package Layout** describes each file. `examples/regression/` holds the regression
check scripts and the scripts that the two experiment jobs of `euler/` run.

## Running the tests

**Pytest suite.** From the root of the repository, after `pip install -e ".[dev]"`:

```bash
python -m pytest tests -v
python -m pytest tests/test_config.py -v      # one file
```

Most tests build or solve a model, so they need Gurobi with a licence.

**Smoke test.** `remote_smoke_test.py` runs the pytest suite and then 12 scenarios with real
Gurobi solves. The inputs are in `examples/smoke/`. Each scenario is solved, checked with `validate`
and plotted.

```bash
python remote_smoke_test.py [--no-pytest] [--dry-run] [--out-dir DIR]
```

| Option | Effect |
|---|---|
| `--no-pytest` | skip the pytest suite |
| `--dry-run` | load and check every scenario input with `load_config`, then exit; no solve, no file written, no Gurobi needed |
| `--out-dir DIR` | output folder (default `./smoke_test_output`) |

The script writes its log, `summary.json` (pass or fail and seconds per scenario, and the pytest
result), the pytest output, the plots, the results and the validation reports into the output
folder. It always exits with code 0: read `summary.json`.

**Job scripts for a Slurm cluster.** The folder `euler/` has Slurm job scripts. Submit them from
the root of the repository, with a folder `logs/` present:

| Script | Job |
|---|---|
| `euler/env_install.sh` | installs the Python environment with `gurobipy` and the package |
| `euler/run_tests.sh` | runs the pytest suite (arguments select test files) |
| `euler/run_smoke.sh` | runs `remote_smoke_test.py` (arguments go to the script) |
| `euler/run_regression.sh` | runs the regression check scripts of `examples/regression/` one by one |

The scripts load the modules and the environment path of the ETH Euler cluster. On another cluster,
change the lines `module load` and the variable `VENV` at the top of each script.

## Further documentation

- `spec/spec.tex`: the specification of the formulation and the tools (the source of truth).
- `scoping.tex` (build: `scoping.pdf`): the project scoping document with the aim, the background,
  the literature and the reasons for each choice.
- `examples/regression/RegressionREADME.md`: the regression checks and the experiment runners.
