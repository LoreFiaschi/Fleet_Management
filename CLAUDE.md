# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

- A Python library, `fleet_management`, that schedules a fleet of trains over a time horizon.
  Each train has `L` components. Each component degrades stochastically and may be repaired
  (imperfect maintenance) or replaced.
- The schedule is a mixed-integer program (MIP) built and solved with Gurobi (`gurobipy`). A cell is one
  component of one train, `(i, l)`. Five degradation models exist; one fleet may mix them.
- Public API (`src/fleet_management/__init__.py`, spec section "Public API"): `solve`,
  `sweep_operating_horizons`, `sweep_horizon_grid`, `plot_management`, `plot_horizon_sweep`, `validate`,
  and the Gamma validators (`validate_gamma_schedule` and its relatives). There is no command-line entry point.
- `import fleet_management` works without Gurobi: the modules import `gurobipy`, or
  `degradation_model.base`, only inside the functions that solve.
- `spec/spec.tex` (version 1.0) is the authoritative mathematical specification. Code comments cite its
  sections by name ("Cell-Builder Interface", "Encoding of the Dynamics", "Loop Constraint", "Input File Schema").
- `scoping.tex` (with `scoping.pdf`, `scoping_bib.bib`) is the research scoping document. The spec calls it
  "the project scoping document".
- **Spec and scoping hold different things; each piece of information lives in exactly one of
  them** (user, 26 Oct 2026: *"spec.tex must contain only exactly what is implemented (or going
  to be implemented). It is not verbose, there are no long explanations, no theorems, etc. All
  these stuff must belong to scoping file. There the aim of the project is given, and all the
  background info and choice explanations are provided."*).
  - `spec/spec.tex` is a Software Design Document (SDD; user's choice of title, 26 Oct
    2026): a reader consults it to learn how the code works and to plan improvements. It
    holds only the codebase and its mathematical constructions: inputs, checks,
    variables, rows, precomputed constants and how to compute them, outputs, tools. Short
    statements, equations and tables. No theorems, proofs, worked examples, rationale or
    literature; at most a pointer to the scoping section that justifies a construction.
  - `scoping.tex`: the aim of the project, the background, the literature, the derivations and
    proofs, and the reasons for each choice.
  - Edit the two together: when a construction changes, its specification goes to the spec and
    its justification to the scoping document.
- The merge records are in `merge/` (`decisions.md`, `reports/`) and `lore-vs-main.md`. They are
  archived at the clean-up after the merge; do not rely on them for the current code.

## Commands

```bash
pip install -e ".[dev]"                 # editable install with pytest (needs a Gurobi licence to solve)
pip install -e ".[dev,dashboard]"       # also streamlit, for the dashboards
python -m pytest tests -v               # full suite
python -m pytest tests/<file>.py::<test_name> -v    # one test
python remote_smoke_test.py [--no-pytest] [--dry-run] [--out-dir DIR]
python -m streamlit run src/fleet_management/validator_dashboard.py     # the dashboards
```

- **Where Gurobi is available.** The user's Windows workstation (`DESKTOP-1L5C3I2`) has no Gurobi and the
  package is not installed there. The tests and the smoke test run on the ETH Euler cluster. Read
  `EULER.md` in the working copy of the project (`C:\Users\utente\Documents\GitHub\Fleet_Management`)
  before any work on Euler: access, folders, Slurm, and the user's workflow. A test that does not solve
  (input checks, the reference model, writers) runs anywhere.
- `remote_smoke_test.py` runs the pytest suite, then 12 scenarios with real solves (inputs in
  `examples/smoke/`). Each scenario is solved, checked with `validate` and plotted.
  - `--no-pytest` skips the suite; `--dry-run` loads and checks every scenario input with `load_config`
    and exits (no solve, no file); `--out-dir` sets the output folder (default `smoke_test_output/`).
  - It writes `smoke_test.log`, `summary.json`, `pytest_output.txt`, `plots/`, `results/`, `inputs/`, and
    always exits 0. Read `summary.json` for pass or fail.
- **Euler jobs in `euler/`** (submit with `sbatch`, from a run folder that has `logs/`):

  | Script | What it does |
  |---|---|
  | `env_install.sh`, `env_uninstall.sh` | create and remove the virtual environment `~/fleet_management/venv` (Python 3.12.8, `gurobipy` 13.0.0) |
  | `run_tests.sh [tests/...]` | the pytest suite, or the given targets; log and JUnit file in `logs/` |
  | `run_smoke.sh [--no-pytest]` | `remote_smoke_test.py` |
  | `run_regression.sh` | the six check scripts of `examples/regression/` |
  | `gamma_horizon_convergence.sbatch`, `vbz_progressive_feasibility.sbatch` | the two experiment jobs |

- No linter, formatter or type checker is configured.
- `results/` (except `results/validator_tests/`, fixtures that the validator tests read), `logs/`,
  `smoke_test_output/` and `experiments/**/runs/` are partly or wholly git-ignored. `input/` is tracked.

## Architecture

### Pipeline

`solve()` (`solver.py`) → `read_input` (YAML, JSON or HDF5, in `config.py`) → `solve_mapping` →
`load_config` (normalise to a `FleetConfig`, run the consistency checks) → `degradation_model.base.solve_mixed`
(build, warm start, optimise, extract) → result dict → `_save_results` (format chosen by the file extension).

- `solve_mapping(data, results_path=None, *, warm_start=None, warm_start_mode="start")` is steps 3 to 5; the sweeps
  call it without a file.
- `FleetConfig` (`config.py`) holds every input broadcast to full shape: scalars to `(F, L)`, profiles to
  `(F, L, M, H*)` (`H*` = `H2`, or `H1` for a transitory profile), one selector array per cell (`model`, `bound_method`, `repair_model`, `reliability_impl`,
  `encoding`), the run options, and the derived constants (support constant, natural ranges `range_mu`,
  `range_v`, `range_R`, `range_K`). Keys of other models may hold `null`; they are ignored where they do not apply.
  `REJECTED_KEYS` lists the keys of removed features with their messages.
- The result dict holds numpy arrays; its keys are `RESULT_KEYS` in `solver.py`. A multi-case sweep report is
  separate (see Tools).

### One shared Gurobi model (`degradation_model/base.py`)

- `FleetModel` is the context `ctx` that the builders read and fill: sizes `F, L, M, H1, H2, T`, the shared
  variables, per-cell arrays and the increment accessor `ctx.increment(key, i, j, l, k)`.
- Shared variables: `x[i, j, k]` of shape `(F, M+1, T)` (`j = 0` is the depot day); `x^m` (repair) and `x^r`
  (replacement) binaries; the shared mean `mu`; `z` (damage removed by a repair); `u` (damage penalty).
  The horizon has `T = H1 + H2` days: `H1` days of the transitory phase, `H2` of the operating phase.
- Model-independent rows: `add_assignment_rows` (at most one activity per train and day, each mission covered),
  `add_depot_rows` (repair or replacement only on a depot day), `add_damage_penalty_rows`.
- **Objective** (`build_objective`, `objective_mode`): `total`, `operating_average` (operating cost per day) or
  `evaluation_total` (needs `evaluation_horizon`). `C_M` is paid once per depot day of a train; `C_R` per unit
  of mean removed by a repair; `C_rep` per replacement; `C_D·u` is the damage penalty (`inf_norm` or `quadratic`).
- **Registry of cell builders.** `register_cell_builder(name, builder)`, `get_cell_builder`, `dispatch_cell`.
  A builder has `name` and three hooks:
  - `prepare(ctx, cfg, cells, opts)`: once per model present; creates the variables of the other tracked
    quantities, anchors, starts `y`, selectors `n`, segment binaries `w`; sets `NonConvex = 2` if some cell keeps an
    exact quadratic row.
  - `add_cell(ctx, i, l)`: the rows of one cell.
  - `extract(ctx, cfg, out)`, optional: model-specific result entries.
- Building order: model and shared variables; `prepare` per model (alphabetical); the user's `gurobi_params` again
  (a user `NonConvex` wins); objective and model-independent rows; `add_cell` for every cell.
- The cap functions of Gamma and inverse Gaussian, and `row_constants` of rainflow, need no Gurobi.
  Such a module registers no builder when `gurobipy` is missing.

### The five models (`degradation_model/<model>.py`)

| Model | Tracked quantities | Reliability row |
|---|---|---|
| `gaussian` | `mu`, `v` | `kappa_G^2 v <= (tau - mu)^2`, in the form of `reliability_impl` |
| `inverse_gaussian` | `mu` | linear cap `mu <= mu_bar` (`ig_mean_cap`) |
| `wiener` | `mu`, `v` (grows by `sigma^2` per mission day; only a replacement lowers it) | as Gaussian |
| `gamma` | `mu` | linear cap `mu <= A_1(beta* tau, eps) / beta*` (`sharp_cap`) |
| `rainflow` | `mu` and by `bound_method` one more: `v` (Cantelli, Bernstein), `R` (Hoeffding), `K` (Chernoff); none for Markov | quadratic (Cantelli, Hoeffding, Bernstein) or linear (Markov, Chernoff) |

- Every model accepts both repair laws (`repair_model`, per cell, default `ardinf`):
  - **ARD∞** (`ardinf`): a repair multiplies the state by the repair factor `a`. Factor `a = 1 - rho` for a
    mean and for `K`; `a = (1 - rho)^2` for a variance and for `R`; `a = 1` for the Wiener variance.
  - **ARD1** (`ard1`): a repair removes a fraction `rho` of the degradation since an anchor `g`, the state just
    after the last intervention: `psi = a·s + (1 - a)·g`. Anchors exist for every tracked quantity except the
    Wiener variance. The starting anchors are the inputs `anchor_mu_0` and so on.
  - Repair map `psi`: `base.repair_factor`; the spec section "Repair Laws".
- Replacement: every tracked quantity and anchor takes its replacement value (`replacement_mu`, ...).

### The rows: one-sided, two encodings

- Every dynamics row is one-sided (`>=`): the tracked states are upper bounds on the true moments. Replacement and
  anchor rows are equalities. `base.add_quantity_rows` builds, per tracked quantity, the start rows, the rows of
  every day and the loop rows.
- `encoding = bigm` (default): big-M rows, with `M_s` the natural range of the quantity (`tau` for a mean).
  `encoding = indicator`: one binary `n` per cell and day (`n + x^m + x^r = 1`) and Gurobi indicator constraints.
  At binary points the two forms have the same feasible set.
- **Linear forms of the quadratic reliability rows** (`reliability_impl`): `exact` (nonconvex quadratic),
  `tangent` (one tangent at `tangent_ref·tau`), `pwl` (`pwl_points` segments with binaries `w`). Linear rows
  ignore the value. Spec section "Linear Forms of the Reliability Rows".
- **Start of the repetition and loop rows.** `y^s` is the value of quantity `s` at the start of every repetition
  of the operating phase; `y^{g,s}` is that of its anchor. The rows of day `H1 + 1` read `y` instead of the
  state of day `H1`. Start rows: `y >= s[H1]`. Loop rows: `s[T] <= y`, and the same for the anchors. Every tracked
  quantity has its own rows. There is no soft loop penalty. Spec section "Loop Constraint".
- Consequences: an ARD1 cell with `rho < 1` that flies a mission with a positive increment in the operating phase
  needs a replacement there; so does a Wiener cell. No input check enforces this.

### Tools

- **Horizon sweeps** (`horizon_sweep.py`): `sweep_operating_horizons` varies `H2` for one `H1`
  (cost gradient, early stop, checkpointed report, optional warm starts and parallel cases);
  `sweep_horizon_grid` runs one operating sweep per `H1`. A failed case is logged in its row and the sweep goes on.
  Parallel workers use `spawn`.
- **Warm starts** (`base.apply_binary_warm_start`, `complete_binary_warm_start`, `apply_relaxation_warm_start`):
  `warm_start` is the result of an earlier solve; `warm_start_mode` is `start`, `hint` or `both`.
  `greedy_warm_start.build_greedy_warm_start` builds a source from the input (a construction pass, then a loop
  pass); it is not exported.
- **Plots** (`utils/plotter.py`): `plot_management`, `plot_horizon_sweep`.
- **Validators** (`validation/`, none calls the optimiser): `validate` (structural), the dashboards
  (`validator_dashboard.py`, `dashboard_inspector.py`, `dashboard_gamma.py`), and the Gamma validators in
  `degradation_model/gamma_utils/` (replay, exact tail, Monte Carlo, dispatcher).
  - **Tolerance rule** (`validation/tolerance.py`; spec section "Tolerance of the checks"): the explicit argument, else the key `tolerance` of the input
    block `validation`, else a default per check from `FeasibilityTol` and `IntFeasTol` of `gurobi_params`
    (Gurobi's defaults `1e-6` and `1e-5`). A state, anchor, start, `z` or cap uses
    `FeasibilityTol + IntFeasTol·M_s`; a binary uses `IntFeasTol`; a probability or a cost uses `FeasibilityTol`.
- **Result files:** YAML, JSON or HDF5, chosen by the extension; the layout is in the spec ("Result Files",
  "HDF5 Layout"). `solver.read_result` reads any of them.

### How to add a model

The steps of the spec section "Cell-Builder Interface":
1. A module `degradation_model/<name>.py` with a class that has `name`, `prepare`, `add_cell` and, if needed,
   `extract`, built from the helpers of `base` (`add_state_vars`, `add_selector_vars`, `add_pwl_vars`;
   `add_selector_rows`, `add_quantity_rows`, `add_upper_cap_rows` or `add_quadratic_reliability_rows`,
   `add_repair_cost_rows`). It must define `ctx.mu_var[i, l, k]` for every `k`.
2. `base.register_cell_builder(name, builder)` at the end of the module (`base._load_builders` imports the module
   `degradation_model/<name>.py` by the model name) and the name in `SUPPORTED_MODELS` of `config.py`.
3. Its input checks and field parsing in `config.py`, and its natural ranges.
4. Its section in `spec/spec.tex` and, in step with it, the justification in `scoping.tex`.
5. Tests: a spec test, and the model in `tests/reference_model.py` (see below).

## The user's standing instructions for this project

- **Spec and scoping:** see the quoted rule in "What this project is" (user, 26 Oct 2026).
- **Euler:** read `EULER.md` before any work on Euler (user's workflow, 7 Nov 2026). Sonnet agents do the work
  there; the main session plans, submits jobs and checks results.
- **Agents:** the global rule `~/.claude/rules/subagents.md` holds the choice of models, the briefs and the limits
  (Sonnet for work on code with a detailed brief; never Haiku or a Fable model; no sub-agents of sub-agents).
- **Removals** go to the Recycle Bin, never a hard delete (global rules `~/.claude/rules/documents.md` and
  `windows-workstation.md`).

## Other files

| Path | Content |
|---|---|
| `Things to change.txt` | the user's list of planned changes, the project's to-do list |
| `scoping.tex`, `scoping.pdf`, `scoping_bib.bib` | the research scoping document |
| `spec/spec.tex`, `spec/references.bib` | the Software Design Document |
| `examples/smoke/` | inputs of the 12 smoke-test scenarios |
| `examples/regression/` | six check scripts (run by `euler/run_regression.sh`), reference inputs, scripts of the two experiment jobs, `RegressionREADME.md` |
| `experiments/` | experiments: inputs, runners, job scripts, one README each |
| `euler/` | Slurm jobs and environment scripts of the ETH Euler cluster |
| `tests/` | the pytest suite; `tests/data/` holds its fixtures |
| `tests/reference_model.py` | the reference model ("oracle"): an independent numpy and scipy re-implementation of what a schedule means under the spec; the `test_oracle_*` files compare the package with it. It uses only `config.load_config` and `read_input` from the package |
| `remote_smoke_test.py` | the smoke test (see Commands) |
| `EULER.md` | in the working copy of the project only (not in this branch): the user's guide to the Euler cluster |
| `ASSESSMENT.md` | a gap analysis from July 2026, written before all five models existed; superseded |
| `merge/`, `lore-vs-main.md` | merge records; archived at the clean-up |
