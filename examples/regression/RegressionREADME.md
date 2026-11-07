# Regression and experiment scripts

This directory contains small checks and command-line runners for the
fleet-management package. They are kept outside the package because they are
executable scenarios, not public functions.

The unit and spec tests are in `tests/` (run `pytest tests -v`). The scripts
here add three things:

1. numerical checks that need only NumPy and SciPy;
2. checks that solve small inputs with Gurobi and validate the result;
3. runners for the horizon sweeps and the feasibility ladder, used by the Euler
   job scripts under `euler/` and `experiments/`.

The scenarios are small. Passing them establishes consistency of the
implementation. It does not establish scalability or physical validity for a
real fleet.

## Running checks

Run commands from the repository root with the project environment activated.

```powershell
python .\examples\regression\check_matched_ard_inputs.py
```

Run every `check_*.py` script in name order and stop at the first failure:

```powershell
$checks = Get-ChildItem .\examples\regression\check_*.py | Sort-Object Name
foreach ($check in $checks) {
    Write-Host "`n=== $($check.Name) ==="
    python $check.FullName
    if ($LASTEXITCODE -ne 0) {
        throw "Regression failed: $($check.Name)"
    }
}
```

Checks marked **solver** need `gurobipy` and a valid Gurobi licence. Checks
marked **numerical** use NumPy, SciPy and Matplotlib only.

## Checks

| Script | Type | What it protects |
|---|---|---|
| `check_gamma_convolution_quadrature.py` | numerical | Moschopoulos tail of a sum of Gamma variables against an independent quadrature. |
| `check_gamma_tail_bound.py` | numerical | Moschopoulos tail of two exponential variables against the closed form. |
| `check_horizon_sweep_visualization.py` | numerical | `plot_horizon_sweep` on hand-made reports: file written, not blank, one report extended by another. |
| `check_gamma_replay_validator.py` | solver | Solve of a mixed Gamma and rainflow fleet with `ard1` repairs, then the replay validator, and the detection of a corrupted mean. |
| `check_matched_ard_inputs.py` | solver | The matched `ardinf` and `ard1` inputs: they differ only in `repair_model`; both solve; replay and Monte Carlo validation pass. |
| `check_unrestricted_maintenance.py` | solver | No fleet-level depot capacity: the key `depot_capacity` is rejected, and two trains can be in the depot on one day. |

## Runners and reports

| Script | What it does |
|---|---|
| `run_horizon_sweep.py` | Calls `sweep_operating_horizons`, or `sweep_horizon_grid` with `--h1`, and prints a table of the cases. |
| `run_progressive_vbz_feasibility.py` | Solves the inputs of `vbz_progressive/` in order, validates each, and writes a summary. |
| `run_gamma_schedule_validation.py` | Runs the replay validator, the Monte Carlo validator, or both, on an input and a result file. |
| `plot_optimization_progress.py` | Plots incumbent, best bound and relative MIP gap from a result file or a Gurobi log. |
| `report_indicator_presolve.py` | Builds models with `encoding: indicator` and compares the Gurobi presolve under three `PreSOS1BigM` settings. |
| `gamma_quadrature_reference.py` | Helper: independent quadrature of the tail of a sum of Gamma variables. Imported by the numerical checks. |

Run the validators on an existing result:

```powershell
python .\examples\regression\run_gamma_schedule_validation.py `
  .\input\gamma_horizon_euler_convergence.yaml `
  .\results\gamma_result.yaml `
  .\results\gamma_validation.yaml `
  --mode both `
  --repetitions 100000 `
  --random-seed 20260908 `
  --maximum-schedule-failure-rate 1e-5
```

The stochastic failure rate is empirical. Its one-sided Clopper-Pearson upper
confidence bound is reported separately. Zero failures in 100,000 replays do
not prove that the true rate is below 1e-5.

Sweep over operating horizons:

```powershell
python .\examples\regression\run_horizon_sweep.py `
  .\input\gamma_horizon_euler_convergence.yaml `
  .\results\gamma_horizon_sweep.yaml `
  --h2-range 2 20 `
  --stop-on-gradient `
  --gradient-tolerance 0.001 `
  --maximum-stopping-gap 0.05
```

The tolerance is a relative change in `J_op/H2` per added unit of `H2`. By
default, two consecutive flat gradients are required. A positive gradient stops
after its first gap-qualified occurrence. A case whose MIP gap exceeds
`--maximum-stopping-gap` is recorded but cannot stop the sweep. The upper value
of `--h2-range` is always a hard limit. The report is rewritten after every
case; a report with `complete: false` is a checkpoint and names
`last_completed_H2`.

Sweep over both phases, with warm starts, ranked by the projected cost over 52
periods:

```powershell
python .\examples\regression\run_horizon_sweep.py `
  .\experiments\gamma_horizon_grid_final\input\horizon_grid.yaml `
  .\results\horizon_grid.yaml `
  --h1 2 4 6 `
  --h2 8 12 16 24 `
  --warm-start `
  --evaluation-horizon 52
```

The grid minimises `J_initialization + (52-H1) * J_op/H2`. Its bound and MIP gap
certify the same projected cost.

A relaxation start can seed the original model for a difficult case:

```yaml
relaxation_warm_start: true
relaxation_time_limit: 300
```

Only the discrete values of the relaxed solution are copied as a MIP start. The
original model must still find and verify an incumbent.

Plot the convergence of a result or of an Euler log:

```powershell
python .\examples\regression\plot_optimization_progress.py `
  .\result.yaml .\convergence.png
```

## Input files

| File | Purpose |
|---|---|
| `mixed_gamma_ard1_public.yaml` | Mixed Gamma and rainflow fleet with `ard1` repair; input of `check_gamma_replay_validator.py`. |
| `matched_mixed_ardinf.yaml` | Controlled mixed numerical case with `ardinf` for both components. |
| `matched_mixed_ard1.yaml` | Companion case with identical data and `ard1` repair. |
| `vbz_progressive/vbz_00_reference.yaml` to `vbz_05_oberes_glattal.yaml` | Feasibility ladder of the VBZ case, from a small reference (4 trains, 1 mission, 2 components, `H = [4, 12]`) to 16 trains, 5 missions, 4 components and `H = [4, 48]`; input of `run_progressive_vbz_feasibility.py`. |

The two matched files are meant for numerical tables. The Gamma cells use
`replacement_mu = 0.001` and `replacement_as_new: false`; the rainflow cells use
`0`.

## Maintenance rules

- A check prints `PASS ...` and exits with a non-zero code on failure.
- Randomised checks use a fixed seed and report it.
- Do not assert wall-clock time as an exact value.
- New public behaviour needs at least one test in `tests/`.
- Prefer extending an existing check to adding a nearly identical file.
