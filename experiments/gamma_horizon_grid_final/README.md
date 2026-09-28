# Final transitory/operating horizon grid

This experiment varies both the preparation horizon `H1` and repeatable
operating horizon `H2` using the finalized explicit-idle formulation.

The controlled two-line case has `F=4`, `M=2`, and `L=2` (one Gamma and one
rainflow component). It uses tangent Cantelli reliability, ARD-infinity repair,
`tau=1.0`, `epsilon=1e-4`, a non-pristine initial fleet, component-specific
intervention costs, 32 Gurobi threads, a 5% MIP-gap target and an optional
phase-I relaxation warm start.

The grid is

```text
H1 = {2, 4, 6}
H2 = {8, 12, 16, 24}
```

Each pair is ranked over a 52-period evaluation horizon using

```text
J_initialization + (52-H1) * J_op/H2.
```

Warm starts are propagated between increasing `H2` values within each fixed
`H1` row. The original, unrelaxed model must still find and verify every
reported incumbent.

## Local preparation

Run the complete regression suite before committing the experiment:

```powershell
$checks = Get-ChildItem .\examples\regression\check_*.py | Sort-Object Name
foreach ($check in $checks) {
    python $check.FullName
    if ($LASTEXITCODE -ne 0) { throw "Regression failed: $($check.Name)" }
}
```

If `experiments/` is ignored, force-add only these small reproducible files:

```powershell
git add -f .\experiments\gamma_horizon_grid_final\README.md
git add -f .\experiments\gamma_horizon_grid_final\gamma_horizon_grid_final.sbatch
git add -f .\experiments\gamma_horizon_grid_final\summarize_grid.py
git add -f .\experiments\gamma_horizon_grid_final\plot_grid.py
git add -f .\experiments\gamma_horizon_grid_final\input\horizon_grid.yaml
git commit -m "Add final H1 H2 horizon grid experiment"
git push origin gamma-tail-integration
```

Do not add the `runs/` directory.

## Submit on Euler

```bash
cd ~/Fleet_Management
git fetch origin
git pull --ff-only
source .venv/bin/activate

mkdir -p runs experiments/gamma_horizon_grid_final/runs
bash -n experiments/gamma_horizon_grid_final/gamma_horizon_grid_final.sbatch

sbatch experiments/gamma_horizon_grid_final/gamma_horizon_grid_final.sbatch
```

The batch job refuses to run with modified tracked files. It validates all
critical dimensions, safety settings, solver parameters and the allocation of
32 CPUs before optimization.

## Monitor

```bash
myjobs -j JOBID
cat runs/gamma-H1-H2-grid-JOBID.err
tail -n 60 \
  experiments/gamma_horizon_grid_final/runs/gamma_horizon_grid_final_JOBID/runner.out
```

## Outputs

The run directory contains the authoritative grid report, compact summaries,
heatmap, environment metadata, input and job snapshots, solver logs and source
snapshots. The exit trap additionally creates:

```text
experiments/gamma_horizon_grid_final/runs/
  gamma_horizon_grid_final_JOBID.tar.gz
  gamma_horizon_grid_final_JOBID.tar.gz.sha256
```

If the job is stopped before post-processing, the archive still preserves all
files written up to that point. The grid report checkpoints after each complete
`H1` row.
