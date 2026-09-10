# Final Gamma operating-horizon stabilization sweep

This experiment evaluates every `H2` in `{4,8,...,64}` using 55 minutes per
case, an exact solver MIP-gap target, and a 16-hour Slurm allocation. It does
not stop early: the 5% threshold is recorded only as the qualification line for
interpreting adjacent cost gradients.

## Submit on Euler

Commit and push the experiment locally, then on Euler run:

```bash
cd ~/Fleet_Management
git pull
cd experiments/gamma_horizon_stabilization
bash -n gamma_horizon_stabilization.sbatch
sbatch gamma_horizon_stabilization.sbatch
```

The job deliberately fails before solving if the Euler repository is dirty.
This prevents final evidence from being produced by uncommitted code.

## Outputs

Job `JOBID` creates:

```text
runs/horizon_JOBID/
  input.yaml
  job.sbatch
  metadata.txt
  modules.txt
  python_packages.txt
  preflight.txt
  run_horizon_sweep.py
  horizon_sweep.yaml
  cases.csv
  summary.yaml
  gamma_horizon_stabilization.png
  runner.out / runner.err
  postprocess.out / postprocess.err
  source_snapshot/
  SHA256SUMS
```

It also creates `runs/gamma_horizon_stabilization_JOBID.tar.gz` and its SHA-256
file. If Slurm terminates the job, the incremental `horizon_sweep.yaml` remains
inside the run directory and the exit trap still attempts to create the archive.

## Download to Windows

From PowerShell in the desired local destination:

```powershell
scp clangenauer@euler.ethz.ch:~/Fleet_Management/experiments/gamma_horizon_stabilization/runs/gamma_horizon_stabilization_JOBID.tar.gz .
scp clangenauer@euler.ethz.ch:~/Fleet_Management/experiments/gamma_horizon_stabilization/runs/gamma_horizon_stabilization_JOBID.tar.gz.sha256 .
tar -xzf .\gamma_horizon_stabilization_JOBID.tar.gz
```

The CSV is convenient for the report table, the PNG is ready for presentation
review, and the full YAML remains the authoritative result.
