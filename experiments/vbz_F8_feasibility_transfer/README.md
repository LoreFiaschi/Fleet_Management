# F6-to-F8 feasible-schedule transfer

This experiment tests whether the feasible binary schedule found for the
mixed-model `F=6` case can help establish feasibility for the otherwise
identical `F=8` case.

The experiment separates three claims:

1. `fixed_transfer` fixes all F8 assignment, repair and no-intervention
   binaries to the embedded F6 schedule. The two added vehicles receive no
   assignment and no maintenance. A feasible solve proves that the schedule is
   compatible with the current F8 formulation.
2. `warm_start` supplies the same binary values as a MIP start, leaves the
   model free, and stops after the first feasible solution.
3. `cold_start` solves the identical free model without a MIP start and stops
   after the first feasible solution.

The warm and cold cases use the same F8 input, four threads, seed 1, a one-hour
solver limit and `SolutionLimit=1`. They compare time to the first feasible
solution; they are not optimality experiments.

Before building F8, the runner also replays the saved F6 result with the current
deterministic Gamma validator. This guards against using an incompatible source
schedule after a formulation change.

## Contents

- `inputs/`: matched F6 and F8 inputs;
- `starts/`: the previously obtained feasible F6 result;
- `run_f8_feasibility_transfer.py`: experiment-only runner;
- `vbz_F8_feasibility_transfer.sbatch`: Euler job;
- `runs/`: per-job evidence and a downloadable archive.

## Preflight on Euler

Run these commands from the repository after pulling the commit containing this
experiment:

```bash
cd ~/Fleet_Management
module purge
module load stack/2024-06 python/3.12.8
module load gurobi/13.0.0
source .venv/bin/activate

bash -n experiments/vbz_F8_feasibility_transfer/vbz_F8_feasibility_transfer.sbatch

python experiments/vbz_F8_feasibility_transfer/run_f8_feasibility_transfer.py \
  --f6-input experiments/vbz_F8_feasibility_transfer/inputs/vbz_F6_M1_L4_H2_48.yaml \
  --f8-input experiments/vbz_F8_feasibility_transfer/inputs/vbz_F8_M1_L4_H2_48.yaml \
  --f6-result experiments/vbz_F8_feasibility_transfer/starts/vbz_F6_M1_L4_H2_48_result.yaml \
  --output-dir experiments/vbz_F8_feasibility_transfer/runs/preflight \
  --summary experiments/vbz_F8_feasibility_transfer/runs/preflight/summary.yaml \
  --build-only
```

The preflight should report three `build_only` cases and identical formulation
counts. It does not call `optimize()`.

## Submit

Submit from the experiment directory so Slurm's stdout and stderr paths resolve
inside `runs/`:

```bash
cd ~/Fleet_Management/experiments/vbz_F8_feasibility_transfer
sbatch vbz_F8_feasibility_transfer.sbatch
```

The job creates `runs/f8_transfer_<job-id>/` and
`runs/f8_transfer_<job-id>.tar.gz`. Preserve the summary, full result files,
deterministic Gamma validation reports, Gurobi logs, exact inputs, source
script, sbatch file, Git commit and dirty-worktree patch.

## Interpretation

- A feasible `fixed_transfer` case is the direct proof that the F6 schedule
  extends to F8 under the current formulation.
- Compare `first_feasible_solution_seconds` between `warm_start` and
  `cold_start`.
- Do not interpret these runs as objective-quality or optimality comparisons:
  every free run intentionally stops at its first feasible solution.
- If the fixed case is feasible but the warm case is not, inspect
  `warm_start.log`; supplying a start does not guarantee that Gurobi accepts or
  repairs it.
