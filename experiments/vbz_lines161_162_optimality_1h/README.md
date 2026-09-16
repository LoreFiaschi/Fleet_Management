# VBZ lines 161/162: tangent-Cantelli, 32-core optimality run

This experiment repeats the established small mixed-model case without changing
its physical input. It uses four vehicles, two missions, one Gamma component,
one rainflow component, `H1=4`, `H2=12`, tangent-Cantelli reliability with
`tangent_ref=0.32`, ARD-infinity repair, and no replacement.

Relative to the archived tangent run, the optimizer allocation and search
strategy change:

- `MIPFocus=3` prioritizes improvement of the dual bound;
- `Heuristics=0.02` reduces time spent searching for additional incumbents;
- `Symmetry=2` requests aggressive symmetry handling;
- `Cuts=2` requests aggressive cutting planes.
- `Threads=32` exactly matches the 32 CPUs requested from Slurm.

The solver receives exactly 3600 seconds. The Slurm allocation is 65 minutes so
that preflight checks and result archiving can finish after the optimizer stops.

This is an optimality experiment for the tangent inner approximation. A 5% gap
would certify the solution relative to that tangent formulation, not to the
original exact quadratic Cantelli formulation.

Submit from the repository root:

```bash
mkdir -p runs
sbatch experiments/vbz_lines161_162_optimality_1h/vbz_tangent_optimality_1h_32c.sbatch
```

The run folder and compressed archive are written below
`experiments/vbz_lines161_162_optimality_1h/runs/`.
