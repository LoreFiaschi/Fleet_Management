# Initial-damage horizon sensitivity

This experiment separates two questions.

1. `gamma_h1_diagnostic.sbatch` fixes `H2=24` and varies
   `H1={2,4,6,8,10,12}`. It directly tests whether additional fleet
   preparation reduces the subsequent average operating cost `J_op/H2`.
2. `gamma_h1_h2_grid.sbatch` evaluates the Cartesian grid
   `H1={2,4,6,8,10,12}` and `H2={12,16,24,32,40}`. It tests whether the H1
   effect persists across operating-horizon lengths.

The initial fleet is heterogeneous. Rows are vehicles and columns are the
Gamma traction-battery and rainflow motor-insulation components:

```text
[[0.10, 0.20],
 [0.15, 0.35],
 [0.35, 0.55],
 [0.45, 0.65]]
```

Both jobs record `J_initialization`, `J_op_average`, projected 52-period cost,
MIP gap, formulation size and solver progress. Plot cells distinguish proven
infeasibility from a time limit without a feasible incumbent.

Submit from the repository root after activating the Euler environment:

```bash
sbatch experiments/gamma_initial_damage_horizon/gamma_h1_diagnostic.sbatch
sbatch experiments/gamma_initial_damage_horizon/gamma_h1_h2_grid.sbatch
```

Run the diagnostic first. The grid is prepared for a later overnight job and
does not need to run concurrently.
