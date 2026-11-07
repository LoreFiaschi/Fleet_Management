# Joint H1 and H2 sweep, 90 minutes per case

This experiment evaluates one grid of the transitory horizon `H1` and the
operating horizon `H2`, with a time limit of 5400 s (90 minutes) for each case.

- `H1 = {2, 6, 10}` and `H2 = {8, 12, 16, 20}`: 12 cases.
- `gamma_h1_h2_daily_5400s.sbatch` calls
  `examples/regression/run_horizon_sweep.py --h1 2 6 10 --h2 8 12 16 20 --warm-start`.
  Within each `H1`, each case starts from the schedule of the preceding case.
- Each case has a time limit of 5400 s, a relaxation start of at most 60 s, a
  MIP gap of 0.05 and 32 threads. The job asks for 20 h in total.

The input `input/horizon_initial_damage_daily_5400s.yaml` has four trains, two
missions and two components: a Gamma traction battery and a rainflow motor
insulation (Cantelli bound, `tangent` form, `epsilon = 1e-4`). The initial
fleet is heterogeneous. Rows are trains and columns are the two components:

```text
[[0.05, 0.05],
 [0.05, 0.05],
 [0.30, 0.50],
 [0.40, 0.60]]
```

The preflight (`preflight_small_grid.py`) checks the input and the allocation.
After the sweep, `summarize.py` writes a CSV and a YAML summary of the report
and `plot_results.py` draws the grid (`--kind grid`). Plot cells distinguish
proven infeasibility from a time limit without a feasible incumbent.

Submit from the repository root after activating the Euler environment:

```bash
sbatch experiments/gamma_h1_h2_daily_5400s/gamma_h1_h2_daily_5400s.sbatch
```
