# Greedy warm-start comparison

This controlled experiment solves the same finalized two-line model twice:

1. baseline without a MIP start;
2. deterministic repair-aware greedy MIP start.

Both cases use the same input, Gurobi parameters, 32 threads, random seed and
one-hour solver limit.  The run is sequential so both cases receive all 32
allocated CPUs.  The existing progress callback records incumbent and bound
trajectories for later comparison.

The greedy constructor reports whether its schedule meets the loop rows
(`repeatability_feasible`).  The F=4, M=2 comparison is expected to meet them.
The run stops with an error if they fail; the schedule is not silently treated
as a feasible schedule of the original problem.
