# Greedy warm-start comparison

This controlled experiment solves the same finalized two-line model twice:

1. baseline without a MIP start;
2. deterministic repair-aware greedy MIP start.

Both cases use the same input, Gurobi parameters, 32 threads, random seed and
one-hour solver limit.  The run is sequential so both cases receive all 32
allocated CPUs.  The existing progress callback records incumbent and bound
trajectories for later comparison.

The greedy constructor reports any mission that would require an artificial
vehicle.  The F=4, M=2 comparison is expected to require none.  A nonzero count
is the explicit trigger for a separate penalized-artificial-vehicle model; it
is not silently treated as a feasible schedule of the original problem.
