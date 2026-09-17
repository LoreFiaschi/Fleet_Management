# Strict Garage-Hardau-inspired feasibility experiment

This experiment is a synthetic reduced scaling application, not a calibrated
VBZ operational case. It uses `F=65`, `M=13`, four representative components,
and a 52-week two-phase horizon (`H1=4`, `H2=48`). The safety configuration is
`tau=1.0`, `epsilon=1e-4`, with the safe linear tangent implementation for the
rainflow reliability rows.

The run first screens one-step rainflow reliability, constructs the
repeatability-aware greedy schedule, and asks the exact fixed-binary original
model to complete it. If the heuristic cannot close the loop, the run falls
back to the existing feasibility-relaxation start generator; a relaxed point is
never reported as a valid solution of the original model. The primary outcome
is whether the original model produces a usable incumbent. Objective, bound and
MIP gap are secondary outcomes within the remaining overnight budget.

The original build-only Garage Hardau input used `F=180`, `M=26`, `L=2`, old
thresholds and exact quadratic reliability. This reduced experiment must not be
described as operationally calibrated data.

