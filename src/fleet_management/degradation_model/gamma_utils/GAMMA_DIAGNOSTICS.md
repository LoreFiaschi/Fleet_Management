# Gamma complexity and timing diagnostics

The modular Gamma workflow has three computational stages. They must be read
separately because a fast Gurobi solve does not imply cheap calibration, and a
large calibration does not add the same number of Gurobi constraints.

| Stage | Purpose | Main complexity driver |
|---|---|---|
| Offline calibration | Construct conservative common-rate shapes | Enumerated finite count vectors and convolution-series terms |
| Gurobi formulation | Optimize assignments and maintenance | Fleet dimensions, Gamma cells, and horizon length |
| Exact validation | Certify the chosen schedule independently of Gurobi | Gamma cells, time steps, surviving history terms, and convolution-series terms |

Generate a report for the uniform Gamma, mixed ARD-inf, and mixed ARD1
Gamma/rainflow fixtures:

```powershell
python .\examples\regression\report_gamma_complexity.py `
    --output .\results\gamma_complexity.yaml
```

The output is intentionally self-describing. Counts are deterministic for the
same input and formulation. Times, iterations, work units and branch-and-bound
nodes are machine- and solver-dependent and should be treated as measurements,
not regression constants.

## Offline calibration fields

- `increment_opportunities`: mission/time entries available in one cell.
- `increment_types`: distinct `(mean, rate)` pairs after compression.
- `seed_types`: nonzero initial and replacement distributions.
- `calibration_lp_variables`: seed plus increment shapes optimized offline.
- `tail_constraints`: finite count vectors certified at the threshold.
- `total_convolution_series_terms`: total Moschopoulos mixture effort.
- `maximum_convolution_remaining_mass`: worst unaccounted positive mixture mass.
- `calibration_seconds`: cell calibration wall time, including reliability-shape inversion.

## Gamma formulation counts

Let `N_gamma` be the number of Gamma vehicle/component cells, `N_gamma_ard1`
the subset using ARD1, `T=H1+H2`, and `I_replacement` equal one when
replacement is enabled.

```text
Gamma shape variables       = N_gamma * T
Gamma ARD1 latch variables  = N_gamma_ard1 * T
Gamma indicator constraints = N_gamma * T * (6 + 3*I_replacement)
                            + N_gamma_ard1 * T * (2 + I_replacement)
Gamma reliability rows      = N_gamma * T
Gamma repeatability rows    = 2 * N_gamma + N_gamma_ard1
Gamma maintenance rows      = N_gamma * T * (2 + I_replacement)
```

Each no-action, repair, and optional replacement transition uses three
indicators: bounding shape `A`, physical mean `mu`, and removed mean `z`.
Indicator constraints are Gurobi general constraints and are not counted again
as ordinary linear rows.

ARD1 additionally carries the physical mean immediately after the previous
action. Its latch is held during no action, set after repair, and reset after
replacement. The Gamma reliability shape still receives no repair credit; this
is conservative because both ARD-inf and ARD1 are pathwise non-increasing.

For the supplied uniform fixture, the formula-generated subtotal must equal the
complete Gurobi model. In a mixed fleet, the reported remainder belongs to the
rainflow block. The mixed ARD1 fixture also verifies the additional latch
variables, indicators and repeatability rows against the actual model totals.

## Exact-validation fields

- `transitions_checked = N_gamma * T`.
- `exact_tail_evaluations`: one replayed tail per Gamma transition.
- `exact_tail_seconds`: time spent evaluating exact convolution tails.
- `replay_and_reporting_seconds`: decisions, history updates, checks and report creation.
- `minimum_conservativeness_margin = p_bound - p_exact_upper`.
- `minimum_reliability_slack = epsilon - p_bound`.

Small negative margins at floating-point scale are accepted within the stated
tolerance. A materially negative margin invalidates the complete report.
