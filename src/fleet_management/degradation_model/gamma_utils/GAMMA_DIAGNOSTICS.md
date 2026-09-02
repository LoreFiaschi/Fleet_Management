# Gamma complexity and timing diagnostics

The current modular Gamma workflow has three computational stages. They must
be interpreted separately because they have different complexity drivers.

| Stage | Purpose | Main complexity driver |
|---|---|---|
| Offline calibration | Construct repeated-increment common-rate bounding shapes | Gamma cells, distinct increment types, and safe repetition counts |
| Gurobi formulation | Optimize assignments and maintenance | Vehicles, missions, components, Gamma cells, and horizon length |
| State replay | Check that the saved schedule reproduces the solver states | Gamma cells and time steps |

Generate the diagnostic report for the uniform Gamma, mixed ARD-infinity, and
mixed ARD1 Gamma/rainflow fixtures with:

```powershell
python .\examples\regression\report_gamma_complexity.py `
    --output .\results\gamma_complexity.yaml
```

The report is intentionally self-describing. Formulation counts are
deterministic for the same input and code version. Wall times, iterations,
work units and branch-and-bound nodes depend on the machine, Gurobi version and
parameter settings; they are measurements rather than regression constants.

## Offline repeated-increment calibration

For one Gamma vehicle/component cell, let increment type `q` have exact
distribution

```text
X_q ~ Gamma(alpha_q, beta_q),    alpha_q = mu_q * beta_q.
```

The current calibration first determines the largest safe repetition count
within the finite planning horizon:

```text
m_q* = max {m <= n_q_max : P(sum(r=1..m) X_qr > tau) <= epsilon}.
```

It then selects one common rate for the cell,

```text
0 < beta* <= min_q beta_q,
```

and finds a bounding shape `alpha_q*` such that, for every repetition count
`m = 1, ..., m_q*`,

```text
P(Gamma(m*alpha_q*, beta*) > tau)
    >= P(Gamma(m*alpha_q, beta_q) > tau).
```

The calibrated shapes can therefore be accumulated linearly in Gurobi at the
common rate. Calibration is performed before model construction; no Gamma CDF,
convolution, or calibration optimization is evaluated inside Gurobi.

The guarantee is specific to the configured threshold, rates, and finite
horizon. Changing `tau`, `epsilon`, an exact Gamma rate, or the horizon requires
recalibration. The repeated-increment construction checks repetitions of each
identified increment type. It must not be described as full stochastic
dominance at every threshold or as an unrestricted-horizon guarantee.

The main report fields are:

- `method`: expected to be `repeated_increment` for the current formulation.
- `increment_opportunities`: mission/time increment entries before compression.
- `increment_types`: distinct increment profiles calibrated for the Gamma cells.
- `minimum_safe_count` and `maximum_safe_count`: range of the calculated `m_q*` values.
- `common_rates`: distinct calibrated `beta*` values used by the cells.
- `minimum_bounded_increment_shape` and `maximum_bounded_increment_shape`: range of calibrated `alpha_q*` values.
- `tail_constraints`: number of repeated-count inequalities checked during calibration.
- `calibration_seconds`: total calibration wall time.

Nonzero initial and replacement states are calibrated as seed states and are
reported separately where the selected fixture uses them.

## Gamma Gurobi formulation counts

The counts below refer only to vehicle/component cells assigned to Gamma.
`T = H1 + H2`, and `I_replacement` equals one when replacement is enabled.

```text
Gamma shape variables      = Gamma component cells * T
Gamma ARD1 latch variables = 2 * Gamma ARD1 component cells * T
Gamma Big-M dynamics rows  = 2 * Gamma component cells * T
                             * (6 + 3*I_replacement)
                           + 4 * Gamma ARD1 component cells * T
                             * (2 + I_replacement)
Gamma reliability rows     = Gamma component cells * T
Gamma repeatability rows   = 2 * Gamma component cells
                           + 2 * Gamma ARD1 component cells
Gamma maintenance rows     = Gamma component cells * T
                              * (2 + I_replacement)
```

The Gamma block uses ordinary linear Big-M rows and introduces no indicator,
general, or quadratic constraints. Conditional state equalities are represented
by two asymmetric Big-M inequalities. The affected states are the bounding
shape, physical expected damage, removed expected damage and, for ARD1, the
post-intervention latch states.

The Big-M constants are time-dependent reachable bounds. Before constraints
are added, each Gamma cell receives safe upper bounds for physical expected
damage, bounding shape, removed damage, and the ARD1 latches. The bounds are an
over-approximation of every possible schedule: normal operation adds the largest
available increment at each step, repair cannot increase a state, and
replacement applies the replacement seed. Physical expected damage is clipped
by `tau`, while bounding shape is clipped by `A_max`. This changes variable
bounds and Big-M coefficients without introducing additional variables or
constraints. Solver output records:

```text
gamma_dynamics_formulation: tight_big_m
gamma_big_m_bound_strategy: time_dependent_reachable
gamma_formulation:
  big_m_implementation:
    conditional_equalities: <count>
    linear_rows: <twice the count>
    minimum_coefficient: <smallest positive M>
    maximum_coefficient: <largest positive M>
```

At fixed common rate, both ARD-infinity and ARD1 scale the Gamma bounding shape
consistently with the physical-mean repair rule. ARD1 additionally stores the
physical mean and bounding shape immediately after the latest intervention.
Normal operation holds these latches, repair updates them, and replacement
resets them.

For the supplied uniform Gamma fixture, the known shared/Gamma subtotal is
compared directly with the complete Gurobi model. In mixed fixtures, the
remaining variables and constraints belong to the remaining-life block.

## Lightweight state replay

The public post-solve check replays the selected schedule from the input data.
It recomputes the physical mean, calibrated bounding shape, removed damage and
ARD1 latch states for every Gamma cell and time step, then compares them with
the saved solver result. It also checks the Gamma reliability and repeatability
inequalities.

Main replay fields are:

- `gamma_cells`: number of Gamma vehicle/component cells replayed.
- `transitions_checked`: number of Gamma vehicle/component cells multiplied by
  the total horizon length `T`.
- `repairs` and `replacements`: interventions encountered in the schedule.
- `maximum_errors`: largest differences between replayed and saved mean, shape,
  removed-damage, latch, reliability and repeatability values.
- `validation_wall_seconds` or `replay_seconds`: complete replay wall time,
  depending on the report schema.

The replay is a software-consistency check. It is intended to detect an
incorrect transition, extraction error, or corrupted result file. It does not
recompute an exact varying-rate failure probability and must not be presented
as an independent probabilistic certificate of the calibrated surrogate.

## Development-only numerical checks

The repository retains convolution, quadrature and randomized property scripts
as internal numerical evidence for the earlier general tail-bound work. They
are useful for regression and research comparison, but they are not called by
the public `solve()` workflow and their complexity is not part of normal
post-solve replay. In particular, fields such as
`minimum_conservativeness_margin`, `minimum_reliability_slack`, convolution
series terms, and remaining mixture mass belong to those internal checks, not
to the lightweight public validator.

## Deterministic scalability sweeps

Two separate diagnostics should be used for scalability:

```powershell
python .\examples\regression\run_formulation_size_sweep.py `
    .\input\gamma_horizon_euler.yaml `
    .\results\gamma_formulation_sweep.yaml

python .\examples\regression\run_horizon_sweep.py `
    .\input\gamma_horizon_euler.yaml `
    .\results\gamma_horizon_sweep.yaml `
    --h2-range 2 32 `
    --stop-on-gradient `
    --gradient-tolerance 0.001 `
    --maximum-stopping-gap 0.05
```

The formulation-size sweep varies `F`, `M`, `L`, and `T` one at a time and
records deterministic variable and constraint counts. These one-at-a-time
slopes describe the selected baseline only because the general formulation
contains interaction terms such as `F*M*T` and `F*L*T`.

The horizon sweep keeps `F`, `M`, `L`, and `H1` fixed while varying `H2`. It
reports total, continuous, integer and binary variables, linear constraints,
calibration time, optimizer time, node count, the
operating objective `J_op/H2`, its best bound and the relative MIP gap. The
operating-average model is deliberately single-objective so this bound and gap
certify the same quantity that is compared across horizons.

The first `H1` steps are an initialization phase. Their decisions and state
transitions remain constrained and influence the operating phase, but their
cost is not part of the objective and no initialization-cost budget is imposed.

With gradient stopping enabled, the sweep continues in increasing `H2` until
the relative operating-cost gradient per added unit of `H2` is sufficiently
flat for the requested number of consecutive comparisons, or becomes positive.
Only adjacent cases that are optimal or satisfy the configured maximum MIP gap
may trigger the stop; otherwise the sweep continues to the hard upper horizon.
`best_proven_H2` is selected only from optimal cases whose reported MIP gap is
at most numerical tolerance. `best_feasible_H2` may
include a time-limit result and must be reported with its bound and MIP gap; a
lower feasible value is not proof of a better operating horizon.
The report is checkpointed after every completed case, so a cluster wall-time
termination preserves all horizons that finished before the interruption.
