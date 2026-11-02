# Test Suite Documentation

Written for the spec/spec.tex v0.5 API (per-component `model` assignment,
`tau`/`rho`/`C_D`/replacement, no `degradation` argument to `solve()`). All
five degradation models are implemented: `gaussian`, `inverse_gaussian`,
`wiener`, `gamma`, `rainflow`. Both `ARD1` and `ARA1` maintenance are
implemented (`ARA1` valid for `wiener`/`gamma` only).

## Test Files Overview

| File | Module under test | Gurobi required? |
|---|---|---|
| `test_solver.py` | `solver.py`: parsing, broadcasting, consistency checks, result I/O | No |
| `test_gaussian.py` | `models/gaussian.py` | Only the dynamics/reliability/replacement tests (validate_inputs tests do not) |
| `test_inverse_gaussian.py` | `models/inverse_gaussian.py` | Only `test_ig_mean_dynamics_and_reliability_cap` (solve_mu_bar/validate_inputs tests do not) |
| `test_wiener.py` | `models/wiener.py` (ARD1 + ARA1) | Only the dynamics/reliability/replacement tests |
| `test_gamma.py` | `models/gamma.py` (ARD1 + ARA1) | Only the dynamics/reliability/replacement tests |
| `test_rainflow.py` | `models/rainflow.py` (Cantelli + Bernstein) | Only the dynamics/reliability/replacement tests |
| `test_plotter.py` | `plotter.py` | No (uses hand-built result dicts, never calls `solve`) |
| `test_end_to_end.py` | Full `solve()` -> save -> `plot_management()` pipeline via the public API | Yes |

## Running the tests

```bash
# From project root
pytest test/ -v

# Only tests that don't need a Gurobi license: test_solver.py and
# test_plotter.py never build a Gurobi model at all.
pytest test/test_solver.py test/test_plotter.py -v
```

The five model test files each mix Gurobi-free tests (validation, offline
root-finding) with ones that solve a real model; see their sections below for
which is which. `test_end_to_end.py` always needs Gurobi.

## A note on test design: "dominates", not "equals"

Every model's dynamics tests check that the solved state **dominates**
(`solved >= recomputed`) a hand-recomputed recursion, not that it equals it
exactly. This isn't a weaker test by accident: the accumulate/repair rows are
one-sided (`>=`) big-M relaxations, and per spec's slack-robustness remark
(Section 5.3) the tracked state may legitimately carry slack wherever neither
cost mechanism (`C_D*u`, `C_R*z`) happens to bind for a given component --
this never endangers correctness, since the tracked state still dominates
the true trajectory componentwise. Two early versions of these tests
asserted exact equality and failed on real (correct) solves purely because
of harmless slack; see `test_gaussian.py`'s
`test_gaussian_mean_dynamics_dominate_recursion` docstring for the full
argument.

A related lesson from the same debugging pass: hand-built "force a specific
schedule via variable bounds" tests must pin the *minimum* needed (usually
just one binary at one step) and leave the rest free. Over-constraining the
rest of the schedule (e.g. forcing missions on every remaining day with no
further maintenance) can make the model *genuinely* infeasible against the
hard loop constraint -- the solver correctly rejecting an unsustainable test
schedule, not a bug. `test_replacement_reset_is_enforced` in each model file
follows the minimal-pinning pattern for this reason.

## test_solver.py

Pure-Python (no Gurobi model is ever built):

- **Broadcasting helpers**: `_broadcast_FL`, `_broadcast_cost_FL`,
  `_broadcast_increment_raw`, `_broadcast_FL_nullable`, `_broadcast_tail_bound`,
  `_wrap_increment` (including the H-interval "slice the period, then double"
  behavior).
- **Parsing**: both example YAML files (`data_example.yaml`,
  `data_example_loop.yaml`, which mix all five models and both maintenance
  types) parse to the expected shapes.
- **Consistency checks** (spec Section 3.6): epsilon range, `F > M`, unknown
  model strings, ARA1 acceptance for wiener/gamma and rejection for
  gaussian/inverse_gaussian/rainflow, `lp`+`quadratic` incompatibility, bad
  `formulation`/`penalty_type`, `warm_start`+`n_workers>1`, `mu_0 < tau`, the
  deprecated `formulation="socp"` alias, `H`-interval shape/ordering
  validation, conditional field requirements (`mu` skippable for all-gamma
  fleets, `v` skippable without gaussian/rainflow, `sigma`/`beta`/`alpha_inc`
  required per model), rainflow `tail_bound` resolution/defaulting/warnings,
  and the Bernstein-only Bhatia-Davis + `tau > 2*kappa*b/3` checks.
- **Result I/O round-trips**: YAML/JSON/HDF5, single- and multi-horizon, using
  hand-built result dicts (via the `make_fake_result`/`make_fake_shared`
  fixtures in `conftest.py`) so no solve is required. Also covers
  serialization of a non-optimal (e.g. infeasible) result.

## test_gaussian.py

- `validate_inputs` happy/unhappy paths (ARA1 rejection, `rho` out of range,
  the `mu_0 >= 3*sqrt(v_0)` check, `mu_0 < tau`, and that masked-out entries
  are skipped) — no Gurobi.
- Dynamics correctness: solves a small mixed fleet for real, then
  **recomputes** the ARD1 mean/variance recursion in plain NumPy from the
  returned `x`/`x_m`/`x_r` and checks the solved `mu`/`v` **dominates** it
  (see "A note on test design" above).
- Loop constraint and exact-reliability-constraint postconditions on the
  solved trajectory.
- `formulation="lp"` and `penalty_type="quadratic"` both solve and satisfy
  their respective constraint forms.
- `test_replacement_reset_is_enforced`: forces `x_r=1` at one step via a
  variable bound (bypassing the optimizer's economic choice), leaving
  everything else free, and checks the replacement big-M rows pin `mu`/`v`
  to `mu_new`/`v_new` exactly at that step.

## test_inverse_gaussian.py

- `solve_mu_bar` (the offline reliability cap, Eq. `ig_mubar`) is cross-checked
  against `scipy.stats.invgauss.cdf` directly, across a range of `eta` — no
  Gurobi. Also covers the non-binding-at-extreme-`eta` case and the
  (mathematically unreachable, by Markov's inequality) "unachievable" branch.
- `validate_inputs` happy/unhappy paths — no Gurobi.
- `test_ig_mean_dynamics_and_reliability_cap`: solves a small IG-only fleet for
  real, checks the solved `mu` dominates the recomputed recursion, stays below
  the precomputed cap, and that `v` is `NaN` (IG doesn't track variance).

## test_wiener.py

- `validate_inputs` happy/unhappy paths for both ARD1 and ARA1 (`sigma > 0`,
  `v_0 > 0` strictly, `rho` range, `mu_0 < tau`) — no Gurobi.
- ARD1 mean dynamics dominate the recomputed recursion; variance dynamics
  (mission-only accumulation via `sigma^2`, no repair term) dominate their
  own recomputed recursion.
- ARA1 mean dynamics dominate an **anchor-aware** recomputed recursion
  (tracks the maintenance-epoch anchor alongside the mean, including the
  anchor-reset-at-replacement fix -- see `maintenance/ara1.py`).
- Loop constraint (including the ARA1 anchor row), exact-reliability
  postconditions (identical form to Gaussian), `lp` formulation, and
  `test_wiener_replacement_reset_is_enforced` (same minimal-pinning pattern
  as Gaussian's).

## test_gamma.py

- `solve_alpha_hat` (the offline reliability cap) is cross-checked against
  `scipy.stats.gamma.sf` directly, across a range of `beta` — no Gurobi. Also
  covers the non-binding-at-extreme-`beta` warning case.
- `validate_inputs` happy/unhappy paths for both ARD1 and ARA1 (`beta > 0`,
  `alpha_inc > 0`, `rho` range, `mu_0 < tau`) — no Gurobi.
- Dynamics correctness: solves a small gamma-only fleet for real and checks
  the output `mu` (= `alpha * beta`, spec's post-processing substitution)
  dominates the recomputed `alpha` recursion scaled by `beta`.
- Reliability cap and loop constraint hold on the recovered `alpha`; `v` is
  `NaN` (gamma doesn't track variance).
- `lp` formulation (gamma's reliability is a precomputed linear cap regardless
  of formulation, so this should solve identically) and
  `test_gamma_replacement_reset_is_enforced`.

## test_rainflow.py

- `_support_constant` (the `b_il = max(b_0, b_new, max(b_inc))` helper) — no
  Gurobi.
- `validate_inputs` happy/unhappy paths for both tail bounds: ARA1 rejection,
  bad `tail_bound` string, the Bernstein-only Bhatia-Davis checks (on both
  the increment and the initial/reset moments) and the
  `tau > 2*kappa*b/3` precondition, and that Cantelli ignores missing `b_*`
  fields entirely — no Gurobi.
- Dynamics correctness (Cantelli): mean/variance dynamics are structurally
  identical to Gaussian's, so the same domination check applies; reliability
  and loop constraint postconditions.
- Dynamics correctness (Bernstein): reliability postcondition using the
  precomputed support constant. Note the `solved_bernstein` fixture uses
  F=3 (not 2): the Bernstein cap (`mu <= tau - c`, tighter than Cantelli's
  plain `tau`) forces more frequent maintenance, and with F=2/M=1 there is no
  schedule slack left for the other train to also satisfy its own
  reliability/loop constraints -- genuine tightness, not a bug (confirmed via
  Gurobi's IIS before widening the fleet; see the fixture's comment).
- Direct cross-check that `V_max^RF` (both tail bounds) matches spec's own
  derivation (the largest variance the exact reliability constraint admits at
  `mu=0`) — no Gurobi.
- `lp` formulation for both tail bounds, and
  `test_rainflow_replacement_reset_is_enforced`.

## test_plotter.py

Every test writes a hand-built result dict to disk (via
`solver._save_results`) and calls `plot_management` on it — `solve()` and
Gurobi are never invoked. Covers: all three I/O formats, default output
naming, PDF output, rejecting unsupported extensions, rejecting a
non-optimal result, multi-horizon (one PNG per H), and every cell-annotation
path (mission number, repair-only gear, replace-only wrench, repair+replace
on different components on the same day, a no-op maintenance day, and idle).

## test_end_to_end.py

Exercises the public API only (`solve`, `plot_management`) end-to-end:
YAML/JSON/HDF5 input, `results_path=None`, the sequential warm-started
horizon loop, the parallel (`n_workers=2`) horizon loop, and a kitchen-sink
mixed fleet covering all five models with ARA1 (wiener + gamma) and a
Bernstein-tail-bound rainflow component solved and plotted together. Requires
Gurobi.
