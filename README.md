# Fleet Management

Vehicle/train fleet scheduling optimization with **per-cell** multi-component
degradation models, solved with Gurobi. Each (vehicle, component) *cell* can use
its own degradation model, reliability bound, and maintenance (repair) model.

Every input is **self-describing**: a top-level `model:` key selects the
degradation model, so `solve()` no longer takes a `degradation` argument.

### Degradation models

| Model | Status | Notes |
|---|---|---|
| `rainflow` | **Implemented** (modular, per-cell) | Palmgren–Miner accumulated damage tracked via mean/variance (+ descriptors); distribution-free reliability bounds with selectable constraint *implementations*. |
| `gamma` | **Implemented via backend** | Solved through the existing gamma backend for gamma-only fleets. A modular per-cell gamma block is a work-in-progress placeholder. |
| `gaussian`, `inverse_gaussian` | **Reserved** | Accepted by the schema but not currently wired through the self-describing entry point (they raise `NotImplementedError`).

Depending on the reliability-constraint implementation, the program is a **MILP**
(linear bounds/surrogates) or a **nonconvex MIQCP** (exact quadratic bounds).

## Prerequisites

- Python >= 3.10
- A valid [Gurobi](https://www.gurobi.com/) license (academic licenses are free)

## Installation

```bash
pip install .
```

For development (editable install):

```bash
pip install -e ".[dev,dashboard]"
```

## Quick start

The model is read from the input file's `model:` key — do **not** pass a
`degradation` argument.

```python
from fleet_management import solve, plot_management

# Solve (model chosen inside the input file via `model: rainflow` / `gamma`)
solve("input/data.yaml", results_path="results/output.yaml")

# Plot the resulting schedule
plot_management("results/output.yaml", plot_file_path="results/schedule.png")
```

## Input file format

Inputs are normalized by a config layer (`config.py`) that validates every field
and broadcasts scalars to full per-cell arrays. Full details and the annotated
template are in `docs/config_and_solver.md` and `spec/input_template.yaml` described.

### Required keys (all models)

| Key | Type | Description |
|---|---|---|
| `model` | str or per-cell array `(F,L)` | Degradation model per cell: `rainflow`, `gamma`, `gaussian`, `inverse_gaussian`. A single string applies to the whole fleet. |
| `F` | int | Number of vehicles (must be > M) |
| `M` | int | Number of missions |
| `H` | int or `[H1, H2]` | Horizon. An int gives transitory = operating = H, `T = 2H`. A pair sets an unequal transitory `H1` (run-up from `mu_0`) and operating `H2` (repeatable cycle), `T = H1 + H2`. |
| `L` | int | Components per vehicle. Inferred from `model`; required only when `model` is a single string and `L > 1`. |
| `tau` | scalar / `(L,)` / `(F,L)` | Failure threshold (> 0). |
| `epsilon` | scalar / `(L,)` / `(F,L)` | Reliability level in (0, 1). |
| `rho` | scalar / `(L,)` / `(F,L)` | Repair efficiency in (0, 1]. |
| `mu_0` | scalar / `(L,)` / `(F,L)` | Initial mean damage. |
| `mu` | scalar / `(M,)` / `(L,M)` / `(L,M,H)` / `(F,L,M)` / `(F,L,M,H)` | Mean damage increment per mission (> 0), broadcast to `(F,L,M,H)`. |
| `C_M`, `C_R`, `C_S`, `C_P` | float | Cost coefficients (maintenance / repair / safety / periodicity). |

### Per-cell selectors

`model`, `bound_method`, and `repair_model` are `(F, L)` string arrays; give a
scalar (whole fleet), a length-`L` vector (per component), or a full `(F, L)`
array (per cell).

### Rainflow-specific keys

| Key | Type | Description |
|---|---|---|
| `bound_method` | str / `(L,)` / `(F,L)` | `markov` \| `cantelli` (default) \| `hoeffding` \| `bernstein` \| `chernoff`.|
| `repair_model` | str / `(L,)` / `(F,L)` | `ard1` (default) or `ardinf`. |
| `v` | scalar / `(M,)` … / `(F,L,M,H)` | Variance increment (> 0); required by cantelli/bernstein. |
| `v_0` | scalar / `(L,)` / `(F,L)` | Initial variance (>= 0). Default 0. |
| `support` | scalar / … | Per-mission increment support width; required by hoeffding/bernstein. |
| `cgf`, `s_chernoff` | scalar / … | Per-mission CGF and tilt; required by chernoff. |
| `replacement_mu`, `replacement_v` | scalar / `(L,)` / `(F,L)` | Post-replacement state (aliases `mu_new`, `v_new`). Default 0. |
| `*_trans` | `(F,L,M,H1)` | Optional transitory-phase profiles (`mu_trans`, `v_trans`, `support_trans`, `cgf_trans`); default reuse the operating profiles. |

### Gamma-specific keys

| Key | Type | Description |
|---|---|---|
| `gamma_beta` | scalar / `(L,)` / `(F,L)` | Gamma scale parameter (> 0). |
| `C_rep` | float | Replacement cost (required for gamma). |

### Options

Optional, passed to the solver (top level of the input, or `solve()` kwargs):
`verbose`, `mip_gap`, `time_limit`, `fast`, `allow_replacement`,
`depot_capacity`, `gurobi_params`, and the Step-3 knobs `reliability_impl`
(`exact` \| `tangent` \| `pwl`), `pwl_points` (default 8), `tangent_ref`
(default 0.5).

### YAML example (rainflow, uniform fleet)

```yaml
model: rainflow            # change to `gamma` to use the gamma backend
F: 6
H: 10                      # int -> H1 = H2 = 10, T = 20; or [H1, H2]
M: 3
L: 2

tau: 1.0
epsilon: 0.1
rho:                       # (F, L) repair efficiency
  - [0.80, 0.75]
  - [0.75, 0.90]
  - [0.90, 0.85]
  - [0.85, 0.70]
  - [0.70, 0.80]
  - [0.80, 0.85]

bound_method: cantelli     # per-fleet; or per-component [.., ..]; or (F, L)
repair_model: ard1

mu_0: 0.02                 # scalar broadcast to every cell
mu: 0.06                   # scalar broadcast to (F, L, M, H); or give a tensor
v: 1.5e-3
v_0: 4.0e-4

C_M: 1.0
C_R: 2.0
C_S: 1.5
C_P: 3.0

# optional
reliability_impl: exact    # exact | tangent | pwl
verbose: 1
mip_gap: 0.12
```

Larger inputs specify `mu`/`v` as `(F, L, M, H)` tensors; see
`spec/input_template.yaml`.

Supported input/output formats: **YAML** (`.yaml`, `.yml`), **JSON** (`.json`),
**HDF5** (`.h5`, `.hdf5`). The output format follows the extension of
`results_path`.

## API reference

### `solve(input_path, results_path=None)`

Reads the problem, chooses the degradation model from the input's `model:` key,
solves, and writes the results.

| Parameter | Type | Description |
|---|---|---|
| `input_path` | `str` | Path to the input data file. |
| `results_path` | `str`, optional | Output file path. Defaults to `"output.yaml"`; a bare name gets `.yaml`. |

### `plot_management(input_file_path, plot_file_path=None)`

Reads solver output and produces a colour-coded schedule grid.

| Parameter | Type | Description |
|---|---|---|
| `input_file_path` | `str` | Path to a solver output file. |
| `plot_file_path` | `str`, optional | Output image path. Defaults to `"output.png"`; a bare name gets `.png`. |

The plot is an `F x (T+1)` grid; each cell is split into `L` horizontal strips
(one per component), coloured green→red by `mu / tau`. It reads a per-cell
`(F, L)` threshold (or scalar / per-component). A blue divider marks the
transitory/operating boundary for two-horizon schedules. Cell annotations:
mission number `j` (assigned), gear (maintenance), or "zzz" cloud (idle).

Supported image formats: **PNG** (`.png`), **PDF** (`.pdf`).

## Output file contents

| Key | Description |
|---|---|
| `status` | Solver status (`"optimal"`, `"time_limit"`, …) |
| `objective` | Optimal objective (or `null`) |
| `degradation` | `rainflow`, `gamma`, or `mixed` |
| `F`, `M`, `H`, `L`, `H1`, `H2`, `T` | Dimensions and horizons |
| `tau` | Failure threshold, `(F, L)` |
| `bound_method`, `repair_model` | Per-cell selectors (scalar when uniform, else nested list) |
| `reliability_impl` | Reliability implementation used (scalar when uniform) |
| `mu_0`, `v_0` | Initial conditions, `(F, L)` |
| `x` | Binary assignment solution, `(F, M+1, T)` |
| `mu` | Mean-damage solution, `(F, L, T)` |
| `v` | Variance solution, `(F, L, T)` (variance-using bounds) |
| `u` | Max aggregate mean per step, `(T,)` |
| `z` | Removed expected damage on maintenance steps, `(F, L, T)` |
| `m`, `r` | Repair / replacement decisions, `(F, L, T)` |
| `mip_gap`, `bound` | Final MIP gap and best bound (when a solution exists) |

Results serialize to plain YAML/JSON (`yaml.safe_load`-compatible).

## GUI usage

> Note: the dashboard and validator below were built for the earlier
> Gaussian/IG input format and may need updating for the self-describing schema
> (`model:`, `tau`, `rho`, per-cell arrays).

```bash
.venv/Scripts/activate
python -m streamlit run src/fleet_management/validator_dashboard.py
```

Assign the input file, solver-results file, and a log path in the GUI; optional
`alpha_override` / `degradation_scale` stress-test the validator. Tabs cover: an
overview (feasible/infeasible assignments, threshold violations, danger zones);
failed/critical assignments with damage breakdown; per-mission analytics; a
per-vehicle damage timeline with mission/maintenance markers; cross-vehicle
component comparison; and a raw-data/export view.

## Validation

> Also predates the new schema; treat as legacy until re-aligned.

### `validate_baseline_assignment_feasibility(input_path, results_path, log_path)`

| Parameter | Type | Description |
|---|---|---|
| `input_path` | `str` | Path to the input data file. |
| `results_path` | `str` | Results file path. |
| `log_path` | `str` | Validation log path, default `"baseline_assignment_feasibility.log"`. |

Writes file paths, fleet parameters, failure threshold, degradation scale, the
actual solver assignment (damage state + increment + threshold check), and a
summary (assigned missions, feasible/infeasible entries, scheduled maintenance).
Supported input formats: YAML, JSON.

## Reliability bounds and implementations

The reliability requirement `P(D > tau) <= epsilon` is enforced per step by a
distribution-free **bound**, selected per cell via `bound_method`:

| Bound | Information used | Constraint type |
|---|---|---|
| `markov` | mean | linear |
| `cantelli` | mean, variance | quadratic |
| `hoeffding` | mean, support | quadratic |
| `bernstein` | mean, variance, support | quadratic |
| `chernoff` | cumulant generating function (fixed tilt `s`) | linear |

Bounds are listed in increasing tightness (each uses more information). See
`reliability_bounds.md` for the exact inequalities.

The three quadratic bounds additionally offer a choice of **implementation** —
how the (nonconvex) inequality is encoded — via `reliability_impl`:

| `reliability_impl` | Encoding | Validity | Solver class |
|---|---|---|---|
| `exact` (default) | nonconvex quadratic, as written | exact | MIQCP (`NonConvex=2`) |
| `tangent` | one supporting tangent of the convex cap | safe (inner) | MILP |
| `pwl` | piecewise tangent over `pwl_points` segments | safe (inner) | MILP + segment binaries |

`tangent` and `pwl` are conservative (their feasible set is a subset of the exact
one — they never accept an unsafe schedule), trading a little optimality for a
linear model. `pwl` tightens toward `exact` as `pwl_points` grows. Markov and
Chernoff are already linear, so they ignore this option. See
`reliability_implementations.md` and `compare_reliability_impls.py`.


## Project structure

```
Fleet_Management/
    pyproject.toml
    README.md
    reliability_bounds.md              # the five bounds (math reference)
    spec/
        input_template.yaml            # annotated input template
    input/
        test_switch_model.yaml         # flip model: rainflow <-> gamma
    docs/
        config_and_solver.md           # schema, config, solver, FleetConfig API
        reliability_implementations.md # bounds x implementations (Step 3)
    compare_reliability_impls.py       # exact/tangent/pwl comparison harness
    src/
        fleet_management/
            __init__.py                # Public API: solve, plot_management
            config.py                  # self-describing schema -> FleetConfig
            solver.py                  # I/O, dispatch, serialization
            degradation_model/
                rainflow.py            # modular per-cell rainflow MILP/MIQCP
                gamma_utils/
                    gamma_gurobi.py    # gamma backend
            utils/
                plotter.py             # schedule visualisation
            # legacy backends (not wired through the new entry point):
            gaussian.py
            inverse_gaussian.py
```

## Further reading

- `docs/config_and_solver.md` — input schema, broadcasting, horizons, validation,
  the `FleetConfig` API, and how `solver.py` dispatches.
- `docs/reliability_implementations.md` — the bound-vs-implementation split and
  how to add new relaxation families.
- `reliability_bounds.md` — the exact probability inequalities.
