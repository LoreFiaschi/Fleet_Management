# Fleet Management

Train fleet scheduling optimization with multi-component stochastic degradation, solved as a
Mixed-Integer (Non-)Linear Program using Gurobi. Each component of each train may use its own
degradation model, and may be repaired (imperfect maintenance) or fully replaced.

Full specification: [`spec/spec.tex`](spec/spec.tex) (v0.5).

**Status.** Currently implemented: **Gaussian** and **inverse Gaussian** degradation models,
`ARD1` maintenance, replacement, mixed-model fleets, both the `exact` (nonconvex quadratic) and
`lp` reliability formulations, and the scalar/interval horizon loop. **Wiener**, **Gamma**,
**Rainflow**, and `ARA1` maintenance are recognized by the input schema but raise
`NotImplementedError` — planned for a follow-up pass.

## Prerequisites

- Python >= 3.9
- A valid [Gurobi](https://www.gurobi.com/) license (academic licenses are free)

## Installation

```bash
pip install .
```

For development (editable install, needed to run the test suite):

```bash
pip install -e ".[dev]"
```

## Quick start

```python
from fleet_management import solve, plot_management

# The degradation model is per-component, read from the input file's 'model'
# field -- there is no separate `degradation` argument.
result = solve("input/data_example.yaml", results_path="results/output.yaml")
print(result["status"], result["objective"])

plot_management("results/output.yaml", plot_file_path="results/schedule.png")
```

For a horizon interval (`H: [H_min, H_max]`), `solve()` returns a dict keyed by `H`, and
`plot_management` writes one image per `H` (`schedule_H5.png`, `schedule_H6.png`, ...). See
[`input/data_example_loop.yaml`](input/data_example_loop.yaml).

## API reference

### `solve(input_path, results_path="output.yaml") -> dict`

Reads the problem data, solves the MILP, optionally writes the result, and always returns it.

| Parameter | Type | Description |
|---|---|---|
| `input_path` | `str` | Path to the input data file (YAML/JSON/HDF5). |
| `results_path` | `str`, optional | Output file path. Pass `None` to skip writing. Defaults to `"output.yaml"`. |

Supported input/output formats: **YAML** (`.yaml`, `.yml`), **JSON** (`.json`), **HDF5**
(`.h5`, `.hdf5`). The output format is determined by the file extension of `results_path`.

### `plot_management(input_file_path, plot_file_path=None)`

Reads solver output and produces a colour-coded schedule grid.

| Parameter | Type | Description |
|---|---|---|
| `input_file_path` | `str` | Path to a solver output file (single- or multi-horizon). |
| `plot_file_path` | `str`, optional | Output image path, or a filename *prefix* for multi-horizon output. Defaults to `"output.png"` / `"output"`. |

Supported image formats: **PNG** (`.png`), **PDF** (`.pdf`).

The plot is an F x 2H grid: each row is a train, each column a time step, each cell split into L
horizontal strips (one per component) coloured green-to-red by `mu / tau`. Strip border style
encodes the degradation model (solid=Gaussian, dashed=inverse Gaussian, dotted=Wiener,
dash-dot=Gamma, long-dash=Rainflow). Cell annotations: a **number** for an assigned mission, a
**gear** for a maintenance day with a repair, a red **R** circle for a replacement (may co-occur
with the gear), and a **"zzz" cloud** for an idle day.

## Input file schema

See `spec/spec.tex` Sections "Input Specification" and "Input File Schema" for the full,
authoritative schema (including consistency checks). Summary of the fields this release reads:

### Always present

| Key | Shape | Description |
|---|---|---|
| `F`, `M`, `L` | int | Trains, missions, components per train (`F > M`) |
| `H` | int or `[H_min, H_max]` | Half-horizon; an interval solves once per `H` in range |
| `tau` | float or (F, L) | Degradation upper bound |
| `epsilon` | float | Reliability threshold, in `(0, 0.01]` |
| `C_M`, `C_D` | float | Maintenance-day and damage-regularisation cost coefficients |
| `penalty_type` | str, optional | `"inf_norm"` (default) or `"quadratic"` (requires `formulation: exact`) |
| `formulation` | str, optional | `"exact"` (default, nonconvex quadratic reliability) or `"lp"` (linear inner approximation). `"socp"` is a deprecated alias for `"exact"` |
| `n_workers`, `warm_start` | optional | Horizon-loop parallelism / sequential warm-starting |
| `model` | (F, L) list of str | Per-component degradation model: `"gaussian"` or `"inverse_gaussian"` (see Status above) |
| `maintenance_type` | (F, L) list of str | Only `"ARD1"` is implemented so far |
| `rho` | float or (F, L) | Repair efficiency, in `(0, 1]` |
| `C_R`, `C_rep` | float, (F,), or (F, L) | Repair and replacement cost |
| `mu_0`, `mu_new` | (F, L) | Initial / post-replacement mean degradation |
| `mu` | (F, M, L) or (F, M, L, H) | Mean increment per mission/step |

### Model-specific

| Key | Applies to | Description |
|---|---|---|
| `v_0`, `v_new`, `v` | Gaussian | Initial/post-replacement variance, and variance increment `(F, M, L)`/`(F, M, L, H)` |
| `eta` | Inverse Gaussian | `mu`/`lambda` ratio, must be positive |
| `v_max_user` | Gaussian, optional | User-defined variance upper bound |

Fields marked "null where not applicable" in the example files may contain `null` at positions
whose component doesn't use that field's model; the parser ignores those entries.

See [`input/data_example.yaml`](input/data_example.yaml) (scalar `H`) and
[`input/data_example_loop.yaml`](input/data_example_loop.yaml) (interval `H`) for complete,
runnable examples of a mixed Gaussian + inverse Gaussian fleet.

## Output

`solve()` returns (and, if `results_path` is given, persists) a dict with `status`, `objective`,
`H`, `F`, `M`, `L`, `model` (echoed), `tail_bound` (always `null` in this release -- reserved for
the Rainflow model), `x` (F x (M+1) x 2H assignment), `x_m`/`x_r` (F x L x 2H repair/replacement
binaries), `mu`/`v` (F x L x 2H; `v` is `NaN` for components that don't track variance, e.g.
inverse Gaussian), `u` (scalar damage-regularisation value), and `z` (F x L x 2H degradation
removed by repair). `tau` is also echoed (F x L) -- not part of the spec's literal output table,
but needed by `plot_management` to normalize the heatmap.

For an interval `H`, `solve()` returns a dict keyed by each `H` value, each mapping to the
single-horizon dict described above.

## Project structure

```
Fleet_Management/
    pyproject.toml
    README.md
    src/
        fleet_management/
            __init__.py           # Public API: solve, plot_management
            solver.py             # I/O, validation, dispatch, horizon loop
            models/
                base.py            # Shared constraint-building helpers
                gaussian.py        # Gaussian component builder
                inverse_gaussian.py  # Inverse Gaussian component builder
            maintenance/
                ard1.py             # Shared ARD1 big-M accumulate/repair pattern
            plotter.py             # Schedule visualisation
    input/
        data_example.yaml          # Example input (mixed models, scalar H)
        data_example_loop.yaml     # Example input (mixed models, interval H)
    test/                          # pytest suite (see test/TEST_DOCUMENTATION.md)
```

## Running the tests

```bash
cd test && pytest -v
```

Most tests need a valid Gurobi license (they build and solve a real, tiny MILP); the
parsing/validation/I/O and plotting tests do not. See `test/TEST_DOCUMENTATION.md` for the split.
