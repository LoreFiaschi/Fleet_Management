# Fleet Management

Fleet scheduling optimization with Gaussian degradation, solved as a Mixed-Integer Linear Program (MILP) using Gurobi.

## Prerequisites

- Python >= 3.9
- A valid [Gurobi](https://www.gurobi.com/) license (academic licenses are free)

## Installation

```bash
pip install .
```

For development (editable install):

```bash
pip install -e ".[dev]"
```

## Quick start

```python
from fleet_management import solve, plot_management

# 1. Solve the optimization problem
solve("input/data.yaml", degradation="gaussian", results_path="results/output.yaml")

# 2. Plot the resulting schedule
plot_management("results/output.yaml", plot_file_path="results/schedule.png")
```

## API reference

### `solve(input_path, degradation, results_path=None)`

Reads the problem data, solves the MILP, and writes the results to a file.

| Parameter | Type | Description |
|---|---|---|
| `input_path` | `str` | Path to the input data file. |
| `degradation` | `str` | Degradation model. Currently supported: `"gaussian"`. |
| `results_path` | `str`, optional | Output file path. Defaults to `"output.yaml"`. |

Supported input/output formats: **YAML** (`.yaml`, `.yml`), **JSON** (`.json`), **HDF5** (`.h5`, `.hdf5`).
The output format is determined by the file extension of `results_path`.

### `plot_management(input_file_path, plot_file_path=None)`

Reads solver output and produces a colour-coded schedule grid.

| Parameter | Type | Description |
|---|---|---|
| `input_file_path` | `str` | Path to a solver output file. |
| `plot_file_path` | `str`, optional | Output image path. Defaults to `"output.png"`. |

Supported image formats: **PNG** (`.png`), **PDF** (`.pdf`).

The plot is an F x (2H+1) grid where each cell is coloured on a green-to-red heatmap (0 to 1) based on the degradation mean. Cell annotations indicate:

- **Number** (j): flight j is assigned (`x[i,j,k] = 1`, j > 0)
- **Gear icon**: maintenance is scheduled (`x[i,0,k] = 1`)
- **"zzz" cloud**: the aircraft is idle

## Input file format

The input file must contain the following keys:

| Key | Type | Description |
|---|---|---|
| `F` | int | Number of flights |
| `H` | int | Time horizon (model spans 2H steps) |
| `M` | int | Number of maintenance levels |
| `mu` | 3D array (F x M x H) | Mean degradation parameters |
| `v` | 3D array (F x M x H) | Variance degradation parameters |
| `epsilon` | float | Reliability threshold, in (0, 0.5) |
| `C_M` | float | Maintenance cost coefficient |
| `C_R` | float | Repair cost coefficient |
| `C_S` | float | Safety cost coefficient |
| `C_P` | float | Penalty cost coefficient |
| `mu_0` | 1D array (F) | Initial mean values per flight |
| `v_0` | 1D array (F) | Initial variance values per flight |
| `verbose` | int, optional | Gurobi verbosity (0 = silent, 1 = normal). Default: 1 |

### YAML example

```yaml
F: 4
H: 3
M: 2
epsilon: 0.1
C_M: 1.0
C_R: 2.0
C_S: 1.5
C_P: 3.0
verbose: 0

mu_0: [0.5, 0.6, 0.55, 0.45]
v_0: [0.02, 0.03, 0.025, 0.018]

mu:
  - [[0.4, 0.5, 0.6], [0.35, 0.45, 0.55]]
  - [[0.5, 0.6, 0.7], [0.45, 0.55, 0.65]]
  - [[0.45, 0.55, 0.65], [0.4, 0.5, 0.6]]
  - [[0.35, 0.45, 0.55], [0.3, 0.4, 0.5]]

v:
  - [[0.01, 0.015, 0.02], [0.008, 0.012, 0.016]]
  - [[0.015, 0.02, 0.025], [0.012, 0.016, 0.02]]
  - [[0.012, 0.018, 0.022], [0.01, 0.014, 0.018]]
  - [[0.008, 0.012, 0.016], [0.006, 0.01, 0.014]]
```

## Output file contents

The output file includes:

| Key | Description |
|---|---|
| `status` | Solver status (`"optimal"` or Gurobi status code) |
| `objective` | Optimal objective value (or `null`) |
| `degradation` | Degradation model used |
| `F`, `M`, `H` | Problem dimensions |
| `mu_0`, `v_0` | Initial conditions |
| `x` | Binary assignment solution (F x (M+1) x 2H) |
| `mu` | Mean degradation solution (F x 2H) |
| `v` | Variance degradation solution (F x 2H) |
| `z` | Auxiliary variable solution (F x 2H) |

## Project structure

```
Fleet_Management/
    pyproject.toml
    README.md
    src/
        fleet_management/
            __init__.py        # Public API: solve, plot_management
            solver.py          # Mid-layer: I/O, validation, dispatch
            gaussian.py        # Gaussian degradation MILP (Gurobi)
            plotter.py         # Schedule visualisation
```
