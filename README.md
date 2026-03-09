# Fleet Management

Train fleet scheduling optimization with degradation models (Gaussian and inverse Gaussian), solved as a Mixed-Integer Linear Program (MILP) using Gurobi.

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

# 1. Solve with Gaussian degradation
solve("input/data.yaml", degradation="gaussian", results_path="results/output.yaml")

# 2. Solve with inverse Gaussian degradation
solve("input/data_ig.yaml", degradation="inverse_gaussian", results_path="results/output_ig.yaml")

# 3. Plot the resulting schedule
plot_management("results/output.yaml", plot_file_path="results/schedule.png")
```

## API reference

### `solve(input_path, degradation, results_path=None)`

Reads the problem data, solves the MILP, and writes the results to a file.

| Parameter | Type | Description |
|---|---|---|
| `input_path` | `str` | Path to the input data file. |
| `degradation` | `str` | Degradation model. Supported: `"gaussian"`, `"inverse_gaussian"`. |
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

The plot is an F x (2H+1) grid where each cell is coloured on a green-to-red heatmap (0 to alpha) based on the degradation mean. Cell annotations indicate:

- **Number** (j): mission j is assigned (`x[i,j,k] = 1`, j > 0)
- **Gear icon**: maintenance is scheduled (`x[i,0,k] = 1`)
- **"zzz" cloud**: the train is idle

## Input file format

### Common keys (both models)

| Key | Type | Description |
|---|---|---|
| `F` | int | Number of trains (must be > M) |
| `H` | int | Time horizon (model spans 2H steps) |
| `M` | int | Number of missions |
| `mu` | array (F x M x H) or (F x M) | Mean degradation parameters (must be < alpha). If 2D, the same values are used for all time steps. |
| `alpha` | float | Upper bound for degradation mean (must be positive) |
| `epsilon` | float | Reliability threshold, in (0, 0.5) |
| `C_M` | float | Maintenance cost coefficient |
| `C_R` | float | Repair cost coefficient |
| `C_S` | float | Safety cost coefficient |
| `C_P` | float | Penalty cost coefficient |
| `mu_0` | 1D array (F) | Initial mean degradation values per train (must be < alpha) |
| `verbose` | int, optional | Gurobi verbosity (0 = silent, 1 = normal). Default: 1 |
| `mip_gap` | float, optional | Relative MIP optimality gap tolerance. Default: Gurobi default (1e-4) |

### Gaussian-specific keys

| Key | Type | Description |
|---|---|---|
| `v` | array (F x M x H) or (F x M) | Variance degradation parameters. If 2D, the same values are used for all time steps. |
| `v_0` | 1D array (F) | Initial variance values per train |

Additional constraints: `mu >= 3*sqrt(v)` and `mu_0 >= 3*sqrt(v_0)`.

### Inverse Gaussian-specific keys

| Key | Type | Description |
|---|---|---|
| `c` | 1D array (F) | Shape parameter per train (must be positive) |

### YAML example

```yaml
F: 6
H: 10
M: 3
alpha: 1.0
epsilon: 0.1
C_M: 1.0
C_R: 2.0
C_S: 1.5
C_P: 3.0
verbose: 1
mip_gap: 0.12

mu_0: [0.1341, 0.1113, 0.1925, 0.1877, 0.1258, 0.166]
v_0: [0.000799, 0.000551, 0.001647, 0.001566, 0.000703, 0.001225]

mu:
  - - [0.1375, 0.1951, 0.1732, 0.1599, 0.1156, 0.1156, 0.1058, 0.1866, 0.1601, 0.1708]
    - [0.1021, 0.1970, 0.1832, 0.1212, 0.1182, 0.1183, 0.1304, 0.1525, 0.1432, 0.1291]
    - [0.1612, 0.1139, 0.1292, 0.1366, 0.1456, 0.1785, 0.1200, 0.1514, 0.1592, 0.1046]
  # ... (6 x 3 x 10 tensor)

v:
  - - [0.000840, 0.001692, 0.001333, 0.001136, 0.000594, 0.000594, 0.000497, 0.001548, 0.001139, 0.001297]
    - [0.000463, 0.001725, 0.001492, 0.000653, 0.000621, 0.000622, 0.000756, 0.001034, 0.000911, 0.000741]
    - [0.001155, 0.000577, 0.000742, 0.000829, 0.000942, 0.001416, 0.000640, 0.001019, 0.001126, 0.000486]
  # ... (6 x 3 x 10 tensor)
```

## Output file contents

The output file includes:

| Key | Description |
|---|---|
| `status` | Solver status (`"optimal"` or Gurobi status code) |
| `objective` | Optimal objective value (or `null`) |
| `degradation` | Degradation model used |
| `F`, `M`, `H` | Problem dimensions (trains, missions, time horizon) |
| `alpha` | Upper bound for degradation mean |
| `mu_0`, `v_0` | Initial conditions (`v_0` only for Gaussian) |
| `x` | Binary assignment solution (F x (M+1) x 2H) |
| `mu` | Mean degradation solution (F x 2H) |
| `v` | Variance degradation solution (F x 2H, Gaussian only) |
| `u` | Max degradation mean per time step (2H) |
| `z` | Degradation level at repair per train (F x 2H, non-zero only when maintenance is scheduled) |

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
            inverse_gaussian.py # Inverse Gaussian degradation MILP (Gurobi)
            plotter.py         # Schedule visualisation
```
