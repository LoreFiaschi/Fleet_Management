# Fleet Management

Train fleet scheduling optimization with multi-component degradation models (Gaussian and inverse Gaussian), solved as a Mixed-Integer Linear Program (MILP) using Gurobi.

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

The plot is an F x (2H+1) grid where each cell is split into L horizontal strips (one per component), coloured on a green-to-red heatmap (0 to alpha) based on the component's degradation mean. When L=1, each cell shows a single colour as before. Cell annotations indicate:

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
| `L` | int, optional | Number of components per train. Default: 1 |
| `mu` | array (F x M x L x H) or (F x M x L) | Mean degradation parameters (must be < alpha). If 3D, the same values are used for all time steps. When L=1, legacy shapes (F x M x H) and (F x M) are also accepted. |
| `alpha` | float | Upper bound for degradation mean (must be positive) |
| `epsilon` | float | Reliability threshold, in (0, 0.5) |
| `xi` | array (F x L) | Fraction of damage repairable in one maintenance day per train and component, in (0, 1] element-wise. When L=1, a 1D array of length F is also accepted. |
| `C_M` | float | Maintenance cost coefficient |
| `C_R` | float | Repair cost coefficient |
| `C_S` | float | Safety cost coefficient |
| `C_P` | float | Penalty cost coefficient |
| `mu_0` | array (F x L) | Initial mean degradation values per train and component (must be < alpha). When L=1, a 1D array of length F is also accepted. |
| `verbose` | int, optional | Gurobi verbosity (0 = silent, 1 = normal). Default: 1 |
| `mip_gap` | float, optional | Relative MIP optimality gap tolerance. Default: Gurobi default (1e-4) |

### Gaussian-specific keys

| Key | Type | Description |
|---|---|---|
| `v` | array (F x M x L x H) or (F x M x L) | Variance degradation parameters. If 3D, the same values are used for all time steps. When L=1, legacy shapes (F x M x H) and (F x M) are also accepted. |
| `v_0` | array (F x L) | Initial variance values per train and component. When L=1, a 1D array of length F is also accepted. |

Additional constraints: `mu >= 3*sqrt(v)` and `mu_0 >= 3*sqrt(v_0)`.

### Inverse Gaussian-specific keys

| Key | Type | Description |
|---|---|---|
| `c` | array (F x L) | Shape parameter per train and component (must be positive). When L=1, a 1D array of length F is also accepted. |

### YAML example

```yaml
F: 6
H: 10
M: 3
L: 2
alpha: 1.0
epsilon: 0.1
xi:
  - [0.8, 0.75]
  - [0.75, 0.9]
  - [0.9, 0.85]
  - [0.85, 0.7]
  - [0.7, 0.8]
  - [0.8, 0.85]
C_M: 1.0
C_R: 2.0
C_S: 1.5
C_P: 3.0
verbose: 1
mip_gap: 0.12

mu_0:
  - [0.1341, 0.1200]
  - [0.1113, 0.1050]
  - [0.1925, 0.1800]
  - [0.1877, 0.1750]
  - [0.1258, 0.1100]
  - [0.1660, 0.1500]
v_0:
  - [0.000799, 0.000640]
  - [0.000551, 0.000490]
  - [0.001647, 0.001440]
  - [0.001566, 0.001361]
  - [0.000703, 0.000538]
  - [0.001225, 0.001000]

mu:  # F x M x L x H tensor
  - - - [0.1375, 0.1951, 0.1732, 0.1599, 0.1156, 0.1156, 0.1058, 0.1866, 0.1601, 0.1708]
      - [0.1300, 0.1850, 0.1650, 0.1500, 0.1100, 0.1100, 0.1000, 0.1780, 0.1520, 0.1620]
    - - [0.1021, 0.1970, 0.1832, 0.1212, 0.1182, 0.1183, 0.1304, 0.1525, 0.1432, 0.1291]
      - [0.0970, 0.1870, 0.1740, 0.1150, 0.1120, 0.1130, 0.1240, 0.1450, 0.1360, 0.1230]
    - - [0.1612, 0.1139, 0.1292, 0.1366, 0.1456, 0.1785, 0.1200, 0.1514, 0.1592, 0.1046]
      - [0.1530, 0.1080, 0.1230, 0.1300, 0.1380, 0.1700, 0.1140, 0.1440, 0.1510, 0.0990]
  # ... (6 x 3 x 2 x 10 tensor)

v:  # F x M x L x H tensor
  - - - [0.000840, 0.001692, 0.001333, 0.001136, 0.000594, 0.000594, 0.000497, 0.001548, 0.001139, 0.001297]
      - [0.000751, 0.001521, 0.001200, 0.001000, 0.000538, 0.000538, 0.000444, 0.001408, 0.001027, 0.001167]
    - - [0.000463, 0.001725, 0.001492, 0.000653, 0.000621, 0.000622, 0.000756, 0.001034, 0.000911, 0.000741]
      - [0.000418, 0.001553, 0.001343, 0.000588, 0.000559, 0.000560, 0.000680, 0.000935, 0.000820, 0.000672]
    - - [0.001155, 0.000577, 0.000742, 0.000829, 0.000942, 0.001416, 0.000640, 0.001019, 0.001126, 0.000486]
      - [0.001040, 0.000519, 0.000668, 0.000751, 0.000848, 0.001284, 0.000576, 0.000921, 0.001013, 0.000438]
  # ... (6 x 3 x 2 x 10 tensor)
```

## Output file contents

The output file includes:

| Key | Description |
|---|---|
| `status` | Solver status (`"optimal"` or Gurobi status code) |
| `objective` | Optimal objective value (or `null`) |
| `degradation` | Degradation model used |
| `F`, `M`, `H`, `L` | Problem dimensions (trains, missions, time horizon, components) |
| `alpha` | Upper bound for degradation mean |
| `mu_0`, `v_0` | Initial conditions (`v_0` only for Gaussian), shape F x L |
| `x` | Binary assignment solution (F x (M+1) x 2H) |
| `mu` | Mean degradation solution (F x L x 2H) |
| `v` | Variance degradation solution (F x L x 2H, Gaussian only) |
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
