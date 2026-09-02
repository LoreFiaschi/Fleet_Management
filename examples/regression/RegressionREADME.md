# Regression and experiment scripts

This directory contains small, deterministic checks for the fleet-management
solver. They are kept outside the production package because they are
executable regression scenarios rather than public solver functions.

The checks serve three purposes:

1. preserve mathematical assumptions that are easy to break during refactoring;
2. verify that YAML inputs still reach the intended public solver backend; and
3. record formulation sizes and known outputs for comparison after model changes.

The scenarios are intentionally small. Passing them establishes consistency of
the implementation; it does **not** establish scalability or physical validity
for a real fleet dataset.

## Running checks

Run commands from the repository root with the project environment activated.

```powershell
python .\examples\regression\check_gamma_public_interface.py
```

Run every `check_*.py` script in name order and stop at the first failure:

```powershell
$checks = Get-ChildItem .\examples\regression\check_*.py | Sort-Object Name
foreach ($check in $checks) {
    Write-Host "`n=== $($check.Name) ==="
    python $check.FullName
    if ($LASTEXITCODE -ne 0) {
        throw "Regression failed: $($check.Name)"
    }
}
```

Checks marked **solver** require `gurobipy` and a valid Gurobi licence. Checks
marked **numerical** use NumPy/SciPy only, although importing the installed
package may still require the dependencies declared in `pyproject.toml`.

## Core public-interface regressions

| Script | Type | What it protects | Why it is kept |
|---|---|---|---|
| `check_gamma_public_interface.py` | solver | YAML → `FleetConfig` → modular Gamma solve | Detects accidental routing back to the legacy Gamma backend. |
| `check_mixed_public_interface.py` | solver | Public mixed Gamma/rainflow solve and lightweight replay | Protects the main mixed-model workflow. |
| `check_mixed_gamma_ard1_public.py` | solver | Public mixed solve with Gamma ARD1 repairs | Protects ARD1 integration, latch states and mixed routing. |
| `check_gamma_cell_integration.py` | solver | Direct modular Gamma-cell construction | Isolates the Gamma block from public file I/O. |
| `check_gamma_replay_validator.py` | solver | Lightweight schedule/state replay and corruption detection | Protects the validator used by the current public workflow. |

## Gamma calibration and numerical evidence

| Script | Type | What it protects | Why it is kept |
|---|---|---|---|
| `check_gamma_repeated_calibration.py` | numerical | Current repeated-increment \(m^*,\beta^*,\alpha^*\) calibration | Directly tests the calibration contract. |
| `check_gamma_tail_bound.py` | numerical | Earlier finite-count calibration and broadcasting | Retained as a comparison/regression path while both calibration methods remain in the code. |
| `check_gamma_convolution_quadrature.py` | numerical | Moschopoulos convolution against independent quadrature | Ensures the production tail calculation is not self-validating. |
| `check_gamma_randomized_properties.py` | numerical | Fixed-seed randomized shapes, rates and count combinations | Broadens coverage beyond hand-selected examples. |
| `gamma_quadrature_reference.py` | support | Independent quadrature helper | Imported by the numerical checks; it is not an executable regression itself. |

## Gamma dynamics and repair regressions

| Script | Type | What it protects | Why it is kept |
|---|---|---|---|
| `check_gamma_big_m_formulation.py` | solver | Tight, time-dependent Big-M Gamma dynamics | Confirms Gamma introduces no indicator constraints and uses bounded states. |
| `check_gamma_repair_integration.py` | solver | Fixed-rate ARD-infinity shape scaling | Verifies physical mean, bounding shape and removed damage. |
| `check_gamma_ard1_integration.py` | solver | ARD1 latch dynamics, repeated repairs and complete repair | Protects the more complex repair state transition. |
| `check_gamma_repair_legacy.py` | numerical | Earlier signed repair-tail calculation | Retained only as mathematical comparison evidence. |

## Exact/internal certification checks

These checks are **not** part of the minimal public post-solve workflow. They
are retained as internal numerical evidence for the Gamma probability routines.

| Script | Type | What it protects |
|---|---|---|
| `check_gamma_exact_validator.py` | solver + numerical | Exact reconstruction of a solved Gamma schedule. |
| `check_gamma_validator_edge_cases.py` | solver + numerical | Zero seeds, repeated repairs and complete-repair edge cases. |

The production workflow uses `gamma_replay_validator.py`, which replays the
reported schedule and states without performing exact convolution.

## Horizon, formulation and performance diagnostics

| Script | Type | What it protects or produces |
|---|---|---|
| `check_operating_average_objective.py` | solver | Single operating-phase objective \(J_{\mathrm{op}}/H_2\), including its bound and MIP gap. The initialization phase remains constrained but has no cost budget. |
| `check_horizon_sweep.py` | solver | Certified objective bounds/MIP gaps, proven-versus-feasible selection and the gap-qualified gradient stopping rule. |
| `run_horizon_sweep.py` | runner | Solves an explicit list or inclusive range of operating horizons and writes a compact YAML report. |
| `check_formulation_sweep.py` | analytical | Deterministic one-factor-at-a-time \(F/M/L/T\) formulation counts. |
| `run_formulation_size_sweep.py` | runner | Writes the full analytical formulation-size sweep report. |
| `check_gamma_complexity_diagnostics.py` | solver | Predicted counts against actual Gurobi model statistics. |
| `report_gamma_complexity.py` | runner | Writes calibration, formulation, optimization and replay diagnostics. |

Examples:

```powershell
python .\examples\regression\run_horizon_sweep.py `
  .\input\gamma_horizon_local.yaml `
  .\results\gamma_horizon_local_sweep.yaml `
  --h2-range 2 20 `
  --stop-on-gradient `
  --gradient-tolerance 0.001 `
  --maximum-stopping-gap 0.05
```

The stopping tolerance is a relative change in `J_op/H2` per added unit of
`H2`. By default, two consecutive flat gradients are required. A positive
gradient stops after its first gap-qualified occurrence. Cases whose MIP gap
exceeds `--maximum-stopping-gap` are recorded but cannot stop the sweep. The
upper value supplied through `--h2-range` is always a hard safety limit.
The YAML output is rewritten after every completed horizon. If Slurm stops the
job before the complete candidate range is evaluated, the file remains a valid
checkpoint with `complete: false` and identifies `last_completed_H2`.
Every case records total, continuous, integer and binary variable counts, as
well as linear constraints and solver diagnostics.

```powershell
python .\examples\regression\run_formulation_size_sweep.py `
  .\input\gamma_horizon_euler.yaml `
  .\results\gamma_formulation_sweep.yaml
```

```powershell
python .\examples\regression\report_gamma_complexity.py `
  --output .\results\gamma_complexity.yaml
```

## Legacy-backend regressions

| Script | Type | Reason retained |
|---|---|---|
| `check_gamma_incumbent_extraction.py` | solver | Confirms that the standalone legacy Gamma backend retains a feasible incumbent when stopped early. |

The legacy backend remains a regression oracle. New model development should
target the modular builder rather than `gamma_gurobi.py`.

## Scenario files

| File | Purpose |
|---|---|
| `gamma_tail_bound_public.yaml` | Uniform modular Gamma public-interface scenario. |
| `mixed_gamma_rainflow_public.yaml` | Mixed Gamma/rainflow ARD-infinity scenario. |
| `mixed_gamma_ard1_public.yaml` | Mixed Gamma/rainflow ARD1 scenario. |

## Maintenance rules

- A regression must print `PASS ...` and exit nonzero on failure.
- Randomized checks must use a fixed seed and report it.
- Do not assert wall-clock timing as an exact value.
- Formulation counts may be asserted exactly for a fixed code/input contract.
- New public solver behavior requires at least one public-interface regression.
- Prefer extending an existing check over adding another nearly identical file.
- Do not delete exact/internal checks merely because they are not in the public
  workflow; remove them only together with the mathematical functionality they
  certify.
