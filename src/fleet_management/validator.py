# Validation for fleet_management
# author: Christoph Langenauer
# date of creation: 10.06.2026

import json
import os
from pathlib import Path

import numpy as np
import yaml

from fleet_management.solver import _read_input, _extract_parameters


SUPPORTED_DEGRADATIONS = {"gaussian", "inverse_gaussian"}
SUPPORTED_RESULT_EXTENSIONS = {".yaml", ".yml", ".json"}


def validate(
    input_path: str,
    degradation: str,
    results_path: str,
    validation_path: str = "validation.yaml",
    tol: float = 1e-6,
) -> dict:
    """
    Validate Gurobi output against the implemented model structure.

    Currently validates:
    - solver status
    - result dimensions
    - x binary
    - assignment constraints
    - demand constraints
    - u >= mu
    - capacity constraint
    - periodic mu constraint
    - periodic v constraint for Gaussian degradation
    - Gaussian objective recomputation

    Parameters
    ----------
    input_path : str
        Path to initial conditions and problem data.

    degradation : str
        Type of degradation model. Supported: "gaussian", "inverse_gaussian".

    results_path : str
        Path to results from Gurobi.

    validation_path : str
        Path to save validation report.

    tol : float
        Numerical tolerance for feasibility checks.

    Returns
    -------
    dict
        Validation report.
    """

    # --- Consistency checks ---
    degradation_lower = degradation.lower()

    if degradation_lower not in SUPPORTED_DEGRADATIONS:
        raise ValueError(
            f"Unsupported degradation type '{degradation}'. "
            f"Supported types: {sorted(SUPPORTED_DEGRADATIONS)}"
        )

    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    results_file = Path(results_path)
    if not results_file.exists():
        raise FileNotFoundError(f"Result file not found: {results_path}")
    if results_file.suffix.lower() not in SUPPORTED_RESULT_EXTENSIONS:
        raise ValueError(
            f"Unsupported result file type '{results_file.suffix}'. "
            f"Supported types: {sorted(SUPPORTED_RESULT_EXTENSIONS)}"
        )

    validation_file = _resolve_validation_path(validation_path)
    if validation_file.suffix.lower() not in SUPPORTED_RESULT_EXTENSIONS:
        raise ValueError(
            f"Unsupported validation file type '{validation_file.suffix}'. "
            f"Supported types: {sorted(SUPPORTED_RESULT_EXTENSIONS)}"
        )

    validation_dir = validation_file.parent
    validation_dir.mkdir(parents=True, exist_ok=True)

    if not os.access(validation_dir, os.W_OK):
        raise PermissionError(f"Validation directory is not writable: {validation_dir}")

    # --- Read and parse input/results ---
    input_data = _read_input(input_file)
    input_params = _extract_parameters(input_data, degradation_lower)

    results_data = _read_results(results_file)
    results_params = _extract_results_parameters(results_data)

    if results_params["degradation"] != degradation_lower:
        raise ValueError(
            f"Degradation mismatch: function argument is '{degradation_lower}', "
            f"but result file contains '{results_params['degradation']}'."
        )

    _check_dimensions(input_params, results_params)

    # --- Validation report ---
    report = {
        "status": results_params["status"],
        "degradation": degradation_lower,
        "solver_objective": results_params["objective"],
        "tolerance": tol,
        "checks": [],
    }

    # --- Basic checks ---
    _add_check(
        report,
        name="solver_status_optimal",
        passed=results_params["status"] == "optimal",
        violation=0.0,
        tol=tol,
    )

    x = results_params["x"]
    mu = results_params["mu"]
    u = results_params["u"]

    F = results_params["F"]
    M = results_params["M"]
    H = results_params["H"]
    L = results_params["L"]

    # x binary
    binary_violation = float(np.max(np.abs(x - np.round(x))))
    _add_check(
        report,
        name="x_binary",
        passed=_is_binary(x, tol),
        violation=binary_violation,
        tol=tol,
    )

    # Assignment constraint:
    # sum_j x[i, j, k] <= 1 for all i, k
    assignment_violation = max(
        0.0,
        float(np.max(np.sum(x, axis=1) - 1.0)),
    )
    _add_check(
        report,
        name="assignment_sum_j_x_le_1",
        passed=assignment_violation <= tol,
        violation=assignment_violation,
        tol=tol,
    )

    # Demand constraint:
    # sum_i x[i, j, k] == 1 for all j, k
    demand_violation = float(np.max(np.abs(np.sum(x, axis=0) - 1.0)))
    _add_check(
        report,
        name="demand_sum_i_x_eq_1",
        passed=demand_violation <= tol,
        violation=demand_violation,
        tol=tol,
    )

    # u >= mu for all i, l, k
    u_violation = max(
        0.0,
        float(np.max(np.max(mu, axis=(0, 1)) - u)),
    )
    _add_check(
        report,
        name="u_ge_mu",
        passed=u_violation <= tol,
        violation=u_violation,
        tol=tol,
    )

    # Capacity constraint:
    # sum_{i,l} mu[i,l,k] <= F - M for all k
    capacity_violation = max(
        0.0,
        float(np.max(np.sum(mu, axis=(0, 1)) - (F - M))),
    )
    _add_check(
        report,
        name="capacity_sum_mu_le_F_minus_M",
        passed=capacity_violation <= tol,
        violation=capacity_violation,
        tol=tol,
    )

    # Periodic mu constraint:
    # mu[i,l,2H-1] <= mu[i,l,H-1]
    mu_periodic_violation = max(
        0.0,
        float(np.max(mu[:, :, 2 * H - 1] - mu[:, :, H - 1])),
    )
    _add_check(
        report,
        name="mu_periodic",
        passed=mu_periodic_violation <= tol,
        violation=mu_periodic_violation,
        tol=tol,
    )

    if degradation_lower == "gaussian":
        v = results_params["v"]

        # Periodic v constraint:
        # v[i,l,2H-1] <= v[i,l,H-1]
        v_periodic_violation = max(
            0.0,
            float(np.max(v[:, :, 2 * H - 1] - v[:, :, H - 1])),
        )
        _add_check(
            report,
            name="v_periodic",
            passed=v_periodic_violation <= tol,
            violation=v_periodic_violation,
            tol=tol,
        )

        _validate_gaussian_objective(report, input_params, results_params, tol)

    report["passed"] = all(check["passed"] for check in report["checks"])
    report["max_violation"] = max(check["violation"] for check in report["checks"])

    _save_validation_report(report, validation_file)

    print(f"Validation written to: {validation_file}")
    print(f"Validation passed: {report['passed']}")
    print(f"Maximum violation: {report['max_violation']:.3e}")

    return report


def _read_results(results_file: Path) -> dict:
    """Read results data from a supported file format."""
    ext = results_file.suffix.lower()

    if ext in (".yaml", ".yml"):
        with open(results_file, "r") as f:
            return yaml.safe_load(f)

    if ext == ".json":
        with open(results_file, "r") as f:
            return json.load(f)

    raise ValueError(f"Unsupported result file type: {ext}")


def _extract_results_parameters(data: dict) -> dict:
    """Extract result parameters from parsed result data."""

    required_keys = {
        "status",
        "objective",
        "degradation",
        "F",
        "M",
        "H",
        "L",
        "alpha",
        "mu_0",
        "x",
        "mu",
        "u",
        "z",
    }

    missing = required_keys - set(data.keys())
    if missing:
        raise KeyError(f"Missing required keys in result file: {sorted(missing)}")

    F = int(data["F"])
    M = int(data["M"])
    H = int(data["H"])
    L = int(data.get("L", 1))

    params = {
        "status": str(data["status"]).lower(),
        "objective": None if data["objective"] is None else float(data["objective"]),
        "degradation": str(data["degradation"]).lower(),
        "F": F,
        "M": M,
        "H": H,
        "L": L,
        "alpha": float(data["alpha"]),
        "mu_0": _as_2d_array(data["mu_0"], F, L, "mu_0"),
        "x": np.asarray(data["x"], dtype=float),
        "mu": np.asarray(data["mu"], dtype=float),
        "u": np.asarray(data["u"], dtype=float),
        "z": np.asarray(data["z"], dtype=float),
    }

    if params["degradation"] == "gaussian":
        if "v" not in data:
            raise KeyError("Gaussian result file is missing required key: 'v'")
        if "v_0" not in data:
            raise KeyError("Gaussian result file is missing required key: 'v_0'")

        params["v"] = np.asarray(data["v"], dtype=float)
        params["v_0"] = _as_2d_array(data["v_0"], F, L, "v_0")

    return params


def _as_2d_array(value, F: int, L: int, name: str) -> np.ndarray:
    """
    Convert value to shape (F, L).

    Accepts:
    - shape (F, L)
    - shape (F,) if L == 1
    """
    arr = np.asarray(value, dtype=float)

    if arr.shape == (F, L):
        return arr

    if L == 1 and arr.shape == (F,):
        return arr[:, np.newaxis]

    raise ValueError(
        f"'{name}' shape {arr.shape} does not match expected shape "
        f"(F={F}, L={L})."
    )


def _check_dimensions(input_params: dict, results_params: dict) -> None:
    """Check that input and output dimensions match."""

    for key in ("F", "M", "H", "L"):
        if input_params[key] != results_params[key]:
            raise ValueError(
                f"Dimension mismatch for '{key}': "
                f"input has {input_params[key]}, "
                f"results have {results_params[key]}."
            )

    F = results_params["F"]
    M = results_params["M"]
    H = results_params["H"]
    L = results_params["L"]

    expected_shapes = {
        "x": (F, M + 1, 2 * H),
        "mu": (F, L, 2 * H),
        "u": (2 * H,),
        "z": (F, 2 * H),
        "mu_0": (F, L),
    }

    if results_params["degradation"] == "gaussian":
        expected_shapes["v"] = (F, L, 2 * H)
        expected_shapes["v_0"] = (F, L)

    for name, expected_shape in expected_shapes.items():
        actual_shape = results_params[name].shape

        if actual_shape != expected_shape:
            raise ValueError(
                f"Result variable '{name}' has shape {actual_shape}, "
                f"expected {expected_shape}."
            )


def _validate_gaussian_objective(
    report: dict,
    input_params: dict,
    results_params: dict,
    tol: float,
) -> None:
    """Recompute the Gaussian objective from saved variables."""

    H = results_params["H"]

    x = results_params["x"]
    mu = results_params["mu"]
    v = results_params["v"]
    u = results_params["u"]
    z = results_params["z"]

    C_M = input_params["C_M"]
    C_R = input_params["C_R"]
    C_S = input_params["C_S"]
    C_P = input_params["C_P"]

    recomputed_objective = 0.0

    recomputed_objective += C_M * float(np.sum(x[:, 0, :]))
    recomputed_objective += C_R * float(np.sum(z))
    recomputed_objective += C_S * float(np.sum(u))

    recomputed_objective += C_P * float(
        np.sum(
            mu[:, :, H - 1]
            - mu[:, :, 2 * H - 1]
            + v[:, :, H - 1]
            - v[:, :, 2 * H - 1]
        )
    )

    report["recomputed_objective"] = recomputed_objective

    if results_params["objective"] is None:
        objective_violation = float("inf")
    else:
        objective_violation = abs(recomputed_objective - results_params["objective"])

    _add_check(
        report,
        name="objective_recomputation",
        passed=objective_violation <= tol,
        violation=objective_violation,
        tol=tol,
    )


def _is_binary(x: np.ndarray, tol: float) -> bool:
    """Check whether all entries of x are numerically binary."""
    rounded = np.round(x)

    close_to_integer = np.all(np.abs(x - rounded) <= tol)
    only_zero_or_one = np.all((rounded == 0) | (rounded == 1))

    return bool(close_to_integer and only_zero_or_one)


def _add_check(
    report: dict,
    name: str,
    passed: bool,
    violation: float,
    tol: float,
) -> None:
    """Append one validation check to the report."""

    report["checks"].append(
        {
            "name": name,
            "passed": bool(passed),
            "violation": float(violation),
            "tolerance": float(tol),
        }
    )


def _resolve_validation_path(validation_path) -> Path:
    """Resolve validation path and add .yaml if no extension is given."""

    if validation_path is None:
        return Path("validation.yaml")

    path = Path(validation_path)

    if path.suffix == "":
        path = path.with_suffix(".yaml")

    return path


def _save_validation_report(report: dict, path: Path) -> None:
    """Save validation report as YAML or JSON."""

    if path.suffix.lower() in (".yaml", ".yml"):
        with open(path, "w") as f:
            yaml.safe_dump(report, f, sort_keys=False)

    elif path.suffix.lower() == ".json":
        with open(path, "w") as f:
            json.dump(report, f, indent=2)

    else:
        raise ValueError(f"Unsupported validation file type: {path.suffix}")