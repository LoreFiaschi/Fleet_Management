# Validation for fleet_management
# author: Christoph Langenauer
# date of creation: 10.06.2026

import json
import os
from pathlib import Path

import numpy as np
import yaml
import pandas as pd

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

    elif degradation_lower == "inverse_gaussian":
        raise NotImplementedError(
            "Inverse-Gaussian-specific validation is not implemented yet. "
            "Only Gaussian validation is currently supported."
        )

    report["passed"] = all(check["passed"] for check in report["checks"])
    report["max_violation"] = max(check["violation"] for check in report["checks"])

    _save_validation_report(report, validation_file)

    print(f"Validation written to: {validation_file}")
    print(f"Validation passed: {report['passed']}")
    print(f"Maximum violation: {report['max_violation']:.3e}")

    return report


def validate_baseline_assignment_feasibility(
    input_path: str,
    results_path: str,
    log_path: str = "results/baseline_assignment_feasibility.log",
    tol: float = 1e-6,
    alpha_override: float | None = None,
    degradation_scale: float = 1.0,
) -> dict:
    """
    Naive baseline feasibility check for the existing Gaussian baseline dataset.

    This is a reporting wrapper around build_assignment_feasibility_dataframe(...).
    It writes a readable log file and returns a summary dictionary.
    """

    input_file = Path(input_path)
    results_file = Path(results_path)
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    df = build_assignment_feasibility_dataframe(
        input_path=input_path,
        results_path=results_path,
        tol=tol,
        alpha_override=alpha_override,
        degradation_scale=degradation_scale,
    )

    if df.empty:
        total_assignments = 0
        feasible_assignments = 0
        infeasible_count = 0
        maintenance_events = 0
        alpha_original = None
        alpha_effective = None
        degradation_scale_used = degradation_scale
    else:
        # Count assignments, not component rows.
        assignment_df = df.drop_duplicates(
            subset=["time_step", "vehicle", "mission"]
        )

        total_assignments = int(len(assignment_df))
        feasible_assignments = int(assignment_df["assignment_feasible"].sum())
        infeasible_count = int(total_assignments - feasible_assignments)

        alpha_original = float(df["alpha_original"].iloc[0])
        alpha_effective = float(df["alpha_effective"].iloc[0])
        degradation_scale_used = float(df["degradation_scale"].iloc[0])

        maintenance_events = int(
            df[["time_step", "vehicle", "maintenance_scheduled_same_vehicle"]]
            .drop_duplicates()
            ["maintenance_scheduled_same_vehicle"]
            .sum()
        )

    lines = []
    lines.append("=" * 88)
    lines.append("Baseline assignment feasibility diagnostic")
    lines.append("=" * 88)
    lines.append(f"Input file:   {input_file}")
    lines.append(f"Results file: {results_file}")

    if not df.empty:
        F = int(df["vehicle"].max()) + 1
        M = int(df["mission"].max()) + 1
        L = int(df["component"].max()) + 1
        H2 = int(df["time_step"].max()) + 1
        H = H2 // 2

        lines.append(f"F={F}, M={M}, L={L}, H={H}, output horizon={H2}")
        lines.append(f"Original failure threshold alpha={alpha_original}")
        lines.append(f"Effective failure threshold alpha={alpha_effective}")
        lines.append(f"Degradation scale={degradation_scale_used}")

    lines.append("")
    lines.append(
        "Interpretation: input mu[i,j,l,k] is treated as expected degradation "
        "increment for vehicle i, mission j, component l, day k."
    )
    lines.append("")
    lines.append("-" * 88)
    lines.append("Actual solver assignments")
    lines.append("-" * 88)

    # Group component rows back into assignment blocks for readable logging.
    grouped = df.groupby(["time_step", "input_day", "vehicle", "mission"], sort=True)

    for (k, input_day, i, j), group in grouped:
        assignment_feasible = bool(group["assignment_feasible"].iloc[0])
        status = (
            "FEASIBLE"
            if assignment_feasible
            else "REQUIRES MAINTENANCE / INFEASIBLE BEFORE ASSIGNMENT"
        )

        lines.append(
            f"k={int(k):02d}, input_day={int(input_day):02d}, "
            f"vehicle={int(i)}, mission={int(j)}: {status}"
        )

        for _, row in group.sort_values("component").iterrows():
            component_status = "OK" if row["feasible"] else "ABOVE THRESHOLD"

            lines.append(
                f"    component={int(row['component'])}: "
                f"before={row['damage_before']:.6f}, "
                f"increment={row['expected_increment']:.6f}, "
                f"after={row['damage_after']:.6f}, "
                f"threshold={row['threshold']:.6f} -> {component_status}"
            )

    # Add maintenance events separately from solver output, because the dataframe
    # only contains active mission assignments.
    results_data = _read_results(results_file)
    results_params = _extract_results_parameters(results_data)
    x = results_params["x"]

    for k in range(x.shape[2]):
        for i in range(x.shape[0]):
            if abs(x[i, 0, k] - 1.0) <= tol:
                lines.append(
                    f"k={k:02d}, vehicle={i}: MAINTENANCE scheduled "
                    f"(x[{i},0,{k}] = {x[i,0,k]:.3f})"
                )

    lines.append("")
    lines.append("-" * 88)
    lines.append("Summary")
    lines.append("-" * 88)
    lines.append(f"Total assigned vehicle-mission-time entries: {total_assignments}")
    lines.append(f"Feasible assigned entries:                   {feasible_assignments}")
    lines.append(f"Infeasible / maintenance-required entries:   {infeasible_count}")
    lines.append(f"Maintenance events found:                    {maintenance_events}")

    failed_assignments = (
        df[~df["assignment_feasible"]]
        .drop_duplicates(subset=["time_step", "vehicle", "mission"])
        if not df.empty
        else pd.DataFrame()
    )

    if not failed_assignments.empty:
        lines.append("")
        lines.append("Infeasible assigned entries:")
        for _, row in failed_assignments.iterrows():
            lines.append(
                f"  k={int(row['time_step']):02d}, "
                f"vehicle={int(row['vehicle'])}, "
                f"mission={int(row['mission'])}, "
                f"violating_components={row['violating_components']}"
            )
    else:
        lines.append("")
        lines.append("No infeasible assigned entries were found by this naive check.")

    lines.append("=" * 88)

    with open(log_file, "w") as f:
        f.write("\n".join(lines))

    report = {
        "passed": infeasible_count == 0,
        "alpha_original": alpha_original,
        "alpha_effective": alpha_effective,
        "degradation_scale": degradation_scale_used,
        "total_assignments": total_assignments,
        "feasible_assignments": feasible_assignments,
        "infeasible_assignments": infeasible_count,
        "maintenance_events": maintenance_events,
        "log_path": str(log_file),
        "details": df.to_dict(orient="records"),
    }

    return report


def build_assignment_feasibility_dataframe(
    input_path: str,
    results_path: str,
    tol: float = 1e-6,
    alpha_override: float | None = None,
    degradation_scale: float = 1.0,
) -> pd.DataFrame:
    """
    Build a component-level assignment feasibility table from input and solver output.

    Each row corresponds to one active assignment x[i,j+1,k] = 1 and one component l.
    The input mu[i,j,l,k] is interpreted as the expected mission degradation increment.
    """

    input_file = Path(input_path)
    results_file = Path(results_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not results_file.exists():
        raise FileNotFoundError(f"Result file not found: {results_path}")

    input_data = _read_input(input_file)
    input_params = _extract_parameters(input_data, "gaussian")

    results_data = _read_results(results_file)
    results_params = _extract_results_parameters(results_data)

    if results_params["degradation"] != "gaussian":
        raise ValueError(
            "This dataframe builder currently expects a Gaussian baseline result file."
        )

    F = input_params["F"]
    M = input_params["M"]
    L = input_params["L"]
    H = input_params["H"]

    alpha_original = float(input_params["alpha"])
    alpha = alpha_original if alpha_override is None else float(alpha_override)
    degradation_scale = float(degradation_scale)

    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha}.")

    if degradation_scale <= 0.0:
        raise ValueError(
            f"degradation_scale must be positive, got {degradation_scale}."
        )

    mu_param = input_params["mu_param"]      # F x M x L x H
    mu_result = results_params["mu"]         # F x L x 2H
    x = results_params["x"]                  # F x (M+1) x 2H

    expected_x_shape = (F, M + 1, 2 * H)
    expected_mu_result_shape = (F, L, 2 * H)
    expected_mu_param_shape = (F, M, L, H)

    if x.shape != expected_x_shape:
        raise ValueError(f"x shape {x.shape}, expected {expected_x_shape}.")

    if mu_result.shape != expected_mu_result_shape:
        raise ValueError(
            f"result mu shape {mu_result.shape}, expected {expected_mu_result_shape}."
        )

    if mu_param.shape != expected_mu_param_shape:
        raise ValueError(
            f"input mu shape {mu_param.shape}, expected {expected_mu_param_shape}."
        )

    rows = []

    for k in range(2 * H):
        # Solver output uses 2H time steps, while the input degradation
        # increments are specified over H days.
        k_input = k % H

        for i in range(F):
            maintenance_scheduled = abs(x[i, 0, k] - 1.0) <= tol

            for j in range(M):
                x_value = x[i, j + 1, k]

                if abs(x_value - 1.0) <= tol:
                    if k == 0:
                        damage_before = input_params["mu_0"][i, :]
                    else:
                        damage_before = mu_result[i, :, k - 1]

                    expected_increment = (
                        degradation_scale * mu_param[i, j, :, k_input]
                    )
                    damage_after = damage_before + expected_increment

                    component_feasible = damage_after <= alpha + tol
                    assignment_feasible = bool(np.all(component_feasible))
                    violating_components = np.where(~component_feasible)[0].tolist()

                    for l in range(L):
                        margin = alpha - damage_after[l]
                        feasible = bool(component_feasible[l])

                        rows.append(
                            {
                                "time_step": k,
                                "input_day": k_input,
                                "vehicle": i,
                                "mission": j,
                                "component": l,
                                "x_value": float(x_value),
                                "maintenance_scheduled_same_vehicle": bool(
                                    maintenance_scheduled
                                ),
                                "damage_before": float(damage_before[l]),
                                "expected_increment": float(expected_increment[l]),
                                "damage_after": float(damage_after[l]),
                                "threshold": float(alpha),
                                "margin_to_threshold": float(margin),
                                "utilization_of_threshold": float(damage_after[l] / alpha),
                                "feasible": feasible,
                                "assignment_feasible": assignment_feasible,
                                "violating_components": violating_components,
                                "status": "OK" if feasible else "ABOVE_THRESHOLD",
                                "alpha_original": alpha_original,
                                "alpha_effective": float(alpha),
                                "degradation_scale": float(degradation_scale),
                            }
                        )

    columns = [
        "time_step",
        "input_day",
        "vehicle",
        "mission",
        "component",
        "x_value",
        "maintenance_scheduled_same_vehicle",
        "damage_before",
        "expected_increment",
        "damage_after",
        "threshold",
        "margin_to_threshold",
        "utilization_of_threshold",
        "feasible",
        "assignment_feasible",
        "violating_components",
        "status",
        "alpha_original",
        "alpha_effective",
        "degradation_scale",
    ]

    return pd.DataFrame(rows, columns=columns)


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