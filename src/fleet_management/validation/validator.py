# Validation for fleet_management
# author: Christoph Langenauer
# date of creation: 10.06.2026

import json
import os
from pathlib import Path

import numpy as np
import yaml
import pandas as pd

from fleet_management.solver import _read_input # , _extract_parameters
from fleet_management.utils.model_registry import extract_degradation_parameters
from fleet_management.degradation_model.gamma_utils.gamma_process import (
    mean_to_shape,
    shape_to_mean,
    shape_to_variance,
    failure_probability,
    reliability_passed,
    loop_constraint_passed,
)

SUPPORTED_DEGRADATIONS = {"gaussian", "inverse_gaussian", "rainflow"}
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
    input_params = extract_degradation_parameters(input_data, degradation_lower)

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
    input_params = extract_degradation_parameters(input_data, "gaussian")

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


def build_gamma_diagnostic_dataframe(
    input_path: str,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """
    Build a component-level diagnostic dataframe for a synthetic Gamma-process
    degradation instance.

    This function does not call the solver. It reads a Gamma input file that
    already contains a synthetic schedule x and propagates the Gamma damage
    state along that schedule.

    Convention
    ----------
    The Gamma process uses the shape-rate parameterisation:

        D ~ Gamma(A, beta)
        E[D]   = A / beta
        Var[D] = A / beta**2

    The input file stores expected damage increments mu[i,j,l,k]. These are
    converted into Gamma shape increments by

        A_increment[i,j,l,k] = beta[l] * mu[i,j,l,k]

    because E[Delta D] = A_increment / beta.

    Parameters
    ----------
    input_path : str
        Path to a Gamma YAML/JSON input file containing:
        F, M, L, H, tau, epsilon, gamma_beta, mu_0, mu, x.

    tol : float
        Numerical tolerance for detecting active binary assignments.

    Returns
    -------
    pandas.DataFrame
        Component-level diagnostic table. Each row corresponds to one
        vehicle-component-time entry.
    """

    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    input_data = _read_input(input_file)

    if input_data is None:
        raise ValueError(
            f"Input file is empty or contains only comments: {input_path}"
        )

    params = extract_degradation_parameters(input_data, "gamma")

    maintenance_policy = str(input_data.get("maintenance_policy", "none")).lower()

    supported_policies = {"none", "replacement"}

    if maintenance_policy not in supported_policies:
        raise ValueError(
            f"Unsupported maintenance_policy '{maintenance_policy}'."
            f"Supported policies: {sorted(supported_policies)}"
        )

    if "x" not in input_data:
        raise KeyError(
            "Gamma diagnostic input file must contain a synthetic schedule 'x'."
        )

    F = params["F"]
    M = params["M"]
    L = params["L"]
    H = params["H"]

    tau = params["tau"]
    epsilon = params["epsilon"]
    gamma_beta = params["gamma_beta"]      # shape: (L,)
    mu_0 = params["mu_0"]                  # shape: (F, L)
    mu_param = params["mu_param"]          # shape: (F, M, L, H)

    x = np.asarray(input_data["x"], dtype=float)
    expected_x_shape = (F, M + 1, 2 * H)

    if x.shape != expected_x_shape:
        raise ValueError(
            f"'x' shape {x.shape} does not match expected shape "
            f"{expected_x_shape}."
        )

    if np.any(x < -tol) or np.any(x > 1.0 + tol):
        raise ValueError("'x' contains values outside [0, 1].")

    horizon = 2 * H

    # Initial expected damage mu_0 is converted to initial Gamma shape.
    # beta has shape (L,), so NumPy broadcasts over vehicles.
    current_shape = mean_to_shape(mu_0, gamma_beta)  # shape: (F, L)

    rows = []

    for k in range(horizon):
        k_input = k % H

        for i in range(F):
            maintenance_or_idle = abs(x[i, 0, k] - 1.0) <= tol

            active_missions = [
                j for j in range(M)
                if abs(x[i, j + 1, k] - 1.0) <= tol
            ]

            if maintenance_or_idle and active_missions:
                raise ValueError(
                    f"Invalid schedule at vehicle={i}, time_step={k}: "
                    "maintenance/idle and mission are both active."
                )

            if len(active_missions) > 1:
                raise ValueError(
                    f"Invalid schedule at vehicle={i}, time_step={k}: "
                    f"multiple missions active: {active_missions}."
                )

            if not maintenance_or_idle and len(active_missions) == 0:
                raise ValueError(
                    f"Invalid schedule at vehicle={i}, time_step={k}: "
                    "no activity selected."
                )

            if maintenance_or_idle:
                for l in range(L):
                    shape_before = float(current_shape[i, l])
                    mean_before = float(shape_to_mean(shape_before, gamma_beta[l]))

                    if maintenance_policy == "replacement":
                        # Synthetic full replacement:
                        # reset accumulated Gamma shape to zero.
                        shape_after = 0.0
                        activity_name = "replacement"
                    else:
                        # No repair/replacement:
                        # carry the state forward unchanged.
                        shape_after = shape_before
                        activity_name = "maintenance_or_idle"

                    current_shape[i, l] = shape_after

                    mean_after = float(shape_to_mean(shape_after, gamma_beta[l]))
                    variance_after = float(
                        shape_to_variance(shape_after, gamma_beta[l])
                    )
                    fail_prob_after = float(
                        failure_probability(shape_after, gamma_beta[l], tau)
                    )
                    passed = bool(fail_prob_after <= epsilon + tol)

                    rows.append(
                        {
                            "degradation": "gamma",
                            "state_interpretation": "gamma_shape_rate",
                            "time_step": k,
                            "input_day": k_input,
                            "vehicle": i,
                            "activity": activity_name,
                            "mission": None,
                            "component": l,
                            "beta": float(gamma_beta[l]),
                            "tau": float(tau),
                            "epsilon": float(epsilon),
                            "shape_before": shape_before,
                            "shape_increment": -shape_before
                            if maintenance_policy == "replacement"
                            else 0.0,
                            "shape_after": shape_after,
                            "mean_before": mean_before,
                            "mean_increment": -mean_before
                            if maintenance_policy == "replacement"
                            else 0.0,
                            "mean_after": mean_after,
                            "variance_after": variance_after,
                            "failure_probability_after": fail_prob_after,
                            "reliability_passed": passed,
                            
                            # Dashboard-compatible aliases / derived fields
                            "damage_before": mean_before,
                            "expected_increment": (-mean_before if maintenance_policy == "replacement" else 0.0),
                            "damage_after": mean_after,
                            "treshold": float(tau),
                            "margin_to_threshold": float(tau - mean_after),
                            "utilization_of_threshold": float (mean_after / tau),
                            "threshold_utilization_percent": float(100.0 * mean_after / tau),
                            "failure_probability_percent": float(100.0 * fail_prob_after),
                            "feasible": passed,

                            "status": (
                                "OK"
                                if passed
                                else "FAILURE_PROBABILITY_ABOVE_EPSILON"
                            ),
                        }
                    )

                continue

            # Exactly one mission is active.
            mission = active_missions[0]

            for l in range(L):
                shape_before = float(current_shape[i, l])
                mean_before = float(shape_to_mean(shape_before, gamma_beta[l]))

                mean_increment = float(mu_param[i, mission, l, k_input])
                shape_increment = float(
                    mean_to_shape(mean_increment, gamma_beta[l])
                )

                shape_after = shape_before + shape_increment
                current_shape[i, l] = shape_after

                mean_after = float(shape_to_mean(shape_after, gamma_beta[l]))
                variance_after = float(
                    shape_to_variance(shape_after, gamma_beta[l])
                )
                fail_prob_after = float(
                    failure_probability(shape_after, gamma_beta[l], tau)
                )
                passed = bool(fail_prob_after <= epsilon + tol)

                rows.append(
                    {
                        "degradation": "gamma",
                        "state_interpretation": "gamma_shape_rate",
                        "time_step": k,
                        "input_day": k_input,
                        "vehicle": i,
                        "activity": "mission",
                        "mission": mission,
                        "component": l,
                        "beta": float(gamma_beta[l]),
                        "tau": float(tau),
                        "epsilon": float(epsilon),
                        "shape_before": shape_before,
                        "shape_increment": shape_increment,
                        "shape_after": float(shape_after),
                        "mean_before": mean_before,
                        "mean_increment": mean_increment,
                        "mean_after": mean_after,
                        "variance_after": variance_after,
                        "failure_probability_after": fail_prob_after,
                        "reliability_passed": passed,

                        # Dashboard-compatible aliases / derived fields
                        "damage_before": mean_before,
                        "expected_increment": mean_increment,
                        "damage_after": mean_after,
                        "threshold": float(tau),
                        "margin_to_threshold": float(tau - mean_after),
                        "utilization_of_threshold": float(mean_after / tau),
                        "threshold_utilization_percent": float(100.0 * mean_after / tau),
                        "failure_probability_percent": float(100.0 * fail_prob_after),
                        "feasible": passed,

                        "status": (
                            "OK"
                            if passed
                            else "FAILURE_PROBABILITY_ABOVE_EPSILON"
                        ),
                    }
                )

    df = pd.DataFrame(rows)

    if not df.empty:
        df["assignment_feasible"] = (
            df.groupby(["time_step", "vehicle"])["feasible"].transform("all").astype(bool)
        )

        violation_map = (
            df.loc[~df["feasible"]].groupby(["time_step", "vehicle"])["component"].apply(lambda s: [int(x) for x in s.tolist()]).to_dict()
        )

        df["violating_components"] = df.apply(
            lambda row: violation_map.get((row["time_step"], row["vehicle"]),[]), axis=1
        )
    
    return df


def validate_gamma_synthetic_diagnostic(
    input_path: str,
    log_path: str = "results/gamma_synthetic_diagnostic.log",
    tol: float = 1e-6,
) -> dict:
    """
    Run a diagnostic validation for a synthetic Gamma-process input file.

    This function:
    - reads the Gamma input file,
    - propagates the Gamma shape state through the provided schedule x,
    - checks P(D > tau) <= epsilon at every vehicle/component/time entry,
    - checks the Gamma loop condition A_2H <= A_H,
    - writes a human-readable log file,
    - returns a summary report.
    """

    input_file = Path(input_path)
    log_file = Path(log_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    log_file.parent.mkdir(parents=True, exist_ok=True)

    input_data = _read_input(input_file)

    if input_data is None:
        raise ValueError(
            f"Input file is empty or contains only comments: {input_path}"
        )

    params = extract_degradation_parameters(input_data, "gamma")

    maintenance_policy = str(
        input_data.get("maintenance_policy", "none")
    ).lower()

    F = params["F"]
    L = params["L"]
    H = params["H"]
    tau = params["tau"]
    epsilon = params["epsilon"]

    df = build_gamma_diagnostic_dataframe(input_path=input_path, tol=tol)

    failed_df = df[~df["reliability_passed"]].copy()

    # Summary diagnostics
    max_failure_probability_idx = df["failure_probability_after"].idxmax()
    max_failure_probability_row = df.loc[max_failure_probability_idx]

    max_failure_probability = float(max_failure_probability_row["failure_probability_after"])

    max_failure_probability_location = {
        "time_step": int(max_failure_probability_row["time_step"]),
        "input_day": int(max_failure_probability_row["input_day"]),
        "vehicle": int(max_failure_probability_row["vehicle"]),
        "component": int(max_failure_probability_row["component"]),
        "activity": str(max_failure_probability_row["activity"]),
        "mission": (None if pd.isna(max_failure_probability_row["mission"]) 
                    else int(max_failure_probability_row["mission"])),
    }

    max_shape_after = float(df["shape_after"].max())

    # For the loop condition, compare shape at k = H-1 and k = 2H-1.
    # These are the last stored states of the first and second half.
    mid_df = df[df["time_step"] == H - 1]
    end_df = df[df["time_step"] == 2 * H - 1]

    shape_mid = (
        mid_df
        .sort_values(["vehicle", "component"])
        ["shape_after"]
        .to_numpy()
        .reshape(F, L)
    )

    shape_end = (
        end_df
        .sort_values(["vehicle", "component"])
        ["shape_after"]
        .to_numpy()
        .reshape(F, L)
    )

    loop_passed_matrix = loop_constraint_passed(
        shape_mid_horizon=shape_mid,
        shape_end_horizon=shape_end,
        tol=tol,
    )

    loop_passed = bool(np.all(loop_passed_matrix))

    lines = []
    lines.append("=" * 88)
    lines.append("Gamma synthetic diagnostic")
    lines.append("=" * 88)
    lines.append(f"Input file: {input_file}")
    lines.append(f"maintenance_policy={maintenance_policy}")
    lines.append(f"F={F}, L={L}, H={H}, output horizon={2 * H}")
    lines.append(f"tau={tau}")
    lines.append(f"epsilon={epsilon}")
    lines.append("")
    lines.append("-" * 88)
    lines.append("Component-level trajectory checks")
    lines.append("-" * 88)

    for _, row in df.iterrows():
        mission_text = (
            "none"
            if pd.isna(row["mission"])
            else str(int(row["mission"]))
        )

        lines.append(
            f"k={int(row['time_step']):02d}, "
            f"input_day={int(row['input_day']):02d}, "
            f"vehicle={int(row['vehicle'])}, "
            f"component={int(row['component'])}, "
            f"activity={row['activity']}, "
            f"mission={mission_text}: {row['status']}"
        )
        lines.append(
            f"    shape: before={row['shape_before']:.6f}, "
            f"increment={row['shape_increment']:.6f}, "
            f"after={row['shape_after']:.6f}, beta={row['beta']:.6f}"
        )
        lines.append(
            f"    mean: before={row['mean_before']:.6f}, "
            f"increment={row['mean_increment']:.6f}, "
            f"after={row['mean_after']:.6f}, "
            f"variance_after={row['variance_after']:.6f}"
        )
        lines.append(
            f"    failure_probability_after="
            f"{row['failure_probability_after']:.6e}, "
            f"epsilon={row['epsilon']:.6e}"
        )

    lines.append("")
    lines.append("-" * 88)
    lines.append("Loop constraint")
    lines.append("-" * 88)
    lines.append("Gamma shared-rate loop condition: A_2H <= A_H")
    lines.append(f"Loop passed: {loop_passed}")

    for i in range(F):
        for l in range(L):
            status = "OK" if loop_passed_matrix[i, l] else "LOOP_VIOLATION"
            lines.append(
                f"vehicle={i}, component={l}: "
                f"A_H={shape_mid[i,l]:.6f}, "
                f"A_2H={shape_end[i,l]:.6f} -> {status}"
            )

    lines.append("")
    lines.append("-" * 88)
    lines.append("Summary")
    lines.append("-" * 88)
    lines.append(f"Rows checked: {len(df)}")
    lines.append(f"Reliability failures: {len(failed_df)}")
    lines.append(f"Loop passed: {loop_passed}")
    lines.append(f"Max failure probability: {max_failure_probability:.3e}")
    lines.append(f"Max shape after: {max_shape_after:.3f}")
    lines.append(
        "Max failure probability location: "
        f"k={max_failure_probability_location['time_step']}, "
        f"input_day={max_failure_probability_location['input_day']}, "
        f"vehicle={max_failure_probability_location['vehicle']}, "
        f"component={max_failure_probability_location['component']}, "
        f"activity={max_failure_probability_location['activity']}, "
        f"mission={max_failure_probability_location['mission']}"
    )
    lines.append("=" * 88)

    with open(log_file, "w") as f:
        f.write("\n".join(lines))

    report = {
        "passed": bool(len(failed_df) == 0 and loop_passed),
        "rows_checked": int(len(df)),
        "reliability_failures": int(len(failed_df)),
        "loop_passed": loop_passed,
        "tau": float(tau),
        "epsilon": float(epsilon),
        "max_failure_probability": max_failure_probability,
        "max_failure_probability_location": max_failure_probability_location,
        "max_shape_after": max_shape_after,
        "log_path": str(log_file),
    }

    print("=" * 72)
    print("Gamma synthetic diagnostic")
    print("=" * 72)
    print(f"Input file: {input_file}")
    print(f"Log file:   {log_file}")
    print(f"Rows checked: {report['rows_checked']}")
    print(f"Reliability failures: {report['reliability_failures']}")
    print(f"Loop passed: {report['loop_passed']}")
    print(f"Max failure probability: {report['max_failure_probability']:.3e}")
    print(f"Max shape after: {report['max_shape_after']:.3f}")
    print(f"Overall passed: {report['passed']}")
    print("=" * 72)

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