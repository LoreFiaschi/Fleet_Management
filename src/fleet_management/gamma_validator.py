"""Independent validator for constant-rate Gamma solver results.

The validator deliberately does not use the Gurobi model.  It reads the
original problem and the serialized solver result, reconstructs every Gamma
shape state from the saved assignment decisions, and checks the mathematical
contract independently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import gamma

from fleet_management.degradation.gamma import maximum_reliable_shape


SUPPORTED_RESULT_EXTENSIONS = {".yaml", ".yml", ".json", ".h5", ".hdf5"}


def validate_gamma_result(
    input_path: str,
    results_path: str,
    validation_path: str | None = None,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Validate a saved exact-Gamma result without using the optimizer.

    Parameters
    ----------
    input_path:
        Original fleet input file.
    results_path:
        Result file produced by ``solve(..., degradation="gamma")``.
    validation_path:
        Optional YAML or JSON path for the validation report.
    tolerance:
        Numerical tolerance used for equality and inequality checks.
    """

    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")

    input_file = Path(input_path)
    result_file = Path(results_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")

    params = _extract_gamma_contract(_read_validator_input(input_file))
    result = _read_gamma_result(result_file)

    F, H, M, L = (params[name] for name in ("F", "H", "M", "L"))
    beta = np.asarray(params["gamma_beta"], dtype=float)
    tau = np.asarray(params["tau"], dtype=float)
    epsilon = float(params["epsilon"])
    mu_param = np.asarray(params["mu_param"], dtype=float)
    mu_0 = np.asarray(params["mu_0"], dtype=float)
    replacement_mu = np.asarray(params["replacement_mu"], dtype=float)

    checks: list[dict[str, Any]] = []

    def add_check(name: str, violation: float, details: str = "") -> None:
        violation = float(max(0.0, violation))
        item: dict[str, Any] = {
            "name": name,
            "passed": bool(violation <= tolerance),
            "maximum_violation": violation,
            "tolerance": float(tolerance),
        }
        if details:
            item["details"] = details
        checks.append(item)

    status = str(result.get("status", "missing")).lower()
    add_check(
        "solver status is optimal",
        0.0 if status == "optimal" else 1.0,
        f"saved status: {status}",
    )
    if status != "optimal":
        report = _finish_report(
            checks=checks,
            result=result,
            dimensions={"F": F, "H": H, "M": M, "L": L},
            tolerance=tolerance,
            summary={"reason": "No optimal solution arrays can be validated."},
        )
        _save_report_if_requested(report, validation_path)
        return report

    x = _required_array(result, "x", (F, M + 1, 2 * H))
    saved_A = _required_array(result, "A", (F, L, 2 * H))
    saved_mu = _required_array(result, "mu", (F, L, 2 * H))
    saved_tail = _required_array(result, "tail_probability", (F, L, 2 * H))
    saved_u = _required_array(result, "u", (2 * H,))
    saved_z = _required_array(result, "z", (F, 2 * H))

    metadata_violation = 0.0
    for name, expected in (("F", F), ("H", H), ("M", M), ("L", L)):
        metadata_violation = max(
            metadata_violation,
            abs(float(result.get(name, np.inf)) - float(expected)),
        )
    add_check("result dimensions match input", metadata_violation)
    add_check(
        "result degradation is Gamma",
        0.0 if str(result.get("degradation", "gamma")).lower() == "gamma" else 1.0,
    )

    integrality_error = float(np.max(np.abs(x - np.rint(x))))
    bounds_error = max(float(np.max(-x)), float(np.max(x - 1.0)), 0.0)
    add_check("assignment variables are binary", max(integrality_error, bounds_error))
    x_binary = np.rint(x).astype(int)

    assignment_violation = float(
        np.max(np.maximum(np.sum(x_binary, axis=1) - 1, 0))
    )
    add_check("each vehicle receives at most one action per step", assignment_violation)

    demand_violation = float(
        np.max(np.abs(np.sum(x_binary, axis=0) - 1))
    )
    add_check("every mission and replacement slot is assigned", demand_violation)

    reconstructed_A = _reconstruct_shapes(
        x=x_binary,
        F=F,
        H=H,
        M=M,
        L=L,
        beta=beta,
        mu_param=mu_param,
        mu_0=mu_0,
        replacement_mu=replacement_mu,
    )
    state_error = float(np.max(np.abs(saved_A - reconstructed_A)))
    add_check("Gamma state transitions", state_error)

    reconstructed_mu = reconstructed_A / beta[np.newaxis, :, np.newaxis]
    mu_error = float(np.max(np.abs(saved_mu - reconstructed_mu)))
    add_check("saved expected damage", mu_error)

    reconstructed_tail = _tail_probabilities(reconstructed_A, beta, tau)
    tail_error = float(np.max(np.abs(saved_tail - reconstructed_tail)))
    add_check("saved Gamma tail probabilities", tail_error)

    reliability_excess = np.maximum(reconstructed_tail - epsilon, 0.0)
    reliability_violation = float(np.max(reliability_excess))
    reliability_failures = int(np.count_nonzero(reliability_excess > tolerance))
    add_check(
        "reliability constraint",
        reliability_violation,
        f"violating component states: {reliability_failures}",
    )

    repeatability_excess = np.maximum(
        reconstructed_A[:, :, 2 * H - 1] - reconstructed_A[:, :, H - 1],
        0.0,
    )
    repeatability_violation = float(np.max(repeatability_excess))
    repeatability_failures = int(np.count_nonzero(repeatability_excess > tolerance))
    add_check(
        "repeatability constraint",
        repeatability_violation,
        f"violating vehicle-components: {repeatability_failures}",
    )

    capacity = np.sum(reconstructed_mu, axis=(0, 1))
    capacity_violation = float(np.max(np.maximum(capacity - (F - M), 0.0)))
    add_check("aggregate capacity constraint", capacity_violation)

    required_u = np.max(reconstructed_mu, axis=(0, 1))
    u_violation = max(
        float(np.max(np.maximum(required_u - saved_u, 0.0))),
        float(np.max(np.maximum(-saved_u, 0.0))),
    )
    add_check("damage regularisation variable u", u_violation)

    z_violation = _replacement_cost_violation(
        x=x_binary,
        z=saved_z,
        reconstructed_mu=reconstructed_mu,
        mu_0=mu_0,
        replacement_mu=replacement_mu,
    )
    add_check("replacement degradation variable z", z_violation)

    maximum_shape = np.array(
        [
            maximum_reliable_shape(float(beta[l]), float(tau[l]), epsilon)
            for l in range(L)
        ]
    )
    if "maximum_shape" in result:
        saved_maximum = np.asarray(result["maximum_shape"], dtype=float)
        add_check(
            "saved maximum reliable shapes",
            float(np.max(np.abs(saved_maximum - maximum_shape))),
        )

    objective = _recompute_objective(
        x=x_binary,
        A=reconstructed_A,
        u=saved_u,
        z=saved_z,
        beta=beta,
        H=H,
        C_M=float(params["C_M"]),
        C_R=float(params["C_R"]),
        C_S=float(params["C_S"]),
        C_P=float(params["C_P"]),
    )
    saved_objective = float(result["objective"])
    add_check("serialized objective value", abs(saved_objective - objective))

    max_tail_index = np.unravel_index(
        int(np.argmax(reconstructed_tail)), reconstructed_tail.shape
    )
    summary = {
        "transitions_checked": int(F * L * 2 * H),
        "maximum_state_error": state_error,
        "maximum_expected_damage_error": mu_error,
        "maximum_tail_probability_error": tail_error,
        "reliability_failures": reliability_failures,
        "repeatability_failures": repeatability_failures,
        "maximum_failure_probability": float(reconstructed_tail[max_tail_index]),
        "maximum_failure_location": {
            "vehicle": int(max_tail_index[0]),
            "component": int(max_tail_index[1]),
            "step": int(max_tail_index[2]),
        },
        "recomputed_objective": objective,
        "saved_objective": saved_objective,
    }
    report = _finish_report(
        checks=checks,
        result=result,
        dimensions={"F": F, "H": H, "M": M, "L": L},
        tolerance=tolerance,
        summary=summary,
    )
    _save_report_if_requested(report, validation_path)
    return report


def _reconstruct_shapes(
    *,
    x: np.ndarray,
    F: int,
    H: int,
    M: int,
    L: int,
    beta: np.ndarray,
    mu_param: np.ndarray,
    mu_0: np.ndarray,
    replacement_mu: np.ndarray,
) -> np.ndarray:
    reconstructed = np.zeros((F, L, 2 * H), dtype=float)
    initial_shape = mu_0 * beta[np.newaxis, :]
    replacement_shape = replacement_mu * beta[np.newaxis, :]

    for k in range(2 * H):
        for i in range(F):
            previous = initial_shape[i] if k == 0 else reconstructed[i, :, k - 1]
            if x[i, 0, k] == 1:
                reconstructed[i, :, k] = replacement_shape[i]
                continue
            increment = np.zeros(L, dtype=float)
            for j in range(1, M + 1):
                increment += (
                    x[i, j, k]
                    * beta
                    * mu_param[i, j - 1, :, k % H]
                )
            reconstructed[i, :, k] = previous + increment
    return reconstructed


def _tail_probabilities(A: np.ndarray, beta: np.ndarray, tau: np.ndarray) -> np.ndarray:
    tail = np.zeros_like(A, dtype=float)
    for l in range(A.shape[1]):
        positive = A[:, l, :] > 0.0
        tail[:, l, :][positive] = gamma.sf(
            tau[l],
            a=A[:, l, :][positive],
            scale=1.0 / beta[l],
        )
    return tail


def _replacement_cost_violation(
    *,
    x: np.ndarray,
    z: np.ndarray,
    reconstructed_mu: np.ndarray,
    mu_0: np.ndarray,
    replacement_mu: np.ndarray,
) -> float:
    violation = float(np.max(np.maximum(-z, 0.0)))
    F, _, horizon = reconstructed_mu.shape
    for i in range(F):
        for k in range(horizon):
            if x[i, 0, k] == 0:
                violation = max(violation, abs(float(z[i, k])))
                continue
            previous = mu_0[i] if k == 0 else reconstructed_mu[i, :, k - 1]
            required = float(np.sum(previous) - np.sum(replacement_mu[i]))
            violation = max(violation, max(required - float(z[i, k]), 0.0))
    return violation


def _recompute_objective(
    *,
    x: np.ndarray,
    A: np.ndarray,
    u: np.ndarray,
    z: np.ndarray,
    beta: np.ndarray,
    H: int,
    C_M: float,
    C_R: float,
    C_S: float,
    C_P: float,
) -> float:
    objective = C_S * float(np.sum(u))
    objective += C_M * float(np.sum(x[:, 0, :]))
    objective += C_R * float(np.sum(z))
    objective += C_P * float(
        np.sum(
            (A[:, :, H - 1] - A[:, :, 2 * H - 1])
            / beta[np.newaxis, :]
        )
    )
    return objective


def _required_array(result: dict[str, Any], name: str, shape: tuple[int, ...]) -> np.ndarray:
    if name not in result or result[name] is None:
        raise KeyError(f"Gamma result is missing required array '{name}'.")
    array = np.asarray(result[name], dtype=float)
    if array.shape != shape:
        raise ValueError(f"Result '{name}' must have shape {shape}, got {array.shape}.")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"Result '{name}' contains non-finite values.")
    return array


def _read_gamma_result(path: Path) -> dict[str, Any]:
    extension = path.suffix.lower()
    if extension not in SUPPORTED_RESULT_EXTENSIONS:
        raise ValueError(
            f"Unsupported result format '{extension}'. "
            f"Supported formats: {sorted(SUPPORTED_RESULT_EXTENSIONS)}"
        )
    if extension in {".yaml", ".yml"}:
        with path.open("r") as stream:
            return yaml.safe_load(stream)
    if extension == ".json":
        with path.open("r") as stream:
            return json.load(stream)

    import h5py

    output: dict[str, Any] = {}
    with h5py.File(path, "r") as handle:
        for key, value in handle.attrs.items():
            output[key] = _plain_hdf5_value(value)
        for key, value in handle.items():
            output[key] = _plain_hdf5_value(value[()])
    return output


def _read_validator_input(path: Path) -> dict[str, Any]:
    """Read the original input without calling the solver input layer."""

    extension = path.suffix.lower()
    if extension in {".yaml", ".yml"}:
        with path.open("r") as stream:
            return yaml.safe_load(stream)
    if extension == ".json":
        with path.open("r") as stream:
            return json.load(stream)
    if extension not in {".h5", ".hdf5"}:
        raise ValueError(f"Unsupported Gamma input format '{extension}'.")

    import h5py

    output: dict[str, Any] = {}
    with h5py.File(path, "r") as handle:
        for key, value in handle.attrs.items():
            output[key] = _plain_hdf5_value(value)
        for key, value in handle.items():
            output[key] = _plain_hdf5_value(value[()])
    return output


def _extract_gamma_contract(data: dict[str, Any]) -> dict[str, Any]:
    """Parse the constant-beta contract independently of ``solver.py``."""

    required = {
        "F", "H", "M", "mu", "tau", "gamma_beta", "epsilon",
        "C_M", "C_R", "C_S", "C_P", "mu_0",
    }
    missing = required - set(data)
    if missing:
        raise KeyError(f"Missing required Gamma input keys: {sorted(missing)}")

    F = int(data["F"])
    H = int(data["H"])
    M = int(data["M"])
    L = int(data.get("L", 1))

    mu_0 = np.asarray(data["mu_0"], dtype=float)
    if L == 1 and mu_0.shape == (F,):
        mu_0 = mu_0[:, np.newaxis]
    if mu_0.shape != (F, L):
        raise ValueError(f"mu_0 must have shape ({F}, {L}), got {mu_0.shape}.")

    replacement_mu = np.asarray(
        data.get("replacement_mu", np.zeros((F, L))), dtype=float
    )
    if L == 1 and replacement_mu.shape == (F,):
        replacement_mu = replacement_mu[:, np.newaxis]
    if replacement_mu.shape != (F, L):
        raise ValueError(
            f"replacement_mu must have shape ({F}, {L}), "
            f"got {replacement_mu.shape}."
        )

    mu_param = _validator_broadcast_mu(
        np.asarray(data["mu"], dtype=float), F, M, L, H
    )
    beta = _validator_component_array(data["gamma_beta"], L, "gamma_beta")
    tau = _validator_component_array(data["tau"], L, "tau")

    if np.any(beta <= 0.0) or np.any(tau <= 0.0):
        raise ValueError("gamma_beta and tau must be positive.")
    if np.any(mu_param < 0.0) or np.any(mu_0 < 0.0) or np.any(replacement_mu < 0.0):
        raise ValueError("Gamma damage inputs cannot be negative.")

    return {
        "F": F,
        "H": H,
        "M": M,
        "L": L,
        "mu_param": mu_param,
        "tau": tau,
        "epsilon": float(data["epsilon"]),
        "gamma_beta": beta,
        "C_M": float(data["C_M"]),
        "C_R": float(data["C_R"]),
        "C_S": float(data["C_S"]),
        "C_P": float(data["C_P"]),
        "mu_0": mu_0,
        "replacement_mu": replacement_mu,
    }


def _validator_broadcast_mu(
    array: np.ndarray, F: int, M: int, L: int, H: int
) -> np.ndarray:
    target = (F, M, L, H)
    if array.shape == target:
        return array
    if array.shape == (F, M, L):
        return np.repeat(array[:, :, :, np.newaxis], H, axis=3)
    if L == 1 and array.shape == (F, M, H):
        return array[:, :, np.newaxis, :]
    if L == 1 and array.shape == (F, M):
        return np.repeat(array[:, :, np.newaxis, np.newaxis], H, axis=3)
    raise ValueError(f"mu cannot be broadcast from {array.shape} to {target}.")


def _validator_component_array(value: Any, L: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(L, float(array))
    if array.shape == (L,):
        return array
    raise ValueError(f"{name} must be scalar or have shape ({L},).")


def _plain_hdf5_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    array = np.asarray(value)
    if array.ndim == 0:
        scalar = array.item()
        return scalar.decode("utf-8") if isinstance(scalar, bytes) else scalar
    return array.tolist()


def _finish_report(
    *,
    checks: list[dict[str, Any]],
    result: dict[str, Any],
    dimensions: dict[str, int],
    tolerance: float,
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "validator": "independent constant-beta Gamma result validator",
        "passed": bool(all(check["passed"] for check in checks)),
        "solver_status": str(result.get("status", "missing")),
        "solver_objective": result.get("objective"),
        "tolerance": float(tolerance),
        "dimensions": dimensions,
        "checks": checks,
        "summary": summary,
    }


def _save_report_if_requested(report: dict[str, Any], path: str | None) -> None:
    if path is None:
        return
    output = Path(path)
    if output.parent != Path(".") and not output.parent.exists():
        raise FileNotFoundError(f"Validation directory does not exist: {output.parent}")
    if output.suffix.lower() in {".yaml", ".yml"}:
        with output.open("w") as stream:
            yaml.safe_dump(report, stream, sort_keys=False)
    elif output.suffix.lower() == ".json":
        with output.open("w") as stream:
            json.dump(report, stream, indent=2)
    else:
        raise ValueError("Validation report must use .yaml, .yml or .json.")