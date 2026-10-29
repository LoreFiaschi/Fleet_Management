# Validation for fleet_management
# author: Christoph Langenauer
# date of creation: 10.06.2026

"""Structural validator of a saved result (spec, section "Structural Validator").

``validate`` reads the two files, checks, writes the report and returns it.
``check_result`` runs the checks on a ``FleetConfig`` and a result dictionary without
files; the dashboards call it. ``read_result``, imported from ``solver``, reads a result
file of ``solve()``.
"""

import json
import os
from pathlib import Path

import numpy as np
import yaml

from fleet_management.config import INPUT_EXTENSIONS, FleetConfig, load_config, read_input
from fleet_management.solver import RESULT_EXTENSIONS, read_result

REPORT_EXTENSIONS = (".yaml", ".yml", ".json")
LOOP_KEYS = ("mu", "v", "R", "K", "g_mu", "g_v", "g_R", "g_K")
SIZE_KEYS = ("F", "M", "L", "H1", "H2")


def check_sizes(cfg: FleetConfig, result: dict) -> None:
    """Raise a ValueError if the result does not fit the input (spec, Validator, Errors)."""
    if result.get("x") is None:
        raise ValueError("the result has no solution: 'x' is missing or null.")

    for key in SIZE_KEYS:
        saved = result.get(key)
        if saved is None or int(saved) != getattr(cfg, key):
            raise ValueError(
                f"'{key}' differs between the result ({saved}) and the input "
                f"({getattr(cfg, key)})."
            )

    F, M, L, T = cfg.F, cfg.M, cfg.L, cfg.T
    shapes = {"x": (F, M + 1, T), "step_costs": (T,)}
    for key in ("m", "r", "idle", "mu", "v", "z", "R", "K", "g_mu", "g_v", "g_R", "g_K"):
        shapes[key] = (F, L, T)
    for key in ("tau", "mu_0", "v_0"):
        shapes[key] = (F, L)
    for key, shape in shapes.items():
        value = result.get(key)
        if value is not None and np.shape(value) != shape:
            raise ValueError(
                f"result array '{key}' has shape {np.shape(value)}; the input gives {shape}."
            )

    for key in ("m", "r", "mu", "z", "u", "objective"):
        if result.get(key) is None:
            raise ValueError(f"the result has 'x' but no '{key}'.")

    for key in ("x", "m", "r", "idle"):
        value = result.get(key)
        if value is not None and not np.all(np.isfinite(value)):
            raise ValueError(f"result array '{key}' has a non-finite entry.")


def recomputed_objective(cfg: FleetConfig, result: dict) -> float:
    """Recompute the objective of the saved schedule (spec, Eqs. a_daycost to a_objective)."""
    x, r = (np.asarray(result[key], dtype=float) for key in ("x", "r"))
    z = np.nan_to_num(np.asarray(result["z"], dtype=float), nan=0.0)

    day = cfg.C_M * x[:, 0, :].sum(axis=0) + np.einsum("il,ilk->k", cfg.C_R, z)
    if cfg.allow_replacement:
        day = day + np.einsum("il,ilk->k", cfg.C_rep, r)
    J_init, J_op = float(day[:cfg.H1].sum()), float(day[cfg.H1:].sum())

    penalty = cfg.C_D * float(result["u"]) if cfg.C_D > 0 else 0.0
    mode = cfg.options["objective_mode"]
    if mode == "total":
        return J_init + J_op + penalty
    if mode == "operating_average":
        return J_op / cfg.H2 + penalty
    E = cfg.options["evaluation_horizon"]
    return J_init + (E - cfg.H1) / cfg.H2 * J_op + penalty


def _loop_violation(value, H1: int, T: int) -> float:
    """Violation of the loop row of one quantity; NaN and null are skipped (spec, check loop_q)."""
    if value is None:
        return 0.0
    q = np.asarray(value, dtype=float)
    excess = q[:, :, T - 1] - q[:, :, H1 - 1]
    excess = excess[np.isfinite(excess)]
    if excess.size == 0:
        return 0.0
    return max(0.0, float(excess.max()))


def check_result(cfg: FleetConfig, result: dict, tol: float = 1e-6) -> dict:
    """Check a result against its input without files; returns the report (spec, Validator)."""
    check_sizes(cfg, result)
    x, m, r = (np.asarray(result[key], dtype=float) for key in ("x", "m", "r"))
    x0 = x[:, 0, :][:, None, :]

    checks = []

    def add(name, violation):
        checks.append(
            {
                "name": name,
                "passed": bool(violation <= tol),
                "violation": float(violation),
                "tolerance": float(tol),
            }
        )

    add("binary", max(float(np.max(np.minimum(np.abs(a), np.abs(a - 1.0)))) for a in (x, m, r)))
    add("one_activity", max(0.0, float(np.max(x.sum(axis=1) - 1.0))))
    add("mission_coverage", float(np.max(np.abs(x[:, 1:, :].sum(axis=0) - 1.0))))

    depot_gating = max(0.0, float(np.max(m + r - x0)))
    if not cfg.allow_replacement:
        depot_gating = max(depot_gating, float(np.max(r)))
    add("depot_gating", depot_gating)

    if result.get("idle") is not None:
        idle = np.asarray(result["idle"], dtype=float)
        add("idle", float(np.max(np.abs(idle - (x0 - m - r)))))

    for key in LOOP_KEYS:
        add(f"loop_{key}", _loop_violation(result.get(key), cfg.H1, cfg.T))

    saved = float(result["objective"])
    recomputed = recomputed_objective(cfg, result)
    add("objective", max(0.0, recomputed - saved) / max(1.0, abs(saved)))

    return {
        "status": result.get("status"),
        "solver_objective": saved,
        "recomputed_objective": recomputed,
        "tolerance": float(tol),
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
        "max_violation": max(check["violation"] for check in checks),
    }


def validate(
    input_path: str,
    results_path: str,
    validation_path: str = "validation.yaml",
    tol: float = 1e-6,
) -> dict:
    """
    Check a saved result against its input and write the report (spec, "Structural Validator").

    The optimiser is not called. The status of the result is echoed, not checked.

    Parameters
    ----------
    input_path : str
        Path to the input file of the solve.

    results_path : str
        Path to the result file written by ``solve()``.

    validation_path : str
        Path to save the validation report (.yaml, .yml or .json). A path without
        extension gets ".yaml"; the folder is created.

    tol : float
        Tolerance of the checks: a check passes if its violation is at most ``tol``.

    Returns
    -------
    dict
        Validation report.
    """

    # --- File checks ---
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    results_file = Path(results_path)
    if not results_file.exists():
        raise FileNotFoundError(f"Result file not found: {results_path}")

    if input_file.suffix.lower() not in INPUT_EXTENSIONS:
        raise ValueError(
            f"Unsupported input file type '{input_file.suffix}'. "
            f"Supported types: {list(INPUT_EXTENSIONS)}"
        )
    if results_file.suffix.lower() not in RESULT_EXTENSIONS:
        raise ValueError(
            f"Unsupported result file type '{results_file.suffix}'. "
            f"Supported types: {list(RESULT_EXTENSIONS)}"
        )

    validation_file = _resolve_validation_path(validation_path)
    if validation_file.suffix.lower() not in REPORT_EXTENSIONS:
        raise ValueError(
            f"Unsupported validation file type '{validation_file.suffix}'. "
            f"Supported types: {list(REPORT_EXTENSIONS)}"
        )

    validation_dir = validation_file.parent
    validation_dir.mkdir(parents=True, exist_ok=True)

    if not os.access(validation_dir, os.W_OK):
        raise PermissionError(f"Validation directory is not writable: {validation_dir}")

    # --- Read, check and write ---
    cfg = load_config(read_input(input_file))
    report = check_result(cfg, read_result(results_file), tol)
    _save_validation_report(report, validation_file)

    return report


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
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(report, f, sort_keys=False)

    elif path.suffix.lower() == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, allow_nan=False)

    else:
        raise ValueError(f"Unsupported validation file type: {path.suffix}")
