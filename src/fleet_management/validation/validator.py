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
from fleet_management.validation.tolerance import (
    binary_tolerance,
    cost_tolerance,
    report_tolerance,
    resolve_tolerance,
    state_tolerance,
)

REPORT_EXTENSIONS = (".yaml", ".yml", ".json")
LOOP_KEYS = ("mu", "v", "R", "K", "g_mu", "g_v", "g_R", "g_K")
SIZE_KEYS = ("F", "M", "L", "H1", "H2")


def check_sizes(cfg: FleetConfig, result: dict) -> None:
    """Raise a ValueError if the result does not fit the input.

    Spec, section "Structural Validator", table "Errors".
    """
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


def start_values(result: dict, key: str):
    """Saved start of the repetitions ``y_<key>`` as an ``F x L`` float array, or None.

    None if the result has no such key or the key holds no number (a result written
    before the start variables existed).
    """
    value = result.get(f"y_{key}")
    if value is None:
        return None
    y = np.asarray(value, dtype=float)
    return y if np.isfinite(y).any() else None


def _loop_cell_violations(value, H1: int, T: int, y=None) -> np.ndarray | None:
    """Violation of the loop row of one quantity at every cell, an ``F x L`` array.

    NaN and null are skipped (spec, check loop_q): such a cell has violation 0, and a null
    quantity gives ``None``. With the start ``y``: ``max{0, q_T - y, q_H1 - y}``. Without it:
    ``max{0, q_T - q_H1}``.
    """
    if value is None:
        return None
    q = np.asarray(value, dtype=float)
    if y is None:
        excess = q[:, :, T - 1] - q[:, :, H1 - 1]
    else:
        excess = np.fmax(q[:, :, T - 1] - y, q[:, :, H1 - 1] - y)
    return np.fmax(excess, 0.0)             # fmax skips NaN


def _loop_violation(value, H1: int, T: int, y=None) -> float:
    """Largest violation of the loop row of one quantity over the cells (spec, check loop_q)."""
    cells = _loop_cell_violations(value, H1, T, y)
    return 0.0 if cells is None else float(cells.max())


def check_result(cfg: FleetConfig, result: dict, tol: float | None = None) -> dict:
    """Check a result against its input without files and return the report.

    Spec, section "Structural Validator": the checks, their order and the report keys.
    The tolerance of a check is ``tol`` if it is a number, else the key ``tolerance`` of the
    input block ``validation``, else a default per check from the Gurobi tolerances
    (``validation/tolerance.py``). A check over several cells passes if no cell exceeds
    its own tolerance; it reports the largest violation and the tolerance at that cell.
    """
    check_sizes(cfg, result)
    single = resolve_tolerance(cfg, tol)
    x, m, r = (np.asarray(result[key], dtype=float) for key in ("x", "m", "r"))
    x0 = x[:, 0, :][:, None, :]

    checks = []

    def add(name, violation, tolerance=None):
        """One check; ``violation`` and ``tolerance`` are numbers or arrays of the same shape."""
        violation = np.asarray(violation, dtype=float)
        tolerance = np.broadcast_to(np.asarray(tolerance, dtype=float), violation.shape)
        worst = int(np.argmax(violation))
        checks.append(
            {
                "name": name,
                "passed": bool(np.all(violation <= tolerance)),
                "violation": float(violation.ravel()[worst]),
                "tolerance": float(tolerance.ravel()[worst]),
            }
        )

    tol_binary = binary_tolerance(cfg, single)
    add("binary", max(float(np.max(np.minimum(np.abs(a), np.abs(a - 1.0)))) for a in (x, m, r)),
        tol_binary)
    add("one_activity", max(0.0, float(np.max(x.sum(axis=1) - 1.0))), tol_binary)
    add("mission_coverage", float(np.max(np.abs(x[:, 1:, :].sum(axis=0) - 1.0))), tol_binary)

    depot_gating = max(0.0, float(np.max(m + r - x0)))
    if not cfg.allow_replacement:
        depot_gating = max(depot_gating, float(np.max(r)))
    add("depot_gating", depot_gating, tol_binary)

    if result.get("idle") is not None:
        idle = np.asarray(result["idle"], dtype=float)
        add("idle", float(np.max(np.abs(idle - (x0 - m - r)))), tol_binary)

    for key in LOOP_KEYS:
        cells = _loop_cell_violations(result.get(key), cfg.H1, cfg.T, start_values(result, key))
        if cells is None:                             # a null quantity: no violation
            cells = np.zeros((cfg.F, cfg.L))
        add(f"loop_{key}", cells, state_tolerance(cfg, key, single))

    saved = float(result["objective"])
    recomputed = recomputed_objective(cfg, result)
    add("objective", max(0.0, recomputed - saved) / max(1.0, abs(saved)),
        cost_tolerance(cfg, single))

    return {
        "status": result.get("status"),
        "solver_objective": saved,
        "recomputed_objective": recomputed,
        "tolerance": report_tolerance(cfg, single),
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
        "max_violation": max(check["violation"] for check in checks),
    }


def validate(
    input_path: str,
    results_path: str,
    validation_path: str = "validation.yaml",
    tol: float | None = None,
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

    tol : float, optional
        One tolerance for every check: a check passes if its violation is at most ``tol``.
        If ``None``, the key ``tolerance`` of the input block ``validation`` is used, else
        a default per check from the Gurobi tolerances of the input (``tolerance.py``).

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
