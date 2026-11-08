"""The solve path and the result files (spec, "Data Flow of One Solve" and "Result Files").

``solve`` checks the paths, reads the input file and calls ``solve_mapping``.
``solve_mapping`` normalises the mapping with ``config.load_config``, solves it with
``degradation_model.base.solve_mixed`` and writes the result file. ``base`` is imported
inside ``solve_mapping``, so the writers import without Gurobi.
"""

import json
import math
import os
from pathlib import Path

import h5py
import numpy as np
import yaml

from fleet_management.config import INPUT_EXTENSIONS, load_config, read_input

_read_input = read_input  # Old name, used by the validation tools.

RESULT_EXTENSIONS = (".yaml", ".yml", ".json", ".h5", ".hdf5")
WARM_START_MODES = ("start", "hint", "both")

# Keys of the result dictionary, in the order of the spec's table.
RESULT_KEYS = (
    "status", "objective", "mip_gap", "bound",
    "F", "M", "L", "H1", "H2", "T", "H", "model", "component_names",
    "tau", "mu_0", "v_0",
    "bound_method", "repair_model", "reliability_impl", "encoding",
    "objective_mode", "evaluation_horizon",
    "x", "m", "r", "idle", "mu", "v", "z", "R", "K",
    "g_mu", "g_v", "g_R", "g_K", "u",
    "step_costs", "J_initialization", "J_op", "J_op_average",
    "peak_damage", "damage_penalty", "operating_objective", "J_total",
    "projected_evaluation_cost", "evaluation_objective", "component_costs",
    "performance", "optimization_progress", "warm_start", "relaxation_warm_start",
)

# Placement in an HDF5 result file (spec, "HDF5 Layout"). Every other key is an attribute.
_HDF5_DATASET_KEYS = (
    "x", "mu", "v", "R", "K", "g_mu", "g_v", "g_R", "g_K",
    "z", "m", "r", "idle", "step_costs", "tau", "mu_0", "v_0",
)
_HDF5_JSON_KEYS = (
    "model", "component_names", "component_costs", "performance",
    "optimization_progress", "warm_start", "relaxation_warm_start",
)
_SELECTOR_KEYS = ("bound_method", "repair_model", "reliability_impl", "encoding")


# ---------------------------------------------------------------------------
# Solve path
# ---------------------------------------------------------------------------
def solve(
    input_path: str,
    results_path: str | None = "output.yaml",
    *,
    warm_start: dict | None = None,
    warm_start_mode: str = "start",
) -> dict:
    """
    Read, normalize, solve and serialize a self-describing fleet input.

    The input file is self-describing: it must carry a top-level ``model:`` key
    (see ``config.load_config``). Uniform and genuinely mixed fleets share this
    entry point.

    Parameters
    ----------
    input_path : str
        Path to an input file containing the problem data.
        Supported formats: YAML (.yaml/.yml), JSON (.json), HDF5 (.h5/.hdf5).
    results_path : str or None, optional
        Path where results will be saved. Defaults to "output.yaml".
        If provided without an extension, ".yaml" is appended.
        ``None`` writes no file.
    warm_start : dict, optional
        Result of an earlier solve, used as a warm start.
    warm_start_mode : {"start", "hint", "both"}, optional
        How the warm start is applied.
    """
    # --- File checks ---
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_file.suffix.lower() not in INPUT_EXTENSIONS:
        raise ValueError(
            f"Unsupported input file type '{input_file.suffix}'. "
            f"Supported types: {list(INPUT_EXTENSIONS)}"
        )
    path = _check_results_path(results_path)

    data = read_input(input_file)
    return solve_mapping(
        data, path, warm_start=warm_start, warm_start_mode=warm_start_mode
    )


def solve_mapping(
    data: dict,
    results_path: str | None = None,
    *,
    warm_start: dict | None = None,
    warm_start_mode: str = "start",
) -> dict:
    """Normalize a mapping, solve it and write the result file if a path is given.

    This is steps 3 to 5 of ``solve``. The result is returned as ``base.solve_mixed``
    gives it. Gurobi errors are not caught.
    """
    if warm_start_mode not in WARM_START_MODES:
        raise ValueError(
            f"warm_start_mode must be one of {WARM_START_MODES} "
            f"(got {warm_start_mode!r})."
        )
    path = _check_results_path(results_path)

    cfg = load_config(data)

    from fleet_management.degradation_model import base  # needs Gurobi

    result = base.solve_mixed(
        cfg, warm_start=warm_start, warm_start_mode=warm_start_mode
    )
    if path is not None:
        _save_results(result, path)
    return result


def _result_format(path: Path) -> str:
    """Return the format of a result file, chosen by its extension."""
    ext = path.suffix.lower()
    if ext in (".yaml", ".yml"):
        return "yaml"
    if ext == ".json":
        return "json"
    if ext in (".h5", ".hdf5"):
        return "hdf5"
    raise ValueError(
        f"Unsupported results file type '{path.suffix}'. "
        f"Supported types: {list(RESULT_EXTENSIONS)}"
    )


def _check_results_path(results_path) -> Path | None:
    """Check the results path and apply the default extension; ``None`` means no file."""
    if results_path is None:
        return None
    path = Path(results_path)
    if path.suffix == "":
        path = path.with_suffix(".yaml")
    _result_format(path)

    results_dir = path.parent
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    if not os.access(results_dir, os.W_OK):
        raise PermissionError(f"Results directory is not writable: {results_dir}")
    return path


def _save_results(result: dict, path: Path) -> None:
    """Save solver results to a file (YAML, JSON, or HDF5)."""
    ext = path.suffix.lower()

    if ext in (".yaml", ".yml"):
        _save_yaml(result, path)
    elif ext == ".json":
        _save_json(result, path)
    elif ext in (".h5", ".hdf5"):
        _save_hdf5(result, path)
    else:
        _save_yaml(result, path)


def _build_serializable_output(result: dict) -> dict:
    """Build a plain dict from solver results for text-based formats."""
    output = {
        "status": result["status"],
        "objective": (
            float(result["objective"])
            if result["objective"] is not None
            else None
        ),
        "degradation": result["degradation"],
        "backend": result.get("backend"),
        "models": _to_builtin(result.get("models", [])),
        "model_assignment": _to_builtin(result.get("model_assignment")),
        "component_names": _to_builtin(result.get("component_names", [])),
        "F": result["F"],
        "M": result["M"],
        "H": result["H"],
        "L": result["L"],
        "mu_0": result["mu_0"].tolist(),
    }

    # Optional scalar parameters
    if result.get("alpha") is not None:
        output["alpha"] = result["alpha"]
    for key in ("bound", "mip_gap"):
        if result.get(key) is not None:
            output[key] = float(result[key])
    for key in (
        "J_initialization",
        "J_op",
        "J_op_average",
        "peak_damage",
        "damage_penalty",
        "operating_objective",
        "evaluation_objective",
        "J_total",
        "projected_evaluation_cost",
    ):
        if result.get(key) is not None:
            output[key] = float(result[key])

    # Two-horizon / rainflow metadata
    for key in (
        "H1",
        "H2",
        "T",
        "method",
        "bound_method",
        "repair_model",
        "reliability_impl",
        "gamma_dynamics_formulation",
        "gamma_big_m_bound_strategy",
        "gamma_calibration_method",
        "objective_mode",
        "evaluation_horizon",
    ):
        if result.get(key) is not None:
            output[key] = _to_builtin(result[key])

    # Optional method-specific arrays
    for key in (
        "tau",
        "gamma_beta",
        "gamma_beta_bound",
        "replacement_mu",
        "repair_rho",
        "maximum_shape",
    ):
        if result.get(key) is not None:
            output[key] = _to_builtin(result[key])

    if result.get("v_0") is not None:
        output["v_0"] = result["v_0"].tolist()

    if result.get("x") is not None:
        output["x"] = result["x"].tolist()
        output["mu"] = result["mu"].tolist()
        output["u"] = (
            result["u"].tolist()
            if hasattr(result["u"], "tolist")
            else float(result["u"])
        )
        output["z"] = result["z"].tolist()

        # Optional solution arrays
        for key in (
            "v",
            "A",
            "tail_probability",
            "gamma_shape_bound",
            "gamma_tail_bound",
            "gamma_mean_latch",
            "gamma_shape_latch",
            "gamma_shape_increment",
            "gamma_shape_increment_trans",
            "gamma_maximum_shape",
            "m",
            "r",
            "idle",
            "step_costs",
        ):
            if result.get(key) is not None:
                output[key] = _to_builtin(result[key])

        if result.get("gamma_calibration") is not None:
            output["gamma_calibration"] = _to_builtin(result["gamma_calibration"])

    # Performance measurements may exist even without a solution.
    if result.get("gamma_formulation") is not None:
        output["gamma_formulation"] = _to_builtin(result["gamma_formulation"])
    if result.get("performance") is not None:
        output["performance"] = _to_builtin(result["performance"])
    for key in (
        "optimization_progress",
        "warm_start",
        "relaxation_warm_start",
        "component_costs",
    ):
        if result.get(key) is not None:
            output[key] = _to_builtin(result[key])

    return output

def _to_builtin(value):                                                 # performance measurement
    """Convert NumPy values and nested containers to serializable Python types."""

    if isinstance(value, dict):
        return {
            str(key): _to_builtin(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]

    if isinstance(value, np.ndarray):
        return [_to_builtin(item) for item in value.tolist()]

    if isinstance(value, np.generic):
        return value.item()

    return value

def _save_yaml(result: dict, path: Path) -> None:
    output = _build_serializable_output(result)
    with open(path, "w") as f:
        # safe_dump emits only plain YAML (no !!python/object tags), so the file
        # round-trips through yaml.safe_load; it also fails loudly if any NumPy
        # object slipped through _build_serializable_output.
        yaml.safe_dump(output, f, default_flow_style=False, sort_keys=False)


def _save_json(result: dict, path: Path) -> None:
    output = _build_serializable_output(result)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)


def _save_hdf5(result: dict, path: Path) -> None:
    with h5py.File(path, "w") as f:
        f.attrs["status"] = (
            result["status"]
            if isinstance(result["status"], str)
            else str(result["status"])
        )

        if result.get("objective") is not None:
            f.attrs["objective"] = float(result["objective"])

        f.attrs["degradation"] = result["degradation"]
        f.attrs["F"] = result["F"]
        f.attrs["M"] = result["M"]
        f.attrs["H"] = result["H"]
        f.attrs["L"] = result["L"]

        # Optional scalar parameter
        if result.get("alpha") is not None:
            f.attrs["alpha"] = result["alpha"]
        for key in ("bound", "mip_gap"):
            if result.get(key) is not None:
                f.attrs[key] = float(result[key])

        # Two-horizon / rainflow metadata
        for key in (
            "H1", "H2", "T", "method", "repair_model",
            "gamma_calibration_method", "objective_mode",
        ):
            if result.get(key) is not None:
                f.attrs[key] = result[key]
        for key in ("component_names", "model_assignment"):
            if result.get(key) is not None:
                f.attrs[key] = json.dumps(_to_builtin(result[key]))
        for key in (
            "J_initialization",
            "J_op",
            "J_op_average",
            "peak_damage",
            "damage_penalty",
            "operating_objective",
            "evaluation_objective",
            "J_total",
            "projected_evaluation_cost",
        ):
            if result.get(key) is not None:
                f.attrs[key] = float(result[key])

        # Method-specific arrays
        for key in (
            "tau",
            "gamma_beta",
            "gamma_beta_bound",
            "replacement_mu",
            "repair_rho",
            "maximum_shape",
        ):
            if result.get(key) is not None:
                f.create_dataset(key, data=result[key])

        f.create_dataset("mu_0", data=result["mu_0"])

        if result.get("v_0") is not None:
            f.create_dataset("v_0", data=result["v_0"])

        if result.get("x") is not None:
            f.create_dataset("x", data=result["x"])
            f.create_dataset("mu", data=result["mu"])
            f.create_dataset("u", data=result["u"])
            f.create_dataset("z", data=result["z"])

            # Optional solution arrays
            for key in (
                "v",
                "A",
                "tail_probability",
                "gamma_shape_bound",
                "gamma_tail_bound",
                "m",
                "r",
                "idle",
                "step_costs",
            ):
                if result.get(key) is not None:
                    f.create_dataset(key, data=result[key])

            if result.get("gamma_calibration") is not None:
                f.attrs["gamma_calibration"] = json.dumps(
                    _to_builtin(result["gamma_calibration"])
                )

        for key in ("gamma_formulation", "performance"):
            if result.get(key) is not None:
                f.attrs[key] = json.dumps(_to_builtin(result[key]))
