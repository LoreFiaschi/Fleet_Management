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

