import json
import os
from pathlib import Path

import h5py
import numpy as np
import yaml

from fleet_management.gaussian import solve_fleet_management

SUPPORTED_DEGRADATIONS = {"gaussian"}
SUPPORTED_EXTENSIONS = {".yaml", ".yml", ".json", ".h5", ".hdf5"}


def solve(input_path: str, degradation: str, results_path: str = None) -> None:
    """
    Mid-layer between the user and the fleet-management solvers.

    Parameters
    ----------
    input_path : str
        Path to an input file containing the problem data.
        Supported formats: YAML (.yaml/.yml), JSON (.json), HDF5 (.h5/.hdf5).
    degradation : str
        Type of degradation model. Currently supported: "gaussian".
    results_path : str, optional
        Path where results will be saved. Defaults to "output.yaml".
        If provided without an extension, ".yaml" is appended.
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
    if input_file.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported input file type '{input_file.suffix}'. "
            f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    results_path = _resolve_results_path(results_path)
    results_dir = results_path.parent
    if results_dir != Path("") and not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    if results_dir != Path("") and not os.access(results_dir, os.W_OK):
        raise PermissionError(f"Results directory is not writable: {results_dir}")

    # --- Read and parse input ---
    data = _read_input(input_file)
    params = _extract_parameters(data)

    # --- Solve ---
    if degradation_lower == "gaussian":
        result = solve_fleet_management(**params)

    # --- Save results ---
    _save_results(result, results_path)


def _read_input(input_file: Path) -> dict:
    """Read input data from a supported file format."""
    ext = input_file.suffix.lower()
    if ext in (".yaml", ".yml"):
        with open(input_file, "r") as f:
            return yaml.safe_load(f)
    elif ext == ".json":
        with open(input_file, "r") as f:
            return json.load(f)
    elif ext in (".h5", ".hdf5"):
        return _read_hdf5(input_file)
    else:
        raise ValueError(f"Unsupported input file type: {ext}")


def _read_hdf5(path: Path) -> dict:
    """Read all solver parameters from an HDF5 file.

    Expected structure:
    - Scalar parameters (F, H, M, epsilon, C_M, C_R, C_S, C_P, verbose)
      stored as attributes on the root group or as scalar datasets.
    - Array parameters (mu, v, mu_0, v_0) stored as datasets.
    """
    data = {}
    scalar_keys = {"F", "H", "M", "epsilon", "C_M", "C_R", "C_S", "C_P", "verbose"}
    array_keys = {"mu", "v", "mu_0", "v_0"}

    with h5py.File(path, "r") as f:
        for key in scalar_keys:
            if key in f.attrs:
                data[key] = float(f.attrs[key])
            elif key in f:
                data[key] = float(f[key][()])
        for key in array_keys:
            if key in f:
                data[key] = f[key][()].tolist()

    return data


def _resolve_results_path(results_path) -> Path:
    """Resolve the results path, applying defaults for missing name or extension."""
    if results_path is None:
        return Path("output.yaml")
    p = Path(results_path)
    if p.suffix == "":
        p = p.with_suffix(".yaml")
    return p


REQUIRED_KEYS = {"F", "H", "M", "mu", "v", "epsilon", "C_M", "C_R", "C_S", "C_P", "mu_0", "v_0"}


def _extract_parameters(data: dict) -> dict:
    """Extract and validate all solver parameters from the parsed input data."""
    missing = REQUIRED_KEYS - set(data.keys())
    if missing:
        raise KeyError(f"Missing required keys in input file: {sorted(missing)}")

    F = int(data["F"])
    H = int(data["H"])
    M = int(data["M"])
    epsilon = float(data["epsilon"])
    C_M = float(data["C_M"])
    C_R = float(data["C_R"])
    C_S = float(data["C_S"])
    C_P = float(data["C_P"])

    mu_param = np.array(data["mu"], dtype=float)
    v_param = np.array(data["v"], dtype=float)
    mu_0 = np.array(data["mu_0"], dtype=float)
    v_0 = np.array(data["v_0"], dtype=float)

    verbose = int(data.get("verbose", 1))

    if mu_param.shape != (F, M, H):
        raise ValueError(
            f"'mu' shape {mu_param.shape} does not match (F={F}, M={M}, H={H})."
        )
    if v_param.shape != (F, M, H):
        raise ValueError(
            f"'v' shape {v_param.shape} does not match (F={F}, M={M}, H={H})."
        )

    return {
        "F": F, "H": H, "M": M,
        "mu_param": mu_param, "v_param": v_param,
        "epsilon": epsilon,
        "C_M": C_M, "C_R": C_R, "C_S": C_S, "C_P": C_P,
        "mu_0": mu_0, "v_0": v_0,
        "verbose": verbose,
    }


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
        "objective": float(result["objective"]) if result["objective"] is not None else None,
        "F": result["F"],
        "M": result["M"],
        "H": result["H"],
    }
    if result["x"] is not None:
        output["x"] = result["x"].tolist()
        output["mu"] = result["mu"].tolist()
        output["v"] = result["v"].tolist()
        output["z"] = result["z"].tolist()
    return output


def _save_yaml(result: dict, path: Path) -> None:
    output = _build_serializable_output(result)
    with open(path, "w") as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)


def _save_json(result: dict, path: Path) -> None:
    output = _build_serializable_output(result)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)


def _save_hdf5(result: dict, path: Path) -> None:
    with h5py.File(path, "w") as f:
        f.attrs["status"] = result["status"] if isinstance(result["status"], str) else str(result["status"])
        if result["objective"] is not None:
            f.attrs["objective"] = float(result["objective"])
        f.attrs["F"] = result["F"]
        f.attrs["M"] = result["M"]
        f.attrs["H"] = result["H"]
        if result["x"] is not None:
            f.create_dataset("x", data=result["x"])
            f.create_dataset("mu", data=result["mu"])
            f.create_dataset("v", data=result["v"])
            f.create_dataset("z", data=result["z"])
