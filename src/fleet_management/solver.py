import json
import os
from pathlib import Path

import h5py
import numpy as np
import yaml

from fleet_management.gaussian import solve_fleet_management as solve_gaussian
from fleet_management.inverse_gaussian import (
    solve_fleet_management as solve_inverse_gaussian,
)
from fleet_management.gamma_gurobi import (
    solve_fleet_management as solve_gamma,
)
"""from fleet_management.model_registry import (
    SUPPORTED_DEGRADATIONS,
    REQUIRED_KEYS_BY_DEGRADATION,
    extract_degradation_parameters,
    broadcast_4d_param,
)"""
from fleet_management.rainflow import solve_fleet_management as solve_rainflow

SUPPORTED_DEGRADATIONS = {"gaussian", "inverse_gaussian", "rainflow", "gamma"}
SUPPORTED_EXTENSIONS = {".yaml", ".yml", ".json", ".h5", ".hdf5"}

_COMMON_KEYS = {"F", "H", "M", "mu", "alpha", "epsilon", "xi", "C_M", "C_R", "C_S", "C_P", "mu_0"}
_GAUSSIAN_KEYS = _COMMON_KEYS | {"v", "v_0"}
_INVERSE_GAUSSIAN_KEYS = _COMMON_KEYS | {"c"}
_RAINFLOW_KEYS = (_COMMON_KEYS - {"alpha"}) | {"v", "v_0"}
_GAMMA_KEYS = {
    "F", "H", "M", "mu", "tau", "gamma_beta", "epsilon",
    "C_M", "C_R", "C_rep", "C_S", "C_P", "mu_0", "repair_rho",
}

REQUIRED_KEYS_BY_DEGRADATION = {
    "gaussian": _GAUSSIAN_KEYS,
    "inverse_gaussian": _INVERSE_GAUSSIAN_KEYS,
    "rainflow": _RAINFLOW_KEYS,
    "gamma": _GAMMA_KEYS,
}

def solve(input_path: str, degradation: str, results_path: str = None) -> dict:         # was -> None, now -> dict for performance measurement
    """
    Mid-layer between the user and the fleet-management solvers.

    Parameters
    ----------
    input_path : str
        Path to an input file containing the problem data.
        Supported formats: YAML (.yaml/.yml), JSON (.json), HDF5 (.h5/.hdf5).
    degradation : str
        Type of degradation model. Supported values are "gamma", "gaussian"
        and "inverse_gaussian".
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
    params = _extract_parameters(data, degradation_lower)

    # --- Solve ---
    if degradation_lower == "gaussian":
        result = solve_gaussian(**params)
    elif degradation_lower == "inverse_gaussian":
        result = solve_inverse_gaussian(**params)
    elif degradation_lower == "rainflow":
        result = solve_rainflow(**params)
    else:  # gamma
        result = solve_gamma(**params)

    performance = result.setdefault("performance", {})          # performance measurement
    result["degradation"] = degradation_lower
    result["mu_0"] = params["mu_0"]
    if "v_0" in params:
        result["v_0"] = params["v_0"]

    # --- Save results ---
    _save_results(result, results_path)

    return result                                               # performance measurement


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
    - Scalar parameters (F, H, M, L, epsilon, C_M, C_R, C_S, C_P, verbose)
      stored as attributes on the root group or as scalar datasets.
    - Array parameters (mu, v, mu_0, v_0, c, xi) stored as datasets.
    """
    data = {}
    scalar_keys = {
    "F",
    "H",
    "M",
    "L",
    "alpha",
    "epsilon",
    "C_M",
    "C_R",
    "C_rep",
    "C_S",
    "C_P",
    "verbose",
    "mip_gap",
    }

    array_keys = {
        # Shared parameters
        "mu",
        "v",
        "mu_0",
        "v_0",
        "c",
        "xi",

        # Parameters used by the gamma method
        "replacement_mu",
        "tau",
        "gamma_beta",
        "repair_rho",

        # Parameters used by the rainflow method
        "support",
        "cgf",
        "mu_trans",
        "v_trans",
        "support_trans",
        "cgf_trans",
    }

    with h5py.File(path, "r") as f:
        # H may be a scalar (single horizon) or a 2-element [H1, H2].
        if "H" in f:
            hval = f["H"][()]
            data["H"] = hval.tolist() if np.ndim(hval) > 0 else float(hval)
        elif "H" in f.attrs:
            hval = f.attrs["H"]
            data["H"] = hval.tolist() if np.ndim(hval) > 0 else float(hval)
        for key in scalar_keys:
            if key in f.attrs:
                data[key] = float(f.attrs[key])
            elif key in f:
                data[key] = float(f[key][()])
        for key in array_keys:
            if key in f:
                value = f[key][()]
                data[key] = value.tolist() if np.ndim(value) else float(value)
            elif key in f.attrs:
                value = f.attrs[key]
                data[key] = value.tolist() if np.ndim(value) else float(value)

    return data


def _resolve_results_path(results_path) -> Path:
    """Resolve the results path, applying defaults for missing name or extension."""
    if results_path is None:
        return Path("output.yaml")
    p = Path(results_path)
    if p.suffix == "":
        p = p.with_suffix(".yaml")
    return p


def _parse_horizon(data: dict):
    """Read H as either a single int or a two-element [H1, H2].

    Returns (H_value, H1, H2) where:
      * H_value is what gets passed to the solver: an int for a single horizon,
        or a (H1, H2) tuple for a transitory + operating split.
      * H2 is the OPERATING period (the length the per-mission profiles are
        broadcast to); for a single horizon H1 == H2 == H.
    """
    H_raw = data["H"]
    if isinstance(H_raw, (list, tuple)):
        if len(H_raw) != 2:
            raise ValueError("'H' must be an int or a two-element list [H1, H2].")
        H1, H2 = int(H_raw[0]), int(H_raw[1])
        if H1 <= 0 or H2 <= 0:
            raise ValueError(f"H1 and H2 must be positive (got {H1}, {H2}).")
        return (H1, H2), H1, H2
    H = int(H_raw)
    return H, H, H


def _extract_parameters(data: dict, degradation: str) -> dict:
    # Extract and validate all solver parameters from the parsed input data.
    required = REQUIRED_KEYS_BY_DEGRADATION[degradation]
    missing = required - set(data.keys())
    if missing:
        raise KeyError(f"Missing required keys in input file: {sorted(missing)}")

    F = int(data["F"])
    H_value, H1, H2 = _parse_horizon(data)
    M = int(data["M"])
    L = int(data.get("L", 1))
    alpha = float(data["alpha"]) if "alpha" in data else None
    epsilon = float(data["epsilon"])
    C_M = float(data["C_M"])
    C_R = float(data["C_R"])
    C_S = float(data["C_S"])
    C_P = float(data["C_P"])

    # Two-horizon (H = [H1, H2]) is only supported by the rainflow solver; the
    # Gaussian / inverse-Gaussian solvers expect a single scalar horizon.
    if isinstance(H_value, tuple) and degradation != "rainflow":
        raise ValueError(
            "A two-horizon 'H' (list [H1, H2]) is only supported for the "
            "'rainflow' degradation model."
        )

    # Optional parameters with defaults
    verbose = int(data.get("verbose", 1))
    mip_gap_raw = data.get("mip_gap", None)
    mip_gap = float(mip_gap_raw) if mip_gap_raw is not None else None
    tau = float(data.get("tau", 1.0))
    time_limit_raw = data.get("time_limit", None)          # in seconds
    time_limit = int(time_limit_raw) if time_limit_raw is not None else None

    # --- Broadcast mu_0: accept (F,) when L=1, or (F, L) ---
    mu_0 = np.array(data["mu_0"], dtype=float)
    if L == 1 and mu_0.shape == (F,):
        mu_0 = mu_0[:, np.newaxis]
    elif mu_0.shape != (F, L):
        raise ValueError(
            f"'mu_0' shape {mu_0.shape} does not match (F={F}, L={L})."
        )

    # --- Broadcast mu_param: accept multiple shapes (operating period H2) ---
    mu_param = np.array(data["mu"], dtype=float)
    mu_param = _broadcast_4d_param(mu_param, F, M, L, H2, "mu")

    if degradation == "gamma":
        C_rep = float(data["C_rep"])
        gamma_beta = _broadcast_component_param(
            data["gamma_beta"], L, "gamma_beta"
        )
        repair_rho = _broadcast_component_param(
            data["repair_rho"], L, "repair_rho"
        )
        tau = _broadcast_component_param(data["tau"], L, "tau")
        replacement_mu = np.array(
            data.get("replacement_mu", np.zeros((F, L))), dtype=float
        )
        if L == 1 and replacement_mu.shape == (F,):
            replacement_mu = replacement_mu[:, np.newaxis]
        elif replacement_mu.shape != (F, L):
            raise ValueError(
                "'replacement_mu' shape "
                f"{replacement_mu.shape} does not match (F={F}, L={L})."
            )
        return {
            "F": F, "H": H1, "M": M, "L": L,
            "mu_param": mu_param, "tau": tau,
            "epsilon": epsilon, "gamma_beta": gamma_beta,
            "C_M": C_M, "C_R": C_R, "C_rep": C_rep,
            "C_S": C_S, "C_P": C_P,
            "mu_0": mu_0, "replacement_mu": replacement_mu,
            "repair_rho": repair_rho,
            "verbose": verbose, "mip_gap": mip_gap,
        }


    # xi belongs to the repair-based legacy backends, not exact Gamma.
    xi = np.array(data["xi"], dtype=float)
    if L == 1 and xi.shape == (F,):
        xi = xi[:, np.newaxis]
    elif xi.shape != (F, L):
        raise ValueError(
            f"'xi' shape {xi.shape} does not match (F={F}, L={L})."
        )

    if degradation == "gaussian":
        alpha = float(data["alpha"])
        # --- Broadcast v_0: accept (F,) when L=1, or (F, L) ---
        v_0 = np.array(data["v_0"], dtype=float)
        if L == 1 and v_0.shape == (F,):
            v_0 = v_0[:, np.newaxis]
        elif v_0.shape != (F, L):
            raise ValueError(
                f"'v_0' shape {v_0.shape} does not match (F={F}, L={L})."
            )

        # --- Broadcast v_param: accept multiple shapes ---
        v_param = np.array(data["v"], dtype=float)
        v_param = _broadcast_4d_param(v_param, F, M, L, H2, "v")

        return {
            "F": F, "H": H_value, "M": M, "L": L,
            "mu_param": mu_param, "v_param": v_param,
            "alpha": alpha, "epsilon": epsilon, "xi": xi,
            "C_M": C_M, "C_R": C_R, "C_S": C_S, "C_P": C_P,
            "mu_0": mu_0, "v_0": v_0,
            "verbose": verbose,
            "mip_gap": mip_gap,
        }

    elif degradation == "rainflow":
        # Rainflow / remaining-life model. Same (mu, v) accumulated-damage state
        # as the Gaussian model, but the threshold is `tau` (Palmgren-Miner
        # limit) and reliability P(D > tau) <= eps is enforced by a concentration
        # bound chosen via `method`. Variance is ALWAYS required (the solver
        # tracks it for every method); `support` / `cgf` / `s` are only needed
        # for the bounds that consume them, and the solver validates that.

        # --- Broadcast v_0: accept (F,) when L=1, or (F, L) ---
        v_0 = np.array(data["v_0"], dtype=float)
        if L == 1 and v_0.shape == (F,):
            v_0 = v_0[:, np.newaxis]
        elif v_0.shape != (F, L):
            raise ValueError(
                f"'v_0' shape {v_0.shape} does not match (F={F}, L={L})."
            )

        # --- Broadcast v_param: accept multiple shapes (operating period H2) ---
        v_param = np.array(data["v"], dtype=float)
        v_param = _broadcast_4d_param(v_param, F, M, L, H2, "v")

        # --- Reliability bound: markov | cantelli | hoeffding | bernstein | chernoff ---
        method = str(data.get("method", "cantelli"))

        # --- Optional per-mission support width (Hoeffding / Bernstein) ---
        support_raw = data.get("support", data.get("support_param", None))
        support_param = (
            _broadcast_4d_param(np.array(support_raw, dtype=float), F, M, L, H2, "support")
            if support_raw is not None else None
        )

        # --- Optional per-mission CGF at s, and the tilt s (Chernoff) ---
        cgf_raw = data.get("cgf", data.get("cgf_param", None))
        cgf_param = (
            _broadcast_4d_param(np.array(cgf_raw, dtype=float), F, M, L, H2, "cgf")
            if cgf_raw is not None else None
        )
        s_raw = data.get("s_chernoff", data.get("s", None))
        s_chernoff = float(s_raw) if s_raw is not None else None

        # --- Optional transitory-phase profiles (broadcast to H1) ---------
        # Used only when H is a two-element [H1, H2]; give the transitory run-up
        # a different regime.  When omitted the transitory phase reuses the
        # operating profiles.
        def _opt_trans(key_names, name):
            raw = None
            for kn in key_names:
                if kn in data and data[kn] is not None:
                    raw = data[kn]
                    break
            if raw is None:
                return None
            return _broadcast_4d_param(np.array(raw, dtype=float), F, M, L, H1, name)

        mu_param_trans = _opt_trans(("mu_trans", "mu_param_trans"), "mu_trans")
        v_param_trans = _opt_trans(("v_trans", "v_param_trans"), "v_trans")
        support_param_trans = _opt_trans(
            ("support_trans", "support_param_trans"), "support_trans")
        cgf_param_trans = _opt_trans(("cgf_trans", "cgf_param_trans"), "cgf_trans")

        return {
            "F": F, "H": H_value, "M": M, "L": L,
            "mu_param": mu_param, "v_param": v_param,
            "tau": tau, "epsilon": epsilon, "xi": xi,
            "C_M": C_M, "C_R": C_R, "C_S": C_S, "C_P": C_P,
            "mu_0": mu_0, "v_0": v_0,
            "method": method,
            "support_param": support_param,
            "cgf_param": cgf_param,
            "s_chernoff": s_chernoff,
            "mu_param_trans": mu_param_trans,
            "v_param_trans": v_param_trans,
            "support_param_trans": support_param_trans,
            "cgf_param_trans": cgf_param_trans,
            "verbose": verbose,
            "mip_gap": mip_gap,
            "time_limit": time_limit,
        }
    
    else:  # inverse_gaussian
        # --- Broadcast c: accept (F,) when L=1, or (F, L) ---
        c = np.array(data["c"], dtype=float)
        if L == 1 and c.shape == (F,):
            c = c[:, np.newaxis]
        elif c.shape != (F, L):
            raise ValueError(
                f"'c' shape {c.shape} does not match (F={F}, L={L})."
            )

        return {
            "F": F, "H": H_value, "M": M, "L": L,
            "mu_param": mu_param, "c": c,
            "alpha": alpha, "epsilon": epsilon, "xi": xi,
            "C_M": C_M, "C_R": C_R, "C_S": C_S, "C_P": C_P,
            "mu_0": mu_0,
            "verbose": verbose,
            "mip_gap": mip_gap,
        }


"""def _broadcast_4d_param(value, F: int, M: int, L: int, H: int, name: str):
    # Backward-compatible wrapper around the central broadcast helper.

    return broadcast_4d_param(value, F, M, L, H, name)"""

def _broadcast_4d_param(arr: np.ndarray, F: int, M: int, L: int, H: int,
                        name: str) -> np.ndarray:
    # Broadcast an array to shape (F, M, L, H), handling legacy shapes.
    #
    # Accepted shapes:
    # - (F, M, L, H) — use directly
    # - (F, M, L)    — repeat along H
    # - (F, M, H) with L=1 — insert L dimension, giving (F, M, 1, H)
    # - (F, M) with L=1 — insert L dimension and repeat along H
    
    if arr.shape == (F, M, L, H):
        return arr
    if arr.ndim == 3 and arr.shape == (F, M, L):
        return np.repeat(arr[:, :, :, np.newaxis], H, axis=3)
    if L == 1 and arr.ndim == 3 and arr.shape == (F, M, H):
        return arr[:, :, np.newaxis, :]
    if L == 1 and arr.ndim == 2 and arr.shape == (F, M):
        arr = arr[:, :, np.newaxis, np.newaxis]
        return np.repeat(arr, H, axis=3)
    raise ValueError(
        f"'{name}' shape {arr.shape} cannot be broadcast to "
        f"(F={F}, M={M}, L={L}, H={H})."
    )


def _broadcast_component_param(value, L: int, name: str) -> np.ndarray:
    """Accept a scalar or one value per component."""

    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(L, float(array))
    if array.shape == (L,):
        return array
    raise ValueError(
        f"'{name}' shape {array.shape} must be scalar or ({L},)."
    )


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
        "F": result["F"],
        "M": result["M"],
        "H": result["H"],
        "L": result["L"],
        "mu_0": result["mu_0"].tolist(),
    }

    # Optional scalar parameters
    if result.get("alpha") is not None:
        output["alpha"] = result["alpha"]

    # Two-horizon / rainflow metadata
    for key in ("H1", "H2", "T", "method", "repair_model"):
        if result.get(key) is not None:
            output[key] = result[key]

    # Optional method-specific arrays
    for key in (
        "tau",
        "gamma_beta",
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
        output["u"] = result["u"].tolist()
        output["z"] = result["z"].tolist()

        # Optional solution arrays
        for key in (
            "v",
            "A",
            "tail_probability",
            "m",
            "r",
        ):
            if result.get(key) is not None:
                output[key] = _to_builtin(result[key])

    # Performance measurements may exist even without a solution.
    if result.get("performance") is not None:
        output["performance"] = _to_builtin(result["performance"])

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
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)


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

        # Two-horizon / rainflow metadata
        for key in ("H1", "H2", "T", "method", "repair_model"):
            if result.get(key) is not None:
                f.attrs[key] = result[key]

        # Method-specific arrays
        for key in (
            "tau",
            "gamma_beta",
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
                "m",
                "r",
            ):
                if result.get(key) is not None:
                    f.create_dataset(key, data=result[key])