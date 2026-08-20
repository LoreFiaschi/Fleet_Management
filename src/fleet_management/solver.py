import json
import os
from pathlib import Path

import h5py
import numpy as np
import yaml

# Three routes: a gamma-only fleet uses the existing gamma backend; a uniform
# rainflow fleet uses the rainflow builder; a genuinely mixed fleet is assembled
# per cell on the shared model layer in base.py.
from fleet_management.degradation_model.gamma_utils.gamma_gurobi import solve_fleet_management as solve_gamma
from fleet_management.degradation_model.rainflow import solve as rainflow_solve
from fleet_management.degradation_model.base import solve_mixed as base_solve_mixed

from fleet_management.config import load_config, FleetConfig

SUPPORTED_EXTENSIONS = {".yaml", ".yml", ".json", ".h5", ".hdf5"}


def solve(input_path: str, results_path: str = None) -> dict:   # was -> None, now -> dict for performance measurement
    """
    Mid-layer between the user and the fleet-management solvers.

    The input file is self-describing: it must carry a top-level ``model:`` key
    (see ``config.load_config``).  There is no separate ``degradation`` argument
    and no legacy path -- every case is treated as mixed, where "mixed" spans
    both a genuinely heterogeneous fleet and one model everywhere.

    Parameters
    ----------
    input_path : str
        Path to an input file containing the problem data.
        Supported formats: YAML (.yaml/.yml), JSON (.json), HDF5 (.h5/.hdf5).
    results_path : str, optional
        Path where results will be saved. Defaults to "output.yaml".
        If provided without an extension, ".yaml" is appended.
    """
    # --- File checks ---
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

    # --- Read, normalize, and validate input via config.load_config ---
    data = _read_input(input_file)
    cfg = load_config(data)

    # --- Solve (uniform single-model is bridged; heterogeneous -> Step 2) ---
    result = _solve_mixed(cfg)

    result.setdefault("performance", {})                        # performance measurement

    # --- Save results ---
    _save_results(result, results_path)

    return result                                               # performance measurement


# ---------------------------------------------------------------------------
# Solve dispatch (all inputs are "mixed"; uniform fleets bridge to a backend)
# ---------------------------------------------------------------------------
def _solve_mixed(cfg: "FleetConfig") -> dict:
    """Solve a normalized FleetConfig — three routes by fleet composition.

    1. **gamma-only**  -> the existing gamma backend (kept as a regression
       oracle while the modular tail-bound block is introduced);
    2. **rainflow-only** -> the rainflow builder (``rainflow.solve``);
    3. **mixed** (cells use different degradation models) -> ``base.solve_mixed``,
       which builds the shared skeleton once and then fills in each cell's
       constraints through that cell's registered model builder.
    """
    models = set(cfg.models)

    if models == {"gamma"}:                                   # 1. gamma-only
        result = solve_gamma(**_cfg_to_gamma_kwargs(cfg))
        result["mu_0"] = cfg.mu_0
        result["degradation"] = "gamma"
        return result

    if models == {"rainflow"}:                                # 2. rainflow-only
        result = rainflow_solve(cfg)
        result["degradation"] = "rainflow"
        return result

    result = base_solve_mixed(cfg)                            # 3. mixed per cell
    result["degradation"] = "mixed"
    return result


def _require_uniform(arr, name):
    """Collapse a per-cell (F, L) array to the single value the gamma backend
    expects, or explain that per-cell variation needs the modular builder."""
    vals = np.unique(np.asarray(arr))
    if vals.size != 1:
        raise NotImplementedError(
            f"the gamma backend takes a single '{name}', but it varies per "
            f"cell ({vals.tolist()}); per-cell '{name}' needs the modular builder."
        )
    return vals.reshape(-1)[0]


def _uniform_over_vehicles(arr, name):
    """Reduce an (F, L) array to (L,), requiring equal rows (the gamma backend
    treats these quantities as per-component, not per-vehicle)."""
    a = np.asarray(arr, dtype=float)
    if not np.allclose(a, a[0:1, :]):
        raise NotImplementedError(
            f"the gamma backend treats '{name}' as per-component, but it varies "
            "per vehicle here; per-vehicle variation needs the Step-2 builder."
        )
    return a[0, :]


def _cfg_to_gamma_kwargs(cfg: "FleetConfig") -> dict:
    """Translate a uniform single-model gamma FleetConfig into solve_gamma
    kwargs for the CURRENT gamma backend.

    Gamma is single-horizon: if the input gave H = [H1, H2], only H1 is used
    (H = cfg.H1).  Component scalars (tau / gamma_beta / repair_rho) are reduced
    to (L,) and the mean profile is transposed to the backend's (F, M, L, H).
    This builds the full kwarg set the backend expects (F, H, M, L, mu_param,
    tau, epsilon, gamma_beta, repair_rho, C_M, C_R, C_rep, C_S, C_P, mu_0,
    replacement_mu, plus verbose / mip_gap)."""
    if "C_rep" not in cfg.costs:
        raise KeyError("gamma requires a fleet-wide 'C_rep' (replacement cost).")
    kw = {
        "F": cfg.F, "H": cfg.H1, "M": cfg.M, "L": cfg.L,
        "mu_param": np.transpose(cfg.mu, (0, 2, 1, 3)),
        "tau": _uniform_over_vehicles(cfg.tau, "tau"),
        "epsilon": float(_require_uniform(cfg.epsilon, "epsilon")),
        "gamma_beta": _uniform_over_vehicles(cfg.gamma_beta, "gamma_beta"),
        "repair_rho": _uniform_over_vehicles(cfg.rho, "rho"),
        "C_M": cfg.costs["C_M"], "C_R": cfg.costs["C_R"], "C_rep": cfg.costs["C_rep"],
        "C_S": cfg.costs.get("C_S", cfg.costs.get("C_D")), "C_P": cfg.costs["C_P"],
        "mu_0": cfg.mu_0, "replacement_mu": cfg.replacement_mu,
    }
    for opt in ("verbose", "mip_gap"):
        if opt in cfg.options:
            kw[opt] = cfg.options[opt]
    return kw


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
    for key in ("H1", "H2", "T", "method", "bound_method", "repair_model"):
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
        output["u"] = result["u"].tolist()
        output["z"] = result["z"].tolist()

        # Optional solution arrays
        for key in (
            "v",
            "A",
            "tail_probability",
            "gamma_shape_bound",
            "gamma_tail_bound",
            "m",
            "r",
        ):
            if result.get(key) is not None:
                output[key] = _to_builtin(result[key])

        if result.get("gamma_calibration") is not None:
            output["gamma_calibration"] = _to_builtin(result["gamma_calibration"])

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

        # Two-horizon / rainflow metadata
        for key in ("H1", "H2", "T", "method", "repair_model"):
            if result.get(key) is not None:
                f.attrs[key] = result[key]

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
            ):
                if result.get(key) is not None:
                    f.create_dataset(key, data=result[key])

            if result.get("gamma_calibration") is not None:
                f.attrs["gamma_calibration"] = json.dumps(
                    _to_builtin(result["gamma_calibration"])
                )
