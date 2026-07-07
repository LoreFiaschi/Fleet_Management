"""Mid-layer between the user and the per-component MILP builders.

Owns I/O (YAML/JSON/HDF5), the full input-validation suite (spec/spec.tex
Section 3, "Consistency Checks"), eager broadcasting of the mission-dependent
tensors, assembly of the single shared Gurobi model (mixed degradation models
share the same assignment variables x/x_m/x_r), the per-(train, component)
dispatch to the correct model module, and the horizon loop (scalar or
[H_min, H_max] interval, optionally parallel across workers with optional
warm-start hinting).

Currently implemented degradation models: "gaussian", "inverse_gaussian".
"wiener", "gamma", and "rainflow" are recognized by the input schema but raise
``NotImplementedError`` -- they are planned for a follow-up pass.
"""

import json
import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import gurobipy as gp
import h5py
import numpy as np
import yaml
from gurobipy import GRB

from fleet_management.models import base, gaussian, inverse_gaussian

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".yaml", ".yml", ".json", ".h5", ".hdf5"}
SUPPORTED_MODELS = {"gaussian", "inverse_gaussian", "wiener", "gamma", "rainflow"}
IMPLEMENTED_MODELS = {"gaussian", "inverse_gaussian"}

_DISPATCH = {
    gaussian.MODEL_NAME: gaussian.build_component,
    inverse_gaussian.MODEL_NAME: inverse_gaussian.build_component,
}

_GRB_STATUS_NAMES = {
    GRB.INFEASIBLE: "infeasible",
    GRB.INF_OR_UNBD: "infeasible_or_unbounded",
    GRB.UNBOUNDED: "unbounded",
    GRB.TIME_LIMIT: "time_limit",
    GRB.SUBOPTIMAL: "suboptimal",
}


# ======================================================================
# Public API
# ======================================================================

def solve(input_path: str, results_path: str = "output.yaml") -> dict:
    """Read, validate, solve, and (optionally) persist a fleet-management problem.

    Parameters
    ----------
    input_path : str
        Path to an input file (YAML, JSON, or HDF5) following the schema in
        spec/spec.tex, Section "Input File Schema".
    results_path : str, optional
        Where to write the result. Defaults to "output.yaml". Pass ``None`` to
        skip writing and only receive the returned dict.

    Returns
    -------
    dict
        A single-horizon result dict (see spec's Output Specification) if
        ``H`` is a scalar, or a dict keyed by ``H`` if ``H`` is
        ``[H_min, H_max]``.
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_file.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported input file type '{input_file.suffix}'. "
            f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    data = _read_input(input_file)
    shared = _parse_and_validate(data)

    if isinstance(shared["H"], list):
        result = _solve_horizon_loop(shared)
    else:
        result = _solve_single_horizon(shared["H"], shared)

    if results_path is not None:
        path = _resolve_results_path(results_path)
        _check_results_dir(path)
        _save_results(result, path, shared)

    return result


# ======================================================================
# Parsing, broadcasting, and validation (spec Sections 3 and "Input File Schema")
# ======================================================================

def _parse_and_validate(data: dict) -> dict:
    F = int(data["F"])
    M = int(data["M"])
    L = int(data["L"])
    if L <= 0:
        raise ValueError("L must be a positive integer.")

    H_raw = data["H"]
    if isinstance(H_raw, (list, tuple)):
        if len(H_raw) != 2:
            raise ValueError("H as a list must have exactly two elements [H_min, H_max].")
        h_min, h_max = int(H_raw[0]), int(H_raw[1])
        if h_min < 1:
            raise ValueError(f"H_min must be >= 1 (got {h_min}).")
        if h_max <= h_min:
            raise ValueError(f"H_max must be > H_min (got H_min={h_min}, H_max={h_max}).")
        H = [h_min, h_max]
        H_period = h_max
    else:
        H = int(H_raw)
        if H < 1:
            raise ValueError(f"H must be >= 1 (got {H}).")
        H_period = H

    if F <= M:
        raise ValueError(f"F must be greater than M (got F={F}, M={M}).")

    epsilon = float(data["epsilon"])
    if not (0 < epsilon <= 0.01):
        raise ValueError(f"epsilon must be in (0, 0.01] (got {epsilon}).")

    C_M = _require_scalar(data, "C_M")
    C_D = _require_scalar(data, "C_D")
    if C_D <= 0:
        raise ValueError("C_D must be positive.")

    penalty_type = data.get("penalty_type", "inf_norm")
    if penalty_type not in ("inf_norm", "quadratic"):
        raise ValueError(f"penalty_type must be 'inf_norm' or 'quadratic' (got '{penalty_type}').")

    formulation = data.get("formulation", "exact")
    if formulation == "socp":
        warnings.warn(
            "formulation='socp' is a deprecated alias for 'exact'.", UserWarning, stacklevel=3
        )
        formulation = "exact"
    if formulation not in ("exact", "lp"):
        raise ValueError(f"formulation must be 'exact' or 'lp' (got '{formulation}').")
    if formulation == "lp" and penalty_type == "quadratic":
        raise ValueError("penalty_type='quadratic' requires formulation='exact'.")

    n_workers = int(data.get("n_workers", 1))
    if n_workers < 1:
        raise ValueError("n_workers must be a positive integer.")
    warm_start = bool(data.get("warm_start", False))
    if warm_start and n_workers > 1:
        raise ValueError("warm_start=True requires n_workers=1 (sequential execution).")

    verbose = int(data.get("verbose", 1))
    mip_gap = data.get("mip_gap", None)
    mip_gap = float(mip_gap) if mip_gap is not None else None
    time_limit = data.get("time_limit", 3600.0)
    time_limit = float(time_limit) if time_limit is not None else None

    # --- Model assignment ---
    model_assignment = np.array(data["model"], dtype=object)
    if model_assignment.shape != (F, L):
        raise ValueError(
            f"'model' must have shape (F, L)=({F}, {L}), got {model_assignment.shape}."
        )
    unknown = sorted({m for m in model_assignment.ravel() if m not in SUPPORTED_MODELS})
    if unknown:
        raise ValueError(f"Unsupported degradation model(s) {unknown}. Supported: {sorted(SUPPORTED_MODELS)}.")
    not_implemented = sorted(set(model_assignment.ravel()) - IMPLEMENTED_MODELS)
    if not_implemented:
        raise NotImplementedError(
            f"Degradation model(s) {not_implemented} are not yet implemented in this "
            f"release (planned for a follow-up pass). Currently supported: "
            f"{sorted(IMPLEMENTED_MODELS)}."
        )

    maintenance_type = np.array(data["maintenance_type"], dtype=object)
    if maintenance_type.shape != (F, L):
        raise ValueError(
            f"'maintenance_type' must have shape (F, L)=({F}, {L}), got {maintenance_type.shape}."
        )
    bad_mt = sorted({mt for mt in maintenance_type.ravel() if mt not in ("ARA1", "ARD1")})
    if bad_mt:
        raise ValueError(f"maintenance_type entries must be 'ARA1' or 'ARD1', got {bad_mt}.")
    if np.any(maintenance_type == "ARA1"):
        raise NotImplementedError(
            "ARA1 maintenance is not yet implemented in this release (planned for a "
            "follow-up pass). All components must use 'ARD1' for now."
        )

    tau = _broadcast_FL(data["tau"], F, L, "tau")
    if not np.all(tau > 0):
        raise ValueError("tau must be positive element-wise.")
    rho = _broadcast_FL(data["rho"], F, L, "rho")
    C_R = _broadcast_cost_FL(data["C_R"], F, L, "C_R")
    C_rep = _broadcast_cost_FL(data["C_rep"], F, L, "C_rep")

    mu_0 = _broadcast_FL(data["mu_0"], F, L, "mu_0")
    mu_new = _broadcast_FL(data.get("mu_new", 0.0), F, L, "mu_new")
    if not np.all(mu_0 < tau):
        raise ValueError("mu_0 must be < tau element-wise.")

    mu_inc_raw = _broadcast_increment_raw(data["mu"], F, M, L, H_period, "mu")

    mask_gaussian = model_assignment == "gaussian"
    mask_ig = model_assignment == "inverse_gaussian"

    v_0 = np.full((F, L), np.nan)
    v_new = np.full((F, L), np.nan)
    v_inc_raw = None
    v_max_user = None
    eta = np.full((F, L), np.nan)

    if np.any(mask_gaussian):
        if "v_0" not in data:
            raise ValueError("'v_0' is required: at least one component uses the 'gaussian' model.")
        v_0_full = _broadcast_FL_nullable(data["v_0"], F, L, "v_0")
        v_0[mask_gaussian] = v_0_full[mask_gaussian]
        v_new_full = _broadcast_FL_nullable(data.get("v_new", 0.0), F, L, "v_new")
        v_new[mask_gaussian] = v_new_full[mask_gaussian]
        if "v" not in data:
            raise ValueError("'v' is required: at least one component uses the 'gaussian' model.")
        v_inc_raw = _broadcast_increment_raw_nullable(data["v"], F, M, L, H_period, "v")
        if "v_max_user" in data and data["v_max_user"] is not None:
            v_max_user = np.full((F, L), np.nan)
            vmu_full = _broadcast_FL_nullable(data["v_max_user"], F, L, "v_max_user")
            v_max_user[mask_gaussian] = vmu_full[mask_gaussian]
            applicable = v_max_user[~np.isnan(v_max_user)]
            if not np.all(applicable > 0):
                raise ValueError("v_max_user must be positive element-wise where specified.")
            if np.any(v_0[mask_gaussian] > v_max_user[mask_gaussian]):
                raise ValueError("v_max_user must be >= v_0 for every applicable component.")

    if np.any(mask_ig):
        if "eta" not in data:
            raise ValueError(
                "'eta' is required: at least one component uses the 'inverse_gaussian' model."
            )
        eta_full = _broadcast_FL_nullable(data["eta"], F, L, "eta")
        eta[mask_ig] = eta_full[mask_ig]
        if not np.all(eta[mask_ig] > 0):
            raise ValueError("eta must be positive element-wise.")

    gaussian.validate_inputs(mask_gaussian, mu_0, v_0, mu_inc_raw, v_inc_raw, tau, rho, maintenance_type)
    inverse_gaussian.validate_inputs(mask_ig, mu_0, mu_inc_raw, tau, rho, eta, maintenance_type)

    return {
        "F": F, "M": M, "L": L, "H": H, "H_period": H_period,
        "epsilon": epsilon, "C_M": C_M, "C_D": C_D,
        "penalty_type": penalty_type, "formulation": formulation,
        "n_workers": n_workers, "warm_start": warm_start,
        "verbose": verbose, "mip_gap": mip_gap, "time_limit": time_limit,
        "model_assignment": model_assignment, "maintenance_type": maintenance_type,
        "tau": tau, "rho": rho, "C_R": C_R, "C_rep": C_rep,
        "mu_0": mu_0, "mu_new": mu_new, "mu_inc_raw": mu_inc_raw,
        "v_0": v_0, "v_new": v_new, "v_inc_raw": v_inc_raw,
        "eta": eta, "v_max_user": v_max_user,
    }


def _require_scalar(data: dict, key: str) -> float:
    val = np.asarray(data[key], dtype=float)
    if val.ndim != 0:
        raise ValueError(f"'{key}' must be a scalar float.")
    return float(val)


def _broadcast_FL(raw, F: int, L: int, name: str) -> np.ndarray:
    """Accept a scalar or an (F, L) array; broadcast the scalar case."""
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 0:
        return np.full((F, L), float(arr))
    if arr.shape == (F, L):
        return arr.astype(float)
    raise ValueError(f"'{name}' must be scalar or shape (F,L)=({F},{L}), got {arr.shape}.")


def _broadcast_cost_FL(raw, F: int, L: int, name: str) -> np.ndarray:
    """Accept a scalar, (F,), or (F, L) array (used for C_R, C_rep)."""
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 0:
        return np.full((F, L), float(arr))
    if arr.shape == (F,):
        return np.repeat(arr[:, np.newaxis], L, axis=1)
    if arr.shape == (F, L):
        return arr.astype(float)
    raise ValueError(f"'{name}' must be scalar, (F,), or (F,L), got {arr.shape}.")


def _broadcast_increment_raw(raw, F: int, M: int, L: int, H: int, name: str) -> np.ndarray:
    """Validate and expand an increment tensor to (F, M, L, H) (not yet doubled to 2H).

    Accepts (F, M, L) -- repeated along a new H axis -- or (F, M, L, H) directly.
    For an H-interval solve, ``H`` here is H_max: the array is expected to cover
    the largest period requested, and each individual solve at H_k <= H_max
    uses the first H_k entries of this array (see ``_wrap_increment``).
    """
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 3:
        if arr.shape != (F, M, L):
            raise ValueError(
                f"'{name}' shape {arr.shape} must be (F,M,L)=({F},{M},{L}) or "
                f"(F,M,L,H)=({F},{M},{L},{H})."
            )
        return np.repeat(arr[..., np.newaxis], H, axis=3)
    if arr.shape != (F, M, L, H):
        raise ValueError(
            f"'{name}' shape {arr.shape} must be (F,M,L)=({F},{M},{L}) or "
            f"(F,M,L,H)=({F},{M},{L},{H})."
        )
    return arr


def _wrap_increment(raw: np.ndarray, F: int, M: int, L: int, H_solve: int) -> np.ndarray:
    """Periodic-wrap a raw (F, M, L, H_period) increment tensor to (F, M, L, 2*H_solve)."""
    sliced = raw[:, :, :, :H_solve]
    return np.concatenate([sliced, sliced], axis=3)


def _to_float_nullable(arr_obj: np.ndarray) -> np.ndarray:
    """Elementwise object -> float, mapping None (YAML/JSON null) to NaN."""
    return np.vectorize(lambda v: np.nan if v is None else float(v), otypes=[float])(arr_obj)


def _broadcast_FL_nullable(raw, F: int, L: int, name: str) -> np.ndarray:
    """Like ``_broadcast_FL``, but (F, L)-shaped input may contain null entries.

    Used for model-specific fields (v_0, v_new, eta, v_max_user) where the spec's
    own input schema puts null at positions that don't apply to that field's model
    (spec/spec.tex, "Input File Schema" -> "Parser behaviour for null entries").
    """
    if raw is None:
        return np.full((F, L), np.nan)
    arr_obj = np.array(raw, dtype=object)
    if arr_obj.ndim == 0:
        return np.full((F, L), np.nan if raw is None else float(raw))
    if arr_obj.shape != (F, L):
        raise ValueError(f"'{name}' must be scalar or shape (F,L)=({F},{L}), got {arr_obj.shape}.")
    return _to_float_nullable(arr_obj)


def _broadcast_increment_raw_nullable(raw, F: int, M: int, L: int, H: int, name: str) -> np.ndarray:
    """Like ``_broadcast_increment_raw``, but entries may be null (see above)."""
    arr_obj = np.array(raw, dtype=object)
    if arr_obj.ndim == 3:
        if arr_obj.shape != (F, M, L):
            raise ValueError(
                f"'{name}' shape {arr_obj.shape} must be (F,M,L)=({F},{M},{L}) or "
                f"(F,M,L,H)=({F},{M},{L},{H})."
            )
        arr = _to_float_nullable(arr_obj)
        return np.repeat(arr[..., np.newaxis], H, axis=3)
    if arr_obj.shape != (F, M, L, H):
        raise ValueError(
            f"'{name}' shape {arr_obj.shape} must be (F,M,L)=({F},{M},{L}) or "
            f"(F,M,L,H)=({F},{M},{L},{H})."
        )
    return _to_float_nullable(arr_obj)


# ======================================================================
# Single-horizon model assembly and solve
# ======================================================================

def _solve_single_horizon(H: int, shared: dict, warm_hint: dict = None) -> dict:
    F, M, L = shared["F"], shared["M"], shared["L"]
    two_h = 2 * H
    mu_inc = _wrap_increment(shared["mu_inc_raw"], F, M, L, H)
    v_inc = None
    if shared["v_inc_raw"] is not None:
        v_inc = _wrap_increment(shared["v_inc_raw"], F, M, L, H)

    model = gp.Model("fleet_management")
    model.Params.OutputFlag = int(shared["verbose"])
    if shared["mip_gap"] is not None:
        model.Params.MIPGap = shared["mip_gap"]
    if shared["time_limit"] is not None:
        model.Params.TimeLimit = shared["time_limit"]

    model_assignment = shared["model_assignment"]
    if shared["formulation"] == "exact" and np.any(model_assignment == "gaussian"):
        model.Params.NonConvex = 2

    x = model.addVars(F, M + 1, two_h, vtype=GRB.BINARY, name="x")
    x_m = model.addVars(F, L, two_h, vtype=GRB.BINARY, name="x_m")
    x_r = model.addVars(F, L, two_h, vtype=GRB.BINARY, name="x_r")
    z = model.addVars(F, L, two_h, lb=0.0, name="z")
    u = model.addVar(lb=0.0, name="u")

    ctx = SimpleNamespace(
        model=model, F=F, H=H, M=M, L=L, two_h=two_h,
        x=x, x_m=x_m, x_r=x_r, z=z, u=u,
        mu={}, v={},
        tau=shared["tau"], rho=shared["rho"],
        mu_0=shared["mu_0"], v_0=shared["v_0"],
        mu_new=shared["mu_new"], v_new=shared["v_new"],
        mu_inc=mu_inc, v_inc=v_inc,
        eta=shared["eta"], v_max_user=shared["v_max_user"],
        phi_inv_sq=base.phi_inv_sq(shared["epsilon"]),
        epsilon=shared["epsilon"], formulation=shared["formulation"],
    )

    base.add_assignment_constraints(model, x, x_m, x_r, F, L, M, two_h)

    for i in range(F):
        for l in range(L):
            _DISPATCH[model_assignment[i, l]](ctx, i, l)

    base.add_damage_regularization(model, u, ctx.mu, F, L, two_h, shared["penalty_type"])

    obj = shared["C_D"] * u
    obj += shared["C_M"] * gp.quicksum(x[i, 0, k] for i in range(F) for k in range(two_h))
    obj += gp.quicksum(
        float(shared["C_R"][i, l]) * z[i, l, k]
        for i in range(F) for l in range(L) for k in range(two_h)
    )
    obj += gp.quicksum(
        float(shared["C_rep"][i, l]) * x_r[i, l, k]
        for i in range(F) for l in range(L) for k in range(two_h)
    )
    model.setObjective(obj, GRB.MINIMIZE)

    if warm_hint is not None:
        _apply_warm_hint(x, x_m, x_r, warm_hint, F, L, M, two_h)

    model.optimize()
    return _extract_result(model, ctx, shared)


def _apply_warm_hint(x, x_m, x_r, hint: dict, F: int, L: int, M: int, two_h: int) -> None:
    """Best-effort VarHintVal seeding from a previous horizon's solution.

    2H changes between horizons, so only the overlapping first ``min(2H_prev,
    2H)`` steps can be hinted; per spec this carries no feasibility guarantee.
    """
    if hint.get("x") is None:
        return
    prev_two_h = np.asarray(hint["x"]).shape[2]
    overlap = min(prev_two_h, two_h)
    prev_x, prev_xm, prev_xr = np.asarray(hint["x"]), np.asarray(hint["x_m"]), np.asarray(hint["x_r"])
    for i in range(F):
        for k in range(overlap):
            for j in range(M + 1):
                x[i, j, k].VarHintVal = float(prev_x[i, j, k])
            for l in range(L):
                x_m[i, l, k].VarHintVal = float(prev_xm[i, l, k])
                x_r[i, l, k].VarHintVal = float(prev_xr[i, l, k])


def _status_name(status_code: int) -> str:
    return _GRB_STATUS_NAMES.get(status_code, f"status_code_{status_code}")


def _extract_result(model: "gp.Model", ctx: SimpleNamespace, shared: dict) -> dict:
    F, L, M, H, two_h = ctx.F, ctx.L, ctx.M, ctx.H, ctx.two_h
    result = {
        "H": H, "F": F, "M": M, "L": L,
        "model": shared["model_assignment"].tolist(),
        "tail_bound": [[None] * L for _ in range(F)],
        # Not part of the spec's literal Output Specification table, but the
        # plotter needs tau to normalize the mu/tau heatmap (spec's plot layout
        # section) and the output table otherwise has no per-component
        # threshold field, so it is echoed here for convenience.
        "tau": shared["tau"].tolist(),
    }
    if model.Status != GRB.OPTIMAL:
        result["status"] = _status_name(model.Status)
        result["objective"] = None
        for key in ("x", "x_m", "x_r", "mu", "v", "u", "z"):
            result[key] = None
        return result

    x_sol = np.zeros((F, M + 1, two_h))
    xm_sol = np.zeros((F, L, two_h))
    xr_sol = np.zeros((F, L, two_h))
    mu_sol = np.zeros((F, L, two_h))
    v_sol = np.full((F, L, two_h), np.nan)
    z_sol = np.zeros((F, L, two_h))

    for i in range(F):
        for k in range(two_h):
            for j in range(M + 1):
                x_sol[i, j, k] = ctx.x[i, j, k].X
            for l in range(L):
                xm_sol[i, l, k] = ctx.x_m[i, l, k].X
                xr_sol[i, l, k] = ctx.x_r[i, l, k].X
                z_sol[i, l, k] = ctx.z[i, l, k].X
                mu_sol[i, l, k] = ctx.mu[i, l, k].X
                if (i, l, k) in ctx.v:
                    v_sol[i, l, k] = ctx.v[i, l, k].X

    result["status"] = "optimal"
    result["objective"] = float(model.ObjVal)
    result["x"] = x_sol
    result["x_m"] = xm_sol
    result["x_r"] = xr_sol
    result["mu"] = mu_sol
    result["v"] = v_sol
    result["u"] = float(ctx.u.X)
    result["z"] = z_sol
    return result


# ======================================================================
# Horizon loop (spec "Solver Loop Behaviour")
# ======================================================================

def _solve_horizon_loop(shared: dict) -> dict:
    h_min, h_max = shared["H"]
    n_workers = shared["n_workers"]
    warm_start = shared["warm_start"]
    results = {}

    if n_workers == 1:
        hint = None
        for h in range(h_min, h_max + 1):
            try:
                res = _solve_single_horizon(h, shared, warm_hint=hint if warm_start else None)
            except Exception as exc:  # noqa: BLE001 -- reported per-horizon, not fatal
                logger.error("Horizon H=%d failed: %s", h, exc)
                results[h] = {"status": f"error: {exc}", "objective": None, "H": h}
                hint = None
                continue
            results[h] = res
            hint = res if warm_start else None
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                h: pool.submit(_solve_single_horizon, h, shared, None)
                for h in range(h_min, h_max + 1)
            }
            for h, fut in futures.items():
                try:
                    results[h] = fut.result()
                except Exception as exc:  # noqa: BLE001
                    logger.error("Horizon H=%d failed: %s", h, exc)
                    results[h] = {"status": f"error: {exc}", "objective": None, "H": h}

    return results


# ======================================================================
# I/O
# ======================================================================

def _read_input(input_file: Path) -> dict:
    ext = input_file.suffix.lower()
    if ext in (".yaml", ".yml"):
        with open(input_file, "r") as f:
            return yaml.safe_load(f)
    if ext == ".json":
        with open(input_file, "r") as f:
            return json.load(f)
    if ext in (".h5", ".hdf5"):
        return _read_hdf5_input(input_file)
    raise ValueError(f"Unsupported input file type: {ext}")


def _decode_h5_value(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray) and value.dtype.kind in ("S", "O"):
        return np.vectorize(lambda b: b.decode() if isinstance(b, bytes) else b)(value).tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _read_hdf5_input(path: Path) -> dict:
    """Generic flat reader: every attribute and dataset becomes a dict entry."""
    data = {}
    with h5py.File(path, "r") as f:
        for key, val in f.attrs.items():
            data[key] = _decode_h5_value(val)
        for key in f.keys():
            data[key] = _decode_h5_value(f[key][()])
    return data


def _resolve_results_path(results_path) -> Path:
    if results_path is None:
        return Path("output.yaml")
    p = Path(results_path)
    if p.suffix == "":
        p = p.with_suffix(".yaml")
    return p


def _check_results_dir(path: Path) -> None:
    results_dir = path.parent
    if results_dir != Path("") and not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    if results_dir != Path("") and not os.access(results_dir, os.W_OK):
        raise PermissionError(f"Results directory is not writable: {results_dir}")


def _is_multi_horizon(H) -> bool:
    return isinstance(H, list)


def _build_serializable_single(res: dict) -> dict:
    out = {
        "status": res["status"],
        "objective": res["objective"],
        "H": res["H"], "F": res["F"], "M": res["M"], "L": res["L"],
        "model": res["model"],
        "tail_bound": res["tail_bound"],
        "tau": res["tau"],
    }
    for key in ("x", "x_m", "x_r", "mu", "v", "z"):
        out[key] = res[key].tolist() if res[key] is not None else None
    out["u"] = res["u"]
    return out


def _save_results(result: dict, path: Path, shared: dict) -> None:
    ext = path.suffix.lower()
    is_multi = _is_multi_horizon(shared["H"])
    if ext == ".json":
        _save_json(result, path, is_multi)
    elif ext in (".h5", ".hdf5"):
        _save_hdf5(result, path, is_multi, shared)
    else:
        _save_yaml(result, path, is_multi)


def _save_yaml(result, path: Path, is_multi: bool) -> None:
    out = (
        {h: _build_serializable_single(r) for h, r in result.items()}
        if is_multi
        else _build_serializable_single(result)
    )
    with open(path, "w") as f:
        yaml.dump(out, f, default_flow_style=False, sort_keys=False)


def _save_json(result, path: Path, is_multi: bool) -> None:
    out = (
        {str(h): _build_serializable_single(r) for h, r in result.items()}
        if is_multi
        else _build_serializable_single(result)
    )
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def _save_hdf5(result, path: Path, is_multi: bool, shared: dict) -> None:
    with h5py.File(path, "w") as f:
        if is_multi:
            for h, res in result.items():
                _write_hdf5_single(f.create_group(f"H{h}"), res, shared)
        else:
            _write_hdf5_single(f, result, shared)


def _write_hdf5_single(root: "h5py.Group", res: dict, shared: dict) -> None:
    meta = root.create_group("metadata")
    meta.attrs["status"] = res["status"]
    if res["objective"] is not None:
        meta.attrs["objective"] = float(res["objective"])
    meta.attrs["F"] = res["F"]
    meta.attrs["H"] = res["H"]
    meta.attrs["M"] = res["M"]
    meta.attrs["L"] = res["L"]

    if res["x"] is not None:
        sol = root.create_group("solution")
        sol.create_dataset("x", data=np.asarray(res["x"], dtype=np.int8))
        sol.create_dataset("x_m", data=np.asarray(res["x_m"], dtype=np.int8))
        sol.create_dataset("x_r", data=np.asarray(res["x_r"], dtype=np.int8))
        sol.create_dataset("mu", data=np.asarray(res["mu"], dtype=np.float64))
        sol.create_dataset("v", data=np.asarray(res["v"], dtype=np.float64))
        sol.create_dataset("u", data=float(res["u"]))
        sol.create_dataset("z", data=np.asarray(res["z"], dtype=np.float64))

    params = root.create_group("parameters")
    params.attrs["epsilon"] = shared["epsilon"]
    params.create_dataset("tau", data=shared["tau"])
    params.create_dataset("rho", data=shared["rho"])
    # Not in the spec's literal HDF5 schema table (which predates per-component
    # model assignment), but required for the plotter to pick border styles
    # when reading an HDF5 result file, so it is persisted here for parity
    # with the YAML/JSON "model" output field.
    params.create_dataset(
        "model", data=np.array(shared["model_assignment"], dtype=h5py.string_dtype())
    )
