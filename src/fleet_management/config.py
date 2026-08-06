"""
Fleet-management input loading, normalization, and validation.

Turns a raw input mapping (parsed YAML / JSON / dict) into a normalized
``FleetConfig`` of global arrays indexed consistently as (F, L) for per-cell
scalars and (F, L, M, H2) for per-mission increment profiles.  Per-cell
selector arrays ``model`` / ``bound_method`` / ``repair_model`` decide what
each (vehicle, component) cell is; model-specific arrays (``v``, ``support``,
``cgf``, ``gamma_beta`` ...) are provided fleet-wide and read only at the cells
whose model / bound requires them.

Self-describing input
---------------------
Every input must carry a top-level ``model:`` key (a single string, a length-L
list, or an (F, L) nested list) that assigns a degradation model to each
(vehicle, component) cell.  There is no separate legacy form and no
``degradation`` argument: a fleet with one model everywhere is simply the
uniform case of the same schema.

All correctness checks are done here in Python: types, canonical shapes /
broadcasting, and per-cell cross-field rules (e.g. a chernoff cell needs
``cgf`` + ``s_chernoff``).  There is no external schema dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


SUPPORTED_MODELS = ("rainflow", "gamma", "gaussian", "inverse_gaussian")
RAINFLOW_BOUNDS = ("markov", "cantelli", "hoeffding", "bernstein", "chernoff")
REPAIR_MODELS = ("ard1", "ardinf")
_NEEDS_SUPPORT = ("hoeffding", "bernstein")
_NEEDS_CGF = ("chernoff",)


# ---------------------------------------------------------------------------
# Normalized representation
# ---------------------------------------------------------------------------
@dataclass
class FleetConfig:
    F: int
    M: int
    L: int
    H: object                       # int or (H1, H2) as written
    H1: int
    H2: int
    T: int
    H_prof: int
    # per-cell (F, L)
    model: np.ndarray = None        # str
    bound_method: np.ndarray = None  # str ('' for non-rainflow cells)
    repair_model: np.ndarray = None  # str
    tau: np.ndarray = None
    epsilon: np.ndarray = None
    rho: np.ndarray = None
    mu_0: np.ndarray = None
    v_0: Optional[np.ndarray] = None
    replacement_mu: np.ndarray = None
    replacement_v: Optional[np.ndarray] = None
    s_chernoff: Optional[np.ndarray] = None
    gamma_beta: Optional[np.ndarray] = None
    # per-mission profiles (F, L, M, H2)  [transitory: H1]
    mu: np.ndarray = None
    v: Optional[np.ndarray] = None
    support: Optional[np.ndarray] = None
    cgf: Optional[np.ndarray] = None
    mu_trans: Optional[np.ndarray] = None
    v_trans: Optional[np.ndarray] = None
    support_trans: Optional[np.ndarray] = None
    cgf_trans: Optional[np.ndarray] = None
    # fleet-wide
    costs: dict = field(default_factory=dict)     # C_M, C_R, C_S, C_P, C_rep
    options: dict = field(default_factory=dict)   # verbose, mip_gap, ...
    # mode
    raw: dict = field(default_factory=dict)

    @property
    def models(self) -> list:
        return sorted(set(self.model.ravel().tolist())) if self.model is not None else []

    @property
    def is_single_model(self) -> bool:
        return len(self.models) <= 1

    def cell(self, i: int, l: int) -> dict:
        """Per-(vehicle, component) view for the modular constraint builders."""
        out = {"model": str(self.model[i, l]),
               "bound_method": str(self.bound_method[i, l]),
               "repair_model": str(self.repair_model[i, l]),
               "tau": float(self.tau[i, l]),
               "epsilon": float(self.epsilon[i, l]),
               "rho": float(self.rho[i, l]),
               "mu_0": float(self.mu_0[i, l]),
               "mu": self.mu[i, l],                                   # (M, H2)
               "replacement_mu": float(self.replacement_mu[i, l])}
        for name, arr in (("v_0", self.v_0), ("replacement_v", self.replacement_v),
                          ("s_chernoff", self.s_chernoff), ("gamma_beta", self.gamma_beta)):
            if arr is not None:
                out[name] = float(arr[i, l])
        for name, arr in (("v", self.v), ("support", self.support), ("cgf", self.cgf),
                          ("mu_trans", self.mu_trans), ("v_trans", self.v_trans),
                          ("support_trans", self.support_trans), ("cgf_trans", self.cgf_trans)):
            if arr is not None:
                out[name] = arr[i, l]                                # (M, H2) or (M, H1)
        return out

    def describe(self) -> str:
        counts = {m: int((self.model == m).sum()) for m in self.models}
        lines = [f"FleetConfig(F={self.F}, M={self.M}, L={self.L}, "
                 f"H={self.H} -> H1={self.H1}, H2={self.H2}, T={self.T})",
                 f"  models (cell counts): {counts}",
                 f"  bound_method:\n{np.array2string(self.bound_method, prefix='    ')}",
                 f"  costs  : {self.costs}",
                 f"  options: {self.options}"]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Horizon
# ---------------------------------------------------------------------------
def _parse_horizon(H_raw):
    if isinstance(H_raw, (list, tuple)):
        if len(H_raw) != 2:
            raise ValueError("'H' must be an int or a two-element list [H1, H2].")
        H1, H2 = int(H_raw[0]), int(H_raw[1])
        if H1 <= 0 or H2 <= 0:
            raise ValueError(f"H1 and H2 must be positive (got {H1}, {H2}).")
        return (H1, H2), H1, H2, H1 + H2
    H = int(H_raw)
    if H <= 0:
        raise ValueError(f"H must be positive (got {H}).")
    return H, H, H, 2 * H


# ---------------------------------------------------------------------------
# Broadcasting helpers  (new (F, L, ...) layout)
# ---------------------------------------------------------------------------
def _fl_scalar(value, F, L, name, default=None):
    """scalar / (L,) / (F,L) -> (F, L) float array.  None if value is None."""
    if value is None:
        value = default
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full((F, L), float(arr))
    if arr.shape == (L,):                           # per-component, tiled over F
        return np.broadcast_to(arr[np.newaxis, :], (F, L)).copy()
    if arr.shape == (F, L):
        return arr
    raise ValueError(f"'{name}' shape {arr.shape} must be scalar, ({L},), or ({F},{L}).")


def _fl_str(value, F, L, name, default=None):
    """single str / length-L list / (F,L) nested -> (F, L) str array."""
    if value is None:
        value = default
    if value is None:
        return None
    if isinstance(value, str):
        return np.full((F, L), value, dtype=object)
    arr = np.array(value, dtype=object)
    if arr.shape == (L,):                           # per-component, tiled over F
        return np.broadcast_to(arr[np.newaxis, :], (F, L)).copy()
    if arr.shape == (F, L):
        return arr
    raise ValueError(f"'{name}' shape {arr.shape} must be a string, ({L},), or ({F},{L}).")


def _flmh_prof(value, F, L, M, H, name):
    """scalar / (M,) / (L,M) / (L,M,H) / (F,L,M) / (F,L,M,H) -> (F, L, M, H)."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full((F, L, M, H), float(arr))
    if arr.shape == (M,):                                   # per-mission
        return np.broadcast_to(arr[None, None, :, None], (F, L, M, H)).copy()
    if arr.shape == (L, M):
        return np.broadcast_to(arr[None, :, :, None], (F, L, M, H)).copy()
    if arr.shape == (L, M, H):
        return np.broadcast_to(arr[None, :, :, :], (F, L, M, H)).copy()
    if arr.shape == (F, L, M):
        return np.repeat(arr[:, :, :, None], H, axis=3)
    if arr.shape == (F, L, M, H):
        return arr
    raise ValueError(
        f"'{name}' shape {arr.shape} must be scalar, ({M},), ({L},{M}), "
        f"({L},{M},{H}), ({F},{L},{M}), or ({F},{L},{M},{H})."
    )


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------
def load_config(data: dict) -> FleetConfig:
    """Normalize a raw input mapping into a ``FleetConfig``.

    The input must be self-describing: a top-level ``model:`` key (a single
    string, a length-L list, or an (F, L) nested list) assigns a degradation
    model to every (vehicle, component) cell.  A uniform ``model`` is just the
    special case of one model everywhere.
    """
    if not isinstance(data, dict):
        raise TypeError("input data must be a mapping (dict).")

    for key in ("F", "H", "M"):
        if key not in data:
            raise KeyError(f"Missing required top-level key '{key}'.")
    if "model" not in data:
        raise KeyError(
            "Missing required top-level key 'model'. Every input must name its "
            f"degradation model(s) (one of {SUPPORTED_MODELS}), as a single "
            "string or an (F, L) / length-L array."
        )
    F, M = int(data["F"]), int(data["M"])
    H_written, H1, H2, T = _parse_horizon(data["H"])
    if F <= 0 or M <= 0:
        raise ValueError(f"F and M must be positive integers (got F={F}, M={M}).")

    model = _fl_str(data["model"], F, _infer_L(data, F), name="model")
    L = model.shape[1]
    if "L" in data and int(data["L"]) != L:
        raise ValueError(f"top-level L={int(data['L'])} disagrees with model shape L={L}.")
    for m in set(model.ravel().tolist()):
        if m not in SUPPORTED_MODELS:
            raise ValueError(f"model contains unknown value {m!r}; supported {SUPPORTED_MODELS}.")

    # Profile horizon: rainflow cells use the operating horizon H2; a fleet made
    # up only of single-horizon models (gamma / gaussian / inverse_gaussian) uses
    # H1.  So with a two-horizon H = [H1, H2], gamma "just takes H1" -- including
    # the length its per-mission profiles are normalized to.
    any_rainflow = any(m == "rainflow" for m in model.ravel().tolist())
    H_prof = H2 if any_rainflow else H1

    # NOTE on two-horizon H = [H1, H2]: rainflow cells use both phases; gamma
    # (and the other single-horizon models) simply use H1 -- that reduction is
    # done in the solver's per-model kwargs builder, so nothing is rejected here.

    def alias(*names):
        for n in names:
            if n in data and data[n] is not None:
                return data[n]
        return None

    cfg = FleetConfig(
        F=F, M=M, L=L, H=H_written, H1=H1, H2=H2, T=T, H_prof=H_prof, model=model,
        bound_method=_fl_str(alias("bound_method", "method"), F, L, "bound_method",
                             default="cantelli"),
        repair_model=_fl_str(data.get("repair_model"), F, L, "repair_model", default="ard1"),
        tau=_fl_scalar(data["tau"], F, L, "tau"),
        epsilon=_fl_scalar(data["epsilon"], F, L, "epsilon"),
        rho=_fl_scalar(alias("rho", "xi", "repair_rho"), F, L, "rho"),
        mu_0=_fl_scalar(data["mu_0"], F, L, "mu_0"),
        v_0=_fl_scalar(data.get("v_0"), F, L, "v_0", default=0.0),
        replacement_mu=_fl_scalar(alias("replacement_mu", "mu_new"), F, L,
                                  "replacement_mu", default=0.0),
        replacement_v=_fl_scalar(alias("replacement_v", "v_new"), F, L,
                                 "replacement_v", default=0.0),
        s_chernoff=_fl_scalar(data.get("s_chernoff"), F, L, "s_chernoff"),
        gamma_beta=_fl_scalar(data.get("gamma_beta"), F, L, "gamma_beta"),
        mu=_flmh_prof(data["mu"], F, L, M, H_prof, "mu"),
        v=_flmh_prof(data.get("v"), F, L, M, H_prof, "v"),
        support=_flmh_prof(data.get("support"), F, L, M, H_prof, "support"),
        cgf=_flmh_prof(data.get("cgf"), F, L, M, H_prof, "cgf"),
        mu_trans=_flmh_prof(data.get("mu_trans"), F, L, M, H1, "mu_trans"),
        v_trans=_flmh_prof(data.get("v_trans"), F, L, M, H1, "v_trans"),
        support_trans=_flmh_prof(data.get("support_trans"), F, L, M, H1, "support_trans"),
        cgf_trans=_flmh_prof(data.get("cgf_trans"), F, L, M, H1, "cgf_trans"),
        costs={k: float(data[k]) for k in ("C_M", "C_R", "C_S", "C_P", "C_rep") if k in data},
        options={k: data[k] for k in ("verbose", "mip_gap", "time_limit", "fast",
                                      "allow_replacement", "depot_capacity",
                                      "gurobi_params") if k in data},
        raw=data,
    )

    _validate_cells(cfg)
    return cfg


def _infer_L(data: dict, F: int) -> int:
    if "L" in data:
        return int(data["L"])
    m = data["model"]
    if isinstance(m, str):
        return 1
    arr = np.array(m, dtype=object)
    if arr.ndim == 1:
        return arr.shape[0]                      # length-L list
    if arr.ndim == 2:
        return arr.shape[1]                      # (F, L)
    raise ValueError("cannot infer L from 'model'; give a top-level 'L'.")


# ---------------------------------------------------------------------------
# Cell-wise cross-field validation
# ---------------------------------------------------------------------------
def _validate_cells(cfg: FleetConfig) -> None:
    F, L = cfg.F, cfg.L

    def has(arr, i, l):
        return arr is not None and np.all(np.isfinite(arr[i, l]))

    for i in range(F):
        for l in range(L):
            m = str(cfg.model[i, l])
            where = f"cell (i={i}, l={l}) model={m}"

            if not (cfg.tau[i, l] > 0):
                raise ValueError(f"{where}: tau must be > 0.")
            if not (0.0 < cfg.epsilon[i, l] < 1.0):
                raise ValueError(f"{where}: epsilon must be in (0, 1).")
            if not (0.0 < cfg.rho[i, l] <= 1.0):
                raise ValueError(f"{where}: rho must be in (0, 1].")
            if not np.all(cfg.mu[i, l] > 0):
                raise ValueError(f"{where}: mu must be positive.")

            if m == "rainflow":
                b = str(cfg.bound_method[i, l])
                if b not in RAINFLOW_BOUNDS:
                    raise ValueError(f"{where}: bound_method {b!r} not in {RAINFLOW_BOUNDS}.")
                if str(cfg.repair_model[i, l]) not in REPAIR_MODELS:
                    raise ValueError(f"{where}: repair_model must be in {REPAIR_MODELS}.")
                if not has(cfg.v, i, l) or not np.all(cfg.v[i, l] > 0):
                    raise ValueError(f"{where}: rainflow needs positive 'v' at this cell.")
                if cfg.v_0 is None or not (cfg.v_0[i, l] >= 0):
                    raise ValueError(f"{where}: rainflow needs 'v_0' >= 0 at this cell.")
                if b in _NEEDS_SUPPORT and (not has(cfg.support, i, l) or not np.all(cfg.support[i, l] > 0)):
                    raise ValueError(f"{where}: bound '{b}' needs positive 'support'.")
                if b in _NEEDS_CGF:
                    if not has(cfg.cgf, i, l) or not np.all(cfg.cgf[i, l] > 0):
                        raise ValueError(f"{where}: bound 'chernoff' needs positive 'cgf'.")
                    if cfg.s_chernoff is None or not (cfg.s_chernoff[i, l] > 0):
                        raise ValueError(f"{where}: bound 'chernoff' needs 's_chernoff' > 0.")

            elif m == "gamma":
                if cfg.gamma_beta is None or not (cfg.gamma_beta[i, l] > 0):
                    raise ValueError(f"{where}: gamma needs 'gamma_beta' > 0 at this cell.")

            # gaussian / inverse_gaussian: reserved (handled by their builders)