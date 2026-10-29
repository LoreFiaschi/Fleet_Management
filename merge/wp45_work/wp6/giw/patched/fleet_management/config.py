"""
Fleet-management input: reader, key checks, broadcasting and consistency checks.

``read_input`` reads one input file (YAML, JSON or HDF5; spec section "Input File
Schema") into a mapping.  ``load_config`` turns a mapping into a ``FleetConfig``:

* The keys are checked first: rejected, unknown, required (spec sections "Rejected
  Keys" and "Input Specification").
* Each selector, cell value and profile is broadcast to its canonical shape, (F, L)
  or (F, L, M, H), and read only at the cells where its key applies (spec sections
  "Input Specification" and "Increment Profiles and the Phase Lookup").
* The checks of the spec section "Consistency Checks" run, then the derived fields
  are set: the Bernstein support constant and the natural ranges of the states.

Every error is a ``ValueError`` that names the key and, for a cell value, the cell.
An input that is not a mapping raises ``TypeError``; a value that is ignored raises
a ``UserWarning``.  The module imports only the standard library, numpy and scipy
(``yaml`` and ``h5py`` are imported inside the reader), so it runs without Gurobi.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import brentq
from scipy.special import log_ndtr, ndtri


# ---------------------------------------------------------------------------
# Keys and options
# ---------------------------------------------------------------------------
SUPPORTED_MODELS = ("gaussian", "inverse_gaussian", "wiener", "gamma", "rainflow")
RAINFLOW_BOUNDS = ("markov", "cantelli", "hoeffding", "bernstein", "chernoff")
REPAIR_MODELS = ("ardinf", "ard1")
RELIABILITY_IMPLS = ("exact", "tangent", "pwl")
ENCODINGS = ("bigm", "indicator")
PENALTY_TYPES = ("inf_norm", "quadratic")
OBJECTIVE_MODES = ("total", "operating_average", "evaluation_total")

REQUIRED_KEYS = ("F", "M", "H", "model", "tau", "epsilon", "rho", "mu_0", "mu",
                 "C_M", "C_R", "C_rep", "C_D")

RUN_OPTION_DEFAULTS = {
    "verbose": 1,
    "mip_gap": 0.1,
    "time_limit": 3600.0,
    "fast": False,
    "gurobi_params": None,
    "objective_mode": "total",
    "evaluation_horizon": None,
    "progress_interval_seconds": 5.0,
    "relaxation_warm_start": False,
    "relaxation_time_limit": 60.0,
    "validation": None,
}

# The keys of the specification tables, in table order.
CANONICAL_KEYS = (
    "F", "M", "L", "H",
    "model", "repair_model", "bound_method", "reliability_impl", "encoding",
    "component_names",
    "tau", "epsilon", "tangent_ref", "pwl_points",
    "rho", "anchor_mu_0", "anchor_v_0", "anchor_R_0", "anchor_K_0",
    "mu_0", "v_0", "support_0", "R_0", "cgf_0",
    "replacement_mu", "replacement_v", "support_new", "R_new", "cgf_new",
    "replacement_as_new", "allow_replacement",
    "mu", "v", "support", "cgf", "s_chernoff", "gamma_beta",
    "mu_trans", "v_trans", "support_trans", "cgf_trans", "gamma_beta_trans",
    "sigma", "eta", "v_max_user", "gamma_beta_0", "gamma_beta_new",
    "C_M", "C_R", "C_rep", "C_D", "penalty_type",
) + tuple(RUN_OPTION_DEFAULTS)

# Rejected keys with the message of the specification, section "Rejected Keys".
_REJECTED_ROWS = (
    (("maintenance_type",),
     "Use 'repair_model'. Its values are 'ardinf' (the repair reduces the whole level) and "
     "'ard1' (it reduces the level accumulated since the last intervention). In a file written "
     "with 'maintenance_type', the value 'ARD1' corresponds to 'ardinf', and the value 'ARA1' "
     "to 'ard1'."),
    (("tail_bound", "method"),
     "Use 'bound_method'; its values are 'markov', 'cantelli', 'hoeffding', 'bernstein' and "
     "'chernoff'."),
    (("formulation",),
     "Use 'reliability_impl'. The value 'exact' keeps its meaning. The value 'lp' corresponds "
     "to 'tangent' with 'tangent_ref' equal to 0."),
    (("beta",),
     "Use 'gamma_beta'. It is a rate, and 'beta' is a scale: 'gamma_beta' equals 1/'beta'."),
    (("alpha_inc",),
     "The Gamma increments are given by the mean 'mu' and the rate 'gamma_beta'. The shape of "
     "an increment is 'mu' times 'gamma_beta'."),
    (("gamma_beta_bound",),
     "Remove it. The formulation computes the smallest rate of each cell from the rates of "
     "the cell."),
    (("gamma_calibration_method",),
     "Remove it. The formulation has no calibration step."),
    (("b_inc",), "Use 'support': the width of the range of the increment."),
    (("b_0",), "Use 'support_0': the width of the range of the initial damage."),
    (("b_new",),
     "Use 'support_new': the width of the range of the damage after a replacement."),
    (("n_workers", "warm_start"),
     "Remove them. Parallel runs and warm starts between horizons are options of the sweep "
     "functions."),
    (("depot_capacity",),
     "Remove it. The availability of the depot is not limited at fleet level."),
    (("C_P",), "Remove it. The loop constraint is a hard constraint and has no penalty."),
    (("C_S",), "Use 'C_D', with the same meaning."),
    (("alpha",), "Use 'tau', with the same meaning."),
    (("xi", "repair_rho"), "Use 'rho', with the same meaning."),
    (("mu_new",), "Use 'replacement_mu', with the same meaning."),
    (("v_new",), "Use 'replacement_v', with the same meaning."),
)
REJECTED_KEYS = {key: message for keys, message in _REJECTED_ROWS for key in keys}


def _check_keys(data) -> None:
    """Check the keys of an input: rejected, unknown, required (before any array is read)."""
    if not isinstance(data, dict):
        raise TypeError("input data must be a mapping (dict).")
    for key in data:
        if key in REJECTED_KEYS:
            raise ValueError(f"key '{key}' is not accepted. {REJECTED_KEYS[key]}")
    unknown = sorted(str(key) for key in data if key not in CANONICAL_KEYS)
    if unknown:
        raise ValueError(f"unknown key(s) {unknown}; "
                         f"the canonical keys are: {', '.join(CANONICAL_KEYS)}.")
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise ValueError(f"missing required key(s): {', '.join(repr(k) for k in missing)}.")


# ---------------------------------------------------------------------------
# Normalized representation
# ---------------------------------------------------------------------------
@dataclass
class FleetConfig:
    """A checked input in the canonical shapes of the specification.

    Selectors are (F, L) object arrays of str, '' where the key does not apply.
    Cell values are (F, L) float64 arrays, NaN where the key does not apply.
    Operating profiles are (F, L, M, H2) arrays and transitory profiles
    (F, L, M, H1), expanded eagerly; a profile is None if its key is absent.
    """
    F: int
    M: int
    L: int
    H: object                       # int or [H1, H2] as written
    H1: int
    H2: int
    T: int
    component_names: list = field(default_factory=list)
    # selectors (F, L)
    model: np.ndarray = None
    repair_model: np.ndarray = None
    bound_method: np.ndarray = None
    reliability_impl: np.ndarray = None  # form that is built: 'exact' at a linear row
    encoding: np.ndarray = None
    # cell values (F, L)
    tau: np.ndarray = None
    epsilon: np.ndarray = None
    tangent_ref: np.ndarray = None
    rho: np.ndarray = None
    anchor_mu_0: np.ndarray = None
    anchor_v_0: np.ndarray = None
    anchor_R_0: np.ndarray = None
    anchor_K_0: np.ndarray = None
    mu_0: np.ndarray = None
    v_0: np.ndarray = None
    support_0: np.ndarray = None
    R_0: np.ndarray = None
    cgf_0: np.ndarray = None
    replacement_mu: np.ndarray = None
    replacement_v: np.ndarray = None
    support_new: np.ndarray = None
    R_new: np.ndarray = None
    cgf_new: np.ndarray = None
    s_chernoff: np.ndarray = None
    sigma: np.ndarray = None
    eta: np.ndarray = None
    v_max_user: np.ndarray = None   # NaN where not given
    gamma_beta_0: np.ndarray = None   # NaN where not given
    gamma_beta_new: np.ndarray = None  # NaN where not given
    C_R: np.ndarray = None
    C_rep: np.ndarray = None
    # scalars
    pwl_points: int = None
    allow_replacement: bool = None
    replacement_as_new: bool = None
    C_M: float = None
    C_D: float = None
    penalty_type: str = None
    # operating profiles (F, L, M, H2)
    mu: Optional[np.ndarray] = None
    v: Optional[np.ndarray] = None
    support: Optional[np.ndarray] = None
    cgf: Optional[np.ndarray] = None
    gamma_beta: Optional[np.ndarray] = None
    # transitory profiles (F, L, M, H1)
    mu_trans: Optional[np.ndarray] = None
    v_trans: Optional[np.ndarray] = None
    support_trans: Optional[np.ndarray] = None
    cgf_trans: Optional[np.ndarray] = None
    gamma_beta_trans: Optional[np.ndarray] = None
    options: dict = field(default_factory=dict)   # the run options, defaults filled
    # derived (F, L), set by _derive
    support_const: Optional[np.ndarray] = None    # Bernstein support constant b
    range_mu: Optional[np.ndarray] = None         # natural ranges of the states
    range_v: Optional[np.ndarray] = None
    range_R: Optional[np.ndarray] = None
    range_K: Optional[np.ndarray] = None

    @property
    def models(self) -> list:
        return sorted(set(self.model.ravel().tolist())) if self.model is not None else []


# ---------------------------------------------------------------------------
# Scalars
# ---------------------------------------------------------------------------
def _integer(value, name: str) -> int:
    """An integer; a float with an integer value is accepted, a bool is not."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"'{name}' must be an integer (got {value!r}).")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    raise ValueError(f"'{name}' must be an integer (got {value!r}).")


def _number(value, name: str) -> float:
    """A real number (not a bool)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)):
        raise ValueError(f"'{name}' must be a number (got {value!r}).")
    return float(value)


def _flag(value, name: str) -> bool:
    """``true`` or ``false``."""
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"'{name}' must be true or false (got {value!r}).")
    return bool(value)


# ---------------------------------------------------------------------------
# Sizes and names
# ---------------------------------------------------------------------------
def _parse_horizon(H_raw):
    if isinstance(H_raw, (list, tuple)):
        if len(H_raw) != 2:
            raise ValueError("'H' must be an integer or a pair [H1, H2].")
        H1, H2 = _integer(H_raw[0], "H1"), _integer(H_raw[1], "H2")
        if H1 < 1 or H2 < 1:
            raise ValueError(f"'H' entries must be >= 1 (got {H1}, {H2}).")
        return [H1, H2], H1, H2, H1 + H2
    H = _integer(H_raw, "H")
    if H < 1:
        raise ValueError(f"'H' must be >= 1 (got {H}).")
    return H, H, H, 2 * H


def _infer_L(data: dict, F: int) -> int:
    L_given = _integer(data["L"], "L") if "L" in data else None
    m = data["model"]
    if isinstance(m, str):
        L = 1 if L_given is None else L_given
    else:
        arr = np.array(m, dtype=object)
        if arr.ndim == 1:
            L = arr.shape[0]                     # length-L list
        elif arr.ndim == 2:
            L = arr.shape[1]                     # (F, L)
        else:
            raise ValueError("'model' must be a string, a list of L strings or an F x L array.")
        if L_given is not None and L_given != L:
            raise ValueError(f"'L'={L_given} differs from the component axis of 'model' ({L}).")
    if L < 1:
        raise ValueError("'L' must be >= 1.")
    return L


def _component_names(value, L: int) -> list[str]:
    """Return one stable, human-readable name for each component index."""
    if value is None:
        return [f"Component {index + 1}" for index in range(L)]
    if not isinstance(value, (list, tuple)):
        raise ValueError("'component_names' must be a list of L strings.")
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"'component_names' entries must be strings (got {item!r}).")
    names = list(value)
    if len(names) != L:
        raise ValueError(
            f"'component_names' has length {len(names)}; expected L={L}"
        )
    names = [name.strip() for name in names]
    if any(not name for name in names):
        raise ValueError("'component_names' entries must be nonempty")
    return names


# ---------------------------------------------------------------------------
# Run options
# ---------------------------------------------------------------------------
def _run_options(data: dict, H1: int) -> dict:
    """Run options with their defaults filled and checked (spec "Run Options")."""
    options = dict(RUN_OPTION_DEFAULTS)
    options.update({key: data[key] for key in RUN_OPTION_DEFAULTS if key in data})

    def fail(key, rule):
        raise ValueError(f"'{key}' must be {rule} (got {options[key]!r}).")

    if isinstance(options["verbose"], (bool, np.bool_)):
        options["verbose"] = int(options["verbose"])
    options["verbose"] = _integer(options["verbose"], "verbose")
    if options["verbose"] not in (0, 1):
        fail("verbose", "0 or 1")
    options["mip_gap"] = _number(options["mip_gap"], "mip_gap")
    if not _le(0.0, options["mip_gap"]):
        fail("mip_gap", "a number >= 0")
    options["mip_gap"] = max(options["mip_gap"], 0.0)   # Gurobi refuses a MIPGap below 0
    for key in ("time_limit", "progress_interval_seconds", "relaxation_time_limit"):
        options[key] = _number(options[key], key)
        if not options[key] > 0:
            fail(key, "a number > 0")
    for key in ("fast", "relaxation_warm_start"):
        options[key] = _flag(options[key], key)
    for key in ("gurobi_params", "validation"):
        if options[key] is not None and not isinstance(options[key], dict):
            fail(key, "a mapping")
    if options["objective_mode"] not in OBJECTIVE_MODES:
        fail("objective_mode", f"one of {OBJECTIVE_MODES}")
    E = options["evaluation_horizon"]
    if E is not None:
        options["evaluation_horizon"] = _integer(E, "evaluation_horizon")
        if options["evaluation_horizon"] <= H1:
            fail("evaluation_horizon", f"an integer > H1 = {H1}")
    elif options["objective_mode"] == "evaluation_total":
        fail("evaluation_horizon", "given when objective_mode is 'evaluation_total'")
    return options


# ---------------------------------------------------------------------------
# Cells, selectors and profiles
# ---------------------------------------------------------------------------
def _first_cell(mask: np.ndarray) -> tuple[int, int]:
    """Indices (i, l) of the first true entry of an (F, L) mask."""
    i, l = np.argwhere(mask)[0]
    return int(i), int(l)


def _holds_text_or_flag(value) -> bool:
    """True if ``value`` is, or contains, a string or a flag (``true`` or ``false``)."""
    if isinstance(value, (str, bytes, bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        return value.dtype.kind in "USb" or (
            value.dtype.kind == "O" and any(_holds_text_or_flag(x) for x in value.ravel()))
    if isinstance(value, (list, tuple)):
        return any(_holds_text_or_flag(x) for x in value)
    return False


def _float_array(raw, name: str) -> np.ndarray:
    """A number or a regular array of numbers as float64 (``None`` becomes NaN).

    A string or a flag is refused, also "0.5" or ``true``: numpy would convert both.
    """
    if _holds_text_or_flag(raw):
        raise ValueError(f"'{name}' must be a number or a regular array of numbers.")
    try:
        return np.asarray(raw, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"'{name}' must be a number or a regular array of numbers.") from None


def _warn_not_applicable(name: str, values: np.ndarray, applies: np.ndarray) -> None:
    """Warn if ``values`` (axes F, L, ...) has entries at the cells where the key does not apply."""
    if values.dtype == object:                      # selector: null is None or ''
        given = np.vectorize(lambda x: x is not None and x != "", otypes=[bool])(values)
    else:
        given = ~np.isnan(values)
    given = given.reshape(given.shape[0], given.shape[1], -1).any(axis=2)
    cells = given & ~applies
    if cells.any():
        i, l = _first_cell(cells)
        warnings.warn(
            f"'{name}' has values at {int(cells.sum())} cell(s) where it does not apply, "
            f"for example cell (i={i}, l={l}); they are ignored.",
            UserWarning, stacklevel=3)


def _selector(data: dict, key: str, F: int, L: int, applies: np.ndarray, default,
              allowed: tuple, fill: str = "") -> np.ndarray:
    """A selector as an (F, L) object array: its entry where it applies, ``fill`` elsewhere."""
    per_cell = False                                # entries given cell by cell (axes F, L)
    if key in data:
        arr = np.array(data[key], dtype=object)
        if arr.shape == ():
            entries = np.full((F, L), arr.item(), dtype=object)
        elif arr.shape == (L,):                     # per component, tiled over F
            entries = np.broadcast_to(arr[np.newaxis, :], (F, L)).copy()
        elif arr.shape == (F, L):
            entries, per_cell = arr, True
        else:
            raise ValueError(f"'{key}' shape {arr.shape} must be a string, ({L},), or ({F},{L}).")
    elif default is not None:
        entries = np.full((F, L), default, dtype=object)
    elif applies.any():
        i, l = _first_cell(applies)
        raise ValueError(f"missing required key '{key}', needed at cell (i={i}, l={l}).")
    else:
        entries = np.full((F, L), None, dtype=object)
    null = np.vectorize(lambda x: x is None or x == "", otypes=[bool])(entries)
    listed = np.vectorize(lambda x: x in allowed, otypes=[bool])(entries)
    if (~null & ~listed).any():
        i, l = _first_cell(~null & ~listed)
        raise ValueError(f"'{key}' has the value {entries[i, l]!r} at cell (i={i}, l={l}); "
                         f"allowed: {allowed}.")
    if (null & applies).any():
        i, l = _first_cell(null & applies)
        raise ValueError(f"'{key}' is null at cell (i={i}, l={l}), where it applies.")
    if per_cell:
        _warn_not_applicable(key, entries, applies)
    out = np.full((F, L), fill, dtype=object)
    out[applies] = entries[applies]
    return out


def _masks(model: np.ndarray, bound_method: np.ndarray, repair_model: np.ndarray,
           reliability_impl: Optional[np.ndarray] = None) -> dict:
    """(F, L) boolean masks of the cell types; the keys are the names of the specification."""
    m = {"all": np.ones(model.shape, dtype=bool)}
    for name in SUPPORTED_MODELS:
        m[name] = model == name
    for bound in RAINFLOW_BOUNDS:
        m[bound] = m["rainflow"] & (bound_method == bound)
    m["ard1"] = repair_model == "ard1"
    m["v_profile"] = m["gaussian"] | m["cantelli"] | m["bernstein"]
    m["variance"] = m["v_profile"] | m["wiener"]
    m["variance_anchor"] = m["ard1"] & m["v_profile"]
    m["support"] = m["hoeffding"] | m["bernstein"]
    m["quadratic"] = m["gaussian"] | m["wiener"] | m["cantelli"] | m["support"]
    if reliability_impl is not None:
        m["tangent"] = reliability_impl == "tangent"
        m["pwl"] = reliability_impl == "pwl"
    return m


def _cell_value(data: dict, key: str, F: int, L: int, applies: np.ndarray, default=None,
                optional: bool = False) -> np.ndarray:
    """A cell value as (F, L) float64: its entry where it applies, NaN elsewhere.

    ``default`` is a number or an (F, L) array. With ``optional``, an absent key and a
    ``null`` entry both mean "not given" and give NaN.
    """
    out = np.full((F, L), np.nan)
    if key not in data:
        if optional or not applies.any():
            return out
        if default is None:
            i, l = _first_cell(applies)
            raise ValueError(f"missing required key '{key}', needed at cell (i={i}, l={l}).")
        out[applies] = np.broadcast_to(default, (F, L))[applies]
        return out
    arr = _float_array(data[key], key)
    if arr.ndim == 0:
        values = np.full((F, L), float(arr))
    elif arr.shape == (L,):                         # per component, tiled over F
        values = np.broadcast_to(arr[np.newaxis, :], (F, L))
    elif arr.shape == (F, L):
        values = arr
        _warn_not_applicable(key, arr, applies)
    else:
        raise ValueError(f"'{key}' shape {arr.shape} must be scalar, ({L},), or ({F},{L}).")
    if not optional and (applies & np.isnan(values)).any():
        i, l = _first_cell(applies & np.isnan(values))
        raise ValueError(f"'{key}' is null at cell (i={i}, l={l}), where it applies.")
    if (applies & np.isinf(values)).any():
        i, l = _first_cell(applies & np.isinf(values))
        raise ValueError(f"'{key}' must be a finite number at cell (i={i}, l={l}).")
    out[applies] = values[applies]
    return out


def _profile(data: dict, key: str, F: int, L: int, M: int, Hs: int, applies: np.ndarray,
             rates: bool = False, required: bool = False) -> Optional[np.ndarray]:
    """A profile as (F, L, M, Hs) float64, expanded eagerly: its entries where it applies,
    NaN elsewhere; ``None`` if the key is absent (an error if ``required`` and a cell applies).

    ``rates`` selects the shapes of ``gamma_beta`` (one rate per component or per cell).
    """
    if key not in data:
        if required and applies.any():
            i, l = _first_cell(applies)
            raise ValueError(f"missing required key '{key}', needed at cell (i={i}, l={l}).")
        return None
    arr = _float_array(data[key], key)
    shape = arr.shape
    per_cell = False                                # entries given cell by cell (axes F, L)
    if shape == ():
        view = arr.reshape(1, 1, 1, 1)
    elif rates and shape == (L,):
        view = arr[None, :, None, None]
    elif rates and shape == (F, L):
        view, per_cell = arr[:, :, None, None], True
    elif not rates and shape == (M,):
        view = arr[None, None, :, None]
    elif not rates and shape == (L, M):
        view = arr[None, :, :, None]
    elif not rates and shape == (L, M, Hs):
        view = arr[None, :, :, :]
    elif shape == (F, L, M):
        view, per_cell = arr[:, :, :, None], True
    elif shape == (F, L, M, Hs):
        view, per_cell = arr, True
    else:
        accepted = (f"({L},), ({F},{L}), ({F},{L},{M}), or ({F},{L},{M},{Hs})" if rates else
                    f"({M},), ({L},{M}), ({L},{M},{Hs}), ({F},{L},{M}), or ({F},{L},{M},{Hs})")
        raise ValueError(f"'{key}' shape {shape} must be scalar, {accepted}.")
    values = np.broadcast_to(view, (F, L, M, Hs)).copy()
    if per_cell:
        _warn_not_applicable(key, arr, applies)
    if (applies & np.isnan(values).any(axis=(2, 3))).any():
        i, l = _first_cell(applies & np.isnan(values).any(axis=(2, 3)))
        raise ValueError(f"'{key}' is null at cell (i={i}, l={l}), where it applies.")
    if (applies & np.isinf(values).any(axis=(2, 3))).any():
        i, l = _first_cell(applies & np.isinf(values).any(axis=(2, 3)))
        raise ValueError(f"'{key}' must be a finite number at cell (i={i}, l={l}).")
    return np.where(applies[:, :, None, None], values, np.nan)


# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
def _ig_unit_log_cdf(x: float, eta: float) -> float:
    """ln F(x) of Z ~ IG(1, 1/eta), x > 0 (spec, "Inverse Gaussian Model")."""
    r = 1.0 / math.sqrt(eta * x)
    return float(np.logaddexp(log_ndtr(r * (x - 1.0)), 2.0 / eta + log_ndtr(-r * (x + 1.0))))


def _ig_unit_log_sf(x: float, eta: float) -> float:
    """ln (1 - F(x)) of Z ~ IG(1, 1/eta), x > 0, without the cancellation of 1 - F."""
    r = 1.0 / math.sqrt(eta * x)
    first = float(log_ndtr(-r * (x - 1.0)))
    second = 2.0 / eta + float(log_ndtr(-r * (x + 1.0)))
    return first + math.log1p(-math.exp(second - first))


def _ig_unit_quantile(eta: float, epsilon: float) -> float:
    """The (1 - epsilon)-quantile q of Z ~ IG(1, 1/eta): Pr(Z > q) = epsilon.

    Root of ln Pr(Z > e^u) = ln epsilon in u = ln q, by brentq on [-50, ln b], with
    b = 1 + sqrt(eta (2/epsilon - 1)): Var Z = eta, so Pr(Z > b) <= epsilon/2 (Cantelli).
    """
    target = math.log(epsilon)

    def gap(u):
        return _ig_unit_log_sf(math.exp(u), eta) - target

    lo, hi = -50.0, math.log(1.0 + math.sqrt(eta * (2.0 / epsilon - 1.0)))
    if not (gap(lo) > 0.0 > gap(hi)):
        raise ValueError(f"no quantile bracket for eta={eta}, epsilon={epsilon}")
    return math.exp(brentq(gap, lo, hi, xtol=1e-14, rtol=1e-15, maxiter=200))


def ig_mean_cap(eta: float, tau: float, epsilon: float) -> float:
    """Mean cap mu_bar = tau / E[Z | Z > q] of an inverse Gaussian cell (spec, Eq. ig_row).

    E[Z; Z > q] = F(1/q), so mu_bar = tau * epsilon / F(1/q), computed in log space.
    The builder ``degradation_model/inverse_gaussian.py`` imports it; it lives here
    because the check of the replacement state needs it without Gurobi.
    """
    q = _ig_unit_quantile(eta, epsilon)
    return tau * math.exp(math.log(epsilon) - _ig_unit_log_cdf(1.0 / q, eta))


# ---------------------------------------------------------------------------
# Consistency checks
# ---------------------------------------------------------------------------
_TOL = 1e-12


def _le(a, b) -> np.ndarray:
    """``a <= b`` up to ``1e-12 * max(1, |a|, |b|)`` (spec "Consistency Checks").

    Every non-strict inequality uses it, also against a constant such as 0 or 1;
    strict inequalities and equalities are exact.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a <= b + _TOL * np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))


def _require(cfg: FleetConfig, ok: np.ndarray, key: str, rule: str) -> None:
    """Raise at the first cell where ``ok`` is false (``ok`` is true where the check is void)."""
    if not ok.all():
        i, l = _first_cell(~ok)
        raise ValueError(f"cell (i={i}, l={l}) model={cfg.model[i, l]}: '{key}' {rule}.")


def _check_state(cfg: FleetConfig) -> None:
    """Reliability level, repair, initial and replacement states, starting anchors, costs."""
    m = _masks(cfg.model, cfg.bound_method, cfg.repair_model, cfg.reliability_impl)

    def check(applies, ok, key, rule):
        _require(cfg, ~applies | ok, key, rule)

    with np.errstate(invalid="ignore"):
        check(m["all"], cfg.tau > 0, "tau", "must be > 0")
        check(m["all"], (cfg.epsilon > 0) & (cfg.epsilon < 1), "epsilon", "must be in (0, 1)")
        check(m["gaussian"] | m["wiener"], cfg.epsilon < 0.5, "epsilon",
              "must be < 0.5 at a gaussian or wiener cell")
        check(m["all"], (cfg.rho > 0) & _le(cfg.rho, 1.0), "rho", "must be in (0, 1]")
        check(m["all"], _le(0.0, cfg.mu_0) & (cfg.mu_0 < cfg.tau), "mu_0",
              "must satisfy 0 <= mu_0 < tau")
        for key, applies in (("v_0", m["variance"]), ("support_0", m["bernstein"]),
                             ("R_0", m["hoeffding"])):
            check(applies, _le(0.0, getattr(cfg, key)), key, "must be >= 0")
        if cfg.allow_replacement:
            for key, applies in (("replacement_mu", m["all"]), ("replacement_v", m["variance"]),
                                 ("support_new", m["bernstein"]), ("R_new", m["hoeffding"])):
                check(applies, _le(0.0, getattr(cfg, key)), key,
                      "must be >= 0 when allow_replacement is true")
        check(m["tangent"], _le(0.0, cfg.tangent_ref) & (cfg.tangent_ref < 1), "tangent_ref",
              "must be in [0, 1) at a cell with the form tangent")
        for key, initial, applies in (
                ("anchor_mu_0", "mu_0", m["ard1"]),
                ("anchor_v_0", "v_0", m["variance_anchor"]),
                ("anchor_R_0", "R_0", m["ard1"] & m["hoeffding"]),
                ("anchor_K_0", "cgf_0", m["ard1"] & m["chernoff"])):
            anchor = getattr(cfg, key)
            check(applies, _le(0.0, anchor) & _le(anchor, getattr(cfg, initial)), key,
                  f"must satisfy 0 <= {key} <= {initial}")
        if cfg.replacement_as_new:
            for key, applies in (("replacement_mu", m["all"]), ("replacement_v", m["variance"]),
                                 ("support_new", m["bernstein"]), ("R_new", m["hoeffding"]),
                                 ("cgf_new", m["chernoff"])):
                check(applies, getattr(cfg, key) == 0, key,
                      "must be 0 when replacement_as_new is true")
        check(m["all"], _le(0.0, cfg.C_R), "C_R", "must be >= 0")
        check(m["all"], _le(0.0, cfg.C_rep), "C_rep", "must be >= 0")
        if cfg.C_D > 0 and cfg.penalty_type == "quadratic":
            check(m["quadratic"], cfg.reliability_impl == "exact", "reliability_impl",
                  "must be 'exact' when penalty_type is 'quadratic' and C_D > 0")


def _lookup(op: np.ndarray, trans: Optional[np.ndarray], H1: int, H2: int) -> np.ndarray:
    """Profile on the days 1..T = H1 + H2 (spec, Eq. ``c_phase``): (F, L, M, T)."""
    out = op[..., np.concatenate([np.arange(H1) % H2, np.arange(H2)])]
    if trans is not None:
        out[..., :H1] = trans
    return out


def _check_increments(cfg: FleetConfig) -> None:
    """Checks of the increment profiles: entry by entry, then after the phase lookup."""
    m = _masks(cfg.model, cfg.bound_method, cfg.repair_model, cfg.reliability_impl)
    tau = cfg.tau[:, :, None, None]
    spread = {"v": m["v_profile"], "support": m["support"], "cgf": m["chernoff"]}

    def check(applies, ok, key, rule):
        _require(cfg, ~applies | ok.all(axis=(2, 3)), key, rule)

    def lookup(key):
        return _lookup(getattr(cfg, key), getattr(cfg, f"{key}_trans"), cfg.H1, cfg.H2)

    with np.errstate(invalid="ignore"):
        for key in ("mu", "mu_trans"):
            profile = getattr(cfg, key)
            if profile is not None:
                check(m["all"], _le(0.0, profile) & (profile < tau), key,
                      "must satisfy 0 <= mu < tau for every mission and day")
        for key, applies in spread.items():
            for name in (key, f"{key}_trans"):
                if getattr(cfg, name) is not None:
                    check(applies, _le(0.0, getattr(cfg, name)), name,
                          "must be >= 0 for every mission and day")
        mu = lookup("mu")
        for key, applies in spread.items():
            if getattr(cfg, key) is not None:
                check(applies, (mu != 0) | (lookup(key) == 0), key,
                      "must be 0 where the mean increment 'mu' is 0")
        if cfg.v is not None:
            v = lookup("v")
            # A variance within the tolerance below 0 counts as 0 under the square root.
            check(m["gaussian"], _le(3.0 * np.sqrt(np.maximum(v, 0.0)), mu), "v",
                  "must satisfy mu >= 3*sqrt(v) for every mission and day")
            if cfg.support is not None:
                check(m["bernstein"], _le(v, lookup("support") ** 2 / 4.0), "v",
                      "must satisfy v <= support^2/4 for every mission and day")
        if cfg.cgf is not None:
            check(m["chernoff"], _le(cfg.s_chernoff[:, :, None, None] * mu, lookup("cgf")), "cgf",
                  "must satisfy cgf >= s_chernoff*mu for every mission and day")


def _derive(cfg: FleetConfig) -> None:
    """Set the Bernstein support constant and the natural ranges of the states."""
    m = _masks(cfg.model, cfg.bound_method, cfg.repair_model, cfg.reliability_impl)
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.log(1.0 / cfg.epsilon)
        kappa_g = ndtri(1.0 - cfg.epsilon)
        b = np.nan_to_num(cfg.support_0, nan=0.0)
        for profile in (cfg.support, cfg.support_trans):
            if profile is not None:
                b = np.maximum(b, np.nan_to_num(profile, nan=0.0).max(axis=(2, 3)))
        if cfg.allow_replacement:
            b = np.maximum(b, np.nan_to_num(cfg.support_new, nan=0.0))
        cfg.support_const = np.where(m["bernstein"], b, np.nan)
        c = kappa * cfg.support_const / 3.0
        tau, eps = cfg.tau, cfg.epsilon
        cfg.range_mu = tau.copy()
        cfg.range_v = np.select(
            [m["gaussian"] | m["wiener"], m["cantelli"], m["bernstein"]],
            [tau ** 2 / kappa_g ** 2, eps * tau ** 2 / (1.0 - eps),
             tau * (tau - 2.0 * c) / (2.0 * kappa)], default=np.nan)
        cfg.range_R = np.where(m["hoeffding"], 2.0 * tau ** 2 / kappa, np.nan)
        cfg.range_K = np.where(m["chernoff"], cfg.s_chernoff * tau - kappa, np.nan)


def _check_models(cfg: FleetConfig) -> None:
    """Checks of one model at a time: Gaussian, inverse Gaussian, Wiener, Gamma and rainflow."""
    m = _masks(cfg.model, cfg.bound_method, cfg.repair_model, cfg.reliability_impl)
    R, s, tau = cfg.allow_replacement, cfg.s_chernoff, cfg.tau

    def check(applies, ok, key, rule):
        _require(cfg, ~applies | ok, key, rule)

    def check_profile(applies, key):
        if getattr(cfg, key) is not None:
            check(applies, (getattr(cfg, key) > 0).all(axis=(2, 3)), key,
                  "must be > 0 for every mission and day")

    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.log(1.0 / cfg.epsilon)
        c = kappa * cfg.support_const / 3.0
        # A variance within the tolerance below 0 counts as 0 under the square root.
        check(m["gaussian"], _le(3.0 * np.sqrt(np.maximum(cfg.v_0, 0.0)), cfg.mu_0), "mu_0",
              "must be >= 3*sqrt(v_0) at a gaussian cell")
        if R:
            check(m["gaussian"],
                  _le(3.0 * np.sqrt(np.maximum(cfg.replacement_v, 0.0)), cfg.replacement_mu),
                  "replacement_mu", "must be >= 3*sqrt(replacement_v) at a gaussian cell")
        check(m["inverse_gaussian"], cfg.eta > 0, "eta", "must be > 0")
        check(m["wiener"], cfg.sigma > 0, "sigma", "must be > 0")
        check_profile(m["gamma"], "gamma_beta")
        check_profile(m["gamma"], "gamma_beta_trans")
        for key in ("gamma_beta_0", "gamma_beta_new"):
            check(m["gamma"] & ~np.isnan(getattr(cfg, key)), getattr(cfg, key) > 0, key,
                  "must be > 0 where given")
        given = m["variance"] & ~np.isnan(cfg.v_max_user)
        check(given, cfg.v_max_user > 0, "v_max_user", "must be > 0")
        check(given, _le(cfg.v_0, cfg.v_max_user), "v_max_user", "must be >= v_0")
        if R:
            check(given, _le(cfg.replacement_v, cfg.v_max_user), "v_max_user",
                  "must be >= replacement_v")
        check(m["bernstein"], _le(cfg.v_0, cfg.support_0 ** 2 / 4.0), "v_0",
              "must be <= support_0^2/4 at a bernstein cell")
        if R:
            check(m["bernstein"], _le(cfg.replacement_v, cfg.support_new ** 2 / 4.0),
                  "replacement_v", "must be <= support_new^2/4 at a bernstein cell")
        check(m["bernstein"], tau > 2.0 * c, "tau",
              "must be > 2*ln(1/epsilon)*b/3, b the support constant of the cell")
        root = np.sqrt(tau * (tau - 2.0 * c))
        check(m["bernstein"] & m["tangent"], cfg.tangent_ref * tau < root, "tangent_ref",
              "must satisfy tangent_ref*tau < sqrt(tau*(tau - 2c)) at a bernstein cell")
        check(m["bernstein"] & m["pwl"], tau / (2.0 * cfg.pwl_points) < root, "pwl_points",
              "must satisfy tau/(2*pwl_points) < sqrt(tau*(tau - 2c)) at a bernstein cell")
        check(m["chernoff"], s > kappa / tau, "s_chernoff", "must be > ln(1/epsilon)/tau")
        check(m["chernoff"], _le(s * cfg.mu_0, cfg.cgf_0), "cgf_0", "must be >= s_chernoff*mu_0")
        check(m["ard1"] & m["chernoff"], _le(s * cfg.anchor_mu_0, cfg.anchor_K_0), "anchor_K_0",
              "must be >= s_chernoff*anchor_mu_0")
        if R:
            check(m["chernoff"], _le(s * cfg.replacement_mu, cfg.cgf_new), "cgf_new",
                  "must be >= s_chernoff*replacement_mu")


def _quad_coefficients(cfg: FleetConfig, m: dict) -> tuple:
    """(chi, c2, c1, side) of the quadratic reliability row at a replacement state.

    Spec, Eq. ``a_quad``: the row is ``chi <= c2 (tau - mu)^2 + c1 (tau - mu)`` with the
    side row ``mu <= side``. NaN at the cells whose row is not quadratic.
    """
    eps, tau, b = cfg.epsilon, cfg.tau, cfg.support_const
    gw = m["gaussian"] | m["wiener"]
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.log(1.0 / eps)
        chi = np.select([gw | m["cantelli"] | m["bernstein"], m["hoeffding"]],
                        [cfg.replacement_v, cfg.R_new], default=np.nan)
        c2 = np.select([gw, m["cantelli"], m["hoeffding"], m["bernstein"]],
                       [1.0 / ndtri(1.0 - eps) ** 2, eps / (1.0 - eps), 2.0 / kappa,
                        1.0 / (2.0 * kappa)], default=np.nan)
        c1 = np.select([gw | m["cantelli"] | m["hoeffding"], m["bernstein"]],
                       [0.0, -b / 3.0], default=np.nan)
        side = np.select([gw | m["cantelli"] | m["hoeffding"], m["bernstein"]],
                         [tau, tau - kappa * b / 3.0], default=np.nan)
    return chi, c2, c1, side


def _cap(mu, tau, c2, c1, impl, tangent_ref, P) -> np.ndarray:
    """Right-hand side of the quadratic row at the mean ``mu``, in the form ``impl``.

    ``pwl``: the largest cap over the segments that contain ``mu`` (-inf if none).
    """
    def f(x):
        return c2 * (tau - x) ** 2 + c1 * (tau - x)

    def fp(x):
        return -2.0 * c2 * (tau - x) - c1

    r = tangent_ref * tau
    tangent = f(r) + fp(r) * (mu - r)
    pwl = np.full(mu.shape, -np.inf)
    for p in range(1, P + 1):
        h = (p - 0.5) * tau / P
        inside = ((p - 1) * tau / P <= mu) & (mu <= p * tau / P)
        pwl = np.where(inside, np.maximum(pwl, f(h) + fp(h) * (mu - h)), pwl)
    return np.where(impl == "tangent", tangent, np.where(impl == "pwl", pwl, f(mu)))


def _check_ranges(cfg: FleetConfig) -> None:
    """Natural ranges of the initial and replacement states; rows of a replacement day."""
    m = _masks(cfg.model, cfg.bound_method, cfg.repair_model, cfg.reliability_impl)
    in_range = "must lie in [0, M], M the natural range of its state at this cell"

    def check(applies, ok, key, rule):
        _require(cfg, ~applies | ok, key, rule)

    with np.errstate(divide="ignore", invalid="ignore"):
        states = [("mu_0", m["all"], "range_mu"), ("v_0", m["variance"], "range_v"),
                  ("R_0", m["hoeffding"], "range_R"), ("cgf_0", m["chernoff"], "range_K")]
        if cfg.allow_replacement:
            states += [("replacement_mu", m["all"], "range_mu"),
                       ("replacement_v", m["variance"], "range_v"),
                       ("R_new", m["hoeffding"], "range_R"),
                       ("cgf_new", m["chernoff"], "range_K")]
        for key, applies, bound in states:
            value = getattr(cfg, key)
            check(applies, _le(0.0, value) & _le(value, getattr(cfg, bound)), key, in_range)
        if not cfg.allow_replacement:
            return
        chi, c2, c1, side = _quad_coefficients(cfg, m)
        mu = cfg.replacement_mu
        check(m["quadratic"], _le(mu, side), "replacement_mu",
              "must satisfy the side row of the reliability row (mu <= tau, or mu <= tau - c "
              "at a bernstein cell)")
        cap = _cap(mu, cfg.tau, c2, c1, cfg.reliability_impl, cfg.tangent_ref, cfg.pwl_points)
        row = ("must satisfy the reliability row on a replacement day, "
               "in the form of reliability_impl")
        check(m["quadratic"] & ~m["hoeffding"], _le(chi, cap), "replacement_v", row)
        check(m["hoeffding"], _le(chi, cap), "R_new", row)
        check(m["markov"], _le(mu, cfg.epsilon * cfg.tau), "replacement_mu",
              "must be <= epsilon*tau at a markov cell")
        ig = m["inverse_gaussian"]
        mu_bar = np.full(mu.shape, np.nan)
        for i, l in zip(*np.nonzero(ig)):
            mu_bar[i, l] = ig_mean_cap(float(cfg.eta[i, l]), float(cfg.tau[i, l]),
                                       float(cfg.epsilon[i, l]))
        check(ig, _le(mu, mu_bar), "replacement_mu",
              "must be <= the mean cap tau/E[Z | Z > q] at an inverse_gaussian cell")


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------
def load_config(data: dict) -> FleetConfig:
    """Check an input mapping and normalize it into a ``FleetConfig``.

    The keys are checked first (rejected, unknown, required), before any array is
    read.  Then come the sizes, the selectors, the cell values, the profiles and the
    run options, and last the checks of the spec section "Consistency Checks": state,
    increments, models, natural ranges and replacement states.  A cell value, a
    selector or a profile is read only at the cells where its key applies.
    """
    _check_keys(data)
    F, M = _integer(data["F"], "F"), _integer(data["M"], "M")
    if F < 1 or M < 1:
        raise ValueError(f"'F' and 'M' must be >= 1 (got F={F}, M={M}).")
    if F <= M:
        raise ValueError(f"'F' must be > 'M' (got F={F}, M={M}).")
    H, H1, H2, T = _parse_horizon(data["H"])
    L = _infer_L(data, F)
    everywhere = np.ones((F, L), dtype=bool)

    model = _selector(data, "model", F, L, everywhere, None, SUPPORTED_MODELS)
    repair_model = _selector(data, "repair_model", F, L, everywhere, "ardinf", REPAIR_MODELS)
    bound_method = _selector(data, "bound_method", F, L, model == "rainflow", "cantelli",
                             RAINFLOW_BOUNDS)
    m = _masks(model, bound_method, repair_model)
    reliability_impl = _selector(data, "reliability_impl", F, L, m["quadratic"], "exact",
                                 RELIABILITY_IMPLS, fill="exact")
    encoding = _selector(data, "encoding", F, L, everywhere, "bigm", ENCODINGS)
    m = _masks(model, bound_method, repair_model, reliability_impl)

    def cv(key, applies, default=None, optional=False):
        return _cell_value(data, key, F, L, applies, default, optional)

    def pf(key, Hs, applies, rates=False, required=False):
        return _profile(data, key, F, L, M, Hs, applies, rates, required)

    tau, epsilon, rho, mu_0 = (cv(key, everywhere) for key in ("tau", "epsilon", "rho", "mu_0"))
    s_chernoff = cv("s_chernoff", m["chernoff"])
    v_0 = cv("v_0", m["variance"], 0.0)
    support_0 = cv("support_0", m["bernstein"], 0.0)
    R_0 = cv("R_0", m["hoeffding"], 0.0)
    cgf_0 = cv("cgf_0", m["chernoff"], s_chernoff * mu_0)
    replacement_mu = cv("replacement_mu", everywhere, 0.0)
    replacement_v = cv("replacement_v", m["variance"], 0.0)
    support_new = cv("support_new", m["bernstein"], 0.0)
    R_new = cv("R_new", m["hoeffding"], 0.0)
    cgf_new = cv("cgf_new", m["chernoff"], s_chernoff * replacement_mu)
    anchor_mu_0 = cv("anchor_mu_0", m["ard1"], mu_0)
    anchor_v_0 = cv("anchor_v_0", m["variance_anchor"], v_0)
    anchor_R_0 = cv("anchor_R_0", m["ard1"] & m["hoeffding"], R_0)
    anchor_K_0 = cv("anchor_K_0", m["ard1"] & m["chernoff"], cgf_0)
    tangent_ref = cv("tangent_ref", m["tangent"], 0.5)
    sigma = cv("sigma", m["wiener"])
    eta = cv("eta", m["inverse_gaussian"])
    v_max_user = cv("v_max_user", m["variance"], optional=True)
    gamma_beta_0 = cv("gamma_beta_0", m["gamma"], optional=True)
    gamma_beta_new = cv("gamma_beta_new", m["gamma"], optional=True)
    C_R, C_rep = cv("C_R", everywhere), cv("C_rep", everywhere)

    profiles = {}
    for key, applies, rates in (("mu", everywhere, False), ("v", m["v_profile"], False),
                                ("support", m["support"], False), ("cgf", m["chernoff"], False),
                                ("gamma_beta", m["gamma"], True)):
        profiles[key] = pf(key, H2, applies, rates, required=True)
        if f"{key}_trans" in data and key not in data:
            raise ValueError(f"'{key}_trans' needs the operating profile '{key}'.")
        profiles[f"{key}_trans"] = pf(f"{key}_trans", H1, applies, rates)

    pwl_points = _integer(data.get("pwl_points", 8), "pwl_points")
    if pwl_points < 1:
        raise ValueError("'pwl_points' must be an integer >= 1.")
    allow_replacement = _flag(data.get("allow_replacement", True), "allow_replacement")
    replacement_as_new = _flag(data.get("replacement_as_new", False), "replacement_as_new")
    costs = {}
    for key in ("C_M", "C_D"):
        costs[key] = _number(data[key], key)
        if not (math.isfinite(costs[key]) and _le(0.0, costs[key])):
            raise ValueError(f"'{key}' must be a finite number >= 0.")
    penalty_type = data.get("penalty_type", "inf_norm")
    if penalty_type not in PENALTY_TYPES:
        raise ValueError(f"'penalty_type' must be one of {PENALTY_TYPES} (got {penalty_type!r}).")

    cfg = FleetConfig(
        F=F, M=M, L=L, H=H, H1=H1, H2=H2, T=T,
        component_names=_component_names(data.get("component_names"), L),
        model=model, repair_model=repair_model, bound_method=bound_method,
        reliability_impl=reliability_impl, encoding=encoding,
        tau=tau, epsilon=epsilon, tangent_ref=tangent_ref, rho=rho,
        anchor_mu_0=anchor_mu_0, anchor_v_0=anchor_v_0, anchor_R_0=anchor_R_0,
        anchor_K_0=anchor_K_0, mu_0=mu_0, v_0=v_0, support_0=support_0, R_0=R_0, cgf_0=cgf_0,
        replacement_mu=replacement_mu, replacement_v=replacement_v, support_new=support_new,
        R_new=R_new, cgf_new=cgf_new, s_chernoff=s_chernoff, sigma=sigma, eta=eta,
        v_max_user=v_max_user, gamma_beta_0=gamma_beta_0, gamma_beta_new=gamma_beta_new,
        C_R=C_R, C_rep=C_rep,
        pwl_points=pwl_points, allow_replacement=allow_replacement,
        replacement_as_new=replacement_as_new, C_M=costs["C_M"], C_D=costs["C_D"],
        penalty_type=penalty_type, **profiles,
        options=_run_options(data, H1),
    )
    _check_state(cfg)
    _check_increments(cfg)
    _derive(cfg)
    _check_models(cfg)
    _check_ranges(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Input reader
# ---------------------------------------------------------------------------
INPUT_EXTENSIONS = (".yaml", ".yml", ".json", ".h5", ".hdf5")


def read_input(path) -> dict:
    """Read an input file (YAML, JSON or HDF5, chosen by the extension) into one mapping."""
    suffix = Path(path).suffix.lower()
    if suffix in (".yaml", ".yml"):
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif suffix in (".h5", ".hdf5"):
        data = _read_hdf5(path)
    else:
        raise ValueError(
            f"Unsupported input file type '{suffix}'. "
            f"Supported types: {list(INPUT_EXTENSIONS)}"
        )
    if not isinstance(data, dict):
        raise TypeError(f"the input file {path} must hold one mapping from key names to values.")
    return data


def _read_hdf5(path) -> dict:
    """One entry per root attribute and per root dataset of a flat HDF5 file."""
    import h5py
    data = {}
    with h5py.File(path, "r") as f:
        for name, value in f.attrs.items():
            data[name] = _hdf5_value(value, text_json=True)
        for name, item in f.items():
            if isinstance(item, h5py.Group):
                raise ValueError(f"HDF5 input must be flat: '{name}' is a group.")
            if name in data:
                raise ValueError(f"HDF5 key '{name}' is both an attribute and a dataset.")
            data[name] = _hdf5_value(item[()], text_json=False)
    return data


def _hdf5_text(item):
    """Nested list of bytes or str -> the same nesting with UTF-8 ``str``."""
    if isinstance(item, list):
        return [_hdf5_text(x) for x in item]
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def _hdf5_value(value, *, text_json: bool):
    """HDF5 value -> Python value: UTF-8 text, JSON text if asked, lists, plain scalars."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        value = str(value)
        if text_json and value[:1] in ("{", "["):
            try:
                return json.loads(value)
            except ValueError:
                raise ValueError(f"HDF5 attribute text {value[:40]!r} is not valid JSON.")
        return value
    if isinstance(value, np.ndarray):
        return _hdf5_text(value.tolist()) if value.dtype.kind in "SOU" else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
