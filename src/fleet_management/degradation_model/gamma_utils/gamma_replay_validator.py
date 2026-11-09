"""Replay validator of the Gamma model and the helpers that the Gamma validators share.

Spec, section "Validators of the Gamma Model". Indices ``i``, ``l``, ``j``, ``k`` of a
report are 0-based array indices; day ``k`` is day ``k + 1`` of the spec.

The replay validator rebuilds the tracked mean and the anchor of every Gamma cell from the
saved schedule and compares them with the saved states. The exact and Monte Carlo
validators import the helpers below: the Gamma cells, the arrays of a result, the
schedule checks, the records of a report and the writer of a report file. No Gurobi
object is read.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from fleet_management.config import FleetConfig, _lookup, load_config, read_input
from fleet_management.solver import _to_plain, read_result
from fleet_management.validation.tolerance import (
    binary_tolerance,
    report_tolerance,
    resolve_tolerance,
    state_tolerance,
)

REPORT_SUFFIXES = (".yaml", ".yml", ".json")


# ===========================================================================
# Helpers shared by the Gamma validators
# ===========================================================================
def _check_tolerance(tolerance: float | None) -> None:
    """``tolerance`` must be ``None`` (a default per check) or positive (spec, "Common rules")."""
    if tolerance is not None and not tolerance > 0:
        raise ValueError("tolerance must be positive.")


def gamma_cells(cfg: FleetConfig, result: dict) -> list[tuple[int, int]]:
    """The Gamma cells ``(i, l)``, ``i`` outer, from the result key ``model`` (spec, "Common rules").

    A result whose ``model`` differs from the input at some cell is refused: the Gamma
    inputs of a cell exist only where the input says ``gamma``.
    """
    names = result.get("model")
    if names is None:
        raise ValueError("result is missing 'model'.")
    names = np.asarray(names).astype(str)
    if names.shape != (cfg.F, cfg.L):
        raise ValueError(f"result 'model' has shape {names.shape}; expected {(cfg.F, cfg.L)}.")
    expected = np.asarray(cfg.model).astype(str)
    for i in range(cfg.F):
        for l in range(cfg.L):
            if names[i, l] != expected[i, l]:
                raise ValueError(
                    f"result 'model' is {names[i, l]!r} at cell (i={i}, l={l}); "
                    f"the input says {expected[i, l]!r}."
                )
    cells = [(i, l) for i in range(cfg.F) for l in range(cfg.L) if names[i, l] == "gamma"]
    if not cells:
        raise ValueError("the configuration has no Gamma cell.")
    return cells


def _array(result: dict, name: str, shape: tuple[int, ...], cells=None) -> np.ndarray:
    """The array ``result[name]`` as floats, with its shape checked.

    Every entry must be finite if ``cells`` is ``None``; otherwise only the entries of
    ``cells`` (the other cells of a merged result hold ``NaN``).
    """
    if result.get(name) is None:
        raise ValueError(f"result is missing {name!r}.")
    value = np.asarray(result[name], dtype=float)     # a None entry becomes NaN
    if value.shape != shape:
        raise ValueError(f"result {name!r} has shape {value.shape}; expected {shape}.")
    entries = value if cells is None else np.array([value[i, l] for i, l in cells])
    if not np.all(np.isfinite(entries)):
        raise ValueError(f"result {name!r} has a non-finite entry.")
    return value


def _binary(result: dict, name: str, shape: tuple[int, ...], tolerance: float,
            optional: bool = False) -> np.ndarray:
    """The array ``result[name]`` rounded to integers; it must lie within ``tolerance`` of {0, 1}."""
    if optional and result.get(name) is None:
        return np.zeros(shape, dtype=np.int64)
    value = _array(result, name, shape)
    rounded = np.rint(value)
    error = max(float(np.max(np.abs(value - rounded))),
                float(np.max(-value)), float(np.max(value - 1.0)), 0.0)
    if error > tolerance:
        raise ValueError(
            f"result {name!r} is not binary within tolerance; maximum error={error:.3e}."
        )
    return rounded.astype(np.int64)


def _increments(cfg: FleetConfig) -> tuple[np.ndarray, np.ndarray]:
    """Mean and rate of every mission increment on the days ``1..T``: two ``(F, L, M, T)`` arrays.

    Eq. ``c_phase``. Mission ``j`` of ``x`` is index ``j - 1`` on axis 2.
    """
    return (_lookup(cfg.mu, cfg.mu_trans, cfg.H1, cfg.H2),
            _lookup(cfg.gamma_beta, cfg.gamma_beta_trans, cfg.H1, cfg.H2))


def gamma_cell_caps(cfg: FleetConfig, cells: list[tuple[int, int]]) -> dict[tuple[int, int], float]:
    """The cap ``A_1(beta* tau, epsilon) / beta*`` of every Gamma cell (Eqs. gm_betastar, gm_row).

    The computation is that of ``GammaCellBuilder.prepare``; ``tau`` is the cap of a cell
    in which no term has a rate.
    """
    from fleet_management.degradation_model.gamma import mean_cap, smallest_rate

    mu_inc, beta_inc = _increments(cfg)
    caps = {}
    for i, l in cells:
        beta_star = smallest_rate(
            mu_inc[i, l], beta_inc[i, l],
            float(cfg.mu_0[i, l]), float(cfg.gamma_beta_0[i, l]),
            float(cfg.replacement_mu[i, l]), float(cfg.gamma_beta_new[i, l]),
            bool(cfg.allow_replacement),
        )
        caps[i, l] = mean_cap(beta_star, float(cfg.tau[i, l]), float(cfg.epsilon[i, l]))
    return caps


def _event(x: np.ndarray, m: np.ndarray, r: np.ndarray, i: int, l: int, k: int) -> str:
    """The event of cell ``(i, l)`` on day ``k`` (spec, "Replay validator", step 3)."""
    if r[i, l, k]:
        return "replacement"
    if m[i, l, k]:
        return "repair"
    if x[i, 1:, k].any():
        return "mission"
    return "idle"


def _schedule_violations(cfg: FleetConfig, result: dict, x: np.ndarray, m: np.ndarray,
                         r: np.ndarray) -> list[str]:
    """The global violations of the saved schedule, one string per occurrence (spec, step 1)."""
    out = []
    horizon = result.get("T")
    if horizon is None or int(horizon) != cfg.T:
        out.append(f"result horizon T={horizon} differs from the input T={cfg.T}")
    for i, k in np.argwhere(x.sum(axis=1) > 1):
        out.append(f"train i={int(i)} has {int(x[i, :, k].sum())} activities on day k={int(k)}")
    for j0, k in np.argwhere(x[:, 1:, :].sum(axis=0) != cfg.d):
        out.append(f"mission j={int(j0) + 1} is flown by {int(x[:, j0 + 1, k].sum())} "
                   f"trains on day k={int(k)}, not {int(cfg.d[j0, k])}")
    for i, l, k in np.argwhere((m == 1) & (r == 1)):
        out.append(f"cell (i={int(i)}, l={int(l)}) has a repair and a replacement "
                   f"on day k={int(k)}")
    for i, l, k in np.argwhere(m + r > 0):
        with_mission = bool(x[i, 1:, k].any())
        without_depot = x[i, 0, k] == 0
        if with_mission or without_depot:
            reasons = " and ".join(
                (["with a mission"] if with_mission else [])
                + (["without a depot day"] if without_depot else []))
            out.append(f"cell (i={int(i)}, l={int(l)}) has an intervention on day "
                       f"k={int(k)} {reasons}")
    return out


def _record(i: int, l: int, k: int, event: str, field: str, expected: float,
            saved: float, error: float) -> dict[str, Any]:
    """One record of ``violations`` (spec, "Replay validator", step 5)."""
    return {"i": int(i), "l": int(l), "k": int(k), "event": event, "field": field,
            "expected": float(expected), "saved": float(saved), "error": float(error)}


def _check_report_path(report_path) -> Path | None:
    """``None``, or ``report_path`` as a path; its suffix is checked before any reading."""
    if report_path is None:
        return None
    path = Path(report_path)
    if path.suffix.lower() not in REPORT_SUFFIXES:
        raise ValueError(
            f"report_path must end in .yaml, .yml or .json (got {str(report_path)!r})."
        )
    return path


def _read_pair(input_path, result_path) -> tuple[FleetConfig, dict]:
    """Configuration of the input file and result of the result file, in any format."""
    return load_config(read_input(input_path)), read_result(result_path)


def _finish(report: dict, destination: Path | None, raise_on_failure: bool) -> dict:
    """Write ``report`` to ``destination`` if given, then raise if it failed and the caller asked."""
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        plain = _to_plain(report)
        with open(destination, "w", encoding="utf-8") as f:
            if destination.suffix.lower() == ".json":
                json.dump(plain, f, indent=2, allow_nan=False)
            else:
                yaml.safe_dump(plain, f, sort_keys=False, allow_unicode=True)
    if raise_on_failure and not report["valid"]:
        raise AssertionError(f"{report['validator']}: the report has valid = false.")
    return report


# ===========================================================================
# Replay validator
# ===========================================================================
def validate_gamma_replay_schedule(
    cfg: FleetConfig,
    result: dict,
    *,
    tolerance: float | None = None,
    raise_on_failure: bool = False,
) -> dict[str, Any]:
    """Replay the saved schedule of every Gamma cell and compare it with the saved states.

    The reference trajectory of ``mu`` and of the anchor ``g_mu`` starts at the initial
    value and the starting anchor of the cell (Eqs. gm_map, a_anchor). The saved states
    are upper bounds of the true moments, so every error is one-sided: it is the amount
    by which the saved value is below the reference (``cap`` and ``repeatability``: above
    the limit). ``repeatability`` compares the saved ``mu`` (and, under ``ard1``, the anchor)
    on days ``H1`` and ``T`` with the saved start ``y_mu`` (``y_g_mu``) of the repetitions;
    a result without these keys is compared on day ``H1`` instead. The fields are ``tracked_mean``, ``mean_anchor``, ``removed_mean``,
    ``cap`` and ``repeatability``. ``valid`` is true if there is no global violation and
    no violation. ``raise_on_failure`` raises ``AssertionError`` if it is false.

    The tolerance of a field is ``tolerance`` if it is a number, else the key ``tolerance`` of
    the input block ``validation``, else ``FeasibilityTol + IntFeasTol * range_mu`` at the
    cell (``IntFeasTol`` for the binaries; ``validation/tolerance.py``). The report keys
    ``tolerance`` and ``maximum_error_tolerances`` give the tolerance and the tolerance at the
    cell of the largest error of each field.
    """
    _check_tolerance(tolerance)
    single = resolve_tolerance(cfg, tolerance)
    tol_binary = binary_tolerance(cfg, single)
    tol_cell = state_tolerance(cfg, "mu", single)
    started = time.perf_counter()
    F, M, L, T = cfg.F, cfg.M, cfg.L, cfg.T
    states = (F, L, T)
    cells = gamma_cells(cfg, result)
    ard1 = [(i, l) for i, l in cells if str(cfg.repair_model[i, l]) == "ard1"]

    x = _binary(result, "x", (F, M + 1, T), tol_binary)
    m = _binary(result, "m", states, tol_binary)
    r = _binary(result, "r", states, tol_binary, optional=True)
    mu = _array(result, "mu", states, cells)
    z = _array(result, "z", states, [])               # NaN where C_R = 0 (not built: skipped)
    for i, l in cells:                                # an infinite entry is read, so it is refused
        if np.isinf(z[i, l]).any():
            raise ValueError(f"result 'z' has a non-finite entry at Gamma cell (i={i}, l={l}).")
    g_mu = _array(result, "g_mu", states, ard1) if ard1 else None
    start_mu = {}                                     # y_mu, y_g_mu: F x L, or None (not saved)
    for y_key in ("y_mu", "y_g_mu"):
        saved_y = result.get(y_key)
        y_array = None if saved_y is None else np.asarray(saved_y, dtype=float)
        if y_array is not None and y_array.shape != (F, L):
            raise ValueError(f"result '{y_key}' has shape {y_array.shape}; expected {(F, L)}.")
        start_mu[y_key] = y_array
    mu_inc, _ = _increments(cfg)
    caps = gamma_cell_caps(cfg, cells)
    global_violations = _schedule_violations(cfg, result, x, m, r)

    maxima = dict.fromkeys(
        ("tracked_mean", "mean_anchor", "removed_mean", "cap", "repeatability"), 0.0)
    maxima_tolerance = dict.fromkeys(maxima, float(tol_cell[cells[0]]))
    violations: list[dict[str, Any]] = []
    repairs = replacements = 0

    def check(i, l, k, event, field, expected, saved, error):
        if error > maxima[field]:
            maxima[field], maxima_tolerance[field] = error, float(tol_cell[i, l])
        if error > tol_cell[i, l]:
            violations.append(_record(i, l, k, event, field, expected, saved, error))

    for i, l in cells:
        anchored = str(cfg.repair_model[i, l]) == "ard1"
        rho = float(cfg.rho[i, l])
        ref_mu = float(cfg.mu_0[i, l])
        ref_g = float(cfg.anchor_mu_0[i, l]) if anchored else 0.0
        prev_mu, prev_g = ref_mu, ref_g               # saved values of day k-1; day 0: the inputs
        events = []
        for k in range(T):
            event = _event(x, m, r, i, l, k)
            events.append(event)
            if event == "replacement":
                replacements += 1
                ref_mu = float(cfg.replacement_mu[i, l])
            elif event == "repair":
                repairs += 1
                ref_mu = (1.0 - rho) * ref_mu + rho * ref_g      # ref_g = 0 under ardinf
            else:
                ref_mu += float(np.dot(x[i, 1:, k], mu_inc[i, l, :, k]))
            if anchored and event in ("replacement", "repair"):
                ref_g = ref_mu
            saved = float(mu[i, l, k])
            check(i, l, k, event, "tracked_mean", ref_mu, saved, max(0.0, ref_mu - saved))
            if anchored:
                saved_g = float(g_mu[i, l, k])
                check(i, l, k, event, "mean_anchor", ref_g, saved_g, max(0.0, ref_g - saved_g))
            if not math.isnan(z[i, l, k]):
                removed = float(m[i, l, k]) * rho * (prev_mu - prev_g)     # Eq. a_z
                check(i, l, k, event, "removed_mean", removed, float(z[i, l, k]),
                      max(0.0, removed - float(z[i, l, k])))
            check(i, l, k, event, "cap", caps[i, l], saved, max(0.0, saved - caps[i, l]))
            prev_mu, prev_g = saved, (float(g_mu[i, l, k]) if anchored else 0.0)
        # repeatability, on the saved states: against the start y of the repetitions if saved,
        # else against the value on day H1 (a result written before y existed)
        for array, y_key in [(mu, "y_mu")] + ([(g_mu, "y_g_mu")] if anchored else []):
            first, last = float(array[i, l, cfg.H1 - 1]), float(array[i, l, T - 1])
            y = start_mu[y_key]
            if y is not None and math.isfinite(float(y[i, l])):
                limit = float(y[i, l])
                error = max(0.0, last - limit, first - limit)
            else:
                limit, error = first, max(0.0, last - first)
            check(i, l, T - 1, events[-1], "repeatability", limit, last, error)

    report = {
        "validator": "Gamma replay validator",
        "valid": not global_violations and not violations,
        "solver_status": str(result.get("status") or "unknown"),   # read_result: None if absent
        "dimensions": {"F": F, "M": M, "L": L, "H1": cfg.H1, "H2": cfg.H2, "T": T},
        "gamma_cells": len(cells),
        "gamma_ard1_cells": len(ard1),
        "transitions_checked": len(cells) * T,
        "repairs": repairs,
        "replacements": replacements,
        "tolerance": report_tolerance(cfg, single),
        "maximum_errors": maxima,
        "maximum_error_tolerances": maxima_tolerance,
        "global_violations": global_violations,
        "violations": violations,
        "timing": {"validation_wall_seconds": time.perf_counter() - started},
    }
    return _finish(report, None, raise_on_failure)


def validate_gamma_replay_files(input_path: str | Path, result_path: str | Path,
                                report_path: str | Path | None = None,
                                **kwargs) -> dict[str, Any]:
    """File version (spec, "Common rules"): input and result paths; the report as YAML or JSON."""
    destination = _check_report_path(report_path)         # before any reading
    raise_on_failure = kwargs.pop("raise_on_failure", False)
    cfg, result = _read_pair(input_path, result_path)
    report = validate_gamma_replay_schedule(cfg, result, **kwargs)
    return _finish(report, destination, raise_on_failure)     # write, then raise
