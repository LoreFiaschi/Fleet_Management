"""PROTOTYPE of steps C1 and C2 of the plan (scratch, written by the Task 1 agent to run the tests)."""

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

REPORT_SUFFIXES = (".yaml", ".yml", ".json")


def _check_tolerance(tolerance):
    if not tolerance > 0:
        raise ValueError("tolerance must be positive.")


def gamma_cells(cfg, result):
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
                    f"the input says {expected[i, l]!r}.")
    cells = [(i, l) for i in range(cfg.F) for l in range(cfg.L) if names[i, l] == "gamma"]
    if not cells:
        raise ValueError("the configuration has no Gamma cell.")
    return cells


def _array(result, name, shape, cells=None):
    if result.get(name) is None:
        raise ValueError(f"result is missing {name!r}.")
    value = np.asarray(result[name], dtype=float)
    if value.shape != shape:
        raise ValueError(f"result {name!r} has shape {value.shape}; expected {shape}.")
    if cells is None:
        entries = value
    else:
        entries = np.array([value[i, l] for i, l in cells])
    if entries.size and np.any(~np.isfinite(entries)):
        raise ValueError(f"result {name!r} has a non-finite entry.")
    return value


def _binary(result, name, shape, tolerance, optional=False):
    if optional and result.get(name) is None:
        return np.zeros(shape, dtype=np.int64)
    value = _array(result, name, shape)
    rounded = np.rint(value)
    error = max(float(np.max(np.abs(value - rounded))),
                float(np.max(-value)), float(np.max(value - 1.0)), 0.0)
    if error > tolerance:
        raise ValueError(f"result {name!r} is not binary within tolerance; maximum error={error:.3e}.")
    return rounded.astype(np.int64)


def _increments(cfg):
    return (_lookup(cfg.mu, cfg.mu_trans, cfg.H1, cfg.H2),
            _lookup(cfg.gamma_beta, cfg.gamma_beta_trans, cfg.H1, cfg.H2))


def gamma_cell_caps(cfg, cells):
    from fleet_management.degradation_model.gamma import mean_cap, smallest_rate
    mu_inc, beta_inc = _increments(cfg)
    caps = {}
    for i, l in cells:
        beta_star = smallest_rate(mu_inc[i, l], beta_inc[i, l], float(cfg.mu_0[i, l]),
                                  float(cfg.gamma_beta_0[i, l]), float(cfg.replacement_mu[i, l]),
                                  float(cfg.gamma_beta_new[i, l]), bool(cfg.allow_replacement))
        caps[i, l] = mean_cap(beta_star, float(cfg.tau[i, l]), float(cfg.epsilon[i, l]))
    return caps


def _event(x, m, r, i, l, k):
    if r[i, l, k]:
        return "replacement"
    if m[i, l, k]:
        return "repair"
    if x[i, 1:, k].any():
        return "mission"
    return "idle"


def _schedule_violations(cfg, result, x, m, r):
    out = []
    T = result.get("T")
    if T is None or int(T) != cfg.T:
        out.append(f"result horizon T={T} differs from the input T={cfg.T}")
    for i, k in np.argwhere(x.sum(axis=1) > 1):
        out.append(f"train i={int(i)} has {int(x[i, :, k].sum())} activities on day k={int(k)}")
    for j0, k in np.argwhere(x[:, 1:, :].sum(axis=0) != 1):
        out.append(f"mission j={int(j0) + 1} is covered {int(x[:, j0 + 1, k].sum())} times on day k={int(k)}")
    for i, l, k in np.argwhere((m == 1) & (r == 1)):
        out.append(f"cell (i={int(i)}, l={int(l)}) has a repair and a replacement on day k={int(k)}")
    for i, l, k in np.argwhere(m + r > 0):
        if x[i, 1:, k].any():
            out.append(f"cell (i={int(i)}, l={int(l)}) has an intervention on day k={int(k)} with a mission")
    for i, l, k in np.argwhere(m + r > 0):
        if x[i, 0, k] == 0:
            out.append(f"cell (i={int(i)}, l={int(l)}) has an intervention on day k={int(k)} without a depot day")
    return out


def _record(i, l, k, event, field, expected, saved, error):
    return {"i": int(i), "l": int(l), "k": int(k), "event": event, "field": field,
            "expected": float(expected), "saved": float(saved), "error": float(error)}


def _check_report_path(report_path):
    if report_path is None:
        return None
    path = Path(report_path)
    if path.suffix.lower() not in REPORT_SUFFIXES:
        raise ValueError(f"report_path must end in .yaml, .yml or .json (got {str(report_path)!r}).")
    return path


def _read_pair(input_path, result_path):
    return load_config(read_input(input_path)), read_result(result_path)


def _finish(report, destination, raise_on_failure):
    if destination:
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


def validate_gamma_replay_schedule(cfg, result, *, tolerance=1e-8, raise_on_failure=False):
    _check_tolerance(tolerance)
    started = time.perf_counter()
    F, M, L, T = cfg.F, cfg.M, cfg.L, cfg.T
    cells = gamma_cells(cfg, result)
    ard1 = [(i, l) for i, l in cells if str(cfg.repair_model[i, l]) == "ard1"]
    x = _binary(result, "x", (F, M + 1, T), tolerance)
    m = _binary(result, "m", (F, L, T), tolerance)
    r = _binary(result, "r", (F, L, T), tolerance, optional=True)
    mu = _array(result, "mu", (F, L, T), cells)
    z = _array(result, "z", (F, L, T), [])
    g_mu = _array(result, "g_mu", (F, L, T), ard1) if ard1 else None
    mu_inc, _ = _increments(cfg)
    caps = gamma_cell_caps(cfg, cells)
    global_violations = _schedule_violations(cfg, result, x, m, r)
    maxima = dict.fromkeys(("tracked_mean", "mean_anchor", "removed_mean", "cap", "repeatability"), 0.0)
    violations = []
    repairs = replacements = 0

    def check(i, l, k, event, field, expected, saved, error):
        maxima[field] = max(maxima[field], error)
        if error > tolerance:
            violations.append(_record(i, l, k, event, field, expected, saved, error))

    for i, l in cells:
        anchored = str(cfg.repair_model[i, l]) == "ard1"
        rho = float(cfg.rho[i, l])
        ref_mu = float(cfg.mu_0[i, l])
        ref_g = float(cfg.anchor_mu_0[i, l]) if anchored else 0.0
        prev_mu, prev_g = ref_mu, ref_g
        events = []
        for k in range(T):
            event = _event(x, m, r, i, l, k)
            events.append(event)
            if event == "replacement":
                replacements += 1
                ref_mu = float(cfg.replacement_mu[i, l])
            elif event == "repair":
                repairs += 1
                ref_mu = (1.0 - rho) * ref_mu + rho * ref_g
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
                removed = float(m[i, l, k]) * rho * (prev_mu - prev_g)
                check(i, l, k, event, "removed_mean", removed, float(z[i, l, k]),
                      max(0.0, removed - float(z[i, l, k])))
            check(i, l, k, event, "cap", caps[i, l], saved, max(0.0, saved - caps[i, l]))
            prev_mu, prev_g = saved, (float(g_mu[i, l, k]) if anchored else 0.0)
        for array in [mu] + ([g_mu] if anchored else []):
            first, last = float(array[i, l, cfg.H1 - 1]), float(array[i, l, T - 1])
            check(i, l, T - 1, events[-1], "repeatability", first, last, max(0.0, last - first))

    report = {
        "validator": "Gamma replay validator",
        "valid": not global_violations and not violations,
        "solver_status": str(result.get("status", "unknown")),
        "dimensions": {"F": F, "M": M, "L": L, "H1": cfg.H1, "H2": cfg.H2, "T": T},
        "gamma_cells": len(cells),
        "gamma_ard1_cells": len(ard1),
        "transitions_checked": len(cells) * T,
        "repairs": repairs,
        "replacements": replacements,
        "maximum_errors": maxima,
        "global_violations": global_violations,
        "violations": violations,
        "timing": {"validation_wall_seconds": time.perf_counter() - started},
    }
    return _finish(report, None, raise_on_failure)


def validate_gamma_replay_files(input_path, result_path, report_path=None, **kwargs):
    destination = _check_report_path(report_path)
    raise_on_failure = kwargs.pop("raise_on_failure", False)
    cfg, result = _read_pair(input_path, result_path)
    report = validate_gamma_replay_schedule(cfg, result, **kwargs)
    return _finish(report, destination, raise_on_failure)
