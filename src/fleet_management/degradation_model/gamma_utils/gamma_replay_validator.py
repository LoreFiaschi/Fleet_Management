"""Lightweight schedule/state replay for modular Gamma solutions.

This module checks deterministic solver bookkeeping.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import yaml

from fleet_management.config import FleetConfig, load_config


def _array(result: dict, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if result.get(name) is None:
        raise ValueError(f"result is missing {name!r}.")
    value = np.asarray(result[name], dtype=float)
    if value.shape != shape:
        raise ValueError(f"result {name!r} has shape {value.shape}; expected {shape}.")
    if np.any(~np.isfinite(value)):
        raise ValueError(f"result {name!r} contains non-finite values.")
    return value


def _binary(result: dict, name: str, shape: tuple[int, ...], tolerance: float,
            optional: bool = False) -> np.ndarray:
    if optional and result.get(name) is None:
        return np.zeros(shape, dtype=np.int64)
    value = _array(result, name, shape)
    rounded = np.rint(value)
    error = max(float(np.max(np.abs(value - rounded))),
                float(np.max(-value)), float(np.max(value - 1.0)), 0.0)
    if error > tolerance:
        raise ValueError(f"{name!r} is not binary; maximum error={error:.3e}.")
    return rounded.astype(np.int64)


def _phase(profile: np.ndarray, transitory: np.ndarray | None,
           i: int, l: int, j: int, k: int, H1: int, H2: int) -> float:
    if k < H1:
        if transitory is not None:
            return float(transitory[i, l, j, k])
        return float(profile[i, l, j, k % H2])
    return float(profile[i, l, j, (k - H1) % H2])


def validate_gamma_replay_schedule(
    cfg: FleetConfig,
    result: dict,
    *,
    tolerance: float = 1e-8,
    raise_on_failure: bool = False,
) -> dict[str, Any]:
    """Replay Gamma dynamics and compare reconstructed and saved states."""
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    started = time.perf_counter()
    F, L, M, T = cfg.F, cfg.L, cfg.M, cfg.T
    states = (F, L, T)
    cells = [(i, l) for i in range(F) for l in range(L)
             if str(cfg.model[i, l]) == "gamma"]
    if not cells:
        raise ValueError("configuration contains no Gamma cells.")

    x = _binary(result, "x", (F, M + 1, T), tolerance)
    m = _binary(result, "m", states, tolerance)
    r = _binary(result, "r", states, tolerance, optional=True)
    mu = _array(result, "mu", states)
    z = _array(result, "z", states)
    shape = _array(result, "gamma_shape_bound", states)
    shape_max = _array(result, "gamma_maximum_shape", (F, L))
    shape_op = _array(result, "gamma_shape_increment", (F, L, M, cfg.H2))
    shape_tr = _array(result, "gamma_shape_increment_trans", (F, L, M, cfg.H1))
    ard1 = {cell for cell in cells if str(cfg.repair_model[cell]) == "ard1"}
    gmu = _array(result, "gamma_mean_latch", states) if ard1 else None
    gshape = _array(result, "gamma_shape_latch", states) if ard1 else None

    rows = result.get("gamma_calibration")
    if not isinstance(rows, list):
        raise ValueError("result is missing Gamma calibration metadata.")
    calibration = {(int(row["i"]), int(row["l"])): row for row in rows}

    global_failures = []
    if result.get("backend") != "modular":
        global_failures.append("result backend is not modular")
    if int(result.get("T", -1)) != T:
        global_failures.append("result horizon differs from configuration")
    if np.max(np.sum(x, axis=1) - 1.0) > tolerance:
        global_failures.append("vehicle assignment violation")
    if np.max(np.abs(np.sum(x[:, 1:, :], axis=0) - 1.0)) > tolerance:
        global_failures.append("mission demand violation")

    maxima = {name: 0.0 for name in (
        "physical_mean", "bounding_shape", "removed_mean", "mean_latch",
        "shape_latch", "reliability", "repeatability")}
    failures = []
    repairs = replacements = 0

    def compare(i, l, k, event, field, saved, expected):
        error = abs(float(saved) - float(expected))
        maxima[field] = max(maxima[field], error)
        if error > tolerance:
            failures.append({"i": i, "l": l, "k": k, "event": event,
                             "field": field, "expected": float(expected),
                             "saved": float(saved), "error": error})

    mu_op = np.asarray(cfg.mu, dtype=float)
    mu_tr = None if cfg.mu_trans is None else np.asarray(cfg.mu_trans, dtype=float)

    for i, l in cells:
        row = calibration.get((i, l))
        if row is None:
            raise ValueError(f"missing Gamma calibration row for {(i, l)}.")
        model = str(cfg.repair_model[i, l])
        if model not in {"ardinf", "ard1"}:
            raise ValueError(f"unsupported Gamma repair model {model!r}.")
        use_latch = model == "ard1"
        rho = float(cfg.rho[i, l])
        remaining = 1.0 - rho
        previous_mu = float(cfg.mu_0[i, l])
        previous_shape = float(row["initial_bounded_shape"])
        previous_gmu = previous_gshape = 0.0
        replacement_mu = float(cfg.replacement_mu[i, l])
        replacement_shape = float(row["replacement_bounded_shape"])

        for k in range(T):
            do_repair, do_replace = bool(m[i, l, k]), bool(r[i, l, k])
            missions = np.flatnonzero(x[i, 1:, k])
            if do_repair and do_replace:
                global_failures.append(f"repair and replacement at {(i,l,k)}")
            if (do_repair or do_replace) and missions.size:
                global_failures.append(f"maintenance and mission at {(i,l,k)}")
            if (do_repair or do_replace) and not x[i, 0, k]:
                global_failures.append(f"maintenance without depot at {(i,l,k)}")

            if do_replace:
                event, replacements = "replacement", replacements + 1
                expected_mu, expected_shape = replacement_mu, replacement_shape
                expected_z = previous_mu - expected_mu
                expected_gmu, expected_gshape = expected_mu, expected_shape
            elif do_repair:
                event, repairs = "repair", repairs + 1
                if use_latch:
                    expected_mu = previous_gmu + remaining * (previous_mu - previous_gmu)
                    expected_shape = previous_gshape + remaining * (previous_shape - previous_gshape)
                else:
                    expected_mu = remaining * previous_mu
                    expected_shape = remaining * previous_shape
                expected_z = previous_mu - expected_mu
                expected_gmu, expected_gshape = expected_mu, expected_shape
            else:
                event = "mission" if missions.size else "idle"
                expected_mu = previous_mu + sum(
                    _phase(mu_op, mu_tr, i, l, int(j), k, cfg.H1, cfg.H2)
                    for j in missions)
                expected_shape = previous_shape + sum(
                    _phase(shape_op, shape_tr, i, l, int(j), k, cfg.H1, cfg.H2)
                    for j in missions)
                expected_z = 0.0
                expected_gmu, expected_gshape = previous_gmu, previous_gshape

            compare(i, l, k, event, "physical_mean", mu[i, l, k], expected_mu)
            compare(i, l, k, event, "bounding_shape", shape[i, l, k], expected_shape)
            compare(i, l, k, event, "removed_mean", z[i, l, k], expected_z)
            if use_latch:
                compare(i, l, k, event, "mean_latch", gmu[i, l, k], expected_gmu)
                compare(i, l, k, event, "shape_latch", gshape[i, l, k], expected_gshape)

            excess = max(0.0, float(shape[i, l, k] - shape_max[i, l]))
            maxima["reliability"] = max(maxima["reliability"], excess)
            if excess > tolerance:
                failures.append({"i": i, "l": l, "k": k, "event": event,
                                 "field": "reliability", "error": excess})
            previous_mu, previous_shape = expected_mu, expected_shape
            previous_gmu, previous_gshape = expected_gmu, expected_gshape

        start, end = cfg.H1 - 1, T - 1
        excess = max(0.0, mu[i,l,end] - mu[i,l,start],
                     shape[i,l,end] - shape[i,l,start])
        if use_latch:
            excess = max(excess, gmu[i,l,end] - gmu[i,l,start],
                         gshape[i,l,end] - gshape[i,l,start])
        maxima["repeatability"] = max(maxima["repeatability"], float(excess))
        if excess > tolerance:
            failures.append({"i": i, "l": l, "k": end,
                             "event": "repeatability", "field": "repeatability",
                             "error": float(excess)})

    report = {
        "validator": "Gamma schedule/state replay",
        "valid": not global_failures and not failures,
        "solver_status": str(result.get("status", "unknown")),
        "dimensions": {"F": F, "M": M, "L": L,
                       "H1": cfg.H1, "H2": cfg.H2, "T": T},
        "gamma_cells": len(cells), "gamma_ard1_cells": len(ard1),
        "transitions_checked": len(cells) * T,
        "repairs": repairs, "replacements": replacements,
        "maximum_errors": maxima,
        "global_violations": global_failures, "violations": failures,
        "timing": {"validation_wall_seconds": time.perf_counter() - started},
    }
    if raise_on_failure and not report["valid"]:
        raise AssertionError(
            f"Gamma replay failed: {len(global_failures)} global and "
            f"{len(failures)} state violations."
        )
    return report


def _read(path: Path) -> dict:
    if path.suffix.lower() in {".yaml", ".yml"}:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError("Gamma replay supports YAML and JSON files.")
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a mapping.")
    return value


def validate_gamma_replay_files(input_path: str | Path, result_path: str | Path,
                                report_path: str | Path | None = None,
                                **kwargs) -> dict[str, Any]:
    """Load an input/result pair, replay it, and optionally save the report."""
    input_file, result_file = Path(input_path), Path(result_path)
    report = validate_gamma_replay_schedule(
        load_config(_read(input_file)), _read(result_file), **kwargs)
    if report_path is not None:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.suffix.lower() in {".yaml", ".yml"}:
            destination.write_text(yaml.safe_dump(report, sort_keys=False),
                                   encoding="utf-8")
        elif destination.suffix.lower() == ".json":
            destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        else:
            raise ValueError("report path must end in YAML or JSON.")
    return report