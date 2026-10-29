"""PROTOTYPE of steps C5 and C6 of the plan (scratch, written by the Task 1 agent to run the tests)."""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.stats import beta as beta_distribution

from fleet_management.config import FleetConfig, _integer, _number
from fleet_management.degradation_model.gamma_utils.gamma_replay_validator import (
    _binary, _check_report_path, _check_tolerance, _event, _finish, _increments, _read_pair,
    gamma_cells, validate_gamma_replay_schedule,
)

VALIDATION_MODES = ("deterministic", "stochastic", "both")
_SETTING_DEFAULTS = {"mode": "deterministic", "repetitions": 100_000, "random_seed": 20260908,
                     "maximum_schedule_failure_rate": None, "confidence_level": 0.95,
                     "batch_size": 20_000}


def _one_sided_binomial_upper(failures, repetitions, confidence_level):
    if failures >= repetitions:
        return 1.0
    return float(beta_distribution.ppf(confidence_level, failures + 1, repetitions - failures))


def _draw(rng, mean, rate, size):
    if mean <= 0.0:
        return np.zeros(size)
    if math.isnan(rate):
        return np.full(size, float(mean))
    return rng.gamma(shape=mean * rate, scale=1.0 / rate, size=size)


def _settings(cfg, **arguments):
    block = cfg.options.get("validation") or {}
    for key in block:
        if key not in _SETTING_DEFAULTS:
            raise ValueError(f"unknown key {key!r} in 'validation'; the keys are {list(_SETTING_DEFAULTS)}.")
    settings = {key: (arguments[key] if arguments[key] is not None else block.get(key, default))
                for key, default in _SETTING_DEFAULTS.items()}
    if settings["mode"] not in VALIDATION_MODES:
        raise ValueError(f"mode must be one of {VALIDATION_MODES} (got {settings['mode']!r}).")
    return settings


def validate_gamma_stochastic_schedule(cfg, result, *, repetitions=100_000, random_seed=20260908,
                                       maximum_schedule_failure_rate=None, confidence_level=0.95,
                                       batch_size=20_000, tolerance=1e-8, raise_on_failure=False):
    repetitions = _integer(repetitions, "repetitions")
    random_seed = _integer(random_seed, "random_seed")
    batch_size = _integer(batch_size, "batch_size")
    confidence_level = _number(confidence_level, "confidence_level")
    if maximum_schedule_failure_rate is not None:
        maximum_schedule_failure_rate = _number(maximum_schedule_failure_rate,
                                                "maximum_schedule_failure_rate")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1).")
    if maximum_schedule_failure_rate is not None and not (0.0 <= maximum_schedule_failure_rate <= 1.0):
        raise ValueError("maximum_schedule_failure_rate must lie in [0, 1].")
    _check_tolerance(tolerance)

    started = time.perf_counter()
    F, L, M, T = cfg.F, cfg.L, cfg.M, cfg.T
    cells = gamma_cells(cfg, result)
    names = np.asarray(result["model"]).astype(str)
    excluded_cells = [
        {"i": i, "l": l, "component": cfg.component_names[l], "model": str(names[i, l]),
         "reason": "no unique sampling distribution is specified"}
        for i in range(F) for l in range(L) if (i, l) not in cells]
    x = _binary(result, "x", (F, M + 1, T), tolerance)
    m = _binary(result, "m", (F, L, T), tolerance)
    r = _binary(result, "r", (F, L, T), tolerance, optional=True)
    mu_inc, beta_inc = _increments(cfg)

    rng = np.random.default_rng(random_seed)
    schedule_failures = 0
    cell_failure_counts = {cell: 0 for cell in cells}
    step_failure_counts = {cell: np.zeros(T, dtype=np.int64) for cell in cells}
    maximum_damage = {cell: 0.0 for cell in cells}
    maximum_excess = {cell: 0.0 for cell in cells}

    for first in range(0, repetitions, batch_size):
        size = min(batch_size, repetitions - first)
        batch_failed = np.zeros(size, dtype=bool)
        for i, l in cells:
            use_latch = str(cfg.repair_model[i, l]) == "ard1"
            beta_0, beta_new = float(cfg.gamma_beta_0[i, l]), float(cfg.gamma_beta_new[i, l])
            if use_latch:
                g_0 = float(cfg.anchor_mu_0[i, l])
                latch = _draw(rng, g_0, beta_0, size)
                state = latch + _draw(rng, float(cfg.mu_0[i, l]) - g_0, beta_0, size)
            else:
                latch = np.zeros(size)
                state = _draw(rng, float(cfg.mu_0[i, l]), beta_0, size)
            cell_failed = np.zeros(size, dtype=bool)
            remaining = max(0.0, 1.0 - float(cfg.rho[i, l]))
            threshold = float(cfg.tau[i, l])
            for k in range(T):
                event = _event(x, m, r, i, l, k)
                if event == "replacement":
                    state = _draw(rng, float(cfg.replacement_mu[i, l]), beta_new, size)
                    if use_latch:
                        latch = state.copy()
                elif event == "repair":
                    state = latch + remaining * (state - latch) if use_latch else remaining * state
                    if use_latch:
                        latch = state.copy()
                else:
                    for j0 in np.flatnonzero(x[i, 1:, k]):
                        state += _draw(rng, float(mu_inc[i, l, j0, k]), float(beta_inc[i, l, j0, k]), size)
                failed = state > threshold
                step_failure_counts[i, l][k] += int(np.count_nonzero(failed))
                cell_failed |= failed
                batch_failed |= failed
                maximum_damage[i, l] = max(maximum_damage[i, l], float(np.max(state)))
                maximum_excess[i, l] = max(maximum_excess[i, l], max(0.0, float(np.max(state - threshold))))
            cell_failure_counts[i, l] += int(np.count_nonzero(cell_failed))
        schedule_failures += int(np.count_nonzero(batch_failed))

    schedule_rate = schedule_failures / repetitions
    schedule_upper = _one_sided_binomial_upper(schedule_failures, repetitions, confidence_level)
    target = maximum_schedule_failure_rate
    observed_target_met = target is None or schedule_rate <= target
    confidence_target_met = target is None or schedule_upper <= target

    cell_results = []
    for i, l in cells:
        step_rates = step_failure_counts[i, l] / repetitions
        worst_step = int(np.argmax(step_rates))
        failures = cell_failure_counts[i, l]
        cell_results.append({
            "i": i, "l": l, "component": cfg.component_names[l],
            "repair_model": str(cfg.repair_model[i, l]),
            "threshold": float(cfg.tau[i, l]), "analytical_epsilon": float(cfg.epsilon[i, l]),
            "failed_replays": failures, "failure_rate": failures / repetitions,
            "failure_rate_upper_confidence_bound": _one_sided_binomial_upper(
                failures, repetitions, confidence_level),
            "worst_time_step": worst_step,
            "worst_step_failures": int(step_failure_counts[i, l][worst_step]),
            "worst_step_failure_rate": float(step_rates[worst_step]),
            "worst_step_upper_confidence_bound": _one_sided_binomial_upper(
                int(step_failure_counts[i, l][worst_step]), repetitions, confidence_level),
            "maximum_sampled_damage": maximum_damage[i, l],
            "maximum_threshold_excess": maximum_excess[i, l],
        })

    report = {
        "validator": "Gamma Monte Carlo validator",
        "mode": "stochastic",
        "valid": bool(observed_target_met),
        "repetitions": repetitions,
        "random_seed": random_seed,
        "maximum_schedule_failure_rate": target,
        "confidence_level": confidence_level,
        "batch_size": batch_size,
        "tolerance": tolerance,
        "scope": "Gamma cells only",
        "complete_fleet_validation": not excluded_cells,
        "excluded_cells": excluded_cells,
        "schedule": {
            "failed_replays": schedule_failures, "failure_rate": schedule_rate,
            "failure_rate_upper_confidence_bound": schedule_upper,
            "maximum_failure_rate": target, "observed_target_met": bool(observed_target_met),
            "confidence_qualified_target_met": bool(confidence_target_met),
        },
        "cells": cell_results,
        "timing": {"validation_wall_seconds": time.perf_counter() - started},
    }
    return _finish(report, None, raise_on_failure)


def validate_gamma_schedule(cfg, result, *, mode=None, repetitions=None, random_seed=None,
                            maximum_schedule_failure_rate=None, confidence_level=None,
                            batch_size=None, tolerance=1e-8, raise_on_failure=False):
    settings = _settings(cfg, mode=mode, repetitions=repetitions, random_seed=random_seed,
                         maximum_schedule_failure_rate=maximum_schedule_failure_rate,
                         confidence_level=confidence_level, batch_size=batch_size)
    mode = settings.pop("mode")
    if mode == "deterministic":
        report = validate_gamma_replay_schedule(cfg, result, tolerance=tolerance, raise_on_failure=False)
        report["mode"] = "deterministic"
    elif mode == "stochastic":
        report = validate_gamma_stochastic_schedule(cfg, result, tolerance=tolerance,
                                                    raise_on_failure=False, **settings)
    else:
        deterministic = validate_gamma_replay_schedule(cfg, result, tolerance=tolerance,
                                                       raise_on_failure=False)
        stochastic = validate_gamma_stochastic_schedule(cfg, result, tolerance=tolerance,
                                                        raise_on_failure=False, **settings)
        report = {"validator": "Gamma replay and Monte Carlo validators", "mode": "both",
                  "valid": deterministic["valid"] and stochastic["valid"],
                  "deterministic": deterministic, "stochastic": stochastic}
    return _finish(report, None, raise_on_failure)


def validate_gamma_schedule_files(input_path, result_path, report_path=None, **kwargs):
    destination = _check_report_path(report_path)
    raise_on_failure = kwargs.pop("raise_on_failure", False)
    cfg, result = _read_pair(input_path, result_path)
    report = validate_gamma_schedule(cfg, result, **kwargs)
    return _finish(report, destination, raise_on_failure)
