"""Monte Carlo validator of the Gamma model and the dispatcher of the Gamma validators.

Spec, section "Validators of the Gamma Model". The saved schedule stays fixed. In each
run, the initial damage, every mission increment and every replacement draw of a Gamma
cell are independent Gamma variables with the means and rates of the input; the sampled
damage follows the repair law of the cell. A run fails if a sampled damage exceeds the
threshold of its cell after some day. This is an empirical stress test; it does not
replace the analytical reliability bound.

Only Gamma cells are sampled: the other models do not specify one distribution. The
dispatcher runs the replay validator, the Monte Carlo validator or both, as the argument
``mode`` or the input block ``validation`` says. Indices of a report are 0-based array
indices.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta as beta_distribution

from fleet_management.config import FleetConfig, _integer, _number
from fleet_management.degradation_model.gamma_utils.gamma_replay_validator import (
    _binary,
    _check_report_path,
    _check_tolerance,
    _event,
    _finish,
    _increments,
    _read_pair,
    gamma_cells,
    validate_gamma_replay_schedule,
)
from fleet_management.validation.tolerance import (
    binary_tolerance,
    report_tolerance,
    resolve_tolerance,
)


VALIDATION_MODES = ("deterministic", "stochastic", "both")

# Defaults of the dispatcher; the keys are those of the input block ``validation``.
_SETTING_DEFAULTS = {
    "mode": "deterministic",
    "repetitions": 100_000,
    "random_seed": 20260908,
    "maximum_schedule_failure_rate": None,
    "confidence_level": 0.95,
    "batch_size": 20_000,
    "tolerance": None,                                # None: the default per check
}


def _one_sided_binomial_upper(
    failures: int,
    repetitions: int,
    confidence_level: float,
) -> float:
    """Exact Clopper--Pearson upper confidence bound for a binomial rate."""
    if failures >= repetitions:
        return 1.0
    return float(
        beta_distribution.ppf(
            confidence_level,
            failures + 1,
            repetitions - failures,
        )
    )


def _draw(rng: np.random.Generator, mean: float, rate: float, size: int) -> np.ndarray:
    """``size`` draws of a Gamma damage of mean ``mean`` and rate ``rate`` (spec, steps 1 and 2).

    A zero mean gives zeros. A damage without a rate (``rate`` is ``NaN``) is the constant
    of its mean.
    """
    if mean <= 0.0:
        return np.zeros(size)
    if math.isnan(rate):
        return np.full(size, float(mean))
    return rng.gamma(shape=mean * rate, scale=1.0 / rate, size=size)


# ===========================================================================
# Monte Carlo validator
# ===========================================================================
def validate_gamma_stochastic_schedule(
    cfg: FleetConfig,
    result: dict,
    *,
    repetitions: int = 100_000,
    random_seed: int = 20260908,
    maximum_schedule_failure_rate: float | None = None,
    confidence_level: float = 0.95,
    batch_size: int = 20_000,
    tolerance: float | None = None,
    raise_on_failure: bool = False,
) -> dict[str, Any]:
    """Replay one fixed schedule under independent exact Gamma realizations.

    Under ARD1 the initial damage of a cell is the sum of two independent terms of means
    ``anchor_mu_0`` and ``mu_0 - anchor_mu_0``, and the repair law is
    ``D <- g + (1 - rho)(D - g)``, ``g <- D``; under ARDinf it is ``D <- (1 - rho) D``.
    ``valid`` is true if no target ``maximum_schedule_failure_rate`` is given or the
    observed schedule failure rate is at most the target. Every failure rate comes with
    its one-sided Clopper--Pearson upper bound at ``confidence_level``.

    The tolerance only decides whether the saved binaries are binary: ``tolerance`` if it is
    a number, else the key ``tolerance`` of the input block ``validation``, else ``IntFeasTol``
    (``validation/tolerance.py``). The report key ``tolerance`` holds the value used, or the
    two Gurobi tolerances.
    """
    repetitions = _integer(repetitions, "repetitions")
    random_seed = _integer(random_seed, "random_seed")
    batch_size = _integer(batch_size, "batch_size")
    confidence_level = _number(confidence_level, "confidence_level")
    if maximum_schedule_failure_rate is not None:
        maximum_schedule_failure_rate = _number(
            maximum_schedule_failure_rate, "maximum_schedule_failure_rate")
    if random_seed < 0:
        raise ValueError("random_seed must be a non-negative integer.")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1).")
    if maximum_schedule_failure_rate is not None and not (
        0.0 <= maximum_schedule_failure_rate <= 1.0
    ):
        raise ValueError("maximum_schedule_failure_rate must lie in [0, 1].")
    _check_tolerance(tolerance)
    single = resolve_tolerance(cfg, tolerance)
    tol_binary = binary_tolerance(cfg, single)

    started = time.perf_counter()
    F, M, L, T = cfg.F, cfg.M, cfg.L, cfg.T
    states = (F, L, T)
    cells = gamma_cells(cfg, result)
    models = np.asarray(result["model"]).astype(str)
    gamma_set = set(cells)
    excluded_cells = [
        {
            "i": i,
            "l": l,
            "component": cfg.component_names[l],
            "model": str(models[i, l]),
            "reason": "no unique sampling distribution is specified",
        }
        for i in range(F)
        for l in range(L)
        if (i, l) not in gamma_set
    ]

    x = _binary(result, "x", (F, M + 1, T), tol_binary)
    m = _binary(result, "m", states, tol_binary)
    r = _binary(result, "r", states, tol_binary, optional=True)
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
            if use_latch:                             # two terms: the anchor and the rest
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
                    state = (
                        latch + remaining * (state - latch)
                        if use_latch else remaining * state
                    )
                    if use_latch:
                        latch = state.copy()
                else:
                    for j0 in np.flatnonzero(x[i, 1:, k]):
                        state += _draw(rng, float(mu_inc[i, l, j0, k]),
                                       float(beta_inc[i, l, j0, k]), size)

                failed = state > threshold
                step_failure_counts[i, l][k] += int(np.count_nonzero(failed))
                cell_failed |= failed
                batch_failed |= failed
                maximum_damage[i, l] = max(maximum_damage[i, l], float(np.max(state)))
                maximum_excess[i, l] = max(
                    maximum_excess[i, l], max(0.0, float(np.max(state - threshold))))

            cell_failure_counts[i, l] += int(np.count_nonzero(cell_failed))

        schedule_failures += int(np.count_nonzero(batch_failed))

    schedule_rate = schedule_failures / repetitions
    schedule_upper = _one_sided_binomial_upper(
        schedule_failures, repetitions, confidence_level)
    target = maximum_schedule_failure_rate
    observed_target_met = target is None or schedule_rate <= target
    confidence_target_met = target is None or schedule_upper <= target

    cell_results = []
    for i, l in cells:
        step_rates = step_failure_counts[i, l] / repetitions
        worst_step = int(np.argmax(step_rates))
        failures = cell_failure_counts[i, l]
        cell_results.append({
            "i": i,
            "l": l,
            "component": cfg.component_names[l],
            "repair_model": str(cfg.repair_model[i, l]),
            "threshold": float(cfg.tau[i, l]),
            "analytical_epsilon": float(cfg.epsilon[i, l]),
            "failed_replays": failures,
            "failure_rate": failures / repetitions,
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
        "tolerance": report_tolerance(cfg, single),
        "scope": "Gamma cells only",
        "complete_fleet_validation": not excluded_cells,
        "excluded_cells": excluded_cells,
        "schedule": {
            "failed_replays": schedule_failures,
            "failure_rate": schedule_rate,
            "failure_rate_upper_confidence_bound": schedule_upper,
            "maximum_failure_rate": target,
            "observed_target_met": bool(observed_target_met),
            "confidence_qualified_target_met": bool(confidence_target_met),
        },
        "cells": cell_results,
        "timing": {"validation_wall_seconds": time.perf_counter() - started},
    }
    return _finish(report, None, raise_on_failure)


# ===========================================================================
# Dispatcher
# ===========================================================================
def _settings(cfg: FleetConfig, **arguments) -> dict[str, Any]:
    """The settings of the dispatcher (spec, "None rule").

    A setting that is ``None`` takes the value of the key of the same name in the input
    block ``validation``, otherwise its default.
    """
    block = cfg.options.get("validation") or {}
    for key in block:
        if key not in _SETTING_DEFAULTS:
            raise ValueError(
                f"unknown key {key!r} in 'validation'; the keys are {list(_SETTING_DEFAULTS)}."
            )
    values = {}
    for key, default in _SETTING_DEFAULTS.items():
        argument = arguments.get(key)
        if argument is None:
            argument = block.get(key)                 # a null in the block counts as absent
        values[key] = argument if argument is not None else default
    if values["mode"] not in VALIDATION_MODES:
        raise ValueError(f"mode must be one of {VALIDATION_MODES} (got {values['mode']!r}).")
    return values


def validate_gamma_schedule(
    cfg: FleetConfig,
    result: dict,
    *,
    mode: str | None = None,
    repetitions: int | None = None,
    random_seed: int | None = None,
    maximum_schedule_failure_rate: float | None = None,
    confidence_level: float | None = None,
    batch_size: int | None = None,
    tolerance: float | None = None,
    raise_on_failure: bool = False,
) -> dict[str, Any]:
    """Run the replay validator, the Monte Carlo validator or both, as ``mode`` selects.

    Mode ``"both"`` returns the keys ``deterministic`` and ``stochastic``, and ``valid`` is
    true if both are valid. The sub-validators never raise on failure: the report is
    complete first, then ``raise_on_failure`` raises if it is not valid.
    """
    settings = _settings(
        cfg,
        mode=mode,
        repetitions=repetitions,
        random_seed=random_seed,
        maximum_schedule_failure_rate=maximum_schedule_failure_rate,
        confidence_level=confidence_level,
        batch_size=batch_size,
        tolerance=tolerance,
    )
    mode = settings.pop("mode")
    tolerance = settings.pop("tolerance")             # None: each validator resolves its own
    if mode == "deterministic":
        report = validate_gamma_replay_schedule(
            cfg, result, tolerance=tolerance, raise_on_failure=False)
        report["mode"] = "deterministic"
    elif mode == "stochastic":
        report = validate_gamma_stochastic_schedule(
            cfg, result, tolerance=tolerance, raise_on_failure=False, **settings)
    else:
        deterministic = validate_gamma_replay_schedule(
            cfg, result, tolerance=tolerance, raise_on_failure=False)
        stochastic = validate_gamma_stochastic_schedule(
            cfg, result, tolerance=tolerance, raise_on_failure=False, **settings)
        report = {
            "validator": "Gamma replay and Monte Carlo validators",
            "mode": "both",
            "valid": deterministic["valid"] and stochastic["valid"],
            "deterministic": deterministic,
            "stochastic": stochastic,
        }
    return _finish(report, None, raise_on_failure)


def validate_gamma_schedule_files(
    input_path: str | Path,
    result_path: str | Path,
    report_path: str | Path | None = None,
    **kwargs,
) -> dict[str, Any]:
    """File version (spec, "Common rules"): input and result paths; the report as YAML or JSON."""
    destination = _check_report_path(report_path)         # before any reading
    raise_on_failure = kwargs.pop("raise_on_failure", False)
    cfg, result = _read_pair(input_path, result_path)
    report = validate_gamma_schedule(cfg, result, **kwargs)
    return _finish(report, destination, raise_on_failure)     # write, then raise
