"""Fixed-schedule Monte Carlo validation for modular Gamma components.

The optimizer's assignments and interventions remain fixed. Independent exact
Gamma initial states, mission increments and replacement states are sampled,
then propagated pathwise through the selected ARD repair dynamics. This is an
empirical stress test; it does not replace the analytical reliability bound.

Remaining-life cells are deliberately not sampled here. Their formulation may
specify moments, support or a CGF for distribution-free bounds, but those data
do not identify one unique probability distribution.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.stats import beta as beta_distribution
import yaml

from fleet_management.config import FleetConfig, load_config
from fleet_management.degradation_model.gamma_utils.gamma_replay_validator import (
    _binary,
    _phase,
    _read,
    validate_gamma_replay_schedule,
)


VALIDATION_MODES = ("deterministic", "stochastic", "both")


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


def _gamma_sample(
    rng: np.random.Generator,
    expected_damage: float,
    rate: float,
    size: int,
) -> np.ndarray:
    """Sample Gamma(mean*rate, rate), including the zero-state limit."""
    mean = float(expected_damage)
    beta = float(rate)
    if mean < 0.0 or beta <= 0.0:
        raise ValueError(
            f"invalid Gamma sampling parameters: mean={mean}, rate={beta}"
        )
    if mean == 0.0:
        return np.zeros(size, dtype=float)
    return rng.gamma(shape=mean * beta, scale=1.0 / beta, size=size)


def _settings(
    cfg: FleetConfig,
    *,
    mode: str | None,
    repetitions: int | None,
    random_seed: int | None,
    maximum_schedule_failure_rate: float | None,
    confidence_level: float | None,
    batch_size: int | None,
) -> dict[str, Any]:
    written = cfg.raw.get("validation", {})
    if written is None:
        written = {}
    if not isinstance(written, dict):
        raise TypeError("top-level 'validation' must be a mapping.")

    selected_mode = str(
        written.get("mode", "deterministic") if mode is None else mode
    ).strip().lower()
    if selected_mode not in VALIDATION_MODES:
        raise ValueError(
            f"validation mode must be one of {VALIDATION_MODES}; "
            f"got {selected_mode!r}."
        )

    values = {
        "mode": selected_mode,
        "repetitions": int(
            written.get("repetitions", 100_000)
            if repetitions is None else repetitions
        ),
        "random_seed": int(
            written.get("random_seed", 20260908)
            if random_seed is None else random_seed
        ),
        "maximum_schedule_failure_rate": (
            written.get("maximum_schedule_failure_rate")
            if maximum_schedule_failure_rate is None
            else maximum_schedule_failure_rate
        ),
        "confidence_level": float(
            written.get("confidence_level", 0.95)
            if confidence_level is None else confidence_level
        ),
        "batch_size": int(
            written.get("batch_size", 20_000)
            if batch_size is None else batch_size
        ),
    }
    if values["repetitions"] <= 0:
        raise ValueError("validation repetitions must be positive.")
    if values["batch_size"] <= 0:
        raise ValueError("validation batch_size must be positive.")
    if not 0.0 < values["confidence_level"] < 1.0:
        raise ValueError("validation confidence_level must lie in (0, 1).")
    target = values["maximum_schedule_failure_rate"]
    if target is not None:
        target = float(target)
        if not 0.0 <= target <= 1.0:
            raise ValueError(
                "maximum_schedule_failure_rate must lie in [0, 1]."
            )
        values["maximum_schedule_failure_rate"] = target
    return values


def validate_gamma_stochastic_schedule(
    cfg: FleetConfig,
    result: dict,
    *,
    repetitions: int = 100_000,
    random_seed: int = 20260908,
    maximum_schedule_failure_rate: float | None = None,
    confidence_level: float = 0.95,
    batch_size: int = 20_000,
    tolerance: float = 1e-8,
    raise_on_failure: bool = False,
) -> dict[str, Any]:
    """Replay one fixed schedule under independent exact Gamma realizations."""
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
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")

    started = time.perf_counter()
    F, L, M, T = cfg.F, cfg.L, cfg.M, cfg.T
    states = (F, L, T)
    gamma_cells = [
        (i, l)
        for i in range(F)
        for l in range(L)
        if str(cfg.model[i, l]) == "gamma"
    ]
    if not gamma_cells:
        raise ValueError("configuration contains no Gamma cells.")
    excluded_cells = [
        {
            "i": i,
            "l": l,
            "component": cfg.component_names[l],
            "model": str(cfg.model[i, l]),
            "reason": "no unique sampling distribution is specified",
        }
        for i in range(F)
        for l in range(L)
        if str(cfg.model[i, l]) != "gamma"
    ]

    x = _binary(result, "x", (F, M + 1, T), tolerance)
    m = _binary(result, "m", states, tolerance)
    r = _binary(result, "r", states, tolerance, optional=True)
    if np.max(np.sum(x, axis=1) - 1.0) > tolerance:
        raise ValueError("result violates the vehicle-assignment constraint.")
    if np.max(np.abs(np.sum(x[:, 1:, :], axis=0) - 1.0)) > tolerance:
        raise ValueError("result violates mission demand.")

    beta_op = np.asarray(cfg.gamma_beta, dtype=float)
    beta_tr = (
        None
        if cfg.gamma_beta_trans is None
        else np.asarray(cfg.gamma_beta_trans, dtype=float)
    )
    # A rate is irrelevant for a degenerate zero seed. Configuration validation
    # requires an exact rate for every nonzero seed; the fallback keeps the
    # sampler robust for older zero-seed inputs.
    beta_0 = (
        np.ones((F, L), dtype=float)
        if cfg.gamma_beta_0 is None
        else np.asarray(cfg.gamma_beta_0, dtype=float)
    )
    beta_new = (
        np.ones((F, L), dtype=float)
        if cfg.gamma_beta_new is None
        else np.asarray(cfg.gamma_beta_new, dtype=float)
    )
    mu_op = np.asarray(cfg.mu, dtype=float)
    mu_tr = (
        None if cfg.mu_trans is None else np.asarray(cfg.mu_trans, dtype=float)
    )

    rng = np.random.default_rng(int(random_seed))
    schedule_failures = 0
    cell_failure_counts = {cell: 0 for cell in gamma_cells}
    step_failure_counts = {
        cell: np.zeros(T, dtype=np.int64) for cell in gamma_cells
    }
    maximum_damage = {cell: 0.0 for cell in gamma_cells}
    maximum_excess = {cell: 0.0 for cell in gamma_cells}

    for first in range(0, repetitions, batch_size):
        size = min(batch_size, repetitions - first)
        batch_failed = np.zeros(size, dtype=bool)

        for i, l in gamma_cells:
            state = _gamma_sample(
                rng,
                float(cfg.mu_0[i, l]),
                float(beta_0[i, l]),
                size,
            )
            latch = np.zeros(size, dtype=float)
            cell_failed = np.zeros(size, dtype=bool)
            rho = float(cfg.rho[i, l])
            remaining = 1.0 - rho
            use_latch = str(cfg.repair_model[i, l]) == "ard1"
            threshold = float(cfg.tau[i, l])

            for k in range(T):
                do_repair = bool(m[i, l, k])
                do_replace = bool(r[i, l, k])
                missions = np.flatnonzero(x[i, 1:, k])
                if do_repair and do_replace:
                    raise ValueError(f"repair and replacement at {(i, l, k)}")
                if (do_repair or do_replace) and missions.size:
                    raise ValueError(f"maintenance and mission at {(i, l, k)}")
                if (do_repair or do_replace) and not x[i, 0, k]:
                    raise ValueError(f"maintenance without access at {(i, l, k)}")

                if do_replace:
                    state = _gamma_sample(
                        rng,
                        float(cfg.replacement_mu[i, l]),
                        float(beta_new[i, l]),
                        size,
                    )
                    if use_latch:
                        latch = state.copy()
                elif do_repair:
                    state = (
                        latch + remaining * (state - latch)
                        if use_latch else remaining * state
                    )
                    if use_latch:
                        latch = state.copy()
                else:
                    for j in missions:
                        mean = _phase(
                            mu_op, mu_tr, i, l, int(j), k, cfg.H1, cfg.H2
                        )
                        rate = _phase(
                            beta_op, beta_tr, i, l, int(j), k, cfg.H1, cfg.H2
                        )
                        state += _gamma_sample(rng, mean, rate, size)

                failed = state > threshold
                step_failure_counts[i, l][k] += int(np.count_nonzero(failed))
                cell_failed |= failed
                batch_failed |= failed
                maximum_damage[i, l] = max(
                    maximum_damage[i, l], float(np.max(state))
                )
                maximum_excess[i, l] = max(
                    maximum_excess[i, l],
                    max(0.0, float(np.max(state - threshold))),
                )

            cell_failure_counts[i, l] += int(np.count_nonzero(cell_failed))

        schedule_failures += int(np.count_nonzero(batch_failed))

    schedule_rate = schedule_failures / repetitions
    schedule_upper = _one_sided_binomial_upper(
        schedule_failures, repetitions, confidence_level
    )
    target = maximum_schedule_failure_rate
    observed_target_met = target is None or schedule_rate <= target
    confidence_target_met = target is None or schedule_upper <= target

    cell_results = []
    for i, l in gamma_cells:
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
                failures, repetitions, confidence_level
            ),
            "worst_time_step": worst_step,
            "worst_step_failures": int(step_failure_counts[i, l][worst_step]),
            "worst_step_failure_rate": float(step_rates[worst_step]),
            "worst_step_upper_confidence_bound": _one_sided_binomial_upper(
                int(step_failure_counts[i, l][worst_step]),
                repetitions,
                confidence_level,
            ),
            "maximum_sampled_damage": maximum_damage[i, l],
            "maximum_threshold_excess": maximum_excess[i, l],
        })

    report = {
        "validator": "Gamma fixed-schedule Monte Carlo",
        "mode": "stochastic",
        "valid": bool(observed_target_met),
        "scope": "Gamma cells only",
        "complete_fleet_validation": not excluded_cells,
        "interpretation": (
            "Empirical stress test of a fixed schedule under exact Gamma "
            "realizations; not an analytical reliability proof."
        ),
        "failure_event": (
            "At least one sampled Gamma state exceeds its component threshold "
            "after a transition."
        ),
        "solver_status": str(result.get("status", "unknown")),
        "dimensions": {
            "F": F,
            "M": M,
            "L": L,
            "H1": cfg.H1,
            "H2": cfg.H2,
            "T": T,
        },
        "gamma_cells": len(gamma_cells),
        "excluded_cells": excluded_cells,
        "repetitions": repetitions,
        "random_seed": int(random_seed),
        "batch_size": batch_size,
        "confidence_level": confidence_level,
        "schedule": {
            "failed_replays": schedule_failures,
            "failure_rate": schedule_rate,
            "failure_rate_upper_confidence_bound": schedule_upper,
            "maximum_failure_rate": target,
            "observed_target_met": bool(observed_target_met),
            "confidence_qualified_target_met": bool(confidence_target_met),
        },
        "cells": cell_results,
        "timing": {
            "validation_wall_seconds": time.perf_counter() - started,
        },
    }
    if raise_on_failure and not report["valid"]:
        raise AssertionError(
            "stochastic validation exceeded maximum_schedule_failure_rate: "
            f"observed {schedule_rate:.6g}, target {target:.6g}."
        )
    return report


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
    tolerance: float = 1e-8,
    raise_on_failure: bool = False,
) -> dict[str, Any]:
    """Run deterministic replay, stochastic replay, or both."""
    settings = _settings(
        cfg,
        mode=mode,
        repetitions=repetitions,
        random_seed=random_seed,
        maximum_schedule_failure_rate=maximum_schedule_failure_rate,
        confidence_level=confidence_level,
        batch_size=batch_size,
    )
    selected = settings.pop("mode")
    if selected == "deterministic":
        report = validate_gamma_replay_schedule(
            cfg,
            result,
            tolerance=tolerance,
            raise_on_failure=raise_on_failure,
        )
        report["mode"] = "deterministic"
        return report
    if selected == "stochastic":
        return validate_gamma_stochastic_schedule(
            cfg,
            result,
            tolerance=tolerance,
            raise_on_failure=raise_on_failure,
            **settings,
        )

    deterministic = validate_gamma_replay_schedule(
        cfg,
        result,
        tolerance=tolerance,
        raise_on_failure=raise_on_failure,
    )
    stochastic = validate_gamma_stochastic_schedule(
        cfg,
        result,
        tolerance=tolerance,
        raise_on_failure=raise_on_failure,
        **settings,
    )
    return {
        "validator": "Gamma deterministic and stochastic validation",
        "mode": "both",
        "valid": bool(deterministic["valid"] and stochastic["valid"]),
        "deterministic": deterministic,
        "stochastic": stochastic,
    }


def _save_report(report: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() in {".yaml", ".yml"}:
        destination.write_text(
            yaml.safe_dump(report, sort_keys=False), encoding="utf-8"
        )
    elif destination.suffix.lower() == ".json":
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        raise ValueError("validation report path must end in YAML or JSON.")


def validate_gamma_schedule_files(
    input_path: str | Path,
    result_path: str | Path,
    report_path: str | Path | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Load an input/result pair and run the selected validation mode."""
    cfg = load_config(_read(Path(input_path)))
    report = validate_gamma_schedule(cfg, _read(Path(result_path)), **kwargs)
    if report_path is not None:
        _save_report(report, Path(report_path))
    return report
