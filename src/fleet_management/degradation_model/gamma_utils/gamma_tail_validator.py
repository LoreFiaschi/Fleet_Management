"""Exact post-solve validation for modular Gamma tail-bound schedules.

The validator is independent of the optimization formulation.  It replays the
saved decisions using the original shape-rate Gamma inputs, evaluates the exact
tail of every accumulated history with the Moschopoulos series, and checks

    exact tail upper bound <= solver Gamma bound <= epsilon.

Initial and replacement states are alternative history seeds.  An ARD-inf
repair with remaining fraction ``c = 1-rho`` transforms each surviving term as

    Gamma(A, beta) -> Gamma(A, beta/c).

The shape is unchanged and the mean contracts by ``c``.  A replacement discards
the complete previous history.  No Gurobi objects or constraints are inspected.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import gamma as gamma_distribution
import yaml

from fleet_management.config import FleetConfig, load_config
from fleet_management.degradation_model.gamma_utils.gamma_tail_bound import (
    moschopoulos_tail_probability,
)


@dataclass(frozen=True)
class ExactGammaTerm:
    """One independent Gamma term in an exact accumulated history."""

    shape: float
    rate: float
    source: str
    created_step: int
    repairs: int = 0

    @property
    def mean(self) -> float:
        return self.shape / self.rate

    def after_ardinf(self, remaining_fraction: float) -> "ExactGammaTerm":
        return ExactGammaTerm(
            shape=self.shape,
            rate=self.rate / remaining_fraction,
            source=self.source,
            created_step=self.created_step,
            repairs=self.repairs + 1,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "rate": self.rate,
            "mean": self.mean,
            "source": self.source,
            "created_step": self.created_step,
            "repairs": self.repairs,
        }


def _required_array(result: dict, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if result.get(name) is None:
        raise ValueError(f"result does not contain required array {name!r}.")
    value = np.asarray(result[name], dtype=float)
    if value.shape != shape:
        raise ValueError(f"result {name!r} has shape {value.shape}; expected {shape}.")
    if np.any(~np.isfinite(value)):
        raise ValueError(f"result {name!r} contains non-finite values.")
    return value


def _binary_decisions(value: np.ndarray, name: str, tolerance: float) -> np.ndarray:
    rounded = np.rint(value)
    error = float(np.max(np.abs(value - rounded)))
    bounds_error = max(float(np.max(-value)), float(np.max(value - 1.0)), 0.0)
    if max(error, bounds_error) > tolerance:
        raise ValueError(
            f"result decision array {name!r} is not binary within tolerance; "
            f"maximum error={max(error, bounds_error):.3e}."
        )
    return rounded.astype(np.int64)


def _phase_value(operating, transitory, i: int, l: int, j: int, k: int,
                 H1: int, H2: int) -> float:
    if k < H1:
        if transitory is not None:
            return float(transitory[i, l, j, k])
        return float(operating[i, l, j, k % H2])
    return float(operating[i, l, j, (k - H1) % H2])


def _seed_term(mean: float, rate: float | None, source: str,
               step: int) -> list[ExactGammaTerm]:
    if mean <= 0.0:
        return []
    if rate is None or not np.isfinite(rate) or rate <= 0.0:
        raise ValueError(f"positive {source} mean requires a finite positive rate.")
    return [ExactGammaTerm(mean * rate, rate, source, step)]


def _exact_tail(history: list[ExactGammaTerm], threshold: float,
                convolution_tolerance: float,
                max_series_terms: int):
    if not history:
        return {
            "estimate": 0.0,
            "upper_bound": 0.0,
            "remaining_mass": 0.0,
            "series_terms": 0,
        }
    convolution = moschopoulos_tail_probability(
        shapes=[term.shape for term in history],
        rates=[term.rate for term in history],
        threshold=threshold,
        tolerance=convolution_tolerance,
        max_series_terms=max_series_terms,
    )
    return {
        "estimate": convolution.estimate,
        "upper_bound": convolution.upper_bound,
        "remaining_mass": convolution.remaining_mass,
        "series_terms": convolution.series_terms,
    }


def validate_gamma_tail_bound_schedule(
    cfg: FleetConfig,
    result: dict,
    *,
    tolerance: float = 1e-8,
    convolution_tolerance: float = 1e-12,
    max_series_terms: int = 100_000,
    include_steps: bool = False,
    raise_on_failure: bool = False,
) -> dict[str, Any]:
    """Replay and certify every Gamma cell in a modular solver result.

    The exact probability used for conservativeness is the numerical
    convolution's certified upper bound, not only its partial-series estimate.
    A failed signed check sets ``report["valid"]`` to ``False`` and, when
    ``raise_on_failure`` is true, raises ``AssertionError``.
    """

    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    if not 0.0 < convolution_tolerance < 1.0:
        raise ValueError("convolution_tolerance must lie in (0, 1).")
    if max_series_terms <= 0:
        raise ValueError("max_series_terms must be positive.")

    F, L, M, T = cfg.F, cfg.L, cfg.M, cfg.T
    gamma_cells = [
        (i, l)
        for i in range(F)
        for l in range(L)
        if str(cfg.model[i, l]) == "gamma"
    ]
    if not gamma_cells:
        raise ValueError("configuration contains no Gamma cells to validate.")

    if result.get("x") is None:
        raise ValueError("result contains no incumbent solution arrays.")
    x = _binary_decisions(
        _required_array(result, "x", (F, M + 1, T)), "x", tolerance
    )
    m = _binary_decisions(
        _required_array(result, "m", (F, L, T)), "m", tolerance
    )
    r = _binary_decisions(
        _required_array(result, "r", (F, L, T)), "r", tolerance
    )
    saved_mu = _required_array(result, "mu", (F, L, T))
    saved_shape = _required_array(result, "gamma_shape_bound", (F, L, T))
    saved_tail = _required_array(result, "gamma_tail_bound", (F, L, T))
    common_rates = _required_array(result, "gamma_beta_bound", (F, L))

    if cfg.gamma_beta is None:
        raise ValueError("Gamma validation requires exact operating gamma_beta rates.")
    beta_operating = np.asarray(cfg.gamma_beta, dtype=float)
    beta_transitory = (
        None
        if cfg.gamma_beta_trans is None
        else np.asarray(cfg.gamma_beta_trans, dtype=float)
    )

    global_violations: list[str] = []
    if str(result.get("backend", "")) != "modular":
        global_violations.append("result backend is not modular")
    if int(result.get("T", -1)) != T:
        global_violations.append("result horizon T does not match configuration")
    if cfg.gamma_beta_bound is not None:
        beta_error = max(
            abs(common_rates[i, l] - float(cfg.gamma_beta_bound[i, l]))
            for i, l in gamma_cells
        )
        if beta_error > tolerance:
            global_violations.append(
                f"saved common rates differ from configuration by {beta_error:.3e}"
            )

    steps: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    cell_histories: list[dict[str, Any]] = []
    minimum_margin = float("inf")
    minimum_slack = float("inf")
    maximum_mean_error = 0.0
    maximum_saved_tail_error = 0.0
    maximum_remaining_mass = 0.0
    maximum_terms = 0
    repairs = 0
    replacements = 0
    worst_margin_step: dict[str, Any] | None = None
    worst_slack_step: dict[str, Any] | None = None

    for i, l in gamma_cells:
        beta_0 = None if cfg.gamma_beta_0 is None else float(cfg.gamma_beta_0[i, l])
        beta_new = (
            None if cfg.gamma_beta_new is None else float(cfg.gamma_beta_new[i, l])
        )
        history = _seed_term(float(cfg.mu_0[i, l]), beta_0, "initial", -1)
        events: list[dict[str, Any]] = []

        for k in range(T):
            repair = bool(m[i, l, k])
            replacement = bool(r[i, l, k])
            mission_indices = np.flatnonzero(x[i, 1:, k])
            reasons: list[str] = []

            if repair and replacement:
                reasons.append("repair and replacement selected simultaneously")
            if (repair or replacement) and mission_indices.size:
                reasons.append("maintenance and mission selected simultaneously")

            mission: int | None = None
            if replacement:
                replacements += 1
                history = _seed_term(
                    float(cfg.replacement_mu[i, l]), beta_new, "replacement", k
                )
                event = "replacement"
            elif repair:
                repairs += 1
                remaining = 1.0 - float(cfg.rho[i, l])
                if remaining < -tolerance or remaining > 1.0 + tolerance:
                    reasons.append("repair remaining fraction lies outside [0,1]")
                elif remaining <= tolerance:
                    history = []
                else:
                    history = [term.after_ardinf(remaining) for term in history]
                event = "repair"
            elif mission_indices.size:
                if mission_indices.size != 1:
                    reasons.append("more than one mission selected")
                mission = int(mission_indices[0])
                mean_increment = _phase_value(
                    cfg.mu, cfg.mu_trans, i, l, mission, k, cfg.H1, cfg.H2
                )
                rate_increment = _phase_value(
                    beta_operating,
                    beta_transitory,
                    i,
                    l,
                    mission,
                    k,
                    cfg.H1,
                    cfg.H2,
                )
                history.append(
                    ExactGammaTerm(
                        mean_increment * rate_increment,
                        rate_increment,
                        f"mission_{mission + 1}",
                        k,
                    )
                )
                event = "mission"
            else:
                event = "idle"

            exact = _exact_tail(
                history,
                float(cfg.tau[i, l]),
                convolution_tolerance,
                max_series_terms,
            )
            exact_mean = float(sum(term.mean for term in history))
            mean_error = abs(float(saved_mu[i, l, k]) - exact_mean)
            bound_shape = float(saved_shape[i, l, k])
            bound_rate = float(common_rates[i, l])
            if bound_shape < -tolerance or bound_rate <= 0.0:
                reasons.append("invalid saved bounding shape or rate")
                calculated_bound_tail = float("nan")
            elif bound_shape <= 0.0:
                calculated_bound_tail = 0.0
            else:
                calculated_bound_tail = float(
                    gamma_distribution.sf(
                        float(cfg.tau[i, l]),
                        a=bound_shape,
                        scale=1.0 / bound_rate,
                    )
                )
            tail_serialization_error = abs(
                float(saved_tail[i, l, k]) - calculated_bound_tail
            )
            margin = calculated_bound_tail - float(exact["upper_bound"])
            slack = float(cfg.epsilon[i, l]) - calculated_bound_tail

            if mean_error > tolerance:
                reasons.append(f"physical mean mismatch {mean_error:.3e}")
            if tail_serialization_error > tolerance:
                reasons.append(
                    f"saved bound-tail mismatch {tail_serialization_error:.3e}"
                )
            if margin < -tolerance:
                reasons.append(f"non-conservative tail margin {margin:.3e}")
            if slack < -tolerance:
                reasons.append(f"reliability violation {-slack:.3e}")

            history_snapshot = [term.as_dict() for term in history]
            step = {
                "i": i,
                "l": l,
                "k": k,
                "event": event,
                "mission": None if mission is None else mission + 1,
                "term_count": len(history),
                "exact_mean": exact_mean,
                "saved_mean": float(saved_mu[i, l, k]),
                "mean_error": mean_error,
                "exact_tail_estimate": float(exact["estimate"]),
                "exact_tail_upper_bound": float(exact["upper_bound"]),
                "convolution_remaining_mass": float(exact["remaining_mass"]),
                "convolution_series_terms": int(exact["series_terms"]),
                "bound_shape": bound_shape,
                "bound_rate": bound_rate,
                "bound_tail": calculated_bound_tail,
                "saved_bound_tail": float(saved_tail[i, l, k]),
                "conservativeness_margin": margin,
                "reliability_slack": slack,
                "passed": not reasons,
            }
            if include_steps:
                steps.append({**step, "history": history_snapshot})
            if reasons:
                violations.append({**step, "reasons": reasons, "history": history_snapshot})
            events.append({
                "k": k,
                "event": event,
                "mission": None if mission is None else mission + 1,
            })

            if margin < minimum_margin:
                minimum_margin = margin
                worst_margin_step = {**step, "history": history_snapshot}
            if slack < minimum_slack:
                minimum_slack = slack
                worst_slack_step = {**step, "history": history_snapshot}
            maximum_mean_error = max(maximum_mean_error, mean_error)
            maximum_saved_tail_error = max(
                maximum_saved_tail_error, tail_serialization_error
            )
            maximum_remaining_mass = max(
                maximum_remaining_mass, float(exact["remaining_mass"])
            )
            maximum_terms = max(maximum_terms, int(exact["series_terms"]))

        cell_histories.append({"i": i, "l": l, "events": events})

    valid = not global_violations and not violations
    report: dict[str, Any] = {
        "valid": valid,
        "solver_status": str(result.get("status", "unknown")),
        "backend": result.get("backend"),
        "dimensions": {
            "F": F,
            "M": M,
            "L": L,
            "H1": cfg.H1,
            "H2": cfg.H2,
            "T": T,
        },
        "gamma_cells": len(gamma_cells),
        "transitions_checked": len(gamma_cells) * T,
        "repairs": repairs,
        "replacements": replacements,
        "minimum_conservativeness_margin": minimum_margin,
        "minimum_reliability_slack": minimum_slack,
        "maximum_mean_error": maximum_mean_error,
        "maximum_saved_tail_error": maximum_saved_tail_error,
        "global_violations": global_violations,
        "violations": violations,
        "worst_conservativeness": worst_margin_step,
        "worst_reliability": worst_slack_step,
        "cell_histories": cell_histories,
        "numerics": {
            "convolution_tolerance": convolution_tolerance,
            "maximum_remaining_mass": maximum_remaining_mass,
            "maximum_series_terms_used": maximum_terms,
            "max_series_terms": max_series_terms,
        },
    }
    if include_steps:
        report["steps"] = steps

    if raise_on_failure and not valid:
        raise AssertionError(
            "exact Gamma schedule validation failed: "
            f"{len(global_violations)} global and {len(violations)} step violations; "
            f"minimum tail margin={minimum_margin:.3e}."
        )
    return report


def _read_mapping(path: Path) -> dict:
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    elif suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError("exact Gamma file validation currently supports YAML and JSON.")
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a mapping.")
    return value


def validate_gamma_tail_bound_files(
    input_path: str | Path,
    result_path: str | Path,
    report_path: str | Path | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Load a public input/result pair, validate it, and optionally save a report."""

    input_file = Path(input_path)
    result_file = Path(result_path)
    if not input_file.is_file():
        raise FileNotFoundError(input_file)
    if not result_file.is_file():
        raise FileNotFoundError(result_file)
    cfg = load_config(_read_mapping(input_file))
    report = validate_gamma_tail_bound_schedule(
        cfg, _read_mapping(result_file), **kwargs
    )
    if report_path is not None:
        destination = Path(report_path)
        if destination.suffix.lower() in {".yaml", ".yml"}:
            destination.write_text(
                yaml.safe_dump(report, sort_keys=False), encoding="utf-8"
            )
        elif destination.suffix.lower() == ".json":
            destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        else:
            raise ValueError("validation report path must end in .yaml, .yml, or .json.")
    return report
