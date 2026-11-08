"""PROTOTYPE of step C3 of the plan (scratch, written by the Task 1 agent to run the tests)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from fleet_management.config import FleetConfig, _integer
from fleet_management.degradation_model.gamma_utils.gamma_replay_validator import (
    _array, _binary, _check_report_path, _check_tolerance, _event, _finish, _increments,
    _read_pair, _record, _schedule_violations, gamma_cells,
)
from fleet_management.degradation_model.gamma_utils.gamma_tail_bound import (
    moschopoulos_tail_probability,
)


@dataclass(frozen=True)
class ExactGammaTerm:
    shape: float
    rate: float
    source: str
    created_step: int
    repairs: int = 0

    @property
    def mean(self) -> float:
        return self.shape / self.rate

    def after_ardinf(self, remaining_fraction: float) -> "ExactGammaTerm":
        return ExactGammaTerm(self.shape, self.rate / remaining_fraction, self.source,
                              self.created_step, self.repairs + 1)

    def as_dict(self) -> dict[str, Any]:
        return {"shape": self.shape, "rate": self.rate, "mean": self.mean, "source": self.source,
                "created_step": self.created_step, "repairs": self.repairs}


def _seed(mean, rate, source, step):
    if mean <= 0.0:
        return [], 0.0
    if math.isnan(rate):
        return [], float(mean)
    return [ExactGammaTerm(mean * rate, rate, source, step)], 0.0


def _exact_tail(terms, constant, tau, convolution_tolerance, max_series_terms):
    if not terms:
        value = 1.0 if constant > tau else 0.0
        return {"estimate": value, "upper_bound": value, "remaining_mass": 0.0, "series_terms": 0}
    if constant >= tau:
        return {"estimate": 1.0, "upper_bound": 1.0, "remaining_mass": 0.0, "series_terms": 0}
    tail = moschopoulos_tail_probability(
        [t.shape for t in terms], [t.rate for t in terms], tau - constant,
        tolerance=convolution_tolerance, max_series_terms=max_series_terms)
    return {"estimate": tail.estimate, "upper_bound": tail.upper_bound,
            "remaining_mass": tail.remaining_mass, "series_terms": tail.series_terms}


def validate_gamma_tail_bound_schedule(cfg, result, *, tolerance=1e-8, convolution_tolerance=1e-12,
                                       max_series_terms=100_000, include_steps=False,
                                       raise_on_failure=False):
    _check_tolerance(tolerance)
    if not 0.0 < convolution_tolerance < 1.0:
        raise ValueError("convolution_tolerance must lie in (0, 1).")
    max_series_terms = _integer(max_series_terms, "max_series_terms")
    if max_series_terms <= 0:
        raise ValueError("max_series_terms must be positive.")
    started = time.perf_counter()
    F, M, L, T = cfg.F, cfg.M, cfg.L, cfg.T
    cells = gamma_cells(cfg, result)
    ard1 = [(i, l) for i, l in cells if str(cfg.repair_model[i, l]) == "ard1"]
    x = _binary(result, "x", (F, M + 1, T), tolerance)
    m = _binary(result, "m", (F, L, T), tolerance)
    r = _binary(result, "r", (F, L, T), tolerance, optional=True)
    mu = _array(result, "mu", (F, L, T), cells)
    g_mu = _array(result, "g_mu", (F, L, T), ard1) if ard1 else None
    mu_inc, beta_inc = _increments(cfg)
    global_violations = _schedule_violations(cfg, result, x, m, r)
    maxima = dict.fromkeys(("reliability", "tracked_mean", "mean_anchor"), 0.0)
    violations, steps = [], []
    repairs = replacements = 0
    tail_seconds = 0.0
    evaluations = 0
    total_terms = max_terms = 0
    max_mass = 0.0
    minimum_slack, worst = math.inf, None

    def check(i, l, k, event, field, expected, saved, error):
        maxima[field] = max(maxima[field], error)
        if error > tolerance:
            violations.append(_record(i, l, k, event, field, expected, saved, error))

    for i, l in cells:
        anchored = str(cfg.repair_model[i, l]) == "ard1"
        remaining = max(0.0, 1.0 - float(cfg.rho[i, l]))
        beta_0, beta_new = float(cfg.gamma_beta_0[i, l]), float(cfg.gamma_beta_new[i, l])
        mu_0, tau, eps = float(cfg.mu_0[i, l]), float(cfg.tau[i, l]), float(cfg.epsilon[i, l])
        if anchored:
            g_0 = float(cfg.anchor_mu_0[i, l])
            frozen, c_frozen = _seed(g_0, beta_0, "initial_anchor", -1)
            active, c_active = _seed(mu_0 - g_0, beta_0, "initial", -1)
        else:
            frozen, c_frozen = [], 0.0
            active, c_active = _seed(mu_0, beta_0, "initial", -1)
        for k in range(T):
            event = _event(x, m, r, i, l, k)
            if event == "replacement":
                replacements += 1
                seed, c_seed = _seed(float(cfg.replacement_mu[i, l]), beta_new, "replacement", k)
                if anchored:
                    frozen, c_frozen, active, c_active = seed, c_seed, [], 0.0
                else:
                    frozen, c_frozen, active, c_active = [], 0.0, seed, c_seed
            elif event == "repair":
                repairs += 1
                repaired = [] if remaining == 0.0 else [t.after_ardinf(remaining) for t in active]
                if anchored:
                    frozen, c_frozen = frozen + repaired, c_frozen + remaining * c_active
                    active, c_active = [], 0.0
                else:
                    active, c_active = repaired, remaining * c_active
            elif event == "mission":
                for j0 in np.flatnonzero(x[i, 1:, k]):
                    mean, rate = float(mu_inc[i, l, j0, k]), float(beta_inc[i, l, j0, k])
                    if mean > 0.0:
                        active.append(ExactGammaTerm(mean * rate, rate, f"mission_{j0 + 1}", k))
            terms = frozen + active
            constant = c_frozen + c_active
            clock = time.perf_counter()
            tail = _exact_tail(terms, constant, tau, convolution_tolerance, max_series_terms)
            tail_seconds += time.perf_counter() - clock
            if tail["series_terms"] > 0:
                evaluations += 1
            total_terms += tail["series_terms"]
            max_terms = max(max_terms, tail["series_terms"])
            max_mass = max(max_mass, tail["remaining_mass"])
            exact_mean = constant + sum(t.mean for t in terms)
            frozen_mean = c_frozen + sum(t.mean for t in frozen)
            upper = tail["upper_bound"]
            slack = eps - upper
            saved = float(mu[i, l, k])
            check(i, l, k, event, "reliability", eps, upper, max(0.0, upper - eps))
            check(i, l, k, event, "tracked_mean", exact_mean, saved, max(0.0, exact_mean - saved))
            saved_anchor = None
            if anchored:
                saved_anchor = float(g_mu[i, l, k])
                check(i, l, k, event, "mean_anchor", frozen_mean, saved_anchor,
                      max(0.0, frozen_mean - saved_anchor))
            step = {"i": i, "l": l, "k": k, "event": event, "epsilon": eps, "constant": constant,
                    "exact_mean": exact_mean, "saved_mean": saved,
                    "frozen_mean": frozen_mean if anchored else None, "saved_anchor": saved_anchor,
                    "exact_tail_estimate": tail["estimate"], "exact_tail_upper_bound": upper,
                    "remaining_mass": tail["remaining_mass"], "series_terms": tail["series_terms"],
                    "reliability_slack": slack}
            if include_steps or slack < minimum_slack:
                step["terms"] = ([{**t.as_dict(), "part": "frozen"} for t in frozen]
                                 + [{**t.as_dict(), "part": "active"} for t in active])
            if slack < minimum_slack:
                minimum_slack, worst = slack, step
            if include_steps:
                steps.append(step)

    report = {
        "validator": "Gamma exact validator",
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
        "timing": {"validation_wall_seconds": time.perf_counter() - started,
                   "exact_tail_seconds": tail_seconds},
        "minimum_reliability_slack": minimum_slack,
        "worst_reliability": worst,
        "maximum_remaining_mass": max_mass,
        "maximum_series_terms_used": max_terms,
        "total_series_terms": total_terms,
        "exact_tail_evaluations": evaluations,
    }
    if include_steps:
        report["steps"] = steps
    return _finish(report, None, raise_on_failure)


def validate_gamma_tail_bound_files(input_path, result_path, report_path=None, **kwargs):
    destination = _check_report_path(report_path)
    raise_on_failure = kwargs.pop("raise_on_failure", False)
    cfg, result = _read_pair(input_path, result_path)
    report = validate_gamma_tail_bound_schedule(cfg, result, **kwargs)
    return _finish(report, destination, raise_on_failure)
