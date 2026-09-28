"""Numerical regression for fixed-schedule Gamma Monte Carlo validation."""

from __future__ import annotations

import numpy as np
from scipy.stats import gamma as gamma_distribution

from fleet_management.config import load_config
from fleet_management.degradation_model.gamma_utils.gamma_stochastic_validator import (
    _one_sided_binomial_upper,
    validate_gamma_schedule,
)


def main() -> None:
    cfg = load_config({
        "F": 1,
        "M": 1,
        "L": 1,
        "H": [1, 1],
        "component_names": ["Battery"],
        "model": "gamma",
        "repair_model": "ardinf",
        "mu": 0.2,
        "gamma_beta": 10.0,
        "gamma_beta_bound": 10.0,
        "gamma_beta_0": 10.0,
        "gamma_beta_new": 10.0,
        "mu_0": 0.0,
        "replacement_mu": 0.0,
        "tau": 0.2,
        "epsilon": 0.99,
        "rho": 0.5,
        "C_M": 1.0,
        "C_R": 1.0,
        "C_D": 1.0,
        "C_rep": 1.0,
        "allow_replacement": False,
    })
    result = {
        "status": "feasible",
        # The one vehicle performs the mission in both time steps.
        "x": [[[0, 0], [1, 1]]],
        "m": [[[0, 0]]],
        "r": [[[0, 0]]],
    }

    kwargs = {
        "mode": "stochastic",
        "repetitions": 50_000,
        "random_seed": 20260908,
        "batch_size": 10_000,
    }
    first = validate_gamma_schedule(cfg, result, **kwargs)
    second = validate_gamma_schedule(cfg, result, **kwargs)

    if first["schedule"] != second["schedule"]:
        raise AssertionError("fixed-seed stochastic validation is not reproducible")
    if not first["complete_fleet_validation"] or first["excluded_cells"]:
        raise AssertionError("pure Gamma test unexpectedly excluded cells")

    # Two independent Gamma(2, rate=10) increments sum to Gamma(4, rate=10).
    expected = float(gamma_distribution.sf(0.2, a=4.0, scale=0.1))
    observed = float(first["schedule"]["failure_rate"])
    if abs(observed - expected) > 0.015:
        raise AssertionError(
            f"Monte Carlo rate {observed:.6f} differs from exact {expected:.6f}"
        )

    zero_failure_upper = _one_sided_binomial_upper(0, 100_000, 0.95)
    if not 2.9e-5 < zero_failure_upper < 3.1e-5:
        raise AssertionError(
            f"unexpected zero-failure confidence bound {zero_failure_upper}"
        )

    print("PASS fixed-schedule Gamma stochastic validator")
    print("repetitions          :", first["repetitions"])
    print("observed failure rate:", observed)
    print("exact failure rate   :", expected)
    print("zero/100000 95% upper:", zero_failure_upper)


if __name__ == "__main__":
    main()
