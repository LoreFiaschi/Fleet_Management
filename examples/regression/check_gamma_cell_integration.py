"""Small end-to-end check for the modular Gamma cell builder.

The case is intentionally mixed so solver dispatch reaches ``base.solve_mixed``
instead of the legacy uniform-Gamma backend. Nonzero initial and replacement
states are calibrated jointly; ARD-inf repair uses a conservative no-tail-credit
transition for the bounding shape.
"""

from __future__ import annotations

import numpy as np

from fleet_management.config import load_config
from fleet_management.degradation_model.base import solve_mixed


def main() -> None:
    cfg = load_config(
        {
            "F": 2,
            "M": 1,
            "L": 2,
            "H": [2, 2],
            "model": ["gamma", "rainflow"],
            "bound_method": ["cantelli", "cantelli"],
            "repair_model": ["ardinf", "ardinf"],
            "tau": [0.6, 0.6],
            "epsilon": [0.1, 0.1],
            "rho": 0.5,
            "mu_0": [0.02, 0.0],
            "v_0": 0.0,
            "replacement_mu": [0.01, 0.0],
            "replacement_v": 0.0,
            "mu": 0.05,
            "v": 0.0002,
            # Exact shape-rate parameters vary by vehicle, phase and time.
            # Values on rainflow cells are inert but keep the fleet-wide array
            # rectangular. The selected common Gamma rate is per component.
            "gamma_beta": [
                [[[20.0, 15.0]], [[10.0, 10.0]]],
                [[[20.0, 12.0]], [[10.0, 10.0]]],
            ],
            "gamma_beta_trans": [
                [[[18.0, 10.0]], [[10.0, 10.0]]],
                [[[16.0, 10.0]], [[10.0, 10.0]]],
            ],
            "gamma_beta_bound": [10.0, 10.0],
            "gamma_calibration_method": "repeated_increment",
            "gamma_beta_0": [14.0, 10.0],
            "gamma_beta_new": [11.0, 10.0],
            "C_M": 1.0,
            "C_R": 0.5,
            "C_D": 2.0,
            "C_rep": 0.2,
            "allow_replacement": True,
            "replacement_as_new": True,
            "mip_gap": 0.0,
            "verbose": 0,
        }
    )

    if cfg.gamma_beta.shape != (2, 2, 1, 2):
        raise AssertionError(f"wrong operating rate shape {cfg.gamma_beta.shape}")
    if cfg.gamma_beta_trans.shape != (2, 2, 1, 2):
        raise AssertionError(f"wrong transitory rate shape {cfg.gamma_beta_trans.shape}")

    result = solve_mixed(cfg)
    if result["status"] != "optimal":
        raise AssertionError(f"expected optimal solve, got {result['status']!r}")

    gamma_component = 0
    tol = 1e-7
    if np.max(result["gamma_tail_bound"][:, gamma_component, :]) > 0.1 + tol:
        raise AssertionError("Gamma reliability limit was violated")
    if any(item["tail_constraints"] <= 0 for item in result["gamma_calibration"]):
        raise AssertionError("repeated-increment calibration generated no checks")
    if any(
        item["method"] != "repeated_increment"
        or not item.get("maximum_safe_counts")
        for item in result["gamma_calibration"]
    ):
        raise AssertionError("m* calibration metadata is missing")

    k_start, k_end = cfg.H1 - 1, cfg.T - 1
    shape = result["gamma_shape_bound"][:, gamma_component, :]
    mean = result["mu"][:, gamma_component, :]
    if np.any(shape[:, k_end] > shape[:, k_start] + tol):
        raise AssertionError("Gamma bounding-shape repeatability was violated")
    if np.any(mean[:, k_end] > mean[:, k_start] + tol):
        raise AssertionError("Gamma physical-mean repeatability was violated")

    summaries = {(item["i"], item["l"]): item for item in result["gamma_calibration"]}
    for i in range(cfg.F):
        for k in range(cfg.T):
            if result["m"][i, gamma_component, k] < 0.5:
                continue
            previous_shape = (
                summaries[i, gamma_component]["initial_bounded_shape"]
                if k == 0 else shape[i, k - 1]
            )
            previous_mean = cfg.mu_0[i, gamma_component] if k == 0 else mean[i, k - 1]
            expected_shape = (
                (1.0 - cfg.rho[i, gamma_component]) * previous_shape
            )
            if abs(shape[i, k] - expected_shape) > tol:
                raise AssertionError("ARD-inf Gamma shape transition is wrong")
            expected_mean = (1.0 - cfg.rho[i, gamma_component]) * previous_mean
            if abs(mean[i, k] - expected_mean) > tol:
                raise AssertionError("ARD-inf Gamma physical-mean transition is wrong")

    print("PASS modular Gamma cell integration")
    print("status       :", result["status"])
    print("objective    :", result["objective"])
    print("models       :", result["models"])
    print("common rates:", result["gamma_beta_bound"][:, gamma_component])
    print("max tail     :", np.max(result["gamma_tail_bound"][:, gamma_component, :]))
    print("Gamma repairs:", int(np.rint(result["m"][:, gamma_component, :]).sum()))
    print("calibration  :", result["gamma_calibration"])


if __name__ == "__main__":
    main()
