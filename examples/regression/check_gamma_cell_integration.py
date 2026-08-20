"""Small end-to-end check for the modular Gamma cell builder.

The case is intentionally mixed so solver dispatch reaches ``base.solve_mixed``
instead of the legacy uniform-Gamma backend.  Gamma starts and replacements
are zero-damage, and imperfect Gamma repair is disabled by the builder.
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
            "mu_0": 0.0,
            "v_0": 0.0,
            "replacement_mu": 0.0,
            "replacement_v": 0.0,
            "mu": 0.05,
            "v": 0.0002,
            "gamma_beta": 10.0,
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

    result = solve_mixed(cfg)
    if result["status"] != "optimal":
        raise AssertionError(f"expected optimal solve, got {result['status']!r}")

    gamma_component = 0
    tol = 1e-7
    if np.max(np.abs(result["m"][:, gamma_component, :])) > tol:
        raise AssertionError("the uncertified Gamma repair decision was used")
    if np.max(result["gamma_tail_bound"][:, gamma_component, :]) > 0.1 + tol:
        raise AssertionError("Gamma reliability limit was violated")

    k_start, k_end = cfg.H1 - 1, cfg.T - 1
    shape = result["gamma_shape_bound"][:, gamma_component, :]
    mean = result["mu"][:, gamma_component, :]
    if np.any(shape[:, k_end] > shape[:, k_start] + tol):
        raise AssertionError("Gamma bounding-shape repeatability was violated")
    if np.any(mean[:, k_end] > mean[:, k_start] + tol):
        raise AssertionError("Gamma physical-mean repeatability was violated")

    print("PASS modular Gamma cell integration")
    print("status       :", result["status"])
    print("objective    :", result["objective"])
    print("models       :", result["models"])
    print("common rates:", result["gamma_beta_bound"][:, gamma_component])
    print("max tail     :", np.max(result["gamma_tail_bound"][:, gamma_component, :]))
    print("calibration  :", result["gamma_calibration"])


if __name__ == "__main__":
    main()
