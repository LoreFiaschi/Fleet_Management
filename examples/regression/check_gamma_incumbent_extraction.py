"""Check that a non-optimal legacy Gamma incumbent is not discarded."""

from __future__ import annotations

import numpy as np

from fleet_management.degradation_model.legacy.gamma_gurobi import (
    solve_fleet_management,
)


def main() -> None:
    result = solve_fleet_management(
        F=2,
        H=2,
        M=1,
        L=1,
        mu_param=np.array([[[[0.05, 0.05]]], [[[0.06, 0.06]]]]),
        tau=np.array([0.6]),
        epsilon=0.1,
        gamma_beta=np.array([10.0]),
        C_M=1.0,
        C_R=0.5,
        C_rep=0.2,
        C_S=2.0,
        C_P=1.0,
        mu_0=np.zeros((2, 1)),
        replacement_mu=np.zeros((2, 1)),
        repair_rho=np.array([0.5]),
        verbose=0,
        mip_gap=0.0,
        gurobi_params={"SolutionLimit": 1},
    )

    if result["performance"]["solutions_found"] < 1:
        raise AssertionError("SolutionLimit run did not produce an incumbent")
    for key in ("objective", "bound", "mip_gap", "x", "m", "r", "A", "mu", "u", "z"):
        if result.get(key) is None:
            raise AssertionError(f"incumbent field {key!r} was discarded")
    if result["status"] not in {"solution_limit", "optimal"}:
        raise AssertionError(f"unexpected status {result['status']!r}")

    print("PASS legacy Gamma incumbent extraction")
    print("status    :", result["status"])
    print("solutions :", result["performance"]["solutions_found"])
    print("objective :", result["objective"])
    print("bound     :", result["bound"])
    print("mip gap   :", result["mip_gap"])


if __name__ == "__main__":
    main()
