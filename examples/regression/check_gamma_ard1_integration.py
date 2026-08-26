"""Regression coverage for fixed-rate Gamma ARD1 repair dynamics."""

from __future__ import annotations

import numpy as np

from fleet_management.config import load_config
from fleet_management.degradation_model.base import (
    build_fleet,
    extract_solution,
    get_cell_builder,
    resolve_run_options,
)


def make_config(*, rho: float, replacement_mu: float):
    return load_config(
        {
            "F": 2,
            "M": 1,
            "L": 1,
            "H": [3, 3],
            "model": "gamma",
            "repair_model": "ard1",
            "tau": 0.8,
            "epsilon": 0.2,
            "rho": rho,
            "mu_0": 0.02,
            "replacement_mu": replacement_mu,
            "mu": 0.02,
            "mu_trans": 0.05,
            "gamma_beta": 10.0,
            "gamma_beta_trans": 10.0,
            "gamma_beta_bound": 10.0,
            "gamma_beta_0": 10.0,
            "gamma_beta_new": 10.0,
            "C_M": 1.0,
            "C_R": 0.5,
            "C_D": 2.0,
            "C_rep": 0.2,
            "allow_replacement": True,
            "depot_capacity": 1,
            "mip_gap": 0.0,
            "verbose": 0,
        }
    )


def solve_forced(cfg, actions):
    ctx = build_fleet(
        cfg,
        resolve_run_options(cfg),
        model_name="gamma_ard1_integration",
    )

    for action, vehicle, step in actions:
        if action == "repair":
            variable = ctx.m_rep[vehicle, 0, step]
        elif action == "replace":
            variable = ctx.r_rep[vehicle, 0, step]
        elif action == "mission":
            variable = ctx.x[vehicle, 1, step]
        else:
            raise ValueError(f"unknown action {action!r}")

        ctx.model.addConstr(
            variable == 1,
            name=f"force_{action}_{vehicle}_{step}",
        )

    ctx.model.optimize()

    if ctx.model.SolCount == 0:
        raise AssertionError("forced Gamma ARD1 model has no incumbent")

    result = extract_solution(ctx, cfg, ctx.model)
    get_cell_builder("gamma").extract(ctx, cfg, result)

    result["backend"] = "modular"
    result["degradation"] = "gamma"

    return result


def assert_close(actual, expected, name, tolerance=1e-8):
    if abs(float(actual) - float(expected)) > tolerance:
        raise AssertionError(
            f"{name}: got {float(actual):.12g}, "
            f"expected {float(expected):.12g}"
        )


def repeated_repair_and_replacement():
    cfg = make_config(rho=0.5, replacement_mu=0.001)

    result = solve_forced(
        cfg,
        [
            ("repair", 0, 0),
            ("mission", 0, 1),

            # This gives the second vehicle a nonzero latch and permits its
            # repeatability conditions to close.
            ("repair", 1, 1),

            ("repair", 0, 2),
            ("mission", 0, 3),
            ("repair", 0, 4),
            ("replace", 0, 5),
        ],
    )

    # Physical expected-damage trajectory.
    expected_mu = [
        0.010,
        0.060,
        0.035,
        0.055,
        0.045,
        0.001,
    ]

    expected_mean_latch = [
        0.010,
        0.010,
        0.035,
        0.035,
        0.045,
        0.001,
    ]

    expected_removed = [
        0.010,
        0.000,
        0.025,
        0.000,
        0.010,
        0.044,
    ]

    # All exact and bounding rates are 10 in this test. Consequently the
    # bounding-shape trajectory is exactly beta_bar times the mean trajectory.
    expected_shape = [
        0.100,
        0.600,
        0.350,
        0.550,
        0.450,
        0.010,
    ]

    expected_shape_latch = [
        0.100,
        0.100,
        0.350,
        0.350,
        0.450,
        0.010,
    ]

    common_rate = float(np.asarray(result["gamma_beta_bound"])[0, 0])

    for k in range(cfg.T):
        assert_close(
            result["mu"][0, 0, k],
            expected_mu[k],
            f"mu[{k}]",
        )
        assert_close(
            result["gamma_mean_latch"][0, 0, k],
            expected_mean_latch[k],
            f"mean_latch[{k}]",
        )
        assert_close(
            result["z"][0, 0, k],
            expected_removed[k],
            f"removed_mean[{k}]",
        )
        assert_close(
            result["gamma_shape_bound"][0, 0, k],
            expected_shape[k],
            f"shape[{k}]",
        )
        assert_close(
            result["gamma_shape_latch"][0, 0, k],
            expected_shape_latch[k],
            f"shape_latch[{k}]",
        )
        assert_close(
            result["gamma_shape_bound"][0, 0, k] / common_rate,
            result["mu"][0, 0, k],
            f"shape-derived mean[{k}]",
        )

    repairs = int(round(np.asarray(result["m"])[:, 0, :].sum()))

    return result, repairs


def complete_and_consecutive_repairs():
    cfg = make_config(rho=1.0, replacement_mu=0.0)

    result = solve_forced(
        cfg,
        [
            ("repair", 0, 0),
            ("repair", 0, 1),
            ("mission", 0, 2),
            ("mission", 0, 3),
            ("repair", 0, 4),
            ("replace", 0, 5),
        ],
    )

    # First complete repair and the immediately repeated repair.
    assert_close(
        result["mu"][0, 0, 0],
        0.0,
        "complete first repair mean",
    )
    assert_close(
        result["gamma_shape_bound"][0, 0, 0],
        0.0,
        "complete first repair shape",
    )
    assert_close(
        result["gamma_mean_latch"][0, 0, 1],
        0.0,
        "consecutive repair mean latch",
    )
    assert_close(
        result["gamma_shape_latch"][0, 0, 1],
        0.0,
        "consecutive repair shape latch",
    )

    # The missions at steps 2 and 3 accumulate new active damage. The complete
    # repair at step 4 must remove everything above the stored zero latch.
    assert_close(
        result["mu"][0, 0, 4],
        0.0,
        "complete later repair mean",
    )
    assert_close(
        result["gamma_shape_bound"][0, 0, 4],
        0.0,
        "complete later repair shape",
    )
    assert_close(
        result["gamma_shape_latch"][0, 0, 4],
        0.0,
        "complete later repair shape latch",
    )

    # Replacement with replacement_mu=0 resets both states to zero.
    assert_close(
        result["mu"][0, 0, 5],
        0.0,
        "zero replacement mean",
    )
    assert_close(
        result["gamma_shape_bound"][0, 0, 5],
        0.0,
        "zero replacement shape",
    )
    assert_close(
        result["gamma_shape_latch"][0, 0, 5],
        0.0,
        "zero replacement shape latch",
    )

    return result


def main() -> None:
    repeated, repairs = repeated_repair_and_replacement()
    complete = complete_and_consecutive_repairs()

    print("PASS fixed-rate Gamma ARD1 repair integration")
    print("repair model         :", repeated["repair_model"])
    print("Gamma repairs        :", repairs)
    print(
        "shape trajectory     :",
        repeated["gamma_shape_bound"][0, 0].tolist(),
    )
    print(
        "shape latch          :",
        repeated["gamma_shape_latch"][0, 0].tolist(),
    )
    print(
        "mean trajectory      :",
        repeated["mu"][0, 0].tolist(),
    )
    print(
        "complete final shape :",
        complete["gamma_shape_bound"][0, 0, -1],
    )
    print("repair shape rule    : fixed beta, scaled shape")


if __name__ == "__main__":
    main()