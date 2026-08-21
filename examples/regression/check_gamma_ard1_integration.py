"""Regression coverage for modular Gamma ARD1 dynamics and validation."""

from __future__ import annotations

from fleet_management.config import load_config
from fleet_management.degradation_model.base import (
    build_fleet,
    extract_solution,
    get_cell_builder,
    resolve_run_options,
)
from fleet_management.degradation_model.gamma_utils.gamma_tail_validator import (
    validate_gamma_tail_bound_schedule,
)
from fleet_management.degradation_model.gamma_utils.gamma_diagnostics import (
    collect_gurobi_model_statistics,
    compare_estimate_with_actual,
    estimate_gamma_formulation,
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
            f"{name}: got {float(actual):.12g}, expected {float(expected):.12g}"
        )


def repeated_repair_and_replacement():
    cfg = make_config(rho=0.5, replacement_mu=0.001)
    result = solve_forced(
        cfg,
        [
            ("repair", 0, 0),
            ("mission", 0, 1),
            # Give vehicle 1 a positive transitory latch so its own ARD1 latch
            # repeatability condition can close while vehicle 0 serves mission 1.
            ("repair", 1, 1),
            ("repair", 0, 2),
            ("mission", 0, 3),
            ("repair", 0, 4),
            ("replace", 0, 5),
        ],
    )

    expected_mu = [0.01, 0.06, 0.035, 0.055, 0.045, 0.001]
    expected_latch = [0.01, 0.01, 0.035, 0.035, 0.045, 0.001]
    expected_removed = [0.01, 0.0, 0.025, 0.0, 0.01, 0.044]
    for k in range(cfg.T):
        assert_close(result["mu"][0, 0, k], expected_mu[k], f"mu[{k}]")
        assert_close(
            result["gamma_mean_latch"][0, 0, k],
            expected_latch[k],
            f"latch[{k}]",
        )
        assert_close(result["z"][0, 0, k], expected_removed[k], f"z[{k}]")

    shape = result["gamma_shape_bound"][0, 0]
    assert_close(shape[0], 0.2, "initial repaired bounding shape")
    assert_close(shape[2], shape[1], "second repair bounding shape")
    assert_close(shape[4], shape[3], "third repair bounding shape")

    report = validate_gamma_tail_bound_schedule(
        cfg,
        result,
        include_steps=True,
        raise_on_failure=True,
    )
    if report["gamma_ard1_cells"] != 2 or report["repairs"] < 3:
        raise AssertionError("ARD1 validator did not record the forced repairs")
    if report["maximum_latch_error"] > 1e-8:
        raise AssertionError("ARD1 exact-history latch reconstruction failed")

    first = next(
        step for step in report["steps"]
        if step["i"] == 0 and step["k"] == 0
    )
    second = next(
        step for step in report["steps"]
        if step["i"] == 0 and step["k"] == 2
    )
    if (first["frozen_term_count"], first["active_term_count"]) != (1, 0):
        raise AssertionError("first ARD1 repair did not freeze the initial term")
    if (second["frozen_term_count"], second["active_term_count"]) != (2, 0):
        raise AssertionError("second ARD1 repair did not freeze the mission term")
    return result, report


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
    assert_close(result["mu"][0, 0, 0], 0.0, "complete first repair mean")
    assert_close(result["mu"][0, 0, 1], 0.0, "consecutive repair mean")
    assert_close(
        result["gamma_mean_latch"][0, 0, 1],
        0.0,
        "consecutive repair latch",
    )

    report = validate_gamma_tail_bound_schedule(
        cfg,
        result,
        include_steps=True,
        raise_on_failure=True,
    )
    repaired = next(
        step for step in report["steps"]
        if step["i"] == 0 and step["k"] == 0
    )
    consecutive = next(
        step for step in report["steps"]
        if step["i"] == 0 and step["k"] == 1
    )
    if repaired["history"] or consecutive["history"]:
        raise AssertionError("complete ARD1 repair retained a repairable term")
    return report


def check_formulation_counts():
    cfg = make_config(rho=0.5, replacement_mu=0.001)
    ctx = build_fleet(
        cfg,
        resolve_run_options(cfg),
        model_name="gamma_ard1_formulation_count",
    )
    estimate = estimate_gamma_formulation(cfg, allow_replacement=True)
    actual = collect_gurobi_model_statistics(ctx.model)
    comparison = compare_estimate_with_actual(estimate, actual)
    if estimate["gamma_ard1_cells"] != 2:
        raise AssertionError("wrong number of Gamma ARD1 cells in estimate")
    if not comparison["known_subtotal_matches_actual"]:
        raise AssertionError(
            "Gamma ARD1 formulation estimate does not match Gurobi: "
            f"{comparison['non_gamma_remainder']}"
        )
    return actual


def main() -> None:
    result, repeated = repeated_repair_and_replacement()
    complete = complete_and_consecutive_repairs()
    formulation = check_formulation_counts()

    print("PASS conservative Gamma ARD1 integration")
    print("repair model          :", result["repair_model"])
    print("ARD1 cells            :", repeated["gamma_ard1_cells"])
    print("repeated repairs      :", repeated["repairs"])
    print("maximum latch error   :", repeated["maximum_latch_error"])
    print("worst tail margin     :", repeated["minimum_conservativeness_margin"])
    print("complete-repair margin:", complete["minimum_conservativeness_margin"])
    print("repair shape rule     : A_after = A_before")
    print("model counts          :", {
        "variables": formulation["variables"],
        "linear": formulation["linear_constraints"],
        "indicators": formulation["indicator_constraints"],
    })


if __name__ == "__main__":
    main()
