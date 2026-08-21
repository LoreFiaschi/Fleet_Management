"""High-value edge cases for exact modular Gamma schedule validation."""

from __future__ import annotations

from typing import Iterable

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


def make_config(
    *,
    rho: float,
    mu_0: float,
    replacement_mu: float,
    tau: float = 0.6,
    epsilon: float = 0.1,
):
    return load_config(
        {
            "F": 2,
            "M": 1,
            "L": 1,
            "H": [2, 2],
            "model": "gamma",
            "repair_model": "ardinf",
            "tau": tau,
            "epsilon": epsilon,
            "rho": rho,
            "mu_0": mu_0,
            "replacement_mu": replacement_mu,
            "mu": 0.05,
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


def solve_forced(cfg, actions: Iterable[tuple[str, int, int]]) -> dict:
    """Solve after forcing ``(action, vehicle, step)`` decisions."""

    ctx = build_fleet(
        cfg,
        resolve_run_options(cfg),
        model_name="gamma_validator_edge_case",
    )
    for action, vehicle, step in actions:
        if action == "repair":
            variable = ctx.m_rep[vehicle, 0, step]
        elif action == "replace":
            variable = ctx.r_rep[vehicle, 0, step]
        elif action == "mission":
            variable = ctx.x[vehicle, 1, step]
        else:
            raise ValueError(f"unknown forced action {action!r}")
        ctx.model.addConstr(
            variable == 1,
            name=f"force_{action}_{vehicle}_{step}",
        )

    ctx.model.optimize()
    if ctx.model.SolCount == 0:
        raise AssertionError("forced Gamma edge case has no incumbent")

    result = extract_solution(ctx, cfg, ctx.model)
    get_cell_builder("gamma").extract(ctx, cfg, result)
    result["backend"] = "modular"
    result["degradation"] = "gamma"
    return result


def report_for(cfg, actions) -> dict:
    return validate_gamma_tail_bound_schedule(
        cfg,
        solve_forced(cfg, actions),
        include_steps=True,
        raise_on_failure=True,
    )


def find_step(report: dict, vehicle: int, step: int) -> dict:
    return next(
        item
        for item in report["steps"]
        if item["i"] == vehicle and item["l"] == 0 and item["k"] == step
    )


def zero_seed_near_threshold() -> dict:
    cfg = make_config(
        rho=0.5,
        mu_0=0.0,
        replacement_mu=0.0,
        tau=0.08,
        epsilon=0.25,
    )
    report = report_for(cfg, [("mission", 0, 0)])
    step = find_step(report, 0, 0)
    if step["event"] != "mission" or len(step["history"]) != 1:
        raise AssertionError("zero-seed case did not begin with one mission term")
    term = step["history"][0]
    if term["source"] != "mission_1" or term["created_step"] != 0:
        raise AssertionError("zero-seed history contains a spurious seed term")
    if abs(step["exact_mean"] - 0.05) > 1e-10:
        raise AssertionError("zero-seed mission mean was reconstructed incorrectly")
    if step["exact_tail_upper_bound"] < 0.15:
        raise AssertionError("near-threshold case is not numerically meaningful")
    return report


def repeated_repairs_then_mission() -> dict:
    # After two repairs the repeatability reference mean is 0.005.  Keep the
    # replacement seed below it so a later reset can close the physical cycle.
    cfg = make_config(rho=0.5, mu_0=0.02, replacement_mu=0.001)
    report = report_for(
        cfg,
        [
            ("repair", 0, 0),
            ("repair", 0, 1),
            ("mission", 0, 2),
        ],
    )
    first = find_step(report, 0, 0)
    second = find_step(report, 0, 1)
    mission = find_step(report, 0, 2)
    if first["event"] != "repair" or second["event"] != "repair":
        raise AssertionError("consecutive repairs were not replayed")
    if abs(first["history"][0]["rate"] - 20.0) > 1e-10:
        raise AssertionError("first repair rate is not beta/(1-rho)")
    if abs(second["history"][0]["rate"] - 40.0) > 1e-10:
        raise AssertionError("second repair did not compound the rate transform")
    if second["history"][0]["repairs"] != 2:
        raise AssertionError("repair count was not retained in exact history")
    if mission["event"] != "mission" or len(mission["history"]) != 2:
        raise AssertionError("mission was not appended after repeated repairs")
    if abs(mission["exact_mean"] - 0.055) > 1e-10:
        raise AssertionError("repaired history plus mission has the wrong mean")
    return report


def complete_repair_then_mission() -> dict:
    cfg = make_config(rho=1.0, mu_0=0.02, replacement_mu=0.01)
    report = report_for(
        cfg,
        [
            ("repair", 0, 0),
            ("mission", 0, 1),
        ],
    )
    repaired = find_step(report, 0, 0)
    mission = find_step(report, 0, 1)
    if repaired["event"] != "repair" or repaired["history"]:
        raise AssertionError("rho=1 repair did not remove the exact history")
    if repaired["exact_mean"] != 0.0 or repaired["exact_tail_upper_bound"] != 0.0:
        raise AssertionError("rho=1 repaired distribution is not degenerate at zero")
    if mission["event"] != "mission" or len(mission["history"]) != 1:
        raise AssertionError("mission after complete repair has the wrong history")
    if mission["history"][0]["source"] != "mission_1":
        raise AssertionError("old seed survived the complete repair")
    if abs(mission["exact_mean"] - 0.05) > 1e-10:
        raise AssertionError("mission mean after complete repair is wrong")
    return report


def main() -> None:
    zero = zero_seed_near_threshold()
    repeated = repeated_repairs_then_mission()
    complete = complete_repair_then_mission()

    print("PASS exact Gamma validator edge cases")
    print("zero-seed worst margin :", zero["minimum_conservativeness_margin"])
    print("zero-seed minimum slack:", zero["minimum_reliability_slack"])
    print("repeated repairs       :", repeated["repairs"])
    print("repeated worst margin  :", repeated["minimum_conservativeness_margin"])
    print("complete repairs       :", complete["repairs"])
    print("complete worst margin  :", complete["minimum_conservativeness_margin"])
    print("maximum remainder      :", max(
        zero["numerics"]["maximum_remaining_mass"],
        repeated["numerics"]["maximum_remaining_mass"],
        complete["numerics"]["maximum_remaining_mass"],
    ))


if __name__ == "__main__":
    main()
