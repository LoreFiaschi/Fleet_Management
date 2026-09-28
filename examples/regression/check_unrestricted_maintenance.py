"""Regression for unrestricted fleet-level maintenance availability."""

from __future__ import annotations

from gurobipy import GRB

from fleet_management.config import load_config
from fleet_management.degradation_model.base import build_fleet, resolve_run_options


def scenario() -> dict:
    return {
        "F": 3,
        "M": 1,
        "L": 1,
        "H": [2, 2],
        "model": "gamma",
        "repair_model": "ardinf",
        "gamma_calibration_method": "repeated_increment",
        "mu": 0.01,
        "gamma_beta": 10.0,
        "gamma_beta_bound": 10.0,
        "gamma_beta_0": 10.0,
        "gamma_beta_new": 10.0,
        "mu_0": 0.02,
        "replacement_mu": 0.02,
        "allow_replacement": False,
        "tau": 0.9,
        "epsilon": 0.2,
        "rho": 0.5,
        "C_M": 1.0,
        "C_R": 0.5,
        "C_D": 1.0,
        "C_rep": 1.0,
        "mip_gap": 0.0,
        "verbose": 0,
    }


def main() -> None:
    obsolete = scenario()
    obsolete["depot_capacity"] = 1
    try:
        load_config(obsolete)
    except ValueError as error:
        if "depot_capacity" not in str(error) or "removed" not in str(error):
            raise AssertionError(f"unclear removed-option error: {error}") from error
    else:
        raise AssertionError("obsolete depot_capacity input was silently accepted")

    cfg = load_config(scenario())
    context = build_fleet(
        cfg,
        resolve_run_options(cfg),
        model_name="unrestricted_maintenance_regression",
    )
    context.model.update()

    constraint_names = {row.ConstrName for row in context.model.getConstrs()}
    if any(name.startswith("depot_cap_") for name in constraint_names):
        raise AssertionError("fleet-level depot-capacity rows still exist")

    # At k=0, two vehicles use maintenance activity while the third serves the
    # required mission. This was infeasible under depot_capacity=1.
    context.model.addConstr(context.x[0, 0, 0] == 1, name="force_maintenance_0")
    context.model.addConstr(context.x[1, 0, 0] == 1, name="force_maintenance_1")
    context.model.addConstr(context.x[2, 1, 0] == 1, name="force_mission_2")
    context.model.optimize()

    if context.model.Status != GRB.OPTIMAL or context.model.SolCount == 0:
        raise AssertionError(
            "two simultaneous maintenance assignments should be feasible; "
            f"status={context.model.Status}, solutions={context.model.SolCount}"
        )

    print("PASS unrestricted fleet-level maintenance")
    print("simultaneous maintenance assignments: 2")
    print("fleet-level depot-capacity rows      : 0")


if __name__ == "__main__":
    main()
