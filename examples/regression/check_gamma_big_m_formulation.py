"""Regression for the modular Gamma tight Big-M dynamics."""

from __future__ import annotations

from pathlib import Path

import yaml

from fleet_management.config import load_config
from fleet_management.degradation_model.base import build_fleet, resolve_run_options
from fleet_management.degradation_model.gamma_utils.gamma_diagnostics import (
    collect_gurobi_model_statistics,
    compare_estimate_with_actual,
    estimate_gamma_formulation,
)


HERE = Path(__file__).resolve().parent


def ard1_config():
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
            "rho": 0.5,
            "mu_0": 0.02,
            "replacement_mu": 0.001,
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
            "verbose": 0,
        }
    )


def check_case(cfg, name: str) -> dict:
    options = resolve_run_options(cfg)
    context = build_fleet(cfg, options, model_name=name)
    actual = collect_gurobi_model_statistics(context.model)
    estimate = estimate_gamma_formulation(
        cfg, allow_replacement=options["allow_replacement"]
    )
    comparison = compare_estimate_with_actual(estimate, actual)

    if actual["indicator_constraints"] != 0:
        raise AssertionError(f"{name} still contains Gamma indicator constraints")
    if actual["general_constraints"] != 0:
        raise AssertionError(f"{name} unexpectedly contains general constraints")
    if not comparison["known_subtotal_matches_actual"]:
        raise AssertionError(
            f"{name} estimate differs from Gurobi: "
            f"{comparison['non_gamma_remainder']}"
        )
    if estimate["gamma_attributable"]["general_constraint_total"] != 0:
        raise AssertionError(f"{name} estimator still counts Gamma indicators")

    return {
        "variables": actual["variables"],
        "linear_constraints": actual["linear_constraints"],
        "indicator_constraints": actual["indicator_constraints"],
    }


def main() -> None:
    ardinf = check_case(
        load_config(
            yaml.safe_load(
                (HERE / "gamma_tail_bound_public.yaml").read_text(
                    encoding="utf-8"
                )
            )
        ),
        "gamma_big_m_ardinf",
    )
    ard1 = check_case(ard1_config(), "gamma_big_m_ard1")

    print("PASS modular Gamma tight Big-M formulation")
    print("ARD-inf model:", ardinf)
    print("ARD1 model   :", ard1)


if __name__ == "__main__":
    main()
