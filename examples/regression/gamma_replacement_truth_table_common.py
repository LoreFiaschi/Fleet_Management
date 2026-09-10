"""Shared fixtures for replacement-enabled Gamma truth-table regressions."""

from __future__ import annotations

import numpy as np

from fleet_management.config import load_config
from fleet_management.degradation_model.base import (
    build_fleet,
    extract_solution,
    get_cell_builder,
    resolve_run_options,
)
from fleet_management.degradation_model.gamma_utils.gamma_diagnostics import (
    collect_gurobi_model_statistics,
)


EVENTS = (
    "mission",
    "replacement",
    "explicit idle",
    "mission",
    "repair",
    "replacement",
    "consecutive replacement",
    "unassigned idle",
)


def make_config(repair_model: str):
    return load_config(
        {
            "F": 2,
            "M": 1,
            "L": 1,
            "H": [4, 4],
            "model": "gamma",
            "repair_model": repair_model,
            "tau": 1.0,
            "epsilon": 0.2,
            "rho": 0.5,
            "mu_0": 0.02,
            "replacement_mu": 0.005,
            "mu": 0.01,
            "mu_trans": 0.01,
            "gamma_beta": 10.0,
            "gamma_beta_trans": 10.0,
            "gamma_beta_bound": 10.0,
            "gamma_beta_0": 10.0,
            "gamma_beta_new": 10.0,
            "C_M": 0.0,
            "C_R": 0.0,
            "C_D": 0.0,
            "C_rep": 0.0,
            "allow_replacement": True,
            "mip_gap": 0.0,
            "verbose": 0,
        }
    )


def assert_close(actual, expected, name: str, tolerance: float = 1e-8) -> None:
    if abs(float(actual) - float(expected)) > tolerance:
        raise AssertionError(
            f"{name}: got {float(actual):.12g}, expected {float(expected):.12g}"
        )


def _force(model, variable, value: int, name: str) -> None:
    model.addConstr(variable == value, name=name)


def solve_truth_table(repair_model: str):
    """Solve every integral replacement transition in one fixed schedule."""
    cfg = make_config(repair_model)
    context = build_fleet(
        cfg,
        resolve_run_options(cfg),
        model_name=f"gamma_{repair_model}_replacement_truth_table",
    )
    context.model.update()
    baseline_statistics = collect_gurobi_model_statistics(context.model)

    # The truth table tests local transitions. The terminal state loop is a
    # separate constraint family and would obscure the deliberately fixed path.
    loop_rows = [
        row
        for row in context.model.getConstrs()
        if row.ConstrName.startswith("loop_")
    ]
    context.model.remove(loop_rows)
    context.model.update()

    mission_steps = {0, 3}
    explicit_idle_steps = {1, 2, 4, 5, 6}
    repair_steps = {4}
    replacement_steps = {1, 5, 6}

    for i in range(cfg.F):
        for j in range(cfg.M + 1):
            for k in range(cfg.T):
                selected = 0
                if i == 0:
                    selected = int(
                        (j == 1 and k in mission_steps)
                        or (j == 0 and k in explicit_idle_steps)
                    )
                else:
                    selected = int(j == 1 and k not in mission_steps)
                _force(
                    context.model,
                    context.x[i, j, k],
                    selected,
                    f"force_x_{i}_{j}_{k}",
                )
        for k in range(cfg.T):
            _force(
                context.model,
                context.m_rep[i, 0, k],
                int(i == 0 and k in repair_steps),
                f"force_m_{i}_{k}",
            )
            _force(
                context.model,
                context.r_rep[i, 0, k],
                int(i == 0 and k in replacement_steps),
                f"force_r_{i}_{k}",
            )

    context.model.optimize()
    if context.model.SolCount == 0:
        raise AssertionError(
            f"forced Gamma {repair_model} replacement truth table is infeasible"
        )

    result = extract_solution(context, cfg, context.model)
    get_cell_builder("gamma").extract(context, cfg, result)
    return cfg, context, result, baseline_statistics


def assert_simultaneous_actions_infeasible(repair_model: str) -> None:
    """Repair and replacement must never be selected together."""
    cfg = make_config(repair_model)
    context = build_fleet(
        cfg,
        resolve_run_options(cfg),
        model_name=f"gamma_{repair_model}_replacement_exclusivity",
    )
    context.model.addConstr(context.m_rep[0, 0, 0] == 1, name="force_repair")
    context.model.addConstr(context.r_rep[0, 0, 0] == 1, name="force_replacement")
    context.model.optimize()
    if context.model.SolCount != 0:
        raise AssertionError("simultaneous repair and replacement were feasible")


def check_common_trajectory(result, expected_mean, expected_removed) -> None:
    expected_mean = np.asarray(expected_mean, dtype=float)
    expected_removed = np.asarray(expected_removed, dtype=float)
    for k in range(expected_mean.size):
        assert_close(result["mu"][0, 0, k], expected_mean[k], f"mu[{k}]")
        assert_close(result["z"][0, 0, k], expected_removed[k], f"z[{k}]")
        assert_close(
            result["gamma_shape_bound"][0, 0, k],
            10.0 * expected_mean[k],
            f"shape[{k}]",
        )
