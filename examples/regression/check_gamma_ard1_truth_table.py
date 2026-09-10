"""Audit every no-replacement Gamma project-ARD1 transition."""

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
    compare_estimate_with_actual,
    estimate_gamma_formulation,
)


def make_config(rho: float):
    return load_config(
        {
            "F": 2,
            "M": 1,
            "L": 1,
            "H": [3, 4],
            "model": "gamma",
            "repair_model": "ard1",
            "tau": 1.0,
            "epsilon": 0.2,
            "rho": rho,
            "mu_0": 0.02,
            "replacement_mu": 0.0,
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
            "allow_replacement": False,
            "mip_gap": 0.0,
            "verbose": 0,
        }
    )


def force_binary(model, variable, value: int, name: str) -> None:
    model.addConstr(variable == value, name=name)


def assert_close(actual, expected, name, tolerance=1e-8) -> None:
    if abs(float(actual) - float(expected)) > tolerance:
        raise AssertionError(
            f"{name}: got {float(actual):.12g}, expected {float(expected):.12g}"
        )


def assert_redesigned_constraints(context) -> None:
    """Verify the no-replacement ARD1 product hull and balance inventory."""
    names = {row.ConstrName for row in context.model.getConstrs()}
    forbidden = (
        "A_gamma_carry_", "mu_gamma_carry_", "z_gamma_zero_",
        "gmu_gamma_hold_", "gA_gamma_hold_", "gmu_gamma_setm_",
        "gA_gamma_setm_", "loop_gmu_gamma_", "loop_gA_gamma_",
    )
    if any(name.startswith(forbidden) for name in names):
        raise AssertionError("legacy ARD1 conditional or latch-loop rows remain")

    for i in range(context.F):
        seed_rows = {
            f"qmu_gamma_ard1_seed_{i}_0_0",
            f"qA_gamma_ard1_seed_{i}_0_0",
        }
        missing = seed_rows - names
        if missing:
            raise AssertionError(f"missing ARD1 seed rows {sorted(missing)}")
        for k in range(context.T):
            balance_rows = {
                f"z_gamma_ard1_balance_{i}_0_{k}",
                f"mu_gamma_ard1_balance_{i}_0_{k}",
                f"A_gamma_ard1_balance_{i}_0_{k}",
                f"gmu_gamma_ard1_balance_{i}_0_{k}",
                f"gA_gamma_ard1_balance_{i}_0_{k}",
            }
            missing = balance_rows - names
            if missing:
                raise AssertionError(f"missing ARD1 balance rows {sorted(missing)}")
        for k in range(1, context.T):
            for prefix in ("qmu_gamma_ard1_product", "qA_gamma_ard1_product"):
                base = f"{prefix}_{i}_0_{k}"
                required = {
                    f"{base}_state_ub", f"{base}_binary_ub", f"{base}_lower"
                }
                missing = required - names
                if missing:
                    raise AssertionError(
                        f"ARD1 product {base!r} is missing rows {sorted(missing)}"
                    )

    expected_local_rows = context.F * context.L * (13 * context.T - 2)
    gamma_rows = [
        name
        for name in names
        if name.startswith(
            (
                "m_gate_", "qmu_gamma_", "qA_gamma_", "A_gamma_",
                "mu_gamma_", "z_gamma_", "gmu_gamma_", "gA_gamma_",
                "rel_gamma_", "loop_A_gamma_", "loop_mu_gamma_",
            )
        )
    ]
    if len(gamma_rows) != expected_local_rows:
        raise AssertionError(
            f"found {len(gamma_rows)} project-ARD1 rows; expected {expected_local_rows}"
        )


def solve_truth_table(rho, expected_mu, expected_latch, expected_removed):
    cfg = make_config(rho)
    context = build_fleet(
        cfg,
        resolve_run_options(cfg),
        model_name="gamma_ard1_truth_table",
    )
    context.model.update()
    assert_redesigned_constraints(context)
    estimate = estimate_gamma_formulation(cfg, allow_replacement=False)
    actual = collect_gurobi_model_statistics(context.model)
    comparison = compare_estimate_with_actual(estimate, actual)
    if not comparison["known_subtotal_matches_actual"]:
        raise AssertionError(
            f"ARD1 product-hull count estimate differs: "
            f"{comparison['non_gamma_remainder']}"
        )
    if estimate["gamma_ard1_product_cells"] != cfg.F * cfg.L:
        raise AssertionError("count estimator missed project-ARD1 product cells")
    if context.r_rep is not None:
        raise AssertionError("no-replacement audit unexpectedly created replacement binaries")
    if context.nb:
        raise AssertionError("no-replacement project-ARD1 still creates nb binaries")
    if context.extras["gamma"]["removed_shape"] is not None:
        raise AssertionError("current project-ARD1 branch unexpectedly exposes removed shape")
    if context.extras["gamma"]["repairable_mean"] is None:
        raise AssertionError("project-ARD1 repairable-mean variables are missing")
    if context.extras["gamma"]["repairable_shape"] is None:
        raise AssertionError("project-ARD1 repairable-shape variables are missing")

    # This test isolates local transition logic. Terminal repeatability is a
    # separate, audited constraint family and would intentionally reject a
    # rising ARD1 latch in the operating phase.
    loop_rows = [
        row
        for row in context.model.getConstrs()
        if row.ConstrName.startswith("loop_")
    ]
    context.model.remove(loop_rows)
    context.model.update()

    # Vehicle 0 traverses every no-replacement truth-table event:
    # mission, explicit idle, first repair, unassigned idle, mission,
    # later repair, consecutive repair.
    vehicle_zero_x = {
        0: 1,       # mission j=1
        1: 0,       # explicit idle/depot j=0
        2: 0,       # repair requires j=0
        4: 1,       # mission j=1
        5: 0,       # repair requires j=0
        6: 0,       # consecutive repair requires j=0
    }
    repair_steps = {2, 5, 6}

    for i in range(cfg.F):
        for j in range(cfg.M + 1):
            for k in range(cfg.T):
                selected = 0
                if i == 0:
                    selected = int(vehicle_zero_x.get(k) == j)
                elif i == 1 and k not in {0, 4}:
                    selected = int(j == 1)
                force_binary(
                    context.model,
                    context.x[i, j, k],
                    selected,
                    f"force_x_{i}_{j}_{k}",
                )
        for k in range(cfg.T):
            force_binary(
                context.model,
                context.m_rep[i, 0, k],
                int(i == 0 and k in repair_steps),
                f"force_m_{i}_{k}",
            )

    context.model.optimize()
    if context.model.SolCount == 0:
        raise AssertionError("forced project-ARD1 truth table has no solution")

    result = extract_solution(context, cfg, context.model)
    get_cell_builder("gamma").extract(context, cfg, result)

    expected_mu = np.asarray(expected_mu, dtype=float)
    expected_latch = np.asarray(expected_latch, dtype=float)
    expected_removed = np.asarray(expected_removed, dtype=float)
    expected_shape = 10.0 * expected_mu
    expected_shape_latch = 10.0 * expected_latch

    for k in range(cfg.T):
        assert_close(result["mu"][0, 0, k], expected_mu[k], f"mu[{k}]")
        assert_close(result["z"][0, 0, k], expected_removed[k], f"z[{k}]")
        assert_close(
            result["gamma_mean_latch"][0, 0, k],
            expected_latch[k],
            f"mean latch[{k}]",
        )
        assert_close(
            result["gamma_shape_bound"][0, 0, k],
            expected_shape[k],
            f"shape[{k}]",
        )
        assert_close(
            result["gamma_shape_latch"][0, 0, k],
            expected_shape_latch[k],
            f"shape latch[{k}]",
        )

    return context, expected_mu, expected_latch, expected_removed


def main() -> None:
    cases = {
        0.5: (
            [0.03, 0.03, 0.015, 0.015, 0.025, 0.020, 0.020],
            [0.00, 0.00, 0.015, 0.015, 0.015, 0.020, 0.020],
            [0.00, 0.00, 0.015, 0.00, 0.00, 0.005, 0.00],
        ),
        1.0: (
            [0.03, 0.03, 0.00, 0.00, 0.01, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.03, 0.00, 0.00, 0.01, 0.00],
        ),
    }
    solved = {
        rho: solve_truth_table(rho, *expected)
        for rho, expected in cases.items()
    }
    context, mean, latch, removed = solved[0.5]
    print("PASS no-replacement Gamma project-ARD1 truth table")
    print("events              : mission, explicit idle, first repair,")
    print("                      unassigned idle, mission, later repair,")
    print("                      consecutive repair")
    print("initial latch       : 0 (current solver and validator convention)")
    print("repair rates tested :", sorted(solved))
    print("physical mean       :", mean.tolist())
    print("mean latch          :", latch.tolist())
    print("removed mean        :", removed.tolist())
    print("no-intervention nb  : 0")
    print("product-hull rows   :", 6 * context.F * context.L * (context.T - 1))
    print("Gamma rows audited  :", context.F * context.L * (13 * context.T - 2))


if __name__ == "__main__":
    main()
