"""Audit every replacement-enabled Gamma project-ARD1 transition."""

from __future__ import annotations

from gamma_replacement_truth_table_common import (
    EVENTS,
    assert_close,
    assert_simultaneous_actions_infeasible,
    check_common_trajectory,
    solve_truth_table,
)
from fleet_management.degradation_model.gamma_utils.gamma_diagnostics import (
    compare_estimate_with_actual,
    estimate_gamma_formulation,
)


def assert_redesigned_constraints(cfg, context, baseline_statistics) -> dict:
    """Audit the replacement-enabled ARD1 product-hull formulation."""
    names = {row.ConstrName for row in context.model.getConstrs()}
    if context.nb:
        raise AssertionError("replacement-enabled ARD1 still creates nb binaries")
    if getattr(context.model, "_tight_big_m_summary", None) is not None:
        raise AssertionError("replacement-enabled ARD1 still creates Big-M rows")
    forbidden_prefixes = (
        "nb_def_",
        "A_gamma_carry_",
        "mu_gamma_carry_",
        "gmu_gamma_hold_",
        "gA_gamma_hold_",
        "gmu_gamma_setm_",
        "gA_gamma_setm_",
        "gmu_gamma_setr_",
        "gA_gamma_setr_",
        "z_gamma_zero_",
        "z_gamma_repl_zero_",
        "A_gamma_repl_",
        "mu_gamma_repl_",
    )
    if any(name.startswith(forbidden_prefixes) for name in names):
        raise AssertionError("legacy replacement-enabled ARD1 rows remain")

    gamma = context.extras["gamma"]
    if gamma["dynamics_formulation"] != "ard1_replacement_product_hull":
        raise AssertionError("replacement-enabled ARD1 selected the wrong formulation")

    product = getattr(context.model, "_binary_product_summary", None)
    expected_products = 6 * cfg.F * cfg.L * (cfg.T - 1)
    if product is None or product["products"] != expected_products:
        raise AssertionError(
            f"recorded {None if product is None else product['products']} "
            f"ARD1 products; expected {expected_products}"
        )
    if product["linear_rows"] != 3 * expected_products:
        raise AssertionError("each ARD1 product must have exactly three hull rows")

    estimate = estimate_gamma_formulation(cfg, allow_replacement=True)
    comparison = compare_estimate_with_actual(estimate, baseline_statistics)
    if not comparison["known_subtotal_matches_actual"]:
        raise AssertionError(
            "ARD1 replacement estimate differs from Gurobi: "
            f"{comparison['non_gamma_remainder']}"
        )
    return product


def main() -> None:
    cfg, context, result, baseline_statistics = solve_truth_table("ard1")
    expected_mean = [0.03, 0.005, 0.005, 0.015, 0.010, 0.005, 0.005, 0.005]
    expected_latch = [0.0, 0.005, 0.005, 0.005, 0.010, 0.005, 0.005, 0.005]
    expected_removed = [0.0, 0.0, 0.0, 0.0, 0.005, 0.0, 0.0, 0.0]
    check_common_trajectory(result, expected_mean, expected_removed)
    for k, expected in enumerate(expected_latch):
        assert_close(
            result["gamma_mean_latch"][0, 0, k],
            expected,
            f"mean latch[{k}]",
        )
        assert_close(
            result["gamma_shape_latch"][0, 0, k],
            10.0 * expected,
            f"shape latch[{k}]",
        )
    product = assert_redesigned_constraints(cfg, context, baseline_statistics)
    assert_simultaneous_actions_infeasible("ard1")

    print("PASS replacement-enabled Gamma project-ARD1 truth table")
    print("events              :", ", ".join(EVENTS))
    print("physical mean       :", expected_mean)
    print("mean latch          :", expected_latch)
    print("removed mean        :", expected_removed)
    print("binary products     :", product["products"])
    print("product-hull rows   :", product["linear_rows"])
    print("formulation         : direct balances and exact product hulls")


if __name__ == "__main__":
    main()
