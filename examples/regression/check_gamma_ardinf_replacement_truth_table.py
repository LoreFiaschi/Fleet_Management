"""Audit every replacement-enabled Gamma ARD-infinity transition."""

from __future__ import annotations

from fleet_management.degradation_model.gamma_utils.gamma_diagnostics import (
    compare_estimate_with_actual,
    estimate_gamma_formulation,
)

from gamma_replacement_truth_table_common import (
    EVENTS,
    assert_simultaneous_actions_infeasible,
    check_common_trajectory,
    solve_truth_table,
)


def assert_redesigned_constraints(context) -> None:
    names = {row.ConstrName for row in context.model.getConstrs()}
    forbidden = (
        "nb_def_",
        "A_gamma_carry_",
        "mu_gamma_carry_",
        "z_gamma_zero_",
        "z_gamma_repl_zero_",
    )
    if any(name.startswith(forbidden) for name in names):
        raise AssertionError("legacy ARD-infinity replacement Big-M rows remain")
    if context.nb:
        raise AssertionError("ARD-infinity replacement still creates nb binaries")

    for i in range(context.F):
        for k in range(context.T):
            required = {
                f"m_gate_{i}_0_{k}",
                f"r_gate_{i}_0_{k}",
                f"maintenance_exclusive_{i}_0_{k}",
                f"mu_gamma_ardinf_balance_{i}_0_{k}",
                f"A_gamma_ardinf_balance_{i}_0_{k}",
                f"rel_gamma_{i}_0_{k}",
            }
            if k == 0:
                required.update(
                    {
                        f"z_gamma_ardinf_seed_{i}_0_0",
                        f"zA_gamma_ardinf_seed_{i}_0_0",
                        f"qRmu_gamma_ardinf_seed_{i}_0_0",
                        f"qRA_gamma_ardinf_seed_{i}_0_0",
                    }
                )
            else:
                for prefix in (
                    "z_gamma_ardinf_product",
                    "zA_gamma_ardinf_product",
                    "qRmu_gamma_ardinf_product",
                    "qRA_gamma_ardinf_product",
                ):
                    base = f"{prefix}_{i}_0_{k}"
                    required.update(
                        {
                            f"{base}_state_ub",
                            f"{base}_binary_ub",
                            f"{base}_lower",
                        }
                    )
            missing = required - names
            if missing:
                raise AssertionError(
                    f"ARD-infinity replacement rows missing: {sorted(missing)}"
                )


def main() -> None:
    cfg, context, result, baseline_statistics = solve_truth_table("ardinf")
    expected_mean = [0.03, 0.005, 0.005, 0.015, 0.0075, 0.005, 0.005, 0.005]
    expected_removed = [0.0, 0.0, 0.0, 0.0, 0.0075, 0.0, 0.0, 0.0]
    check_common_trajectory(result, expected_mean, expected_removed)
    assert_redesigned_constraints(context)
    assert_simultaneous_actions_infeasible("ardinf")

    estimate = estimate_gamma_formulation(cfg, allow_replacement=True)
    comparison = compare_estimate_with_actual(estimate, baseline_statistics)
    if not comparison["known_subtotal_matches_actual"]:
        raise AssertionError(
            "ARD-infinity replacement count estimate differs: "
            f"{comparison['non_gamma_remainder']}"
        )
    if context.extras["gamma"]["dynamics_formulation"] != (
        "ardinf_replacement_product_hull"
    ):
        raise AssertionError("wrong ARD-infinity replacement formulation metadata")

    print("PASS replacement-enabled Gamma ARD-infinity truth table")
    print("events              :", ", ".join(EVENTS))
    print("physical mean       :", expected_mean)
    print("removed mean        :", expected_removed)
    print("no-intervention nb  : 0")
    print("conditional Big-M   : 0")


if __name__ == "__main__":
    main()
