"""Regression for the modular Gamma product-hull formulations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from fleet_management.config import load_config
from fleet_management.degradation_model.base import (
    build_fleet,
    resolve_run_options,
    solve_mixed,
)
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
            "verbose": 0,
        }
    )


def ardinf_no_replacement_config():
    return load_config(
        {
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
    )


def check_ard1_replacement_product_hull() -> dict:
    cfg = ard1_config()
    options = resolve_run_options(cfg)
    context = build_fleet(cfg, options, model_name="gamma_ard1_replacement_product_hull")
    actual = collect_gurobi_model_statistics(context.model)
    estimate = estimate_gamma_formulation(
        cfg, allow_replacement=options["allow_replacement"]
    )
    comparison = compare_estimate_with_actual(estimate, actual)

    if actual["indicator_constraints"] != 0:
        raise AssertionError("ARD1 still contains Gamma indicator constraints")
    if actual["general_constraints"] != 0:
        raise AssertionError("ARD1 unexpectedly contains general constraints")
    if not comparison["known_subtotal_matches_actual"]:
        raise AssertionError(
            "ARD1 estimate differs from Gurobi: "
            f"{comparison['non_gamma_remainder']}"
        )
    if estimate["gamma_attributable"]["general_constraint_total"] != 0:
        raise AssertionError("ARD1 estimator still counts Gamma indicators")

    if context.nb:
        raise AssertionError("replacement-enabled ARD1 still creates nb binaries")
    if getattr(context.model, "_tight_big_m_summary", None) is not None:
        raise AssertionError("replacement-enabled ARD1 still creates Big-M rows")
    if context.extras["gamma"]["dynamics_formulation"] != (
        "ard1_replacement_product_hull"
    ):
        raise AssertionError("replacement-enabled ARD1 selected wrong dynamics")

    product = getattr(context.model, "_binary_product_summary", None)
    expected_products = 6 * cfg.F * cfg.L * (cfg.T - 1)
    if product is None or product["products"] != expected_products:
        raise AssertionError("replacement-enabled ARD1 product count is wrong")
    if product["linear_rows"] != 3 * expected_products:
        raise AssertionError("each ARD1 binary product must have three hull rows")

    gamma = context.extras["gamma"]
    strict_state_bounds = 0
    strict_latch_bounds = 0
    for i, l in gamma["cells"]:
        bounds = gamma["reachable_upper_bounds"][i, l]
        shape_limit = float(gamma["maximum_shape"][i, l])
        mean_limit = float(context.tau[i, l])
        for key in ("mean", "shape", "removed_mean"):
            values = np.asarray(bounds[key], dtype=float)
            if values.shape != (context.T,) or np.any(~np.isfinite(values)):
                raise AssertionError(f"ARD1 has invalid reachable {key} bounds")
            if np.any(values < 0.0):
                raise AssertionError(f"ARD1 has negative reachable {key} bounds")
        if np.any(bounds["mean"] > mean_limit + 1e-12):
            raise AssertionError("ARD1 mean bounds exceed tau")
        if np.any(bounds["shape"] > shape_limit + 1e-12):
            raise AssertionError("ARD1 shape bounds exceed A_max")

        for k in range(context.T):
            if abs(context.mu_var[i, l, k].UB - bounds["mean"][k]) > 1e-12:
                raise AssertionError("ARD1 did not apply its mean bound")
            if abs(gamma["A_var"][i, l, k].UB - bounds["shape"][k]) > 1e-12:
                raise AssertionError("ARD1 did not apply its shape bound")
            if abs(context.z_var[i, l, k].UB - bounds["removed_mean"][k]) > 1e-12:
                raise AssertionError("ARD1 did not apply its removed-mean bound")
        strict_state_bounds += int(np.count_nonzero(bounds["mean"] < mean_limit - 1e-12))
        strict_state_bounds += int(np.count_nonzero(bounds["shape"] < shape_limit - 1e-12))

        if (i, l) in gamma["ard1_cells"]:
            if np.any(bounds["mean_latch"] > bounds["mean"] + 1e-12):
                raise AssertionError("ARD1 mean-latch bounds exceed state bounds")
            if np.any(bounds["shape_latch"] > bounds["shape"] + 1e-12):
                raise AssertionError("ARD1 shape-latch bounds exceed state bounds")
            for k in range(context.T):
                if abs(
                    gamma["mean_latch"][i, l, k].UB
                    - bounds["mean_latch"][k]
                ) > 1e-12:
                    raise AssertionError("ARD1 did not apply its mean-latch bound")
                if abs(
                    gamma["shape_latch"][i, l, k].UB
                    - bounds["shape_latch"][k]
                ) > 1e-12:
                    raise AssertionError("ARD1 did not apply its shape-latch bound")
            strict_latch_bounds += int(
                np.count_nonzero(bounds["mean_latch"] < mean_limit - 1e-12)
            )
            strict_latch_bounds += int(
                np.count_nonzero(bounds["shape_latch"] < shape_limit - 1e-12)
            )

    if strict_state_bounds == 0:
        raise AssertionError("ARD1 produced no tighter time-dependent bounds")

    return {
        "variables": actual["variables"],
        "linear_constraints": actual["linear_constraints"],
        "indicator_constraints": actual["indicator_constraints"],
        "binary_products": product["products"],
        "product_hull_rows": product["linear_rows"],
        "strict_state_bounds": strict_state_bounds,
        "strict_latch_bounds": strict_latch_bounds,
    }


def check_ardinf_product_hull() -> dict:
    cfg = ardinf_no_replacement_config()
    options = resolve_run_options(cfg)
    context = build_fleet(cfg, options, model_name="gamma_ardinf_product_hull")
    actual = collect_gurobi_model_statistics(context.model)
    estimate = estimate_gamma_formulation(cfg, allow_replacement=False)
    comparison = compare_estimate_with_actual(estimate, actual)

    if not comparison["known_subtotal_matches_actual"]:
        raise AssertionError(
            "ARD-inf product-hull estimate differs from Gurobi: "
            f"{comparison['non_gamma_remainder']}"
        )
    if context.extras["gamma"]["dynamics_formulation"] != "ardinf_product_hull":
        raise AssertionError("no-replacement ARD-inf selected the wrong dynamics")
    if context.nb:
        raise AssertionError("no-replacement ARD-inf still creates nb binaries")
    if getattr(context.model, "_tight_big_m_summary", None) is not None:
        raise AssertionError("no-replacement ARD-inf still creates conditional Big-M rows")

    product = getattr(context.model, "_binary_product_summary", None)
    expected_products = 2 * cfg.F * cfg.L * (cfg.T - 1)
    if product is None or product["products"] != expected_products:
        raise AssertionError(
            f"recorded {None if product is None else product['products']} "
            f"binary products; expected {expected_products}"
        )
    if product["linear_rows"] != 3 * expected_products:
        raise AssertionError("each binary product must have exactly three hull rows")

    names = {row.ConstrName for row in context.model.getConstrs()}
    forbidden = ("A_gamma_carry_", "mu_gamma_carry_", "z_gamma_zero_")
    if any(name.startswith(forbidden) for name in names):
        raise AssertionError("legacy conditional state rows remain in product branch")

    # Exercise both special cases: k=0 multiplies a constant seed by m, while
    # k=1 uses the three-row hull for a bounded continuous previous state.
    context.model.addConstr(context.m_rep[0, 0, 0] == 1, name="force_seed_repair")
    context.model.addConstr(context.m_rep[1, 0, 1] == 1, name="force_later_repair")
    context.model.optimize()
    if context.model.SolCount == 0:
        raise AssertionError("forced ARD-inf product-hull case is infeasible")

    gamma = context.extras["gamma"]
    shape = gamma["A_var"]
    removed_shape = gamma["removed_shape"]
    tolerance = 1e-8
    maximum_error = 0.0
    for i in range(cfg.F):
        rho = float(cfg.rho[i, 0])
        for k in range(cfg.T):
            repair = float(context.m_rep[i, 0, k].X)
            previous_mean = (
                float(cfg.mu_0[i, 0])
                if k == 0 else float(context.mu_var[i, 0, k - 1].X)
            )
            previous_shape = (
                float(gamma["initial_shape"][i, 0])
                if k == 0 else float(shape[i, 0, k - 1].X)
            )
            maximum_error = max(
                maximum_error,
                abs(float(context.z_var[i, 0, k].X) - rho * previous_mean * repair),
                abs(float(removed_shape[i, 0, k].X) - rho * previous_shape * repair),
            )
    if maximum_error > tolerance:
        raise AssertionError(
            f"ARD-inf product identity error is {maximum_error:.3e}"
        )

    # Also exercise the normal solve/extract/diagnostics route, rather than
    # validating only the manually built model above.
    result = solve_mixed(cfg)
    if result["status"] != "optimal":
        raise AssertionError(
            f"unforced ARD-inf product-hull status is {result['status']!r}"
        )
    if result["gamma_dynamics_formulation"] != "ardinf_product_hull":
        raise AssertionError("result metadata lost the product-hull identity")
    recorded = result["gamma_formulation"]["binary_product_implementation"]
    if recorded["products"] != expected_products:
        raise AssertionError("result diagnostics lost the binary-product count")

    return {
        "variables": actual["variables"],
        "continuous_variables": actual["continuous_variables"],
        "binary_variables": actual["binary_variables"],
        "linear_constraints": actual["linear_constraints"],
        "binary_products": product["products"],
        "product_hull_rows": product["linear_rows"],
        "maximum_product_error": maximum_error,
    }


def check_ardinf_replacement_product_hull() -> dict:
    cfg = load_config(
        yaml.safe_load(
            (HERE / "gamma_tail_bound_public.yaml").read_text(encoding="utf-8")
        )
    )
    options = resolve_run_options(cfg)
    context = build_fleet(
        cfg,
        options,
        model_name="gamma_ardinf_replacement_product_hull",
    )
    actual = collect_gurobi_model_statistics(context.model)
    estimate = estimate_gamma_formulation(cfg, allow_replacement=True)
    comparison = compare_estimate_with_actual(estimate, actual)

    if not comparison["known_subtotal_matches_actual"]:
        raise AssertionError(
            "ARD-inf replacement product-hull estimate differs from Gurobi: "
            f"{comparison['non_gamma_remainder']}"
        )
    if context.extras["gamma"]["dynamics_formulation"] != (
        "ardinf_replacement_product_hull"
    ):
        raise AssertionError("replacement-enabled ARD-inf selected wrong dynamics")
    if context.nb:
        raise AssertionError("replacement-enabled ARD-inf still creates nb binaries")
    if getattr(context.model, "_tight_big_m_summary", None) is not None:
        raise AssertionError("replacement-enabled ARD-inf still creates Big-M rows")

    product = getattr(context.model, "_binary_product_summary", None)
    expected_products = 4 * cfg.F * cfg.L * (cfg.T - 1)
    if product is None or product["products"] != expected_products:
        raise AssertionError(
            f"recorded {None if product is None else product['products']} "
            f"replacement products; expected {expected_products}"
        )
    if product["linear_rows"] != 3 * expected_products:
        raise AssertionError("each replacement product must have three hull rows")

    names = {row.ConstrName for row in context.model.getConstrs()}
    forbidden = (
        "nb_def_", "A_gamma_carry_", "mu_gamma_carry_",
        "z_gamma_zero_", "A_gamma_repl_", "mu_gamma_repl_",
        "z_gamma_repl_zero_",
    )
    if any(name.startswith(forbidden) for name in names):
        raise AssertionError("legacy ARD-inf replacement conditional rows remain")

    return {
        "variables": actual["variables"],
        "continuous_variables": actual["continuous_variables"],
        "binary_variables": actual["binary_variables"],
        "linear_constraints": actual["linear_constraints"],
        "binary_products": product["products"],
        "product_hull_rows": product["linear_rows"],
    }


def main() -> None:
    product_hull = check_ardinf_product_hull()
    ardinf = check_ardinf_replacement_product_hull()
    ard1 = check_ard1_replacement_product_hull()

    print("PASS modular Gamma dynamic formulations")
    print("ARD-inf, no replacement:", product_hull)
    print("ARD-inf, replacement   :", ardinf)
    print("ARD1                   :", ard1)


if __name__ == "__main__":
    main()
