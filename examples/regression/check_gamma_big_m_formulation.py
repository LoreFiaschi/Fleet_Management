"""Regression for the modular Gamma tight Big-M dynamics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
                raise AssertionError(f"{name} has invalid reachable {key} bounds")
            if np.any(values < 0.0):
                raise AssertionError(f"{name} has negative reachable {key} bounds")
        if np.any(bounds["mean"] > mean_limit + 1e-12):
            raise AssertionError(f"{name} mean bounds exceed tau")
        if np.any(bounds["shape"] > shape_limit + 1e-12):
            raise AssertionError(f"{name} shape bounds exceed A_max")

        for k in range(context.T):
            if abs(context.mu_var[i, l, k].UB - bounds["mean"][k]) > 1e-12:
                raise AssertionError(f"{name} did not apply its mean bound")
            if abs(gamma["A_var"][i, l, k].UB - bounds["shape"][k]) > 1e-12:
                raise AssertionError(f"{name} did not apply its shape bound")
            if abs(context.z_var[i, l, k].UB - bounds["removed_mean"][k]) > 1e-12:
                raise AssertionError(f"{name} did not apply its removed-mean bound")
        strict_state_bounds += int(np.count_nonzero(bounds["mean"] < mean_limit - 1e-12))
        strict_state_bounds += int(np.count_nonzero(bounds["shape"] < shape_limit - 1e-12))

        if (i, l) in gamma["ard1_cells"]:
            if np.any(bounds["mean_latch"] > bounds["mean"] + 1e-12):
                raise AssertionError(f"{name} mean-latch bounds exceed state bounds")
            if np.any(bounds["shape_latch"] > bounds["shape"] + 1e-12):
                raise AssertionError(f"{name} shape-latch bounds exceed state bounds")
            for k in range(context.T):
                if abs(
                    gamma["mean_latch"][i, l, k].UB
                    - bounds["mean_latch"][k]
                ) > 1e-12:
                    raise AssertionError(f"{name} did not apply its mean-latch bound")
                if abs(
                    gamma["shape_latch"][i, l, k].UB
                    - bounds["shape_latch"][k]
                ) > 1e-12:
                    raise AssertionError(f"{name} did not apply its shape-latch bound")
            strict_latch_bounds += int(
                np.count_nonzero(bounds["mean_latch"] < mean_limit - 1e-12)
            )
            strict_latch_bounds += int(
                np.count_nonzero(bounds["shape_latch"] < shape_limit - 1e-12)
            )

    if strict_state_bounds == 0:
        raise AssertionError(f"{name} produced no tighter time-dependent bounds")

    return {
        "variables": actual["variables"],
        "linear_constraints": actual["linear_constraints"],
        "indicator_constraints": actual["indicator_constraints"],
        "strict_state_bounds": strict_state_bounds,
        "strict_latch_bounds": strict_latch_bounds,
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
