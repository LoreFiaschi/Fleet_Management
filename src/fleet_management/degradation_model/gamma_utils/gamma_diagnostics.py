"""Complexity and performance diagnostics for the modular Gamma formulation.

The estimator explains counts before optimization.  Actual Gurobi statistics
are collected separately and can be compared with the estimate.  In a mixed
fleet, the difference between actual totals and the shared-plus-Gamma subtotal
belongs to the other degradation blocks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import gurobipy as gp
from gurobipy import GRB


def estimate_gamma_formulation(cfg, *, allow_replacement: bool) -> dict[str, Any]:
    """Return formula-generated shared and Gamma formulation counts."""

    F, M, L, T = cfg.F, cfg.M, cfg.L, cfg.T
    gamma_cells = sum(
        str(cfg.model[i, l]) == "gamma"
        for i in range(F)
        for l in range(L)
    )
    gamma_ard1_cells = sum(
        str(cfg.model[i, l]) == "gamma"
        and str(cfg.repair_model[i, l]) == "ard1"
        for i in range(F)
        for l in range(L)
    )
    replacement = int(bool(allow_replacement))
    gamma_ardinf_product_cells = sum(
        str(cfg.model[i, l]) == "gamma"
        and str(cfg.repair_model[i, l]) == "ardinf"
        and not allow_replacement
        for i in range(F)
        for l in range(L)
    )
    gamma_ard1_product_cells = sum(
        str(cfg.model[i, l]) == "gamma"
        and str(cfg.repair_model[i, l]) == "ard1"
        and not allow_replacement
        for i in range(F)
        for l in range(L)
    )
    gamma_product_cells = (
        gamma_ardinf_product_cells + gamma_ard1_product_cells
    )
    gamma_big_m_cells = gamma_cells - gamma_product_cells
    gamma_ard1_big_m_cells = gamma_ard1_cells - gamma_ard1_product_cells

    shared_binary = {
        "assignment_x": F * (M + 1) * T,
        "repair_m": F * L * T,
        "replacement_r": replacement * F * L * T,
        "no_intervention_nb": (
            (F * L - gamma_product_cells) * T
        ),
    }
    shared_continuous = {
        "physical_mean_mu": F * L * T,
        "removed_mean_z": F * L * T,
        "safety_u": T,
    }
    shared_linear = {
        "vehicle_assignment": F * T,
        "mission_demand": M * T,
        "safety_regularisation": F * T,
    }

    gamma_variables = {
        "bounding_shape_A": gamma_cells * T,
        "ardinf_removed_shape": gamma_ardinf_product_cells * T,
        "ard1_repairable_mean": gamma_ard1_product_cells * T,
        "ard1_repairable_shape": gamma_ard1_product_cells * T,
        "ard1_physical_mean_latch": gamma_ard1_cells * T,
        "ard1_bounding_shape_latch": gamma_ard1_cells * T,
    }
    gamma_linear = {
        # No-replacement product-hull cells need only m<=x0. Other branches additionally
        # define nb and, when enabled, gate replacement.
        "maintenance_gating": (
            gamma_product_cells * T
            + gamma_big_m_cells * T * (2 + replacement)
        ),
        "tail_reliability": gamma_cells * T,
        "shape_repeatability": gamma_cells,
        "physical_mean_repeatability": gamma_cells,
        "ard1_mean_latch_repeatability": 0,
        "ard1_shape_latch_repeatability": 0,
        # No-replacement ARD-inf: two state balances at every step, two
        # binary-times-constant seed equalities, and two three-row product
        # hulls at every later step.
        "ardinf_product_hull_dynamics": (
            gamma_ardinf_product_cells * (8 * T - 4)
        ),
        # No-replacement ARD1: two selected active-damage products, five
        # balance equations per step, and no latch terminal rows.
        "ard1_product_hull_dynamics": (
            gamma_ard1_product_cells * (11 * T - 4)
        ),
        # Every conditional equality in the remaining branches becomes an
        # upper and lower Big-M row.
        "big_m_state_dynamics": (
            2 * gamma_big_m_cells * T * (6 + 3 * replacement)
        ),
        "ard1_mean_latch_big_m_dynamics": (
            2 * gamma_ard1_big_m_cells * T * (2 + replacement)
        ),
        "ard1_shape_latch_big_m_dynamics": (
            2 * gamma_ard1_big_m_cells * T * (2 + replacement)
        ),
    }
    gamma_general = {}

    shared_variable_total = sum(shared_binary.values()) + sum(shared_continuous.values())
    shared_linear_total = sum(shared_linear.values())
    gamma_variable_total = sum(gamma_variables.values())
    gamma_linear_total = sum(gamma_linear.values())
    gamma_general_total = sum(gamma_general.values())
    uniform_gamma = gamma_cells == F * L

    return {
        "dimensions": {"F": F, "M": M, "L": L, "T": T},
        "gamma_cells": gamma_cells,
        "gamma_ard1_cells": gamma_ard1_cells,
        "gamma_ardinf_product_cells": gamma_ardinf_product_cells,
        "gamma_ard1_product_cells": gamma_ard1_product_cells,
        "gamma_big_m_cells": gamma_big_m_cells,
        "uniform_gamma": uniform_gamma,
        "allow_replacement": bool(allow_replacement),
        "definitions": {
            "linear_constraints": (
                "Ordinary Gurobi rows reported by Model.NumConstrs."
            ),
            "general_constraints": (
                "Gamma introduces no general constraints. Nonzero mixed-fleet "
                "totals belong to other blocks."
            ),
            "shared": (
                "Variables and rows created once for the complete fleet."
            ),
            "gamma_attributable": (
                "Additional state and cell constraints introduced for Gamma cells."
            ),
        },
        "formulas": {
            "gamma_shape_variables": "Gamma component cells * T",
            "gamma_ard1_latch_variables": "2 * Gamma ARD1 component cells * T",
            "gamma_ardinf_removed_shape_variables": (
                "Gamma ARD-inf/no-replacement component cells * T"
            ),
            "gamma_ard1_repairable_variables": (
                "2 * Gamma ARD1/no-replacement component cells * T"
            ),
            "gamma_ardinf_product_hull_rows": (
                "Gamma ARD-inf/no-replacement component cells * (8*T - 4)"
            ),
            "gamma_ard1_product_hull_rows": (
                "Gamma ARD1/no-replacement component cells * (11*T - 4)"
            ),
            "gamma_big_m_state_rows": (
                "2 * remaining Gamma component cells * T * "
                "(6 + 3*I_replacement)"
            ),
            "gamma_ard1_latch_big_m_rows": (
                "4 * replacement-enabled Gamma ARD1 component cells * T * "
                "(2 + I_replacement)"
            ),
            "gamma_reliability_rows": "Gamma component cells * T",
            "gamma_repeatability_rows": (
                "2 * Gamma component cells"
            ),
            "gamma_maintenance_gating_rows": (
                "no-replacement product-hull cells*T + remaining Gamma cells*T*"
                "(2 + I_replacement)"
            ),
        },
        "shared": {
            "binary_variables": shared_binary,
            "continuous_variables": shared_continuous,
            "linear_constraints": shared_linear,
            "variable_total": shared_variable_total,
            "linear_constraint_total": shared_linear_total,
        },
        "gamma_attributable": {
            "continuous_variables": gamma_variables,
            "linear_constraints": gamma_linear,
            "general_constraints": gamma_general,
            "variable_total": gamma_variable_total,
            "linear_constraint_total": gamma_linear_total,
            "general_constraint_total": gamma_general_total,
        },
        "known_subtotal": {
            "variables": shared_variable_total + gamma_variable_total,
            "linear_constraints": shared_linear_total + gamma_linear_total,
            "general_constraints": gamma_general_total,
            "quadratic_constraints": 0,
        },
        "interpretation": (
            "For a uniform Gamma fleet the known subtotal should equal the "
            "actual Gurobi totals. For a mixed fleet, remaining variables and "
            "constraints are contributed by non-Gamma degradation blocks. "
            "Gamma without replacement uses exact binary-product hulls and no "
            "no-intervention binary. Replacement-enabled Gamma dynamics use "
            "two bounded Big-M rows per conditional equality. Gamma contributes "
            "no indicator constraints."
        ),
    }


def collect_gurobi_model_statistics(model) -> dict[str, Any]:
    """Collect actual formulation and optimizer counters from a Gurobi model."""

    model.update()
    general_constraints = list(model.getGenConstrs())
    indicator_constraints = sum(
        int(item.GenConstrType == GRB.GENCONSTR_INDICATOR)
        for item in general_constraints
    )

    def optional_float(attribute: str):
        try:
            return float(getattr(model, attribute))
        except (AttributeError, TypeError):
            return None

    return {
        "gurobi_version": ".".join(str(value) for value in gp.gurobi.version()),
        "variables": int(model.NumVars),
        "continuous_variables": int(model.NumVars - model.NumIntVars),
        "integer_variables": int(model.NumIntVars),
        "binary_variables": int(model.NumBinVars),
        "linear_constraints": int(model.NumConstrs),
        "general_constraints": int(model.NumGenConstrs),
        "indicator_constraints": int(indicator_constraints),
        "quadratic_constraints": int(model.NumQConstrs),
        "nonzeros": int(model.NumNZs),
        "solutions_found": int(model.SolCount),
        "gurobi_runtime_seconds": optional_float("Runtime"),
        "branch_and_bound_nodes": optional_float("NodeCount"),
        "simplex_iterations": optional_float("IterCount"),
        "barrier_iterations": optional_float("BarIterCount"),
        "work_units": optional_float("Work"),
    }


def compare_estimate_with_actual(estimate: dict, actual: dict) -> dict[str, Any]:
    """Explain the portion of actual totals not covered by shared/Gamma counts."""

    known = estimate["known_subtotal"]
    remainder = {
        "variables": actual["variables"] - known["variables"],
        "linear_constraints": (
            actual["linear_constraints"] - known["linear_constraints"]
        ),
        "general_constraints": (
            actual["general_constraints"] - known["general_constraints"]
        ),
        "quadratic_constraints": (
            actual["quadratic_constraints"] - known["quadratic_constraints"]
        ),
    }
    return {
        "known_subtotal_matches_actual": all(value == 0 for value in remainder.values()),
        "non_gamma_remainder": remainder,
        "interpretation": (
            "A zero remainder is expected for a uniform Gamma fleet. Positive "
            "mixed-fleet remainders are variables and constraints belonging to "
            "the other degradation blocks."
        ),
    }
