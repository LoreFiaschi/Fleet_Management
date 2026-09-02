"""A-priori one-factor-at-a-time formulation-size sweeps.

This module normalizes derived configurations and evaluates the analytical
shared/Gamma formulation count. It does not build or optimize a Gurobi model,
so its output describes deterministic formulation growth rather than
feasibility or computational runtime.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Iterable, Mapping

import yaml

from fleet_management.config import load_config
from fleet_management.degradation_model.gamma_utils.gamma_diagnostics import (
    estimate_gamma_formulation,
)


def _positive_unique(values: Iterable[int], name: str) -> list[int]:
    result = [int(value) for value in values]
    if not result or any(value <= 0 for value in result):
        raise ValueError(f"{name} candidates must be positive integers")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} candidates must not contain duplicates")
    return result


def _written_horizons(data: dict) -> tuple[int, int]:
    written = data["H"]
    if isinstance(written, (list, tuple)):
        if len(written) != 2:
            raise ValueError("H must be an integer or [H1, H2]")
        return int(written[0]), int(written[1])
    value = int(written)
    return value, value


def _case_data(source: dict, parameter: str, value: int, H1: int) -> dict:
    case = deepcopy(source)
    if parameter in {"F", "M", "L"}:
        case[parameter] = value
    elif parameter == "T":
        if value <= H1:
            raise ValueError(f"T={value} must exceed fixed H1={H1}")
        case["H"] = [H1, value - H1]
    else:
        raise ValueError(f"unsupported sweep parameter {parameter!r}")
    return case


def _count_case(data: dict, parameter: str, value: int) -> dict:
    cfg = load_config(data)
    allow_replacement = bool(cfg.options.get("allow_replacement", True))
    estimate = estimate_gamma_formulation(
        cfg, allow_replacement=allow_replacement
    )

    shared_binary = estimate["shared"]["binary_variables"]
    shared_continuous = estimate["shared"]["continuous_variables"]
    gamma_variables = estimate["gamma_attributable"]["continuous_variables"]
    shared_rows = estimate["shared"]["linear_constraints"]
    gamma_rows = estimate["gamma_attributable"]["linear_constraints"]
    known = estimate["known_subtotal"]

    binary_total = sum(shared_binary.values())
    continuous_total = sum(shared_continuous.values()) + sum(
        gamma_variables.values()
    )
    repeatability_rows = sum(
        gamma_rows[name]
        for name in (
            "shape_repeatability",
            "physical_mean_repeatability",
            "ard1_mean_latch_repeatability",
            "ard1_shape_latch_repeatability",
        )
    )

    return {
        "changed_parameter": parameter,
        "changed_value": value,
        "dimensions": {
            "F": cfg.F,
            "M": cfg.M,
            "L": cfg.L,
            "H1": cfg.H1,
            "H2": cfg.H2,
            "T": cfg.T,
        },
        "gamma_cells": estimate["gamma_cells"],
        "gamma_ard1_cells": estimate["gamma_ard1_cells"],
        "allow_replacement": allow_replacement,
        "counts": {
            "variables": known["variables"],
            "integer_variables": binary_total,
            "binary_variables": binary_total,
            "continuous_variables": continuous_total,
            "linear_constraints": known["linear_constraints"],
            "general_constraints": known["general_constraints"],
            "quadratic_constraints": known["quadratic_constraints"],
        },
        "selected_breakdown": {
            "assignment_variables": shared_binary["assignment_x"],
            "physical_mean_variables": shared_continuous["physical_mean_mu"],
            "gamma_shape_variables": gamma_variables["bounding_shape_A"],
            "gamma_ard1_latch_variables": (
                gamma_variables["ard1_physical_mean_latch"]
                + gamma_variables["ard1_bounding_shape_latch"]
            ),
            "gamma_big_m_state_rows": gamma_rows["big_m_state_dynamics"],
            "gamma_ard1_latch_big_m_rows": (
                gamma_rows["ard1_mean_latch_big_m_dynamics"]
                + gamma_rows["ard1_shape_latch_big_m_dynamics"]
            ),
            "gamma_reliability_rows": gamma_rows["tail_reliability"],
            "gamma_repeatability_rows": repeatability_rows,
        },
        "normalized_counts": {
            "variables_per_time_step": known["variables"] / cfg.T,
            "linear_constraints_per_time_step": (
                known["linear_constraints"] / cfg.T
            ),
            "variables_per_gamma_cell_time_step": (
                known["variables"] / (estimate["gamma_cells"] * cfg.T)
                if estimate["gamma_cells"] else None
            ),
            "linear_constraints_per_gamma_cell_time_step": (
                known["linear_constraints"]
                / (estimate["gamma_cells"] * cfg.T)
                if estimate["gamma_cells"] else None
            ),
        },
    }


def _growth(cases: list[dict], parameter: str) -> dict | None:
    if len(cases) < 2:
        return None
    first, last = cases[0], cases[-1]
    delta = last["changed_value"] - first["changed_value"]
    if delta == 0:
        return None
    result = {
        "from_value": first["changed_value"],
        "to_value": last["changed_value"],
        "parameter_increase": delta,
    }
    for name in (
        "variables", "integer_variables", "binary_variables",
        "continuous_variables", "linear_constraints",
    ):
        increase = last["counts"][name] - first["counts"][name]
        result[f"added_{name}"] = increase
        result[f"{name}_per_unit_{parameter}"] = increase / delta
    return result


def sweep_formulation_dimensions(
    input_path: str | Path,
    *,
    candidates: Mapping[str, Iterable[int]] | None = None,
    output_path: str | Path | None = None,
) -> dict:
    """Calculate deterministic formulation counts while varying F/M/L/T."""
    source = Path(input_path)
    if source.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("formulation-sweep input must be YAML")
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("formulation-sweep input must contain a YAML mapping")

    H1, H2 = _written_horizons(data)
    defaults = {
        "F": [2, 4, 6, 8],
        "M": [1, 2, 3, 4],
        "L": [1, 2, 4, 8],
        "T": [H1 + H2, H1 + 2 * H2, H1 + 3 * H2, H1 + 4 * H2],
    }
    supplied = {} if candidates is None else dict(candidates)
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError(f"unsupported sweep parameters: {sorted(unknown)}")

    sweeps = {}
    for parameter in ("F", "M", "L", "T"):
        values = _positive_unique(
            supplied.get(parameter, defaults[parameter]), parameter
        )
        cases = [
            _count_case(_case_data(data, parameter, value, H1), parameter, value)
            for value in values
        ]
        sweeps[parameter] = {
            "varied_parameter": parameter,
            "cases": cases,
            "first_to_last_growth": _growth(cases, parameter),
        }

    baseline_cfg = load_config(data)
    report = {
        "report": "Deterministic one-factor-at-a-time formulation-size sweep",
        "input": str(source),
        "method": "analytical pre-solve count; no Gurobi model is optimized",
        "interpretation": {
            "counts": (
                "Counts are deterministic properties of the normalized input "
                "and current shared/Gamma formulation."
            ),
            "not_measured": (
                "The sweep does not test feasibility, runtime, branch-and-bound "
                "nodes, memory consumption, or MIP convergence."
            ),
            "one_factor_at_a_time": (
                "Within each sweep only the named dimension changes; all other "
                "base-scenario parameters remain fixed."
            ),
        },
        "baseline_dimensions": {
            "F": baseline_cfg.F,
            "M": baseline_cfg.M,
            "L": baseline_cfg.L,
            "H1": baseline_cfg.H1,
            "H2": baseline_cfg.H2,
            "T": baseline_cfg.T,
        },
        "formulas": {
            "assignment_variables": "F * (M + 1) * T",
            "gamma_cells": "F * L for a uniform Gamma fleet",
            "gamma_shape_variables": "F * L * T for a uniform Gamma fleet",
            "gamma_big_m_state_rows": (
                "12 * F * L * T without replacement; "
                "18 * F * L * T with replacement"
            ),
            "gamma_reliability_rows": "F * L * T for a uniform Gamma fleet",
        },
        "sweeps": sweeps,
    }

    if output_path is not None:
        target = Path(output_path)
        if target.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError("output_path must end in .yaml or .yml")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    return report
