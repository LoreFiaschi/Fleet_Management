"""Small outer loop for comparing operating-horizon lengths."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

import numpy as np
import yaml

from fleet_management.solver import solve


FORMULATION_KEYS = (
    "variables",
    "continuous_variables",
    "integer_variables",
    "binary_variables",
    "linear_constraints",
    "general_constraints",
    "indicator_constraints",
    "quadratic_constraints",
    "nonzeros",
)


def _gamma_calibration_summary(result: dict) -> dict:
    cells = result.get("gamma_calibration", [])
    safe_counts = [
        int(value)
        for cell in cells
        for value in cell.get("maximum_safe_counts", [])
    ]
    common_rates = sorted({float(cell["common_rate"]) for cell in cells})

    bounded = result.get("gamma_shape_increment")
    bounded_values = (
        np.asarray(bounded, dtype=float).ravel()
        if bounded is not None
        else np.asarray([], dtype=float)
    )
    return {
        "method": result.get("gamma_calibration_method"),
        "gamma_cells": len(cells),
        "common_rates": common_rates,
        "minimum_safe_count": min(safe_counts) if safe_counts else None,
        "maximum_safe_count": max(safe_counts) if safe_counts else None,
        "minimum_bounded_increment_shape": (
            float(np.min(bounded_values)) if bounded_values.size else None
        ),
        "maximum_bounded_increment_shape": (
            float(np.max(bounded_values)) if bounded_values.size else None
        ),
        "increment_types": sum(int(cell["increment_types"]) for cell in cells),
        "increment_opportunities": sum(
            int(cell["increment_opportunities"]) for cell in cells
        ),
        "tail_constraints": sum(int(cell["tail_constraints"]) for cell in cells),
        "calibration_seconds": result.get("performance", {}).get(
            "gamma_calibration_seconds"
        ),
    }


def _formulation_growth(rows: list[dict]) -> dict | None:
    """Summarize first-to-last model growth per added time step."""
    if len(rows) < 2:
        return None
    first, last = rows[0], rows[-1]
    delta_t = int(last["T"] - first["T"])
    if delta_t == 0:
        return None

    growth = {
        "from_H2": first["H2"],
        "to_H2": last["H2"],
        "added_time_steps": delta_t,
    }
    for key in (
        "variables",
        "binary_variables",
        "linear_constraints",
        "general_constraints",
        "indicator_constraints",
        "quadratic_constraints",
        "nonzeros",
    ):
        start = first["formulation"][key]
        stop = last["formulation"][key]
        growth[f"added_{key}"] = stop - start
        growth[f"{key}_per_added_time_step"] = (stop - start) / delta_t
    return growth


def sweep_operating_horizons(
    input_path: str | Path,
    operating_horizons: Iterable[int],
    *,
    transitory_budget: float | None = None,
    output_path: str | Path | None = None,
) -> dict:
    """Solve one scenario for each requested ``H2`` and select the best.

    This intentionally performs no interpolation of time-indexed input data.
    Scalars and time-constant mission profiles broadcast naturally; an explicit
    profile whose length does not match a candidate horizon is rejected by the
    normal configuration validation.
    """

    source = Path(input_path)
    if source.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("the initial horizon sweep supports YAML inputs only")
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("horizon-sweep input must contain a YAML mapping")

    written_horizon = data.get("H")
    if isinstance(written_horizon, (list, tuple)):
        if len(written_horizon) != 2:
            raise ValueError("H must be an integer or [H1, H2]")
        H1 = int(written_horizon[0])
    else:
        H1 = int(written_horizon)

    candidates = [int(value) for value in operating_horizons]
    if not candidates or any(value <= 0 for value in candidates):
        raise ValueError("operating_horizons must contain positive integers")
    if len(set(candidates)) != len(candidates):
        raise ValueError("operating_horizons must not contain duplicates")

    budget = data.get("transitory_budget") if transitory_budget is None else transitory_budget
    if budget is None:
        raise ValueError("a transitory_budget is required for a horizon sweep")
    budget = float(budget)

    rows: list[dict] = []
    with TemporaryDirectory(prefix="fleet-horizon-sweep-") as directory:
        temporary = Path(directory)
        for H2 in candidates:
            case = deepcopy(data)
            case["H"] = [H1, H2]
            case["objective_mode"] = "operating_average"
            case["transitory_budget"] = budget
            case_input = temporary / f"H2_{H2}_input.yaml"
            case_output = temporary / f"H2_{H2}_result.yaml"
            case_input.write_text(
                yaml.safe_dump(case, sort_keys=False), encoding="utf-8"
            )
            result = solve(str(case_input), str(case_output))
            performance = result.get("performance", {})
            gamma_formulation = result.get("gamma_formulation", {})
            formulation = {
                key: performance.get(key) for key in FORMULATION_KEYS
            }
            formulation.update({
                "gamma_cells": gamma_formulation.get("gamma_cells"),
                "gamma_ard1_cells": gamma_formulation.get("gamma_ard1_cells"),
                "known_shared_gamma_subtotal": gamma_formulation.get(
                    "known_subtotal"
                ),
                "known_subtotal_matches_actual": gamma_formulation.get(
                    "comparison", {}
                ).get("known_subtotal_matches_actual"),
            })
            rows.append({
                "H1": H1,
                "H2": H2,
                "T": H1 + H2,
                "dimensions": {
                    "F": int(result.get("F", data["F"])),
                    "M": int(result.get("M", data["M"])),
                    "L": int(result.get("L", data["L"])),
                    "H1": H1,
                    "H2": H2,
                    "T": H1 + H2,
                },
                "status": result.get("status"),
                "objective": result.get("objective"),
                "J_trans": result.get("J_trans"),
                "J_op": result.get("J_op"),
                "J_op_average": result.get("J_op_average"),
                "objective_bound": result.get("bound"),
                "mip_gap": result.get("mip_gap"),
                "optimizer_seconds": performance.get("optimizer_call_seconds"),
                "solutions_found": performance.get("solutions_found"),
                "formulation": formulation,
                "calibration": _gamma_calibration_summary(result),
                "timing": {
                    "model_construction_seconds": performance.get(
                        "model_construction_seconds"
                    ),
                    "gamma_calibration_seconds": performance.get(
                        "gamma_calibration_seconds"
                    ),
                    "optimizer_call_seconds": performance.get(
                        "optimizer_call_seconds"
                    ),
                    "solution_extraction_seconds": performance.get(
                        "solution_extraction_seconds"
                    ),
                    "backend_wall_seconds": performance.get(
                        "backend_wall_seconds"
                    ),
                    "branch_and_bound_nodes": performance.get(
                        "branch_and_bound_nodes"
                    ),
                    "simplex_iterations": performance.get("simplex_iterations"),
                    "work_units": performance.get("work_units"),
                },
            })

    eligible = [
        row for row in rows
        if row["J_op_average"] is not None
        and row["status"] not in {"infeasible", "inf_or_unbounded", "unbounded"}
        and row["J_trans"] <= budget + 1e-8
    ]
    best = min(eligible, key=lambda row: row["J_op_average"]) if eligible else None
    report = {
        "input": str(source),
        "objective": "minimize J_op / H2 subject to J_trans <= B_trans",
        "varied_parameter": "H2",
        "fixed_dimensions": {
            "F": int(data["F"]),
            "M": int(data["M"]),
            "L": int(data["L"]),
            "H1": H1,
        },
        "H1": H1,
        "transitory_budget": budget,
        "cases": rows,
        "formulation_growth": _formulation_growth(rows),
        "best_H2": None if best is None else best["H2"],
        "best_J_op_average": None if best is None else best["J_op_average"],
    }

    if output_path is not None:
        target = Path(output_path)
        if target.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError("output_path must end in .yaml or .yml")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    return report