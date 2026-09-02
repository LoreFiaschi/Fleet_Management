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
        "continuous_variables",
        "integer_variables",
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


def _select_horizon_cases(
    rows: list[dict],
) -> tuple[dict | None, dict | None]:
    """Return the best proven case and the best feasible."""
    feasible = [
        row
        for row in rows
        if row["J_op_average"] is not None
        and row["status"]
        not in {"infeasible", "inf_or_unbounded", "unbounded"}
    ]
    proven = [
        row
        for row in feasible
        if row["status"] == "optimal"
        and row.get("mip_gap") is not None
        and np.isfinite(row["mip_gap"])
        and float(row["mip_gap"]) <= 1e-8
    ]

    best_feasible = (
        min(feasible, key=lambda row: row["J_op_average"])
        if feasible
        else None
    )
    best_proven = (
        min(proven, key=lambda row: row["J_op_average"])
        if proven
        else None
    )
    return best_proven, best_feasible


def _is_feasible_case(row: dict) -> bool:
    return (
        row.get("J_op_average") is not None
        and row.get("status")
        not in {"infeasible", "inf_or_unbounded", "unbounded"}
    )


def _gap_qualifies_for_stopping(row: dict, maximum_mip_gap: float) -> bool:
    """Whether one result is accurate enough to inform an early stop."""
    gap = row.get("mip_gap")
    return gap is not None and np.isfinite(gap) and float(gap) <= maximum_mip_gap


def _annotate_latest_gradient(
    rows: list[dict],
    *,
    gradient_tolerance: float,
    maximum_mip_gap: float,
) -> None:
    """Add the observed cost slope and its stopping eligibility to the last row.

    The normalized gradient is the relative change in ``J_op/H2`` per added
    unit of ``H2``.  A gradient is allowed to stop an adaptive sweep only when
    both adjacent cases are feasible and either optimal or within the selected
    MIP-gap threshold.
    """
    current = rows[-1]
    current.update({
        "cost_gradient_per_H2": None,
        "relative_cost_gradient_per_H2": None,
        "gradient_classification": "unavailable",
        "gradient_stopping_eligible": False,
    })
    if len(rows) < 2:
        return

    previous = rows[-2]
    if not (
        _is_feasible_case(previous)
        and _is_feasible_case(current)
    ):
        return

    delta_horizon = int(current["H2"]) - int(previous["H2"])
    if delta_horizon <= 0:
        raise ValueError("adaptive horizon candidates must be strictly increasing")
    previous_cost = float(previous["J_op_average"])
    current_cost = float(current["J_op_average"])
    gradient = (current_cost - previous_cost) / delta_horizon
    scale = max(abs(previous_cost), np.finfo(float).eps)
    relative_gradient = gradient / scale

    if relative_gradient > gradient_tolerance:
        classification = "increase"
    elif relative_gradient < -gradient_tolerance:
        classification = "decrease"
    else:
        classification = "flat"

    current.update({
        "cost_gradient_per_H2": gradient,
        "relative_cost_gradient_per_H2": relative_gradient,
        "gradient_classification": classification,
        "gradient_stopping_eligible": (
            _gap_qualifies_for_stopping(previous, maximum_mip_gap)
            and _gap_qualifies_for_stopping(current, maximum_mip_gap)
        ),
    })


def _gradient_stop_reason(
    rows: list[dict],
    *,
    minimum_cases: int,
    flat_gradients_required: int,
) -> str | None:
    """Return the reason for stopping after the latest evaluated horizon."""
    if len(rows) < minimum_cases:
        return None
    latest = rows[-1]
    if not latest.get("gradient_stopping_eligible", False):
        return None
    if latest.get("gradient_classification") == "increase":
        return "cost_increase"
    if latest.get("gradient_classification") != "flat":
        return None

    eligible_flat = 0
    for row in reversed(rows[1:]):
        if (
            row.get("gradient_stopping_eligible", False)
            and row.get("gradient_classification") == "flat"
        ):
            eligible_flat += 1
        else:
            break
    if eligible_flat >= flat_gradients_required:
        return "flat_gradient"
    return None


def _assemble_sweep_report(
    *,
    source: Path,
    data: dict,
    H1: int,
    candidates: list[int],
    rows: list[dict],
    stop_on_gradient: bool,
    gradient_tolerance: float,
    flat_gradients_required: int,
    minimum_cases: int,
    maximum_mip_gap_for_stopping: float,
    stopping_reason: str | None,
    complete: bool,
) -> dict:
    """Create the self-describing final report or an incremental checkpoint."""
    best_proven, best_feasible = _select_horizon_cases(rows)
    reason = stopping_reason
    if reason is None:
        reason = "candidate_range_exhausted" if complete else "in_progress"
    return {
        "input": str(source),
        "complete": bool(complete),
        "last_completed_H2": None if not rows else rows[-1]["H2"],
        "objective": "minimize J_op / H2 over the operating phase",
        "varied_parameter": "H2",
        "fixed_dimensions": {
            "F": int(data["F"]),
            "M": int(data["M"]),
            "L": int(data["L"]),
            "H1": H1,
        },
        "H1": H1,
        "planned_operating_horizons": candidates,
        "evaluated_operating_horizons": [row["H2"] for row in rows],
        "stopping_rule": {
            "enabled": bool(stop_on_gradient),
            "interpretation": (
                "Stop after a gap-qualified positive relative cost gradient, "
                "or after the requested number of consecutive gap-qualified "
                "gradients whose magnitude is within the tolerance."
            ),
            "relative_gradient_tolerance_per_H2": gradient_tolerance,
            "flat_gradients_required": flat_gradients_required,
            "minimum_cases": minimum_cases,
            "maximum_mip_gap_for_stopping": maximum_mip_gap_for_stopping,
            "reason": reason,
            "stopped_early": bool(complete and len(rows) < len(candidates)),
        },
        "cases": rows,
        "formulation_growth": _formulation_growth(rows),
        "selection_interpretation": {
            "best_proven": (
                "Minimum J_op/H2 among cases with status='optimal' and a "
                "reported relative MIP gap no larger than 1e-8."
            ),
            "best_feasible": (
                "Minimum feasible J_op/H2 found, including cases stopped "
                "by a time or solution limit; it is not necessarily optimal."
            ),
            "legacy_fields": (
                "best_H2 and best_J_op_average are aliases for the proven "
                "selection, retained for compatibility."
            ),
        },
        "best_proven_H2": (
            None if best_proven is None else best_proven["H2"]
        ),
        "best_proven_J_op_average": (
            None if best_proven is None else best_proven["J_op_average"]
        ),
        "best_feasible_H2": (
            None if best_feasible is None else best_feasible["H2"]
        ),
        "best_feasible_J_op_average": (
            None if best_feasible is None else best_feasible["J_op_average"]
        ),
        "best_feasible_status": (
            None if best_feasible is None else best_feasible["status"]
        ),
        # Backward-compatible aliases now refer only to a proven optimum.
        "best_H2": None if best_proven is None else best_proven["H2"],
        "best_J_op_average": (
            None if best_proven is None else best_proven["J_op_average"]
        ),
    }


def _write_sweep_report(report: dict, target: Path) -> None:
    target.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")


def sweep_operating_horizons(
    input_path: str | Path,
    operating_horizons: Iterable[int],
    *,
    output_path: str | Path | None = None,
    stop_on_gradient: bool = False,
    gradient_tolerance: float = 1e-3,
    flat_gradients_required: int = 2,
    minimum_cases: int = 3,
    maximum_mip_gap_for_stopping: float = 0.05,
) -> dict:
    """Solve one scenario for each requested ``H2`` and select the best.

    This intentionally performs no interpolation of time-indexed input data.
    Scalars and time-constant mission profiles broadcast naturally; an explicit
    profile whose length does not match a candidate horizon is rejected by the
    normal configuration validation.  With ``stop_on_gradient=True``, the
    candidates are evaluated in increasing order until the normalized operating
    cost gradient is flat for the requested number of consecutive comparisons
    or becomes positive.  Only optimal or sufficiently small-gap adjacent cases
    are allowed to trigger this early stopping rule.
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

    candidates = sorted(int(value) for value in operating_horizons)
    if not candidates or any(value <= 0 for value in candidates):
        raise ValueError("operating_horizons must contain positive integers")
    if len(set(candidates)) != len(candidates):
        raise ValueError("operating_horizons must not contain duplicates")
    if not np.isfinite(gradient_tolerance) or gradient_tolerance < 0.0:
        raise ValueError("gradient_tolerance must be finite and non-negative")
    if flat_gradients_required <= 0:
        raise ValueError("flat_gradients_required must be positive")
    if minimum_cases < 2:
        raise ValueError("minimum_cases must be at least two")
    if (
        not np.isfinite(maximum_mip_gap_for_stopping)
        or maximum_mip_gap_for_stopping < 0.0
    ):
        raise ValueError(
            "maximum_mip_gap_for_stopping must be finite and non-negative"
        )

    target: Path | None = None
    if output_path is not None:
        target = Path(output_path)
        if target.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError("output_path must end in .yaml or .yml")
        target.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    stopping_reason: str | None = None
    with TemporaryDirectory(prefix="fleet-horizon-sweep-") as directory:
        temporary = Path(directory)
        for H2 in candidates:
            case = deepcopy(data)
            case["H"] = [H1, H2]
            case["objective_mode"] = "operating_average"
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
            objective_bound = result.get("bound")
            mip_gap = result.get("mip_gap")
            objective = result.get("objective")
            absolute_gap = (
                None
                if objective is None or objective_bound is None
                else max(0.0, float(objective) - float(objective_bound))
            )
            row = {
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
                "J_op": result.get("J_op"),
                "J_op_average": result.get("J_op_average"),
                "objective_bound": objective_bound,
                "absolute_mip_gap": absolute_gap,
                "mip_gap": mip_gap,
                "requested_mip_gap": performance.get("requested_mip_gap"),
                "time_limit_seconds": performance.get(
                    "requested_time_limit_seconds"
                ),
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
            }
            rows.append(row)
            _annotate_latest_gradient(
                rows,
                gradient_tolerance=gradient_tolerance,
                maximum_mip_gap=maximum_mip_gap_for_stopping,
            )
            if stop_on_gradient:
                stopping_reason = _gradient_stop_reason(
                    rows,
                    minimum_cases=minimum_cases,
                    flat_gradients_required=flat_gradients_required,
                )
            if target is not None:
                checkpoint = _assemble_sweep_report(
                    source=source, data=data, H1=H1,
                    candidates=candidates, rows=rows,
                    stop_on_gradient=stop_on_gradient,
                    gradient_tolerance=gradient_tolerance,
                    flat_gradients_required=flat_gradients_required,
                    minimum_cases=minimum_cases,
                    maximum_mip_gap_for_stopping=(
                        maximum_mip_gap_for_stopping
                    ),
                    stopping_reason=stopping_reason,
                    complete=stopping_reason is not None,
                )
                _write_sweep_report(checkpoint, target)
            if stopping_reason is not None:
                break

    report = _assemble_sweep_report(
        source=source, data=data, H1=H1,
        candidates=candidates, rows=rows,
        stop_on_gradient=stop_on_gradient,
        gradient_tolerance=gradient_tolerance,
        flat_gradients_required=flat_gradients_required,
        minimum_cases=minimum_cases,
        maximum_mip_gap_for_stopping=maximum_mip_gap_for_stopping,
        stopping_reason=stopping_reason,
        complete=True,
    )
    if target is not None:
        _write_sweep_report(report, target)
    return report
