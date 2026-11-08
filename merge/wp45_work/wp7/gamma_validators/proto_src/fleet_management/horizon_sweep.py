"""
Horizon sweeps (spec, "Horizon Sweeps"): ``sweep_operating_horizons`` and
``sweep_horizon_grid``.

A sweep solves one input for several pairs ``(H1, H2)`` and ranks the cases.
Each case is solved with ``solver.solve_mapping``, without a file; ``base`` and
Gurobi load only inside that function, so this module imports without Gurobi.

* ``sweep_operating_horizons`` varies ``H2`` for one ``H1``. It annotates the
  cost gradient of every row, applies the early-stop rule, and rewrites its
  report after every case (a checkpoint).
* ``sweep_horizon_grid`` runs one operating sweep for each ``H1`` and ranks all
  cases.
* An error inside a case is logged and recorded in the row of the case; the
  sweep goes on.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

from fleet_management.config import OBJECTIVE_MODES, _infer_L, _integer, read_input
from fleet_management.solver import WARM_START_MODES, _to_plain, solve_mapping


logger = logging.getLogger(__name__)


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

TIMING_KEYS = (
    "model_construction_seconds",
    "optimizer_call_seconds",
    "solution_extraction_seconds",
    "backend_wall_seconds",
    "branch_and_bound_nodes",
    "simplex_iterations",
    "work_units",
)

COST_KEYS = (
    "J_initialization",
    "J_op",
    "J_op_average",
    "peak_damage",
    "damage_penalty",
    "operating_objective",
    "J_total",
    "projected_evaluation_cost",
    "evaluation_objective",
)

GRADIENT_KEYS = (
    "cost_gradient_per_H2",
    "relative_cost_gradient_per_H2",
    "gradient_classification",
    "gradient_stopping_eligible",
)

# The keys of a case row, in order (spec, "Keys of a case row").
ROW_KEYS = (
    ("H1", "H2", "T", "dimensions", "status", "objective_mode",
     "evaluation_horizon", "objective")
    + COST_KEYS
    + ("warm_start", "objective_bound", "mip_gap", "absolute_mip_gap",
       "requested_mip_gap", "requested_time_limit_seconds",
       "optimizer_seconds", "solutions_found", "formulation", "timing")
    + GRADIENT_KEYS
)

# A case without a solution: its status is one of these (spec, "Feasible case").
NO_SOLUTION_STATUSES = ("infeasible", "unbounded", "inf_or_unbounded")

SWEEP_OBJECTIVES = {
    "operating_average": "minimize J_op/H2 + C_D*u; rank by J_op/H2",
    "evaluation_total": (
        "minimize J_initialization + (evaluation_horizon - H1)*J_op/H2 + C_D*u; "
        "rank by J_op/H2"
    ),
    "total": "minimize J_initialization + J_op + C_D*u; rank by J_op/H2",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _is_integer(value) -> bool:
    """An integer, not a bool."""
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _is_nonnegative_number(value) -> bool:
    """A finite number >= 0, not a bool."""
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and bool(np.isfinite(value))
        and value >= 0
    )


def _positive_integers(values: Iterable, name: str) -> list[int]:
    """The values as a list of positive integers; at least one is required."""
    values = list(values)
    if not values:
        raise ValueError(f"{name} must contain at least one value.")
    for value in values:
        if not _is_integer(value) or value <= 0:
            raise ValueError(f"{name} must contain positive integers (got {value!r}).")
    return [int(value) for value in values]


def _horizon_H1(data: dict) -> int:
    """H1 of the input: the first entry of ``H``, or ``H`` itself (spec, ``input_path``)."""
    H = data.get("H")
    if isinstance(H, (list, tuple)):
        if len(H) != 2:
            raise ValueError("'H' must be an integer or a pair [H1, H2].")
        return _integer(H[0], "H1")
    return _integer(H, "H")


def _input_sizes(data: dict) -> dict:
    """``F``, ``M`` and ``L`` of the input; ``None`` for all three if they cannot be read.

    ``L`` is optional in the input, so it is inferred as the input reader does. An input
    whose sizes cannot be read fails in every case, with the message of the checks.
    """
    try:
        F = _integer(data["F"], "F")
        M = _integer(data["M"], "M")
        L = _infer_L(data, F)
    except (KeyError, TypeError, ValueError):
        return {"F": None, "M": None, "L": None}
    return {"F": F, "M": M, "L": L}



# ---------------------------------------------------------------------------
# Rows of a case
# ---------------------------------------------------------------------------
def _case_row(result: dict) -> dict:
    """The row of a case, from its result (spec, "Keys of a case row").

    The row keeps no schedule. NumPy values become plain data; NaN and infinity
    become ``None``. The four gradient keys are filled by ``_annotate_gradients``.
    """
    performance = result.get("performance") or {}
    H1, H2 = int(result["H1"]), int(result["H2"])
    objective, bound = result.get("objective"), result.get("bound")
    if objective is None or bound is None:
        absolute_gap = None
    elif np.isfinite(objective) and np.isfinite(bound):
        absolute_gap = max(0.0, float(objective) - float(bound))
    else:
        absolute_gap = None

    row = {
        "H1": H1,
        "H2": H2,
        "T": H1 + H2,
        "dimensions": {
            key: int(result[key]) for key in ("F", "M", "L", "H1", "H2", "T")
        },
        "status": result.get("status"),
        "objective_mode": result.get("objective_mode"),
        "evaluation_horizon": result.get("evaluation_horizon"),
        "objective": objective,
    }
    row.update({key: result.get(key) for key in COST_KEYS})
    row.update({
        "warm_start": result.get("warm_start"),
        "objective_bound": bound,
        "mip_gap": result.get("mip_gap"),
        "absolute_mip_gap": absolute_gap,
        "requested_mip_gap": performance.get("requested_mip_gap"),
        "requested_time_limit_seconds": performance.get(
            "requested_time_limit_seconds"
        ),
        "optimizer_seconds": performance.get("optimizer_call_seconds"),
        "solutions_found": performance.get("solutions_found"),
        "formulation": {key: performance.get(key) for key in FORMULATION_KEYS},
        "timing": {key: performance.get(key) for key in TIMING_KEYS},
    })
    row.update({key: None for key in GRADIENT_KEYS})
    return _to_plain(row)


def _error_row(H1: int, H2: int, sizes: dict, error: Exception) -> dict:
    """The row of a case that raised ``error``: ``None`` everywhere except the sizes."""
    row = {key: None for key in ROW_KEYS}
    row.update({
        "H1": H1,
        "H2": H2,
        "T": H1 + H2,
        "dimensions": {**sizes, "H1": H1, "H2": H2, "T": H1 + H2},
        "status": f"error: {error}",
    })
    return row


def _source_of(result: dict) -> dict | None:
    """The warm-start source that a result gives to the next case; ``None`` without a schedule."""
    if result.get("x") is None:
        return None
    return {key: result.get(key) for key in ("H1", "H2", "x", "m", "r")}


def _solve_case(
    case: dict,
    warm_start: dict | None,
    warm_start_mode: str,
    keep_source: bool,
) -> tuple[dict, dict | None]:
    """Solve one case input and return its row and, if asked, its warm-start source.

    A module-level function, so that a worker process can run it. It catches nothing:
    the caller records the error in the row of the case.
    """
    result = solve_mapping(
        case, None, warm_start=warm_start, warm_start_mode=warm_start_mode
    )
    return _case_row(result), _source_of(result) if keep_source else None



# ---------------------------------------------------------------------------
# Feasibility, ranking, gradients and the stop rule
# ---------------------------------------------------------------------------
def _is_feasible(row: dict) -> bool:
    """Feasible case (spec, "Cost Measures and Ranking"): a solution and a status that allows one."""
    return (
        row.get("objective") is not None
        and row.get("status") not in NO_SOLUTION_STATUSES
    )


def _is_proven(row: dict) -> bool:
    """Proven case: feasible and ``"optimal"`` at the MIP gap used for the case."""
    return _is_feasible(row) and row.get("status") == "optimal"


def _select_horizon_cases(rows: list[dict]) -> tuple[dict | None, dict | None]:
    """Return the best proven and the best feasible case of an operating sweep.

    The ranking value is ``J_op_average``; a tie goes to the smaller ``H2``.
    """
    def key(row: dict) -> tuple:
        return row["J_op_average"], row["H2"]

    feasible = [row for row in rows if _is_feasible(row)]
    proven = [row for row in feasible if _is_proven(row)]
    return (
        min(proven, key=key) if proven else None,
        min(feasible, key=key) if feasible else None,
    )


def _select_grid_cases(
    rows: list[dict], metric: str
) -> tuple[dict | None, dict | None]:
    """Return the best proven and the best feasible case of a grid.

    The ranking value is ``metric``; a tie goes to the smaller ``H1``, then to the
    smaller ``H2``.
    """
    def key(row: dict) -> tuple:
        return row[metric], row["H1"], row["H2"]

    feasible = [row for row in rows if _is_feasible(row)]
    proven = [row for row in feasible if _is_proven(row)]
    return (
        min(proven, key=key) if proven else None,
        min(feasible, key=key) if feasible else None,
    )


def _qualifies(row: dict, maximum_mip_gap_for_stopping: float | None) -> bool:
    """Whether a case reports a finite MIP gap at most its threshold (spec, "Definitions").

    The threshold is ``maximum_mip_gap_for_stopping`` if it is a number, otherwise the
    MIP gap used for the case (``requested_mip_gap`` of its row).
    """
    threshold = (
        row.get("requested_mip_gap")
        if maximum_mip_gap_for_stopping is None
        else maximum_mip_gap_for_stopping
    )
    gap = row.get("mip_gap")
    return (
        gap is not None
        and threshold is not None
        and bool(np.isfinite(gap))
        and float(gap) <= float(threshold)
    )


def _annotate_gradients(
    rows: list[dict],
    *,
    gradient_tolerance: float,
    maximum_mip_gap_for_stopping: float | None,
) -> None:
    """Set the gradient keys of every row; ``rows`` is ordered by ``H2``.

    The gradient of row ``n`` is the change of ``J_op_average`` from row ``n - 1`` per
    added unit of ``H2``. Its relative value is Eq. d_gradient. It is available only if
    both rows are feasible. The class is ``"increase"``, ``"decrease"`` or ``"flat"``
    with the tolerance ``gradient_tolerance``. The gradient is eligible to stop if both
    rows qualify. Every call recomputes all rows, so the annotations stay right when the
    rows arrive out of order.
    """
    for n, row in enumerate(rows):
        row.update({
            "cost_gradient_per_H2": None,
            "relative_cost_gradient_per_H2": None,
            "gradient_classification": "unavailable",
            "gradient_stopping_eligible": False,
        })
        if n == 0:
            continue
        previous = rows[n - 1]
        if not (_is_feasible(previous) and _is_feasible(row)):
            continue

        previous_cost = float(previous["J_op_average"])
        gradient = (float(row["J_op_average"]) - previous_cost) / (
            int(row["H2"]) - int(previous["H2"])
        )
        # Eq. d_gradient
        relative = gradient / max(abs(previous_cost), float(np.finfo(float).eps))
        if relative > gradient_tolerance:
            classification = "increase"
        elif relative < -gradient_tolerance:
            classification = "decrease"
        else:
            classification = "flat"
        row.update({
            "cost_gradient_per_H2": gradient,
            "relative_cost_gradient_per_H2": relative,
            "gradient_classification": classification,
            "gradient_stopping_eligible": (
                _qualifies(previous, maximum_mip_gap_for_stopping)
                and _qualifies(row, maximum_mip_gap_for_stopping)
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



# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def _formulation_growth(rows: list[dict]) -> dict | None:
    """Growth of the model from the first to the last row with counts (spec, ``formulation_growth``).

    ``rows`` is ordered by ``H2``. A row without counts (a case error) is skipped.
    The result is ``None`` with fewer than two rows with counts.
    """
    counted = [row for row in rows if row["formulation"] is not None]
    if len(counted) < 2:
        return None
    first, last = counted[0], counted[-1]
    added_time_steps = last["T"] - first["T"]

    growth = {
        "from_H2": first["H2"],
        "to_H2": last["H2"],
        "added_time_steps": added_time_steps,
    }
    for key in FORMULATION_KEYS:
        start, stop = first["formulation"][key], last["formulation"][key]
        if start is None or stop is None:
            growth[f"added_{key}"] = None
            growth[f"{key}_per_added_time_step"] = None
        else:
            growth[f"added_{key}"] = stop - start
            growth[f"{key}_per_added_time_step"] = (stop - start) / added_time_steps
    return growth


def _assemble_sweep_report(
    *,
    source: Path,
    sizes: dict,
    H1: int,
    settings: dict,
    rows: list[dict],
    last_completed_H2: int | None,
    stopping_reason: str | None,
    complete: bool,
) -> dict:
    """The report of an operating sweep, or a checkpoint (spec, "Report keys").

    A checkpoint has ``complete = False`` and the reason ``"in_progress"``, also after
    the case that stops the sweep.
    """
    candidates = settings["candidates"]
    best_proven, best_feasible = _select_horizon_cases(rows)
    reason = (
        (stopping_reason or "candidate_range_exhausted") if complete else "in_progress"
    )
    return {
        "input": str(source),
        "complete": bool(complete),
        "last_completed_H2": last_completed_H2,
        "objective": SWEEP_OBJECTIVES[settings["objective_mode"]],
        "varied_parameter": "H2",
        "fixed_dimensions": dict(sizes),
        "H1": H1,
        "planned_operating_horizons": list(candidates),
        "evaluated_operating_horizons": [row["H2"] for row in rows],
        "warm_start": {
            "enabled": bool(settings["warm_start"]),
            "mode": settings["warm_start_mode"] if settings["warm_start"] else None,
        },
        "n_workers": settings["n_workers"],
        "stopping_rule": {
            "enabled": bool(settings["stop_on_gradient"]),
            "relative_gradient_tolerance_per_H2": settings["gradient_tolerance"],
            "flat_gradients_required": settings["flat_gradients_required"],
            "minimum_cases": settings["minimum_cases"],
            "maximum_mip_gap_for_stopping": settings["maximum_mip_gap_for_stopping"],
            "reason": reason,
            "stopped_early": bool(complete and len(rows) < len(candidates)),
        },
        "cases": rows,
        "formulation_growth": _formulation_growth(rows),
        "best_proven_H2": None if best_proven is None else best_proven["H2"],
        "best_feasible_H2": None if best_feasible is None else best_feasible["H2"],
        "best_proven_J_op_average": (
            None if best_proven is None else best_proven["J_op_average"]
        ),
        "best_feasible_J_op_average": (
            None if best_feasible is None else best_feasible["J_op_average"]
        ),
        "best_feasible_status": (
            None if best_feasible is None else best_feasible["status"]
        ),
    }


def _best_case(row: dict | None) -> dict | None:
    """The entry ``best_proven`` or ``best_feasible`` of a grid report (spec, "Report keys")."""
    if row is None:
        return None
    best = {
        key: row[key] for key in ("H1", "H2", "T", "status", "mip_gap", "objective")
    }
    best.update({
        key: row[key] for key in COST_KEYS if key not in ("J_op", "J_total")
    })
    return best


def _assemble_grid_report(
    *,
    source: Path,
    sizes: dict,
    h1_values: list[int],
    h2_values: list[int],
    subreports: list[dict],
    warm_start: bool,
    evaluation_horizon: int | None,
    complete: bool,
) -> dict:
    """The report of a grid, or a checkpoint (spec, ``sweep_horizon_grid``, "Report keys").

    ``subreports`` are the reports of the operating sweeps, by ``H1``. ``cases`` holds
    copies of all their rows, by ``H1``, then ``H2``, so that no row appears twice in
    the written file.
    """
    cases = [deepcopy(row) for report in subreports for row in report["cases"]]
    metric = (
        "projected_evaluation_cost"
        if evaluation_horizon is not None
        else "J_op_average"
    )
    best_proven, best_feasible = _select_grid_cases(cases, metric)
    objective = (
        "minimize J_initialization + (evaluation_horizon - H1)*J_op/H2 + C_D*u; "
        "rank by projected_evaluation_cost"
        if evaluation_horizon is not None
        else SWEEP_OBJECTIVES["operating_average"]
    )
    return {
        "input": str(source),
        "complete": bool(complete),
        "objective": objective,
        "comparison_metric": metric,
        "mip_certificate_metric": "objective",
        "evaluation_horizon": evaluation_horizon,
        "varied_parameters": ["H1", "H2"],
        "fixed_dimensions": dict(sizes),
        "planned_transitory_horizons": list(h1_values),
        "planned_operating_horizons": list(h2_values),
        "warm_start_enabled": bool(warm_start),
        "H1_sweeps": subreports,
        "cases": cases,
        "best_proven": _best_case(best_proven),
        "best_feasible": _best_case(best_feasible),
    }


def _write_sweep_report(report: dict, target: Path) -> None:
    target.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")



# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
def _check_output_path(output_path: str | Path | None) -> Path | None:
    """Check the report file (spec, ``output_path``) and create its parent folders."""
    if output_path is None:
        return None
    target = Path(output_path)
    if target.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError("output_path must end in .yaml or .yml.")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _check_sweep_arguments(
    H1: int,
    operating_horizons: Iterable[int],
    *,
    output_path: str | Path | None,
    stop_on_gradient: bool,
    gradient_tolerance: float,
    flat_gradients_required: int,
    minimum_cases: int,
    maximum_mip_gap_for_stopping: float | None,
    warm_start: bool,
    warm_start_mode: str,
    n_workers: int,
    objective_mode: str,
    evaluation_horizon: int | None,
) -> dict:
    """Check the arguments of an operating sweep (spec, "Arguments") and return its settings.

    Every error is a ``ValueError``, raised before the first case. The parent folders of
    ``output_path`` are created only when all other checks pass.
    """
    candidates = sorted(_positive_integers(operating_horizons, "operating_horizons"))
    if len(set(candidates)) != len(candidates):
        raise ValueError("operating_horizons must not contain duplicates.")

    if not _is_nonnegative_number(gradient_tolerance):
        raise ValueError("gradient_tolerance must be a finite number >= 0.")
    if not _is_integer(flat_gradients_required) or flat_gradients_required < 1:
        raise ValueError("flat_gradients_required must be an integer >= 1.")
    if not _is_integer(minimum_cases) or minimum_cases < 2:
        raise ValueError("minimum_cases must be an integer >= 2.")
    if maximum_mip_gap_for_stopping is not None and not _is_nonnegative_number(
        maximum_mip_gap_for_stopping
    ):
        raise ValueError(
            "maximum_mip_gap_for_stopping must be None or a finite number >= 0."
        )

    if warm_start_mode not in WARM_START_MODES:
        raise ValueError(
            f"warm_start_mode must be one of {WARM_START_MODES} "
            f"(got {warm_start_mode!r})."
        )
    if not _is_integer(n_workers) or n_workers < 1:
        raise ValueError("n_workers must be an integer >= 1.")
    if n_workers > 1 and warm_start:
        raise ValueError(
            "warm_start=True needs n_workers=1: a warm start uses the preceding case."
        )
    if n_workers > 1 and stop_on_gradient:
        raise ValueError(
            "stop_on_gradient=True needs n_workers=1: the early stop needs the "
            "cases in order."
        )

    if objective_mode not in OBJECTIVE_MODES:
        raise ValueError(
            f"objective_mode must be one of {OBJECTIVE_MODES} "
            f"(got {objective_mode!r})."
        )
    if evaluation_horizon is not None and not (
        _is_integer(evaluation_horizon) and evaluation_horizon > H1
    ):
        raise ValueError(
            f"evaluation_horizon must be an integer > H1 = {H1} "
            f"(got {evaluation_horizon!r})."
        )
    if objective_mode == "evaluation_total" and evaluation_horizon is None:
        raise ValueError("objective_mode 'evaluation_total' needs evaluation_horizon.")

    target = _check_output_path(output_path)
    return {
        "candidates": candidates,
        "target": target,
        "stop_on_gradient": bool(stop_on_gradient),
        "gradient_tolerance": float(gradient_tolerance),
        "flat_gradients_required": int(flat_gradients_required),
        "minimum_cases": int(minimum_cases),
        "maximum_mip_gap_for_stopping": (
            None
            if maximum_mip_gap_for_stopping is None
            else float(maximum_mip_gap_for_stopping)
        ),
        "warm_start": bool(warm_start),
        "warm_start_mode": warm_start_mode,
        "n_workers": int(n_workers),
        "objective_mode": objective_mode,
        "evaluation_horizon": (
            None if evaluation_horizon is None else int(evaluation_horizon)
        ),
    }



# ---------------------------------------------------------------------------
# The case loop
# ---------------------------------------------------------------------------
def _sweep(
    data: dict,
    source: Path,
    sizes: dict,
    H1: int,
    settings: dict,
    on_checkpoint=None,
) -> dict:
    """Solve the cases of an operating sweep and return its report (spec, "Steps").

    ``settings`` is the result of ``_check_sweep_arguments``. The case input is the input
    data with ``H = [H1, H2]`` and with ``objective_mode`` and ``evaluation_horizon`` (if
    given) replaced. After each case the row is appended, the rows are ordered by ``H2``,
    the gradients are annotated, and a checkpoint is written to ``settings["target"]``
    (if set) and passed to ``on_checkpoint`` (if given). A case that raises an
    ``Exception`` is logged at level ERROR and recorded as an error row; the sweep goes on.

    With one worker the cases run in increasing order of ``H2``; the warm start of a case
    is the schedule of the preceding case that has one, and the stop rule may end the
    sweep. With more workers every case is submitted at once, in its own process, and
    the rows arrive in the order of completion.
    """
    candidates = settings["candidates"]
    target = settings["target"]
    rows: list[dict] = []
    last_completed_H2: int | None = None
    stopping_reason: str | None = None

    def case_input(H2: int) -> dict:
        case = deepcopy(data)
        case["H"] = [H1, H2]
        case["objective_mode"] = settings["objective_mode"]
        if settings["evaluation_horizon"] is not None:
            case["evaluation_horizon"] = settings["evaluation_horizon"]
        return case

    def failed(H2: int, error: Exception) -> dict:
        logger.error("horizon sweep case H1=%d, H2=%d failed: %s", H1, H2, error)
        return _error_row(H1, H2, sizes, error)

    def report(complete: bool) -> dict:
        return _assemble_sweep_report(
            source=source, sizes=sizes, H1=H1, settings=settings, rows=rows,
            last_completed_H2=last_completed_H2, stopping_reason=stopping_reason,
            complete=complete,
        )

    def record(row: dict, H2: int) -> None:
        nonlocal last_completed_H2
        rows.append(row)
        rows.sort(key=lambda item: item["H2"])
        last_completed_H2 = H2
        _annotate_gradients(
            rows,
            gradient_tolerance=settings["gradient_tolerance"],
            maximum_mip_gap_for_stopping=settings["maximum_mip_gap_for_stopping"],
        )

    def checkpoint() -> None:
        partial = report(False)
        if target is not None:
            _write_sweep_report(partial, target)
        if on_checkpoint is not None:
            on_checkpoint(partial)

    if settings["n_workers"] == 1:
        previous: dict | None = None
        for H2 in candidates:
            warm = previous if settings["warm_start"] else None
            try:
                row, found = _solve_case(
                    case_input(H2), warm, settings["warm_start_mode"],
                    settings["warm_start"],
                )
            except Exception as error:
                row, found = failed(H2, error), None
            if found is not None:
                previous = found
            record(row, H2)
            if settings["stop_on_gradient"]:
                stopping_reason = _gradient_stop_reason(
                    rows,
                    minimum_cases=settings["minimum_cases"],
                    flat_gradients_required=settings["flat_gradients_required"],
                )
            checkpoint()
            if stopping_reason is not None:
                break
    else:
        with ProcessPoolExecutor(max_workers=settings["n_workers"]) as pool:
            futures = {
                pool.submit(
                    _solve_case, case_input(H2), None, settings["warm_start_mode"],
                    False,
                ): H2
                for H2 in candidates
            }
            for future in as_completed(futures):
                H2 = futures[future]
                try:
                    row = future.result()[0]
                except Exception as error:
                    row = failed(H2, error)
                record(row, H2)
                checkpoint()

    final = report(True)
    if target is not None:
        _write_sweep_report(final, target)
    return final


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------
def sweep_operating_horizons(
    input_path: str | Path,
    operating_horizons: Iterable[int],
    *,
    output_path: str | Path | None = None,
    stop_on_gradient: bool = False,
    gradient_tolerance: float = 1e-3,
    flat_gradients_required: int = 2,
    minimum_cases: int = 3,
    maximum_mip_gap_for_stopping: float | None = None,
    warm_start: bool = False,
    warm_start_mode: str = "start",
    n_workers: int = 1,
    objective_mode: str = "operating_average",
    evaluation_horizon: int | None = None,
) -> dict:
    """Solve one input for several values of ``H2`` with ``H1`` fixed, and rank the cases.

    The steps (spec, ``sweep_operating_horizons``):

    1. Read the input file with the input reader; the data must be a mapping.
    2. Check every argument; an invalid argument is a ``ValueError`` before the first case.
    3. For each ``H2``, the case input is the input data with ``H = [H1, H2]`` and with
       ``objective_mode`` and ``evaluation_horizon`` (if given) replaced. Time-indexed
       data are not interpolated.
    4. Solve the case input with ``solve_mapping`` and ``results_path=None``; keep the
       row of the case, not its schedule.
    5. After each case: append its row, annotate the gradient, test the stop rule, and
       write a checkpoint if ``output_path`` is given.
    6. At the end, write the final report and return it.
    """
    source = Path(input_path)
    data = read_input(source)
    H1 = _horizon_H1(data)
    settings = _check_sweep_arguments(
        H1, operating_horizons,
        output_path=output_path,
        stop_on_gradient=stop_on_gradient,
        gradient_tolerance=gradient_tolerance,
        flat_gradients_required=flat_gradients_required,
        minimum_cases=minimum_cases,
        maximum_mip_gap_for_stopping=maximum_mip_gap_for_stopping,
        warm_start=warm_start,
        warm_start_mode=warm_start_mode,
        n_workers=n_workers,
        objective_mode=objective_mode,
        evaluation_horizon=evaluation_horizon,
    )
    return _sweep(data, source, _input_sizes(data), H1, settings)


def sweep_horizon_grid(
    input_path: str | Path,
    transitory_horizons: Iterable[int],
    operating_horizons: Iterable[int],
    *,
    output_path: str | Path | None = None,
    warm_start: bool = True,
    evaluation_horizon: int | None = None,
    **sweep_options,
) -> dict:
    """Vary ``H1`` and ``H2``: run an operating sweep for each ``H1`` and rank all cases.

    ``sweep_options`` are the keyword arguments of ``sweep_operating_horizons`` except
    ``objective_mode``. The objective mode of every sweep is ``"evaluation_total"`` if an
    evaluation horizon is given, otherwise ``"operating_average"``. Warm starts, the
    early stop and the capture of case errors act within each ``H1``. All arguments are
    checked before the first case. The grid report is rewritten after every case and at
    the end (spec, ``sweep_horizon_grid``).
    """
    source = Path(input_path)
    data = read_input(source)
    h1_values = sorted(set(_positive_integers(transitory_horizons, "transitory_horizons")))
    h2_values = sorted(set(_positive_integers(operating_horizons, "operating_horizons")))
    if "objective_mode" in sweep_options:
        raise ValueError(
            "sweep_horizon_grid sets objective_mode itself; remove it from sweep_options."
        )
    if evaluation_horizon is not None and not (
        _is_integer(evaluation_horizon) and evaluation_horizon > max(h1_values)
    ):
        raise ValueError(
            "evaluation_horizon must be an integer that exceeds every H1 candidate."
        )

    objective_mode = (
        "evaluation_total" if evaluation_horizon is not None else "operating_average"
    )
    settings = {
        h1: _check_sweep_arguments(
            h1, h2_values,
            output_path=None,
            warm_start=warm_start,
            objective_mode=objective_mode,
            evaluation_horizon=evaluation_horizon,
            **sweep_options,
        )
        for h1 in h1_values
    }
    target = _check_output_path(output_path)

    sizes = _input_sizes(data)
    done: list[dict] = []

    def write(subreports: list[dict], complete: bool) -> dict:
        report = _assemble_grid_report(
            source=source, sizes=sizes, h1_values=h1_values, h2_values=h2_values,
            subreports=subreports, warm_start=warm_start,
            evaluation_horizon=evaluation_horizon, complete=complete,
        )
        if target is not None:
            _write_sweep_report(report, target)
        return report

    for h1 in h1_values:
        subreport = _sweep(
            data, source, sizes, h1, settings[h1],
            on_checkpoint=lambda partial: write(done + [partial], False),
        )
        done.append(subreport)
    return write(done, True)
