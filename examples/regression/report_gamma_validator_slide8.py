"""Generate Slide 8 evidence for replacement-enabled Gamma validation.

The script runs the forced transition truth table for both ARD-infinity and
ARD1.  Each fixed schedule contains missions, idle periods, one repair, and
four replacements.  It then performs:

1. deterministic schedule/state replay;
2. a fixed-seed Monte Carlo replay of the same schedule; and
3. a deliberate state corruption that the deterministic validator must reject.

Run this file from the Fleet_Management repository root.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from gamma_replacement_truth_table_common import make_config
from fleet_management.degradation_model.base import (
    build_fleet,
    extract_solution,
    get_cell_builder,
    resolve_run_options,
)
from fleet_management.degradation_model.gamma_utils.gamma_replay_validator import (
    validate_gamma_replay_schedule,
)
from fleet_management.degradation_model.gamma_utils.gamma_stochastic_validator import (
    validate_gamma_stochastic_schedule,
)


CASES = ("ardinf", "ard1")
EVENTS = (
    "mission",
    "replacement",
    "explicit idle",
    "mission",
    "repair",
    "replacement",
    "consecutive replacement",
    "mission with paired-vehicle replacement",
)
STATE_ERROR_FIELDS = (
    "physical_mean",
    "bounding_shape",
    "removed_mean",
    "mean_latch",
    "shape_latch",
)


def _plain(value: Any) -> Any:
    """Convert NumPy objects recursively so PyYAML can serialize them."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _count_violations(report: dict[str, Any], field: str) -> int:
    return sum(
        1 for row in report["violations"] if row.get("field") == field
    )


def _maximum_state_error(report: dict[str, Any]) -> float:
    return max(float(report["maximum_errors"].get(name, 0.0))
               for name in STATE_ERROR_FIELDS)


def _force(model: Any, variable: Any, value: int, name: str) -> None:
    model.addConstr(variable == value, name=name)


def _solve_repeatable_schedule(repair_model: str):
    """Build a fixed schedule that exercises interventions and closes the loop."""
    cfg = make_config(repair_model)
    context = build_fleet(
        cfg,
        resolve_run_options(cfg),
        model_name=f"gamma_{repair_model}_validator_slide8",
    )

    # Vehicle 0 performs the mission at k=0, 3 and 7. Vehicle 1 covers all
    # other periods, then receives a replacement at k=7. This final paired
    # mission/replacement makes both vehicle trajectories repeatable while
    # retaining every local event used by the replacement truth table.
    vehicle_0_missions = {0, 3, 7}
    vehicle_0_depot = {1, 2, 4, 5, 6}

    for i in range(cfg.F):
        for j in range(cfg.M + 1):
            for k in range(cfg.T):
                if j == 1:
                    selected = int(
                        (i == 0 and k in vehicle_0_missions)
                        or (i == 1 and k not in vehicle_0_missions)
                    )
                else:
                    selected = int(
                        (i == 0 and k in vehicle_0_depot)
                        or (i == 1 and k == 7)
                    )
                _force(
                    context.model,
                    context.x[i, j, k],
                    selected,
                    f"slide8_force_x_{i}_{j}_{k}",
                )

        for k in range(cfg.T):
            _force(
                context.model,
                context.m_rep[i, 0, k],
                int(i == 0 and k == 4),
                f"slide8_force_m_{i}_{k}",
            )
            _force(
                context.model,
                context.r_rep[i, 0, k],
                int((i == 0 and k in {1, 5, 6}) or (i == 1 and k == 7)),
                f"slide8_force_r_{i}_{k}",
            )

    context.model.optimize()
    if context.model.SolCount == 0:
        raise AssertionError(
            f"forced Gamma {repair_model} Slide 8 schedule is infeasible"
        )

    result = extract_solution(context, cfg, context.model)
    get_cell_builder("gamma").extract(context, cfg, result)
    # The public solve() wrapper normally attaches these identity fields.
    result["backend"] = "modular"
    result["degradation"] = "gamma"
    return cfg, context, result


def _run_case(
    repair_model: str,
    *,
    repetitions: int,
    random_seed: int,
    batch_size: int,
    tolerance: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg, context, result = _solve_repeatable_schedule(repair_model)

    # Replacement is a resolved run option stored on the built model context,
    # not a direct FleetConfig dataclass attribute.
    if not bool(context.allow_replacement):
        raise AssertionError(f"{repair_model}: replacement is not enabled")

    deterministic = validate_gamma_replay_schedule(
        cfg,
        result,
        tolerance=tolerance,
        raise_on_failure=True,
    )
    stochastic = validate_gamma_stochastic_schedule(
        cfg,
        result,
        repetitions=repetitions,
        random_seed=random_seed,
        confidence_level=0.95,
        batch_size=batch_size,
        tolerance=tolerance,
        # Report the empirical rate without turning it into a pass/fail target.
        maximum_schedule_failure_rate=None,
        raise_on_failure=False,
    )

    corrupted = dict(result)
    corrupted["mu"] = np.asarray(corrupted["mu"], dtype=float).copy()
    corrupted["mu"][0, 0, 0] += 0.01
    corruption = validate_gamma_replay_schedule(
        cfg,
        corrupted,
        tolerance=tolerance,
        raise_on_failure=False,
    )
    if corruption["valid"]:
        raise AssertionError(
            f"{repair_model}: validator accepted deliberate state corruption"
        )

    schedule = stochastic["schedule"]
    summary = {
        "repair_model": repair_model,
        "replacement_enabled": True,
        "forced_events": list(EVENTS),
        "gamma_cells": int(deterministic["gamma_cells"]),
        "transitions_replayed": int(deterministic["transitions_checked"]),
        "repairs": int(deterministic["repairs"]),
        "replacements": int(deterministic["replacements"]),
        "deterministic_replay": "PASS",
        "maximum_state_mismatch": _maximum_state_error(deterministic),
        "reliability_violations": _count_violations(
            deterministic, "reliability"
        ),
        "repeatability_violations": _count_violations(
            deterministic, "repeatability"
        ),
        "failed_stochastic_replays": int(schedule["failed_replays"]),
        "total_stochastic_replays": int(stochastic["repetitions"]),
        "stochastic_failure_rate": float(schedule["failure_rate"]),
        "failure_rate_95pct_upper_bound": float(
            schedule["failure_rate_upper_confidence_bound"]
        ),
        "random_seed": int(stochastic["random_seed"]),
        "corruption_detected": True,
        "corrupted_maximum_state_mismatch": _maximum_state_error(corruption),
    }
    full_report = {
        "case": summary,
        "deterministic": deterministic,
        "stochastic": stochastic,
        "corruption_test": corruption,
    }

    # Avoid keeping two Gurobi models alive while running the second case.
    context.model.dispose()
    return _plain(summary), _plain(full_report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/validator_slide8"),
    )
    parser.add_argument("--repetitions", type=int, default=100_000)
    parser.add_argument("--random-seed", type=int, default=20260913)
    parser.add_argument("--batch-size", type=int, default=20_000)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    args = parser.parse_args()

    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []

    for repair_model in CASES:
        summary, full_report = _run_case(
            repair_model,
            repetitions=args.repetitions,
            random_seed=args.random_seed,
            batch_size=args.batch_size,
            tolerance=args.tolerance,
        )
        summaries.append(summary)
        destination = args.output_dir / f"{repair_model}_validation.yaml"
        destination.write_text(
            yaml.safe_dump(full_report, sort_keys=False),
            encoding="utf-8",
        )

    summary_yaml = args.output_dir / "validator_slide8_summary.yaml"
    summary_yaml.write_text(
        yaml.safe_dump(
            {
                "experiment": "replacement-enabled Gamma validator evidence",
                "cases": summaries,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    summary_csv = args.output_dir / "validator_slide8_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    print("PASS replacement-enabled Gamma validator evidence")
    for row in summaries:
        print(
            f"{row['repair_model']:>6}: deterministic PASS, "
            f"repairs/replacements={row['repairs']}/{row['replacements']}, "
            f"maximum mismatch={row['maximum_state_mismatch']:.3e}, "
            f"failed stochastic replays="
            f"{row['failed_stochastic_replays']}/"
            f"{row['total_stochastic_replays']}, "
            f"corruption detected"
        )
    print("Wrote", summary_yaml)
    print("Wrote", summary_csv)


if __name__ == "__main__":
    main()
