"""Create compact CSV/YAML summaries for either horizon experiment."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import yaml


FIELDS = (
    "H1", "H2", "T", "status", "J_initialization", "J_op_average",
    "projected_evaluation_cost", "objective_bound", "mip_gap",
    "optimizer_seconds", "solutions_found", "variables",
    "continuous_variables", "binary_variables", "linear_constraints",
    "branch_and_bound_nodes",
)


def flatten(case: dict) -> dict:
    formulation = case.get("formulation") or {}
    timing = case.get("timing") or {}
    return {
        "H1": case.get("H1"), "H2": case.get("H2"), "T": case.get("T"),
        "status": case.get("status"),
        "J_initialization": case.get("J_initialization"),
        "J_op_average": case.get("J_op_average"),
        "projected_evaluation_cost": case.get("projected_evaluation_cost"),
        "objective_bound": case.get("objective_bound"),
        "mip_gap": case.get("mip_gap"),
        "optimizer_seconds": case.get("optimizer_seconds"),
        "solutions_found": case.get("solutions_found"),
        "variables": formulation.get("variables"),
        "continuous_variables": formulation.get("continuous_variables"),
        "binary_variables": formulation.get("binary_variables"),
        "linear_constraints": formulation.get("linear_constraints"),
        "branch_and_bound_nodes": timing.get("branch_and_bound_nodes"),
    }


def outcome(case: dict) -> str:
    if str(case.get("status", "")).lower() == "infeasible":
        return "proven_infeasible"
    if case.get("objective") is None:
        return "no_incumbent"
    return "incumbent_found"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    report = yaml.safe_load(args.report.read_text(encoding="utf-8"))
    cases = report.get("cases", [])
    rows = [flatten(case) for case in cases]
    args.output_directory.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_directory / "cases.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "complete": report.get("complete"),
        "objective": report.get("objective"),
        "evaluation_horizon": report.get("evaluation_horizon"),
        "planned_transitory_horizons": report.get("planned_transitory_horizons"),
        "planned_operating_horizons": report.get("planned_operating_horizons"),
        "evaluated_cases": len(rows),
        "outcomes": dict(Counter(outcome(case) for case in cases)),
        "best_proven": report.get("best_proven"),
        "best_observed_feasible": report.get("best_feasible"),
    }
    summary_path = args.output_directory / "summary.yaml"
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(summary, sort_keys=False))
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
