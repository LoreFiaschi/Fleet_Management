"""Write searchable CSV and compact YAML summaries of a horizon sweep."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml


FIELDS = (
    "H2", "T", "status", "J_op_average", "objective_bound", "mip_gap",
    "relative_cost_gradient_per_H2", "continuous_variables",
    "integer_variables", "linear_constraints", "general_constraints",
    "indicator_constraints", "quadratic_constraints",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output_directory", type=Path)
    arguments = parser.parse_args()

    report = yaml.safe_load(arguments.report.read_text(encoding="utf-8"))
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in report["cases"]:
        formulation = case["formulation"]
        rows.append({
            "H2": case["H2"],
            "T": case["T"],
            "status": case["status"],
            "J_op_average": case.get("J_op_average"),
            "objective_bound": case.get("objective_bound"),
            "mip_gap": case.get("mip_gap"),
            "relative_cost_gradient_per_H2": case.get(
                "relative_cost_gradient_per_H2"
            ),
            "continuous_variables": formulation["continuous_variables"],
            "integer_variables": formulation["integer_variables"],
            "linear_constraints": formulation["linear_constraints"],
            "general_constraints": formulation.get("general_constraints", 0),
            "indicator_constraints": formulation.get("indicator_constraints", 0),
            "quadratic_constraints": formulation.get("quadratic_constraints", 0),
        })

    csv_path = arguments.output_directory / "cases.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "complete": report.get("complete"),
        "planned_operating_horizons": report.get("planned_operating_horizons"),
        "evaluated_operating_horizons": report.get("evaluated_operating_horizons"),
        "best_proven_H2": report.get("best_proven_H2"),
        "best_proven_J_op_average": report.get("best_proven_J_op_average"),
        "best_feasible_H2": report.get("best_feasible_H2"),
        "best_feasible_J_op_average": report.get("best_feasible_J_op_average"),
        "best_feasible_status": report.get("best_feasible_status"),
        "stopping_rule": report.get("stopping_rule"),
        "formulation_growth": report.get("formulation_growth"),
    }
    summary_path = arguments.output_directory / "summary.yaml"
    summary_path.write_text(
        yaml.safe_dump(summary, sort_keys=False), encoding="utf-8"
    )
    print("CSV    :", csv_path)
    print("summary:", summary_path)


if __name__ == "__main__":
    main()

