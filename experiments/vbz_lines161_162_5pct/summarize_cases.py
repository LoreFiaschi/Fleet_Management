#!/usr/bin/env python3
"""Combine the completed 5%-target cases into CSV and YAML reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


CASE_ROUTES = {
    "line162_control": "Line 162",
    "lines161_162": "Lines 161 and 162",
}

FIELDS = [
    "case",
    "routes",
    "F",
    "M",
    "L",
    "H1",
    "H2",
    "T",
    "annual_repetitions",
    "status",
    "usable_incumbent",
    "qualified_at_5_percent",
    "objective",
    "bound",
    "mip_gap",
    "J_op",
    "J_op_average",
    "gurobi_runtime_seconds",
    "variables",
    "continuous_variables",
    "binary_variables",
    "linear_constraints",
    "indicator_constraints",
    "quadratic_constraints",
    "branch_and_bound_nodes",
    "simplex_iterations",
    "input",
    "result",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.run_dir.glob("case_*_summary.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["routes"] = CASE_ROUTES.get(row["case"], row["case"])
        row["annual_repetitions"] = 4
        row["usable_incumbent"] = row.get("objective") is not None
        gap = row.get("mip_gap")
        row["qualified_at_5_percent"] = (
            gap is not None and float(gap) <= 0.05 + 1e-9
        )
        rows.append(row)

    csv_path = args.run_dir / "cases.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "experiment": "vbz_lines161_162_5pct",
        "interpretation": {
            "period": "one aggregate week",
            "H1_weeks": 4,
            "H2_weeks": 12,
            "annual_extension": "repeat H2 four times after H1",
            "data_status": "synthetic degradation and cost inputs",
        },
        "mip_gap_target": 0.05,
        "cases": rows,
    }
    (args.run_dir / "summary.yaml").write_text(
        yaml.safe_dump(summary, sort_keys=False), encoding="utf-8"
    )
    print(yaml.safe_dump(summary, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
