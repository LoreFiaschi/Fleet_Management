#!/usr/bin/env python3
"""Run and summarize the one-hour optimality-focused mixed-model case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from fleet_management.solver import solve


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()

    result = solve(str(args.input), str(args.output))
    performance = result.get("performance", {})
    summary = {
        "experiment": "vbz_lines161_162_tangent_optimality_1h_32c",
        "status": result.get("status"),
        "objective": result.get("objective"),
        "bound": result.get("bound", result.get("best_bound")),
        "mip_gap": result.get("mip_gap"),
        "qualified_at_5_percent": (
            result.get("mip_gap") is not None
            and float(result["mip_gap"]) <= 0.05 + 1e-9
        ),
        "J_op": result.get("J_op"),
        "J_op_average": result.get("J_op_average"),
        "F": result.get("F"),
        "M": result.get("M"),
        "L": result.get("L"),
        "H1": result.get("H1"),
        "H2": result.get("H2"),
        "T": result.get("T"),
        "reliability_impl": result.get("reliability_impl"),
        "variables": performance.get("variables"),
        "continuous_variables": performance.get("continuous_variables"),
        "binary_variables": performance.get("binary_variables"),
        "linear_constraints": performance.get("linear_constraints"),
        "indicator_constraints": performance.get("indicator_constraints"),
        "quadratic_constraints": performance.get("quadratic_constraints"),
        "gurobi_runtime_seconds": performance.get("gurobi_runtime_seconds"),
        "branch_and_bound_nodes": performance.get("branch_and_bound_nodes"),
        "simplex_iterations": performance.get("simplex_iterations"),
        "work_units": performance.get("work_units"),
        "solutions_found": performance.get("solutions_found"),
        "input": str(args.input),
        "result": str(args.output),
    }
    args.summary.write_text(
        yaml.safe_dump(summary, sort_keys=False), encoding="utf-8"
    )
    print("RESULT SUMMARY")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
