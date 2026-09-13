#!/usr/bin/env python3
"""Measure the current Gamma formulation for four intervention variants.

The script uses the repository's validated Gamma horizon input as a template,
forces F=4, M=1, L=1 and H=[4, 4] (therefore T=8), solves the four very small
models, and writes aggregate Gurobi formulation counts to CSV and YAML.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from fleet_management.solver import solve


VARIANTS = (
    ("ardinf_no_replacement", "ardinf", False),
    ("ardinf_replacement", "ardinf", True),
    ("ard1_no_replacement", "ard1", False),
    ("ard1_replacement", "ard1", True),
)

COUNT_ALIASES = {
    "variables": ("variables", "num_variables", "total_variables"),
    "continuous_variables": (
        "continuous_variables",
        "num_continuous_variables",
    ),
    "integer_variables": ("integer_variables", "num_integer_variables"),
    "binary_variables": ("binary_variables", "num_binary_variables"),
    "linear_constraints": ("linear_constraints", "num_linear_constraints"),
    "general_constraints": ("general_constraints", "num_general_constraints"),
    "indicator_constraints": (
        "indicator_constraints",
        "num_indicator_constraints",
    ),
    "quadratic_constraints": (
        "quadratic_constraints",
        "num_quadratic_constraints",
    ),
    "nonzeros": ("nonzeros", "num_nonzeros"),
}


def first_present(sources: list[dict[str, Any]], aliases: tuple[str, ...]) -> Any:
    for source in sources:
        for key in aliases:
            if key in source and source[key] is not None:
                return source[key]
    return None


def extract_counts(result: dict[str, Any]) -> dict[str, Any]:
    sources = [
        result.get("formulation", {}),
        result.get("performance", {}),
        result,
    ]
    counts = {
        target: first_present(sources, aliases)
        for target, aliases in COUNT_ALIASES.items()
    }
    required = (
        "variables",
        "continuous_variables",
        "integer_variables",
        "linear_constraints",
    )
    missing = [key for key in required if counts[key] is None]
    if missing:
        available = sorted({key for source in sources for key in source})
        raise KeyError(
            f"Missing formulation counters {missing}. Available result keys: {available}"
        )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-input",
        type=Path,
        default=Path(
            "experiments/gamma_horizon_stabilization/input/"
            "gamma_horizon_stabilization.yaml"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/gamma_formulation_counts/results"),
    )
    args = parser.parse_args()

    base = yaml.safe_load(args.base_input.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for name, repair_model, allow_replacement in VARIANTS:
        cfg = deepcopy(base)
        cfg.update(
            {
                "F": 4,
                "M": 1,
                "L": 1,
                "H": [4, 4],
                "model": ["gamma"],
                "repair_model": [repair_model],
                "allow_replacement": allow_replacement,
                # Counts are available after model construction. The cases are
                # tiny, but cap optimization in case one variant is difficult.
                "time_limit": 120,
                "mip_gap": 1.0,
                "verbose": 0,
            }
        )
        cfg.pop("depot_capacity", None)

        params = dict(cfg.get("gurobi_params", {}))
        params.update({"Threads": 1, "Seed": 1})
        cfg["gurobi_params"] = params

        input_path = args.output_dir / f"{name}.yaml"
        result_path = args.output_dir / f"{name}_result.yaml"
        input_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        print(f"Building {name} ...", flush=True)
        result = solve(str(input_path), str(result_path))
        row = {
            "case": name,
            "repair_model": repair_model,
            "allow_replacement": allow_replacement,
            "F": 4,
            "M": 1,
            "L": 1,
            "H1": 4,
            "H2": 4,
            "T": 8,
            **extract_counts(result),
        }
        rows.append(row)
        print(
            f"  variables={row['variables']} "
            f"(continuous={row['continuous_variables']}, "
            f"integer={row['integer_variables']}), "
            f"linear_constraints={row['linear_constraints']}"
        )

    csv_path = args.output_dir / "gamma_formulation_counts_F4_M1_L1_T8.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    yaml_path = args.output_dir / "gamma_formulation_counts_F4_M1_L1_T8.yaml"
    yaml_path.write_text(yaml.safe_dump({"cases": rows}, sort_keys=False), encoding="utf-8")

    print(f"\nCSV:  {csv_path}")
    print(f"YAML: {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
