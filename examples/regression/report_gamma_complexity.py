"""Generate a self-describing Gamma complexity and timing report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from fleet_management import solve, validate_gamma_replay_files


HERE = Path(__file__).resolve().parent
CASES = {
    "uniform_gamma": HERE / "gamma_tail_bound_public.yaml",
    "mixed_gamma_rainflow": HERE / "mixed_gamma_rainflow_public.yaml",
    "mixed_gamma_ard1": HERE / "mixed_gamma_ard1_public.yaml",
}


def build_case_report(name: str, input_path: Path, directory: Path) -> dict:
    result_path = directory / f"{name}_result.yaml"
    validation_path = directory / f"{name}_validation.yaml"
    solve(str(input_path), str(result_path))
    saved_result = yaml.safe_load(result_path.read_text(encoding="utf-8"))
    validation = validate_gamma_replay_files(
        input_path,
        result_path,
        validation_path,
        raise_on_failure=True,
    )

    return {
        "input": input_path.name,
        "outcome": {
            "status": saved_result["status"],
            "objective": saved_result["objective"],
            "backend": saved_result["backend"],
            "models": saved_result["models"],
        },
        "dimensions": {
            key: saved_result[key] for key in ("F", "M", "L", "H1", "H2", "T")
        },
        "offline_calibration": {
            "total_seconds": saved_result["performance"][
                "gamma_calibration_seconds"
            ],
            "cells": saved_result["gamma_calibration"],
        },
        "gurobi_formulation": saved_result["gamma_formulation"],
        "solve_performance": saved_result["performance"],
        "state_replay": {
            "valid": validation["valid"],
            "gamma_cells": validation["gamma_cells"],
            "gamma_ard1_cells": validation["gamma_ard1_cells"],
            "transitions_checked": validation["transitions_checked"],
            "repairs": validation["repairs"],
            "replacements": validation["replacements"],
            "maximum_errors": validation["maximum_errors"],
            "timing": validation["timing"],
        },
    }


def build_complexity_report() -> dict:
    with TemporaryDirectory(prefix="gamma-complexity-") as temporary:
        directory = Path(temporary)
        cases = {
            name: build_case_report(name, path, directory)
            for name, path in CASES.items()
        }

    return {
        "report": "Gamma calibration, formulation, solve and replay diagnostics",
        "report_version": 2,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Separates offline tail calibration, Gurobi formulation/solve, and "
            "lightweight post-solve schedule/state replay."
        ),
        "interpretation": {
            "deterministic_values": (
                "Variable and constraint counts are formulation properties and "
                "should reproduce exactly for the same code and input."
            ),
            "machine_dependent_values": (
                "All fields ending in _seconds, Gurobi work, iterations and node "
                "counts depend on hardware, software versions and solver settings."
            ),
            "replay_check": (
                "maximum_errors compares replayed physical mean, bounding shape, "
                "removed mean and ARD1 latch states with the serialized solution."
            ),
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional .yaml/.yml file; without it the report is printed.",
    )
    arguments = parser.parse_args()
    report = build_complexity_report()
    rendered = yaml.safe_dump(report, sort_keys=False)
    if arguments.output is None:
        print(rendered, end="")
        return
    if arguments.output.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("--output must end in .yaml or .yml")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
