"""Solve a progressive VBZ-size ladder and checkpoint a compact summary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
import subprocess
from typing import Any

import yaml

from fleet_management import solve, validate_gamma_schedule_files


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        check=False,
    )
    return completed.stdout.strip()


def _write(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    arguments = parser.parse_args()

    for path in arguments.inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "report": "Progressive VBZ regional feasibility ladder",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "git_commit": _git_commit(),
        "interpretation": (
            "Each rung changes one dimension. A feasible time-limit result is "
            "a valid schedule but not an optimality certificate."
        ),
        "cases": [],
    }

    errors = []
    for index, input_path in enumerate(arguments.inputs):
        result_path = arguments.output_dir / f"{input_path.stem}_result.yaml"
        validation_path = (
            arguments.output_dir / f"{input_path.stem}_validation.yaml"
        )
        try:
            result = solve(str(input_path), str(result_path))
            feasible = result.get("objective") is not None
            validation = None
            if feasible:
                validation = validate_gamma_schedule_files(
                    input_path,
                    result_path,
                    validation_path,
                    mode="deterministic",
                    raise_on_failure=True,
                )
            performance = result.get("performance") or {}
            report["cases"].append({
                "rung": index,
                "input": str(input_path),
                "status": result.get("status"),
                "feasible_schedule_found": feasible,
                "objective": result.get("objective"),
                "objective_bound": result.get("bound"),
                "mip_gap": result.get("mip_gap"),
                "dimensions": {
                    key: result.get(key) for key in ("F", "M", "L", "H1", "H2", "T")
                },
                "deterministic_gamma_replay": (
                    None if validation is None else validation.get("valid")
                ),
                "formulation": {
                    key: performance.get(key) for key in (
                        "continuous_variables", "integer_variables",
                        "linear_constraints", "general_constraints",
                        "quadratic_constraints",
                    )
                },
                "result": str(result_path),
                "validation": str(validation_path) if validation else None,
            })
        except Exception as error:
            errors.append(f"{input_path}: {type(error).__name__}: {error}")
            report["cases"].append({
                "rung": index,
                "input": str(input_path),
                "status": "runner_error",
                "error": errors[-1],
            })
        _write(report, arguments.summary)

    print("\nPROGRESSIVE VBZ FEASIBILITY SUMMARY")
    print("rung  F   M   L   H2   status          feasible   gap")
    for case in report["cases"]:
        dimensions = case.get("dimensions") or {}
        gap = case.get("mip_gap")
        shown_gap = "-" if gap is None else f"{100.0 * float(gap):.2f}%"
        print(
            f"{case['rung']:>4}  {str(dimensions.get('F', '-')):>2}  "
            f"{str(dimensions.get('M', '-')):>2}  "
            f"{str(dimensions.get('L', '-')):>2}  "
            f"{str(dimensions.get('H2', '-')):>3}   "
            f"{str(case.get('status')):<15} "
            f"{str(case.get('feasible_schedule_found', False)):<8}   "
            f"{shown_gap}"
        )
    print("Report:", arguments.summary)
    if errors:
        raise RuntimeError("; ".join(errors))


if __name__ == "__main__":
    main()
