"""Build mixed fleet formulations and report their exact Gurobi sizes.

The models are assembled and updated, but ``Model.optimize()`` is deliberately
not called.  The resulting counts therefore describe the original formulation
and are independent of solver hardware, time limits, presolve and MIP progress.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
import time

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import yaml

from fleet_management.config import load_config
from fleet_management.degradation_model.base import build_fleet, resolve_run_options
from fleet_management.degradation_model.gamma_utils.gamma_diagnostics import (
    compare_estimate_with_actual,
    estimate_gamma_formulation,
)


def _read_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return data


def _model_statistics(model: gp.Model) -> dict:
    """Read formulation attributes that are available before optimization."""
    model.update()
    general_constraints = list(model.getGenConstrs())
    indicator_constraints = sum(
        int(item.GenConstrType == GRB.GENCONSTR_INDICATOR)
        for item in general_constraints
    )
    return {
        "variables": int(model.NumVars),
        "continuous_variables": int(model.NumVars - model.NumIntVars),
        "integer_variables": int(model.NumIntVars),
        "binary_variables": int(model.NumBinVars),
        "linear_constraints": int(model.NumConstrs),
        "general_constraints": int(model.NumGenConstrs),
        "indicator_constraints": int(indicator_constraints),
        "quadratic_constraints": int(model.NumQConstrs),
        "constraint_objects": int(
            model.NumConstrs + model.NumGenConstrs + model.NumQConstrs
        ),
        "nonzeros": int(model.NumNZs),
    }


def build_case(input_path: Path) -> dict:
    data = _read_yaml(input_path)
    cfg = load_config(data)
    options = resolve_run_options(cfg)

    started = time.perf_counter()
    context = build_fleet(
        cfg,
        options,
        model_name=f"formulation_size_{input_path.stem}",
    )
    try:
        statistics = _model_statistics(context.model)
        construction_seconds = time.perf_counter() - started
        gamma_estimate = estimate_gamma_formulation(
            cfg,
            allow_replacement=context.allow_replacement,
        )
        comparison = compare_estimate_with_actual(gamma_estimate, statistics)
    finally:
        context.model.dispose()

    return {
        "case": input_path.stem,
        "input": str(input_path),
        "dimensions": {
            "F": cfg.F,
            "M": cfg.M,
            "L": cfg.L,
            "H1": cfg.H1,
            "H2": cfg.H2,
            "T": cfg.T,
        },
        "components": [
            {
                "name": cfg.component_names[l],
                "model": str(cfg.model[0, l]),
                "bound_method": str(cfg.bound_method[0, l]),
                "repair_model": str(cfg.repair_model[0, l]),
            }
            for l in range(cfg.L)
        ],
        "allow_replacement": bool(context.allow_replacement),
        "construction_seconds": construction_seconds,
        "counts": statistics,
        "gamma_cells": gamma_estimate["gamma_cells"],
        "non_gamma_remainder": comparison["non_gamma_remainder"],
    }


def create_report(input_paths: list[Path]) -> dict:
    return {
        "report": "Exact build-only mixed-formulation sizes",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "gurobi_version": ".".join(str(value) for value in gp.gurobi.version()),
        "method": (
            "Build and update each Gurobi model without calling optimize; "
            "counts describe the original, unpresolved formulation."
        ),
        "interpretation": {
            "linear_constraints": "Ordinary rows reported by Model.NumConstrs.",
            "general_constraints": (
                "Gurobi general constraints; indicator constraints are a subset."
            ),
            "quadratic_constraints": (
                "Quadratic reliability rows, if selected by the remaining-life model."
            ),
            "constraint_objects": (
                "Linear + general + quadratic constraint objects; these categories "
                "are also reported separately and should not be confused with "
                "presolved rows."
            ),
        },
        "cases": [build_case(path) for path in input_paths],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build mixed fleet models and report exact formulation counts."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more mixed-model YAML inputs.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination .yaml or .yml report.",
    )
    arguments = parser.parse_args()

    if arguments.output.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("--output must end in .yaml or .yml")
    for path in arguments.inputs:
        if path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(f"input must be YAML: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)

    report = create_report(arguments.inputs)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        yaml.safe_dump(report, sort_keys=False),
        encoding="utf-8",
    )

    print("\nEXACT BUILD-ONLY FORMULATION COUNTS")
    print(
        "case                              continuous    integer     linear  "
        " general  indicator  quadratic"
    )
    for case in report["cases"]:
        counts = case["counts"]
        print(
            f"{case['case']:<34} "
            f"{counts['continuous_variables']:>10} "
            f"{counts['integer_variables']:>10} "
            f"{counts['linear_constraints']:>10} "
            f"{counts['general_constraints']:>8} "
            f"{counts['indicator_constraints']:>10} "
            f"{counts['quadratic_constraints']:>10}"
        )
    print("Report:", arguments.output)


if __name__ == "__main__":
    main()
