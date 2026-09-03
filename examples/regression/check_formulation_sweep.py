"""Regression for deterministic F/M/L/T formulation-size sweeps."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from fleet_management.formulation_size_sweep import sweep_formulation_dimensions


def main() -> None:
    scenario = {
        "F": 2,
        "M": 1,
        "L": 1,
        "component_names": ["Battery"],
        "H": [2, 2],
        "model": "gamma",
        "repair_model": "ardinf",
        "gamma_calibration_method": "repeated_increment",
        "mu": 0.02,
        "gamma_beta": 20.0,
        "gamma_beta_bound": 10.0,
        "mu_0": 0.0,
        "replacement_mu": 0.0,
        "allow_replacement": False,
        "tau": 0.6,
        "epsilon": 0.1,
        "rho": 0.5,
        "depot_capacity": 1,
        "C_M": 1.0,
        "C_R": 0.5,
        "C_D": 2.0,
        "C_rep": 0.2,
        "objective_mode": "operating_average",
    }

    with TemporaryDirectory(prefix="formulation-sweep-") as directory:
        root = Path(directory)
        input_path = root / "input.yaml"
        output_path = root / "report.yaml"
        input_path.write_text(yaml.safe_dump(scenario), encoding="utf-8")
        report = sweep_formulation_dimensions(
            input_path,
            candidates={
                "F": [2, 3],
                "M": [1, 2],
                "L": [1, 2],
                "T": [4, 6],
            },
            output_path=output_path,
        )
        saved = yaml.safe_load(output_path.read_text(encoding="utf-8"))

    if saved["method"] != report["method"]:
        raise AssertionError("saved formulation report changed")
    if set(report["sweeps"]) != {"F", "M", "L", "T"}:
        raise AssertionError("formulation report is missing a dimension sweep")

    for parameter, sweep in report["sweeps"].items():
        cases = sweep["cases"]
        if len(cases) != 2:
            raise AssertionError(f"{parameter} sweep has the wrong case count")
        if cases[1]["counts"]["variables"] <= cases[0]["counts"]["variables"]:
            raise AssertionError(f"{parameter} variable count did not increase")
        if (
            cases[1]["counts"]["linear_constraints"]
            <= cases[0]["counts"]["linear_constraints"]
        ):
            raise AssertionError(f"{parameter} row count did not increase")
        if sweep["first_to_last_growth"] is None:
            raise AssertionError(f"{parameter} growth summary is missing")

    baseline = report["sweeps"]["F"]["cases"][0]
    if baseline["counts"] != {
        "variables": 60,
        "integer_variables": 32,
        "binary_variables": 32,
        "continuous_variables": 28,
        "linear_constraints": 152,
        "general_constraints": 0,
        "quadratic_constraints": 0,
    }:
        raise AssertionError(
            f"analytical baseline changed: {baseline['counts']}"
        )
    if baseline["counts"]["integer_variables"] != 32:
        raise AssertionError("integer-variable count was not retained")

    print("PASS deterministic formulation-size sweep")
    for parameter, sweep in report["sweeps"].items():
        print(parameter, sweep["first_to_last_growth"])


if __name__ == "__main__":
    main()
