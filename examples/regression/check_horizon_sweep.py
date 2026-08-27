"""Regression for the simple outer operating-horizon sweep."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from fleet_management import sweep_operating_horizons


def main() -> None:
    scenario = {
        "F": 2,
        "M": 1,
        "L": 1,
        "H": [2, 2],
        "model": "gamma",
        "repair_model": "ardinf",
        "tau": 0.6,
        "epsilon": 0.1,
        "rho": 0.5,
        "mu_0": 0.01,
        "replacement_mu": 0.005,
        "mu": 0.02,
        "gamma_beta": 10.0,
        "gamma_beta_bound": 10.0,
        "gamma_beta_0": 10.0,
        "gamma_beta_new": 10.0,
        "gamma_calibration_method": "repeated_increment",
        "C_M": 1.0,
        "C_R": 0.5,
        "C_D": 2.0,
        "C_rep": 0.2,
        "allow_replacement": True,
        "depot_capacity": 1,
        "mip_gap": 0.0,
        "verbose": 0,
        "transitory_budget": 10.0,
    }

    with TemporaryDirectory(prefix="horizon-sweep-regression-") as directory:
        root = Path(directory)
        input_path = root / "scenario.yaml"
        output_path = root / "sweep.yaml"
        input_path.write_text(
            yaml.safe_dump(scenario, sort_keys=False), encoding="utf-8"
        )
        report = sweep_operating_horizons(
            input_path, [2, 3], output_path=output_path
        )
        saved = yaml.safe_load(output_path.read_text(encoding="utf-8"))

    if len(report["cases"]) != 2:
        raise AssertionError("wrong number of horizon cases")
    if [row["H2"] for row in report["cases"]] != [2, 3]:
        raise AssertionError("H2 candidates were not retained")
    if any(row["status"] != "optimal" for row in report["cases"]):
        raise AssertionError("a horizon candidate did not solve optimally")
    if any(row["J_trans"] > 10.0 + 1e-8 for row in report["cases"]):
        raise AssertionError("a horizon candidate violates B_trans")
    if report["best_H2"] not in {2, 3}:
        raise AssertionError("no best horizon was selected")
    expected_best = min(
        report["cases"], key=lambda row: row["J_op_average"]
    )["H2"]
    if report["best_H2"] != expected_best:
        raise AssertionError("best H2 does not minimize J_op/H2")
    if saved["best_H2"] != report["best_H2"]:
        raise AssertionError("saved sweep report changed the selected horizon")
    for row in report["cases"]:
        if row["dimensions"] != {
            "F": 2, "M": 1, "L": 1,
            "H1": 2, "H2": row["H2"], "T": row["T"],
        }:
            raise AssertionError("sweep dimensions are incomplete")
        if row["formulation"]["variables"] <= 0:
            raise AssertionError("sweep formulation has no variables")
        if row["formulation"]["linear_constraints"] <= 0:
            raise AssertionError("sweep formulation has no linear constraints")
        if row["calibration"]["gamma_cells"] != 2:
            raise AssertionError("sweep Gamma-cell count is wrong")
        if row["calibration"]["maximum_safe_count"] is None:
            raise AssertionError("sweep omitted repeated-calibration m*")
    if report["formulation_growth"] is None:
        raise AssertionError("sweep omitted formulation growth diagnostics")

    print("PASS operating-horizon sweep")
    for row in report["cases"]:
        print(
            f"H2={row['H2']}: J_trans={row['J_trans']:.6g}, "
            f"J_op/H2={row['J_op_average']:.6g}"
        )
    print("best H2:", report["best_H2"])


if __name__ == "__main__":
    main()