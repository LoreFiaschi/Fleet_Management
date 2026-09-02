"""Regression for the simple outer operating-horizon sweep."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from fleet_management import sweep_operating_horizons
from fleet_management.horizon_sweep import (
    _annotate_latest_gradient,
    _gradient_stop_reason,
    _select_horizon_cases,
)


def main() -> None:
    synthetic = [
        {
            "H2": 12,
            "status": "optimal",
            "J_op_average": 0.66,
            "mip_gap": 0.0,
        },
        {
            "H2": 16,
            "status": "time_limit",
            "J_op_average": 0.61,
            "mip_gap": 0.10,
        },
    ]

    proven, feasible = _select_horizon_cases(synthetic)

    if proven is None or proven["H2"] != 12:
        raise AssertionError(
            "time-limit case displaced the proven optimum"
        )

    if feasible is None or feasible["H2"] != 16:
        raise AssertionError(
            "best feasible case was not retained"
        )

    gradient_rows = []
    for H2, cost in ((4, 1.0), (8, 0.8), (12, 0.8002)):
        gradient_rows.append({
            "H2": H2,
            "status": "optimal",
            "J_op_average": cost,
            "mip_gap": 0.0,
        })
        _annotate_latest_gradient(
            gradient_rows,
            gradient_tolerance=1e-3,
            maximum_mip_gap=0.05,
        )
    if gradient_rows[-1]["gradient_classification"] != "flat":
        raise AssertionError("near-zero horizon gradient was not classified as flat")
    if _gradient_stop_reason(
        gradient_rows, minimum_cases=3, flat_gradients_required=1
    ) != "flat_gradient":
        raise AssertionError("flat gradient did not trigger the stopping rule")

    gradient_rows.append({
        "H2": 16,
        "status": "time_limit",
        "J_op_average": 0.84,
        "mip_gap": 0.02,
    })
    _annotate_latest_gradient(
        gradient_rows,
        gradient_tolerance=1e-3,
        maximum_mip_gap=0.05,
    )
    if _gradient_stop_reason(
        gradient_rows, minimum_cases=3, flat_gradients_required=2
    ) != "cost_increase":
        raise AssertionError("gap-qualified cost increase did not stop the sweep")
    
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
        adaptive = sweep_operating_horizons(
            input_path,
            [2, 3, 4, 5],
            stop_on_gradient=True,
            gradient_tolerance=1.0,
            flat_gradients_required=1,
            minimum_cases=3,
            maximum_mip_gap_for_stopping=0.05,
        )
        saved = yaml.safe_load(output_path.read_text(encoding="utf-8"))

    if len(report["cases"]) != 2:
        raise AssertionError("wrong number of horizon cases")
    if [row["H2"] for row in report["cases"]] != [2, 3]:
        raise AssertionError("H2 candidates were not retained")
    if any(row["status"] != "optimal" for row in report["cases"]):
        raise AssertionError("a horizon candidate did not solve optimally")
    if report["best_H2"] not in {2, 3}:
        raise AssertionError("no best horizon was selected")
    expected_best = min(
        report["cases"], key=lambda row: row["J_op_average"]
    )["H2"]
    if report["best_proven_H2"] != expected_best:
        raise AssertionError(
            "best proven H2 does not minimize J_op/H2"
        )

    if report["best_feasible_H2"] != expected_best:
        raise AssertionError(
            "best feasible H2 does not minimize J_op/H2"
        )

    if report["best_feasible_status"] != "optimal":
        raise AssertionError(
            "optimal regression feasible case has wrong status"
        )

    if report["best_H2"] != report["best_proven_H2"]:
        raise AssertionError(
            "legacy best_H2 is not the proven selection"
        )
    if saved["best_H2"] != report["best_H2"]:
        raise AssertionError("saved sweep report changed the selected horizon")
    if not saved.get("complete", False):
        raise AssertionError("finished sweep report was left as an incomplete checkpoint")
    for row in report["cases"]:
        if row["dimensions"] != {
            "F": 2, "M": 1, "L": 1,
            "H1": 2, "H2": row["H2"], "T": row["T"],
        }:
            raise AssertionError("sweep dimensions are incomplete")
        if row["formulation"]["variables"] <= 0:
            raise AssertionError("sweep formulation has no variables")
        if row["formulation"]["continuous_variables"] <= 0:
            raise AssertionError("sweep omitted continuous-variable counts")
        if row["formulation"]["integer_variables"] <= 0:
            raise AssertionError("sweep omitted integer-variable counts")
        if row["formulation"]["linear_constraints"] <= 0:
            raise AssertionError("sweep formulation has no linear constraints")
        if row["calibration"]["gamma_cells"] != 2:
            raise AssertionError("sweep Gamma-cell count is wrong")
        if row["calibration"]["maximum_safe_count"] is None:
            raise AssertionError("sweep omitted repeated-calibration m*")
        if row["objective_bound"] is None:
            raise AssertionError("sweep omitted the objective bound")
        if row["mip_gap"] is None:
            raise AssertionError("sweep omitted the relative MIP gap")
        if row["mip_gap"] > 1e-8:
            raise AssertionError("optimal regression case has a nonzero MIP gap")
    if report["formulation_growth"] is None:
        raise AssertionError("sweep omitted formulation growth diagnostics")
    if len(adaptive["cases"]) != 3:
        raise AssertionError("adaptive sweep did not stop after its third case")
    if adaptive["stopping_rule"]["reason"] != "flat_gradient":
        raise AssertionError("adaptive sweep saved the wrong stopping reason")
    if not adaptive["stopping_rule"]["stopped_early"]:
        raise AssertionError("adaptive sweep did not report its early stop")

    print("PASS operating-horizon sweep")
    for row in report["cases"]:
        print(
            f"H2={row['H2']}: J_op/H2={row['J_op_average']:.6g}, "
            f"gap={row['mip_gap']:.2%}"
        )
    print("best proven H2   :", report["best_proven_H2"])
    print("best feasible H2 :", report["best_feasible_H2"])


if __name__ == "__main__":
    main()
