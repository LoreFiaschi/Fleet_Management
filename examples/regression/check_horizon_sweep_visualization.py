"""Regression for the operating-horizon sweep visualisation."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.image as mpimg
import numpy as np
import yaml

from fleet_management import plot_horizon_sweep


def main() -> None:
    if plot_horizon_sweep.__module__ != "fleet_management.utils.mixed_plotter":
        raise AssertionError("public horizon plot does not route to mixed_plotter.py")

    cases = []
    for H2, cost, runtime, variables, constraints, status in (
        (4, 0.911251, 0.49, 232, 577, "optimal"),
        (8, 0.798001, 9.58, 348, 861, "optimal"),
        (12, 0.661948, 87.46, 464, 1145, "optimal"),
        (16, 0.617563, 900.0, 580, 1429, "time_limit"),
    ):
        cases.append({
            "H1": 4,
            "H2": H2,
            "T": 4 + H2,
            "status": status,
            "J_trans": 0.24,
            "J_op": cost * H2,
            "J_op_average": cost,
            "optimizer_seconds": runtime,
            "formulation": {
                "variables": variables,
                "linear_constraints": constraints,
            },
            "timing": {"optimizer_call_seconds": runtime},
        })

    report = {
        "objective": "minimize J_op / H2 subject to J_trans <= B_trans",
        "fixed_dimensions": {"F": 4, "M": 1, "L": 1, "H1": 4},
        "transitory_budget": 10.0,
        "cases": cases,
        "best_proven_H2": 12,
        "best_proven_J_op_average": 0.661948,
        "best_incumbent_H2": 16,
        "best_incumbent_J_op_average": 0.617563,
        "best_incumbent_status": "time_limit",
    }

    with TemporaryDirectory(prefix="horizon-sweep-visualisation-") as directory:
        root = Path(directory)
        report_path = root / "horizon_sweep.yaml"
        image_path = root / "horizon_sweep.png"
        report_path.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
        plot_horizon_sweep(str(report_path), str(image_path))

        if not image_path.is_file() or image_path.stat().st_size < 30_000:
            raise AssertionError("horizon-sweep visualisation was not created")
        image = mpimg.imread(image_path)
        if image.ndim != 3 or min(image.shape[:2]) < 600:
            raise AssertionError(f"unexpected horizon figure shape {image.shape}")
        if float(np.std(image)) < 0.05:
            raise AssertionError("horizon-sweep visualisation appears blank")

    print("PASS operating-horizon sweep visualisation")
    print("best proven H2   : 12")
    print("best incumbent H2: 16 (time limit)")
    print("panels            : operating cost, runtime and formulation growth")


if __name__ == "__main__":
    main()