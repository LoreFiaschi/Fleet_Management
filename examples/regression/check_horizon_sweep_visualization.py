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
    for H2, cost, bound, gap, runtime, continuous, integer, constraints, status in (
        (4, 0.911251, 0.911251, 0.0, 0.49, 104, 128, 576, "optimal"),
        (8, 0.798001, 0.798001, 0.0, 9.58, 156, 192, 860, "optimal"),
        (12, 0.661948, 0.661948, 0.0, 87.46, 208, 256, 1144, "optimal"),
        (16, 0.617563, 0.550000, 0.1094, 900.0, 260, 320, 1428, "time_limit"),
    ):
        cases.append({
            "H1": 4,
            "H2": H2,
            "T": 4 + H2,
            "status": status,
            "J_op": cost * H2,
            "J_op_average": cost,
            "objective_bound": bound,
            "mip_gap": gap,
            "optimizer_seconds": runtime,
            "formulation": {
                "variables": continuous + integer,
                "continuous_variables": continuous,
                "integer_variables": integer,
                "linear_constraints": constraints,
            },
            "timing": {"optimizer_call_seconds": runtime},
        })

    report = {
        "objective": "minimize J_op / H2 over the operating phase",
        "fixed_dimensions": {"F": 4, "M": 1, "L": 1, "H1": 4},
        "cases": cases,
        "best_proven_H2": 12,
        "best_proven_J_op_average": 0.661948,
        "best_feasible_H2": 16,
        "best_feasible_J_op_average": 0.617563,
        "best_feasible_status": "time_limit",
    }

    with TemporaryDirectory(prefix="horizon-sweep-visualisation-") as directory:
        root = Path(directory)
        report_path = root / "horizon_sweep.yaml"
        extension_path = root / "horizon_sweep_extension.yaml"
        image_path = root / "horizon_sweep.png"
        report_path.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
        extension = {
            "objective": report["objective"],
            "fixed_dimensions": report["fixed_dimensions"],
            "cases": [
                cases[-1],
                {
                    "H1": 4,
                    "H2": 20,
                    "T": 24,
                    "status": "time_limit",
                    "J_op": None,
                    "J_op_average": None,
                    "objective_bound": 0.45,
                    "mip_gap": None,
                    "optimizer_seconds": 900.0,
                    "formulation": {
                        "variables": 696,
                        "continuous_variables": 312,
                        "integer_variables": 384,
                        "linear_constraints": 1712,
                    },
                    "timing": {"optimizer_call_seconds": 900.0},
                },
            ],
        }
        extension_path.write_text(
            yaml.safe_dump(extension, sort_keys=False), encoding="utf-8"
        )
        plot_horizon_sweep(
            [str(report_path), str(extension_path)], str(image_path)
        )

        if not image_path.is_file() or image_path.stat().st_size < 30_000:
            raise AssertionError("horizon-sweep visualisation was not created")
        image = mpimg.imread(image_path)
        if image.ndim != 3 or min(image.shape[:2]) < 600:
            raise AssertionError(f"unexpected horizon figure shape {image.shape}")
        if float(np.std(image)) < 0.05:
            raise AssertionError("horizon-sweep visualisation appears blank")

    print("PASS operating-horizon sweep visualisation")
    print("best proven H2   : 12")
    print("best feasible H2: 16 (time limit)")
    print("no feasible H2    : 20")
    print("panels            : operating cost, MIP gap and formulation growth")


if __name__ == "__main__":
    main()
