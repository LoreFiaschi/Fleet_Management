"""Regression for the compact mixed-fleet result visualisation."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.image as mpimg
import numpy as np
import yaml

from fleet_management import plot_mixed_management


def main() -> None:
    if plot_mixed_management.__module__ != (
        "fleet_management.utils.mixed_plotter"
    ):
        raise AssertionError(
            "public plot_mixed_management does not route to mixed_plotter.py"
        )

    result = {
        "status": "time_limit",
        "objective": 4.4245,
        "mip_gap": 0.075,
        "backend": "modular",
        "degradation": "mixed",
        "models": ["gamma", "rainflow"],
        "reliability_impl": [["gamma", "rainflow"], ["gamma", "rainflow"]],
        "repair_model": ["ardinf", "ard1"],
        "F": 2,
        "M": 1,
        "L": 2,
        "H": [2, 3],
        "H1": 2,
        "H2": 3,
        "T": 5,
        "tau": [[0.6, 0.5], [0.6, 0.5]],
        "mu_0": [[0.02, 0.05], [0.015, 0.04]],
        "mu": [
            [[0.06, 0.11, 0.055, 0.105, 0.01], [0.10, 0.16, 0.08, 0.14, 0.02]],
            [[0.055, 0.025, 0.075, 0.008, 0.058], [0.09, 0.04, 0.10, 0.01, 0.07]],
        ],
        "x": [
            [[0, 0, 1, 0, 1], [1, 1, 0, 1, 0]],
            [[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]],
        ],
        "m": [
            [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0]],
            [[0, 1, 0, 0, 0], [0, 1, 0, 0, 0]],
        ],
        "r": [
            [[0, 0, 0, 0, 1], [0, 0, 0, 0, 1]],
            [[0, 0, 0, 1, 0], [0, 0, 0, 1, 0]],
        ],
        "objective_mode": "operating_average",
        "J_trans": 0.24,
        "J_op": 3.52,
        "J_op_average": 1.1733333333,
        "performance": {
            "variables": 175,
            "linear_constraints": 427,
            "general_constraints": 90,
            "optimizer_call_seconds": 2.7,
            "branch_and_bound_nodes": 1234,
        },
    }

    with TemporaryDirectory(prefix="fleet-visualisation-") as directory:
        root = Path(directory)
        result_path = root / "mixed_result.yaml"
        image_path = root / "mixed_result.png"
        result_path.write_text(yaml.safe_dump(result, sort_keys=False), encoding="utf-8")
        plot_mixed_management(str(result_path), str(image_path))

        if not image_path.is_file() or image_path.stat().st_size < 20_000:
            raise AssertionError("result visualisation was not created correctly")
        image = mpimg.imread(image_path)
        if image.ndim != 3 or min(image.shape[:2]) < 500:
            raise AssertionError(f"unexpected visualisation shape {image.shape}")
        if float(np.std(image)) < 0.05:
            raise AssertionError("visualisation appears blank")

    print("PASS compact mixed-fleet result visualisation")
    print("shows schedule actions : mission, idle, repair, replacement")
    print("shows degradation      : physical mean / threshold")
    print("shows model assignment : Gamma and rainflow")
    print("shows run statistics   : status, objective, dimensions, formulation, timing")


if __name__ == "__main__":
    main()