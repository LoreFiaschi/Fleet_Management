"""Run the two uniform-model smoke cases and compare stable solution values.

This is deliberately a standalone regression command, not a pytest module.
It requires the project dependencies and a working Gurobi installation/license.
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from fleet_management import solve  # noqa: E402


EXPECTED_FILES = {
    "gamma": HERE / "expected_uniform_gamma.yaml",
    "rainflow": HERE / "expected_uniform_rainflow.yaml",
}


def _compare(name: str, actual: Any, expected: Any, atol: float) -> None:
    """Compare one saved check and raise a readable assertion on a mismatch."""
    if isinstance(expected, str):
        if str(actual) != expected:
            raise AssertionError(f"{name}: expected {expected!r}, got {actual!r}")
        return

    expected_array = np.asarray(expected)
    actual_array = np.asarray(actual)
    if expected_array.ndim == 0:
        if not math.isclose(
            float(actual_array), float(expected_array), rel_tol=0.0, abs_tol=atol
        ):
            raise AssertionError(
                f"{name}: expected {float(expected_array)}, got {float(actual_array)}"
            )
        return

    np.testing.assert_allclose(
        actual_array,
        expected_array,
        rtol=0.0,
        atol=atol,
        err_msg=f"regression mismatch for {name}",
    )


def run_case(case: str) -> None:
    expected_path = EXPECTED_FILES[case]
    with expected_path.open(encoding="utf-8") as stream:
        snapshot = yaml.safe_load(stream)

    input_path = HERE / snapshot["fixture"]
    atol = float(snapshot.get("atol", 1.0e-6))

    with tempfile.TemporaryDirectory(prefix=f"fleet-{case}-regression-") as directory:
        result_path = Path(directory) / "solver_output.yaml"
        result = solve(str(input_path), str(result_path))

    for key, expected_value in snapshot["checks"].items():
        if key not in result:
            raise AssertionError(f"{case}: solver result is missing key {key!r}")
        _compare(f"{case}.{key}", result[key], expected_value, atol)

    print(
        f"PASS {case}: objective={float(result['objective']):.6f}, "
        f"checked {len(snapshot['checks'])} values"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("all", *EXPECTED_FILES),
        default="all",
        help="run one fixture or both (default: both)",
    )
    args = parser.parse_args()

    cases = EXPECTED_FILES if args.case == "all" else (args.case,)
    for case in cases:
        run_case(case)


if __name__ == "__main__":
    main()