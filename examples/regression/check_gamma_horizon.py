"""Solve and verify the unequal-horizon uniform Gamma regression instance."""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parent
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from fleet_management import solve, validate_gamma_result  # noqa: E402


def _compare(name: str, actual: Any, expected: Any, atol: float) -> None:
    if isinstance(expected, str):
        if str(actual) != expected:
            raise AssertionError(f"{name}: expected {expected!r}, got {actual!r}")
        return

    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if expected_array.ndim == 0:
        if not math.isclose(
            float(actual_array), float(expected_array), rel_tol=0.0, abs_tol=atol
        ):
            raise AssertionError(f"{name}: expected {expected}, got {actual}")
        return

    np.testing.assert_allclose(
        actual_array,
        expected_array,
        rtol=0.0,
        atol=atol,
        err_msg=f"regression mismatch for {name}",
    )


def main() -> None:
    expected_path = HERE / "expected_gamma_unequal_horizon.yaml"
    with expected_path.open(encoding="utf-8") as stream:
        snapshot = yaml.safe_load(stream)

    input_path = HERE / snapshot["fixture"]
    atol = float(snapshot["atol"])
    with tempfile.TemporaryDirectory(prefix="fleet-gamma-horizon-") as directory:
        result_path = Path(directory) / "output.yaml"
        report_path = Path(directory) / "validation.yaml"
        result = solve(str(input_path), str(result_path))
        report = validate_gamma_result(
            str(input_path), str(result_path), str(report_path), tolerance=atol
        )

    for key, expected in snapshot["checks"].items():
        if key not in result:
            raise AssertionError(f"solver result is missing key {key!r}")
        _compare(key, result[key], expected, atol)

    if not report["passed"]:
        failed = [check["name"] for check in report["checks"] if not check["passed"]]
        raise AssertionError(f"independent Gamma validation failed: {failed}")

    print(
        f"PASS unequal Gamma horizon: H1={result['H1']}, H2={result['H2']}, "
        f"T={result['T']}, objective={float(result['objective']):.6f}"
    )


if __name__ == "__main__":
    main()
