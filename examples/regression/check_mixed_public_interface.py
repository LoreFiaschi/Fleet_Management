"""Public YAML-to-validation regression for a mixed degradation fleet."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml

from fleet_management import solve, validate_gamma_tail_bound_files


HERE = Path(__file__).resolve().parent
INPUT = HERE / "mixed_gamma_rainflow_public.yaml"


def main() -> None:
    with TemporaryDirectory(prefix="mixed-public-") as directory:
        result_path = Path(directory) / "result.yaml"
        validation_path = Path(directory) / "gamma_validation.yaml"
        result = solve(str(INPUT), str(result_path))
        saved = yaml.safe_load(result_path.read_text(encoding="utf-8"))
        report = validate_gamma_tail_bound_files(
            INPUT,
            result_path,
            validation_path,
            include_steps=True,
            raise_on_failure=True,
        )

        if not validation_path.is_file():
            raise AssertionError("mixed Gamma validation report was not saved")

    if result["status"] != "optimal":
        raise AssertionError(f"expected optimal solve, got {result['status']!r}")
    if result["backend"] != "modular":
        raise AssertionError(f"expected modular backend, got {result['backend']!r}")
    if result["degradation"] != "mixed":
        raise AssertionError("mixed degradation identity was not preserved")
    if result["models"] != ["gamma", "rainflow"]:
        raise AssertionError(f"unexpected model list {result['models']!r}")
    if (result["H1"], result["H2"], result["T"]) != (2, 3, 5):
        raise AssertionError("unequal mixed horizon was not preserved")

    expected_shapes = {
        "x": (2, 2, 5),
        "mu": (2, 2, 5),
        "gamma_shape_bound": (2, 2, 5),
        "gamma_tail_bound": (2, 2, 5),
    }
    for key, expected in expected_shapes.items():
        actual = np.asarray(result[key]).shape
        if actual != expected:
            raise AssertionError(f"{key} shape {actual} != {expected}")

    # Gamma is component 0. Component 1 is rainflow and must not acquire Gamma
    # states merely because the exported arrays are fleet-shaped.
    gamma_component = 0
    rainflow_component = 1
    gamma_shape = np.asarray(result["gamma_shape_bound"], dtype=float)
    gamma_tail = np.asarray(result["gamma_tail_bound"], dtype=float)
    if not np.any(gamma_shape[:, gamma_component, :] > 0.0):
        raise AssertionError("Gamma cells contain no bounding states")
    if np.any(np.abs(gamma_shape[:, rainflow_component, :]) > 1e-12):
        raise AssertionError("rainflow cells received Gamma bounding shapes")
    if np.any(np.abs(gamma_tail[:, rainflow_component, :]) > 1e-12):
        raise AssertionError("rainflow cells received Gamma tail probabilities")
    if np.max(gamma_tail[:, gamma_component, :]) > 0.1 + 1e-8:
        raise AssertionError("mixed Gamma reliability limit was violated")

    # Two vehicles times one Gamma component times five steps. Rainflow cells
    # are intentionally excluded from the exact Gamma replay.
    if not report["valid"]:
        raise AssertionError("exact mixed Gamma report is invalid")
    if report["gamma_cells"] != 2:
        raise AssertionError("validator did not select exactly the Gamma cells")
    if report["transitions_checked"] != 10:
        raise AssertionError("wrong number of mixed Gamma transitions")

    for key in ("backend", "degradation", "models", "reliability_impl"):
        if saved.get(key) != result.get(key):
            raise AssertionError(f"{key} changed during YAML serialization")

    print("PASS public mixed Gamma/rainflow validation")
    print("status             :", result["status"])
    print("objective          :", result["objective"])
    print("models             :", result["models"])
    print("horizon            :", [result["H1"], result["H2"], result["T"]])
    print("Gamma cells        :", report["gamma_cells"])
    print("Gamma transitions  :", report["transitions_checked"])
    print("Gamma replacements :", report["replacements"])
    print("Gamma repairs      :", report["repairs"])
    print("worst tail margin  :", report["minimum_conservativeness_margin"])
    print("reliability slack  :", report["minimum_reliability_slack"])


if __name__ == "__main__":
    main()
