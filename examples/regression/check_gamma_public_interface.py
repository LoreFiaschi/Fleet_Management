"""End-to-end regression for public modular Gamma tail-bound routing.

This intentionally uses ``fleet_management.solve`` and a YAML file rather than
calling ``base.solve_mixed`` directly.  On the pre-routing implementation it
must fail because every uniform Gamma fleet is sent to the constant-rate legacy
backend, which cannot accept the varying exact rates in this fixture.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml

from fleet_management import solve


HERE = Path(__file__).resolve().parent
INPUT = HERE / "gamma_tail_bound_public.yaml"


def require_keys(mapping: dict, keys: tuple[str, ...], where: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise AssertionError(f"{where} is missing keys: {missing}")


def main() -> None:
    with TemporaryDirectory(prefix="gamma-public-") as directory:
        output_path = Path(directory) / "result.yaml"
        result = solve(str(INPUT), str(output_path))
        saved = yaml.safe_load(output_path.read_text(encoding="utf-8"))

    require_keys(
        result,
        (
            "status",
            "objective",
            "backend",
            "degradation",
            "models",
            "reliability_impl",
            "H1",
            "H2",
            "T",
            "x",
            "mu",
            "gamma_shape_bound",
            "gamma_tail_bound",
            "gamma_beta_bound",
            "gamma_calibration",
        ),
        "in-memory result",
    )
    require_keys(
        saved,
        (
            "status",
            "objective",
            "backend",
            "degradation",
            "models",
            "reliability_impl",
            "H1",
            "H2",
            "T",
            "x",
            "mu",
            "gamma_shape_bound",
            "gamma_tail_bound",
            "gamma_beta_bound",
            "gamma_calibration",
        ),
        "saved YAML result",
    )

    if result["status"] != "optimal":
        raise AssertionError(f"expected optimal solve, got {result['status']!r}")
    if result["backend"] != "modular":
        raise AssertionError(f"expected modular backend, got {result['backend']!r}")
    if result["degradation"] != "gamma" or result["models"] != ["gamma"]:
        raise AssertionError("uniform Gamma model identity was not preserved")
    if result["reliability_impl"] != "gamma_finite_tail":
        raise AssertionError(
            f"wrong reliability implementation {result['reliability_impl']!r}"
        )
    if (result["H1"], result["H2"], result["T"]) != (2, 3, 5):
        raise AssertionError("unequal public-interface horizon was not preserved")

    expected_shapes = {
        "x": (2, 2, 5),
        "mu": (2, 1, 5),
        "gamma_shape_bound": (2, 1, 5),
        "gamma_tail_bound": (2, 1, 5),
        "gamma_beta_bound": (2, 1),
    }
    for key, expected in expected_shapes.items():
        actual = np.asarray(result[key]).shape
        if actual != expected:
            raise AssertionError(f"{key} shape {actual} != {expected}")

    tail = np.asarray(result["gamma_tail_bound"], dtype=float)
    if np.max(tail) > 0.1 + 1e-8:
        raise AssertionError("Gamma reliability limit was violated")
    if len(result["gamma_calibration"]) != 2:
        raise AssertionError("expected one Gamma calibration summary per cell")
    if any(item["tail_constraints"] <= 0 for item in result["gamma_calibration"]):
        raise AssertionError("Gamma calibration generated no tail constraints")
    if any(
        item["worst_calibration_margin"] < -1e-10
        for item in result["gamma_calibration"]
    ):
        raise AssertionError("Gamma calibration is not conservative")

    # The saved result must contain the same public identity and numerical data.
    if saved["backend"] != result["backend"]:
        raise AssertionError("backend identity was lost during YAML serialization")
    if saved["models"] != result["models"]:
        raise AssertionError("model identity was lost during YAML serialization")
    if not np.allclose(saved["gamma_tail_bound"], result["gamma_tail_bound"]):
        raise AssertionError("Gamma tail states changed during YAML serialization")

    print("PASS public modular Gamma tail-bound routing")
    print("status       :", result["status"])
    print("objective    :", result["objective"])
    print("backend      :", result["backend"])
    print("horizon      :", [result["H1"], result["H2"], result["T"]])
    print("common rates :", np.asarray(result["gamma_beta_bound"]).ravel())
    print("max tail     :", float(np.max(tail)))


if __name__ == "__main__":
    main()
