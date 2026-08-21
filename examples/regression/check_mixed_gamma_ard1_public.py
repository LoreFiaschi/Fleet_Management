"""Public mixed Gamma/rainflow ARD1 solve, serialization, and validation."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml

from fleet_management import solve, validate_gamma_tail_bound_files


HERE = Path(__file__).resolve().parent
INPUT = HERE / "mixed_gamma_ard1_public.yaml"


def main() -> None:
    with TemporaryDirectory(prefix="mixed-gamma-ard1-") as directory:
        result_path = Path(directory) / "result.yaml"
        validation_path = Path(directory) / "validation.yaml"
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
            raise AssertionError("public ARD1 validation report was not written")

    if result["status"] != "optimal":
        raise AssertionError(f"expected optimal solve, got {result['status']!r}")
    if result["backend"] != "modular" or result["degradation"] != "mixed":
        raise AssertionError("public mixed ARD1 case used the wrong solver route")
    if result["models"] != ["gamma", "rainflow"]:
        raise AssertionError(f"unexpected model list {result['models']!r}")
    if result["repair_model"] != "ard1":
        raise AssertionError("ARD1 selector was not preserved in the result")
    if (result["H1"], result["H2"], result["T"]) != (2, 3, 5):
        raise AssertionError("unequal ARD1 horizon was not preserved")

    expected_shape = (2, 2, 5)
    latch = np.asarray(result["gamma_mean_latch"], dtype=float)
    if latch.shape != expected_shape:
        raise AssertionError(
            f"Gamma ARD1 latch shape {latch.shape} != {expected_shape}"
        )
    if np.any(np.abs(latch[:, 1, :]) > 1e-12):
        raise AssertionError("rainflow cells received Gamma latch values")
    if "gamma_mean_latch" not in saved:
        raise AssertionError("Gamma ARD1 latch was omitted from saved YAML")
    if not np.allclose(saved["gamma_mean_latch"], latch, atol=1e-12, rtol=0.0):
        raise AssertionError("Gamma ARD1 latch changed during serialization")

    gamma_repairs = int(np.rint(np.asarray(result["m"])[:, 0, :]).sum())
    gamma_replacements = int(np.rint(np.asarray(result["r"])[:, 0, :]).sum())
    if gamma_repairs < 1:
        raise AssertionError("public optimum contains no Gamma ARD1 repair")
    if gamma_replacements < 1:
        raise AssertionError("public optimum contains no Gamma replacement")

    if not report["valid"]:
        raise AssertionError("exact public Gamma ARD1 validation failed")
    if report["gamma_cells"] != 2 or report["gamma_ard1_cells"] != 2:
        raise AssertionError("validator selected the wrong Gamma ARD1 cells")
    if report["repairs"] != gamma_repairs:
        raise AssertionError("validator repair count differs from solver result")
    if report["replacements"] != gamma_replacements:
        raise AssertionError("validator replacement count differs from solver result")
    if report["maximum_latch_error"] > 1e-8:
        raise AssertionError("exact validator did not reproduce the ARD1 latch")
    if report["minimum_conservativeness_margin"] < -1e-8:
        raise AssertionError("Gamma ARD1 reliability envelope is non-conservative")
    if report["minimum_reliability_slack"] < -1e-8:
        raise AssertionError("Gamma ARD1 reliability limit was violated")

    repaired_steps = [
        step for step in report["steps"] if step["event"] == "repair"
    ]
    if not repaired_steps or not all(
        step["repair_model"] == "ard1" for step in repaired_steps
    ):
        raise AssertionError("exact replay did not identify ARD1 repair events")
    if not any(step["frozen_term_count"] > 0 for step in repaired_steps):
        raise AssertionError("ARD1 repair did not freeze any exact Gamma history")

    calibration = result["gamma_calibration"]
    if not calibration or not all(
        cell["repair_bound"] == "ard1_no_tail_credit" for cell in calibration
    ):
        raise AssertionError("Gamma ARD1 repair-bound metadata is incorrect")

    formulation = result["gamma_formulation"]
    if formulation["gamma_cells"] != 2 or formulation["gamma_ard1_cells"] != 2:
        raise AssertionError("ARD1 formulation estimate has wrong cell counts")
    if formulation["known_subtotal"] != {
        "variables": 145,
        "linear_constraints": 81,
        "general_constraints": 120,
        "quadratic_constraints": 0,
    }:
        raise AssertionError("public mixed ARD1 formulation baseline changed")

    print("PASS public mixed Gamma/rainflow ARD1 validation")
    print("status              :", result["status"])
    print("objective           :", result["objective"])
    print("horizon             :", [result["H1"], result["H2"], result["T"]])
    print("Gamma ARD1 cells    :", report["gamma_ard1_cells"])
    print("Gamma repairs       :", gamma_repairs)
    print("Gamma replacements  :", gamma_replacements)
    print("maximum latch error :", report["maximum_latch_error"])
    print("worst tail margin   :", report["minimum_conservativeness_margin"])
    print("reliability slack   :", report["minimum_reliability_slack"])
    print("known Gamma subtotal:", formulation["known_subtotal"])


if __name__ == "__main__":
    main()
