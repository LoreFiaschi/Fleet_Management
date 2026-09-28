"""Regression for the matched ARD-infinity/ARD1 numerical inputs."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from fleet_management import solve, validate_gamma_schedule_files


HERE = Path(__file__).resolve().parent
CASES = {
    "ardinf": HERE / "matched_mixed_ardinf.yaml",
    "ard1": HERE / "matched_mixed_ard1.yaml",
}


def _read(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return data


def main() -> None:
    inputs = {name: _read(path) for name, path in CASES.items()}
    expected_repairs = {
        "ardinf": ["ardinf", "ardinf"],
        "ard1": ["ard1", "ard1"],
    }
    for name, expected in expected_repairs.items():
        if inputs[name].get("repair_model") != expected:
            raise AssertionError(f"{name} has repair_model={inputs[name].get('repair_model')}")

    compared = {}
    for name, data in inputs.items():
        compared[name] = dict(data)
        compared[name].pop("repair_model")
    if compared["ardinf"] != compared["ard1"]:
        raise AssertionError("matched inputs differ outside repair_model")

    summaries = {}
    with TemporaryDirectory(prefix="matched-ard-") as directory:
        directory = Path(directory)
        for name, input_path in CASES.items():
            result_path = directory / f"{name}_result.yaml"
            report_path = directory / f"{name}_validation.yaml"
            result = solve(str(input_path), str(result_path))
            report = validate_gamma_schedule_files(
                input_path,
                result_path,
                report_path,
                mode="both",
                repetitions=10_000,
                batch_size=5_000,
                raise_on_failure=True,
            )
            if result["status"] != "optimal":
                raise AssertionError(f"{name} status is {result['status']!r}")
            if not report["deterministic"]["valid"]:
                raise AssertionError(f"{name} deterministic replay failed")
            summaries[name] = {
                "objective": result["objective"],
                "continuous": result["performance"]["continuous_variables"],
                "integer": result["performance"]["integer_variables"],
                "linear": result["performance"]["linear_constraints"],
                "general": result["performance"]["general_constraints"],
                "gamma_failures": report["stochastic"]["schedule"][
                    "failed_replays"
                ],
            }

    print("PASS matched ARD-infinity/ARD1 numerical inputs")
    for name, summary in summaries.items():
        print(f"{name:>6}: {summary}")


if __name__ == "__main__":
    main()
