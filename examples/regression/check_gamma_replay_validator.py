"""Regression for the lightweight public Gamma schedule/state replay."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from fleet_management import solve, validate_gamma_replay_files


HERE = Path(__file__).resolve().parent
INPUT = HERE / "mixed_gamma_ard1_public.yaml"
TOLERANCE = 1e-8


def main() -> None:
    with TemporaryDirectory(prefix="gamma-replay-") as directory:
        directory = Path(directory)
        result_path = directory / "result.yaml"
        report_path = directory / "replay.yaml"
        corrupt_path = directory / "corrupt.yaml"

        result = solve(str(INPUT), str(result_path))
        report = validate_gamma_replay_files(
            INPUT, result_path, report_path, raise_on_failure=True
        )

        if result["status"] != "optimal":
            raise AssertionError(f"expected optimal solve, got {result['status']!r}")
        if not report["valid"]:
            raise AssertionError("valid solver output failed Gamma replay")
        if report["gamma_cells"] != 2 or report["gamma_ard1_cells"] != 2:
            raise AssertionError("replay selected the wrong Gamma cells")
        if report["repairs"] < 1:
            raise AssertionError("ARD1 replay case contains no Gamma repair")
        if max(report["maximum_errors"].values()) > TOLERANCE:
            raise AssertionError(
                f"valid replay has excessive error {report['maximum_errors']}"
            )
        if not report_path.is_file():
            raise AssertionError("Gamma replay report was not written")

        corrupt = yaml.safe_load(result_path.read_text(encoding="utf-8"))
        corrupt["mu"][0][0][0] += 0.01
        corrupt_path.write_text(
            yaml.safe_dump(corrupt, sort_keys=False), encoding="utf-8"
        )
        rejected = validate_gamma_replay_files(INPUT, corrupt_path)
        if rejected["valid"]:
            raise AssertionError("Gamma replay accepted a corrupted mean state")
        if rejected["maximum_errors"]["physical_mean"] < 0.009:
            raise AssertionError("corrupted mean was not diagnosed")

        try:
            validate_gamma_replay_files(
                INPUT, corrupt_path, raise_on_failure=True
            )
        except AssertionError:
            pass
        else:
            raise AssertionError("raise_on_failure did not reject corruption")

    print("PASS lightweight Gamma schedule/state replay")
    print("Gamma cells          :", report["gamma_cells"])
    print("transitions checked  :", report["transitions_checked"])
    print("repairs              :", report["repairs"])
    print("replacements         :", report["replacements"])
    print("maximum replay errors:", report["maximum_errors"])
    print("corruption detected  :", not rejected["valid"])


if __name__ == "__main__":
    main()