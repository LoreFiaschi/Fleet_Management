"""Regression for the operating-phase average-cost objective."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from fleet_management import solve


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "gamma_tail_bound_public.yaml"


def main() -> None:
    data = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    data["objective_mode"] = "operating_average"

    with TemporaryDirectory(prefix="operating-objective-") as directory:
        root = Path(directory)
        input_path = root / "input.yaml"
        result_path = root / "result.yaml"
        input_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        result = solve(str(input_path), str(result_path))
        saved = yaml.safe_load(result_path.read_text(encoding="utf-8"))

    if result["status"] != "optimal":
        raise AssertionError(f"phase-objective case is {result['status']!r}")
    if result["objective_mode"] != "operating_average":
        raise AssertionError("wrong objective mode")
    expected = result["J_op"] / result["H2"]
    if abs(result["objective"] - expected) > 1e-8:
        raise AssertionError("objective is not J_op/H2")
    if result.get("bound") is None or result.get("mip_gap") is None:
        raise AssertionError("operating objective has no bound/MIP-gap certificate")
    if result["mip_gap"] > 1e-8:
        raise AssertionError("optimal operating objective has a nonzero MIP gap")
    for key in ("objective_mode", "J_op", "J_op_average"):
        if key not in saved:
            raise AssertionError(f"saved result is missing {key}")
    if abs(saved["J_op_average"] - expected) > 1e-8:
        raise AssertionError("saved operating average is incorrect")

    print("PASS operating-phase average objective")
    print("objective mode   :", result["objective_mode"])
    print("J_op             :", result["J_op"])
    print("H2               :", result["H2"])
    print("J_op / H2        :", result["J_op_average"])
    print("objective bound  :", result["bound"])
    print("MIP gap          :", result["mip_gap"])


if __name__ == "__main__":
    main()
