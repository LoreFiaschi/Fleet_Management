"""Regression for the operating-phase average-cost objective."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
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
    expected_average = result["J_op"] / result["H2"]
    expected_objective = expected_average + result["damage_penalty"]
    if abs(result["objective"] - expected_objective) > 1e-8:
        raise AssertionError("objective is not J_op/H2 + C_D*u")
    if result.get("bound") is None or result.get("mip_gap") is None:
        raise AssertionError("operating objective has no bound/MIP-gap certificate")
    if result["mip_gap"] > 1e-8:
        raise AssertionError("optimal operating objective has a nonzero MIP gap")
    for key in (
        "objective_mode",
        "J_op",
        "J_op_average",
        "peak_damage",
        "damage_penalty",
        "operating_objective",
    ):
        if key not in saved:
            raise AssertionError(f"saved result is missing {key}")
    if abs(saved["J_op_average"] - expected_average) > 1e-8:
        raise AssertionError("saved operating average is incorrect")

    costs = result["component_costs"]
    m = np.asarray(result["m"], dtype=float)
    z = np.asarray(result["z"], dtype=float)
    r = np.asarray(result["r"], dtype=float)
    mu = np.asarray(result["mu"], dtype=float)
    u = float(result["u"])

    c_m = np.asarray(costs["C_M"], dtype=float)
    c_r = np.asarray(costs["C_R"], dtype=float)
    c_rep = np.asarray(costs["C_rep"], dtype=float)

    # Per-step accounting contains additive action costs only.
    manual_steps = np.asarray([
        np.sum(c_m[np.newaxis, :] * m[:, :, k])
        + np.sum(c_r[np.newaxis, :] * z[:, :, k])
        + np.sum(c_rep[np.newaxis, :] * r[:, :, k])
        for k in range(result["T"])
    ])

    if not np.allclose(
        manual_steps, result["step_costs"], atol=1e-8, rtol=0.0
    ):
        raise AssertionError("additive step-cost accounting is inconsistent")

    manual_peak = float(np.max(np.sum(mu, axis=1)))
    if abs(u - manual_peak) > 1e-8:
        raise AssertionError("saved u is not the maximum fleet damage")

    manual_penalty = float(costs["C_D"]) * manual_peak
    if abs(result["damage_penalty"] - manual_penalty) > 1e-8:
        raise AssertionError("damage penalty is not C_D*u")

    manual_init = float(np.sum(manual_steps[:result["H1"]]))
    manual_op = float(np.sum(manual_steps[result["H1"]:]))

    if abs(result["J_initialization"] - manual_init) > 1e-8:
        raise AssertionError("initialization cost was not recorded correctly")

    if abs(result["J_op"] - manual_op) > 1e-8:
        raise AssertionError("operating cost was not recorded correctly")

    print("PASS operating-phase average objective")
    print("objective mode   :", result["objective_mode"])
    print("J_op             :", result["J_op"])
    print("H2               :", result["H2"])
    print("J_op / H2        :", result["J_op_average"])
    print("objective bound  :", result["bound"])
    print("MIP gap          :", result["mip_gap"])


if __name__ == "__main__":
    main()
