"""Verify that Gurobi accepts the repeatability-aware greedy MIP start."""

from __future__ import annotations

import gc
import tempfile
from pathlib import Path

import gurobipy as gp
import yaml

from fleet_management.config import load_config
from fleet_management.greedy_warm_start import build_greedy_warm_start
from fleet_management.solver import solve


def main() -> None:
    raw = {
        "F": 4,
        "M": 2,
        "L": 1,
        "H": [4, 12],
        "model": "gamma",
        "bound_method": "cantelli",
        "repair_model": "ardinf",
        "tau": 1.0,
        "epsilon": 0.1,
        "rho": 0.5,
        "mu_0": 0.0,
        "mu": [0.03, 0.02],
        "mu_trans": [0.03, 0.02],
        "gamma_beta": 20.0,
        "gamma_beta_trans": 20.0,
        "gamma_beta_bound": 10.0,
        "gamma_beta_0": 20.0,
        "gamma_beta_new": 20.0,
        "gamma_calibration_method": "repeated_increment",
        "C_M": 1.0,
        "C_R": 0.5,
        "C_D": 2.0,
        "C_rep": 4.0,
        "allow_replacement": False,
        "objective_mode": "operating_average",
        "mip_gap": 0.20,
        "time_limit": 10,
        "verbose": 1,
        "gurobi_params": {"Threads": 4, "Seed": 1},
    }
    cfg = load_config(raw)
    start, report = build_greedy_warm_start(cfg)
    assert report.feasible_assignment
    assert report.repeatability_feasible
    assert report.repairs > 0

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        root = Path(directory)
        log_path = root / "gurobi.log"
        raw["gurobi_params"]["LogFile"] = str(log_path)
        input_path = root / "input.yaml"
        result_path = root / "result.yaml"
        input_path.write_text(
            yaml.safe_dump(raw, sort_keys=False), encoding="utf-8"
        )
        result = solve(str(input_path), str(result_path), warm_start=start)
        log = log_path.read_text(encoding="utf-8", errors="replace")
        # On Windows the default Gurobi environment can retain the LogFile
        # handle beyond solve(). Release model cycles and the environment before
        # TemporaryDirectory attempts to remove the file.
        gc.collect()
        gp.disposeDefaultEnv()

    accepted_messages = [
        line.strip()
        for line in log.splitlines()
        if "MIP start" in line
    ]
    rejected = any(
        "did not produce a new incumbent" in line
        for line in accepted_messages
    )
    accepted = any(
        "Loaded user MIP start" in line
        or "User MIP start produced solution" in line
        for line in accepted_messages
    )
    assert result.get("warm_start", {}).get("applied") is True
    completion = result["warm_start"].get("completion", {})
    assert completion.get("status") == "completed", completion
    assert accepted and not rejected, accepted_messages

    print("PASS repeatability-aware greedy MIP start accepted")
    print("repairs                  :", report.repairs)
    print("repeatability failures   :", report.repeatability_failures)
    print("maximum terminal excess  :", report.maximum_repeatability_excess)
    print("solver status             :", result.get("status"))
    print("solver objective          :", result.get("objective"))
    print("start completion          :", completion)
    print("Gurobi MIP-start messages :", accepted_messages)


if __name__ == "__main__":
    main()
