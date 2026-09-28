"""Shared Euler preflight for the initial-damage horizon experiments."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import gurobipy
import numpy as np
import yaml

import fleet_management
from fleet_management.config import load_config


def main() -> None:
    path = Path(sys.argv[1])
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = load_config(raw)
    expected_mu0 = np.asarray([
        [0.10, 0.20], [0.15, 0.35], [0.35, 0.55], [0.45, 0.65]
    ])
    params = raw["gurobi_params"]
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", "32"))

    assert (cfg.F, cfg.M, cfg.L) == (4, 2, 2)
    assert cfg.models == ["gamma", "rainflow"]
    assert np.allclose(cfg.mu_0, expected_mu0)
    assert np.allclose(cfg.tau, 1.0)
    assert np.allclose(cfg.epsilon, 1e-4)
    assert raw["reliability_impl"] == "tangent"
    assert raw["allow_replacement"] is False
    assert float(raw["mip_gap"]) == 0.05
    assert int(raw["time_limit"]) == 1500
    assert raw["relaxation_warm_start"] is True
    assert int(raw["relaxation_time_limit"]) == 120
    assert int(params["Threads"]) == allocated == 32

    print(f"PASS {path}")
    print(f"F={cfg.F}, M={cfg.M}, L={cfg.L}")
    print("Initial means:", cfg.mu_0.tolist())
    print("Initial variances:", cfg.v_0.tolist())
    print("Slurm CPUs / Gurobi threads:", allocated, "/", params["Threads"])
    print("Gurobi:", gurobipy.gurobi.version())
    print("fleet_management:", fleet_management.__file__)


if __name__ == "__main__":
    main()
