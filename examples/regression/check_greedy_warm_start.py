"""Small structural regression for the deterministic greedy MIP start."""

from __future__ import annotations

import numpy as np

from fleet_management.config import load_config
from fleet_management.greedy_warm_start import build_greedy_warm_start


def main() -> None:
    raw = {
        "F": 4, "M": 2, "L": 1, "H": [2, 3],
        "model": "gamma", "bound_method": "cantelli",
        "repair_model": "ardinf", "tau": 1.0, "epsilon": 0.1,
        "rho": 0.5, "mu_0": 0.0, "mu": [0.03, 0.02],
        "mu_trans": [0.03, 0.02], "gamma_beta": 20.0,
        "gamma_beta_trans": 20.0, "gamma_beta_bound": 10.0,
        "gamma_beta_0": 20.0, "gamma_beta_new": 20.0,
        "C_M": 1.0, "C_R": 0.5, "C_D": 2.0,
        "C_rep": 4.0, "allow_replacement": False,
    }
    cfg = load_config(raw)
    start, report = build_greedy_warm_start(cfg)
    x = np.asarray(start["x"])
    m = np.asarray(start["m"])
    idle = np.asarray(start["idle"])

    assert report.feasible_assignment
    assert report.artificial_assignments == 0
    assert np.allclose(x[:, 1:, :].sum(axis=0), 1.0)
    assert np.all(x.sum(axis=1) <= 1.0 + 1e-12)
    assert np.allclose(idle + m, x[:, 0, :][:, None, :])

    impossible = dict(raw)
    impossible.update({"F": 1, "M": 2})
    cfg_impossible = load_config(impossible)
    _, fallback = build_greedy_warm_start(cfg_impossible)
    assert not fallback.feasible_assignment
    assert fallback.artificial_assignments == cfg_impossible.T

    print("PASS greedy warm-start structure")
    print("real-case artificial assignments:", report.artificial_assignments)
    print("fallback artificial assignments :", fallback.artificial_assignments)


if __name__ == "__main__":
    main()
