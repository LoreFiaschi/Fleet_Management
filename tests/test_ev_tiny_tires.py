from pathlib import Path

import numpy as np
import yaml

from fleet_management.ev_solver import solve_ev


def test_ev_tiny_tires_solves(tmp_path):
    output_path = tmp_path / "output_ev_tiny_tires.yaml"

    result = solve_ev(
        input_path="input/ev_tiny_tires.yaml",
        results_path=str(output_path),
    )

    assert result["status"] == "optimal"
    assert output_path.exists()

    with open(output_path, "r") as f:
        data = yaml.safe_load(f)
    
    x = np.asarray(data["x"], dtype=float)
    D = np.asarray(data["D"], dtype=float)

    F = data["F"]
    M = data["M"]
    H = data["H"]
    L = data["L"]

    assert x.shape == (F, M + 1, 2 * H)
    assert D.shape == (F, L, 2 * H)

    # Each mission is served exactly once.
    for k in range(2 * H):
        for j in range(1, M + 1):
            assert abs(np.sum(x[:, j, k]) - 1.0) <= 1e-6
    
    # Damage below threshold
    assert np.max(D) <= data["alpha"] + 1e-6
    