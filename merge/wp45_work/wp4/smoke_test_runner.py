import warnings

import numpy as np
import pytest

from fleet_management.config import load_config


def _base():
    return {"F": 2, "M": 1, "H": 2, "model": "rainflow", "tau": 1.0, "epsilon": 0.1,
            "rho": 0.5, "mu_0": 0.0, "mu": 0.1, "v": 0.001, "v_0": 0.0,
            "C_M": 1.0, "C_R": 1.0, "C_rep": 1.0, "C_D": 1.0}


def test_loads():
    cfg = load_config(_base())
    assert cfg.mu.shape == (2, 1, 1, 2)


def test_raises_on_bad_tau():
    data = _base()
    data["tau"] = -1.0
    with pytest.raises(ValueError, match="tau"):
        load_config(data)


def test_warns_stub():
    with pytest.warns(UserWarning, match="hello"):
        warnings.warn("hello there", UserWarning)


def test_tmp_path(tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    assert (tmp_path / "a.txt").exists()


def test_expected_failure_is_reported():
    with pytest.raises(ValueError):
        pass
