"""Tests of step 3 (spec alignment) for the validation and sweep tools: M9, M10, M12, M14.

Inputs and results are the hand-written files of ``tests/data/tools/validation`` and
``tests/data/tools/sweeps``, copied and changed in ``tmp_path``. No Gurobi is needed.
"""
import copy
import math
from pathlib import Path

import pytest
import yaml

import fleet_management.horizon_sweep as hs
from fleet_management.config import load_config, read_input
from fleet_management.degradation_model.gamma_utils.gamma_replay_validator import (
    validate_gamma_replay_schedule,
)
from fleet_management.degradation_model.gamma_utils.gamma_stochastic_validator import (
    _SETTING_DEFAULTS,
    _settings,
    validate_gamma_stochastic_schedule,
)
from fleet_management.solver import read_result

DATA = Path(__file__).parent / "data" / "tools"
VALIDATION = DATA / "validation"
SWEEPS = DATA / "sweeps"


def load_yaml(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path, data):
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    return str(path)


def gamma_pair(tmp_path, input_changes=None, z_entry=None):
    """Configuration and result of the ardinf Gamma example; ``z_entry`` replaces z[0][0][1]."""
    inp = load_yaml(VALIDATION / "gamma_ardinf_input.yaml")
    res = load_yaml(VALIDATION / "gamma_ardinf_result.yaml")
    inp.update(input_changes or {})
    if z_entry is not None:
        res["z"][0][0][1] = z_entry
    in_path = write_yaml(tmp_path / "in.yaml", inp)
    res_path = write_yaml(tmp_path / "res.yaml", res)
    return load_config(read_input(in_path)), read_result(res_path)


# ---------------------------------------------------------------------------
# M9: replay validator, non-finite z at a Gamma cell
# ---------------------------------------------------------------------------
def test_replay_valid_baseline(tmp_path):
    cfg, res = gamma_pair(tmp_path)
    assert validate_gamma_replay_schedule(cfg, res)["valid"]


@pytest.mark.parametrize("bad", [math.inf, -math.inf])
def test_replay_infinite_z_raises(tmp_path, bad):
    cfg, res = gamma_pair(tmp_path, z_entry=bad)
    with pytest.raises(ValueError, match="non-finite"):
        validate_gamma_replay_schedule(cfg, res)


def test_replay_nan_z_is_skipped(tmp_path):
    cfg, res = gamma_pair(tmp_path, z_entry=math.nan)
    report = validate_gamma_replay_schedule(cfg, res)
    assert report["valid"]


# ---------------------------------------------------------------------------
# M10: random_seed must be a non-negative integer
# ---------------------------------------------------------------------------
def test_stochastic_negative_seed_raises(tmp_path):
    cfg, res = gamma_pair(tmp_path)
    with pytest.raises(ValueError, match="random_seed"):
        validate_gamma_stochastic_schedule(cfg, res, repetitions=10, random_seed=-1)


def test_stochastic_zero_seed_passes(tmp_path):
    cfg, res = gamma_pair(tmp_path)
    report = validate_gamma_stochastic_schedule(cfg, res, repetitions=10, random_seed=0)
    assert report["random_seed"] == 0


# ---------------------------------------------------------------------------
# M12: a null key of the block ``validation`` counts as absent
# ---------------------------------------------------------------------------
def test_dispatcher_null_block_keys_take_defaults(tmp_path):
    cfg, _ = gamma_pair(tmp_path, {"validation": {"mode": None, "repetitions": None}})
    assert _settings(cfg) == _SETTING_DEFAULTS


def test_dispatcher_block_value_and_null_mix(tmp_path):
    cfg, _ = gamma_pair(tmp_path, {"validation": {"mode": None, "repetitions": 50}})
    settings = _settings(cfg)
    assert settings["mode"] == "deterministic"
    assert settings["repetitions"] == 50


# ---------------------------------------------------------------------------
# M14: the operating sweep refuses H1 < 1 before the first case
# ---------------------------------------------------------------------------
class NeverCalled:
    def __init__(self):
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("solve_mapping must not be called")


@pytest.mark.parametrize("H", [[0, 3], 0])
def test_sweep_refuses_h1_below_one(tmp_path, monkeypatch, H):
    data = load_yaml(SWEEPS / "gaussian_small.yaml")
    data["H"] = H
    path = write_yaml(tmp_path / "in.yaml", copy.deepcopy(data))
    fake = NeverCalled()
    monkeypatch.setattr(hs, "solve_mapping", fake)
    with pytest.raises(ValueError, match="H1"):
        hs.sweep_operating_horizons(path, [1, 2])
    assert fake.calls == 0


def test_grid_refuses_h1_below_one(tmp_path, monkeypatch):
    path = write_yaml(tmp_path / "in.yaml", load_yaml(SWEEPS / "gaussian_small.yaml"))
    fake = NeverCalled()
    monkeypatch.setattr(hs, "solve_mapping", fake)
    with pytest.raises(ValueError):
        hs.sweep_horizon_grid(path, [0, 1], [1, 2])
    assert fake.calls == 0
