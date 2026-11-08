"""Tests of the start of the repetitions (plan Y3) for the three tools that read the loop rows.

The structural validator (checks ``loop_q``), the Gamma replay validator (field
``repeatability``) and the dashboard loop table use the saved start ``y_q`` when the result
has it, and the value on day H1 otherwise. Inputs and results are the hand-written files of
``tests/data/tools/validation``, changed in ``tmp_path``. No Gurobi is needed.

Gaussian pair (F = 2, L = 1, H1 = 1, T = 3): ``mu`` of train 0 is 0.3 on day H1 and 0.25 on
day T; train 1 has 0.05 on day H1 and a replacement on day T. Gamma pair: ``mu`` of train 0
is 0.15 on day H1 and 0.06 on day T (anchor 0.04 and 0); train 1 has 0.07 and 0 (anchor 0.07
and 0).
"""
from pathlib import Path

import numpy as np
import pytest
import yaml

from fleet_management.config import load_config, read_input
from fleet_management.degradation_model.gamma_utils.gamma_replay_validator import (
    validate_gamma_replay_schedule,
)
from fleet_management.solver import read_result
from fleet_management.validation.dashboard_inspector import loop_table
from fleet_management.validation.validator import check_result, validate

VALIDATION = Path(__file__).parent / "data" / "tools" / "validation"
TOL = 1e-6


def load_yaml(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path, data):
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    return path


def pair(tmp_path, name, **changes):
    """Config and result of ``<name>``; ``changes`` overwrite keys of the result."""
    cfg = load_config(read_input(VALIDATION / f"{name}_input.yaml"))
    data = load_yaml(VALIDATION / f"{name}_result.yaml")
    data.update(changes)
    return cfg, read_result(write_yaml(tmp_path / f"{name}_result.yaml", data)), data


def checks_of(report):
    return {check["name"]: check for check in report["checks"]}


# ---------------------------------------------------------------------------
# Structural validator
# ---------------------------------------------------------------------------
# Train 1 of the Gaussian pair: mu is 0.05 on day H1 and 0.1 on day T (above day H1).
GAUSS_MU = [[[0.3, 0.15, 0.25]], [[0.05, 0.25, 0.1]]]


def test_validator_start_passes_above_day_h1(tmp_path):
    cfg, result, _ = pair(tmp_path, "gaussian", mu=GAUSS_MU, y_mu=[[0.3], [0.12]])
    check = checks_of(check_result(cfg, result, TOL))["loop_mu"]
    assert check["passed"] and check["violation"] == 0.0


def test_validator_start_fails_when_day_t_above_y(tmp_path):
    cfg, result, _ = pair(tmp_path, "gaussian", mu=GAUSS_MU, y_mu=[[0.3], [0.08]])
    report = check_result(cfg, result, TOL)
    check = checks_of(report)["loop_mu"]
    assert not check["passed"] and check["violation"] == pytest.approx(0.1 - 0.08)
    assert not report["passed"]


def test_validator_start_fails_when_day_h1_above_y(tmp_path):
    mu = [[[0.3, 0.15, 0.25]], [[0.05, 0.25, 0.0]]]
    cfg, result, _ = pair(tmp_path, "gaussian", mu=mu, y_mu=[[0.3], [0.04]])
    check = checks_of(check_result(cfg, result, TOL))["loop_mu"]
    assert not check["passed"] and check["violation"] == pytest.approx(0.05 - 0.04)


def test_validator_without_start_follows_the_old_rule(tmp_path):
    cfg, result, _ = pair(tmp_path, "gaussian", mu=GAUSS_MU)
    check = checks_of(check_result(cfg, result, TOL))["loop_mu"]
    assert not check["passed"] and check["violation"] == pytest.approx(0.1 - 0.05)
    cfg, result, _ = pair(tmp_path, "gaussian")
    assert check_result(cfg, result, TOL)["passed"]


def test_validator_start_skips_nan_and_keeps_check_names(tmp_path):
    names = [c["name"] for c in check_result(*pair(tmp_path, "gaussian")[:2], TOL)["checks"]]
    cfg, result, _ = pair(tmp_path, "gaussian", y_mu=[[0.3], [0.05]],
                          y_v=[[0.0014], [0.0001]])
    report = check_result(cfg, result, TOL)
    assert [c["name"] for c in report["checks"]] == names
    assert report["passed"]
    assert np.isnan(np.asarray(result["y_mu"], dtype=float)).sum() == 0
    # y_g_mu holds only NaN: no start, and g_mu is not saved: nothing to check
    assert checks_of(report)["loop_g_mu"]["violation"] == 0.0


def test_validate_files_use_the_start(tmp_path):
    data = load_yaml(VALIDATION / "gaussian_result.yaml")
    data["mu"] = GAUSS_MU
    data["y_mu"] = [[0.3], [0.12]]
    result_path = write_yaml(tmp_path / "result.yaml", data)
    report = validate(str(VALIDATION / "gaussian_input.yaml"), str(result_path),
                      str(tmp_path / "report.yaml"))
    assert report["passed"]
    data["y_mu"] = [[0.3], [0.08]]
    result_path = write_yaml(tmp_path / "result.yaml", data)
    report = validate(str(VALIDATION / "gaussian_input.yaml"), str(result_path),
                      str(tmp_path / "report.yaml"))
    assert not report["passed"]


# ---------------------------------------------------------------------------
# Gamma replay validator
# ---------------------------------------------------------------------------
# Train 1 of the Gamma pair: mu is 0.07 on day H1 and 0.1 on day T (a saved value is an upper
# bound of the replayed 0, so it is accepted).
GAMMA_MU = [[[0.15, 0.0, 0.06]], [[0.07, 0.12, 0.1]]]
GAMMA_Y_G = [[0.04], [0.07]]


def repeatability(report):
    return [v for v in report["violations"] if v["field"] == "repeatability"]


def test_replay_start_passes_above_day_h1(tmp_path):
    cfg, result, _ = pair(tmp_path, "gamma_ard1", mu=GAMMA_MU, y_mu=[[0.15], [0.12]],
                          y_g_mu=GAMMA_Y_G)
    report = validate_gamma_replay_schedule(cfg, result)
    assert report["valid"] and report["maximum_errors"]["repeatability"] == 0.0


def test_replay_start_fails_when_day_t_above_y(tmp_path):
    cfg, result, _ = pair(tmp_path, "gamma_ard1", mu=GAMMA_MU, y_mu=[[0.15], [0.09]],
                          y_g_mu=GAMMA_Y_G)
    report = validate_gamma_replay_schedule(cfg, result)
    (record,) = repeatability(report)
    assert not report["valid"]
    assert record["i"] == 1 and record["k"] == cfg.T - 1
    assert record["expected"] == pytest.approx(0.09) and record["saved"] == pytest.approx(0.1)
    assert record["error"] == pytest.approx(0.1 - 0.09)


def test_replay_start_fails_when_day_h1_above_y(tmp_path):
    cfg, result, _ = pair(tmp_path, "gamma_ard1", y_mu=[[0.14], [0.12]], y_g_mu=GAMMA_Y_G)
    report = validate_gamma_replay_schedule(cfg, result)
    (record,) = repeatability(report)
    assert record["i"] == 0 and record["expected"] == pytest.approx(0.14)
    assert record["error"] == pytest.approx(0.15 - 0.14)


def test_replay_start_checks_the_anchor(tmp_path):
    cfg, result, _ = pair(tmp_path, "gamma_ard1", y_mu=[[0.15], [0.12]],
                          y_g_mu=[[0.04], [0.06]])
    report = validate_gamma_replay_schedule(cfg, result)
    (record,) = repeatability(report)
    assert record["i"] == 1 and record["error"] == pytest.approx(0.07 - 0.06)


def test_replay_without_start_follows_the_old_rule(tmp_path):
    cfg, result, _ = pair(tmp_path, "gamma_ard1")
    assert validate_gamma_replay_schedule(cfg, result)["valid"]
    cfg, result, _ = pair(tmp_path, "gamma_ard1", mu=GAMMA_MU)
    report = validate_gamma_replay_schedule(cfg, result)
    (record,) = repeatability(report)
    assert record["expected"] == pytest.approx(0.07)
    assert record["error"] == pytest.approx(0.1 - 0.07)


# ---------------------------------------------------------------------------
# Dashboard loop table
# ---------------------------------------------------------------------------
def row_of(table, train, quantity):
    return table[(table["train"] == train) & (table["quantity"] == quantity)].iloc[0]


def test_loop_table_start_passes_above_day_h1(tmp_path):
    cfg, result, _ = pair(tmp_path, "gaussian", mu=GAUSS_MU, y_mu=[[0.3], [0.12]])
    table = loop_table(cfg, result, TOL)
    assert list(table.columns) == ["train", "component", "quantity", "value_at_H1", "y",
                                   "value_at_T", "margin_T", "margin_H1", "margin", "holds"]
    row = row_of(table, 2, "mu")
    assert row["value_at_H1"] == pytest.approx(0.05) and row["y"] == pytest.approx(0.12)
    assert row["value_at_T"] == pytest.approx(0.1)
    assert row["margin_T"] == pytest.approx(0.02) and row["margin_H1"] == pytest.approx(0.07)
    assert row["margin"] == pytest.approx(0.02) and bool(row["holds"])
    assert bool(table["holds"].all())


def test_loop_table_start_fails_when_day_t_above_y(tmp_path):
    cfg, result, _ = pair(tmp_path, "gaussian", mu=GAUSS_MU, y_mu=[[0.3], [0.08]])
    row = row_of(loop_table(cfg, result, TOL), 2, "mu")
    assert row["margin_T"] == pytest.approx(-0.02) and not bool(row["holds"])


def test_loop_table_start_fails_when_day_h1_above_y(tmp_path):
    mu = [[[0.3, 0.15, 0.25]], [[0.05, 0.25, 0.0]]]
    cfg, result, _ = pair(tmp_path, "gaussian", mu=mu, y_mu=[[0.3], [0.04]])
    row = row_of(loop_table(cfg, result, TOL), 2, "mu")
    assert row["margin_H1"] == pytest.approx(-0.01) and row["margin_T"] == pytest.approx(0.04)
    assert not bool(row["holds"])


def test_loop_table_without_start_follows_the_old_rule(tmp_path):
    cfg, result, _ = pair(tmp_path, "gaussian", mu=GAUSS_MU)
    table = loop_table(cfg, result, TOL)
    row = row_of(table, 2, "mu")
    assert np.isnan(row["y"]) and np.isnan(row["margin_H1"])
    assert row["margin_T"] == pytest.approx(0.05 - 0.1) and row["margin"] == row["margin_T"]
    assert not bool(row["holds"])
    assert bool(row_of(table, 1, "mu")["holds"])
