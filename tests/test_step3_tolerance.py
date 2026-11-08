"""Tests of the tolerances of the validators (step 3, item M19).

The tolerance of a check is resolved in this order: the explicit argument, the key
``tolerance`` of the input block ``validation``, then a default per check from the Gurobi
tolerances ``FeasibilityTol`` and ``IntFeasTol`` (``validation/tolerance.py``). Inputs and
results are the hand-built pairs of ``tests/data/tools/validation/``: the Gaussian pair for
the structural validator (tau = 1, so the natural range of ``mu`` is 1) and the Gamma ARDinf
pair for the Gamma validators. None of the tests needs Gurobi.
"""

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from fleet_management.config import load_config, read_input
from fleet_management.degradation_model.gamma_utils.gamma_replay_validator import (
    validate_gamma_replay_schedule,
)
from fleet_management.degradation_model.gamma_utils.gamma_stochastic_validator import (
    validate_gamma_schedule,
    validate_gamma_stochastic_schedule,
)
from fleet_management.degradation_model.gamma_utils.gamma_tail_validator import (
    validate_gamma_tail_bound_schedule,
)
from fleet_management.solver import read_result
from fleet_management.validation.tolerance import (
    gurobi_tolerances,
    resolve_tolerance,
    state_tolerance,
)
from fleet_management.validation.validator import check_result, validate

DATA = Path(__file__).parent / "data" / "tools" / "validation"
DEFAULTS = {"FeasibilityTol": 1e-6, "IntFeasTol": 1e-5}
STATE_TOL = 1e-6 + 1e-5 * 1.0          # FeasibilityTol + IntFeasTol * range_mu, with tau = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_yaml(name):
    with open(DATA / name, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def pair(tmp_path, input_name, result_name, **input_changes):
    """The configuration and the result of a pair, with some input keys changed."""
    data = load_yaml(input_name)
    data.update(input_changes)
    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    result = tmp_path / "result.yaml"
    result.write_text(yaml.safe_dump(load_yaml(result_name), sort_keys=False), encoding="utf-8")
    return load_config(read_input(path)), read_result(result), path, result


def records(report):
    return {c["name"]: c for c in report["checks"]}


# ---------------------------------------------------------------------------
# The helper module
# ---------------------------------------------------------------------------
def test_gurobi_tolerances_defaults_and_case_insensitive_names(tmp_path):
    cfg, _, _, _ = pair(tmp_path, "gaussian_input.yaml", "gaussian_result.yaml")
    assert gurobi_tolerances(cfg) == (1e-6, 1e-5)
    cfg, _, _, _ = pair(tmp_path, "gaussian_input.yaml", "gaussian_result.yaml",
                        gurobi_params={"feasibilitytol": 1e-7, "INTFEASTOL": 1e-6})
    assert gurobi_tolerances(cfg) == (1e-7, 1e-6)
    cfg, _, _, _ = pair(tmp_path, "gaussian_input.yaml", "gaussian_result.yaml",
                        gurobi_params={"Threads": 2})        # other parameters are ignored
    assert gurobi_tolerances(cfg) == (1e-6, 1e-5)


def test_resolution_order(tmp_path):
    cfg, _, _, _ = pair(tmp_path, "gaussian_input.yaml", "gaussian_result.yaml")
    assert resolve_tolerance(cfg, None) is None                 # step 3: a default per check
    assert resolve_tolerance(cfg, 0.5) == 0.5                   # step 1
    cfg, _, _, _ = pair(tmp_path, "gaussian_input.yaml", "gaussian_result.yaml",
                        validation={"tolerance": 0.2})
    assert resolve_tolerance(cfg, None) == 0.2                  # step 2
    assert resolve_tolerance(cfg, 0.5) == 0.5                   # the argument comes first
    cfg, _, _, _ = pair(tmp_path, "gaussian_input.yaml", "gaussian_result.yaml",
                        validation={"tolerance": None})
    assert resolve_tolerance(cfg, None) is None                 # a null counts as absent


def test_state_tolerance_uses_the_range_of_each_cell(tmp_path):
    cfg, _, _, _ = pair(tmp_path, "gaussian_input.yaml", "gaussian_result.yaml",
                        tau=[[1.0], [2.0]])
    table = state_tolerance(cfg, "mu")
    assert table.shape == (2, 1)
    assert table[0, 0] == pytest.approx(1e-6 + 1e-5 * 1.0, rel=1e-12)
    assert state_tolerance(cfg, "g_mu", None, 1, 0) == pytest.approx(1e-6 + 1e-5 * 2.0, rel=1e-12)
    assert np.all(state_tolerance(cfg, "mu", 0.3) == 0.3)


@pytest.mark.parametrize("value", [-1, 0, float("nan"), float("inf"), "x", True])
def test_input_block_tolerance_must_be_a_finite_number_above_zero(tmp_path, value):
    with pytest.raises(ValueError, match="validation.tolerance"):
        pair(tmp_path, "gaussian_input.yaml", "gaussian_result.yaml",
             validation={"tolerance": value})


# ---------------------------------------------------------------------------
# Structural validator
# ---------------------------------------------------------------------------
def test_default_report_tolerance_and_per_check_values(tmp_path):
    cfg, result, _, _ = pair(tmp_path, "gaussian_input.yaml", "gaussian_result.yaml")
    report = check_result(cfg, result)
    assert report["passed"] is True
    assert report["tolerance"] == DEFAULTS
    checks = records(report)
    for name in ("binary", "one_activity", "mission_coverage", "depot_gating", "idle"):
        assert checks[name]["tolerance"] == 1e-5               # IntFeasTol
    assert checks["objective"]["tolerance"] == 1e-6            # FeasibilityTol
    for name in ("loop_mu", "loop_v", "loop_g_mu"):
        assert checks[name]["tolerance"] >= 1e-6               # FeasibilityTol + IntFeasTol * M_s
    assert checks["loop_mu"]["tolerance"] == pytest.approx(STATE_TOL, rel=1e-12)


def test_gurobi_params_change_the_defaults(tmp_path):
    cfg, result, _, _ = pair(tmp_path, "gaussian_input.yaml", "gaussian_result.yaml",
                             gurobi_params={"FeasibilityTol": 1e-7, "IntFeasTol": 1e-6})
    report = check_result(cfg, result)
    assert report["tolerance"] == {"FeasibilityTol": 1e-7, "IntFeasTol": 1e-6}
    checks = records(report)
    assert checks["binary"]["tolerance"] == 1e-6
    assert checks["objective"]["tolerance"] == 1e-7
    assert checks["loop_mu"]["tolerance"] == pytest.approx(1e-7 + 1e-6 * 1.0, rel=1e-12)
    result["m"][0][0][0] = 5e-6                                 # passes at 1e-5, fails at 1e-6
    assert records(check_result(cfg, result))["binary"]["passed"] is False


@pytest.mark.parametrize("factor, passed", [(0.5, True), (2.0, False)])
def test_state_gap_against_the_default(tmp_path, factor, passed):
    cfg, result, _, _ = pair(tmp_path, "gaussian_input.yaml", "gaussian_result.yaml")
    gap = factor * (1e-6 + 1e-5 * 1.0)                          # tau = 1
    result["mu"][0][0][2] = result["mu"][0][0][0] + gap         # mu_T above mu_H1 by the gap
    report = check_result(cfg, result)
    check = records(report)["loop_mu"]
    assert check["passed"] is passed
    assert check["violation"] == pytest.approx(gap, rel=1e-6)
    assert check["tolerance"] == pytest.approx(STATE_TOL, rel=1e-12)
    assert report["passed"] is passed


def test_state_gap_is_judged_at_its_own_cell(tmp_path):
    """tau = 1 and 2: a gap of 1.5e-5 passes at the cell of tau = 2 and fails at that of tau = 1."""
    cfg, result, _, _ = pair(tmp_path, "gaussian_input.yaml", "gaussian_result.yaml",
                             tau=[[1.0], [2.0]])
    result["mu"][1][0][2] = result["mu"][1][0][0] + 1.5e-5
    check = records(check_result(cfg, result))["loop_mu"]
    assert check["passed"] is True
    assert check["tolerance"] == pytest.approx(1e-6 + 2e-5, rel=1e-12)    # at the cell of the gap
    result["mu"][0][0][2] = result["mu"][0][0][0] + 1.5e-5
    check = records(check_result(cfg, result))["loop_mu"]
    assert check["passed"] is False


@pytest.mark.parametrize("gap, passed", [(5e-6, True), (2e-5, False)])
def test_binary_gap_against_the_default(tmp_path, gap, passed):
    cfg, result, _, _ = pair(tmp_path, "gaussian_input.yaml", "gaussian_result.yaml")
    result["m"][0][0][0] = gap                                  # a 0 that is off by the gap
    check = records(check_result(cfg, result))["binary"]
    assert check["passed"] is passed
    assert check["violation"] == pytest.approx(gap, rel=1e-9)
    assert check["tolerance"] == 1e-5


def test_explicit_argument_and_block_give_one_value_for_every_check(tmp_path):
    cfg, result, _, _ = pair(tmp_path, "gaussian_input.yaml", "gaussian_result.yaml",
                             validation={"tolerance": 0.2})
    report = check_result(cfg, result)
    assert report["tolerance"] == 0.2
    assert {c["tolerance"] for c in report["checks"]} == {0.2}
    report = check_result(cfg, result, tol=0.5)                 # the argument comes first
    assert report["tolerance"] == 0.5
    assert {c["tolerance"] for c in report["checks"]} == {0.5}


def test_validate_reads_the_block_and_the_defaults(tmp_path):
    _, _, path, result = pair(tmp_path, "gaussian_input.yaml", "gaussian_result.yaml",
                              validation={"tolerance": 0.2})
    assert validate(path, result, tmp_path / "a.yaml")["tolerance"] == 0.2
    assert validate(path, result, tmp_path / "b.yaml", tol=0.4)["tolerance"] == 0.4
    _, _, path, result = pair(tmp_path, "gaussian_input.yaml", "gaussian_result.yaml")
    report = validate(path, result, tmp_path / "c.yaml")
    assert report["tolerance"] == DEFAULTS
    assert yaml.safe_load((tmp_path / "c.yaml").read_text(encoding="utf-8")) == report


# ---------------------------------------------------------------------------
# Gamma validators
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("factor, valid", [(0.5, True), (2.0, False)])
def test_replay_state_gap_against_the_default(tmp_path, factor, valid):
    cfg, result, _, _ = pair(tmp_path, "gamma_ardinf_input.yaml", "gamma_ardinf_result.yaml")
    gap = factor * STATE_TOL
    result["mu"][0][0][0] = 0.15 - gap                          # the replayed mean is 0.15
    report = validate_gamma_replay_schedule(cfg, result)
    assert report["valid"] is valid
    assert report["maximum_errors"]["tracked_mean"] == pytest.approx(gap, rel=1e-6)
    assert report["maximum_error_tolerances"]["tracked_mean"] == pytest.approx(STATE_TOL, rel=1e-12)
    assert report["tolerance"] == DEFAULTS


def test_replay_gurobi_params_change_the_defaults(tmp_path):
    cfg, result, _, _ = pair(tmp_path, "gamma_ardinf_input.yaml", "gamma_ardinf_result.yaml",
                             gurobi_params={"FeasibilityTol": 1e-7, "IntFeasTol": 1e-6})
    result["mu"][0][0][0] = 0.15 - 0.5 * STATE_TOL              # passes at 1.1e-5, fails at 1.1e-6
    report = validate_gamma_replay_schedule(cfg, result)
    assert report["valid"] is False
    assert report["tolerance"] == {"FeasibilityTol": 1e-7, "IntFeasTol": 1e-6}
    assert report["maximum_error_tolerances"]["tracked_mean"] == pytest.approx(1.1e-6, rel=1e-12)


def test_replay_explicit_argument_and_block(tmp_path):
    cfg, result, _, _ = pair(tmp_path, "gamma_ardinf_input.yaml", "gamma_ardinf_result.yaml",
                             validation={"tolerance": 0.02})
    result["mu"][0][0][0] = 0.15 - 0.01
    report = validate_gamma_replay_schedule(cfg, result)        # the block: 0.02
    assert report["valid"] is True and report["tolerance"] == 0.02
    report = validate_gamma_replay_schedule(cfg, result, tolerance=0.005)
    assert report["valid"] is False and report["tolerance"] == 0.005


@pytest.mark.parametrize("function", [validate_gamma_replay_schedule,
                                      validate_gamma_tail_bound_schedule])
@pytest.mark.parametrize("gap, passed", [(5e-6, True), (2e-5, False)])
def test_gamma_binary_gap_against_the_default(tmp_path, function, gap, passed):
    cfg, result, _, _ = pair(tmp_path, "gamma_ardinf_input.yaml", "gamma_ardinf_result.yaml")
    result["m"][0][0][0] = gap
    if passed:
        assert function(cfg, result)["valid"] is True
    else:
        with pytest.raises(ValueError, match="not binary"):
            function(cfg, result)


def test_stochastic_binary_gap_and_report_tolerance(tmp_path):
    cfg, result, _, _ = pair(tmp_path, "gamma_ardinf_input.yaml", "gamma_ardinf_result.yaml")
    result["m"][0][0][0] = 5e-6
    report = validate_gamma_stochastic_schedule(cfg, result, repetitions=100)
    assert report["tolerance"] == DEFAULTS
    result["m"][0][0][0] = 2e-5
    with pytest.raises(ValueError, match="not binary"):
        validate_gamma_stochastic_schedule(cfg, result, repetitions=100)
    with pytest.raises(ValueError, match="not binary"):          # 1e-5 is below 2e-5
        validate_gamma_stochastic_schedule(cfg, result, repetitions=100, tolerance=1e-5)
    assert validate_gamma_stochastic_schedule(
        cfg, result, repetitions=100, tolerance=1e-4)["tolerance"] == 1e-4


def test_exact_validator_per_check_tolerances(tmp_path):
    cfg, result, _, _ = pair(tmp_path, "gamma_ardinf_input.yaml", "gamma_ardinf_result.yaml")
    report = validate_gamma_tail_bound_schedule(cfg, result)
    assert report["tolerance"] == DEFAULTS
    tolerances = report["maximum_error_tolerances"]
    assert tolerances["reliability"] == 1e-6                    # a probability: FeasibilityTol
    assert tolerances["tracked_mean"] == pytest.approx(STATE_TOL, rel=1e-12)
    report = validate_gamma_tail_bound_schedule(cfg, result, tolerance=0.01)
    assert report["tolerance"] == 0.01
    assert set(report["maximum_error_tolerances"].values()) == {0.01}


def test_dispatcher_reads_the_block_tolerance(tmp_path):
    cfg, result, _, _ = pair(tmp_path, "gamma_ardinf_input.yaml", "gamma_ardinf_result.yaml",
                             validation={"tolerance": 0.3})
    report = validate_gamma_schedule(cfg, result)               # mode: deterministic
    assert report["tolerance"] == 0.3
    report = validate_gamma_schedule(cfg, result, tolerance=0.1)    # the argument comes first
    assert report["tolerance"] == 0.1
    report = validate_gamma_schedule(cfg, result, mode="both", repetitions=100)
    assert report["deterministic"]["tolerance"] == 0.3
    assert report["stochastic"]["tolerance"] == 0.3
    cfg, result, _, _ = pair(tmp_path, "gamma_ardinf_input.yaml", "gamma_ardinf_result.yaml")
    assert validate_gamma_schedule(cfg, result)["tolerance"] == DEFAULTS


def test_dispatcher_refuses_a_non_positive_argument(tmp_path):
    cfg, result, _, _ = pair(tmp_path, "gamma_ardinf_input.yaml", "gamma_ardinf_result.yaml")
    with pytest.raises(ValueError, match="tolerance must be positive"):
        validate_gamma_schedule(cfg, copy.deepcopy(result), tolerance=0)
