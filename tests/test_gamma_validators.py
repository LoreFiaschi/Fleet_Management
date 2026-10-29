"""Tests of the Gamma validators (spec, "Validators of the Gamma Model").

The replay, exact and Monte Carlo validators and the dispatcher check a saved schedule
against the input. None of them calls Gurobi, so no test here is skipped.

The hand case is that of ``test_gamma.py``: two trains, one mission, one Gamma component,
H = [2, 2]. Train 0 flies, is repaired, flies and is replaced; train 1 flies with mean 0.
Case A is ARD1, case B is ARDinf, case C is ARD1 with a rate for the initial damage and for
the replacement. The result dictionaries are built as ``solver.read_result`` returns them:
every key of the result table (``None`` if absent), float arrays, ``NaN`` where the result
file has ``null``. The expected tails are exact values (``gammaincc``) or values of the
independent quadrature of ``examples/regression/gamma_quadrature_reference.py``.
"""

import json
import math

import numpy as np
import pytest
import yaml
from scipy.special import gammaincc

from fleet_management.config import load_config
from fleet_management.degradation_model.gamma_utils import gamma_tail_bound, gamma_tail_validator
from fleet_management.degradation_model.gamma_utils.gamma_replay_validator import (
    validate_gamma_replay_files,
    validate_gamma_replay_schedule,
)
from fleet_management.degradation_model.gamma_utils.gamma_stochastic_validator import (
    _one_sided_binomial_upper,
    validate_gamma_schedule,
    validate_gamma_schedule_files,
    validate_gamma_stochastic_schedule,
)
from fleet_management.degradation_model.gamma_utils.gamma_tail_bound import (
    moschopoulos_tail_probability,
)
from fleet_management.degradation_model.gamma_utils.gamma_tail_validator import (
    ExactGammaTerm,
    validate_gamma_tail_bound_files,
    validate_gamma_tail_bound_schedule,
)
from fleet_management.solver import RESULT_KEYS, _save_results

NAN = math.nan
_TOL = 1e-12            # means and exact tails: the series stops below a remaining mass of 1e-12
_CAP_20 = 0.7445173245655496    # A_1(20, 0.1) / 20, tau = 1 (test_gamma.py: 0.744517324566)
_REPLAY_KEYS = (
    "validator", "valid", "solver_status", "dimensions", "gamma_cells", "gamma_ard1_cells",
    "transitions_checked", "repairs", "replacements", "maximum_errors", "global_violations",
    "violations", "timing",
)
_RECORD_KEYS = {"i", "l", "k", "event", "field", "expected", "saved", "error"}


# ---------------------------------------------------------------------------
# Helpers: the hand case
# ---------------------------------------------------------------------------
def _data(**changes):
    """Two trains, one mission, one Gamma component, H = [2, 2]; train 1 flies with mean 0."""
    data = {"F": 2, "M": 1, "H": [2, 2], "model": "gamma", "repair_model": "ard1",
            "tau": 1.0, "epsilon": 0.1, "rho": 0.5, "mu_0": 0.2, "replacement_mu": 0.1,
            "mu": [[[0.3]], [[0.0]]], "gamma_beta": 20.0,
            "C_M": 1.0, "C_R": 2.0, "C_rep": 3.0, "C_D": 0.0}
    data.update(changes)
    return data


_SCHEDULE_X = ((0, 1, 0), (0, 0, 1), (0, 1, 2), (0, 0, 3), (1, 1, 1), (1, 1, 3))
_CASES = {   # case: input changes, train 0 mu, g_mu and z, g_mu of train 1 (its mu is 0.2, its z 0)
    "A": ({}, [0.5, 0.35, 0.65, 0.1], [0.2, 0.35, 0.35, 0.1], [0.0, 0.15, 0.0, 0.0], 0.2),
    "B": ({"repair_model": "ardinf"}, [0.5, 0.25, 0.55, 0.1], [NAN] * 4,
          [0.0, 0.25, 0.0, 0.0], NAN),
    "C": ({"gamma_beta_0": 20.0, "anchor_mu_0": 0.05, "gamma_beta_new": 30.0},
          [0.5, 0.275, 0.575, 0.1], [0.05, 0.275, 0.275, 0.1], [0.0, 0.225, 0.0, 0.0], 0.05),
}
_EXACT_UPPER = {   # upper bound of the exact tail: train 0 on days 0..3, train 1 on every day
    "A": ([gammaincc(6, 16), gammaincc(6, 32), 0.015106503928459624, 0.0], [0.0] * 4),
    "B": ([gammaincc(6, 16), gammaincc(6, 36), 0.0042209224829627395, 0.0], [0.0] * 4),
    "C": ([gammaincc(10, 20), 1.0539648750813092e-06, 0.009116347927228055,
           gammaincc(3, 30)], [gammaincc(4, 20)] * 4),
}


def _cfg(case="A", **changes):
    """The configuration of a case; ``changes`` replace keys of the input."""
    data = _data(**_CASES[case][0])
    data.update(changes)
    return load_config(data)


def _result(case="A", **changes):
    """The result of a case, as ``solver.read_result`` returns it; ``changes`` replace keys.

    Every call builds new arrays, so a test may edit them in place.
    """
    _, mu, g_mu, z, g_mu_1 = _CASES[case]
    x = np.zeros((2, 2, 4))
    for i, j, k in _SCHEDULE_X:
        x[i, j, k] = 1.0
    m = np.zeros((2, 1, 4))
    m[0, 0, 1] = 1.0
    r = np.zeros((2, 1, 4))
    r[0, 0, 3] = 1.0
    result = dict.fromkeys(RESULT_KEYS)
    result.update(
        status="optimal", F=2, M=1, L=1, H1=2, H2=2, T=4, H=[2, 2],
        model=[["gamma"], ["gamma"]], x=x, m=m, r=r,
        mu=np.array([[mu], [[0.2] * 4]], dtype=float),
        g_mu=np.array([[g_mu], [[g_mu_1] * 4]], dtype=float),
        z=np.array([[z], [[0.0] * 4]], dtype=float),
    )
    result.update(changes)
    return result


def _assert_record(record, i, l, k, event, field, expected, saved, error, tol=_TOL):
    assert set(record) == _RECORD_KEYS
    assert (record["i"], record["l"], record["k"]) == (i, l, k)
    assert (record["event"], record["field"]) == (event, field)
    for key, value in (("expected", expected), ("saved", saved), ("error", error)):
        assert abs(record[key] - value) <= tol, (key, record[key], value)


def _assert_plain(report):
    """A report holds plain Python types: it dumps to JSON (NaN is not allowed)."""
    json.dumps(report, allow_nan=False)


def _steps(report):
    return {(step["i"], step["l"], step["k"]): step for step in report["steps"]}


# ---------------------------------------------------------------------------
# Replay validator
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case, ard1_cells", [("A", 2), ("B", 0), ("C", 2)])
def test_replay_accepts_the_hand_cases(case, ard1_cells):
    report = validate_gamma_replay_schedule(_cfg(case), _result(case))
    assert report["valid"]
    assert report["global_violations"] == [] and report["violations"] == []
    assert set(report["maximum_errors"]) == {
        "tracked_mean", "mean_anchor", "removed_mean", "cap", "repeatability"}
    assert all(value <= _TOL for value in report["maximum_errors"].values())
    assert (report["repairs"], report["replacements"]) == (1, 1)
    assert report["transitions_checked"] == 8
    assert report["gamma_cells"] == 2
    assert report["gamma_ard1_cells"] == ard1_cells
    _assert_plain(report)


def test_replay_one_sided():
    # The saved states are upper bounds: a saved mean above the reference is accepted.
    result = _result("A")
    result["mu"][0, 0, 2] = 0.70
    assert validate_gamma_replay_schedule(_cfg("A"), result)["valid"]


@pytest.mark.parametrize("key, index, value, k, event, field, expected, saved, error, tol", [
    ("mu", (0, 0, 2), 0.64, 2, "mission", "tracked_mean", 0.65, 0.64, 0.01, _TOL),
    ("z", (0, 0, 1), 0.1, 1, "repair", "removed_mean", 0.15, 0.1, 0.05, _TOL),
    ("mu", (0, 0, 2), 0.8, 2, "mission", "cap", _CAP_20, 0.8, 0.8 - _CAP_20, 1e-9),
    ("mu", (0, 0, 3), 0.4, 3, "replacement", "repeatability", 0.35, 0.4, 0.05, _TOL),
])
def test_replay_records(key, index, value, k, event, field, expected, saved, error, tol):
    result = _result("A")
    result[key][index] = value
    report = validate_gamma_replay_schedule(_cfg("A"), result)
    assert not report["valid"]
    assert report["global_violations"] == []
    assert len(report["violations"]) == 1
    _assert_record(report["violations"][0], 0, 0, k, event, field, expected, saved, error, tol)
    assert abs(report["maximum_errors"][field] - error) <= tol
    with pytest.raises(AssertionError):
        validate_gamma_replay_schedule(_cfg("A"), result, raise_on_failure=True)


def test_replay_skips_nan_repair_cost():
    # C_R = 0 gives no repair cost, so z is NaN there (spec, "removed_mean").
    result = _result("A")
    result["z"][1, 0, :] = NAN
    report = validate_gamma_replay_schedule(_cfg("A", C_R=[[2.0], [0.0]]), result)
    assert report["valid"]
    assert report["maximum_errors"]["removed_mean"] == 0.0      # a NaN error would not dump
    _assert_plain(report)


def test_replay_global_violations():
    cfg = _cfg("A")
    report = validate_gamma_replay_schedule(cfg, _result("A", T=5))
    assert report["global_violations"] == ["result horizon T=5 differs from the input T=4"]
    assert not report["valid"]
    assert report["violations"] == []

    result = _result("A")
    result["x"][1, 1, 0] = 1.0      # both trains fly mission 1 on day 0
    report = validate_gamma_replay_schedule(cfg, result)
    assert report["global_violations"] == ["mission j=1 is covered 2 times on day k=0"]
    assert not report["valid"]

    result = _result("A")
    result["m"][0, 0, 0] = 1.0      # a repair on a day with a mission and no depot day
    report = validate_gamma_replay_schedule(cfg, result)
    assert "cell (i=0, l=0) has an intervention on day k=0 with a mission" in report[
        "global_violations"]
    assert "cell (i=0, l=0) has an intervention on day k=0 without a depot day" in report[
        "global_violations"]
    assert not report["valid"]

    result = _result("A")
    result["x"][0, 0, 0] = 1.0      # train 0 is in the depot and flies on day 0
    result["r"][0, 0, 1] = 1.0      # a repair and a replacement on day 1
    report = validate_gamma_replay_schedule(cfg, result)
    assert report["global_violations"] == [
        "train i=0 has 2 activities on day k=0",
        "cell (i=0, l=0) has a repair and a replacement on day k=1",
    ]
    assert not report["valid"]

    report = validate_gamma_replay_schedule(cfg, _result("A", status=None))
    assert report["solver_status"] == "unknown" and report["valid"]


def test_replay_anchor_array():
    # g_mu is read at ARD1 cells only; a missing key and a key with the value None are the same.
    for how in ("absent", "none"):
        result = _result("B")
        if how == "absent":
            del result["g_mu"]
        else:
            result["g_mu"] = None
        assert validate_gamma_replay_schedule(_cfg("B"), result)["valid"]

        result = _result("A")
        if how == "absent":
            del result["g_mu"]
        else:
            result["g_mu"] = None
        with pytest.raises(ValueError, match="missing 'g_mu'"):
            validate_gamma_replay_schedule(_cfg("A"), result)


# ---------------------------------------------------------------------------
# Exact validator
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", ["A", "B", "C"])
def test_exact_tails_of_the_hand_cases(case):
    report = validate_gamma_tail_bound_schedule(_cfg(case), _result(case), include_steps=True)
    assert report["valid"]
    assert report["global_violations"] == [] and report["violations"] == []
    steps = _steps(report)
    assert len(steps) == 8
    _, mu_0, g_mu_0, _, g_mu_1 = _CASES[case]
    uppers = _EXACT_UPPER[case]
    for i, (expected, mu, g_mu) in enumerate(
            [(uppers[0], mu_0, g_mu_0), (uppers[1], [0.2] * 4, [g_mu_1] * 4)]):
        for k in range(4):
            step = steps[i, 0, k]
            assert abs(step["exact_tail_upper_bound"] - expected[k]) <= _TOL
            assert step["exact_tail_estimate"] <= expected[k] + 1e-14
            assert step["remaining_mass"] <= _TOL
            assert abs(step["exact_mean"] - mu[k]) <= _TOL      # the hand results are exact
            assert abs(step["saved_mean"] - mu[k]) <= _TOL
            assert abs(step["reliability_slack"] - (0.1 - step["exact_tail_upper_bound"])) <= _TOL
            if case == "B":                                      # no anchor under ARDinf
                assert step["frozen_mean"] is None and step["saved_anchor"] is None
            else:
                assert abs(step["frozen_mean"] - g_mu[k]) <= _TOL
                assert abs(step["saved_anchor"] - g_mu[k]) <= _TOL
    assert [steps[0, 0, k]["event"] for k in range(4)] == [
        "mission", "repair", "mission", "replacement"]
    assert [steps[1, 0, k]["event"] for k in range(4)] == ["idle", "mission", "idle", "mission"]
    worst = report["worst_reliability"]
    assert (worst["i"], worst["l"], worst["k"]) == (0, 0, 2)
    assert abs(report["minimum_reliability_slack"] - (0.1 - uppers[0][2])) <= _TOL


def test_exact_terms_of_case_c():
    # Day 1 of case C: the repair moves the terms of the initial damage and of the mission
    # of day 0 to the frozen part; the repaired rates are divided by 1 - rho.
    report = validate_gamma_tail_bound_schedule(_cfg("C"), _result("C"), include_steps=True)
    terms = _steps(report)[0, 0, 1]["terms"]
    expected = [(1.0, 20.0, "frozen"), (3.0, 40.0, "frozen"), (6.0, 40.0, "frozen")]
    assert len(terms) == len(expected)
    for term, (shape, rate, part) in zip(terms, expected):
        assert abs(term["shape"] - shape) <= _TOL and abs(term["rate"] - rate) <= _TOL
        assert term["part"] == part


def test_exact_records():
    result = _result("A")
    cfg = _cfg("A", epsilon=0.01)
    report = validate_gamma_tail_bound_schedule(cfg, result)
    upper = _EXACT_UPPER["A"][0][2]
    assert not report["valid"] and len(report["violations"]) == 1
    _assert_record(report["violations"][0], 0, 0, 2, "mission", "reliability",
                   0.01, upper, upper - 0.01)
    with pytest.raises(AssertionError):
        validate_gamma_tail_bound_schedule(cfg, result, raise_on_failure=True)

    result = _result("A")
    result["mu"][0, 0, 2] = 0.64
    report = validate_gamma_tail_bound_schedule(_cfg("A"), result)
    assert not report["valid"] and len(report["violations"]) == 1
    _assert_record(report["violations"][0], 0, 0, 2, "mission", "tracked_mean", 0.65, 0.64, 0.01)

    result = _result("A")
    result["g_mu"][0, 0, 2] = 0.30
    report = validate_gamma_tail_bound_schedule(_cfg("A"), result)
    assert not report["valid"] and len(report["violations"]) == 1
    _assert_record(report["violations"][0], 0, 0, 2, "mission", "mean_anchor", 0.35, 0.30, 0.05)

    # A global violation alone makes the exact report not valid, as the replay report.
    report = validate_gamma_tail_bound_schedule(_cfg("A"), _result("A", T=5))
    assert report["global_violations"] == ["result horizon T=5 differs from the input T=4"]
    assert report["violations"] == [] and not report["valid"]


def test_exact_full_repair():
    # rho = 1 under ARDinf removes the repairable terms: the damage is 0 after the repair.
    result = _result("B")
    result["mu"][0, 0] = [0.5, 0.0, 0.3, 0.1]
    report = validate_gamma_tail_bound_schedule(_cfg("B", rho=1.0), result, include_steps=True)
    assert report["valid"]
    steps = _steps(report)
    assert steps[0, 0, 1]["exact_tail_upper_bound"] == 0.0
    assert abs(steps[0, 0, 1]["exact_mean"]) <= _TOL
    assert abs(steps[0, 0, 2]["exact_tail_upper_bound"] - gammaincc(6, 20)) <= _TOL


def test_exact_tail_edge_cases():
    tail = gamma_tail_validator._exact_tail      # (terms, constant, tau, tolerance, max terms)
    for terms, constant, expected in (
            ([], 0.2, 0.0),                                    # no term, constant below tau
            ([], 1.2, 1.0),                                    # no term, constant above tau
            ([ExactGammaTerm(1.0, 1.0, "s", 0)], 1.0, 1.0)):   # constant not below tau
        value = tail(terms, constant, 1.0, 1e-12, 100_000)
        assert value["upper_bound"] == expected and value["estimate"] == expected
        assert value["remaining_mass"] == 0.0 and value["series_terms"] == 0


def test_exact_report_keys():
    report = validate_gamma_tail_bound_schedule(_cfg("A"), _result("A"))
    for key in _REPLAY_KEYS:
        assert key in report
    assert report["validator"] == "Gamma exact validator"
    assert report["solver_status"] == "optimal"
    assert report["dimensions"] == {"F": 2, "M": 1, "L": 1, "H1": 2, "H2": 2, "T": 4}
    assert (report["gamma_cells"], report["gamma_ard1_cells"]) == (2, 2)
    assert report["transitions_checked"] == 8
    assert set(report["maximum_errors"]) == {"reliability", "tracked_mean", "mean_anchor"}
    assert {"validation_wall_seconds", "exact_tail_seconds"} <= set(report["timing"])
    for key in ("minimum_reliability_slack", "worst_reliability", "maximum_remaining_mass",
                "maximum_series_terms_used", "total_series_terms", "exact_tail_evaluations"):
        assert key in report
    assert report["exact_tail_evaluations"] == 3        # train 0, days 0 to 2
    assert 1 <= report["maximum_series_terms_used"] <= report["total_series_terms"]
    assert report["maximum_remaining_mass"] <= _TOL
    assert "steps" not in report
    _assert_plain(report)
    assert "steps" in validate_gamma_tail_bound_schedule(
        _cfg("A"), _result("A"), include_steps=True)


def test_exact_series_limit():
    cfg, result = _cfg("A"), _result("A")
    with pytest.raises(RuntimeError, match="did not reach"):
        validate_gamma_tail_bound_schedule(cfg, result, max_series_terms=5)
    with pytest.raises(ValueError, match="convolution_tolerance"):
        validate_gamma_tail_bound_schedule(cfg, result, convolution_tolerance=1.0)


# ---------------------------------------------------------------------------
# Monte Carlo validator
# ---------------------------------------------------------------------------
_MC_RUNS = {"repetitions": 20000, "batch_size": 7000}


def _mc_data(**changes):
    """Two trains, one mission of mean 0.5 and rate 4, H = [1, 1], ARDinf, no initial damage."""
    data = {"F": 2, "M": 1, "H": [1, 1], "model": "gamma", "repair_model": "ardinf",
            "tau": 1.0, "epsilon": 0.1, "rho": 0.5, "mu_0": 0.0, "mu": 0.5, "gamma_beta": 4.0,
            "C_M": 1.0, "C_R": 2.0, "C_rep": 3.0, "C_D": 0.0}
    data.update(changes)
    return data


def _mc_result(flights=((0, 1, 0), (0, 1, 1)), repairs=(), model=None, replacements=()):
    """Two days. ``flights`` are entries (i, j, k) of x; ``repairs`` and ``replacements`` are
    cells (i, l, k) of m and r."""
    x = np.zeros((2, 2, 2))
    for i, j, k in flights:
        x[i, j, k] = 1.0
    m = np.zeros((2, 1, 2))
    for i, l, k in repairs:
        m[i, l, k] = 1.0
    r = np.zeros((2, 1, 2))
    for i, l, k in replacements:
        r[i, l, k] = 1.0
    result = dict.fromkeys(RESULT_KEYS)
    result.update(
        status="optimal", F=2, M=1, L=1, H1=1, H2=1, T=2, H=[1, 1],
        model=[["gamma"], ["gamma"]] if model is None else model,
        x=x, m=m, r=r,
    )
    return result


def _near(rate, probability, runs=_MC_RUNS["repetitions"]):
    """The rate lies within five standard errors of the probability."""
    return abs(rate - probability) <= 5.0 * math.sqrt(probability * (1.0 - probability) / runs)


def test_monte_carlo_matches_the_exact_tail():
    # Train 0 flies both days: the damage is Gamma(4, 4) on day 1, and it never decreases.
    report = validate_gamma_stochastic_schedule(
        load_config(_mc_data()), _mc_result(), **_MC_RUNS)
    schedule, cells = report["schedule"], report["cells"]
    assert _near(schedule["failure_rate"], gammaincc(4, 4))     # 0.43347012036670896
    assert cells[0]["worst_time_step"] == 1
    assert cells[0]["failed_replays"] == schedule["failed_replays"]
    assert cells[1]["failed_replays"] == 0
    assert report["valid"]                                      # no target given
    assert (report["repetitions"], report["batch_size"]) == (20000, 7000)
    _assert_plain(report)


def test_monte_carlo_known_constant():
    # mu_0 = 0.6 without a rate is the constant 0.6: day 1 fails if Gamma(4, 4) > 0.4.
    report = validate_gamma_stochastic_schedule(
        load_config(_mc_data(mu_0=0.6)), _mc_result(), **_MC_RUNS)
    assert _near(report["schedule"]["failure_rate"], gammaincc(4, 1.6))    # 0.9211865127702811
    assert report["cells"][0]["maximum_sampled_damage"] >= 0.6


def test_monte_carlo_splits_the_initial_damage():
    # ARD1, rho = 1: the repair of day 0 brings the damage to the anchor. The anchor is the
    # Gamma(2.4, 4) term of mean g_0 = mu_0; with anchor_mu_0 = 0 the whole damage is removed.
    data = _mc_data(repair_model="ard1", rho=1.0, mu_0=[[0.6], [0.0]], gamma_beta_0=4.0,
                    mu=[[[0.5]], [[0.0]]])
    result = _mc_result(flights=((0, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1)),
                        repairs=((0, 0, 0),))
    report = validate_gamma_stochastic_schedule(load_config(data), result, **_MC_RUNS)
    assert _near(report["cells"][0]["failure_rate"], gammaincc(2.4, 4))    # 0.1418232030653136
    report = validate_gamma_stochastic_schedule(
        load_config({**data, "anchor_mu_0": 0.0}), result, **_MC_RUNS)
    assert report["cells"][0]["failed_replays"] == 0
    assert report["schedule"]["failed_replays"] == 0


def test_monte_carlo_replacement_draw():
    # Train 0 is replaced on day 0 (a Gamma(1.6, 4) draw of mean 0.4) and flies on day 1, so
    # its damage on day 1 is Gamma(3.6, 4): rate gammaincc(3.6, 4) = 0.352. The constant 0.4
    # (rate ignored) would give gammaincc(2, 2.4) = 0.308, 13 standard errors away.
    data = _mc_data(replacement_mu=0.4, gamma_beta_new=4.0)
    result = _mc_result(flights=((0, 0, 0), (1, 1, 0), (0, 1, 1)), replacements=((0, 0, 0),))
    report = validate_gamma_stochastic_schedule(load_config(data), result, **_MC_RUNS)
    assert _near(report["cells"][0]["failure_rate"], gammaincc(3.6, 4))
    assert report["cells"][0]["worst_time_step"] == 1
    assert _near(report["cells"][1]["failure_rate"], gammaincc(2, 4))     # train 1, day 0


def test_monte_carlo_is_reproducible():
    cfg, result = load_config(_mc_data()), _mc_result()
    reports = [validate_gamma_stochastic_schedule(cfg, result, **_MC_RUNS) for _ in range(2)]
    first, second = ({key: value for key, value in report.items() if key != "timing"}
                     for report in reports)
    assert first == second


def test_monte_carlo_target():
    cfg, result = load_config(_mc_data()), _mc_result()
    report = validate_gamma_stochastic_schedule(
        cfg, result, maximum_schedule_failure_rate=0.5, **_MC_RUNS)
    assert report["valid"]
    assert report["schedule"]["observed_target_met"]
    assert report["schedule"]["confidence_qualified_target_met"]
    assert report["schedule"]["maximum_failure_rate"] == 0.5
    report = validate_gamma_stochastic_schedule(
        cfg, result, maximum_schedule_failure_rate=0.4, **_MC_RUNS)
    assert not report["valid"]
    with pytest.raises(AssertionError):
        validate_gamma_stochastic_schedule(
            cfg, result, maximum_schedule_failure_rate=0.4, raise_on_failure=True, **_MC_RUNS)


def test_clopper_pearson():
    for failures, runs, expected in ((0, 100, 0.02951304960703993),
                                     (3, 100, 0.07571079374983006),
                                     (100, 100, 1.0)):
        assert math.isclose(_one_sided_binomial_upper(failures, runs, 0.95), expected,
                            rel_tol=1e-12)


def test_monte_carlo_excludes_other_cells():
    model = [["gamma"], ["inverse_gaussian"]]
    report = validate_gamma_stochastic_schedule(
        load_config(_mc_data(model=model, eta=0.5)), _mc_result(model=model), **_MC_RUNS)
    assert not report["complete_fleet_validation"]
    assert report["scope"] == "Gamma cells only"
    assert len(report["cells"]) == 1 and report["cells"][0]["i"] == 0
    assert len(report["excluded_cells"]) == 1
    excluded = report["excluded_cells"][0]
    assert (excluded["i"], excluded["l"], excluded["model"]) == (1, 0, "inverse_gaussian")
    assert "component" in excluded and "reason" in excluded


@pytest.mark.parametrize("argument, value", [
    ("repetitions", 0), ("repetitions", 2.5), ("batch_size", 0), ("confidence_level", 1.0),
    ("maximum_schedule_failure_rate", 1.5),
])
def test_monte_carlo_argument_errors(argument, value):
    arguments = {**_MC_RUNS, argument: value}
    with pytest.raises(ValueError):
        validate_gamma_stochastic_schedule(load_config(_mc_data()), _mc_result(), **arguments)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def test_dispatcher_default_mode():
    report = validate_gamma_schedule(_cfg("A"), _result("A"))
    assert report["mode"] == "deterministic"
    assert report["validator"] == "Gamma replay validator"
    assert report["valid"]


def test_dispatcher_reads_the_validation_block():
    cfg = _cfg("A", validation={"mode": "stochastic", "repetitions": 500, "batch_size": 200})
    report = validate_gamma_schedule(cfg, _result("A"))
    assert report["mode"] == "stochastic"
    assert (report["repetitions"], report["batch_size"]) == (500, 200)
    assert report["random_seed"] == 20260908
    report = validate_gamma_schedule(cfg, _result("A"), repetitions=300)     # the argument wins
    assert (report["repetitions"], report["batch_size"]) == (300, 200)


def test_dispatcher_both():
    cfg = _cfg("A")
    report = validate_gamma_schedule(cfg, _result("A"), mode="both", repetitions=500)
    assert report["mode"] == "both"
    assert report["deterministic"]["valid"] and report["stochastic"]["valid"]
    assert report["valid"]
    result = _result("A")
    result["mu"][0, 0, 2] = 0.64
    # The combined report exists although the replay fails (it does not raise first).
    report = validate_gamma_schedule(cfg, result, mode="both", repetitions=500)
    assert not report["valid"]
    assert not report["deterministic"]["valid"] and report["stochastic"]["valid"]
    with pytest.raises(AssertionError):
        validate_gamma_schedule(cfg, result, mode="both", repetitions=500, raise_on_failure=True)


@pytest.mark.parametrize("mode", ["Both", "monte_carlo", " stochastic"])
def test_dispatcher_rejects_modes(mode):
    with pytest.raises(ValueError, match="mode must be one of"):
        validate_gamma_schedule(_cfg("A"), _result("A"), mode=mode)


def test_dispatcher_rejects_unknown_validation_keys():
    with pytest.raises(ValueError, match="unknown key"):
        validate_gamma_schedule(_cfg("A", validation={"repetition": 5}), _result("A"))


# ---------------------------------------------------------------------------
# Moschopoulos series
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shapes, rates, threshold, expected", [
    ([6, 6], [40, 20], 0.8, 0.015106503928459624),
    ([7, 9], [20, 40], 1.0, 0.009116347927228055),
])
def test_series_matches_the_quadrature(shapes, rates, threshold, expected):
    tail = moschopoulos_tail_probability(shapes, rates, threshold)
    assert abs(tail.upper_bound - expected) <= _TOL
    assert tail.estimate <= tail.upper_bound
    assert tail.remaining_mass <= _TOL


def test_series_with_equal_rates_is_the_gamma_tail():
    tail = moschopoulos_tail_probability([2, 3], [5, 5], 1.0)
    assert abs(tail.upper_bound - gammaincc(5, 5)) <= _TOL
    assert (tail.series_terms, tail.remaining_mass) == (1, 0.0)


def test_series_errors():
    with pytest.raises(RuntimeError, match="did not reach"):
        moschopoulos_tail_probability([6, 6], [40, 20], 0.8, max_series_terms=5)
    with pytest.raises(ValueError):
        moschopoulos_tail_probability([6, 6], [40, 20], 0.0)


def test_tail_bound_keeps_only_the_series():
    names = vars(gamma_tail_bound)
    removed = [name for name in names if name.startswith("calculate_")]
    removed += [name for name in ("required_shape_for_tail", "enumerate_count_vectors",
                                  "shapes_from_expected_damage") if name in names]
    assert removed == []
    assert callable(names["moschopoulos_tail_probability"]) and "ConvolutionTail" in names


# ---------------------------------------------------------------------------
# Files and errors
# ---------------------------------------------------------------------------
_FILE_VERSIONS = {
    "replay": validate_gamma_replay_files,
    "exact": validate_gamma_tail_bound_files,
    "dispatcher": validate_gamma_schedule_files,
}


def _write_input(tmp_path, data):
    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _load_report(path):
    text = path.read_text(encoding="utf-8")
    return json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)


@pytest.mark.parametrize("result_suffix", [".yaml", ".json", ".h5"])
@pytest.mark.parametrize("report_suffix", [".yaml", ".json"])
def test_file_versions(tmp_path, result_suffix, report_suffix):
    input_path = _write_input(tmp_path, _data())
    result_path = tmp_path / f"result{result_suffix}"
    _save_results(_result("A"), result_path)
    for name, function in _FILE_VERSIONS.items():
        report_path = tmp_path / name / "out" / f"report{report_suffix}"    # folders are created
        report = function(input_path, result_path, report_path)
        assert report["valid"]
        assert _load_report(report_path)["valid"] is True


@pytest.mark.parametrize("result_suffix", [".h5", ".yaml"])
def test_file_versions_keep_nan(tmp_path, result_suffix):
    # z is NaN where C_R = 0 and g_mu is NaN under ARDinf: neither must be refused.
    input_path = _write_input(tmp_path, _data(repair_model="ardinf", C_R=[[2.0], [0.0]]))
    result = _result("B")
    result["z"][1, 0, :] = NAN
    result_path = tmp_path / f"result{result_suffix}"
    _save_results(result, result_path)
    for name, function in _FILE_VERSIONS.items():
        assert function(input_path, result_path)["valid"], name


def test_report_path_is_checked_first(tmp_path):
    input_path = _write_input(tmp_path, _data())
    for name, function in _FILE_VERSIONS.items():
        with pytest.raises(ValueError, match="report_path must end in"):
            function(input_path, tmp_path / "missing.yaml", "report.txt")


def test_file_version_writes_before_raising(tmp_path):
    input_path = _write_input(tmp_path, _data())
    result = _result("A")
    result["mu"][0, 0, 2] = 0.64
    result_path = tmp_path / "result.yaml"
    _save_results(result, result_path)
    for name, function in _FILE_VERSIONS.items():
        report_path = tmp_path / name / "report.yaml"
        with pytest.raises(AssertionError):
            function(input_path, result_path, report_path, raise_on_failure=True)
        assert _load_report(report_path)["valid"] is False


_SCHEDULE_VALIDATORS = {
    "replay": validate_gamma_replay_schedule,
    "exact": validate_gamma_tail_bound_schedule,
    "monte_carlo": lambda cfg, result, **kwargs: validate_gamma_stochastic_schedule(
        cfg, result, repetitions=100, batch_size=100, **kwargs),
}


@pytest.mark.parametrize("name", ["replay", "exact", "monte_carlo"])
def test_errors(name):
    validate = _SCHEDULE_VALIDATORS[name]
    cfg = _cfg("A")
    with pytest.raises(ValueError, match="no Gamma cell"):
        validate(_cfg("A", model="inverse_gaussian", eta=0.5),
                 _result("A", model=[["inverse_gaussian"], ["inverse_gaussian"]]))
    for how in ("absent", "none"):
        result = _result("A")
        if how == "absent":
            del result["model"]
        else:
            result["model"] = None
        with pytest.raises(ValueError, match="missing 'model'"):
            validate(cfg, result)
    with pytest.raises(ValueError, match="result 'model'"):
        validate(cfg, _result("A", model=[["gamma"], ["inverse_gaussian"]]))
    result = _result("A")
    result["x"][0, 1, 0] = 0.5
    with pytest.raises(ValueError, match="not binary"):
        validate(cfg, result)
    with pytest.raises(ValueError, match="tolerance must be positive"):
        validate(cfg, _result("A"), tolerance=0)


@pytest.mark.parametrize("name", ["replay", "exact"])
def test_errors_of_the_state_arrays(name):
    validate = _SCHEDULE_VALIDATORS[name]
    result = _result("A")
    result["mu"][0, 0, 0] = NAN
    with pytest.raises(ValueError, match="non-finite"):
        validate(_cfg("A"), result)
    result = _result("A", mu=np.zeros((2, 1, 3)))
    with pytest.raises(ValueError, match="shape"):
        validate(_cfg("A"), result)
