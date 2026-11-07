"""Tests of the Validation Tools of the spec (Validation Tools, spec lines 2381-2710).

Every expected number comes from the spec and is derived by hand or by an independent scipy
computation (``scipy.stats.gamma.sf``, ``scipy.integrate.quad``, ``scipy.stats.beta.ppf``); no value
is taken from the code under test. The inputs and the results are the hand-written files in
``tests/data/tools/validation/``, each with a comment that derives its numbers by the spec's day maps.
Broken copies are made from them in ``tmp_path``. None of the tests needs Gurobi.
"""

import copy
import functools
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy import integrate, optimize, special, stats

import fleet_management
from fleet_management.config import load_config, read_input
from fleet_management.degradation_model.gamma_utils.gamma_replay_validator import (
    validate_gamma_replay_files,
    validate_gamma_replay_schedule,
)
from fleet_management.degradation_model.gamma_utils.gamma_stochastic_validator import (
    validate_gamma_schedule,
    validate_gamma_schedule_files,
    validate_gamma_stochastic_schedule,
)
from fleet_management.degradation_model.gamma_utils.gamma_tail_validator import (
    validate_gamma_tail_bound_files,
    validate_gamma_tail_bound_schedule,
)
from fleet_management.solver import read_result
from fleet_management.validation.dashboard_gamma import load_gamma_diagnostic
from fleet_management.validation.dashboard_inspector import load_inspector
from fleet_management.validation.validator import check_result, validate

DATA = Path(__file__).parent / "data" / "tools" / "validation"

CHECK_ORDER = ["binary", "one_activity", "mission_coverage", "depot_gating", "idle",
               "loop_mu", "loop_v", "loop_R", "loop_K", "loop_g_mu", "loop_g_v", "loop_g_R",
               "loop_g_K", "objective"]                                    # table, spec 2423-2431
REPORT_KEYS = ["status", "solver_objective", "recomputed_objective", "tolerance", "checks",
               "passed", "max_violation"]                                  # spec 2437-2445
REPLAY_KEYS = ["validator", "valid", "solver_status", "dimensions", "gamma_cells",
               "gamma_ard1_cells", "transitions_checked", "repairs", "replacements",
               "maximum_errors", "global_violations", "violations", "timing"]   # spec 2564-2571
STEP_KEYS = ["i", "l", "k", "event", "epsilon", "constant", "exact_mean", "saved_mean",
             "frozen_mean", "saved_anchor", "exact_tail_estimate", "exact_tail_upper_bound",
             "remaining_mass", "series_terms", "reliability_slack", "terms"]   # spec 2629
RECORD_KEYS = ["i", "l", "k", "event", "field", "expected", "saved", "error"]   # spec 2549


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_yaml(name):
    with open(DATA / name, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path, data):
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    return str(path)


def put(res, key, i, j, k, value):
    """Set res[key][i][j][k] (arrays are nested lists, the first axis is the train)."""
    res[key][i][j][k] = value


def run_validate(tmp_path, inp, res, tol=1e-6, name="case"):
    """Write an input and a result mapping, run ``validate``, return its report."""
    in_path = write_yaml(tmp_path / f"{name}_in.yaml", inp)
    res_path = write_yaml(tmp_path / f"{name}_res.yaml", res)
    return validate(in_path, res_path, str(tmp_path / f"{name}_rep.yaml"), tol=tol)


def records(report):
    return {c["name"]: c for c in report["checks"]}


def pair(tmp_path, inp, res, name="p"):
    """Write a pair and return (cfg, result, input path, result path)."""
    in_path = write_yaml(tmp_path / f"{name}_in.yaml", inp)
    res_path = write_yaml(tmp_path / f"{name}_res.yaml", res)
    return load_config(read_input(in_path)), read_result(res_path), in_path, res_path


def near(a, b, rel=1e-9, abs_=1e-12):
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_)


def gaussian_without_idle():
    res = load_yaml("gaussian_result.yaml")
    del res["idle"]
    return res


# --- independent tails and caps ---------------------------------------------------------------
def gsf(shape, rate, threshold=1.0):
    """Pr(Gamma(shape, rate) > threshold), scipy."""
    return float(stats.gamma.sf(threshold, shape, scale=1.0 / rate))


def conv_tail(a1, r1, a2, r2, threshold=1.0):
    """Pr(X + Y > threshold), X ~ Gamma(a1, r1), Y ~ Gamma(a2, r2): convolution by quad."""
    inner = integrate.quad(
        lambda x: stats.gamma.pdf(x, a1, scale=1.0 / r1) * stats.gamma.sf(threshold - x, a2,
                                                                          scale=1.0 / r2),
        0.0, threshold, epsabs=1e-15, epsrel=1e-12, limit=400)[0]
    return gsf(a1, r1, threshold) + inner


def _w(A, t):
    """w(A) = max_{c in (0,1]} Q(A/c, t/c) (spec 1290-1292): grid of 2000 values of log c, then refine."""
    logs = np.linspace(np.log(1e-8), 0.0, 2000)
    q = lambda s: special.gammaincc(A / np.exp(s), t / np.exp(s))
    vals = q(logs)
    j = int(np.argmax(vals))
    lo, hi = logs[max(j - 1, 0)], logs[min(j + 1, len(logs) - 1)]
    best = optimize.minimize_scalar(lambda s: -q(s), bounds=(lo, hi), method="bounded",
                                    options={"xatol": 1e-12})
    return max(float(vals[j]), float(-best.fun))


@functools.lru_cache(maxsize=None)
def gamma_A1(t, eps):
    """Sharp cap A1(t, eps) of spec 1293-1298, computed as in wp8_plan_oracle.md section 2 (eps < 1/2)."""
    A_hat = optimize.brentq(lambda a: special.gammaincc(a, t) - eps, 1e-9, t)
    f = lambda A: _w(A, t) - eps
    if f(A_hat) <= 0.0:
        return A_hat
    return optimize.brentq(f, eps * t, A_hat, xtol=1e-14)


def gamma_cap(beta_star, tau, eps):
    """Cap A1(beta* tau, eps) / beta* of the tracked mean (Eq. gm_row)."""
    return gamma_A1(beta_star * tau, eps) / beta_star


# ---------------------------------------------------------------------------
# 1. validate: the valid pair, one broken copy per check
# ---------------------------------------------------------------------------
def test_validate_valid_pair_report(tmp_path):
    """Spec 2437-2445, 2423-2431, 2433: the valid pair passes; report keys, order, objective."""
    out = tmp_path / "out" / "report.yaml"
    report = validate(str(DATA / "gaussian_input.yaml"), str(DATA / "gaussian_result.yaml"),
                      str(out))
    for key in REPORT_KEYS:
        assert key in report
    assert report["status"] == "optimal"
    assert report["tolerance"] == 1e-6
    assert report["passed"] is True
    assert report["max_violation"] <= 1e-6
    # the checks in the order of the table (the file has the key idle, so idle is listed)
    assert [c["name"] for c in report["checks"]] == CHECK_ORDER
    for c in report["checks"]:
        assert set(["name", "passed", "violation", "tolerance"]) <= set(c)
        assert c["passed"] is True and c["violation"] <= 1e-6 and c["tolerance"] == 1e-6
    # objective derived by hand in the result file: 1.1 + 12.3 + 0.45 = 13.85
    assert near(report["solver_objective"], 13.85)
    assert near(report["recomputed_objective"], 13.85, 1e-9)
    # the report is also written to validation_path
    with open(out, encoding="utf-8") as handle:
        saved = yaml.safe_load(handle)
    assert saved["passed"] is True
    assert [c["name"] for c in saved["checks"]] == CHECK_ORDER
    assert near(saved["recomputed_objective"], 13.85)


def test_validate_checks_without_idle_keep_the_order(tmp_path):
    """Spec 2428, 2435: without the key idle the other checks keep the order; eight loop checks listed."""
    report = run_validate(tmp_path, load_yaml("gaussian_input.yaml"), gaussian_without_idle())
    names = [c["name"] for c in report["checks"]]
    assert names == [n for n in CHECK_ORDER if n in names]
    assert [n for n in names if n.startswith("loop_")] == CHECK_ORDER[5:13]
    assert names[0] == "binary" and names[-1] == "objective"
    assert report["passed"] is True


def _nonzero(report):
    return {c["name"]: c["violation"] for c in report["checks"] if c["violation"] > 1e-12}


def test_validate_binary_violation(tmp_path):
    """Spec 2424: an entry 0.4 of x is at distance 0.4 from {0,1}; the other rows change as implied.

    x[0][1][0] (train 0 flies mission 1 on day 1) = 0.4: mission_coverage = |0.4 - 1| = 0.6
    (spec 2426); one_activity sum 0.4 <= 1 (no violation); the depot row and the cost are unchanged."""
    res = gaussian_without_idle()
    put(res, "x", 0, 1, 0, 0.4)
    report = run_validate(tmp_path, load_yaml("gaussian_input.yaml"), res)
    v = _nonzero(report)
    assert near(records(report)["binary"]["violation"], 0.4)
    assert v.keys() == {"binary", "mission_coverage"}
    assert near(v["mission_coverage"], 0.6)
    assert report["passed"] is False and records(report)["binary"]["passed"] is False
    assert near(report["max_violation"], 0.6)


def test_validate_binary_covers_m_and_r(tmp_path):
    """Spec 2424: x^m and x^r are binary too. r[0][0][0] = 0.7 is at distance 0.3 from {0,1}; as implied,
    depot_gating (x^m + x^r <= x_i0k, here 0.7 > 0) is 0.7."""
    res = gaussian_without_idle()
    put(res, "r", 0, 0, 0, 0.7)
    report = run_validate(tmp_path, load_yaml("gaussian_input.yaml"), res)
    v = _nonzero(report)
    assert near(v["binary"], 0.3)
    assert near(v["depot_gating"], 0.7)
    assert set(v) <= {"binary", "depot_gating", "objective"}
    res = gaussian_without_idle()
    put(res, "m", 1, 0, 1, 0.2)         # distance 0.2 from 0; day 2 of train 1 flies: gating 0.2 > x_100? = 0
    report = run_validate(tmp_path, load_yaml("gaussian_input.yaml"), res, name="m")
    assert near(records(report)["binary"]["violation"], 0.2)


def test_validate_one_activity_violation(tmp_path):
    """Spec 2425: train 0 both at the depot and flying on day 1: sum_j x = 2, violation 1.
    The extra depot day costs C_M = 1 more (Eq. a_daycost): the saved objective is 14.85."""
    res = gaussian_without_idle()
    put(res, "x", 0, 0, 0, 1)
    res["objective"] = 14.85
    report = run_validate(tmp_path, load_yaml("gaussian_input.yaml"), res)
    assert set(_nonzero(report)) == {"one_activity"}
    assert near(records(report)["one_activity"]["violation"], 1.0)
    assert near(report["recomputed_objective"], 14.85)


def test_validate_mission_coverage_violation(tmp_path):
    """Spec 2426: nobody flies mission 1 on day 1 (sum_i x_i1k = 0): violation |0 - 1| = 1."""
    res = gaussian_without_idle()
    put(res, "x", 0, 1, 0, 0)
    report = run_validate(tmp_path, load_yaml("gaussian_input.yaml"), res)
    assert set(_nonzero(report)) == {"mission_coverage"}
    assert near(records(report)["mission_coverage"]["violation"], 1.0)


def test_validate_mission_coverage_two_flyers(tmp_path):
    """Spec 2426: two trains fly mission 1 on day 3 (sum = 2): violation |2 - 1| = 1, and the depot
    train 1 flies on a depot day: one_activity violation 1 as well (x_100 + x_111 = 2)."""
    res = gaussian_without_idle()
    put(res, "x", 1, 1, 2, 1)
    report = run_validate(tmp_path, load_yaml("gaussian_input.yaml"), res)
    v = _nonzero(report)
    assert near(v["mission_coverage"], 1.0)
    assert near(v["one_activity"], 1.0)


def test_validate_depot_gating_violation(tmp_path):
    """Spec 2427: a repair of train 0 on day 1, when it flies (x_{0,0,1} = 0): m + r - x_i0k = 1."""
    res = gaussian_without_idle()
    put(res, "m", 0, 0, 0, 1)
    report = run_validate(tmp_path, load_yaml("gaussian_input.yaml"), res)
    assert set(_nonzero(report)) == {"depot_gating"}
    assert near(records(report)["depot_gating"]["violation"], 1.0)


def test_validate_depot_gating_replacement_not_allowed(tmp_path):
    """Spec 2427: x^r = 0 is required when allow_replacement is false. The replacement of train 1 on
    day 3 (a depot day) then violates the row; the excess of x^r over 0 is 1."""
    inp = load_yaml("gaussian_input.yaml")
    inp["allow_replacement"] = False
    report = run_validate(tmp_path, inp, gaussian_without_idle())
    rec = records(report)["depot_gating"]
    assert rec["passed"] is False
    assert near(rec["violation"], 1.0)
    assert report["passed"] is False


def test_validate_idle_violation(tmp_path):
    """Spec 2428, 1734: idle_{1,0,3} saved as 1, but x_{1,0,3} - m - r = 1 - 0 - 1 = 0 (replacement day)."""
    res = load_yaml("gaussian_result.yaml")
    put(res, "idle", 1, 0, 2, 1.0)
    report = run_validate(tmp_path, load_yaml("gaussian_input.yaml"), res)
    assert set(_nonzero(report)) == {"idle"}
    assert near(records(report)["idle"]["violation"], 1.0)
    assert report["checks"][4]["name"] == "idle"


def test_validate_idle_on_a_plain_depot_day(tmp_path):
    """Spec 2428: a depot day without intervention has idle = x_i0k - m - r = 1. Removing the repair of
    train 0 on day 2 (m = 0, z = 0) makes the saved idle (0) wrong by 1."""
    res = load_yaml("gaussian_result.yaml")
    put(res, "m", 0, 0, 1, 0)
    put(res, "z", 0, 0, 1, 0.0)
    res["objective"] = 13.85 - 2 * 0.15           # z cost C_R z = 0.3 removed (Eq. a_daycost)
    report = run_validate(tmp_path, load_yaml("gaussian_input.yaml"), res)
    assert near(records(report)["idle"]["violation"], 1.0)
    assert near(report["recomputed_objective"], 13.55)


def test_validate_loop_mu_and_v(tmp_path):
    """Spec 2429, 1608: loop_mu = max(0, max_i,l (mu_T - mu_H1)); H1 = 1, T = 3.
    mu[0][0][2] = 0.4 against mu_H1 = 0.3 gives 0.1; v[1][0][2] = 0.0021 against 0.0001 gives 0.002."""
    res = gaussian_without_idle()
    put(res, "mu", 0, 0, 2, 0.4)
    report = run_validate(tmp_path, load_yaml("gaussian_input.yaml"), res)
    assert set(_nonzero(report)) == {"loop_mu"}
    assert near(records(report)["loop_mu"]["violation"], 0.1)
    res = gaussian_without_idle()
    put(res, "v", 1, 0, 2, 0.0021)
    report = run_validate(tmp_path, load_yaml("gaussian_input.yaml"), res, name="v")
    assert set(_nonzero(report)) == {"loop_v"}
    assert near(records(report)["loop_v"]["violation"], 0.002)


# violation of each further loop check: day H1 = day 1 (index 0), day T = day 3 (index 2)
LOOP_CASES = [("R", [0.1, 0.2, 0.35], 0.25), ("K", [0.5, 0.6, 0.9], 0.4),
              ("g_mu", [0.2, 0.2, 0.25], 0.05), ("g_v", [0.001, 0.002, 0.0035], 0.0025),
              ("g_R", [0.3, 0.3, 0.8], 0.5), ("g_K", [0.7, 0.7, 0.9], 0.2)]


@pytest.mark.parametrize("key, series, expected", LOOP_CASES)
def test_validate_each_loop_check(tmp_path, key, series, expected):
    """Spec 2429: loop_<q> for q in R, K, g_mu, g_v, g_R, g_K is max(0, q_T - q_H1) = last - first.
    The key holds numbers for train 1 only; train 0 holds null, which is skipped (spec 2429)."""
    res = gaussian_without_idle()
    res[key] = [[[None, None, None]], [series]]
    report = run_validate(tmp_path, load_yaml("gaussian_input.yaml"), res)
    assert set(_nonzero(report)) == {f"loop_{key}"}
    assert near(records(report)[f"loop_{key}"]["violation"], expected, 1e-9, 1e-12)
    assert records(report)[f"loop_{key}"]["passed"] is False


def test_validate_loop_skips_missing_numbers(tmp_path):
    """Spec 2435: a key with no pair of numbers at days H1 and T has violation 0 (here NaN at day H1)."""
    res = gaussian_without_idle()
    res["g_R"] = [[[None, 0.5, 0.9]], [[None, 0.5, 0.9]]]
    report = run_validate(tmp_path, load_yaml("gaussian_input.yaml"), res)
    assert records(report)["loop_g_R"]["violation"] == 0.0
    assert report["passed"] is True


def test_validate_objective_violation(tmp_path):
    """Spec 2430: saved o = 13.0 below recomputed r = 13.85: violation (r - o)/max(1,|o|) = 0.85/13."""
    res = gaussian_without_idle()
    res["objective"] = 13.0
    report = run_validate(tmp_path, load_yaml("gaussian_input.yaml"), res)
    assert set(_nonzero(report)) == {"objective"}
    assert near(records(report)["objective"]["violation"], 0.85 / 13.0, 1e-9)
    assert report["passed"] is False
    assert near(report["solver_objective"], 13.0)


def test_validate_objective_above_recomputed_passes(tmp_path):
    """Spec 2430: a saved objective above the recomputed one is not a violation."""
    res = gaussian_without_idle()
    res["objective"] = 20.0
    report = run_validate(tmp_path, load_yaml("gaussian_input.yaml"), res)
    assert records(report)["objective"]["violation"] == 0.0
    assert report["passed"] is True


def test_validate_small_objective_uses_max_one(tmp_path):
    """Spec 2430: the denominator is max(1, |o|): saved 0.5 against recomputed 13.85 gives 13.35/1."""
    res = gaussian_without_idle()
    res["objective"] = 0.5
    report = run_validate(tmp_path, load_yaml("gaussian_input.yaml"), res)
    assert near(records(report)["objective"]["violation"], 13.35, 1e-9)


def test_validate_tolerance_argument(tmp_path):
    """Spec 2407, 2420: a check passes if its violation is at most tol. The objective violation 0.85/13
    = 0.0654 passes at tol = 0.1 and fails at tol = 1e-6; the report echoes tol."""
    res = gaussian_without_idle()
    res["objective"] = 13.0
    inp = load_yaml("gaussian_input.yaml")
    loose = run_validate(tmp_path, inp, res, tol=0.1, name="loose")
    assert loose["tolerance"] == 0.1
    assert records(loose)["objective"]["passed"] is True and loose["passed"] is True
    assert near(records(loose)["objective"]["violation"], 0.85 / 13.0, 1e-9)
    tight = run_validate(tmp_path, inp, res, tol=1e-6, name="tight")
    assert tight["passed"] is False


def test_validate_status_rule(tmp_path):
    """Spec 2419: the status is not a check; a result with status time_limit and a valid schedule passes
    and the report echoes the status."""
    res = load_yaml("gaussian_result.yaml")
    res["status"] = "time_limit"
    report = run_validate(tmp_path, load_yaml("gaussian_input.yaml"), res)
    assert report["status"] == "time_limit"
    assert report["passed"] is True
    assert "status" not in [c["name"] for c in report["checks"]]


# ---------------------------------------------------------------------------
# 2. Recomputed objective in each mode (spec 2433, Eq. a_objective, a_daycost, a_phases)
# ---------------------------------------------------------------------------
def test_recomputed_objective_operating_average(tmp_path):
    """Spec 2433, 672-680: J_op/H2 + C_D u. Day costs of the pair (derived in the result file):
    c_1 = 1.1, c_2 = 1.3, c_3 = 11, so J_op = 12.3, J_op/H2 = 6.15, C_D u = 0.45: 6.6."""
    inp = load_yaml("gaussian_input.yaml")
    inp["objective_mode"] = "operating_average"
    res = load_yaml("gaussian_result.yaml")
    res["objective"] = 6.6
    report = run_validate(tmp_path, inp, res)
    assert near(report["recomputed_objective"], 6.6)
    assert report["passed"] is True


def test_recomputed_objective_evaluation_total(tmp_path):
    """Spec 2433, 677-678: J_init + (E - H1)/H2 J_op + C_D u with E = 5, H1 = 1, H2 = 2:
    1.1 + (4/2)*12.3 + 0.45 = 26.15."""
    inp = load_yaml("gaussian_input.yaml")
    inp["objective_mode"] = "evaluation_total"
    inp["evaluation_horizon"] = 5
    res = load_yaml("gaussian_result.yaml")
    res["objective"] = 26.15
    report = run_validate(tmp_path, inp, res)
    assert near(report["recomputed_objective"], 26.15)
    assert report["passed"] is True


def test_recomputed_objective_uses_saved_u_and_nan_z(tmp_path):
    """Spec 2433: the saved u enters the objective (u = 0.5 instead of 0.3: +1.5*0.2 = +0.3), and z is 0
    where it is null: with C_R = 0 and z null the repair cost 0.4 drops out. 13.85 - 0.4 + 0.3 = 13.75."""
    inp = load_yaml("gaussian_input.yaml")
    inp["C_R"] = 0.0
    res = load_yaml("gaussian_result.yaml")
    res["z"] = [[[None, None, None]], [[None, None, None]]]
    res["u"] = 0.5
    res["objective"] = 13.75
    report = run_validate(tmp_path, inp, res)
    assert near(report["recomputed_objective"], 13.75)
    assert report["passed"] is True


def test_recomputed_objective_c_d_zero(tmp_path):
    """Spec 683: with C_D = 0 the term C_D u is absent: 13.85 - 0.45 = 13.4."""
    inp = load_yaml("gaussian_input.yaml")
    inp["C_D"] = 0.0
    res = load_yaml("gaussian_result.yaml")
    res["objective"] = 13.4
    report = run_validate(tmp_path, inp, res)
    assert near(report["recomputed_objective"], 13.4)


# ---------------------------------------------------------------------------
# 3. validate: errors and report path (spec 2402-2415)
# ---------------------------------------------------------------------------
def test_validate_missing_files(tmp_path):
    """Spec 2412: FileNotFoundError when the input or the result is missing."""
    rep = str(tmp_path / "r.yaml")
    with pytest.raises(FileNotFoundError):
        validate(str(tmp_path / "none.yaml"), str(DATA / "gaussian_result.yaml"), rep)
    with pytest.raises(FileNotFoundError):
        validate(str(DATA / "gaussian_input.yaml"), str(tmp_path / "none.yaml"), rep)


def test_validate_unsupported_extensions(tmp_path):
    """Spec 2414, 2406: ValueError for an unsupported extension of the result, the input or the report."""
    inp, res = load_yaml("gaussian_input.yaml"), load_yaml("gaussian_result.yaml")
    ok_in = write_yaml(tmp_path / "in.yaml", inp)
    ok_res = write_yaml(tmp_path / "res.yaml", res)
    txt_res = write_yaml(tmp_path / "res.txt", res)
    txt_in = write_yaml(tmp_path / "in.txt", inp)
    with pytest.raises(ValueError):
        validate(ok_in, txt_res, str(tmp_path / "r.yaml"))
    with pytest.raises(ValueError):
        validate(txt_in, ok_res, str(tmp_path / "r.yaml"))
    with pytest.raises(ValueError):
        validate(ok_in, ok_res, str(tmp_path / "r.txt"))


def test_validate_result_without_x(tmp_path):
    """Spec 2414: the result has no x (missing or null)."""
    inp = load_yaml("gaussian_input.yaml")
    res = load_yaml("gaussian_result.yaml")
    del res["x"]
    with pytest.raises(ValueError):
        run_validate(tmp_path, inp, res, name="missing")
    res = load_yaml("gaussian_result.yaml")
    res["x"] = None
    with pytest.raises(ValueError):
        run_validate(tmp_path, inp, res, name="null")


def test_validate_input_failing_a_check(tmp_path):
    """Spec 2414, 530: an input that fails the consistency checks (tau < 0) raises ValueError."""
    inp = load_yaml("gaussian_input.yaml")
    inp["tau"] = -1.0
    with pytest.raises(ValueError):
        run_validate(tmp_path, inp, load_yaml("gaussian_result.yaml"))


@pytest.mark.parametrize("name, value", [("F", 3), ("M", 2), ("L", 2), ("H1", 2), ("H2", 3)])
def test_validate_size_mismatch_is_named(tmp_path, name, value):
    """Spec 2414: the sizes F, M, L, H1, H2 differ between result and input; the message names the size."""
    res = load_yaml("gaussian_result.yaml")
    res[name] = value
    with pytest.raises(ValueError, match=rf"\b{name}\b"):
        run_validate(tmp_path, load_yaml("gaussian_input.yaml"), res)


def test_validate_shape_mismatch(tmp_path):
    """Spec 2414: an array whose shape differs from the input (x with 2 days instead of 3)."""
    res = load_yaml("gaussian_result.yaml")
    res["x"] = [[[0, 1], [1, 0]], [[1, 0], [0, 1]]]
    with pytest.raises(ValueError):
        run_validate(tmp_path, load_yaml("gaussian_input.yaml"), res)
    res = load_yaml("gaussian_result.yaml")
    res["mu"] = [[[0.3, 0.15]], [[0.05, 0.25]]]
    with pytest.raises(ValueError):
        run_validate(tmp_path, load_yaml("gaussian_input.yaml"), res, name="mu")


@pytest.mark.parametrize("key", ["m", "r", "mu", "z", "u", "objective"])
def test_validate_missing_required_key(tmp_path, key):
    """Spec 2414: the result has x but lacks one of m, r, mu, z, u, objective."""
    res = load_yaml("gaussian_result.yaml")
    del res[key]
    with pytest.raises(ValueError, match=rf"\b{key}\b"):
        run_validate(tmp_path, load_yaml("gaussian_input.yaml"), res)


@pytest.mark.parametrize("key", ["x", "m", "r", "idle"])
def test_validate_non_finite_entry(tmp_path, key):
    """Spec 2414: a non-finite entry in x, m, r or idle (a null in the file) raises ValueError."""
    res = load_yaml("gaussian_result.yaml")
    j = 1 if key == "x" else 0
    res[key][0][j][0] = None
    with pytest.raises(ValueError):
        run_validate(tmp_path, load_yaml("gaussian_input.yaml"), res)


def test_validate_report_extensions_and_folder(tmp_path):
    """Spec 2406: .yaml, .yml and .json are accepted; no extension gets .yaml; the folder is created."""
    in_path, res_path = str(DATA / "gaussian_input.yaml"), str(DATA / "gaussian_result.yaml")
    validate(in_path, res_path, str(tmp_path / "a" / "b" / "report.json"))
    with open(tmp_path / "a" / "b" / "report.json", encoding="utf-8") as handle:
        assert json.load(handle)["passed"] is True
    validate(in_path, res_path, str(tmp_path / "c" / "report.yml"))
    with open(tmp_path / "c" / "report.yml", encoding="utf-8") as handle:
        assert yaml.safe_load(handle)["passed"] is True
    validate(in_path, res_path, str(tmp_path / "d" / "report"))
    assert (tmp_path / "d" / "report.yaml").exists()
    assert not (tmp_path / "d" / "report").exists()


def test_validate_default_report_path(tmp_path):
    """Spec 2398, 2406: the default validation_path is validation.yaml (in the working directory)."""
    import os
    previous = os.getcwd()
    os.chdir(tmp_path)
    try:
        validate(str(DATA / "gaussian_input.yaml"), str(DATA / "gaussian_result.yaml"))
    finally:
        os.chdir(previous)
    assert (tmp_path / "validation.yaml").exists()


def test_check_result_returns_report_without_writing(tmp_path):
    """Spec 2435: check_result(cfg, result, tol) returns the report without writing; same numbers."""
    cfg, result, _, _ = pair(tmp_path, load_yaml("gaussian_input.yaml"),
                             load_yaml("gaussian_result.yaml"))
    report = check_result(cfg, result)
    assert [c["name"] for c in report["checks"]] == CHECK_ORDER
    assert report["passed"] is True
    assert near(report["recomputed_objective"], 13.85)
    assert report["tolerance"] == 1e-6
    assert sorted(p.name for p in tmp_path.iterdir()) == ["p_in.yaml", "p_res.yaml"]
    assert check_result(cfg, result, tol=0.5)["tolerance"] == 0.5


def test_public_api_exports_validate():
    """Spec 2398: fleet_management.validate is the structural validator."""
    assert fleet_management.validate is validate


# ---------------------------------------------------------------------------
# 4. Dashboards (spec 2447-2502; no Streamlit)
# ---------------------------------------------------------------------------
def test_dashboard_modules_load_without_streamlit():
    """Spec 2458: Streamlit is imported only inside main and the page functions; every module loads
    without it. The import of streamlit is made to fail."""
    names = ["fleet_management.validator_dashboard", "fleet_management.validation.dashboard_inspector",
             "fleet_management.validation.dashboard_gamma"]
    saved = {n: sys.modules.pop(n, None) for n in names}
    previous = sys.modules.get("streamlit", "absent")
    sys.modules["streamlit"] = None            # `import streamlit` raises ImportError
    try:
        modules = [importlib.import_module(n) for n in names]
        assert callable(modules[0].main)
        assert callable(modules[1].load_inspector) and callable(modules[1].render_result_inspector)
        assert callable(modules[2].load_gamma_diagnostic) and callable(modules[2].render_gamma_diagnostic)
    finally:
        if previous == "absent":
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = previous
        for n, module in saved.items():
            if module is not None:
                sys.modules[n] = module


def test_inspector_loads_pair_and_runs_check_result(tmp_path):
    """Spec 2456, 2490: the loaded data of the Gaussian pair; the Validation tab holds check_result with
    the sidebar tolerance."""
    state = load_inspector(str(DATA / "gaussian_input.yaml"), str(DATA / "gaussian_result.yaml"), 1e-6)
    assert state["report"]["passed"] is True
    assert state["report"]["tolerance"] == 1e-6
    assert near(state["report"]["recomputed_objective"], 13.85)
    state = load_inspector(str(DATA / "gaussian_input.yaml"), str(DATA / "gaussian_result.yaml"), 0.01)
    assert state["report"]["tolerance"] == 0.01
    assert state["tol"] == 0.01


def test_inspector_gaussian_margins_by_hand(tmp_path):
    """Spec 2480: margin of a G cell min{(tau-mu)^2 - kappa_G^2 v, tau-mu}, kappa_G = Phi^-1(0.95).
    Train 0 mu 0.3, 0.15, 0.25 and v 0.0014, 0.00035, 0.00085 (result file); first day: 0.7^2 - 2.7055435*0.0014."""
    state = load_inspector(str(DATA / "gaussian_input.yaml"), str(DATA / "gaussian_result.yaml"), 1e-6)
    margins = np.asarray(state["margins"], dtype=float)
    assert margins.shape == (2, 1, 3)
    kg2 = float(stats.norm.ppf(0.95)) ** 2
    mu = np.array([[0.3, 0.15, 0.25], [0.05, 0.25, 0.0]])
    v = np.array([[0.0014, 0.00035, 0.00085], [0.0001, 0.0011, 0.0]])
    expected = np.minimum((1 - mu) ** 2 - kg2 * v, 1 - mu)
    assert np.allclose(margins[:, 0, :], expected, rtol=1e-9, atol=1e-12)
    assert near(float(margins[0, 0, 0]), 0.49 - 2.705543454095413 * 0.0014, 1e-8)
    assert near(float(margins[0, 0, 0]), 0.48621224, 1e-6)


def _expected_margin_table_margins():
    """The ten margins x two days of margins_result.yaml by the table of spec 2479-2488 (independent)."""
    tau = 1.0
    out = np.full((2, 5, 2), np.nan)
    kg2 = float(stats.norm.ppf(0.95)) ** 2
    mu = {(0, 0): (0.30, 0.25), (0, 1): (0.20, 0.10), (0, 2): (0.20, 0.10), (0, 3): (0.05, 0.04),
          (0, 4): (0.30, 0.20), (1, 0): (0.30, 0.20), (1, 1): (0.30, 0.20), (1, 2): (0.10, 0.10),
          (1, 3): (0.30, 0.25), (1, 4): (0.90, 0.50)}
    for d in range(2):
        m = lambda c: mu[c][d]
        # G, W: min{(tau-mu)^2 - kappa_G^2 v, tau-mu}
        for c, v in {(0, 0): (0.0014, 0.0010), (1, 3): (0.004, 0.002), (1, 4): (0.01, 0.001)}.items():
            out[c[0], c[1], d] = min((tau - m(c)) ** 2 - kg2 * v[d], tau - m(c))
        # IG: mu_bar - mu; mu_bar = tau eps / E[Z; Z > q], Z ~ IG(mean 1, shape 1/eta) (scipy)
        eta, eps = 0.15, 0.1
        dist = stats.invgauss(mu=eta, scale=1.0 / eta)
        q = dist.isf(eps)
        e_z = integrate.quad(lambda z: z * dist.pdf(z), q, np.inf)[0]
        out[0, 1, d] = tau * eps / e_z - m((0, 1))
        # Ga: A1(beta* tau, eps)/beta* - mu with beta* = 5
        out[0, 2, d] = gamma_cap(5.0, tau, 0.1) - m((0, 2))
        # Mk: eps tau - mu
        out[0, 3, d] = 0.1 * tau - m((0, 3))
        # Ca: min{(tau-mu)^2 - kappa_C^2 v, tau-mu}, kappa_C^2 = (1-eps)/eps = 9
        v = (0.01, 0.005)[d]
        out[0, 4, d] = min((tau - m((0, 4))) ** 2 - 9.0 * v, tau - m((0, 4)))
        # Ho: min{(tau-mu)^2 - kappa/2 R, tau-mu}, kappa = ln 10
        kappa = math.log(10.0)
        R = (0.02, 0.01)[d]
        out[1, 0, d] = min((tau - m((1, 0))) ** 2 - kappa / 2.0 * R, tau - m((1, 0)))
        # Be: b = 0.05, c = kappa b/3: min{(tau-c-mu)^2 - c^2 - 2 kappa v, tau-c-mu}
        c_b = kappa * 0.05 / 3.0
        v = (0.0002, 0.0001)[d]
        out[1, 1, d] = min((tau - c_b - m((1, 1))) ** 2 - c_b ** 2 - 2 * kappa * v, tau - c_b - m((1, 1)))
        # Ch: ln eps - (K - s tau), s = 5
        K = (0.4, 0.3)[d]
        out[1, 2, d] = math.log(0.1) - (K - 5.0 * tau)
    return out


def test_inspector_margin_of_every_row_of_the_table():
    """Spec 2479-2488: one margin per row of the table (G, IG, Ga, Mk, Ca, Ho, Be, Ch, W, G), days 1 and 2.
    The hand values are in the comment of margins_result.yaml."""
    state = load_inspector(str(DATA / "margins_input.yaml"), str(DATA / "margins_result.yaml"), 1e-6)
    margins = np.asarray(state["margins"], dtype=float)
    assert margins.shape == (2, 5, 2)
    expected = _expected_margin_table_margins()
    assert np.allclose(margins, expected, rtol=1e-6, atol=1e-9), (margins, expected)
    # a few literal values of the hand derivation
    assert near(float(margins[0, 3, 0]), 0.05, 1e-9)                    # Mk: 0.1 - 0.05
    assert near(float(margins[0, 4, 0]), 0.40, 1e-9)                    # Ca: 0.49 - 9*0.01
    assert near(float(margins[1, 0, 0]), 0.4669741, 1e-6)               # Ho
    assert near(float(margins[1, 1, 0]), 0.4353520, 1e-6)               # Be
    assert near(float(margins[1, 2, 0]), 2.2974149, 1e-6)               # Ch
    assert near(float(margins[1, 3, 0]), 0.4791778, 1e-6)               # W
    assert near(float(margins[1, 4, 0]), -0.0170554, 1e-5)              # G, a violation
    assert near(float(margins[0, 1, 0]), 0.5493234 - 0.2, 1e-6)         # IG cap 0.5493234
    assert near(float(margins[0, 2, 0]), 0.5538641 - 0.2, 1e-6)         # Ga cap 0.5538641


def test_inspector_margin_table_overview_and_tolerance():
    """Spec 2476, 2465, 2490: the row holds if the margin is at least -tolerance; the Overview counts the
    violations, the largest mu/tau and the models; the ten smallest margins are listed in order."""
    inp, res = str(DATA / "margins_input.yaml"), str(DATA / "margins_result.yaml")
    state = load_inspector(inp, res, 1e-6)
    ov = state["overview"]
    assert ov["reliability_violations"] == 1                  # only G (1,4) on day 1: -0.0170554
    assert near(ov["largest_mu_over_tau"], 0.9)
    assert ov["status"] == "optimal" and near(ov["objective"], 2.0)
    assert (ov["H1"], ov["H2"], ov["T"]) == (1, 1, 2)
    assert sorted(ov["models"]) == sorted(["gaussian", "inverse_gaussian", "gamma", "wiener",
                                           "rainflow (markov)", "rainflow (cantelli)",
                                           "rainflow (hoeffding)", "rainflow (bernstein)",
                                           "rainflow (chernoff)"])
    smallest = ov["smallest_margins"]
    assert len(smallest) == 10
    values = list(smallest["margin"])
    assert values == sorted(values)
    assert near(values[0], -0.0170554, 1e-5)
    assert len(state["table"]) == 20                          # ten cells, two days, every margin a number
    assert int((~state["table"]["holds"]).sum()) == 1
    # a tolerance of 0.02 admits the margin -0.0170554
    loose = load_inspector(inp, res, 0.02)
    assert loose["overview"]["reliability_violations"] == 0
    assert bool(loose["table"]["holds"].all())


def test_inspector_nan_entries_are_left_out(tmp_path):
    """Spec 2476: NaN entries are left out of the margins. Removing v of the G cell (0,0) on day 2 leaves
    19 rows in the table and a NaN margin."""
    res = load_yaml("margins_result.yaml")
    res["v"][0][0][1] = None
    cfg_in = str(DATA / "margins_input.yaml")
    res_path = write_yaml(tmp_path / "res.yaml", res)
    state = load_inspector(cfg_in, res_path, 1e-6)
    assert np.isnan(np.asarray(state["margins"], dtype=float)[0, 0, 1])
    assert len(state["table"]) == 19


def test_inspector_repeatability_table():
    """Spec 2490: Repeatability shows both values and the margin q_H1 - q_T for each cell and saved
    quantity with numbers at days H1 and T. Train 1 of the Gaussian pair, mu: 0.05 and 0.0."""
    state = load_inspector(str(DATA / "gaussian_input.yaml"), str(DATA / "gaussian_result.yaml"), 1e-6)
    loops = state["loops"]
    mu_rows = loops[loops["quantity"] == "mu"]
    assert len(mu_rows) == 2
    row = mu_rows[mu_rows["train"] == 2].iloc[0]
    assert near(float(row["value_at_H1"]), 0.05) and near(float(row["value_at_T"]), 0.0)
    assert near(float(row["margin"]), 0.05)
    row = mu_rows[mu_rows["train"] == 1].iloc[0]
    assert near(float(row["margin"]), 0.3 - 0.25, 1e-9)
    assert bool(loops["holds"].all())
    assert set(loops["quantity"]) == {"mu", "v"}              # only the quantities saved with numbers


def test_inspector_model_mismatch_raises(tmp_path):
    """Spec 2456: a result whose model differs from the input at a cell raises ValueError."""
    res = load_yaml("gaussian_result.yaml")
    res["model"] = [["gaussian"], ["wiener"]]
    in_path = write_yaml(tmp_path / "in.yaml", load_yaml("gaussian_input.yaml"))
    res_path = write_yaml(tmp_path / "res.yaml", res)
    with pytest.raises(ValueError):
        load_inspector(in_path, res_path, 1e-6)
    del res["model"]
    res_path = write_yaml(tmp_path / "res2.yaml", res)
    with pytest.raises(ValueError):
        load_inspector(in_path, res_path, 1e-6)


def test_inspector_size_mismatch_raises(tmp_path):
    """Spec 2456: the errors of validate on sizes apply to the dashboard loaders."""
    res = load_yaml("gaussian_result.yaml")
    res["F"] = 3
    in_path = write_yaml(tmp_path / "in.yaml", load_yaml("gaussian_input.yaml"))
    res_path = write_yaml(tmp_path / "res.yaml", res)
    with pytest.raises(ValueError):
        load_inspector(in_path, res_path, 1e-6)
    with pytest.raises(FileNotFoundError):
        load_inspector(in_path, str(tmp_path / "none.yaml"), 1e-6)


def test_gamma_diagnostic_without_gamma_cell_raises():
    """Spec 2502: an input without a Gamma cell raises ValueError."""
    with pytest.raises(ValueError):
        load_gamma_diagnostic(str(DATA / "gaussian_input.yaml"), str(DATA / "gaussian_result.yaml"), 1e-6)


def test_gamma_diagnostic_model_mismatch_raises(tmp_path):
    """Spec 2456: a result whose model differs at a cell raises ValueError (Gamma diagnostic loader)."""
    res = load_yaml("gamma_ardinf_result.yaml")
    res["model"] = [["gamma"], ["gaussian"]]
    in_path = write_yaml(tmp_path / "in.yaml", load_yaml("gamma_ardinf_input.yaml"))
    res_path = write_yaml(tmp_path / "res.yaml", res)
    with pytest.raises(ValueError):
        load_gamma_diagnostic(in_path, res_path, 1e-6)


def test_gamma_diagnostic_loads_valid_pair():
    """Spec 2493-2502: the overview (repairs, replacements, valid, violations, global violations,
    maximum errors of the replay report); the mean table with the cap; the tail table with the exact
    bound. Pair gamma_ardinf: 2 repairs, 1 replacement, 6 cell-days."""
    state = load_gamma_diagnostic(str(DATA / "gamma_ardinf_input.yaml"),
                                  str(DATA / "gamma_ardinf_result.yaml"), 1e-6)
    ov = state["overview"]
    assert ov["valid"] is True and ov["repairs"] == 2 and ov["replacements"] == 1
    assert ov["violations"] == 0 and list(ov["global_violations"]) == []
    assert "maximum_errors" in ov
    cap = gamma_cap(5.0, 1.0, 0.1)
    means = state["means"]
    assert len(means) == 6
    for _, row in means.iterrows():
        assert near(float(row["cap"]), cap, 1e-6)
        assert near(float(row["margin"]), cap - float(row["mu"]), 1e-6, 1e-9)
        assert bool(row["holds"]) is True
    tails = state["tails"]
    assert len(tails) == 6
    expected = {(1, 1): gsf(.75, 5), (1, 2): gsf(.75, 10), (1, 3): gsf(1.35, 10), (2, 1): gsf(.5, 10),
                (2, 2): conv_tail(.5, 10, .25, 5), (2, 3): 0.0}
    for _, row in tails.iterrows():
        bound = expected[(int(row["train"]), int(row["day"]))]
        assert near(float(row["tail_upper_bound"]), bound, 1e-6, 1e-10)
        assert near(float(row["epsilon"]), 0.1)
        assert near(float(row["slack"]), 0.1 - bound, 1e-6, 1e-10)
        assert bool(row["holds"]) is True


def test_gamma_diagnostic_marks_days_that_do_not_hold(tmp_path):
    """Spec 2502: a day holds if epsilon minus the bound is at least -tolerance. With epsilon = 0.001 the
    train-1 day-1 tail 0.0035261 fails; the cap A1(5, 0.001)/5 = 0.0778 is below the saved means 0.15
    (train 0 day 1), 0.135 (train 0 day 3) and 0.10 (train 1 day 2); 0.075 and 0.05 hold."""
    inp = load_yaml("gamma_ardinf_input.yaml")
    inp["epsilon"] = 0.001
    cfg_in = write_yaml(tmp_path / "in.yaml", inp)
    state = load_gamma_diagnostic(cfg_in, str(DATA / "gamma_ardinf_result.yaml"), 1e-6)
    expected_tails = {(1, 1): gsf(.75, 5), (1, 2): gsf(.75, 10), (1, 3): gsf(1.35, 10),
                      (2, 1): gsf(.5, 10), (2, 2): conv_tail(.5, 10, .25, 5), (2, 3): 0.0}
    holds = {(int(r["train"]), int(r["day"])): bool(r["holds"]) for _, r in state["tails"].iterrows()}
    assert holds == {k: (0.001 - b >= -1e-6) for k, b in expected_tails.items()}
    assert holds[(1, 1)] is False and sum(not h for h in holds.values()) == 1
    cap = gamma_cap(5.0, 1.0, 0.001)
    mu = {(1, 1): 0.15, (1, 2): 0.075, (1, 3): 0.135, (2, 1): 0.05, (2, 2): 0.10, (2, 3): 0.0}
    mean_holds = {(int(r["train"]), int(r["day"])): bool(r["holds"]) for _, r in state["means"].iterrows()}
    assert mean_holds == {k: (cap - m >= -1e-6) for k, m in mu.items()}
    assert state["overview"]["valid"] is False               # the cap row is violated (replay validator)


# ---------------------------------------------------------------------------
# 5. Gamma replay validator (spec 2525-2571, 2507-2523)
# ---------------------------------------------------------------------------
def gamma_ardinf_pair(tmp_path, edit=None, inp_edit=None, name="g"):
    inp, res = load_yaml("gamma_ardinf_input.yaml"), load_yaml("gamma_ardinf_result.yaml")
    if inp_edit:
        inp_edit(inp)
    if edit:
        edit(res)
    return pair(tmp_path, inp, res, name)


def gamma_ard1_pair(tmp_path, edit=None, inp_edit=None, name="h"):
    inp, res = load_yaml("gamma_ard1_input.yaml"), load_yaml("gamma_ard1_result.yaml")
    if inp_edit:
        inp_edit(inp)
    if edit:
        edit(res)
    return pair(tmp_path, inp, res, name)


def test_replay_valid_pairs(tmp_path):
    """Spec 2562-2571: both valid pairs have valid = true, no record, and the counts derived by hand:
    ardinf 2 Gamma cells, 0 ard1 cells, 2 repairs, 1 replacement; ard1 pair 1 repair, 2 replacements."""
    cfg, result, _, _ = gamma_ardinf_pair(tmp_path)
    report = validate_gamma_replay_schedule(cfg, result)
    for key in REPLAY_KEYS:
        assert key in report
    assert report["valid"] is True and report["violations"] == [] and report["global_violations"] == []
    assert report["gamma_cells"] == 2 and report["gamma_ard1_cells"] == 0
    assert report["repairs"] == 2 and report["replacements"] == 1
    assert report["transitions_checked"] == 6                # 2 cells x 3 days
    assert report["solver_status"] == "optimal"
    cfg, result, _, _ = gamma_ard1_pair(tmp_path)
    report = validate_gamma_replay_schedule(cfg, result)
    assert report["valid"] is True and report["violations"] == []
    assert report["gamma_cells"] == 2 and report["gamma_ard1_cells"] == 2
    assert report["repairs"] == 1 and report["replacements"] == 2


def _only_record(report):
    assert report["valid"] is False
    assert len(report["violations"]) == 1, report["violations"]
    rec = report["violations"][0]
    for key in RECORD_KEYS:
        assert key in rec
    return rec


def test_replay_tracked_mean_record(tmp_path):
    """Spec 2554, 2549: saved mu below the reference. Train 1, day 2 (mission): reference 0.10, saved 0.09:
    error 0.01, recorded with 0-based i = 1, l = 0, k = 1 and the event mission."""
    cfg, result, _, _ = gamma_ardinf_pair(tmp_path, lambda r: put(r, "mu", 1, 0, 1, 0.09))
    report = validate_gamma_replay_schedule(cfg, result)
    rec = _only_record(report)
    assert (rec["i"], rec["l"], rec["k"], rec["event"], rec["field"]) == (1, 0, 1, "mission", "tracked_mean")
    assert near(rec["expected"], 0.10, 1e-9) and near(rec["saved"], 0.09)
    assert near(rec["error"], 0.01, 1e-6, 1e-12)
    assert near(report["maximum_errors"]["tracked_mean"], 0.01, 1e-6, 1e-12)


def test_replay_mean_anchor_record(tmp_path):
    """Spec 2555: under ard1 the same for g_mu. Train 1, day 2: reference anchor 0.07, saved 0.06."""
    cfg, result, _, _ = gamma_ard1_pair(tmp_path, lambda r: put(r, "g_mu", 1, 0, 1, 0.06))
    rec = _only_record(validate_gamma_replay_schedule(cfg, result))
    assert (rec["i"], rec["l"], rec["k"], rec["event"], rec["field"]) == (1, 0, 1, "mission", "mean_anchor")
    assert near(rec["expected"], 0.07, 1e-9) and near(rec["saved"], 0.06)
    assert near(rec["error"], 0.01, 1e-6, 1e-12)


def test_replay_removed_mean_record_ardinf(tmp_path):
    """Spec 2556: z below x^m rho (mu_{k-1} - g_{k-1}), g = 0 under ardinf, mu_0 on day 1.
    Train 1, day 1 repair: 0.5*(0.1 - 0) = 0.05; saved z 0.04: error 0.01."""
    cfg, result, _, _ = gamma_ardinf_pair(tmp_path, lambda r: put(r, "z", 1, 0, 0, 0.04))
    rec = _only_record(validate_gamma_replay_schedule(cfg, result))
    assert (rec["i"], rec["l"], rec["k"], rec["event"], rec["field"]) == (1, 0, 0, "repair", "removed_mean")
    assert near(rec["expected"], 0.05, 1e-9) and near(rec["saved"], 0.04)
    assert near(rec["error"], 0.01, 1e-6, 1e-12)


def test_replay_removed_mean_record_ard1(tmp_path):
    """Spec 2556: under ard1 the anchor enters: day 1 repair of train 1: 0.5*(mu_0 - g_0) = 0.5*(0.1 - 0.04) = 0.03;
    saved z 0.02: error 0.01."""
    cfg, result, _, _ = gamma_ard1_pair(tmp_path, lambda r: put(r, "z", 1, 0, 0, 0.02))
    rec = _only_record(validate_gamma_replay_schedule(cfg, result))
    assert (rec["i"], rec["l"], rec["k"], rec["field"]) == (1, 0, 0, "removed_mean")
    assert near(rec["expected"], 0.03, 1e-9) and near(rec["saved"], 0.02)
    assert near(rec["error"], 0.01, 1e-6, 1e-12)


def test_replay_removed_mean_skipped_where_z_is_nan(tmp_path):
    """Spec 2556: skipped where z is NaN (C_R = 0): no record although the repair removes mean."""
    def edit_input(inp):
        inp["C_R"] = 0.0

    def edit(res):
        res["z"] = [[[None, None, None]], [[None, None, None]]]
    cfg, result, _, _ = gamma_ardinf_pair(tmp_path, edit, edit_input)
    report = validate_gamma_replay_schedule(cfg, result)
    assert report["valid"] is True and report["violations"] == []


def test_replay_cap_record(tmp_path):
    """Spec 2557, 1298: mu above A1(beta* tau, eps)/beta*; beta* = min(5, 10, 5) = 5, t = 5, eps = 0.1.
    Train 1, day 2 (mission) saved 0.9: error 0.9 - A1/5, expected = the cap (independent scipy cap)."""
    cfg, result, _, _ = gamma_ardinf_pair(tmp_path, lambda r: put(r, "mu", 1, 0, 1, 0.9))
    rec = _only_record(validate_gamma_replay_schedule(cfg, result))
    cap = gamma_cap(5.0, 1.0, 0.1)
    assert (rec["i"], rec["l"], rec["k"], rec["event"], rec["field"]) == (1, 0, 1, "mission", "cap")
    assert near(rec["expected"], cap, 1e-6)
    assert near(rec["saved"], 0.9)
    assert near(rec["error"], 0.9 - cap, 1e-6, 1e-9)


def test_replay_repeatability_record(tmp_path):
    """Spec 2558, 2560: mu_T above mu_H1: train 0, day 3 saved 0.16 against 0.15 at day H1: error 0.01,
    recorded on day T (k = 2) with its event mission; expected = the saved value on day H1."""
    cfg, result, _, _ = gamma_ardinf_pair(tmp_path, lambda r: put(r, "mu", 0, 0, 2, 0.16))
    rec = _only_record(validate_gamma_replay_schedule(cfg, result))
    assert (rec["i"], rec["l"], rec["k"], rec["event"], rec["field"]) == (0, 0, 2, "mission", "repeatability")
    assert near(rec["expected"], 0.15) and near(rec["saved"], 0.16)
    assert near(rec["error"], 0.01, 1e-6, 1e-12)


def test_replay_repeatability_of_the_anchor(tmp_path):
    """Spec 2558: the same for the saved anchor. Train 0 (ard1), g_T = 0.05 against g_H1 = 0.04: error 0.01.
    The anchor above the reference is not a tracked-anchor violation (one-sided, spec 2555)."""
    cfg, result, _, _ = gamma_ard1_pair(tmp_path, lambda r: put(r, "g_mu", 0, 0, 2, 0.05))
    rec = _only_record(validate_gamma_replay_schedule(cfg, result))
    assert (rec["i"], rec["l"], rec["k"], rec["field"]) == (0, 0, 2, "repeatability")
    assert near(rec["expected"], 0.04) and near(rec["saved"], 0.05)
    assert near(rec["error"], 0.01, 1e-6, 1e-12)


def test_replay_tolerance_decides(tmp_path):
    """Spec 2549: a record only if the error is above tolerance: error 0.01 is no violation at tolerance 0.02."""
    cfg, result, _, _ = gamma_ardinf_pair(tmp_path, lambda r: put(r, "mu", 1, 0, 1, 0.09))
    assert validate_gamma_replay_schedule(cfg, result, tolerance=0.02)["valid"] is True
    assert validate_gamma_replay_schedule(cfg, result, tolerance=0.005)["valid"] is False


def test_replay_global_violation_more_than_one_activity(tmp_path):
    """Spec 2541: train 0 at the depot and flying on day 1: one string; the event of the day is mission."""
    cfg, result, _, _ = gamma_ardinf_pair(tmp_path, lambda r: put(r, "x", 0, 0, 0, 1))
    report = validate_gamma_replay_schedule(cfg, result)
    assert len(report["global_violations"]) == 1 and report["valid"] is False
    assert all(isinstance(s, str) for s in report["global_violations"])
    assert report["violations"] == []


def test_replay_global_violation_mission_not_covered(tmp_path):
    """Spec 2542: nobody flies on day 1: one string (the validator does not stop)."""
    cfg, result, _, _ = gamma_ardinf_pair(tmp_path, lambda r: put(r, "x", 0, 1, 0, 0))
    report = validate_gamma_replay_schedule(cfg, result)
    assert len(report["global_violations"]) == 1 and report["valid"] is False
    assert report["violations"] == []


def test_replay_global_violation_repair_and_replacement_together(tmp_path):
    """Spec 2543: train 1 repairs and replaces on day 3: one string. The event is replacement (spec 2514),
    so the repair row reads rho*(mu_2 - 0) = 0.05; z[1][0][2] = 0.05 makes that error 0."""
    def edit(res):
        put(res, "m", 1, 0, 2, 1)
        put(res, "z", 1, 0, 2, 0.05)
    cfg, result, _, _ = gamma_ardinf_pair(tmp_path, edit)
    report = validate_gamma_replay_schedule(cfg, result)
    assert len(report["global_violations"]) == 1 and report["valid"] is False
    assert report["violations"] == []


def test_replay_global_violation_intervention_on_mission_day(tmp_path):
    """Spec 2544: a repair of train 0 on day 1 while it flies is a global violation. The event is repair; the
    repair row 0.5*(mu_0 - 0) = 0.05 is met by z = 0.05. The day has a mission and no depot day, so both
    reasons hold; the spec gives one string per occurrence (line 2538), so exactly one string."""
    def edit(res):
        put(res, "m", 0, 0, 0, 1)
        put(res, "z", 0, 0, 0, 0.05)
    cfg, result, _, _ = gamma_ardinf_pair(tmp_path, edit)
    report = validate_gamma_replay_schedule(cfg, result)
    assert len(report["global_violations"]) == 1 and report["valid"] is False
    assert report["violations"] == []


def test_replay_global_violation_horizon(tmp_path):
    """Spec 2540: a result horizon other than T: one string. (T = 4 saved against H1 + H2 = 3.)"""
    cfg, result, _, _ = gamma_ardinf_pair(tmp_path, lambda r: r.__setitem__("T", 4))
    report = validate_gamma_replay_schedule(cfg, result)
    assert len(report["global_violations"]) == 1 and report["valid"] is False


def test_replay_raise_on_failure(tmp_path):
    """Spec 2522: raise_on_failure=True and valid = false raise AssertionError; a valid report does not raise."""
    cfg, result, _, _ = gamma_ardinf_pair(tmp_path, lambda r: put(r, "mu", 1, 0, 1, 0.09))
    with pytest.raises(AssertionError):
        validate_gamma_replay_schedule(cfg, result, raise_on_failure=True)
    assert validate_gamma_replay_schedule(cfg, result)["valid"] is False
    cfg, result, _, _ = gamma_ardinf_pair(tmp_path, name="ok")
    assert validate_gamma_replay_schedule(cfg, result, raise_on_failure=True)["valid"] is True


@pytest.mark.parametrize("tolerance", [0.0, -1e-8, -1.0])
def test_validators_reject_non_positive_tolerance(tmp_path, tolerance):
    """Spec 2512: tolerance must be positive (replay, exact, Monte Carlo and dispatcher)."""
    cfg, result, _, _ = gamma_ardinf_pair(tmp_path)
    with pytest.raises(ValueError):
        validate_gamma_replay_schedule(cfg, result, tolerance=tolerance)
    with pytest.raises(ValueError):
        validate_gamma_tail_bound_schedule(cfg, result, tolerance=tolerance)
    with pytest.raises(ValueError):
        validate_gamma_stochastic_schedule(cfg, result, repetitions=100, tolerance=tolerance)
    with pytest.raises(ValueError):
        validate_gamma_schedule(cfg, result, tolerance=tolerance)


def test_replay_errors_on_inputs(tmp_path):
    """Spec 2521: ValueError when the result has no model, a model of another shape or one that differs
    from the input; a required array is missing, wrongly shaped or not finite where read; a binary
    array is not within tolerance of {0,1}."""
    cases = {
        "no model": lambda r: r.pop("model"),
        "model shape": lambda r: r.__setitem__("model", [["gamma"]]),
        "model differs": lambda r: r.__setitem__("model", [["gamma"], ["gaussian"]]),
        "missing mu": lambda r: r.pop("mu"),
        "missing x": lambda r: r.pop("x"),
        "missing m": lambda r: r.pop("m"),
        "mu shape": lambda r: r.__setitem__("mu", [[[0.15, 0.075]], [[0.05, 0.10]]]),
        "mu not finite": lambda r: put(r, "mu", 0, 0, 1, None),
        "x not binary": lambda r: put(r, "x", 0, 1, 0, 0.5),
        "m not binary": lambda r: put(r, "m", 0, 0, 1, 0.5),
    }
    for label, edit in cases.items():
        cfg, result, _, _ = gamma_ardinf_pair(tmp_path, edit, name="e")
        with pytest.raises(ValueError):
            validate_gamma_replay_schedule(cfg, result)


def test_replay_replacement_array_is_optional(tmp_path):
    """Spec 2527: r is optional; a result without r does not raise."""
    cfg, result, _, _ = gamma_ardinf_pair(tmp_path, lambda r: r.pop("r"))
    report = validate_gamma_replay_schedule(cfg, result)
    assert "valid" in report


def test_replay_input_without_gamma_cell_raises(tmp_path):
    """Spec 2521: cfg has no Gamma cell."""
    cfg, result, _, _ = pair(tmp_path, load_yaml("gaussian_input.yaml"), load_yaml("gaussian_result.yaml"))
    with pytest.raises(ValueError):
        validate_gamma_replay_schedule(cfg, result)
    with pytest.raises(ValueError):
        validate_gamma_tail_bound_schedule(cfg, result)
    with pytest.raises(ValueError):
        validate_gamma_stochastic_schedule(cfg, result, repetitions=10)


def test_replay_files_report_and_suffix_order(tmp_path):
    """Spec 2511, 2516: the file version writes the report (YAML or JSON), creates the folder, and checks the
    suffix of report_path before it reads a file (a missing input with a bad suffix is a ValueError)."""
    in_path, res_path = str(DATA / "gamma_ardinf_input.yaml"), str(DATA / "gamma_ardinf_result.yaml")
    report = validate_gamma_replay_files(in_path, res_path, str(tmp_path / "x" / "y" / "rep.json"))
    with open(tmp_path / "x" / "y" / "rep.json", encoding="utf-8") as handle:
        saved = json.load(handle, parse_constant=lambda c: (_ for _ in ()).throw(AssertionError(c)))
    assert saved["valid"] is True and saved["repairs"] == report["repairs"]
    validate_gamma_replay_files(in_path, res_path, str(tmp_path / "z" / "rep.yaml"))
    with open(tmp_path / "z" / "rep.yaml", encoding="utf-8") as handle:
        assert yaml.safe_load(handle)["valid"] is True
    with pytest.raises(ValueError):
        validate_gamma_replay_files(str(tmp_path / "missing.yaml"), res_path, str(tmp_path / "rep.txt"))
    with pytest.raises(ValueError):
        validate_gamma_tail_bound_files(str(tmp_path / "missing.yaml"), res_path, str(tmp_path / "rep.txt"))
    with pytest.raises(ValueError):
        validate_gamma_schedule_files(str(tmp_path / "missing.yaml"), res_path, str(tmp_path / "rep.txt"))
    with pytest.raises(FileNotFoundError):
        validate_gamma_replay_files(str(tmp_path / "missing.yaml"), res_path)


def test_replay_files_write_before_raising(tmp_path):
    """Spec 2516: the report is written before the AssertionError is raised (and holds no NaN token)."""
    res = load_yaml("gamma_ardinf_result.yaml")
    put(res, "mu", 1, 0, 1, 0.09)
    in_path = str(DATA / "gamma_ardinf_input.yaml")
    res_path = write_yaml(tmp_path / "bad.yaml", res)
    out = tmp_path / "rep" / "report.json"
    with pytest.raises(AssertionError):
        validate_gamma_replay_files(in_path, res_path, str(out), raise_on_failure=True)
    with open(out, encoding="utf-8") as handle:
        text = handle.read()
    assert "NaN" not in text and "Infinity" not in text
    saved = json.loads(text)
    assert saved["valid"] is False and len(saved["violations"]) == 1
    out_yaml = tmp_path / "rep" / "report.yaml"
    with pytest.raises(AssertionError):
        validate_gamma_replay_files(in_path, res_path, str(out_yaml), raise_on_failure=True)
    assert out_yaml.exists()


# ---------------------------------------------------------------------------
# 6. Events of the days (spec 2514)
# ---------------------------------------------------------------------------
def test_events_of_the_days_in_every_validator(tmp_path):
    """Spec 2514: replacement, else repair, else mission (train flies a mission), else idle; also on a day
    with a global violation. Pair gamma_ardinf: train 0 mission, repair, mission; train 1 repair, mission,
    replacement. Variant: nobody flies day 1 (train 0 has no activity) -> train 0's event on day 1 is idle."""
    cfg, result, _, _ = gamma_ardinf_pair(tmp_path)
    steps = validate_gamma_tail_bound_schedule(cfg, result, include_steps=True)["steps"]
    events = {(s["i"], s["k"]): s["event"] for s in steps}
    assert events == {(0, 0): "mission", (0, 1): "repair", (0, 2): "mission",
                      (1, 0): "repair", (1, 1): "mission", (1, 2): "replacement"}
    cfg, result, _, _ = gamma_ardinf_pair(tmp_path, lambda r: put(r, "x", 0, 1, 0, 0), name="idle")
    steps = validate_gamma_tail_bound_schedule(cfg, result, include_steps=True)["steps"]
    events = {(s["i"], s["k"]): s["event"] for s in steps}
    assert events[(0, 0)] == "idle"
    # the replay validator applies the same events: a tracked-mean error on that idle day reports it
    def edit(r):
        put(r, "x", 0, 1, 0, 0)
        put(r, "mu", 0, 0, 0, 0.05)       # reference on an idle day: mu_0 = 0.1, saved 0.05
    cfg, result, _, _ = gamma_ardinf_pair(tmp_path, edit, name="idle2")
    report = validate_gamma_replay_schedule(cfg, result)
    recs = [v for v in report["violations"] if v["field"] == "tracked_mean"]
    assert len(recs) == 1 and recs[0]["event"] == "idle" and recs[0]["k"] == 0
    assert near(recs[0]["expected"], 0.1) and near(recs[0]["saved"], 0.05)
    # repair and replacement on one day: replacement
    def edit2(r):
        put(r, "m", 1, 0, 2, 1)
        put(r, "z", 1, 0, 2, 0.05)
    cfg, result, _, _ = gamma_ardinf_pair(tmp_path, edit2, name="both")
    steps = validate_gamma_tail_bound_schedule(cfg, result, include_steps=True)["steps"]
    assert {(s["i"], s["k"]): s["event"] for s in steps}[(1, 2)] == "replacement"


# ---------------------------------------------------------------------------
# 7. Exact validator (spec 2573-2630)
# ---------------------------------------------------------------------------
EXPECTED_TAILS_ARDINF = {   # terms by hand (see the comment of gamma_ardinf_result.yaml), threshold tau = 1
    (0, 0): lambda: gsf(.75, 5),             # Gamma(.5,5) + Gamma(.25,5)
    (0, 1): lambda: gsf(.75, 10),            # repair: both rates x 1/(1-rho) = 2
    (0, 2): lambda: gsf(1.35, 10),           # + Gamma(.6, 10)
    (1, 0): lambda: gsf(.5, 10),             # repair: Gamma(.5, 10)
    (1, 1): lambda: conv_tail(.5, 10, .25, 5),    # + Gamma(.25, 5): two rates
    (1, 2): lambda: 0.0,                     # replacement by the constant 0: no term, c = 0 <= tau
}
EXPECTED_TAILS_ARD1 = {
    (0, 0): lambda: gsf(.75, 5),             # Gamma(.2,5) frozen + Gamma(.3,5) + Gamma(.25,5)
    (0, 1): lambda: 0.0,                     # replacement
    (0, 2): lambda: gsf(.6, 10),             # Gamma(.6, 10)
    (1, 0): lambda: conv_tail(.2, 5, .3, 10),     # repair: Gamma(.3,5) -> Gamma(.3,10), frozen
    (1, 1): lambda: conv_tail(.45, 5, .3, 10),    # + Gamma(.25, 5)
    (1, 2): lambda: 0.0,
}


def _steps(report):
    return {(s["i"], s["k"]): s for s in report["steps"]}


def test_exact_validator_tails_against_scipy_ardinf(tmp_path):
    """Spec 2593-2604: tails of every day of the ardinf pair: one Gamma term (scipy.stats.gamma.sf), equal
    rates, two rates (scipy.integrate.quad of the convolution), a repair multiplying the rates by
    1/(1-rho) = 2, a replacement by a constant. The upper bound is the estimate plus a mass <= 1e-12."""
    cfg, result, _, _ = gamma_ardinf_pair(tmp_path)
    report = validate_gamma_tail_bound_schedule(cfg, result, include_steps=True)
    steps = _steps(report)
    assert len(steps) == 6
    for key, make in EXPECTED_TAILS_ARDINF.items():
        expected = make()
        s = steps[key]
        assert abs(s["exact_tail_estimate"] - expected) <= 1e-10 + 1e-8 * expected, (key, s, expected)
        assert expected - 1e-10 <= s["exact_tail_upper_bound"] <= expected + 1e-10 + 1e-8 * expected
        assert s["exact_tail_upper_bound"] >= s["exact_tail_estimate"]
        assert 0.0 <= s["remaining_mass"] <= 1e-12 + 1e-15
        assert near(s["epsilon"], 0.1)
        assert near(s["reliability_slack"], 0.1 - s["exact_tail_upper_bound"], 1e-9, 1e-12)
    for key, mean in {(0, 0): 0.15, (0, 1): 0.075, (0, 2): 0.135, (1, 0): 0.05, (1, 1): 0.10, (1, 2): 0.0}.items():
        assert near(steps[key]["exact_mean"], mean, 1e-9, 1e-12)      # by the day map of the result file
        assert near(steps[key]["saved_mean"], mean, 1e-9, 1e-12)
    assert report["valid"] is True and report["violations"] == []


def test_exact_validator_tails_against_scipy_ard1(tmp_path):
    """Spec 2599-2601: ard1 frozen part. The repaired terms join the frozen part (train 1 day 1: Gamma(.2,5)
    + Gamma(.3,10), frozen mean 0.04 + 0.03 = 0.07); a replacement term is frozen (constant 0, mean 0)."""
    cfg, result, _, _ = gamma_ard1_pair(tmp_path)
    report = validate_gamma_tail_bound_schedule(cfg, result, include_steps=True)
    steps = _steps(report)
    for key, make in EXPECTED_TAILS_ARD1.items():
        expected = make()
        s = steps[key]
        assert abs(s["exact_tail_estimate"] - expected) <= 1e-10 + 1e-8 * expected, (key, s, expected)
        assert expected - 1e-10 <= s["exact_tail_upper_bound"] <= expected + 1e-10 + 1e-8 * expected
    means = {(0, 0): 0.15, (0, 1): 0.0, (0, 2): 0.06, (1, 0): 0.07, (1, 1): 0.12, (1, 2): 0.0}
    frozen = {(0, 0): 0.04, (0, 1): 0.0, (0, 2): 0.0, (1, 0): 0.07, (1, 1): 0.07, (1, 2): 0.0}
    for key in means:
        assert near(steps[key]["exact_mean"], means[key], 1e-9, 1e-12)
        assert near(steps[key]["frozen_mean"], frozen[key], 1e-9, 1e-12)
    anchors = {(0, 0): 0.04, (0, 1): 0.0, (0, 2): 0.0, (1, 0): 0.07, (1, 1): 0.07, (1, 2): 0.0}
    for key, a in anchors.items():
        assert near(steps[key]["saved_anchor"], a, 1e-9, 1e-12)
    assert report["valid"] is True


def test_exact_validator_known_constant_shifts_the_threshold(tmp_path):
    """Spec 2602: without gamma_beta_0 the initial damage is the known constant c = 0.1 and the tail is
    Pr(sum of the terms > tau - c). Train 0 day 1: terms Gamma(.25, 5): Pr(> 0.9). Day 2 (repair):
    the constant scales by 1 - rho (c = 0.05) and the term rate by 2: Gamma(.25, 10), Pr(> 0.95)."""
    def edit_input(inp):
        del inp["gamma_beta_0"]
    cfg, result, _, _ = gamma_ardinf_pair(tmp_path, inp_edit=edit_input)
    steps = _steps(validate_gamma_tail_bound_schedule(cfg, result, include_steps=True))
    s0, s1 = steps[(0, 0)], steps[(0, 1)]
    assert near(s0["constant"], 0.1, 1e-9)
    assert abs(s0["exact_tail_estimate"] - gsf(.25, 5, 0.9)) <= 1e-10 + 1e-8 * gsf(.25, 5, 0.9)
    assert near(s1["constant"], 0.05, 1e-9)
    assert abs(s1["exact_tail_estimate"] - gsf(.25, 10, 0.95)) <= 1e-10 + 1e-8 * gsf(.25, 10, 0.95)
    assert near(s0["exact_mean"], 0.15, 1e-9) and near(s1["exact_mean"], 0.075, 1e-9)


def test_exact_validator_constant_not_below_tau(tmp_path):
    """Spec 2602: the tail is 1 if c >= tau and some term exists; without terms it is 1 if c > tau, else 0.
    The constants are set on the loaded configuration (an input check forbids them). Initial damage
    c = tau = 1: day 1 of train 0 has a term: tail 1. Replacement constant tau on train 1's day 3: no
    term, c = tau: tail 0; replacement constant 1.2 > tau: tail 1."""
    def edit_input(inp):
        del inp["gamma_beta_0"]
    cfg, result, _, _ = gamma_ardinf_pair(tmp_path, inp_edit=edit_input)
    cfg1 = copy.deepcopy(cfg)
    cfg1.mu_0[0, 0] = 1.0
    steps = _steps(validate_gamma_tail_bound_schedule(cfg1, result, include_steps=True))
    assert near(steps[(0, 0)]["exact_tail_estimate"], 1.0) and near(steps[(0, 0)]["exact_tail_upper_bound"], 1.0)
    cfg2 = copy.deepcopy(cfg)
    cfg2.replacement_mu[1, 0] = 1.0
    steps = _steps(validate_gamma_tail_bound_schedule(cfg2, result, include_steps=True))
    assert near(steps[(1, 2)]["constant"], 1.0)
    assert steps[(1, 2)]["exact_tail_estimate"] == 0.0 and steps[(1, 2)]["exact_tail_upper_bound"] == 0.0
    cfg3 = copy.deepcopy(cfg)
    cfg3.replacement_mu[1, 0] = 1.2
    steps = _steps(validate_gamma_tail_bound_schedule(cfg3, result, include_steps=True))
    assert near(steps[(1, 2)]["constant"], 1.2)
    assert near(steps[(1, 2)]["exact_tail_estimate"], 1.0) and near(steps[(1, 2)]["exact_tail_upper_bound"], 1.0)


def test_exact_validator_report_keys_and_include_steps(tmp_path):
    """Spec 2622-2630: the keys of the replay report are all included; minimum slack and worst step;
    include_steps False gives no list; the keys of a step."""
    cfg, result, _, _ = gamma_ardinf_pair(tmp_path)
    plain = validate_gamma_tail_bound_schedule(cfg, result)
    assert "steps" not in plain
    report = validate_gamma_tail_bound_schedule(cfg, result, include_steps=True)
    for key in REPLAY_KEYS + ["minimum_reliability_slack", "worst_reliability", "maximum_remaining_mass",
                              "maximum_series_terms_used", "total_series_terms", "exact_tail_evaluations", "steps"]:
        assert key in report, key
    assert "exact_tail_seconds" in report["timing"]
    for s in report["steps"]:
        for key in STEP_KEYS:
            assert key in s, key
    assert len(report["steps"]) == 6
    # minimum of epsilon minus the upper bound: the largest bound is train 0 day 1, 0.0035261
    worst = max(s["exact_tail_upper_bound"] for s in report["steps"])
    assert near(report["minimum_reliability_slack"], 0.1 - worst, 1e-9, 1e-12)
    assert near(report["minimum_reliability_slack"], 0.1 - gsf(.75, 5), 1e-6, 1e-9)
    assert (report["worst_reliability"]["i"], report["worst_reliability"]["k"]) == (0, 0)
    assert report["maximum_remaining_mass"] <= 1e-12 + 1e-15
    assert report["maximum_series_terms_used"] <= report["total_series_terms"]
    assert report["exact_tail_evaluations"] >= 1


def test_exact_validator_reliability_record(tmp_path):
    """Spec 2611, 2615: with epsilon = 0.001 only train 0 day 1 (tail 0.0035261) fails; the record has
    field reliability, expected epsilon, saved the upper bound (spec 2615), 0-based i, l, k."""
    def edit_input(inp):
        inp["epsilon"] = 0.001
    cfg, result, _, _ = gamma_ardinf_pair(tmp_path, inp_edit=edit_input)
    report = validate_gamma_tail_bound_schedule(cfg, result, include_steps=True)
    assert report["valid"] is False
    recs = [v for v in report["violations"] if v["field"] == "reliability"]
    assert len(report["violations"]) == 1 and len(recs) == 1
    rec = recs[0]
    assert (rec["i"], rec["l"], rec["k"], rec["event"]) == (0, 0, 0, "mission")
    assert near(rec["expected"], 0.001)
    assert abs(rec["saved"] - gsf(.75, 5)) <= 1e-10 + 1e-8 * gsf(.75, 5)
    assert rec["error"] > 0 and near(rec["error"], rec["saved"] - 0.001, 1e-6, 1e-12)
    for key in ("reliability", "tracked_mean", "mean_anchor"):
        assert key in report["maximum_errors"]
    assert near(report["maximum_errors"]["reliability"], gsf(.75, 5) - 0.001, 1e-6, 1e-9)
    assert near(report["minimum_reliability_slack"], 0.001 - gsf(.75, 5), 1e-6, 1e-9)
    assert (report["worst_reliability"]["i"], report["worst_reliability"]["k"]) == (0, 0)


def test_exact_validator_pass_rule_means(tmp_path):
    """Spec 2612-2615: the saved mean must be at least E D_k - tolerance. Train 1 day 2: exact mean 0.10
    (0.05 + 0.05), saved 0.09: a tracked_mean record, expected E D_k. Under ard1, a saved anchor below the
    exact mean of the frozen part (train 1, day 2: 0.07) gives a mean_anchor record."""
    cfg, result, _, _ = gamma_ardinf_pair(tmp_path, lambda r: put(r, "mu", 1, 0, 1, 0.09))
    report = validate_gamma_tail_bound_schedule(cfg, result)
    recs = [v for v in report["violations"] if v["field"] == "tracked_mean"]
    assert report["valid"] is False and len(recs) == 1 and len(report["violations"]) == 1
    assert (recs[0]["i"], recs[0]["l"], recs[0]["k"]) == (1, 0, 1)
    assert near(recs[0]["expected"], 0.10, 1e-9) and near(recs[0]["saved"], 0.09)
    assert near(recs[0]["error"], 0.01, 1e-6, 1e-12)
    assert near(report["maximum_errors"]["tracked_mean"], 0.01, 1e-6, 1e-12)
    cfg, result, _, _ = gamma_ard1_pair(tmp_path, lambda r: put(r, "g_mu", 1, 0, 1, 0.06), name="a")
    report = validate_gamma_tail_bound_schedule(cfg, result)
    recs = [v for v in report["violations"] if v["field"] == "mean_anchor"]
    assert len(recs) == 1 and len(report["violations"]) == 1
    assert (recs[0]["i"], recs[0]["k"]) == (1, 1)
    assert near(recs[0]["expected"], 0.07, 1e-9) and near(recs[0]["saved"], 0.06)
    assert near(report["maximum_errors"]["mean_anchor"], 0.01, 1e-6, 1e-12)


def test_exact_validator_global_violation_makes_it_invalid(tmp_path):
    """Spec 2615: valid is true if there is no global violation and every day passes."""
    cfg, result, _, _ = gamma_ardinf_pair(tmp_path, lambda r: put(r, "x", 0, 0, 0, 1))
    report = validate_gamma_tail_bound_schedule(cfg, result)
    assert len(report["global_violations"]) == 1 and report["valid"] is False


def test_exact_validator_series_term_limit(tmp_path):
    """Spec 2619: RuntimeError when the series does not reach convolution_tolerance within max_series_terms.
    Train 1 day 2 convolves two different rates (10 and 5): one term is not enough."""
    cfg, result, _, _ = gamma_ardinf_pair(tmp_path)
    with pytest.raises(RuntimeError):
        validate_gamma_tail_bound_schedule(cfg, result, max_series_terms=1)
    assert validate_gamma_tail_bound_schedule(cfg, result, max_series_terms=100000)["valid"] is True


@pytest.mark.parametrize("name, value", [("convolution_tolerance", 0.0), ("convolution_tolerance", 1.0),
                                         ("convolution_tolerance", -0.5), ("convolution_tolerance", 1.5),
                                         ("max_series_terms", 0), ("max_series_terms", -3)])
def test_exact_validator_argument_errors(tmp_path, name, value):
    """Spec 2587-2588: convolution_tolerance in (0,1); max_series_terms positive."""
    cfg, result, _, _ = gamma_ardinf_pair(tmp_path)
    with pytest.raises(ValueError):
        validate_gamma_tail_bound_schedule(cfg, result, **{name: value})


def test_exact_validator_files_and_raise(tmp_path):
    """Spec 2516, 2522: the file version writes the report, then raises when raise_on_failure is true."""
    res = load_yaml("gamma_ardinf_result.yaml")
    put(res, "mu", 1, 0, 1, 0.09)
    in_path = str(DATA / "gamma_ardinf_input.yaml")
    res_path = write_yaml(tmp_path / "bad.yaml", res)
    out = tmp_path / "t" / "tail.yaml"
    with pytest.raises(AssertionError):
        validate_gamma_tail_bound_files(in_path, res_path, str(out), raise_on_failure=True,
                                        include_steps=True)
    with open(out, encoding="utf-8") as handle:
        saved = yaml.safe_load(handle)
    assert saved["valid"] is False and len(saved["steps"]) == 6
    ok = validate_gamma_tail_bound_files(in_path, str(DATA / "gamma_ardinf_result.yaml"),
                                         str(tmp_path / "t" / "ok.json"))
    assert ok["valid"] is True


# ---------------------------------------------------------------------------
# 8. Monte Carlo validator (spec 2632-2680)
# ---------------------------------------------------------------------------
def mixed_pair(tmp_path, edit=None):
    inp, res = load_yaml("mixed_mc_input.yaml"), load_yaml("mixed_mc_result.yaml")
    if edit:
        edit(res)
    return pair(tmp_path, inp, res, "m")


# the damage of the Gamma cell (0,0) on day 3 is the constant 0.1 plus Gamma(0.9, 2) (spec 2655-2656, three
# missions of shape 0.3, rate 2) and only grows; the schedule fails when it exceeds tau = 1 on day 3
MC_EXACT_TAIL = float(stats.gamma.sf(0.9, 0.9, scale=0.5))


def test_stochastic_failure_rate_matches_exact_tail(tmp_path):
    """Spec 2655-2659, 2665: the failure rate of the one Gamma cell is within 4 standard errors of the
    exact tail Pr(0.1 + Gamma(0.9, 2) > 1) = 0.1403 (seeded, 20000 runs); schedule and cell agree."""
    cfg, result, _, _ = mixed_pair(tmp_path)
    report = validate_gamma_stochastic_schedule(cfg, result, repetitions=20000, random_seed=11)
    n = 20000
    se = math.sqrt(MC_EXACT_TAIL * (1 - MC_EXACT_TAIL) / n)
    assert abs(report["schedule"]["failure_rate"] - MC_EXACT_TAIL) <= 4 * se
    cell = report["cells"][0]
    assert (cell["i"], cell["l"]) == (0, 0)
    assert cell["failed_replays"] == report["schedule"]["failed_replays"]
    assert near(cell["failure_rate"], cell["failed_replays"] / n, 1e-12)
    assert near(cell["threshold"], 1.0) and near(cell["analytical_epsilon"], 0.5)
    assert cell["component"] == "GammaPart"
    assert report["scope"] == "Gamma cells only"


def test_stochastic_report_keys_excluded_cells_and_scope(tmp_path):
    """Spec 2673-2679: the mixed fleet has three other cells (gaussian), so the validation is not complete
    and excluded_cells lists each with i, l, component, model, reason."""
    cfg, result, _, _ = mixed_pair(tmp_path)
    report = validate_gamma_stochastic_schedule(cfg, result, repetitions=200)
    for key in ("validator", "mode", "valid", "timing", "scope", "complete_fleet_validation",
                "excluded_cells", "schedule", "cells"):
        assert key in report
    assert report["mode"] == "stochastic"
    assert report["complete_fleet_validation"] is False
    ex = {(e["i"], e["l"]): e for e in report["excluded_cells"]}
    assert set(ex) == {(0, 1), (1, 0), (1, 1)}
    assert ex[(0, 1)]["component"] == "GaussPart" and ex[(1, 0)]["component"] == "GammaPart"
    assert all(e["model"] == "gaussian" and e["reason"] for e in ex.values())
    for key in ("failed_replays", "failure_rate", "failure_rate_upper_confidence_bound",
                "maximum_failure_rate", "observed_target_met", "confidence_qualified_target_met"):
        assert key in report["schedule"]
    for key in ("i", "l", "component", "repair_model", "threshold", "analytical_epsilon", "failed_replays",
                "failure_rate", "failure_rate_upper_confidence_bound", "worst_time_step",
                "maximum_sampled_damage", "maximum_threshold_excess"):
        assert key in report["cells"][0], key
    # a pure Gamma fleet is complete
    cfg2, result2, _, _ = gamma_ardinf_pair(tmp_path)
    report2 = validate_gamma_stochastic_schedule(cfg2, result2, repetitions=200)
    assert report2["complete_fleet_validation"] is True and report2["excluded_cells"] == []
    assert len(report2["cells"]) == 2


def test_stochastic_same_arguments_same_report(tmp_path):
    """Spec 2646: the same arguments give the same report (the timing differs)."""
    cfg, result, _, _ = mixed_pair(tmp_path)
    kwargs = dict(repetitions=3000, random_seed=5, batch_size=700)
    a = validate_gamma_stochastic_schedule(cfg, result, **kwargs)
    b = validate_gamma_stochastic_schedule(cfg, result, **kwargs)
    a.pop("timing"), b.pop("timing")
    assert a == b
    c = validate_gamma_stochastic_schedule(cfg, result, **dict(kwargs, random_seed=6))
    assert c["schedule"]["failed_replays"] != a["schedule"]["failed_replays"]


def test_stochastic_clopper_pearson_bound(tmp_path):
    """Spec 2666: the bound is the confidence-quantile of Beta(f+1, n-f), and 1 if f = n."""
    cfg, result, _, _ = mixed_pair(tmp_path)
    for confidence in (0.95, 0.99, 0.5):
        report = validate_gamma_stochastic_schedule(cfg, result, repetitions=4000, random_seed=3,
                                                    confidence_level=confidence)
        f, n = report["schedule"]["failed_replays"], 4000
        bound = float(stats.beta.ppf(confidence, f + 1, n - f))
        assert near(report["schedule"]["failure_rate_upper_confidence_bound"], bound, 1e-9)
        assert near(report["cells"][0]["failure_rate_upper_confidence_bound"], bound, 1e-9)
        assert bound >= f / n
    # f = n: a known initial damage above tau (set on the configuration) fails every run
    cfg_all = copy.deepcopy(cfg)
    cfg_all.mu_0[0, 0] = 2.0
    report = validate_gamma_stochastic_schedule(cfg_all, result, repetitions=300)
    assert report["schedule"]["failed_replays"] == 300
    assert report["schedule"]["failure_rate_upper_confidence_bound"] == 1.0
    assert report["cells"][0]["failure_rate"] == 1.0


def test_stochastic_pass_rule(tmp_path):
    """Spec 2669: valid if no target is given, or the observed failure rate is at most the target.
    The observed rate is about 0.14: target 0.9 passes, target 0.01 fails."""
    cfg, result, _, _ = mixed_pair(tmp_path)
    assert validate_gamma_stochastic_schedule(cfg, result, repetitions=2000)["valid"] is True
    ok = validate_gamma_stochastic_schedule(cfg, result, repetitions=2000, maximum_schedule_failure_rate=0.9)
    assert ok["valid"] is True and ok["schedule"]["observed_target_met"] is True
    assert near(ok["schedule"]["maximum_failure_rate"], 0.9)
    bad = validate_gamma_stochastic_schedule(cfg, result, repetitions=2000, maximum_schedule_failure_rate=0.01)
    assert bad["valid"] is False and bad["schedule"]["observed_target_met"] is False
    with pytest.raises(AssertionError):
        validate_gamma_stochastic_schedule(cfg, result, repetitions=2000,
                                           maximum_schedule_failure_rate=0.01, raise_on_failure=True)


def test_stochastic_no_error_for_global_violation(tmp_path):
    """Spec 2661: no error for a global violation of the schedule; each day applies its event."""
    cfg, result, _, _ = gamma_ardinf_pair(tmp_path, lambda r: put(r, "x", 0, 1, 0, 0))
    report = validate_gamma_stochastic_schedule(cfg, result, repetitions=200)
    assert report["valid"] is True


@pytest.mark.parametrize("name, value", [
    ("repetitions", True), ("repetitions", 0), ("repetitions", -5), ("repetitions", 2.5),
    ("random_seed", True), ("random_seed", 1.5), ("batch_size", True), ("batch_size", 0),
    ("batch_size", 2.5), ("confidence_level", 1.0), ("confidence_level", 0.0),
    ("confidence_level", 1.5), ("maximum_schedule_failure_rate", 1.5),
    ("maximum_schedule_failure_rate", -0.1)])
def test_stochastic_argument_errors(tmp_path, name, value):
    """Spec 2645-2649: a flag, zero, a negative or a fractional count, a confidence level outside (0,1) and
    a target outside [0,1] raise ValueError."""
    cfg, result, _, _ = mixed_pair(tmp_path)
    kwargs = {"repetitions": 100}
    kwargs[name] = value
    with pytest.raises(ValueError):
        validate_gamma_stochastic_schedule(cfg, result, **kwargs)


def test_stochastic_accepts_integral_floats(tmp_path):
    """Spec 2645-2649: a number with zero fractional part is accepted for repetitions, seed and batch size;
    targets 0 and 1 are in [0, 1]."""
    cfg, result, _, _ = mixed_pair(tmp_path)
    a = validate_gamma_stochastic_schedule(cfg, result, repetitions=100.0, random_seed=7.0, batch_size=50.0)
    b = validate_gamma_stochastic_schedule(cfg, result, repetitions=100, random_seed=7, batch_size=50)
    a.pop("timing"), b.pop("timing")
    assert a == b
    validate_gamma_stochastic_schedule(cfg, result, repetitions=100, maximum_schedule_failure_rate=0.0)
    validate_gamma_stochastic_schedule(cfg, result, repetitions=100, maximum_schedule_failure_rate=1.0)


# ---------------------------------------------------------------------------
# 9. Dispatcher (spec 2682-2709)
# ---------------------------------------------------------------------------
def test_dispatcher_default_mode_is_deterministic(tmp_path):
    """Spec 2703-2704: the default mode is deterministic; the report is the replay report plus mode."""
    cfg, result, _, _ = gamma_ardinf_pair(tmp_path)
    report = validate_gamma_schedule(cfg, result)
    assert report["mode"] == "deterministic"
    for key in REPLAY_KEYS:
        assert key in report
    assert report["valid"] is True and report["repairs"] == 2


def test_dispatcher_three_modes(tmp_path):
    """Spec 2695-2698: deterministic runs the replay validator, stochastic the Monte Carlo validator, both
    runs both with the keys deterministic and stochastic and valid = both valid."""
    cfg, result, _, _ = gamma_ardinf_pair(tmp_path)
    stoch = validate_gamma_schedule(cfg, result, mode="stochastic", repetitions=200)
    assert stoch["mode"] == "stochastic" and "schedule" in stoch and "cells" in stoch
    both = validate_gamma_schedule(cfg, result, mode="both", repetitions=200)
    assert both["mode"] == "both"
    assert both["deterministic"]["valid"] is True and both["stochastic"]["valid"] is True
    assert "schedule" in both["stochastic"] and "repairs" in both["deterministic"]
    assert both["valid"] is True
    # an invalid replay makes both invalid although the Monte Carlo validator has no target
    cfg2, result2, _, _ = gamma_ardinf_pair(tmp_path, lambda r: put(r, "mu", 1, 0, 1, 0.09), name="b")
    both2 = validate_gamma_schedule(cfg2, result2, mode="both", repetitions=200)
    assert both2["deterministic"]["valid"] is False and both2["stochastic"]["valid"] is True
    assert both2["valid"] is False
    # and a failed target makes it invalid the other way
    both3 = validate_gamma_schedule(cfg, result, mode="both", repetitions=200,
                                    maximum_schedule_failure_rate=0.0, random_seed=1)
    if both3["stochastic"]["schedule"]["failed_replays"] > 0:
        assert both3["stochastic"]["valid"] is False and both3["valid"] is False


def test_dispatcher_none_rule_reads_the_validation_block(tmp_path):
    """Spec 2702: an argument that is None takes the value of the key of the input block validation, else
    its default. The block gives mode stochastic and 500 repetitions; explicit arguments win."""
    def edit_input(inp):
        inp["validation"] = {"mode": "stochastic", "repetitions": 500, "random_seed": 3,
                             "maximum_schedule_failure_rate": 0.9, "confidence_level": 0.9, "batch_size": 100}
    cfg, result, _, _ = gamma_ardinf_pair(tmp_path, inp_edit=edit_input)
    report = validate_gamma_schedule(cfg, result)
    assert report["mode"] == "stochastic"
    assert report["repetitions"] == 500 and report["random_seed"] == 3
    assert near(report["confidence_level"], 0.9) and report["batch_size"] == 100
    assert near(report["schedule"]["maximum_failure_rate"], 0.9)
    assert report["schedule"]["failed_replays"] <= 500
    over = validate_gamma_schedule(cfg, result, mode="deterministic")
    assert over["mode"] == "deterministic" and "repairs" in over
    over = validate_gamma_schedule(cfg, result, repetitions=300)
    assert over["mode"] == "stochastic" and over["repetitions"] == 300
    # without a block the defaults hold
    cfg0, result0, _, _ = gamma_ardinf_pair(tmp_path, name="nb")
    assert validate_gamma_schedule(cfg0, result0, mode="stochastic", repetitions=100)["confidence_level"] == 0.95


def test_dispatcher_errors(tmp_path):
    """Spec 2709: ValueError for another mode (case sensitive) and for a key of the block validation outside
    the six keys."""
    cfg, result, _, _ = gamma_ardinf_pair(tmp_path)
    for mode in ("Both", "STOCHASTIC", "replay", "", "deterministic "):
        with pytest.raises(ValueError):
            validate_gamma_schedule(cfg, result, mode=mode)

    def edit_input(inp):
        inp["validation"] = {"mode": "deterministic", "unknown_key": 1}
    with pytest.raises(ValueError):
        cfg2, result2, _, _ = gamma_ardinf_pair(tmp_path, inp_edit=edit_input, name="k")
        validate_gamma_schedule(cfg2, result2)


def test_dispatcher_raise_on_failure_acts_on_the_dispatcher_report(tmp_path):
    """Spec 2704: the validators run without raising; raise_on_failure applies to the dispatcher's report."""
    cfg, result, _, _ = gamma_ardinf_pair(tmp_path, lambda r: put(r, "mu", 1, 0, 1, 0.09))
    report = validate_gamma_schedule(cfg, result)
    assert report["valid"] is False
    with pytest.raises(AssertionError):
        validate_gamma_schedule(cfg, result, raise_on_failure=True)
    with pytest.raises(AssertionError):
        validate_gamma_schedule(cfg, result, mode="both", repetitions=100, raise_on_failure=True)
    # in mode stochastic the same schedule is valid (no target): no raise
    assert validate_gamma_schedule(cfg, result, mode="stochastic", repetitions=100,
                                   raise_on_failure=True)["valid"] is True


def test_dispatcher_files(tmp_path):
    """Spec 2691-2693, 2511: the file version reads the block validation of the input file and writes the report."""
    inp = load_yaml("gamma_ardinf_input.yaml")
    inp["validation"] = {"mode": "both", "repetitions": 200, "random_seed": 4}
    in_path = write_yaml(tmp_path / "in.yaml", inp)
    out = tmp_path / "d" / "rep.yaml"
    report = validate_gamma_schedule_files(in_path, str(DATA / "gamma_ardinf_result.yaml"), str(out))
    assert report["mode"] == "both" and report["valid"] is True
    with open(out, encoding="utf-8") as handle:
        saved = yaml.safe_load(handle)
    assert saved["mode"] == "both" and "deterministic" in saved and "stochastic" in saved
    assert saved["stochastic"]["repetitions"] == 200
    with pytest.raises(ValueError):
        validate_gamma_schedule_files(in_path, str(DATA / "gamma_ardinf_result.yaml"), str(tmp_path / "x.txt"))
    # the public name is exported
    assert fleet_management.validate_gamma_schedule is validate_gamma_schedule
    assert fleet_management.validate_gamma_replay_schedule is validate_gamma_replay_schedule
