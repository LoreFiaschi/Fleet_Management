"""Several trains per mission, checked against the specification (parts without Gurobi, and solves).

Every expected value comes from spec/spec.tex, not from the code. Sections used: "Sizes and Horizon"
(the keys ``vehicles_per_mission`` and ``vehicles_per_mission_trans``, the number d_jk of trains that
mission j needs on day k), "Increment Profiles and the Phase Lookup" (Eq. c_phase, without the axes i
and l), "Consistency Checks" (item "Sizes and selectors"), "Assignment Constraints" (Eq. a_coverage),
"Greedy Heuristic" (step 2 of the construction pass), "Structural Validator" (check
``mission_coverage``) and "Validators of the Gamma Model" (the replay validator, global violations).

Inputs are small mappings written inline: a Gamma fleet with one component, F = 5, M = 2, H = [2, 3].
Tests that solve need Gurobi and skip without it (``pytest.importorskip`` inside the test); every
other test runs anywhere. The oracle (``tests/reference_model.py``) does its own phase lookup.
"""

import copy

import numpy as np
import pytest

import reference_model as rm
from fleet_management.config import load_config
from fleet_management.degradation_model.gamma_utils.gamma_replay_validator import (
    validate_gamma_replay_schedule,
)
from fleet_management.greedy_warm_start import build_greedy_warm_start
from fleet_management.solver import RESULT_KEYS
from fleet_management.validation.validator import check_result

KEY = "vehicles_per_mission"
TRANS = "vehicles_per_mission_trans"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _data(**changes):
    """A Gamma fleet with one component: F = 5, M = 2, H = [2, 3] (T = 5); ``changes`` replace keys."""
    data = {"F": 5, "M": 2, "H": [2, 3], "model": "gamma", "tau": 1.0, "epsilon": 0.1, "rho": 0.5,
            "mu_0": 0.2, "mu": 0.05, "gamma_beta": 20.0,
            "C_M": 1.0, "C_R": 2.0, "C_rep": 3.0, "C_D": 0.0,
            "verbose": 0, "mip_gap": 0, "time_limit": 60, "gurobi_params": {"Threads": 1}}
    data.update(changes)
    return data


def _cfg(**changes):
    return load_config(_data(**changes))


def _rejected(match, **changes):
    """``load_config`` of the input raises a ValueError whose message contains ``match`` (a plain string)."""
    with pytest.raises(ValueError) as caught:
        load_config(_data(**changes))
    assert match in str(caught.value), str(caught.value)


def _flights(cfg):
    """A schedule ``x`` (F, M+1, T) that meets ``d``: on every day the missions take the trains in order
    0, 1, 2, ... (mission 1 first); the other trains are idle. No intervention."""
    x = np.zeros((cfg.F, cfg.M + 1, cfg.T), dtype=int)
    d = rm.mission_demand(cfg)
    for k in range(cfg.T):
        i = 0
        for j in range(1, cfg.M + 1):
            for _ in range(d[j - 1, k]):
                x[i, j, k] = 1
                i += 1
    return x


def _result(cfg, x, **changes):
    """A result dictionary with the keys that ``check_result`` reads; the objective is large, so its check passes."""
    F, L, T = cfg.F, cfg.L, cfg.T
    res = {"status": "optimal", "F": F, "M": cfg.M, "L": L, "H1": cfg.H1, "H2": cfg.H2, "T": T,
           "x": x, "m": np.zeros((F, L, T)), "r": np.zeros((F, L, T)),
           "mu": np.zeros((F, L, T)), "z": np.zeros((F, L, T)), "u": 0.0, "objective": 1e9}
    res.update(changes)
    return res


def _coverage(report):
    return next(c for c in report["checks"] if c["name"] == "mission_coverage")


# ---------------------------------------------------------------------------
# Parsing (spec "Sizes and Horizon", Eq. c_phase)
# ---------------------------------------------------------------------------
def test_default_is_one_train_per_mission():
    """Spec "Sizes and Horizon": the default of ``vehicles_per_mission`` is 1; d has shape (M, T)."""
    cfg = _cfg()
    assert cfg.d.shape == (2, 5) and cfg.d.dtype == np.int64
    assert (cfg.d == 1).all()
    assert cfg.vehicles_per_mission.shape == (2, 3) and cfg.vehicles_per_mission.dtype == np.int64
    assert (cfg.vehicles_per_mission == 1).all()
    assert cfg.vehicles_per_mission_trans is None


# H = [2, 3], M = 2. Day k <= 2 (no transitory key) reads the operating index (k-1) mod 3; day k > 2 reads (k-3) mod 3.
CASES = {
    "scalar": ({KEY: 2}, None, [[2] * 5, [2] * 5]),
    "per_mission": ({KEY: [2, 1]}, None, [[2] * 5, [1] * 5]),
    "per_mission_and_day": ({KEY: [[1, 2, 3], [0, 1, 0]]}, None,
                            [[1, 2, 1, 2, 3], [0, 1, 0, 1, 0]]),
    "trans_per_mission": ({TRANS: [1, 2]}, None, [[1, 1, 1, 1, 1], [2, 2, 1, 1, 1]]),
    "trans_per_mission_and_day": ({TRANS: [[0, 1], [2, 0]]}, None,
                                  [[0, 1, 1, 1, 1], [2, 0, 1, 1, 1]]),
    "trans_scalar_with_operating": ({KEY: [[1, 2, 3], [0, 1, 0]], TRANS: 2}, None,
                                    [[2, 2, 1, 2, 3], [2, 2, 0, 1, 0]]),
    "both_full": ({KEY: [[1, 2, 3], [0, 1, 0]], TRANS: [[0, 1], [2, 0]]}, None,
                  [[0, 1, 1, 2, 3], [2, 0, 0, 1, 0]]),
}


@pytest.mark.parametrize("name", list(CASES))
def test_phase_lookup(name):
    """Eq. c_phase without i and l: ``cfg.d`` has the hand-computed values, and equals the oracle's own lookup."""
    changes, _, expected = CASES[name]
    cfg = _cfg(**changes)
    np.testing.assert_array_equal(cfg.d, np.array(expected))
    np.testing.assert_array_equal(rm.mission_demand(cfg), np.array(expected))
    assert cfg.d.dtype == np.int64


def test_shapes_of_the_stored_fields():
    """Spec "Sizes and Horizon": the operating field is (M, H2), the transitory field (M, H1) or None; a missing day axis is repeated."""
    cfg = _cfg(**{KEY: [2, 1], TRANS: 1})
    assert cfg.vehicles_per_mission.shape == (2, 3)
    assert cfg.vehicles_per_mission_trans.shape == (2, 2)
    np.testing.assert_array_equal(cfg.vehicles_per_mission, [[2, 2, 2], [1, 1, 1]])
    np.testing.assert_array_equal(cfg.vehicles_per_mission_trans, [[1, 1], [1, 1]])


def test_transitory_key_without_the_operating_key():
    """The transitory key may be given alone: the operating values are then the default 1."""
    cfg = _cfg(**{TRANS: [[2, 2], [0, 0]]})
    assert (cfg.vehicles_per_mission == 1).all()
    np.testing.assert_array_equal(cfg.d, [[2, 2, 1, 1, 1], [0, 0, 1, 1, 1]])


def test_integer_valued_floats_are_accepted():
    """An integer accepts a number with zero fractional part (3.0): the stored value is an integer."""
    cfg = _cfg(F=7, **{KEY: [3.0, 2.0]})
    np.testing.assert_array_equal(cfg.d, [[3] * 5, [2] * 5])
    assert cfg.d.dtype == np.int64 and cfg.vehicles_per_mission.dtype == np.int64
    cfg = _cfg(F=7, **{KEY: 3.0, TRANS: [[1.0, 0.0], [2.0, 2.0]]})
    np.testing.assert_array_equal(cfg.d, [[1, 0, 3, 3, 3], [2, 2, 3, 3, 3]])


def test_zero_is_allowed():
    """A mission with d_jk = 0 is not flown on day k (spec "Sizes and Horizon"): entries are integers >= 0."""
    cfg = _cfg(**{KEY: 0})
    assert (cfg.d == 0).all()


# ---------------------------------------------------------------------------
# Errors (spec "Consistency Checks", item "Sizes and selectors")
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key, value, match", [
    (KEY, [1, 1, 1], "shape (3,) must be scalar"),                    # M + 1 entries
    (KEY, [[1, 1], [1, 1]], "shape (2, 2) must be scalar"),           # H2 = 3, not 2
    (KEY, [[1, 1, 1]], "shape (1, 3) must be scalar"),                # M = 2 rows
    (KEY, [[1, 1], [1, 1], [1, 1]], "shape (3, 2) must be scalar"),   # axes exchanged
    (KEY, [[[1], [1], [1]], [[1], [1], [1]]], "shape (2, 3, 1) must be scalar"),
    (TRANS, [[1, 1, 1], [1, 1, 1]], "shape (2, 3) must be scalar"),   # H1 = 2, not 3
    (TRANS, [1, 1, 1], "shape (3,) must be scalar"),
])
def test_wrong_shape(key, value, match):
    """Spec "Sizes and Horizon": a shape is a scalar, (M) or (M, H*), with H* = H2 for the operating key and H1 for the transitory key."""
    with pytest.raises(ValueError) as caught:
        load_config(_data(**{key: value}))
    message = str(caught.value)
    assert match in message and message.startswith(f"'{key}' shape")
    Hs = 3 if key == KEY else 2
    assert f"({2},{Hs})" in message


@pytest.mark.parametrize("key", [KEY, TRANS])
@pytest.mark.parametrize("value, j", [
    (-1, 1),
    ([1, -1], 2),
    ([2.5, 1], 1),
    ([1, 1.5], 2),
    (True, 1),
    ([1, False], 2),
    (None, 1),
    ([1, None], 2),
    (float("nan"), 1),
    ([float("nan"), 1], 1),
    (float("inf"), 1),
    ([1, float("-inf")], 2),
    ([[1, 1, 1], [1, -1, 1]] , 2),      # (M, H2) or (M, H1) rows; wrong shape for TRANS, see below
])
def test_bad_entry(key, value, j):
    """Spec "Consistency Checks": every entry is an integer >= 0. The message names the key and the mission j (1-based)."""
    if key == TRANS and isinstance(value, list) and isinstance(value[0], list):
        value = [[1, 1], [1, -1]]
    with pytest.raises(ValueError) as caught:
        load_config(_data(**{key: value}))
    message = str(caught.value)
    assert f"'{key}' must hold integers >= 0" in message
    assert f"at mission j={j}" in message


def test_fleet_too_small_at_the_boundary():
    """F equal to the largest daily sum of d fails; one more train passes. The largest sum is not on day 1.

    H = [1, 2], M = 2, d = [[1, 1], [1, 2]] on the operating days: the sums are 2 (operating index 1; the
    transitory day 1 reads the same index) and 3 (index 2). Days k = 1, 2, 3 have the sums 2, 2, 3, so the
    first day of the largest sum is k = 3. The sums per operating index are 2 and 3.
    """
    changes = {"H": [1, 2], KEY: [[1, 1], [1, 2]]}
    with pytest.raises(ValueError) as caught:
        load_config(_data(F=3, **changes))
    message = str(caught.value)
    assert "largest number of trains" in message
    assert "got F=3, largest=3 on day k=3" in message
    cfg = load_config(_data(F=4, **changes))
    assert cfg.d.sum(axis=0).tolist() == [2, 2, 3]


def test_fleet_check_reports_the_first_day_of_the_largest_sum():
    """Two days reach the largest sum: the first is reported. H = [2, 1], transitory d sums 2 and 4, operating 4."""
    changes = {"H": [2, 1], KEY: [1, 1], TRANS: [[1, 2], [1, 2]]}
    with pytest.raises(ValueError) as caught:
        load_config(_data(F=4, **changes))
    assert "largest=4 on day k=2" in str(caught.value)
    load_config(_data(F=5, **changes))


def test_fleet_check_reads_the_transitory_phase():
    """A transitory day alone can break the check: operating sums 2, transitory sum 6 on day 1."""
    changes = {"H": [1, 1], KEY: [1, 1], TRANS: [3, 3]}
    with pytest.raises(ValueError) as caught:
        load_config(_data(F=6, **changes))
    assert "largest number of trains" in str(caught.value) and "day k=1" in str(caught.value)
    load_config(_data(F=7, **changes))


def test_fleet_larger_than_the_missions_still_required_when_some_d_are_zero():
    """F > M stays (spec "Sizes and selectors"), also if d = 0 everywhere: the check F > M runs first and keeps its message."""
    with pytest.raises(ValueError) as caught:
        load_config(_data(F=2, **{KEY: [0, 0]}))
    assert "'F' must be > 'M'" in str(caught.value)
    cfg = load_config(_data(F=3, **{KEY: [0, 0]}))
    assert (cfg.d == 0).all()
    with pytest.raises(ValueError) as caught:
        load_config(_data(F=2, **{KEY: [0, 1]}))
    assert "'F' must be > 'M'" in str(caught.value)


def test_fleet_check_with_a_zero_row():
    """A mission with d = 0 adds no train to the sum: M = 2, d = [3, 0], F = 4 passes, F = 3 fails (3 is not > 3)."""
    load_config(_data(F=4, **{KEY: [3, 0]}))
    with pytest.raises(ValueError) as caught:
        load_config(_data(F=3, **{KEY: [3, 0]}))
    assert "largest number of trains" in str(caught.value)


# ---------------------------------------------------------------------------
# Oracle (tests/reference_model.py): enumeration and coverage row with d
# ---------------------------------------------------------------------------
def test_oracle_enumeration_counts():
    """F = 4, M = 2, d = [2, 1], T = 2, no replacement: per day C(4,2) * 2 choices of the flyers, and the
    last train is idle, in the depot, or repaired (3 states): 36 per day, 36^2 = 1296 schedules."""
    cfg = _cfg(F=4, H=[1, 1], allow_replacement=False, **{KEY: [2, 1]})
    count = 0
    for x, m, r in rm.enumerate_schedules(cfg):
        count += 1
        assert (x.sum(axis=1) <= 1).all()                                  # Eq. a_one_activity
        assert (x[:, 1, :].sum(axis=0) == 2).all() and (x[:, 2, :].sum(axis=0) == 1).all()   # Eq. a_coverage
        assert (m + r <= x[:, 0:1, :]).all()
    assert count == 36 ** 2


def test_oracle_enumeration_with_a_zero_mission():
    """F = 3, M = 2, d = [2, 0]: mission 2 has no train. Per day C(3,2) = 3 flyers, the last train has 3 states: 9; T = 2: 81."""
    cfg = _cfg(F=3, H=[1, 1], allow_replacement=False, **{KEY: [2, 0]})
    count = 0
    for x, m, r in rm.enumerate_schedules(cfg):
        count += 1
        assert (x[:, 2, :] == 0).all()
        assert (x[:, 1, :].sum(axis=0) == 2).all()
    assert count == 81


def test_oracle_enumeration_with_a_day_varying_demand():
    """F = 3, M = 1, H = [1, 1], transitory key d = 2: day 1 has d = 2, day 2 has d = 1. Per day (no replacement):
    d = 2: 3 flyer sets * 3 states of the last train = 9; d = 1: 3 flyer sets * 3^2 = 27. Total 9 * 27 = 243."""
    cfg = _cfg(F=3, M=1, H=[1, 1], mu=0.05, allow_replacement=False, **{TRANS: 2})
    np.testing.assert_array_equal(rm.mission_demand(cfg), [[2, 1]])
    count = sum(1 for _ in rm.enumerate_schedules(cfg))
    assert count == 9 * 27


def test_oracle_coverage_row_uses_d():
    """``row_violations`` has the row ``coverage_j<j>`` with the violation |sum_i x_ijk - d_jk| (Eq. a_coverage)."""
    cfg = _cfg(F=4, H=[1, 1], **{KEY: [2, 1]})
    x = _flights(cfg)
    m = np.zeros((4, 1, 2), dtype=int)
    traj = rm.reference_trajectory(cfg, x, m, np.zeros_like(m))
    sol = {"x": x, "m": m, "r": np.zeros_like(m), "z": rm.minimal_z(cfg, traj, m), "u": rm.u_value(cfg, traj["mu"])}
    for name in ("mu", "v", "R", "K", "g_mu", "g_v", "g_R", "g_K"):
        sol[name] = traj[name][..., 1:]
    assert [v for v in rm.row_violations(cfg, sol) if v.row.startswith("coverage")] == []
    bad = x.copy()
    bad[1, 1, 1] = 0                                         # mission 1 has one train on day 2
    found = [v for v in rm.row_violations(cfg, dict(sol, x=bad)) if v.row.startswith("coverage")]
    assert [(v.row, v.k) for v in found] == [("coverage_j1", 2)]
    assert found[0].violation == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Greedy heuristic (spec "Greedy Heuristic", step 2 of the construction pass)
# ---------------------------------------------------------------------------
def test_greedy_meets_d_on_every_day():
    """Every day mission j has exactly d_jk trains in x, a train has at most one activity, a mission with d = 0 has none."""
    cfg = _cfg(**{KEY: [[2, 1, 0], [1, 2, 1]], TRANS: [[0, 2], [2, 0]]})
    source, _ = build_greedy_warm_start(cfg)
    x = np.asarray(source["x"])
    d = rm.mission_demand(cfg)
    assert x.shape == (5, 3, 5)
    assert (x.sum(axis=1) <= 1).all()                                       # Eq. a_one_activity
    np.testing.assert_array_equal(x[:, 1:, :].sum(axis=0), d)               # Eq. a_coverage
    assert (x[:, 1:, :].sum(axis=0)[d == 0] == 0).all()
    assert (d == 0).any()
    assert (x[:, 1, 0] == 0).all() and x[:, 2, 0].sum() == 2                # d_{1,1} = 0, d_{2,1} = 2


def test_greedy_default_demand_is_unchanged():
    """With d = 1 the heuristic gives one train per mission and day."""
    cfg = _cfg()
    x = np.asarray(build_greedy_warm_start(cfg)[0]["x"])
    assert (x[:, 1:, :].sum(axis=0) == 1).all()


def test_greedy_tie_rule():
    """Symmetric case: F = 4, M = 2, d = [2, 1], one component, equal increments, equal states.

    Step 1: the missions have the same ratio, a tie goes to the larger j: mission 2 first.
    Step 2: each time the free train of the smallest key, a tie to the smaller i: mission 2 takes train 0
    (index 0); then mission 1 takes train 1, and (train 1 now has the larger state) train 2.
    Day 1 is a transitory day that reads the operating index 1.
    """
    cfg = _cfg(F=4, H=[1, 2], **{KEY: [2, 1]})
    source, report = build_greedy_warm_start(cfg)
    x = np.asarray(source["x"])
    assert x[0, 2, 0] == 1
    assert x[1, 1, 0] == 1 and x[2, 1, 0] == 1
    assert x[3, :, 0].sum() == 0
    assert report.assignment_order[0] == [2, 1]


def test_greedy_tie_rule_one_mission_two_trains():
    """F = 3, M = 1, d = 2, equal states: the trains 0 and 1 fly (a tie goes to the smaller index)."""
    cfg = _cfg(F=3, M=1, H=[1, 1], **{KEY: 2})
    x = np.asarray(build_greedy_warm_start(cfg)[0]["x"])
    assert x[0, 1, 0] == 1 and x[1, 1, 0] == 1 and x[2, 1, 0] == 0


# ---------------------------------------------------------------------------
# Structural validator (spec "Structural Validator", check mission_coverage)
# ---------------------------------------------------------------------------
def test_validator_accepts_a_schedule_that_meets_d():
    """d = [2, 1], F = 4: the schedule meets d on every day, mission_coverage has violation 0."""
    cfg = _cfg(F=4, H=[1, 1], **{KEY: [2, 1]})
    x = _flights(cfg)
    assert (x[:, 1, :].sum(axis=0) == 2).all()
    report = check_result(cfg, _result(cfg, x))
    assert _coverage(report)["passed"] and _coverage(report)["violation"] == 0.0
    assert report["passed"]


def test_validator_meets_a_day_varying_d():
    """d from the operating key (M, H2) and the transitory key: a schedule that meets d passes."""
    cfg = _cfg(**{KEY: [[1, 2, 0], [2, 0, 1]], TRANS: [[0, 1], [1, 1]]})
    report = check_result(cfg, _result(cfg, _flights(cfg)))
    assert _coverage(report)["violation"] == 0.0 and report["passed"]


def test_validator_one_train_removed():
    """Spec check table: the violation is max |sum_i x_ijk - d_jk|; one train removed from the mission with d = 2 gives 1."""
    cfg = _cfg(F=4, H=[1, 1], **{KEY: [2, 1]})
    x = _flights(cfg)
    x[1, 1, 1] = 0                                          # mission 1 has one train on day 2
    report = check_result(cfg, _result(cfg, x))
    record = _coverage(report)
    assert record["violation"] == pytest.approx(1.0) and not record["passed"]
    assert not report["passed"]
    assert next(c for c in report["checks"] if c["name"] == "one_activity")["passed"]


def test_validator_one_train_too_many():
    """A third train on mission 1 (d = 2): violation |3 - 2| = 1."""
    cfg = _cfg(F=4, H=[1, 1], **{KEY: [2, 1]})
    x = _flights(cfg)
    x[3, 1, 0] = 1
    assert _coverage(check_result(cfg, _result(cfg, x)))["violation"] == pytest.approx(1.0)


def test_validator_mission_with_zero_demand():
    """A mission with d = 0 that is flown by one train: violation |1 - 0| = 1."""
    cfg = _cfg(F=4, H=[1, 1], **{KEY: [2, 0]})
    x = _flights(cfg)
    assert _coverage(check_result(cfg, _result(cfg, x)))["passed"]
    x[2, 2, 0] = 1
    assert _coverage(check_result(cfg, _result(cfg, x)))["violation"] == pytest.approx(1.0)


def test_validator_violation_is_the_largest_difference():
    """Mission 1 misses two trains on one day (d = 3 flown by 1): violation 2, not 1."""
    cfg = _cfg(F=5, H=[1, 1], **{KEY: [3, 1]})
    x = _flights(cfg)
    x[1, 1, 0] = 0
    x[2, 1, 0] = 0
    assert _coverage(check_result(cfg, _result(cfg, x)))["violation"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Gamma replay validator (spec "Validators of the Gamma Model", step 1)
# ---------------------------------------------------------------------------
def _replay_result(cfg, x):
    F, L, T = cfg.F, cfg.L, cfg.T
    res = dict.fromkeys(RESULT_KEYS)
    res.update(status="optimal", F=F, M=cfg.M, L=L, H1=cfg.H1, H2=cfg.H2, T=T, H=[cfg.H1, cfg.H2],
               model=[["gamma"] * L for _ in range(F)],
               x=np.asarray(x, dtype=float), m=np.zeros((F, L, T)), r=np.zeros((F, L, T)),
               mu=np.full((F, L, T), 0.5), g_mu=np.full((F, L, T), np.nan), z=np.zeros((F, L, T)))
    return res


def test_replay_global_violation_for_the_number_of_trains():
    """Mission j on day k with a number of trains other than d_jk: one string per occurrence.

    The days of these strings are 0-based, as in the other global violations of this validator.
    d = 2 for mission 1: the schedule gives 1 train on day 0 and 3 trains on day 1.
    """
    cfg = _cfg(F=4, M=1, H=[1, 1], repair_model="ardinf", **{KEY: 2})
    x = _flights(cfg)
    assert validate_gamma_replay_schedule(cfg, _replay_result(cfg, x))["global_violations"] == []
    x[1, 1, 0] = 0
    x[2, 1, 1] = 1
    report = validate_gamma_replay_schedule(cfg, _replay_result(cfg, x))
    assert report["global_violations"] == [
        "mission j=1 is flown by 1 trains on day k=0, not 2",
        "mission j=1 is flown by 3 trains on day k=1, not 2",
    ]
    assert not report["valid"]


def test_replay_global_violation_with_zero_demand():
    """d = 0 on the transitory day: one train that flies is reported with 'not 0'."""
    cfg = _cfg(F=3, M=1, H=[1, 1], repair_model="ardinf", **{KEY: 1, TRANS: 0})
    x = np.zeros((3, 2, 2))
    x[0, 1, 0] = 1                                          # d = 0 on day 0 but one train flies
    x[0, 1, 1] = 1                                          # d = 1 on day 1: met
    report = validate_gamma_replay_schedule(cfg, _replay_result(cfg, x))
    assert report["global_violations"] == ["mission j=1 is flown by 1 trains on day k=0, not 0"]


# ---------------------------------------------------------------------------
# Solves (need Gurobi): the rows demand_<j>_<k>, the validator and the oracle
# ---------------------------------------------------------------------------
def _solution(res):
    """The mapping that ``row_violations`` reads: binaries rounded, the rest as reported."""
    sol = {key: np.round(np.asarray(res[key], dtype=float)).astype(np.int8) for key in ("x", "m", "r")}
    for key in ("mu", "v", "R", "K", "g_mu", "g_v", "g_R", "g_K", "z",
                "y_mu", "y_v", "y_R", "y_K", "y_g_mu", "y_g_v", "y_g_R", "y_g_K"):
        sol[key] = np.asarray(res[key], dtype=float)
    sol["u"] = float(res["u"])
    return sol


def _solve_and_check(data):
    """Solve ``data``; check status, the rows demand, the validator and the oracle. Returns ``(cfg, res)``."""
    pytest.importorskip("gurobipy")
    from fleet_management.solver import solve_mapping

    cfg = load_config(copy.deepcopy(data))
    res = solve_mapping(copy.deepcopy(data), None)
    assert res["status"] == "optimal"
    x = np.asarray(res["x"], dtype=float)
    assert np.all(np.abs(x - np.round(x)) <= 1e-5)
    x = np.round(x).astype(int)
    d = rm.mission_demand(cfg)
    np.testing.assert_array_equal(d, cfg.d)
    np.testing.assert_array_equal(x[:, 1:, :].sum(axis=0), d)               # sum_i x_ijk = d_jk
    assert (x.sum(axis=1) <= 1).all()
    report = check_result(cfg, res)
    assert report["passed"], [c for c in report["checks"] if not c["passed"]]
    assert _coverage(report)["violation"] <= 1e-5
    sol = _solution(res)                                                     # the oracle accepts the schedule
    assert rm.row_violations(cfg, sol) == []
    rep = rm.costs(cfg, sol["x"], sol["r"], res["z"], float(res["u"]))
    assert float(res["objective"]) == pytest.approx(rep["objective"], rel=1e-6, abs=1e-9)
    return cfg, res


# A schedule with d = [2, 1], F = 5, H = [1, 3] that the oracle accepts (checked by hand): mu_0 = 0.5, increment 0.05.
SOLVE_BASE = dict(F=5, H=[1, 3], mu_0=0.5, **{KEY: [2, 1]})


def test_solve_two_missions_with_several_trains():
    """F = 5, M = 2, d = [2, 1]: optimal; sum_i x_ijk = d_jk on every day; validate passes; the oracle agrees on the objective."""
    cfg, res = _solve_and_check(_data(**SOLVE_BASE))
    d = np.asarray(res["x"]).round()[:, 1:, :].sum(axis=0)
    np.testing.assert_array_equal(d, [[2] * 4, [1] * 4])


def test_solve_with_the_indicator_encoding():
    """The same instance with ``encoding: indicator`` has the same optimal objective as the big-M encoding."""
    cfg_a, res_a = _solve_and_check(_data(**SOLVE_BASE))
    cfg_b, res_b = _solve_and_check(_data(encoding="indicator", **SOLVE_BASE))
    assert float(res_b["objective"]) == pytest.approx(float(res_a["objective"]), rel=1e-6, abs=1e-6)


def _brute_force_cost(cfg):
    """The best cost over every schedule of the oracle (exact for ardinf cells: ``best_onesided``)."""
    best = rm.best_onesided(cfg)
    assert best["exact"] and best["cost"] is not None
    return best["cost"]


SMALL = dict(F=3, M=1, H=[1, 2], mu_0=0.5, allow_replacement=False)


def test_solve_day_varying_demand_matches_brute_force():
    """M = 1, F = 3, d = [[1, 2]] (operating days 2 and 3 read index 1 and 2; day 1 reads index 1): the optimum
    equals the best cost of the enumeration of every schedule (oracle; every cell is ardinf, so it is exact)."""
    data = _data(**SMALL, **{KEY: [[1, 2]]})
    cfg, res = _solve_and_check(data)
    np.testing.assert_array_equal(cfg.d, [[1, 1, 2]])
    assert float(res["objective"]) == pytest.approx(_brute_force_cost(cfg), rel=1e-6, abs=1e-6)


def test_solve_transitory_demand_matches_brute_force():
    """M = 1, F = 3, transitory d = 2 on day 1, operating d = 1: the optimum equals the best cost of the enumeration."""
    data = _data(**SMALL, **{TRANS: 2})
    cfg, res = _solve_and_check(data)
    np.testing.assert_array_equal(cfg.d, [[2, 1, 1]])
    assert float(res["objective"]) == pytest.approx(_brute_force_cost(cfg), rel=1e-6, abs=1e-6)


def test_solve_transitory_demand_with_the_indicator_encoding():
    """The transitory case with the indicator encoding: the same optimum as the enumeration."""
    data = _data(encoding="indicator", **SMALL, **{TRANS: 2})
    cfg, res = _solve_and_check(data)
    assert float(res["objective"]) == pytest.approx(_brute_force_cost(cfg), rel=1e-6, abs=1e-6)


def test_oracle_brute_force_costs_of_the_small_cases():
    """No Gurobi: the oracle finds a feasible schedule in both small cases (best costs 4.6 and 3.1, checked by hand)."""
    cost_day = _brute_force_cost(load_config(_data(**SMALL, **{KEY: [[1, 2]]})))
    cost_trans = _brute_force_cost(load_config(_data(**SMALL, **{TRANS: 2})))
    assert cost_day == pytest.approx(4.6, abs=1e-9)
    assert cost_trans == pytest.approx(3.1, abs=1e-9)


def test_oracle_accepts_the_hand_schedule_of_the_solve_case():
    """No Gurobi: F = 5, M = 2, d = [2, 1], H = [1, 3]. Day 1 is a transitory day. On the operating days 2 to 4 the trains
    0 to 4 follow F F D, F D F, D F F, F D F, I F D (F flies, D repairs in the depot, I idle); the oracle finds no broken row."""
    cfg = load_config(_data(**SOLVE_BASE))
    F, M, T = 5, 2, 4
    x = np.zeros((F, M + 1, T), dtype=int)
    m = np.zeros((F, 1, T), dtype=int)
    x[0, 1, 0] = x[1, 1, 0] = x[2, 2, 0] = 1
    patterns = ["FFD", "FDF", "DFF", "FDF", "IFD"]
    for i, pattern in enumerate(patterns):
        for k, c in enumerate(pattern, start=1):
            if c == "D":
                x[i, 0, k] = m[i, 0, k] = 1
    for k in (1, 2, 3):
        flyers = [i for i, pattern in enumerate(patterns) if pattern[k - 1] == "F"]
        x[flyers[0], 1, k] = x[flyers[1], 1, k] = x[flyers[2], 2, k] = 1
    r = np.zeros_like(m)
    traj = rm.reference_trajectory(cfg, x, m, r)
    sol = {"x": x, "m": m, "r": r, "z": rm.minimal_z(cfg, traj, m), "u": rm.u_value(cfg, traj["mu"])}
    for name in ("mu", "v", "R", "K", "g_mu", "g_v", "g_R", "g_K"):
        sol[name] = traj[name][..., 1:]
    assert rm.row_violations(cfg, sol) == []
    assert _coverage(check_result(cfg, _result(cfg, x)))["violation"] == 0.0
