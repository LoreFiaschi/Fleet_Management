"""Warm starts and the greedy heuristic, against spec section "Warm Starts" (lines 2109-2273 of the lore copy).

Every expected value is derived by hand from the spec (comments give the derivation) and was
checked with a script that follows the spec steps, not the package code.  Tests that solve need
Gurobi and skip without it (``pytest.importorskip`` inside the test); the sweep tests replace
``fleet_management.horizon_sweep.solve_mapping`` by a fake (spec 1949); the greedy tests need
no Gurobi.
"""
from pathlib import Path

import numpy as np
import pytest
import yaml

DATA = Path(__file__).parent / "data" / "tools" / "warm_starts"
BASE = DATA / "zero_increments.yaml"


# ===========================================================================
# Helpers
# ===========================================================================
def _read(name):
    with open(DATA / name, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _variant(tmp_path, **changes):
    """Write the base input (F = 2, M = 1, L = 1, H = [2, 3]) with some keys replaced."""
    data = _read("zero_increments.yaml")
    data.update(changes)
    path = tmp_path / "variant.yaml"
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle)
    return str(path)


def _solve(path, warm_start=None, mode="start"):
    pytest.importorskip("gurobipy")
    from fleet_management.solver import solve
    return solve(str(path), None, warm_start=warm_start, warm_start_mode=mode)


def _pair(source_horizon):
    """(H1, H2) from the report value, whether it is a list, a tuple or a mapping (spec 1838)."""
    if isinstance(source_horizon, dict):
        return (source_horizon["H1"], source_horizon["H2"])
    return tuple(source_horizon)


def _sigma(k, H1, H2, H1s, H2s):
    """Eq. d_daymap (spec 2142-2146), 1-based days."""
    if k <= min(H1, H1s):
        return k
    return H1s + 1 + (max(0, k - H1 - 1) % H2s)


def _schedule(F, M, L, T, flights, depots=(), repairs=(), replacements=()):
    """Arrays x (F, M+1, T), m, r (F, L, T) from a hand description, all indices 1-based.

    flights: {day: {mission: train}}; depots: [(train, day)]; repairs, replacements:
    [(train, component, day)].
    """
    x = np.zeros((F, M + 1, T), dtype=int)
    m = np.zeros((F, L, T), dtype=int)
    r = np.zeros((F, L, T), dtype=int)
    for k, missions in flights.items():
        for j, i in missions.items():
            x[i - 1, j, k - 1] = 1
    for i, k in depots:
        x[i - 1, 0, k - 1] = 1
    for i, l, k in repairs:
        m[i - 1, l - 1, k - 1] = 1
    for i, l, k in replacements:
        r[i - 1, l - 1, k - 1] = 1
    return x, m, r


# Source of the day-mapping tests, case 1: F' = 3, M' = 1, L' = 1, H1' = 1, H2' = 2 (T' = 3).
# Train 3 does not exist in the target (F = 2): its entries must be ignored.
def _source_case1():
    x = np.zeros((3, 2, 3), dtype=int)
    x[0, 1, 0] = 1                        # source day 1: train 1 flies
    x[1, 1, 1] = 1; x[0, 0, 1] = 1        # source day 2: train 2 flies, train 1 in the depot
    x[0, 1, 2] = 1; x[1, 0, 2] = 1        # source day 3: train 1 flies, train 2 in the depot
    x[2, 0, 0] = 1                        # train 3: ignored
    m = np.zeros((3, 1, 3), dtype=int)
    m[0, 0, 1] = 1                        # train 1 repaired on source day 2
    m[2, 0, 0] = 1                        # train 3: ignored
    r = np.zeros((3, 1, 3), dtype=int)
    r[1, 0, 2] = 1                        # train 2 replaced on source day 3
    return {"H1": 1, "H2": 2, "x": x, "m": m, "r": r}


# Source, case 2: F' = 2, M' = 1, L' = 2, H1' = 3, H2' = 2 (T' = 5). Component 2 does not exist
# in the target (L = 1): its entries must be ignored.
def _source_case2():
    x = np.zeros((2, 2, 5), dtype=int)
    x[0, 1, 0] = 1                        # source day 1: train 1 flies
    x[1, 1, 1] = 1                        # day 2: train 2 flies
    x[0, 1, 2] = 1; x[1, 0, 2] = 1        # day 3: train 1 flies, train 2 in the depot (never mapped)
    x[1, 1, 3] = 1; x[0, 0, 3] = 1        # day 4: train 2 flies, train 1 in the depot
    x[0, 1, 4] = 1; x[1, 0, 4] = 1        # day 5: train 1 flies, train 2 in the depot
    m = np.zeros((2, 2, 5), dtype=int)
    m[0, 0, 3] = 1                        # train 1, component 1 repaired on source day 4
    m[1, 0, 4] = 1                        # train 2, component 1 repaired on source day 5
    m[0, 1, 0] = 1                        # component 2: ignored
    r = np.zeros((2, 2, 5), dtype=int)
    r[1, 0, 2] = 1                        # replacement on source day 3, which no target day uses
    return {"H1": 3, "H2": 2, "x": x, "m": m, "r": r}


# A full source with the target's own dimensions (F = 2, M = 1, L = 1, T = 5), used for the modes.
def _source_full():
    x, m, r = _schedule(2, 1, 1, 5, {1: {1: 1}, 2: {1: 2}, 3: {1: 2}, 4: {1: 1}, 5: {1: 2}},
                        depots=[(1, 2), (1, 3), (2, 4), (1, 5)],
                        repairs=[(1, 1, 2), (1, 1, 3), (1, 1, 5)], replacements=[(2, 1, 4)])
    return {"H1": 2, "H2": 3, "x": x, "m": m, "r": r}


# Number of variables that receive a value (spec 1836, 2151-2155): x has F (M+1) T entries, m has
# F L T (0 where the source has no m), r has F L T if allow_replacement and the source has r, and
# the selector n of the indicator form has F L T.
def _values(F=2, M=1, L=1, T=5, with_r=True, indicator=False):
    return F * (M + 1) * T + F * L * T + (F * L * T if with_r else 0) + (F * L * T if indicator else 0)


def _check_mapping(result, expected, count, source_horizon):
    """Check the report of a mapped start and, if Gurobi returned the start, its schedule.

    SolutionLimit is 1 and every schedule is feasible, so Gurobi normally returns the accepted
    start itself.  If the returned solution differs from the start, the arrays say nothing about
    the mapping and only the report is checked (spec 1835-1838); the count is derived by hand.
    """
    report = result["warm_start"]
    assert report["applied"] is True
    assert report["mode"] == "start"
    assert _pair(report["source_horizon"]) == source_horizon
    assert report["values"] == count
    assert result["x"] is not None
    x, m, r = expected
    same = (np.array_equal(np.rint(result["x"]).astype(int), x)
            and np.array_equal(np.rint(result["m"]).astype(int), m)
            and np.array_equal(np.rint(result["r"]).astype(int), r))
    if not same:
        assert result["warm_start"]["values"] == count   # the fallback: the count of mapped values


# ===========================================================================
# Interface (spec 2126-2135)
# ===========================================================================
def test_invalid_mode_raises(tmp_path):
    """Another warm_start_mode raises ValueError (spec 2128)."""
    pytest.importorskip("gurobipy")
    for mode in ("warm", "Start", "", None):
        with pytest.raises(ValueError):
            _solve(BASE, warm_start=_source_full(), mode=mode)
    with pytest.raises(ValueError):
        _solve(BASE, warm_start=None, mode="both ")


def test_invalid_source_horizons_raise(tmp_path):
    """H1 or H2 not an integer >= 1 (zero, fractional, a flag) raises ValueError (spec 2132)."""
    pytest.importorskip("gurobipy")
    x = np.zeros((2, 2, 3), dtype=int)
    bad = [
        {"H1": 0, "H2": 3, "x": x},
        {"H1": 1, "H2": 0, "x": x},
        {"H1": 1.5, "H2": 2, "x": x},
        {"H1": 1, "H2": 2.5, "x": x},
        {"H1": True, "H2": 2, "x": x},
        {"H1": 1, "H2": True, "x": x},
        {"H1": -1, "H2": 4, "x": x},
    ]
    for source in bad:
        with pytest.raises(ValueError):
            _solve(BASE, warm_start=source)


def test_invalid_source_axes_raise(tmp_path):
    """x, m or r without three axes or without T' days on the last raises ValueError (spec 2132)."""
    pytest.importorskip("gurobipy")
    good_x = np.zeros((2, 2, 5), dtype=int)
    bad = [
        {"H1": 2, "H2": 3, "x": np.zeros((2, 5), dtype=int)},               # two axes
        {"H1": 2, "H2": 3, "x": np.zeros((2, 2, 4), dtype=int)},            # wrong last axis
        {"H1": 2, "H2": 3, "x": np.zeros((2, 2, 5, 1), dtype=int)},         # four axes
        {"H1": 2, "H2": 3, "x": good_x, "m": np.zeros((2, 5), dtype=int)},
        {"H1": 2, "H2": 3, "x": good_x, "m": np.zeros((2, 1, 4), dtype=int)},
        {"H1": 2, "H2": 3, "x": good_x, "r": np.zeros((2, 5), dtype=int)},
        {"H1": 2, "H2": 3, "x": good_x, "r": np.zeros((2, 1, 6), dtype=int)},
    ]
    for source in bad:
        with pytest.raises(ValueError):
            _solve(BASE, warm_start=source)


def test_integer_valued_horizons_are_accepted(tmp_path):
    """A number with zero fractional part is an integer (spec 2132): H1 = 2.0, H2 = 3.0 is applied."""
    source = _source_full()
    source["H1"], source["H2"] = 2.0, 3.0
    result = _solve(BASE, warm_start=source)
    assert result["warm_start"]["applied"] is True
    assert _pair(result["warm_start"]["source_horizon"]) == (2, 3)


def test_source_without_x_has_no_effect(tmp_path):
    """A source whose x is missing or null has no effect: applied false, mode null (spec 2131, 1835-1837)."""
    for source in ({"H1": 2, "H2": 3}, {"H1": 2, "H2": 3, "x": None},
                   {"H1": 2, "H2": 3, "x": None, "m": np.zeros((2, 1, 5), dtype=int)}):
        result = _solve(BASE, warm_start=source)
        report = result["warm_start"]
        assert report["applied"] is False
        assert report["mode"] is None
        assert report["values"] == 0
        assert result["x"] is not None


def test_no_source_report(tmp_path):
    """Without a source: applied false, values 0, mode and source_horizon null, completion not requested (spec 1835-1847)."""
    result = _solve(BASE)
    report = result["warm_start"]
    assert report["applied"] is False
    assert report["values"] == 0
    assert report["mode"] is None
    assert report["source_horizon"] is None
    assert report["completion"]["requested"] is False
    assert report["completion"]["status"] == "not_requested"


# ===========================================================================
# Day mapping, Eq. d_daymap (spec 2140-2155)
# ===========================================================================
def test_day_map_formula_by_hand():
    """sigma(k) for H1 = 2, H2 = 3 by hand (the values used in the expected arrays below)."""
    # source (H1', H2') = (1, 2): k = 1 -> 1; otherwise 2 + (max(0, k - 3) mod 2): 2, 2, 3, 2.
    assert [_sigma(k, 2, 3, 1, 2) for k in range(1, 6)] == [1, 2, 2, 3, 2]
    # source (3, 2): k = 1, 2 -> k; otherwise 4 + ((k - 3) mod 2): 4, 5, 4.
    assert [_sigma(k, 2, 3, 3, 2) for k in range(1, 6)] == [1, 2, 4, 5, 4]


def test_day_map_short_source_and_larger_fleet(tmp_path):
    """Source (H1', H2') = (1, 2) with F' = 3 > F: x, m, r mapped by sigma = [1, 2, 2, 3, 2] (spec 2140-2154).

    Target days 1..5 take source days 1, 2, 2, 3, 2:
      day 1 = source 1: train 1 flies;
      days 2, 3, 5 = source 2: train 2 flies, train 1 in the depot with a repair;
      day 4 = source 3: train 1 flies, train 2 in the depot with a replacement.
    Train 3 of the source has no counterpart.  Values: x 2*2*5 = 20, m 10, r 10 = 40.
    """
    expected = _schedule(2, 1, 1, 5, {1: {1: 1}, 2: {1: 2}, 3: {1: 2}, 4: {1: 1}, 5: {1: 2}},
                         depots=[(1, 2), (1, 3), (2, 4), (1, 5)],
                         repairs=[(1, 1, 2), (1, 1, 3), (1, 1, 5)], replacements=[(2, 1, 4)])
    result = _solve(BASE, warm_start=_source_case1())
    _check_mapping(result, expected, _values(), (1, 2))


def test_day_map_long_source_and_more_components(tmp_path):
    """Source (3, 2) with L' = 2 > L: sigma = [1, 2, 4, 5, 4]; component 2 of the source is ignored (spec 2133, 2140-2154).

    Day 1 = source 1: train 1 flies.  Day 2 = source 2: train 2 flies.
    Days 3, 5 = source 4: train 2 flies, train 1 in the depot, repair of train 1.
    Day 4 = source 5: train 1 flies, train 2 in the depot, repair of train 2.
    The replacement of source day 3 is used by no target day: r is zero.  Values 40.
    """
    expected = _schedule(2, 1, 1, 5, {1: {1: 1}, 2: {1: 2}, 3: {1: 2}, 4: {1: 1}, 5: {1: 2}},
                         depots=[(1, 3), (2, 4), (1, 5)],
                         repairs=[(1, 1, 3), (2, 1, 4), (1, 1, 5)])
    result = _solve(BASE, warm_start=_source_case2())
    _check_mapping(result, expected, _values(), (3, 2))


def test_value_counts(tmp_path):
    """Counts of mapped variables: a source without m gives m = 0 (counted), without r no r (spec 2152-2155, 1836)."""
    full = _source_full()
    only_x = {"H1": 2, "H2": 3, "x": full["x"]}
    # x 20 + m 10 (zeros) = 30
    assert _solve(BASE, warm_start=only_x)["warm_start"]["values"] == _values(with_r=False)
    # x and m: also 30
    with_m = {"H1": 2, "H2": 3, "x": full["x"], "m": full["m"]}
    assert _solve(BASE, warm_start=with_m)["warm_start"]["values"] == _values(with_r=False)
    # allow_replacement false: r is not mapped even if the source has r: x 20 + m 10 = 30
    no_rep = _variant(tmp_path, allow_replacement=False)
    assert _solve(no_rep, warm_start=full)["warm_start"]["values"] == _values(with_r=False)
    # full source: 40
    assert _solve(BASE, warm_start=full)["warm_start"]["values"] == _values()


def test_value_count_indicator_selector(tmp_path):
    """In the indicator form the selectors n are fixed by the mapped values and count too: 40 + 10 = 50 (spec 2155)."""
    path = _variant(tmp_path, encoding="indicator")
    result = _solve(path, warm_start=_source_full())
    assert result["warm_start"]["values"] == _values(indicator=True)


# ===========================================================================
# Modes and completion (spec 2157-2178, 1842-1852)
# ===========================================================================
def test_modes_report_and_values(tmp_path):
    """Modes start, hint and both: mode echoed, applied true, values counted once (spec 2160-2164, 1836)."""
    source = _source_full()
    for mode in ("start", "hint", "both"):
        report = _solve(BASE, warm_start=source, mode=mode)["warm_start"]
        assert report["applied"] is True
        assert report["mode"] == mode
        assert _pair(report["source_horizon"]) == (2, 3)
        assert report["values"] == _values()
        # a result-like source does not request a completion
        assert report["completion"]["requested"] is False
        assert report["completion"]["status"] == "not_requested"


def test_hint_never_completes(tmp_path):
    """Mode hint: no completion even if the source requests it; requested is the flag of the source (spec 2162, 2171)."""
    source = dict(_source_full(), complete_binary_start=True)
    completion = _solve(BASE, warm_start=source, mode="hint")["warm_start"]["completion"]
    assert completion["requested"] is True
    assert completion["status"] == "not_requested"


def test_completion_runs_for_start_and_both(tmp_path):
    """A source with complete_binary_start runs the completion in the modes start and both (spec 2161, 2163, 2168-2177).

    Every schedule is feasible, so the copy with all binaries fixed has a solution: status completed.
    All mapped binaries are fixed: fixed_integer_variables equals the number of values.
    """
    source = dict(_source_full(), complete_binary_start=True)
    for mode in ("start", "both"):
        report = _solve(BASE, warm_start=source, mode=mode)["warm_start"]
        completion = report["completion"]
        assert completion["requested"] is True
        assert completion["status"] == "completed"
        assert completion["fixed_integer_variables"] == report["values"] == _values()
        assert completion["values_copied"] > 0
        assert completion["runtime_seconds"] >= 0
        assert completion["iis_constraints"] == []


def test_greedy_source_is_completed(tmp_path):
    """The greedy source requests the completion (spec 2168, 2230): requested true, status completed (spec 2161)."""
    pytest.importorskip("gurobipy")
    from fleet_management.config import load_config
    from fleet_management.greedy_warm_start import build_greedy_warm_start
    source, _ = build_greedy_warm_start(load_config(_read("zero_increments.yaml")))
    report = _solve(BASE, warm_start=source, mode="start")["warm_start"]
    assert report["applied"] is True
    assert report["mode"] == "start"
    assert _pair(report["source_horizon"]) == (2, 3)
    assert report["completion"]["requested"] is True
    assert report["completion"]["status"] == "completed"
    # the same greedy source in the mode hint: no completion
    hint = _solve(BASE, warm_start=source, mode="hint")["warm_start"]
    assert hint["completion"]["requested"] is True
    assert hint["completion"]["status"] == "not_requested"


def test_completion_not_run_without_values(tmp_path):
    """The completion runs only if the mapping set a value (spec 2169): a source without x runs none."""
    source = {"H1": 2, "H2": 3, "complete_binary_start": True}
    completion = _solve(BASE, warm_start=source)["warm_start"]["completion"]
    assert completion["status"] == "not_requested"


# ===========================================================================
# Relaxation start (spec 2180-2195, 1854-1863)
# ===========================================================================
def test_relaxation_disabled(tmp_path):
    """Option false: enabled false, status disabled (spec 1858-1859)."""
    report = _solve(BASE)["relaxation_warm_start"]
    assert report["enabled"] is False
    assert report["status"] == "disabled"


def test_relaxation_skipped_with_a_start_or_hint(tmp_path):
    """Option true and a start or hint applied: status skipped (spec 2186, 1860)."""
    path = _variant(tmp_path, relaxation_warm_start=True)
    for mode in ("start", "hint", "both"):
        result = _solve(path, warm_start=_source_full(), mode=mode)
        assert result["warm_start"]["applied"] is True
        report = result["relaxation_warm_start"]
        assert report["enabled"] is True
        assert report["status"] == "skipped"


def test_relaxation_runs_without_a_start(tmp_path):
    """Option true and nothing applied: the status is a Gurobi status string, values_copied >= 0 (spec 2185-2193, 1861-1863)."""
    path = _variant(tmp_path, relaxation_warm_start=True, relaxation_time_limit=10)
    for source in (None, {"H1": 2, "H2": 3}):
        result = _solve(path, warm_start=source)
        assert result["warm_start"]["applied"] is False
        report = result["relaxation_warm_start"]
        assert report["enabled"] is True
        assert isinstance(report["status"], str)
        assert report["status"] not in ("disabled", "skipped")
        assert report["values_copied"] >= 0
        assert report["runtime_seconds"] >= 0
        assert "relaxation_objective" in report


# ===========================================================================
# Warm starts in sweeps (spec 2197-2204), with a fake solve (spec 1949)
# ===========================================================================
def _fake_solver(monkeypatch, outcomes, calls):
    """Replace horizon_sweep.solve_mapping; outcomes[(H1, H2)] is "ok", "error" or "nosol"."""
    from fleet_management import horizon_sweep

    def fake(data, results_path=None, *, warm_start=None, warm_start_mode="start"):
        H1, H2 = data["H"]
        F, M, L, T = data["F"], data["M"], data["L"], H1 + H2
        calls.append({"H1": H1, "H2": H2, "warm_start": warm_start, "mode": warm_start_mode})
        outcome = outcomes.get((H1, H2), "ok")
        if outcome == "error":
            raise RuntimeError("fake failure")
        result = {
            "status": "optimal", "objective": float(H2), "bound": float(H2), "mip_gap": 0.0,
            "F": F, "M": M, "L": L, "H1": H1, "H2": H2, "T": T,
            "objective_mode": data.get("objective_mode"), "evaluation_horizon": None,
            "x": np.zeros((F, M + 1, T), dtype=int),
            "m": np.zeros((F, L, T), dtype=int),
            "r": np.zeros((F, L, T), dtype=int),
            "J_initialization": 1.0, "J_op": float(H2), "J_op_average": 1.0,
            "performance": {"requested_mip_gap": 0.1, "requested_time_limit_seconds": 20.0},
        }
        if outcome == "nosol":
            result.update({"status": "infeasible", "objective": None, "bound": None,
                           "mip_gap": None, "x": None, "m": None, "r": None,
                           "J_initialization": None, "J_op": None, "J_op_average": None})
        return result

    monkeypatch.setattr(horizon_sweep, "solve_mapping", fake)


def _source_horizon_of(call):
    warm = call["warm_start"]
    return None if warm is None else (warm["H1"], warm["H2"])


def test_sweep_sources_skip_error_and_no_solution_cases(monkeypatch):
    """Operating sweep with warm_start: the source is the nearest preceding case with a solution (spec 2201-2203).

    H1 = 2, H2 in 2..6: the case H2 = 3 raises, H2 = 4 has no solution.  Sources by hand:
    H2 = 2: none (first case); 3: case 2; 4: case 2 (3 is an error); 5: case 2 (4 has no solution);
    6: case 5.  The mode is passed through.
    """
    from fleet_management.horizon_sweep import sweep_operating_horizons
    calls = []
    _fake_solver(monkeypatch, {(2, 3): "error", (2, 4): "nosol"}, calls)
    report = sweep_operating_horizons(str(BASE), [2, 3, 4, 5, 6], warm_start=True,
                                      warm_start_mode="hint")
    assert [(c["H1"], c["H2"]) for c in calls] == [(2, 2), (2, 3), (2, 4), (2, 5), (2, 6)]
    assert [_source_horizon_of(c) for c in calls] == [None, (2, 2), (2, 2), (2, 2), (2, 5)]
    for call in calls[1:]:
        warm = call["warm_start"]
        assert np.asarray(warm["x"]).shape[-1] == warm["H1"] + warm["H2"]
        assert call["mode"] == "hint"
    assert [row["H2"] for row in report["cases"]] == [2, 3, 4, 5, 6]
    assert report["cases"][1]["status"].startswith("error")


def test_sweep_without_warm_start_has_no_sources(monkeypatch):
    """With warm_start false (the default) no case gets a source (spec 2201)."""
    from fleet_management.horizon_sweep import sweep_operating_horizons
    calls = []
    _fake_solver(monkeypatch, {}, calls)
    sweep_operating_horizons(str(BASE), [2, 3, 4])
    assert [c["warm_start"] for c in calls] == [None, None, None]


def test_grid_first_case_of_each_h1_has_no_source(monkeypatch):
    """Grid, H1 in {1, 2}, H2 in {2, 3}: the first case of each H1 has no source, even though an earlier H1 has cases (spec 2202).

    Calls in order (1,2), (1,3), (2,2), (2,3); sources: none, case (1,2), none, case (2,2).
    """
    from fleet_management.horizon_sweep import sweep_horizon_grid
    calls = []
    _fake_solver(monkeypatch, {}, calls)
    sweep_horizon_grid(str(BASE), [2, 1], [3, 2])
    assert [(c["H1"], c["H2"]) for c in calls] == [(1, 2), (1, 3), (2, 2), (2, 3)]
    assert [_source_horizon_of(c) for c in calls] == [None, (1, 2), None, (2, 2)]


def test_grid_error_case_is_not_a_source(monkeypatch):
    """Grid: the case (1, 2) raises, so (1, 3) has no source; the sources never cross H1 (spec 2201-2203)."""
    from fleet_management.horizon_sweep import sweep_horizon_grid
    calls = []
    _fake_solver(monkeypatch, {(1, 2): "error"}, calls)
    sweep_horizon_grid(str(BASE), [1, 2], [2, 3])
    assert [(c["H1"], c["H2"]) for c in calls] == [(1, 2), (1, 3), (2, 2), (2, 3)]
    assert [_source_horizon_of(c) for c in calls] == [None, None, None, (2, 2)]


# ===========================================================================
# Greedy heuristic (spec 2206-2273); no Gurobi
# ===========================================================================
REPORT_KEYS = {"repairs", "replacements", "repeatability_feasible", "repeatability_failures",
               "maximum_repeatability_excess", "maximum_proxy_ratio", "assignment_order"}
SOURCE_KEYS = {"H1", "H2", "x", "m", "r", "complete_binary_start", "completion_time_limit"}


def _greedy(name, **kwargs):
    from fleet_management.config import load_config
    from fleet_management.greedy_warm_start import build_greedy_warm_start
    cfg = load_config(_read(name))
    return build_greedy_warm_start(cfg, **kwargs)


def _check_greedy(name, F, M, L, H1, H2, flights, depots=(), repairs=(), replacements=(),
                  order=None, failures=0, excess=0.0, proxy=None, **kwargs):
    """Run the heuristic on a hand input and compare the schedule and the report with the hand values."""
    T = H1 + H2
    x, m, r = _schedule(F, M, L, T, flights, depots, repairs, replacements)
    source, report = _greedy(name, **kwargs)
    # keys and plain values of the source (spec 2227-2232)
    assert set(source) == SOURCE_KEYS
    assert (source["H1"], source["H2"]) == (H1, H2)
    assert source["complete_binary_start"] is True
    assert source["completion_time_limit"] == 60
    assert np.asarray(source["x"]).shape == (F, M + 1, T)
    assert np.asarray(source["m"]).shape == (F, L, T)
    assert np.asarray(source["r"]).shape == (F, L, T)
    # the schedule
    assert np.array_equal(np.rint(source["x"]).astype(int), x), name
    assert np.array_equal(np.rint(source["m"]).astype(int), m), name
    assert np.array_equal(np.rint(source["r"]).astype(int), r), name
    # the report (spec 2263-2272)
    rep = report.as_dict()
    assert set(rep) == REPORT_KEYS
    assert type(rep["repairs"]) is int and rep["repairs"] == int(m.sum())
    assert type(rep["replacements"]) is int and rep["replacements"] == int(r.sum())
    assert type(rep["repeatability_feasible"]) is bool
    assert rep["repeatability_feasible"] is (failures == 0)
    assert type(rep["repeatability_failures"]) is int and rep["repeatability_failures"] == failures
    assert type(rep["maximum_repeatability_excess"]) is float
    assert rep["maximum_repeatability_excess"] == pytest.approx(excess, abs=1e-12)
    assert type(rep["maximum_proxy_ratio"]) is float
    if proxy is not None:
        assert rep["maximum_proxy_ratio"] == pytest.approx(proxy, abs=1e-12)
    if order is not None:
        assert rep["assignment_order"] == order
    assert all(type(j) is int for day in rep["assignment_order"] for j in day)
    assert len(rep["assignment_order"]) == T
    return source, rep


# ---- construction pass --------------------------------------------------------------------
def test_greedy_construction_repairs_above_trigger():
    """greedy_A, repair_trigger 0.45: mission order, train choice and repairs by hand (spec 2237-2243).

    F = 3, M = 2, L = 1, H = [2, 1], tau = 1, rho = 0.5, state mu_0 = (0.1, 0.2, 0.0).
    Increments (all trains): day 1 j1 0.30, j2 0.50; day 2 j1 0.10, j2 0.20; day 3 zero.
    Day 1: order by the largest increment: j2 (0.50), j1 (0.30) -> [2, 1].
      j2: keys 0.6, 0.7, 0.5 -> train 3 (state 0.5); j1: trains 1, 2: 0.4, 0.5 -> train 1 (0.4).
      Train 2 idle with 0.2 <= 0.45: no repair.  State (0.4, 0.2, 0.5).
    Day 2: [2, 1]; j2: 0.6, 0.4, 0.7 -> train 2 (0.4); j1: trains 1, 3: 0.5, 0.6 -> train 1 (0.5).
      Train 3 idle with 0.5 > 0.45: depot and repair, state 0.25.  State (0.5, 0.4, 0.25).
    Day 3 (increments 0, tie of missions -> larger j first, [2, 1]):
      j2 -> smallest state: train 3 (0.25); j1 -> trains 1, 2 (0.5, 0.4): train 2.
      Train 1 idle with 0.5 > 0.45: depot and repair.
    Loop pass: no cell has an excess (the means end at or below their day-H1 values).
    Proxy ratio: the largest state is 0.5.
    """
    _check_greedy("greedy_A.yaml", 3, 2, 1, 2, 1,
                  flights={1: {2: 3, 1: 1}, 2: {2: 2, 1: 1}, 3: {2: 3, 1: 2}},
                  depots=[(3, 2), (1, 3)], repairs=[(3, 1, 2), (1, 1, 3)],
                  order=[[2, 1], [2, 1], [2, 1]], proxy=0.5, repair_trigger=0.45)


def test_greedy_default_trigger_is_070():
    """greedy_A with the default trigger 0.70: no state exceeds 0.7, so no repair; the same assignments otherwise.

    Days 1 and 2 as in the test above.  State after day 2: (0.5, 0.4, 0.5).  Day 3 (zero
    increments, [2, 1]): j2 -> smallest state: train 2 (0.4); j1 -> trains 1, 3 tie (0.5, 0.5):
    smaller i, train 1.  Train 3 idle with 0.5 <= 0.7.
    """
    _check_greedy("greedy_A.yaml", 3, 2, 1, 2, 1,
                  flights={1: {2: 3, 1: 1}, 2: {2: 2, 1: 1}, 3: {2: 2, 1: 1}},
                  order=[[2, 1], [2, 1], [2, 1]], proxy=0.5)


def test_greedy_ties_mission_larger_j_train_smaller_i():
    """greedy_B: all states 0, both missions with increment 0.2 on day 1 and 0 on day 2 (spec 2237-2239).

    F = 3, M = 2, H = [1, 1].  Day 1: mission tie -> [2, 1].  j2: all keys 0.2 -> smallest i,
    train 1.  j1: trains 2, 3 tie -> train 2.  Train 3 idle.  State (0.2, 0.2, 0).
    Day 2 (zero increments, tie -> [2, 1]): j2 -> train 3 (key 0); j1 -> trains 1, 2 tie (0.2, 0.2)
    -> train 1.  Train 2 idle.  No state exceeds 0.7.  Proxy ratio 0.2.
    """
    _check_greedy("greedy_B.yaml", 3, 2, 1, 1, 1,
                  flights={1: {2: 1, 1: 2}, 2: {2: 3, 1: 1}},
                  order=[[2, 1], [2, 1]], proxy=0.2)


def test_greedy_train_key_is_max_ratio_then_sum():
    """greedy_C: two components; the key is (largest ratio, sum of ratios) (spec 2238-2239).

    F = 2, M = 1, L = 2, H = [1, 1].  State train 1 (0.3, 0.3), train 2 (0.3, 0.0); increment
    (0.1, 0.1) on day 1.  Keys: train 1 (0.4, 0.8); train 2 (0.4, 0.5).  The largest ratios tie;
    the sum decides: train 2 flies (state (0.4, 0.1)); train 1 idle.
    Day 2 (zero increments): train 1 key (0.3, 0.6), train 2 key (0.4, 0.5): train 1 flies.
    Proxy ratio 0.4.
    """
    _check_greedy("greedy_C.yaml", 2, 1, 2, 1, 1,
                  flights={1: {1: 2}, 2: {1: 1}}, order=[[1], [1]], proxy=0.4)


def test_greedy_repair_factor_is_one_minus_rho_for_every_law():
    """greedy_D, repair_model ard1, default trigger 0.70: the proxy state is multiplied by (1 - rho) (spec 2241).

    F = 2, M = 1, H = [2, 1], rho = 0.5, state (0.25, 0.25); increments day 1: 0.5, day 2: 0.125,
    day 3: 0.
    Day 1: keys 0.75, 0.75 tie -> train 1 flies, state 0.75; train 2 idle (0.25).
    Day 2: keys 0.875, 0.375 -> train 2 flies (0.375); train 1 idle with 0.75 > 0.7: repair,
      state 0.375 (the ard1 law with the anchor 0.25 would give 0.5).
    Day 3 (zero): keys 0.375, 0.375 tie -> train 1 flies (with 0.5 train 2 would).
    The true ard1 trajectory has no loop excess: train 1 ends at 0.5 = its value on day 2, the
    anchor at 0.5 = 0.5; train 2 ends at 0.375 = 0.375.  Proxy ratio 0.75.
    """
    _check_greedy("greedy_D.yaml", 2, 1, 1, 2, 1,
                  flights={1: {1: 1}, 2: {1: 2}, 3: {1: 1}}, depots=[(1, 2)],
                  repairs=[(1, 1, 2)], order=[[1], [1], [1]], proxy=0.75)


def test_greedy_trigger_is_strict():
    """greedy_D with repair_trigger 0.75: a state equal to r tau is not repaired (spec 2241: "> r tau").

    Day 1: train 1 flies, state 0.75.  Day 2: train 2 flies; train 1 idle with 0.75, not above
    0.75: idle.  Day 3: keys 0.75 (train 1) and 0.375 (train 2): train 2 flies.  Loop pass:
    train 1 ends at 0.75 = 0.75 on day 2, no excess.
    """
    _check_greedy("greedy_D.yaml", 2, 1, 1, 2, 1,
                  flights={1: {1: 1}, 2: {1: 2}, 3: {1: 2}}, order=[[1], [1], [1]],
                  proxy=0.75, repair_trigger=0.75)


def test_greedy_repairs_only_the_components_above_the_trigger():
    """greedy_E, repair_trigger 0.5: only the component above r tau is repaired (spec 2241).

    F = 2, M = 1, L = 2, H = [1, 1], rho = 0.5, state train 1 (0.75, 0.25), train 2 (0.25, 0.25),
    increment (0.125, 0.125) on day 1.  Keys: train 1 (0.875, 1.0), train 2 (0.375, 0.75):
    train 2 flies.  Train 1 idle: component 1 has 0.75 > 0.5, component 2 has 0.25: depot day,
    repair of component 1 only.  Day 2 (zero): train 1 (0.375, 0.25) key (0.375, 0.625); train 2
    (0.375, 0.375) key (0.375, 0.75): train 1 flies.  No loop excess.
    """
    _check_greedy("greedy_E.yaml", 2, 1, 2, 1, 1,
                  flights={1: {1: 2}, 2: {1: 1}}, depots=[(1, 1)], repairs=[(1, 1, 1)],
                  order=[[1], [1]], repair_trigger=0.5)


def test_greedy_trigger_errors_and_bounds():
    """repair_trigger outside [0, 1] raises ValueError; 0 and 1 are accepted (spec 2219, 2224)."""
    for bad in (1.2, -0.1, 1.0000001, 2):
        with pytest.raises(ValueError):
            _greedy("greedy_B.yaml", repair_trigger=bad)
    for good in (0.0, 1.0):
        source, report = _greedy("greedy_B.yaml", repair_trigger=good)
        assert set(source) == SOURCE_KEYS
        assert set(report.as_dict()) == REPORT_KEYS


# ---- loop pass ----------------------------------------------------------------------------
def test_greedy_loop_repair_tie_goes_to_the_later_day():
    """greedy_L1: one repair in the operating phase; two free days give the same excess (spec 2256).

    F = 2, M = 1, H = [1, 3], rho = 1, state 0.  Increments: day 1 zero; operating 0.25, 0, 0.
    Construction: day 1 tie -> train 1; day 2 keys 0.25, 0.25 -> train 1 (state 0.25); days 3, 4
    (increment 0): train 2 (key 0 < 0.25).  No state above 0.7: no repair.
    Loop pass, train 1 (flies days 1, 2): means 0, 0.25, 0.25, 0.25; the mean on day 4 exceeds
    that of day H1 = 1 (0) by 0.25 = the excess.  Free days: 3, 4.  A repair on day 3 or on day 4
    resets the mean to 0 (rho = 1) and gives excess 0.  The tie goes to the later day: day 4.
    Train 2 flies days 3, 4 with zero increments: no excess.  Excess after the pass 0; the largest
    state is 0.25.
    """
    _check_greedy("greedy_L1.yaml", 2, 1, 1, 1, 3,
                  flights={1: {1: 1}, 2: {1: 1}, 3: {1: 2}, 4: {1: 2}}, depots=[(1, 4)],
                  repairs=[(1, 1, 4)], order=[[1]] * 4, proxy=0.25)


def test_greedy_loop_replacement_when_no_repair_helps():
    """greedy_L2, repair_model ard1: the repairs do not lower the excess, so replacements are placed (spec 2256-2257).

    F = 2, M = 1, H = [1, 2], rho = 0.5, state (0.5, 0.25), anchors equal to the state,
    increments 0.125 on day 1 and on both operating days, tau = 1.
    Construction: day 1 keys 0.625, 0.375 -> train 2; day 2 keys 0.625, 0.5 -> train 2 (0.5);
    day 3 keys 0.625, 0.625 tie -> train 1.  Idle states 0.5 <= 0.7: no repair.
    Train 1 (flies day 3): mean 0.5, 0.5, 0.625; excess 0.125 (day 3 against day 1: 0.625 - 0.5).
      Free day: 2.  A repair on day 2 gives (1 - rho) 0.5 + rho 0.5 = 0.5 (state equals anchor):
      nothing changes, the excess is not lowered.  A replacement on day 2 gives mean 0, then 0.125
      on day 3: no loop excess (0.125 <= 0.5; the anchor 0 <= 0.5).
    Train 2 (flies days 1, 2): means 0.375, 0.5, 0.5; excess 0.125 (0.5 - 0.375).  Free day: 3.
      A repair on day 3 gives 0.5 * 0.5 + 0.5 * 0.25 = 0.375 and moves the anchor to 0.375, which
      exceeds the anchor 0.25 of day 1 by 0.125: the excess is 0.125 again, not lowered.  A
      replacement on day 3 gives mean 0 and anchor 0: excess 0.
    Largest state: 0.625.  Excess after the pass 0.
    """
    _check_greedy("greedy_L2.yaml", 2, 1, 1, 1, 2,
                  flights={1: {1: 2}, 2: {1: 2}, 3: {1: 1}}, depots=[(1, 2), (2, 3)],
                  replacements=[(1, 1, 2), (2, 1, 3)], order=[[1]] * 3, proxy=0.625)


def test_greedy_failure_is_counted_and_start_is_returned():
    """greedy_F: a cell stays above its day-H1 value after its best repair; one failure counted (spec 2258, 2261, 2269-2270).

    F = 2, M = 1, H = [1, 3], rho = 1, increments 0 on day 1 and 0.25 on every operating day.
    Construction: day 1 tie -> train 1; day 2 tie (0.25, 0.25) -> train 1 (0.25); day 3 keys 0.5,
    0.25 -> train 2 (0.25); day 4 tie (0.5, 0.5) -> train 1 (0.5).  No repair (all <= 0.7).
    Train 1 flies days 1, 2, 4: means 0, 0.25, 0.25, 0.5; excess 0.5.  Free day: 3.  A repair on
    day 3 gives 0, 0.25 on day 4: excess 0.25 < 0.5, kept.  No free day is left (day 3 has the
    repair, days 2 and 4 are flown) and replacement has no free day: one failure; the excess left is 0.25.
    Train 2 flies day 3 only: means 0, 0, 0.25, 0.25; excess 0.25.  Free days 2 and 4.  A repair on
    day 2 changes nothing (mean 0); on day 4 it gives mean 0: excess 0.  Day 4 is chosen.
    Report: 2 repairs, 0 replacements, 1 failure, not feasible; the largest excess term, before
    the division by tau = 1, is 0.25; the largest state is 0.5.
    """
    _check_greedy("greedy_F.yaml", 2, 1, 1, 1, 3,
                  flights={1: {1: 1}, 2: {1: 1}, 3: {1: 2}, 4: {1: 1}},
                  depots=[(1, 3), (2, 4)], repairs=[(1, 1, 3), (2, 1, 4)],
                  order=[[1]] * 4, failures=1, excess=0.25, proxy=0.5)


def test_greedy_wiener_rule_tie_to_repair_and_to_later_day():
    """greedy_W: at a Wiener cell a repair and a replacement are tried on every free day (spec 2260).

    F = 2, M = 1, L = 1, H = [1, 3], tau = 1, epsilon = 0.1, rho = 1, sigma^2 = 2^-8 per flight,
    state (0.5, 0.25), variance 0, V_max = 1 / Phi^-1(0.9)^2 = 0.6089.  Mean increments: day 1
    0.25; operating 0.25, 0.125, 0.125.  Repair trigger 0.95.
    Construction: day 1 keys 0.75, 0.5 -> train 2 (0.5); day 2 keys 0.75, 0.75 tie -> train 1
    (0.75); day 3 keys 0.875, 0.625 -> train 2 (0.625); day 4 keys 0.875, 0.75 -> train 2.
    Train 1 flies day 2 only.  Means 0.5, 0.75, 0.75, 0.75; excess 0.25 in the mean (and
    sigma^2 / V_max in the variance).  Free days 3, 4.  A repair (rho = 1) gives mean 0 but
    leaves the variance excess sigma^2 / V_max.  A replacement (mean 0, variance 0) gives excess 0 on
    day 3 and on day 4 alike: the later day, 4, is kept.
    Train 2 flies days 1, 3, 4.  Means 0.5, 0.5, 0.625, 0.75; the variance on day 4 is
    3 sigma^2 and on day 1 sigma^2.  Free day: 2.  A repair on day 2 gives mean 0 (day 4: 0.25,
    below 0.5) and the variance excess 2 sigma^2 / V_max.  A replacement on day 2 has replacement_v =
    sigma^2, the variance of day 1, so it gives the same variances and the same means: the two tie
    and the repair is kept.  It lowers the excess from 0.25, so it stays; then no free day is
    left and a failure is counted.  The largest excess term before the division is the variance
    increase 2 sigma^2 = 2^-7 = 0.0078125.  The largest mean is 0.75.
    """
    _check_greedy("greedy_W.yaml", 2, 1, 1, 1, 3,
                  flights={1: {1: 2}, 2: {1: 1}, 3: {1: 2}, 4: {1: 2}},
                  depots=[(1, 4), (2, 2)], repairs=[(2, 1, 2)], replacements=[(1, 1, 4)],
                  order=[[1]] * 4, failures=1, excess=0.0078125, proxy=0.75,
                  repair_trigger=0.95)
