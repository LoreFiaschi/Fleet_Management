"""Tests of the warm starts: the greedy heuristic and the warm-start functions of ``base.py``.

The greedy heuristic (``greedy_warm_start.py``) needs neither Gurobi nor the solver. The
tests of ``apply_binary_warm_start``, ``complete_binary_warm_start`` and ``solve_mixed``
need gurobipy; the two at the end need a real Gurobi (a search over the binaries).
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from fleet_management import greedy_warm_start
from fleet_management.config import load_config

# ---------------------------------------------------------------------------
# The greedy heuristic (no Gurobi)
# ---------------------------------------------------------------------------
_COSTS = {"C_M": 1.0, "C_R": 1.0, "C_rep": 5.0, "C_D": 0.0}

_CASES = {
    # A: Gaussian, ardinf, tangent form; two trains, one mission, H = [2, 2]
    "A_gaussian": dict(F=2, M=1, H=[2, 2], model="gaussian", tau=1.0, epsilon=0.1, rho=0.5,
                       mu_0=0.125, v_0=0.0, mu=0.25, v=0.00390625, reliability_impl="tangent",
                       **_COSTS),
    # F: as A with three trains, mu_0 = 0.25 and rho = 1 (a repair empties the state)
    "F_gaussian": dict(F=3, M=1, H=[2, 2], model="gaussian", tau=1.0, epsilon=0.1, rho=1.0,
                       mu_0=0.25, v_0=0.0, mu=0.25, v=0.00390625, reliability_impl="tangent",
                       **_COSTS),
    # G: linear rows only (inverse Gaussian, Gamma, rainflow Markov and Chernoff)
    "G_linear": dict(F=3, M=1, H=[2, 2],
                     model=["inverse_gaussian", "gamma", "rainflow", "rainflow"],
                     bound_method=["markov", "markov", "markov", "chernoff"], tau=1.0,
                     epsilon=0.5, rho=1.0, mu_0=0.0625, eta=0.25, gamma_beta=8.0,
                     s_chernoff=4.0, mu=[[0.125], [0.0625], [0.0625], [0.125]],
                     cgf=[[0.0], [0.0], [0.0], [0.625]], **_COSTS),
    # B: Wiener; the variance falls only by replacement
    "B_wiener": dict(F=2, M=1, H=[2, 2], model="wiener", tau=1.0, epsilon=0.1, rho=0.5,
                     mu_0=0.125, v_0=0.0, sigma=0.0625, mu=0.25, reliability_impl="tangent",
                     **_COSTS),
    # H: one Wiener and one Gaussian component per train
    "H_wiener_gaussian": dict(F=2, M=1, H=[2, 2], model=["wiener", "gaussian"], tau=1.0,
                              epsilon=0.1, rho=0.5, mu_0=0.125, v_0=0.0, sigma=[0.0625, 0.0],
                              mu=0.25, v=[[0.0], [0.00390625]], reliability_impl="tangent",
                              **_COSTS),
    # C: Wiener without replacement
    "C_wiener_norepl": dict(F=2, M=1, H=[2, 2], model="wiener", tau=1.0, epsilon=0.1, rho=0.5,
                            mu_0=0.125, v_0=0.0, sigma=0.0625, mu=0.25,
                            reliability_impl="tangent", allow_replacement=False, **_COSTS),
    # D: mixed fleet, ard1, three trains, two missions, three components
    "D_mixed_ard1": dict(F=3, M=2, H=[2, 3], model=["inverse_gaussian", "gamma", "rainflow"],
                         bound_method="chernoff", repair_model="ard1", tau=1.0, epsilon=0.1,
                         rho=0.5, mu_0=0.0625, eta=0.25, gamma_beta=8.0, s_chernoff=4.0,
                         mu=[[0.125, 0.25], [0.0625, 0.125], [0.125, 0.0625]],
                         cgf=[[0.625, 1.25], [0.3125, 0.625], [0.625, 0.3125]], **_COSTS),
}

# case -> (activities per train and day, repairs, replacements as (i, l, day) with the day
# counted from 1, report: repairs, replacements, failures, maximum excess, maximum ratio).
# The activity is the mission j, 0 for the depot and -1 for idle. At a Wiener cell the
# repairs and the replacements are tried together, so B and H differ from the spec order.
_GREEDY = {
    "A_gaussian": ([[1, -1, 1, 0], [-1, 1, 0, 1]],
                   [(0, 0, 4), (1, 0, 3)], [],
                   (2, 0, 1, 0.0625, 0.625)),
    "F_gaussian": ([[1, -1, 0, 1], [-1, 1, -1, -1], [-1, -1, 1, 0]],
                   [(0, 0, 3), (2, 0, 4)], [],
                   (2, 0, 0, 0.0, 0.75)),
    "G_linear": ([[1, -1, 0, 1], [-1, 1, -1, -1], [-1, -1, 1, 0]],
                 [(0, l, 3) for l in range(4)] + [(2, l, 4) for l in range(4)], [],
                 (8, 0, 0, 0.0, 0.3125)),
    "B_wiener": ([[1, -1, 1, 0], [-1, 1, 0, 1]],
                 [], [(0, 0, 4), (1, 0, 3)],
                 (0, 2, 0, 0.0, 0.625)),
    "H_wiener_gaussian": ([[1, -1, 1, 0], [-1, 1, 0, 1]],
                          [(0, 1, 4), (1, 1, 3)], [(0, 0, 4), (1, 0, 3)],
                          (2, 2, 1, 0.0625, 0.625)),
    "C_wiener_norepl": ([[1, -1, 1, 0], [-1, 1, 0, 1]],
                        [(0, 0, 4), (1, 0, 3)], [],
                        (2, 0, 2, 0.0625, 0.625)),
    "D_mixed_ard1": ([[2, -1, 2, 0, 2], [1, 1, 0, 2, 1], [-1, 2, 1, 1, 0]],
                     [(0, l, 4) for l in range(3)] + [(1, l, 3) for l in range(3)] + [(2, 2, 5)],
                     [(2, 0, 5), (2, 1, 5)],
                     (7, 2, 8, 0.78125, 0.8125)),
}


def _config(name, **changes):
    return load_config(dict(_CASES[name], verbose=0, **changes))


def _activities(x):
    return [[int(np.argmax(x[i, :, k])) if x[i, :, k].sum() > 0.5 else -1
             for k in range(x.shape[2])] for i in range(x.shape[0])]


def _schedule(cfg, entries):
    out = np.zeros((cfg.F, cfg.L, cfg.T))
    for i, l, day in entries:
        out[i, l, day - 1] = 1.0
    return out


@pytest.mark.parametrize("name", list(_GREEDY))
def test_greedy_cases(name):
    activities, repairs, replacements, expected = _GREEDY[name]
    cfg = _config(name)
    source, report = greedy_warm_start.build_greedy_warm_start(cfg)
    assert _activities(source["x"]) == activities
    np.testing.assert_array_equal(source["m"], _schedule(cfg, repairs))
    np.testing.assert_array_equal(source["r"], _schedule(cfg, replacements))
    report = report.as_dict()
    n_repairs, n_replacements, failures, excess, ratio = expected
    assert (report["repairs"], report["replacements"]) == (n_repairs, n_replacements)
    assert report["repeatability_failures"] == failures
    assert report["repeatability_feasible"] == (failures == 0)
    assert report["maximum_repeatability_excess"] == pytest.approx(excess)
    assert report["maximum_proxy_ratio"] == pytest.approx(ratio)
    assert report["assignment_order"] == [list(range(cfg.M, 0, -1))] * cfg.T


@pytest.mark.parametrize("trigger", [-0.1, 1.1, float("nan")])
def test_repair_trigger_range(trigger):
    with pytest.raises(ValueError, match="must lie in"):
        greedy_warm_start.build_greedy_warm_start(_config("F_gaussian"), repair_trigger=trigger)


def test_source_and_report_keys():
    source, report = greedy_warm_start.build_greedy_warm_start(_config("F_gaussian"))
    assert list(source) == ["H1", "H2", "x", "m", "r", "complete_binary_start",
                            "completion_time_limit"]
    assert source["complete_binary_start"] is True
    assert source["completion_time_limit"] == 60.0
    assert (source["H1"], source["H2"]) == (2, 2)
    assert source["x"].shape == (3, 2, 4)
    assert source["m"].shape == (3, 1, 4) and source["r"].shape == (3, 1, 4)
    assert list(report.as_dict()) == [
        "repairs", "replacements", "repeatability_feasible", "repeatability_failures",
        "maximum_repeatability_excess", "maximum_proxy_ratio", "assignment_order"]
    assert yaml.safe_load(yaml.safe_dump(report.as_dict())) == report.as_dict()


def test_greedy_needs_no_gurobi():
    # the heuristic and the caps it imports run without gurobipy
    src = Path(greedy_warm_start.__file__).resolve().parents[1]
    code = "\n".join([
        "import sys",
        "sys.modules['gurobipy'] = None",
        "from fleet_management.config import load_config",
        "from fleet_management.greedy_warm_start import build_greedy_warm_start",
        f"cfg = load_config({dict(_CASES['G_linear'], verbose=0)!r})",
        "source, report = build_greedy_warm_start(cfg)",
        "assert report.repairs == 8 and report.repeatability_feasible",
    ])
    path = os.pathsep.join([str(src), os.environ.get("PYTHONPATH", "")])
    env = dict(os.environ, PYTHONPATH=path)
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


# ---------------------------------------------------------------------------
# Day mapping, start, hint and completion (need gurobipy)
# ---------------------------------------------------------------------------
def _base():
    pytest.importorskip("gurobipy")
    from fleet_management.degradation_model import base
    return base


def _model_input(**changes):
    """Three trains, one mission, one inverse Gaussian component, H = [2, 3], indicator encoding."""
    data = {"F": 3, "M": 1, "H": [2, 3], "model": "inverse_gaussian", "encoding": "indicator",
            "tau": 1.0, "epsilon": 0.5, "rho": 1.0, "mu_0": 0.0625, "eta": 0.25, "mu": 0.125,
            "C_M": 1.0, "C_R": 1.0, "C_rep": 5.0, "C_D": 0.0, "verbose": 0}
    data.update(changes)
    return data


def _source(**changes):
    """A source of two trains with H1 = 1 and H2 = 3 (four days)."""
    x = np.zeros((2, 2, 4))
    x[0, 1], x[0, 0] = [1, 0, 1, 0], [0, 1, 0, 0]
    x[1, 1], x[1, 0] = [0, 1, 0, 1], [0, 0, 1, 0]
    m = np.zeros((2, 1, 4))
    m[0, 0], m[1, 0] = [0, 1, 0, 0], [0, 0, 1, 0]
    r = np.zeros((2, 1, 4))
    r[0, 0] = [0, 0, 0, 1]
    source = {"H1": 1, "H2": 3, "x": x, "m": m, "r": r}
    source.update(changes)
    return source


def _context(base, data):
    cfg = load_config(data)
    ctx = base.build_fleet(cfg, base.resolve_run_options(cfg))
    ctx.model.update()
    return ctx


def test_warm_start_day():
    base = _base()
    assert [base.warm_start_day(k, 2, 1, 3) for k in range(5)] == [0, 1, 1, 2, 3]
    assert [base.warm_start_day(k, 1, 3, 2) for k in range(4)] == [0, 3, 4, 3]


# source (i, a) -> values on the five days; train 2 has no value
_EXPECTED = {
    "x": {(0, 0): [0, 1, 1, 0, 0], (0, 1): [1, 0, 0, 1, 0],
          (1, 0): [0, 0, 0, 1, 0], (1, 1): [0, 1, 1, 0, 1]},
    "m": {(0, 0): [0, 1, 1, 0, 0], (1, 0): [0, 0, 0, 1, 0]},
    "r": {(0, 0): [0, 0, 0, 0, 1], (1, 0): [0, 0, 0, 0, 0]},
    "n": {(0, 0): [1, 0, 0, 1, 0], (1, 0): [1, 1, 1, 0, 1]},
}


@pytest.mark.parametrize("mode", ["start", "hint", "both"])
def test_apply_modes(mode):
    base = _base()
    ctx = _context(base, _model_input())
    report = base.apply_binary_warm_start(ctx, _source(), mode)
    ctx.model.update()
    assert report == {"applied": True, "values": 50, "mode": mode, "source_horizon": [1, 3]}
    groups = {"x": ctx.x, "m": ctx.m_rep, "r": ctx.r_rep, "n": ctx.nb}
    for attribute, active in (("Start", mode in ("start", "both")),
                              ("VarHintVal", mode in ("hint", "both"))):
        for name, variables in groups.items():
            for (i, a, k), var in variables.items():
                value = getattr(var, attribute)
                if active and i < 2:
                    assert value == _EXPECTED[name][i, a][k], (attribute, name, i, a, k)
                else:
                    assert value >= 1e100, (attribute, name, i, a, k)


def test_apply_partial_sources():
    base = _base()
    ctx = _context(base, _model_input())
    report = base.apply_binary_warm_start(ctx, _source(m=None, r=None))
    ctx.model.update()
    assert report["values"] == 30
    for i in range(3):
        for k in range(5):
            assert (ctx.m_rep[i, 0, k].Start == 0.0) == (i < 2)     # no m: zero, for the trains of x
            assert ctx.r_rep[i, 0, k].Start >= 1e100
            assert ctx.nb[i, 0, k].Start >= 1e100
    # without replacement: no r variable, and r' is not read; the selector is 1 - m
    for source in (_source(), _source(r=None)):
        ctx = _context(base, _model_input(allow_replacement=False))
        report = base.apply_binary_warm_start(ctx, source)
        ctx.model.update()
        assert report["values"] == 40
        assert [ctx.nb[0, 0, k].Start for k in range(5)] == [1, 0, 0, 1, 1]
    # no x: no effect
    empty = {"applied": False, "values": 0, "mode": None, "source_horizon": None}
    ctx = _context(base, _model_input())
    assert base.apply_binary_warm_start(ctx, _source(x=None)) == empty
    assert base.apply_binary_warm_start(ctx, None) == empty
    # a source that does not fit
    source = _source()
    del source["H1"]
    with pytest.raises(ValueError, match="must carry"):
        base.apply_binary_warm_start(ctx, source)
    with pytest.raises(ValueError, match="must have the shape"):
        base.apply_binary_warm_start(ctx, _source(x=np.zeros((2, 2, 5))))


@pytest.mark.parametrize("changes, message", [
    ({"H1": 2.7}, "must carry the integers"),
    ({"H1": True}, "must carry the integers"),
    ({"H2": 0}, "must carry the integers"),
    ({"H2": float("nan")}, "must carry the integers"),
    ({"m": np.zeros((2, 1, 3))}, "'m' of a warm-start source must have the shape"),
    ({"r": np.zeros((2, 4))}, "'r' of a warm-start source must have the shape"),
])
def test_invalid_sources(changes, message):
    base = _base()
    ctx = _context(base, _model_input())
    with pytest.raises(ValueError, match=message):
        base.apply_binary_warm_start(ctx, _source(**changes))
    # a number with zero fractional part is an integer, as for the input keys
    report = base.apply_binary_warm_start(ctx, _source(H1=1.0, H2=3.0))
    assert report["values"] == 50 and report["source_horizon"] == [1, 3]


def test_invalid_mode():
    base = _base()
    ctx = _context(base, _model_input())
    with pytest.raises(ValueError, match="warm_start_mode must be"):
        base.apply_binary_warm_start(ctx, _source(), "hints")
    with pytest.raises(ValueError, match="warm_start_mode must be"):
        base.complete_binary_warm_start(ctx.model, _source(), {"applied": True}, "hints")
    with pytest.raises(ValueError, match="warm_start_mode must be"):
        base.solve_mixed(load_config(_model_input()), warm_start_mode="hints")


@pytest.mark.parametrize("mode", ["start", "both"])
def test_completion(mode):
    base = _base()
    cfg = _config("G_linear")
    source, _ = greedy_warm_start.build_greedy_warm_start(cfg)
    out = base.solve_mixed(cfg, warm_start=source, warm_start_mode=mode)
    completion = out["warm_start"]["completion"]
    assert completion["requested"] is True and completion["status"] == "completed"
    assert completion["fixed_integer_variables"] == 120
    assert completion["values_copied"] == out["performance"]["variables"]
    assert out["warm_start"]["values"] == 120 and out["warm_start"]["source_horizon"] == [2, 2]
    assert out["warm_start"]["mode"] == mode
    assert out["status"] == "optimal"


def test_hint_skips_completion():
    base = _base()
    cfg = _config("G_linear")
    source, _ = greedy_warm_start.build_greedy_warm_start(cfg)
    ctx = base.build_fleet(cfg, base.resolve_run_options(cfg))
    ctx.model.update()
    report = base.apply_binary_warm_start(ctx, source, "hint")
    report = base.complete_binary_warm_start(ctx.model, source, report, "hint")
    assert report["applied"] is True and report["mode"] == "hint"
    assert report["completion"]["requested"] is True
    assert report["completion"]["status"] == "not_requested"
    assert report["completion"]["fixed_integer_variables"] == 0


def test_failed_completion():
    # the Wiener variance cannot fall without a replacement, and none is allowed
    base = _base()
    cfg = _config("C_wiener_norepl", gurobi_params={"DualReductions": 0})
    source, report = greedy_warm_start.build_greedy_warm_start(cfg)
    assert not report.repeatability_feasible
    ctx = base.build_fleet(cfg, base.resolve_run_options(cfg))
    ctx.model.update()
    report = base.apply_binary_warm_start(ctx, source)
    report = base.complete_binary_warm_start(ctx.model, source, report)
    completion = report["completion"]
    assert completion["requested"] is True and completion["fixed_integer_variables"] > 0
    # Gurobi may report INF_OR_UNBD if the copy does not keep DualReductions = 0
    assert completion["status"] in ("infeasible", "infeasible_or_unbounded")
    if completion["status"] == "infeasible":
        assert 0 < len(completion["iis_constraints"]) <= 50
        assert all(isinstance(name, str) for name in completion["iis_constraints"])
    else:
        assert completion["iis_constraints"] == []


# ---------------------------------------------------------------------------
# Gurobi only: the emulator of the tests cannot search over the binaries
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["start", "hint", "both"])
def test_solve_mixed_modes(mode):
    base = _base()
    cfg = _config("G_linear", relaxation_warm_start=True)
    source, _ = greedy_warm_start.build_greedy_warm_start(cfg)
    out = base.solve_mixed(cfg, warm_start=source, warm_start_mode=mode)
    assert out["warm_start"]["mode"] == mode and out["warm_start"]["applied"] is True
    assert out["relaxation_warm_start"]["status"] == "skipped"


def test_relaxation_start():
    base = _base()
    out = base.solve_mixed(_config("G_linear", relaxation_warm_start=True))
    relaxation = out["relaxation_warm_start"]
    assert relaxation["enabled"] is True
    assert relaxation["status"] not in ("disabled", "skipped")
    assert isinstance(relaxation["runtime_seconds"], float)
