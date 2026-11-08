"""Tests of the step-3 changes of ``degradation_model/base.py`` (spec alignment).

* M1: an entry of ``gurobi_params`` is applied after the ``prepare`` hooks, so the user's
  ``NonConvex`` wins (spec, run options: ``gurobi_params`` is set after every other option).
* M4: a Gurobi error inside the warm-start completion or the relaxation start does not stop
  the solve; the status is ``"error"`` (spec, "Start, Hints and Completion").
* M5: a best bound or MIP gap that is not finite is reported as ``None`` ("if available").
* M15: ``completion_time_limit`` of a source must be a finite number >= 0 (invalid source).

M5 and M15 call plain functions and need no Gurobi model. The other tests build models of
at most a few hundred variables and skip without Gurobi.
"""
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

base = pytest.importorskip("fleet_management.degradation_model.base")

DATA = Path(__file__).parent / "data"
WARM = DATA / "tools" / "warm_starts" / "zero_increments.yaml"
GAUSSIAN = DATA / "io" / "valid" / "gaussian.yaml"


# ===========================================================================
# Helpers
# ===========================================================================
def _gurobi():
    """The module ``gurobipy``; the test is skipped without a working one (a stub counts as none)."""
    gp = pytest.importorskip("gurobipy")
    try:
        gp.Model("probe").dispose()
    except RuntimeError:
        pytest.importorskip("gurobipy_is_not_installed_here")
    return gp


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class _Proxy:
    """A Gurobi model that forwards everything, except that ``copy`` returns a proxy whose
    ``optimize`` raises ``GurobiError`` (the completion and the relaxation solve a copy)."""

    def __init__(self, real, gp, failing=False, log=None):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_gp", gp)
        object.__setattr__(self, "_failing", failing)
        object.__setattr__(self, "_log", [] if log is None else log)
        object.__setattr__(self, "disposed", False)

    def __getattr__(self, name):
        return getattr(self._real, name)

    def __setattr__(self, name, value):
        setattr(self._real, name, value)

    def copy(self):
        clone = _Proxy(self._real.copy(), self._gp, failing=True, log=self._log)
        self._log.append(clone)
        return clone

    def optimize(self, *args):
        if self._failing:
            raise self._gp.GurobiError(10005, "test")
        return self._real.optimize(*args)

    def dispose(self):
        object.__setattr__(self, "disposed", True)
        self._real.dispose()


def _failing_copies(monkeypatch, gp):
    """Make every model copy of ``solve_mixed`` fail in ``optimize``; return the list of copies."""
    copies = []
    original = base.build_fleet

    def build_fleet(cfg, opts, *args, **kwargs):
        ctx = original(cfg, opts, *args, **kwargs)
        ctx.model = _Proxy(ctx.model, gp, log=copies)
        return ctx

    monkeypatch.setattr(base, "build_fleet", build_fleet)
    return copies


def _solve(data, warm_start=None):
    from fleet_management.solver import solve_mapping
    return solve_mapping(data, warm_start=warm_start, warm_start_mode="start")


def _source(**changes):
    """A source for the input ``zero_increments.yaml`` (F = 2, M = 1, L = 1, H = [2, 3])."""
    source = {
        "x": np.zeros((2, 2, 5)), "H1": 2, "H2": 3,
        "complete_binary_start": True, "completion_time_limit": 5,
    }
    source.update(changes)
    return source


# ===========================================================================
# M5: a bound or a gap that is not finite is None
# ===========================================================================
@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan, np.inf, np.float64("nan")])
def test_finite_or_none_rejects_non_finite(value):
    assert base.finite_or_none(value) is None


@pytest.mark.parametrize("value", [0.0, 1.5, -3.0, 1e30])
def test_finite_or_none_keeps_finite(value):
    assert base.finite_or_none(value) == value


@pytest.mark.parametrize("bound", [math.inf, -math.inf, math.nan, 12.5])
def test_extract_solution_reports_a_non_finite_bound_as_none(bound):
    """The result entry ``bound`` is None for ``inf`` or ``NaN`` (spec: "if available")."""
    from types import SimpleNamespace
    model = SimpleNamespace(ObjBound=bound, MIPGap=math.inf, SolCount=0, Status=2)
    ctx = SimpleNamespace(
        model=model, F=1, M=1, L=1, T=2, H1=1, H2=1,
        opts={"objective_mode": "evaluation_total", "evaluation_horizon": 2},
    )
    cfg = SimpleNamespace(
        model=np.array([["gaussian"]]), component_names=["c"], tau=1.0, mu_0=0.0, v_0=0.0,
        bound_method=np.array([[""]]), repair_model=np.array([["ard1"]]),
        reliability_impl=np.array([["exact"]]), encoding=np.array([["bigm"]]), H=[1, 1],
    )
    out = base.extract_solution(ctx, cfg)
    assert out["bound"] == (bound if math.isfinite(bound) else None)
    assert out["objective"] is None and out["mip_gap"] is None


# ===========================================================================
# M15: completion_time_limit of a source
# ===========================================================================
@pytest.mark.parametrize("value", [None, -1, -0.5, math.inf, -math.inf, math.nan, True, False,
                                   "60", [1.0]])
def test_invalid_completion_time_limit_raises(value):
    with pytest.raises(ValueError, match="completion_time_limit"):
        base._check_completion_time_limit({"completion_time_limit": value})


@pytest.mark.parametrize("source", [{}, {"completion_time_limit": 0}, {"completion_time_limit": 0.0},
                                    {"completion_time_limit": 60}, {"completion_time_limit": 2.5},
                                    {"completion_time_limit": np.float64(3.0)}])
def test_valid_completion_time_limit_is_accepted(source):
    base._check_completion_time_limit(source)


@pytest.mark.parametrize("value", [None, -1, math.inf, math.nan, True])
def test_invalid_source_raises_before_any_model_is_touched(value):
    """``apply_binary_warm_start`` raises with the other source checks; ``ctx`` is not used."""
    source = _source(completion_time_limit=value)
    with pytest.raises(ValueError, match="completion_time_limit"):
        base.apply_binary_warm_start(None, source, "start")


def test_source_without_x_is_not_checked():
    """A source without ``x`` has no effect (spec 2130), so nothing in it is checked."""
    report = base.apply_binary_warm_start(None, {"completion_time_limit": None}, "start")
    assert report["applied"] is False


def test_solve_raises_for_invalid_completion_time_limit():
    _gurobi()
    data = _read(WARM)
    with pytest.raises(ValueError, match="completion_time_limit"):
        _solve(data, _source(completion_time_limit=-1))


# ===========================================================================
# M1: gurobi_params wins over the flags of the prepare hooks
# ===========================================================================
@pytest.mark.parametrize("entry, expected", [({"NonConvex": 1}, 1), ({}, 2), ({"NonConvex": 0}, 0)])
def test_gurobi_params_nonconvex_is_final(entry, expected):
    _gurobi()
    from fleet_management.config import load_config
    data = _read(GAUSSIAN)
    data["gurobi_params"] = dict(entry)
    if not entry:
        del data["gurobi_params"]
    cfg = load_config(data)
    ctx = base.build_fleet(cfg, base.resolve_run_options(cfg))
    try:
        assert int(ctx.model.Params.NonConvex) == expected
    finally:
        ctx.model.dispose()


def test_gurobi_params_still_win_over_the_run_options():
    """The first application stays: ``gurobi_params`` also beats ``mip_gap`` and ``time_limit``."""
    _gurobi()
    from fleet_management.config import load_config
    data = _read(GAUSSIAN)
    data["mip_gap"] = 0.2
    data["time_limit"] = 30
    data["gurobi_params"] = {"MIPGap": 0.01, "TimeLimit": 7, "NonConvex": 1}
    cfg = load_config(data)
    ctx = base.build_fleet(cfg, base.resolve_run_options(cfg))
    try:
        assert float(ctx.model.Params.MIPGap) == pytest.approx(0.01)
        assert float(ctx.model.Params.TimeLimit) == pytest.approx(7.0)
        assert int(ctx.model.Params.NonConvex) == 1
    finally:
        ctx.model.dispose()


# ===========================================================================
# M4: a Gurobi error in the completion or the relaxation start does not stop the solve
# ===========================================================================
def test_failing_completion_gives_status_error(monkeypatch):
    gp = _gurobi()
    copies = _failing_copies(monkeypatch, gp)
    result = _solve(_read(WARM), _source())
    completion = result["warm_start"]["completion"]
    assert completion["requested"] is True
    assert completion["status"] == "error"
    assert completion["iis_constraints"] == []
    assert completion["values_copied"] == 0
    assert len(copies) == 1 and copies[0].disposed
    assert "status" in result and result["warm_start"]["applied"] is True


def test_failing_relaxation_start_gives_status_error(monkeypatch):
    gp = _gurobi()
    copies = _failing_copies(monkeypatch, gp)
    data = _read(WARM)
    data["relaxation_warm_start"] = True
    result = _solve(data, None)
    relaxation = result["relaxation_warm_start"]
    assert relaxation["enabled"] is True
    assert relaxation["status"] == "error"
    assert relaxation["values_copied"] == 0
    assert relaxation["relaxation_objective"] is None
    assert len(copies) == 1 and copies[0].disposed
    assert "status" in result


def test_failing_iis_gives_status_error(monkeypatch):
    """``computeIIS`` of an infeasible completion may fail too: status ``error``, no rows."""
    gp = _gurobi()

    class IisProxy(_Proxy):
        def optimize(self, *args):
            return self._real.optimize(*args)

        def computeIIS(self):
            raise self._gp.GurobiError(10005, "test")

        def copy(self):
            clone = IisProxy(self._real.copy(), self._gp, failing=True, log=self._log)
            self._log.append(clone)
            return clone

    # x = 0 fixes every binary x to 0, so no train flies the mission: the completion is infeasible
    data = _read(WARM)
    copies = []
    original = base.build_fleet

    def build_fleet(cfg, opts, *args, **kwargs):
        ctx = original(cfg, opts, *args, **kwargs)
        ctx.model = IisProxy(ctx.model, gp, log=copies)
        return ctx

    monkeypatch.setattr(base, "build_fleet", build_fleet)
    result = _solve(data, _source())
    completion = result["warm_start"]["completion"]
    assert completion["status"] == "error"
    assert completion["iis_constraints"] == []
    assert len(copies) == 1 and copies[0].disposed
