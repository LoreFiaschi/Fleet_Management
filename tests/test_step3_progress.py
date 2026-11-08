"""Test of M5b: the records of ``optimization_progress`` report a value that is not finite as ``None``.

``incumbent``, ``best_bound`` and ``relative_gap`` are ``None`` when Gurobi gives ``inf``,
``-inf`` or ``NaN``. The functions are called with fake models; no Gurobi model is built.
"""
import math

import pytest

base = pytest.importorskip("fleet_management.degradation_model.base")
GRB = base.GRB


class _FakeFinal:
    def __init__(self, obj, bound, gap, solutions=1):
        self.ObjVal, self.ObjBound, self.MIPGap = obj, bound, gap
        self.SolCount, self.Runtime, self.NodeCount = solutions, 1.5, 10
        self._optimization_progress = []


class _FakeCallback:
    def __init__(self, incumbent, bound):
        self.values = {
            GRB.Callback.RUNTIME: 2.0,
            GRB.Callback.MIP_OBJBST: incumbent,
            GRB.Callback.MIP_OBJBND: bound,
            GRB.Callback.MIP_NODCNT: 5.0,
            GRB.Callback.MIP_SOLCNT: 1,
        }
        self._optimization_progress = []
        self._optimization_progress_last_time = -math.inf
        self._optimization_progress_last_solutions = -1

    def cbGet(self, what):
        return self.values[what]


def _record_final(obj, bound, gap):
    model = _FakeFinal(obj, bound, gap)
    base._append_final_progress(model)
    return model._optimization_progress[-1]


def _record_callback(incumbent, bound):
    model = _FakeCallback(incumbent, bound)
    base._make_progress_callback(1.0)(model, GRB.Callback.MIP)
    return model._optimization_progress[-1]


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_final_record_not_finite_is_none(bad):
    record = _record_final(bad, bad, bad)
    assert record["incumbent"] is None
    assert record["best_bound"] is None
    assert record["relative_gap"] is None


def test_final_record_finite_is_kept():
    record = _record_final(10.0, 9.0, 0.1)
    assert (record["incumbent"], record["best_bound"], record["relative_gap"]) == (10.0, 9.0, 0.1)


def test_final_record_each_entry_apart():
    record = _record_final(10.0, -math.inf, math.inf)
    assert record["incumbent"] == 10.0
    assert record["best_bound"] is None
    assert record["relative_gap"] is None


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_callback_record_not_finite_is_none(bad):
    record = _record_callback(bad, bad)
    assert record["incumbent"] is None
    assert record["best_bound"] is None
    assert record["relative_gap"] is None


def test_callback_record_nan_bound_only():
    record = _record_callback(10.0, math.nan)
    assert record["incumbent"] == 10.0
    assert record["best_bound"] is None
    assert record["relative_gap"] is None


def test_callback_record_finite_and_gap_formula():
    record = _record_callback(10.0, 8.0)
    assert record["incumbent"] == 10.0
    assert record["best_bound"] == 8.0
    assert record["relative_gap"] == pytest.approx(0.2)


def test_callback_record_gurobi_infinity_is_none():
    record = _record_callback(1e100, 8.0)
    assert record["incumbent"] is None
    assert record["relative_gap"] is None
