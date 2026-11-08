"""Tests of the rounding of the binary decisions in the result (item M18 of step 3).

Gurobi returns a binary variable within its integrality tolerance (0.99999901 was seen);
the result holds exactly 0 or 1 for ``x``, ``m`` and ``r``, and ``idle`` and the cost
report are computed from the rounded values. The test of the helper needs no Gurobi;
the solve needs it and is skipped without it.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fleet_management.degradation_model import base

DATA = Path(__file__).resolve().parent / "data" / "io" / "valid"


def _vars(values):
    return {key: SimpleNamespace(X=value) for key, value in values.items()}


def test_read_binary_values_rounds_to_zero_or_one():
    got = base._read_binary_values(
        _vars({(0, 0): 0.99999901, (0, 1): 1e-7, (1, 0): 1.0000002, (1, 1): -3e-9}), (2, 3))
    assert got[0, 0] == 1.0 and got[0, 1] == 0.0 and got[1, 0] == 1.0 and got[1, 1] == 0.0
    assert np.isnan(got[0, 2]) and np.isnan(got[1, 2])      # an absent key stays NaN


def test_cost_report_from_rounded_arrays():
    x = base._read_binary_values(_vars({(0, 0, 0): 0.99999901, (0, 0, 1): 1e-7}), (1, 1, 2))
    r = base._read_binary_values(_vars({(0, 0, 0): 0.99999901, (0, 0, 1): 1e-7}), (1, 1, 2))
    rep = base.cost_report(x, r, np.zeros((1, 1, 2)), 2.0, [[3.0]], [[5.0]], 0.0, 0.0, 1, 1, None)
    assert rep["step_costs"][0] == 2.0 + 5.0 and rep["step_costs"][1] == 0.0


def test_solved_result_has_exact_binaries_and_idle():
    gp = pytest.importorskip("gurobipy")
    try:
        gp.Model("probe").dispose()
    except RuntimeError:
        pytest.importorskip("gurobipy_is_not_installed_here")
    from fleet_management.solver import solve
    res = solve(str(DATA / "gaussian.yaml"), None)
    x, m, r = res["x"], res["m"], res["r"]
    for a in (x, m, r):
        assert np.all((a == 0.0) | (a == 1.0))
    assert np.array_equal(res["idle"], x[:, 0, :][:, None, :] - m - r)
