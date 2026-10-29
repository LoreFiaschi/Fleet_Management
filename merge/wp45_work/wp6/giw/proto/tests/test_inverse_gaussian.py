"""Tests of the inverse Gaussian cell builder, ``degradation_model/inverse_gaussian.py``.

The tests need ``gurobipy`` and are skipped without it. Every model has at most
about 100 variables and rows, far below the size limit of the free Gurobi
licence. The cap ``mu_bar`` itself (``config.ig_mean_cap``) is tested in
``tests/test_config.py``, without Gurobi.

The reference case is worked by hand: two trains, one mission, one component,
horizon ``H1 = H2 = 2``, ``eta = 0.5``, ``epsilon = 0.1``, ``tau = 1``, so the
cap is ``mu_bar = 0.3823093163``. Train 0 flies, is repaired, flies and is
replaced. Train 1 flies on days 2 and 4 and adds no degradation.
"""

import numpy as np
import pytest

gp = pytest.importorskip("gurobipy")

from fleet_management.config import load_config  # noqa: E402
from fleet_management.degradation_model import base, inverse_gaussian  # noqa: E402

_INFEASIBLE = ("infeasible", "inf_or_unbounded")
_MU_BAR = 0.3823093163                         # tau / E[Z | Z > q], hand_giw.py


def _data(**changes) -> dict:
    """The input mapping of the hand-worked case; ``changes`` replace keys."""
    data = {
        "F": 2, "M": 1, "H": [2, 2], "model": "inverse_gaussian", "repair_model": "ard1",
        "tau": 1.0, "epsilon": 0.1, "rho": 0.5, "eta": 0.5,
        "mu_0": 0.1, "replacement_mu": 0.05,
        "mu": [[[[0.15, 0.15]]], [[[0.0, 0.0]]]],
        "C_M": 1.0, "C_R": 2.0, "C_rep": 3.0, "C_D": 0.0,
        "verbose": 0, "mip_gap": 0.0, "time_limit": 60.0,
    }
    data.update(changes)
    return data


# (train, activity, day index) of every binary equal to 1; activity 0 is the depot
_SCHEDULE = {
    "x": [(0, 1, 0), (0, 0, 1), (0, 1, 2), (0, 0, 3), (1, 1, 1), (1, 1, 3)],
    "m": [(0, 0, 1)],
    "r": [(0, 0, 3)],
}
# expected arrays of train 0 (hand_giw.py); train 1 keeps mu = 0.1
_TRAIN_0 = {
    "ard1": {"mu": [0.25, 0.175, 0.325, 0.05], "g_mu": [0.1, 0.175, 0.175, 0.05],
             "z": [0.0, 0.075, 0.0, 0.0]},
    "ardinf": {"mu": [0.25, 0.125, 0.275, 0.05], "z": [0.0, 0.125, 0.0, 0.0]},
}


def _build(data):
    cfg = load_config(data)
    ctx = base.build_fleet(cfg, base.resolve_run_options(cfg))
    ctx.model.update()
    return cfg, ctx


def _fix(ctx, schedule) -> None:
    """Fix every binary to the schedule by bounds; the selector ``n`` is ``1 - m - r``."""
    ones = {name: set(keys) for name, keys in schedule.items()}
    for variables, name in ((ctx.x, "x"), (ctx.m_rep, "m"), (ctx.r_rep, "r")):
        for key, variable in (variables or {}).items():
            variable.LB = variable.UB = float(key in ones[name])
    for key, variable in ctx.nb.items():
        variable.LB = variable.UB = 1.0 - float(key in ones["m"]) - float(key in ones["r"])


def _solve_fixed(data, schedule=_SCHEDULE):
    """Build, fix the schedule, minimise the states (the rows are then exact), extract."""
    cfg, ctx = _build(data)
    _fix(ctx, schedule)
    states = list(ctx.mu_var.values()) + list(ctx.z_var.values())
    ctx.model.setObjective(gp.quicksum(states), gp.GRB.MINIMIZE)
    ctx.model.optimize()
    return ctx, base.extract_solution(ctx, cfg)


def test_registered():
    assert isinstance(base.get_cell_builder("inverse_gaussian"),
                      inverse_gaussian.InverseGaussianCellBuilder)
    assert base.get_cell_builder("inverse_gaussian").name == "inverse_gaussian"


@pytest.mark.parametrize("encoding", ["bigm", "indicator"])
@pytest.mark.parametrize("law", ["ard1", "ardinf"])
def test_dynamics(encoding, law):
    ctx, out = _solve_fixed(_data(encoding=encoding, repair_model=law))
    assert out["status"] == "optimal"
    expected = _TRAIN_0[law]
    np.testing.assert_allclose(out["mu"][0, 0], expected["mu"], atol=1e-6)
    np.testing.assert_allclose(out["z"][0, 0], expected["z"], atol=1e-6)
    np.testing.assert_allclose(out["mu"][1, 0], [0.1] * 4, atol=1e-6)
    if law == "ard1":
        np.testing.assert_allclose(out["g_mu"][0, 0], expected["g_mu"], atol=1e-6)
    else:
        assert np.isnan(out["g_mu"]).all()
    for key in ("v", "g_v", "R", "K"):
        assert np.isnan(out[key]).all()


def test_cap_rows():
    # reliability_impl is ignored at a cell with a linear row: no segment binaries
    cfg, ctx = _build(_data(reliability_impl="pwl"))
    rows = {row.ConstrName: row for row in ctx.model.getConstrs()}
    for i in range(2):
        for k in range(4):
            row = rows[f"rel_ig_{i}_0_{k}"]
            assert abs(row.RHS - _MU_BAR) < 1e-9 and row.Sense == "<"
    assert not any(name.startswith("rel_side") for name in rows)
    assert not ctx.state_vars["v"] and not ctx.pwl_w
    assert ctx.model.NumQConstrs == 0
    assert getattr(ctx.model.Params, "NonConvex", None) != 2


# H = [1, 1]: train 0 flies on day 1 and is idle on day 2, so mu = 0.1 + mu_inc on
# both days; train 1 is idle, then flies with zero increment
@pytest.mark.parametrize("scale, feasible", [(0.99, True), (1.01, False)])
def test_cap_boundary(scale, feasible):
    increment = scale * _MU_BAR - 0.1
    data = _data(H=[1, 1], mu=[[[[increment]]], [[[0.0]]]])
    schedule = {"x": [(0, 1, 0), (1, 1, 1)], "m": [], "r": []}
    ctx, out = _solve_fixed(data, schedule)
    assert (out["status"] == "optimal") == feasible
    if not feasible:
        assert out["status"] in _INFEASIBLE


def test_free_schedule():
    cfg = load_config(_data(mu=0.15))
    out = base.solve_mixed(cfg)
    assert out["status"] == "optimal"
    assert (out["mu"] <= _MU_BAR + 1e-6).all()
    for key in ("mu", "g_mu"):
        assert (out[key][:, :, cfg.T - 1] <= out[key][:, :, cfg.H1 - 1] + 1e-6).all()
