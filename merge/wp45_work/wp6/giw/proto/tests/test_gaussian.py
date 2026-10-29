"""Tests of the Gaussian cell builder, ``degradation_model/gaussian.py``.

The tests need ``gurobipy`` and are skipped without it. Every model has at most
about 150 variables and rows, far below the size limit of the free Gurobi
licence.

The reference case is worked by hand: two trains, one mission, one component,
horizon ``H1 = H2 = 2``. Train 0 flies, is repaired, flies and is replaced.
Train 1 flies on days 2 and 4 and adds no degradation. The input goes through
``load_config``, so the builder reads the fields that the input layer fills.
"""

import numpy as np
import pytest
from scipy.special import ndtri

gp = pytest.importorskip("gurobipy")

from fleet_management.config import load_config  # noqa: E402
from fleet_management.degradation_model import base, gaussian  # noqa: E402

_INFEASIBLE = ("infeasible", "inf_or_unbounded")


def _data(**changes) -> dict:
    """The input mapping of the hand-worked case; ``changes`` replace keys."""
    data = {
        "F": 2, "M": 1, "H": [2, 2], "model": "gaussian", "repair_model": "ard1",
        "reliability_impl": "tangent", "tau": 1.0, "epsilon": 0.1, "rho": 0.5,
        "mu_0": 0.2, "replacement_mu": 0.1,
        "mu": [[[[0.3, 0.3]]], [[[0.0, 0.0]]]],
        "v": [[[[0.01, 0.01]]], [[[0.0, 0.0]]]],
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
# expected arrays of train 0 (hand_giw.py); train 1 keeps mu = 0.2 and v = 0
_TRAIN_0 = {
    "ard1": {"mu": [0.5, 0.35, 0.65, 0.1], "v": [0.01, 0.0025, 0.0125, 0.0],
             "g_mu": [0.2, 0.35, 0.35, 0.1], "g_v": [0.0, 0.0025, 0.0025, 0.0],
             "z": [0.0, 0.15, 0.0, 0.0]},
    "ardinf": {"mu": [0.5, 0.25, 0.55, 0.1], "v": [0.01, 0.0025, 0.0125, 0.0],
               "z": [0.0, 0.25, 0.0, 0.0]},
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
    states = (list(ctx.mu_var.values()) + list(ctx.state_vars["v"].values())
              + list(ctx.z_var.values()))
    ctx.model.setObjective(gp.quicksum(states), gp.GRB.MINIMIZE)
    ctx.model.optimize()
    return ctx, base.extract_solution(ctx, cfg)


def _row_names(ctx) -> set:
    return {row.ConstrName for row in ctx.model.getConstrs()}


def test_registered():
    assert isinstance(base.get_cell_builder("gaussian"), gaussian.GaussianCellBuilder)
    assert base.get_cell_builder("gaussian").name == "gaussian"


@pytest.mark.parametrize("encoding", ["bigm", "indicator"])
@pytest.mark.parametrize("law", ["ard1", "ardinf"])
def test_dynamics(encoding, law):
    ctx, out = _solve_fixed(_data(encoding=encoding, repair_model=law))
    assert out["status"] == "optimal"
    expected = _TRAIN_0[law]
    for key in ("mu", "v", "z"):
        np.testing.assert_allclose(out[key][0, 0], expected[key], atol=1e-6)
    np.testing.assert_allclose(out["mu"][1, 0], [0.2] * 4, atol=1e-6)
    np.testing.assert_allclose(out["v"][1, 0], [0.0] * 4, atol=1e-6)
    if law == "ard1":
        np.testing.assert_allclose(out["g_mu"][0, 0], expected["g_mu"], atol=1e-6)
        np.testing.assert_allclose(out["g_v"][0, 0], expected["g_v"], atol=1e-6)
    else:
        assert np.isnan(out["g_mu"]).all() and np.isnan(out["g_v"]).all()
    assert np.isnan(out["R"]).all() and np.isnan(out["K"]).all()


def test_rows_and_variables():
    cfg, ctx = _build(_data())
    assert len(ctx.state_vars["v"]) == 8
    assert len(ctx.anchor_vars["mu"]) == 8 and len(ctx.anchor_vars["v"]) == 8
    assert not ctx.nb and not ctx.pwl_w
    names = _row_names(ctx)
    for k in range(4):
        for prefix in ("rel_side", "rel", "v_carry", "v_repair", "gv_set_lb", "z"):
            assert f"{prefix}_0_0_{k}" in names
    for name in ("loop_mu_0_0", "loop_v_0_0", "loop_gmu_0_0", "loop_gv_0_0"):
        assert name in names
    assert not any(name.startswith("v_max_user") for name in names)
    assert ctx.model.NumQConstrs == 0
    assert getattr(ctx.model.Params, "NonConvex", None) != 2


def test_exact_form_is_quadratic():
    cfg, ctx = _build(_data(reliability_impl="exact"))
    assert ctx.model.Params.NonConvex == 2
    names = {row.QCName for row in ctx.model.getQConstrs()}
    assert names == {f"rel_{i}_0_{k}" for i in range(2) for k in range(4)}


def test_pwl_form():
    cfg, ctx = _build(_data(reliability_impl="pwl", pwl_points=4))
    assert len(ctx.pwl_w) == 2 * 4 * 4
    assert {f"rel_seg_0_0_{k}" for k in range(4)} <= _row_names(ctx)
    assert ctx.model.NumQConstrs == 0
    assert getattr(ctx.model.Params, "NonConvex", None) != 2


@pytest.mark.parametrize("v_max_user, feasible", [(0.013, True), (0.012, False)])
def test_v_max_user(v_max_user, feasible):
    data = _data(v_max_user=[[v_max_user], [None]])
    cfg, ctx = _build(data)
    names = _row_names(ctx)
    assert {f"v_max_user_0_0_{k}" for k in range(4)} <= names
    assert not any(name.startswith("v_max_user_1_") for name in names)
    ctx, out = _solve_fixed(data)
    assert (out["status"] == "optimal") == feasible
    if not feasible:
        assert out["status"] in _INFEASIBLE


# epsilon = 0.001: at mu = 0.65, v = 0.0125 the exact cap is 0.0128278, the tangent at
# 0.5 tau gives 0.0104717 and the pwl segment [0.625, 0.75] of P = 8 gives 0.0126806
@pytest.mark.parametrize("impl, feasible", [("tangent", False), ("exact", True), ("pwl", True)])
def test_reliability_forms(impl, feasible):
    ctx, out = _solve_fixed(_data(epsilon=0.001, reliability_impl=impl))
    assert (out["status"] == "optimal") == feasible
    if not feasible:
        assert out["status"] in _INFEASIBLE


def test_free_schedule():
    data = _data(mu=0.3, v=0.01, reliability_impl="exact")
    cfg = load_config(data)
    out = base.solve_mixed(cfg)
    assert out["status"] == "optimal"
    mu, v, H1, T = out["mu"], out["v"], cfg.H1, cfg.T
    kappa2 = float(ndtri(0.9)) ** 2
    assert (kappa2 * v <= (1.0 - mu) ** 2 + 1e-6).all() and (mu <= 1.0 + 1e-6).all()
    for key in ("mu", "v", "g_mu", "g_v"):
        assert (out[key][:, :, T - 1] <= out[key][:, :, H1 - 1] + 1e-6).all()
