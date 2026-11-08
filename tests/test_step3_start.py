"""Tests of the start of the repetition ``y`` (the loop rows return to ``y``, not to day ``H1``).

For every cell and tracked quantity ``s`` (and, at an ``ard1`` cell, every anchor ``g``) the
model has a variable ``y`` in ``[0, M_s]``, the start rows ``y >= s_{H1}``, ``y^g >= g_{H1}``, the
rows of day ``H1 + 1`` reading ``y`` instead of day ``H1`` (both encodings, the Wiener variance
row, the anchor keep rows, the repair-cost row) and the loop rows ``s_T <= y``, ``g_T <= y^g``.
The result carries ``y_mu, y_v, y_R, y_K, y_g_mu, y_g_v, y_g_R, y_g_K`` (``F x L``, ``NaN`` where
a cell does not track the quantity or anchor).

* (a) a hand case that the old rows refuse and the new rows accept (Gurobi);
* (b) the rows of the interface, by name and coefficient, on a small model (Gurobi);
* (c) the result keys and their ``NaN`` pattern (the writers and ``read_result``, no Gurobi;
  plus one free solve with Gurobi).

The Gurobi tests build models of at most about 100 variables and rows and solve in well under
5 s. They skip without a working ``gurobipy``.
"""

import math

import numpy as np
import pytest

from fleet_management import solver
from fleet_management.config import load_config

Y_KEYS = ("y_mu", "y_v", "y_R", "y_K", "y_g_mu", "y_g_v", "y_g_R", "y_g_K")


def _gurobi():
    """``gurobipy`` and ``base``; the test is skipped without a working gurobipy (a stub counts as none)."""
    gp = pytest.importorskip("gurobipy")
    try:
        gp.Model("probe").dispose()
    except RuntimeError:
        pytest.importorskip("gurobipy_is_not_installed_here")
    from fleet_management.degradation_model import base
    return gp, base


def _build(data):
    _, base = _gurobi()
    cfg = load_config(data)
    ctx = base.build_fleet(cfg, base.resolve_run_options(cfg))
    ctx.model.update()
    return cfg, ctx


# ===========================================================================
# (a) A schedule that replaces the component on day H1 and then rises
# ===========================================================================
# Inverse Gaussian cell (one mean, no variance), law ardinf, rho = 0.5, tau = 2, eta = 0.5,
# epsilon = 0.1: the cap is mu_bar = 2 * 0.3823093163 = 0.7646186326 on every day.
# F = 2, M = 1, H = [2, 2], T = 4; mission increment 0.2 on train 0 (train 1 adds nothing and
# keeps mu = 0.5), replacement value 0, mu_0 = 0.5, C_M = 1, C_R = 10, C_rep = 3.
# The flight plan is fixed (train 0 flies on days 1 and 3 and is in the depot on days 2 and 4;
# train 1 the other way round); the repairs and replacements are free. Write s_k for mu of train 0
# on day k and y for the start of the repetition.
#
#   day 1  fly:    s_1 = 0.5 + 0.2 = 0.7  (below the cap)
#   day 2  depot (day H1): nothing, repair (s_2 = 0.5 s_1, z_2 = 0.5 s_1) or replace (s_2 = 0)
#   day 3  fly:    s_3 = y + 0.2           (the rows of day H1 + 1 read y)
#   day 4  depot:  nothing, repair (s_4 = 0.5 s_3, z_4 = 0.5 s_3) or replace (s_4 = 0)
#   rows:  y >= s_2 and s_4 <= y; cap s_k <= 0.7646 on every day.
#
# The steady state of the operating phase "fly, repair" is the fixed point of
# y -> 0.5 (y + 0.2), which is y = 0.2: s_3 = 0.4, s_4 = 0.2, z_4 = 0.5 * 0.4 = 0.2.
#
# Schedules (cost without the depot days, which are fixed: 4 * C_M = 4):
#   nothing on day 2 : y >= s_2 = 0.7, s_3 >= 0.9 > cap                        infeasible
#   repair 2, repair 4:  s_2 = 0.35, y = 0.35, s_3 = 0.55, z = 10 (0.35 + 0.275)      6.25
#   repair 2, replace 4: 10 * 0.35 + 3                                                 6.5
#   replace 2, replace 4: 3 + 3                                                        6
#   replace 2, repair 4 : 3 + 10 * 0.2 (y = 0.2, s_4 = 0.2 <= y)                       5   <- optimum
# The optimum is replace 2, repair 4, with objective 5 + 4 = 9, y = 0.2 and
# mu = [>= 0.7, 0, 0.4, 0.2].
#
# The old rows (s_4 <= s_2 = 0) refuse it: s_4 = 0.2 > 0. Under the old rows the best schedule
# replaces on days 2 and 4: 6 + 4 = 10 (the other feasible one costs 6.25 + 4 = 10.25).
_HAND = {
    "F": 2, "M": 1, "H": [2, 2], "model": "inverse_gaussian", "repair_model": "ardinf",
    "tau": 2.0, "epsilon": 0.1, "rho": 0.5, "eta": 0.5, "mu_0": 0.5, "replacement_mu": 0.0,
    "mu": [[[[0.2, 0.2]]], [[[0.0, 0.0]]]],
    "C_M": 1.0, "C_R": 10.0, "C_rep": 3.0, "C_D": 0.0,
    "verbose": 0, "mip_gap": 0.0, "time_limit": 60.0,
}
_FLIGHTS = {(0, 1, 0), (0, 1, 2), (1, 1, 1), (1, 1, 3), (0, 0, 1), (0, 0, 3), (1, 0, 0), (1, 0, 2)}


def _solve_hand(encoding, old_loop_row):
    gp, base = _gurobi()
    cfg = load_config({**_HAND, "encoding": encoding})
    ctx = base.build_fleet(cfg, base.resolve_run_options(cfg))
    for key, var in ctx.x.items():
        var.LB = var.UB = 1.0 if key in _FLIGHTS else 0.0
    if old_loop_row:       # the old row s_T <= s_{H1} on top of the new rows
        ctx.model.addConstr(ctx.mu_var[0, 0, cfg.T - 1] <= ctx.mu_var[0, 0, cfg.H1 - 1],
                            name="old_loop_mu_0_0")
    ctx.model.optimize()
    return cfg, base.extract_solution(ctx, cfg)


@pytest.mark.parametrize("encoding", ["bigm", "indicator"])
def test_replacement_on_day_h1_then_rising_operating_phase(encoding):
    cfg, out = _solve_hand(encoding, old_loop_row=False)
    H1, T = cfg.H1, cfg.T
    assert out["status"] == "optimal"
    assert math.isclose(out["objective"], 9.0, abs_tol=1e-6)
    np.testing.assert_allclose(out["r"][0, 0], [0, 1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(out["m"][0, 0], [0, 0, 0, 1], atol=1e-6)
    mu = out["mu"][0, 0]
    assert mu[0] >= 0.7 - 1e-6 and abs(mu[1]) <= 1e-6
    np.testing.assert_allclose(mu[2:], [0.4, 0.2], atol=1e-6)
    # y is at least the fixed point of y -> 0.5 (y + 0.2) and above the state on day H1
    fixed_point = 0.2
    assert out["y_mu"][0, 0] >= fixed_point - 1e-6
    np.testing.assert_allclose(out["y_mu"][0, 0], fixed_point, atol=1e-6)
    assert out["y_mu"][0, 0] > mu[H1 - 1] + 0.1
    # the old loop row is violated by this schedule
    assert mu[T - 1] > mu[H1 - 1] + 0.1
    # a mean-only ardinf cell: every other key of the start is NaN
    for key in Y_KEYS[1:]:
        assert np.isnan(out[key]).all(), key


@pytest.mark.parametrize("encoding", ["bigm", "indicator"])
def test_old_loop_row_refuses_the_schedule_and_costs_more(encoding):
    _, out = _solve_hand(encoding, old_loop_row=True)
    assert out["status"] == "optimal"
    assert math.isclose(out["objective"], 10.0, abs_tol=1e-6)
    np.testing.assert_allclose(out["r"][0, 0], [0, 1, 0, 1], atol=1e-6)


# ===========================================================================
# (b) The rows of the interface, by name and coefficient
# ===========================================================================
def _data(model="gaussian", **changes):
    """Two trains, one mission, H = [2, 2]; the day H1 + 1 has index 2 (0-based)."""
    data = {
        "F": 2, "M": 1, "H": [2, 2], "model": model, "repair_model": "ard1",
        "reliability_impl": "tangent", "tau": 1.0, "epsilon": 0.1, "rho": 0.5,
        "mu_0": 0.2, "replacement_mu": 0.1,
        "mu": [[[[0.3, 0.3]]], [[[0.0, 0.0]]]],
        "C_M": 1.0, "C_R": 2.0, "C_rep": 3.0, "C_D": 0.0,
        "verbose": 0, "mip_gap": 0.0, "time_limit": 60.0,
    }
    if model == "gaussian":
        data["v"] = [[[[0.01, 0.01]]], [[[0.0, 0.0]]]]
    if model == "wiener":
        data["sigma"] = 0.1
    data.update(changes)
    return data


def _normalise(expr, sense, rhs):
    """``({variable name: coefficient}, sense, rhs)`` with the sense ``>`` or ``=``."""
    coefs = {}
    for n in range(expr.size()):
        name = expr.getVar(n).VarName
        coefs[name] = coefs.get(name, 0.0) + expr.getCoeff(n)
    rhs = rhs - expr.getConstant()
    if sense == "<":
        coefs, rhs, sense = {k: -v for k, v in coefs.items()}, -rhs, ">"
    return {k: v for k, v in coefs.items() if abs(v) > 1e-12}, sense, rhs


def _linear_rows(ctx):
    """name -> normalised linear row, for every linear row of the model."""
    rows = {}
    for constr in ctx.model.getConstrs():
        rows[constr.ConstrName] = _normalise(ctx.model.getRow(constr), constr.Sense, constr.RHS)
    return rows


def _indicator_rows(ctx):
    """name -> (selector name, normalised row), for every indicator constraint."""
    rows = {}
    for gc in ctx.model.getGenConstrs():
        binvar, binval, expr, sense, rhs = ctx.model.getGenConstrIndicator(gc)
        assert binval in (True, 1)
        rows[gc.GenConstrName] = (binvar.VarName, _normalise(expr, sense, rhs))
    return rows


def _assert_row(row, coefs, sense, rhs, tol=1e-9):
    got, got_sense, got_rhs = row
    assert set(got) == set(coefs), (sorted(got), sorted(coefs))
    for name, value in coefs.items():
        assert math.isclose(got[name], value, rel_tol=tol, abs_tol=tol), (name, got[name], value)
    assert got_sense == sense
    assert math.isclose(got_rhs, rhs, rel_tol=tol, abs_tol=tol), (got_rhs, rhs)


def _name(var):
    return var.VarName


def test_variables_and_registry_ard1():
    _, ctx = _build(_data("gaussian"))
    cells = {(0, 0), (1, 0)}
    assert set(ctx.start_vars["mu"]) == cells and set(ctx.start_vars["v"]) == cells
    assert set(ctx.start_anchor_vars["mu"]) == cells and set(ctx.start_anchor_vars["v"]) == cells
    assert not ctx.start_vars["R"] and not ctx.start_vars["K"]
    assert not ctx.start_anchor_vars["R"] and not ctx.start_anchor_vars["K"]
    for q in ("mu", "v"):
        upper = float(ctx.state_range[q][0, 0])
        for var in (ctx.start_vars[q][0, 0], ctx.start_anchor_vars[q][0, 0]):
            assert var.LB == 0.0 and math.isclose(var.UB, upper, rel_tol=1e-12)
            assert var.VType == "C"


def test_variables_ardinf_and_wiener_have_no_variance_anchor():
    _, ctx = _build(_data("gaussian", repair_model="ardinf"))
    assert set(ctx.start_vars["mu"]) == {(0, 0), (1, 0)} and set(ctx.start_vars["v"]) == {(0, 0), (1, 0)}
    assert not ctx.start_anchor_vars["mu"] and not ctx.start_anchor_vars["v"]
    _, ctx = _build(_data("wiener"))
    assert set(ctx.start_anchor_vars["mu"]) == {(0, 0), (1, 0)}
    assert set(ctx.start_vars["v"]) == {(0, 0), (1, 0)} and not ctx.start_anchor_vars["v"]


def test_rows_of_the_start_bigm_ard1():
    _, ctx = _build(_data("gaussian"))
    rows = _linear_rows(ctx)
    var = {"mu": ctx.mu_var, "v": ctx.state_vars["v"], "gmu": ctx.anchor_vars["mu"],
           "gv": ctx.anchor_vars["v"]}
    y = {"mu": ctx.start_vars["mu"][0, 0], "v": ctx.start_vars["v"][0, 0],
         "gmu": ctx.start_anchor_vars["mu"][0, 0], "gv": ctx.start_anchor_vars["v"][0, 0]}
    M_mu, M_v = float(ctx.state_range["mu"][0, 0]), float(ctx.state_range["v"][0, 0])
    xm, xr = ctx.m_rep[0, 0, 2], ctx.r_rep[0, 0, 2]
    x = ctx.x[0, 1, 2]

    # start rows: y >= value on day H1 (index 1)
    for key in ("mu", "v", "gmu", "gv"):
        _assert_row(rows[f"start_{key}_0_0"], {_name(y[key]): 1.0, _name(var[key][0, 0, 1]): -1.0}, ">", 0.0)
    # loop rows: value on day T (index 3) <= y
    for key in ("mu", "v", "gmu", "gv"):
        _assert_row(rows[f"loop_{key}_0_0"], {_name(y[key]): 1.0, _name(var[key][0, 0, 3]): -1.0}, ">", 0.0)

    # day H1 + 1 (index 2) of the mean: carry and repair read y
    _assert_row(rows["mu_carry_0_0_2"],
                {_name(var["mu"][0, 0, 2]): 1.0, _name(y["mu"]): -1.0, _name(x): -0.3,
                 _name(xm): M_mu, _name(xr): M_mu}, ">", 0.0)
    _assert_row(rows["mu_repair_0_0_2"],
                {_name(var["mu"][0, 0, 2]): 1.0, _name(y["mu"]): -0.5, _name(y["gmu"]): -0.5,
                 _name(xm): -M_mu}, ">", -M_mu)
    # ... and of the variance (repair factor (1 - rho)^2 = 0.25)
    _assert_row(rows["v_carry_0_0_2"],
                {_name(var["v"][0, 0, 2]): 1.0, _name(y["v"]): -1.0, _name(x): -0.01,
                 _name(xm): M_v, _name(xr): M_v}, ">", 0.0)
    _assert_row(rows["v_repair_0_0_2"],
                {_name(var["v"][0, 0, 2]): 1.0, _name(y["v"]): -0.25, _name(y["gv"]): -0.75,
                 _name(xm): -M_v}, ">", -M_v)
    # anchor keep rows read y^g
    for key, M in (("gmu", M_mu), ("gv", M_v)):
        _assert_row(rows[f"{key}_keep_lb_0_0_2"],
                    {_name(var[key][0, 0, 2]): -1.0, _name(y[key]): 1.0, _name(xm): M, _name(xr): M}, ">", 0.0)
        _assert_row(rows[f"{key}_keep_ub_0_0_2"],
                    {_name(var[key][0, 0, 2]): 1.0, _name(y[key]): -1.0, _name(xm): M, _name(xr): M}, ">", 0.0)
    # repair-cost row: z >= rho (y^mu - y^{g,mu}) - rho tau (1 - x^m), rho = 0.5, tau = 1
    _assert_row(rows["z_0_0_2"],
                {_name(ctx.z_var[0, 0, 2]): 1.0, _name(y["mu"]): -0.5, _name(y["gmu"]): 0.5,
                 _name(xm): -0.5}, ">", -0.5)

    # rows of the other days do not read y
    for name in ("mu_carry_0_0_3", "mu_repair_0_0_3", "mu_carry_0_0_1", "mu_repair_0_0_1", "z_0_0_3", "z_0_0_1",
                 "gmu_keep_lb_0_0_3", "v_carry_0_0_3"):
        assert not {_name(y[key]) for key in y} & set(rows[name][0]), name
    # the replacement and anchor-set rows of day H1 + 1 read no previous day
    for name in ("mu_repl_lb_0_0_2", "mu_repl_ub_0_0_2", "gmu_set_lb_0_0_2", "gmu_set_ub_0_0_2"):
        assert not {_name(y[key]) for key in y} & set(rows[name][0]), name

    # the rows that read each y are exactly the listed ones: no reliability row, no row of another day
    expected = {
        "mu": {"start_mu_0_0", "loop_mu_0_0", "mu_carry_0_0_2", "mu_repair_0_0_2", "z_0_0_2"},
        "gmu": {"start_gmu_0_0", "loop_gmu_0_0", "mu_repair_0_0_2", "gmu_keep_lb_0_0_2",
                "gmu_keep_ub_0_0_2", "z_0_0_2"},
        "v": {"start_v_0_0", "loop_v_0_0", "v_carry_0_0_2", "v_repair_0_0_2"},
        "gv": {"start_gv_0_0", "loop_gv_0_0", "v_repair_0_0_2", "gv_keep_lb_0_0_2", "gv_keep_ub_0_0_2"},
    }
    for key, names in expected.items():
        assert {n for n, row in rows.items() if _name(y[key]) in row[0]} == names, key


def test_rows_of_the_start_bigm_ardinf():
    _, ctx = _build(_data("gaussian", repair_model="ardinf"))
    rows = _linear_rows(ctx)
    y = ctx.start_vars["mu"][0, 0]
    M_mu = float(ctx.state_range["mu"][0, 0])
    xm = ctx.m_rep[0, 0, 2]
    # ardinf: the repair row has psi = (1 - rho) y, and y^{g,mu} = 0 in the repair-cost row
    _assert_row(rows["mu_repair_0_0_2"], {_name(ctx.mu_var[0, 0, 2]): 1.0, _name(y): -0.5, _name(xm): -M_mu},
                ">", -M_mu)
    _assert_row(rows["z_0_0_2"], {_name(ctx.z_var[0, 0, 2]): 1.0, _name(y): -0.5, _name(xm): -0.5}, ">", -0.5)
    _assert_row(rows["start_mu_0_0"], {_name(y): 1.0, _name(ctx.mu_var[0, 0, 1]): -1.0}, ">", 0.0)
    _assert_row(rows["loop_mu_0_0"], {_name(y): 1.0, _name(ctx.mu_var[0, 0, 3]): -1.0}, ">", 0.0)
    assert not [name for name in rows if name.startswith(("start_g", "loop_g"))]


def test_rows_of_the_start_wiener_variance_row():
    _, ctx = _build(_data("wiener"))
    rows = _linear_rows(ctx)
    yv, ymu = ctx.start_vars["v"][0, 0], ctx.start_vars["mu"][0, 0]
    M_v = float(ctx.state_range["v"][0, 0])
    xr, x = ctx.r_rep[0, 0, 2], ctx.x[0, 1, 2]
    v = ctx.state_vars["v"]
    # Eq. w_vrow on day H1 + 1: one row, repair keeps the variance, only a replacement resets it
    _assert_row(rows["v_carry_0_0_2"],
                {_name(v[0, 0, 2]): 1.0, _name(yv): -1.0, _name(x): -0.1 ** 2, _name(xr): M_v}, ">", 0.0)
    assert "v_repair_0_0_2" not in rows
    _assert_row(rows["start_v_0_0"], {_name(yv): 1.0, _name(v[0, 0, 1]): -1.0}, ">", 0.0)
    _assert_row(rows["loop_v_0_0"], {_name(yv): 1.0, _name(v[0, 0, 3]): -1.0}, ">", 0.0)
    assert not [name for name in rows if name.startswith(("start_gv", "loop_gv"))]
    assert {n for n, row in rows.items() if _name(yv) in row[0]} == {
        "start_v_0_0", "loop_v_0_0", "v_carry_0_0_2"}
    # the mean (ard1) of the Wiener cell has its anchor start
    assert "start_gmu_0_0" in rows and _name(ymu) in rows["mu_carry_0_0_2"][0]


def test_rows_of_the_start_indicator_encoding():
    _, ctx = _build(_data("gaussian", encoding="indicator"))
    lin, gen = _linear_rows(ctx), _indicator_rows(ctx)
    var = {"mu": ctx.mu_var, "v": ctx.state_vars["v"], "gmu": ctx.anchor_vars["mu"],
           "gv": ctx.anchor_vars["v"]}
    y = {"mu": ctx.start_vars["mu"][0, 0], "v": ctx.start_vars["v"][0, 0],
         "gmu": ctx.start_anchor_vars["mu"][0, 0], "gv": ctx.start_anchor_vars["v"][0, 0]}
    n, xm, x = ctx.nb[0, 0, 2], ctx.m_rep[0, 0, 2], ctx.x[0, 1, 2]

    # the start, loop and repair-cost rows are linear rows in both encodings
    for key in ("mu", "v", "gmu", "gv"):
        _assert_row(lin[f"start_{key}_0_0"], {_name(y[key]): 1.0, _name(var[key][0, 0, 1]): -1.0}, ">", 0.0)
        _assert_row(lin[f"loop_{key}_0_0"], {_name(y[key]): 1.0, _name(var[key][0, 0, 3]): -1.0}, ">", 0.0)
    _assert_row(lin["z_0_0_2"],
                {_name(ctx.z_var[0, 0, 2]): 1.0, _name(y["mu"]): -0.5, _name(y["gmu"]): 0.5,
                 _name(xm): -0.5}, ">", -0.5)

    # indicator rows of day H1 + 1 (Eqs. a_ind_state, a_ind_anchor)
    sel, row = gen["mu_carry_0_0_2"]
    assert sel == _name(n)
    _assert_row(row, {_name(var["mu"][0, 0, 2]): 1.0, _name(y["mu"]): -1.0, _name(x): -0.3}, ">", 0.0)
    sel, row = gen["mu_repair_0_0_2"]
    assert sel == _name(xm)
    _assert_row(row, {_name(var["mu"][0, 0, 2]): 1.0, _name(y["mu"]): -0.5, _name(y["gmu"]): -0.5}, ">", 0.0)
    sel, row = gen["v_carry_0_0_2"]
    assert sel == _name(n)
    _assert_row(row, {_name(var["v"][0, 0, 2]): 1.0, _name(y["v"]): -1.0, _name(x): -0.01}, ">", 0.0)
    sel, row = gen["v_repair_0_0_2"]
    assert sel == _name(xm)
    _assert_row(row, {_name(var["v"][0, 0, 2]): 1.0, _name(y["v"]): -0.25, _name(y["gv"]): -0.75}, ">", 0.0)
    for key in ("gmu", "gv"):
        sel, row = gen[f"{key}_keep_0_0_2"]
        assert sel == _name(n)
        _assert_row(row, {_name(var[key][0, 0, 2]): 1.0, _name(y[key]): -1.0}, "=", 0.0)

    # the indicator constraints that read y are exactly those of day H1 + 1
    expected = {
        "mu": {"mu_carry_0_0_2", "mu_repair_0_0_2"},
        "gmu": {"mu_repair_0_0_2", "gmu_keep_0_0_2"},
        "v": {"v_carry_0_0_2", "v_repair_0_0_2"},
        "gv": {"v_repair_0_0_2", "gv_keep_0_0_2"},
    }
    for key, names in expected.items():
        assert {nm for nm, (_, row) in gen.items() if _name(y[key]) in row[0]} == names, key
    # no other linear row reads y: only start, loop and the repair-cost row
    for key in ("mu", "gmu"):
        assert {nm for nm, row in lin.items() if _name(y[key]) in row[0]} == {
            f"start_{key}_0_0", f"loop_{key}_0_0", "z_0_0_2"}
    for key in ("v", "gv"):
        assert {nm for nm, row in lin.items() if _name(y[key]) in row[0]} == {
            f"start_{key}_0_0", f"loop_{key}_0_0"}


# ===========================================================================
# (c) The result keys and their NaN pattern
# ===========================================================================
def _hand_result():
    """A small result with the start of the repetition: two trains, components (Gaussian, rainflow-like)."""
    F, M, L, H1, H2, T = 2, 1, 2, 1, 2, 3
    nan = float("nan")
    y_mu = np.array([[0.5, 0.25], [0.125, 0.0625]])
    y_v = np.array([[0.01, nan], [0.02, nan]])
    all_nan = np.full((F, L), nan)
    result = dict.fromkeys(solver.RESULT_KEYS)
    result.update(
        status="optimal", objective=1.5, F=F, M=M, L=L, H1=H1, H2=H2, T=T, H=[H1, H2],
        mu=np.arange(F * L * T, dtype=float).reshape(F, L, T) / 10.0,
        y_mu=y_mu, y_v=y_v, y_R=all_nan.copy(), y_K=all_nan.copy(),
        y_g_mu=np.array([[0.1, nan], [0.2, nan]]), y_g_v=all_nan.copy(),
        y_g_R=all_nan.copy(), y_g_K=all_nan.copy(),
    )
    return result


def test_result_keys_and_hdf5_datasets_hold_the_start_keys():
    keys = list(solver.RESULT_KEYS)
    assert len(keys) == len(set(keys)) == 59
    for key in Y_KEYS:
        assert key in keys and key in solver._HDF5_DATASET_KEYS, key
    # in the table the start keys follow the anchors and come before u
    assert keys.index("g_K") + 1 == keys.index("y_mu") and keys.index("y_g_K") + 1 == keys.index("u")


def _check_start_arrays(back):
    expected = _hand_result()
    for key in Y_KEYS:
        assert isinstance(back[key], np.ndarray) and back[key].dtype == np.float64, key
        assert back[key].shape == (2, 2), key
        np.testing.assert_array_equal(np.isnan(back[key]), np.isnan(expected[key]), err_msg=key)
        np.testing.assert_allclose(back[key], expected[key], equal_nan=True, atol=1e-12, err_msg=key)


@pytest.mark.parametrize("name", ["r.yaml", "r.json", "r.h5"])
def test_start_keys_round_trip_through_the_writers(tmp_path, name):
    path = tmp_path / name
    solver._save_results(_hand_result(), path)
    _check_start_arrays(solver.read_result(path))


def test_start_keys_in_the_text_and_hdf5_files(tmp_path):
    import json

    import h5py
    result = _hand_result()
    solver._save_results(result, tmp_path / "r.json")
    with open(tmp_path / "r.json", encoding="utf-8") as handle:
        data = json.load(handle)
    assert data["y_mu"] == [[0.5, 0.25], [0.125, 0.0625]]
    assert data["y_v"] == [[0.01, None], [0.02, None]]
    assert data["y_R"] == [[None, None], [None, None]]
    solver._save_results(result, tmp_path / "r.h5")
    with h5py.File(tmp_path / "r.h5", "r") as handle:
        for key in Y_KEYS:
            assert key in handle and key not in handle.attrs, key
            assert handle[key].dtype == np.float64 and handle[key].shape == (2, 2), key
        assert np.isnan(handle["y_v"][0, 1]) and handle["y_mu"][1, 1] == 0.0625


def test_a_result_without_the_start_keys_still_reads(tmp_path):
    result = _hand_result()
    for key in Y_KEYS:
        result[key] = None
    solver._save_results(result, tmp_path / "r.yaml")
    back = solver.read_result(tmp_path / "r.yaml")
    assert all(back[key] is None for key in Y_KEYS)
    assert back["status"] == "optimal"


def test_free_solve_returns_a_consistent_start():
    """Gaussian ard1 fleet solved free: y covers day H1 and day T, NaN where not tracked."""
    gp, base = _gurobi()
    cfg = load_config(_data("gaussian"))
    out = base.solve_mixed(cfg)
    assert out["status"] == "optimal"
    H1, T = cfg.H1, cfg.T
    for key in ("y_mu", "y_v", "y_g_mu", "y_g_v"):
        assert out[key].shape == (cfg.F, cfg.L) and np.isfinite(out[key]).all(), key
    for key in ("y_R", "y_K", "y_g_R", "y_g_K"):
        assert np.isnan(out[key]).all(), key
    tol = 1e-6
    for state, anchor in (("mu", "y_mu"), ("v", "y_v"), ("g_mu", "y_g_mu"), ("g_v", "y_g_v")):
        y = out[anchor]
        assert np.all(out[state][:, :, H1 - 1] <= y + tol), state
        assert np.all(out[state][:, :, T - 1] <= y + tol), state
