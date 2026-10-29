"""Tests of the inverse Gaussian model, ``degradation_model/inverse_gaussian.py``.

The cap ``mu_bar`` (``inverse_gaussian.ig_mean_cap``) needs neither Gurobi nor
the solver: ``test_inverse_gaussian_cap`` and ``test_cap_without_gurobi`` always
run. The builder tests need ``gurobipy`` and are skipped without it; every model
has at most about 100 variables and rows, far below the size limit of the free
Gurobi licence. The check of the replacement state that uses the cap is tested in
``tests/test_config.py``.

The reference case is worked by hand: two trains, one mission, one component,
horizon ``H1 = H2 = 2``, ``eta = 0.5``, ``epsilon = 0.1``, ``tau = 1``, so the
cap is ``mu_bar = 0.3823093163``. Train 0 flies, is repaired, flies and is
replaced. Train 1 flies on days 2 and 4 and adds no degradation.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import invgauss

from fleet_management.config import load_config
from fleet_management.degradation_model import inverse_gaussian

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


def _gurobi():
    """``gurobipy`` and ``base``; the calling test is skipped without gurobipy."""
    gp = pytest.importorskip("gurobipy")
    from fleet_management.degradation_model import base
    return gp, base


def _build(data):
    _, base = _gurobi()
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
    gp, base = _gurobi()
    cfg, ctx = _build(data)
    _fix(ctx, schedule)
    states = list(ctx.mu_var.values()) + list(ctx.z_var.values())
    ctx.model.setObjective(gp.quicksum(states), gp.GRB.MINIMIZE)
    ctx.model.optimize()
    return ctx, base.extract_solution(ctx, cfg)


# ---------------------------------------------------------------------------
# The cap mu_bar (no Gurobi)
# ---------------------------------------------------------------------------
def test_inverse_gaussian_cap():
    # (eta, tau, epsilon, q, mu_bar), computed by brentq and checked by quadrature
    for eta, tau, eps, q, mu_bar in ((0.5, 1.0, 0.1, 1.8859953361, 0.3823093163),
                                     (0.1, 1.0, 0.01, 1.9488253601, 0.4627586840),
                                     (2.0, 2.0, 0.05, 3.5525960458, 0.3469170694)):
        assert abs(inverse_gaussian._ig_unit_quantile(eta, eps) - q) <= 1e-9 * q
        assert abs(inverse_gaussian.ig_mean_cap(eta, tau, eps) - mu_bar) <= 1e-9 * mu_bar
        assert abs(invgauss.sf(q, eta, scale=1.0 / eta) - eps) <= 1e-8 * eps
    for eta in (1e-3, 1.0, 1e3):
        for eps in (1e-9, 0.5, 0.99):
            cap = inverse_gaussian.ig_mean_cap(eta, 1.0, eps)
            assert 0.0 < cap <= 1.0 / inverse_gaussian._ig_unit_quantile(eta, eps)


def test_cap_without_gurobi():
    # config.py imports the cap inside its check; both load without gurobipy
    src = Path(inverse_gaussian.__file__).resolve().parents[2]
    code = "\n".join([
        "import sys",
        "sys.modules['gurobipy'] = None",
        "from fleet_management.degradation_model import inverse_gaussian",
        "from fleet_management.config import load_config",
        "assert inverse_gaussian.base is None",
        "assert abs(inverse_gaussian.ig_mean_cap(0.5, 1.0, 0.1) - 0.3823093163) < 1e-9",
        "load_config(dict(F=2, M=1, H=[1, 2], model='inverse_gaussian', tau=1.0, epsilon=0.1,",
        "                 rho=0.5, eta=0.15, mu_0=0.1, mu=0.05, C_M=1.0, C_R=1.0, C_rep=1.0,",
        "                 C_D=1.0, replacement_mu=0.54))",
    ])
    path = os.pathsep.join([str(src), os.environ.get("PYTHONPATH", "")])
    env = dict(os.environ, PYTHONPATH=path)
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


# ---------------------------------------------------------------------------
# Builder (needs gurobipy)
# ---------------------------------------------------------------------------
def test_registered():
    _, base = _gurobi()
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
            row = rows[f"rel_{i}_0_{k}"]
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
    _, base = _gurobi()
    cfg = load_config(_data(mu=0.15))
    out = base.solve_mixed(cfg)
    assert out["status"] == "optimal"
    assert (out["mu"] <= _MU_BAR + 1e-6).all()
    for key in ("mu", "g_mu"):
        assert (out[key][:, :, cfg.T - 1] <= out[key][:, :, cfg.H1 - 1] + 1e-6).all()
