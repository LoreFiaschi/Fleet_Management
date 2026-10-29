"""Tests of the Gamma model ``degradation_model/gamma.py``: the sharp cap and the cell builder.

The cap functions need neither Gurobi nor the solver. The model tests build the
hand-worked case of ``test_base.py`` with Gamma cells; they need gurobipy.
"""

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import gammaincc

from fleet_management.config import load_config
from fleet_management.degradation_model import gamma

NAN = math.nan


# ---------------------------------------------------------------------------
# The sharp cap A_1 (no Gurobi)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("t, eps, expected", [
    (1.25, 0.10, 0.452900672495),
    (11.0, 0.05, 6.36399044578),
    (40.0, 0.01, 26.5638726679),
])
def test_one_rate_cap_reaches_the_level(t, eps, expected):
    a_hat = gamma.one_rate_cap(t, eps)
    assert math.isclose(a_hat, expected, rel_tol=1e-9)
    assert abs(gammaincc(a_hat, t) - eps) <= 1e-10
    assert gammaincc(0.9 * a_hat, t) < eps < gammaincc(1.1 * a_hat, t)


@pytest.mark.parametrize("t, eps", [
    (0.0, 0.05), (-1.0, 0.05), (math.inf, 0.05), (NAN, 0.05),
    (11.0, 0.0), (11.0, 1.0), (11.0, -0.1), (11.0, 1.1), (11.0, NAN),
])
def test_one_rate_cap_rejects_invalid_input(t, eps):
    with pytest.raises(ValueError, match="Gamma cap"):
        gamma.one_rate_cap(t, eps)


def test_sharp_cap_reference_values():
    # epsilon = 0.01; scoping document, "The sharp cap and the exact one-rate cap":
    # A_1 / A_hat = 0.142, 0.641, 0.961, 1, 1 at t = 0.01, 0.1, 0.3, 1, 10
    expected = {0.01: 3.528887629e-4, 0.1: 3.52888763e-3, 0.3: 1.058666289e-2,
                1.0: 4.365272936e-2, 10.0: 3.971715971}
    for t, value in expected.items():
        assert math.isclose(gamma.sharp_cap(t, 0.01), value, rel_tol=1e-8)


def test_sharp_cap_ratio_table():
    # A_1 / A_hat at t = 0.01, 0.1, 0.3, 1, 3, 10, 100, to 4 digits
    table = {0.001: [0.1433, 0.6470, 0.9648, 1.0, 1.0, 1.0, 1.0],
             0.01: [0.1419, 0.6415, 0.9606, 1.0, 1.0, 1.0, 1.0],
             0.1: [0.1274, 0.5847, 0.9094, 1.0, 1.0, 1.0, 1.0]}
    for eps, ratios in table.items():
        for t, ratio in zip((0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0), ratios):
            a_hat = gamma.one_rate_cap(t, eps)
            assert abs(gamma.sharp_cap(t, eps) / a_hat - ratio) <= 6e-5
            if t >= 1.0:
                assert gamma.sharp_cap(t, eps) == a_hat


def test_sharp_cap_below_the_threshold_is_proportional():
    # A_1(t, eps) = r*(eps) t below t_1(eps); scoping: r* = 0.00355, 0.0353, 0.331
    r_star = {0.001: 0.00355012891, 0.01: 0.0352888763, 0.1: 0.33061707}
    for eps, r in r_star.items():
        for t in (0.01, 0.1, 0.3):
            assert math.isclose(gamma.sharp_cap(t, eps) / t, r, rel_tol=1e-7)


def test_sharp_cap_is_safe_and_sharp():
    for eps in (0.001, 0.01, 0.1):
        for t in (0.01, 0.1, 0.3):
            a_1 = gamma.sharp_cap(t, eps)
            assert gamma.worst_one_term_tail(a_1, t) <= eps
            assert gamma.worst_one_term_tail(a_1 * (1.0 + 1e-6), t) > eps


def test_worst_one_term_tail():
    # the exact one-rate cap is not safe below t_1: tail 1.565 eps at t = 0.1 (scoping example)
    a_hat = gamma.one_rate_cap(0.1, 0.01)
    assert math.isclose(gamma.worst_one_term_tail(a_hat, 0.1) / 0.01, 1.564868, rel_tol=1e-6)
    # above t_1 the worst case is c = 1, the single Gamma law
    a_hat = gamma.one_rate_cap(10.0, 0.01)
    assert math.isclose(gamma.worst_one_term_tail(a_hat, 10.0), gammaincc(a_hat, 10.0),
                        rel_tol=1e-12)


def test_sharp_cap_at_large_epsilon():
    # for eps >= 1/2 every A < t has w(A) < 1/2 <= eps; the bisection returns a value below t
    for eps in (0.5, 0.7):
        for t in (0.01, 1.0, 100.0):
            a_1 = gamma.sharp_cap(t, eps)
            assert t * (1.0 - 2e-9) <= a_1 < t


def test_smallest_rate():
    mu = [[0.3, 0.0], [0.1, 0.2]]                 # two missions, two days
    beta = [[20.0, 1.0], [30.0, 5.0]]
    assert gamma.smallest_rate(mu, beta, 0.0, NAN, 0.0, NAN, True) == 5.0   # rate 1 has mean 0
    assert gamma.smallest_rate(mu, beta, 0.2, 4.0, 0.0, NAN, True) == 4.0   # initial damage
    assert gamma.smallest_rate(mu, beta, 0.0, 4.0, 0.0, NAN, True) == 5.0   # mu_0 = 0: no term
    assert gamma.smallest_rate(mu, beta, 0.2, NAN, 0.0, NAN, True) == 5.0   # known initial damage
    assert gamma.smallest_rate(mu, beta, 0.0, NAN, 0.1, 3.0, True) == 3.0   # replacement draw
    assert gamma.smallest_rate(mu, beta, 0.0, NAN, 0.1, 3.0, False) == 5.0  # no replacement
    assert gamma.smallest_rate(mu, beta, 0.0, NAN, 0.0, 3.0, True) == 5.0   # mu_new = 0: no term
    assert gamma.smallest_rate([[0.0, 0.0]], [[2.0, 2.0]], 0.2, NAN, 0.1, NAN, True) is None


def test_mean_cap():
    assert gamma.mean_cap(None, 0.6, 0.1) == 0.6
    assert math.isclose(gamma.mean_cap(20.0, 1.0, 0.1), 0.744517324566, rel_tol=1e-9)
    assert math.isclose(gamma.mean_cap(2.0, 1.0, 0.1), 0.414912231273, rel_tol=1e-9)
    assert math.isclose(gamma.mean_cap(0.2, 1.0, 0.1), 0.330617069585, rel_tol=1e-8)
    assert math.isclose(gamma.mean_cap(4.0, 0.5, 0.1), 0.5 * gamma.mean_cap(2.0, 1.0, 0.1),
                        rel_tol=1e-15)


def test_cap_without_gurobi():
    # config.py and the Gamma validators import the cap functions without gurobipy
    src = Path(gamma.__file__).resolve().parents[2]
    code = "\n".join([
        "import sys",
        "sys.modules['gurobipy'] = None",
        "from fleet_management.degradation_model import gamma",
        "from fleet_management.config import load_config",
        "assert gamma.base is None",
        "assert abs(gamma.mean_cap(20.0, 1.0, 0.1) - 0.744517324566) < 1e-9",
        "load_config(dict(F=2, M=1, H=[1, 2], model='gamma', tau=1.0, epsilon=0.1, rho=0.5,",
        "                 mu_0=0.1, mu=0.05, gamma_beta=20.0, C_M=1.0, C_R=1.0, C_rep=1.0,",
        "                 C_D=1.0, replacement_mu=0.7))",
    ])
    path = os.pathsep.join([str(src), os.environ.get("PYTHONPATH", "")])
    env = dict(os.environ, PYTHONPATH=path)
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


# ---------------------------------------------------------------------------
# Model (needs gurobipy): the hand-worked case of test_base.py with Gamma cells
# ---------------------------------------------------------------------------
_CAP_20 = 0.744517324566        # A_1(20, 0.1) / 20, tau = 1
_CAP_2 = 0.414912231273         # A_1(2, 0.1) / 2
_CAP_5 = 0.553864057892         # A_1(5, 0.1) / 5
_SCHEDULE_X = ((0, 1, 0), (0, 0, 1), (0, 1, 2), (0, 0, 3), (1, 1, 1), (1, 1, 3))
_EXPECTED = {   # train 0: flies, repaired, flies, replaced
    "ard1": {"mu": [0.5, 0.35, 0.65, 0.1], "g_mu": [0.2, 0.35, 0.35, 0.1],
             "z": [0.0, 0.15, 0.0, 0.0]},
    "ardinf": {"mu": [0.5, 0.25, 0.55, 0.1], "g_mu": [NAN] * 4, "z": [0.0, 0.25, 0.0, 0.0]},
}
_NO_SOLUTION = ("infeasible", "inf_or_unbounded")


def _input(**changes):
    """Two trains, one mission, one Gamma component, H = [2, 2]; train 1 flies with mean 0."""
    data = {"F": 2, "M": 1, "H": [2, 2], "model": "gamma", "repair_model": "ard1",
            "encoding": "bigm", "tau": 1.0, "epsilon": 0.1, "rho": 0.5, "mu_0": 0.2,
            "replacement_mu": 0.1, "mu": [[[0.3]], [[0.0]]], "gamma_beta": 20.0,
            "C_M": 1.0, "C_R": 2.0, "C_rep": 3.0, "C_D": 0.0,
            "verbose": 0, "mip_gap": 0.0, "time_limit": 60,
            "gurobi_params": {"DualReductions": 0}}
    data.update(changes)
    return data


def _base():
    pytest.importorskip("gurobipy")
    from fleet_management.degradation_model import base
    return base


def _solve_fixed(base, cfg):
    """Build the model, fix the schedule, minimise mu + g_mu + z and optimise."""
    import gurobipy as gp
    ctx = base.build_fleet(cfg, base.resolve_run_options(cfg))
    for key, var in ctx.x.items():
        var.LB = var.UB = 1.0 if key in _SCHEDULE_X else 0.0
    for key, var in ctx.m_rep.items():
        var.LB = var.UB = 1.0 if key == (0, 0, 1) else 0.0
    for key, var in ctx.r_rep.items():
        var.LB = var.UB = 1.0 if key == (0, 0, 3) else 0.0
    states = (list(ctx.mu_var.values()) + list(ctx.anchor_vars["mu"].values())
              + list(ctx.z_var.values()))
    ctx.model.setObjective(gp.quicksum(states), gp.GRB.MINIMIZE)
    ctx.model.optimize()
    return ctx


def test_builder_is_registered():
    base = _base()
    assert base.get_cell_builder("gamma").name == "gamma"


@pytest.mark.parametrize("encoding", ["bigm", "indicator"])
@pytest.mark.parametrize("repair_model", ["ard1", "ardinf"])
def test_dynamics_and_cap(repair_model, encoding):
    base = _base()
    cfg = load_config(_input(repair_model=repair_model, encoding=encoding))
    ctx = _solve_fixed(base, cfg)
    out = base.extract_solution(ctx, cfg)
    assert out["status"] == "optimal"
    for key, values in _EXPECTED[repair_model].items():
        np.testing.assert_allclose(out[key][0, 0], values, atol=1e-6)
    np.testing.assert_allclose(out["mu"][1, 0], [0.2] * 4, atol=1e-6)
    np.testing.assert_allclose(out["z"][1, 0], [0.0] * 4, atol=1e-6)
    assert ctx.extras["gamma"]["beta_star"] == {(0, 0): 20.0, (1, 0): None}
    rows = {row.ConstrName: row for row in ctx.model.getConstrs()}
    for k in range(ctx.T):
        assert math.isclose(rows[f"rel_0_0_{k}"].RHS, _CAP_20, rel_tol=1e-9)
        assert rows[f"rel_1_0_{k}"].RHS == 1.0          # no term with a rate: mu <= tau
    assert ("loop_gmu_0_0" in rows) == (repair_model == "ard1")
    assert ctx.model.NumQConstrs == 0 and ctx.model.Params.NonConvex == -1


@pytest.mark.parametrize("repair_model", ["ard1", "ardinf"])
def test_cap_binds(repair_model):
    # gamma_beta_new = 2 gives beta* = 2 and the cap 0.4149 < mu = 0.5 on day 1
    base = _base()
    ctx = _solve_fixed(base, load_config(_input(repair_model=repair_model, gamma_beta_new=2.0)))
    assert ctx.extras["gamma"]["beta_star"] == {(0, 0): 2.0, (1, 0): 2.0}
    assert math.isclose(ctx.extras["gamma"]["cap"][0, 0], _CAP_2, rel_tol=1e-9)
    assert base.status_string(ctx.model.Status) in _NO_SOLUTION


def test_cap_depends_on_the_repair_law():
    # gamma_beta_0 = 5 gives the cap 0.5539: the ard1 peak 0.65 breaks it, the ardinf peak 0.55 not
    base = _base()
    ctx = _solve_fixed(base, load_config(_input(repair_model="ard1", gamma_beta_0=5.0)))
    assert ctx.extras["gamma"]["beta_star"] == {(0, 0): 5.0, (1, 0): 5.0}
    assert base.status_string(ctx.model.Status) in _NO_SOLUTION
    ctx = _solve_fixed(base, load_config(_input(repair_model="ardinf", gamma_beta_0=5.0)))
    assert math.isclose(ctx.extras["gamma"]["cap"][0, 0], _CAP_5, rel_tol=1e-9)
    assert base.status_string(ctx.model.Status) == "optimal"


@pytest.mark.parametrize("changes, expected", [
    ({"gamma_beta_new": 2.0, "allow_replacement": False}, (20.0, None)),
    ({"gamma_beta_new": 2.0, "replacement_mu": 0.0}, (20.0, None)),
    ({"gamma_beta_trans": 2.0}, (2.0, None)),
    ({"gamma_beta_trans": 2.0, "mu_trans": 0.0}, (20.0, None)),
    ({"mu_trans": [[[0.0]], [[0.1]]]}, (20.0, 20.0)),
    ({"gamma_beta_0": 5.0, "mu_0": 0.0}, (20.0, None)),
    ({"gamma_beta": [[20.0], [3.0]], "mu": [[[0.3]], [[0.1]]]}, (20.0, 3.0)),
])
def test_smallest_rate_of_each_cell(changes, expected):
    base = _base()
    cfg = load_config(_input(**changes))
    ctx = base.build_context(cfg, base.resolve_run_options(cfg))
    gamma.GammaCellBuilder().prepare(ctx, cfg, ctx.cells_of("gamma"), ctx.opts)
    stars = ctx.extras["gamma"]["beta_star"]
    assert (stars[0, 0], stars[1, 0]) == expected
