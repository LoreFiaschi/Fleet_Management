"""Tests for fleet_management.models.rainflow.

validate_inputs tests (including the Bernstein-only Bhatia-Davis checks and
the tau > 2*kappa*b/3 precondition) are pure numpy (no Gurobi). The dynamics
tests build and solve a real (tiny) Gurobi model and check the *domination*
invariant, same reasoning as Gaussian/IG/Wiener/Gamma. Requires a Gurobi
license for the dynamics/reliability/replacement tests.
"""

import math

import numpy as np
import pytest

from fleet_management import solver
from fleet_management.models import rainflow


# ======================================================================
# _support_constant -- pure
# ======================================================================

def test_support_constant_takes_max_of_all_three():
    b_inc = np.array([0.1, 0.2, 0.05])
    assert rainflow._support_constant(0.15, 0.05, b_inc) == pytest.approx(0.2)
    assert rainflow._support_constant(0.5, 0.05, b_inc) == pytest.approx(0.5)
    assert rainflow._support_constant(0.05, 0.4, b_inc) == pytest.approx(0.4)


# ======================================================================
# validate_inputs -- pure numpy
# ======================================================================

def _base_inputs(F=1, L=1, tail_bound="cantelli"):
    mask = np.ones((F, L), dtype=bool)
    mu_0 = np.full((F, L), 0.1)
    v_0 = np.full((F, L), 0.0004)
    mu_new = np.full((F, L), 0.0)
    v_new = np.full((F, L), 0.0)
    mu_inc = np.full((F, 1, L, 3), 0.08)
    v_inc = np.full((F, 1, L, 3), 0.0004)
    tau = np.ones((F, L))
    rho = np.full((F, L), 0.7)
    mt = [["ARD1"] * L for _ in range(F)]
    tb = np.full((F, L), tail_bound, dtype=object)
    b_inc = np.full((F, 1, L, 3), 0.3)
    b_0 = np.full((F, L), 0.3)
    b_new = np.full((F, L), 0.0)
    return dict(
        mask=mask, mu_0=mu_0, v_0=v_0, mu_new=mu_new, v_new=v_new,
        mu_inc=mu_inc, v_inc=v_inc, tau=tau, rho=rho, maintenance_type=mt,
        tail_bound=tb, epsilon=0.01, b_inc=b_inc, b_0=b_0, b_new=b_new,
    )


def test_validate_inputs_happy_path_cantelli():
    kw = _base_inputs(tail_bound="cantelli")
    rainflow.validate_inputs(**kw)


def test_validate_inputs_happy_path_bernstein():
    kw = _base_inputs(tail_bound="bernstein")
    rainflow.validate_inputs(**kw)


def test_validate_inputs_rejects_ara1():
    kw = _base_inputs()
    kw["maintenance_type"][0][0] = "ARA1"
    with pytest.raises(ValueError):
        rainflow.validate_inputs(**kw)


def test_validate_inputs_rejects_bad_tail_bound():
    kw = _base_inputs()
    kw["tail_bound"][0, 0] = "bogus"
    with pytest.raises(ValueError):
        rainflow.validate_inputs(**kw)


def test_validate_inputs_bernstein_rejects_b_inc_below_mu_inc():
    kw = _base_inputs(tail_bound="bernstein")
    kw["b_inc"][0, :, 0, :] = 0.01  # below mu_inc=0.08
    with pytest.raises(ValueError, match="b_inc must be >= mu_inc"):
        rainflow.validate_inputs(**kw)


def test_validate_inputs_bernstein_rejects_bhatia_davis_v_inc():
    kw = _base_inputs(tail_bound="bernstein")
    kw["v_inc"][0, :, 0, :] = 10.0  # way above mu_inc*(b_inc-mu_inc)
    with pytest.raises(ValueError, match="Bhatia-Davis"):
        rainflow.validate_inputs(**kw)


def test_validate_inputs_bernstein_rejects_b0_below_mu0():
    kw = _base_inputs(tail_bound="bernstein")
    kw["b_0"][0, 0] = 0.01
    with pytest.raises(ValueError, match="b_0 must be >= mu_0"):
        rainflow.validate_inputs(**kw)


def test_validate_inputs_bernstein_rejects_bhatia_davis_v0():
    kw = _base_inputs(tail_bound="bernstein")
    kw["v_0"][0, 0] = 10.0
    with pytest.raises(ValueError, match="Bhatia-Davis"):
        rainflow.validate_inputs(**kw)


def test_validate_inputs_bernstein_tau_precondition():
    kw = _base_inputs(tail_bound="bernstein")
    kw["b_0"][0, 0] = 10.0
    kw["b_inc"][0, :, 0, :] = 10.0
    with pytest.raises(ValueError, match=r"tau must be > 2\*kappa\*b/3"):
        rainflow.validate_inputs(**kw)


def test_validate_inputs_cantelli_ignores_missing_b_fields():
    """Cantelli doesn't need b_inc/b_0/b_new at all."""
    kw = _base_inputs(tail_bound="cantelli")
    kw["b_inc"] = None
    kw["b_0"] = None
    kw["b_new"] = None
    rainflow.validate_inputs(**kw)


# ======================================================================
# Dynamics correctness via a real (tiny) Gurobi solve
# ======================================================================

def _recompute_ard1(x_m, x_r, delta, mu_0_il, mu_new_il, rho_il, two_h, sq=False):
    out = np.zeros(two_h)
    prev = mu_0_il
    for k in range(two_h):
        if x_r[k] > 0.5:
            prev = mu_new_il
        elif x_m[k] > 0.5:
            prev = ((1 - rho_il) ** 2 if sq else (1 - rho_il)) * prev
        else:
            prev = prev + delta[k]
        out[k] = prev
    return out


@pytest.fixture
def solved_cantelli(make_tiny_problem):
    data = make_tiny_problem(
        [["rainflow", "rainflow"], ["rainflow", "rainflow"], ["rainflow", "rainflow"]],
        F=3, M=1, L=2, H=3, mu_inc=0.09, v_inc=0.0004, rho=0.6, tail_bound="cantelli",
    )
    shared = solver._parse_and_validate(data)
    result = solver._solve_single_horizon(shared["H"], shared)
    assert result["status"] == "optimal"
    return result, shared


@pytest.fixture
def solved_bernstein(make_tiny_problem):
    # F=3 (not 2): the Bernstein cap (mu <= tau - c, tighter than Cantelli's
    # plain tau) forces more frequent maintenance on the rainflow component;
    # with F=2/M=1 there is no slack left for the other train to also
    # satisfy its own reliability/loop constraints while covering every
    # mission the rainflow train can't take -- genuine schedule tightness,
    # not a solver bug (confirmed via IIS before widening the fleet).
    data = make_tiny_problem(
        [["rainflow"], ["gaussian"], ["gaussian"]], F=3, M=1, L=1, H=3,
        mu_inc=0.09, v_inc=0.0004, rho=0.6, tail_bound="bernstein",
        b_inc=0.3, b_0=0.3, b_new=0.0,
    )
    shared = solver._parse_and_validate(data)
    result = solver._solve_single_horizon(shared["H"], shared)
    assert result["status"] == "optimal"
    return result, shared


def test_rainflow_cantelli_mean_and_variance_dominate_recursion(solved_cantelli):
    result, shared = solved_cantelli
    F, L, H, M = shared["F"], shared["L"], shared["H"], shared["M"]
    two_h = 2 * H
    mu_inc_2h = solver._wrap_increment(shared["mu_inc_raw"], F, M, L, H)
    v_inc_2h = solver._wrap_increment(shared["v_inc_raw"], F, M, L, H)
    for i in range(F):
        for l in range(L):
            delta_mu = np.array([
                sum(result["x"][i, j, k] * mu_inc_2h[i, j - 1, l, k] for j in range(1, M + 1))
                for k in range(two_h)
            ])
            delta_v = np.array([
                sum(result["x"][i, j, k] * v_inc_2h[i, j - 1, l, k] for j in range(1, M + 1))
                for k in range(two_h)
            ])
            expected_mu = _recompute_ard1(
                result["x_m"][i, l], result["x_r"][i, l], delta_mu,
                shared["mu_0"][i, l], shared["mu_new"][i, l], shared["rho"][i, l], two_h,
            )
            expected_v = _recompute_ard1(
                result["x_m"][i, l], result["x_r"][i, l], delta_v,
                shared["v_0"][i, l], shared["v_new"][i, l], shared["rho"][i, l], two_h, sq=True,
            )
            assert np.all(result["mu"][i, l] >= expected_mu - 1e-6), (i, l)
            assert np.all(result["v"][i, l] >= expected_v - 1e-6), (i, l)


def test_rainflow_cantelli_reliability_and_loop_constraint(solved_cantelli):
    result, shared = solved_cantelli
    epsilon = shared["epsilon"]
    kappa_c = math.sqrt((1 - epsilon) / epsilon)
    for i in range(shared["F"]):
        for l in range(shared["L"]):
            tau_il = shared["tau"][i, l]
            H = shared["H"]
            for k in range(2 * H):
                mu, v = result["mu"][i, l, k], max(result["v"][i, l, k], 0.0)
                assert mu + kappa_c * np.sqrt(v) <= tau_il + 1e-4
            assert result["mu"][i, l, 2 * H - 1] <= result["mu"][i, l, H - 1] + 1e-6
            assert result["v"][i, l, 2 * H - 1] <= result["v"][i, l, H - 1] + 1e-6


def test_rainflow_bernstein_reliability_holds(solved_bernstein):
    result, shared = solved_bernstein
    epsilon = shared["epsilon"]
    kappa = math.log(1.0 / epsilon)
    i, l = 0, 0
    tau_il = shared["tau"][i, l]
    b_il = rainflow._support_constant(
        shared["b_0"][i, l], shared["b_new"][i, l],
        solver._wrap_increment(shared["b_inc_raw"], shared["F"], shared["M"], shared["L"], shared["H"])[i, :, l, :],
    )
    c_il = kappa * b_il / 3
    for k in range(2 * shared["H"]):
        mu, v = result["mu"][i, l, k], max(result["v"][i, l, k], 0.0)
        assert mu + c_il + math.sqrt(c_il ** 2 + 2 * kappa * v) <= tau_il + 1e-4


def test_rainflow_v_max_rf_matches_derivation_cantelli():
    """At mu=0 the exact Cantelli constraint admits exactly V_max^RF =
    tau^2*eps/(1-eps) -- spec's own derivation, cross-checked directly."""
    tau, epsilon = 1.0, 0.01
    kappa_c = math.sqrt((1 - epsilon) / epsilon)
    v_max_rf = tau ** 2 * epsilon / (1 - epsilon)
    assert kappa_c ** 2 * v_max_rf == pytest.approx(tau ** 2)


def test_rainflow_v_max_rf_matches_derivation_bernstein():
    """At mu=0 the exact Bernstein constraint admits exactly
    V_max^RF = tau*(tau-2*kappa*b/3)/(2*kappa)."""
    tau, epsilon, b = 1.0, 0.01, 0.1
    kappa = math.log(1.0 / epsilon)
    c = kappa * b / 3
    v_max_rf = tau * (tau - 2 * c) / (2 * kappa)
    assert c ** 2 + 2 * kappa * v_max_rf == pytest.approx((tau - c) ** 2)


def test_rainflow_lp_formulation_solves_both_tail_bounds(make_tiny_problem):
    for tb in ("cantelli", "bernstein"):
        data = make_tiny_problem(
            [["rainflow"], ["gaussian"], ["gaussian"]], F=3, M=1, L=1, H=2,
            formulation="lp", tail_bound=tb, b_inc=0.3, b_0=0.3, b_new=0.0,
        )
        shared = solver._parse_and_validate(data)
        result = solver._solve_single_horizon(shared["H"], shared)
        assert result["status"] == "optimal", tb


def test_rainflow_replacement_reset_is_enforced():
    """Directly force x_r=1 for one component at one step and check the
    replacement big-M rows pin mu/v to mu_new/v_new exactly at that step."""
    import gurobipy as gp
    from gurobipy import GRB
    from types import SimpleNamespace

    from fleet_management.models import base

    F, L, M, H = 2, 1, 1, 2
    two_h = 2 * H
    forced_k = 1

    model = gp.Model("test_rainflow_replacement_reset")
    model.Params.OutputFlag = 0
    model.Params.NonConvex = 2

    x = model.addVars(F, M + 1, two_h, vtype=GRB.BINARY, name="x")
    x_m = model.addVars(F, L, two_h, vtype=GRB.BINARY, name="x_m")
    x_r = model.addVars(F, L, two_h, vtype=GRB.BINARY, name="x_r")
    z = model.addVars(F, L, two_h, lb=0.0, name="z")
    u = model.addVar(lb=0.0, name="u")

    x_r[0, 0, forced_k].LB = x_r[0, 0, forced_k].UB = 1.0

    mu_new_val, v_new_val = 0.02, 0.0001
    ctx = SimpleNamespace(
        model=model, F=F, H=H, M=M, L=L, two_h=two_h,
        x=x, x_m=x_m, x_r=x_r, z=z, u=u, mu={}, v={},
        tau=np.ones((F, L)), rho=np.full((F, L), 0.7),
        mu_0=np.full((F, L), 0.1), v_0=np.full((F, L), 0.0004),
        mu_new=np.full((F, L), mu_new_val), v_new=np.full((F, L), v_new_val),
        mu_inc=np.full((F, M, L, two_h), 0.05), v_inc=np.full((F, M, L, two_h), 0.0002),
        v_max_user=None, tail_bound=np.full((F, L), "cantelli", dtype=object),
        b_inc=None, b_0=np.full((F, L), np.nan), b_new=np.full((F, L), np.nan),
        epsilon=0.01, formulation="exact",
    )

    rainflow.build_component(ctx, 0, 0)
    base.add_assignment_constraints(model, x, x_m, x_r, F, L, M, two_h)
    model.setObjective(gp.LinExpr(0.0), GRB.MINIMIZE)
    model.optimize()

    assert model.Status == GRB.OPTIMAL
    assert abs(ctx.mu[0, 0, forced_k].X - mu_new_val) < 1e-6
    assert abs(ctx.v[0, 0, forced_k].X - v_new_val) < 1e-6
