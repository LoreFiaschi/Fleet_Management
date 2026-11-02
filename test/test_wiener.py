"""Tests for fleet_management.models.wiener.

validate_inputs tests are pure numpy (no Gurobi). The dynamics tests build
and solve a real (tiny) Gurobi model via solver._solve_single_horizon and
check the solution against a hand-recomputed recursion -- using the
*domination* invariant (solved >= recomputed), not equality, per the same
slack-robustness reasoning established for Gaussian/IG in Milestone 1 (see
test_gaussian.py's test_gaussian_mean_dynamics_dominate_recursion docstring).
Requires a Gurobi license for the dynamics/reliability/replacement tests.
"""

import numpy as np
import pytest

from fleet_management import solver
from fleet_management.models import wiener


# ======================================================================
# validate_inputs (no Gurobi)
# ======================================================================

def _base_inputs(F=1, L=1):
    mask = np.ones((F, L), dtype=bool)
    mu_0 = np.full((F, L), 0.1)
    v_0 = np.full((F, L), 0.0004)
    mu_inc = np.full((F, 1, L, 3), 0.08)
    tau = np.ones((F, L))
    rho = np.full((F, L), 0.7)
    sigma = np.full((F, L), 0.03)
    mt = [["ARD1"] * L for _ in range(F)]
    return mask, mu_0, v_0, mu_inc, tau, rho, sigma, mt


def test_validate_inputs_happy_path_ard1():
    wiener.validate_inputs(*_base_inputs())


def test_validate_inputs_happy_path_ara1():
    mask, mu_0, v_0, mu_inc, tau, rho, sigma, mt = _base_inputs()
    mt[0][0] = "ARA1"
    wiener.validate_inputs(mask, mu_0, v_0, mu_inc, tau, rho, sigma, mt)


def test_validate_inputs_rejects_bad_maintenance_type():
    mask, mu_0, v_0, mu_inc, tau, rho, sigma, mt = _base_inputs()
    mt[0][0] = "bogus"
    with pytest.raises(ValueError):
        wiener.validate_inputs(mask, mu_0, v_0, mu_inc, tau, rho, sigma, mt)


def test_validate_inputs_rejects_nonpositive_sigma():
    mask, mu_0, v_0, mu_inc, tau, rho, sigma, mt = _base_inputs()
    sigma[0, 0] = 0.0
    with pytest.raises(ValueError):
        wiener.validate_inputs(mask, mu_0, v_0, mu_inc, tau, rho, sigma, mt)


def test_validate_inputs_rejects_nonpositive_v0():
    mask, mu_0, v_0, mu_inc, tau, rho, sigma, mt = _base_inputs()
    v_0[0, 0] = 0.0
    with pytest.raises(ValueError):
        wiener.validate_inputs(mask, mu_0, v_0, mu_inc, tau, rho, sigma, mt)


def test_validate_inputs_rejects_mu0_ge_tau():
    mask, mu_0, v_0, mu_inc, tau, rho, sigma, mt = _base_inputs()
    mu_0[0, 0] = 2.0
    with pytest.raises(ValueError):
        wiener.validate_inputs(mask, mu_0, v_0, mu_inc, tau, rho, sigma, mt)


# ======================================================================
# Dynamics correctness via a real (tiny) Gurobi solve
# ======================================================================

def _recompute_ard1_mu(x_m, x_r, mu_inc_2h, mu_0_il, mu_new_il, rho_il, two_h):
    mu = np.zeros(two_h)
    prev = mu_0_il
    for k in range(two_h):
        if x_r[k] > 0.5:
            prev = mu_new_il
        elif x_m[k] > 0.5:
            prev = (1 - rho_il) * prev
        else:
            prev = prev + mu_inc_2h[k]
        mu[k] = prev
    return mu


def _recompute_ara1_mu(x_m, x_r, mu_inc_2h, mu_0_il, mu_new_il, rho_il, two_h):
    """Anchor-aware recursion, including the anchor-reset-at-replacement fix
    (see maintenance/ara1.py's module docstring for why the reset is needed)."""
    mu = np.zeros(two_h)
    mu_prev = mu_0_il
    anchor_prev = mu_0_il
    for k in range(two_h):
        if x_r[k] > 0.5:
            mu_prev = mu_new_il
            anchor_prev = mu_new_il
        elif x_m[k] > 0.5:
            new_mu = mu_prev - rho_il * (mu_prev - anchor_prev)
            anchor_prev = mu_prev
            mu_prev = new_mu
        else:
            mu_prev = mu_prev + mu_inc_2h[k]
        mu[k] = mu_prev
    return mu


def _recompute_wiener_v(x, x_r_row, sigma_il, v_0_il, v_new_il, M, two_h):
    v = np.zeros(two_h)
    prev = v_0_il
    for k in range(two_h):
        if x_r_row[k] > 0.5:
            prev = v_new_il
        else:
            n_missions = sum(x[j, k] for j in range(1, M + 1))
            prev = prev + sigma_il ** 2 * n_missions
        v[k] = prev
    return v


@pytest.fixture
def solved_ard1(make_tiny_problem):
    data = make_tiny_problem(
        [["wiener", "wiener"], ["wiener", "wiener"], ["wiener", "wiener"]],
        F=3, M=1, L=2, H=3, mu_inc=0.09, rho=0.6, sigma=0.03,
    )
    shared = solver._parse_and_validate(data)
    result = solver._solve_single_horizon(shared["H"], shared)
    assert result["status"] == "optimal"
    return result, shared


@pytest.fixture
def solved_ara1(make_tiny_problem):
    data = make_tiny_problem(
        [["wiener"], ["gaussian"]], F=2, M=1, L=1, H=3, mu_inc=0.09, rho=0.6,
        sigma=0.03, maintenance_type=[["ARA1"], ["ARD1"]],
    )
    shared = solver._parse_and_validate(data)
    result = solver._solve_single_horizon(shared["H"], shared)
    assert result["status"] == "optimal"
    return result, shared


def test_wiener_ard1_mean_dominates_recursion(solved_ard1):
    result, shared = solved_ard1
    F, L, H, M = shared["F"], shared["L"], shared["H"], shared["M"]
    two_h = 2 * H
    mu_inc_2h = solver._wrap_increment(shared["mu_inc_raw"], F, M, L, H)
    for i in range(F):
        for l in range(L):
            delta = np.array([
                sum(result["x"][i, j, k] * mu_inc_2h[i, j - 1, l, k] for j in range(1, M + 1))
                for k in range(two_h)
            ])
            expected = _recompute_ard1_mu(
                result["x_m"][i, l], result["x_r"][i, l], delta,
                shared["mu_0"][i, l], shared["mu_new"][i, l], shared["rho"][i, l], two_h,
            )
            assert np.all(result["mu"][i, l] >= expected - 1e-6), (i, l)


def test_wiener_variance_dominates_mission_only_recursion(solved_ard1):
    result, shared = solved_ard1
    F, L, H, M = shared["F"], shared["L"], shared["H"], shared["M"]
    two_h = 2 * H
    for i in range(F):
        for l in range(L):
            expected_v = _recompute_wiener_v(
                result["x"][i], result["x_r"][i, l], shared["sigma"][i, l],
                shared["v_0"][i, l], shared["v_new"][i, l], M, two_h,
            )
            assert np.all(result["v"][i, l] >= expected_v - 1e-6), (i, l)


def test_wiener_ara1_mean_dominates_anchor_aware_recursion(solved_ara1):
    result, shared = solved_ara1
    M, H = shared["M"], shared["H"]
    two_h = 2 * H
    mu_inc_2h = solver._wrap_increment(shared["mu_inc_raw"], shared["F"], M, shared["L"], H)
    i, l = 0, 0  # the wiener/ARA1 component
    delta = np.array([
        sum(result["x"][i, j, k] * mu_inc_2h[i, j - 1, l, k] for j in range(1, M + 1))
        for k in range(two_h)
    ])
    expected = _recompute_ara1_mu(
        result["x_m"][i, l], result["x_r"][i, l], delta,
        shared["mu_0"][i, l], shared["mu_new"][i, l], shared["rho"][i, l], two_h,
    )
    assert np.all(result["mu"][i, l] >= expected - 1e-6)


def test_wiener_loop_constraint_holds(solved_ard1):
    result, shared = solved_ard1
    H = shared["H"]
    for i in range(shared["F"]):
        for l in range(shared["L"]):
            assert result["mu"][i, l, 2 * H - 1] <= result["mu"][i, l, H - 1] + 1e-6
            assert result["v"][i, l, 2 * H - 1] <= result["v"][i, l, H - 1] + 1e-6


def test_wiener_reliability_constraint_holds_exact(solved_ard1):
    result, shared = solved_ard1
    from scipy.stats import norm

    phi_inv = norm.ppf(1 - shared["epsilon"])
    for i in range(shared["F"]):
        for l in range(shared["L"]):
            tau_il = shared["tau"][i, l]
            for k in range(2 * shared["H"]):
                mu = result["mu"][i, l, k]
                v = max(result["v"][i, l, k], 0.0)
                assert mu + phi_inv * np.sqrt(v) <= tau_il + 1e-4


def test_wiener_lp_formulation_solves(make_tiny_problem):
    data = make_tiny_problem(
        [["wiener"], ["gaussian"]], F=2, M=1, L=1, H=2, formulation="lp",
    )
    shared = solver._parse_and_validate(data)
    result = solver._solve_single_horizon(shared["H"], shared)
    assert result["status"] == "optimal"
    from scipy.stats import norm
    phi_inv_sq = norm.ppf(1 - shared["epsilon"]) ** 2
    tau_il = shared["tau"][0, 0]
    for k in range(2 * shared["H"]):
        mu, v = result["mu"][0, 0, k], result["v"][0, 0, k]
        assert phi_inv_sq * v + 2 * tau_il * mu <= tau_il ** 2 + 1e-4


def test_wiener_replacement_reset_is_enforced():
    """Directly force x_r=1 for one component at one step (bypassing the
    optimizer's economic choice) and check the replacement big-M rows pin
    mu/v to mu_new/v_new exactly at that step -- same pattern as
    test_gaussian.py's test_replacement_reset_is_enforced (only x_r at the
    forced step is pinned; everything else is left free for the solver, to
    avoid over-constraining the rest of the schedule into genuine
    infeasibility against the loop constraint)."""
    import gurobipy as gp
    from gurobipy import GRB
    from types import SimpleNamespace

    from fleet_management.models import base

    F, L, M, H = 2, 1, 1, 2
    two_h = 2 * H
    forced_k = 1

    model = gp.Model("test_wiener_replacement_reset")
    model.Params.OutputFlag = 0
    model.Params.NonConvex = 2

    x = model.addVars(F, M + 1, two_h, vtype=GRB.BINARY, name="x")
    x_m = model.addVars(F, L, two_h, vtype=GRB.BINARY, name="x_m")
    x_r = model.addVars(F, L, two_h, vtype=GRB.BINARY, name="x_r")
    z = model.addVars(F, L, two_h, lb=0.0, name="z")
    u = model.addVar(lb=0.0, name="u")

    # x_r=1 forces x[0,0,forced_k]=1 via the maintenance-day compatibility
    # constraint, and x_m=0 via the no-repair-and-replace constraint.
    x_r[0, 0, forced_k].LB = x_r[0, 0, forced_k].UB = 1.0

    mu_new_val, v_new_val = 0.02, 0.0001
    ctx = SimpleNamespace(
        model=model, F=F, H=H, M=M, L=L, two_h=two_h,
        x=x, x_m=x_m, x_r=x_r, z=z, u=u, mu={}, v={}, mu_last={},
        tau=np.ones((F, L)), rho=np.full((F, L), 0.7),
        mu_0=np.full((F, L), 0.1), v_0=np.full((F, L), 0.0004),
        mu_new=np.full((F, L), mu_new_val), v_new=np.full((F, L), v_new_val),
        mu_inc=np.full((F, M, L, two_h), 0.05),
        sigma=np.full((F, L), 0.03), v_max_user=None,
        maintenance_type=[["ARD1"], ["ARD1"]],
        phi_inv_sq=base.phi_inv_sq(0.01), epsilon=0.01, formulation="exact",
    )

    wiener.build_component(ctx, 0, 0)
    base.add_assignment_constraints(model, x, x_m, x_r, F, L, M, two_h)
    model.setObjective(gp.LinExpr(0.0), GRB.MINIMIZE)
    model.optimize()

    assert model.Status == GRB.OPTIMAL
    assert abs(ctx.mu[0, 0, forced_k].X - mu_new_val) < 1e-6
    assert abs(ctx.v[0, 0, forced_k].X - v_new_val) < 1e-6
