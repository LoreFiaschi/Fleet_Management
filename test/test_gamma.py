"""Tests for fleet_management.models.gamma.

solve_alpha_hat/validate_inputs tests are pure numpy+scipy (no Gurobi). The
dynamics tests build and solve a real (tiny) Gurobi model and check the
*domination* invariant (solved >= recomputed recursion), same reasoning as
Gaussian/IG/Wiener. Requires a Gurobi license for the dynamics/reliability/
replacement tests.
"""

import warnings

import numpy as np
import pytest
from scipy.stats import gamma as gamma_dist

from fleet_management import solver
from fleet_management.models import gamma


# ======================================================================
# solve_alpha_hat (offline reliability cap) -- pure numpy/scipy
# ======================================================================

@pytest.mark.parametrize("beta,tau,epsilon", [
    (0.5, 1.0, 0.01),
    (0.2, 1.0, 0.001),
    (1.0, 2.0, 0.005),
    (0.05, 1.0, 0.01),
    (10.0, 1.0, 0.01),
])
def test_solve_alpha_hat_matches_scipy_gamma(beta, tau, epsilon):
    alpha_hat = gamma.solve_alpha_hat(beta, tau, epsilon)
    p_fail = gamma_dist.sf(tau, a=alpha_hat, scale=beta)
    assert p_fail == pytest.approx(epsilon, abs=1e-4)


def test_solve_alpha_hat_non_binding_warns_and_caps():
    with pytest.warns(UserWarning, match="non-binding"):
        alpha_hat = gamma.solve_alpha_hat(beta=1000.0, tau=1.0, epsilon=0.01)
    assert alpha_hat == pytest.approx(1.0 / 1000.0, rel=1e-6)


def test_solve_alpha_hat_is_below_alpha_bar():
    alpha_hat = gamma.solve_alpha_hat(0.5, 1.0, 0.01)
    assert 0 < alpha_hat < 1.0 / 0.5


# ======================================================================
# validate_inputs -- pure numpy
# ======================================================================

def _base_inputs(F=1, L=1):
    mask = np.ones((F, L), dtype=bool)
    mu_0 = np.full((F, L), 0.1)
    alpha_inc = np.full((F, 1, L, 3), 0.03)
    tau = np.ones((F, L))
    rho = np.full((F, L), 0.7)
    beta = np.full((F, L), 0.5)
    mt = [["ARD1"] * L for _ in range(F)]
    return mask, mu_0, alpha_inc, tau, rho, beta, mt


def test_validate_inputs_happy_path_ard1():
    gamma.validate_inputs(*_base_inputs())


def test_validate_inputs_happy_path_ara1():
    mask, mu_0, alpha_inc, tau, rho, beta, mt = _base_inputs()
    mt[0][0] = "ARA1"
    gamma.validate_inputs(mask, mu_0, alpha_inc, tau, rho, beta, mt)


def test_validate_inputs_rejects_bad_maintenance_type():
    mask, mu_0, alpha_inc, tau, rho, beta, mt = _base_inputs()
    mt[0][0] = "bogus"
    with pytest.raises(ValueError):
        gamma.validate_inputs(mask, mu_0, alpha_inc, tau, rho, beta, mt)


def test_validate_inputs_rejects_nonpositive_beta():
    mask, mu_0, alpha_inc, tau, rho, beta, mt = _base_inputs()
    beta[0, 0] = 0.0
    with pytest.raises(ValueError):
        gamma.validate_inputs(mask, mu_0, alpha_inc, tau, rho, beta, mt)


def test_validate_inputs_rejects_nonpositive_alpha_inc():
    mask, mu_0, alpha_inc, tau, rho, beta, mt = _base_inputs()
    alpha_inc[0, :, 0, :] = -0.01
    with pytest.raises(ValueError):
        gamma.validate_inputs(mask, mu_0, alpha_inc, tau, rho, beta, mt)


def test_validate_inputs_rejects_mu0_ge_tau():
    mask, mu_0, alpha_inc, tau, rho, beta, mt = _base_inputs()
    mu_0[0, 0] = 2.0
    with pytest.raises(ValueError):
        gamma.validate_inputs(mask, mu_0, alpha_inc, tau, rho, beta, mt)


# ======================================================================
# Dynamics correctness via a real (tiny) Gurobi solve
# ======================================================================

def _recompute_ard1_alpha(x_m, x_r, alpha_inc_2h, alpha_0_il, alpha_new_il, rho_il, two_h):
    alpha = np.zeros(two_h)
    prev = alpha_0_il
    for k in range(two_h):
        if x_r[k] > 0.5:
            prev = alpha_new_il
        elif x_m[k] > 0.5:
            prev = (1 - rho_il) * prev
        else:
            prev = prev + alpha_inc_2h[k]
        alpha[k] = prev
    return alpha


@pytest.fixture
def solved_ard1(make_tiny_problem):
    # mu_0=0.05 keeps alpha_0=mu_0/beta=0.1 comfortably below alpha_hat
    # (~0.162 for beta=0.5, tau=1, eps=0.01) -- with mu_0=0.1 (the fixture's
    # default), alpha_0=0.2 already exceeds alpha_hat, and with M=1 the one
    # train forced onto the mission on day 0 (no maintenance possible that
    # day) has no way to satisfy reliability at k=0, a genuine structural
    # infeasibility from the test's own initial condition, not a model bug.
    data = make_tiny_problem(
        [["gamma", "gamma"], ["gamma", "gamma"], ["gamma", "gamma"]],
        F=3, M=1, L=2, H=3, alpha_inc=0.05, rho=0.6, beta=0.5, mu_0=0.05,
    )
    shared = solver._parse_and_validate(data)
    result = solver._solve_single_horizon(shared["H"], shared)
    assert result["status"] == "optimal"
    return result, shared


def test_gamma_mu_dominates_alpha_beta_recursion(solved_ard1):
    """result['mu'] is alpha*beta (spec's post-processing substitution) --
    check it dominates the recomputed alpha recursion scaled by beta."""
    result, shared = solved_ard1
    F, L, H, M = shared["F"], shared["L"], shared["H"], shared["M"]
    two_h = 2 * H
    alpha_inc_2h = solver._wrap_increment(shared["alpha_inc_raw"], F, M, L, H)
    beta = shared["beta"]
    for i in range(F):
        for l in range(L):
            beta_il = beta[i, l]
            alpha_0_il = shared["mu_0"][i, l] / beta_il
            alpha_new_il = shared["mu_new"][i, l] / beta_il
            delta = np.array([
                sum(result["x"][i, j, k] * alpha_inc_2h[i, j - 1, l, k] for j in range(1, M + 1))
                for k in range(two_h)
            ])
            expected_alpha = _recompute_ard1_alpha(
                result["x_m"][i, l], result["x_r"][i, l], delta,
                alpha_0_il, alpha_new_il, shared["rho"][i, l], two_h,
            )
            assert np.all(result["mu"][i, l] >= expected_alpha * beta_il - 1e-6), (i, l)


def test_gamma_reliability_cap_and_loop_constraint_hold(solved_ard1):
    result, shared = solved_ard1
    F, L, H = shared["F"], shared["L"], shared["H"]
    for i in range(F):
        for l in range(L):
            beta_il = shared["beta"][i, l]
            alpha_hat = gamma.solve_alpha_hat(beta_il, shared["tau"][i, l], shared["epsilon"])
            alpha_sol = result["mu"][i, l] / beta_il
            assert np.all(alpha_sol <= alpha_hat + 1e-6)
            assert result["mu"][i, l, 2 * H - 1] <= result["mu"][i, l, H - 1] + 1e-6
            # v is not tracked by Gamma -- must be NaN in the output.
            assert np.all(np.isnan(result["v"][i, l]))


def test_gamma_lp_formulation_solves(make_tiny_problem):
    # Gamma's reliability is a precomputed linear cap regardless of
    # formulation, so "lp" should solve identically well.
    data = make_tiny_problem(
        [["gamma"], ["gaussian"]], F=2, M=1, L=1, H=2, formulation="lp",
    )
    shared = solver._parse_and_validate(data)
    result = solver._solve_single_horizon(shared["H"], shared)
    assert result["status"] == "optimal"


def test_gamma_replacement_reset_is_enforced():
    """Directly force x_r=1 for one component at one step and check the
    replacement big-M rows pin alpha (hence mu=alpha*beta) to alpha_new*beta
    exactly at that step."""
    import gurobipy as gp
    from gurobipy import GRB
    from types import SimpleNamespace

    from fleet_management.models import base

    F, L, M, H = 2, 1, 1, 2
    two_h = 2 * H
    forced_k = 1

    model = gp.Model("test_gamma_replacement_reset")
    model.Params.OutputFlag = 0

    x = model.addVars(F, M + 1, two_h, vtype=GRB.BINARY, name="x")
    x_m = model.addVars(F, L, two_h, vtype=GRB.BINARY, name="x_m")
    x_r = model.addVars(F, L, two_h, vtype=GRB.BINARY, name="x_r")
    z = model.addVars(F, L, two_h, lb=0.0, name="z")
    u = model.addVar(lb=0.0, name="u")

    x_r[0, 0, forced_k].LB = x_r[0, 0, forced_k].UB = 1.0

    beta_il = 0.5
    mu_new_val = 0.0
    ctx = SimpleNamespace(
        model=model, F=F, H=H, M=M, L=L, two_h=two_h,
        x=x, x_m=x_m, x_r=x_r, z=z, u=u, mu={}, alpha_last={},
        tau=np.ones((F, L)), rho=np.full((F, L), 0.6),
        mu_0=np.full((F, L), 0.1), mu_new=np.full((F, L), mu_new_val),
        alpha_inc=np.full((F, M, L, two_h), 0.03),
        beta=np.full((F, L), beta_il),
        maintenance_type=[["ARD1"], ["ARD1"]],
        epsilon=0.01,
    )

    gamma.build_component(ctx, 0, 0)
    base.add_assignment_constraints(model, x, x_m, x_r, F, L, M, two_h)
    model.setObjective(gp.LinExpr(0.0), GRB.MINIMIZE)
    model.optimize()

    assert model.Status == GRB.OPTIMAL
    mu_forced = ctx.mu[0, 0, forced_k]
    mu_forced_val = mu_forced.getValue() if hasattr(mu_forced, "getValue") else mu_forced.X
    assert abs(mu_forced_val - mu_new_val) < 1e-6
