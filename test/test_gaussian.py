"""Tests for fleet_management.models.gaussian.

The validate_inputs tests are pure numpy (no Gurobi). The dynamics tests build
and solve a real (tiny) Gurobi model via solver._solve_single_horizon and then
check the solution against a hand-recomputed ARD1 recursion -- this validates
that build_component's constraints reproduce the intended recursion, whatever
schedule the optimizer actually picks, rather than asserting a specific
schedule. Requires a Gurobi license.
"""

import numpy as np
import pytest

from fleet_management import solver
from fleet_management.models import gaussian


# ======================================================================
# validate_inputs (no Gurobi)
# ======================================================================

def _base_inputs(F=1, L=1):
    mask = np.ones((F, L), dtype=bool)
    mu_0 = np.full((F, L), 0.1)
    v_0 = np.full((F, L), 0.0004)
    mu_inc = np.full((F, 1, L, 3), 0.08)
    v_inc = np.full((F, 1, L, 3), 0.0004)
    tau = np.ones((F, L))
    rho = np.full((F, L), 0.7)
    mt = [["ARD1"] * L for _ in range(F)]
    return mask, mu_0, v_0, mu_inc, v_inc, tau, rho, mt


def test_validate_inputs_happy_path():
    gaussian.validate_inputs(*_base_inputs())


def test_validate_inputs_rejects_ara1():
    mask, mu_0, v_0, mu_inc, v_inc, tau, rho, mt = _base_inputs()
    mt[0][0] = "ARA1"
    with pytest.raises(ValueError):
        gaussian.validate_inputs(mask, mu_0, v_0, mu_inc, v_inc, tau, rho, mt)


def test_validate_inputs_rejects_rho_out_of_range():
    mask, mu_0, v_0, mu_inc, v_inc, tau, rho, mt = _base_inputs()
    rho[0, 0] = 1.5
    with pytest.raises(ValueError):
        gaussian.validate_inputs(mask, mu_0, v_0, mu_inc, v_inc, tau, rho, mt)


def test_validate_inputs_rejects_3sigma_violation_mu0():
    mask, mu_0, v_0, mu_inc, v_inc, tau, rho, mt = _base_inputs()
    mu_0[0, 0] = 0.01  # too small relative to v_0=0.0004 (3*sqrt = 0.06)
    with pytest.raises(ValueError):
        gaussian.validate_inputs(mask, mu_0, v_0, mu_inc, v_inc, tau, rho, mt)


def test_validate_inputs_rejects_mu_ge_tau():
    mask, mu_0, v_0, mu_inc, v_inc, tau, rho, mt = _base_inputs()
    mu_0[0, 0] = 1.5
    with pytest.raises(ValueError):
        gaussian.validate_inputs(mask, mu_0, v_0, mu_inc, v_inc, tau, rho, mt)


def test_validate_inputs_skips_entries_outside_mask():
    # An out-of-range value at a masked-out (i, l) must not raise.
    mask, mu_0, v_0, mu_inc, v_inc, tau, rho, mt = _base_inputs(F=2, L=1)
    mask[1, 0] = False
    rho[1, 0] = 99.0  # invalid, but not checked since mask is False there
    gaussian.validate_inputs(mask, mu_0, v_0, mu_inc, v_inc, tau, rho, mt)


# ======================================================================
# Dynamics correctness via a real (tiny) Gurobi solve
# ======================================================================

def _recompute_ard1(x_m, x_r, mu_inc_2h, mu_0_il, mu_new_il, rho_il, two_h):
    """Reference ARD1 recursion for one component, from an already-solved
    schedule's x_m/x_r/mu_inc, independent of build_component's constraints.
    """
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


@pytest.fixture
def solved_mixed(make_tiny_problem):
    """A small F=3, M=1, L=2 (gaussian, gaussian) fleet, solved for real."""
    data = make_tiny_problem(
        [["gaussian", "gaussian"], ["gaussian", "gaussian"], ["gaussian", "gaussian"]],
        F=3, M=1, L=2, H=3, mu_inc=0.09, v_inc=0.0004, rho=0.6,
    )
    shared = solver._parse_and_validate(data)
    result = solver._solve_single_horizon(shared["H"], shared)
    assert result["status"] == "optimal"
    return result, shared


def test_gaussian_mean_dynamics_dominate_recursion(solved_mixed):
    """The solved mu must be a valid *upper bound* on the exact ARD1
    recursion, not necessarily equal to it: the accumulation row is a
    one-sided (>=) big-M relaxation, and per spec's slack-robustness remark
    (Section 5.3) the tracked state may carry slack wherever neither cost
    mechanism (C_D*u, C_R*z) happens to bind for that component -- this does
    not endanger correctness, since the tracked state still dominates the
    true trajectory componentwise. Exact equality is therefore not a safe
    invariant to assert in general (it depends on incidental cost pressure
    and solver tie-breaking); domination is the actual guarantee and is what
    we check here.
    """
    result, shared = solved_mixed
    F, L, H = shared["F"], shared["L"], shared["H"]
    two_h = 2 * H
    mu_inc_2h = solver._wrap_increment(shared["mu_inc_raw"], F, shared["M"], L, H)
    for i in range(F):
        for l in range(L):
            delta = np.array([
                sum(
                    result["x"][i, j, k] * mu_inc_2h[i, j - 1, l, k]
                    for j in range(1, shared["M"] + 1)
                )
                for k in range(two_h)
            ])
            expected = _recompute_ard1(
                result["x_m"][i, l], result["x_r"][i, l], delta,
                shared["mu_0"][i, l], shared["mu_new"][i, l], shared["rho"][i, l], two_h,
            )
            assert np.all(result["mu"][i, l] >= expected - 1e-6), (i, l)


def test_gaussian_loop_constraint_holds(solved_mixed):
    result, shared = solved_mixed
    H = shared["H"]
    for i in range(shared["F"]):
        for l in range(shared["L"]):
            assert result["mu"][i, l, 2 * H - 1] <= result["mu"][i, l, H - 1] + 1e-6
            assert result["v"][i, l, 2 * H - 1] <= result["v"][i, l, H - 1] + 1e-6


def test_gaussian_reliability_constraint_holds_exact(solved_mixed):
    result, shared = solved_mixed
    from scipy.stats import norm

    phi_inv = norm.ppf(1 - shared["epsilon"])
    for i in range(shared["F"]):
        for l in range(shared["L"]):
            tau_il = shared["tau"][i, l]
            for k in range(2 * shared["H"]):
                mu = result["mu"][i, l, k]
                v = max(result["v"][i, l, k], 0.0)
                assert mu + phi_inv * np.sqrt(v) <= tau_il + 1e-4


def test_gaussian_lp_formulation_solves(make_tiny_problem):
    data = make_tiny_problem(
        [["gaussian"], ["gaussian"]], F=2, M=1, L=1, H=2, formulation="lp",
    )
    shared = solver._parse_and_validate(data)
    result = solver._solve_single_horizon(shared["H"], shared)
    assert result["status"] == "optimal"
    from scipy.stats import norm
    phi_inv_sq = norm.ppf(1 - shared["epsilon"]) ** 2
    for i in range(2):
        for k in range(2 * shared["H"]):
            tau_il = shared["tau"][i, 0]
            mu, v = result["mu"][i, 0, k], result["v"][i, 0, k]
            assert phi_inv_sq * v + 2 * tau_il * mu <= tau_il ** 2 + 1e-4


def test_quadratic_penalty_solves(make_tiny_problem):
    data = make_tiny_problem(
        [["gaussian"], ["gaussian"]], F=2, M=1, L=1, H=2,
        formulation="exact", penalty_type="quadratic",
    )
    shared = solver._parse_and_validate(data)
    result = solver._solve_single_horizon(shared["H"], shared)
    assert result["status"] == "optimal"
    for i in range(2):
        for k in range(2 * shared["H"]):
            assert result["u"] + 1e-6 >= result["mu"][i, 0, k] ** 2


def test_replacement_reset_is_enforced():
    """Directly force x_r=1 for one component at one step (via a variable
    bound, bypassing the optimizer's economic choice entirely) and check
    that the replacement big-M rows pin mu/v to mu_new/v_new exactly at that
    step. F=2 so train 1 can absorb the mission on train 0's forced
    maintenance day (spec requires F > M; train 1's own component is left
    unconstrained since only train 0's is under test).

    Only x_r at the forced step is pinned; every other assignment/repair/
    replacement choice (including train 0's own schedule on other days) is
    left free for the solver to pick. Forcing more than that risks
    over-constraining the schedule into genuine infeasibility -- e.g. forcing
    missions on every remaining day with no further maintenance would make
    mu accumulate past what the hard loop constraint (mu_2H <= mu_H) allows,
    which is the model correctly rejecting an unsustainable schedule, not a
    bug under test here.
    """
    import gurobipy as gp
    from gurobipy import GRB
    from types import SimpleNamespace

    from fleet_management.models import base

    F, L, M, H = 2, 1, 1, 2
    two_h = 2 * H
    forced_k = 1

    model = gp.Model("test_replacement_reset")
    model.Params.OutputFlag = 0
    model.Params.NonConvex = 2

    x = model.addVars(F, M + 1, two_h, vtype=GRB.BINARY, name="x")
    x_m = model.addVars(F, L, two_h, vtype=GRB.BINARY, name="x_m")
    x_r = model.addVars(F, L, two_h, vtype=GRB.BINARY, name="x_r")
    z = model.addVars(F, L, two_h, lb=0.0, name="z")
    u = model.addVar(lb=0.0, name="u")

    # x_r=1 forces x[0,0,forced_k]=1 via the maintenance-day compatibility
    # constraint, and x_m=0 via the no-repair-and-replace constraint, so
    # neither needs to be pinned explicitly.
    x_r[0, 0, forced_k].LB = x_r[0, 0, forced_k].UB = 1.0

    mu_new_val, v_new_val = 0.02, 0.0001
    ctx = SimpleNamespace(
        model=model, F=F, H=H, M=M, L=L, two_h=two_h,
        x=x, x_m=x_m, x_r=x_r, z=z, u=u, mu={}, v={},
        tau=np.ones((F, L)), rho=np.full((F, L), 0.7),
        mu_0=np.full((F, L), 0.1), v_0=np.full((F, L), 0.0004),
        mu_new=np.full((F, L), mu_new_val), v_new=np.full((F, L), v_new_val),
        mu_inc=np.full((F, M, L, two_h), 0.05), v_inc=np.full((F, M, L, two_h), 0.0002),
        eta=np.full((F, L), np.nan), v_max_user=None,
        phi_inv_sq=base.phi_inv_sq(0.01), epsilon=0.01, formulation="exact",
    )

    gaussian.build_component(ctx, 0, 0)
    base.add_assignment_constraints(model, x, x_m, x_r, F, L, M, two_h)
    model.setObjective(gp.LinExpr(0.0), GRB.MINIMIZE)
    model.optimize()

    assert model.Status == GRB.OPTIMAL
    assert abs(ctx.mu[0, 0, forced_k].X - mu_new_val) < 1e-6
    assert abs(ctx.v[0, 0, forced_k].X - v_new_val) < 1e-6
