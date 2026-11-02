"""Tests for fleet_management.models.inverse_gaussian.

solve_mu_bar / validate_inputs tests are pure numpy+scipy (no Gurobi). The
dynamics test builds and solves a real (tiny) Gurobi model. Requires a Gurobi
license for the dynamics test only.
"""

import numpy as np
import pytest
from scipy.stats import invgauss

from fleet_management import solver
from fleet_management.models import inverse_gaussian


# ======================================================================
# solve_mu_bar (offline reliability cap) -- pure numpy/scipy
# ======================================================================

@pytest.mark.parametrize("eta,tau,epsilon", [
    (0.15, 1.0, 0.01),
    (0.05, 1.0, 0.001),
    (0.3, 2.0, 0.005),
    (1.0, 1.0, 0.01),
    (5.0, 1.0, 0.01),
])
def test_solve_mu_bar_matches_scipy_invgauss(eta, tau, epsilon):
    mu_bar = inverse_gaussian.solve_mu_bar(eta, tau, epsilon)
    lam = mu_bar / eta
    p_fail = 1 - invgauss.cdf(tau, mu=mu_bar / lam, scale=lam)
    assert p_fail == pytest.approx(epsilon, abs=1e-4)


def test_solve_mu_bar_non_binding_at_extreme_eta():
    # At very large eta (huge coefficient of variation), the IG distribution
    # becomes so right-skewed that most mass sits well below the mean, so
    # Pr(D > tau) stays small even at mu = tau: the reliability cap is
    # non-binding and mu_bar collapses to (approximately) tau.
    mu_bar = inverse_gaussian.solve_mu_bar(eta=1e5, tau=1.0, epsilon=0.01)
    assert mu_bar == pytest.approx(1.0, abs=1e-6)


def test_solve_mu_bar_never_unachievable_for_valid_epsilon():
    # By Markov's inequality, Pr(D > tau) <= E[D]/tau = mu/tau -> 0 as mu -> 0,
    # for *any* eta > 0. So the reliability target (epsilon > 0) can never be
    # unachievable near mu = 0: solve_mu_bar's defensive ValueError branch is
    # a numerical safety net, not a reachable modelling scenario. Sweep a wide
    # range of eta and confirm a finite root (or the non-binding cap) is
    # always found instead of an exception.
    for eta in [1e-6, 1e-3, 0.05, 0.5, 5.0, 50.0, 500.0, 1e4, 1e6]:
        mu_bar = inverse_gaussian.solve_mu_bar(eta, tau=1.0, epsilon=0.01)
        assert 0 < mu_bar <= 1.0


def test_solve_mu_bar_is_below_tau():
    mu_bar = inverse_gaussian.solve_mu_bar(0.15, 1.0, 0.01)
    assert 0 < mu_bar < 1.0


def test_solve_mu_bar_monotone_in_eta():
    # Larger eta (more dispersion per unit mean) should require a smaller
    # mean to hold Pr(D > tau) <= epsilon -- i.e. mu_bar decreases in eta,
    # for eta not so large the target becomes unachievable.
    bars = [inverse_gaussian.solve_mu_bar(eta, 1.0, 0.01) for eta in [0.05, 0.15, 0.5, 1.0]]
    assert all(b1 >= b2 for b1, b2 in zip(bars, bars[1:]))


# ======================================================================
# validate_inputs -- pure numpy
# ======================================================================

def _base_inputs(F=1, L=1):
    mask = np.ones((F, L), dtype=bool)
    mu_0 = np.full((F, L), 0.1)
    mu_inc = np.full((F, 1, L, 3), 0.08)
    tau = np.ones((F, L))
    rho = np.full((F, L), 0.7)
    eta = np.full((F, L), 0.15)
    mt = [["ARD1"] * L for _ in range(F)]
    return mask, mu_0, mu_inc, tau, rho, eta, mt


def test_validate_inputs_happy_path():
    inverse_gaussian.validate_inputs(*_base_inputs())


def test_validate_inputs_rejects_ara1():
    mask, mu_0, mu_inc, tau, rho, eta, mt = _base_inputs()
    mt[0][0] = "ARA1"
    with pytest.raises(ValueError):
        inverse_gaussian.validate_inputs(mask, mu_0, mu_inc, tau, rho, eta, mt)


def test_validate_inputs_rejects_nonpositive_eta():
    mask, mu_0, mu_inc, tau, rho, eta, mt = _base_inputs()
    eta[0, 0] = -0.1
    with pytest.raises(ValueError):
        inverse_gaussian.validate_inputs(mask, mu_0, mu_inc, tau, rho, eta, mt)


def test_validate_inputs_rejects_mu0_ge_tau():
    mask, mu_0, mu_inc, tau, rho, eta, mt = _base_inputs()
    mu_0[0, 0] = 2.0
    with pytest.raises(ValueError):
        inverse_gaussian.validate_inputs(mask, mu_0, mu_inc, tau, rho, eta, mt)


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


def test_ig_mean_dynamics_and_reliability_cap(make_tiny_problem):
    """The solved mu must dominate (be >=) the exact ARD1 recursion, not
    necessarily equal it -- see the docstring of
    test_gaussian_mean_dynamics_dominate_recursion in test_gaussian.py for
    why exact equality is not a safe invariant (spec's slack-robustness
    remark, Section 5.3)."""
    data = make_tiny_problem(
        [["inverse_gaussian", "inverse_gaussian"],
         ["inverse_gaussian", "inverse_gaussian"],
         ["inverse_gaussian", "inverse_gaussian"]],
        F=3, M=1, L=2, H=3, mu_inc=0.09, rho=0.6, eta=0.15,
    )
    shared = solver._parse_and_validate(data)
    result = solver._solve_single_horizon(shared["H"], shared)
    assert result["status"] == "optimal"

    F, L, M, H = shared["F"], shared["L"], shared["M"], shared["H"]
    two_h = 2 * H
    mu_inc_2h = solver._wrap_increment(shared["mu_inc_raw"], F, M, L, H)
    mu_bar = inverse_gaussian.solve_mu_bar(0.15, shared["tau"][0, 0], shared["epsilon"])

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
            assert np.all(result["mu"][i, l] <= mu_bar + 1e-6)
            assert result["mu"][i, l, 2 * H - 1] <= result["mu"][i, l, H - 1] + 1e-6
            # IG components don't track variance: output must be NaN there.
            assert np.all(np.isnan(result["v"][i, l]))
