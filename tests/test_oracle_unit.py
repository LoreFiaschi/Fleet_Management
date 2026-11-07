"""Unit tests of the reference model (``tests/reference_model.py``) on numbers derived by hand from the spec.

No Gurobi is needed. Spec = ``spec/spec.tex`` (the ``lore`` copy). Every expected number has its
derivation in a comment; the numbers were also checked in a separate computation before they were
written here. Task O1 of WP8: day profiles, cell kinds, constants, trajectories of the Gaussian,
inverse Gaussian and Wiener models, reliability slacks of the Gaussian and Wiener rows, the
property "each linear form implies the exact row", repair cost, damage penalty, costs, the
independent inverse Gaussian cap and the enumeration of schedules.

Task O2 of WP8 (the last part of the file): trajectories and reliability rows of the Gamma model
and of the five rainflow bounds, the independent Gamma cap, ``row_violations``,
``repeated_safety`` and ``best_reference``.
"""

import copy
import math
import os
import warnings

import numpy as np
import pytest
from scipy import special

from fleet_management.config import load_config, read_input
from fleet_management.degradation_model.gamma import mean_cap, sharp_cap
from fleet_management.degradation_model.inverse_gaussian import ig_mean_cap

import reference_model as rm


def close(actual, expected, rel=1e-9, abs_=1e-12):
    """``actual`` equals ``expected`` up to ``rel`` relative and ``abs_`` absolute error (arrays too)."""
    assert np.allclose(actual, expected, rtol=rel, atol=abs_, equal_nan=True), \
        f"{actual!r} != {expected!r}"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
# Gaussian cell of the plan: tau 1, eps 0.05, rho 0.5, mu_0 0.1, v_0 0.0004, increment mu 0.2, v 0.001.
G_BASE = dict(F=2, M=1, H=[2, 2], model="gaussian", tau=1.0, epsilon=0.05, rho=0.5,
              mu_0=0.1, v_0=0.0004, mu=0.2, v=0.001, C_M=2.0, C_R=10.0, C_rep=7.0, C_D=3.0,
              replacement_mu=0.05, replacement_v=0.0001)
IG_BASE = dict(F=2, M=1, H=[2, 2], model="inverse_gaussian", tau=1.0, epsilon=0.05, rho=0.5,
               eta=0.5, mu_0=0.1, mu=0.2, C_M=2.0, C_R=10.0, C_rep=7.0, C_D=3.0,
               replacement_mu=0.05)
W_BASE = dict(F=2, M=1, H=[2, 2], model="wiener", tau=1.0, epsilon=0.05, rho=0.5, sigma=0.02,
              mu_0=0.1, v_0=0.0004, mu=0.2, C_M=2.0, C_R=10.0, C_rep=7.0, C_D=3.0,
              replacement_mu=0.05, replacement_v=0.0001)
ANCHORS = dict(anchor_mu_0=0.05, anchor_v_0=0.0002)


def make(base, **changes):
    return load_config(dict(base, **changes))


def hand_schedule():
    """F = 2 trains, M = 1 mission, L = 1 component, T = 4 days (H = [2, 2]).

    Train 0 (the cell under study): day 1 flies the mission, day 2 repair (depot), day 3 idle,
    day 4 replacement (depot). Train 1: idle on day 1, flies the mission on days 2, 3, 4.
    Every mission is flown by exactly one train each day (spec Eq. a_coverage).
    """
    F, M, L, T = 2, 1, 1, 4
    x = np.zeros((F, M + 1, T), dtype=int)
    m = np.zeros((F, L, T), dtype=int)
    r = np.zeros((F, L, T), dtype=int)
    x[0, 1, 0] = 1
    x[1, 1, 1:] = 1
    x[0, 0, 1] = 1
    x[0, 0, 3] = 1
    m[0, 0, 1] = 1
    r[0, 0, 3] = 1
    return x, m, r


# ---------------------------------------------------------------------------
# 1. day_profile (spec Eq. c_phase, 438-449; 457)
# ---------------------------------------------------------------------------
def test_day_profile_operating_only_wraps_with_period_h2():
    # H1 = 5, H2 = 2, operating profile [0.1, 0.2], no transitory profile (T = 7).
    # k <= H1 without a transitory profile reads index ((k-1) mod 2): days 1..5 -> 0,1,0,1,0.
    # k > H1 reads index ((k-5-1) mod 2): day 6 -> 0, day 7 -> 1.
    cfg = make(G_BASE, H=[5, 2], mu=[[[0.1, 0.2]]], v=0.0, F=2)
    prof = rm.day_profile(cfg, "mu")
    assert prof.shape == (2, 1, 1, 7)
    close(prof[0, 0, 0], [0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 0.2])
    close(prof[1, 0, 0], [0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 0.2])


def test_day_profile_transitory_given():
    # H1 = 2, H2 = 3. Operating [0.1, 0.2, 0.3], transitory [0.05, 0.06].
    # Days 1, 2 read the transitory profile (index k-1); day 3 + : index ((k-3) mod 3) of the
    # operating profile: days 3, 4, 5 -> 0.1, 0.2, 0.3.
    cfg = make(G_BASE, H=[2, 3], mu=[[[0.1, 0.2, 0.3]]], mu_trans=[[[0.05, 0.06]]], v=0.0)
    prof = rm.day_profile(cfg, "mu")
    close(prof[0, 0, 0], [0.05, 0.06, 0.1, 0.2, 0.3])


def test_day_profile_operating_and_transitory_wrap_when_h1_exceeds_h2():
    # H1 = 4 > H2 = 3, no transitory profile: days 1..4 -> idx 0,1,2,0; days 5..7 -> idx 0,1,2.
    cfg = make(G_BASE, H=[4, 3], mu=[[[0.1, 0.2, 0.3]]], v=0.0)
    close(rm.day_profile(cfg, "mu")[0, 0, 0], [0.1, 0.2, 0.3, 0.1, 0.1, 0.2, 0.3])


def test_day_profile_each_key_is_looked_up_on_its_own():
    # mu has a transitory profile, v has none (spec 448: each profile key on its own).
    # H = [2, 2]: mu days: [0.01, 0.02] (transitory), then [0.1, 0.2]; v days: wrap [1e-5, 2e-5] twice.
    cfg = make(G_BASE, H=[2, 2], mu=[[[0.1, 0.2]]], mu_trans=[[[0.01, 0.02]]],
               v=[[[0.00001, 0.00002]]], v_0=0.0004)
    close(rm.day_profile(cfg, "mu")[0, 0, 0], [0.01, 0.02, 0.1, 0.2])
    close(rm.day_profile(cfg, "v")[0, 0, 0], [0.00001, 0.00002, 0.00001, 0.00002])


def test_day_profile_per_train_and_mission():
    # F = 3, L = 1, M = 2, H = [3, 2] (T = 5). value[i, j, h] = 0.01 (1 + 4 i + 2 j + h).
    # Days 1..3 read h = 0,1,0 (wrap with period 2); days 4, 5 read h = 0,1.
    vals = np.array([[[[0.01 * (1 + 4 * i + 2 * j + h) for h in range(2)] for j in range(2)]]
                     for i in range(3)])                       # (F, L, M, H2) = (3, 1, 2, 2)
    cfg = make(G_BASE, F=3, M=2, H=[3, 2], mu=vals.tolist(), v=0.0)
    prof = rm.day_profile(cfg, "mu")
    assert prof.shape == (3, 1, 2, 5)
    for i in range(3):
        for j in range(2):
            h = [(1 + 4 * i + 2 * j + hh) * 0.01 for hh in range(2)]
            close(prof[i, 0, j], [h[0], h[1], h[0], h[0], h[1]])


def test_day_profile_absent_key_is_none():
    assert rm.day_profile(make(G_BASE), "support") is None


def test_day_profile_gamma_beta_without_day_axis_is_repeated():
    # spec 457: a shape without a day axis is repeated along the days. One rate per component.
    cfg = load_config(dict(F=2, M=1, H=[2, 3], model="gamma", tau=1.0, epsilon=0.05, rho=0.5,
                           mu_0=0.1, mu=0.2, gamma_beta=[4.0], C_M=1.0, C_R=1.0, C_rep=1.0,
                           C_D=0.0))
    prof = rm.day_profile(cfg, "gamma_beta")
    assert prof.shape == (2, 1, 1, 5)
    close(prof, 4.0)


# ---------------------------------------------------------------------------
# cell_kind, initial and replacement values
# ---------------------------------------------------------------------------
def test_cell_kind_of_the_three_models():
    for base, kind, tracked in ((G_BASE, "G", ("mu", "v")), (IG_BASE, "IG", ("mu",)),
                                (W_BASE, "W", ("mu", "v"))):
        info = rm.cell_kind(make(base), 0, 0)
        assert info["kind"] == kind
        assert info["tracked"] == tracked
        assert info["law"] == "ardinf"
    assert rm.cell_kind(make(G_BASE), 0, 0)["quadratic"] is True
    assert rm.cell_kind(make(IG_BASE), 0, 0)["quadratic"] is False
    # a linear row is built in the form exact whatever reliability_impl says (spec 919-920)
    assert rm.cell_kind(make(IG_BASE, reliability_impl="tangent"), 0, 0)["impl"] == "exact"
    assert rm.cell_kind(make(G_BASE, reliability_impl="pwl"), 0, 0)["impl"] == "pwl"


def test_cell_kind_of_rainflow_bounds():
    # spec 1347-1371: Mk tracks mu; Ca mu, v; Ho mu, R; Be mu, v; Ch mu, K.
    for bound, kind, tracked in (("markov", "Mk", ("mu",)), ("cantelli", "Ca", ("mu", "v")),
                                 ("hoeffding", "Ho", ("mu", "R")),
                                 ("bernstein", "Be", ("mu", "v")), ("chernoff", "Ch", ("mu", "K"))):
        info = rm.cell_kind(make(RF[bound]), 0, 0)
        assert (info["kind"], info["tracked"], info["bound"]) == (kind, tracked, bound)


def test_initial_and_new_defaults_and_overrides():
    # spec 774-778: a starting anchor defaults to the initial value; spec 421-425: replacement
    # values default to 0.
    cfg = make(G_BASE, repair_model="ard1", replacement_mu=0.0, replacement_v=0.0)
    ini = rm.initial_and_new(cfg)[(0, 0)]
    assert ini["init"] == {"mu": 0.1, "v": 0.0004}
    assert ini["anchor0"] == {"mu": 0.1, "v": 0.0004}
    assert ini["new"] == {"mu": 0.0, "v": 0.0}
    cfg = make(G_BASE, repair_model="ard1", **ANCHORS)
    ini = rm.initial_and_new(cfg)[(0, 0)]
    assert ini["anchor0"] == {"mu": 0.05, "v": 0.0002}
    assert ini["new"] == {"mu": 0.05, "v": 0.0001}


def test_initial_and_new_replacement_as_new_sets_replacement_values_to_zero():
    # spec 857-860: replacement_as_new makes every replacement value 0 (the inputs are then 0).
    cfg = make(dict(G_BASE, replacement_mu=0.0, replacement_v=0.0), replacement_as_new=True)
    assert rm.initial_and_new(cfg)[(0, 0)]["new"] == {"mu": 0.0, "v": 0.0}


def test_initial_and_new_chernoff_defaults():
    # spec 1391-1392: K_0 = s mu_0 and K_new = s mu_new by default; s = 5, mu_0 = 0.1, mu_new = 0.
    cfg = make(RF["chernoff"])
    ini = rm.initial_and_new(cfg)[(0, 0)]
    close(ini["init"]["K"], 0.5)
    close(ini["new"]["K"], 0.0)
    close(ini["init"]["mu"], 0.1)


# ---------------------------------------------------------------------------
# Rainflow cells (used by the constants tests)
# ---------------------------------------------------------------------------
_RF_COMMON = dict(F=2, M=1, H=[2, 2], model="rainflow", tau=1.0, epsilon=0.1, rho=0.5, mu_0=0.1,
                  mu=0.2, C_M=1.0, C_R=1.0, C_rep=1.0, C_D=0.0)
RF = {
    "markov": dict(_RF_COMMON, bound_method="markov"),
    "cantelli": dict(_RF_COMMON, bound_method="cantelli", v=0.01),
    "hoeffding": dict(_RF_COMMON, bound_method="hoeffding", support=0.3),
    "bernstein": dict(_RF_COMMON, bound_method="bernstein", v=0.01, support=0.3),
    "chernoff": dict(_RF_COMMON, bound_method="chernoff", cgf=1.0, s_chernoff=5.0),
}


# ---------------------------------------------------------------------------
# Constants (spec 324-343, 862-884, 1022, 1399-1405, 1508-1529)
# ---------------------------------------------------------------------------
def test_constants_gaussian_hand_values():
    # eps = 0.05: kappa_G = Phi^{-1}(0.95) = 1.6448536; kappa_G^2 = 2.7055435;
    # V_max = tau^2 / kappa_G^2 = 1 / 2.7055435 = 0.3696115 (spec 325-326, 965-966).
    # The row table (875-884): c2 = 1/kappa_G^2, c1 = 0, side = tau.
    const = rm.constants(make(G_BASE))[(0, 0)]
    close(const["kappa_G"], 1.6448536, rel=1e-7)
    close(const["V_max"], 0.3696115, rel=1e-6)
    close(const["V_max"], 1.0 / 1.6448536 ** 2, rel=1e-6)
    close(const["M"]["mu"], 1.0)
    close(const["M"]["v"], const["V_max"])
    close(const["c2"], const["V_max"])
    assert const["c1"] == 0.0 and const["side"] == 1.0 and const["chi"] == "v"


def test_constants_wiener_equals_gaussian():
    # spec 1195-1199: the Wiener row is the Gaussian row with the same V_max.
    g = rm.constants(make(G_BASE))[(0, 0)]
    w = rm.constants(make(W_BASE))[(0, 0)]
    for key in ("kappa_G", "V_max", "c2", "c1", "side"):
        close(w[key], g[key])


def test_constants_cantelli():
    # eps = 0.1: kappa_C = sqrt(0.9 / 0.1) = 3, kappa_C^2 = 9, V_C = eps tau^2/(1-eps) = 1/9.
    # c2 = eps/(1-eps) = 1/9 (spec 880, 1403).
    const = rm.constants(make(RF["cantelli"]))[(0, 0)]
    close(const["kappa_C"], 3.0)
    close(const["V_C"], 1.0 / 9.0)
    close(const["c2"], 1.0 / 9.0)
    close(const["M"]["v"], 1.0 / 9.0)


def test_constants_hoeffding():
    # eps = 0.1: kappa = ln 10 = 2.3025851; range of R = 2 tau^2 / kappa = 2 / ln 10 = 0.8685890;
    # f(mu) = 2 (tau - mu)^2 / kappa, so c2 = 2 / kappa (spec 881, 1405).
    const = rm.constants(make(RF["hoeffding"]))[(0, 0)]
    close(const["kappa"], math.log(10.0))
    close(const["M"]["R"], 2.0 / math.log(10.0))
    close(const["M"]["R"], 0.8685890, rel=1e-6)
    close(const["c2"], 2.0 / math.log(10.0))


def test_constants_chernoff():
    # s = 5, eps = 0.1, tau = 1: range of K = s tau - kappa = 5 - ln 10 = 2.6974149 (spec 1405);
    # the row K - s tau <= ln eps is K <= s tau + ln eps = 5 - ln 10 (spec 1525).
    const = rm.constants(make(RF["chernoff"]))[(0, 0)]
    close(const["M"]["K"], 5.0 - math.log(10.0))
    close(const["M"]["K"], 2.6974149, rel=1e-6)
    close(const["K_cap"], 5.0 - math.log(10.0))


def test_constants_bernstein():
    # eps = 0.1, tau = 1, b = max(b0, bnew, max b_inc) = 0.3 (support 0.3, b0 = bnew = 0).
    # kappa = ln 10 = 2.3025851; c = kappa b / 3 = 0.2302585;
    # V_B = tau (tau - 2c) / (2 kappa) = 0.5394830 / 4.6051702 = 0.1171472;
    # c2 = 1/(2 kappa) = 0.2171472, c1 = -b/3 = -0.1, side = tau - c = 0.7697415 (spec 882, 1404, 1508-1519).
    const = rm.constants(make(RF["bernstein"]))[(0, 0)]
    close(const["b"], 0.3)
    close(const["c"], 0.2302585, rel=1e-6)
    close(const["V_B"], 0.1171472, rel=1e-6)
    close(const["c2"], 0.2171472, rel=1e-6)
    close(const["c1"], -0.1)
    close(const["side"], 0.7697415, rel=1e-6)
    close(const["M"]["v"], const["V_B"])


def test_constants_bernstein_support_constant_takes_the_largest_width():
    # spec 1510: b = max(b0, bnew, max b_inc); bnew only when replacement is allowed.
    # support profile max 0.3, support_0 0.5 (v_0 <= b0^2/4 = 0.0625 holds with v_0 = 0), support_new 0.4.
    cfg = make(RF["bernstein"], support_0=0.5, support_new=0.4)
    close(rm.constants(cfg)[(0, 0)]["b"], 0.5)
    cfg = make(RF["bernstein"], support_0=0.1, support_new=0.4)
    close(rm.constants(cfg)[(0, 0)]["b"], 0.4)
    cfg = make(RF["bernstein"], support_0=0.1, support_new=0.4, allow_replacement=False)
    close(rm.constants(cfg)[(0, 0)]["b"], 0.3)


def test_constants_markov_and_ig_linear_rows():
    # Markov: mu <= eps tau = 0.1 (spec 1492).
    close(rm.constants(make(RF["markov"]))[(0, 0)]["mu_cap"], 0.1)
    const = rm.constants(make(IG_BASE))[(0, 0)]
    assert 0.0 < const["mu_bar"] < 1.0
    close(const["M"]["mu"], 1.0)


# ---------------------------------------------------------------------------
# 2. reference_trajectory (spec 829, Eqs. g_map_mu, g_map_v, ig_map, w_map_v, a_anchor)
# ---------------------------------------------------------------------------
def test_trajectory_gaussian_ardinf():
    # Train 0, schedule of hand_schedule(), rho = 0.5, (1-rho)^2 = 0.25.
    # mu: 0.1 | day 1 mission: 0.1 + 0.2 = 0.3 | day 2 repair: 0.5 * 0.3 = 0.15 | day 3 idle: 0.15 |
    #     day 4 replacement: mu_new = 0.05.
    # v:  0.0004 | 0.0004 + 0.001 = 0.0014 | 0.25 * 0.0014 = 0.00035 | 0.00035 | v_new = 0.0001.
    # Train 1 (idle, then mission on days 2-4): mu 0.1, 0.1, 0.3, 0.5, 0.7; v 0.0004, 0.0004, 0.0014, 0.0024, 0.0034.
    x, m, r = hand_schedule()
    t = rm.reference_trajectory(make(G_BASE), x, m, r)
    close(t["mu"][0, 0], [0.1, 0.3, 0.15, 0.15, 0.05])
    close(t["v"][0, 0], [0.0004, 0.0014, 0.00035, 0.00035, 0.0001])
    close(t["mu"][1, 0], [0.1, 0.1, 0.3, 0.5, 0.7])
    close(t["v"][1, 0], [0.0004, 0.0004, 0.0014, 0.0024, 0.0034])
    for name in ("R", "K", "g_mu", "g_v", "g_R", "g_K"):
        assert np.isnan(t[name]).all(), name


def test_trajectory_gaussian_ard1_with_starting_anchors():
    # anchor_mu_0 = 0.05, anchor_v_0 = 0.0002. Day 1 as ardinf: mu 0.3, v 0.0014.
    # Day 2 repair, ARD1: mu = 0.5 * 0.3 + 0.5 * 0.05 = 0.175;
    #   v = 0.25 * 0.0014 + (1 - 0.25) * 0.0002 = 0.00035 + 0.00015 = 0.0005. Anchors become 0.175, 0.0005.
    # Day 3 idle: states and anchors unchanged. Day 4 replacement: mu 0.05, v 0.0001; anchors equal them.
    # Day 1 (a mission): the anchors stay at the starting anchors.
    x, m, r = hand_schedule()
    t = rm.reference_trajectory(make(G_BASE, repair_model="ard1", **ANCHORS), x, m, r)
    close(t["mu"][0, 0], [0.1, 0.3, 0.175, 0.175, 0.05])
    close(t["v"][0, 0], [0.0004, 0.0014, 0.0005, 0.0005, 0.0001])
    close(t["g_mu"][0, 0], [0.05, 0.05, 0.175, 0.175, 0.05])
    close(t["g_v"][0, 0], [0.0002, 0.0002, 0.0005, 0.0005, 0.0001])
    # train 1 never has an intervention: its anchors stay at the starting anchors
    close(t["g_mu"][1, 0], [0.05] * 5)
    close(t["g_v"][1, 0], [0.0002] * 5)
    # the anchor after a replacement equals the replacement value
    close(t["g_mu"][0, 0, 4], 0.05)
    close(t["g_v"][0, 0, 4], 0.0001)


def test_trajectory_ard1_with_default_anchors_equals_initial_values():
    # Default starting anchors are the initial values (spec 776): mu 0.1, v 0.0004.
    # Day 2 repair: mu = 0.5 * 0.3 + 0.5 * 0.1 = 0.2; v = 0.25 * 0.0014 + 0.75 * 0.0004 = 0.00065.
    x, m, r = hand_schedule()
    t = rm.reference_trajectory(make(G_BASE, repair_model="ard1"), x, m, r)
    close(t["mu"][0, 0, 2], 0.2)
    close(t["v"][0, 0, 2], 0.00065)


def test_trajectory_replacement_as_new():
    # replacement_as_new: the replacement values are 0 (spec 857-860): day 4 gives mu = 0 and v = 0.
    cfg = make(dict(G_BASE, replacement_mu=0.0, replacement_v=0.0), replacement_as_new=True)
    x, m, r = hand_schedule()
    t = rm.reference_trajectory(cfg, x, m, r)
    close(t["mu"][0, 0, 4], 0.0)
    close(t["v"][0, 0, 4], 0.0)


def test_trajectory_without_replacement_array():
    # r = None means no replacement: day 4 is a depot day without an intervention -> idle carry (no increment).
    x, m, r = hand_schedule()
    t = rm.reference_trajectory(make(G_BASE), x, m, None)
    close(t["mu"][0, 0], [0.1, 0.3, 0.15, 0.15, 0.15])


def test_trajectory_rejects_repair_and_replacement_on_one_day():
    x, m, r = hand_schedule()
    m[0, 0, 3] = 1
    with pytest.raises(ValueError):
        rm.reference_trajectory(make(G_BASE), x, m, r)


def test_trajectory_inverse_gaussian_ardinf_and_ard1():
    # Same schedule, mean only (spec Eq. ig_map). ARDINF: 0.1, 0.3, 0.15, 0.15, 0.05.
    # ARD1 with anchor_mu_0 = 0.05: day 2: 0.5 * 0.3 + 0.5 * 0.05 = 0.175; anchors 0.05, 0.05, 0.175, 0.175, 0.05.
    x, m, r = hand_schedule()
    t = rm.reference_trajectory(make(IG_BASE), x, m, r)
    close(t["mu"][0, 0], [0.1, 0.3, 0.15, 0.15, 0.05])
    close(t["mu"][1, 0], [0.1, 0.1, 0.3, 0.5, 0.7])
    assert np.isnan(t["v"]).all() and np.isnan(t["g_mu"]).all()
    t = rm.reference_trajectory(make(IG_BASE, repair_model="ard1", anchor_mu_0=0.05), x, m, r)
    close(t["mu"][0, 0], [0.1, 0.3, 0.175, 0.175, 0.05])
    close(t["g_mu"][0, 0], [0.05, 0.05, 0.175, 0.175, 0.05])
    assert np.isnan(t["v"]).all() and np.isnan(t["g_v"]).all()


def test_trajectory_wiener_ardinf_repair_keeps_the_variance():
    # sigma = 0.02, sigma^2 = 0.0004 per mission. Train 0: v 0.0004 | day 1 mission: 0.0008 |
    # day 2 repair: kept, 0.0008 | day 3 idle: 0.0008 | day 4 replacement: v_new = 0.0001.
    # mu as Gaussian ARDINF: 0.1, 0.3, 0.15, 0.15, 0.05. Train 1: v = 0.0004 + 0.0004 n, n missions so far.
    x, m, r = hand_schedule()
    t = rm.reference_trajectory(make(W_BASE), x, m, r)
    close(t["mu"][0, 0], [0.1, 0.3, 0.15, 0.15, 0.05])
    close(t["v"][0, 0], [0.0004, 0.0008, 0.0008, 0.0008, 0.0001])
    close(t["v"][1, 0], [0.0004, 0.0004, 0.0008, 0.0012, 0.0016])
    assert np.isnan(t["g_mu"]).all() and np.isnan(t["g_v"]).all()


def test_trajectory_wiener_ard1_has_a_mean_anchor_and_no_variance_anchor():
    # ARD1, anchor_mu_0 = 0.05: day 2 mean = 0.5 * 0.3 + 0.5 * 0.05 = 0.175; the variance is still
    # kept (spec 1144) and has no anchor (spec 764-765, 1139).
    x, m, r = hand_schedule()
    t = rm.reference_trajectory(make(W_BASE, repair_model="ard1", anchor_mu_0=0.05), x, m, r)
    close(t["mu"][0, 0], [0.1, 0.3, 0.175, 0.175, 0.05])
    close(t["g_mu"][0, 0], [0.05, 0.05, 0.175, 0.175, 0.05])
    close(t["v"][0, 0], [0.0004, 0.0008, 0.0008, 0.0008, 0.0001])
    assert np.isnan(t["g_v"]).all()


def test_trajectory_uses_the_phase_lookup_of_the_increments():
    # H = [2, 2]; mu operating [0.2, 0.3], transitory [0.05, 0.06]; train 1 flies days 2-4.
    # Train 1 (no intervention): mu_0 0.1; day 2 (k <= H1, transitory index 1): + 0.06 = 0.16;
    # day 3 (operating index 0): + 0.2 = 0.36; day 4 (operating index 1): + 0.3 = 0.66. v = 0.
    cfg = make(IG_BASE, mu=[[[0.2, 0.3]]], mu_trans=[[[0.05, 0.06]]])
    x, m, r = hand_schedule()
    t = rm.reference_trajectory(cfg, x, m, r)
    close(t["mu"][1, 0], [0.1, 0.1, 0.16, 0.36, 0.66])


# ---------------------------------------------------------------------------
# 3. reliability_slack, Gaussian and Wiener rows (spec Eq. g_row, a_tangent, a_pwl_*)
# ---------------------------------------------------------------------------
# tau = 1, eps = 0.05: c2 = 1/kappa_G^2 = 0.3696115. f(mu) = c2 (1 - mu)^2. At mu = 0.3:
# f = 0.49 * 0.3696115 = 0.1811096 (exact cap).
def test_slack_gaussian_exact_inside_and_outside():
    cfg = make(G_BASE)
    # inside: v = 0.15 -> slack 0.1811096 - 0.15 = 0.0311096; outside: v = 0.2 -> 0.1811096 - 0.2 = -0.0188904.
    close(rm.reliability_slack(cfg, 0, 0, {"mu": 0.3, "v": 0.15}), 0.0311096, abs_=1e-7)
    close(rm.reliability_slack(cfg, 0, 0, {"mu": 0.3, "v": 0.2}), -0.0188904, abs_=1e-7)
    # on the boundary the slack is 0
    close(rm.reliability_slack(cfg, 0, 0, {"mu": 0.3, "v": 0.49 / 1.6448536269514722 ** 2}), 0.0,
          abs_=1e-12)


def test_slack_gaussian_side_row():
    # mu = 1.05 > tau: the exact row alone would hold (c2 * 0.05^2 = 0.00092 > v = 0) but the side
    # row mu <= tau has slack 1 - 1.05 = -0.05, the smaller of the two.
    close(rm.reliability_slack(make(G_BASE), 0, 0, {"mu": 1.05, "v": 0.0}), -0.05)


def test_slack_gaussian_tangent():
    # tangent_ref 0.5, mu_r = 0.5: f(0.5) = 0.25 c2 = 0.0924029, f'(0.5) = -2 c2 (0.5) = -0.3696115.
    # Cap at mu = 0.3: 0.0924029 + 0.3696115 * 0.2 = 0.1663252 (below the exact 0.1811096).
    cfg = make(G_BASE, reliability_impl="tangent", tangent_ref=0.5)
    close(rm.reliability_slack(cfg, 0, 0, {"mu": 0.3, "v": 0.15}), 0.1663252 - 0.15, abs_=1e-7)
    close(rm.reliability_slack(cfg, 0, 0, {"mu": 0.3, "v": 0.17}), 0.1663252 - 0.17, abs_=1e-7)
    # a point inside the exact row and outside the tangent row: v = 0.17 < 0.1811096 but > 0.1663252
    exact = rm.reliability_slack(make(G_BASE), 0, 0, {"mu": 0.3, "v": 0.17})
    assert exact > 0 > rm.reliability_slack(cfg, 0, 0, {"mu": 0.3, "v": 0.17})


def test_slack_gaussian_tangent_default_reference_is_one_half():
    # tangent_ref default 0.5 (spec 394): the same cap as above.
    cfg = make(G_BASE, reliability_impl="tangent")
    close(rm.reliability_slack(cfg, 0, 0, {"mu": 0.3, "v": 0.15}), 0.1663252 - 0.15, abs_=1e-7)


def test_slack_gaussian_tangent_other_reference():
    # tangent_ref 0.2: mu_r = 0.2: f(0.2) = 0.64 c2 = 0.2365514; f'(0.2) = -2 c2 0.8 = -0.5913784.
    # Cap at mu = 0.3: 0.2365514 - 0.5913784 * 0.1 = 0.1774136.
    cfg = make(G_BASE, reliability_impl="tangent", tangent_ref=0.2)
    close(rm.reliability_slack(cfg, 0, 0, {"mu": 0.3, "v": 0.15}), 0.1774136 - 0.15, abs_=1e-7)


def test_slack_gaussian_pwl():
    # P = 4: segments [0,.25], [.25,.5], [.5,.75], [.75,1], midpoints .125, .375, .625, .875.
    # The tangent of c2 (tau - mu)^2 at m is f(mu) - c2 (mu - m)^2. At mu = 0.3 only segment 2 applies:
    # cap = 0.1811096 - 0.3696115 * 0.075^2 = 0.1811096 - 0.0020791 = 0.1790306.
    cfg = make(G_BASE, reliability_impl="pwl", pwl_points=4)
    close(rm.reliability_slack(cfg, 0, 0, {"mu": 0.3, "v": 0.15}), 0.1790306 - 0.15, abs_=1e-7)
    close(rm.reliability_slack(cfg, 0, 0, {"mu": 0.3, "v": 0.18}), 0.1790306 - 0.18, abs_=1e-7)


def test_slack_gaussian_pwl_at_a_segment_boundary_takes_the_larger_cap():
    # mu = 0.25 belongs to segments 1 and 2 (spec Eq. a_pwl_seg has closed segments). Both midpoints are
    # 0.125 away, so both caps are f(0.25) - c2 * 0.125^2 = 0.3696115 * (0.5625 - 0.015625) = 0.2021313.
    # The larger of two equal caps is that value.
    cfg = make(G_BASE, reliability_impl="pwl", pwl_points=4)
    close(rm.reliability_slack(cfg, 0, 0, {"mu": 0.25, "v": 0.15}), 0.2021313 - 0.15, abs_=1e-7)


def test_slack_gaussian_pwl_one_segment_is_the_tangent_at_half_tau():
    # P = 1: the single tangent at tau / 2 (spec 915-916): the same cap 0.1663252 at mu = 0.3.
    cfg = make(G_BASE, reliability_impl="pwl", pwl_points=1)
    close(rm.reliability_slack(cfg, 0, 0, {"mu": 0.3, "v": 0.15}), 0.1663252 - 0.15, abs_=1e-7)


def test_slack_gaussian_pwl_outside_every_segment():
    # mu > tau lies in no segment: the row cannot hold (spec Eq. a_pwl_seg).
    cfg = make(G_BASE, reliability_impl="pwl", pwl_points=4)
    assert rm.reliability_slack(cfg, 0, 0, {"mu": 1.01, "v": 0.0}) < 0


def test_slack_gaussian_with_user_variance_bound():
    # v_max_user = 0.1 (spec 1030-1031): at mu = 0.3, v = 0.15 the exact row has slack +0.0311096
    # but v <= 0.1 has slack 0.1 - 0.15 = -0.05; at v = 0.05 the slacks are 0.1311096 and 0.05.
    cfg = make(G_BASE, v_max_user=0.1)
    close(rm.reliability_slack(cfg, 0, 0, {"mu": 0.3, "v": 0.15}), -0.05)
    close(rm.reliability_slack(cfg, 0, 0, {"mu": 0.3, "v": 0.05}), 0.05)


def test_slack_wiener_is_the_gaussian_row():
    # spec 1195-1199: same row, same forms, same optional row.
    for changes, state, expected in (
            ({}, {"mu": 0.3, "v": 0.15}, 0.0311096),
            ({"reliability_impl": "tangent"}, {"mu": 0.3, "v": 0.15}, 0.1663252 - 0.15),
            ({"reliability_impl": "pwl", "pwl_points": 4}, {"mu": 0.3, "v": 0.15}, 0.1790306 - 0.15),
            ({"v_max_user": 0.1}, {"mu": 0.3, "v": 0.15}, -0.05)):
        close(rm.reliability_slack(make(W_BASE, **changes), 0, 0, state), expected, abs_=1e-7)


def test_slack_inverse_gaussian_is_a_linear_row_in_exact_form():
    # mu <= mu_bar (spec Eq. ig_row); reliability_impl is ignored (spec 1100-1101).
    cap = rm.constants(make(IG_BASE))[(0, 0)]["mu_bar"]
    for impl in ("exact", "tangent", "pwl"):
        cfg = make(IG_BASE, reliability_impl=impl)
        close(rm.reliability_slack(cfg, 0, 0, {"mu": 0.1}), cap - 0.1)
        assert rm.reliability_slack(cfg, 0, 0, {"mu": cap + 0.01}) < 0


def test_slack_bernstein_with_positive_b_uses_the_shifted_quadratic():
    # eps = 0.1, tau = 1, b = 0.3: kappa = ln 10, c = 0.2302585; f(mu) = ((1-mu)^2 - 2 c (1-mu)) / (2 kappa)
    # (spec table 1543). At mu = 0.4: (0.36 - 2 * 0.2302585 * 0.6) / 4.6051702 = 0.0836 / 4.6051702.. computed:
    # 0.36 - 0.2763102 = 0.0836898; / 4.6051702 = 0.0181730.
    # quadratic_slack with c2 = 1/(2 kappa), c1 = -b/3 must give the same value.
    const = rm.constants(make(RF["bernstein"]))[(0, 0)]
    slack = rm.quadratic_slack(const["c2"], const["c1"], const["side"], const["tau"],
                               chi=0.01, mu=0.4, impl="exact")
    close(slack, 0.0181730 - 0.01, abs_=1e-7)
    # the side row mu <= tau - c = 0.7697415 binds: at mu = 0.8 the slack is 0.7697415 - 0.8 = -0.0302585
    slack = rm.quadratic_slack(const["c2"], const["c1"], const["side"], const["tau"],
                               chi=0.0, mu=0.8, impl="exact")
    close(slack, 0.7697415 - 0.8, abs_=1e-7)


# ---------------------------------------------------------------------------
# 4. Property: each linear form implies the exact row (spec 921-922)
# ---------------------------------------------------------------------------
def _form_cap(const, impl, mu, tangent_ref=0.5, P=8):
    """The largest chi that the form allows at mu (the side row left out)."""
    return rm.quadratic_slack(const["c2"], const["c1"], math.inf, const["tau"], chi=0.0, mu=mu,
                              impl=impl, tangent_ref=tangent_ref, P=P)


def _quadratic_cells():
    cfgs = {"G": make(G_BASE), "W": make(W_BASE)}
    for bound in ("cantelli", "hoeffding", "bernstein"):
        cfgs[bound] = make(RF[bound])
    return {name: rm.constants(cfg)[(0, 0)] for name, cfg in cfgs.items()}


def test_linear_forms_imply_the_exact_row_at_random_points():
    rng = np.random.default_rng(20261107)
    forms = [("tangent", {"tangent_ref": 0.5}), ("tangent", {"tangent_ref": 0.1}),
             ("tangent", {"tangent_ref": 0.7}), ("pwl", {"P": 1}), ("pwl", {"P": 3}),
             ("pwl", {"P": 8})]
    for name, const in _quadratic_cells().items():
        tau, side = const["tau"], const["side"]
        for impl, kw in forms:
            checked = 0
            for _ in range(1000):
                mu = rng.uniform(0.0, side)
                cap = _form_cap(const, impl, mu, **kw)
                if cap < 0:
                    continue                     # no chi >= 0 satisfies the form
                chi = rng.uniform(0.0, cap)
                form_slack = rm.quadratic_slack(const["c2"], const["c1"], side, tau, chi, mu,
                                                impl=impl, **kw)
                exact_slack = rm.quadratic_slack(const["c2"], const["c1"], side, tau, chi, mu,
                                                 impl="exact")
                assert form_slack >= -1e-12, (name, impl, kw, mu, chi)
                assert exact_slack >= -1e-12, (name, impl, kw, mu, chi)
                checked += 1
            assert checked > 100, (name, impl, kw)


def test_linear_form_caps_never_exceed_the_exact_cap():
    # The tangent of the convex quadratic lies below it everywhere on [0, tau] (c2 > 0).
    rng = np.random.default_rng(7)
    for name, const in _quadratic_cells().items():
        for mu in rng.uniform(0.0, const["tau"], 300):
            exact = _form_cap(const, "exact", mu)
            for impl, kw in (("tangent", {"tangent_ref": 0.3}), ("pwl", {"P": 5})):
                assert _form_cap(const, impl, mu, **kw) <= exact + 1e-12, (name, impl, mu)


# ---------------------------------------------------------------------------
# 5. minimal_z, u_value, costs (spec Eq. a_z, c_u, a_daycost, a_phases, a_objective, d_projected)
# ---------------------------------------------------------------------------
def test_minimal_z_ardinf():
    # Train 0 repairs on day 2: z = max(0, rho (mu_1 - 0)) = 0.5 * 0.3 = 0.15. Every other day 0
    # (the replacement day 4 included).
    x, m, r = hand_schedule()
    cfg = make(G_BASE)
    z = rm.minimal_z(cfg, rm.reference_trajectory(cfg, x, m, r), m)
    close(z[0, 0], [0.0, 0.15, 0.0, 0.0])
    close(z[1, 0], [0.0, 0.0, 0.0, 0.0])


def test_minimal_z_ard1():
    # anchor_mu_0 = 0.05: z = rho (mu_1 - g_1) = 0.5 * (0.3 - 0.05) = 0.125.
    x, m, r = hand_schedule()
    cfg = make(G_BASE, repair_model="ard1", **ANCHORS)
    z = rm.minimal_z(cfg, rm.reference_trajectory(cfg, x, m, r), m)
    close(z[0, 0], [0.0, 0.125, 0.0, 0.0])


def test_minimal_z_is_nan_where_the_repair_cost_is_zero_and_never_negative():
    x, m, r = hand_schedule()
    cfg = make(G_BASE, C_R=0.0)
    z = rm.minimal_z(cfg, rm.reference_trajectory(cfg, x, m, r), m)
    assert np.isnan(z).all()
    # a repair on a day when the anchor lies above the mean would give a negative value: z is clipped at 0.
    # Here: ARD1, anchor 0.1 = mu_0, repair on day 1 (before any mission): rho (0.1 - 0.1) = 0.
    x2 = np.zeros((2, 2, 4), dtype=int)
    x2[1, 1, :] = 1
    x2[0, 0, 0] = 1
    m2 = np.zeros((2, 1, 4), dtype=int)
    m2[0, 0, 0] = 1
    cfg = make(G_BASE, repair_model="ard1")
    z = rm.minimal_z(cfg, rm.reference_trajectory(cfg, x2, m2, None), m2)
    close(z[0, 0, 0], 0.0)


def test_u_value_inf_norm_and_quadratic():
    # Reference mu, train 0: 0.3, 0.15, 0.15, 0.05; train 1: 0.1, 0.3, 0.5, 0.7 (days 1..4).
    # inf_norm: max over trains and days of the sum over components (L = 1) = 0.7.
    # quadratic: 0.7^2 = 0.49.
    x, m, r = hand_schedule()
    cfg = make(G_BASE)
    t = rm.reference_trajectory(cfg, x, m, r)
    close(rm.u_value(cfg, t["mu"]), 0.7)
    close(rm.u_value(cfg, t["mu"][..., 1:]), 0.7)
    close(rm.u_value(make(G_BASE, penalty_type="quadratic"), t["mu"]), 0.49)


def test_u_value_sums_over_components():
    # Two components on one train: mu = 0.3 and 0.4 on a day -> inf_norm sum 0.7, quadratic 0.09 + 0.16 = 0.25.
    cfg = make(dict(G_BASE, model=["gaussian", "gaussian"]), L=2)
    mu = np.zeros((2, 2, 4))
    mu[0, 0, 2], mu[0, 1, 2] = 0.3, 0.4
    close(rm.u_value(cfg, mu), 0.7)
    close(rm.u_value(make(dict(G_BASE, model=["gaussian", "gaussian"]), L=2,
                          penalty_type="quadratic"), mu), 0.25)


def _hand_costs(**changes):
    cfg = make(G_BASE, repair_model="ard1", **dict(ANCHORS, **changes))
    x, m, r = hand_schedule()
    t = rm.reference_trajectory(cfg, x, m, r)
    z = rm.minimal_z(cfg, t, m)
    u = rm.u_value(cfg, t["mu"])
    return rm.costs(cfg, x, r, z, u)


def test_costs_total_mode():
    # C_M = 2, C_R = 10, C_rep = 7, C_D = 3; ARD1 with anchors: z[0, day 2] = 0.125; u = 0.7 (inf_norm).
    # Depot days: train 0 on days 2 and 4. Day costs (Eq. a_daycost):
    #   c1 = 0; c2 = 2 + 10 * 0.125 = 3.25; c3 = 0; c4 = 2 + 7 = 9.
    # H1 = H2 = 2: J_init = c1 + c2 = 3.25, J_op = c3 + c4 = 9, J_op/H2 = 4.5.
    # total: 3.25 + 9 + 3 * 0.7 = 14.35.
    out = _hand_costs()
    close(out["step_costs"], [0.0, 3.25, 0.0, 9.0])
    close(out["J_initialization"], 3.25)
    close(out["J_op"], 9.0)
    close(out["J_op_average"], 4.5)
    close(out["peak_damage"], 0.7)
    close(out["damage_penalty"], 2.1)
    close(out["J_total"], 14.35)
    close(out["operating_objective"], 4.5 + 2.1)
    assert out["projected_evaluation_cost"] is None and out["evaluation_objective"] is None
    close(out["objective"], 14.35)


def test_costs_operating_average_mode():
    # operating_average: J_op / H2 + C_D u = 4.5 + 2.1 = 6.6.
    close(_hand_costs(objective_mode="operating_average")["objective"], 6.6)


def test_costs_evaluation_total_mode():
    # E = 6, H1 = 2, H2 = 2: P_E = J_init + (E - H1) J_op / H2 = 3.25 + 4 * 4.5 = 21.25 (Eq. d_projected);
    # the objective J_init + (E - H1)/H2 J_op + C_D u = 21.25 + 2.1 = 23.35.
    out = _hand_costs(objective_mode="evaluation_total", evaluation_horizon=6)
    close(out["projected_evaluation_cost"], 21.25)
    close(out["evaluation_objective"], 23.35)
    close(out["objective"], 23.35)
    close(out["J_total"], 14.35)


def test_costs_quadratic_penalty_and_zero_penalty_weight():
    # quadratic penalty: u = 0.49, C_D u = 1.47: total 3.25 + 9 + 1.47 = 13.72.
    close(_hand_costs(penalty_type="quadratic")["objective"], 13.72)
    # C_D = 0: the term is absent (spec 683-684) but u is still reported.
    out = _hand_costs(C_D=0.0)
    close(out["damage_penalty"], 0.0)
    close(out["objective"], 12.25)
    close(out["peak_damage"], 0.7)


def test_costs_nan_z_counts_as_zero_and_no_replacement_cost_when_not_allowed():
    # C_R = 0 -> z is NaN -> counts 0 (spec 1777): c2 = 2 only; total 2 + 9 + 2.1 = 13.1.
    cfg = make(G_BASE, C_R=0.0)
    x, m, r = hand_schedule()
    t = rm.reference_trajectory(cfg, x, m, r)
    z = rm.minimal_z(cfg, t, m)
    close(rm.costs(cfg, x, r, z, rm.u_value(cfg, t["mu"]))["J_total"], 13.1)
    # allow_replacement false: no replacement cost whatever r says (spec 669-670): c4 = 2 only.
    cfg = make(dict(G_BASE, C_R=0.0), allow_replacement=False)
    t = rm.reference_trajectory(cfg, x, m, None)
    z = rm.minimal_z(cfg, t, m)
    out = rm.costs(cfg, x, r, z, rm.u_value(cfg, t["mu"]))
    close(out["step_costs"], [0.0, 2.0, 0.0, 2.0])


# ---------------------------------------------------------------------------
# 6. Independent inverse Gaussian cap against the package (spec 1094-1119)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("eta, eps", [(0.5, 0.05), (2.0, 0.1), (0.1, 0.01), (1.0, 0.3)])
def test_ig_cap_matches_the_package(eta, eps):
    # mu_bar = tau / E[Z | Z > q], Z ~ IG(1, 1/eta). The oracle integrates z f(z) over (q, inf) with scipy;
    # the package solves a log-CDF equation. They must agree within 1e-6 relative (tau = 1 and tau = 2.5).
    for tau in (1.0, 2.5):
        close(rm.ig_cap(eta, tau, eps), ig_mean_cap(eta, tau, eps), rel=1e-6)


def test_ig_cap_small_eta_limit_by_hand():
    # eta -> 0: Z -> 1 + sqrt(eta) N(0, 1). With eta = 1e-4 and eps = 0.1:
    # q = 1 + 1.2816 * 0.01; E[Z | Z > q] = 1 + 0.01 * phi(1.2816) / 0.1 = 1 + 0.01 * 1.7550 = 1.01755;
    # mu_bar = 1 / 1.01755 = 0.98275 (tau = 1). The normal approximation is exact to about 1e-3 here.
    close(rm.ig_cap(1e-4, 1.0, 0.1), 0.98275, rel=2e-3)


def test_ig_cap_is_below_tau_and_scales_with_tau():
    # E[Z | Z > q] > 1 for every eps < 1 (it exceeds the mean), so mu_bar < tau; the cap is linear in tau.
    for eta, eps in ((0.5, 0.05), (2.0, 0.4)):
        cap = rm.ig_cap(eta, 1.0, eps)
        assert 0.0 < cap < 1.0
        close(rm.ig_cap(eta, 3.0, eps), 3.0 * cap, rel=1e-9)


# ---------------------------------------------------------------------------
# 7. enumerate_schedules (spec 704-732)
# ---------------------------------------------------------------------------
def _enum_cfg(L=1, **changes):
    base = dict(G_BASE, H=[1, 1])
    if L == 2:
        base["model"] = ["gaussian", "gaussian"]
        return make(base, L=2, **changes)
    return make(base, **changes)


def test_enumerate_schedules_count_with_and_without_replacement():
    # F = 2, M = 1, L = 1, T = 2. Per day: 2 choices of the flying train; the other train is idle, in
    # the depot without intervention, repaired, or (with replacement) replaced: 4 choices (3 without).
    # With replacement: (2 * 4)^2 = 64 schedules; without: (2 * 3)^2 = 36.
    assert sum(1 for _ in rm.enumerate_schedules(_enum_cfg())) == 64
    assert sum(1 for _ in rm.enumerate_schedules(_enum_cfg(allow_replacement=False))) == 36


def test_enumerate_schedules_count_with_two_components():
    # L = 2, T = 1 (H = [1, 1] gives T = 2; use T = 2 below): per day 2 * (1 idle + 3^2 depot states) = 20
    # with replacement, 2 * (1 + 2^2) = 10 without. T = 2: 20^2 = 400 and 10^2 = 100.
    assert sum(1 for _ in rm.enumerate_schedules(_enum_cfg(L=2))) == 400
    assert sum(1 for _ in rm.enumerate_schedules(_enum_cfg(L=2, allow_replacement=False))) == 100


def test_enumerate_schedules_satisfy_the_assignment_rows_and_are_distinct():
    cfg = _enum_cfg(L=2)
    seen = set()
    for x, m, r in rm.enumerate_schedules(cfg):
        assert x.shape == (2, 2, 2) and m.shape == (2, 2, 2) and r.shape == (2, 2, 2)
        assert (x.sum(axis=1) <= 1).all()                              # Eq. a_one_activity
        assert (x[:, 1:, :].sum(axis=0) == 1).all()                    # Eq. a_coverage
        assert (m + r <= x[:, 0:1, :]).all()                           # Eq. a_depot
        seen.add((x.tobytes(), m.tobytes(), r.tobytes()))
    assert len(seen) == 400


def test_enumerate_schedules_lists_idle_and_depot_without_intervention_apart():
    # T = 2 schedule where train 1 is idle both days vs one where it is in the depot without
    # intervention both days: both exist and differ only in x[1, 0, :].
    cfg = _enum_cfg()
    found = {"idle": False, "depot": False}
    for x, m, r in rm.enumerate_schedules(cfg):
        if x[0, 1, :].all() and not m.any() and not r.any():
            if not x[1, 0, :].any():
                found["idle"] = True
            if x[1, 0, :].all():
                found["depot"] = True
    assert found == {"idle": True, "depot": True}


def test_enumerate_schedules_two_missions():
    # F = 3, M = 2, L = 1, T = 2 (H = [1, 1]): per day, the ordered pairs of distinct flying trains
    # 3 * 2 = 6, the remaining train has 4 states: 24 per day with replacement, 18 without.
    cfg = make(dict(G_BASE, F=3, M=2, H=[1, 1]))
    assert sum(1 for _ in rm.enumerate_schedules(cfg)) == 24 ** 2


# ===========================================================================
# Task O2: Gamma, rainflow bounds, row_violations, repeated_safety, best_reference
# ===========================================================================
# Gamma cell (spec 1212-1315): tau 1, eps 0.05, one rate beta = 4 for every mission; the replacement
# draw is the known constant 0.05 (no gamma_beta_new).
GA_BASE = dict(F=2, M=1, H=[2, 2], model="gamma", tau=1.0, epsilon=0.05, rho=0.5, mu_0=0.1,
               mu=0.2, gamma_beta=[4.0], C_M=2.0, C_R=10.0, C_rep=7.0, C_D=3.0,
               replacement_mu=0.05)

# Rainflow cells for the trajectories of hand_schedule(): tau 1, eps 0.1, rho 0.5, mu_0 0.1, mu inc 0.2,
# mu_new 0.05; each bound adds its own quantity (spec 1347-1371).
_RF_TR = dict(F=2, M=1, H=[2, 2], model="rainflow", tau=1.0, epsilon=0.1, rho=0.5, mu_0=0.1,
              mu=0.2, C_M=1.0, C_R=1.0, C_rep=1.0, C_D=0.0, replacement_mu=0.05)
RF_TR = {
    "markov": dict(_RF_TR, bound_method="markov"),
    # Cantelli: v inc 0.01, v_0 0.004, v_new 0.001.
    "cantelli": dict(_RF_TR, bound_method="cantelli", v=0.01, v_0=0.004, replacement_v=0.001),
    # Hoeffding: width 0.3 per mission, so delta R = 0.3^2 = 0.09; R_0 0.01, R_new 0.002.
    "hoeffding": dict(_RF_TR, bound_method="hoeffding", support=0.3, R_0=0.01, R_new=0.002),
    # Bernstein: as Cantelli; v_0 <= b0^2/4 needs support_0 >= 0.126 (0.2), v_new needs support_new >= 0.063 (0.1).
    "bernstein": dict(_RF_TR, bound_method="bernstein", v=0.01, support=0.3, v_0=0.004,
                      support_0=0.2, replacement_v=0.001, support_new=0.1),
    # Chernoff: cgf 1.0 per mission (a constant 0.2 at s = 5: K = s * 0.2), K_0 = s mu_0 = 0.5,
    # K_new = s mu_new = 0.25 (defaults, spec 1391-1392).
    "chernoff": dict(_RF_TR, bound_method="chernoff", cgf=1.0, s_chernoff=5.0),
}
RF_TR_ANCHORS = {
    "markov": dict(anchor_mu_0=0.05), "cantelli": dict(anchor_mu_0=0.05, anchor_v_0=0.002),
    "hoeffding": dict(anchor_mu_0=0.05, anchor_R_0=0.005),
    "bernstein": dict(anchor_mu_0=0.05, anchor_v_0=0.002),
    "chernoff": dict(anchor_mu_0=0.05, anchor_K_0=0.25),
}
_ALL_QUANTITIES = ("mu", "v", "R", "K")
# the schedule of hand_schedule() for train 0 (mission, repair, idle, replacement) and train 1 (idle, mission x3)
MU_0_ARDINF = [0.1, 0.3, 0.15, 0.15, 0.05]
MU_1 = [0.1, 0.1, 0.3, 0.5, 0.7]
MU_0_ARD1 = [0.1, 0.3, 0.175, 0.175, 0.05]
MU_G_ARD1 = [0.05, 0.05, 0.175, 0.175, 0.05]


def _trajectory(base, law, anchors):
    changes = {"repair_model": law}
    if law == "ard1":
        changes.update(anchors)
    x, m, r = hand_schedule()
    return rm.reference_trajectory(make(base, **changes), x, m, r)


def _check_quantity(t, q, train0, train1, anchor0=None):
    """The reference values of ``q`` for the two trains; ``anchor0`` is the anchor of train 0 (ard1) or None."""
    close(t[q][0, 0], train0)
    close(t[q][1, 0], train1)
    if anchor0 is None:
        assert np.isnan(t["g_" + q]).all(), q
    else:
        close(t["g_" + q][0, 0], anchor0)
        close(t["g_" + q][1, 0], [anchor0[0]] * 5)       # train 1 has no intervention: the starting anchor stays


def _check_untracked(t, tracked):
    for q in _ALL_QUANTITIES:
        if q not in tracked:
            assert np.isnan(t[q]).all() and np.isnan(t["g_" + q]).all(), q


# ---------------------------------------------------------------------------
# 2. reference_trajectory: Gamma and rainflow (spec Eqs. gm_map, rf_*; 1381-1396)
# Schedule (hand_schedule): train 0 mission (day 1), repair (2), idle (3), replacement (4); train 1 idle
# on day 1 and flies days 2-4. rho = 0.5, so the factors are 1 - rho = 0.5 and (1-rho)^2 = 0.25.
# Mean of train 0, ardinf: 0.1 + 0.2 = 0.3; 0.5 * 0.3 = 0.15; 0.15; mu_new = 0.05. Train 1: 0.1 + 0.2 n.
# Mean, ard1 with anchor_mu_0 = 0.05: day 2 = 0.5 * 0.3 + 0.5 * 0.05 = 0.175, anchor 0.175 from day 2,
# the replacement sets both to 0.05.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["gamma", "markov"])
def test_trajectory_mean_only_cells_gamma_and_markov(name):
    base = GA_BASE if name == "gamma" else RF_TR["markov"]
    t = _trajectory(base, "ardinf", {})
    _check_quantity(t, "mu", MU_0_ARDINF, MU_1)
    _check_untracked(t, ("mu",))
    t = _trajectory(base, "ard1", dict(anchor_mu_0=0.05))
    _check_quantity(t, "mu", MU_0_ARD1, MU_1, MU_G_ARD1)
    _check_untracked(t, ("mu",))


@pytest.mark.parametrize("bound", ["cantelli", "bernstein"])
def test_trajectory_rainflow_variance_bounds(bound):
    # v of train 0, ardinf: 0.004 + 0.01 = 0.014; 0.25 * 0.014 = 0.0035; 0.0035; v_new = 0.001.
    # Train 1: 0.004, 0.004, 0.014, 0.024, 0.034 (0.01 per mission).
    # ard1, anchor_v_0 = 0.002: day 2 = 0.25 * 0.014 + (1 - 0.25) * 0.002 = 0.0035 + 0.0015 = 0.005;
    # anchors 0.002, 0.002, 0.005, 0.005, 0.001.
    v1 = [0.004, 0.004, 0.014, 0.024, 0.034]
    t = _trajectory(RF_TR[bound], "ardinf", {})
    _check_quantity(t, "mu", MU_0_ARDINF, MU_1)
    _check_quantity(t, "v", [0.004, 0.014, 0.0035, 0.0035, 0.001], v1)
    _check_untracked(t, ("mu", "v"))
    t = _trajectory(RF_TR[bound], "ard1", RF_TR_ANCHORS[bound])
    _check_quantity(t, "mu", MU_0_ARD1, MU_1, MU_G_ARD1)
    _check_quantity(t, "v", [0.004, 0.014, 0.005, 0.005, 0.001], v1,
                    [0.002, 0.002, 0.005, 0.005, 0.001])


def test_trajectory_rainflow_hoeffding_budget():
    # delta R = 0.3^2 = 0.09 per mission. R of train 0, ardinf: 0.01 + 0.09 = 0.1; 0.25 * 0.1 = 0.025; 0.025;
    # R_new = 0.002. Train 1: 0.01, 0.01, 0.1, 0.19, 0.28.
    # ard1, anchor_R_0 = 0.005: day 2 = 0.25 * 0.1 + 0.75 * 0.005 = 0.025 + 0.00375 = 0.02875;
    # anchors 0.005, 0.005, 0.02875, 0.02875, 0.002.
    r1 = [0.01, 0.01, 0.1, 0.19, 0.28]
    t = _trajectory(RF_TR["hoeffding"], "ardinf", {})
    _check_quantity(t, "mu", MU_0_ARDINF, MU_1)
    _check_quantity(t, "R", [0.01, 0.1, 0.025, 0.025, 0.002], r1)
    _check_untracked(t, ("mu", "R"))
    t = _trajectory(RF_TR["hoeffding"], "ard1", RF_TR_ANCHORS["hoeffding"])
    _check_quantity(t, "mu", MU_0_ARD1, MU_1, MU_G_ARD1)
    _check_quantity(t, "R", [0.01, 0.1, 0.02875, 0.02875, 0.002], r1,
                    [0.005, 0.005, 0.02875, 0.02875, 0.002])


def test_trajectory_rainflow_chernoff_quantity_has_factor_one_minus_rho():
    # K: K_0 = 0.5, delta K = 1.0 per mission, factor 1 - rho = 0.5 (not (1-rho)^2), K_new = 0.25.
    # Train 0, ardinf: 0.5 + 1 = 1.5; 0.5 * 1.5 = 0.75; 0.75; 0.25. Train 1: 0.5, 0.5, 1.5, 2.5, 3.5.
    # ard1, anchor_K_0 = 0.25: day 2 = 0.5 * 1.5 + 0.5 * 0.25 = 0.875; anchors 0.25, 0.25, 0.875, 0.875, 0.25.
    k1 = [0.5, 0.5, 1.5, 2.5, 3.5]
    t = _trajectory(RF_TR["chernoff"], "ardinf", {})
    _check_quantity(t, "mu", MU_0_ARDINF, MU_1)
    _check_quantity(t, "K", [0.5, 1.5, 0.75, 0.75, 0.25], k1)
    _check_untracked(t, ("mu", "K"))
    t = _trajectory(RF_TR["chernoff"], "ard1", RF_TR_ANCHORS["chernoff"])
    _check_quantity(t, "mu", MU_0_ARD1, MU_1, MU_G_ARD1)
    _check_quantity(t, "K", [0.5, 1.5, 0.875, 0.875, 0.25], k1, [0.25, 0.25, 0.875, 0.875, 0.25])


def test_trajectory_rainflow_replacement_as_new_sets_every_quantity_to_zero():
    # replacement_as_new (spec 1393): day 4 gives 0 for mu and for the quantity of the bound.
    x, m, r = hand_schedule()
    for bound, q in (("cantelli", "v"), ("hoeffding", "R"), ("chernoff", "K")):
        base = dict(RF_TR[bound], replacement_mu=0.0)
        for key in ("replacement_v", "R_new"):
            base.pop(key, None)
        t = rm.reference_trajectory(make(base, replacement_as_new=True), x, m, r)
        close(t["mu"][0, 0, 4], 0.0)
        close(t[q][0, 0, 4], 0.0)


# ---------------------------------------------------------------------------
# 3. reliability_slack: rainflow rows and the Gamma row (spec 1490-1555, 1274-1310, 875-918)
# ---------------------------------------------------------------------------
def test_slack_markov_row():
    # mu <= eps tau = 0.1 (spec Eq. rf_markov): at mu 0.05 the slack is 0.05, at 0.15 it is -0.05.
    cfg = make(RF["markov"])
    close(rm.reliability_slack(cfg, 0, 0, {"mu": 0.05}), 0.05)
    close(rm.reliability_slack(cfg, 0, 0, {"mu": 0.15}), -0.05)


def test_slack_chernoff_row():
    # K - s tau <= ln eps (Eq. rf_chernoff), s = 5, tau = 1, eps = 0.1: K <= 5 - ln 10 = 2.6974149.
    # K = 2: slack 0.6974149; K = 3: -0.3025851. The row does not read mu.
    cfg = make(RF["chernoff"])
    close(rm.reliability_slack(cfg, 0, 0, {"mu": 0.9, "K": 2.0}), 0.6974149, abs_=1e-7)
    close(rm.reliability_slack(cfg, 0, 0, {"mu": 0.1, "K": 3.0}), -0.3025851, abs_=1e-7)


def test_slack_cantelli_forms():
    # eps = 0.1: f(mu) = (tau - mu)^2 / 9 (table 1540). At mu = 0.4, v = 0.03:
    #   exact:   f = 0.36 / 9 = 0.04, slack 0.01;
    #   tangent (mu_r = 0.5): f(0.5) = 0.25 / 9 = 0.0277778, f'(0.5) = -2 * 0.5 / 9 = -0.1111111,
    #            cap = 0.0277778 + 0.1111111 * 0.1 = 0.0388889, slack 0.0088889;
    #   pwl, P = 4: mu in segment 2 = [0.25, 0.5], midpoint 0.375; the tangent of c2 (tau - mu)^2 at m is
    #            f(mu) - c2 (mu - m)^2 = 0.04 - (1/9) 0.025^2 = 0.0399306, slack 0.0099306.
    state = {"mu": 0.4, "v": 0.03}
    close(rm.reliability_slack(make(RF["cantelli"]), 0, 0, state), 0.01, abs_=1e-9)
    close(rm.reliability_slack(make(RF["cantelli"], reliability_impl="tangent"), 0, 0, state),
          0.0088889, abs_=1e-7)
    close(rm.reliability_slack(make(RF["cantelli"], reliability_impl="pwl", pwl_points=4), 0, 0, state),
          0.0099306, abs_=1e-7)
    # outside: v = 0.05 -> exact slack -0.01
    close(rm.reliability_slack(make(RF["cantelli"]), 0, 0, {"mu": 0.4, "v": 0.05}), -0.01, abs_=1e-9)


def test_slack_cantelli_side_row_and_user_bound():
    # mu = 1.1 > tau: f(1.1) = 0.01 / 9 holds for v = 0 but the side row has slack 1 - 1.1 = -0.1.
    close(rm.reliability_slack(make(RF["cantelli"]), 0, 0, {"mu": 1.1, "v": 0.0}), -0.1)
    # v_max_user = 0.02 (spec 1554-1555): at mu = 0.4, v = 0.03 the row has slack 0.01, the user row -0.01.
    cfg = make(RF["cantelli"], v_max_user=0.02)
    close(rm.reliability_slack(cfg, 0, 0, {"mu": 0.4, "v": 0.03}), -0.01, abs_=1e-9)
    close(rm.reliability_slack(cfg, 0, 0, {"mu": 0.4, "v": 0.015}), 0.005, abs_=1e-9)


def test_slack_hoeffding_forms():
    # eps = 0.1, kappa = ln 10 = 2.3025851, f(mu) = 2 (tau - mu)^2 / kappa (table 1542). At mu = 0.4, R = 0.3:
    #   exact:   f = 0.72 / 2.3025851 = 0.3126920, slack 0.0126920;
    #   tangent (mu_r = 0.5): f(0.5) = 0.5 / kappa = 0.2171472, f'(0.5) = -2 (2/kappa) 0.5 = -0.8685890,
    #            cap = 0.2171472 + 0.0868589 = 0.3040061, slack 0.0040061;
    #   pwl, P = 2: mu in [0, 0.5], midpoint 0.25: cap = f(0.4) - (2/kappa) 0.15^2 = 0.3126920 - 0.0195432
    #            = 0.2931488, slack -0.0068512.
    state = {"mu": 0.4, "R": 0.3}
    close(rm.reliability_slack(make(RF["hoeffding"]), 0, 0, state), 0.0126920, abs_=1e-7)
    close(rm.reliability_slack(make(RF["hoeffding"], reliability_impl="tangent"), 0, 0, state),
          0.0040061, abs_=1e-7)
    close(rm.reliability_slack(make(RF["hoeffding"], reliability_impl="pwl", pwl_points=2), 0, 0, state),
          -0.0068512, abs_=1e-7)
    # side row: mu = 1.2 -> -0.2 (R = 0 satisfies the quadratic row)
    close(rm.reliability_slack(make(RF["hoeffding"]), 0, 0, {"mu": 1.2, "R": 0.0}), -0.2)


def test_slack_bernstein_forms_with_positive_b():
    # eps = 0.1, b = 0.3: kappa = 2.3025851, c = kappa b / 3 = 0.2302585, c2 = 1/(2 kappa) = 0.2171472,
    # c1 = -b/3 = -0.1, f(mu) = ((tau - mu)^2 - 2c (tau - mu)) / (2 kappa). At mu = 0.4, v = 0.01:
    #   exact:   (0.36 - 2 * 0.2302585 * 0.6) / 4.6051702 = 0.0836898 / 4.6051702 = 0.0181730, slack 0.0081730;
    #   tangent (mu_r = 0.5 < sqrt(tau (tau - 2c)) = 0.7345): f(0.5) = (0.25 - 0.2302585) / 4.6051702 =
    #            0.0042868, f'(0.5) = -2 c2 0.5 - c1 = -0.2171472 + 0.1 = -0.1171472,
    #            cap = 0.0042868 + 0.1171472 * 0.1 = 0.0160015, slack 0.0060015;
    #   pwl, P = 4: segment 2 (midpoint 0.375): cap = f(0.4) - c2 0.025^2 = 0.0181730 - 0.0001357 = 0.0180373,
    #            slack 0.0080373.
    state = {"mu": 0.4, "v": 0.01}
    close(rm.reliability_slack(make(RF["bernstein"]), 0, 0, state), 0.0081730, abs_=1e-7)
    close(rm.reliability_slack(make(RF["bernstein"], reliability_impl="tangent"), 0, 0, state),
          0.0060015, abs_=1e-7)
    close(rm.reliability_slack(make(RF["bernstein"], reliability_impl="pwl", pwl_points=4), 0, 0, state),
          0.0080373, abs_=1e-7)
    # side row mu <= tau - c = 0.7697415: at mu = 0.8 (v = 0) the slack is -0.0302585
    close(rm.reliability_slack(make(RF["bernstein"]), 0, 0, {"mu": 0.8, "v": 0.0}), -0.0302585, abs_=1e-7)
    # v_max_user = 0.005 binds at v = 0.01 (slack -0.005 against 0.0081730)
    cfg = make(RF["bernstein"], v_max_user=0.005)
    close(rm.reliability_slack(cfg, 0, 0, state), -0.005, abs_=1e-9)


def test_slack_gamma_row_is_a_linear_cap_in_exact_form():
    # beta* = 4 (one rate), t = beta* tau = 4, eps = 0.05: A_1(4, 0.05) = 1.5536699 solves Q(A, 4) = 0.05
    # (Q(1.5536699, 4) = 0.05, checked with special.gammaincc), cap = A_1 / beta* = 0.3884175.
    # mu = 0.3: slack 0.0884175; mu = 0.4: -0.0115825. reliability_impl is ignored (spec 1301-1302).
    close(special.gammaincc(1.5536699, 4.0), 0.05, rel=1e-6)
    for impl in ("exact", "tangent", "pwl"):
        cfg = make(GA_BASE, reliability_impl=impl)
        close(rm.reliability_slack(cfg, 0, 0, {"mu": 0.3}), 0.0884175, abs_=1e-7)
        close(rm.reliability_slack(cfg, 0, 0, {"mu": 0.4}), -0.0115825, abs_=1e-7)


def test_gamma_smallest_rate_cases():
    # beta* = min of the rates of the terms present (Eq. gm_betastar, 1274-1283).
    def beta_star(**changes):
        return rm.constants(make(GA_BASE, **changes))[(0, 0)]["beta_star"]
    assert beta_star() == 4.0                                   # only the mission rate
    assert beta_star(gamma_beta_0=2.0) == 2.0                   # mu_0 > 0 with a rate
    assert beta_star(gamma_beta_new=1.5) == 1.5                 # mu_new 0.05 > 0 with a rate
    assert beta_star(gamma_beta_new=1.5, allow_replacement=False) == 4.0   # beta_new only if allowed
    # a rate per day: rates [3, 5], both increments positive -> 3
    per_day = [[[[3.0, 5.0]]], [[[3.0, 5.0]]]]
    assert beta_star(gamma_beta=per_day, mu=[[[0.2, 0.3]]]) == 3.0
    # rates [5, 3] but the increment of day 2 is 0: its rate does not count -> 5
    per_day = [[[[5.0, 3.0]]], [[[5.0, 3.0]]]]
    cfg = make(GA_BASE, gamma_beta=per_day, mu=[[[0.2, 0.0]]])
    const = rm.constants(cfg)[(0, 0)]
    assert const["beta_star"] == 5.0
    close(const["mu_cap"], rm.gamma_cap(5.0, 0.05) / 5.0)
    close(const["mu_cap"], 0.4318558, abs_=1e-7)


def test_gamma_cell_without_any_rate_has_the_row_mu_below_tau():
    # No mission increment is positive, mu_0 and mu_new have no rate: no term has a rate, and the row is
    # mu <= tau (spec 1282-1283).
    cfg = make(GA_BASE, mu=0.0)
    const = rm.constants(cfg)[(0, 0)]
    assert const["beta_star"] is None
    close(const["mu_cap"], 1.0)
    close(rm.reliability_slack(cfg, 0, 0, {"mu": 0.7}), 0.3)


# ---------------------------------------------------------------------------
# 6. The independent Gamma cap against the package (spec 1287-1310)
# ---------------------------------------------------------------------------
def _w_dense(A, t):
    """w(A) = max over c in (0, 1] of Q(A/c, t/c), on a grid of 400001 values of log c in [log 1e-8, 0]."""
    c = np.exp(np.linspace(math.log(1e-8), 0.0, 400001))
    return float(special.gammaincc(A / c, t / c).max())


# Values of the oracle (scratch computation, brentq on w(A) - eps with a bounded search of w over log c):
#   (4, 0.05):  1.5536699  the maximum over c is at c = 1: A_1 = A^ with Q(A^, 4) = 0.05;
#   (10, 0.2):  7.7982968  likewise;
#   (0.3, 0.2): 0.1820018  below A^ = 0.2181945: the maximum over c is above Q at c = 1, A_1 < A^;
#   (1, 0.4):   0.9533342  below A^ = 1.0746803 (> t = 1): the same effect, and A^ >= t;
#   (3, 0.6):   3 (eps >= 1/2: every A < t qualifies, the supremum is t).
@pytest.mark.parametrize("t, eps, expected", [(4.0, 0.05, 1.5536699), (10.0, 0.2, 7.7982968),
                                              (0.3, 0.2, 0.1820018), (1.0, 0.4, 0.9533342)])
def test_gamma_cap_oracle_and_package(t, eps, expected):
    a1 = rm.gamma_cap(t, eps)
    close(a1, expected, rel=1e-6)
    # the definition (spec 1293-1294): w(A_1) <= eps, and a slightly larger A has w > eps
    assert _w_dense(a1, t) <= eps * (1.0 + 1e-9)
    assert _w_dense(a1 * (1.0 + 1e-4), t) > eps
    # against the package (within 1e-6 relative), also through the cap of the mean (spec 1311)
    close(a1, sharp_cap(t, eps), rel=1e-6)
    close(a1 / (t / 1.0), mean_cap(t, 1.0, eps), rel=1e-6)       # beta* = t, tau = 1


def test_gamma_cap_binding_at_c_equal_one_is_the_tail_root():
    # when w(A^) = Q(A^, t), the cap solves Q(A, t) = eps: (4, 0.05) and (10, 0.2).
    for t, eps in ((4.0, 0.05), (10.0, 0.2)):
        close(special.gammaincc(rm.gamma_cap(t, eps), t), eps, rel=1e-9)
    # (0.3, 0.2) and (1, 0.4): the cap lies below the root, so Q(A_1, t) < eps
    for t, eps in ((0.3, 0.2), (1.0, 0.4)):
        assert special.gammaincc(rm.gamma_cap(t, eps), t) < eps * (1.0 - 1e-3)


def test_gamma_cap_for_eps_at_least_one_half_is_the_supremum_t():
    # spec 1294-1295, 1308: for eps >= 1/2 every A < t qualifies (w(A) < 1/2). The package returns a value
    # just below t (the lower end of a bisection of width 1e-9 t).
    for eps in (0.5, 0.6, 0.9):
        assert rm.gamma_cap(3.0, eps) == 3.0
        close(sharp_cap(3.0, eps), 3.0, rel=1e-6)
    assert _w_dense(3.0 * (1.0 - 1e-3), 3.0) < 0.5               # the claim w(A) < 1/2 below t


# ---------------------------------------------------------------------------
# row_violations (spec 698-1625). Hand solution: Gaussian cell, ard1, H = [1, 2] (T = 3), F = 2, M = 1.
# Day 1: train 0 flies, train 1 is repaired. Day 2: train 1 flies, train 0 is replaced.
# Day 3: train 0 flies, train 1 is replaced. Inputs as G_BASE with anchor_mu_0 0.05, anchor_v_0 0.0002.
# ---------------------------------------------------------------------------
G_RV = dict(G_BASE, H=[1, 2], repair_model="ard1", **ANCHORS)


def rv_schedule():
    F, M, L, T = 2, 1, 1, 3
    x = np.zeros((F, M + 1, T), dtype=int)
    m = np.zeros((F, L, T), dtype=int)
    r = np.zeros((F, L, T), dtype=int)
    x[0, 1, 0] = 1; x[1, 0, 0] = 1; m[1, 0, 0] = 1       # day 1: train 0 flies, train 1 depot + repair
    x[1, 1, 1] = 1; x[0, 0, 1] = 1; r[0, 0, 1] = 1       # day 2: train 1 flies, train 0 depot + replacement
    x[0, 1, 2] = 1; x[1, 0, 2] = 1; r[1, 0, 2] = 1       # day 3: train 0 flies, train 1 depot + replacement
    return x, m, r


def rv_solution(cfg, schedule=None):
    """The solution of the reference trajectory of a schedule with the smallest ``z`` and ``u`` (arrays for days 1..T)."""
    x, m, r = rv_schedule() if schedule is None else schedule
    t = rm.reference_trajectory(cfg, x, m, r)
    sol = {"x": x, "m": m, "r": r, "z": rm.minimal_z(cfg, t, m), "u": rm.u_value(cfg, t["mu"])}
    for name in ("mu", "v", "R", "K", "g_mu", "g_v", "g_R", "g_K"):
        sol[name] = t[name][..., 1:].copy()
    return sol


def test_row_violations_hand_solution_values_and_no_violation():
    # mu, train 0: 0.1 + 0.2 = 0.3; replaced 0.05; 0.05 + 0.2 = 0.25.
    # mu, train 1: repair on day 1: 0.5 * 0.1 + 0.5 * 0.05 = 0.075; flies 0.275; replaced 0.05.
    # v, train 0: 0.0004 + 0.001 = 0.0014; 0.0001; 0.0001 + 0.001 = 0.0011.
    # v, train 1: repair: 0.25 * 0.0004 + 0.75 * 0.0002 = 0.00025; flies 0.00125; replaced 0.0001.
    # Anchors mu: train 0: 0.05 (flying keeps it), replaced 0.05, 0.05; train 1: 0.075 after the repair, 0.075, 0.05.
    # Anchors v: train 0: 0.0002, 0.0001, 0.0001; train 1: 0.00025, 0.00025, 0.0001.
    # z, train 1 day 1: rho (mu_0 - g_0) = 0.5 (0.1 - 0.05) = 0.025. u = max mu = 0.3 (inf_norm).
    cfg = make(G_RV)
    sol = rv_solution(cfg)
    close(sol["mu"][0, 0], [0.3, 0.05, 0.25])
    close(sol["mu"][1, 0], [0.075, 0.275, 0.05])
    close(sol["v"][0, 0], [0.0014, 0.0001, 0.0011])
    close(sol["v"][1, 0], [0.00025, 0.00125, 0.0001])
    close(sol["g_mu"][0, 0], [0.05, 0.05, 0.05])
    close(sol["g_mu"][1, 0], [0.075, 0.075, 0.05])
    close(sol["g_v"][0, 0], [0.0002, 0.0001, 0.0001])
    close(sol["g_v"][1, 0], [0.00025, 0.00025, 0.0001])
    close(sol["z"][:, 0], [[0.0, 0.0, 0.0], [0.025, 0.0, 0.0]])
    close(sol["u"], 0.3)
    # loop rows (Eq. a_loop): 0.25 <= 0.3, 0.05 <= 0.075, v: 0.0011 <= 0.0014, 0.0001 <= 0.00025; anchors likewise
    assert rm.row_violations(cfg, sol) == []
    # the same schedule under ardinf has no violation either
    cfg0 = make(dict(G_RV, repair_model="ardinf"))
    assert rm.row_violations(cfg0, rv_solution(cfg0)) == []


def _edit_one_activity(s): s["x"][0, 0, 0] = 1                     # train 0: mission and depot on day 1
def _edit_coverage(s): s["x"][1, 1, 1] = 0                         # mission uncovered on day 2
def _edit_depot(s): s["x"][1, 0, 0] = 0                            # repair of train 1 without a depot day
def _edit_carry(s): s["mu"][0, 0, 0] = 0.29                        # day 1 flight needs mu >= 0.3
def _edit_repair(s):
    s["mu"][1, 0, 0] = 0.07                                        # repair row needs mu >= 0.075
    s["g_mu"][1, 0, 0:2] = 0.07                                    # anchors follow, so that no anchor row fails
def _edit_replace(s):
    s["mu"][0, 0, 1] = 0.04                                        # replacement value is 0.05
    s["g_mu"][0, 0, 1:3] = 0.04
def _edit_anchor_set(s): s["g_mu"][1, 0, 0:2] = 0.07               # anchor after the repair must equal mu = 0.075
def _edit_anchor_keep(s): s["g_mu"][0, 0, 0] = 0.06                # anchor must stay 0.05 on a mission day
def _edit_reliability(s): s["v"][0, 0, 0] = 0.2                    # exact cap at mu = 0.3 is 0.1811096
def _edit_loop(s):
    s["mu"][0, 0, 2] = 0.35                                        # mu_T = 0.35 > mu_{H1} = 0.3
    s["u"] = 0.35
def _edit_z_row(s): s["z"][1, 0, 0] = 0.02                         # z >= 0.025 on the repair day
def _edit_z_negative(s): s["z"][0, 0, 1] = -0.01                   # z >= 0 (the a_z row is slack there: -0.375)
def _edit_u_row(s): s["u"] = 0.29                                  # u >= max_i sum mu = 0.3


@pytest.mark.parametrize("edit, row, i, l, k", [
    (_edit_one_activity, "one_activity", 0, None, 1),
    (_edit_coverage, "coverage_j1", None, None, 2),
    (_edit_depot, "depot", 1, 0, 1),
    (_edit_carry, "mu_carry", 0, 0, 1),
    (_edit_repair, "mu_repair", 1, 0, 1),
    (_edit_replace, "mu_replace", 0, 0, 2),
    (_edit_anchor_set, "g_mu_set", 1, 0, 1),
    (_edit_anchor_keep, "g_mu_keep", 0, 0, 1),
    (_edit_reliability, "reliability", 0, 0, 1),
    (_edit_loop, "mu_loop", 0, 0, 3),
    (_edit_z_row, "z_row", 1, 0, 1),
    (_edit_z_negative, "z_nonneg", 0, 0, 2),
    (_edit_u_row, "u_row", 0, None, 1),
])
def test_row_violations_each_broken_row_is_reported_alone(edit, row, i, l, k):
    # every edit breaks exactly one row of the hand solution (the comment at each edit says which and why;
    # the one-sided dynamics rows accept a larger value, so the edits that lower a value fix the anchors too).
    cfg = make(G_RV)
    sol = rv_solution(cfg)
    edit(sol)
    found = rm.row_violations(cfg, sol)
    assert [(v.row, v.i, v.l, v.k) for v in found] == [(row, i, l, k)], found
    assert found[0].violation > 0


def test_row_violations_amounts_and_one_sided_rows():
    cfg = make(G_RV)
    sol = rv_solution(cfg)
    sol["mu"][0, 0, 0] = 0.29
    close(rm.row_violations(cfg, sol)[0].violation, 0.01, abs_=1e-12)       # 0.3 - 0.29
    # a larger value satisfies every one-sided row of the day (spec 783): mu = 0.31 on day 1 of train 0,
    # with u = 0.31 (the u row) and the loop row 0.25 <= 0.31 holding.
    sol = rv_solution(cfg)
    sol["mu"][0, 0, 0] = 0.31
    sol["u"] = 0.31
    assert rm.row_violations(cfg, sol) == []


def test_row_violations_tolerance_rules():
    cfg = make(G_RV)
    # mu_carry fails by 0.01: accepted with tol = 0.02, flagged with the default
    sol = rv_solution(cfg)
    sol["mu"][0, 0, 0] = 0.29
    assert rm.row_violations(cfg, sol, tol=0.02) == []
    assert len(rm.row_violations(cfg, sol)) == 1
    # a row with a big-M constant M_s = tau = 1 allows tol + 2e-5 M_s = 2.1e-5 by default:
    # a failure of 1e-5 is accepted, 3e-5 is not; with bigm_tol = 0 both are flagged (tol = 1e-6).
    sol = rv_solution(cfg)
    sol["mu"][0, 0, 0] = 0.3 - 1e-5
    assert rm.row_violations(cfg, sol) == []
    assert len(rm.row_violations(cfg, sol, bigm_tol=0.0)) == 1
    sol["mu"][0, 0, 0] = 0.3 - 3e-5
    assert len(rm.row_violations(cfg, sol)) == 1
    # a fleet row (no big-M constant) allows tol only: x = 1 + 5e-6 on a one-activity row fails by 5e-6 > 1e-6
    sol = rv_solution(cfg)
    sol["x"] = sol["x"].astype(float)
    sol["x"][0, 1, 0] = 1.0 + 5e-6
    assert any(v.row == "one_activity" for v in rm.row_violations(cfg, sol))


def test_row_violations_nan_counts_as_a_failure():
    cfg = make(G_RV)
    sol = rv_solution(cfg)
    sol["mu"][0, 0, 0] = np.nan
    assert {v.row for v in rm.row_violations(cfg, sol)} >= {"mu_carry", "reliability", "mu_loop"}


def test_row_violations_without_replacement_array_and_loop_rows():
    # allow_replacement false, no 'r' key. Train 0 flies on all 3 days with no intervention: mu = 0.1 + 0.2 n,
    # v = 0.0004 + 0.001 n, so mu_T = 0.7 > mu_{H1} = 0.3 and v_T = 0.0034 > 0.0014: two loop rows of train 0 fail;
    # train 1 idles (state constant, rows hold). Reliability holds on all 3 days (slack > 0).
    cfg = make(dict(G_BASE, H=[1, 2], allow_replacement=False))
    x = np.zeros((2, 2, 3), dtype=int)
    x[0, 1, :] = 1
    m = np.zeros((2, 1, 3), dtype=int)
    t = rm.reference_trajectory(cfg, x, m)
    sol = {"x": x, "m": m, "u": rm.u_value(cfg, t["mu"]), "z": rm.minimal_z(cfg, t, m)}
    for name in ("mu", "v", "R", "K", "g_mu", "g_v", "g_R", "g_K"):
        sol[name] = t[name][..., 1:]
    found = rm.row_violations(cfg, sol)
    assert sorted((v.row, v.i, v.k) for v in found) == [("mu_loop", 0, 3), ("v_loop", 0, 3)]
    close([v.violation for v in sorted(found, key=lambda v: v.row)], [0.4, 0.002], abs_=1e-9)


def test_row_violations_z_rows_in_mode_operating_average():
    # Eq. a_z is built only on operating days in the mode operating_average (spec 1564-1565). Day 1 (the only
    # transitory day) has the repair of train 1: z = 0 there breaks the row in mode total and not in the mode.
    cfg_total = make(G_RV)
    cfg_avg = make(G_RV, objective_mode="operating_average")
    sol = rv_solution(cfg_total)
    sol["z"][1, 0, 0] = 0.0
    assert [(v.row, v.k) for v in rm.row_violations(cfg_total, sol)] == [("z_row", 1)]
    assert rm.row_violations(cfg_avg, sol) == []
    # z >= 0 is built on the operating days in both modes. Day 3 of train 1 (a replacement): the a_z row there
    # is z >= 0.5 (0.275 - 0.075) - 0.5 = -0.4, so z = -0.3 satisfies it and fails only z >= 0.
    sol = rv_solution(cfg_avg)
    sol["z"][1, 0, 2] = -0.3
    assert [(v.row, v.k) for v in rm.row_violations(cfg_avg, sol)] == [("z_nonneg", 3)]


# every model, bound and law: the reference trajectory of the hand schedule satisfies every row, and the
# reliability row is reported when the state is pushed beyond it.
# Schedule of rv_schedule(): replacement values 0 (not given), so a replaced component restarts at 0 and
# each loop row holds (train 0: s_T = new + increment <= s_{H1} = s_0 + increment; train 1: s_T = new = 0).
_SC = dict(F=2, M=1, H=[1, 2], tau=1.0, rho=0.5, mu_0=0.1, mu=0.2, C_M=1.0, C_R=1.0, C_rep=1.0, C_D=0.0)
SAFE_CELLS = {
    "G": dict(_SC, model="gaussian", epsilon=0.05, v_0=0.0004, v=0.001),
    "IG": dict(_SC, model="inverse_gaussian", epsilon=0.05, eta=0.5),
    "W": dict(_SC, model="wiener", epsilon=0.05, sigma=0.02, v_0=0.0004),
    "Ga": dict(_SC, model="gamma", epsilon=0.05, gamma_beta=[4.0]),
    "Mk": dict(_SC, model="rainflow", bound_method="markov", epsilon=0.1, mu_0=0.02, mu=0.05),
    "Ca": dict(_SC, model="rainflow", bound_method="cantelli", epsilon=0.1, v=0.01, v_0=0.004),
    "Ho": dict(_SC, model="rainflow", bound_method="hoeffding", epsilon=0.1, support=0.3, R_0=0.01),
    "Be": dict(_SC, model="rainflow", bound_method="bernstein", epsilon=0.1, v=0.01, v_0=0.004,
               support=0.3, support_0=0.2),
    "Ch": dict(_SC, model="rainflow", bound_method="chernoff", epsilon=0.1, cgf=1.0, s_chernoff=5.0),
}


@pytest.mark.parametrize("kind", sorted(SAFE_CELLS))
@pytest.mark.parametrize("law", ["ardinf", "ard1"])
def test_row_violations_reference_solution_of_every_kind_has_no_violation(kind, law):
    cfg = make(SAFE_CELLS[kind], repair_model=law)
    assert rm.cell_kind(cfg, 0, 0)["kind"] == kind
    sol = rv_solution(cfg)
    assert rm.row_violations(cfg, sol) == []
    # the reliability row is the only one to fail when the state is pushed beyond it: mu = 0.99 for every kind
    # (beyond the Markov, Gamma and IG caps, with the quadratic rows' f(0.99) below the tracked v or R), and
    # K = 2.9 > s tau + ln eps = 2.6974 for the Chernoff cell (whose row does not read mu).
    if kind == "Ch":
        sol["K"][0, 0, 0] = 2.9
    else:
        sol["mu"][0, 0, 0] = 0.99
    assert [v.row for v in rm.row_violations(cfg, sol)] == ["reliability"]


# ---------------------------------------------------------------------------
# 8. repeated_safety (spec 1612-1617)
# ---------------------------------------------------------------------------
def test_repeated_safety_holds_for_a_schedule_whose_loop_rows_hold():
    # The hand schedule of rv_schedule() satisfies the loop rows (the previous tests). Then every repetition of
    # the operating phase is safe (spec 1614-1617): the slack never turns negative over 20 cycles.
    # The smallest slack is on day 1 of train 0: mu = 0.3, v = 0.0014: f(0.3) - v = 0.49 / kappa_G^2 - 0.0014
    #   = 0.49 * 0.3696115 - 0.0014 = 0.1811096 - 0.0014 = 0.1797096 (every later day has a larger slack).
    x, m, r = rv_schedule()
    for law in ("ardinf", "ard1"):
        cfg = make(G_RV, repair_model=law)
        assert rm.row_violations(cfg, rv_solution(cfg)) == []
        for cycles in (0, 1, 5, 20):
            close(rm.repeated_safety(cfg, x, m, r, cycles), 0.1797096, abs_=1e-7)
            assert rm.repeated_safety(cfg, x, m, r, cycles) >= -1e-9


@pytest.mark.parametrize("kind", sorted(SAFE_CELLS))
def test_repeated_safety_holds_for_every_kind_on_the_loop_safe_schedule(kind):
    x, m, r = rv_schedule()
    for law in ("ardinf", "ard1"):
        cfg = make(SAFE_CELLS[kind], repair_model=law)
        assert rm.repeated_safety(cfg, x, m, r, 20) >= -1e-9


def test_repeated_safety_detects_a_schedule_that_breaks_the_loop_rows():
    # Train 0 flies every day (the loop rows fail, see test_row_violations_without_replacement_array...).
    # After 3 days mu = 0.1 + 0.2 * 3 = 0.7, v = 0.0004 + 0.001 * 3 = 0.0034:
    #   slack = f(0.7) - v = 0.3696115 * 0.09 - 0.0034 = 0.0332650 - 0.0034 = 0.0298650 > 0 (cycles = 0).
    # With c extra cycles of 2 flying days each, mu = 0.1 + 0.2 (3 + 2c); c = 20: mu = 8.7 and the side row
    # mu <= tau has slack 1 - 8.7 = -7.7, the smallest of the rows.
    cfg = make(dict(G_BASE, H=[1, 2], allow_replacement=False))
    x = np.zeros((2, 2, 3), dtype=int)
    x[0, 1, :] = 1
    m = np.zeros((2, 1, 3), dtype=int)
    close(rm.repeated_safety(cfg, x, m, None, 0), 0.0298650, abs_=1e-7)
    close(rm.repeated_safety(cfg, x, m, None, 20), -7.7, abs_=1e-9)
    assert rm.repeated_safety(cfg, x, m, None, 1) < 0          # c = 1: mu = 0.1 + 0.2 * 5 = 1.1 > tau


# ---------------------------------------------------------------------------
# best_reference (spec 698-1625 through row_violations and costs)
# ---------------------------------------------------------------------------
BEST_BASE = dict(F=2, M=1, H=[1, 2], model="rainflow", bound_method="markov", tau=1.0, epsilon=0.1,
                 rho=0.5, mu_0=0.04, mu=0.05, C_M=1.0, C_R=10.0, C_rep=3.0, C_D=0.0,
                 replacement_mu=0.0)


def test_best_reference_tiny_input_by_hand():
    # Markov cells, tau 1, eps 0.1: the cap is mu <= 0.1. rho = 0.5, mu_0 = 0.04, increment 0.05 on every
    # mission day, replacement value 0, ardinf. F = 2 trains, M = 1, H = [1, 2] (T = 3), loop row mu_3 <= mu_1.
    # Let X be the train that flies on day 3 and Y the other one.
    #  * mu_3(X) >= 0.05 (X flies on day 3), so the loop row needs mu_1(X) >= 0.05 > 0.04: X must fly on day 1
    #    (mu_1 = 0.09) and cannot fly on day 2 (0.14 > 0.1): Y flies on day 2.
    #  * X on day 2: a repair gives 0.045, then day 3 gives 0.095 > mu_1 = 0.09: the loop row fails; a replacement
    #    gives 0 and 0.05 <= 0.09: X is replaced on day 2.
    #  * Y flew on day 2 and is in the depot on day 3. Without intervention mu_3(Y) = mu_2(Y) > mu_1(Y).
    #    A repair gives 0.5 (mu_1(Y) + 0.05) which is <= mu_1(Y) only if mu_1(Y) >= 0.05, but mu_1(Y) <= 0.04:
    #    Y is replaced on day 3 (mu_3 = 0 <= mu_1).
    #  * Y on day 1 has 4 options (idle, depot without intervention, repair, replacement): all feasible
    #    (mu_2(Y) <= 0.09, mu_3(Y) = 0).
    # Feasible schedules: 2 (choice of X) * 4 = 8. Costs (C_M = 1, C_rep = 3, C_R = 10, C_D = 0, mode total):
    #    X replaced day 2: 1 + 3; Y replaced day 3: 1 + 3; total 8 with Y idle on day 1;
    #    + Y depot day 1 without intervention: 8 + 1 = 9;
    #    + Y repaired day 1: z = rho mu_0 = 0.02, 1 + 10 * 0.02 = 1.2: 9.2;
    #    + Y replaced day 1: 1 + 3 = 4: 12.
    # Two schedules for each of 8, 9, 9.2, 12 (X = train 0 or 1).
    cfg = make(BEST_BASE)
    best = rm.best_reference(cfg)
    assert best["n_total"] == (2 * 4) ** 3 == 512
    assert best["n_feasible"] == 8
    close(sorted(best["costs"]), [8.0, 8.0, 9.0, 9.0, 9.2, 9.2, 12.0, 12.0])
    close(best["cost"], 8.0)
    x, m, r = best["schedule"]
    # the first schedule found with cost 8 has X = train 0: flies days 1 and 3, replaced on day 2; Y = train 1
    # flies day 2 and is replaced on day 3.
    assert x[0, 1].tolist() == [1, 0, 1] and x[1, 1].tolist() == [0, 1, 0]
    assert r[0, 0].tolist() == [0, 1, 0] and r[1, 0].tolist() == [0, 0, 1]
    assert x[1, 0].tolist() == [0, 0, 1] and x[0, 0].tolist() == [0, 1, 0]
    assert not m.any()


def test_best_reference_without_a_feasible_schedule():
    # H = [1, 1]: the operating phase is the single day 2; the train that flies on it has mu_2 = mu_1 + 0.05
    # > mu_1 (no intervention on a flying day), so the loop row fails for every schedule: no feasible schedule.
    # Total schedules: (2 flyers * 4 states of the other train)^2 = 64.
    best = rm.best_reference(make(dict(BEST_BASE, H=[1, 1])))
    assert best["cost"] is None and best["schedule"] is None
    assert best["n_feasible"] == 0 and best["n_total"] == 64


# ---------------------------------------------------------------------------
# Task O5: least one-sided solution and best_onesided (spec 780-836, 1602-1617, 1562-1594)
# ---------------------------------------------------------------------------
def os_schedule(x_fly, y_fly, x_dep, y_dep, x_rep, y_rep, x_repl, y_repl):
    """F = 2, M = 1, L = 1, T = 3 schedule. Per train: flies (day vector), depot day, repair, replacement."""
    x = np.zeros((2, 2, 3), dtype=np.int8)
    x[0, 1], x[1, 1], x[0, 0], x[1, 0] = x_fly, y_fly, x_dep, y_dep
    m = np.zeros((2, 1, 3), dtype=np.int8)
    r = np.zeros((2, 1, 3), dtype=np.int8)
    m[0, 0], m[1, 0], r[0, 0], r[1, 0] = x_rep, y_rep, x_repl, y_repl
    return x, m, r


def test_least_solution_rescues_a_schedule_that_breaks_the_loop_row():
    # Markov cells, ardinf, tau 1, eps 0.2 (cap mu <= 0.2), rho 0.5, mu_0 0.04, increment d = 0.05 per mission,
    # H = [1, 2] (T = 3, loop row mu_3 <= mu_1). Train X flies days 1 and 3 and is repaired on day 2; train Y
    # flies day 2 and is repaired on day 3 (Y idle on day 1).
    # Exact trajectory of X: 0.04, 0.09, 0.5 * 0.09 = 0.045, 0.045 + 0.05 = 0.095 > mu_1 = 0.09: loop row broken.
    # One-sided rows, with y = mu_1(X): mu_1 >= 0.04 + 0.05 and mu_1 >= mu_3, mu_2 >= 0.5 mu_1, mu_3 >= mu_2 + 0.05.
    # The operating-phase map is y -> a y + d with a = 1 - rho = 0.5 (a mission, then a repair, then the next
    # mission): its fixed point is y = a y + d, y = d / (1 - a) = 0.1. The Kleene iteration climbs from 0.09
    # (0.095, 0.0975, ...) to y = 0.1 (below the cap 0.2). Least solution of X: mu = 0.04, 0.1, 0.05, 0.1.
    # Y: y = mu_1(Y) >= 0.04; mu_2 >= y + 0.05; mu_3 >= 0.5 (y + 0.05) = 0.5 y + 0.025; mu_1 >= mu_3 gives
    # y = 0.5 y + 0.025, y = 0.05 (a repair after one mission). Least solution of Y: 0.04, 0.05, 0.1, 0.05.
    # z = rho mu_{k-1} on a repair day (ardinf, Eq. a_z): X day 2: 0.5 * 0.1 = 0.05; Y day 3: 0.5 * 0.1 = 0.05.
    # u = max mu = 0.1 (inf_norm, C_D = 0 but u is still the value of Eq. a_u_inf).
    cfg = make(BEST_BASE, epsilon=0.2)
    x, m, r = os_schedule([1, 0, 1], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [0] * 3, [0] * 3)
    traj = rm.reference_trajectory(cfg, x, m, r)
    close(traj["mu"][0, 0], [0.04, 0.09, 0.045, 0.095])
    exact_rows = rm.row_violations(cfg, dict(
        x=x, m=m, r=r, mu=traj["mu"][..., 1:], v=traj["v"][..., 1:], R=traj["R"][..., 1:], K=traj["K"][..., 1:],
        g_mu=traj["g_mu"][..., 1:], g_v=traj["g_v"][..., 1:], g_R=traj["g_R"][..., 1:], g_K=traj["g_K"][..., 1:]),
        tol=1e-9, bigm_tol=0.0, quad_rel=0.0)
    # (exact Y: 0.04, 0.04, 0.09, 0.045, and 0.045 > mu_1 = 0.04: its loop row is broken too)
    assert [(v.row, v.i) for v in exact_rows] == [("mu_loop", 0), ("mu_loop", 1)]
    ls = rm.least_solution(cfg, x, m, r)
    assert ls["feasible"] and ls["row_violations"] == []
    close(ls["mu"][0, 0], [0.04, 0.10, 0.05, 0.10], abs_=1e-10)
    close(ls["mu"][1, 0], [0.04, 0.05, 0.10, 0.05], abs_=1e-10)
    close(ls["z"], [[[0.0, 0.05, 0.0]], [[0.0, 0.0, 0.05]]], abs_=1e-10)
    close(ls["u"], 0.1, abs_=1e-10)
    close(rm.minimal_z(cfg, {"mu": ls["mu"], "g_mu": ls["g_mu"]}, m), ls["z"], abs_=1e-12)
    # the least solution is below any other one-sided solution: a larger mu_1 is also accepted by the rows
    # (it is not the least), checked through row_violations with mu_1(X) = 0.12.
    sol = dict(x=x, m=m, r=r, z=ls["z"], u=ls["u"])
    for name in ("mu", "v", "R", "K", "g_mu", "g_v", "g_R", "g_K"):
        sol[name] = ls[name][..., 1:].copy()
    sol["mu"][0, 0] = [0.12, 0.06, 0.12]
    sol["z"] = np.array(ls["z"])
    sol["z"][0, 0, 1] = 0.06
    sol["u"] = 0.12
    assert rm.row_violations(cfg, sol, tol=1e-9, bigm_tol=0.0, quad_rel=0.0) == []
    assert (sol["mu"][0, 0] >= ls["mu"][0, 0, 1:] - 1e-12).all()


def test_least_solution_reports_no_solution_when_the_iteration_diverges():
    # Same cells. X flies days 1 and 3 and makes no intervention on day 2 (Y flies day 2): the operating phase has
    # a mission and no intervention, so mu_3 >= mu_1 + 0.05 > mu_1 and the loop row mu_1 >= mu_3 makes the
    # lower bounds climb by 0.05 per pass until they exceed the range: no solution (the schedule is infeasible).
    cfg = make(BEST_BASE, epsilon=0.2)
    x, m, r = os_schedule([1, 0, 1], [0, 1, 0], [0] * 3, [0, 0, 1], [0] * 3, [0, 0, 1], [0] * 3, [0] * 3)
    ls = rm.least_solution(cfg, x, m, r)
    assert not ls["feasible"] and "cell (0, 0)" in ls["reason"] and "mu" in ls["reason"]
    # one pass per 0.05 would need only 4 passes to pass the cap 0.2; with max_passes = 2 the schedule of the
    # previous test, which needs more than two passes to reach 0.1 within 1e-12, is also reported
    x, m, r = os_schedule([1, 0, 1], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [0] * 3, [0] * 3)
    ls = rm.least_solution(cfg, x, m, r, max_passes=2)
    assert not ls["feasible"] and "max_passes" in ls["reason"]


ARD1_OS = dict(BEST_BASE, epsilon=0.5, repair_model="ard1", mu_0=[[0.08], [0.12]], replacement_mu=0.10)


def test_least_solution_with_anchors():
    # Markov cells, ard1, eps 0.5 (cap mu <= 0.5), rho 0.5 (a = 0.5), d = 0.05, replacement value 0.10,
    # mu_0 = 0.08 for X and 0.12 for Y, starting anchors g_0 = mu_0, H = [1, 2] (T = 3).
    # X: day 1 repair (flying train Y), day 2 flies, day 3 replacement. Y: day 1 flies, day 2 replacement, day 3 flies.
    # Exact X: s_1 = 0.5 * 0.08 + 0.5 * 0.08 = 0.08, g_1 = 0.08; s_2 = 0.13, g_2 = 0.08; s_3 = 0.10, g_3 = 0.10.
    # Loop rows: s_3 = 0.10 > s_1 = 0.08 and g_3 = 0.10 > g_1 = 0.08: broken.
    # One-sided: day 1 is an intervention day, so g_1 = s_1 (two lower bounds) and both can rise: the loop rows
    # s_1 >= s_3 = 0.10 and g_1 >= g_3 = 0.10 give s_1 = g_1 = 0.10; then s_2 >= s_1 + 0.05 = 0.15 (g_2 = g_1 = 0.10
    # on a carry day); s_3 = 0.10 (replacement). Least solution of X: s = 0.08, 0.10, 0.15, 0.10 and
    # g = 0.08, 0.10, 0.10, 0.10. Repair cost: z = rho max(0, mu_0 - g_0) = 0 (ard1, day 1 of X: mu_0 = g_0).
    # Exact Y: s = 0.12, 0.17, 0.10, 0.15; g = 0.12, 0.12, 0.10, 0.10: both loop rows hold (0.15 <= 0.17, 0.10 <= 0.12).
    cfg = make(ARD1_OS)
    x, m, r = os_schedule([0, 1, 0], [1, 0, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [0] * 3, [0, 0, 1], [0, 1, 0])
    traj = rm.reference_trajectory(cfg, x, m, r)
    close(traj["mu"][0, 0], [0.08, 0.08, 0.13, 0.10])
    close(traj["g_mu"][0, 0], [0.08, 0.08, 0.08, 0.10])
    ls = rm.least_solution(cfg, x, m, r)
    assert ls["feasible"] and ls["row_violations"] == []
    close(ls["mu"][0, 0], [0.08, 0.10, 0.15, 0.10], abs_=1e-10)
    close(ls["g_mu"][0, 0], [0.08, 0.10, 0.10, 0.10], abs_=1e-10)
    close(ls["mu"][1, 0], [0.12, 0.17, 0.10, 0.15], abs_=1e-10)
    close(ls["g_mu"][1, 0], [0.12, 0.12, 0.10, 0.10], abs_=1e-10)
    close(ls["z"], 0.0, abs_=1e-10)
    # X repaired on days 1 and 3 and flying on day 2 (a replacement instead of the day-3 repair is not used):
    # s_3 >= 0.5 s_2 + 0.5 g_2 = 0.5 (s_1 + 0.05) + 0.5 s_1 = s_1 + 0.025 > s_1 = g_1 >= g_3 = s_3: the repair
    # cannot lower the state below its anchor (known defect of the ard1 repair, merge/decisions.md), so the
    # lower bounds climb by 0.025 per pass and there is no solution.
    x, m, r = os_schedule([0, 1, 0], [1, 0, 1], [1, 0, 1], [0, 1, 0], [1, 0, 1], [0] * 3, [0] * 3, [0, 1, 0])
    ls = rm.least_solution(cfg, x, m, r)
    assert not ls["feasible"] and "cell (0, 0)" in ls["reason"]


@pytest.mark.parametrize("name", ["bf_G_no_replacement", "bf_IG_ardinf"])
def test_best_onesided_is_at_most_best_reference(name):
    # Every schedule that is feasible with its exact trajectory is also feasible one-sidedly, with the same
    # least solution (the exact trajectory satisfies every lower-bound row, and the least solution lies above it
    # and below any solution), so best_onesided <= best_reference.
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "oracle", name + ".yaml")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = load_config(read_input(path))
    ref = rm.best_reference(cfg)
    ons = rm.best_onesided(cfg)
    assert ons["n_total"] == ref["n_total"] and ons["n_feasible"] >= ref["n_feasible"]
    assert ons["cost"] <= ref["cost"] + 1e-12
    assert ons["exact"] is True
    x, m, r = ons["schedule"]
    assert rm.least_solution(cfg, x, m, r)["row_violations"] == []


def test_best_onesided_tiny_input_by_hand():
    # The input of test_best_reference_tiny_input_by_hand with eps 0.2. Cap mu <= 0.2, ardinf: exact optimum.
    # X flies on day 3. The loop row, one-sidedly, is satisfiable for X in two ways: flying on day 1 with a repair
    # on day 2 (the fixed point of the previous test, mu_1 = 0.1 <= 0.2) at repair cost 10 * 0.5 * 0.1 = 0.5 plus
    # one depot day (C_M = 1): 1.5; or a replacement on day 2: 1 + 3 = 4. The other train Y flies day 2 and
    # needs an intervention on day 3: a repair (mu_3 = 0.5 (y + 0.05) <= y, y = 0.05, cost 1 + 10 * 0.5 * 0.1 = 1.5,
    # where mu_2(Y) = y + 0.05 = 0.1) or a replacement (cost 4). Y on day 1: idle (cost 0). Best: 1.5 + 1.5 = 3.0.
    cfg = make(BEST_BASE, epsilon=0.2)
    ons = rm.best_onesided(cfg)
    close(ons["cost"], 3.0, abs_=1e-9)
    assert ons["exact"] and ons["cost"] <= rm.best_reference(cfg)["cost"] + 1e-12
