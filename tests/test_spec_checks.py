"""Rejected keys and consistency checks, checked against the specification (no Gurobi needed).

Every expected outcome comes from spec/spec.tex: Input Specification (lines 349-500), Rejected Keys
(500-525), Consistency Checks (527-600), the reliability rows of the models (862-922, 1013-1031,
1092-1119, 1195-1199, 1274-1310, 1486-1555) and the keys by cell type (2745-2761). The location is
cited in each docstring or comment.

Inputs are the files of tests/data/io/valid/. An invalid input is a valid mapping with one key
changed (helpers ``_g``, ``_ig``, ``_w``, ``_ga``, ``_rf`` build a fresh mapping from a library file).
Each test asserts ``ValueError`` and that the message names the key (spec 530); nothing more
specific about the message is asserted, except the words "unknown key" (spec 503).
"""

import copy
import math
import re
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy import integrate, stats
from scipy.optimize import brentq
from scipy.special import gammaincc, ndtri

from fleet_management.config import load_config

DATA = Path(__file__).parent / "data" / "io" / "valid"
_DROP = object()          # in a helper's keywords: remove the key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base(name):
    """A fresh mapping read from the library file tests/data/io/valid/<name>.yaml."""
    with open(DATA / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _update(data, **kw):
    for key, value in kw.items():
        if value is _DROP:
            data.pop(key, None)
        else:
            data[key] = value
    return data


def _g(**kw):
    """Gaussian fleet, F = 3, M = 1, L = 2, H = [2, 3]: only scalar and list values (minimal.yaml)."""
    return _update(_base("minimal"), **kw)


def _ig(**kw):
    return _update(_base("inverse_gaussian"), **kw)


def _w(**kw):
    data = _base("wiener")
    data.pop("v_max_user")
    return _update(data, **kw)


def _ga(**kw):
    return _update(_base("gamma"), **kw)


RF = ["markov", "cantelli", "hoeffding", "bernstein", "chernoff"]


def _rf(bound, **kw):
    """A rainflow fleet with the one component of the given bound (L = 1), cut from rainflow_bounds.yaml."""
    index = RF.index(bound)
    out = {}
    for key, value in _base("rainflow_bounds").items():
        if isinstance(value, list) and len(value) == 5:
            value = value[index]
            if value is None or value == [None]:
                continue
        out[key] = value
    out["L"] = 1
    return _update(out, **kw)


def _mixed(**kw):
    """Cells of two models: component 0 Gaussian, component 1 inverse Gaussian."""
    kw.setdefault("eta", 0.15)
    return _g(model=["gaussian", "inverse_gaussian"], **kw)


def _named(key, msg):
    pattern = r"(?<![A-Za-z0-9_])" + re.escape(key) + r"(?![A-Za-z0-9_])"
    return re.search(pattern, msg) is not None


def _refuse(data, *keys):
    """load_config raises ValueError whose message names every key (a tuple: any one of its members)."""
    with pytest.raises(ValueError) as info:
        load_config(data)
    msg = str(info.value)
    for key in keys:
        alternatives = key if isinstance(key, tuple) else (key,)
        assert any(_named(k, msg) for k in alternatives), f"message {msg!r} does not name {alternatives}"
    return msg


def _accept(data):
    return load_config(data)


# ---------------------------------------------------------------------------
# Formulas of the spec used to compute the boundaries
# ---------------------------------------------------------------------------

def _kappa_G(eps):
    return float(ndtri(1 - eps))                      # spec 1022


def _vmax(tau, eps):
    return tau ** 2 / _kappa_G(eps) ** 2              # spec 566


def _f(mu, tau, c2, c1=0.0):
    """f(mu) = c2 (tau - mu)^2 + c1 (tau - mu), spec 868."""
    return c2 * (tau - mu) ** 2 + c1 * (tau - mu)


def _fprime(mu, tau, c2, c1=0.0):
    return -2 * c2 * (tau - mu) - c1                  # spec 872


def _tangent_cap(mu, tau, c2, c1, mu_r):
    """Spec 897."""
    return _f(mu_r, tau, c2, c1) + _fprime(mu_r, tau, c2, c1) * (mu - mu_r)


def _pwl_cap(mu, tau, c2, c1, P):
    """Spec 902-912: the cap of the segment [(p-1) tau/P, p tau/P] that contains mu."""
    p = min(P, max(1, math.ceil(mu * P / tau)))
    mid = (p - 0.5) * tau / P
    return _tangent_cap(mu, tau, c2, c1, mid)


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["gaussian", "gaussian_ard1_forms", "inverse_gaussian", "wiener",
                                  "gamma", "rainflow_bounds", "mixed_example", "minimal"])
def test_library_is_valid(name):
    """Each file of the library passes every check (the starting point of every test below)."""
    _accept(_base(name))


@pytest.mark.parametrize("bound", RF)
def test_single_rainflow_inputs_are_valid(bound):
    _accept(_rf(bound))


# ---------------------------------------------------------------------------
# Rejected keys (spec 500-525)
# ---------------------------------------------------------------------------

# (rejected key, a value for it, the canonical keys the message must name; spec 506-525).
# A row that only says "Remove it" has no canonical key.
REJECTED = [
    ("maintenance_type", "ARD1", ["repair_model"]),                  # 507
    ("tail_bound", "cantelli", ["bound_method"]),                    # 508
    ("method", "cantelli", ["bound_method"]),                        # 508
    ("formulation", "exact", ["reliability_impl"]),                  # 509
    ("beta", 0.05, ["gamma_beta"]),                                  # 510
    ("alpha_inc", 1.0, ["mu", "gamma_beta"]),                        # 511
    ("gamma_beta_bound", 1.0, []),                                   # 512
    ("gamma_calibration_method", "x", []),                           # 513
    ("b_inc", 0.1, ["support"]),                                     # 514
    ("b_0", 0.1, ["support_0"]),                                     # 515
    ("b_new", 0.1, ["support_new"]),                                 # 516
    ("n_workers", 2, []),                                            # 517
    ("warm_start", True, []),                                        # 517
    ("depot_capacity", 1, []),                                       # 518
    ("C_P", 1.0, []),                                                # 519
    ("C_S", 1.0, ["C_D"]),                                           # 520
    ("alpha", 1.0, ["tau"]),                                         # 521
    ("xi", 0.8, ["rho"]),                                            # 522
    ("repair_rho", 0.8, ["rho"]),                                    # 522
    ("mu_new", 0.05, ["replacement_mu"]),                            # 523
    ("v_new", 0.0001, ["replacement_v"]),                            # 524
]


def test_rejected_table_has_the_keys_of_the_spec():
    """The table of 506-525 has 21 keys (the plan said 22)."""
    assert len(REJECTED) == 21
    assert len({row[0] for row in REJECTED}) == 21


@pytest.mark.parametrize("key,value,canonical", REJECTED)
def test_rejected_key_names_canonical_key(key, value, canonical):
    """Spec 502, 506-525: a rejected key raises an error that names the canonical key; it is not 'unknown key'."""
    data = _g()
    data[key] = value
    msg = _refuse(data, *canonical)
    assert "unknown key" not in msg.lower()


def test_unknown_key_message():
    """Spec 503: any other key raises 'unknown key' with the list of the canonical keys."""
    data = _g()
    data["no_such_key"] = 1
    msg = _refuse(data)
    assert "unknown key" in msg.lower()
    for key in ["tau", "epsilon", "rho", "mu_0", "C_M"]:
        assert _named(key, msg)


def test_key_check_runs_before_arrays_are_read():
    """Spec 504: an unknown key together with a malformed mu gives the unknown-key error."""
    data = _g(mu=[0.3] * 7, tau="not a number")
    data["no_such_key"] = 1
    msg = _refuse(data)
    assert "unknown key" in msg.lower()


def test_rejected_key_checked_before_arrays_are_read():
    """Spec 504: a rejected key together with a malformed mu gives the rejected-key error."""
    data = _g(mu=[0.3] * 7, tau="not a number")
    data["maintenance_type"] = "ARD1"
    _refuse(data, "repair_model")


# ---------------------------------------------------------------------------
# Required keys (spec 536, 2750-2760)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["F", "M", "H", "model", "tau", "epsilon", "rho", "mu_0", "mu",
                                 "C_M", "C_R", "C_rep", "C_D"])
def test_required_key_of_every_cell(key):
    """Spec 536: each of these keys is required."""
    data = _g()
    del data[key]
    _refuse(data, key)


@pytest.mark.parametrize("make,key", [
    (_g, "v"), (_ig, "eta"), (_w, "sigma"), (_ga, "gamma_beta"),
    (lambda: _rf("cantelli"), "v"),
    (lambda: _rf("hoeffding"), "support"),
    (lambda: _rf("bernstein"), "v"), (lambda: _rf("bernstein"), "support"),
    (lambda: _rf("chernoff"), "cgf"), (lambda: _rf("chernoff"), "s_chernoff"),
])
def test_required_key_of_a_cell_type(make, key):
    """Spec 536 and the table 2750-2760: G needs v, IG eta, W sigma, Ga gamma_beta, Ca v, Ho support,
    Be v and support, Ch cgf and s_chernoff."""
    data = make()
    del data[key]
    _refuse(data, key)


def test_markov_needs_only_the_common_keys():
    """Spec 2756: a Markov cell requires only bound_method (and the keys of every cell)."""
    data = _rf("markov")
    for key in ["v", "support", "cgf", "s_chernoff", "eta", "sigma", "gamma_beta"]:
        assert key not in data
    _accept(data)


def test_markov_without_bound_method_needs_the_keys_of_the_default():
    """Spec 383, 2756-2757: without bound_method the default is cantelli, which requires v."""
    data = _rf("markov")
    del data["bound_method"]
    _refuse(data, "v")


@pytest.mark.parametrize("make,dropped", [
    (lambda: _ig(), ["v", "support", "cgf", "sigma", "gamma_beta"]),
    (lambda: _g(), ["eta", "sigma", "gamma_beta", "support", "cgf", "s_chernoff"]),
])
def test_keys_of_other_types_are_not_required(make, dropped):
    """Spec 2750-2760: a key that no cell of the fleet needs is not required."""
    data = make()
    for key in dropped:
        data.pop(key, None)
    _accept(data)


# ---------------------------------------------------------------------------
# Shapes and entries (spec 537-538)
# ---------------------------------------------------------------------------

def test_shape_cell_value_wrong():
    """Spec 537: a cell value with the shape of no row (2 x 2 for F = 3, L = 2) is an error."""
    _refuse(_g(tau=[[1.0, 1.0], [1.0, 1.0]]), "tau")


def test_shape_profile_wrong_length():
    """Spec 537: three numbers are not a profile shape for L = 2, M = 1."""
    _refuse(_g(mu=[0.3, 0.3, 0.3]), "mu")


def test_shape_operating_profile_last_axis():
    """Spec 537: the last axis of an operating profile has H2 = 3 entries; four is an error."""
    _refuse(_g(mu=[[[0.3] * 4], [[0.3] * 4]]), "mu")


def test_shape_transitory_profile_last_axis():
    """Spec 537: the last axis of a transitory profile has H1 = 2 entries; three is an error."""
    _refuse(_g(mu_trans=[[[0.3] * 3], [[0.3] * 3]]), ("mu_trans", "mu"))


def test_shape_component_names_length():
    """Spec 537, 543: component_names has L entries."""
    _accept(_g(component_names=["A", "B"]))
    _refuse(_g(component_names=["A", "B", "C"]), "component_names")
    _refuse(_g(component_names=["A"]), "component_names")


@pytest.mark.parametrize("key,value", [
    ("rho", [0.8, None]), ("tau", [1.0, None]), ("epsilon", [None, 0.05]), ("mu_0", [0.1, None]),
    ("C_R", [2.0, None]), ("C_rep", [None, 10.0]),
])
def test_null_where_the_key_applies(key, value):
    """Spec 538: null at a cell where the key applies is an error."""
    _refuse(_g(**{key: value}), key)


def test_null_in_a_selector_and_a_profile():
    """Spec 538: null is an error also in a selector and in a profile entry."""
    _refuse(_g(model=["gaussian", None]), "model")
    _refuse(_g(v=[[0.0025], [None]]), "v")
    _refuse(_g(repair_model=["ardinf", None]), "repair_model")


@pytest.mark.parametrize("key", ["tau", "epsilon", "rho", "mu_0", "C_rep"])
@pytest.mark.parametrize("bad", [float("inf"), float("nan"), "abc"])
def test_entries_are_finite_numbers(key, bad):
    """Spec 538: at a cell where a key applies, every entry is a finite number."""
    _refuse(_g(**{key: bad}), key)


@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
def test_profile_entries_are_finite(bad):
    _refuse(_g(mu=[[bad], [0.3]]), "mu")


# ---------------------------------------------------------------------------
# Sizes and selectors (spec 541-549)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("F,ok", [(0, False), (-1, False), (1, False), (2, True), (3, True)])
def test_F_positive_and_greater_than_M(F, ok):
    """Spec 543 and 369: F is a positive integer and F > M (M = 1): F = M fails, F = M + 1 passes."""
    if ok:
        assert load_config(_g(F=F)).F == F
    else:
        _refuse(_g(F=F), "F")


@pytest.mark.parametrize("M,ok", [(0, False), (-1, False), (1, True), (2, True), (3, False), (4, False)])
def test_M_positive_and_less_than_F(M, ok):
    """Spec 543: M is a positive integer and F > M (F = 3): M = 2 passes, M = F fails."""
    if ok:
        assert load_config(_g(M=M)).M == M
    else:
        _refuse(_g(M=M), ("M", "F"))


@pytest.mark.parametrize("L", [0, -1])
def test_L_positive(L):
    """Spec 543: L is a positive integer."""
    _refuse(_g(L=L), "L")


def test_L_equals_length_of_model_axis():
    """Spec 543: a given L equals the length of the component axis of model."""
    assert load_config(_g(L=2)).L == 2
    _refuse(_g(L=1), "L")
    _refuse(_g(L=3), "L")


def test_L_for_one_model_string():
    """Spec 371: L is inferred from the component axis of model; with one string and no L it is 1,
    and a given L > 1 is read (L is the way to ask for more components)."""
    assert load_config(_g(model="gaussian")).L == 1
    assert load_config(_g(model="gaussian", L=2)).L == 2


@pytest.mark.parametrize("names", [["A", ""], ["A", "   "], [1, "B"], ["A", None]])
def test_component_names_entries(names):
    """Spec 544: every entry of component_names is a string that is not empty after blanks are removed."""
    _refuse(_g(component_names=names), "component_names")


@pytest.mark.parametrize("H,ok", [
    (1, True), (3, True), ([1, 1], True), ([2, 3], True),
    (0, False), (-2, False), ([0, 3], False), ([2, 0], False), ([2, 3, 4], False), ("a", False),
    (2.5, False),
])
def test_H_integer_at_least_one_or_a_pair(H, ok):
    """Spec 545: H is an integer >= 1, or a pair of integers H1, H2 >= 1."""
    if ok:
        _accept(_g(H=H))
    else:
        _refuse(_g(H=H), "H")


@pytest.mark.parametrize("value", ["Gaussian", "foo", "", None])
def test_model_entries(value):
    """Spec 546: model entries are the five names, in lower case."""
    _refuse(_g(model=["gaussian", value]), "model")


@pytest.mark.parametrize("value", ["ARD1", "ara1", "foo"])
def test_repair_model_entries(value):
    """Spec 546: repair_model entries are ard1 or ardinf."""
    _refuse(_g(repair_model=value), "repair_model")


@pytest.mark.parametrize("make", [_g, _ig, _w, _ga])
def test_repair_model_ard1_is_accepted_for_every_model(make):
    """Spec 382, 546: every model accepts ard1 and ardinf."""
    for law in ["ard1", "ardinf"]:
        _accept(make(repair_model=law))


def test_bound_method_checked_also_at_a_cell_that_is_not_rainflow():
    """Spec 547, 362: a bad selector entry is an error also at a Gaussian cell; a valid entry is ignored there."""
    _refuse(_g(bound_method="chernof"), "bound_method")
    _refuse(_g(bound_method="Cantelli"), "bound_method")
    _accept(_g(bound_method="markov"))
    _accept(_g(bound_method=["markov", "chernoff"]))


@pytest.mark.parametrize("bound", ["markov", "cantelli", "hoeffding", "bernstein", "chernoff"])
def test_bound_method_values_accepted_at_a_gaussian_cell(bound):
    """Spec 547."""
    _accept(_g(bound_method=bound))


def test_reliability_impl_checked_also_at_a_linear_row():
    """Spec 548: a bad value is an error at an IG cell too; a valid value is ignored there."""
    _refuse(_ig(reliability_impl="foo"), "reliability_impl")
    _refuse(_ga(reliability_impl="lp"), "reliability_impl")
    _refuse(_rf("markov", reliability_impl="Exact"), "reliability_impl")
    for impl in ["exact", "tangent", "pwl"]:
        _accept(_ig(reliability_impl=impl))
        _accept(_ga(reliability_impl=impl))
        _accept(_rf("markov", reliability_impl=impl))
        _accept(_rf("chernoff", reliability_impl=impl))


def test_encoding_entries():
    """Spec 548: encoding entries are bigm or indicator."""
    for value in ["bigm", "indicator"]:
        _accept(_g(encoding=value))
    _refuse(_g(encoding="big-m"), "encoding")
    _refuse(_g(encoding="BIGM"), "encoding")


def test_penalty_type_values():
    """Spec 549: penalty_type is inf_norm or quadratic."""
    _accept(_g(penalty_type="inf_norm"))
    _accept(_g(penalty_type="quadratic"))
    _refuse(_g(penalty_type="l2"), "penalty_type")


def test_quadratic_penalty_requires_exact_at_quadratic_rows():
    """Spec 549, 479: with C_D > 0, quadratic needs the form exact at every G cell."""
    _accept(_g(penalty_type="quadratic", reliability_impl="exact"))
    _refuse(_g(penalty_type="quadratic", reliability_impl="tangent"), ("penalty_type", "reliability_impl"))
    _refuse(_g(penalty_type="quadratic", reliability_impl="pwl"), ("penalty_type", "reliability_impl"))
    _refuse(_g(penalty_type="quadratic", reliability_impl=["exact", "tangent"]),
            ("penalty_type", "reliability_impl"))
    _refuse(_w(penalty_type="quadratic", reliability_impl="tangent"), ("penalty_type", "reliability_impl"))


def test_quadratic_penalty_with_zero_C_D_allows_every_form():
    """Spec 549: the condition holds only if C_D > 0."""
    _accept(_g(penalty_type="quadratic", reliability_impl="tangent", C_D=0.0))
    _accept(_g(penalty_type="quadratic", reliability_impl="pwl", C_D=0.0))


@pytest.mark.parametrize("bound,fails", [("markov", False), ("cantelli", True), ("hoeffding", True),
                                         ("bernstein", True), ("chernoff", False)])
def test_quadratic_penalty_rainflow_rows(bound, fails):
    """Spec 549: the quadratic rows are G, W and RF with Ca, Ho or Be; Mk and Ch rows are linear."""
    data = _rf(bound, penalty_type="quadratic", reliability_impl="tangent")
    if fails:
        _refuse(data, ("penalty_type", "reliability_impl"))
    else:
        _accept(data)


@pytest.mark.parametrize("make", [_ig, _ga])
def test_quadratic_penalty_linear_rows_allow_every_form(make):
    """Spec 549: IG and Ga rows are linear."""
    _accept(make(penalty_type="quadratic", reliability_impl="pwl"))


# ---------------------------------------------------------------------------
# Reliability, repair and initial state (spec 554-561)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tau", [0.0, -1.0, float("inf"), float("nan")])
def test_tau_positive_and_finite(tau):
    """Spec 554."""
    _refuse(_g(tau=tau), "tau")


def test_tau_one_bad_cell():
    _refuse(_g(tau=[1.0, 0.0]), "tau")


@pytest.mark.parametrize("eps", [0.0, 1.0, -0.1, 1.5])
def test_epsilon_in_open_unit_interval(eps):
    """Spec 555 (tested at an IG cell, where the G bound 0.5 does not apply)."""
    _refuse(_ig(epsilon=eps), "epsilon")


@pytest.mark.parametrize("make", [_g, _w])
def test_epsilon_below_half_at_G_and_W(make):
    """Spec 555: eps < 0.5 at G and W cells; 0.5 fails, 0.49 passes."""
    _refuse(make(epsilon=0.5), "epsilon")
    _refuse(make(epsilon=0.7), "epsilon")
    _accept(make(epsilon=0.49))


def test_epsilon_half_has_no_tolerance():
    """Spec 532, 555: a strict inequality has no tolerance, so 0.5 (1 + 1e-13) fails at a G cell."""
    _refuse(_g(epsilon=0.5 * (1 + 1e-13)), "epsilon")


def test_epsilon_at_least_half_allowed_at_other_cells():
    """Spec 531, 555: eps < 0.5 is a check of G and W; an IG cell and a Cantelli cell accept 0.5."""
    _accept(_ig(epsilon=0.5))
    _accept(_rf("cantelli", epsilon=0.5))
    _accept(_rf("markov", epsilon=0.5))


def test_epsilon_bound_runs_only_at_G_cells_of_a_mixed_fleet():
    """Spec 531, 555: eps = 0.6 passes at the IG column and fails at the G column."""
    _accept(_mixed(epsilon=[0.05, 0.6]))
    _refuse(_mixed(epsilon=[0.6, 0.05]), "epsilon")


@pytest.mark.parametrize("rho", [0.0, -0.1, 1.5, float("nan")])
def test_rho_in_half_open_interval(rho):
    """Spec 556: rho in (0, 1]."""
    _refuse(_g(rho=rho), "rho")


def test_rho_one_passes():
    _accept(_g(rho=1.0))
    _accept(_g(rho=0.001))


@pytest.mark.parametrize("mu_0", [-0.1, 1.0, 1.5])
def test_mu_0_in_range(mu_0):
    """Spec 557: 0 <= mu_0 < tau (tau = 1); mu_0 = tau fails."""
    _refuse(_g(mu_0=mu_0), "mu_0")


def test_mu_0_boundary_passes():
    """Spec 557: mu_0 = 0 and mu_0 just below tau pass."""
    _accept(_g(mu_0=0.0))
    _accept(_g(mu_0=0.999))


def test_mu_0_below_tau_per_cell():
    """Spec 557: tau varies by component; mu_0 = 1.5 is below tau = 2 and not below tau = 1."""
    _accept(_g(tau=[2.0, 2.0], mu_0=1.5))
    _refuse(_g(tau=[2.0, 1.0], mu_0=1.5), "mu_0")


@pytest.mark.parametrize("make,key,extra", [
    (_g, "v_0", {}), (_w, "v_0", {}), (lambda: _rf("cantelli"), "v_0", {}),
    (lambda: _rf("bernstein"), "v_0", {}),
    (lambda: _rf("bernstein"), "support_0", {"v_0": 0.0}),    # v_0 <= support_0^2 / 4, spec 588
    (lambda: _rf("hoeffding"), "R_0", {}),
])
def test_initial_values_non_negative(make, key, extra):
    """Spec 558: v_0, support_0 and R_0 are >= 0; zero passes."""
    data = _update(make(), **extra)
    data[key] = -1e-3
    _refuse(data, key)
    data[key] = 0.0
    _accept(data)


@pytest.mark.parametrize("make,key", [
    (_g, "replacement_mu"), (_g, "replacement_v"), (_w, "replacement_v"),
    (lambda: _rf("bernstein"), "support_new"), (lambda: _rf("hoeffding"), "R_new"),
    (lambda: _rf("cantelli"), "replacement_v"), (_ig, "replacement_mu"),
])
def test_replacement_values_non_negative_only_if_allowed(make, key):
    """Spec 558: replacement values are >= 0 if allow_replacement is true; with false they are not checked."""
    data = make()
    data[key] = -1e-3
    _refuse(data, key)
    data["allow_replacement"] = False
    _accept(data)


def test_negative_replacement_v_passes_without_replacement():
    """Spec 558, 566: with allow_replacement false, replacement values are not checked, also out of range."""
    _accept(_g(allow_replacement=False, replacement_v=-0.5, replacement_mu=-2.0))
    _accept(_g(allow_replacement=False, replacement_v=1e3, replacement_mu=5.0))


def test_tangent_ref_in_half_open_interval():
    """Spec 559: tangent_ref in [0, 1) at a tangent cell (gaussian_ard1_forms: cell (0, 0) is tangent)."""
    base = _base("gaussian_ard1_forms")
    for bad in [-0.1, 1.0, 1.5]:
        data = copy.deepcopy(base)
        data["tangent_ref"] = bad
        _refuse(data, "tangent_ref")
    for good in [0.0, 0.5, 0.999]:
        data = copy.deepcopy(base)
        data["tangent_ref"] = good
        _accept(data)


def test_tangent_ref_read_only_at_tangent_cells():
    """Spec 394, 2748: a scalar tangent_ref is read only at cells with the form tangent."""
    _accept(_g(reliability_impl="exact", tangent_ref=1.5))
    _accept(_g(reliability_impl="pwl", tangent_ref=1.5))
    _refuse(_g(reliability_impl="tangent", tangent_ref=1.5), "tangent_ref")


def test_pwl_points_integer_at_least_one():
    """Spec 559: pwl_points is an integer >= 1 (gaussian_ard1_forms has a pwl cell)."""
    base = _base("gaussian_ard1_forms")
    for bad in [0, -1, 2.5]:
        data = copy.deepcopy(base)
        data["pwl_points"] = bad
        _refuse(data, "pwl_points")
    for good in [1, 2, 8]:
        data = copy.deepcopy(base)
        data["pwl_points"] = good
        _accept(data)


def test_anchor_mu_0_range():
    """Spec 560: 0 <= anchor_mu_0 <= mu_0 at ard1 cells (gaussian_ard1_forms, mu_0 = 0.12)."""
    base = _base("gaussian_ard1_forms")
    for bad in [-0.01, 0.13]:
        data = copy.deepcopy(base)
        data["anchor_mu_0"] = [[bad, 0.06], [0.05, 0.12], [0.0, 0.10]]
        _refuse(data, "anchor_mu_0")
    for good in [0.0, 0.12]:
        data = copy.deepcopy(base)
        data["anchor_mu_0"] = [[good, 0.06], [0.05, 0.12], [0.0, 0.10]]
        _accept(data)


def test_anchor_v_0_range():
    """Spec 560: 0 <= anchor_v_0 <= v_0 at ard1 cells at G (v_0 = 0.001)."""
    base = _base("gaussian_ard1_forms")
    for bad in [-1e-4, 0.0011]:
        data = copy.deepcopy(base)
        data["anchor_v_0"] = bad
        _refuse(data, "anchor_v_0")
    for good in [0.0, 0.001]:
        data = copy.deepcopy(base)
        data["anchor_v_0"] = good
        _accept(data)


def test_anchors_read_only_at_ard1_cells():
    """Spec 560, 402: a scalar anchor outside its range passes at ardinf cells, which do not read it."""
    _accept(_g(repair_model="ardinf", anchor_mu_0=5.0, anchor_v_0=5.0, mu_0=0.1))
    _refuse(_g(repair_model="ard1", anchor_mu_0=5.0), "anchor_mu_0")


def test_anchor_R_0_range():
    """Spec 560: 0 <= anchor_R_0 <= R_0 at ard1 cells at Ho (R_0 = 0.001)."""
    _refuse(_rf("hoeffding", repair_model="ard1", anchor_R_0=0.002), "anchor_R_0")
    _refuse(_rf("hoeffding", repair_model="ard1", anchor_R_0=-0.001), "anchor_R_0")
    _accept(_rf("hoeffding", repair_model="ard1", anchor_R_0=0.001))
    _accept(_rf("hoeffding", repair_model="ard1", anchor_R_0=0.0))
    _accept(_rf("hoeffding", repair_model="ardinf", anchor_R_0=0.002))


def test_anchor_K_0_range():
    """Spec 560, 592: 0 <= anchor_K_0 <= K_0 (K_0 = s mu_0 = 0.5 by default), and g^K_0 >= s g^mu_0."""
    _refuse(_rf("chernoff", repair_model="ard1", anchor_K_0=0.6), "anchor_K_0")
    _refuse(_rf("chernoff", repair_model="ard1", anchor_K_0=-0.1), "anchor_K_0")
    _accept(_rf("chernoff", repair_model="ard1", anchor_K_0=0.5))
    _accept(_rf("chernoff", repair_model="ard1", anchor_K_0=0.0, anchor_mu_0=0.0))
    _accept(_rf("chernoff", repair_model="ardinf", anchor_K_0=0.6))


def test_anchor_K_0_at_least_s_anchor_mu_0():
    """Spec 592: at ard1 cells g^K_0 >= s g^mu_0 (s = 5, anchor_mu_0 = 0.1: 0.45 fails, 0.5 passes)."""
    _refuse(_rf("chernoff", repair_model="ard1", anchor_mu_0=0.1, anchor_K_0=0.45), "anchor_K_0")
    _accept(_rf("chernoff", repair_model="ard1", anchor_mu_0=0.1, anchor_K_0=0.5))
    _accept(_rf("chernoff", repair_model="ardinf", anchor_mu_0=0.1, anchor_K_0=0.45))


@pytest.mark.parametrize("make,key,value,extra", [
    (_g, "replacement_mu", 0.05, {}), (_g, "replacement_v", 0.0001, {"replacement_mu": 0.05}),
    (_w, "replacement_v", 0.0001, {}),
    (lambda: _rf("bernstein"), "support_new", 0.03, {}), (lambda: _rf("hoeffding"), "R_new", 0.0005, {}),
    (lambda: _rf("chernoff"), "cgf_new", 0.1, {}), (lambda: _rf("cantelli"), "replacement_v", 0.0001, {}),
    (_ig, "replacement_mu", 0.05, {}),
])
def test_replacement_as_new_needs_zero_inputs(make, key, value, extra):
    """Spec 561: with replacement_as_new true every replacement input read at a cell is 0; false: no check."""
    data = _update(make(), **extra)
    data[key] = value
    _accept(data)
    data["replacement_as_new"] = True
    for other in ["replacement_mu", "replacement_v", "support_new", "R_new", "cgf_new"]:
        if other != key:
            data.pop(other, None)
    if key != "replacement_mu":
        data["replacement_mu"] = 0.0
    _refuse(data, key)
    data[key] = 0.0
    _accept(data)


def test_replacement_as_new_ignores_keys_not_read():
    """Spec 561: 'at every cell that reads them': replacement_v is not read at an IG cell."""
    data = _ig(replacement_as_new=True, replacement_v=1.0, replacement_mu=0.0)
    _accept(data)


# ---------------------------------------------------------------------------
# Natural ranges (spec 566)
# ---------------------------------------------------------------------------

def test_G_initial_variance_natural_range():
    """Spec 566: v_0 <= V_max = tau^2 / kappa_G^2. kappa_G = ndtri(1 - eps), spec 1022.
    With eps = 1e-4, V_max = 0.0723; mu_0 = 0.95 keeps mu_0 >= 3 sqrt(v_0) (spec 579)."""
    eps, tau = 1e-4, 1.0
    vmax = _vmax(tau, eps)
    common = dict(epsilon=eps, mu_0=0.95)
    _accept(_g(v_0=vmax * (1 - 1e-3), **common))
    _refuse(_g(v_0=vmax * (1 + 1e-3), **common), "v_0")


def test_W_initial_variance_natural_range():
    """Spec 566: V_max = tau^2 / kappa_G^2 at W cells (tau = 2)."""
    eps, tau = 0.05, 2.0
    vmax = _vmax(tau, eps)
    _accept(_w(tau=tau, v_0=vmax * (1 - 1e-3)))
    _refuse(_w(tau=tau, v_0=vmax * (1 + 1e-3)), "v_0")


def test_Cantelli_initial_variance_natural_range():
    """Spec 566: eps tau^2 / (1 - eps) at Ca cells."""
    eps, tau = 0.1, 1.0
    top = eps * tau ** 2 / (1 - eps)
    _accept(_rf("cantelli", v_0=top * (1 - 1e-3)))
    _refuse(_rf("cantelli", v_0=top * (1 + 1e-3)), "v_0")


def test_Bernstein_initial_variance_natural_range():
    """Spec 566: tau (tau - 2c) / (2 kappa) at Be cells, c = kappa b / 3, kappa = ln(1/eps).
    support_0 = b = 0.6 so that v_0 <= b0^2 / 4 = 0.09 (spec 588) is weaker than the range 0.0171."""
    eps, tau, b = 0.1, 1.0, 0.6
    kappa = math.log(1 / eps)
    c = kappa * b / 3
    top = tau * (tau - 2 * c) / (2 * kappa)
    _accept(_rf("bernstein", support_0=b, v_0=top * (1 - 1e-3)))
    _refuse(_rf("bernstein", support_0=b, v_0=top * (1 + 1e-3)), "v_0")


def test_Hoeffding_initial_budget_natural_range():
    """Spec 566: R_0 <= 2 tau^2 / kappa at Ho cells."""
    eps, tau = 0.1, 1.0
    top = 2 * tau ** 2 / math.log(1 / eps)
    _accept(_rf("hoeffding", R_0=top * (1 - 1e-3)))
    _refuse(_rf("hoeffding", R_0=top * (1 + 1e-3)), "R_0")


def test_Chernoff_initial_quantity_natural_range():
    """Spec 566: K_0 <= s tau - kappa at Ch cells (s = 5, eps = 0.1)."""
    eps, tau, s = 0.1, 1.0, 5.0
    top = s * tau - math.log(1 / eps)
    _accept(_rf("chernoff", cgf_0=top * (1 - 1e-3)))
    _refuse(_rf("chernoff", cgf_0=top * (1 + 1e-3)), "cgf_0")


def test_mean_replacement_value_natural_range():
    """Spec 566: replacement_mu lies in [0, tau]; tau = 1.5 here makes 1.6 out of range at a Gaussian cell."""
    _refuse(_g(tau=1.5, replacement_mu=1.6), "replacement_mu")


# ---------------------------------------------------------------------------
# Replacement state against the rows (spec 567)
# ---------------------------------------------------------------------------

def test_G_exact_replacement_state():
    """Spec 567, 1018: kappa_G^2 v_new <= (tau - mu_new)^2 (mu_new = 0.9, v_new at the cap).
    kappa_G = ndtri(1 - eps), spec 1022."""
    eps, tau, mu_new = 0.05, 1.0, 0.9
    cap = _f(mu_new, tau, 1 / _kappa_G(eps) ** 2)
    _accept(_g(replacement_mu=mu_new, replacement_v=cap * (1 - 1e-3)))
    _refuse(_g(replacement_mu=mu_new, replacement_v=cap * (1 + 1e-3)), "replacement_v")
    _accept(_g(replacement_mu=mu_new, replacement_v=cap * (1 + 1e-3), allow_replacement=False))


def test_W_exact_replacement_state():
    """Spec 567, 1195-1199: the Gaussian row at W cells (mu_new = 0.5)."""
    eps, tau, mu_new = 0.05, 1.0, 0.5
    cap = _f(mu_new, tau, 1 / _kappa_G(eps) ** 2)
    _accept(_w(replacement_mu=mu_new, replacement_v=cap * (1 - 1e-3)))
    _refuse(_w(replacement_mu=mu_new, replacement_v=cap * (1 + 1e-3)), "replacement_v")


def test_G_tangent_replacement_state():
    """Spec 567, 897: in the form tangent the cap is the tangent line at mu_r = tangent_ref tau.
    mu_new = 0.65, tangent_ref = 0.4: the tangent cap (0.0222) is below the exact cap (0.0453)
    and below mu_new^2 / 9 = 0.0469 (spec 579), so a v_new between them fails only in the form tangent."""
    eps, tau, mu_new, ref = 0.05, 1.0, 0.65, 0.4
    c2 = 1 / _kappa_G(eps) ** 2
    tangent = _tangent_cap(mu_new, tau, c2, 0.0, ref * tau)
    exact = _f(mu_new, tau, c2)
    assert tangent < 0.03 < exact
    kw = dict(replacement_mu=mu_new, tangent_ref=ref)
    _accept(_g(reliability_impl="exact", replacement_v=0.03, **kw))
    _refuse(_g(reliability_impl="tangent", replacement_v=0.03, **kw), "replacement_v")
    _accept(_g(reliability_impl="tangent", replacement_v=tangent * (1 - 1e-3), **kw))
    _refuse(_g(reliability_impl="tangent", replacement_v=tangent * (1 + 1e-3), **kw), "replacement_v")


def test_G_pwl_replacement_state_uses_the_segment_of_mu_new():
    """Spec 567, 902-912: in the form pwl the cap is that of the segment that contains mu_new.
    P = 4, mu_new = 0.8 lies in [0.75, 1], midpoint 0.875. The cap of the segment [0.5, 0.75]
    (midpoint 0.625) at mu_new would be 0.0035, so v_new = 0.01 passes only if the right segment is used."""
    eps, tau, mu_new, P = 0.05, 1.0, 0.8, 4
    c2 = 1 / _kappa_G(eps) ** 2
    cap = _pwl_cap(mu_new, tau, c2, 0.0, P)
    other = _tangent_cap(mu_new, tau, c2, 0.0, 0.625)
    exact = _f(mu_new, tau, c2)
    assert other < 0.01 < cap < exact
    kw = dict(replacement_mu=mu_new, reliability_impl="pwl", pwl_points=P)
    _accept(_g(replacement_v=0.01, **kw))
    _accept(_g(replacement_v=cap * (1 - 1e-3), **kw))
    _refuse(_g(replacement_v=cap * (1 + 1e-3), **kw), "replacement_v")
    _refuse(_g(replacement_v=0.0135, **kw), "replacement_v")
    _accept(_g(reliability_impl="exact", replacement_mu=mu_new, replacement_v=0.0135))


def test_Cantelli_replacement_state():
    """Spec 567, 1498, 1540: v_new <= eps (tau - mu_new)^2 / (1 - eps)."""
    eps, tau, mu_new = 0.1, 1.0, 0.02
    cap = _f(mu_new, tau, eps / (1 - eps))
    _accept(_rf("cantelli", replacement_mu=mu_new, replacement_v=cap * (1 - 1e-3)))
    _refuse(_rf("cantelli", replacement_mu=mu_new, replacement_v=cap * (1 + 1e-3)), "replacement_v")


def test_Hoeffding_replacement_state():
    """Spec 567, 1504, 1542: R_new <= 2 (tau - mu_new)^2 / kappa."""
    eps, tau, mu_new = 0.1, 1.0, 0.02
    cap = _f(mu_new, tau, 2 / math.log(1 / eps))
    _accept(_rf("hoeffding", replacement_mu=mu_new, R_new=cap * (1 - 1e-3)))
    _refuse(_rf("hoeffding", replacement_mu=mu_new, R_new=cap * (1 + 1e-3)), "R_new")


def test_Bernstein_replacement_state():
    """Spec 567, 1518, 1543: v_new <= ((tau - mu)^2 - 2 c (tau - mu)) / (2 kappa), c = kappa b / 3.
    support_new = b = 0.6 makes the row (0.0125) bind before b_new^2 / 4 = 0.09 (spec 588)
    and before the natural range 0.0171 (spec 566)."""
    eps, tau, mu_new, b = 0.1, 1.0, 0.02, 0.6
    kappa = math.log(1 / eps)
    cap = _f(mu_new, tau, 1 / (2 * kappa), -b / 3)
    kw = dict(support_new=b, replacement_mu=mu_new)
    _accept(_rf("bernstein", replacement_v=cap * (1 - 1e-3), **kw))
    _refuse(_rf("bernstein", replacement_v=cap * (1 + 1e-3), **kw), "replacement_v")
    _accept(_rf("bernstein", replacement_v=cap * (1 + 1e-3), allow_replacement=False, **kw))


def test_Markov_replacement_state():
    """Spec 567, 1492: mu_new <= eps tau (eps = 0.1, tau = 1)."""
    _accept(_rf("markov", replacement_mu=0.1 * (1 - 1e-3)))
    _refuse(_rf("markov", replacement_mu=0.1 * (1 + 1e-3)), "replacement_mu")


def test_Chernoff_replacement_state():
    """Spec 567, 1525-1529: K_new <= s tau - kappa (s = 5, eps = 0.1)."""
    top = 5.0 - math.log(10.0)
    _accept(_rf("chernoff", cgf_new=top * (1 - 1e-3)))
    _refuse(_rf("chernoff", cgf_new=top * (1 + 1e-3)), "cgf_new")
    _accept(_rf("chernoff", cgf_new=top * (1 + 1e-3), allow_replacement=False))


def test_user_variance_bound_replacement_state():
    """Spec 567, 583: v_new <= V_user at G cells where v_max_user is given."""
    kw = dict(v_0=0.001, v_max_user=0.001, replacement_mu=0.3)     # mu_new >= 3 sqrt(v_new), spec 579
    _refuse(_g(replacement_v=0.002, **kw), ("replacement_v", "v_max_user"))
    _accept(_g(replacement_v=0.001, **kw))
    _accept(_g(replacement_v=0.002, allow_replacement=False, **kw))


def _ig_cap(eta, tau, eps):
    """The cap of spec 1097, from the law of Z in spec 1107: mu_bar = tau / E[Z | Z > q], P(Z > q) = eps.
    Z is inverse Gaussian with mean 1 and shape 1/eta, that is scipy invgauss(mu=eta, scale=1/eta)."""
    dist = stats.invgauss(mu=eta, scale=1 / eta)
    q = dist.isf(eps)
    qbar = integrate.quad(lambda x: x * dist.pdf(x), q, np.inf)[0] / eps
    return tau / qbar


def test_IG_cdf_matches_the_spec_formula():
    """Sanity check of the helper: scipy's law equals F of spec 1107 (to 1e-12) at a few points."""
    from scipy.special import ndtr
    for eta in [0.15, 0.2]:
        dist = stats.invgauss(mu=eta, scale=1 / eta)
        for x in [0.5, 1.0, 1.7, 3.0]:
            r = 1 / math.sqrt(eta * x)
            spec_F = ndtr(r * (x - 1)) + math.exp(2 / eta) * ndtr(-r * (x + 1))
            assert abs(dist.cdf(x) - spec_F) < 1e-9


def test_IG_replacement_state():
    """Spec 567, 1097: replacement_mu <= mu_bar = tau / E[Z | Z > q]. Computed here with
    scipy.stats.invgauss and scipy.integrate.quad (the bound is the smaller one of the two cells,
    eta = 0.15 and 0.20). The margin is 0.5 percent."""
    caps = [_ig_cap(eta, 1.0, 0.05) for eta in (0.15, 0.20)]
    top = min(caps)
    assert 0.05 < top < 1.0
    _accept(_ig(replacement_mu=top * 0.995))
    _refuse(_ig(replacement_mu=top * 1.005), "replacement_mu")
    _accept(_ig(replacement_mu=top * 1.005, allow_replacement=False))


def test_Gamma_replacement_state():
    """Spec 567, 1298, 1306, 1308: replacement_mu <= A_1(beta* tau, eps) / beta*.
    Two bounds that the spec gives are used, so no rough computation of w(A) is needed.
    Inside: A_1 >= eps t (spec 1308, 'w(eps t) <= eps holds'), so every mu_new <= eps tau passes.
    Outside: A_1 <= A_hat, the root of Q(A_hat, t) = eps (spec 1306, 1308), so mu_new above
    A_hat / beta* fails at the cell where this bound is smallest.
    beta* is the smallest rate of the cell among the terms with a positive mean (spec 1276):
    the cell rates of gamma.yaml, with beta_0 = 40 at cell (0, 0) and beta_new = 40, all >= the mission rates."""
    eps, tau = 0.05, 1.0
    rates = np.array(_base("gamma")["gamma_beta"])[:, :, 0]        # (F, L), M = 1
    bounds = []
    for beta_star in rates.ravel():
        t = beta_star * tau
        a_hat = brentq(lambda a: gammaincc(a, t) - eps, 1e-6, 10 * t, xtol=1e-12)
        bounds.append(a_hat / beta_star)
    inside = eps * tau * 0.99
    outside = min(bounds) * 1.01
    assert inside < outside < tau
    _accept(_ga(replacement_mu=inside))
    _refuse(_ga(replacement_mu=outside), "replacement_mu")
    _accept(_ga(replacement_mu=outside, allow_replacement=False))


# ---------------------------------------------------------------------------
# Increments (spec 571-575)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [1.0, 1.5, -0.1])
def test_increment_mean_in_range(bad):
    """Spec 572: 0 <= mu_inc < tau (tau = 1)."""
    _refuse(_g(mu=bad, v=0.0), "mu")


def test_increment_mean_boundary_passes():
    """Spec 572: mu_inc = 0 passes (with v_inc = 0); mu_inc just below tau passes at an IG cell."""
    _accept(_g(mu=0.0, v=0.0))
    _accept(_ig(mu=0.999))


def test_increment_mean_checked_every_day_and_mission():
    """Spec 572: every mission j and day k. A profile (L, M, H2) with one bad entry on the last day fails."""
    good = [[[0.3, 0.3, 0.3]], [[0.3, 0.3, 0.3]]]
    _accept(_g(mu=good, v=0.0025))
    bad = copy.deepcopy(good)
    bad[1][0][2] = 1.0
    _refuse(_g(mu=bad, v=0.0025), "mu")


def test_increment_mean_checked_in_transitory_phase():
    """Spec 572: days 1..T include the transitory days; mu_trans = 1.0 fails (H1 = 2)."""
    _refuse(_g(mu_trans=[[[1.0, 0.3]], [[0.3, 0.3]]]), ("mu_trans", "mu"))


def test_zero_mean_needs_zero_other_parameters():
    """Spec 573: where mu_inc = 0, v_inc = 0; v_inc = 0.0025 fails."""
    _refuse(_g(mu=0.0, v=0.0025), "v")
    _accept(_g(mu=0.0, v=0.0))


def test_zero_mean_needs_zero_support_and_cgf():
    """Spec 573: where mu_inc = 0, b_inc and K_inc are 0."""
    _refuse(_rf("hoeffding", mu=0.0), "support")
    _accept(_rf("hoeffding", mu=0.0, support=[0.0]))
    _refuse(_rf("chernoff", mu=0.0), "cgf")
    _accept(_rf("chernoff", mu=0.0, cgf=[0.0]))
    _refuse(_rf("bernstein", mu=0.0, support=[0.0]), "v")
    _accept(_rf("bernstein", mu=0.0, support=[0.0], v=[0.0]))


def test_zero_mean_lookup_per_profile_in_transitory_phase():
    """Spec 439-448, 573: each profile key is looked up on its own. mu_trans = 0 with no v_trans reads
    the operating v on the transitory days, which is > 0 there, so the check fails."""
    _refuse(_g(mu_trans=0.0), ("v", "v_trans", "mu_trans"))
    _accept(_g(mu_trans=0.0, v_trans=0.0))


@pytest.mark.parametrize("make,key,value", [
    (_g, "v", -0.001), (lambda **kw: _rf("cantelli", **kw), "v", [-0.001]),
    (lambda **kw: _rf("hoeffding", **kw), "support", [-0.1]),
    (lambda **kw: _rf("bernstein", **kw), "support", [-0.1]),
    (lambda **kw: _rf("chernoff", **kw), "cgf", [-0.1]),
])
def test_increment_parameters_non_negative(make, key, value):
    """Spec 574: v_inc, b_inc, K_inc are >= 0."""
    _refuse(make(**{key: value}), key)


def test_transitory_profile_needs_operating_profile():
    """Spec 574, 458: a transitory profile has its operating profile (Ho: support_trans without support)."""
    data = _rf("hoeffding", support_trans=[0.1])
    _accept(data)
    del data["support"]
    _refuse(data, ("support", "support_trans"))


# ---------------------------------------------------------------------------
# Gaussian, inverse Gaussian, Wiener, Gamma (spec 577-584)
# ---------------------------------------------------------------------------

def test_G_initial_mean_at_least_three_sigma():
    """Spec 579: mu_0 >= 3 sqrt(v_0) (mu_0 = 0.1: v_0 <= 0.001111)."""
    _accept(_g(v_0=0.00111))
    _refuse(_g(v_0=0.00112), "v_0")


def test_G_increment_mean_at_least_three_sigma():
    """Spec 579: mu_inc >= 3 sqrt(v_inc) (mu_inc = 0.3: v_inc <= 0.01)."""
    _accept(_g(v=0.0099))
    _refuse(_g(v=0.0102), "v")


def test_G_increment_checked_every_day():
    """Spec 579: for every mission j and day k; one bad day of a profile (L, M, H2) fails."""
    mu = [[[0.3, 0.3, 0.3]], [[0.3, 0.3, 0.3]]]
    v = [[[0.0099, 0.0099, 0.0099]], [[0.0099, 0.0099, 0.0099]]]
    _accept(_g(mu=mu, v=v))
    v[0][0][2] = 0.0102
    _refuse(_g(mu=mu, v=v), "v")


def test_G_transitory_increment_mean_at_least_three_sigma():
    """Spec 579, 443: on the transitory days the transitory values are used."""
    _refuse(_g(mu_trans=0.3, v_trans=0.02), ("v_trans", "v"))
    _accept(_g(mu_trans=0.3, v_trans=0.0099))


def test_G_replacement_mean_at_least_three_sigma():
    """Spec 579: mu_new >= 3 sqrt(v_new) if allow_replacement is true (mu_new = 0.3: v_new <= 0.01)."""
    _accept(_g(replacement_mu=0.3, replacement_v=0.0099))
    _refuse(_g(replacement_mu=0.3, replacement_v=0.0102), ("replacement_v", "replacement_mu"))
    _accept(_g(replacement_mu=0.3, replacement_v=0.0102, allow_replacement=False))


def test_G_mean_over_sigma_check_not_run_at_other_cells():
    """Spec 531, 579: at W cells mu_0 >= 3 sqrt(v_0) is not a check (v_0 = 0.1 > (mu_0 / 3)^2)."""
    _accept(_w(v_0=0.1))


@pytest.mark.parametrize("eta", [0.0, -0.1, float("nan"), float("inf")])
def test_IG_eta_positive(eta):
    """Spec 580."""
    _refuse(_ig(eta=eta), "eta")


def test_IG_eta_one_bad_cell():
    _refuse(_ig(eta=[0.15, 0.0]), "eta")


@pytest.mark.parametrize("sigma", [0.0, -0.1, float("nan")])
def test_W_sigma_positive(sigma):
    """Spec 581."""
    _refuse(_w(sigma=sigma), "sigma")


def test_W_sigma_small_positive_passes():
    _accept(_w(sigma=1e-6))


def test_W_initial_variance_non_negative():
    """Spec 581."""
    _refuse(_w(v_0=-1e-3), "v_0")
    _accept(_w(v_0=0.0))


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_Ga_rate_positive_and_finite(value):
    """Spec 582: every entry of gamma_beta is finite and > 0."""
    _refuse(_ga(gamma_beta=value), "gamma_beta")


def test_Ga_rate_one_entry_of_the_array():
    """Spec 582: an (F, L, M) array with one bad entry."""
    rates = [[[20.0], [25.0]], [[20.0], [25.0]], [[22.0], [0.0]]]
    _refuse(_ga(gamma_beta=rates), "gamma_beta")


def test_Ga_transitory_rate_positive():
    """Spec 582: gamma_beta_trans entries are finite and > 0."""
    _accept(_ga(gamma_beta_trans=30.0))
    for value in [0.0, -1.0, float("nan")]:
        _refuse(_ga(gamma_beta_trans=value), ("gamma_beta_trans", "gamma_beta"))


@pytest.mark.parametrize("key", ["gamma_beta_0", "gamma_beta_new"])
@pytest.mark.parametrize("value", [0.0, -5.0, float("inf")])
def test_Ga_initial_and_replacement_rate_when_given(key, value):
    """Spec 582: gamma_beta_0 and gamma_beta_new, where given, are finite and > 0.
    (NaN is not tested: in the stored form NaN means 'not given', spec 2729-2734.)"""
    _refuse(_ga(**{key: value}), key)


def test_Ga_rate_per_cell_with_null_means_not_given():
    """Spec 360, 582: null at a cell means 'not given' and is not checked; a given 0 at one cell fails."""
    _accept(_ga(gamma_beta_0=[[40.0, None], [None, None], [None, None]]))
    _refuse(_ga(gamma_beta_0=[[40.0, None], [None, None], [None, 0.0]]), "gamma_beta_0")


def test_user_variance_bound_positive_and_above_initial_variance():
    """Spec 583: V_user > 0 and V_user >= v_0 (v_0 = 0.001; equality passes)."""
    _refuse(_g(v_0=0.0, v_max_user=0.0), "v_max_user")
    _refuse(_g(v_0=0.0, v_max_user=-1.0), "v_max_user")
    _refuse(_g(v_0=0.001, v_max_user=0.0009), "v_max_user")
    _accept(_g(v_0=0.001, v_max_user=0.001))
    _accept(_g(v_0=0.001, v_max_user=0.5))


@pytest.mark.parametrize("make", [_w, lambda: _rf("cantelli"), lambda: _rf("bernstein")])
def test_user_variance_bound_other_cells(make):
    """Spec 583: the same at W, Ca, Be cells (v_0 = 0.0005)."""
    data = make()
    data["v_max_user"] = 0.0
    _refuse(data, "v_max_user")
    data["v_max_user"] = 0.0004
    _refuse(data, "v_max_user")
    data["v_max_user"] = 0.0005
    _accept(data)


def test_user_variance_bound_null_is_not_given():
    """Spec 360: null at v_max_user means not given."""
    _accept(_g(v_max_user=[[0.01, None], [None, None], [None, None]]))


def test_user_variance_bound_not_checked_at_IG_cell():
    """Spec 531, 583: v_max_user applies at G, W, Ca, Be only; a bad scalar is ignored at an IG cell."""
    _accept(_ig(v_max_user=-1.0))


# ---------------------------------------------------------------------------
# Rainflow (spec 586-593)
# ---------------------------------------------------------------------------

def test_Be_increment_variance_at_most_quarter_support_squared():
    """Spec 588: v_inc <= b_inc^2 / 4 (b_inc = 0.1: 0.0025; equality passes)."""
    _accept(_rf("bernstein", v=[0.0025]))
    _refuse(_rf("bernstein", v=[0.0026]), "v")


def test_Be_initial_variance_at_most_quarter_support_squared():
    """Spec 588: v_0 <= b0^2 / 4 (b0 = 0.05: 0.000625)."""
    _accept(_rf("bernstein", v_0=0.000625))
    _refuse(_rf("bernstein", v_0=0.0007), "v_0")


def test_Be_replacement_variance_at_most_quarter_support_squared():
    """Spec 588: v_new <= b_new^2 / 4 (b_new = 0.03: 0.000225) if allow_replacement is true."""
    _accept(_rf("bernstein", replacement_v=0.0002))
    _refuse(_rf("bernstein", replacement_v=0.0003), ("replacement_v", "support_new"))
    _accept(_rf("bernstein", replacement_v=0.0003, allow_replacement=False))


def test_Be_tau_above_two_kappa_b_over_three():
    """Spec 589: tau > 2 kappa b / 3, kappa = ln(1/eps) = ln 10: b = 0.65 gives 0.9978 < 1, b = 0.66 gives 1.013.
    allow_replacement is false and v_0 = 0 (the ranges and rows of the replacement state are not the point)."""
    kw = dict(allow_replacement=False, v_0=0.0, support_0=0.0, v=[0.002])
    _accept(_rf("bernstein", support=[0.65], **kw))
    _refuse(_rf("bernstein", support=[0.66], **kw), "tau")


def test_Be_tangent_ref_condition():
    """Spec 590, 900: tangent_ref tau < sqrt(tau (tau - 2c)), c = kappa b / 3. With c = 0.42, tau = 1
    the right side is sqrt(0.16) = 0.4: tangent_ref 0.39 passes, 0.41 fails."""
    eps, tau, c = 0.1, 1.0, 0.42
    kappa = math.log(1 / eps)
    b = 3 * c / kappa
    top = math.sqrt(tau * (tau - 2 * c))
    assert abs(top - 0.4) < 1e-12
    kw = dict(support=[b], allow_replacement=False, reliability_impl="tangent")
    _accept(_rf("bernstein", tangent_ref=0.39, **kw))
    _refuse(_rf("bernstein", tangent_ref=0.41, **kw), "tangent_ref")


def test_Be_pwl_condition():
    """Spec 590, 916: tau / (2P) < sqrt(tau (tau - 2c)) = 0.4 (c = 0.42): P = 1 gives 0.5 and fails, P = 2 gives 0.25."""
    eps, tau, c = 0.1, 1.0, 0.42
    b = 3 * c / math.log(1 / eps)
    kw = dict(support=[b], allow_replacement=False, reliability_impl="pwl")
    _accept(_rf("bernstein", pwl_points=2, **kw))
    _refuse(_rf("bernstein", pwl_points=1, **kw), ("pwl_points", "reliability_impl"))


def test_Be_tangent_condition_not_checked_in_form_exact():
    """Spec 590: the condition belongs to the forms tangent and pwl; with exact a large tangent_ref is not read."""
    eps, c = 0.1, 0.42
    b = 3 * c / math.log(1 / eps)
    _accept(_rf("bernstein", support=[b], allow_replacement=False, reliability_impl="exact",
                tangent_ref=0.9, pwl_points=1))


def test_Ch_s_above_kappa_over_tau():
    """Spec 591: s > ln(1/eps) / tau = 2.3026 (eps = 0.1). mu_0 and replacement_mu are 0 so that
    K_0 = K_new = 0 stay in the range s tau - kappa (spec 566)."""
    kw = dict(mu_0=0.0, replacement_mu=0.0)
    _accept(_rf("chernoff", s_chernoff=2.31, **kw))
    _refuse(_rf("chernoff", s_chernoff=2.30, **kw), "s_chernoff")
    _refuse(_rf("chernoff", s_chernoff=0.5, **kw), "s_chernoff")


def test_Ch_increment_cgf_at_least_s_mu():
    """Spec 592: K_inc >= s mu_inc (s = 5, mu_inc = 0.25: 1.25; equality passes)."""
    _accept(_rf("chernoff", cgf=[1.25]))
    _refuse(_rf("chernoff", cgf=[1.2]), "cgf")


def test_Ch_initial_cgf_at_least_s_mu_0():
    """Spec 592: K_0 >= s mu_0 (mu_0 = 0.1: 0.5)."""
    _accept(_rf("chernoff", cgf_0=0.5))
    _refuse(_rf("chernoff", cgf_0=0.4), "cgf_0")


def test_Ch_replacement_cgf_at_least_s_mu_new():
    """Spec 592: K_new >= s mu_new (mu_new = 0.02: 0.1) if allow_replacement is true."""
    _accept(_rf("chernoff", cgf_new=0.1))
    _refuse(_rf("chernoff", cgf_new=0.09), "cgf_new")
    _accept(_rf("chernoff", cgf_new=0.09, allow_replacement=False))


# ---------------------------------------------------------------------------
# Run options and costs (spec 597-599)
# ---------------------------------------------------------------------------

def test_objective_mode_values():
    """Spec 597."""
    for mode in ["total", "operating_average"]:
        _accept(_g(objective_mode=mode))
    _refuse(_g(objective_mode="average"), "objective_mode")
    _refuse(_g(objective_mode="Total"), "objective_mode")


def test_evaluation_total_needs_evaluation_horizon():
    """Spec 597: in the mode evaluation_total, evaluation_horizon is given."""
    _refuse(_g(objective_mode="evaluation_total"), "evaluation_horizon")
    _accept(_g(objective_mode="evaluation_total", evaluation_horizon=3))


@pytest.mark.parametrize("mode", ["total", "operating_average", "evaluation_total"])
def test_evaluation_horizon_above_H1(mode):
    """Spec 597: if given, in any mode, E is an integer > H1 (H1 = 2): E = 2 fails, E = 3 passes."""
    _refuse(_g(objective_mode=mode, evaluation_horizon=2), "evaluation_horizon")
    _refuse(_g(objective_mode=mode, evaluation_horizon=1), "evaluation_horizon")
    _refuse(_g(objective_mode=mode, evaluation_horizon=2.5), "evaluation_horizon")
    _accept(_g(objective_mode=mode, evaluation_horizon=3))
    _accept(_g(objective_mode=mode, evaluation_horizon=3.0))
    _accept(_g(objective_mode=mode, evaluation_horizon=40))


def test_mip_gap_non_negative():
    """Spec 598: mip_gap >= 0."""
    _refuse(_g(mip_gap=-1.0), "mip_gap")
    _refuse(_g(mip_gap=-0.01), "mip_gap")
    assert _accept(_g(mip_gap=0.0)).options["mip_gap"] == 0
    assert _accept(_g(mip_gap=0.05)).options["mip_gap"] == 0.05


def test_mip_gap_tolerance_and_set_to_zero():
    """Spec 532, 598: -1e-13 passes the tolerance 1e-12 and is set to 0; -1e-10 fails."""
    cfg = _accept(_g(mip_gap=-1e-13))
    assert cfg.options["mip_gap"] == 0
    assert not (math.copysign(1.0, cfg.options["mip_gap"]) < 0 and cfg.options["mip_gap"] != 0)
    _refuse(_g(mip_gap=-1e-10), "mip_gap")


@pytest.mark.parametrize("key", ["time_limit", "progress_interval_seconds", "relaxation_time_limit"])
def test_positive_run_options(key):
    """Spec 598: these options are > 0 (strict: 0 fails)."""
    _refuse(_g(**{key: 0}), key)
    _refuse(_g(**{key: -1.0}), key)
    _accept(_g(**{key: 1e-3}))
    _accept(_g(**{key: 10}))


@pytest.mark.parametrize("key", ["gurobi_params", "validation"])
def test_mapping_run_options(key):
    """Spec 598: gurobi_params and validation are mappings."""
    _refuse(_g(**{key: "abc"}), key)
    _refuse(_g(**{key: [1, 2]}), key)
    _accept(_g(**{key: {}}))


@pytest.mark.parametrize("key", ["fast", "relaxation_warm_start", "allow_replacement", "replacement_as_new"])
def test_flag_options(key):
    """Spec 598: these options are flags (true or false); a string or a number such as 2 is an error."""
    _accept(_g(**{key: True}))
    _accept(_g(**{key: False}))
    _refuse(_g(**{key: "yes"}), key)
    _refuse(_g(**{key: 2}), key)


@pytest.mark.parametrize("key", ["C_M", "C_D", "C_R", "C_rep"])
@pytest.mark.parametrize("bad", [-1.0, float("inf"), float("nan")])
def test_costs_finite_and_non_negative(key, bad):
    """Spec 599."""
    _refuse(_g(**{key: bad}), key)


@pytest.mark.parametrize("key", ["C_M", "C_D", "C_R", "C_rep"])
def test_costs_zero_passes(key):
    """Spec 599: zero is allowed."""
    _accept(_g(**{key: 0.0}))


@pytest.mark.parametrize("key", ["C_R", "C_rep"])
def test_cell_costs_one_bad_cell(key):
    _refuse(_g(**{key: [2.0, -1.0]}), key)


# ---------------------------------------------------------------------------
# Tolerance (spec 532)
# ---------------------------------------------------------------------------

def test_tolerance_anchor_mu_0_below_mu_0():
    """Spec 532, 560: anchor_mu_0 <= mu_0 passes when it holds up to 1e-12 max(1, |a|, |b|)
    (mu_0 = 0.12): anchor = mu_0 (1 + 5e-13) passes, mu_0 (1 + 1e-10) fails."""
    base = _base("gaussian_ard1_forms")
    mu_0 = base["mu_0"]
    for factor, ok in [(1 + 5e-13, True), (1 + 1e-10, False)]:
        data = copy.deepcopy(base)
        data["anchor_mu_0"] = mu_0 * factor
        if ok:
            _accept(data)
        else:
            _refuse(data, "anchor_mu_0")


def test_tolerance_rho_at_most_one():
    """Spec 532, 556: rho <= 1 with the tolerance: 1 + 5e-13 passes, 1 + 1e-10 fails."""
    _accept(_g(rho=1 + 5e-13))
    _refuse(_g(rho=1 + 1e-10), "rho")


def test_tolerance_non_negative_values():
    """Spec 532, 558: v_0 >= 0 holds up to 1e-12: -5e-13 passes, -1e-10 fails."""
    _accept(_g(v_0=-5e-13))
    _refuse(_g(v_0=-1e-10), "v_0")


def test_tolerance_scales_with_the_larger_value():
    """Spec 532: the tolerance is 1e-12 max(1, |a|, |b|). G check mu_0 >= 3 sqrt(v_0) with tau = 1000,
    mu_0 = 300, v_0 = 1e4 (1 + x): the excess is 1.5e4 x ... 3 * 50 x = 1.5e2 x * 1e-0; for x = 1e-13 it is
    1.5e-11 <= 3e-10 (passes), for x = 1e-10 it is 1.5e-8 > 3e-10 (fails)."""
    kw = dict(tau=1000.0, mu_0=300.0)
    _accept(_g(v_0=1e4 * (1 + 1e-13), **kw))
    _refuse(_g(v_0=1e4 * (1 + 1e-10), **kw), "v_0")


def test_tolerance_increment_three_sigma():
    """Spec 532, 579: mu_inc = 0.3 >= 3 sqrt(v_inc); v_inc = 0.01 (1 + 1e-13) passes, 0.01 (1 + 1e-10) fails."""
    _accept(_g(v=0.01 * (1 + 1e-13)))
    _refuse(_g(v=0.01 * (1 + 1e-10)), "v")


def test_tolerance_user_variance_bound():
    """Spec 532, 583: V_user >= v_0 (v_0 = 0.001)."""
    _accept(_g(v_0=0.001, v_max_user=0.001 * (1 - 5e-10)))
    _refuse(_g(v_0=0.001, v_max_user=0.001 * (1 - 1e-6)), "v_max_user")


def test_tolerance_Ch_cgf_at_least_s_mu():
    """Spec 532, 592: K_inc >= s mu_inc (1.25): 1.25 - 5e-13 passes, 1.25 - 1e-10 fails."""
    _accept(_rf("chernoff", cgf=[1.25 - 5e-13]))
    _refuse(_rf("chernoff", cgf=[1.25 - 1e-10]), "cgf")


def test_tolerance_Be_variance_bound():
    """Spec 532, 588: v_inc <= b^2 / 4 = 0.0025: 0.0025 (1 + 5e-11) passes (excess 1.25e-13), 0.0025 (1 + 1e-6) fails."""
    _accept(_rf("bernstein", v=[0.0025 * (1 + 5e-11)]))
    _refuse(_rf("bernstein", v=[0.0025 * (1 + 1e-6)]), "v")


@pytest.mark.parametrize("make,key,value", [
    (_g, "tau", 0.0), (_ig, "eta", 0.0), (_w, "sigma", 0.0), (_ga, "gamma_beta", 0.0),
    (_g, "epsilon", 0.0), (_ig, "epsilon", 1.0),
])
def test_strict_inequalities_fail_at_equality(make, key, value):
    """Spec 532: a strict inequality has no tolerance: tau > 0, eta > 0, sigma > 0, rates > 0, eps in (0, 1)."""
    _refuse(make(**{key: value}), key)


def test_strict_F_greater_than_M():
    """Spec 532, 543: F > M is strict: F = M fails."""
    _refuse(_g(F=2, M=2), ("F", "M"))
    _accept(_g(F=3, M=2))


def test_strict_mu_0_below_tau():
    """Spec 532, 557: mu_0 < tau is strict: mu_0 = tau fails, also mu_0 = tau (1 + 1e-13)."""
    _refuse(_g(mu_0=1.0), "mu_0")
    _refuse(_g(mu_0=1.0 * (1 + 1e-13)), "mu_0")


def test_strict_increment_below_tau():
    """Spec 532, 572: mu_inc < tau is strict."""
    _refuse(_ig(mu=1.0), "mu")
    _refuse(_ig(mu=1.0 * (1 + 1e-13)), "mu")


def test_strict_evaluation_horizon_above_H1():
    """Spec 532, 597: E > H1 is strict (H1 = 2)."""
    _refuse(_g(evaluation_horizon=2), "evaluation_horizon")


def test_strict_Ch_s_above_bound():
    """Spec 532, 591: s > kappa / tau is strict: s = kappa (1 - 1e-13) fails (within the tolerance, still refused)."""
    kappa = math.log(10.0)
    kw = dict(mu_0=0.0, replacement_mu=0.0)
    _refuse(_rf("chernoff", s_chernoff=kappa * (1 - 1e-13), **kw), "s_chernoff")


# ---------------------------------------------------------------------------
# A check of one model runs only at that model's cells (spec 531)
# ---------------------------------------------------------------------------

def test_G_checks_do_not_run_at_IG_cells():
    """Spec 531: a value that breaks a G check passes at an IG cell: v = 1 (G needs mu >= 3 sqrt(v)),
    v_0 = -1, v_max_user = -1, replacement_v = -1 (read only at G, W, Ca, Be cells)."""
    _accept(_ig(v=1.0, v_0=-1.0, v_max_user=-1.0, replacement_v=-1.0))
    _accept(_ig(support=-1.0, support_0=-1.0, R_0=-1.0, cgf_0=-1.0))


def test_G_checks_do_not_run_at_W_and_Ga_cells():
    """Spec 531: G and IG constants are not checked at W and Ga cells."""
    _accept(_w(eta=-1.0, gamma_beta=-1.0, support=-1.0))
    _accept(_ga(eta=-1.0, sigma=-1.0, v=-1.0, v_0=-1.0))


def test_W_check_does_not_run_at_G_cell():
    """Spec 531: sigma > 0 is a Wiener check; sigma = -1 passes at a Gaussian cell."""
    _accept(_g(sigma=-1.0, eta=-1.0, gamma_beta=-1.0))


def test_eta_check_only_at_IG_cell_of_a_mixed_fleet():
    """Spec 531, 580: eta = 0 at the G column passes and at the IG column fails."""
    _accept(_mixed(eta=[0.0, 0.15]))
    _refuse(_mixed(eta=[0.15, 0.0]), "eta")


def test_G_variance_checks_only_at_G_column_of_a_mixed_fleet():
    """Spec 531, 579: v_0 = 0.5 breaks mu_0 >= 3 sqrt(v_0) at the G column; at the IG column it is not read.
    The array F x L: entries at the IG column are ignored with a UserWarning (spec 362)."""
    import warnings
    row = [0.0, 0.5]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _accept(_mixed(v_0=[row] * 3))
    _refuse(_mixed(v_0=[[0.5, 0.0]] * 3), "v_0")


def test_Ga_and_RF_checks_do_not_run_at_G_cells():
    """Spec 531: Ho, Be and Ch constants are not checked at a G cell (support = -1, cgf = -1, R_0 = -1)."""
    _accept(_g(support=-1.0, cgf=-1.0, R_0=-1.0, s_chernoff=-1.0, support_0=-1.0))


def test_Be_check_does_not_run_at_Ca_cell():
    """Spec 531, 588: v <= b^2 / 4 is a Bernstein check; v = 0.5 passes at a Cantelli cell if the other
    Cantelli conditions hold (v = 0.5 is not constrained at Ca, only v >= 0)."""
    _accept(_rf("cantelli", v=[0.5], support=[0.0001]))
