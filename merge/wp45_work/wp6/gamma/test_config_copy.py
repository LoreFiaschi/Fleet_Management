"""Tests of the input layer ``config.py``: reader, keys, shapes, checks, derived fields.

They need neither Gurobi nor the solver: ``load_config`` and ``read_input`` import
only the standard library, numpy and scipy (YAML and HDF5 inside the reader).
"""

import dataclasses
import json
import re
import warnings

import h5py
import numpy as np
import pytest
import yaml

from fleet_management import config
from fleet_management.config import load_config, read_input


# The example of the specification, section "YAML Example (Mixed Models)".
EXAMPLE_YAML = """
F: 3
M: 2
L: 2
H: [2, 3]               # H1 = 2, H2 = 3, so T = 5
component_names: [Wheelset, Brake pad]

model: [[gaussian, inverse_gaussian], [wiener, gamma], [rainflow, rainflow]]
repair_model: [[ard1, ard1], [ardinf, ard1], [ard1, ard1]]
bound_method: [[null, null], [null, null], [bernstein, chernoff]]
reliability_impl: [[tangent, null], [pwl, null], [exact, null]]
encoding: [[bigm, bigm], [bigm, bigm], [bigm, indicator]]

tau: 1.0
epsilon: [[0.05, 0.05], [0.05, 0.05], [0.10, 0.10]]
tangent_ref: 0.4        # tangent cell only
pwl_points: 6           # pwl cell only
rho: [0.8, 0.7]         # per component, the same for every train

mu_0: [[0.12, 0.10], [0.15, 0.08], [0.09, 0.14]]
v_0: [[0.0004, null], [0.0005, null], [0.0004, null]]
support_0: [[null, null], [null, null], [0.05, null]]
anchor_mu_0: [[0.10, 0.06], [null, 0.05], [0.00, 0.10]]
anchor_v_0: [[0.0002, null], [null, null], [0.0002, null]]
anchor_K_0: [[null, null], [null, null], [null, 0.50]]

replacement_mu: 0.02
replacement_v: [[0.00004, null], [0.00004, null], [0.00004, null]]
support_new: [[null, null], [null, null], [0.03, null]]
replacement_as_new: false
allow_replacement: true

mu: [[0.06, 0.05], [0.04, 0.07]]        # profile (L, M)
mu_trans: [[[0.07, 0.06], [0.06, 0.05]], [[0.04, 0.04], [0.07, 0.07]]]  # (L, M, H1)
v: [[0.0001, 0.0001], [null, null]]
support: [[0.05, 0.04], [null, null]]
cgf: [[null, null], [0.22, 0.38]]       # at s_chernoff = 5
s_chernoff: [[null, null], [null, null], [null, 5.0]]

sigma: [[null, null], [0.03, null], [null, null]]
eta: [[null, 0.15], [null, null], [null, null]]
v_max_user: [[null, null], [0.05, null], [null, null]]
gamma_beta: [[null, null], [null, 20.0], [null, null]]
gamma_beta_new: [[null, null], [null, 20.0], [null, null]]  # no gamma_beta_0

C_M: 1.0
C_R: [2.0, 1.5]
C_rep: [10.0, 8.0]
C_D: 1.5
penalty_type: inf_norm
objective_mode: evaluation_total
evaluation_horizon: 30

verbose: 0
mip_gap: 0.05
time_limit: 600
fast: false
gurobi_params: {Threads: 4}
progress_interval_seconds: 10
relaxation_warm_start: false
"""


def _example(**changes):
    data = yaml.safe_load(EXAMPLE_YAML)
    data.update(changes)
    return data


def _small(model, bound=None, **changes):
    """Two trains, one mission, one component: a valid input of one model."""
    data = {"F": 2, "M": 1, "H": [1, 2], "model": model, "tau": 1.0, "epsilon": 0.1,
            "rho": 0.5, "mu_0": 0.1, "mu": 0.05, "C_M": 1.0, "C_R": 1.0, "C_rep": 1.0,
            "C_D": 1.0}
    if model == "gaussian":
        data["v"] = 0.0001
    elif model == "inverse_gaussian":
        data["eta"] = 0.15
    elif model == "wiener":
        data["sigma"] = 0.03
    elif model == "gamma":
        data["gamma_beta"] = 20.0
    elif model == "rainflow":
        data["bound_method"] = bound
        if bound == "cantelli":
            data["v"] = 0.0001
        elif bound == "hoeffding":
            data["support"] = 0.05
        elif bound == "bernstein":
            data.update(v=0.0001, support=0.05)
        elif bound == "chernoff":
            data.update(cgf=0.3, s_chernoff=5.0)
    data.update(changes)
    return data


def _holds_string(value):
    if isinstance(value, str):
        return True
    if isinstance(value, (list, tuple)):
        return any(_holds_string(item) for item in value)
    return False


def _write_input_hdf5(data, path):
    """Write an input mapping in the flat HDF5 layout of the specification."""
    with h5py.File(path, "w") as f:
        for key, value in data.items():
            if key == "H" and isinstance(value, list):
                f.attrs[key] = np.asarray(value, dtype=np.int64)
            elif isinstance(value, dict) or (isinstance(value, list) and _holds_string(value)):
                f.attrs[key] = json.dumps(value)
            elif isinstance(value, list):
                f.create_dataset(key, data=np.array(value, dtype=float))
            else:
                f.attrs[key] = value


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------
def test_read_input_yaml_and_json_agree(tmp_path):
    (tmp_path / "in.yaml").write_text(EXAMPLE_YAML, encoding="utf-8")
    (tmp_path / "in.json").write_text(json.dumps(yaml.safe_load(EXAMPLE_YAML)),
                                      encoding="utf-8")
    assert read_input(tmp_path / "in.yaml") == read_input(tmp_path / "in.json")


def test_read_input_hdf5_layout(tmp_path):
    path = tmp_path / "in.h5"
    with h5py.File(path, "w") as f:
        f.attrs["F"] = 3
        f.attrs["H"] = np.asarray([2, 3], dtype=np.int64)
        f.attrs["penalty_type"] = "inf_norm"
        f.attrs["gurobi_params"] = '{"Threads": 4}'
        f.attrs["fast"] = False
        f.create_dataset("mu", data=np.array([[0.05, np.nan]]))
        f.create_dataset("bound_method", data=np.array([["", "cantelli"]], dtype=object),
                         dtype=h5py.string_dtype())
    data = read_input(path)
    assert data["F"] == 3 and type(data["F"]) is int
    assert data["H"] == [2, 3]
    assert data["penalty_type"] == "inf_norm"
    assert data["mu"][0][0] == 0.05 and np.isnan(data["mu"][0][1])
    assert data["bound_method"] == [["", "cantelli"]]
    assert data["gurobi_params"] == {"Threads": 4}
    assert data["fast"] is False


def test_read_input_rejects_extension(tmp_path):
    (tmp_path / "in.txt").write_text("F: 1", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported input file type"):
        read_input(tmp_path / "in.txt")


def test_read_input_rejects_non_mapping(tmp_path):
    (tmp_path / "in.yaml").write_text("- 1\n", encoding="utf-8")
    with pytest.raises(TypeError, match="one mapping"):
        read_input(tmp_path / "in.yaml")


def test_read_input_rejects_hdf5_group(tmp_path):
    path = tmp_path / "in.h5"
    with h5py.File(path, "w") as f:
        f.create_group("tau")
    with pytest.raises(ValueError, match="must be flat"):
        read_input(path)


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------
def test_rejected_keys():
    phrase_of = {
        "maintenance_type": "repair_model", "tail_bound": "bound_method",
        "method": "bound_method", "formulation": "reliability_impl",
        "beta": "gamma_beta", "alpha_inc": "gamma_beta", "b_inc": "support",
        "b_0": "support_0", "b_new": "support_new", "n_workers": "sweep",
        "warm_start": "sweep", "C_S": "C_D", "alpha": "tau", "xi": "rho",
        "repair_rho": "rho", "mu_new": "replacement_mu", "v_new": "replacement_v",
        "gamma_beta_bound": "Remove it", "gamma_calibration_method": "Remove it",
        "depot_capacity": "Remove it", "C_P": "Remove it",
    }
    assert len(phrase_of) == 21 and set(phrase_of) == set(config.REJECTED_KEYS)
    for key, phrase in phrase_of.items():
        with pytest.raises(ValueError, match=f"key '{key}' is not accepted") as info:
            load_config(_example(**{key: 1}))
        assert phrase in str(info.value)


def test_unknown_key():
    with pytest.raises(ValueError, match="unknown key") as info:
        load_config(_example(colour=1))
    assert "bound_method" in str(info.value)


def test_missing_required_keys():
    assert len(config.REQUIRED_KEYS) == 13
    for key in config.REQUIRED_KEYS:
        data = _example()
        del data[key]
        with pytest.raises(ValueError, match="missing required key"):
            load_config(data)


def test_key_check_runs_before_arrays():
    with pytest.raises(ValueError, match="'xi'"):
        load_config(_example(mu="abc", xi=0.5))


def test_input_must_be_a_mapping():
    with pytest.raises(TypeError, match="mapping"):
        load_config([1])


# ---------------------------------------------------------------------------
# Parsing, shapes and null
# ---------------------------------------------------------------------------
def test_example_loads():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_config(_example())
    assert not [w for w in caught if issubclass(w.category, UserWarning)]
    assert (cfg.F, cfg.M, cfg.L, cfg.H1, cfg.H2, cfg.T) == (3, 2, 2, 2, 3, 5)
    assert cfg.H == [2, 3]
    assert cfg.mu.shape == (3, 2, 2, 3) and cfg.mu_trans.shape == (3, 2, 2, 2)
    assert cfg.bound_method[0, 0] == "" and cfg.reliability_impl[0, 1] == "exact"
    assert np.isnan(cfg.v_0[0, 1])
    assert cfg.cgf_0[2, 1] == 5.0 * 0.14
    assert cfg.anchor_v_0[2, 0] == 0.0002
    assert cfg.tangent_ref[0, 0] == 0.4 and int(np.isnan(cfg.tangent_ref).sum()) == 5
    assert np.isnan(cfg.v_max_user[0, 0]) and cfg.v_max_user[1, 0] == 0.05
    assert cfg.C_M == 1.0
    assert (cfg.C_R == np.array([[2.0, 1.5]] * 3)).all()
    assert cfg.pwl_points == 6
    expected = dict(config.RUN_OPTION_DEFAULTS)
    expected.update(objective_mode="evaluation_total", evaluation_horizon=30, verbose=0,
                    mip_gap=0.05, time_limit=600, fast=False, gurobi_params={"Threads": 4},
                    progress_interval_seconds=10, relaxation_warm_start=False)
    assert cfg.options == expected


def test_example_three_formats(tmp_path):
    data = _example()
    (tmp_path / "in.yaml").write_text(EXAMPLE_YAML, encoding="utf-8")
    (tmp_path / "in.json").write_text(json.dumps(data), encoding="utf-8")
    _write_input_hdf5(data, tmp_path / "in.h5")
    cfgs = [load_config(read_input(tmp_path / name)) for name in ("in.yaml", "in.json", "in.h5")]
    for cfg in cfgs[1:]:
        for item in dataclasses.fields(cfg):
            first, other = getattr(cfgs[0], item.name), getattr(cfg, item.name)
            if isinstance(first, np.ndarray):
                np.testing.assert_array_equal(first, other)
            else:
                assert first == other, item.name


def test_profile_shapes():
    ref = 0.040 + 0.0001 * np.arange(36).reshape(3, 2, 2, 3)
    cases = ((ref[0, 0, :, 0], ref[0, 0, :, 0][None, None, :, None]),
             (ref[0, :, :, 0], ref[0, :, :, 0][None, :, :, None]),
             (ref[0], ref[0][None]),
             (ref[..., 0], ref[..., 0][..., None]),
             (ref, ref))
    for given, full in cases:
        cfg = load_config(_example(mu=given.tolist()))
        np.testing.assert_array_equal(cfg.mu, np.broadcast_to(full, ref.shape))
    with pytest.raises(ValueError, match="'mu' shape"):
        load_config(_example(mu=[0.05] * 4))


def test_cell_value_shapes():
    for tau in (1.0, [1.0, 1.0], [[1.0, 1.0]] * 3):
        assert load_config(_example(tau=tau)).tau.shape == (3, 2)
    with pytest.raises(ValueError, match="'tau' shape"):
        load_config(_example(tau=[1.0] * 3))


def test_cost_shapes():
    for C_R in (2.0, [2.0, 1.5], [[2.0, 1.5]] * 3):
        assert load_config(_example(C_R=C_R)).C_R.shape == (3, 2)
    assert (load_config(_example(C_R=2.0)).C_R == 2.0).all()
    with pytest.raises(ValueError, match="'C_M' must be a number"):
        load_config(_example(C_M=[1.0, 1.0]))


def test_gamma_beta_shapes():
    for beta in (20.0, [20.0], [[20.0], [20.0]], np.full((2, 1, 1), 20.0).tolist(),
                 np.full((2, 1, 1, 2), 20.0).tolist()):
        assert load_config(_small("gamma", gamma_beta=beta)).gamma_beta.shape == (2, 1, 1, 2)
    cfg = load_config(_small("gamma", gamma_beta_trans=20.0))
    assert cfg.gamma_beta_trans.shape == (2, 1, 1, 1)


def test_null_where_key_applies():
    with pytest.raises(ValueError, match="'tau' is null at cell \\(i=0, l=1\\)"):
        load_config(_example(tau=[[1.0, None], [1.0, 1.0], [1.0, 1.0]]))


def test_value_where_key_does_not_apply_warns():
    with pytest.warns(UserWarning, match="'sigma' has values at 1 cell"):
        cfg = load_config(_example(sigma=[[0.03, None], [0.03, None], [None, None]]))
    assert np.isnan(cfg.sigma[0, 0])


def test_broadcast_value_does_not_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_config(_example(sigma=0.03))
    assert not [w for w in caught if issubclass(w.category, UserWarning)]
    assert cfg.sigma[1, 0] == 0.03 and int(np.isnan(cfg.sigma).sum()) == 5


def test_profile_value_where_key_does_not_apply_warns():
    with pytest.warns(UserWarning, match="'v' has values"):
        load_config(_example(v=np.full((3, 2, 2), 0.0001).tolist()))


def test_non_finite_and_non_numeric():
    with pytest.raises(ValueError, match="finite"):
        load_config(_example(rho=float("inf")))
    for rho in ("high", "0.5", True, [0.8, "0.7"], [[0.8, 0.7], [0.8, False], [0.8, 0.7]]):
        with pytest.raises(ValueError, match="'rho' must be a number"):
            load_config(_example(rho=rho))
    with pytest.raises(ValueError, match="'mu' must be a number"):
        load_config(_example(mu=[["0.06", 0.05], [0.04, 0.07]]))


def test_selector_defaults():
    cfg = load_config(_small("gaussian"))
    assert (cfg.repair_model == "ardinf").all() and (cfg.encoding == "bigm").all()
    assert (cfg.reliability_impl == "exact").all()
    data = _small("rainflow", "cantelli")
    del data["bound_method"]
    assert (load_config(data).bound_method == "cantelli").all()


def test_selector_values():
    for changes, phrase in (
            ({"model": "weibull"}, "'model' has the value 'weibull'"),
            ({"repair_model": "ARD1"}, "'repair_model' has the value 'ARD1'"),
            ({"encoding": "big"}, "'encoding' has the value 'big'"),
            ({"penalty_type": "l2"}, "'penalty_type' must be one of"),
            ({"bound_method": [["foo", None], [None, None], ["bernstein", "chernoff"]]},
             "'bound_method' has the value 'foo'"),
            ({"reliability_impl": [["tangent", "lp"], ["pwl", None], ["exact", None]]},
             "'reliability_impl' has the value 'lp'")):
        with pytest.raises(ValueError, match=phrase):
            load_config(_example(**changes))


def test_component_names():
    data = _example()
    del data["component_names"]
    assert load_config(data).component_names == ["Component 1", "Component 2"]
    for names in (["a"], ["a", " "], ["a", 5], "ab"):
        with pytest.raises(ValueError, match="'component_names'"):
            load_config(_example(component_names=names))


def test_cell_type_required_keys():
    for model, bound, key in (
            ("gaussian", None, "v"), ("inverse_gaussian", None, "eta"),
            ("wiener", None, "sigma"), ("gamma", None, "gamma_beta"),
            ("rainflow", "hoeffding", "support"), ("rainflow", "bernstein", "support"),
            ("rainflow", "chernoff", "cgf"), ("rainflow", "chernoff", "s_chernoff")):
        data = _small(model, bound)
        del data[key]
        with pytest.raises(ValueError, match=f"missing required key '{key}'"):
            load_config(data)


def test_transitory_needs_operating():
    with pytest.raises(ValueError, match="'v_trans' needs"):
        load_config(_small("inverse_gaussian", v_trans=0.0001))


def test_sizes():
    for changes, phrase in (({"M": 2}, "'F' must be > 'M'"), ({"F": 0}, "'F' and 'M'"),
                            ({"F": 2.5}, "'F' must be an integer"),
                            ({"F": True}, "'F' must be an integer"),
                            ({"H": 0}, "'H' must be >= 1"), ({"H": [1]}, "'H' must be"),
                            ({"H": [1, 0]}, "'H' entries"), ({"H": 1.5}, "'H' must be an integer")):
        with pytest.raises(ValueError, match=phrase):
            load_config(_small("gaussian", **changes))
    assert load_config(_small("gaussian", H=2.0)).H == 2
    with pytest.raises(ValueError, match="'L'"):
        load_config(_example(L=3))


def test_run_options():
    assert load_config(_small("gaussian")).options == config.RUN_OPTION_DEFAULTS
    for changes, key in (
            ({"objective_mode": "avg"}, "objective_mode"),
            ({"objective_mode": "evaluation_total"}, "evaluation_horizon"),
            ({"evaluation_horizon": 1}, "evaluation_horizon"),
            ({"evaluation_horizon": 2.5}, "evaluation_horizon"),
            ({"mip_gap": -0.1}, "mip_gap"), ({"time_limit": 0}, "time_limit"),
            ({"progress_interval_seconds": 0}, "progress_interval_seconds"),
            ({"relaxation_time_limit": 0}, "relaxation_time_limit"),
            ({"gurobi_params": [1]}, "gurobi_params"), ({"validation": "x"}, "validation"),
            ({"fast": "yes"}, "fast"), ({"relaxation_warm_start": 1}, "relaxation_warm_start"),
            ({"verbose": 2}, "verbose"), ({"allow_replacement": "no"}, "allow_replacement"),
            ({"replacement_as_new": 0}, "replacement_as_new")):
        with pytest.raises(ValueError, match=f"'{key}'"):
            load_config(_small("gaussian", **changes))


def test_cost_values():
    for changes, key in (({"C_M": -1.0}, "C_M"), ({"C_D": -1.0}, "C_D"),
                         ({"C_D": float("inf")}, "C_D"), ({"C_R": [[-1.0], [1.0]]}, "C_R")):
        with pytest.raises(ValueError, match=f"'{key}'"):
            load_config(_small("gaussian", **changes))


# ---------------------------------------------------------------------------
# State checks and derived fields
# ---------------------------------------------------------------------------
def _fails(model, bound, key, rule="", **changes):
    """``load_config`` of a small input raises a ValueError that names ``key`` (and ``rule``)."""
    with pytest.raises(ValueError, match=f"'{key}' {re.escape(rule)}"):
        load_config(_small(model, bound, **changes))


def test_tau_epsilon_rho():
    _fails("gaussian", None, "tau", tau=0.0)
    _fails("gaussian", None, "epsilon", epsilon=0.0)
    _fails("gaussian", None, "epsilon", epsilon=1.0)
    with pytest.raises(ValueError, match="must be < 0.5"):
        load_config(_small("gaussian", epsilon=0.5))
    _fails("gaussian", None, "rho", rho=0.0)
    _fails("gaussian", None, "rho", rho=1.5)
    load_config(_small("rainflow", "cantelli", epsilon=0.6))
    load_config(_small("gaussian", rho=1.0))


def test_mu_0_range():
    _fails("gaussian", None, "mu_0", mu_0=1.0)
    _fails("gaussian", None, "mu_0", mu_0=-0.1)


def test_nonnegative_values():
    _fails("rainflow", "bernstein", "support_0", support_0=-0.01)
    _fails("rainflow", "hoeffding", "R_0", R_0=-0.01)
    _fails("gaussian", None, "replacement_v", replacement_v=-0.0001)
    load_config(_small("gaussian", replacement_v=-0.0001, allow_replacement=False))


def test_tangent_ref_and_pwl_points():
    _fails("gaussian", None, "tangent_ref", reliability_impl="tangent", tangent_ref=1.0)
    load_config(_small("gaussian", reliability_impl="exact", tangent_ref=1.0))
    _fails("gaussian", None, "pwl_points", pwl_points=0)


def test_anchors():
    cfg = load_config(_small("gaussian", repair_model="ard1", v_0=0.0001))
    assert (cfg.anchor_mu_0 == cfg.mu_0).all() and (cfg.anchor_v_0 == cfg.v_0).all()
    _fails("gaussian", None, "anchor_mu_0", repair_model="ard1", anchor_mu_0=0.2)
    _fails("gaussian", None, "anchor_mu_0", repair_model="ard1", anchor_mu_0=-0.1)
    _fails("gaussian", None, "anchor_v_0", repair_model="ard1", v_0=0.0001, anchor_v_0=0.001)
    _fails("rainflow", "hoeffding", "anchor_R_0", repair_model="ard1", anchor_R_0=0.01)
    _fails("rainflow", "chernoff", "anchor_K_0", repair_model="ard1", anchor_K_0=0.6)
    load_config(_small("gaussian", anchor_mu_0=0.2))


def test_replacement_as_new():
    _fails("gaussian", None, "replacement_mu", replacement_as_new=True, replacement_mu=0.02)
    cfg = load_config(_small("rainflow", "chernoff", replacement_as_new=True))
    assert (cfg.cgf_new == 0).all()
    _fails("rainflow", "chernoff", "cgf_new", replacement_as_new=True, cgf_new=0.1)
    # the flag is the only condition (spec "Consistency Checks"), also without replacements
    _fails("gaussian", None, "replacement_mu", replacement_as_new=True, replacement_mu=0.02,
           allow_replacement=False)


def test_quadratic_penalty():
    _fails("gaussian", None, "reliability_impl", penalty_type="quadratic",
           reliability_impl="tangent")
    load_config(_small("gaussian", penalty_type="quadratic", reliability_impl="tangent",
                       C_D=0.0))
    load_config(_small("inverse_gaussian", penalty_type="quadratic",
                       reliability_impl="tangent"))


def test_derived_fields():
    cfg = load_config(_example())
    close = dict(rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(cfg.range_v[0, 0], 0.3696115094681951, **close)
    np.testing.assert_allclose(cfg.range_v[2, 0], 0.20048057428495922, **close)
    np.testing.assert_allclose(cfg.range_K[2, 1], 2.697414907005954, **close)
    assert np.isnan(cfg.range_R).all() and np.isnan(cfg.range_v[0, 1])
    assert (cfg.range_mu == cfg.tau).all()
    assert cfg.support_const[2, 0] == 0.05 and int(np.isnan(cfg.support_const).sum()) == 5
    support_trans = [[[0.06, 0.06], [0.06, 0.06]], [[None, None], [None, None]]]
    support_new = [[None, None], [None, None], [0.08, None]]
    assert load_config(_example(support_trans=support_trans)).support_const[2, 0] == 0.06
    cfg = load_config(_example(support_trans=support_trans, support_new=support_new))
    assert cfg.support_const[2, 0] == 0.08
    cfg = load_config(_example(support_trans=support_trans, support_new=support_new,
                               allow_replacement=False))
    assert cfg.support_const[2, 0] == 0.06


# ---------------------------------------------------------------------------
# Increments
# ---------------------------------------------------------------------------
def test_increment_mean_range():
    _fails("gaussian", None, "mu", mu=-0.01)
    _fails("gaussian", None, "mu", mu=1.0)
    load_config(_small("inverse_gaussian", mu=0.0))


def test_zero_mean_needs_zero_spread():
    with pytest.raises(ValueError, match="'v' must be 0 where"):
        load_config(_small("gaussian", mu=0.0))
    load_config(_small("gaussian", mu=0.0, v=0.0))
    _fails("rainflow", "hoeffding", "support", mu=0.0)
    _fails("rainflow", "chernoff", "cgf", mu=0.0)


def test_negative_spread():
    _fails("rainflow", "cantelli", "v", v=-0.0001)
    _fails("rainflow", "hoeffding", "support", support=-0.01)
    _fails("rainflow", "chernoff", "cgf", cgf=-0.1)


def test_lookup_pairs_transitory_mean_with_operating_variance():
    # day 1 takes mu from mu_trans and v from the operating profile
    with pytest.raises(ValueError, match="'v' must be 0 where"):
        load_config(_small("gaussian", mu_trans=0.0))


def test_gaussian_three_sigma_increments():
    _fails("gaussian", None, "v", mu=0.02)


def test_bernstein_popoviciu_increments():
    _fails("rainflow", "bernstein", "v", v=0.001)


def test_chernoff_cgf_increments():
    _fails("rainflow", "chernoff", "cgf", cgf=0.2)


def test_tolerance():
    # every non-strict inequality passes up to 1e-12 * max(1, |a|, |b|) (spec "Consistency Checks")
    assert 5.0 * 0.14 != 0.7
    load_config(_small("rainflow", "chernoff", mu_0=0.14, cgf_0=0.7))
    load_config(_small("gaussian", v_0=-1e-13, rho=1.0 + 1e-13, C_M=-1e-13))
    assert load_config(_small("gaussian", mip_gap=-1e-13)).options["mip_gap"] == 0.0
    _fails("rainflow", "hoeffding", "R_0", "must be >= 0", R_0=-1e-11)
    _fails("gaussian", None, "rho", "must be in (0, 1]", rho=1.0 + 1e-11)


# ---------------------------------------------------------------------------
# Model checks
# ---------------------------------------------------------------------------
def test_gaussian_three_sigma_states():
    _fails("gaussian", None, "mu_0", mu_0=0.02, v_0=0.0001)
    _fails("gaussian", None, "replacement_mu", replacement_mu=0.01, replacement_v=0.0001)
    load_config(_small("gaussian", replacement_mu=0.01, replacement_v=0.0001,
                       allow_replacement=False))


def test_wiener_inverse_gaussian_constants():
    _fails("inverse_gaussian", None, "eta", eta=0.0)
    _fails("wiener", None, "sigma", sigma=0.0)
    load_config(_small("wiener", v_0=0.0))


def test_gamma_rates():
    _fails("gamma", None, "gamma_beta", gamma_beta=0.0)
    _fails("gamma", None, "gamma_beta_trans", gamma_beta_trans=float("inf"))
    _fails("gamma", None, "gamma_beta_0", gamma_beta_0=0.0)
    _fails("gamma", None, "gamma_beta_new", gamma_beta_new=-1.0)
    cfg = load_config(_small("gamma", mu_0=0.1))
    assert np.isnan(cfg.gamma_beta_0).all() and np.isnan(cfg.gamma_beta_new).all()


def test_v_max_user():
    _fails("gaussian", None, "v_max_user", v_max_user=0.0)
    _fails("gaussian", None, "v_max_user", v_0=0.0001, v_max_user=0.00005)
    changes = dict(replacement_mu=0.05, replacement_v=0.0001, v_max_user=0.00005)
    _fails("gaussian", None, "v_max_user", **changes)
    load_config(_small("gaussian", allow_replacement=False, **changes))


def test_bernstein_states_and_tau():
    _fails("rainflow", "bernstein", "v_0", v_0=0.0001)
    load_config(_small("rainflow", "bernstein", v_0=0.0001, support_0=0.05))
    _fails("rainflow", "bernstein", "replacement_v", replacement_v=0.0001)
    load_config(_small("rainflow", "bernstein", replacement_v=0.0001, support_new=0.05))
    _fails("rainflow", "bernstein", "tau", support=1.0)


def test_bernstein_tangent_and_pwl_conditions():
    # sqrt(tau (tau - 2c)) = 0.48215 for support 0.5, tau = 1, epsilon = 0.1
    _fails("rainflow", "bernstein", "tangent_ref", support=0.5, reliability_impl="tangent",
           tangent_ref=0.6)
    load_config(_small("rainflow", "bernstein", support=0.5, reliability_impl="tangent",
                       tangent_ref=0.4))
    _fails("rainflow", "bernstein", "pwl_points", support=0.5, reliability_impl="pwl",
           pwl_points=1)
    load_config(_small("rainflow", "bernstein", support=0.5, reliability_impl="pwl",
                       pwl_points=2))


def test_chernoff_conditions():
    _fails("rainflow", "chernoff", "s_chernoff", s_chernoff=2.0)
    _fails("rainflow", "chernoff", "cgf_0", cgf_0=0.4)
    _fails("rainflow", "chernoff", "anchor_K_0", repair_model="ard1", anchor_mu_0=0.1,
           anchor_K_0=0.4)
    _fails("rainflow", "chernoff", "cgf_new", replacement_mu=0.02, cgf_new=0.0)


# ---------------------------------------------------------------------------
# Natural ranges and replacement states
# ---------------------------------------------------------------------------
def test_natural_ranges():
    in_range = "must lie in [0, M]"
    _fails("wiener", None, "v_0", in_range, v_0=0.7)                    # V_max = 0.60887
    _fails("rainflow", "cantelli", "v_0", in_range, v_0=0.2)            # 0.11111
    _fails("rainflow", "hoeffding", "R_0", in_range, R_0=1.0)           # 0.86859
    _fails("rainflow", "chernoff", "cgf_0", in_range, cgf_0=3.0)        # 2.69741
    _fails("gaussian", None, "replacement_mu", in_range, replacement_mu=1.5)
    load_config(_small("gaussian", replacement_mu=1.5, allow_replacement=False))


def test_replacement_rows():
    # cap of the reliability row at the replacement mean, for tau = 1 and epsilon = 0.1
    row = "must satisfy the reliability row on a replacement day"
    _fails("gaussian", None, "replacement_v", row, replacement_mu=0.9,
           replacement_v=0.01)                                                          # 0.0060887
    load_config(_small("gaussian", replacement_mu=0.9, replacement_v=0.005))
    changes = dict(reliability_impl="tangent", tangent_ref=0.4, replacement_mu=0.9,
                   replacement_v=0.0001)                                                # -0.14613
    _fails("gaussian", None, "replacement_v", row, **changes)
    load_config(_small("gaussian", **dict(changes, reliability_impl="exact")))
    changes = dict(reliability_impl="pwl", pwl_points=6, replacement_mu=0.95)
    _fails("wiener", None, "replacement_v", row, replacement_v=0.001, **changes)        # 0.00084566
    load_config(_small("wiener", replacement_v=0.0005, **changes))
    _fails("rainflow", "markov", "replacement_mu", "must be <= epsilon*tau",
           replacement_mu=0.2)
    load_config(_small("rainflow", "markov", replacement_mu=0.05))
    _fails("rainflow", "cantelli", "replacement_v", row, replacement_mu=0.9,
           replacement_v=0.002)                                                         # 0.0011111
    _fails("rainflow", "hoeffding", "R_new", row, replacement_mu=0.9, R_new=0.01)       # 0.0086859
    _fails("rainflow", "bernstein", "replacement_mu", "must satisfy the side row",
           replacement_mu=0.97)                                                         # 0.96162


def test_replacement_rows_off():
    load_config(_small("rainflow", "markov", replacement_mu=0.2, allow_replacement=False))


def test_gamma_replacement_cap():
    # mean cap A_1(beta* tau, epsilon)/beta* at tau = 1, epsilon = 0.1: 0.744517 for beta* = 20,
    # 0.414912 for beta* = 2, 0.330617 for beta* = 0.2 (below the threshold t_1)
    cap = "must be <= the Gamma cap"
    _fails("gamma", None, "replacement_mu", cap, replacement_mu=0.7453)
    load_config(_small("gamma", replacement_mu=0.7437))
    _fails("gamma", None, "replacement_mu", cap, gamma_beta_new=2.0, replacement_mu=0.5)
    load_config(_small("gamma", gamma_beta_new=2.0, replacement_mu=0.41))
    load_config(_small("gamma", replacement_mu=0.5))
    load_config(_small("gamma", gamma_beta_new=2.0, replacement_mu=0.5, allow_replacement=False))
    # day 1 pairs the transitory rate with the operating mean (phase lookup)
    _fails("gamma", None, "replacement_mu", cap, gamma_beta_trans=2.0, replacement_mu=0.5)
    load_config(_small("gamma", gamma_beta_trans=2.0, mu_trans=0.0, replacement_mu=0.5))
    _fails("gamma", None, "replacement_mu", cap, gamma_beta_0=2.0, replacement_mu=0.5)
    load_config(_small("gamma", gamma_beta_0=2.0, mu_0=0.0, replacement_mu=0.5))
    _fails("gamma", None, "replacement_mu", cap, gamma_beta=0.2, replacement_mu=0.34)
    load_config(_small("gamma", gamma_beta=0.2, replacement_mu=0.32))
    # no term with a rate: only the natural range mu_new <= tau
    load_config(_small("gamma", mu=0.0, replacement_mu=0.9))
