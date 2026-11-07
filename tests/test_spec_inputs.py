"""Input reading and normalisation, checked against the specification (no Gurobi needed).

Every expected value comes from spec/spec.tex (Input Specification, lines 349-500; Input File
Schema, lines 2712-2828), not from the code. The location is cited in each docstring or comment.
Inputs are the files of tests/data/io/valid/; an invalid input is a valid mapping with one key
changed.
"""

import copy
import dataclasses
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml

from fleet_management.config import FleetConfig, load_config, read_input

DATA = Path(__file__).parent / "data" / "io" / "valid"
LIBRARY = ["gaussian", "gaussian_ard1_forms", "inverse_gaussian", "wiener", "gamma",
           "rainflow_bounds", "mixed_example", "minimal"]


def _base(name):
    with open(DATA / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _cfg(data):
    return load_config(data)


def _same_array(a, b):
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def _same_value(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        assert a is not None and b is not None
        _same_array(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _same_value(a[key], b[key])
    elif isinstance(a, float) and np.isnan(a):
        assert isinstance(b, float) and np.isnan(b)
    else:
        assert a == b


def _same_config(a, b):
    """Every field of two FleetConfig objects is equal (NaN equals NaN)."""
    for fld in dataclasses.fields(FleetConfig):
        _same_value(getattr(a, fld.name), getattr(b, fld.name))


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", LIBRARY)
def test_library_file_loads(name):
    """Every file of the library passes load_config (spec 2766 for the example)."""
    cfg = _cfg(_base(name))
    assert isinstance(cfg, FleetConfig)
    assert (cfg.F, cfg.M) == ((3, 2) if name == "mixed_example" else (3, 1))


def test_library_run_options():
    """The library files set verbose 0, mip_gap 0, time_limit 60 (plan, section 2)."""
    for name in ["gaussian", "gaussian_ard1_forms", "inverse_gaussian", "wiener", "gamma",
                 "rainflow_bounds"]:
        options = _cfg(_base(name)).options
        assert options["verbose"] == 0 and options["mip_gap"] == 0
        assert options["time_limit"] == 60


def test_library_example_sizes():
    """The YAML example: F=3, M=2, L=2, H=[2,3], T=5 (spec 2765, 2772)."""
    cfg = _cfg(_base("mixed_example"))
    assert (cfg.F, cfg.M, cfg.L, cfg.H1, cfg.H2, cfg.T) == (3, 2, 2, 2, 3, 5)
    assert cfg.component_names == ["Wheelset", "Brake pad"]


def test_library_rainflow_one_component_per_bound():
    """rainflow_bounds.yaml has one component per bound (plan, section 2)."""
    cfg = _cfg(_base("rainflow_bounds"))
    assert cfg.L == 5
    for i in range(cfg.F):
        assert list(cfg.bound_method[i]) == ["markov", "cantelli", "hoeffding", "bernstein",
                                             "chernoff"]


# ---------------------------------------------------------------------------
# Defaults (tables 366-496)
# ---------------------------------------------------------------------------
def test_defaults_selectors_and_names():
    """Defaults of the selector table (spec 381-386) on the minimal input."""
    cfg = _cfg(_base("minimal"))
    assert (cfg.F, cfg.L) == (3, 2)
    assert np.all(cfg.repair_model == "ardinf")
    assert np.all(cfg.reliability_impl == "exact")
    assert np.all(cfg.encoding == "bigm")
    assert cfg.component_names == ["Component 1", "Component 2"]


def test_default_bound_method_is_cantelli():
    """bound_method defaults to cantelli at a rainflow cell (spec 383)."""
    data = _base("minimal")
    data["model"] = ["rainflow", "rainflow"]
    cfg = _cfg(data)
    assert np.all(cfg.bound_method == "cantelli")


def test_default_tangent_ref_and_pwl_points():
    """tangent_ref 0.5 at tangent cells, pwl_points 8 (spec 394-395)."""
    data = _base("minimal")
    data["reliability_impl"] = "tangent"
    cfg = _cfg(data)
    assert np.all(cfg.tangent_ref == 0.5)
    assert cfg.pwl_points == 8
    data["reliability_impl"] = "pwl"
    assert _cfg(data).pwl_points == 8


def test_default_anchors_equal_initial_values_g():
    """Starting anchors default to the initial values at ard1 cells (spec 402-403)."""
    data = _base("minimal")
    data["repair_model"] = "ard1"
    data["v_0"] = 0.001
    cfg = _cfg(data)
    assert np.all(cfg.anchor_mu_0 == cfg.mu_0) and np.all(cfg.mu_0 == 0.1)
    assert np.all(cfg.anchor_v_0 == 0.001)
    data["v_0"] = [0.001, 0.0005]
    cfg = _cfg(data)
    _same_array(cfg.anchor_v_0, cfg.v_0)
    del data["v_0"]
    assert np.all(_cfg(data).anchor_v_0 == 0.0)


def test_default_anchors_equal_initial_values_rainflow():
    """anchor_R_0 = R_0 (Ho), anchor_K_0 = K_0 (Ch), anchor_v_0 = v_0 (Ca, Be) (spec 403-405)."""
    data = _base("rainflow_bounds")
    data["repair_model"] = "ard1"
    cfg = _cfg(data)
    ho, ca, be, ch = 2, 1, 3, 4
    assert np.all(cfg.anchor_R_0[:, ho] == cfg.R_0[:, ho]) and np.all(cfg.R_0[:, ho] == 0.001)
    assert np.all(cfg.anchor_v_0[:, [ca, be]] == 0.0005)
    assert np.all(cfg.anchor_K_0[:, ch] == cfg.cgf_0[:, ch])
    assert np.all(cfg.anchor_K_0[:, ch] == 5.0 * 0.10)
    mk = 0
    assert np.all(cfg.anchor_mu_0[:, mk] == 0.02) and np.all(cfg.anchor_mu_0[:, 1:] == 0.10)


def test_default_initial_state_rainflow():
    """v_0, support_0, R_0 default to 0; cgf_0 = s mu_0 (spec 412-415)."""
    data = _base("rainflow_bounds")
    for key in ["v_0", "support_0", "R_0"]:
        del data[key]
    cfg = _cfg(data)
    ca, be, ho, ch = 1, 3, 2, 4
    assert np.all(cfg.v_0[:, [ca, be]] == 0.0)
    assert np.all(cfg.support_0[:, be] == 0.0)
    assert np.all(cfg.R_0[:, ho] == 0.0)
    assert np.all(cfg.cgf_0[:, ch] == 5.0 * 0.10)


def test_default_initial_state_gaussian_wiener():
    """v_0 defaults to 0 at G and W cells (spec 412)."""
    data = _base("wiener")
    del data["v_0"]
    assert np.all(_cfg(data).v_0 == 0.0)
    data = _base("minimal")
    assert np.all(_cfg(data).v_0 == 0.0)


def test_default_replacement_state():
    """Replacement values default to 0; cgf_new = s mu_new (spec 421-425)."""
    data = _base("rainflow_bounds")
    for key in ["replacement_v", "support_new", "R_new"]:
        del data[key]
    cfg = _cfg(data)
    assert np.all(cfg.replacement_mu == 0.02)
    assert np.all(cfg.replacement_v[:, [1, 3]] == 0.0)
    assert np.all(cfg.support_new[:, 3] == 0.0)
    assert np.all(cfg.R_new[:, 2] == 0.0)
    assert np.all(cfg.cgf_new[:, 4] == 5.0 * 0.02)
    del data["replacement_mu"]
    cfg = _cfg(data)
    assert np.all(cfg.replacement_mu == 0.0)
    assert np.all(cfg.cgf_new[:, 4] == 0.0)
    assert np.all(_cfg(_base("minimal")).replacement_mu == 0.0)
    assert np.all(_cfg(_base("minimal")).replacement_v == 0.0)


def test_default_flags_and_penalty():
    """replacement_as_new false, allow_replacement true, penalty_type inf_norm (spec 426-427, 479)."""
    cfg = _cfg(_base("minimal"))
    assert cfg.replacement_as_new is False
    assert cfg.allow_replacement is True
    assert cfg.penalty_type == "inf_norm"


def test_default_run_options():
    """Defaults of the run options table (spec 485-495) on an input without them."""
    options = _cfg(_base("minimal")).options
    assert options["verbose"] == 1
    assert options["mip_gap"] == 0.1
    assert options["time_limit"] == 3600
    assert options["fast"] is False
    assert options["objective_mode"] == "total"
    assert options["progress_interval_seconds"] == 5.0
    assert options["relaxation_warm_start"] is False
    assert options["relaxation_time_limit"] == 60.0
    assert options.get("gurobi_params") is None
    assert options.get("evaluation_horizon") is None
    assert options.get("validation") is None


def test_default_gamma_rates_not_given():
    """gamma_beta_0 and gamma_beta_new have no default: NaN when omitted (spec 360, 468-469)."""
    data = _base("gamma")
    del data["gamma_beta_new"]
    del data["gamma_beta_0"]
    cfg = _cfg(data)
    assert np.all(np.isnan(cfg.gamma_beta_0)) and np.all(np.isnan(cfg.gamma_beta_new))


# ---------------------------------------------------------------------------
# Sizes (366-376)
# ---------------------------------------------------------------------------
def test_L_inferred_from_model_list():
    """L is the length of the component axis of model (spec 371)."""
    data = _base("minimal")
    assert _cfg(data).L == 2
    data["model"] = ["gaussian", "gaussian", "gaussian"]
    data.update(rho=0.8)
    assert _cfg(data).L == 3
    data["model"] = [["gaussian", "gaussian"]] * 3
    assert _cfg(data).L == 2


def test_L_required_for_one_model_string():
    """L is required when model is one string and L > 1 (spec 371)."""
    data = _base("gaussian")
    assert _cfg(data).L == 2
    del data["L"]
    with pytest.raises(ValueError):
        _cfg(data)


def test_L_one_when_model_string_without_L():
    """A single string model without L and with L = 1 data is a one-component fleet (spec 371)."""
    data = _base("minimal")
    data["model"] = "gaussian"
    cfg = _cfg(data)
    assert cfg.L == 1


def test_horizon_scalar():
    """A scalar H means H1 = H2 = H and T = 2H (spec 372, 375)."""
    data = _base("minimal")
    data["H"] = 3
    cfg = _cfg(data)
    assert (cfg.H1, cfg.H2, cfg.T) == (3, 3, 6)


def test_horizon_pair():
    """[H1, H2] gives T = H1 + H2 (spec 372, 375)."""
    data = _base("minimal")
    data["H"] = [2, 4]
    cfg = _cfg(data)
    assert (cfg.H1, cfg.H2, cfg.T) == (2, 4, 6)


# ---------------------------------------------------------------------------
# Cell values and selectors (354-357)
# ---------------------------------------------------------------------------
def test_cell_value_shapes_agree():
    """A scalar, a list of L numbers and an F x L array give the same F x L array (spec 354)."""
    data = _base("minimal")
    data["rho"] = 0.7
    scalar = _cfg(data).rho
    data["rho"] = [0.7, 0.7]
    lst = _cfg(data).rho
    data["rho"] = [[0.7, 0.7]] * 3
    grid = _cfg(data).rho
    assert scalar.shape == (3, 2)
    _same_array(scalar, np.full((3, 2), 0.7))
    _same_array(lst, scalar)
    _same_array(grid, scalar)


def test_cell_value_list_copied_to_every_train():
    """A list of L numbers is one per component, copied to every train (spec 354)."""
    data = _base("minimal")
    data["rho"] = [0.8, 0.7]
    _same_array(_cfg(data).rho, np.tile([0.8, 0.7], (3, 1)))


def test_cell_value_grid_kept():
    """An F x L array is read cell by cell (spec 354)."""
    data = _base("minimal")
    grid = [[0.8, 0.7], [0.6, 0.5], [0.9, 0.4]]
    data["rho"] = grid
    _same_array(_cfg(data).rho, grid)


@pytest.mark.parametrize("bad", [[0.8, 0.7, 0.6], [[0.8], [0.7], [0.6]], [[0.8, 0.7]]])
def test_cell_value_no_other_broadcasting(bad):
    """No other broadcasting is accepted: F entries, an (F, 1) array, a (1, L) array (spec 357)."""
    data = _base("minimal")
    data["rho"] = bad
    with pytest.raises(ValueError):
        _cfg(data)


def test_selector_shapes_agree():
    """A selector has the same shapes as a cell value (spec 355)."""
    data = _base("minimal")
    data["repair_model"] = "ard1"
    one = _cfg(data).repair_model
    data["repair_model"] = ["ard1", "ard1"]
    lst = _cfg(data).repair_model
    data["repair_model"] = [["ard1", "ard1"]] * 3
    grid = _cfg(data).repair_model
    assert one.shape == (3, 2)
    assert np.all(one == "ard1")
    _same_array(lst, one)
    _same_array(grid, one)
    data["repair_model"] = ["ard1", "ardinf"]
    cfg = _cfg(data)
    for i in range(3):
        assert list(cfg.repair_model[i]) == ["ard1", "ardinf"]
    data["repair_model"] = [["ard1", "ardinf"], ["ardinf", "ard1"], ["ard1", "ard1"]]
    cfg = _cfg(data)
    assert [list(row) for row in cfg.repair_model] == data["repair_model"]


def test_selector_no_other_broadcasting():
    """A selector list of F strings (F != L) is refused (spec 355, 357)."""
    data = _base("minimal")
    data["repair_model"] = ["ard1", "ard1", "ard1"]
    with pytest.raises(ValueError):
        _cfg(data)


# ---------------------------------------------------------------------------
# Profiles (432-436)
# ---------------------------------------------------------------------------
def _profile_input(**profiles):
    """A Gaussian input with F=3, M=2, L=2, H=[2, 3] and the given profiles."""
    data = _base("minimal")
    data["M"] = 2
    data["H"] = [2, 3]
    data.update(profiles)
    return data


def test_profile_scalar():
    """A scalar profile expands to (F, L, M, H2) (spec 434)."""
    cfg = _cfg(_profile_input(mu=0.2))
    assert cfg.mu.shape == (3, 2, 2, 3)
    assert np.all(cfg.mu == 0.2)


def test_profile_six_shapes_same_numbers():
    """The five array shapes of one profile expand to the same (F, L, M, H2) array (spec 434)."""
    per_mission = [0.2, 0.3]
    shapes = {
        "M": per_mission,
        "LM": [per_mission] * 2,
        "LMH": [[[e] * 3 for e in per_mission]] * 2,
        "FLM": [[per_mission] * 2] * 3,
        "FLMH": [[[[e] * 3 for e in per_mission]] * 2] * 3,
    }
    expected = np.broadcast_to(np.array(per_mission)[None, None, :, None], (3, 2, 2, 3))
    for name, arr in shapes.items():
        cfg = _cfg(_profile_input(mu=arr))
        assert cfg.mu.shape == (3, 2, 2, 3), name
        _same_array(cfg.mu, expected)


def test_profile_missing_axes_repeated():
    """A missing axis is repeated; present axes keep their numbers (spec 434)."""
    values = np.linspace(0.16, 0.45, 2 * 2).reshape(2, 2)           # (L, M)
    cfg = _cfg(_profile_input(mu=values.tolist()))
    _same_array(cfg.mu, np.broadcast_to(values[None, :, :, None], (3, 2, 2, 3)))
    days = np.linspace(0.16, 0.45, 2 * 2 * 3).reshape(2, 2, 3)      # (L, M, H2)
    cfg = _cfg(_profile_input(mu=days.tolist()))
    _same_array(cfg.mu, np.broadcast_to(days[None], (3, 2, 2, 3)))
    trains = np.linspace(0.16, 0.45, 3 * 2 * 2).reshape(3, 2, 2)    # (F, L, M)
    cfg = _cfg(_profile_input(mu=trains.tolist()))
    _same_array(cfg.mu, np.broadcast_to(trains[:, :, :, None], (3, 2, 2, 3)))
    full = np.linspace(0.16, 0.45, 3 * 2 * 2 * 3).reshape(3, 2, 2, 3)
    cfg = _cfg(_profile_input(mu=full.tolist()))
    _same_array(cfg.mu, full)
    per_mission = np.array([0.2, 0.35])
    cfg = _cfg(_profile_input(mu=per_mission.tolist()))
    _same_array(cfg.mu, np.broadcast_to(per_mission[None, None, :, None], (3, 2, 2, 3)))


def test_profile_ambiguous_shape_cannot_occur():
    """Spec 435: (L, M, H*) and (F, L, M) coincide only if F = L = M, which F > M excludes (366-369).

    Here the shapes (L, M, H*) = (2, 2, 3) and (F, L, M) = (3, 2, 2) are different, and each is
    read as its own shape.
    """
    cfg_a = _cfg(_profile_input(mu=np.full((2, 2, 3), 0.2).tolist()))
    cfg_b = _cfg(_profile_input(mu=np.full((3, 2, 2), 0.2).tolist()))
    _same_array(cfg_a.mu, cfg_b.mu)
    data = _profile_input(mu=0.2)
    data["F"] = 2
    data["M"] = 2
    with pytest.raises(ValueError):
        _cfg(data)


def test_profile_transitory_has_H1_entries():
    """A transitory profile has H1 entries per day axis; absent means None (spec 376, 433)."""
    assert _cfg(_profile_input(mu=0.2)).mu_trans is None
    values = np.linspace(0.16, 0.45, 2 * 2 * 2).reshape(2, 2, 2)    # (L, M, H1)
    cfg = _cfg(_profile_input(mu=0.2, mu_trans=values.tolist()))
    assert cfg.mu_trans.shape == (3, 2, 2, 2)
    _same_array(cfg.mu_trans, np.broadcast_to(values[None], (3, 2, 2, 2)))
    full = np.linspace(0.16, 0.45, 3 * 2 * 2 * 2).reshape(3, 2, 2, 2)
    cfg = _cfg(_profile_input(mu=0.2, mu_trans=full.tolist()))
    _same_array(cfg.mu_trans, full)
    cfg = _cfg(_profile_input(mu=0.2, mu_trans=0.25))
    assert np.all(cfg.mu_trans == 0.25) and cfg.mu_trans.shape == (3, 2, 2, 2)


def test_profile_v_expands_like_mu():
    """The profile v (G cells) is expanded like mu (spec 453)."""
    values = np.array([[0.001, 0.002], [0.0015, 0.0025]])
    cfg = _cfg(_profile_input(mu=0.3, v=values.tolist()))
    _same_array(cfg.v, np.broadcast_to(values[None, :, :, None], (3, 2, 2, 3)))


def _gamma_input(**keys):
    data = _base("gamma")
    data.update(keys)
    return data


def test_gamma_beta_shapes():
    """gamma_beta: scalar, (L), (F, L), (F, L, M), (F, L, M, H2) (spec 436, 457)."""
    shape = (3, 2, 1, 3)
    cfg = _cfg(_gamma_input(gamma_beta=20.0))
    assert cfg.gamma_beta.shape == shape and np.all(cfg.gamma_beta == 20.0)
    per_component = np.array([20.0, 25.0])
    cfg = _cfg(_gamma_input(gamma_beta=per_component.tolist()))
    _same_array(cfg.gamma_beta, np.broadcast_to(per_component[None, :, None, None], shape))
    per_cell = np.linspace(15.0, 30.0, 6).reshape(3, 2)
    cfg = _cfg(_gamma_input(gamma_beta=per_cell.tolist()))
    _same_array(cfg.gamma_beta, np.broadcast_to(per_cell[:, :, None, None], shape))
    per_mission = np.linspace(15.0, 30.0, 6).reshape(3, 2, 1)
    cfg = _cfg(_gamma_input(gamma_beta=per_mission.tolist()))
    _same_array(cfg.gamma_beta, np.broadcast_to(per_mission[:, :, :, None], shape))
    per_day = np.linspace(15.0, 30.0, 18).reshape(3, 2, 1, 3)
    cfg = _cfg(_gamma_input(gamma_beta=per_day.tolist()))
    _same_array(cfg.gamma_beta, per_day)


def test_gamma_beta_trans_shapes():
    """gamma_beta_trans: scalar, (L), (F, L), (F, L, M), (F, L, M, H1) (spec 436, 459)."""
    shape = (3, 2, 1, 2)
    assert _cfg(_gamma_input()).gamma_beta_trans is None
    cfg = _cfg(_gamma_input(gamma_beta_trans=21.0))
    assert cfg.gamma_beta_trans.shape == shape and np.all(cfg.gamma_beta_trans == 21.0)
    per_component = np.array([21.0, 26.0])
    cfg = _cfg(_gamma_input(gamma_beta_trans=per_component.tolist()))
    _same_array(cfg.gamma_beta_trans, np.broadcast_to(per_component[None, :, None, None], shape))
    per_cell = np.linspace(15.0, 30.0, 6).reshape(3, 2)
    cfg = _cfg(_gamma_input(gamma_beta_trans=per_cell.tolist()))
    _same_array(cfg.gamma_beta_trans, np.broadcast_to(per_cell[:, :, None, None], shape))
    per_mission = np.linspace(15.0, 30.0, 6).reshape(3, 2, 1)
    cfg = _cfg(_gamma_input(gamma_beta_trans=per_mission.tolist()))
    _same_array(cfg.gamma_beta_trans, np.broadcast_to(per_mission[:, :, :, None], shape))
    per_day = np.linspace(15.0, 30.0, 12).reshape(3, 2, 1, 2)
    cfg = _cfg(_gamma_input(gamma_beta_trans=per_day.tolist()))
    _same_array(cfg.gamma_beta_trans, per_day)


def test_gamma_beta_0_and_new_cells():
    """gamma_beta_0, gamma_beta_new: values at the given cells, NaN where null (spec 360, 468)."""
    cfg = _cfg(_base("gamma"))
    assert cfg.gamma_beta_0[0, 0] == 40.0
    assert np.isnan(cfg.gamma_beta_0).sum() == 5
    assert np.all(cfg.gamma_beta_new == 40.0)


# ---------------------------------------------------------------------------
# Integers (364)
# ---------------------------------------------------------------------------
def test_integer_with_zero_fraction_accepted():
    """F: 3.0 is accepted and read as 3 (spec 364)."""
    data = _base("minimal")
    data["F"] = 3.0
    data["M"] = 1.0
    cfg = _cfg(data)
    assert cfg.F == 3 and cfg.M == 1
    assert isinstance(cfg.F, int) and not isinstance(cfg.F, bool)
    data["H"] = [2.0, 3.0]
    assert _cfg(data).T == 5


@pytest.mark.parametrize("key,value", [("F", 2.5), ("F", True), ("M", 1.5), ("M", False),
                                       ("H", 2.5), ("H", True)])
def test_integer_other_numbers_and_flags_refused(key, value):
    """Any other number and a flag are refused at an integer key (spec 364)."""
    data = _base("minimal")
    data[key] = value
    with pytest.raises(ValueError):
        _cfg(data)


def test_integer_pwl_points():
    """pwl_points is an integer key (spec 395, 364)."""
    data = _base("minimal")
    data["reliability_impl"] = "pwl"
    data["pwl_points"] = 6.0
    assert _cfg(data).pwl_points == 6
    data["pwl_points"] = 6.5
    with pytest.raises(ValueError):
        _cfg(data)
    data["pwl_points"] = True
    with pytest.raises(ValueError):
        _cfg(data)


# ---------------------------------------------------------------------------
# null rules (359-362)
# ---------------------------------------------------------------------------
def test_null_accepted_where_key_does_not_apply():
    """null entries at cells where a key does not apply are accepted (spec 359)."""
    cfg = _cfg(_base("mixed_example"))
    ig = (0, 1)
    assert np.isnan(cfg.v_0[ig]) and np.isnan(cfg.sigma[ig])
    assert cfg.eta[ig] == 0.15
    data = _base("inverse_gaussian")
    data["v_0"] = [[None, None]] * 3
    cfg = _cfg(data)
    assert np.all(np.isnan(cfg.v_0))


def test_null_key_omitted_where_it_does_not_apply():
    """A key that applies nowhere may be omitted (spec 359)."""
    cfg = _cfg(_base("inverse_gaussian"))
    assert np.all(np.isnan(cfg.v_0)) and np.all(np.isnan(cfg.sigma))


@pytest.mark.parametrize("key,bad", [
    ("mu_0", [[None, 0.1], [0.1, 0.1], [0.1, 0.1]]),
    ("tau", [[1.0, 1.0], [None, 1.0], [1.0, 1.0]]),
    ("epsilon", [[0.05, None], [0.05, 0.05], [0.05, 0.05]]),
    ("rho", [[0.8, 0.8], [0.8, 0.8], [0.8, None]]),
    ("C_R", [[None, 2.0], [2.0, 2.0], [2.0, 2.0]]),
    ("v", [[[0.0025], [None]], [[0.0025], [0.0025]], [[0.0025], [0.0025]]]),
    ("repair_model", [[None, "ardinf"]] * 3),
    ("model", [["gaussian", None]] * 3),
])
def test_null_refused_where_key_applies(key, bad):
    """null where a key applies is an error (spec 359, 538)."""
    data = _base("gaussian")
    del data["component_names"]
    data["rho"] = 0.8
    data["C_R"] = 2.0
    data["mu_0"] = 0.1
    data["mu"] = 0.3
    data[key] = bad
    with pytest.raises(ValueError):
        _cfg(data)


def test_null_refused_at_rainflow_bound_method():
    """bound_method applies at rainflow cells: null there is an error (spec 383, 359)."""
    data = _base("rainflow_bounds")
    data["bound_method"] = [["markov", "cantelli", "hoeffding", "bernstein", None]] * 3
    with pytest.raises(ValueError):
        _cfg(data)


def test_null_means_not_given_at_v_max_user():
    """At v_max_user, null at a cell means not given: NaN (spec 360, 467)."""
    cfg = _cfg(_base("wiener"))
    assert cfg.v_max_user[0, 0] == 0.05
    assert np.isnan(cfg.v_max_user).sum() == 5
    data = _base("wiener")
    del data["v_max_user"]
    assert np.all(np.isnan(_cfg(data).v_max_user))


def test_null_means_not_given_at_gamma_rates():
    """At gamma_beta_0 and gamma_beta_new, null means not given (spec 360, 468-469)."""
    data = _base("gamma")
    data["gamma_beta_new"] = [[None, 40.0], [None, None], [40.0, None]]
    cfg = _cfg(data)
    assert np.isnan(cfg.gamma_beta_new[0, 0]) and cfg.gamma_beta_new[0, 1] == 40.0
    assert np.isnan(cfg.gamma_beta_new[1]).all() and cfg.gamma_beta_new[2, 0] == 40.0
    cfg = _cfg(_base("mixed_example"))
    assert cfg.gamma_beta_new[1, 1] == 20.0
    assert np.isnan(cfg.gamma_beta_0).all()


def test_non_null_entry_where_key_does_not_apply_warns():
    """A non-null F x L entry where the key does not apply: UserWarning, ignored (spec 362)."""
    data = _base("inverse_gaussian")
    data["v_0"] = [[0.001, 0.001]] * 3
    with pytest.warns(UserWarning):
        cfg = _cfg(data)
    assert np.all(np.isnan(cfg.v_0))


def test_non_null_profile_entry_where_key_does_not_apply_warns():
    """A profile with axes F and L, non-null at a cell where the key does not apply (spec 362)."""
    data = _base("inverse_gaussian")
    data["v"] = [[[0.0025], [0.0025]]] * 3
    with pytest.warns(UserWarning):
        cfg = _cfg(data)
    assert cfg.v is None or np.all(np.isnan(cfg.v))


def test_broadcast_value_where_key_does_not_apply_is_silent():
    """A scalar or a list is read only where the key applies, with no warning (spec 361)."""
    data = _base("inverse_gaussian")
    data["v_0"] = 0.001
    data["sigma"] = [0.03, 0.04]
    data["v"] = 0.0025
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = _cfg(data)
    assert not [w for w in caught if issubclass(w.category, UserWarning)]
    assert np.all(np.isnan(cfg.v_0)) and np.all(np.isnan(cfg.sigma))


def test_selector_entry_where_key_does_not_apply_is_checked():
    """bound_method: chernof at a Gaussian cell is an error (spec 362, 547)."""
    data = _base("gaussian")
    data["bound_method"] = [["chernof", None]] + [[None, None]] * 2
    with pytest.raises(ValueError):
        _cfg(data)
    data["bound_method"] = "chernof"
    with pytest.raises(ValueError):
        _cfg(data)


def test_valid_selector_entry_where_key_does_not_apply_warns():
    """A valid bound_method entry at a Gaussian cell is ignored with a UserWarning (spec 362)."""
    data = _base("gaussian")
    data["bound_method"] = [["cantelli", None]] + [[None, None]] * 2
    with pytest.warns(UserWarning):
        _cfg(data)


def test_reliability_impl_ignored_at_linear_row():
    """A reliability_impl value at an IG cell is ignored: form exact is built (spec 384, 548)."""
    data = _base("inverse_gaussian")
    data["reliability_impl"] = "pwl"
    cfg = _cfg(data)
    assert np.all(cfg.reliability_impl == "exact")
    data["reliability_impl"] = "linear"
    with pytest.raises(ValueError):
        _cfg(data)


def test_reliability_impl_kept_at_quadratic_row():
    """The forms tangent, pwl and exact of the YAML example are kept (spec 2778, 384)."""
    cfg = _cfg(_base("mixed_example"))
    assert cfg.reliability_impl[0, 0] == "tangent"
    assert cfg.reliability_impl[1, 0] == "pwl"
    assert cfg.reliability_impl[2, 0] == "exact"
    assert cfg.reliability_impl[0, 1] == "exact"


# ---------------------------------------------------------------------------
# Case sensitivity (363)
# ---------------------------------------------------------------------------
def test_selector_values_case_sensitive():
    """model: Gaussian is an error (spec 363)."""
    data = _base("minimal")
    data["model"] = ["Gaussian", "gaussian"]
    with pytest.raises(ValueError):
        _cfg(data)
    data = _base("minimal")
    data["repair_model"] = "ARD1"
    with pytest.raises(ValueError):
        _cfg(data)


def test_key_names_case_sensitive():
    """The key Tau is an error (spec 363)."""
    data = _base("minimal")
    data["Tau"] = data.pop("tau")
    with pytest.raises(ValueError):
        _cfg(data)


# ---------------------------------------------------------------------------
# Formats (2714-2743)
# ---------------------------------------------------------------------------
_SELECTORS = ("model", "repair_model", "bound_method", "reliability_impl", "encoding")


def _write_hdf5(data, path):
    """Write an input mapping as HDF5 by the layout of spec 2729-2734."""
    import h5py
    with h5py.File(path, "w") as f:
        for key, value in data.items():
            if key in _SELECTORS and isinstance(value, list):
                strings = np.array(value, dtype=object)
                strings = np.where(strings == None, "", strings)           # noqa: E711
                f.create_dataset(key, data=strings, dtype=h5py.string_dtype())
            elif key in ("component_names", "gurobi_params", "validation"):
                f.attrs[key] = json.dumps(value)
            elif key == "H" and isinstance(value, list):
                f.attrs[key] = np.array(value, dtype=np.int64)
            elif isinstance(value, list):
                f.create_dataset(key, data=np.array(value, dtype=np.float64))
            elif value is None:
                continue
            else:
                f.attrs[key] = value


def test_format_json_equals_yaml(tmp_path):
    """The YAML example written as JSON gives the same configuration (spec 2716, 2719)."""
    path = tmp_path / "example.json"
    path.write_text(json.dumps(_base("mixed_example")), encoding="utf-8")
    _same_config(_cfg(read_input(path)), _cfg(_base("mixed_example")))


@pytest.mark.parametrize("suffix", [".h5", ".hdf5"])
def test_format_hdf5_equals_yaml(tmp_path, suffix):
    """The YAML example written as HDF5 gives the same configuration (spec 2716, 2729-2734)."""
    pytest.importorskip("h5py")
    path = tmp_path / f"example{suffix}"
    _write_hdf5(_base("mixed_example"), path)
    _same_config(_cfg(read_input(path)), _cfg(_base("mixed_example")))


@pytest.mark.parametrize("suffix", [".yml", ".YAML", ".yaml"])
def test_format_yaml_extensions(tmp_path, suffix):
    """.yml and .yaml are YAML; the extension is read in lower case (spec 2716)."""
    path = tmp_path / f"example{suffix}"
    shutil.copy(DATA / "mixed_example.yaml", path)
    _same_config(_cfg(read_input(path)), _cfg(_base("mixed_example")))


def test_format_upper_case_other_extensions(tmp_path):
    """.JSON and .H5 are read in lower case too (spec 2716)."""
    path = tmp_path / "example.JSON"
    path.write_text(json.dumps(_base("minimal")), encoding="utf-8")
    assert _cfg(read_input(path)).F == 3
    pytest.importorskip("h5py")
    path = tmp_path / "example.H5"
    _write_hdf5(_base("minimal"), path)
    assert _cfg(read_input(path)).F == 3


def test_format_other_extension_refused(tmp_path):
    """Another extension raises an error (spec 2716)."""
    path = tmp_path / "example.txt"
    shutil.copy(DATA / "minimal.yaml", path)
    with pytest.raises(ValueError):
        read_input(path)


def test_format_yaml_safe_mode(tmp_path):
    """YAML is read in the safe mode: a Python object tag is refused (spec 2721)."""
    path = tmp_path / "unsafe.yaml"
    path.write_text("F: !!python/object/apply:os.getcwd []\n", encoding="utf-8")
    with pytest.raises(Exception):
        read_input(path)
