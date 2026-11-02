"""Tests for fleet_management.solver: parsing, broadcasting, consistency
checks, and result I/O. None of these tests build or solve a Gurobi model.
"""

import copy

import numpy as np
import pytest
import yaml

from fleet_management import solver
from conftest import INPUT_DIR


# ======================================================================
# Broadcasting helpers
# ======================================================================

def test_broadcast_FL_scalar():
    out = solver._broadcast_FL(1.5, 3, 2, "x")
    assert out.shape == (3, 2)
    assert np.all(out == 1.5)


def test_broadcast_FL_exact_shape():
    raw = [[1.0, 2.0], [3.0, 4.0]]
    out = solver._broadcast_FL(raw, 2, 2, "x")
    assert np.array_equal(out, np.array(raw))


def test_broadcast_FL_bad_shape_raises():
    with pytest.raises(ValueError):
        solver._broadcast_FL([1.0, 2.0, 3.0], 2, 2, "x")


def test_broadcast_cost_FL_accepts_F_vector():
    out = solver._broadcast_cost_FL([1.0, 2.0], 2, 3, "C_R")
    assert out.shape == (2, 3)
    assert np.array_equal(out[0], [1.0, 1.0, 1.0])
    assert np.array_equal(out[1], [2.0, 2.0, 2.0])


def test_broadcast_increment_raw_3d_repeats_over_H():
    raw = np.ones((2, 1, 2))
    out = solver._broadcast_increment_raw(raw, F=2, M=1, L=2, H=4, name="mu")
    assert out.shape == (2, 1, 2, 4)
    assert np.all(out == 1.0)


def test_broadcast_increment_raw_4d_passthrough():
    raw = np.arange(2 * 1 * 2 * 3, dtype=float).reshape(2, 1, 2, 3)
    out = solver._broadcast_increment_raw(raw, F=2, M=1, L=2, H=3, name="mu")
    assert np.array_equal(out, raw)


def test_broadcast_FL_nullable_handles_none():
    raw = [[1.0, None], [None, 2.0]]
    out = solver._broadcast_FL_nullable(raw, 2, 2, "v_0")
    assert np.isnan(out[0, 1]) and np.isnan(out[1, 0])
    assert out[0, 0] == 1.0 and out[1, 1] == 2.0


def test_wrap_increment_periodic():
    F, M, L, H = 1, 1, 1, 3
    raw = np.array([[[[10.0, 20.0, 30.0]]]])
    wrapped = solver._wrap_increment(raw, F, M, L, H)
    assert wrapped.shape == (F, M, L, 2 * H)
    assert np.array_equal(wrapped[0, 0, 0], [10.0, 20.0, 30.0, 10.0, 20.0, 30.0])


def test_wrap_increment_slices_period_for_smaller_H():
    # Interval-H case: the raw array covers H_max=6; a solve at H=4 only uses
    # the first 4 entries of the period, doubled.
    F, M, L = 1, 1, 1
    raw = np.array([[[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]]])
    wrapped = solver._wrap_increment(raw, F, M, L, H_solve=4)
    assert wrapped.shape == (F, M, L, 8)
    assert np.array_equal(wrapped[0, 0, 0], [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])


# ======================================================================
# Parsing the example YAML files
# ======================================================================

@pytest.fixture
def base_data():
    with open(INPUT_DIR / "data_example.yaml") as f:
        return yaml.safe_load(f)


def test_parse_scalar_H_example(base_data):
    shared = solver._parse_and_validate(base_data)
    assert shared["H"] == 5
    assert shared["F"] == 4 and shared["M"] == 2 and shared["L"] == 2
    assert shared["tau"].shape == (4, 2)
    assert shared["mu_inc_raw"].shape == (4, 2, 2, 5)
    assert shared["v_inc_raw"].shape == (4, 2, 2, 5)
    assert np.isnan(shared["v_0"][0, 1])  # train0-comp1 is inverse_gaussian
    assert shared["v_0"][0, 0] == pytest.approx(0.0004)


def test_parse_interval_H_example():
    with open(INPUT_DIR / "data_example_loop.yaml") as f:
        data = yaml.safe_load(f)
    shared = solver._parse_and_validate(data)
    assert shared["H"] == [4, 6]
    assert shared["H_period"] == 6
    assert shared["warm_start"] is True


# ======================================================================
# Consistency checks (spec Section 3.6)
# ======================================================================

def _expect(base_data, mutator, exc_type):
    bad = copy.deepcopy(base_data)
    mutator(bad)
    with pytest.raises(exc_type):
        solver._parse_and_validate(bad)


def test_epsilon_out_of_range_raises(base_data):
    _expect(base_data, lambda d: d.update(epsilon=0.5), ValueError)
    _expect(base_data, lambda d: d.update(epsilon=0.0), ValueError)


def test_F_must_exceed_M(base_data):
    _expect(base_data, lambda d: d.update(M=10), ValueError)


def test_unknown_model_string_raises(base_data):
    _expect(base_data, lambda d: d["model"].__setitem__(0, ["bogus", d["model"][0][1]]), ValueError)


def test_bad_maintenance_type_string_raises(base_data):
    _expect(
        base_data,
        lambda d: d["maintenance_type"].__setitem__(0, ["bogus", d["maintenance_type"][0][1]]),
        ValueError,
    )


@pytest.mark.parametrize("model_name", ["gaussian", "inverse_gaussian", "rainflow"])
def test_ara1_rejected_for_ard1_only_models(model_name, make_tiny_problem):
    """Gaussian, IG, and Rainflow only support ARD1 -- ARA1 is a ValueError
    from their own validate_inputs (not a blanket solver-level rejection,
    since ARA1 is valid for wiener/gamma)."""
    data = make_tiny_problem([[model_name], [model_name]], F=2, M=1, L=1, H=2,
                              maintenance_type=[["ARA1"], ["ARD1"]])
    with pytest.raises(ValueError):
        solver._parse_and_validate(data)


@pytest.mark.parametrize("model_name", ["wiener", "gamma"])
def test_ara1_accepted_for_wiener_and_gamma(model_name, make_tiny_problem):
    data = make_tiny_problem([[model_name], [model_name]], F=2, M=1, L=1, H=2,
                              maintenance_type=[["ARA1"], ["ARD1"]])
    shared = solver._parse_and_validate(data)
    assert shared["maintenance_type"][0][0] == "ARA1"


def test_lp_and_quadratic_incompatible(base_data):
    _expect(base_data, lambda d: d.update(formulation="lp", penalty_type="quadratic"), ValueError)


def test_bad_formulation_raises(base_data):
    _expect(base_data, lambda d: d.update(formulation="bogus"), ValueError)


def test_bad_penalty_type_raises(base_data):
    _expect(base_data, lambda d: d.update(penalty_type="bogus"), ValueError)


def test_warm_start_requires_single_worker(base_data):
    _expect(base_data, lambda d: d.update(warm_start=True, n_workers=2), ValueError)


def test_mu_0_must_be_below_tau(base_data):
    _expect(base_data, lambda d: d["mu_0"].__setitem__(0, [1.5, 0.1]), ValueError)


def test_socp_alias_warns_and_maps_to_exact(base_data):
    data = copy.deepcopy(base_data)
    data["formulation"] = "socp"
    with pytest.warns(UserWarning):
        shared = solver._parse_and_validate(data)
    assert shared["formulation"] == "exact"


def test_H_interval_validation():
    with open(INPUT_DIR / "data_example_loop.yaml") as f:
        base = yaml.safe_load(f)
    _expect(base, lambda d: d.update(H=[6, 4]), ValueError)  # H_max <= H_min
    _expect(base, lambda d: d.update(H=[0, 4]), ValueError)  # H_min < 1
    _expect(base, lambda d: d.update(H=[4, 5, 6]), ValueError)  # wrong length


# ======================================================================
# Milestone-2 field requirements and consistency checks
# ======================================================================

def test_mu_not_required_for_all_gamma_fleet(make_tiny_problem):
    data = make_tiny_problem([["gamma"], ["gamma"]], F=2, M=1, L=1, H=2)
    assert "mu" not in data
    shared = solver._parse_and_validate(data)
    assert shared["mu_inc_raw"] is None


def test_mu_required_when_non_gamma_present(make_tiny_problem):
    data = make_tiny_problem([["gamma"], ["gaussian"]], F=2, M=1, L=1, H=2)
    del data["mu"]
    with pytest.raises(ValueError, match="'mu' is required"):
        solver._parse_and_validate(data)


def test_v_inc_not_required_for_wiener_only(make_tiny_problem):
    data = make_tiny_problem([["wiener"], ["wiener"]], F=2, M=1, L=1, H=2)
    assert "v" not in data
    shared = solver._parse_and_validate(data)
    assert shared["v_inc_raw"] is None


def test_wiener_requires_sigma(make_tiny_problem):
    data = make_tiny_problem([["wiener"], ["gaussian"]], F=2, M=1, L=1, H=2)
    del data["sigma"]
    with pytest.raises(ValueError, match="'sigma' is required"):
        solver._parse_and_validate(data)


def test_wiener_sigma_must_be_positive(make_tiny_problem):
    data = make_tiny_problem([["wiener"], ["gaussian"]], F=2, M=1, L=1, H=2)
    data["sigma"][0][0] = -0.1
    with pytest.raises(ValueError, match="sigma must be positive"):
        solver._parse_and_validate(data)


def test_wiener_v0_must_be_strictly_positive(make_tiny_problem):
    data = make_tiny_problem([["wiener"], ["gaussian"]], F=2, M=1, L=1, H=2)
    data["v_0"][0][0] = 0.0
    with pytest.raises(ValueError, match="v_0 must be positive"):
        solver._parse_and_validate(data)


def test_gamma_requires_beta_and_alpha_inc(make_tiny_problem):
    data = make_tiny_problem([["gamma"], ["gaussian"]], F=2, M=1, L=1, H=2)
    del data["beta"]
    with pytest.raises(ValueError, match="'beta' is required"):
        solver._parse_and_validate(data)

    data2 = make_tiny_problem([["gamma"], ["gaussian"]], F=2, M=1, L=1, H=2)
    del data2["alpha_inc"]
    with pytest.raises(ValueError, match="'alpha_inc' is required"):
        solver._parse_and_validate(data2)


def test_gamma_beta_must_be_positive(make_tiny_problem):
    data = make_tiny_problem([["gamma"], ["gaussian"]], F=2, M=1, L=1, H=2)
    data["beta"][0][0] = -0.5
    with pytest.raises(ValueError, match="beta must be positive"):
        solver._parse_and_validate(data)


def test_gamma_alpha_inc_must_be_positive(make_tiny_problem):
    data = make_tiny_problem([["gamma"], ["gaussian"]], F=2, M=1, L=1, H=2)
    data["alpha_inc"][0][0][0] = -0.1
    with pytest.raises(ValueError, match="alpha_inc must be positive"):
        solver._parse_and_validate(data)


def test_rainflow_tail_bound_defaults_to_cantelli(make_tiny_problem):
    data = make_tiny_problem([["rainflow"], ["gaussian"]], F=2, M=1, L=1, H=2)
    del data["tail_bound"]
    shared = solver._parse_and_validate(data)
    assert shared["tail_bound"][0, 0] == "cantelli"


def test_rainflow_bad_tail_bound_string_raises(make_tiny_problem):
    data = make_tiny_problem([["rainflow"], ["gaussian"]], F=2, M=1, L=1, H=2,
                              tail_bound="bogus")
    with pytest.raises(ValueError, match="tail_bound"):
        solver._parse_and_validate(data)


def test_tail_bound_at_non_rainflow_position_warns(make_tiny_problem):
    data = make_tiny_problem([["rainflow"], ["gaussian"]], F=2, M=1, L=1, H=2,
                              tail_bound=[["cantelli"], ["cantelli"]])
    with pytest.warns(UserWarning, match="non-rainflow"):
        solver._parse_and_validate(data)


def test_rainflow_bernstein_requires_b_fields(make_tiny_problem):
    data = make_tiny_problem([["rainflow"], ["gaussian"]], F=2, M=1, L=1, H=2,
                              tail_bound="bernstein")
    del data["b_inc"]
    with pytest.raises(ValueError, match="'b_inc' is required"):
        solver._parse_and_validate(data)


def test_rainflow_bernstein_bhatia_davis_v_inc_violation(make_tiny_problem):
    data = make_tiny_problem([["rainflow"], ["gaussian"]], F=2, M=1, L=1, H=2,
                              tail_bound="bernstein")
    data["v"][0][0][0] = 10.0  # way above mu_inc*(b_inc - mu_inc)
    with pytest.raises(ValueError, match="Bhatia-Davis"):
        solver._parse_and_validate(data)


def test_rainflow_bernstein_tau_precondition(make_tiny_problem):
    data = make_tiny_problem([["rainflow"], ["gaussian"]], F=2, M=1, L=1, H=2,
                              tail_bound="bernstein", b_inc=10.0, b_0=10.0)
    with pytest.raises(ValueError, match=r"tau must be > 2\*kappa\*b/3"):
        solver._parse_and_validate(data)


def test_v_max_user_applies_to_wiener_and_rainflow(make_tiny_problem):
    for model_name in ("wiener", "rainflow"):
        data = make_tiny_problem([[model_name], ["gaussian"]], F=2, M=1, L=1, H=2,
                                  v_max_user=0.5)
        shared = solver._parse_and_validate(data)
        assert shared["v_max_user"][0, 0] == 0.5


# ======================================================================
# Result I/O (YAML / JSON / HDF5), single- and multi-horizon
# ======================================================================

@pytest.mark.parametrize("ext", [".yaml", ".json", ".h5"])
def test_save_and_read_roundtrip_single_horizon(tmp_path, make_fake_result, make_fake_shared, ext):
    from fleet_management import plotter

    result = make_fake_result(F=2, M=1, L=2, H=3)
    shared = make_fake_shared(result)
    path = tmp_path / f"out{ext}"

    solver._save_results(result, path, shared)
    loaded = plotter._read_output(path)

    assert loaded["status"] == "optimal"
    assert (loaded["F"], loaded["M"], loaded["H"], loaded["L"]) == (2, 1, 3, 2)
    assert np.allclose(np.asarray(loaded["mu"]), result["mu"])
    assert np.allclose(np.asarray(loaded["tau"]), np.asarray(result["tau"]))
    assert loaded["model"][0][0] == "gaussian"
    assert loaded["model"][0][1] == "inverse_gaussian"


@pytest.mark.parametrize("ext", [".yaml", ".json", ".h5"])
def test_save_and_read_roundtrip_multi_horizon(tmp_path, make_fake_result, make_fake_shared, ext):
    from fleet_management import plotter

    r1 = make_fake_result(F=2, M=1, L=2, H=3, seed=0)
    r2 = make_fake_result(F=2, M=1, L=2, H=4, seed=1)
    multi_result = {3: r1, 4: r2}
    shared = make_fake_shared(r1, H=[3, 4])
    path = tmp_path / f"multi{ext}"

    solver._save_results(multi_result, path, shared)
    loaded = plotter._read_output(path)
    # JSON can't have int dict keys, so its round-trip yields string keys
    # ("3", "4"); normalize before indexing.
    by_h = {int(k): v for k, v in loaded.items()}

    assert set(by_h.keys()) == {3, 4}
    assert by_h[3]["H"] == 3 and by_h[4]["H"] == 4


def test_non_optimal_status_serializes_with_null_arrays(tmp_path):
    result = {
        "H": 3, "F": 2, "M": 1, "L": 1,
        "model": [["gaussian"], ["gaussian"]],
        "tail_bound": [[None], [None]],
        "tau": [[1.0], [1.0]],
        "status": "infeasible", "objective": None,
        "x": None, "x_m": None, "x_r": None, "mu": None, "v": None, "u": None, "z": None,
    }
    shared = {"H": 3, "epsilon": 0.01, "tau": np.ones((2, 1)), "rho": np.ones((2, 1)) * 0.5,
              "model_assignment": np.array(result["model"], dtype=object)}
    path = tmp_path / "out.yaml"
    solver._save_results(result, path, shared)
    with open(path) as f:
        loaded = yaml.safe_load(f)
    assert loaded["status"] == "infeasible"
    assert loaded["x"] is None
