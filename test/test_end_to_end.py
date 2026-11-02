"""End-to-end tests: full solve() -> save -> plot_management() pipeline
through the public API, for YAML/JSON/HDF5 input, scalar and interval
horizons, and both sequential/parallel horizon loops. Requires a Gurobi
license.
"""

import json

import h5py
import numpy as np
import pytest
import yaml

import matplotlib
matplotlib.use("Agg")

from fleet_management import plot_management, solve


def _write_yaml(path, data):
    with open(path, "w") as f:
        yaml.dump(data, f)


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def _write_hdf5_input(path, data):
    """Generic flat writer mirroring solver._read_hdf5_input's expectations."""
    with h5py.File(path, "w") as f:
        for key, val in data.items():
            if isinstance(val, bool):
                f.attrs[key] = val
            elif isinstance(val, (int, float, str)):
                f.attrs[key] = val
            else:
                try:
                    f.create_dataset(key, data=np.array(val, dtype=float))
                except (TypeError, ValueError):
                    f.create_dataset(key, data=np.array(val, dtype=object),
                                      dtype=h5py.string_dtype())


@pytest.fixture
def mixed_data(make_tiny_problem):
    return make_tiny_problem(
        [["gaussian", "inverse_gaussian"], ["gaussian", "gaussian"], ["inverse_gaussian", "gaussian"]],
        F=3, M=1, L=2, H=2,
    )


@pytest.fixture
def gaussian_only_data(make_tiny_problem):
    return make_tiny_problem([["gaussian"], ["gaussian"]], F=2, M=1, L=1, H=2)


def test_yaml_end_to_end_mixed(tmp_path, mixed_data):
    in_path = tmp_path / "in.yaml"
    _write_yaml(in_path, mixed_data)
    out_path = tmp_path / "out.yaml"

    result = solve(str(in_path), str(out_path))
    assert result["status"] == "optimal"
    assert out_path.exists()

    png_path = tmp_path / "schedule.png"
    plot_management(str(out_path), str(png_path))
    assert png_path.exists() and png_path.stat().st_size > 0


def test_json_end_to_end_gaussian_only(tmp_path, gaussian_only_data):
    in_path = tmp_path / "in.json"
    _write_json(in_path, gaussian_only_data)
    out_path = tmp_path / "out.json"

    result = solve(str(in_path), str(out_path))
    assert result["status"] == "optimal"
    with open(out_path) as f:
        saved = json.load(f)
    assert saved["status"] == "optimal"
    assert saved["F"] == 2 and saved["L"] == 1


def test_hdf5_end_to_end_gaussian_only(tmp_path, gaussian_only_data):
    in_path = tmp_path / "in.h5"
    _write_hdf5_input(in_path, gaussian_only_data)
    out_path = tmp_path / "out.h5"

    result = solve(str(in_path), str(out_path))
    assert result["status"] == "optimal"

    with h5py.File(out_path, "r") as f:
        assert f["metadata"].attrs["status"] == "optimal"
        assert f["solution"]["mu"].shape == (2, 1, 4)

    png_path = tmp_path / "schedule.png"
    plot_management(str(out_path), str(png_path))
    assert png_path.exists()


def test_solve_with_results_path_none_still_returns_dict(mixed_data, tmp_path):
    in_path = tmp_path / "in.yaml"
    _write_yaml(in_path, mixed_data)
    result = solve(str(in_path), results_path=None)
    assert result["status"] == "optimal"
    assert not (tmp_path / "output.yaml").exists()


def test_horizon_loop_sequential_warm_start(tmp_path, make_tiny_problem):
    data = make_tiny_problem(
        [["gaussian"], ["gaussian"]], F=2, M=1, L=1, H=[2, 3], warm_start=True,
    )
    in_path = tmp_path / "in.yaml"
    _write_yaml(in_path, data)
    out_path = tmp_path / "out.yaml"

    result = solve(str(in_path), str(out_path))
    assert set(result.keys()) == {2, 3}
    for h in (2, 3):
        assert result[h]["status"] == "optimal"
        assert result[h]["mu"].shape == (2, 1, 2 * h)

    prefix = tmp_path / "schedule"
    plot_management(str(out_path), str(prefix) + ".png")
    assert (tmp_path / "schedule_H2.png").exists()
    assert (tmp_path / "schedule_H3.png").exists()


def test_horizon_loop_parallel(tmp_path, make_tiny_problem):
    data = make_tiny_problem(
        [["gaussian"], ["gaussian"]], F=2, M=1, L=1, H=[2, 3], n_workers=2, warm_start=False,
    )
    in_path = tmp_path / "in.yaml"
    _write_yaml(in_path, data)

    result = solve(str(in_path), results_path=None)
    assert set(result.keys()) == {2, 3}
    assert all(result[h]["status"] == "optimal" for h in (2, 3))


def test_all_five_models_mixed_fleet_with_ara1_and_bernstein(tmp_path, make_tiny_problem):
    """Kitchen-sink Milestone-2 integration test: one train per model, ARA1
    for the wiener and gamma components, Bernstein tail bound for rainflow,
    solved and plotted end-to-end through the public API. F=6, M=2 gives
    each train ~2/3 of the days free for maintenance -- enough slack that
    every model's own reliability/loop constraints can be satisfied
    independently (see test_rainflow.py's solved_bernstein fixture comment
    for why a tight F/M ratio can make this genuinely infeasible)."""
    model_grid = [
        ["gaussian"], ["inverse_gaussian"], ["wiener"], ["wiener"], ["gamma"], ["rainflow"],
    ]
    maintenance_type = [
        ["ARD1"], ["ARD1"], ["ARD1"], ["ARA1"], ["ARA1"], ["ARD1"],
    ]
    data = make_tiny_problem(
        model_grid, F=6, M=2, L=1, H=3, mu_inc=0.07, v_inc=0.0003, rho=0.6,
        maintenance_type=maintenance_type, tail_bound="bernstein",
        b_inc=0.3, b_0=0.3, b_new=0.0,
    )
    in_path = tmp_path / "in.yaml"
    _write_yaml(in_path, data)
    out_path = tmp_path / "out.yaml"

    result = solve(str(in_path), str(out_path))
    assert result["status"] == "optimal"
    assert result["model"] == model_grid
    assert result["tail_bound"][5][0] == "bernstein"
    for i in range(6):
        if i not in (1, 4):  # IG and Gamma don't track variance
            assert not np.any(np.isnan(result["v"][i, 0]))
        else:
            assert np.all(np.isnan(result["v"][i, 0]))

    png_path = tmp_path / "schedule.png"
    plot_management(str(out_path), str(png_path))
    assert png_path.exists() and png_path.stat().st_size > 0
