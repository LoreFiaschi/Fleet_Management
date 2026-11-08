"""Tests of the solve path and of the result files of ``fleet_management.solver``.

None of them needs Gurobi: ``base.solve_mixed`` is replaced by a stub.
"""

import inspect
import json
import sys
import types
from unittest import mock

import h5py
import numpy as np
import pytest
import yaml

from fleet_management import config, solver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _result(solution: bool = True) -> dict:
    """A result dictionary with the 59 keys of the spec, plus one key outside the table."""
    F, M, L, H1, H2, T = 2, 1, 2, 1, 2, 3
    nan = float("nan")
    arrays = np.arange(F * L * T, dtype=float).reshape(F, L, T)
    v = arrays / 100.0
    v[:, 1, :] = nan
    v_0 = np.full((F, L), 0.001)
    v_0[:, 1] = nan
    all_nan = np.full((F, L, T), nan)
    y_mu = np.array([[0.5, 0.25], [0.5, 0.25]])        # start of the repetition, (F, L)
    y_v = y_mu / 100.0
    y_v[:, 1] = nan
    y_nan = np.full((F, L), nan)
    result = {
        "status": "optimal",
        "objective": 12.5,
        "mip_gap": 0.01,
        "bound": 12.4,
        "F": F, "M": M, "L": L, "H1": H1, "H2": H2, "T": T,
        "H": [H1, H2],
        "model": np.array([["gaussian", "rainflow"]] * F, dtype=object),
        "component_names": ["Wheelset", "Brake pad"],
        "tau": np.full((F, L), 1.0),
        "mu_0": np.full((F, L), 0.1),
        "v_0": v_0,
        "bound_method": [[None, "cantelli"], [None, "cantelli"]],
        "repair_model": "ardinf",
        "reliability_impl": "exact",
        "encoding": [["bigm", "indicator"], ["bigm", "bigm"]],
        "objective_mode": "total",
        "evaluation_horizon": None,
        "x": np.zeros((F, M + 1, T)),
        "m": np.zeros((F, L, T)),
        "r": np.zeros((F, L, T)),
        "idle": np.ones((F, L, T)),
        "mu": arrays / 10.0,
        "v": v,
        "z": np.zeros((F, L, T)),
        "R": all_nan.copy(),
        "K": all_nan.copy(),
        "g_mu": all_nan.copy(),
        "g_v": all_nan.copy(),
        "g_R": all_nan.copy(),
        "g_K": all_nan.copy(),
        "y_mu": y_mu,
        "y_v": y_v,
        "y_R": y_nan.copy(),
        "y_K": y_nan.copy(),
        "y_g_mu": y_nan.copy(),
        "y_g_v": y_nan.copy(),
        "y_g_R": y_nan.copy(),
        "y_g_K": y_nan.copy(),
        "u": np.float64(1.2),
        "step_costs": np.array([1.0, 2.0, 3.0]),
        "J_initialization": 1.0,
        "J_op": 5.0,
        "J_op_average": 2.5,
        "peak_damage": 1.2,
        "damage_penalty": 1.2,
        "operating_objective": 3.7,
        "J_total": 12.5,
        "projected_evaluation_cost": None,
        "evaluation_objective": None,
        "component_costs": {
            "C_M": 1.0, "C_D": 1.0,
            "C_R": np.full((F, L), 0.5), "C_rep": np.full((F, L), 3.0),
        },
        "performance": {
            "gurobi_version": "13.0.0",
            "variables": 120,
            "solutions_found": None,
            "gurobi_runtime_seconds": np.float64(1.5),
        },
        "optimization_progress": [
            {
                "runtime_seconds": 0.5, "incumbent": None, "best_bound": 3.2,
                "relative_gap": None, "nodes": 0, "solutions": 0,
            },
        ],
        "warm_start": {
            "applied": True, "values": 40, "mode": "start", "source_horizon": [1, 2],
            "completion": {
                "requested": False, "status": "not_requested",
                "fixed_integer_variables": 0, "values_copied": 0,
                "runtime_seconds": 0.0, "iis_constraints": [],
            },
        },
        "relaxation_warm_start": {
            "enabled": False, "status": "disabled", "relaxation_objective": None,
            "values_copied": 0, "runtime_seconds": 0.0,
        },
        "extra_key": 1,
    }
    if not solution:
        for key in (
            "objective", "mip_gap", "bound",
            "x", "m", "r", "idle", "mu", "v", "z", "R", "K",
            "g_mu", "g_v", "g_R", "g_K",
            "y_mu", "y_v", "y_R", "y_K", "y_g_mu", "y_g_v", "y_g_R", "y_g_K", "u",
            "step_costs", "J_initialization", "J_op", "J_op_average",
            "peak_damage", "damage_penalty", "operating_objective", "J_total",
            "component_costs",
        ):
            result[key] = None
        result["status"] = "infeasible"
    return result


def _stub_base(result: dict, calls: list):
    """Replace ``fleet_management.degradation_model.base`` by a stub that returns ``result``."""
    def solve_mixed(cfg, *, warm_start=None, warm_start_mode="start"):
        calls.append((cfg, warm_start, warm_start_mode))
        return result

    package = types.ModuleType("fleet_management.degradation_model")
    package.base = types.SimpleNamespace(solve_mixed=solve_mixed)
    return mock.patch.dict(sys.modules, {"fleet_management.degradation_model": package})


# ---------------------------------------------------------------------------
# B1: the solve path
# ---------------------------------------------------------------------------
def test_results_path_checks(tmp_path) -> None:
    assert solver._check_results_path(None) is None
    assert solver._check_results_path(tmp_path / "out") == tmp_path / "out.yaml"
    assert solver._check_results_path(str(tmp_path / "a.JSON")) == tmp_path / "a.JSON"
    with pytest.raises(ValueError, match="Unsupported results file type '.txt'"):
        solver._check_results_path(tmp_path / "out.txt")
    with pytest.raises(FileNotFoundError, match="Results directory does not exist"):
        solver._check_results_path(tmp_path / "missing" / "out.yaml")
    with mock.patch("fleet_management.solver.os.access", return_value=False):
        with pytest.raises(PermissionError, match="not writable"):
            solver._check_results_path(tmp_path / "out.yaml")


def test_solve_checks_input_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Input file not found"):
        solver.solve(tmp_path / "missing.yaml", tmp_path / "out.yaml")
    (tmp_path / "in.txt").write_text("F: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported input file type '.txt'"):
        solver.solve(tmp_path / "in.txt", tmp_path / "out.yaml")


def test_solve_checks_results_path_before_reading_input(tmp_path) -> None:
    (tmp_path / "in.yaml").write_text("{{{", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported results file type"):
        solver.solve(tmp_path / "in.yaml", tmp_path / "r.txt")


def test_solve_mapping_rejects_warm_start_mode() -> None:
    with pytest.raises(ValueError, match="warm_start_mode"):
        solver.solve_mapping({}, warm_start_mode="warm")


def test_solve_mapping_checks_input_without_gurobi() -> None:
    with pytest.raises(ValueError):
        solver.solve_mapping({"F": 2})


def test_solve_mapping_wiring(tmp_path) -> None:
    sentinel = object()
    calls = []
    result = _result()
    with mock.patch("fleet_management.solver.load_config", return_value=sentinel):
        with _stub_base(result, calls):
            out = solver.solve_mapping(
                {"F": 2}, tmp_path / "r.json",
                warm_start={"x": None}, warm_start_mode="hint",
            )
    assert out is result
    assert (tmp_path / "r.json").exists()
    assert calls == [(sentinel, {"x": None}, "hint")]


def test_solve_writes_or_not(tmp_path) -> None:
    (tmp_path / "in.yaml").write_text("F: 2\n", encoding="utf-8")
    calls = []
    result = _result()
    with mock.patch("fleet_management.solver.load_config", return_value=object()):
        with _stub_base(result, calls):
            out = solver.solve(tmp_path / "in.yaml", tmp_path / "out")
            assert out is result
            assert (tmp_path / "out.yaml").exists()
            solver.solve(tmp_path / "in.yaml", None)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["in.yaml", "out.yaml"]
    assert len(calls) == 2


def test_solve_default_results_path() -> None:
    parameters = inspect.signature(solver.solve).parameters
    assert parameters["results_path"].default == "output.yaml"
    assert parameters["warm_start_mode"].default == "start"
    assert inspect.signature(solver.solve_mapping).parameters["results_path"].default is None


def test_read_input_alias() -> None:
    assert solver._read_input is config.read_input


# ---------------------------------------------------------------------------
# B2: YAML and JSON result files
# ---------------------------------------------------------------------------
def _read_yaml(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_json(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _check_text_data(result: dict, data: dict) -> None:
    expected = [key for key in solver.RESULT_KEYS if result.get(key) is not None]
    assert list(data) == expected
    assert "extra_key" not in data
    assert data["status"] == "optimal"
    assert data["objective"] == 12.5
    assert data["u"] == 1.2
    assert data["v"][0][0] == [0.0, 0.01, 0.02]
    assert data["v"][0][1] == [None, None, None]
    assert data["R"][1][0] == [None, None, None]
    assert data["v_0"] == [[0.001, None], [0.001, None]]
    assert data["H"] == [1, 2]
    assert data["model"] == [["gaussian", "rainflow"], ["gaussian", "rainflow"]]
    assert data["component_names"] == ["Wheelset", "Brake pad"]
    assert data["bound_method"] == [[None, "cantelli"], [None, "cantelli"]]
    assert data["repair_model"] == "ardinf"
    assert data["encoding"] == [["bigm", "indicator"], ["bigm", "bigm"]]
    assert data["component_costs"]["C_R"] == [[0.5, 0.5], [0.5, 0.5]]
    assert data["performance"]["solutions_found"] is None
    assert data["performance"]["gurobi_runtime_seconds"] == 1.5
    assert data["optimization_progress"][0]["incumbent"] is None
    assert data["warm_start"]["completion"]["iis_constraints"] == []


def test_yaml_round_trip(tmp_path) -> None:
    result = _result()
    solver._save_results(result, tmp_path / "r.yaml")
    _check_text_data(result, _read_yaml(tmp_path / "r.yaml"))
    solver._save_results(result, tmp_path / "r.yml")
    _check_text_data(result, _read_yaml(tmp_path / "r.yml"))


def test_json_round_trip(tmp_path) -> None:
    result = _result()
    solver._save_results(result, tmp_path / "r.json")
    _check_text_data(result, _read_json(tmp_path / "r.json"))
    text = (tmp_path / "r.json").read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text


def test_no_solution_text_files(tmp_path) -> None:
    result = _result(solution=False)
    solver._save_results(result, tmp_path / "r.yaml")
    solver._save_results(result, tmp_path / "r.json")
    for data in (_read_yaml(tmp_path / "r.yaml"), _read_json(tmp_path / "r.json")):
        assert "objective" in data
        assert data["objective"] is None
        assert data["status"] == "infeasible"
        for key in ("x", "mu", "step_costs", "mip_gap"):
            assert key not in data
        assert "performance" in data
        assert "model" in data


def test_infinite_float_written_as_null(tmp_path) -> None:
    result = _result()
    result["bound"] = float("inf")
    solver._save_results(result, tmp_path / "r.yaml")
    solver._save_results(result, tmp_path / "r.json")
    for data in (_read_yaml(tmp_path / "r.yaml"), _read_json(tmp_path / "r.json")):
        assert "bound" in data
        assert data["bound"] is None


def test_save_results_rejects_extension(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unsupported results file type '.csv'"):
        solver._save_results(_result(), tmp_path / "r.csv")


# ---------------------------------------------------------------------------
# B3: HDF5 result files
# ---------------------------------------------------------------------------
def test_hdf5_round_trip(tmp_path) -> None:
    result = _result()
    solver._save_results(result, tmp_path / "r.h5")
    solver._save_results(result, tmp_path / "r.hdf5")
    with h5py.File(tmp_path / "r.h5", "r") as f:
        assert set(f.keys()) == set(solver._HDF5_DATASET_KEYS)
        assert len(solver._HDF5_DATASET_KEYS) == 25
        for key in f.keys():
            assert f[key].dtype == np.float64
        assert np.isnan(f["v"][0, 1, 0])
        assert f["v"][0, 0, 1] == 0.01
        assert np.isnan(f["R"][()]).all()
        assert f["x"].shape == (2, 2, 3)
        assert f["tau"].shape == (2, 2)
        assert f["y_mu"].shape == (2, 2) and f["y_mu"][0, 0] == 0.5
        assert np.isnan(f["y_v"][0, 1]) and np.isnan(f["y_g_mu"][()]).all()

        attrs = f.attrs
        assert attrs["status"] == "optimal"
        assert attrs["objective"] == 12.5
        assert attrs["u"] == 1.2
        assert list(attrs["H"]) == [1, 2]
        assert attrs["F"] == 2 and attrs["T"] == 3
        assert attrs["objective_mode"] == "total"
        assert attrs["repair_model"] == "ardinf"
        assert attrs["reliability_impl"] == "exact"
        assert json.loads(attrs["bound_method"]) == [[None, "cantelli"], [None, "cantelli"]]
        assert json.loads(attrs["encoding"]) == [["bigm", "indicator"], ["bigm", "bigm"]]
        assert json.loads(attrs["model"]) == [["gaussian", "rainflow"]] * 2
        assert json.loads(attrs["component_names"]) == ["Wheelset", "Brake pad"]
        assert json.loads(attrs["component_costs"])["C_R"] == [[0.5, 0.5], [0.5, 0.5]]
        assert json.loads(attrs["performance"])["solutions_found"] is None
        assert json.loads(attrs["optimization_progress"])[0]["incumbent"] is None
        assert json.loads(attrs["warm_start"])["source_horizon"] == [1, 2]
        assert json.loads(attrs["relaxation_warm_start"])["status"] == "disabled"
        assert "evaluation_horizon" not in attrs
        assert "projected_evaluation_cost" not in attrs
        assert "extra_key" not in attrs
        assert "extra_key" not in f
    with h5py.File(tmp_path / "r.hdf5", "r") as f:
        assert set(f.keys()) == set(solver._HDF5_DATASET_KEYS)


def test_hdf5_keeps_non_finite_numbers(tmp_path) -> None:
    result = _result()
    result["bound"] = float("inf")
    result["mip_gap"] = float("nan")
    solver._save_results(result, tmp_path / "r.h5")
    with h5py.File(tmp_path / "r.h5", "r") as f:
        assert f.attrs["bound"] == float("inf")
        assert np.isnan(f.attrs["mip_gap"])


def test_hdf5_selector_that_is_one_string_is_an_attribute(tmp_path) -> None:
    result = _result()
    result["encoding"] = "bigm"
    result["bound_method"] = np.array([[None, "cantelli"]] * 2, dtype=object)
    solver._save_results(result, tmp_path / "r.h5")
    with h5py.File(tmp_path / "r.h5", "r") as f:
        assert f.attrs["encoding"] == "bigm"
        assert json.loads(f.attrs["bound_method"]) == [[None, "cantelli"]] * 2


def test_no_solution_hdf5(tmp_path) -> None:
    solver._save_results(_result(solution=False), tmp_path / "r.h5")
    with h5py.File(tmp_path / "r.h5", "r") as f:
        assert f.attrs["status"] == "infeasible"
        assert "objective" not in f.attrs
        assert "mip_gap" not in f.attrs
        assert "x" not in f
        assert set(f.keys()) == {"tau", "mu_0", "v_0"}
        assert "model" in f.attrs
        assert json.loads(f.attrs["performance"])["variables"] == 120
