"""Tests of the outputs of ``solve`` against the spec (Output Specification, Data Flow, Public API).

Every expected value comes from the spec (``spec/spec.tex``, lines of the ``lore`` copy named in each test),
derived here from the input file by an independent computation. Tests that solve need Gurobi and skip without it.
"""

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from fleet_management.solver import read_result, solve

DATA = Path(__file__).parent / "data" / "io"
SRC = Path(__file__).resolve().parents[1] / "src"

# Keys of the result dictionary (spec 1716-1753).
RESULT_KEYS = (
    "status", "objective", "mip_gap", "bound",
    "F", "M", "L", "H1", "H2", "T", "H", "model", "component_names",
    "tau", "mu_0", "v_0",
    "bound_method", "repair_model", "reliability_impl", "encoding",
    "objective_mode", "evaluation_horizon",
    "x", "m", "r", "idle", "mu", "v", "z", "R", "K",
    "g_mu", "g_v", "g_R", "g_K", "u",
    "step_costs", "J_initialization", "J_op", "J_op_average",
    "peak_damage", "damage_penalty", "operating_objective", "J_total",
    "projected_evaluation_cost", "evaluation_objective", "component_costs",
    "performance", "optimization_progress", "warm_start", "relaxation_warm_start",
)
SOLUTION_KEYS = ("x", "m", "r", "idle", "mu", "v", "z", "R", "K", "g_mu", "g_v", "g_R", "g_K", "u")  # 1730-1740
COST_KEYS = (
    "step_costs", "J_initialization", "J_op", "J_op_average", "peak_damage", "damage_penalty",
    "operating_objective", "J_total", "projected_evaluation_cost", "evaluation_objective", "component_costs",
)  # 1742-1748
SELECTORS = ("bound_method", "repair_model", "reliability_impl", "encoding")
PERFORMANCE_KEYS = (
    "gurobi_version", "variables", "continuous_variables", "integer_variables", "binary_variables",
    "linear_constraints", "general_constraints", "indicator_constraints", "quadratic_constraints", "nonzeros",
    "solutions_found", "gurobi_runtime_seconds", "branch_and_bound_nodes", "simplex_iterations",
    "barrier_iterations", "work_units", "model_construction_seconds", "optimizer_call_seconds",
    "solution_extraction_seconds", "backend_wall_seconds", "requested_time_limit_seconds",
    "requested_mip_gap", "threads", "seed",
)  # 1793-1811
PROGRESS_KEYS = {"runtime_seconds", "incumbent", "best_bound", "relative_gap", "nodes", "solutions"}  # 1818-1820
HDF5_DATASETS = (
    "x", "mu", "v", "R", "K", "g_mu", "g_v", "g_R", "g_K", "z", "m", "r", "idle", "step_costs", "tau", "mu_0", "v_0",
)  # 2741
HDF5_ATTRIBUTES = (
    "status", "objective", "mip_gap", "F", "M", "L", "H", "H1", "H2", "T", "objective_mode", "u",
    "J_initialization", "J_op", "J_op_average", "peak_damage", "damage_penalty", "operating_objective", "J_total",
)  # 2740, the keys that a solved fleet of the library always has
HDF5_JSON = (
    "model", "component_names", "component_costs", "performance", "optimization_progress", "warm_start",
    "relaxation_warm_start",
)  # 2742
MODE_KEY = {"total": "J_total", "operating_average": "operating_objective", "evaluation_total": "evaluation_objective"}

# One solve each; (library file, changes to the mapping).
CASES = [
    ("gaussian", {}),
    ("gaussian_ard1_forms", {}),
    ("inverse_gaussian", {}),
    ("wiener", {}),
    ("gamma", {}),
    ("rainflow_bounds", {}),
    ("mixed_example", {"time_limit": 60}),
    ("gaussian", {"C_D": 0}),
    ("gaussian", {"penalty_type": "quadratic"}),
    ("gaussian", {"C_R": [0.0, 1.5]}),
    ("gaussian", {"objective_mode": "operating_average"}),
    ("gaussian", {"objective_mode": "evaluation_total", "evaluation_horizon": 10}),
    ("gaussian", {"H": 2}),
]

_CACHE = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _data(name, changes=None):
    with open(DATA / "valid" / f"{name}.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data.update(changes or {})
    return data


def _write_input(folder, data, name="input.yaml"):
    path = Path(folder) / name
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return path


def _solved(name, changes=None, ext=".yaml"):
    """Solve a library file with changes once; return (result, result path, mapping)."""
    pytest.importorskip("gurobipy")
    key = (name, json.dumps(changes or {}, sort_keys=True), ext)
    if key not in _CACHE:
        folder = Path(tempfile.mkdtemp(prefix="spec_outputs_"))
        data = _data(name, changes)
        path = _write_input(folder, data)
        out = folder / f"result{ext}"
        _CACHE[key] = (solve(str(path), str(out)), out, data)
    return _CACHE[key]


def _solved_infeasible(ext=".yaml"):
    pytest.importorskip("gurobipy")
    key = ("infeasible_wiener", ext)
    if key not in _CACHE:
        folder = Path(tempfile.mkdtemp(prefix="spec_outputs_"))
        out = folder / f"result{ext}"
        _CACHE[key] = (solve(str(DATA / "infeasible_wiener.yaml"), str(out)), out)
    return _CACHE[key]


def _grid(value, F, L):
    """Broadcast a scalar, a list of L entries or an F x L nested list to an F x L nested list."""
    if not isinstance(value, (list, tuple)):
        return [[value] * L for _ in range(F)]
    if len(value) == F and all(isinstance(row, (list, tuple)) and len(row) == L for row in value):
        return [list(row) for row in value]
    assert len(value) == L
    return [list(value) for _ in range(F)]


def _sizes(data):
    model = data["model"]
    if "L" in data:
        L = data["L"]
    elif isinstance(model, list):
        L = len(model[0]) if isinstance(model[0], list) else len(model)
    else:
        L = 1
    H = data["H"]
    H1, H2 = (H, H) if isinstance(H, int) else H
    return data["F"], data["M"], L, H1, H2


def _cells(data):
    """Model, repair law and bound method of each cell, with the defaults of the spec."""
    F, M, L, H1, H2 = _sizes(data)
    models = _grid(data["model"], F, L)
    repairs = _grid(data.get("repair_model", "ardinf"), F, L)
    bounds = _grid(data.get("bound_method"), F, L)
    for i in range(F):
        for l in range(L):
            if repairs[i][l] is None:
                repairs[i][l] = "ardinf"
            if models[i][l] == "rainflow" and bounds[i][l] is None:
                bounds[i][l] = "cantelli"
    return models, repairs, bounds


def _tracked(model, bound, repair):
    """Which keys hold values at a cell (spec 1736-1739); the others are NaN."""
    rf = model == "rainflow"
    var = model in ("gaussian", "wiener") or (rf and bound in ("cantelli", "bernstein"))
    ard1 = repair == "ard1"
    return {
        "v": var,
        "R": rf and bound == "hoeffding",
        "K": rf and bound == "chernoff",
        "g_mu": ard1,
        "g_v": ard1 and (model == "gaussian" or (rf and bound in ("cantelli", "bernstein"))),
        "g_R": ard1 and rf and bound == "hoeffding",
        "g_K": ard1 and rf and bound == "chernoff",
    }


def _arr(value):
    return np.asarray(value, dtype=float)


def _close(a, b, tol=1e-6):
    return np.allclose(_arr(a), _arr(b), rtol=tol, atol=tol, equal_nan=True)


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _same(a, b):
    """Equality of a result value and the same value read back (NaN equal to None and to NaN)."""
    if isinstance(a, dict) or isinstance(b, dict):
        assert isinstance(a, dict) and isinstance(b, dict)
        assert set(a) == set(b)
        return all(_same(a[k], b[k]) for k in a)
    if isinstance(b, np.ndarray) and b.dtype.kind in "US":
        b = b.tolist()
    if isinstance(a, np.ndarray) and a.dtype.kind in "US":
        a = a.tolist()
    if isinstance(a, (str, bytes)) or isinstance(b, (str, bytes)):
        return _text(a) == _text(b)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if any(isinstance(item, (dict, str)) for item in list(a) + list(b)):
            return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if a is None or b is None:
        other = b if a is None else a
        return other is None or (np.ndim(other) == 0 and not np.isfinite(float(other)))
    try:
        return _close(a, b, 1e-9)
    except (TypeError, ValueError):
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
        return a == b


def _plain_no_nan(value):
    """True if a loaded YAML or JSON value holds no NaN and no infinity."""
    if isinstance(value, dict):
        return all(_plain_no_nan(v) for v in value.values())
    if isinstance(value, list):
        return all(_plain_no_nan(v) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _load_json_strict(path):
    def refuse(token):
        raise AssertionError(f"JSON constant {token} in {path}")

    with open(path, encoding="utf-8") as f:
        return json.load(f, parse_constant=refuse)


# ---------------------------------------------------------------------------
# Data flow (spec 221, 1880; no Gurobi)
# ---------------------------------------------------------------------------

def test_missing_input_file_raises_file_not_found(tmp_path):
    """Spec 221: the input file must exist (FileNotFoundError)."""
    with pytest.raises(FileNotFoundError):
        solve(str(tmp_path / "missing.yaml"), str(tmp_path / "out.yaml"))


def test_input_extension_txt_raises_value_error(tmp_path):
    """Spec 221: the input extension must be .yaml, .yml, .json, .h5 or .hdf5 (ValueError)."""
    path = tmp_path / "input.txt"
    path.write_text("F: 3\n", encoding="utf-8")
    with pytest.raises(ValueError):
        solve(str(path), str(tmp_path / "out.yaml"))


def test_result_extension_checked_before_input_is_read(tmp_path):
    """Spec 1880: the extension of the result path is checked before the input file is read.

    The input file exists and cannot be parsed; the error is the ValueError of the extension.
    """
    path = tmp_path / "input.yaml"
    path.write_text("a: [\n", encoding="utf-8")
    with pytest.raises(ValueError) as info:
        solve(str(path), str(tmp_path / "out.txt"))
    assert not isinstance(info.value, yaml.YAMLError)


def test_result_folder_checked_before_input_is_read(tmp_path):
    """Spec 1880: the directory of the result path must exist (FileNotFoundError), checked before the input is read."""
    path = tmp_path / "input.yaml"
    path.write_text("a: [\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        solve(str(path), str(tmp_path / "no_such_folder" / "out.yaml"))


def test_missing_input_is_checked_before_bad_result_extension(tmp_path):
    """Spec 221, step 1: the input file is checked first, so a missing input gives FileNotFoundError even when
    the result extension is bad. (Spec 1880 says the result path is checked before the input is read; a missing
    file is not read, and step 1 comes first.)"""
    with pytest.raises(FileNotFoundError):
        solve(str(tmp_path / "missing.yaml"), str(tmp_path / "out.txt"))


def test_missing_result_folder_with_valid_input_raises_file_not_found(tmp_path):
    """Spec 1880: a missing result folder gives FileNotFoundError."""
    path = _write_input(tmp_path, _data("gaussian"))
    with pytest.raises(FileNotFoundError):
        solve(str(path), str(tmp_path / "no_such_folder" / "out.yaml"))


@pytest.mark.parametrize("name", ["out.txt", "out.csv"])
def test_bad_result_extension_with_valid_input_raises_value_error(tmp_path, name):
    """Spec 1874: any other result extension raises a ValueError."""
    path = _write_input(tmp_path, _data("gaussian"))
    with pytest.raises(ValueError):
        solve(str(path), str(tmp_path / name))


# ---------------------------------------------------------------------------
# Result dictionary (Gurobi)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, changes", CASES)
def test_result_keys_shapes_and_problem_entries(name, changes):
    """Spec 1716-1753: exactly the keys of the table, the shapes, and the problem entries."""
    res, _, data = _solved(name, changes)
    F, M, L, H1, H2 = _sizes(data)
    T = H1 + H2
    assert set(res) == set(RESULT_KEYS)
    assert res["status"] == "optimal"
    assert (res["F"], res["M"], res["L"], res["H1"], res["H2"], res["T"]) == (F, M, L, H1, H2, T)
    assert list(np.atleast_1d(res["H"])) == list(np.atleast_1d(data["H"]))
    assert np.ndim(res["H"]) == np.ndim(data["H"])
    assert [list(row) for row in res["model"]] == _grid(data["model"], F, L)
    assert list(res["component_names"]) == list(data.get("component_names", [f"Component {l + 1}" for l in range(L)]))
    assert res["objective_mode"] == data.get("objective_mode", "total")
    assert res["evaluation_horizon"] == data.get("evaluation_horizon")
    for key in ("tau", "mu_0", "v_0"):
        assert np.asarray(res[key]).shape == (F, L)
    assert _close(res["tau"], _grid(data["tau"], F, L))
    assert _close(res["mu_0"], _grid(data["mu_0"], F, L))
    assert isinstance(res["x"], np.ndarray) and res["x"].shape == (F, M + 1, T)
    for key in ("m", "r", "idle", "mu", "v", "z", "R", "K", "g_mu", "g_v", "g_R", "g_K"):
        assert isinstance(res[key], np.ndarray), key
        assert res[key].shape == (F, L, T), key
    assert np.asarray(res["step_costs"]).shape == (T,)
    assert res["bound"] is None or res["bound"] <= res["objective"] + 1e-6
    assert res["mip_gap"] >= 0
    if "mip_gap" in data:
        assert res["mip_gap"] <= data["mip_gap"] + 1e-9


@pytest.mark.parametrize("name, changes", CASES)
def test_selectors(name, changes):
    """Spec 1727: one string if all cells agree, else an F x L list; bound_method is null at a non-rainflow cell."""
    res, _, data = _solved(name, changes)
    F, M, L, H1, H2 = _sizes(data)
    models, repairs, bounds = _cells(data)
    given_rel = _grid(data.get("reliability_impl"), F, L)
    expected = {
        "bound_method": [[bounds[i][l] if models[i][l] == "rainflow" else None for l in range(L)] for i in range(F)],
        "repair_model": repairs,
        "encoding": [[e if e is not None else "bigm" for e in row] for row in _grid(data.get("encoding", "bigm"), F, L)],
        # the form applies to G and W cells; the spec fixes the default exact there (spec 2753 table)
        "reliability_impl": [
            [given_rel[i][l] if given_rel[i][l] is not None else ("exact" if models[i][l] in ("gaussian", "wiener") else None)
             for l in range(L)] for i in range(F)
        ],
    }
    for key in SELECTORS:
        value = res[key]
        exp = expected[key]
        flat = [e for row in exp for e in row]
        if isinstance(value, str):
            assert all(e is None or e == value for e in flat), key
            continue
        if value is None:
            assert all(e is None for e in flat), key
            continue
        grid = _grid(list(value), F, L)
        for i in range(F):
            for l in range(L):
                if exp[i][l] is not None:
                    assert grid[i][l] == exp[i][l], (key, i, l)
                elif key == "bound_method":
                    assert grid[i][l] is None, (key, i, l)
        if key != "bound_method" and None not in flat and len(set(flat)) == 1:
            pytest.fail(f"{key}: all cells agree but the result is a list")


@pytest.mark.parametrize("name, changes", CASES)
def test_nan_rules(name, changes):
    """Spec 1736-1739: NaN at the cells that do not track the key; z NaN where C_R = 0."""
    res, _, data = _solved(name, changes)
    F, M, L, H1, H2 = _sizes(data)
    models, repairs, bounds = _cells(data)
    c_r = _grid(data["C_R"], F, L)
    for i in range(F):
        for l in range(L):
            tracked = _tracked(models[i][l], bounds[i][l], repairs[i][l])
            tracked["z"] = c_r[i][l] > 0
            for key, has in tracked.items():
                nan = np.isnan(res[key][i, l, :])
                assert nan.all() != has, (key, i, l)
                assert nan.all() or not nan.any(), (key, i, l)


@pytest.mark.parametrize("name, changes", CASES)
def test_idle(name, changes):
    """Spec 1734: idle = x_{i0k} - m - r, and 0 on an idle day of the train."""
    res, _, data = _solved(name, changes)
    x0 = res["x"][:, 0, :][:, None, :]
    assert _close(res["idle"], x0 - res["m"] - res["r"])
    assert np.all(res["idle"] > -1e-6)
    assert np.all(res["idle"][np.broadcast_to(x0 < 0.5, res["idle"].shape)] < 1e-6)


@pytest.mark.parametrize("name, changes", CASES)
def test_u_by_equation_c_u(name, changes):
    """Spec 1771-1776 (Eq. c_u): u from mu, also with C_D = 0; peak_damage and damage_penalty (1745)."""
    res, _, data = _solved(name, changes)
    mu = res["mu"]
    if data.get("penalty_type", "inf_norm") == "quadratic":
        u = np.max(np.sum(mu ** 2, axis=1))
    else:
        u = np.max(np.sum(mu, axis=1))
    assert abs(res["u"] - u) <= 1e-6 * (1 + abs(u))
    assert abs(res["peak_damage"] - res["u"]) <= 1e-9
    assert abs(res["damage_penalty"] - data["C_D"] * res["u"]) <= 1e-6 * (1 + abs(res["damage_penalty"]))


@pytest.mark.parametrize("name, changes", CASES)
def test_cost_report(name, changes):
    """Spec 657-669 (Eqs. a_daycost, a_phases), 1743-1748, 1777, 1897-1902 (Eq. d_projected)."""
    res, _, data = _solved(name, changes)
    F, M, L, H1, H2 = _sizes(data)
    c_m, c_d = data["C_M"], data["C_D"]
    c_r, c_rep = _arr(_grid(data["C_R"], F, L)), _arr(_grid(data["C_rep"], F, L))
    x, r, z = res["x"], res["r"], np.nan_to_num(res["z"], nan=0.0)
    c = c_m * x[:, 0, :].sum(axis=0) + (c_r[:, :, None] * z).sum(axis=(0, 1)) + (c_rep[:, :, None] * r).sum(axis=(0, 1))
    assert _close(res["step_costs"], c)
    j_init, j_op = c[:H1].sum(), c[H1:].sum()
    assert _close(res["J_initialization"], j_init)
    assert _close(res["J_op"], j_op)
    assert _close(res["J_op_average"], j_op / H2)
    assert _close(res["operating_objective"], j_op / H2 + c_d * res["u"])
    assert _close(res["J_total"], j_init + j_op + c_d * res["u"])
    E = data.get("evaluation_horizon")
    if E is None:
        assert res["projected_evaluation_cost"] is None
        assert res["evaluation_objective"] is None
    else:
        p_e = j_init + (E - H1) * j_op / H2
        assert _close(res["projected_evaluation_cost"], p_e)
        assert _close(res["evaluation_objective"], p_e + c_d * res["u"])
    costs = res["component_costs"]
    assert set(costs) == {"C_M", "C_D", "C_R", "C_rep"}
    assert _close(costs["C_M"], c_m) and _close(costs["C_D"], c_d)
    assert _close(costs["C_R"], c_r) and _close(costs["C_rep"], c_rep)
    assert np.asarray(costs["C_R"]).shape == (F, L) and np.asarray(costs["C_rep"]).shape == (F, L)


@pytest.mark.parametrize("name, changes", CASES)
def test_objective_against_key_of_the_mode(name, changes):
    """Spec 1778-1783: objective equals the key of the mode when u has no slack, else it is larger."""
    res, _, data = _solved(name, changes)
    key = res[MODE_KEY[data.get("objective_mode", "total")]]
    tol = 1e-5 * (1 + abs(key))
    assert res["objective"] >= key - tol
    if data.get("mip_gap", 0) == 0:
        assert abs(res["objective"] - key) <= tol


def test_z_on_transitory_days_in_operating_average_mode():
    """Spec 1762-1770 (Eq. c_zpost): z = rho (mu_{k-1} - g_{k-1}) on a repair day of the transitory phase, else 0."""
    res, _, data = _solved("gaussian", {"objective_mode": "operating_average"})
    F, M, L, H1, H2 = _sizes(data)
    rho = _arr(_grid(data["rho"], F, L))
    mu_0 = _arr(_grid(data["mu_0"], F, L))
    mu_prev = np.concatenate([mu_0[:, :, None], res["mu"][:, :, :H1 - 1]], axis=2)
    expected = np.where(res["m"][:, :, :H1] > 0.5, rho[:, :, None] * mu_prev, 0.0)  # ardinf: g = 0
    assert _close(res["z"][:, :, :H1], expected)


# ---------------------------------------------------------------------------
# No solution (Gurobi)
# ---------------------------------------------------------------------------

def test_infeasible_input_has_no_solution():
    """Spec 1759 and 1201-1205: W cells, no replacement, a mission in the operating phase: infeasible."""
    res, _ = _solved_infeasible()
    assert res["status"] in ("infeasible", "inf_or_unbounded")
    for key in SOLUTION_KEYS + COST_KEYS + ("objective", "mip_gap"):
        assert res[key] is None, key
    assert set(res) == set(RESULT_KEYS)
    assert isinstance(res["performance"], dict)  # 1750: the reports are always present
    assert isinstance(res["optimization_progress"], list)


# ---------------------------------------------------------------------------
# Reports (Gurobi)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, changes", CASES[:7])
def test_performance_report(name, changes):
    """Spec 1788-1812: every key of the table; the requested gap and time limit are the run options."""
    res, _, data = _solved(name, changes)
    perf = res["performance"]
    assert isinstance(perf, dict)
    assert set(PERFORMANCE_KEYS) <= set(perf)
    assert abs(perf["requested_mip_gap"] - data["mip_gap"]) < 1e-12
    assert abs(perf["requested_time_limit_seconds"] - data["time_limit"]) < 1e-9
    assert perf["solutions_found"] >= 1


@pytest.mark.parametrize("name, changes", CASES[:7])
def test_optimization_progress_report(name, changes):
    """Spec 1814-1829: records with the keys of the table; the last record is the final state."""
    res, _, _ = _solved(name, changes)
    records = res["optimization_progress"]
    assert isinstance(records, list) and len(records) >= 1
    for record in records:
        assert set(record) == PROGRESS_KEYS
    assert records[-1]["solutions"] == res["performance"]["solutions_found"]
    runtimes = [rec["runtime_seconds"] for rec in records]
    assert runtimes == sorted(runtimes)


# ---------------------------------------------------------------------------
# Result files (spec 1866-1884)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ext", [".yaml", ".yml", ".json", ".h5", ".hdf5"])
def test_result_file_extensions(ext):
    """Spec 1870-1873: each of the five extensions writes a file, and it can be read back."""
    res, out, _ = _solved("gaussian", {}, ext)
    assert out.exists()
    back = read_result(out)
    assert back["status"] == "optimal"
    assert _same(res["mu"], back["mu"])


def test_default_result_path_is_output_yaml(tmp_path, monkeypatch):
    """Spec 1878: the default of results_path is output.yaml (in the working folder)."""
    pytest.importorskip("gurobipy")
    path = _write_input(tmp_path, _data("gaussian"), "in.yaml")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    solve(str(path))
    assert (work / "output.yaml").exists()


def test_result_path_without_extension_gets_yaml(tmp_path):
    """Spec 1879: a path without extension gets the extension .yaml."""
    pytest.importorskip("gurobipy")
    path = _write_input(tmp_path, _data("gaussian"), "in.yaml")
    solve(str(path), str(tmp_path / "result"))
    assert (tmp_path / "result.yaml").exists()
    assert not (tmp_path / "result").exists()


def test_none_writes_no_file(tmp_path, monkeypatch):
    """Spec 1878: results_path None writes no file."""
    pytest.importorskip("gurobipy")
    path = _write_input(tmp_path, _data("gaussian"), "in.yaml")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    res = solve(str(path), None)
    assert res["status"] == "optimal"
    assert os.listdir(work) == []
    assert sorted(os.listdir(tmp_path)) == ["in.yaml", "work"]


def test_yaml_file_holds_plain_data_and_null():
    """Spec 1881: the YAML file loads with yaml.safe_load; NaN is written as null (v at IG cells)."""
    res, out, _ = _solved("inverse_gaussian")
    with open(out, encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    assert _plain_no_nan(loaded)
    flat = np.asarray(loaded["v"], dtype=object).ravel()
    assert len(flat) == res["v"].size and all(v is None for v in flat)
    assert loaded["status"] == "optimal"


def test_json_file_holds_plain_data_and_null():
    """Spec 1881: the JSON file loads with json.load and holds no NaN token; NaN is written as null."""
    res, out, _ = _solved("inverse_gaussian", {}, ".json")
    loaded = _load_json_strict(out)
    assert _plain_no_nan(loaded)
    flat = np.asarray(loaded["v"], dtype=object).ravel()
    assert len(flat) == res["v"].size and all(v is None for v in flat)


@pytest.mark.parametrize("ext", [".yaml", ".json"])
def test_none_keys_are_not_written(ext):
    """Spec 1882: a key with value None is not written (here evaluation_horizon and the two projected costs)."""
    _, out, _ = _solved("gaussian", {}, ext)
    loaded = _load_json_strict(out) if ext == ".json" else yaml.safe_load(open(out, encoding="utf-8"))
    for key in ("evaluation_horizon", "projected_evaluation_cost", "evaluation_objective"):
        assert key not in loaded
    assert isinstance(loaded["objective"], float)


@pytest.mark.parametrize("ext", [".yaml", ".json"])
def test_objective_none_is_written_as_null(ext):
    """Spec 1882: objective is written as null when None; the other None keys are absent."""
    res, out = _solved_infeasible(ext)
    loaded = _load_json_strict(out) if ext == ".json" else yaml.safe_load(open(out, encoding="utf-8"))
    assert "objective" in loaded and loaded["objective"] is None
    for key in SOLUTION_KEYS + COST_KEYS + ("mip_gap",):
        assert key not in loaded, key
    assert loaded["status"] == res["status"]
    assert "performance" in loaded


def test_hdf5_layout_of_a_solved_fleet():
    """Spec 2736-2743: attributes, float64 datasets with NaN kept, JSON text attributes."""
    res, out, data = _solved("gaussian", {}, ".h5")
    F, M, L, H1, H2 = _sizes(data)
    T = H1 + H2
    with h5py.File(out, "r") as f:
        for key in HDF5_ATTRIBUTES:
            assert key in f.attrs, key
            assert key not in f, key
        assert list(np.atleast_1d(f.attrs["H"])) == [H1, H2]
        assert _text(f.attrs["status"]) == "optimal"
        assert _text(f.attrs["objective_mode"]) == "total"
        assert abs(float(f.attrs["objective"]) - res["objective"]) <= 1e-9 * (1 + abs(res["objective"]))
        for key in ("repair_model", "reliability_impl", "encoding"):
            assert key in f.attrs and key not in f
        assert _text(f.attrs["repair_model"]) == "ardinf"
        shapes = {key: (F, L, T) for key in HDF5_DATASETS}
        shapes.update({"x": (F, M + 1, T), "step_costs": (T,), "tau": (F, L), "mu_0": (F, L), "v_0": (F, L)})
        for key in HDF5_DATASETS:
            assert key in f and key not in f.attrs, key
            assert f[key].dtype == np.float64, key
            assert f[key].shape == shapes[key], key
        for key in ("R", "K", "g_mu", "g_v", "g_R", "g_K"):  # NaN kept at ardinf and non-rainflow cells
            assert np.isnan(f[key][()]).all(), key
        assert not np.isnan(f["v"][()]).any()
        for key in HDF5_JSON:
            assert key in f.attrs and key not in f, key
            text = _text(f.attrs[key])
            assert text[0] in "[{", key
            json.loads(text)
        assert json.loads(_text(f.attrs["model"])) == [["gaussian", "gaussian"]] * F
        assert json.loads(_text(f.attrs["component_names"])) == ["Wheelset", "Brake pad"]
        for key in ("evaluation_horizon", "projected_evaluation_cost", "evaluation_objective"):
            assert key not in f.attrs and key not in f


def test_hdf5_keeps_nan_of_an_inverse_gaussian_fleet():
    """Spec 2737: an array of numbers is a float64 dataset with NaN kept (v at IG cells)."""
    res, out, _ = _solved("inverse_gaussian", {}, ".h5")
    with h5py.File(out, "r") as f:
        assert f["v"].dtype == np.float64
        assert np.isnan(f["v"][()]).all()


def test_hdf5_selectors_that_differ_between_cells_are_json_text():
    """Spec 2742: a selector that differs between cells is JSON text in an attribute."""
    res, out, data = _solved("mixed_example", {"time_limit": 60}, ".h5")
    F, M, L, H1, H2 = _sizes(data)
    with h5py.File(out, "r") as f:
        for key in ("reliability_impl", "bound_method", "repair_model"):  # the example gives different values
            assert key in f.attrs and key not in f, key
            parsed = json.loads(_text(f.attrs[key]))
            assert np.asarray(parsed, dtype=object).shape == (F, L), key
        assert json.loads(_text(f.attrs["bound_method"]))[2] == ["bernstein", "chernoff"]
        assert json.loads(_text(f.attrs["bound_method"]))[0] == [None, None]
        assert list(np.atleast_1d(f.attrs["H"])) == [2, 3]
        assert float(f.attrs["evaluation_horizon"]) == 30


def test_hdf5_objective_none_is_not_stored():
    """Spec 1882, 2737: HDF5 has no null; objective and every other None key are absent."""
    res, out = _solved_infeasible(".h5")
    with h5py.File(out, "r") as f:
        for key in ("objective", "mip_gap") + SOLUTION_KEYS + COST_KEYS:
            assert key not in f.attrs and key not in f, key
        assert _text(f.attrs["status"]) == res["status"]
    back = read_result(out)
    assert back["objective"] is None and back["x"] is None


# ---------------------------------------------------------------------------
# read_result (spec 1883)
# ---------------------------------------------------------------------------

def test_read_result_partial_yaml_has_exactly_the_keys_of_the_table():
    """Spec 1883: exactly the keys of the result table; a missing key is None; an extra key is dropped."""
    res = read_result(DATA / "results" / "partial.yaml")
    assert set(res) == set(RESULT_KEYS)
    assert "extra_key" not in res
    assert res["status"] == "optimal" and res["F"] == 3 and res["M"] == 1 and res["L"] == 2
    assert list(res["component_names"]) == ["Wheelset", "Brake pad"]
    assert _close(res["tau"], np.ones((3, 2)))
    for key in RESULT_KEYS:
        if key not in ("status", "F", "M", "L", "component_names", "tau"):
            assert res[key] is None, key


def test_read_result_float_keys_with_null_become_float_arrays():
    """Spec 1883: a float64 key becomes a float array, with NaN at a null entry."""
    res = read_result(DATA / "results" / "nulls.yaml")
    assert isinstance(res["x"], np.ndarray) and res["x"].dtype == np.float64
    assert res["x"].shape == (1, 1, 3) and _close(res["x"], [[[1.0, 0.0, 1.0]]])
    assert res["v"].dtype == np.float64 and res["v"].shape == (1, 1, 3)
    assert res["v"][0, 0, 0] == 0.5 and np.isnan(res["v"][0, 0, 1]) and res["v"][0, 0, 2] == 1.0
    assert res["step_costs"].dtype == np.float64
    assert _close(res["step_costs"], [1.0, 2.0, np.nan])
    assert abs(res["u"] - 0.4) < 1e-12
    assert res["mu"] is None


def test_read_result_json_file(tmp_path):
    """Spec 1883: a JSON result file is read; null becomes NaN in a float64 key; an extra key is dropped."""
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"status": "optimal", "mu": [[[0.1, None]]], "bogus": 1, "F": 2}), encoding="utf-8")
    res = read_result(path)
    assert set(res) == set(RESULT_KEYS)
    assert res["status"] == "optimal" and res["F"] == 2 and "bogus" not in res
    assert np.isnan(res["mu"][0, 0, 1]) and res["mu"][0, 0, 0] == 0.1
    assert res["objective"] is None


def test_read_result_hdf5_file(tmp_path):
    """Spec 1883, 2727, 2737: float64 datasets, attributes, JSON text; an extra root entry is dropped."""
    path = tmp_path / "r.h5"
    with h5py.File(path, "w") as f:
        f.attrs["status"] = "optimal"
        f.attrs["F"] = 2
        f.attrs["component_names"] = json.dumps(["A", "B"])
        f.attrs["bogus"] = 7
        f.create_dataset("mu", data=np.array([[[0.1, np.nan]]], dtype=np.float64))
    res = read_result(path)
    assert set(res) == set(RESULT_KEYS)
    assert res["status"] == "optimal" and res["F"] == 2 and "bogus" not in res
    assert list(res["component_names"]) == ["A", "B"]
    assert res["mu"].dtype == np.float64 and np.isnan(res["mu"][0, 0, 1])
    assert res["objective"] is None and res["x"] is None


def test_read_result_other_extension_raises_value_error(tmp_path):
    """Spec 1883: another extension raises a ValueError."""
    path = tmp_path / "r.txt"
    path.write_text("status: optimal\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_result(path)


@pytest.mark.parametrize("ext", [".yaml", ".json", ".h5"])
def test_round_trip_of_a_solved_result(ext):
    """Spec 1883: the writer and read_result agree for the three formats (arrays equal with NaN equal, scalars equal)."""
    res, out, _ = _solved("mixed_example", {"time_limit": 60}, ext)
    back = read_result(out)
    assert set(back) == set(RESULT_KEYS)
    for key in RESULT_KEYS:
        assert _same(res[key], back[key]), key


# ---------------------------------------------------------------------------
# Public API (spec 232-240; no Gurobi)
# ---------------------------------------------------------------------------

EXPORTED = (
    "solve", "sweep_operating_horizons", "sweep_horizon_grid", "plot_management", "plot_horizon_sweep", "validate",
    "validate_gamma_replay_schedule", "validate_gamma_replay_files", "validate_gamma_stochastic_schedule",
    "validate_gamma_schedule", "validate_gamma_schedule_files",
)
NOT_EXPORTED = ("solve_mapping", "load_config", "build_greedy_warm_start", "validate_gamma_tail_bound_schedule")

_PROBE = """
import json, sys
sys.modules['gurobipy'] = None
import fleet_management as fm
public = list(fm.__all__) if hasattr(fm, '__all__') else [n for n in dir(fm) if not n.startswith('_')]
print(json.dumps({"public": public, "callable": {n: callable(getattr(fm, n, None)) for n in public},
                  "attrs": dir(fm), "gurobipy_blocked": sys.modules['gurobipy'] is None}))
"""


def _probe(tmp_path):
    env = dict(os.environ)
    if SRC.exists():
        env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    done = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_import_without_gurobipy_exports_the_public_api(tmp_path):
    """Spec 232-240: the package exports the listed functions, and imports without gurobipy."""
    info = _probe(tmp_path)
    assert info["gurobipy_blocked"]
    for name in EXPORTED:
        assert name in info["public"], name
        assert info["callable"][name], name


def test_internal_functions_are_not_exported(tmp_path):
    """Spec 240: solve_mapping, load_config, build_greedy_warm_start and the tail-bound validator are not exported."""
    info = _probe(tmp_path)
    for name in NOT_EXPORTED:
        assert name not in info["public"], name
        assert name not in info["attrs"], name
