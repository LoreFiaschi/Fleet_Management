"""Tests of the structural validator (spec, section "Structural Validator").

The fixture is ``results/validator_tests/input.yaml`` (the YAML example of the spec: three
trains, two missions, two components, H1 = 2, H2 = 3) and the valid result
``results/validator_tests/result.yaml`` built from it by hand, with every row at equality.
None of the tests needs Gurobi.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from fleet_management import solver
from fleet_management.config import read_input
from fleet_management.validation.validator import validate

FIXTURES = Path(__file__).resolve().parents[1] / "results" / "validator_tests"
INPUT = FIXTURES / "input.yaml"
RESULT = FIXTURES / "result.yaml"

CHECK_NAMES = [
    "binary", "one_activity", "mission_coverage", "depot_gating", "idle",
    "loop_mu", "loop_v", "loop_R", "loop_K", "loop_g_mu", "loop_g_v", "loop_g_R", "loop_g_K",
    "objective",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _input(tmp_path, **changes):
    """Write the fixture input with some keys changed; return its path."""
    data = read_input(INPUT)
    data.update(changes)
    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _run(tmp_path, result, input_path=INPUT, **kwargs):
    """Save the result dictionary, validate it against the input and return the report."""
    path = tmp_path / "result.yaml"
    solver._save_results(result, path)
    return validate(input_path, path, tmp_path / "validation.yaml", **kwargs)


def _failed(report):
    """The failed checks of a report: name -> violation."""
    return {c["name"]: c["violation"] for c in report["checks"] if not c["passed"]}


def test_valid_result_passes(tmp_path):
    report = validate(INPUT, RESULT, tmp_path / "validation.yaml")
    assert report["passed"]
    assert [c["name"] for c in report["checks"]] == CHECK_NAMES
    assert report["status"] == "optimal"
    assert report["solver_objective"] == pytest.approx(533.882, rel=1e-12)
    assert report["recomputed_objective"] == pytest.approx(533.882, rel=1e-12)
    assert report["tolerance"] == 1e-6
    assert report["max_violation"] <= 1e-12
    assert yaml.safe_load((tmp_path / "validation.yaml").read_text(encoding="utf-8")) == report


@pytest.mark.parametrize("suffix", [".yaml", ".yml", ".json", ".h5", ".hdf5"])
def test_every_result_format_gives_the_same_report(tmp_path, suffix):
    reference = validate(INPUT, RESULT, tmp_path / "reference.yaml")
    path = tmp_path / f"result{suffix}"
    solver._save_results(solver.read_result(RESULT), path)
    report = validate(INPUT, path, tmp_path / "validation.yaml")
    assert report["passed"] == reference["passed"]
    assert report["checks"] == reference["checks"]
    assert report["recomputed_objective"] == reference["recomputed_objective"]


def test_read_result(tmp_path):
    base = solver.read_result(RESULT)
    assert np.all(np.isnan(base["R"]))
    assert base["H"] == [2, 3]
    for suffix in (".json", ".h5"):
        path = tmp_path / f"result{suffix}"
        solver._save_results(base, path)
        other = solver.read_result(path)
        for key in solver._HDF5_DATASET_KEYS:
            np.testing.assert_array_equal(other[key], base[key])
        assert other["H"] == [2, 3]
    solver._save_results(dict(base, idle=None), tmp_path / "no_idle.yaml")
    assert solver.read_result(tmp_path / "no_idle.yaml")["idle"] is None


def test_report_path_rules(tmp_path):
    validate(INPUT, RESULT, tmp_path / "new" / "report")
    assert (tmp_path / "new" / "report.yaml").exists()
    report = validate(INPUT, RESULT, tmp_path / "report.json")
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8")) == report


@pytest.mark.parametrize(
    "changes, expected",
    [
        ({"objective_mode": "total"}, 58.882),
        ({"objective_mode": "operating_average"}, 19.84),
        ({"objective_mode": "evaluation_total"}, 533.882),
        ({"objective_mode": "total", "C_D": 0.0}, 58.042),
    ],
)
def test_recomputed_objective_by_mode(tmp_path, changes, expected):
    report = _run(tmp_path, solver.read_result(RESULT), _input(tmp_path, **changes))
    assert report["recomputed_objective"] == pytest.approx(expected, rel=1e-12)
    assert report["passed"]


def test_objective_below_the_recomputed_value_fails(tmp_path):
    result = solver.read_result(RESULT)
    result["objective"] -= 2.0
    assert _failed(_run(tmp_path, result)) == {
        "objective": pytest.approx(0.003760232532779827, rel=1e-12)
    }


def test_objective_above_the_recomputed_value_passes(tmp_path):
    result = solver.read_result(RESULT)
    result["objective"] += 5.0
    assert _run(tmp_path, result)["passed"]


def test_status_is_echoed_not_checked(tmp_path):
    result = solver.read_result(RESULT)
    result["status"] = "time_limit"
    report = _run(tmp_path, result)
    assert report["passed"]
    assert report["status"] == "time_limit"


def test_file_errors(tmp_path):
    report_path = tmp_path / "validation.yaml"
    with pytest.raises(FileNotFoundError, match="Input file not found"):
        validate(tmp_path / "missing.yaml", RESULT, report_path)
    with pytest.raises(FileNotFoundError, match="Result file not found"):
        validate(INPUT, tmp_path / "missing.yaml", report_path)
    (tmp_path / "result.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported result file type"):
        validate(INPUT, tmp_path / "result.txt", report_path)
    with pytest.raises(ValueError, match="Unsupported validation file type"):
        validate(INPUT, RESULT, tmp_path / "report.txt")
    (tmp_path / "input.csv").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported input file type"):
        validate(tmp_path / "input.csv", RESULT, report_path)


# ---------------------------------------------------------------------------
# One failing check at a time
# ---------------------------------------------------------------------------
def _edit(result, key, index, value):
    """Set ``result[key][index] = value``; an index of None adds ``value`` to a scalar."""
    if index is None:
        result[key] += value
    else:
        result[key][index] = value


@pytest.mark.parametrize(
    "name, edits, violation",
    [
        ("binary", [("x", (2, 0, 0), 0.3), ("idle", (2, slice(None), 0), 0.3),
                    ("objective", None, 0.3)], 0.3),
        ("one_activity", [("x", (0, 1, 2), 1.0), ("x", (1, 1, 2), 0.0)], 1.0),
        ("mission_coverage", [("x", (1, 1, 2), 0.0)], 1.0),
        ("depot_gating", [("m", (0, 0, 0), 1.0), ("idle", (0, 0, 0), -1.0)], 1.0),
        ("idle", [("idle", (2, 0, 1), 0.0)], 1.0),
    ],
)
def test_violation_is_detected(tmp_path, name, edits, violation):
    result = solver.read_result(RESULT)
    for key, index, value in edits:
        _edit(result, key, index, value)
    assert _failed(_run(tmp_path, result)) == {name: pytest.approx(violation, rel=1e-12)}


def test_binary_is_the_distance_to_0_and_1(tmp_path):
    result = solver.read_result(RESULT)
    result["r"][0, 0, 2] = 2.0          # an integer, but not in {0, 1}
    assert _failed(_run(tmp_path, result))["binary"] == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize(
    "key, i, l",
    [("mu", 0, 0), ("v", 1, 0), ("R", 0, 0), ("K", 2, 1),
     ("g_mu", 1, 1), ("g_v", 2, 0), ("g_R", 0, 0), ("g_K", 2, 1)],
)
def test_loop_row_violation_is_detected(tmp_path, key, i, l):
    result = solver.read_result(RESULT)
    a = result[key]
    if np.isnan(a[i, l, 1]):
        a[i, l, 1] = 0.1
    a[i, l, 4] = a[i, l, 1] + 0.5
    assert _failed(_run(tmp_path, result)) == {f"loop_{key}": pytest.approx(0.5, rel=1e-12)}


def test_nan_entries_are_skipped(tmp_path):
    report = validate(INPUT, RESULT, tmp_path / "validation.yaml")
    violations = {c["name"]: c["violation"] for c in report["checks"]}
    assert violations["loop_R"] == 0.0
    assert violations["loop_g_R"] == 0.0
    result = solver.read_result(RESULT)
    result["K"] = None
    report = _run(tmp_path, result)
    assert [c["name"] for c in report["checks"]] == CHECK_NAMES
    assert report["passed"]


def test_replacement_not_allowed(tmp_path):
    report = _run(tmp_path, solver.read_result(RESULT), _input(tmp_path, allow_replacement=False))
    assert _failed(report) == {"depot_gating": 1.0}
    assert report["recomputed_objective"] == pytest.approx(29.882, rel=1e-12)


def test_null_z_counts_as_zero_and_idle_is_optional(tmp_path):
    result = solver.read_result(RESULT)
    result["z"][:, 0, :] = np.nan       # z is not built at cells with C_R = 0
    result["idle"] = None
    report = _run(tmp_path, result, _input(tmp_path, C_R=[0.0, 1.5]))
    assert report["passed"]
    assert report["recomputed_objective"] == pytest.approx(533.882, rel=1e-12)
    assert [c["name"] for c in report["checks"]] == [n for n in CHECK_NAMES if n != "idle"]


# ---------------------------------------------------------------------------
# Results and inputs that cannot be checked
# ---------------------------------------------------------------------------
def test_result_without_solution(tmp_path):
    result = solver.read_result(RESULT)
    result["x"] = None
    with pytest.raises(ValueError, match="no solution: 'x' is missing or null"):
        _run(tmp_path, result)


@pytest.mark.parametrize("key, value", [("F", 4), ("H1", 3)])
def test_sizes_must_match(tmp_path, key, value):
    result = solver.read_result(RESULT)
    result[key] = value
    with pytest.raises(ValueError, match=f"'{key}' differs between the result"):
        _run(tmp_path, result)


def test_shapes_must_match(tmp_path):
    result = solver.read_result(RESULT)
    result["mu"] = result["mu"][:, :, :4]
    message = r"result array 'mu' has shape \(3, 2, 4\); the input gives \(3, 2, 5\)"
    with pytest.raises(ValueError, match=message):
        _run(tmp_path, result)


def test_malformed_result_or_input(tmp_path):
    result = solver.read_result(RESULT)
    result["x"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="result array 'x' has a non-finite entry"):
        _run(tmp_path, result)
    result = solver.read_result(RESULT)
    result["u"] = None
    with pytest.raises(ValueError, match="the result has 'x' but no 'u'"):
        _run(tmp_path, result)
    with pytest.raises(ValueError, match="tau"):
        _run(tmp_path, solver.read_result(RESULT), _input(tmp_path, tau=-1.0))
