"""Tests of ``fleet_management.utils.plotter`` (spec, "Plotting API and Layout") and of the
public API (spec, "Public API").

None of them needs Gurobi. Result files are written by the writers of ``solver``;
sweep reports are written as the sweep writes them (``yaml.safe_dump``) or as JSON.
"""

import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml

from fleet_management import config, solver
from fleet_management.utils import plotter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _result(solution: bool = True) -> dict:
    """A result of two trains, one mission, two components and four days (H = [2, 2]).

    Marks by row (train, component) and day 1..4:
    (1, Wheelset) M1 I M1 R; (1, Brake pad) M1 I M1 P;
    (2, Wheelset) R M1 D M1; (2, Brake pad) D M1 P M1.
    """
    F, M, L, H1, H2 = 2, 1, 2, 2, 2
    T = H1 + H2
    x = np.zeros((F, M + 1, T))
    x[0, 1, [0, 2]] = 1.0      # train 1 flies mission 1 on days 1 and 3
    x[0, 0, 3] = 1.0           # train 1 is in the depot on day 4; idle on day 2
    x[1, 0, [0, 2]] = 1.0      # train 2 is in the depot on days 1 and 3
    x[1, 1, [1, 3]] = 1.0      # train 2 flies mission 1 on days 2 and 4
    m = np.zeros((F, L, T))
    m[0, 0, 3] = 1.0
    m[1, 0, 0] = 1.0
    r = np.zeros((F, L, T))
    r[0, 1, 3] = 1.0
    r[1, 1, 2] = 1.0
    nan = float("nan")
    result = {
        "status": "time_limit", "objective": 7.5, "mip_gap": 0.031, "bound": 7.3,
        "F": F, "M": M, "L": L, "H1": H1, "H2": H2, "T": T, "H": [H1, H2],
        "model": [["gaussian", "rainflow"], ["gaussian", "rainflow"]],
        "component_names": ["Wheelset", "Brake pad"],
        "tau": np.full((F, L), 1.0),
        "mu_0": np.full((F, L), 0.1),
        "v_0": np.array([[0.001, nan], [0.001, nan]]),
        "bound_method": [[None, "cantelli"], [None, "cantelli"]],
        "repair_model": [["ard1", "ardinf"], ["ard1", "ardinf"]],
        "reliability_impl": "exact",
        "encoding": "bigm",
        "objective_mode": "operating_average",
        "x": x, "m": m, "r": r,
        "idle": x[:, 0, :][:, None, :] - m - r,
        "mu": np.linspace(0.1, 1.2, F * L * T).reshape(F, L, T),
        "v": np.full((F, L, T), nan),
        "J_op_average": 1.25,
        "performance": {
            "variables": 120, "continuous_variables": 80, "integer_variables": 40,
            "linear_constraints": 300, "optimizer_call_seconds": 2.5,
            "branch_and_bound_nodes": 17.0,
        },
    }
    if not solution:
        for key in ("objective", "mip_gap", "x", "m", "r", "idle", "mu", "v",
                    "J_op_average"):
            result[key] = None
        result["status"] = "infeasible"
    return result


EXPECTED_MARKS = {
    (0, 1): "M1", (0, 2): "I", (0, 3): "M1", (0, 4): "R",
    (1, 1): "M1", (1, 2): "I", (1, 3): "M1", (1, 4): "P",
    (2, 1): "R", (2, 2): "M1", (2, 3): "D", (2, 4): "M1",
    (3, 1): "D", (3, 2): "M1", (3, 3): "P", (3, 4): "M1",
}


def _write(result: dict, path: Path) -> Path:
    solver._save_results(result, path)
    return path


def _schedule_figure(path: Path):
    view = plotter._normalise_solution(config.read_input(path), plotter._read_presolve_log(path))
    figure = plotter._draw_solution(view)
    figure.canvas.draw()
    return figure


def _axes(figure, title: str):
    return next(ax for ax in figure.axes if ax.get_title(loc="left") == title)


def _marks(ax) -> dict:
    """Map (row, column) -> text of every mark of the grid."""
    return {
        (int(round(text.get_position()[1])), int(round(text.get_position()[0]))): text.get_text()
        for text in ax.texts if text.get_text() not in ("transitory", "operating")
    }


def _summary(ax) -> dict:
    """Map label -> value of the summary panel (labels at x = 0, values at x = 0.70)."""
    labels = {text.get_position()[1]: text.get_text()
              for text in ax.texts if text.get_position()[0] == 0.0}
    values = {text.get_position()[1]: text.get_text()
              for text in ax.texts if text.get_position()[0] == 0.70}
    return {labels[y]: values[y] for y in labels}


@contextlib.contextmanager
def _inside(directory: Path):
    previous = os.getcwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)


# ---------------------------------------------------------------------------
# Schedule plot: layout
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("extension", [".yaml", ".json", ".h5"])
def test_schedule_layout(tmp_path, extension) -> None:
    figure = _schedule_figure(_write(_result(), tmp_path / f"r{extension}"))
    grid = _axes(figure, "Schedule and tracked mean degradation")
    assert [t.get_text() for t in grid.get_yticklabels()] == [
        "Train 1 — Wheelset\ngaussian", "Train 1 — Brake pad\nrainflow",
        "Train 2 — Wheelset\ngaussian", "Train 2 — Brake pad\nrainflow",
    ]
    assert [t.get_text() for t in grid.get_xticklabels()] == ["initial", "1", "2", "3", "4"]
    assert _marks(grid) == EXPECTED_MARKS
    dashed = [line for line in grid.lines if line.get_linestyle() == "--"]
    assert len(dashed) == 1 and list(dashed[0].get_xdata()) == [2.5, 2.5]
    assert {"transitory", "operating"} <= {t.get_text() for t in grid.texts}
    assert [t.get_text() for t in grid.get_legend().get_texts()] == [
        "P replacement", "R repair", "M$_j$ mission $j$",
        "D depot, no intervention", "I idle train",
    ]
    assert any(ax.get_ylabel() == "tracked mean / failure threshold" for ax in figure.axes)
    plt.close(figure)


def test_schedule_summary(tmp_path) -> None:
    figure = _schedule_figure(_write(_result(), tmp_path / "r.yaml"))
    summary = _summary(_axes(figure, "Run summary"))
    assert summary["Status"] == "time limit"
    assert summary["Objective"] == "7.5"
    assert summary["Dimensions"] == "F=2, M=1, L=2"
    assert summary["Horizon"] == "H1=2, H2=2, T=4"
    assert summary["Models"] == "gaussian, rainflow"
    assert summary["Repair laws"] == "ard1, ardinf"
    assert summary["Objective mode"] == "operating average"
    assert summary["J_op / H2"] == "1.25"
    assert summary["MIP gap"] == "3.10%"
    assert summary["Counts from"] == "performance"
    assert summary["Variables"] == "120"
    assert summary["Continuous variables"] == "80"
    assert summary["Integer variables"] == "40"
    assert summary["Linear rows"] == "300"
    assert summary["Optimizer"] == "2.50 s"
    assert summary["B&B nodes"] == "17"
    assert summary["Repairs"] == "2"
    assert summary["Replacements"] == "2"
    plt.close(figure)


def test_schedule_counts_from_runner_log(tmp_path) -> None:
    (tmp_path / "runner.out").write_text(
        "Optimize a model with 300 rows, 120 columns\n"
        "Presolved: 210 rows, 95 columns, 640 nonzeros\n"
        "Variable types: 60 continuous, 35 integer (35 binary)\n",
        encoding="utf-8",
    )
    figure = _schedule_figure(_write(_result(), tmp_path / "r.yaml"))
    summary = _summary(_axes(figure, "Run summary"))
    assert summary["Counts from"] == "runner.out (presolved)"
    assert summary["Variables"] == "95"
    assert summary["Continuous variables"] == "60"
    assert summary["Integer variables"] == "35"
    assert summary["Linear rows"] == "210"
    plt.close(figure)


def test_schedule_defaults(tmp_path) -> None:
    result = _result()
    for key in ("m", "r", "component_names", "model"):
        result[key] = None
    figure = _schedule_figure(_write(result, tmp_path / "r.yaml"))
    grid = _axes(figure, "Schedule and tracked mean degradation")
    assert [t.get_text() for t in grid.get_yticklabels()][:2] == [
        "Train 1 — Component 1\nunknown", "Train 1 — Component 2\nunknown",
    ]
    marks = _marks(grid)
    assert marks[(2, 1)] == "D" and marks[(3, 3)] == "D" and marks[(0, 4)] == "D"
    plt.close(figure)


# ---------------------------------------------------------------------------
# Schedule plot: files and errors
# ---------------------------------------------------------------------------
def test_plot_management_writes_png_and_pdf(tmp_path) -> None:
    path = _write(_result(), tmp_path / "r.h5")
    plotter.plot_management(str(path), str(tmp_path / "schedule"))
    image = mpimg.imread(tmp_path / "schedule.png")
    assert image.ndim == 3 and float(np.std(image)) > 0.05
    plotter.plot_management(str(path), str(tmp_path / "schedule.pdf"))
    assert (tmp_path / "schedule.pdf").read_bytes()[:4] == b"%PDF"
    with _inside(tmp_path):
        plotter.plot_management(str(path))
    assert (tmp_path / "output.png").is_file()


@pytest.mark.parametrize("extension", [".yaml", ".json", ".h5"])
def test_plot_management_without_solution(tmp_path, extension) -> None:
    path = _write(_result(solution=False), tmp_path / f"r{extension}")
    with pytest.raises(ValueError, match="missing keys: mu, x"):
        plotter.plot_management(str(path), str(tmp_path / "p.png"))


def test_plot_management_empty_result(tmp_path) -> None:
    path = tmp_path / "r.yaml"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing keys: F, M, L, H1, H2, tau, mu_0, mu, x"):
        plotter.plot_management(str(path), str(tmp_path / "p.png"))


def test_plot_management_path_errors(tmp_path) -> None:
    path = _write(_result(), tmp_path / "r.yaml")
    with pytest.raises(FileNotFoundError):
        plotter.plot_management(str(tmp_path / "absent.yaml"), str(tmp_path / "p.png"))
    (tmp_path / "r.txt").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported input file type"):
        plotter.plot_management(str(tmp_path / "r.txt"), str(tmp_path / "p.png"))
    with pytest.raises(ValueError, match="Unsupported plot file type"):
        plotter.plot_management(str(path), str(tmp_path / "p.jpg"))
    with pytest.raises(FileNotFoundError):
        plotter.plot_management(str(path), str(tmp_path / "absent" / "p.png"))
    with mock.patch.object(plotter.os, "access", return_value=False):
        with pytest.raises(PermissionError):
            plotter.plot_management(str(path), str(tmp_path / "p.png"))


def test_plot_management_shape_errors(tmp_path) -> None:
    cases = [
        ("tau", np.ones(2), "tau has shape"),
        ("mu_0", np.ones((2, 3)), "mu_0 has shape"),
        ("x", np.zeros((2, 3, 4)), "x has shape"),
        ("m", np.zeros((2, 2, 3)), "m has shape"),
        ("H2", 3, "H1 \\+ H2 must equal T"),
        ("T", 5, "the key T is 5"),
        ("component_names", ["Wheelset"], "component_names has length 1"),
        ("model", ["gaussian", "rainflow"], "model has shape"),
    ]
    for key, value, message in cases:
        result = _result()
        result[key] = value
        path = _write(result, tmp_path / f"{key}.yaml")
        with pytest.raises(ValueError, match=message):
            plotter.plot_management(str(path), str(tmp_path / "p.png"))


def test_parse_gurobi_presolve() -> None:
    text = "Presolved: 1,210 rows, 95 columns, 640 nonzeros\nVariable types: 60 continuous, 35 integer (35 binary)\n"
    assert plotter._parse_gurobi_presolve(text) == {
        "linear_rows": 1210, "linear_constraints": 1210, "variables": 95,
        "continuous_variables": 60, "integer_variables": 35, "binary_variables": 35,
    }
    assert plotter._parse_gurobi_presolve("no presolve here") == {}
    two_solves = "Presolved: 50 rows, 40 columns, 90 nonzeros\n" + text
    assert plotter._parse_gurobi_presolve(two_solves)["variables"] == 95


# ---------------------------------------------------------------------------
# Sweep plot
# ---------------------------------------------------------------------------
def _case(H2, status, cost, gap, bound=None):
    """A case row with the keys of the spec ("Keys of a case row") that the plot reads."""
    error = str(status).startswith("error")
    return {
        "H1": 4, "H2": H2, "T": 4 + H2,
        "dimensions": {"F": 4, "M": 1, "L": 1, "H1": 4, "H2": H2, "T": 4 + H2},
        "status": status, "objective": None if cost is None else cost * H2,
        "J_op": None if cost is None else cost * H2, "J_op_average": cost,
        "objective_bound": bound, "mip_gap": gap,
        "requested_mip_gap": None if error else 0.01,
        "formulation": None if error else {
            "variables": 30 * H2, "continuous_variables": 10 * H2,
            "integer_variables": 20 * H2, "linear_constraints": 50 * H2,
        },
        "timing": None if error else {"optimizer_call_seconds": 1.0},
    }


def _report(cases, maximum_gap=0.05, F=4, H1=4) -> dict:
    return {
        "input": "input.yaml", "complete": True, "objective": "minimize J_op / H2",
        "varied_parameter": "H2", "fixed_dimensions": {"F": F, "M": 1, "L": 1}, "H1": H1,
        "stopping_rule": {"enabled": False, "maximum_mip_gap_for_stopping": maximum_gap},
        "cases": cases,
    }


SWEEP_CASES = [
    _case(4, "optimal", 0.911251, 0.0, 0.911251),
    _case(8, "optimal", 0.798001, 0.0, 0.798001),
    _case(12, "time_limit", 0.617563, 0.1094, 0.55),
    _case(16, "error: out of memory", None, None),
]


def _sweep_figure(report: dict):
    figure = plotter._draw_horizon_sweep(plotter._normalise_horizon_sweep(report))
    figure.canvas.draw()
    return figure


def test_sweep_layout() -> None:
    figure = _sweep_figure(_report(SWEEP_CASES))
    assert figure.get_suptitle() == "Operating-horizon sweep (F=4, M=1, L=1, H1=4)"
    cost = _axes(figure, "Operating objective")
    assert [t.get_text() for t in cost.get_legend().get_texts()] == [
        "proven (optimal)", "feasible, not proven", "no feasible solution", "best feasible",
    ]
    assert {"0.911", "0.798"} <= {t.get_text() for t in cost.texts}
    gap = _axes(figure, "MIP gap")
    lines = [line for line in gap.lines if line.get_label() == "5% stopping gap"]
    assert len(lines) == 1 and list(lines[0].get_ydata()) == [5.0, 5.0]
    growth = _axes(figure, "Formulation growth")
    assert [line.get_label() for line in growth.lines] == [
        "continuous variables", "integer variables", "linear constraints",
    ]
    plt.close(figure)


def test_sweep_requested_gap_marks() -> None:
    figure = _sweep_figure(_report(SWEEP_CASES, maximum_gap=None))
    gap = _axes(figure, "MIP gap")
    assert not [line for line in gap.lines if line.get_label().endswith("stopping gap")]
    marks = [c for c in gap.collections if c.get_label() == "requested MIP gap"]
    assert len(marks) == 1
    assert marks[0].get_offsets().tolist() == [[4.0, 1.0], [8.0, 1.0], [12.0, 1.0]]
    plt.close(figure)


def test_sweep_selection_of_equal_H2(tmp_path) -> None:
    first = _report(SWEEP_CASES)
    second = _report([_case(12, "optimal", 0.6, 0.0, 0.6), _case(20, "time_limit", None, None, 0.45)])
    merged = plotter._merge_horizon_sweep_reports([first, second])
    view = plotter._normalise_horizon_sweep(merged)
    assert [row["H2"] for row in view["rows"]] == [4, 8, 12, 16, 20]
    assert view["rows"][2]["cost"] == 0.6 and view["rows"][2]["is_proven"]
    paths = [tmp_path / "a.yaml", tmp_path / "b.json"]
    paths[0].write_text(yaml.safe_dump(first, sort_keys=False), encoding="utf-8")
    paths[1].write_text(json.dumps(second), encoding="utf-8")
    plotter.plot_horizon_sweep([str(p) for p in paths], str(tmp_path / "sweep"))
    image = mpimg.imread(tmp_path / "sweep.png")
    assert image.ndim == 3 and float(np.std(image)) > 0.05
    with _inside(tmp_path):
        plotter.plot_horizon_sweep(str(paths[0]))
    assert (tmp_path / "horizon_sweep.png").is_file()


def test_sweep_errors(tmp_path) -> None:
    good = tmp_path / "a.yaml"
    good.write_text(yaml.safe_dump(_report(SWEEP_CASES)), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        plotter.plot_horizon_sweep(str(tmp_path / "absent.yaml"), str(tmp_path / "p.png"))
    (tmp_path / "a.h5").write_bytes(b"")
    with pytest.raises(ValueError, match="YAML or JSON"):
        plotter.plot_horizon_sweep(str(tmp_path / "a.h5"), str(tmp_path / "p.png"))
    with pytest.raises(ValueError, match="At least one"):
        plotter.plot_horizon_sweep([], str(tmp_path / "p.png"))
    for name, report, message in (
        ("f.yaml", _report(SWEEP_CASES, F=5), "changes fixed F"),
        ("h.yaml", _report(SWEEP_CASES, H1=5), "changes fixed H1"),
        ("c.yaml", {k: v for k, v in _report(SWEEP_CASES).items() if k != "cases"}, "no cases list"),
    ):
        (tmp_path / name).write_text(yaml.safe_dump(report), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            plotter.plot_horizon_sweep([str(good), str(tmp_path / name)], str(tmp_path / "p.png"))
    with pytest.raises(ValueError, match="cases"):
        plotter.plot_horizon_sweep(str(tmp_path / "c.yaml"), str(tmp_path / "p.png"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
PUBLIC_API = {
    "solve": "fleet_management.solver",
    "sweep_operating_horizons": "fleet_management.horizon_sweep",
    "sweep_horizon_grid": "fleet_management.horizon_sweep",
    "plot_management": "fleet_management.utils.plotter",
    "plot_horizon_sweep": "fleet_management.utils.plotter",
    "validate": "fleet_management.validation.validator",
    "validate_gamma_replay_schedule": "fleet_management.degradation_model.gamma_utils.gamma_replay_validator",
    "validate_gamma_replay_files": "fleet_management.degradation_model.gamma_utils.gamma_replay_validator",
    "validate_gamma_stochastic_schedule": "fleet_management.degradation_model.gamma_utils.gamma_stochastic_validator",
    "validate_gamma_schedule": "fleet_management.degradation_model.gamma_utils.gamma_stochastic_validator",
    "validate_gamma_schedule_files": "fleet_management.degradation_model.gamma_utils.gamma_stochastic_validator",
}


def test_public_api_without_gurobi() -> None:
    """``import fleet_management`` works with gurobipy blocked, and exports exactly the spec's names."""
    src = Path(plotter.__file__).resolve().parents[2]
    code = (
        "import sys\n"
        "sys.modules['gurobipy'] = None\n"
        "import fleet_management\n"
        "for name in fleet_management.__all__:\n"
        "    print(name, getattr(fleet_management, name).__module__)\n"
    )
    env = dict(os.environ, PYTHONPATH=str(src) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               MPLBACKEND="Agg")
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                               env=env, check=False)
    assert completed.returncode == 0, completed.stderr
    exported = dict(line.split() for line in completed.stdout.splitlines())
    assert list(exported) == list(PUBLIC_API)
    assert exported == PUBLIC_API
