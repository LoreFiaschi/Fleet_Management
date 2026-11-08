"""Tests of step 3 for the sweep plot and the result inspector (spec, Plotting API and Layout, Dashboards).

M6: two reports without H1 plot, and the title leaves out H1.
M7: panels (a) and (b) draw the line and the labels of the cases with a feasible solution only.
M8: the selection for equal H2 also runs for one report.
M11: the Overview's largest mu/tau counts every cell and day with a number.

The reports and the result are written by hand in ``tmp_path``; no Gurobi is needed. The figure is kept by
wrapping ``matplotlib.figure.Figure.savefig`` with ``monkeypatch``.
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import yaml
from matplotlib.figure import Figure

from fleet_management.utils.plotter import plot_horizon_sweep
from fleet_management.validation.dashboard_inspector import load_inspector

VALIDATION = Path(__file__).parent / "data" / "tools" / "validation"

FORMULATION = {"variables": 10, "continuous_variables": 6, "integer_variables": 4, "linear_constraints": 12}


def capture(monkeypatch):
    kept = []
    original = Figure.savefig

    def savefig(self, *args, **kwargs):
        kept.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", savefig)
    return kept


def write_yaml(path, data):
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def case(h2, status="optimal", cost=1.0, bound=1.0, gap=0.0):
    return {"H2": h2, "status": status, "J_op_average": cost, "objective_bound": bound, "mip_gap": gap,
            "requested_mip_gap": 0.01, "formulation": dict(FORMULATION)}


def report(cases, with_h1=False):
    out = {"objective": "operating_average", "fixed_dimensions": {"F": 3, "M": 1, "L": 2}, "cases": cases}
    if with_h1:
        out["H1"] = 2
    return out


def texts_at(ax, h2):
    return [t.get_text() for t in ax.texts if getattr(t, "xy", None) is not None and float(t.xy[0]) == h2]


def line_x(ax):
    """The x values of every line drawn in the panel."""
    return [float(x) for line in ax.lines for x in np.asarray(line.get_xdata(), float)]


def line_points_at(ax, h2):
    """The y values that the grey lines draw at H2 = h2 (NaN excluded)."""
    ys = []
    for line in ax.lines:
        xs = np.asarray(line.get_xdata(), float)
        yv = np.asarray(line.get_ydata(), float)
        for x, y in zip(xs, yv):
            if x == h2 and np.isfinite(y):
                ys.append(float(y))
    return ys


def test_m6_two_reports_without_h1(monkeypatch, tmp_path):
    first = write_yaml(tmp_path / "a.yaml", report([case(1), case(2, cost=2.0)]))
    second = write_yaml(tmp_path / "b.yaml", report([case(3, cost=1.5), case(4, cost=1.2)]))
    kept = capture(monkeypatch)
    plot_horizon_sweep([str(first), str(second)], str(tmp_path / "sweep.png"))
    assert (tmp_path / "sweep.png").exists()
    title = kept[0]._suptitle.get_text()
    assert "F=3" in title and "M=1" in title and "L=2" in title
    assert "H1" not in title


def test_m7_infeasible_case_has_no_line_point_and_no_label(monkeypatch, tmp_path):
    cases = [
        case(1, gap=0.0, cost=3.0),
        case(2, gap=0.2, cost=2.0, status="time_limit"),
        # no feasible solution, yet the report gives a cost and a gap by hand
        case(3, status="infeasible", gap=0.5, cost=9.0),
        case(4, gap=0.0, cost=1.0),
    ]
    path = write_yaml(tmp_path / "r.yaml", report(cases, with_h1=True))
    kept = capture(monkeypatch)
    plot_horizon_sweep(str(path), str(tmp_path / "sweep.png"))
    cost_ax, gap_ax = kept[0].axes[0], kept[0].axes[1]
    assert texts_at(gap_ax, 3.0) == []
    assert line_points_at(gap_ax, 3.0) == []
    assert line_points_at(cost_ax, 3.0) == []
    # the feasible cases keep their labels and line points
    assert texts_at(gap_ax, 2.0) == ["20.0%"]
    assert line_points_at(gap_ax, 2.0) == [20.0]
    assert line_points_at(cost_ax, 4.0) == [1.0]


def test_m8_equal_h2_in_one_report(monkeypatch, tmp_path):
    cases = [
        case(1, cost=3.0),
        case(2, cost=2.0, gap=0.1, status="time_limit"),
        case(2, cost=2.0, gap=0.3, status="time_limit"),  # same cost and bound: the earlier case stays
        case(3, cost=1.0),
    ]
    path = write_yaml(tmp_path / "r.yaml", report(cases, with_h1=True))
    kept = capture(monkeypatch)
    plot_horizon_sweep(str(path), str(tmp_path / "sweep.png"))
    gap_ax = kept[0].axes[1]
    points = []
    for collection in gap_ax.collections:
        if "proven" not in collection.get_label():
            continue   # the reference marks of the requested gap are not cases
        for x, y in np.asarray(collection.get_offsets(), float).tolist():
            if x == 2.0:
                points.append(y)
    assert len(points) == 1
    assert abs(points[0] - 10.0) < 1e-9
    assert texts_at(gap_ax, 2.0) == ["10.0%"]


def test_m11_largest_mu_over_tau_counts_cells_without_margin(tmp_path):
    result = yaml.safe_load((VALIDATION / "gaussian_result.yaml").read_text(encoding="utf-8"))
    result["mu"][1][0][0] = 0.95   # the largest mu/tau (tau = 1) ...
    result["v"][1][0][0] = float("nan")   # ... in a cell whose margin is NaN
    result_path = write_yaml(tmp_path / "result.yaml", result)
    state = load_inspector(str(VALIDATION / "gaussian_input.yaml"), str(result_path), 1e-6)
    assert math.isnan(float(state["margins"][1, 0, 0]))
    assert len(state["table"]) == 5   # the cell with the NaN margin is not in the table
    assert abs(state["overview"]["largest_mu_over_tau"] - 0.95) < 1e-12
