"""Tests of the dashboards (spec, section "Dashboards").

The fixture is ``results/validator_tests/input.yaml`` with the valid result ``result.yaml``
(three trains, two missions, two components, H1 = 2, H2 = 3, T = 5). The tests call the
functions without the prefix ``render_``: none of them needs Streamlit or Gurobi.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.figure import Figure

from fleet_management import solver
from fleet_management.validation import dashboard_gamma, dashboard_inspector
from fleet_management.validation.dashboard_gamma import (
    gamma_cells,
    gamma_mean_figure,
    gamma_mean_table,
    gamma_overview,
    gamma_tail_figure,
    gamma_tail_heatmap_figure,
    gamma_tail_table,
    load_gamma_diagnostic,
)
from fleet_management.validation.dashboard_inspector import (
    cell_label,
    component_figure,
    day_marks,
    load_inspector,
    load_pair,
    loop_table,
    margin_heatmap_figure,
    margin_table,
    overview,
    reliability_caps,
    reliability_margins,
    timeline_figure,
)

FIXTURES = Path(__file__).resolve().parents[1] / "results" / "validator_tests"
INPUT = FIXTURES / "input.yaml"
RESULT = FIXTURES / "result.yaml"

# Margins of the fixture: rows train 1 Wheelset, train 1 Brake pad, train 2 Wheelset, ...;
# columns days 1 to 5.
MARGINS = np.array([
    [0.654747228, 0.575976674, 0.960291778, 0.846021224, 0.756250670],
    [0.352346972, 0.282346972, 0.472346972, 0.432346972, 0.362346972],
    [0.620312239, 0.526677250, 0.440242261, 0.960291778, 0.843856789],
    [0.522690506, 0.482690506, 0.442690506, 0.652690506, 0.612690506],
    [0.756412851, 0.756412851, 0.671289976, 0.591167100, 0.884998013],
    [1.997414907, 2.137414907, 1.757414907, 1.377414907, 2.597414907],
])


def test_modules_import_without_streamlit():
    for module in (dashboard_inspector, dashboard_gamma):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] != "streamlit" for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] != "streamlit"


def test_load_pair(tmp_path):
    cfg, result = load_pair(INPUT, RESULT)
    assert cfg.T == 5
    with pytest.raises(FileNotFoundError, match="Input file not found"):
        load_pair(tmp_path / "missing.yaml", RESULT)
    with pytest.raises(FileNotFoundError, match="Result file not found"):
        load_pair(INPUT, tmp_path / "missing.yaml")
    result["model"][0][0] = "wiener"
    path = tmp_path / "result.yaml"
    solver._save_results(result, path)
    with pytest.raises(ValueError, match="'model' differs"):
        load_pair(INPUT, path)


def test_cell_label():
    cfg, _ = load_pair(INPUT, RESULT)
    assert cell_label(cfg, 2, 0) == "rainflow (bernstein)"
    assert cell_label(cfg, 2, 1) == "rainflow (chernoff)"
    assert cell_label(cfg, 0, 1) == "inverse_gaussian"


def test_reliability_caps():
    cfg, _ = load_pair(INPUT, RESULT)
    caps = reliability_caps(cfg)
    assert caps[0, 1] == pytest.approx(0.49234697216756246, rel=1e-9)
    assert caps[1, 1] == pytest.approx(0.6726905060677382, rel=1e-9)
    assert int(np.isnan(caps).sum()) == 4


def test_reliability_margins():
    cfg, result = load_pair(INPUT, RESULT)
    margins = reliability_margins(cfg, result)
    assert margins.shape == (3, 2, 5)
    np.testing.assert_allclose(margins.reshape(6, 5), MARGINS, rtol=0.0, atol=1e-9)


def test_margin_violation():
    cfg, result = load_pair(INPUT, RESULT)
    result["mu"][0, 0, 2] = 0.99
    margins = reliability_margins(cfg, result)
    assert margins[0, 0, 2] == pytest.approx(-8.221738163816342e-06, rel=1e-6)
    table = margin_table(cfg, result, margins, tol=1e-6)
    violated = table[~table["holds"]]
    assert len(violated) == 1
    row = violated.iloc[0]
    assert (row["train"], row["component"], row["day"]) == (1, "Wheelset", 3)
    assert margin_table(cfg, result, margins, tol=1e-5)["holds"].all()
    # the side row mu <= tau decides when the quadratic row alone would hold
    result["mu"][0, 0, 2], result["v"][0, 0, 2] = 1.2, 0.0
    assert reliability_margins(cfg, result)[0, 0, 2] == pytest.approx(-0.2)


def test_margins_of_the_bounds_outside_the_fixture():
    # Markov, Cantelli and Hoeffding rows (Eqs. rf_markov, rf_cantelli, rf_hoeffding)
    nan = np.nan
    cfg = SimpleNamespace(
        F=1, L=3, T=1, model=np.array([["rainflow"] * 3], dtype=object),
        bound_method=np.array([["markov", "cantelli", "hoeffding"]], dtype=object),
        tau=np.ones((1, 3)), epsilon=np.full((1, 3), 0.1),
        s_chernoff=np.full((1, 3), nan), support_const=np.full((1, 3), nan))
    result = {"mu": [[[0.05], [0.5], [0.5]]], "v": [[[nan], [0.02], [nan]]],
              "R": [[[nan], [nan], [0.1]]], "K": None}
    margins = reliability_margins(cfg, result)[0, :, 0]
    assert margins.tolist() == pytest.approx([0.1 - 0.05, 0.25 - 9.0 * 0.02,
                                              0.25 - np.log(10.0) / 2.0 * 0.1])


def test_margin_table_and_overview():
    cfg, result = load_pair(INPUT, RESULT)
    table = margin_table(cfg, result, reliability_margins(cfg, result), tol=1e-6)
    assert len(table) == 30
    assert list(table.columns) == ["train", "component", "model", "day", "phase", "mu", "tau",
                                   "mu_over_tau", "margin", "holds"]
    assert table["phase"].tolist()[:5] == ["transitory"] * 2 + ["operating"] * 3
    assert table["mu_over_tau"].tolist() == pytest.approx((table["mu"] / table["tau"]).tolist())
    summary = overview(cfg, result, table)
    assert summary["reliability_violations"] == 0
    assert summary["largest_mu_over_tau"] == pytest.approx(0.33)
    first = summary["smallest_margins"].iloc[0]
    assert (first["train"], first["component"], first["day"]) == (1, "Brake pad", 2)
    assert first["margin"] == pytest.approx(0.282346972, abs=1e-9)
    assert summary["models"] == ["gamma", "gaussian", "inverse_gaussian", "rainflow (bernstein)",
                                 "rainflow (chernoff)", "wiener"]
    assert summary["status"] == "optimal"
    assert (summary["H1"], summary["H2"], summary["T"]) == (2, 3, 5)


def test_loop_table():
    cfg, result = load_pair(INPUT, RESULT)
    loops = loop_table(cfg, result, tol=1e-6)
    assert len(loops) == 18
    assert loops["quantity"].value_counts().to_dict() == {"mu": 6, "v": 3, "K": 1, "g_mu": 5,
                                                         "g_v": 2, "g_K": 1}
    assert loops["holds"].all()
    row = loops[(loops["train"] == 2) & (loops["component"] == "Wheelset")
                & (loops["quantity"] == "mu")].iloc[0]
    assert row["value_at_H1"] == pytest.approx(0.27)
    assert row["value_at_T"] == pytest.approx(0.08)
    assert row["margin"] == pytest.approx(0.19)


def test_day_marks():
    _, result = load_pair(INPUT, RESULT)
    assert day_marks(result, 2, 0) == ["I", "D", "M2", "M2", "P"]
    assert day_marks(result, 2, 1) == ["I", "R", "M2", "M2", "P"]
    assert day_marks(result, 0, 0) == ["M1", "M2", "P", "M1", "M2"]
    assert day_marks(result, 1, 1) == ["M2", "M1", "M1", "P", "M1"]
    result["m"][2, 0, 4] = 1.0      # a replacement counts before a repair
    assert day_marks(result, 2, 0)[4] == "P"


def test_figures():
    cfg, result = load_pair(INPUT, RESULT)
    margins = reliability_margins(cfg, result)
    timeline = timeline_figure(cfg, result, 2)
    heatmap = margin_heatmap_figure(cfg, margins)
    component = component_figure(cfg, result, 0)
    for figure in (timeline, heatmap, component):
        assert isinstance(figure, Figure)
    assert len(timeline.axes[0].lines) >= 4
    assert len(heatmap.axes[0].images) == 1
    assert timeline.axes[0].lines[0].get_ydata()[0] == pytest.approx(cfg.mu_0[2, 0])
    assert list(heatmap.axes[0].lines[0].get_xdata()) == [cfg.H1 - 0.5, cfg.H1 - 0.5]


def test_load_inspector():
    state = load_inspector(INPUT, RESULT, 1e-6)
    assert set(state) == {"cfg", "result", "margins", "table", "overview", "loops", "report",
                          "tol"}
    assert state["report"]["passed"]
    assert state["overview"]["reliability_violations"] == 0


def test_gamma_tables():
    cfg, result = load_pair(INPUT, RESULT)
    assert gamma_cells(cfg) == [(1, 1)]
    with pytest.raises(ValueError, match="no Gamma cell"):
        gamma_cells(SimpleNamespace(model=np.array([["gaussian"]], dtype=object)))

    means = gamma_mean_table(cfg, result, reliability_caps(cfg), tol=1e-6)
    assert len(means) == 5
    assert means.loc[means["day"] == 3, "margin"].iloc[0] == pytest.approx(0.4426905060677382)

    steps = [{"i": 1, "l": 1, "k": 0, "event": "mission", "exact_tail_upper_bound": 0.001},
             {"i": 1, "l": 1, "k": 1, "event": "mission", "exact_tail_upper_bound": 0.06}]
    tails = gamma_tail_table(cfg, {"steps": steps}, tol=1e-6)
    assert tails["day"].tolist() == [1, 2]
    assert tails["slack"].tolist() == pytest.approx([0.049, -0.01])
    assert tails["holds"].tolist() == [True, False]
    assert set(tails["component"]) == {"Brake pad"}

    replay = {"valid": False, "repairs": 1, "replacements": 2, "global_violations": ["text"],
              "violations": [{"i": 1}, {"i": 2}], "maximum_errors": {"cap": 0.5}}
    assert gamma_overview(replay) == {
        "valid": False, "repairs": 1, "replacements": 2, "global_violations": ["text"],
        "violations": 2, "maximum_errors": {"cap": 0.5},
    }

    heatmap = gamma_tail_heatmap_figure(cfg, tails)
    for figure in (gamma_mean_figure(cfg, means, 1, 1), gamma_tail_figure(cfg, tails, 1, 1),
                   heatmap):
        assert isinstance(figure, Figure)
    assert [text.get_text() for text in heatmap.axes[0].texts] == ["!"]


def test_load_gamma_diagnostic():
    state = load_gamma_diagnostic(INPUT, RESULT, 1e-6)
    assert state["overview"]["valid"]
    assert state["overview"]["repairs"] == 0
    assert state["overview"]["replacements"] == 1
    assert len(state["tails"]) == 5
    assert state["tails"]["holds"].all()
    assert len(state["means"]) == 5
