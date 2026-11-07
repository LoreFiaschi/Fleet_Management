"""Tests of the two plots against the spec, Section "Plotting API and Layout" (lines 2276-2378 of the
lore copy of spec.tex). The inputs are hand-written files in tests/data/tools/plots/; no Gurobi is needed.

The only seam is ``matplotlib.figure.Figure.savefig``, wrapped with ``monkeypatch`` to keep the figure
that a plot function saves, so that the tests can read its axes, texts, lines, patches, legend and
colour bar. Each test also writes its image to ``tmp_path``.
"""

import json
import os
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.text
import numpy as np
import pytest
import yaml
from matplotlib.figure import Figure

import fleet_management
from fleet_management import plot_horizon_sweep, plot_management

DATA = Path(__file__).parent / "data" / "tools" / "plots"
COLORBAR_LABEL = "tracked mean / failure threshold"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def capture(monkeypatch):
    """Keep every figure saved through Figure.savefig; return the list."""
    kept = []
    original = Figure.savefig

    def savefig(self, *args, **kwargs):
        kept.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", savefig)
    return kept


def draw_schedule(monkeypatch, tmp_path, result, name="plot.png"):
    """Plot a result file (a path in DATA, or a mapping written first) and return the figure."""
    kept = capture(monkeypatch)
    if isinstance(result, dict):
        result_path = write_yaml(tmp_path, "result.yaml", result)
    else:
        result_path = result
    image = tmp_path / name
    plot_management(str(result_path), str(image))
    assert image.is_file() and image.stat().st_size > 0
    assert len(kept) == 1
    return kept[0]


def draw_sweep(monkeypatch, tmp_path, reports, name="sweep.png"):
    kept = capture(monkeypatch)
    image = tmp_path / name
    plot_horizon_sweep(reports, str(image))
    assert image.is_file() and image.stat().st_size > 0
    assert len(kept) == 1
    return kept[0]


def write_yaml(folder, name, data):
    path = Path(folder) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def load(name):
    return yaml.safe_load((DATA / name).read_text(encoding="utf-8"))


def norm_mark(text):
    return text.replace("$", "").replace("_", "").replace("{", "").replace("}", "").strip()


def grid_axes(fig):
    """The axes of the grid: the one with the most patches (one rectangle per cell and column)."""
    return max(fig.axes, key=lambda ax: len(ax.patches))


def summary_axes(fig):
    for ax in fig.axes:
        if any(t.get_text() == "Status" for t in ax.texts):
            return ax
    raise AssertionError("no summary panel")


def colorbar_axes(fig):
    for ax in fig.axes:
        if ax.get_ylabel() == COLORBAR_LABEL:
            return ax
    raise AssertionError("no colour bar")


def row_labels(ax):
    return [t.get_text() for t in ax.get_yticklabels()]


def column_labels(ax):
    return [t.get_text() for t in ax.get_xticklabels()]


def marks(ax, rows, days):
    """Mark texts of the grid as {(row, day): mark}; days from 1 (the column 'initial' is 0)."""
    found = {}
    for t in ax.texts:
        x, y = t.get_position()
        mark = norm_mark(t.get_text())
        if re.fullmatch(r"P|R|D|I|M\d+", mark) and 1 <= x <= days and 0 <= y <= rows - 1:
            assert (round(y), round(x)) not in found, "two marks on one field"
            found[(round(y), round(x))] = mark
    return found


def cell_colors(ax, rows, columns):
    """Face colours of the rectangles as an array (rows, columns, 4)."""
    out = np.full((rows, columns, 4), np.nan)
    for p in ax.patches:
        x, y = p.get_xy()
        col, row = int(round(x + p.get_width() / 2)), int(round(y + p.get_height() / 2))
        if 0 <= row < rows and 0 <= col < columns:
            out[row, col] = p.get_facecolor()
    assert not np.isnan(out).any()
    return out


def summary(fig):
    """The summary panel as {label: Text of its value}."""
    ax = summary_axes(fig)
    x_label = min(t.get_position()[0] for t in ax.texts)
    labels = [t for t in ax.texts if t.get_position()[0] == x_label]
    values = [t for t in ax.texts if t.get_position()[0] != x_label]
    out = {}
    for lab in labels:
        near = [v for v in values if abs(v.get_position()[1] - lab.get_position()[1]) < 1e-6]
        assert len(near) == 1, lab.get_text()
        out[lab.get_text()] = near[0]
    return out


def find_value(panel, predicate):
    hits = [(k, v) for k, v in panel.items() if predicate(k.lower())]
    assert len(hits) == 1, (list(panel), hits)
    return hits[0][1]


def number(text):
    return float(text.get_text().replace(",", "").replace("%", "").strip())


def kind_of_color(color):
    r, g, b = mcolors.to_rgb(color)
    if g > r and g > b:
        return "green"
    if r > g > b and r >= 0.6 and r - b >= 0.4 and 0.3 <= g / r <= 0.9:
        return "amber"
    if max(r, g, b) - min(r, g, b) < 0.15 and 0.2 < (r + g + b) / 3 < 0.8:
        return "grey"
    return "other"


# ---------------------------------------------------------------------------
# schedule plot: rows, columns, lines, colour
# ---------------------------------------------------------------------------

def test_schedule_rows_and_columns(monkeypatch, tmp_path):
    """Spec 2317-2318: one row per cell, ordered by train then component, two-line labels; columns
    'initial' and 1..T (2299)."""
    fig = draw_schedule(monkeypatch, tmp_path, DATA / "schedule_marks.yaml", "schedule_marks.png")
    ax = grid_axes(fig)
    labels = row_labels(ax)
    assert len(labels) == 4
    expected = [("1", "Pad", "gaussian"), ("1", "Door", "gamma"), ("2", "Pad", "wiener"), ("2", "Door", "inverse_gaussian")]
    for label, (train, name, model) in zip(labels, expected):
        lines = label.split("\n")
        assert len(lines) == 2
        assert lines[0].startswith(f"Train {train}") and lines[0].endswith(name)
        assert re.search(r"[-–—]", lines[0][len(f"Train {train}"):-len(name)])
        assert lines[1] == model
    assert column_labels(ax) == ["initial", "1", "2", "3", "4"]
    assert len(ax.patches) == 4 * 5


def test_schedule_default_names_and_unknown_model(monkeypatch, tmp_path):
    """Spec 2300, 2303: without the key model every row shows 'unknown'; the default name of component
    l is 'Component l'."""
    fig = draw_schedule(monkeypatch, tmp_path, DATA / "schedule_precedence.yaml")
    labels = row_labels(grid_axes(fig))
    assert len(labels) == 1
    first, second = labels[0].split("\n")
    assert first.startswith("Train 1") and first.endswith("Component 1")
    assert second == "unknown"
    fig = draw_schedule(monkeypatch, tmp_path, DATA / "schedule_no_mr.yaml")
    firsts = [lab.split("\n")[0] for lab in row_labels(grid_axes(fig))]
    assert [f.split()[-2:] for f in firsts] == [["Component", "1"], ["Component", "2"]]


def test_schedule_transitory_operating_and_dashed_line(monkeypatch, tmp_path):
    """Spec 2318: a vertical dashed line after day H1 (H1 = 2, so between the columns 2 and 3); the
    labels 'transitory' and 'operating' above the grid, on the two sides of the line."""
    fig = draw_schedule(monkeypatch, tmp_path, DATA / "schedule_marks.yaml")
    ax = grid_axes(fig)
    dashed = [l for l in ax.lines
              if l.get_linestyle() in ("--", "dashed") and len(set(np.asarray(l.get_xdata(), float))) == 1]
    assert len(dashed) == 1
    x_line = float(dashed[0].get_xdata()[0])
    assert x_line == 2.5
    texts = {t.get_text(): t for t in ax.texts}
    assert texts["transitory"].get_position()[0] < x_line < texts["operating"].get_position()[0]
    # the same for H1 = 1 (precedence file: columns initial, 1..5)
    fig = draw_schedule(monkeypatch, tmp_path, DATA / "schedule_precedence.yaml")
    ax = grid_axes(fig)
    dashed = [l for l in ax.lines
              if l.get_linestyle() in ("--", "dashed") and len(set(np.asarray(l.get_xdata(), float))) == 1]
    assert [float(l.get_xdata()[0]) for l in dashed] == [1.5]


def test_schedule_heavier_line_between_trains(monkeypatch, tmp_path):
    """Spec 2317: a heavier line separates the trains. Rows 0,1 are train 1 and rows 2,3 train 2, so the
    horizontal line at y = 1.5 is wider than the ones inside a train (y = 0.5 and 2.5)."""
    fig = draw_schedule(monkeypatch, tmp_path, DATA / "schedule_marks.yaml")
    ax = grid_axes(fig)
    width = {}
    for l in ax.lines:
        y = np.asarray(l.get_ydata(), float)
        if len(y) == 2 and y[0] == y[1] and l.get_linestyle() in ("-", "solid"):
            width[float(y[0])] = l.get_linewidth()
    assert {0.5, 1.5, 2.5} <= set(width)
    assert width[1.5] > width[0.5]
    assert width[1.5] > width[2.5]


def test_schedule_colour_scale_and_colorbar(monkeypatch, tmp_path):
    """Spec 2319: a green-to-red scale of mu/tau clipped to [0,1]; colour bar 'tracked mean / failure
    threshold'. Equal ratios have equal colours, also in the column 'initial' (mu_0/tau)."""
    fig = draw_schedule(monkeypatch, tmp_path, DATA / "schedule_marks.yaml")
    ax = grid_axes(fig)
    col = cell_colors(ax, 4, 5)
    # cell (train 1, Pad): tau 1, mu_0 0, mu 0, 0.5, 1, 2
    r0, g0 = col[0, 0, 0], col[0, 0, 1]
    assert g0 > r0                                   # ratio 0: green side
    assert np.allclose(col[0, 0], col[0, 1])         # mu_0 = 0 and mu = 0 give the same colour
    assert col[0, 3, 0] > col[0, 3, 1]               # ratio 1: red side
    assert np.allclose(col[0, 3], col[0, 4])         # ratio 2 is clipped to ratio 1
    redness = [col[0, c, 0] - col[0, c, 1] for c in (1, 2, 3)]
    assert redness[0] < redness[1] < redness[2]      # 0 < 0.5 < 1 goes from green to red
    # ratio 0.5 in four places: Pad day 2 (0.5/1), Train 2 Pad initial (0.5/1), Door initial (1.0/2)
    assert np.allclose(col[0, 2], col[2, 0])
    assert np.allclose(col[0, 2], col[1, 0])
    # Door of train 1, tau 2: mu 2 is ratio 1, like mu 1 with tau 1
    assert np.allclose(col[1, 3], col[0, 3]) and np.allclose(col[1, 4], col[0, 3])
    cb = colorbar_axes(fig)
    assert tuple(cb.get_ylim()) == (0.0, 1.0)


# ---------------------------------------------------------------------------
# schedule plot: marks
# ---------------------------------------------------------------------------

def test_schedule_marks_each_field(monkeypatch, tmp_path):
    """Spec 2320-2329: one mark per field of a day (4 rows x 4 days), none in the column 'initial'.
    P replacement, R repair, M1 mission, D depot without intervention, I idle train."""
    fig = draw_schedule(monkeypatch, tmp_path, DATA / "schedule_marks.yaml")
    ax = grid_axes(fig)
    found = marks(ax, 4, 4)
    expected = {
        (0, 1): "P", (1, 1): "R", (0, 2): "M1", (1, 2): "M1", (0, 3): "D", (1, 3): "D", (0, 4): "M1", (1, 4): "M1",
        (2, 1): "M1", (3, 1): "M1", (2, 2): "I", (3, 2): "I", (2, 3): "M1", (3, 3): "M1", (2, 4): "R", (3, 4): "D",
    }
    assert found == expected
    # no text mark in the column 'initial' (x = 0)
    assert not [t for t in ax.texts if abs(t.get_position()[0]) < 0.25
                and re.fullmatch(r"P|R|D|I|M\d+", norm_mark(t.get_text()))]


def test_schedule_marks_first_rule_and_threshold(monkeypatch, tmp_path):
    """Spec 2320, 2324-2328: the first rule that applies gives the mark (P before R before M_j before D
    before I); a value counts as 1 only if it is above 0.5 (0.5 is 0, 0.51 is 1); the mark of a mission
    shows j."""
    fig = draw_schedule(monkeypatch, tmp_path, DATA / "schedule_precedence.yaml")
    found = marks(grid_axes(fig), 1, 5)
    assert found == {(0, 1): "P", (0, 2): "R", (0, 3): "R", (0, 4): "M2", (0, 5): "I"}


def test_schedule_marks_without_m_and_r(monkeypatch, tmp_path):
    """Spec 2302: missing m and r are read as zero, so a depot day without them is D and the repair and
    replacement counts are zero; status time_limit is plotted (2301)."""
    fig = draw_schedule(monkeypatch, tmp_path, DATA / "schedule_no_mr.yaml")
    found = marks(grid_axes(fig), 2, 2)
    assert found == {(0, 1): "D", (1, 1): "D", (0, 2): "M1", (1, 2): "M1"}
    panel = summary(fig)
    assert number(find_value(panel, lambda k: "repairs" in k)) == 0
    assert number(find_value(panel, lambda k: "replacement" in k)) == 0


def test_schedule_legend_five_marks(monkeypatch, tmp_path):
    """Spec 2320: a legend names the five marks."""
    fig = draw_schedule(monkeypatch, tmp_path, DATA / "schedule_marks.yaml")
    legends = [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
    assert len(legends) == 1
    texts = [norm_mark(t.get_text()) for t in legends[0].get_texts()]
    assert len(texts) == 5
    assert [t[0] for t in texts] == ["P", "R", "M", "D", "I"]
    assert "replacement" in texts[0] and "repair" in texts[1] and "mission" in texts[2]
    assert "depot" in texts[3] and "idle" in texts[4]


# ---------------------------------------------------------------------------
# schedule plot: summary panel
# ---------------------------------------------------------------------------

def test_schedule_summary_values(monkeypatch, tmp_path):
    """Spec 2331-2342: the panel 'Run summary' with the values of the result."""
    fig = draw_schedule(monkeypatch, tmp_path, DATA / "schedule_marks.yaml")
    ax = summary_axes(fig)
    assert ax.get_title(loc="left") == "Run summary" or any(
        t.get_text() == "Run summary" for t in fig.findobj(matplotlib.text.Text))
    panel = summary(fig)
    assert find_value(panel, lambda k: k == "status").get_text() == "optimal"
    assert "12.5" in find_value(panel, lambda k: k == "objective").get_text()
    ints = lambda s: [int(v) for v in re.findall(r"=\s*(\d+)", s)]
    assert ints(find_value(panel, lambda k: k.startswith("dimension")).get_text()) == [2, 1, 2]    # F, M, L
    assert ints(find_value(panel, lambda k: k.startswith("horizon")).get_text()) == [2, 2, 4]      # H1, H2, T
    models = find_value(panel, lambda k: k.startswith("models")).get_text()
    assert all(name in models for name in ("gaussian", "gamma", "wiener", "inverse_gaussian"))
    laws = find_value(panel, lambda k: "law" in k).get_text()
    assert "ard1" in laws and "ardinf" in laws
    assert "operating" in find_value(panel, lambda k: k.startswith("objective mode")).get_text().lower()
    assert number(find_value(panel, lambda k: "h2" in k and "j" in k)) == 3.25
    gap = find_value(panel, lambda k: "gap" in k)
    assert "%" in gap.get_text() and abs(number(gap) - 1.23) < 1e-9          # mip_gap 0.0123 in percent


def test_schedule_summary_counts_from_performance(monkeypatch, tmp_path):
    """Spec 2339-2341, 2343: without a runner.out the counts come from performance; repairs and
    replacements are counted over all cells and days (m has two ones, r has one)."""
    fig = draw_schedule(monkeypatch, tmp_path, DATA / "schedule_marks.yaml")
    panel = summary(fig)
    assert find_value(panel, lambda k: k.startswith("counts")).get_text() == "performance"
    assert number(find_value(panel, lambda k: k == "variables")) == 1234
    assert number(find_value(panel, lambda k: "continuous" in k)) == 1000
    assert number(find_value(panel, lambda k: "integer" in k)) == 234
    assert number(find_value(panel, lambda k: "row" in k or "constraint" in k)) == 5678
    assert number(find_value(panel, lambda k: "node" in k)) == 4321
    assert number(find_value(panel, lambda k: "repairs" in k)) == 2
    assert number(find_value(panel, lambda k: "replacement" in k)) == 1


def test_schedule_status_colours(monkeypatch, tmp_path):
    """Spec 2334: status in green if optimal, otherwise amber; 'unknown' in grey without the key."""
    fig = draw_schedule(monkeypatch, tmp_path, DATA / "schedule_marks.yaml")
    value = find_value(summary(fig), lambda k: k == "status")
    assert value.get_text() == "optimal" and kind_of_color(value.get_color()) == "green"
    fig = draw_schedule(monkeypatch, tmp_path, DATA / "schedule_no_mr.yaml")
    value = find_value(summary(fig), lambda k: k == "status")
    assert value.get_text().replace(" ", "_") == "time_limit" and kind_of_color(value.get_color()) == "amber"
    fig = draw_schedule(monkeypatch, tmp_path, DATA / "schedule_precedence.yaml")
    value = find_value(summary(fig), lambda k: k == "status")
    assert value.get_text() == "unknown" and kind_of_color(value.get_color()) == "grey"


def _copy_result(folder):
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "result.yaml"
    shutil.copy(DATA / "schedule_marks.yaml", target)
    return target


LOG_TWO_SOLVES = """Gurobi Optimizer version 13.0.0
Optimize a model with 500 rows, 400 columns and 2000 nonzeros
Variable types: 99 continuous, 98 integer (97 binary)
Presolved: 10 rows, 8 columns, 30 nonzeros
Variable types: 5 continuous, 3 integer (3 binary)
Explored 1 nodes
Optimize a model with 700 rows, 600 columns and 3000 nonzeros
Variable types: 90 continuous, 89 integer (88 binary)
Presolved: 120 rows, 90 columns, 400 nonzeros
Variable types: 60 continuous, 30 integer (25 binary)
Variable types: 77 continuous, 66 integer (55 binary)
Explored 9 nodes
"""


def test_schedule_summary_counts_from_runner_out(monkeypatch, tmp_path):
    """Spec 2343: a runner.out beside the result gives the presolved counts (last 'Presolved:' line and
    the first 'Variable types:' line after it), and the panel says so."""
    result = _copy_result(tmp_path / "run")
    (tmp_path / "run" / "runner.out").write_text(LOG_TWO_SOLVES, encoding="utf-8")
    fig = draw_schedule(monkeypatch, tmp_path, result)
    panel = summary(fig)
    source = find_value(panel, lambda k: k.startswith("counts")).get_text()
    assert source != "performance"
    assert any(w in source.lower() for w in ("presolve", "runner", "log"))
    assert number(find_value(panel, lambda k: k == "variables")) == 90         # 'columns' of the last Presolved
    assert number(find_value(panel, lambda k: "continuous" in k)) == 60
    assert number(find_value(panel, lambda k: "integer" in k)) == 30
    assert number(find_value(panel, lambda k: "row" in k or "constraint" in k)) == 120
    # the nodes and the repair counts are not from the log
    assert number(find_value(panel, lambda k: "node" in k)) == 4321
    assert number(find_value(panel, lambda k: "repairs" in k)) == 2


@pytest.mark.parametrize("log", [
    # 'Variable types' only before the 'Presolved' line: no pair
    "Variable types: 99 continuous, 98 integer (97 binary)\nPresolved: 120 rows, 90 columns, 400 nonzeros\n",
    # a 'Presolved' line alone
    "Presolved: 120 rows, 90 columns, 400 nonzeros\nExplored 1 nodes\n",
    # no relevant line
    "Gurobi Optimizer version 13.0.0\nExplored 1 nodes\n",
])
def test_schedule_runner_out_without_pair_is_not_used(monkeypatch, tmp_path, log):
    """Spec 2343: a log without such a pair is not used; the counts come from performance."""
    result = _copy_result(tmp_path / "run")
    (tmp_path / "run" / "runner.out").write_text(log, encoding="utf-8")
    fig = draw_schedule(monkeypatch, tmp_path, result)
    panel = summary(fig)
    assert find_value(panel, lambda k: k.startswith("counts")).get_text() == "performance"
    assert number(find_value(panel, lambda k: k == "variables")) == 1234
    assert number(find_value(panel, lambda k: "continuous" in k)) == 1000


# ---------------------------------------------------------------------------
# schedule plot: files, formats and errors
# ---------------------------------------------------------------------------

def test_schedule_default_image_name(monkeypatch, tmp_path):
    """Spec 2293: the default image is 'output.png', in the working directory."""
    monkeypatch.chdir(tmp_path)
    plot_management(str(DATA / "schedule_marks.yaml"))
    assert (tmp_path / "output.png").is_file()
    assert (tmp_path / "output.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_schedule_path_without_extension_gets_png(tmp_path):
    """Spec 2293: a path without extension gets .png."""
    plot_management(str(DATA / "schedule_marks.yaml"), str(tmp_path / "image"))
    assert (tmp_path / "image.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert not (tmp_path / "image").exists()


def test_schedule_pdf_accepted(tmp_path):
    """Spec 2293: .png or .pdf."""
    plot_management(str(DATA / "schedule_marks.yaml"), str(tmp_path / "image.pdf"))
    assert (tmp_path / "image.pdf").read_bytes()[:4] == b"%PDF"


def test_schedule_json_and_yml_results(monkeypatch, tmp_path):
    """Spec 2292: the result file may be YAML or JSON (.yml is YAML, 1871-1872)."""
    data = load("schedule_marks.yaml")
    json_path = tmp_path / "result.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")
    fig = draw_schedule(monkeypatch, tmp_path, json_path, "from_json.png")
    assert len(marks(grid_axes(fig), 4, 4)) == 16
    yml_path = write_yaml(tmp_path, "result.yml", data)
    fig = draw_schedule(monkeypatch, tmp_path, yml_path, "from_yml.png")
    assert len(marks(grid_axes(fig), 4, 4)) == 16


def test_schedule_missing_files(tmp_path):
    """Spec 2308: FileNotFoundError for a missing result file and for a missing image folder."""
    with pytest.raises(FileNotFoundError):
        plot_management(str(tmp_path / "absent.yaml"), str(tmp_path / "x.png"))
    with pytest.raises(FileNotFoundError):
        plot_management(str(DATA / "schedule_marks.yaml"), str(tmp_path / "no_folder" / "x.png"))


def test_schedule_unwritable_folder(tmp_path):
    """Spec 2309: PermissionError when the image folder is not writable (only where the system can
    make a folder read-only for the current user)."""
    folder = tmp_path / "locked"
    folder.mkdir()
    os.chmod(folder, 0o500)
    try:
        if os.access(folder, os.W_OK):
            return
        with pytest.raises(PermissionError):
            plot_management(str(DATA / "schedule_marks.yaml"), str(folder / "x.png"))
    finally:
        os.chmod(folder, 0o700)


def test_schedule_not_a_mapping(tmp_path):
    """Spec 2310: TypeError when the data of the result file are not a mapping."""
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(TypeError):
        plot_management(str(path), str(tmp_path / "x.png"))


def test_schedule_extension_errors(tmp_path):
    """Spec 2311: ValueError when either file has another extension."""
    bad = tmp_path / "result.txt"
    shutil.copy(DATA / "schedule_marks.yaml", bad)
    with pytest.raises(ValueError):
        plot_management(str(bad), str(tmp_path / "x.png"))
    with pytest.raises(ValueError):
        plot_management(str(DATA / "schedule_marks.yaml"), str(tmp_path / "x.gif"))


@pytest.mark.parametrize("key", ["F", "M", "L", "H1", "H2", "tau", "mu_0", "mu", "x"])
def test_schedule_missing_key(tmp_path, key):
    """Spec 2311: ValueError when the result lacks one of F, M, L, H1, H2, tau, mu_0, mu, x; the message
    names the key."""
    data = load("schedule_marks.yaml")
    del data[key]
    path = write_yaml(tmp_path, "result.yaml", data)
    with pytest.raises(ValueError, match=re.escape(key)):
        plot_management(str(path), str(tmp_path / "x.png"))


def test_schedule_missing_keys_all_named(tmp_path):
    """Spec 2311: the message names the keys (all that are missing)."""
    data = load("schedule_marks.yaml")
    del data["tau"], data["mu_0"]
    path = write_yaml(tmp_path, "result.yaml", data)
    with pytest.raises(ValueError) as info:
        plot_management(str(path), str(tmp_path / "x.png"))
    assert "tau" in str(info.value) and "mu_0" in str(info.value)


def _mutate_tau(d): d["tau"] = [[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]
def _mutate_mu0(d): d["mu_0"] = [0.0, 1.0]
def _mutate_mu_days(d): d["mu"] = [[[0.0] * 3] * 2] * 2          # T = 3 against H1 + H2 = 4
def _mutate_mu_rows(d): d["mu"] = d["mu"][:1]
def _mutate_x_depot(d): d["x"] = [[[0] * 4] * 3] * 2             # M + 1 = 3 rows instead of 2
def _mutate_x_days(d): d["x"] = [[[0] * 3] * 2] * 2
def _mutate_m(d): d["m"] = [[[0] * 3] * 2] * 2
def _mutate_r(d): d["r"] = [[[0] * 4] * 2]
def _mutate_h2(d): d["H2"] = 3                                   # H1 + H2 = 5, T = 4
def _mutate_t_key(d): d["T"] = 7                                 # the key T must equal the length of mu
def _mutate_names_string(d): d["component_names"] = "Pad"
def _mutate_names_length(d): d["component_names"] = ["Pad", "Door", "Extra"]


@pytest.mark.parametrize("mutate", [
    _mutate_tau, _mutate_mu0, _mutate_mu_days, _mutate_mu_rows, _mutate_x_depot, _mutate_x_days, _mutate_m,
    _mutate_r, _mutate_h2, _mutate_t_key, _mutate_names_string, _mutate_names_length,
])
def test_schedule_shape_and_name_errors(tmp_path, mutate):
    """Spec 2299, 2311: ValueError for a shape that differs from the output section, H1 + H2 != T, a
    key T that differs from the length of mu, and component_names that is not a list of L entries."""
    data = load("schedule_marks.yaml")
    mutate(data)
    path = write_yaml(tmp_path, "result.yaml", data)
    with pytest.raises(ValueError):
        plot_management(str(path), str(tmp_path / "x.png"))


@pytest.mark.parametrize("key", ["tau", "mu_0", "mu", "x", "m", "r"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_schedule_non_finite_entry(tmp_path, key, bad):
    """Spec 2311: ValueError when an entry of tau, mu_0, mu, x, m or r is not a finite number."""
    data = load("schedule_marks.yaml")
    arr = np.array(data[key], dtype=float)
    arr.flat[0] = bad
    data[key] = arr.tolist()
    path = write_yaml(tmp_path, "result.yaml", data)
    with pytest.raises(ValueError):
        plot_management(str(path), str(tmp_path / "x.png"))


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_schedule_tau_not_positive(tmp_path, bad):
    """Spec 2311: ValueError when an entry of tau is not positive."""
    data = load("schedule_marks.yaml")
    data["tau"][1][0] = bad
    path = write_yaml(tmp_path, "result.yaml", data)
    with pytest.raises(ValueError):
        plot_management(str(path), str(tmp_path / "x.png"))


# ---------------------------------------------------------------------------
# sweep plot
# ---------------------------------------------------------------------------

def sweep_panels(fig):
    assert len(fig.axes) >= 3
    return fig.axes[0], fig.axes[1], fig.axes[2]


def collection(ax, kind):
    """Points of the collection whose label (legend entry) names the kind of mark."""
    def match(label):
        label = label.lower()
        if kind == "proven":
            return "proven" in label and "not" not in label
        if kind == "feasible":
            return "feasible" in label and "not proven" in label
        if kind == "none":
            return "no feasible" in label
        if kind == "best":
            return "best" in label
        raise ValueError(kind)
    hits = [c for c in ax.collections if match(c.get_label())]
    assert len(hits) == 1, (kind, [c.get_label() for c in ax.collections])
    return [tuple(p) for p in np.asarray(hits[0].get_offsets(), float).tolist()]


def axes_fraction_y(ax, kind):
    """Vertical positions of a collection in axes fractions (0 bottom, 1 top), whatever its transform."""
    hit = [c for c in ax.collections if c.get_label() and "no feasible" in c.get_label().lower()]
    assert len(hit) == 1
    c = hit[0]
    display = c.get_offset_transform().transform(np.asarray(c.get_offsets(), float))
    return ax.transAxes.inverted().transform(display)[:, 1]


def assert_points(points, expected):
    points = sorted(points)
    assert len(points) == len(expected), (points, expected)
    for (x, y), (ex, ey) in zip(points, sorted(expected)):
        assert x == ex and abs(y - ey) < 1e-6, (points, expected)


def annotations(ax):
    """Text written beside a point as {(x, y): text}."""
    out = {}
    for t in ax.texts:
        xy = getattr(t, "xy", None)
        assert xy is not None
        out[(float(xy[0]), float(xy[1]))] = t.get_text()
    return out


def suptitle_text(fig):
    return fig._suptitle.get_text() if fig._suptitle is not None else None


def test_sweep_selection_and_marks_panel_a(monkeypatch, tmp_path):
    """Spec 2368, 2375: for equal H2 the plot keeps the case with a solution, then the smaller J_op/H2
    (numpy.isclose), then the larger bound, then the earlier report. Panel (a) marks proven, feasible not
    proven, no feasible solution, best feasible."""
    fig = draw_sweep(monkeypatch, tmp_path, [str(DATA / "sweep_a.yaml"), str(DATA / "sweep_b.yaml")], "sweep_ab.png")
    a, _, _ = sweep_panels(fig)
    # H2=1 sweep_a (the other has no solution); 2 sweep_b (isclose J, larger bound); 4 sweep_b (the case of
    # sweep_a is an error row); 5 sweep_a only; 6 sweep_a (same J, same bound, earlier report);
    # 7 sweep_b (smaller J); 8 sweep_b (same J, the other has no bound)
    assert_points(collection(a, "proven"), [(1, 5.0), (2, 4.0), (5, 3.0), (6, 2.5)])
    assert_points(collection(a, "feasible"), [(4, 3.5), (7, 2.0), (8, 1.5)])
    assert_points(collection(a, "best"), [(8, 1.5)])
    none = collection(a, "none")
    assert sorted(p[0] for p in none) == [3.0, 9.0]                    # infeasible case and error row
    # a case without a feasible solution is marked at the bottom of the panel
    assert all(0.0 <= f <= 0.25 for f in axes_fraction_y(a, "none"))


def test_sweep_values_shown_panel_a(monkeypatch, tmp_path):
    """Spec 2375: the values of the proven cases and of the best feasible case are shown."""
    fig = draw_sweep(monkeypatch, tmp_path, [str(DATA / "sweep_a.yaml"), str(DATA / "sweep_b.yaml")])
    a, _, _ = sweep_panels(fig)
    shown = annotations(a)
    for x, y in [(1, 5.0), (2, 4.0), (5, 3.0), (6, 2.5), (8, 1.5)]:
        near = [t for (px, py), t in shown.items() if px == x and abs(py - y) < 1e-6]
        assert len(near) == 1, (x, y, shown)
        assert abs(float(near[0]) - y) < 1e-6


def test_sweep_gap_panel_b(monkeypatch, tmp_path):
    """Spec 2376: relative MIP gap in percent of the feasible cases, each value beside its point; a case
    without a feasible solution is marked at the top; a horizontal line at the stopping gap of the first
    report (0.05, so 5 on a percent axis). The gaps are those of the kept cases."""
    fig = draw_sweep(monkeypatch, tmp_path, [str(DATA / "sweep_a.yaml"), str(DATA / "sweep_b.yaml")])
    _, b, _ = sweep_panels(fig)
    expected = {1: 0.0, 2: 2.5, 4: 20.0, 5: 0.0, 6: 0.1, 7: 25.0, 8: 35.0}
    points = collection(b, "proven") + collection(b, "feasible")
    assert sorted(p[0] for p in points) == sorted(expected)
    for x, y in points:
        assert abs(y - expected[int(x)]) < 1e-6
    shown = annotations(b)
    for x, y in points:
        near = [t for (px, py), t in shown.items() if px == x and abs(py - y) < 1e-6]
        assert len(near) == 1
        assert abs(float(near[0].replace("%", "")) - y) < 1e-6
    none = collection(b, "none")
    assert sorted(p[0] for p in none) == [3.0, 9.0]
    assert all(0.75 <= f <= 1.0 for f in axes_fraction_y(b, "none"))
    hlines = [l for l in b.lines if len(l.get_ydata()) == 2 and l.get_ydata()[0] == l.get_ydata()[1]]
    assert len(hlines) == 1
    assert abs(float(hlines[0].get_ydata()[0]) - 5.0) < 1e-9


def test_sweep_gap_panel_requested_gap_marks(monkeypatch, tmp_path):
    """Spec 2376: when the stopping gap is not a number, a mark at the requested MIP gap of each case
    that has one (0.02 at H2=1, 0.04 at H2=2; none at H2=3), and no horizontal stopping line."""
    fig = draw_sweep(monkeypatch, tmp_path, str(DATA / "sweep_nogap.yaml"))
    _, b, _ = sweep_panels(fig)
    hlines = [l for l in b.lines if len(l.get_ydata()) == 2 and l.get_ydata()[0] == l.get_ydata()[1]]
    assert hlines == []
    points = [tuple(p) for c in b.collections for p in np.asarray(c.get_offsets(), float).tolist()]
    points += [(float(x), float(y)) for l in b.lines if len(l.get_xdata()) > 2
               for x, y in zip(l.get_xdata(), l.get_ydata())]
    at = lambda x, y: any(px == x and abs(py - y) < 1e-6 for px, py in points)
    assert at(1, 2.0) and at(2, 4.0)
    # H2=3 has the gap 16.67 only; there is no requested-gap mark for it
    assert sorted({round(py, 2) for px, py in points if px == 3 and np.isfinite(py)}) == [16.67]


def test_sweep_panel_c_counts(monkeypatch, tmp_path):
    """Spec 2377: numbers of continuous variables, integer variables and linear constraints over H2.
    The error row of H2=9 has no formulation."""
    fig = draw_sweep(monkeypatch, tmp_path, [str(DATA / "sweep_a.yaml"), str(DATA / "sweep_b.yaml")])
    _, _, c = sweep_panels(fig)
    series = {}
    for l in c.lines:
        label = l.get_label().lower()
        series[label] = (np.asarray(l.get_xdata(), float), np.asarray(l.get_ydata(), float))
    pick = lambda word: [v for k, v in series.items() if word in k]
    assert len(pick("continuous")) == len(pick("integer")) == len(pick("linear")) == 1
    # kept cases: H2=1 a, 2 b, 3 a, 4 b, 5 a, 6 a, 7 b, 8 b, 9 a (no formulation)
    cont = [10, 22, 30, 44, 50, 60, 77, 88]
    integer = [5, 12, 15, 24, 25, 30, 42, 48]
    rows = [20, 42, 60, 84, 100, 120, 147, 168]
    for word, expected in (("continuous", cont), ("integer", integer), ("linear", rows)):
        x, y = pick(word)[0]
        assert list(x[:8]) == list(range(1, 9))
        assert list(y[:8]) == expected
        assert len(x) == 8 or np.isnan(y[8])


def test_sweep_earlier_report_wins_a_tie(monkeypatch, tmp_path):
    """Spec 2368: a remaining tie keeps the case of the earlier report in the list. At H2=6 the two reports
    differ only in the MIP gap (0.1 % in sweep_a, 0.2 % in sweep_b)."""
    for order, expected in (((("sweep_a.yaml", "sweep_b.yaml")), 0.1), ((("sweep_b.yaml", "sweep_a.yaml")), 0.2)):
        fig = draw_sweep(monkeypatch, tmp_path, [str(DATA / n) for n in order])
        _, b, _ = sweep_panels(fig)
        gap = {p[0]: p[1] for p in collection(b, "proven")}
        assert abs(gap[6.0] - expected) < 1e-6


def test_sweep_title_with_sizes(monkeypatch, tmp_path):
    """Spec 2369-2370: F, M, L from fixed_dimensions; H1 from fixed_dimensions if present, else the key
    H1 (sweep_a has the key, sweep_b the entry); a size that the reports do not give is left out."""
    expected = "Operating-horizon sweep (F=3, M=1, L=2, H1=2)"
    for reports in ([DATA / "sweep_a.yaml"], [DATA / "sweep_b.yaml"], [DATA / "sweep_a.yaml", DATA / "sweep_b.yaml"]):
        fig = draw_sweep(monkeypatch, tmp_path, [str(p) for p in reports])
        assert suptitle_text(fig) == expected or any(t.get_text() == expected for t in fig.texts)
    fig = draw_sweep(monkeypatch, tmp_path, str(DATA / "sweep_nogap.yaml"))
    assert suptitle_text(fig) == "Operating-horizon sweep (L=2, H1=3)"


def test_sweep_single_path_and_list_agree(monkeypatch, tmp_path):
    """Spec 2355: one path, or a list of paths, of reports."""
    fig1 = draw_sweep(monkeypatch, tmp_path, str(DATA / "sweep_a.yaml"))
    fig2 = draw_sweep(monkeypatch, tmp_path, [str(DATA / "sweep_a.yaml")])
    a1, _, _ = sweep_panels(fig1)
    a2, _, _ = sweep_panels(fig2)
    assert_points(collection(a1, "proven"), collection(a2, "proven"))
    assert_points(collection(a1, "proven"), [(1, 5.0), (5, 3.0), (6, 2.5)])
    # sweep_a alone: H2=2 is time_limit (feasible, not proven); H2=4 and 9 are error rows
    assert_points(collection(a1, "feasible"), [(2, 4.0), (7, 3.0), (8, 1.5)])
    assert sorted(p[0] for p in collection(a1, "none")) == [3.0, 4.0, 9.0]


def test_sweep_json_report(monkeypatch, tmp_path):
    """Spec 2355: reports in YAML or JSON."""
    data = load("sweep_a.yaml")
    path = tmp_path / "report.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    fig = draw_sweep(monkeypatch, tmp_path, str(path))
    a, _, _ = sweep_panels(fig)
    assert len(collection(a, "proven")) == 3


def test_sweep_default_name_extensions_and_pdf(monkeypatch, tmp_path):
    """Spec 2356: as for plot_management; the default is 'horizon_sweep.png'; no extension gets .png;
    .pdf accepted."""
    monkeypatch.chdir(tmp_path)
    plot_horizon_sweep(str(DATA / "sweep_a.yaml"))
    assert (tmp_path / "horizon_sweep.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    plot_horizon_sweep(str(DATA / "sweep_a.yaml"), str(tmp_path / "pic"))
    assert (tmp_path / "pic.png").is_file()
    plot_horizon_sweep(str(DATA / "sweep_a.yaml"), str(tmp_path / "pic.pdf"))
    assert (tmp_path / "pic.pdf").read_bytes()[:4] == b"%PDF"


def test_sweep_file_errors(tmp_path):
    """Spec 2361-2363: FileNotFoundError for a missing report (and an image folder, 2356), TypeError for
    data that are not a mapping, ValueError for another file type and for an empty list of paths."""
    with pytest.raises(FileNotFoundError):
        plot_horizon_sweep(str(tmp_path / "absent.yaml"), str(tmp_path / "x.png"))
    with pytest.raises(FileNotFoundError):
        plot_horizon_sweep([str(DATA / "sweep_a.yaml"), str(tmp_path / "absent.yaml")], str(tmp_path / "x.png"))
    with pytest.raises(FileNotFoundError):
        plot_horizon_sweep(str(DATA / "sweep_a.yaml"), str(tmp_path / "no_folder" / "x.png"))
    bad = tmp_path / "list.yaml"
    bad.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(TypeError):
        plot_horizon_sweep(str(bad), str(tmp_path / "x.png"))
    txt = tmp_path / "report.txt"
    shutil.copy(DATA / "sweep_a.yaml", txt)
    with pytest.raises(ValueError):
        plot_horizon_sweep(str(txt), str(tmp_path / "x.png"))
    with pytest.raises(ValueError):
        plot_horizon_sweep([], str(tmp_path / "x.png"))


def _sweep_f(d): d["fixed_dimensions"]["F"] = 4
def _sweep_m(d): d["fixed_dimensions"]["M"] = 2
def _sweep_l(d): d["fixed_dimensions"]["L"] = 3
def _sweep_h1(d): d["fixed_dimensions"]["H1"] = 3


@pytest.mark.parametrize("mutate", [_sweep_f, _sweep_m, _sweep_l, _sweep_h1])
def test_sweep_reports_differing_in_size(tmp_path, mutate):
    """Spec 2355, 2363: ValueError when the reports differ in F, M, L or H1."""
    other = load("sweep_b.yaml")
    mutate(other)
    second = write_yaml(tmp_path, "other.yaml", other)
    with pytest.raises(ValueError):
        plot_horizon_sweep([str(DATA / "sweep_a.yaml"), str(second)], str(tmp_path / "x.png"))


def test_sweep_cases_errors(tmp_path):
    """Spec 2355, 2363: ValueError when a report has no list cases, when the reports hold no case, and
    when a case has no H2."""
    data = load("sweep_a.yaml")
    del data["cases"]
    with pytest.raises(ValueError):
        plot_horizon_sweep(str(write_yaml(tmp_path, "nocases.yaml", data)), str(tmp_path / "x.png"))
    data["cases"] = "none"
    with pytest.raises(ValueError):
        plot_horizon_sweep(str(write_yaml(tmp_path, "strcases.yaml", data)), str(tmp_path / "x.png"))
    data["cases"] = []
    with pytest.raises(ValueError):
        plot_horizon_sweep(str(write_yaml(tmp_path, "empty.yaml", data)), str(tmp_path / "x.png"))
    data = load("sweep_a.yaml")
    del data["cases"][2]["H2"]
    with pytest.raises(ValueError):
        plot_horizon_sweep(str(write_yaml(tmp_path, "noh2.yaml", data)), str(tmp_path / "x.png"))


def test_schedule_summary_panel_texts_do_not_overlap(monkeypatch, tmp_path):
    """Spec 2331-2342: every value of the summary panel is readable. A result with the five models and both
    repair laws gives a value that needs more than one line; no two texts of the panel may overlap."""
    result = load("schedule_marks.yaml")
    for key in ("model", "repair_model", "tau", "mu_0"):
        for row in result[key]:
            row.append(row[-1])
    result["model"][0][2] = "rainflow"
    result["model"][1][2] = "rainflow"
    result["repair_model"][0][2] = "ard1"
    result["component_names"].append("Seat")
    result["L"] = 3
    for key in ("m", "r", "mu"):
        for row in result[key]:
            row.append(list(row[-1]))
    fig = draw_schedule(monkeypatch, tmp_path, result)
    ax = summary_axes(fig)
    models = [v for k, v in summary(fig).items() if k == "Models"][0].get_text()
    assert all(name in models.replace("\n", " ") for name in
               ("gaussian", "gamma", "wiener", "inverse_gaussian", "rainflow"))
    fig.canvas.draw()
    boxes = [(t.get_text(), t.get_window_extent()) for t in ax.texts if t.get_text().strip()]
    assert len(boxes) > 10
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            assert not boxes[a][1].overlaps(boxes[b][1]), (boxes[a][0], boxes[b][0])
