"""Tests for fleet_management.plotter.

All tests use hand-built result dicts (via the ``make_fake_result`` fixture or
constructed directly) written to disk with ``solver._save_results``, then
rendered with ``plot_management``. No Gurobi model is ever built.
"""

import numpy as np
import pytest

from fleet_management import plotter, solver


def _write(tmp_path, result, shared, name="out.yaml"):
    path = tmp_path / name
    solver._save_results(result, path, shared)
    return path


@pytest.mark.parametrize("ext", [".yaml", ".json", ".h5"])
def test_plot_management_produces_file(tmp_path, make_fake_result, make_fake_shared, ext):
    result = make_fake_result(F=3, M=1, L=2, H=3)
    shared = make_fake_shared(result)
    path = _write(tmp_path, result, shared, name=f"out{ext}")
    out_png = tmp_path / "schedule.png"
    plotter.plot_management(str(path), str(out_png))
    assert out_png.exists() and out_png.stat().st_size > 0


def test_plot_management_default_output_name(tmp_path, make_fake_result, make_fake_shared, monkeypatch):
    result = make_fake_result(F=2, M=1, L=1, H=2)
    shared = make_fake_shared(result)
    path = _write(tmp_path, result, shared)
    monkeypatch.chdir(tmp_path)
    plotter.plot_management(str(path))
    assert (tmp_path / "output.png").exists()


def test_plot_management_pdf_extension(tmp_path, make_fake_result, make_fake_shared):
    result = make_fake_result(F=2, M=1, L=1, H=2)
    shared = make_fake_shared(result)
    path = _write(tmp_path, result, shared)
    out_pdf = tmp_path / "schedule.pdf"
    plotter.plot_management(str(path), str(out_pdf))
    assert out_pdf.exists() and out_pdf.stat().st_size > 0


def test_plot_management_rejects_unsupported_output_extension(tmp_path, make_fake_result, make_fake_shared):
    result = make_fake_result(F=2, M=1, L=1, H=2)
    shared = make_fake_shared(result)
    path = _write(tmp_path, result, shared)
    with pytest.raises(ValueError):
        plotter.plot_management(str(path), str(tmp_path / "schedule.svg"))


def test_plot_management_missing_input_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        plotter.plot_management(str(tmp_path / "does_not_exist.yaml"))


def test_plot_management_rejects_non_optimal(tmp_path):
    result = {
        "H": 2, "F": 1, "M": 1, "L": 1,
        "model": [["gaussian"]], "tail_bound": [[None]], "tau": [[1.0]],
        "status": "infeasible", "objective": None,
        "x": None, "x_m": None, "x_r": None, "mu": None, "v": None, "u": None, "z": None,
    }
    shared = {"H": 2, "epsilon": 0.01, "tau": np.ones((1, 1)), "rho": np.ones((1, 1)) * 0.5,
              "model_assignment": np.array(result["model"], dtype=object)}
    path = _write(tmp_path, result, shared)
    with pytest.raises(ValueError):
        plotter.plot_management(str(path))


def test_plot_management_multi_horizon_writes_one_file_per_H(tmp_path, make_fake_result, make_fake_shared):
    r1 = make_fake_result(F=2, M=1, L=1, H=2, seed=0)
    r2 = make_fake_result(F=2, M=1, L=1, H=3, seed=1)
    multi = {2: r1, 3: r2}
    shared = make_fake_shared(r1, H=[2, 3])
    path = _write(tmp_path, multi, shared, name="multi.yaml")
    prefix = tmp_path / "schedule"
    plotter.plot_management(str(path), str(prefix) + ".png")
    assert (tmp_path / "schedule_H2.png").exists()
    assert (tmp_path / "schedule_H3.png").exists()


# ======================================================================
# Cell-annotation coverage: mission number, gear, wrench, both, idle
# ======================================================================

def _annotated_result(F=1, M=1, L=2, H=2):
    """A hand-built (F, L)=(1,2) fixture exercising every annotation path over
    four consecutive days: mission -> repair-only -> replace-only -> repair+
    replace-on-different-components -> idle. H is bumped up to fit all of it.
    """
    H = 3  # 2H = 6 days
    two_h = 2 * H
    model = [["gaussian", "inverse_gaussian"]]
    tau = np.ones((F, L))
    mu = np.full((F, L, two_h), 0.2)
    x = np.zeros((F, M + 1, two_h))
    x_m = np.zeros((F, L, two_h))
    x_r = np.zeros((F, L, two_h))

    x[0, 1, 0] = 1  # day0: mission 1

    x[0, 0, 1] = 1  # day1: maintenance day, repair comp0 only
    x_m[0, 0, 1] = 1

    x[0, 0, 2] = 1  # day2: maintenance day, replace comp1 only
    x_r[0, 1, 2] = 1

    x[0, 0, 3] = 1  # day3: maintenance day, repair comp0 AND replace comp1
    x_m[0, 0, 3] = 1
    x_r[0, 1, 3] = 1
    # day4: all zero -> idle
    x[0, 0, 5] = 1  # day5: maintenance day, no action at all (blank cell)

    result = {
        "status": "optimal", "objective": 1.0,
        "H": H, "F": F, "M": M, "L": L,
        "model": model, "tail_bound": [[None] * L],
        "tau": tau.tolist(),
        "x": x, "x_m": x_m, "x_r": x_r,
        "mu": mu, "v": np.full((F, L, two_h), np.nan),
        "u": 0.5, "z": np.zeros((F, L, two_h)),
    }
    shared = {"H": H, "epsilon": 0.01, "tau": tau, "rho": np.full((F, L), 0.7),
              "model_assignment": np.array(model, dtype=object)}
    return result, shared


def test_plot_annotations_all_paths_render_without_error(tmp_path):
    result, shared = _annotated_result()
    path = _write(tmp_path, result, shared)
    out_png = tmp_path / "annotated.png"
    plotter.plot_management(str(path), str(out_png))
    assert out_png.exists() and out_png.stat().st_size > 0


def test_border_styles_cover_all_five_models():
    assert set(plotter._BORDER_STYLES) == {
        "gaussian", "inverse_gaussian", "wiener", "gamma", "rainflow"
    }
