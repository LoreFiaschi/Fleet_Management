"""Probe: the schedule plot accepts what solver.read_result gives (numpy arrays, NaN, numpy scalars).

Usage: python probe_nan.py   (loads the package from the worktree without __init__.py)
"""
import importlib.util
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

SRC = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
package = types.ModuleType("fleet_management")
package.__path__ = [str(SRC / "fleet_management")]
sys.modules["fleet_management"] = package
from fleet_management import solver  # noqa: E402
from fleet_management.utils import plotter  # noqa: E402

spec = importlib.util.spec_from_file_location("tests_module", r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\tests\test_plotter.py")
tests = importlib.util.module_from_spec(spec)
sys.modules["pytest"] = types.SimpleNamespace(mark=types.SimpleNamespace(parametrize=lambda *a: (lambda f: f)))
spec.loader.exec_module(tests)

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad\wp7\plots\tmp")
out.mkdir(exist_ok=True)

for extension in (".yaml", ".json", ".h5"):
    result = tests._result()
    result["mu"] = result["mu"].copy()
    result["mu"][0, 0, 1] = float("nan")      # a null in the tracked mean
    result["x"] = result["x"].copy()
    result["x"][1, 1, 0] = float("nan")       # a null in the schedule
    path = out / f"nan{extension}"
    tests._write(result, path)
    data = solver.read_result(path)
    assert isinstance(data["mu"], np.ndarray) and np.isnan(data["mu"][0, 0, 1])
    view = plotter._normalise_solution(data, {})
    figure = plotter._draw_solution(view)
    figure.canvas.draw()
    print(extension, "ok; mark at (row 2, day 1):", [t.get_text() for t in figure.axes[0].texts if t.get_position() == (1, 2)])
    plotter.plot_management(str(path), str(out / f"nan{extension[1:]}.png"))
print("probe done")
