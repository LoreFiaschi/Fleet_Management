"""Render the two plots of the prototype to PNG files in the scratchpad.

Usage: python render.py <plotter module path> <test module path> <output folder>
"""
import importlib.util
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import yaml

SRC = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
package = types.ModuleType("fleet_management")
package.__path__ = [str(SRC / "fleet_management")]
sys.modules["fleet_management"] = package
utils = types.ModuleType("fleet_management.utils")
utils.__path__ = [str(SRC / "fleet_management" / "utils")]
sys.modules["fleet_management.utils"] = utils


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


plotter = load("fleet_management.utils.plotter", sys.argv[1])
utils.plotter = plotter
sys.modules["pytest"] = types.SimpleNamespace(mark=types.SimpleNamespace(parametrize=lambda *a: (lambda f: f)))
tests = load("tests_module", sys.argv[2])
out = Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
result_path = out / "result.yaml"
tests._write(tests._result(), result_path)
plotter.plot_management(str(result_path), str(out / "schedule.png"))
report = out / "sweep.yaml"
report.write_text(yaml.safe_dump(tests._report(tests.SWEEP_CASES), sort_keys=False), encoding="utf-8")
plotter.plot_horizon_sweep(str(report), str(out / "sweep.png"))
report2 = out / "sweep_requested.yaml"
report2.write_text(yaml.safe_dump(tests._report(tests.SWEEP_CASES, maximum_gap=None), sort_keys=False),
                   encoding="utf-8")
plotter.plot_horizon_sweep(str(report2), str(out / "sweep_requested.png"))
print("written to", out)
