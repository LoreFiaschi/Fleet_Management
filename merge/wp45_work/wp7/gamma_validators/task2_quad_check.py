"""Run examples/regression/check_gamma_convolution_quadrature.py against the worktree src.

The package is loaded without its __init__.py (other WP7 parts may be unfinished) and without
gurobipy. Usage: python task2_quad_check.py
"""
import runpy
import sys
import types
from pathlib import Path

ROOT = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge")
package = types.ModuleType("fleet_management")
package.__path__ = [str(ROOT / "src" / "fleet_management")]
sys.modules["fleet_management"] = package
sys.modules["gurobipy"] = None
script = ROOT / "examples" / "regression" / "check_gamma_convolution_quadrature.py"
sys.path.insert(0, str(script.parent))
runpy.run_path(str(script), run_name="__main__")
