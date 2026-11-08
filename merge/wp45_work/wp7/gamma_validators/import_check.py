"""Import the four gamma_utils modules of the worktree without gurobipy; print their public names.

Usage: python import_check.py
Prints one line per module and "ok" at the end. gurobipy is blocked, as on a machine
without Gurobi. The package is loaded without its __init__.py (other WP7 parts may be
unfinished), so this checks only the gamma_utils modules and their imports.
"""
import importlib
import sys
import types
from pathlib import Path

SRC = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
package = types.ModuleType("fleet_management")
package.__path__ = [str(SRC / "fleet_management")]
sys.modules["fleet_management"] = package
sys.modules["gurobipy"] = None

for name in ("gamma_tail_bound", "gamma_replay_validator", "gamma_tail_validator",
             "gamma_stochastic_validator"):
    module = importlib.import_module(f"fleet_management.degradation_model.gamma_utils.{name}")
    public = sorted(n for n in vars(module) if not n.startswith("_")
                    and callable(getattr(module, n))
                    and getattr(getattr(module, n), "__module__", "") == module.__name__)
    print(f"{name}: {', '.join(public)}")
assert "fleet_management.degradation_model.base" not in sys.modules, "base was imported"
print("ok")
