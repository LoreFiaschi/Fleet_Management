"""Check that inverse_gaussian.py imports with gurobipy unavailable and that ig_mean_cap works.

Usage: python ig_nogurobi_check.py
Runs in a fresh interpreter: a sys.meta_path finder refuses ``gurobipy`` (ModuleNotFoundError
with name "gurobipy"), then the package is loaded from the merge worktree without running its
__init__.py (as the plan's test runner does) and the module is imported.
"""
import importlib
import importlib.abc
import sys
import types
from pathlib import Path

SRC = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")


class RefuseGurobi(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "gurobipy" or fullname.startswith("gurobipy."):
            raise ModuleNotFoundError("No module named 'gurobipy'", name="gurobipy")
        return None


sys.meta_path.insert(0, RefuseGurobi())
package = types.ModuleType("fleet_management")
package.__path__ = [str(SRC / "fleet_management")]
sys.modules["fleet_management"] = package

module = importlib.import_module("fleet_management.degradation_model.inverse_gaussian")
print("imported; base is None:", module.base is None)
print("gurobipy in sys.modules:", "gurobipy" in sys.modules)
assert module.base is None

reference = ((0.5, 1.0, 0.1, 1.8859953361, 0.3823093163),
             (0.1, 1.0, 0.01, 1.9488253601, 0.4627586840),
             (2.0, 2.0, 0.05, 3.5525960458, 0.3469170694))
ok = True
for eta, tau, eps, q, mu_bar in reference:
    q_got = module._ig_unit_quantile(eta, eps)
    cap_got = module.ig_mean_cap(eta, tau, eps)
    good = abs(q_got - q) <= 1e-9 * q and abs(cap_got - mu_bar) <= 1e-9 * mu_bar
    ok = ok and good
    print(f"eta={eta} tau={tau} eps={eps}: q={q_got:.10f} (ref {q}), cap={cap_got:.10f} "
          f"(ref {mu_bar}) -> {'ok' if good else 'MISMATCH'}")
print("all three reference values:", "ok" if ok else "FAILED")
sys.exit(0 if ok else 1)
