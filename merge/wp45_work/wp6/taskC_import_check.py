"""Import check of Task C: config.py and its two cap checks work without gurobipy.

A sys.meta_path finder refuses every import of gurobipy. The real package (with its
__init__.py) is imported from the worktree, then load_config runs on the small inverse
Gaussian and Gamma inputs of the two new tests.
"""
import importlib.abc
import sys

SRC = r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src"


class RefuseGurobi(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "gurobipy" or name.startswith("gurobipy."):
            raise ModuleNotFoundError("gurobipy refused by the check", name="gurobipy")
        return None


sys.meta_path.insert(0, RefuseGurobi())
sys.path.insert(0, SRC)

try:
    import gurobipy      # noqa: F401
except ModuleNotFoundError:
    print("gurobipy refused: ok")
else:
    raise SystemExit("gurobipy imported: the check is void")

import fleet_management                     # noqa: E402
import fleet_management.config as config    # noqa: E402
print("import fleet_management, fleet_management.config:", config.__file__)

small = {"F": 2, "M": 1, "H": [1, 2], "tau": 1.0, "epsilon": 0.1, "rho": 0.5, "mu_0": 0.1,
         "mu": 0.05, "C_M": 1.0, "C_R": 1.0, "C_rep": 1.0, "C_D": 1.0}


def small_input(model, **changes):
    data = dict(small, model=model)
    data.update({"inverse_gaussian": {"eta": 0.15}, "gamma": {"gamma_beta": 20.0}}[model])
    data.update(changes)
    return data


def rejected(model, **changes):
    try:
        config.load_config(small_input(model, **changes))
    except ValueError as error:
        return str(error)
    raise SystemExit(f"{model} {changes}: not rejected")


config.load_config(small_input("inverse_gaussian", replacement_mu=0.54))
print("inverse_gaussian, replacement_mu 0.54: loads")
print("inverse_gaussian, replacement_mu 0.56:", rejected("inverse_gaussian", replacement_mu=0.56))
config.load_config(small_input("gamma", replacement_mu=0.7437))
print("gamma, replacement_mu 0.7437: loads")
print("gamma, replacement_mu 0.7453:", rejected("gamma", replacement_mu=0.7453))

assert "gurobipy" not in sys.modules, "gurobipy was imported"
loaded = sorted(name for name in sys.modules if name.startswith("fleet_management.degradation_model"))
print("modules loaded:", loaded)
from fleet_management.degradation_model import gamma, inverse_gaussian     # noqa: E402
print("gamma.base is None:", gamma.base is None, "; inverse_gaussian.base is None:",
      inverse_gaussian.base is None)
print("import check: ok")
