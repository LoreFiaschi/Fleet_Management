"""Load base.py of the merge worktree without Gurobi, through the stub.

Usage: python load_base.py
Prints the public names of the module. Exits with an error if it cannot load.
"""
import importlib.util
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
BASE = pathlib.Path(
    r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge"
    r"\src\fleet_management\degradation_model\base.py"
)


def load():
    sys.path.insert(0, str(HERE / "stubs"))
    spec = importlib.util.spec_from_file_location("wp5_base", BASE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["wp5_base"] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    base = load()
    print(sorted(name for name in vars(base) if not name.startswith("_")))
