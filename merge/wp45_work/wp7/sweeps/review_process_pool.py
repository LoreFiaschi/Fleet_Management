"""Review: a real ProcessPoolExecutor sweep (n_workers = 2) on this computer, without Gurobi.

Each worker process runs the real solve_mapping, which fails at the import of gurobipy, so
every row must be an error row; the check is that the pool path works on Windows (spawn,
pickling of _solve_case and its arguments, capture of the error, order of the rows).
The stub package below runs at the top level, so the spawned workers also skip
fleet_management/__init__.py (another package that it imports is being rewritten).
"""
import logging
import sys
import types
from pathlib import Path

SRC = r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, SRC)
_package = types.ModuleType("fleet_management")
_package.__path__ = [SRC + r"\fleet_management"]
sys.modules["fleet_management"] = _package

if __name__ == "__main__":
    import yaml
    from fleet_management import horizon_sweep

    logging.basicConfig(level=logging.ERROR)
    folder = HERE / "tmp" / "process_pool"
    folder.mkdir(parents=True, exist_ok=True)
    data = {"F": 3, "M": 1, "H": [2, 9], "model": "gaussian", "tau": 1.0, "epsilon": 0.1,
            "rho": 0.5, "mu_0": 0.1, "mu": 0.2, "v": 0.001, "reliability_impl": "tangent",
            "C_M": 1.0, "C_R": 1.0, "C_rep": 5.0, "C_D": 0.0}
    path = folder / "input.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    report = horizon_sweep.sweep_operating_horizons(
        path, [4, 2, 3], n_workers=2, output_path=folder / "report.yaml")
    print("rows:", [(row["H2"], row["status"]) for row in report["cases"]])
    print("complete:", report["complete"], "n_workers:", report["n_workers"],
          "last_completed_H2:", report["last_completed_H2"])
    assert [row["H2"] for row in report["cases"]] == [2, 3, 4]
    assert all(row["status"].startswith("error: ") for row in report["cases"])
    print("process pool path: OK")
