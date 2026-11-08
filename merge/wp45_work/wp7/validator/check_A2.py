"""Check of step A2: the module imports without gurobipy; read_result of the three files."""
import sys
import types
from pathlib import Path

import numpy as np

SRC = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
HERE = Path(__file__).resolve().parent
package = types.ModuleType("fleet_management")
package.__path__ = [str(SRC / "fleet_management")]
sys.modules["fleet_management"] = package

from fleet_management.validation import validator  # noqa: E402
from fleet_management import solver  # noqa: E402

assert "gurobipy" not in sys.modules
assert validator.read_result is solver.read_result

results = {s: solver.read_result(HERE / f"result.{s}") for s in ("yaml", "json", "h5")}
base = results["yaml"]
for s, res in results.items():
    assert list(res) == list(solver.RESULT_KEYS), s
    for key in solver._HDF5_DATASET_KEYS:
        if base[key] is None:
            assert res[key] is None, (s, key)
        else:
            assert res[key].dtype == np.float64, (s, key)
            np.testing.assert_array_equal(res[key], base[key], err_msg=f"{s} {key}")
    assert np.all(np.isnan(res["R"])), s
    assert list(res["H"]) == [2, 3] or res["H"] == [2, 3], (s, res["H"])
    print(s, "ok; H", res["H"], "R all NaN", "v finite at", int(np.isfinite(res["v"]).sum()))

try:
    solver.read_result(HERE / "result.txt")
except ValueError as exc:
    print("ValueError:", exc)
print("A2 check passed")
