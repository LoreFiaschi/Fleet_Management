import sys
sys.path.insert(0, r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
import numpy as np
import yaml
from fleet_management import horizon_sweep as h

T = 5
perf = {k: 10 * T for k in h.FORMULATION_KEYS}
perf.update({k: 0.1 for k in h.TIMING_KEYS})
perf.update({"requested_mip_gap": 1e-4, "requested_time_limit_seconds": 60.0,
             "optimizer_call_seconds": 0.5, "solutions_found": 1})
result = {
    "status": "optimal", "objective": np.float64(1.0), "bound": 0.9, "mip_gap": np.float64(0.1),
    "F": 3, "M": 1, "L": 1, "H1": 2, "H2": 3, "T": 5, "objective_mode": "operating_average",
    "evaluation_horizon": None, "x": np.zeros((3, 2, 5)), "m": np.zeros((3, 1, 5)), "r": np.zeros((3, 1, 5)),
    "warm_start": {"applied": False}, "performance": perf,
    "J_initialization": 2.0, "J_op": np.float64(3.0), "J_op_average": 1.0, "peak_damage": float("nan"),
}
row = h._case_row(result)
assert list(row) == list(h.ROW_KEYS), list(row)
assert row["peak_damage"] is None and row["absolute_mip_gap"] == 0.09999999999999998 or abs(row["absolute_mip_gap"] - 0.1) < 1e-12
assert type(row["objective"]) is float and type(row["J_op"]) is float
yaml.safe_dump(row)
err = h._error_row(2, 5, {"F": 3, "M": 1, "L": 1}, RuntimeError("boom"))
assert err["status"] == "error: boom"
assert list(err) == list(h.ROW_KEYS)
assert err["dimensions"] == {"F": 3, "M": 1, "L": 1, "H1": 2, "H2": 5, "T": 7}
assert all(v is None for k, v in err.items() if k not in ("H1", "H2", "T", "dimensions", "status"))
src = h._source_of(result)
assert list(src) == ["H1", "H2", "x", "m", "r"]
assert h._source_of({"x": None}) is None
# _solve_case with a stub
h.solve_mapping = lambda case, path, *, warm_start, warm_start_mode: dict(result)
r, s = h._solve_case({}, None, "start", True)
assert s is not None and r == row
r, s = h._solve_case({}, None, "start", False)
assert s is None
# result without a solution
bad = {"status": "infeasible", "objective": None, "bound": None, "mip_gap": None, "F": 3, "M": 1, "L": 1,
       "H1": 2, "H2": 3, "T": 5, "performance": perf}
rb = h._case_row(bad)
assert rb["absolute_mip_gap"] is None and rb["status"] == "infeasible"
print("A2 check OK")
