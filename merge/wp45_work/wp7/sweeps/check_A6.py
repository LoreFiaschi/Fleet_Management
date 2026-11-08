import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
import numpy as np
import yaml
from fleet_management import horizon_sweep as h

TABLE = {2: ("optimal", 1.0, 0.0), 3: ("optimal", 0.9, 0.0), 4: ("time_limit", 0.8, 0.05), 5: "error",
         6: ("infeasible", None, None), 7: ("optimal", 0.9, 0.0)}
calls = []


def stub(data, results_path=None, *, warm_start=None, warm_start_mode="start"):
    assert results_path is None
    H1, H2 = data["H"]
    calls.append((H1, H2, None if warm_start is None else warm_start["H2"], warm_start_mode))
    entry = TABLE[H2]
    if entry == "error":
        raise RuntimeError(f"boom {H2}")
    status, J, gap = entry
    T = H1 + H2
    perf = {k: 10 * T for k in h.FORMULATION_KEYS}
    perf.update({k: 0.1 for k in h.TIMING_KEYS})
    perf.update({"requested_mip_gap": 0.1, "requested_time_limit_seconds": 60.0, "optimizer_call_seconds": 0.5,
                 "solutions_found": 1})
    res = {"status": status, "objective": J, "bound": None if J is None else J * (1 - gap), "mip_gap": gap,
           "F": 3, "M": 1, "L": 1, "H1": H1, "H2": H2, "T": T, "objective_mode": data["objective_mode"],
           "evaluation_horizon": data.get("evaluation_horizon"), "warm_start": {"applied": warm_start is not None},
           "performance": perf, "J_op_average": J}
    if J is not None:
        res.update({"x": np.zeros((3, 2, T)), "m": np.zeros((3, 1, T)), "r": np.zeros((3, 1, T))})
    return res


h.solve_mapping = stub
data = {"F": 3, "M": 1, "H": [2, 9], "model": "gaussian"}
sizes = h._input_sizes(data)
base = dict(output_path=None, stop_on_gradient=False, gradient_tolerance=1e-3, flat_gradients_required=2,
            minimum_cases=3, maximum_mip_gap_for_stopping=None, warm_start=True, warm_start_mode="both", n_workers=1,
            objective_mode="operating_average", evaluation_horizon=None)
settings = h._check_sweep_arguments(2, [7, 2, 3, 4, 5, 6], **base)
logging.basicConfig(level=logging.CRITICAL)
report = h._sweep(data, Path("in.yaml"), sizes, 2, settings)
assert [c[2] for c in calls] == [None, 2, 3, 4, 4, 4], calls
assert all(c[3] == "both" for c in calls)
assert report["evaluated_operating_horizons"] == [2, 3, 4, 5, 6, 7]
assert report["cases"][3]["status"] == "error: boom 5"
assert report["best_proven_H2"] == 3 and report["best_feasible_H2"] == 4
assert report["formulation_growth"]["added_variables"] == 50
assert [c["gradient_classification"] for c in report["cases"]] == ["unavailable", "decrease", "decrease"] + ["unavailable"] * 3
assert report["stopping_rule"]["reason"] == "candidate_range_exhausted" and report["last_completed_H2"] == 7

# parallel with threads
calls.clear()
h.ProcessPoolExecutor = ThreadPoolExecutor
settings = h._check_sweep_arguments(2, [2, 3, 4], **dict(base, warm_start=False, n_workers=3))
seen = []
report = h._sweep(data, Path("in.yaml"), sizes, 2, settings, on_checkpoint=lambda p: seen.append(p["complete"]))
assert seen == [False, False, False] and report["complete"] is True
assert [c["H2"] for c in report["cases"]] == [2, 3, 4] and report["n_workers"] == 3
assert all(c[2] is None for c in calls)
print("A6 check OK")
