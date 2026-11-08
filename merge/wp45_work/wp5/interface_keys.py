"""Interface check 2 of the WP5 review: the keys of solve_mixed against solver.RESULT_KEYS.

Usage: python interface_keys.py
Runs base.solve_mixed through the emulator (fakegp) on the hand case, once with a
solution (complete MIP start) and once without one (an infeasible start), and
compares the key set with RESULT_KEYS of the WP4 writer. It also writes both
results through the WP4 YAML and JSON writers into this folder, to check that
every value is plain data. Prints ``failures: 0`` when all pass.
"""
import json
import pathlib
import sys

import numpy as np
import yaml

import hand_rows as h
from hand_rows import base, make_cfg
from hand_rows_solve import schedule

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
import fleet_management.solver as solver  # noqa: E402  (config and writers; no Gurobi)

failures = 0


def check(label, ok):
    global failures
    print(("PASS " if ok else "FAIL ") + label)
    failures += 0 if ok else 1


def solve(cfg, **kwargs):
    base.CELL_BUILDERS["gaussian"] = h.MeanVarianceBuilder()
    try:
        return base.solve_mixed(cfg, **kwargs)
    finally:
        del base.CELL_BUILDERS["gaussian"]


expected = set(solver.RESULT_KEYS)
check("RESULT_KEYS has 51 distinct keys", len(solver.RESULT_KEYS) == 51 == len(expected))

out = solve(make_cfg(C_D=0.5), warm_start=schedule())
check("with a solution: keys equal RESULT_KEYS", set(out) == expected)
check("with a solution: the spec's NaN rules (R, K, g_R, g_K all NaN)",
      all(np.isnan(out[key]).all() for key in ("R", "K", "g_R", "g_K")))
check("with a solution: bound_method is None (no rainflow cell)", out["bound_method"] is None)
check("with a solution: no evaluation horizon gives None", out["projected_evaluation_cost"] is None
      and out["evaluation_objective"] is None and out["evaluation_horizon"] is None)

# No solution: the start puts both trains on the mission on day 1 (coverage row broken).
bad = schedule()
bad["x"] = bad["x"].copy()
bad["x"][1, 1, 0] = 1.0
none_out = solve(make_cfg(C_D=0.5), warm_start=bad)
check("without a solution: status is not optimal", none_out["status"] != "optimal")
check("without a solution: keys equal RESULT_KEYS", set(none_out) == expected)
solution_and_costs = ("objective", "mip_gap", "x", "m", "r", "idle", "mu", "v", "z", "R", "K",
                      "g_mu", "g_v", "g_R", "g_K", "u", "step_costs", "J_initialization", "J_op",
                      "J_op_average", "peak_damage", "damage_penalty", "operating_objective",
                      "J_total", "projected_evaluation_cost", "evaluation_objective",
                      "component_costs")
check("without a solution: solution and cost keys are None",
      all(none_out[key] is None for key in solution_and_costs))

for label, result in (("solution", out), ("no_solution", none_out)):
    for suffix, loader in ((".yaml", yaml.safe_load), (".json", json.load)):
        path = HERE / f"interface_{label}{suffix}"
        solver._save_results(result, path)
        with open(path, encoding="utf-8") as f:
            data = loader(f)
        written = set(data)
        dropped = {key for key in expected if result.get(key) is None and key != "objective"}
        check(f"{label}{suffix}: written keys = RESULT_KEYS minus the None keys",
              written == expected - dropped)
    import h5py
    path = HERE / f"interface_{label}.h5"
    solver._save_results(result, path)
    with h5py.File(path, "r") as f:
        written = set(f.keys()) | set(f.attrs.keys())
    dropped = {key for key in expected if result.get(key) is None}
    check(f"{label}.h5: stored keys = RESULT_KEYS minus the None keys", written == expected - dropped)

print(f"failures: {failures}")
