"""T10, T12, T13 of the plan through solve_mixed (step A10), with the emulator.

Usage: python hand_rows_solve.py
The emulator cannot branch, so every solve here starts from a complete MIP start (the schedule
of the hand case), which the emulator takes as fixed binaries. A solve without a start (T10's
free-binary case, T11 through solve_mixed) needs real Gurobi.
"""
import numpy as np

import hand_rows as h
from hand_rows import OPTIONS, check, make_cfg, base
from hand_rows_extract import KEYS_WITHOUT_REPORTS

PERFORMANCE_KEYS = {
    "gurobi_version", "variables", "continuous_variables", "integer_variables", "binary_variables",
    "linear_constraints", "general_constraints", "indicator_constraints", "quadratic_constraints",
    "nonzeros", "solutions_found", "gurobi_runtime_seconds", "branch_and_bound_nodes",
    "simplex_iterations", "barrier_iterations", "work_units", "model_construction_seconds",
    "optimizer_call_seconds", "solution_extraction_seconds", "backend_wall_seconds",
    "requested_time_limit_seconds", "requested_mip_gap", "threads", "seed",
}


def schedule():
    x = np.zeros((2, 2, 4))
    for key in h.SCHEDULE_X:
        x[key] = 1.0
    m = np.zeros((2, 1, 4)); m[0, 0, 1] = 1.0
    r = np.zeros((2, 1, 4)); r[0, 0, 3] = 1.0
    return {"x": x, "m": m, "r": r, "H1": 2, "H2": 2}


def solve(cfg, **kwargs):
    base.CELL_BUILDERS["gaussian"] = h.MeanVarianceBuilder()
    try:
        return base.solve_mixed(cfg, **kwargs)
    finally:
        del base.CELL_BUILDERS["gaussian"]


def main():
    cfg = make_cfg(C_D=0.5)
    out = solve(cfg, warm_start=schedule())
    check("T10 status optimal", out["status"], "optimal")
    check("T10 keys: the 47 of the result and the 4 reports",
          set(out), KEYS_WITHOUT_REPORTS | {"performance", "optimization_progress", "warm_start", "relaxation_warm_start"})
    check("T10 performance keys (24)", set(out["performance"]), PERFORMANCE_KEYS)
    check("T10 requested_mip_gap 0.0 and time limit 60", (out["performance"]["requested_mip_gap"], out["performance"]["requested_time_limit_seconds"]), (0.0, 60.0))
    check("T10 threads and seed are int", (type(out["performance"]["threads"]), type(out["performance"]["seed"])), (int, int))
    check("T10 relaxation status disabled", (out["relaxation_warm_start"]["enabled"], out["relaxation_warm_start"]["status"]), (False, "disabled"))
    check("T10 model, R all NaN, v finite", (out["model"], bool(np.isnan(out["R"]).all()), bool(np.isfinite(out["v"]).all())),
          ([["gaussian"], ["gaussian"]], True, True))
    check("T10 progress has a final record", len(out["optimization_progress"]) >= 1, True)
    check("T10 sizes: indicator constraints 0 at bigm", out["performance"]["indicator_constraints"], 0)
    check("T13 warm_start applied, mode start", (out["warm_start"]["applied"], out["warm_start"]["mode"]), (True, "start"))
    check("T13 warm_start source_horizon and completion",
          (out["warm_start"]["source_horizon"], out["warm_start"]["completion"]["status"]),
          ({"H1": 2, "H2": 2}, "not_requested"))
    check("T10 objective is the report J_total", abs(out["objective"] - out["J_total"]) < 1e-6, True)

    cfg = make_cfg(C_D=0.5, options=dict(OPTIONS, relaxation_warm_start=True), encoding="indicator")
    out = solve(cfg, warm_start=schedule())
    check("T13 relaxation skipped when a start was applied",
          (out["relaxation_warm_start"]["enabled"], out["relaxation_warm_start"]["status"]), (True, "skipped"))
    check("T13 relaxation report keys", set(out["relaxation_warm_start"]),
          {"enabled", "status", "relaxation_objective", "values_copied", "runtime_seconds"})
    check("T13 indicator encoding: indicator constraints counted", out["performance"]["indicator_constraints"] > 0, True)

    for mode in ("bogus",):
        try:
            solve(make_cfg(), warm_start_mode=mode)
            check("T12 warm_start_mode bogus raises ValueError", 0, 1)
        except ValueError:
            check("T12 warm_start_mode bogus raises ValueError", 1, 1)
    for mode in ("hint", "both"):
        try:
            solve(make_cfg(), warm_start=schedule(), warm_start_mode=mode)
            check(f"A10 mode {mode} with a source raises NotImplementedError", 0, 1)
        except NotImplementedError:
            check(f"A10 mode {mode} with a source raises NotImplementedError", 1, 1)
    print("failures:", h.failures)


if __name__ == "__main__":
    main()
