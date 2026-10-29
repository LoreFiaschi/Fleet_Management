"""Check the reference greedy schedules against the real WP6 rows, with the linear emulator.

Each case of greedy_ref.py whose rows are linear is built by base.build_fleet; the binaries
are fixed to the greedy schedule and the remaining LP is solved. A schedule reported
feasible must give an LP with a solution (one direction only).
"""
import sys

SCRATCH = r"C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad"
sys.path.insert(0, SCRATCH + r"\wp6\giw\fakegp")
sys.path.insert(0, r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
sys.path.insert(0, SCRATCH + r"\wp7\sweeps")

import numpy as np  # noqa: E402
import gurobipy  # noqa: E402
from fleet_management.config import load_config  # noqa: E402
from fleet_management.degradation_model import base  # noqa: E402
import greedy_ref  # noqa: E402

assert "fakegp" in gurobipy.__file__


JOINT = "--joint" in sys.argv
WIENER = ("wiener",) if "--wiener" in sys.argv else ()   # decision 2 of wp7_interface.md


def check(name):
    data = dict(greedy_ref.CASES[name], verbose=0)
    cfg = load_config(data)
    x, m, r, rep = greedy_ref.greedy(cfg, joint=JOINT, joint_models=WIENER)
    ctx = base.build_fleet(cfg, base.resolve_run_options(cfg))
    for key, var in ctx.x.items():
        var.LB = var.UB = float(x[key])
    for key, var in ctx.m_rep.items():
        var.LB = var.UB = float(m[key])
    if ctx.r_rep is not None:
        for key, var in ctx.r_rep.items():
            var.LB = var.UB = float(r[key])
    ctx.model.optimize()
    status = base.status_string(ctx.model.Status)
    print(f"{name:18s} greedy feasible={rep['repeatability_feasible']!s:5s} "
          f"failures={rep['repeatability_failures']}  LP status={status}")
    # One direction only: the rows are one-sided, so the tracked states may exceed the
    # reference trajectory and a schedule with failures can still be feasible.
    return (not rep["repeatability_feasible"]) or status == "optimal"


ok = all(check(name) for name in greedy_ref.CASES if name != "E_rainflow")
print("agreement:", ok)
