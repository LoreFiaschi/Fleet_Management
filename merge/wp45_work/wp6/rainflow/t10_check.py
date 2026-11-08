"""Local check of the logic of T10 through the emulator.

The quadratic row cannot be built without Gurobi, so the reliability helper is
replaced by one that adds only the side rows ``mu <= mu_side``. The check shows
that with b = 0.3 the side row alone makes the case infeasible, with b = 0 it
leaves it feasible, and that the RHS of ``rel_side_0_0_0`` is tau - c.
"""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

import run_local  # noqa: F401  (installs the emulator and the stubs)

path = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\tests\test_rainflow.py")
spec = importlib.util.spec_from_file_location("test_rainflow", path)
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)

base = t.base


def side_only(ctx, i, l, q, *, c2, c1, mu_side):
    for k in range(ctx.T):
        ctx.model.addConstr(ctx.mu_var[i, l, k] <= mu_side, name=f"rel_side_{i}_{l}_{k}")


base.add_quadratic_reliability_rows = side_only
fails = 0
for b, feasible in ((0.3, False), (0.0, True)):
    zeros = np.zeros((2, 1, 1, 2))
    mu_trans = zeros.copy()
    mu_trans[0, 0, 0] = [0.8, 0.0]
    cfg = t._cfg("bernstein", "ardinf", reliability_impl="exact", mu=zeros, v=zeros,
                 support=zeros, mu_trans=mu_trans, v_0=np.zeros((2, 1)),
                 replacement_v=np.zeros((2, 1)), support_const=np.full((2, 1), b))
    ctx = t._build(cfg)
    for variables, name in ((ctx.x, "x"), (ctx.m_rep, "m"), (ctx.r_rep, "r")):
        for key, variable in variables.items():
            variable.LB = variable.UB = float(t._SCHEDULE[name][key])
    t._min_states(ctx)
    ctx.model.optimize()
    status = base.status_string(ctx.model.Status)
    rhs = ctx.model.getConstrByName("rel_side_0_0_0").RHS
    want = 1.0 - math.log(2.5) * b / 3.0
    ok = (status == "optimal") == feasible and abs(rhs - want) < 1e-12
    fails += not ok
    print("ok  " if ok else "FAIL", f"b={b}", status, "RHS", rhs, "want", want)
    if status == "optimal":
        print("   mu train 0:", [ctx.mu_var[0, 0, k].X for k in range(4)])
print("failures:", fails)
sys.exit(1 if fails else 0)
