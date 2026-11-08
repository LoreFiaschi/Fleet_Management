"""Review: every greedy start without failures must be feasible for the real rows (WP6 builders),
checked on random fleets with linear rows only (tangent form, big-M encoding), with the emulator.

Run: python review_lp_random.py [cases]
"""
import sys
import types

HERE = r"C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad\wp7\sweeps"
SRC = r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src"
sys.path.insert(0, HERE + r"\fakegp")
sys.path.insert(0, SRC)
sys.path.insert(0, HERE)
pkg = types.ModuleType("fleet_management")
pkg.__path__ = [SRC + r"\fleet_management"]
sys.modules["fleet_management"] = pkg

import numpy as np  # noqa: E402
from fleet_management.config import load_config  # noqa: E402
from fleet_management.degradation_model import base  # noqa: E402
from fleet_management.greedy_warm_start import build_greedy_warm_start  # noqa: E402
from review_random_greedy import random_input  # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    rng = np.random.default_rng(11)
    checked = feasible_starts = 0
    for _ in range(n):
        data = random_input(rng)
        data.update(reliability_impl="tangent", encoding="bigm")
        cfg = load_config(data)
        source, report = build_greedy_warm_start(cfg)
        if not report.repeatability_feasible:
            continue
        ctx = base.build_fleet(cfg, base.resolve_run_options(cfg))
        for name, variables in (("x", ctx.x), ("m", ctx.m_rep), ("r", ctx.r_rep or {})):
            for key, var in variables.items():
                var.LB = var.UB = float(source[name][key])
        ctx.model.optimize()
        status = base.status_string(ctx.model.Status)
        checked += 1
        if status != "optimal":
            print("INFEASIBLE start reported feasible:", data)
            return
        feasible_starts += 1
    print(f"{checked} starts without failures; all {feasible_starts} feasible for the real rows")


if __name__ == "__main__":
    main()
