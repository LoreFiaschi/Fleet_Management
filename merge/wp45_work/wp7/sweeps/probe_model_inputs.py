"""Probe: the model inputs of section 3.2 build with the emulator, and solve_mixed runs case G
with a complete start (binaries from the reference greedy). Uses the current base.py."""
import sys

HERE = r"C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad\wp7\sweeps"
sys.path.insert(0, HERE + r"\fakegp")
sys.path.insert(0, r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
sys.path.insert(0, HERE)

from fleet_management.config import load_config  # noqa: E402
from fleet_management.degradation_model import base  # noqa: E402
import greedy_ref  # noqa: E402

data = {"F": 3, "M": 1, "H": [2, 3], "model": "inverse_gaussian", "encoding": "indicator", "tau": 1.0,
        "epsilon": 0.5, "rho": 1.0, "mu_0": 0.0625, "eta": 0.25, "mu": 0.125, "C_M": 1.0, "C_R": 1.0,
        "C_rep": 5.0, "C_D": 0.0, "verbose": 0}
cfg = load_config(data)
ctx = base.build_fleet(cfg, base.resolve_run_options(cfg))
print("IG indicator model:", len(ctx.x), "x,", len(ctx.m_rep), "m,", len(ctx.r_rep), "r,", len(ctx.nb), "n")

cfg = load_config(dict(greedy_ref.CASES["G_linear"], verbose=0))
x, m, r, rep = greedy_ref.greedy(cfg)
source = {"H1": cfg.H1, "H2": cfg.H2, "x": x, "m": m, "r": r, "complete_binary_start": True,
          "completion_time_limit": 60.0}
out = base.solve_mixed(cfg, warm_start=source)
ws = out["warm_start"]
print("G solve_mixed:", out["status"], "values", ws["values"], "completion", ws["completion"]["status"],
      ws["completion"]["fixed_integer_variables"], ws["completion"]["values_copied"],
      out["performance"]["variables"])
