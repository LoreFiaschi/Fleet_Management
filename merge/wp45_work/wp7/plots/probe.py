"""Probe: result files written by the real writers, read back by config.read_input;
matplotlib facts the planned tests rely on. Reads only; writes into the scratchpad."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
from fleet_management import config, solver  # noqa: E402

OUT = Path(__file__).resolve().parent / "probe_out"
OUT.mkdir(exist_ok=True)

F, M, L, H1, H2 = 2, 1, 2, 2, 2
T = H1 + H2
x = np.zeros((F, M + 1, T))
x[0, 1, [0, 2]] = 1
x[0, 0, 3] = 1
x[1, 0, [0, 2]] = 1
x[1, 1, [1, 3]] = 1
m = np.zeros((F, L, T)); m[0, 0, 3] = 1; m[1, 0, 0] = 1
r = np.zeros((F, L, T)); r[0, 1, 3] = 1; r[1, 1, 2] = 1
nan = float("nan")
result = {
    "status": "time_limit", "objective": 7.5, "mip_gap": 0.031, "bound": 7.3,
    "F": F, "M": M, "L": L, "H1": H1, "H2": H2, "T": T, "H": [H1, H2],
    "model": [["gaussian", "rainflow"], ["gaussian", "rainflow"]],
    "component_names": ["Wheelset", "Brake pad"],
    "tau": np.full((F, L), 1.0), "mu_0": np.full((F, L), 0.1),
    "v_0": np.array([[0.001, nan], [0.001, nan]]),
    "bound_method": [[None, "cantelli"], [None, "cantelli"]],
    "repair_model": [["ard1", "ardinf"], ["ard1", "ardinf"]],
    "reliability_impl": "exact", "encoding": "bigm",
    "objective_mode": "operating_average", "evaluation_horizon": None,
    "x": x, "m": m, "r": r, "idle": x[:, 0, :][:, None, :] - m - r,
    "mu": np.linspace(0.1, 1.2, F * L * T).reshape(F, L, T),
    "v": np.full((F, L, T), nan),
    "J_op_average": 1.25,
    "performance": {"variables": 120, "continuous_variables": 80,
                    "integer_variables": 40, "linear_constraints": 300,
                    "optimizer_call_seconds": 2.5, "branch_and_bound_nodes": 17.0},
}
for ext in (".yaml", ".json", ".h5"):
    p = OUT / f"r{ext}"
    solver._save_results(result, p)
    d = config.read_input(p)
    print(ext, type(d["H"]), d["H"], type(d["model"]), d["model"][0], type(d["performance"]),
          d["status"], d["repair_model"], "v" in d, np.asarray(d.get("v"), dtype=float)[0, 1, 0],
          type(d["F"]), np.asarray(d["tau"]).shape, d["objective"])

print("None->nan:", np.asarray([[None, 1.0]], dtype=float))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_yticks(range(2))
ax.set_yticklabels(["a\nb", "c"])
print("ticklabels before draw:", [t.get_text() for t in ax.get_yticklabels()])
ax.set_xticks(range(3))
ax.set_xticklabels(["initial", *range(1, 3)])
print("xticklabels:", [t.get_text() for t in ax.get_xticklabels()])
line = ax.axvline(2.5, linestyle="--")
print("axvline:", line.get_xdata(), line.get_linestyle())
h = ax.axhline(5.0, linestyle="--", label="5% stopping gap")
print("axhline:", h.get_ydata(), h.get_label())
fig.suptitle("Operating-horizon sweep (F=4, M=1, L=1, H1=4)")
print("suptitle:", fig.get_suptitle())
sm = plt.cm.ScalarMappable(cmap="viridis"); sm.set_array([])
cb = fig.colorbar(sm, ax=ax, label="tracked mean / failure threshold")
print("cbar ylabel:", [a.get_ylabel() for a in fig.axes])
sc = ax.scatter([4, 8], [1.0, 1.0], marker="_", label="requested MIP gap")
print("scatter offsets:", sc.get_offsets().tolist(), sc.get_label())
ax.set_title("Run summary", loc="left")
print("title left:", ax.get_title(loc="left"))
leg = ax.legend()
print("legend:", [t.get_text() for t in leg.get_texts()])
plt.close(fig)
print("matplotlib", matplotlib.__version__)
