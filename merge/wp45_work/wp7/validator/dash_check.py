import sys, types
from pathlib import Path
import numpy as np
SRC = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
p = types.ModuleType("fleet_management"); p.__path__ = [str(SRC / "fleet_management")]; sys.modules["fleet_management"] = p
from fleet_management.config import read_input
from fleet_management import solver
d = read_input(Path(__file__).parent / "result.yaml")
H1, T = 2, 5
count = {}
for q in ("mu","v","R","K","g_mu","g_v","g_R","g_K"):
    a = np.asarray(d[q], dtype=float)
    ok = np.isfinite(a[:, :, H1-1]) & np.isfinite(a[:, :, T-1])
    count[q] = int(ok.sum())
print(count, sum(count.values()))
x = np.asarray(d["x"]); m = np.asarray(d["m"]); r = np.asarray(d["r"])
def marks(i, l):
    out = []
    for k in range(T):
        if r[i,l,k] > .5: out.append("P")
        elif m[i,l,k] > .5: out.append("R")
        elif any(x[i,j,k] > .5 for j in range(1, x.shape[1])): out.append("M%d" % next(j for j in range(1, x.shape[1]) if x[i,j,k] > .5))
        elif x[i,0,k] > .5: out.append("D")
        else: out.append("I")
    return out
for c in ((2,0),(2,1),(0,0),(1,1)): print(c, marks(*c))
