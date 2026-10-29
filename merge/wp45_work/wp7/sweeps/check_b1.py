"""Check of step B1: the construction pass of case A (and the other cases against the reference)."""
import sys

SP = r"C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad\wp7\sweeps"
sys.path.insert(0, r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
sys.path.insert(0, SP)

import numpy as np  # noqa: E402
from fleet_management import greedy_warm_start as g  # noqa: E402
from fleet_management.config import load_config  # noqa: E402
import greedy_ref  # noqa: E402

assert "gurobipy" not in sys.modules


def activities(x):
    return [[int(np.argmax(x[i, :, k])) if x[i, :, k].sum() > 0.5 else -1 for k in range(x.shape[2])]
            for i in range(x.shape[0])]


cfg = load_config(dict(greedy_ref.CASES["A_gaussian"], verbose=0))
x, m, r, order, ratio = g._construction_pass(cfg, g._profiles(cfg)["mu"], 0.70)
print("A activities", activities(x), "ratio", ratio)
assert activities(x) == [[1, -1, 1, -1], [-1, 1, -1, 1]], activities(x)
assert order == [[1]] * 4

# every case: the construction pass equals the first loop of the reference (no loop pass)
for name, data in greedy_ref.CASES.items():
    cfg = load_config(dict(data, verbose=0))
    x, m, r, order, ratio = g._construction_pass(cfg, g._profiles(cfg)["mu"], 0.70)
    print(name, activities(x), order[0], round(ratio, 6))
print("B1 ok")
