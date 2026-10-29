"""Check of step B2: margin_table, overview, loop_table, day_marks, load_inspector."""
import sys
import types
from pathlib import Path

import numpy as np

SRC = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
package = types.ModuleType("fleet_management")
package.__path__ = [str(SRC / "fleet_management")]
sys.modules["fleet_management"] = package

from fleet_management.validation.dashboard_inspector import (  # noqa: E402
    day_marks, load_inspector, load_pair, margin_table, reliability_margins)

FIX = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\results\validator_tests")
state = load_inspector(FIX / "input.yaml", FIX / "result.yaml", 1e-6)
print(sorted(state))
table, ov, loops = state["table"], state["overview"], state["loops"]
print(table.shape, list(table.columns))
print(table.head(3).to_string())
assert len(table) == 30
assert ov["reliability_violations"] == 0
assert abs(ov["largest_mu_over_tau"] - 0.33) < 1e-12
first = ov["smallest_margins"].iloc[0]
print(first.to_dict())
assert (first["train"], first["component"], first["day"]) == (1, "Brake pad", 2)
assert abs(first["margin"] - 0.282346972) < 1e-9
print(ov["models"], ov["status"], ov["objective"])
assert ov["models"] == ["gamma", "gaussian", "inverse_gaussian", "rainflow (bernstein)",
                        "rainflow (chernoff)", "wiener"]
assert ov["status"] == "optimal"
print(len(loops), loops["quantity"].value_counts().to_dict(), loops["holds"].all())
assert len(loops) == 18
assert loops["quantity"].value_counts().to_dict() == {"mu": 6, "g_mu": 5, "v": 3, "g_v": 2,
                                                     "K": 1, "g_K": 1}
row = loops[(loops["train"] == 2) & (loops["component"] == "Wheelset") & (loops["quantity"] == "mu")]
print(row.to_string())
assert len(row) == 1 and abs(row.iloc[0]["margin"] - 0.19) < 1e-12
assert abs(row.iloc[0]["value_at_H1"] - 0.27) < 1e-12 and abs(row.iloc[0]["value_at_T"] - 0.08) < 1e-12

# marks
res = state["result"]
for cell, want in {(2, 0): "I D M2 M2 P", (2, 1): "I R M2 M2 P", (0, 0): "M1 M2 P M1 M2",
                   (1, 1): "M2 M1 M1 P M1"}.items():
    got = " ".join(day_marks(res, *cell))
    print(cell, got)
    assert got == want

# test 23: a violated row
cfg, result = load_pair(FIX / "input.yaml", FIX / "result.yaml")
result["mu"] = result["mu"].copy()
result["mu"][0, 0, 2] = 0.99
mg = reliability_margins(cfg, result)
print(mg[0, 0, 2])
assert abs(mg[0, 0, 2] - (-8.221738163816342e-06)) < 1e-6 * 8.22e-06
t = margin_table(cfg, result, mg, 1e-6)
bad = t[~t["holds"]]
print(bad.to_string())
assert len(bad) == 1 and bad.iloc[0]["train"] == 1 and bad.iloc[0]["component"] == "Wheelset" \
    and bad.iloc[0]["day"] == 3
print("B2 ok")
