"""Check of step B3: build_greedy_warm_start against the reference (rule of decision 2) and the table of 3.2.

Run with gurobipy blocked: python check_b3.py
"""
import sys

sys.modules["gurobipy"] = None          # the module must run without Gurobi
SP = r"C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad\wp7\sweeps"
sys.path.insert(0, r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
sys.path.insert(0, SP)

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from fleet_management import greedy_warm_start as g  # noqa: E402
from fleet_management.config import load_config  # noqa: E402
import greedy_ref  # noqa: E402


def activities(x):
    return [[int(np.argmax(x[i, :, k])) if x[i, :, k].sum() > 0.5 else -1 for k in range(x.shape[2])]
            for i in range(x.shape[0])]


# 1. every case of the reference: identical schedule and report
for name, data in greedy_ref.CASES.items():
    cfg = load_config(dict(data, verbose=0))
    source, report = g.build_greedy_warm_start(cfg)
    xr, mr, rr, rep = greedy_ref.greedy(cfg, joint_models=("wiener",))
    np.testing.assert_array_equal(source["x"], xr)
    np.testing.assert_array_equal(source["m"], mr)
    np.testing.assert_array_equal(source["r"], rr)
    d = report.as_dict()
    assert list(d) == ["repairs", "replacements", "repeatability_feasible", "repeatability_failures",
                       "maximum_repeatability_excess", "maximum_proxy_ratio", "assignment_order"]
    for key in d:
        a, b = d[key], rep[key]
        if isinstance(b, float) or hasattr(b, "dtype"):
            assert abs(a - float(b)) <= 1e-12, (name, key, a, b)
        else:
            assert a == b, (name, key, a, b)
    assert type(d["maximum_repeatability_excess"]) is float and type(d["maximum_proxy_ratio"]) is float
    assert type(d["repairs"]) is int and type(d["repeatability_feasible"]) is bool
    assert all(type(j) is int for day in d["assignment_order"] for j in day)
    yaml.safe_dump(d)
    print(f"{name:18s} ok  repairs={d['repairs']} replacements={d['replacements']} "
          f"failures={d['repeatability_failures']} excess={d['maximum_repeatability_excess']:.6g} "
          f"ratio={d['maximum_proxy_ratio']:.6g}")

# 2. the table of section 3.2 (day numbers 1-based)
TABLE = {
    "A_gaussian": ([[1, -1, 1, 0], [-1, 1, 0, 1]], (2, 0, 1, 0.0625, 0.625)),
    "F_gaussian": ([[1, -1, 0, 1], [-1, 1, -1, -1], [-1, -1, 1, 0]], (2, 0, 0, 0.0, 0.75)),
    "G_linear": ([[1, -1, 0, 1], [-1, 1, -1, -1], [-1, -1, 1, 0]], (8, 0, 0, 0.0, 0.3125)),
    "B_wiener": ([[1, -1, 1, 0], [-1, 1, 0, 1]], (0, 2, 0, 0.0, 0.625)),
    "H_wiener_gaussian": ([[1, -1, 1, 0], [-1, 1, 0, 1]], (2, 2, 1, 0.0625, 0.625)),
    "C_wiener_norepl": ([[1, -1, 1, 0], [-1, 1, 0, 1]], (2, 0, 2, 0.0625, 0.625)),
    "D_mixed_ard1": ([[2, -1, 2, 0, 2], [1, 1, 0, 2, 1], [-1, 2, 1, 1, 0]], (7, 2, 8, 0.78125, 0.8125)),
}
for name, (acts, (rp, rl, fl, ex, ra)) in TABLE.items():
    cfg = load_config(dict(greedy_ref.CASES[name], verbose=0))
    source, report = g.build_greedy_warm_start(cfg)
    assert activities(source["x"]) == acts, (name, activities(source["x"]))
    d = report.as_dict()
    assert (d["repairs"], d["replacements"], d["repeatability_failures"]) == (rp, rl, fl), (name, d)
    assert abs(d["maximum_repeatability_excess"] - ex) < 1e-12 and abs(d["maximum_proxy_ratio"] - ra) < 1e-12, (name, d)
    assert d["assignment_order"] == ([[2, 1]] * 5 if name == "D_mixed_ard1" else [[1]] * cfg.T)
print("table of 3.2 ok")

# 3. source keys, shapes, repair_trigger
cfg = load_config(dict(greedy_ref.CASES["F_gaussian"], verbose=0))
source, report = g.build_greedy_warm_start(cfg)
assert list(source) == ["H1", "H2", "x", "m", "r", "complete_binary_start", "completion_time_limit"]
assert source["completion_time_limit"] == 60.0 and source["complete_binary_start"] is True
assert source["x"].shape == (3, 2, 4) and source["m"].shape == (3, 1, 4) and source["r"].shape == (3, 1, 4)
for bad in (-0.1, 1.1, float("nan")):
    try:
        g.build_greedy_warm_start(cfg, repair_trigger=bad)
    except ValueError as error:
        assert "must lie in" in str(error)
    else:
        raise AssertionError("no error for " + repr(bad))
assert "gurobipy" not in [k for k, v in sys.modules.items() if v is not None]
print("B3 ok")
