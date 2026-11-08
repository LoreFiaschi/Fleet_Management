"""Reference implementation of the checks of spec "Structural Validator", for the plan's
expected values. It reads the fixture written by fixture_values.py, applies each corruption of
the plan, and prints the failing checks with their violations.

Usage: python reference_checks.py
"""
import copy
import math
import sys
import types
from pathlib import Path

import numpy as np
from scipy.special import ndtri

SRC = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
HERE = Path(__file__).resolve().parent
package = types.ModuleType("fleet_management")
package.__path__ = [str(SRC / "fleet_management")]
sys.modules["fleet_management"] = package

from fleet_management.config import load_config, read_input  # noqa: E402
from fleet_management import solver  # noqa: E402

LOOP = ("mu", "v", "R", "K", "g_mu", "g_v", "g_R", "g_K")


def read_result(path):
    data = read_input(path)
    out = {key: data.get(key) for key in solver.RESULT_KEYS}
    for key in solver._HDF5_DATASET_KEYS:
        if out[key] is not None:
            out[key] = np.asarray(out[key], dtype=float)
    return out


def checks(cfg, res, tol=1e-6):
    x, m, r = res["x"], res["m"], res["r"]
    out = {}
    out["binary"] = max(float(np.max(np.minimum(np.abs(a), np.abs(a - 1)))) for a in (x, m, r))
    out["one_activity"] = max(0.0, float(np.max(x.sum(axis=1) - 1)))
    out["mission_coverage"] = float(np.max(np.abs(x[:, 1:, :].sum(axis=0) - 1)))
    g = max(0.0, float(np.max(m + r - x[:, 0, :][:, None, :])))
    if not cfg.allow_replacement:
        g = max(g, float(np.max(r)))
    out["depot_gating"] = g
    if res.get("idle") is not None:
        out["idle"] = float(np.max(np.abs(res["idle"] - (x[:, 0, :][:, None, :] - m - r))))
    for q in LOOP:
        a = res.get(q)
        if a is None:
            out["loop_" + q] = 0.0
            continue
        d = a[:, :, cfg.T - 1] - a[:, :, cfg.H1 - 1]
        d = d[np.isfinite(d)]
        out["loop_" + q] = max(0.0, float(d.max())) if d.size else 0.0
    z = np.nan_to_num(res["z"], nan=0.0)
    day = cfg.C_M * x[:, 0, :].sum(axis=0) + np.einsum("il,ilk->k", cfg.C_R, z)
    if cfg.allow_replacement:
        day = day + np.einsum("il,ilk->k", cfg.C_rep, r)
    Ji, Jo = day[:cfg.H1].sum(), day[cfg.H1:].sum()
    pen = cfg.C_D * float(res["u"]) if cfg.C_D > 0 else 0.0
    mode = cfg.options["objective_mode"]
    if mode == "total":
        rec = Ji + Jo + pen
    elif mode == "operating_average":
        rec = Jo / cfg.H2 + pen
    else:
        rec = Ji + (cfg.options["evaluation_horizon"] - cfg.H1) / cfg.H2 * Jo + pen
    o = float(res["objective"])
    out["objective"] = max(0.0, rec - o) / max(1.0, abs(o))
    failed = {k: v for k, v in out.items() if v > tol}
    return out, failed, rec


cfg = load_config(read_input(HERE / "input.yaml"))
base = read_result(HERE / "result.yaml")
T, H1 = cfg.T, cfg.H1

all_checks, failed, rec = checks(cfg, base)
print("valid: names", list(all_checks), "failed", failed, "recomputed", repr(rec))


def corrupt(fn):
    res = copy.deepcopy(base)
    fn(res)
    return res


def loop_case(q, i, l):
    def fn(res):
        a = res[q]
        if not np.isfinite(a[i, l, H1 - 1]):
            a[i, l, H1 - 1] = 0.1
        a[i, l, T - 1] = a[i, l, H1 - 1] + 0.5
    return fn


cases = {
    "binary": lambda res: (res["x"].__setitem__((2, 0, 0), 0.3),
                           res["idle"].__setitem__((2, slice(None), 0), 0.3),
                           res.__setitem__("objective", res["objective"] + 0.3)),
    "one_activity": lambda res: (res["x"].__setitem__((0, 1, 2), 1.0),
                                 res["x"].__setitem__((1, 1, 2), 0.0)),
    "mission_coverage": lambda res: res["x"].__setitem__((1, 1, 2), 0.0),
    "depot_gating": lambda res: (res["m"].__setitem__((0, 0, 0), 1.0),
                                 res["idle"].__setitem__((0, 0, 0), -1.0)),
    "idle": lambda res: res["idle"].__setitem__((2, 0, 1), 0.0),
    "objective": lambda res: res.__setitem__("objective", res["objective"] - 2.0),
    "objective_above": lambda res: res.__setitem__("objective", res["objective"] + 5.0),
}
for q, cell in {"mu": (0, 0), "v": (1, 0), "R": (0, 0), "K": (2, 1), "g_mu": (1, 1),
                "g_v": (2, 0), "g_R": (0, 0), "g_K": (2, 1)}.items():
    cases["loop_" + q] = loop_case(q, *cell)

for name, fn in cases.items():
    _, failed, rec = checks(cfg, corrupt(fn))
    print(f"{name:18s} failed {failed}")

# allow_replacement false
data = read_input(HERE / "input.yaml")
data["allow_replacement"] = False
cfg_nr = load_config(data)
_, failed, rec = checks(cfg_nr, base)
print("allow_replacement false: failed", failed, "recomputed", repr(rec))

# objective modes and C_D = 0
for mode, extra in (("total", {}), ("operating_average", {}), ("evaluation_total", {}),
                    ("total", {"C_D": 0.0})):
    data = read_input(HERE / "input.yaml")
    data["objective_mode"] = mode
    data.update(extra)
    _, failed, rec = checks(load_config(data), base)
    print("mode", mode, extra, "recomputed", repr(rec), "failed", failed)

# margin of the Gaussian cell when mu[0, 0, 2] = 0.99
mu, v = 0.99, base["v"][0, 0, 2]
kg = ndtri(1 - 0.05)
print("violated G margin", repr(min((1 - mu) ** 2 - kg ** 2 * v, 1 - mu)), "v", v, "kg^2", kg ** 2)
print("objective saved", repr(base["objective"]), "obj violation case", repr(2.0 / abs(base["objective"] - 2.0)))
