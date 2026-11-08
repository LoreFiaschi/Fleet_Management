"""Hand-built valid result of the validator fixture, and the expected values of the plan.

Usage: python fixture_values.py
Writes result.yaml, result.json, result.h5 next to this script with the real writers
(solver._save_results). Prints the numbers that the plan quotes.
"""
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

from fleet_management.config import load_config, read_input, _lookup  # noqa: E402
from fleet_management import solver  # noqa: E402
from fleet_management.degradation_model.inverse_gaussian import ig_mean_cap  # noqa: E402
from fleet_management.degradation_model.gamma import mean_cap, smallest_rate  # noqa: E402

cfg = load_config(read_input(HERE / "input.yaml"))
F, M, L, H1, H2, T = cfg.F, cfg.M, cfg.L, cfg.H1, cfg.H2, cfg.T
print("sizes", F, M, L, H1, H2, T)

# ---- schedule (0-based day index k = day - 1) ----
x = np.zeros((F, M + 1, T))
m = np.zeros((F, L, T))
r = np.zeros((F, L, T))
plan = {  # day -> {train: activity}; activity: mission j >= 1, 0 depot, None idle
    1: {0: 1, 1: 2, 2: None},
    2: {0: 2, 1: 1, 2: 0},
    3: {0: 0, 1: 1, 2: 2},
    4: {0: 1, 1: 0, 2: 2},
    5: {0: 2, 1: 1, 2: 0},
}
for day, acts in plan.items():
    for i, j in acts.items():
        if j is not None:
            x[i, j, day - 1] = 1.0
m[2, 1, 1] = 1.0                       # day 2: repair of train 3, component 2
r[0, :, 2] = 1.0                       # day 3: train 1 replaced
r[1, :, 3] = 1.0                       # day 4: train 2 replaced
r[2, :, 4] = 1.0                       # day 5: train 3 replaced

# ---- trajectories by the day maps, rows at equality ----
mu_inc = _lookup(cfg.mu, cfg.mu_trans, H1, H2)                       # (F, L, M, T)
v_inc = _lookup(cfg.v, cfg.v_trans, H1, H2) if cfg.v is not None else None
K_inc = _lookup(cfg.cgf, cfg.cgf_trans, H1, H2) if cfg.cgf is not None else None
nan = np.full((F, L, T), np.nan)
arr = {q: nan.copy() for q in ("mu", "v", "R", "K", "g_mu", "g_v", "g_R", "g_K", "z")}


def tracked(i, l):
    model = cfg.model[i, l]
    if model in ("gaussian", "wiener"):
        return ["mu", "v"]
    if model == "rainflow":
        return ["mu"] + {"markov": [], "cantelli": ["v"], "hoeffding": ["R"],
                         "bernstein": ["v"], "chernoff": ["K"]}[cfg.bound_method[i, l]]
    return ["mu"]


for i in range(F):
    for l in range(L):
        ard1 = cfg.repair_model[i, l] == "ard1"
        rho = cfg.rho[i, l]
        wiener = cfg.model[i, l] == "wiener"
        for q in tracked(i, l):
            s = {"mu": cfg.mu_0, "v": cfg.v_0, "R": cfg.R_0, "K": cfg.cgf_0}[q][i, l]
            g = {"mu": cfg.anchor_mu_0, "v": cfg.anchor_v_0, "R": cfg.anchor_R_0,
                 "K": cfg.anchor_K_0}[q][i, l] if ard1 else 0.0
            new = {"mu": cfg.replacement_mu, "v": cfg.replacement_v, "R": cfg.R_new,
                   "K": cfg.cgf_new}[q][i, l]
            a = (1 - rho) if q in ("mu", "K") else (1 - rho) ** 2
            anchored = ard1 and not (wiener and q == "v")
            for k in range(T):
                if r[i, l, k]:
                    s_new, g_new = new, new
                elif m[i, l, k]:
                    if wiener and q == "v":
                        s_new = s
                    else:
                        s_new = a * s + (1 - a) * g if ard1 else a * s
                    g_new = s_new
                else:
                    if q == "mu":
                        inc = sum(x[i, j + 1, k] * mu_inc[i, l, j, k] for j in range(M))
                    elif q == "v" and wiener:
                        inc = cfg.sigma[i, l] ** 2 * x[i, 1:, k].sum()
                    elif q == "v":
                        inc = sum(x[i, j + 1, k] * v_inc[i, l, j, k] for j in range(M))
                    elif q == "K":
                        inc = sum(x[i, j + 1, k] * K_inc[i, l, j, k] for j in range(M))
                    s_new, g_new = s + inc, g
                if q == "mu" and cfg.C_R[i, l] > 0:
                    gp = g if ard1 else 0.0
                    arr["z"][i, l, k] = rho * (s - gp) if m[i, l, k] else 0.0
                s, g = s_new, g_new
                arr[q][i, l, k] = s
                if anchored:
                    arr["g_" + q][i, l, k] = g

u = float(np.max(np.sum(arr["mu"], axis=1)))
z0 = np.nan_to_num(arr["z"], nan=0.0)
step = np.array([cfg.C_M * x[:, 0, k].sum() + (cfg.C_R * z0[:, :, k]).sum()
                 + (cfg.C_rep * r[:, :, k]).sum() for k in range(T)])
J_init, J_op = step[:H1].sum(), step[H1:].sum()
E = cfg.options["evaluation_horizon"]
objectives = {
    "total": J_init + J_op + cfg.C_D * u,
    "operating_average": J_op / H2 + cfg.C_D * u,
    "evaluation_total": J_init + (E - H1) / H2 * J_op + cfg.C_D * u,
}
print("step_costs", step.tolist())
print("J_init", J_init, "J_op", J_op, "u", u)
for mode, value in objectives.items():
    print("objective", mode, repr(value))

result = {
    "status": "optimal", "objective": objectives["evaluation_total"], "mip_gap": 0.0,
    "bound": objectives["evaluation_total"],
    "F": F, "M": M, "L": L, "H1": H1, "H2": H2, "T": T, "H": cfg.H,
    "model": cfg.model.astype(str).tolist(), "component_names": cfg.component_names,
    "tau": cfg.tau, "mu_0": cfg.mu_0, "v_0": cfg.v_0,
    "bound_method": [[None, None], [None, None], ["bernstein", "chernoff"]],
    "repair_model": cfg.repair_model.astype(str).tolist(),
    "reliability_impl": cfg.reliability_impl.astype(str).tolist(),
    "encoding": cfg.encoding.astype(str).tolist(),
    "objective_mode": "evaluation_total", "evaluation_horizon": E,
    "x": x, "m": m, "r": r, "idle": x[:, 0, :][:, None, :] - m - r,
    **arr, "u": u,
    "step_costs": step, "J_initialization": J_init, "J_op": J_op, "J_op_average": J_op / H2,
    "peak_damage": u, "damage_penalty": cfg.C_D * u,
    "operating_objective": J_op / H2 + cfg.C_D * u, "J_total": J_init + J_op + cfg.C_D * u,
    "projected_evaluation_cost": J_init + (E - H1) * J_op / H2,
    "evaluation_objective": objectives["evaluation_total"],
    "component_costs": {"C_M": cfg.C_M, "C_D": cfg.C_D, "C_R": cfg.C_R, "C_rep": cfg.C_rep},
    "performance": {}, "optimization_progress": [],
    "warm_start": {"applied": False, "values": 0, "mode": None, "source_horizon": None,
                   "completion": {"requested": False, "status": "not_requested",
                                  "fixed_integer_variables": 0, "values_copied": 0,
                                  "runtime_seconds": 0.0, "iis_constraints": []}},
    "relaxation_warm_start": {"enabled": False, "status": "disabled"},
}
for ext in ("yaml", "json", "h5"):
    solver._save_results(result, HERE / f"result.{ext}")

# ---- loop rows ----
for q in ("mu", "v", "R", "K", "g_mu", "g_v", "g_R", "g_K"):
    d = arr[q][:, :, T - 1] - arr[q][:, :, H1 - 1]
    print("loop", q, "max excess", np.nanmax(d) if np.isfinite(d).any() else "all NaN")

# ---- trajectories ----
for q in ("mu", "v", "K", "g_mu", "g_v", "g_K", "z"):
    print(q)
    print(np.array2string(arr[q], precision=6, max_line_width=150))

# ---- reliability margins, exact form; min with the side row ----
margin = np.full((F, L, T), np.nan)
mu, v, K = arr["mu"], arr["v"], arr["K"]
for i in range(F):
    for l in range(L):
        tau, eps = cfg.tau[i, l], cfg.epsilon[i, l]
        model = cfg.model[i, l]
        if model in ("gaussian", "wiener"):
            kg = ndtri(1 - eps)
            margin[i, l] = np.minimum((tau - mu[i, l]) ** 2 - kg ** 2 * v[i, l], tau - mu[i, l])
        elif model == "inverse_gaussian":
            cap = ig_mean_cap(cfg.eta[i, l], tau, eps)
            print("IG cap", (i, l), repr(cap))
            margin[i, l] = cap - mu[i, l]
        elif model == "gamma":
            mi = _lookup(cfg.mu, cfg.mu_trans, H1, H2)[i, l]
            bi = _lookup(cfg.gamma_beta, cfg.gamma_beta_trans, H1, H2)[i, l]
            bs = smallest_rate(mi, bi, cfg.mu_0[i, l], cfg.gamma_beta_0[i, l],
                               cfg.replacement_mu[i, l], cfg.gamma_beta_new[i, l],
                               cfg.allow_replacement)
            cap = mean_cap(bs, tau, eps)
            print("Gamma beta*", bs, "cap", repr(cap))
            margin[i, l] = cap - mu[i, l]
        else:
            kappa = math.log(1 / eps)
            bound = cfg.bound_method[i, l]
            if bound == "bernstein":
                b = cfg.support_const[i, l]
                c = kappa * b / 3
                print("Bernstein b", b, "c", c)
                margin[i, l] = np.minimum((tau - c - mu[i, l]) ** 2 - c ** 2 - 2 * kappa * v[i, l],
                                          tau - c - mu[i, l])
            elif bound == "chernoff":
                s = cfg.s_chernoff[i, l]
                margin[i, l] = math.log(eps) - (K[i, l] - s * tau)
print("margins")
print(np.array2string(margin, precision=9, max_line_width=150))
print("min margin", np.nanmin(margin), np.unravel_index(np.nanargmin(margin), margin.shape))
print("max mu/tau", np.nanmax(mu / cfg.tau[:, :, None]))

# ---- read back ----
for ext in ("yaml", "json", "h5"):
    back = read_input(HERE / f"result.{ext}")
    xb = np.asarray(back["x"], dtype=float)
    Rb = np.asarray(back["R"], dtype=float)
    print(ext, "keys", len(back), "x equal", np.array_equal(xb, x), "R all NaN", np.isnan(Rb).all(),
          "H", back["H"], "bound_method", back["bound_method"], "objective", back["objective"])
