"""Prototype of the per-cell loops of steps C2, C3 and C5, run on the test cases of section 3.

Checks the numbers that section 3 expects; not the module code.
"""
import math
import sys
import types
from pathlib import Path

import numpy as np
from scipy.special import gammaincc

SRC = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
package = types.ModuleType("fleet_management")
package.__path__ = [str(SRC / "fleet_management")]
sys.modules["fleet_management"] = package
sys.modules["gurobipy"] = None
from fleet_management.config import load_config, _lookup  # noqa: E402
from fleet_management.degradation_model.gamma import mean_cap, smallest_rate  # noqa: E402
from fleet_management.degradation_model.gamma_utils.gamma_tail_bound import moschopoulos_tail_probability  # noqa: E402
from fleet_management.degradation_model.gamma_utils.gamma_tail_validator import ExactGammaTerm  # noqa: E402
from fleet_management.degradation_model.gamma_utils.gamma_stochastic_validator import _one_sided_binomial_upper  # noqa


def data(**changes):
    d = {"F": 2, "M": 1, "H": [2, 2], "model": "gamma", "repair_model": "ard1", "tau": 1.0,
         "epsilon": 0.1, "rho": 0.5, "mu_0": 0.2, "replacement_mu": 0.1, "mu": [[[0.3]], [[0.0]]],
         "gamma_beta": 20.0, "C_M": 1.0, "C_R": 2.0, "C_rep": 3.0, "C_D": 0.0}
    d.update(changes)
    return d


TABLE = {
    "A": ({}, [0.5, 0.35, 0.65, 0.1], [0.2, 0.35, 0.35, 0.1], [0, 0.15, 0, 0], 0.2),
    "B": ({"repair_model": "ardinf"}, [0.5, 0.25, 0.55, 0.1], None, [0, 0.25, 0, 0], None),
    "C": ({"gamma_beta_0": 20.0, "anchor_mu_0": 0.05, "gamma_beta_new": 30.0},
          [0.5, 0.275, 0.575, 0.1], [0.05, 0.275, 0.275, 0.1], [0, 0.225, 0, 0], 0.05),
}


def result(case):
    _, mu0, g0, z0, g1 = TABLE[case]
    x = np.zeros((2, 2, 4))
    for i, j, k in ((0, 1, 0), (0, 0, 1), (0, 1, 2), (0, 0, 3), (1, 1, 1), (1, 1, 3)):
        x[i, j, k] = 1
    m = np.zeros((2, 1, 4)); m[0, 0, 1] = 1
    r = np.zeros((2, 1, 4)); r[0, 0, 3] = 1
    mu = np.array([[mu0], [[0.2] * 4]], dtype=float)
    g = np.full((2, 1, 4), np.nan) if g0 is None else np.array([[g0], [[g1] * 4]], dtype=float)
    z = np.array([[z0], [[0.0] * 4]], dtype=float)
    return {"x": x, "m": m, "r": r, "mu": mu, "g_mu": g, "z": z, "T": 4}


def event_of(x, m, r, i, l, k):
    if r[i, l, k]:
        return "replacement"
    if m[i, l, k]:
        return "repair"
    if x[i, 1:, k].any():
        return "mission"
    return "idle"


def replay(cfg, res, tolerance=1e-8):
    x, m, r = (np.rint(res[n]).astype(int) for n in ("x", "m", "r"))
    mu, z, g_mu = res["mu"], res["z"], res["g_mu"]
    mu_inc = _lookup(cfg.mu, cfg.mu_trans, cfg.H1, cfg.H2)
    beta_inc = _lookup(cfg.gamma_beta, cfg.gamma_beta_trans, cfg.H1, cfg.H2)
    T = cfg.T
    maxima, violations = {}, []

    def check(i, l, k, event, field, expected, saved, error):
        maxima[field] = max(maxima.get(field, 0.0), error)
        if error > tolerance:
            violations.append((i, l, k, event, field, expected, saved, error))

    for i in range(2):
        l = 0
        beta_star = smallest_rate(mu_inc[i, l], beta_inc[i, l], float(cfg.mu_0[i, l]), float(cfg.gamma_beta_0[i, l]),
                                  float(cfg.replacement_mu[i, l]), float(cfg.gamma_beta_new[i, l]),
                                  bool(cfg.allow_replacement))
        cap = mean_cap(beta_star, float(cfg.tau[i, l]), float(cfg.epsilon[i, l]))
        anchored = str(cfg.repair_model[i, l]) == "ard1"
        rho = float(cfg.rho[i, l])
        ref_mu = float(cfg.mu_0[i, l])
        ref_g = float(cfg.anchor_mu_0[i, l]) if anchored else 0.0
        prev_mu, prev_g = ref_mu, ref_g
        events = []
        for k in range(T):
            event = event_of(x, m, r, i, l, k)
            events.append(event)
            if event == "replacement":
                ref_mu = float(cfg.replacement_mu[i, l])
            elif event == "repair":
                ref_mu = (1.0 - rho) * ref_mu + rho * ref_g
            else:
                ref_mu += float(np.dot(x[i, 1:, k], mu_inc[i, l, :, k]))
            if anchored and event in ("replacement", "repair"):
                ref_g = ref_mu
            saved = float(mu[i, l, k])
            check(i, l, k, event, "tracked_mean", ref_mu, saved, max(0.0, ref_mu - saved))
            if anchored:
                saved_g = float(g_mu[i, l, k])
                check(i, l, k, event, "mean_anchor", ref_g, saved_g, max(0.0, ref_g - saved_g))
            if not math.isnan(z[i, l, k]):
                removed = float(m[i, l, k]) * rho * (prev_mu - prev_g)
                check(i, l, k, event, "removed_mean", removed, float(z[i, l, k]), max(0.0, removed - float(z[i, l, k])))
            check(i, l, k, event, "cap", cap, saved, max(0.0, saved - cap))
            prev_mu, prev_g = saved, (float(g_mu[i, l, k]) if anchored else 0.0)
        for array in [mu] + ([g_mu] if anchored else []):
            first, last = float(array[i, l, cfg.H1 - 1]), float(array[i, l, T - 1])
            check(i, l, T - 1, events[-1], "repeatability", first, last, max(0.0, last - first))
    return maxima, violations


def seed(mean, rate, source, step):
    if mean <= 0.0:
        return [], 0.0
    if math.isnan(rate):
        return [], float(mean)
    return [ExactGammaTerm(mean * rate, rate, source, step)], 0.0


def exact_tail(terms, constant, tau):
    if not terms:
        return 1.0 if constant > tau else 0.0
    if constant >= tau:
        return 1.0
    return moschopoulos_tail_probability([t.shape for t in terms], [t.rate for t in terms], tau - constant).upper_bound


def exact(cfg, res):
    x, m, r = (np.rint(res[n]).astype(int) for n in ("x", "m", "r"))
    mu_inc = _lookup(cfg.mu, cfg.mu_trans, cfg.H1, cfg.H2)
    beta_inc = _lookup(cfg.gamma_beta, cfg.gamma_beta_trans, cfg.H1, cfg.H2)
    out = {}
    for i in range(2):
        l = 0
        anchored = str(cfg.repair_model[i, l]) == "ard1"
        remaining = max(0.0, 1.0 - float(cfg.rho[i, l]))
        beta_0, beta_new = float(cfg.gamma_beta_0[i, l]), float(cfg.gamma_beta_new[i, l])
        mu_0, tau = float(cfg.mu_0[i, l]), float(cfg.tau[i, l])
        if anchored:
            g_0 = float(cfg.anchor_mu_0[i, l])
            frozen, c_frozen = seed(g_0, beta_0, "initial_anchor", -1)
            active, c_active = seed(mu_0 - g_0, beta_0, "initial", -1)
        else:
            frozen, c_frozen = [], 0.0
            active, c_active = seed(mu_0, beta_0, "initial", -1)
        rows = []
        for k in range(cfg.T):
            event = event_of(x, m, r, i, l, k)
            if event == "replacement":
                s, c_s = seed(float(cfg.replacement_mu[i, l]), beta_new, "replacement", k)
                if anchored:
                    frozen, c_frozen, active, c_active = s, c_s, [], 0.0
                else:
                    frozen, c_frozen, active, c_active = [], 0.0, s, c_s
            elif event == "repair":
                repaired = [] if remaining == 0.0 else [t.after_ardinf(remaining) for t in active]
                if anchored:
                    frozen, c_frozen = frozen + repaired, c_frozen + remaining * c_active
                    active, c_active = [], 0.0
                else:
                    active, c_active = repaired, remaining * c_active
            elif event == "mission":
                for j0 in np.flatnonzero(x[i, 1:, k]):
                    mean, rate = float(mu_inc[i, l, j0, k]), float(beta_inc[i, l, j0, k])
                    if mean > 0.0:
                        active.append(ExactGammaTerm(mean * rate, rate, f"mission_{j0 + 1}", k))
            terms = frozen + active
            constant = c_frozen + c_active
            upper = exact_tail(terms, constant, tau)
            exact_mean = constant + sum(t.mean for t in terms)
            frozen_mean = c_frozen + sum(t.mean for t in frozen)
            rows.append((k, event, upper, round(exact_mean, 12), round(frozen_mean, 12),
                         [(round(t.shape, 9), round(t.rate, 9), "f") for t in frozen]
                         + [(round(t.shape, 9), round(t.rate, 9), "a") for t in active]))
        out[i] = rows
    return out


EXPECTED = {
    "A": [gammaincc(6, 16), gammaincc(6, 32), 0.015106503928459624, 0.0],
    "B": [gammaincc(6, 16), gammaincc(6, 36), 0.0042209224829627395, 0.0],
    "C": [gammaincc(10, 20), 1.0539648750813092e-06, 0.009116347927228055, gammaincc(3, 30)],
}
for case, (changes, *_rest) in TABLE.items():
    cfg = load_config(data(**changes))
    res = result(case)
    maxima, violations = replay(cfg, res)
    print(case, "replay maxima", {k: f"{v:.1e}" for k, v in maxima.items()}, "violations", violations)
    rows = exact(cfg, res)
    for (k, event, upper, mean, fmean, terms), want in zip(rows[0], EXPECTED[case]):
        print(f"  {case} k={k} {event:11s} upper={upper:.17g} diff={abs(upper - want):.1e} mean={mean} "
              f"frozen={fmean} terms={terms}")
    print("  train1 uppers", [f"{row[2]:.6g}" for row in rows[1]], "Q(4,20)=", gammaincc(4, 20))

# one-change records of 3.2
cfg = load_config(data())
for key, index, value in (("mu", (0, 0, 2), 0.64), ("z", (0, 0, 1), 0.1), ("mu", (0, 0, 2), 0.8),
                          ("mu", (0, 0, 3), 0.4), ("mu", (0, 0, 2), 0.70)):
    res = result("A")
    res[key] = res[key].copy()
    res[key][index] = value
    print("A", key, index, value, "->", replay(cfg, res)[1])
# exact with epsilon 0.01 and full repair
print("B rho=1:", [row[2] for row in exact(load_config(data(repair_model="ardinf", rho=1.0)), result("B"))[0]],
      "Q(6,20)=", gammaincc(6, 20))
print("CP:", _one_sided_binomial_upper(0, 100, 0.95), _one_sided_binomial_upper(3, 100, 0.95),
      _one_sided_binomial_upper(100, 100, 0.95))
