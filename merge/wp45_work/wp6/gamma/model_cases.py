"""Model cases of the WP6 Gamma plan, through the linear emulator (fakegp) and the real config.py.

Usage: python model_cases.py [path of gamma module]
Default module: gamma_planned.py beside this script (the plan's code). After the executor's
step, pass the worktree's gamma.py. Prints one line per check and "failures: N".
"""
import importlib.util
import math
import os
import pathlib
import sys
import types

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SCRATCH = HERE.parent.parent
SRC = pathlib.Path(os.environ.get("FM_SRC") or r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
sys.path.insert(0, str(SCRATCH / "wp5" / "fakegp"))
package = types.ModuleType("fleet_management")          # skip fleet_management/__init__.py
package.__path__ = [str(SRC / "fleet_management")]
sys.modules["fleet_management"] = package

from fleet_management.degradation_model import base       # noqa: E402  (fake gurobipy)
from fleet_management.config import load_config           # noqa: E402

module_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "gamma_planned.py"
spec = importlib.util.spec_from_file_location("gamma_under_test", module_path)
gamma = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gamma)
base.CELL_BUILDERS["gamma"] = gamma.GammaCellBuilder()    # the module registered it; make sure

FAILURES = []


def check(label, got, expected, tol=1e-7):
    if isinstance(expected, (list, tuple, np.ndarray)) and not isinstance(expected, str):
        got_a, exp_a = np.asarray(got, dtype=float), np.asarray(expected, dtype=float)
        ok = got_a.shape == exp_a.shape and bool(np.all(
            (np.isnan(got_a) & np.isnan(exp_a)) | (np.abs(got_a - exp_a) <= tol)))
    elif isinstance(expected, float) and isinstance(got, (float, int)) and got is not None:
        ok = abs(got - expected) <= tol * max(1.0, abs(expected))
    else:
        ok = got == expected
    print(("ok   " if ok else "FAIL ") + f"{label}: {got!r}" + ("" if ok else f" (expected {expected!r})"))
    if not ok:
        FAILURES.append(label)


def gamma_input(**changes):
    data = {"F": 2, "M": 1, "H": [2, 2], "model": "gamma", "repair_model": "ard1",
            "encoding": "bigm", "tau": 1.0, "epsilon": 0.1, "rho": 0.5, "mu_0": 0.2,
            "replacement_mu": 0.1, "mu": [[[0.3]], [[0.0]]], "gamma_beta": 20.0,
            "C_M": 1.0, "C_R": 2.0, "C_rep": 3.0, "C_D": 0.0,
            "verbose": 0, "mip_gap": 0.0, "time_limit": 60}
    data.update(changes)
    return data


SCHEDULE_X = [(0, 1, 0), (0, 0, 1), (0, 1, 2), (0, 0, 3), (1, 1, 1), (1, 1, 3)]


def fix_schedule(ctx):
    for key, var in ctx.x.items():
        var.LB = var.UB = 1.0 if key in SCHEDULE_X else 0.0
    for key, var in ctx.m_rep.items():
        var.LB = var.UB = 1.0 if key == (0, 0, 1) else 0.0
    if ctx.r_rep is not None:
        for key, var in ctx.r_rep.items():
            var.LB = var.UB = 1.0 if key == (0, 0, 3) else 0.0
    for (i, l, k), var in ctx.nb.items():
        used = ctx.m_rep[i, l, k].LB + (ctx.r_rep[i, l, k].LB if ctx.r_rep is not None else 0.0)
        var.LB = var.UB = 1.0 - used


def solve_fixed(cfg):
    ctx = base.build_fleet(cfg, base.resolve_run_options(cfg))
    fix_schedule(ctx)
    terms = list(ctx.mu_var.values()) + list(ctx.anchor_vars["mu"].values()) + list(ctx.z_var.values())
    ctx.model.setObjective(base.gp.quicksum(terms))
    ctx.model.optimize()
    return ctx


def trajectory(variables, i, T):
    return [variables[i, 0, k].X if (i, 0, k) in variables else math.nan for k in range(T)]


CAP20 = 0.744517324566        # A_1(20, 0.1) / 20 = A_hat(20, 0.1) / 20
CAP2 = 0.414912231273         # A_1(2, 0.1) / 2
CAP5 = 0.553864057892         # A_1(5, 0.1) / 5
EXPECTED = {
    "ard1": dict(mu=[0.5, 0.35, 0.65, 0.1], g=[0.2, 0.35, 0.35, 0.1], z=[0.0, 0.15, 0.0, 0.0]),
    "ardinf": dict(mu=[0.5, 0.25, 0.55, 0.1], g=[math.nan] * 4, z=[0.0, 0.25, 0.0, 0.0]),
}

# --- G1: dynamics, anchors, repair cost and the cap, both laws and both encodings
for law in ("ard1", "ardinf"):
    for encoding in ("bigm", "indicator"):
        tag = f"G1 {law} {encoding}"
        ctx = solve_fixed(load_config(gamma_input(repair_model=law, encoding=encoding)))
        check(f"{tag} status", ctx.model.Status, base.GRB.OPTIMAL)
        if ctx.model.Status != base.GRB.OPTIMAL:
            continue
        T = ctx.T
        check(f"{tag} mu train 0", trajectory(ctx.mu_var, 0, T), EXPECTED[law]["mu"])
        check(f"{tag} gmu train 0", trajectory(ctx.anchor_vars["mu"], 0, T), EXPECTED[law]["g"])
        check(f"{tag} z train 0", trajectory(ctx.z_var, 0, T), EXPECTED[law]["z"])
        check(f"{tag} mu train 1", trajectory(ctx.mu_var, 1, T), [0.2] * 4)
        check(f"{tag} z train 1", trajectory(ctx.z_var, 1, T), [0.0] * 4)
        extras = ctx.extras["gamma"]
        check(f"{tag} beta_star", (extras["beta_star"][0, 0], extras["beta_star"][1, 0]), (20.0, None))
        check(f"{tag} cap train 0", extras["cap"][0, 0], CAP20, tol=1e-9)
        check(f"{tag} cap train 1", extras["cap"][1, 0], 1.0)
        names = set(ctx.model.getConstrs()) | {n for *_, n in ctx.model.indicators}
        check(f"{tag} cap rows rel_i_0_k", all(f"rel_{i}_0_{k}" in names for i in (0, 1) for k in range(T)), True)
        check(f"{tag} loop rows", ("loop_mu_0_0" in names, "loop_gmu_0_0" in names), (True, law == "ard1"))
        check(f"{tag} selector vars", len(ctx.nb), 8 if encoding == "indicator" else 0)
        check(f"{tag} no quadratic rows, NonConvex unset",
              (ctx.model.NumQConstrs, getattr(ctx.model.Params, "NonConvex", None)), (0, None))
        rhs = []
        for tc, name in ctx.model.rows:
            if name == "rel_0_0_2":
                rhs.append(-tc.expr.const)
        check(f"{tag} RHS of rel_0_0_2", rhs, [CAP20], tol=1e-9)

# --- G2: the cap binds: beta_new = 2 with replacement allowed gives cap 0.4149 < mu = 0.5 on day 1
for law in ("ard1", "ardinf"):
    ctx = solve_fixed(load_config(gamma_input(repair_model=law, gamma_beta_new=2.0)))
    check(f"G2 {law} beta_star", (ctx.extras["gamma"]["beta_star"][0, 0], ctx.extras["gamma"]["beta_star"][1, 0]),
          (2.0, 2.0))
    check(f"G2 {law} cap", ctx.extras["gamma"]["cap"][0, 0], CAP2, tol=1e-9)
    check(f"G2 {law} infeasible", ctx.model.Status in (base.GRB.INFEASIBLE, base.GRB.INF_OR_UNBD), True)

# --- G3: beta_0 = 5: cap 0.55386; ard1 peak 0.65 is infeasible, ardinf peak 0.55 is feasible
ctx = solve_fixed(load_config(gamma_input(repair_model="ard1", gamma_beta_0=5.0)))
check("G3 ard1 beta_star (train 0, train 1)",
      (ctx.extras["gamma"]["beta_star"][0, 0], ctx.extras["gamma"]["beta_star"][1, 0]), (5.0, 5.0))
check("G3 ard1 infeasible", ctx.model.Status in (base.GRB.INFEASIBLE, base.GRB.INF_OR_UNBD), True)
ctx = solve_fixed(load_config(gamma_input(repair_model="ardinf", gamma_beta_0=5.0)))
check("G3 ardinf cap", ctx.extras["gamma"]["cap"][0, 0], CAP5, tol=1e-9)
check("G3 ardinf optimal", ctx.model.Status, base.GRB.OPTIMAL)


# --- G4: beta* sources, prepare only
def prepared(**changes):
    cfg = load_config(gamma_input(**changes))
    ctx = base.build_context(cfg, base.resolve_run_options(cfg))
    base.CELL_BUILDERS["gamma"].prepare(ctx, cfg, ctx.cells_of("gamma"), ctx.opts)
    stars = ctx.extras["gamma"]["beta_star"]
    return stars[0, 0], stars[1, 0]


check("G4 beta_new ignored without replacement", prepared(gamma_beta_new=2.0, allow_replacement=False),
      (20.0, None))
check("G4 beta_new ignored when replacement_mu = 0", prepared(gamma_beta_new=2.0, replacement_mu=0.0),
      (20.0, None))
check("G4 transitory rate on days 1-2", prepared(gamma_beta_trans=2.0), (2.0, None))
check("G4 transitory rate where mu_trans = 0", prepared(gamma_beta_trans=2.0, mu_trans=0.0), (20.0, None))
check("G4 transitory mean with operating rate", prepared(mu_trans=[[[0.0]], [[0.1]]]), (20.0, 20.0))
check("G4 beta_0 ignored when mu_0 = 0", prepared(gamma_beta_0=5.0, mu_0=0.0), (20.0, None))
check("G4 rate per cell", prepared(gamma_beta=[[20.0], [3.0]], mu=[[[0.3]], [[0.1]]]), (20.0, 3.0))


# --- C: the planned config.py check, run after load_config on the same cfg
def config_check(cfg):
    """The block of section 5, as a function (the same code)."""
    from fleet_management import config
    m = config._masks(cfg.model, cfg.bound_method, cfg.repair_model, cfg.reliability_impl)
    if not cfg.allow_replacement or not m["gamma"].any():
        return

    def check_row(applies, ok, key, rule):
        config._require(cfg, ~applies | ok, key, rule)

    with np.errstate(divide="ignore", invalid="ignore"):
        mu = cfg.replacement_mu
        mu_inc = config._lookup(cfg.mu, cfg.mu_trans, cfg.H1, cfg.H2)
        beta_inc = config._lookup(cfg.gamma_beta, cfg.gamma_beta_trans, cfg.H1, cfg.H2)
        cap = np.full(cfg.tau.shape, np.nan)
        for i, l in np.argwhere(m["gamma"]):
            beta_star = gamma.smallest_rate(
                mu_inc[i, l], beta_inc[i, l], float(cfg.mu_0[i, l]), float(cfg.gamma_beta_0[i, l]),
                float(cfg.replacement_mu[i, l]), float(cfg.gamma_beta_new[i, l]), True)
            cap[i, l] = gamma.mean_cap(beta_star, float(cfg.tau[i, l]), float(cfg.epsilon[i, l]))
        check_row(m["gamma"], config._le(mu, cap), "replacement_mu",
                  "must be <= the Gamma cap A_1(beta_star*tau, epsilon)/beta_star at a gamma cell")


def small(**changes):
    data = {"F": 2, "M": 1, "H": [1, 2], "model": "gamma", "tau": 1.0, "epsilon": 0.1,
            "rho": 0.5, "mu_0": 0.1, "mu": 0.05, "C_M": 1.0, "C_R": 1.0, "C_rep": 1.0,
            "C_D": 1.0, "gamma_beta": 20.0}
    data.update(changes)
    return data


def outcome(**changes):
    try:
        config_check(load_config(small(**changes)))
    except ValueError as error:
        return "fails" if "must be <= the Gamma cap" in str(error) else f"other error: {error}"
    return "passes"


check("C1 at 0.999 cap(20)", outcome(replacement_mu=0.999 * CAP20), "passes")
check("C1 at 1.001 cap(20)", outcome(replacement_mu=1.001 * CAP20), "fails")
check("C2 beta_new 2, mu_new 0.5", outcome(gamma_beta_new=2.0, replacement_mu=0.5), "fails")
check("C2 beta_new 2, mu_new 0.41", outcome(gamma_beta_new=2.0, replacement_mu=0.41), "passes")
check("C2 no beta_new, mu_new 0.5", outcome(replacement_mu=0.5), "passes")
check("C2 no replacement", outcome(gamma_beta_new=2.0, replacement_mu=0.5, allow_replacement=False), "passes")
check("C3 transitory rate 2 on day 1", outcome(gamma_beta_trans=2.0, replacement_mu=0.5), "fails")
check("C3 transitory rate where mu_trans = 0", outcome(gamma_beta_trans=2.0, mu_trans=0.0, replacement_mu=0.5),
      "passes")
check("C4 beta_0 2", outcome(gamma_beta_0=2.0, replacement_mu=0.5), "fails")
check("C4 beta_0 2 with mu_0 = 0", outcome(gamma_beta_0=2.0, mu_0=0.0, replacement_mu=0.5), "passes")
check("C5 below t1: rate 0.2, cap 0.3306 tau", outcome(gamma_beta=0.2, replacement_mu=0.34), "fails")
check("C5 below t1: 0.32 passes", outcome(gamma_beta=0.2, replacement_mu=0.32), "passes")
check("C6 no rate (mu = 0, no beta_0): only mu_new <= tau", outcome(mu=0.0, replacement_mu=0.9), "passes")
check("C7 tolerance: cap*(1 + 1e-13)", outcome(replacement_mu=CAP20 * (1 + 1e-13)), "passes")
cfg = load_config(small(gamma_beta=0.2))
check("C5 mean_cap(0.2, 1, 0.1) = r*(0.1)", gamma.mean_cap(0.2, 1.0, 0.1), 0.330617069585, tol=1e-9)

print(f"failures: {len(FAILURES)}")
