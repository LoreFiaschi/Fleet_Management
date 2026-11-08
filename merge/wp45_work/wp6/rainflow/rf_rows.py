"""Row check of the rainflow builder on the hand case, without Gurobi.

Usage: python rf_rows.py [path to rainflow.py]
Default path: the prototype ``proto_rainflow.py`` beside this script. Loads
``base.py`` of the worktree and the builder through the linear emulator
``wp5/fakegp`` (scipy LP with every binary fixed). Builds each case with the
schedule fixed (x, m, r, and n at indicator cells), minimises the states and
compares with the expected values of the plan (section 3). Only linear forms:
Markov and Chernoff (exact), quadratic bounds in the form tangent.
Prints one line per check and ``failures: N`` at the end.
"""
import importlib.util
import math
import pathlib
import sys
import types

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SCRATCH = HERE.parent.parent
SRC = pathlib.Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
BUILDER = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "proto_rainflow.py"

sys.path.insert(0, str(SCRATCH / "wp5" / "fakegp"))
for name, path in (("fleet_management", SRC / "fleet_management"),
                   ("fleet_management.degradation_model", SRC / "fleet_management" / "degradation_model")):
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package
import gurobipy as gp  # noqa: E402  (the emulator)
from fleet_management.degradation_model import base  # noqa: E402

spec = importlib.util.spec_from_file_location("fleet_management.degradation_model.rainflow", BUILDER)
rainflow = importlib.util.module_from_spec(spec)
sys.modules["fleet_management.degradation_model.rainflow"] = rainflow
spec.loader.exec_module(rainflow)

NAN = float("nan")
OPTIONS = dict(verbose=0, mip_gap=0.0, time_limit=60.0, fast=False, gurobi_params=None,
               objective_mode="total", evaluation_horizon=None, progress_interval_seconds=5.0,
               relaxation_warm_start=False, relaxation_time_limit=60.0, validation=None)

# per bound: second quantity (start, replacement, increment profile value) and inputs
BOUND_INPUTS = {
    "markov": dict(eps=(0.7, 0.6)),
    "cantelli": dict(eps=(0.2, 0.11), v_0=0.004, replacement_v=0.002, v=0.01),
    "hoeffding": dict(eps=(0.05, 0.01), R_0=0.01, R_new=0.005, support=0.2),
    "bernstein": dict(eps=(0.4, 0.3), v_0=0.004, replacement_v=0.002, v=0.01, support=0.25,
                      support_0=0.3, support_new=0.1),
    "chernoff": dict(eps=(0.5, 0.4), s_chernoff=4.0, cgf_0=1.0, cgf_new=0.5, cgf=1.5),
}
EXPECTED = {   # train 0, days 1..4: (state, anchor) per law; train 1 keeps its start
    "mu": {"ard1": ([0.5, 0.35, 0.65, 0.1], [0.2, 0.35, 0.35, 0.1]),
           "ardinf": ([0.5, 0.25, 0.55, 0.1], None)},
    "v": {"ard1": ([0.014, 0.0065, 0.0165, 0.002], [0.004, 0.0065, 0.0065, 0.002]),
          "ardinf": ([0.014, 0.0035, 0.0135, 0.002], None)},
    "R": {"ard1": ([0.05, 0.02, 0.06, 0.005], [0.01, 0.02, 0.02, 0.005]),
          "ardinf": ([0.05, 0.0125, 0.0525, 0.005], None)},
    "K": {"ard1": ([2.5, 1.75, 3.25, 0.5], [1.0, 1.75, 1.75, 0.5]),
          "ardinf": ([2.5, 1.25, 2.75, 0.5], None)},
}
Z = {"ard1": [0.0, 0.15, 0.0, 0.0], "ardinf": [0.0, 0.25, 0.0, 0.0]}
RESULT_KEY = {"mu": ("mu", "g_mu"), "v": ("v", "g_v"), "R": ("R", "g_R"), "K": ("K", "g_K")}


def make_cfg(bound, law, encoding="bigm", form="tangent", eps_index=0, **changes):
    F, L, M, H1, H2 = 2, 1, 1, 2, 2
    inputs = BOUND_INPUTS[bound]
    eps = inputs["eps"][eps_index]
    tau = 1.0
    kappa = math.log(1.0 / eps)

    def cell(value):
        return np.full((F, L), value, dtype=float)

    def sel(value):
        return np.full((F, L), value, dtype=object)

    def profile(value):
        array = np.zeros((F, L, M, H2))
        array[0] = value
        return array

    quadratic = bound in ("cantelli", "hoeffding", "bernstein")
    has_v = bound in ("cantelli", "bernstein")
    v_0 = inputs.get("v_0", NAN)
    R_0 = inputs.get("R_0", NAN)
    s = inputs.get("s_chernoff", NAN)
    cgf_0 = inputs.get("cgf_0", NAN)
    b = max(inputs.get("support_0", 0.0), inputs.get("support_new", 0.0),
            inputs.get("support", 0.0)) if bound == "bernstein" else NAN
    c = kappa * b / 3.0
    ard1 = law == "ard1"
    d = dict(
        F=F, M=M, L=L, H=[H1, H2], H1=H1, H2=H2, T=H1 + H2, component_names=["Component 1"],
        model=sel("rainflow"), repair_model=sel(law), bound_method=sel(bound),
        reliability_impl=sel(form if quadratic else "exact"), encoding=sel(encoding),
        tau=cell(tau), epsilon=cell(eps), tangent_ref=cell(0.5 if form == "tangent" else NAN),
        rho=cell(0.5), mu_0=cell(0.2), v_0=cell(v_0), support_0=cell(inputs.get("support_0", NAN)),
        R_0=cell(R_0), cgf_0=cell(cgf_0),
        anchor_mu_0=cell(0.2 if ard1 else NAN), anchor_v_0=cell(v_0 if ard1 else NAN),
        anchor_R_0=cell(R_0 if ard1 else NAN), anchor_K_0=cell(cgf_0 if ard1 else NAN),
        replacement_mu=cell(0.1), replacement_v=cell(inputs.get("replacement_v", NAN)),
        support_new=cell(inputs.get("support_new", NAN)), R_new=cell(inputs.get("R_new", NAN)),
        cgf_new=cell(inputs.get("cgf_new", NAN)), s_chernoff=cell(s),
        sigma=cell(NAN), eta=cell(NAN), v_max_user=cell(NAN), gamma_beta_0=cell(NAN),
        gamma_beta_new=cell(NAN), C_R=cell(2.0), C_rep=cell(3.0),
        pwl_points=8, allow_replacement=True, replacement_as_new=False,
        C_M=1.0, C_D=0.0, penalty_type="inf_norm",
        mu=profile(0.3), v=profile(inputs["v"]) if has_v else None,
        support=profile(inputs["support"]) if "support" in inputs else None,
        cgf=profile(inputs["cgf"]) if "cgf" in inputs else None, gamma_beta=None,
        mu_trans=None, v_trans=None, support_trans=None, cgf_trans=None, gamma_beta_trans=None,
        support_const=cell(b),
        range_mu=cell(tau),
        range_v=cell(eps * tau ** 2 / (1 - eps) if bound == "cantelli"
                     else tau * (tau - 2 * c) / (2 * kappa) if bound == "bernstein" else NAN),
        range_R=cell(2 * tau ** 2 / kappa if bound == "hoeffding" else NAN),
        range_K=cell(s * tau - kappa if bound == "chernoff" else NAN),
        options=dict(OPTIONS),
    )
    for key, value in changes.items():
        d[key] = cell(value) if key in ("v_max_user",) else value
    return types.SimpleNamespace(**d)


def schedule():
    x, m, r = np.zeros((2, 2, 4)), np.zeros((2, 1, 4)), np.zeros((2, 1, 4))
    for index in [(0, 1, 0), (0, 0, 1), (0, 1, 2), (0, 0, 3), (1, 1, 1), (1, 1, 3)]:
        x[index] = 1.0
    m[0, 0, 1] = 1.0
    r[0, 0, 3] = 1.0
    return x, m, r


def solve_fixed(cfg):
    ctx = base.build_fleet(cfg, base.resolve_run_options(cfg))
    x, m, r = schedule()
    for variables, values in ((ctx.x, x), (ctx.m_rep, m), (ctx.r_rep, r)):
        for key, variable in variables.items():
            variable.LB = variable.UB = float(values[key])
    for (i, l, k), variable in ctx.nb.items():
        variable.LB = variable.UB = 1.0 - m[i, l, k] - r[i, l, k]
    states = list(ctx.mu_var.values()) + list(ctx.z_var.values())
    for q in ("v", "R", "K"):
        states += list(ctx.state_vars[q].values())
    ctx.model.setObjective(gp.quicksum(states), gp.GRB.MINIMIZE)
    ctx.model.optimize()
    return ctx, base.extract_solution(ctx, cfg)


FAILURES = []


def check(label, ok):
    print(("ok   " if ok else "FAIL ") + label)
    if not ok:
        FAILURES.append(label)


def close(a, b):
    return np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), atol=1e-7, equal_nan=True)


def main():
    starts = {"mu": 0.2, "v": 0.004, "R": 0.01, "K": 1.0}
    for bound, second in rainflow.BOUND_QUANTITY.items():
        for law in ("ard1", "ardinf"):
            for encoding in ("bigm", "indicator"):
                tag = f"{bound} {law} {encoding}"
                ctx, out = solve_fixed(make_cfg(bound, law, encoding))
                check(f"{tag}: status optimal", out["status"] == "optimal")
                if out["status"] != "optimal":
                    continue
                tracked = ["mu"] + ([second[0]] if second else [])
                for q in ("mu", "v", "R", "K"):
                    key, gkey = RESULT_KEY[q]
                    if q not in tracked:
                        check(f"{tag}: {key} and {gkey} all NaN",
                              bool(np.isnan(out[key]).all() and np.isnan(out[gkey]).all()))
                        continue
                    states, anchors = EXPECTED[q][law]
                    check(f"{tag}: {key} train 0", close(out[key][0, 0], states))
                    check(f"{tag}: {key} train 1", close(out[key][1, 0], [starts[q]] * 4))
                    if law == "ard1":
                        check(f"{tag}: {gkey} train 0", close(out[gkey][0, 0], anchors))
                        check(f"{tag}: {gkey} train 1", close(out[gkey][1, 0], [starts[q]] * 4))
                    else:
                        check(f"{tag}: {gkey} all NaN", bool(np.isnan(out[gkey]).all()))
                check(f"{tag}: z train 0", close(out["z"][0, 0], Z[law]))
            # infeasible epsilon: ard1 infeasible, ardinf optimal (tangent form)
            _, out = solve_fixed(make_cfg(bound, law, eps_index=1))
            expected = "infeasible" if law == "ard1" else "optimal"
            check(f"{bound} {law} small epsilon: status {expected}", out["status"] == expected)
    # v_max_user (Cantelli): 0.015 is below the ard1 peak 0.0165 and above the ardinf peak 0.0135
    for law, expected in (("ard1", "infeasible"), ("ardinf", "optimal")):
        _, out = solve_fixed(make_cfg("cantelli", law, v_max_user=0.015))
        check(f"cantelli {law} v_max_user 0.015: status {expected}", out["status"] == expected)
    # variables and row names (emulator: getConstrs returns the names of the linear rows)
    for bound, second in rainflow.BOUND_QUANTITY.items():
        for law in ("ard1", "ardinf"):
            cfg = make_cfg(bound, law, v_max_user=0.05 if bound in ("cantelli", "bernstein") else NAN)
            ctx = base.build_fleet(cfg, base.resolve_run_options(cfg))
            names = set(ctx.model.getConstrs())
            tracked = ["mu"] + ([second[0]] if second else [])
            for q in ("v", "R", "K"):
                check(f"{bound} {law}: {q} variables only if tracked",
                      (len(ctx.state_vars[q]) == 8) == (q in tracked) and
                      (len(ctx.anchor_vars[q]) == (8 if q in tracked and law == "ard1" else 0)))
            loops = {f"loop_{q}_{i}_0" for q in tracked for i in (0, 1)}
            if law == "ard1":
                loops |= {f"loop_g{q}_{i}_0" for q in tracked for i in (0, 1)}
            check(f"{bound} {law}: loop rows", {n for n in names if n.startswith("loop_")} == loops)
            quadratic = bound in rainflow.QUADRATIC_BOUNDS
            check(f"{bound} {law}: side rows only at quadratic rows",
                  ("rel_side_0_0_0" in names) == quadratic and "rel_0_0_3" in names)
            check(f"{bound} {law}: v_max_user rows only at Ca, Be",
                  ("v_max_user_1_0_3" in names) == (bound in ("cantelli", "bernstein")))
    # two bounds in one fleet: component 0 Cantelli, component 1 Chernoff, both laws
    for law in ("ard1", "ardinf"):
        a, b = make_cfg("cantelli", law), make_cfg("chernoff", law)
        d = dict(vars(a))
        for key, va in vars(a).items():
            vb = getattr(b, key)
            if va is None and isinstance(vb, np.ndarray):
                d[key] = np.concatenate([np.full(vb.shape, NAN), vb], axis=1)
            elif isinstance(va, np.ndarray) and vb is None:
                d[key] = np.concatenate([va, np.full(va.shape, NAN)], axis=1)
            elif isinstance(va, np.ndarray) and va.ndim >= 2 and va.shape[:2] == (2, 1):
                d[key] = np.concatenate([va, vb], axis=1)
        d["L"], d["component_names"] = 2, ["Component 1", "Component 2"]
        cfg = types.SimpleNamespace(**d)
        ctx = base.build_fleet(cfg, base.resolve_run_options(cfg))
        x, m1, r1 = schedule()
        m, r = np.concatenate([m1, m1], axis=1), np.concatenate([r1, r1], axis=1)
        for variables, values in ((ctx.x, x), (ctx.m_rep, m), (ctx.r_rep, r)):
            for key, variable in variables.items():
                variable.LB = variable.UB = float(values[key])
        states = list(ctx.mu_var.values()) + list(ctx.z_var.values())
        for q in ("v", "R", "K"):
            states += list(ctx.state_vars[q].values())
        ctx.model.setObjective(gp.quicksum(states), gp.GRB.MINIMIZE)
        ctx.model.optimize()
        out = base.extract_solution(ctx, cfg)
        tag = f"mixed {law}"
        check(f"{tag}: optimal", out["status"] == "optimal")
        check(f"{tag}: v keys at l = 0 only, K keys at l = 1 only",
              {key[1] for key in ctx.state_vars["v"]} == {0} and {key[1] for key in ctx.state_vars["K"]} == {1})
        check(f"{tag}: mu both components", close(out["mu"][0, 0], EXPECTED["mu"][law][0])
              and close(out["mu"][0, 1], EXPECTED["mu"][law][0]))
        check(f"{tag}: v of component 0, K of component 1",
              close(out["v"][0, 0], EXPECTED["v"][law][0]) and close(out["K"][0, 1], EXPECTED["K"][law][0])
              and bool(np.isnan(out["v"][:, 1]).all()) and bool(np.isnan(out["K"][:, 0]).all()))
    # row constants
    rc = rainflow.row_constants
    check("row_constants markov", rc("markov", 1.0, 0.6) == {"q": "mu", "cap": 0.6})
    got = rc("bernstein", 1.0, 0.4, b=0.3)
    kappa = math.log(2.5)
    check("row_constants bernstein", got["q"] == "v" and close(
        [got["c2"], got["c1"], got["mu_side"]], [1 / (2 * kappa), -0.1, 1 - 0.1 * kappa]))
    got = rc("chernoff", 1.0, 0.5, s=4.0)
    check("row_constants chernoff", got["q"] == "K" and close(got["cap"], 4.0 + math.log(0.5)))
    try:
        rc("nope", 1.0, 0.5)
        check("row_constants unknown bound raises", False)
    except ValueError:
        check("row_constants unknown bound raises", True)
    print(f"failures: {len(FAILURES)}")


if __name__ == "__main__":
    main()
