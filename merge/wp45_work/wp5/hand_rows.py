"""Row check of base.py on the hand-worked case of the plan (section 4), without Gurobi.

Usage: python hand_rows.py
Loads base.py through the linear emulator in ``fakegp`` (scipy LP with every binary
fixed), builds the two test cells of the plan (Gaussian-like and Wiener-like
variance), fixes the schedule, minimises the states and compares the trajectory
with the expected arrays of section 4.4 (T4, T5, T6; z of T4 once A5 exists).
Only the tangent form of the reliability row is emulated (no quadratic rows).
"""
import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
from scipy.stats import norm

HERE = pathlib.Path(__file__).resolve().parent
BASE = pathlib.Path(
    r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge"
    r"\src\fleet_management\degradation_model\base.py"
)
sys.path.insert(0, str(HERE / "fakegp"))
spec = importlib.util.spec_from_file_location("wp5_base_fake", BASE)
base = importlib.util.module_from_spec(spec)
sys.modules["wp5_base_fake"] = base
spec.loader.exec_module(base)

NAN = float("nan")
OPTIONS = dict(verbose=0, mip_gap=0.0, time_limit=60.0, fast=False, gurobi_params=None,
               objective_mode="total", evaluation_horizon=None, progress_interval_seconds=5.0,
               relaxation_warm_start=False, relaxation_time_limit=60.0, validation=None)


def make_cfg(**changes):
    F, L = 2, 1

    def cell(value):
        return np.full((F, L), value, dtype=float)

    def sel(value):
        return np.full((F, L), value, dtype=object)

    mu = np.zeros((F, L, 1, 2))
    mu[0] = 0.3
    v = np.zeros((F, L, 1, 2))
    v[0] = 0.01
    d = dict(
        F=F, M=1, L=L, H=[2, 2], H1=2, H2=2, T=4, component_names=["Component 1"],
        model=sel("gaussian"), repair_model=sel("ard1"), bound_method=sel(""),
        reliability_impl=sel("tangent"), encoding=sel("bigm"),
        tau=cell(1.0), epsilon=cell(0.1), tangent_ref=cell(0.5), rho=cell(0.5),
        mu_0=cell(0.2), v_0=cell(0.0), anchor_mu_0=cell(0.2), anchor_v_0=cell(0.0),
        anchor_R_0=cell(NAN), anchor_K_0=cell(NAN), replacement_mu=cell(0.1),
        replacement_v=cell(0.0), R_0=cell(NAN), cgf_0=cell(NAN), R_new=cell(NAN),
        cgf_new=cell(NAN), sigma=np.array([[0.1], [0.0]]),
        C_M=1.0, C_D=0.0, penalty_type="inf_norm", C_R=cell(2.0), C_rep=cell(3.0),
        pwl_points=8, allow_replacement=True, replacement_as_new=False,
        mu=mu, v=v, support=None, cgf=None, gamma_beta=None,
        range_mu=cell(1.0), range_v=cell(1 / norm.ppf(0.9) ** 2),
        range_R=cell(NAN), range_K=cell(NAN), options=dict(OPTIONS),
    )
    for key, value in changes.items():
        if key in ("repair_model", "encoding", "reliability_impl", "model") and isinstance(value, str):
            value = sel(value)
        d[key] = value
    cfg = SimpleNamespace(**d)
    if str(cfg.repair_model[0, 0]) == "ardinf":
        cfg.anchor_mu_0 = cell(NAN)
        cfg.anchor_v_0 = cell(NAN)
    return cfg


SCHEDULE_X = [(0, 1, 0), (0, 0, 1), (0, 1, 2), (0, 0, 3), (1, 1, 1), (1, 1, 3)]


def fix_schedule(ctx, repair_day=1, replace_day=3):
    for key, var in ctx.x.items():
        var.LB = var.UB = 1.0 if key in SCHEDULE_X else 0.0
    for (i, l, k), var in ctx.m_rep.items():
        var.LB = var.UB = 1.0 if (i, l, k) == (0, 0, repair_day) else 0.0
    if ctx.r_rep is not None:
        for (i, l, k), var in ctx.r_rep.items():
            var.LB = var.UB = 1.0 if (i, l, k) == (0, 0, replace_day) else 0.0
    for (i, l, k), var in ctx.nb.items():
        used = (ctx.m_rep[i, l, k].LB + (ctx.r_rep[i, l, k].LB if ctx.r_rep is not None else 0.0))
        var.LB = var.UB = 1.0 - used


class MeanVarianceBuilder:
    def prepare(self, ctx, cfg, cells, opts):
        base.add_state_vars(ctx, "mu", cells)
        base.add_state_vars(ctx, "v", cells)
        base.add_selector_vars(ctx, cells)
        base.add_pwl_vars(ctx, cells)

    def add_cell(self, ctx, i, l):
        T = ctx.T
        tau, eps = float(ctx.tau[i, l]), float(ctx.eps[i, l])
        base.add_selector_rows(ctx, i, l)
        base.add_quantity_rows(ctx, i, l, "mu", [base.day_increment(ctx, i, l, k, "mu") for k in range(T)])
        base.add_quantity_rows(ctx, i, l, "v", [base.day_increment(ctx, i, l, k, "v") for k in range(T)])
        if hasattr(base, "add_quadratic_reliability_rows"):
            kappa2 = norm.ppf(1 - eps) ** 2
            base.add_quadratic_reliability_rows(ctx, i, l, "v", c2=1 / kappa2, c1=0.0, mu_side=tau)
        if hasattr(base, "add_repair_cost_rows"):
            base.add_repair_cost_rows(ctx, i, l)


class WienerLikeBuilder(MeanVarianceBuilder):
    def prepare(self, ctx, cfg, cells, opts):
        base.add_state_vars(ctx, "mu", cells)
        base.add_state_vars(ctx, "v", cells, anchored=False)
        base.add_selector_vars(ctx, cells)
        base.add_pwl_vars(ctx, cells)
        ctx.extras["sigma"] = np.asarray(cfg.sigma, dtype=float)

    def add_cell(self, ctx, i, l):
        T = ctx.T
        tau, eps = float(ctx.tau[i, l]), float(ctx.eps[i, l])
        sigma = float(ctx.extras["sigma"][i, l])
        base.add_selector_rows(ctx, i, l)
        base.add_quantity_rows(ctx, i, l, "mu", [base.day_increment(ctx, i, l, k, "mu") for k in range(T)])
        base.add_quantity_rows(ctx, i, l, "v", [sigma ** 2 * base.mission_count(ctx, i, k) for k in range(T)],
                               repair_keeps_state=True)
        if hasattr(base, "add_quadratic_reliability_rows"):
            kappa2 = norm.ppf(1 - eps) ** 2
            base.add_quadratic_reliability_rows(ctx, i, l, "v", c2=1 / kappa2, c1=0.0, mu_side=tau)
        if hasattr(base, "add_repair_cost_rows"):
            base.add_repair_cost_rows(ctx, i, l)


def build(cfg, builder, objective_states=True):
    """build_context + prepare + shared rows + add_cell, by hand (build_fleet may not exist yet)."""
    opts = base.resolve_run_options(cfg)
    ctx = base.build_context(cfg, opts)
    cells = ctx.all_cells()
    builder.prepare(ctx, cfg, cells, opts)
    if hasattr(base, "add_assignment_rows"):
        base.add_assignment_rows(ctx)
    if hasattr(base, "add_depot_rows"):
        base.add_depot_rows(ctx)
    for i, l in cells:
        builder.add_cell(ctx, i, l)
    return ctx


def min_states(ctx):
    expr = base.gp.quicksum(ctx.mu_var.values())
    expr = expr + base.gp.quicksum(ctx.state_vars["v"].values())
    expr = expr + base.gp.quicksum(ctx.z_var.values())
    ctx.model.setObjective(expr, base.GRB.MINIMIZE)


def read(ctx, table, i=0):
    return np.array([table[i, 0, k].X for k in range(ctx.T)])


EXPECT = {
    "ard1": dict(mu=[0.5, 0.35, 0.65, 0.1], vg=[0.01, 0.0025, 0.0125, 0.0], vw=[0.01, 0.01, 0.02, 0.0],
                 g_mu=[0.2, 0.35, 0.35, 0.1], g_v=[0.0, 0.0025, 0.0025, 0.0], z=[0.0, 0.15, 0.0, 0.0]),
    "ardinf": dict(mu=[0.5, 0.25, 0.55, 0.1], vg=[0.01, 0.0025, 0.0125, 0.0], vw=[0.01, 0.01, 0.02, 0.0],
                   g_mu=None, g_v=None, z=[0.0, 0.25, 0.0, 0.0]),
}

failures = 0


def check(label, got, want, tol=1e-6):
    global failures
    try:
        ok = np.allclose(np.asarray(got, dtype=float), np.asarray(want, dtype=float), atol=tol, equal_nan=True)
    except (TypeError, ValueError):
        ok = got == want
    failures += 0 if ok else 1
    print(("PASS " if ok else "FAIL ") + label + ("" if ok else f"  got {got} want {want}"))


def run_dynamics():
    for encoding in ("bigm", "indicator"):
        for repair in ("ard1", "ardinf"):
            for name, builder in (("gaussian-like", MeanVarianceBuilder()), ("wiener-like", WienerLikeBuilder())):
                cfg = make_cfg(encoding=encoding, repair_model=repair)
                ctx = build(cfg, builder)
                fix_schedule(ctx)
                min_states(ctx)
                ctx.model.optimize()
                tag = f"{name} {encoding} {repair}"
                if ctx.model.SolCount == 0:
                    check(tag + " solved", 0, 1)
                    continue
                e = EXPECT[repair]
                check(tag + " mu", read(ctx, ctx.mu_var), e["mu"])
                check(tag + " v", read(ctx, ctx.state_vars["v"]), e["vg"] if name == "gaussian-like" else e["vw"])
                check(tag + " mu train 1", read(ctx, ctx.mu_var, 1), [0.2] * 4)
                if repair == "ard1":
                    check(tag + " g_mu", read(ctx, ctx.anchor_vars["mu"]), e["g_mu"])
                    if name == "gaussian-like":
                        check(tag + " g_v", read(ctx, ctx.anchor_vars["v"]), e["g_v"])
                    else:
                        check(tag + " no g_v", len(ctx.anchor_vars["v"]), 0)
                else:
                    check(tag + " no anchors", len(ctx.anchor_vars["mu"]) + len(ctx.anchor_vars["v"]), 0)
                if len(ctx.z_var):
                    check(tag + " z", read(ctx, ctx.z_var), e["z"])


class RowOnlyBuilder:
    def __init__(self, c2, c1, mu_side):
        self.args = dict(c2=c2, c1=c1, mu_side=mu_side)

    def prepare(self, ctx, cfg, cells, opts):
        base.add_state_vars(ctx, "v", cells, anchored=False)
        base.add_pwl_vars(ctx, cells)

    def add_cell(self, ctx, i, l):
        base.add_quadratic_reliability_rows(ctx, i, l, "v", **self.args)


def run_reliability():
    """T8 of the plan for the forms the emulator can solve (tangent, pwl)."""
    kappa2 = norm.ppf(0.9) ** 2
    f = lambda m: (1.0 - m) ** 2 / kappa2
    v_a = 0.99 * f(0.375)

    def solve(form, ref=0.5, points=8, mu=0.375, v=v_a, mu_side=1.0, segment=None):
        cfg = make_cfg(H=[1, 1], H1=1, H2=1, T=2, reliability_impl=form,
                       tangent_ref=np.full((2, 1), ref), pwl_points=points)
        ctx = build(cfg, RowOnlyBuilder(1 / kappa2, 0.0, mu_side))
        ctx.mu_var[0, 0, 0].LB = ctx.mu_var[0, 0, 0].UB = mu
        ctx.state_vars["v"][0, 0, 0].LB = ctx.state_vars["v"][0, 0, 0].UB = v
        for key, var in ctx.x.items():
            var.LB = var.UB = 1.0 if key in ((0, 1, 0), (0, 1, 1)) else 0.0
        for var in list(ctx.m_rep.values()) + list(ctx.r_rep.values()):
            var.LB = var.UB = 0.0
        for (i, l, k, p), w in ctx.pwl_w.items():
            w.LB = w.UB = 1.0 if p == (segment if (i, k) == (0, 0) else 0) else 0.0
        ctx.model.setObjective(base.gp.LinExpr(), base.GRB.MINIMIZE)
        ctx.model.optimize()
        return ctx.model.SolCount > 0

    check("T8 tangent ref 0.375 optimal", solve("tangent", ref=0.375), True)
    check("T8 tangent ref 0.5 infeasible", solve("tangent", ref=0.5), False)
    check("T8 pwl P=4 optimal (w on segment 1)", solve("pwl", points=4, segment=1), True)
    check("T8 pwl P=4, v above the cap, infeasible", solve("pwl", points=4, segment=1, v=1.01 * f(0.375)), False)
    check("T8 pwl P=4, w on a wrong segment, infeasible", solve("pwl", points=4, segment=0), False)
    check("T8 pwl P=1 infeasible", solve("pwl", points=1), False)
    check("T8 side row: mu 0.65 > side 0.6 infeasible (tangent row alone allows it)", solve("tangent", ref=0.375, mu=0.65, v=0.0, mu_side=0.6), False)
    check("T8 side row: mu 0.55 <= side 0.6 optimal", solve("tangent", ref=0.375, mu=0.55, v=0.0, mu_side=0.6), True)
    check("T8 tangent alone: mu 0.7 infeasible (tangent at 0.375 crosses 0 at 0.6875)", solve("tangent", ref=0.375, mu=0.7, v=0.0), False)
    try:
        solve("bogus")
        check("T8 unknown form raises ValueError", 0, 1)
    except ValueError as error:
        check("T8 unknown form raises ValueError", 1, 1 if "unknown reliability_impl 'bogus'" in str(error) else 0)


def row_names(ctx):
    return [name for _, name in ctx.model.rows]


def run_variables():
    """T3 of the plan: variables and rows exist only where the spec builds them."""
    def built(**changes):
        cfg = make_cfg(**changes)
        return build(cfg, MeanVarianceBuilder())

    ctx = built(C_D=0.0)
    check("T3 C_D = 0: no u, no u rows", (ctx.u_var is None, any(n.startswith("u_") for n in row_names(ctx))), (True, False))
    ctx = built(C_D=0.5)
    base.add_damage_penalty_rows(ctx, "inf_norm")
    check("T3 C_D = 0.5: u and 8 rows u_*", (ctx.u_var is not None, sum(n.startswith("u_") for n in row_names(ctx))), (True, 8))
    ctx = built(C_R=np.array([[2.0], [0.0]]))
    check("T3 z keys, C_R = [[2],[0]]", sorted(ctx.z_var.keys()), [(0, 0, k) for k in range(4)] if False else sorted((0, 0, k) for k in range(4)))
    ctx = built(options=dict(OPTIONS, objective_mode="operating_average"), C_R=np.array([[2.0], [0.0]]))
    check("T3 z keys, operating_average", sorted(ctx.z_var.keys()), [(0, 0, 2), (0, 0, 3)])
    check("T3 operating_average: no z row on k < 2", [n for n in row_names(ctx) if n.startswith("z_")], ["z_0_0_2", "z_0_0_3"])
    ctx = built(allow_replacement=False)
    check("T3 no replacement: r_rep None, depot row, no repl rows",
          (ctx.r_rep is None, "depot_0_0_0" in row_names(ctx), any("repl" in n for n in row_names(ctx))), (True, True, False))
    check("T3 bigm: nb empty", len(built(encoding="bigm").nb), 0)
    check("T3 indicator: nb 8", len(built(encoding="indicator").nb), 8)
    ctx = built(reliability_impl="pwl", pwl_points=4)
    check("T3 pwl P=4: 32 w", len(ctx.pwl_w), 32)
    check("T3 pwl P=1: no w", len(built(reliability_impl="pwl", pwl_points=1).pwl_w), 0)
    check("T3 ardinf: no anchors", len(built(repair_model="ardinf").anchor_vars["mu"]), 0)
    check("T3 ard1: 8 anchors", len(built().anchor_vars["mu"]), 8)
    ctx = built()
    check("T3 bigm no selector rows", any(n.startswith("sel_") for n in row_names(ctx)), False)
    ctx = built(encoding="indicator")
    check("T3 indicator selector rows 8", sum(n.startswith("sel_") for n in row_names(ctx)), 8)
    check("T3 shared variables: no idle, no nb at bigm", hasattr(ctx, "idle"), False)
    ctx = built(allow_replacement=False, encoding="indicator")
    check("T3 no replacement, indicator: no repl indicators", any("repl" in name for *_, name in ctx.model.indicators), False)


def run_objective():
    """T9 of the plan, for the modes the emulator can solve (inf_norm); hand-computed values."""
    cases = [
        ("total", None, 5.3 + 0.5 * 0.65),
        ("operating_average", None, 2.0 + 0.5 * 0.65),
        ("evaluation_total", 6, 1.3 + 2 * 4.0 + 0.5 * 0.65),
    ]
    for mode, E, want in cases:
        cfg = make_cfg(C_D=0.5, options=dict(OPTIONS, objective_mode=mode, evaluation_horizon=E))
        ctx = build(cfg, MeanVarianceBuilder())
        base.add_damage_penalty_rows(ctx, "inf_norm")
        base.build_objective(ctx, cfg, cfg.options)
        fix_schedule(ctx)
        ctx.model.optimize()
        check(f"T9 objective {mode}", ctx.model.ObjVal if ctx.model.SolCount else None, want)
    cfg = make_cfg(C_D=0.0, allow_replacement=False, options=dict(OPTIONS, objective_mode="total"))
    ctx = build(cfg, MeanVarianceBuilder())
    base.build_objective(ctx, cfg, cfg.options)
    check("T9 no u term when C_D = 0 (objective has no u)", all(v.VarName != "u" for v in ctx.model.objective.terms), True)
    try:
        base.build_objective(ctx, cfg, dict(OPTIONS, objective_mode="bogus"))
        check("T9 unknown mode raises ValueError", 0, 1)
    except ValueError:
        check("T9 unknown mode raises ValueError", 1, 1)


def run_assembly():
    """T1 of the plan and the building order of ``build_fleet`` (A7)."""
    try:
        base.get_cell_builder("no_such_model")
        check("T1 unknown builder raises NotImplementedError naming gamma", 0, 1)
    except NotImplementedError as error:
        check("T1 unknown builder raises NotImplementedError naming gamma", "gamma" in str(error), True)

    real_import = base.importlib
    calls = []

    def fake_import(module):
        calls.append(module)
        raise ModuleNotFoundError(f"No module named {module!r}", name=module if "other" not in module else "elsewhere")

    base.importlib = SimpleNamespace(import_module=fake_import)
    try:
        base._load_builders(["gamma"])
        check("A7 registered model: module not imported", calls, [])
        base._load_builders(["nomodel"])
        check("A7 module named after the model is imported, its absence is swallowed",
              calls, ["fleet_management.degradation_model.nomodel"])
        base.importlib = SimpleNamespace(import_module=lambda module: (_ for _ in ()).throw(
            ModuleNotFoundError("No module named 'scipy_x'", name="scipy_x")))
        try:
            base._load_builders(["other"])
            check("A7 a missing dependency inside the module is re-raised", 0, 1)
        except ModuleNotFoundError as error:
            check("A7 a missing dependency inside the module is re-raised", error.name, "scipy_x")
    finally:
        base.importlib = real_import

    cfg = make_cfg(model=np.full((2, 1), "no_such_model", dtype=object), C_D=0.5)
    real_model = base.gp.Model

    def no_model(*args, **kwargs):
        raise AssertionError("a Gurobi model was created")

    base.gp.Model = no_model
    base.importlib = SimpleNamespace(import_module=fake_import)
    try:
        base.build_fleet(cfg, cfg.options)
        check("A7 unknown model fails before any Gurobi object", 0, 1)
    except NotImplementedError:
        check("A7 unknown model fails before any Gurobi object", 1, 1)
    except AssertionError:
        check("A7 unknown model fails before any Gurobi object", 0, 1)
    finally:
        base.gp.Model = real_model
        base.importlib = real_import

    base.CELL_BUILDERS["gaussian"] = MeanVarianceBuilder()
    try:
        cfg = make_cfg(C_D=0.5)
        ctx = base.build_fleet(cfg, cfg.options)
        names = row_names(ctx)
        check("A7 build_fleet rows: assign 8, demand 4, depot 8, u 8, loop_mu 2, loop_gmu 2",
              (sum(n.startswith("assign_") for n in names), sum(n.startswith("demand_") for n in names),
               sum(n.startswith("depot_") for n in names), sum(n.startswith("u_") for n in names),
               sum(n.startswith("loop_mu_") for n in names), sum(n.startswith("loop_gmu_") for n in names)),
              (8, 4, 8, 8, 2, 2))
        check("A7 names are unique", len(names), len(set(names)))
        var_names = [v.VarName for v in ctx.model.vars]
        check("A7 variable names are unique", len(var_names), len(set(var_names)))
        fix_schedule(ctx)
        ctx.model.optimize()
        check("A7 build_fleet objective at the hand schedule (total, C_D 0.5)", ctx.model.ObjVal, 5.3 + 0.5 * 0.65)
    finally:
        del base.CELL_BUILDERS["gaussian"]


if __name__ == "__main__":
    run_dynamics()
    run_reliability()
    run_variables()
    run_objective()
    run_assembly()
    print("failures:", failures)
