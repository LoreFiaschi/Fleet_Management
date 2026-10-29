"""T4, T5, T7, T9, T11 of the plan through build_fleet and extract_solution (step A9).

Usage: python hand_rows_extract.py
Uses the emulator and the helpers of hand_rows.py (the schedule of the hand case is fixed;
the LP of the emulator gives the states). Prints PASS/FAIL lines and `failures: N`.
"""
import numpy as np

import hand_rows as h
from hand_rows import OPTIONS, EXPECT, check, make_cfg, fix_schedule, min_states, base

KEYS_WITHOUT_REPORTS = {
    "status", "objective", "mip_gap", "bound", "F", "M", "L", "H1", "H2", "T", "H", "model",
    "component_names", "tau", "mu_0", "v_0", "bound_method", "repair_model", "reliability_impl",
    "encoding", "objective_mode", "evaluation_horizon",
    "x", "m", "r", "idle", "mu", "v", "z", "R", "K", "g_mu", "g_v", "g_R", "g_K", "u",
    "step_costs", "J_initialization", "J_op", "J_op_average", "peak_damage", "damage_penalty",
    "operating_objective", "J_total", "projected_evaluation_cost", "evaluation_objective",
    "component_costs",
}


def fleet(cfg, builder=None):
    base.CELL_BUILDERS["gaussian"] = builder or h.MeanVarianceBuilder()
    try:
        return base.build_fleet(cfg, cfg.options)
    finally:
        del base.CELL_BUILDERS["gaussian"]


def solve(cfg, builder=None, states=True, **fix):
    ctx = fleet(cfg, builder)
    fix_schedule(ctx, **fix)
    if states:
        min_states(ctx)
    ctx.model.optimize()
    return ctx, base.extract_solution(ctx, cfg)


def run_t4_t5():
    for encoding in ("bigm", "indicator"):
        for repair in ("ard1", "ardinf"):
            for name, builder in (("gaussian-like", h.MeanVarianceBuilder()), ("wiener-like", h.WienerLikeBuilder())):
                cfg = make_cfg(encoding=encoding, repair_model=repair)
                ctx, out = solve(cfg, builder)
                tag = f"extract {name} {encoding} {repair}"
                e = EXPECT[repair]
                nan4 = [np.nan] * 4
                check(tag + " status", out["status"], "optimal")
                check(tag + " mu", out["mu"][0, 0], e["mu"])
                check(tag + " v", out["v"][0, 0], e["vg"] if name == "gaussian-like" else e["vw"])
                check(tag + " g_mu", out["g_mu"][0, 0], e["g_mu"] or nan4)
                if name == "gaussian-like":
                    check(tag + " g_v", out["g_v"][0, 0], e["g_v"] or nan4)
                else:
                    check(tag + " g_v all NaN", bool(np.isnan(out["g_v"]).all()), True)
                check(tag + " z", out["z"][0, 0], e["z"])
                check(tag + " train 1 mu", out["mu"][1, 0], [0.2] * 4)
                check(tag + " train 1 z", out["z"][1, 0], [0.0] * 4)
                check(tag + " R, K, g_R, g_K all NaN", all(bool(np.isnan(out[k]).all()) for k in ("R", "K", "g_R", "g_K")), True)
                check(tag + " idle all zero", bool(np.allclose(out["idle"], 0.0)), True)


def run_keys_and_costs():
    cfg = make_cfg(C_D=0.5)
    ctx, out = solve(cfg, states=False)
    check("A9 keys of the result without the reports (47)", set(out), KEYS_WITHOUT_REPORTS)
    check("A9 key order starts with the quality keys", list(out)[:4], ["status", "objective", "mip_gap", "bound"])
    check("A9 selectors", (out["model"], out["repair_model"], out["encoding"], out["reliability_impl"], out["bound_method"]),
          ([["gaussian"], ["gaussian"]], "ard1", "bigm", "tangent", None))
    check("A9 sizes", (out["F"], out["M"], out["L"], out["H1"], out["H2"], out["T"], out["H"],
                       out["objective_mode"], out["evaluation_horizon"]), (2, 1, 1, 2, 2, 4, [2, 2], "total", None))
    check("A9 tau, mu_0, v_0 are float arrays",
          all(isinstance(out[k], np.ndarray) and out[k].dtype == float for k in ("mu_0", "tau", "v_0")), True)
    check("A9 u", out["u"], 0.65)
    check("A9 objective equals J_total", out["objective"], out["J_total"])
    check("A9 step_costs", out["step_costs"], [0.0, 1.3, 0.0, 4.0])
    check("A9 no evaluation horizon: projected and evaluation objective None",
          (out["projected_evaluation_cost"], out["evaluation_objective"]), (None, None))
    check("A9 component_costs", (out["component_costs"]["C_M"], out["component_costs"]["C_D"], out["component_costs"]["C_R"].shape),
          (1.0, 0.5, (2, 1)))
    check("A9 float scalars", all(type(out[k]) is float for k in ("objective", "mip_gap", "bound", "u", "J_total", "J_op")), True)
    for mode, E, key in (("total", None, "J_total"), ("operating_average", None, "operating_objective"),
                         ("evaluation_total", 6, "evaluation_objective")):
        cfg = make_cfg(C_D=0.5, options=dict(OPTIONS, objective_mode=mode, evaluation_horizon=E))
        ctx, out = solve(cfg, states=False)
        check(f"T9 objective equals the report key {key}", abs(out["objective"] - out[key]) <= 1e-6 * max(1, abs(out[key])), True)
    cfg = make_cfg(C_D=0.5, options=dict(OPTIONS, objective_mode="total", evaluation_horizon=6))
    ctx, out = solve(cfg, states=False)
    check("A9 mode total with E = 6: P_E 9.3, evaluation objective 9.625",
          (out["projected_evaluation_cost"], out["evaluation_objective"]), (9.3, 9.3 + 0.5 * 0.65))


def run_t7():
    options = dict(OPTIONS, objective_mode="operating_average")
    cfg = make_cfg(options=options, C_R=np.array([[2.0], [0.0]]))
    ctx, out = solve(cfg)
    check("T7 z_var has no key with k < 2", all(k >= 2 for (_, _, k) in ctx.z_var), True)
    check("T7 z train 0 = [0, 0.15, 0, 0]", out["z"][0, 0], [0.0, 0.15, 0.0, 0.0])
    check("T7 z train 1 all NaN", bool(np.isnan(out["z"][1, 0]).all()), True)
    cfg = make_cfg(options=options, C_R=np.array([[2.0], [0.0]]), repair_model="ardinf")
    ctx, out = solve(cfg)
    check("T7 ardinf z train 0 = [0, 0.25, 0, 0]", out["z"][0, 0], [0.0, 0.25, 0.0, 0.0])
    check("T7 ardinf z train 1 all NaN", bool(np.isnan(out["z"][1, 0]).all()), True)


def run_no_replacement_and_no_solution():
    cfg = make_cfg(allow_replacement=False)
    ctx = fleet(cfg)
    for key, var in ctx.x.items():                       # train 1 flies every day, train 0 repairs on day 1
        var.LB = var.UB = 1.0 if key[0] == 1 and key[1] == 1 else 0.0
    for k in (1, 2):                                      # train 0 in the depot on days 1 and 2
        ctx.x[0, 0, k].LB = ctx.x[0, 0, k].UB = 1.0
    for key, var in ctx.m_rep.items():
        var.LB = var.UB = 1.0 if key == (0, 0, 1) else 0.0
    min_states(ctx)
    ctx.model.optimize()
    out = base.extract_solution(ctx, cfg)
    check("A9 no replacement: r all zeros, status optimal", (out["status"], float(np.abs(out["r"]).sum())), ("optimal", 0.0))
    check("A9 no replacement: idle = depot - m - r", out["idle"][0, 0], [0.0, 0.0, 1.0, 0.0])
    mu = np.full((2, 1, 1, 2), 0.9)
    cfg = make_cfg(mu=mu, allow_replacement=False, C_D=0.5)
    ctx, out = solve(cfg, states=False, replace_day=None)
    check("T11 no solution: status and bound", (out["status"], out["bound"]), ("infeasible", None))
    check("T11 objective, mip_gap, every solution and cost key are None",
          all(out[k] is None for k in ("objective", "mip_gap") + base._SOLUTION_KEYS + base._COST_KEYS), True)
    check("T11 the other keys are present", set(out), KEYS_WITHOUT_REPORTS)
    check("T11 objective_mode, model present", (out["objective_mode"], out["model"]), ("total", [["gaussian"], ["gaussian"]]))


if __name__ == "__main__":
    run_t4_t5()
    run_keys_and_costs()
    run_t7()
    run_no_replacement_and_no_solution()
    print("failures:", h.failures)
