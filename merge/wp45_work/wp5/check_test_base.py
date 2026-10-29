"""Local checks of tests/test_base.py (Task B), on a computer without Gurobi and pytest.

Usage: python check_test_base.py
1. ast parse; line endings; lines longer than 88 characters.
2. names_check.py on the test file.
3. Load the test module through a stub pytest and the stub gurobipy, with the real
   base.py (load_base.py); check constants, the fixture ``_cfg``, the expected
   arrays against hand_case.py, and T2 against the real FleetModel.increment.
4. List the ``base.<name>`` used by the file, with a flag for names outside section 2.
"""
import ast
import contextlib
import importlib.util
import io
import pathlib
import re
import subprocess
import sys
import types

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
TEST = pathlib.Path(
    r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\tests\test_base.py"
)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "stubs"))

SECTION_2 = {
    # context and registry
    "FleetModel", "CELL_BUILDERS", "get_cell_builder", "register_cell_builder", "dispatch_cell",
    # section 2.2 to 2.5
    "add_state_vars", "add_selector_vars", "add_pwl_vars", "mission_count", "day_increment",
    "add_selector_rows", "add_quantity_rows", "add_repair_cost_rows",
    "add_quadratic_reliability_rows", "add_upper_cap_rows", "phase_day", "cost_days",
    "repair_factor", "collapse_selector", "transitory_repair_cost", "peak_damage",
    "cost_report", "resolve_run_options", "build_context", "build_objective",
    "add_assignment_rows", "add_depot_rows", "add_damage_penalty_rows", "build_fleet",
    "extract_solution", "solve_mixed",
}


def step1():
    source = TEST.read_text(encoding="utf-8")
    ast.parse(source)
    data = TEST.read_bytes()
    print("ast.parse OK; lines", data.count(b"\n"), "; CRLF", data.count(b"\r\n"))
    long_lines = [
        (n, len(line)) for n, line in enumerate(source.splitlines(), 1) if len(line) > 88
    ]
    print("lines over 88 characters:", long_lines or "none")
    return source


def step2():
    result = subprocess.run(
        [sys.executable, "names_check.py", str(TEST)],
        cwd=HERE, capture_output=True, text=True, encoding="utf-8",
    )
    print(result.stdout.strip().splitlines()[0], result.stderr.strip()[:300])
    lines = result.stdout.strip().splitlines()
    for line in lines[1:]:
        if line.startswith("undefined names inside"):
            break
        print(line)


def load_test_module():
    from load_base import load

    base = load()
    stub_gurobipy = importlib.import_module("gurobipy")

    class _Mark:
        def parametrize(self, names, values, **kwargs):
            def decorate(function):
                function.__dict__.setdefault("_params", []).append((names, values))
                return function

            return decorate

    fake_pytest = types.ModuleType("pytest")
    fake_pytest.importorskip = lambda name: stub_gurobipy
    fake_pytest.mark = _Mark()
    fake_pytest.param = lambda *args, **kwargs: args
    fake_pytest.fixture = lambda function: function
    fake_pytest.raises = contextlib.nullcontext
    sys.modules["pytest"] = fake_pytest
    package = types.ModuleType("fleet_management")
    sub = types.ModuleType("fleet_management.degradation_model")
    sub.base = base
    package.degradation_model = sub
    sys.modules["fleet_management"] = package
    sys.modules["fleet_management.degradation_model"] = sub
    spec = importlib.util.spec_from_file_location("test_base_local", TEST)
    module = importlib.util.module_from_spec(spec)
    sys.modules["test_base_local"] = module
    spec.loader.exec_module(module)
    return base, module


def check(label, condition):
    print(("PASS " if condition else "FAIL ") + label)
    return bool(condition)


def step3(base, t):
    ok = True
    ok &= check("51 result keys", len(t._RESULT_KEYS) == 51)
    ok &= check("24 performance keys", len(t._PERFORMANCE_KEYS) == 24)
    ok &= check("14 solution keys, 11 cost keys", len(t._SOLUTION_KEYS) == 14 and len(t._COST_KEYS) == 11)
    ok &= check(
        "result keys = 22 header + 4 report keys + solution + cost keys",
        len(t._RESULT_KEYS - set(t._SOLUTION_KEYS) - set(t._COST_KEYS)) == 26,
    )
    s = t._SCHEDULE
    ok &= check("schedule shapes", s["x"].shape == (2, 2, 4) and s["m"].shape == (2, 1, 4) and s["r"].shape == (2, 1, 4))
    ok &= check("schedule ones: x 6, m 1, r 1", s["x"].sum() == 6 and s["m"].sum() == 1 and s["r"].sum() == 1)
    ok &= check("one mission flown per day", np.all(s["x"][:, 1, :].sum(axis=0) == 1))
    ok &= check("depot rows: m + r <= x0", np.all(s["m"][:, 0, :] + s["r"][:, 0, :] <= s["x"][:, 0, :]))

    cfg = t._cfg()
    ok &= check("sizes", (cfg.F, cfg.M, cfg.L, cfg.H, cfg.H1, cfg.H2, cfg.T) == (2, 1, 1, [2, 2], 2, 2, 4))
    ok &= check("selectors", cfg.model.shape == (2, 1) and cfg.model[0, 0] == "gaussian"
                and cfg.repair_model[0, 0] == "ard1" and cfg.bound_method[0, 0] == ""
                and cfg.reliability_impl[0, 0] == "tangent" and cfg.encoding[0, 0] == "bigm")
    ok &= check("cell values", all(np.asarray(getattr(cfg, n)).shape == (2, 1) for n in t._CELL_VALUES))
    ok &= check("defaults", cfg.tau[0, 0] == 1.0 and cfg.epsilon[0, 0] == 0.1 and cfg.tangent_ref[0, 0] == 0.5
                and cfg.rho[0, 0] == 0.5 and cfg.mu_0[0, 0] == 0.2 and cfg.v_0[0, 0] == 0.0
                and cfg.anchor_mu_0[0, 0] == 0.2 and cfg.anchor_v_0[0, 0] == 0.0
                and np.isnan(cfg.anchor_R_0).all() and np.isnan(cfg.anchor_K_0).all()
                and cfg.replacement_mu[0, 0] == 0.1 and cfg.replacement_v[0, 0] == 0.0
                and np.isnan(cfg.R_0).all() and np.isnan(cfg.cgf_new).all()
                and cfg.sigma[:, 0].tolist() == [0.1, 0.0]
                and cfg.C_R[0, 0] == 2.0 and cfg.C_rep[1, 0] == 3.0
                and cfg.C_M == 1.0 and cfg.C_D == 0.0 and cfg.penalty_type == "inf_norm"
                and cfg.pwl_points == 8 and cfg.allow_replacement is True and cfg.replacement_as_new is False
                and cfg.range_mu[0, 0] == 1.0 and abs(cfg.range_v[0, 0] - 0.6088746) < 1e-6
                and np.isnan(cfg.range_R).all() and np.isnan(cfg.range_K).all())
    ok &= check("profiles", cfg.mu.shape == (2, 1, 1, 2) and cfg.mu[0, 0, 0].tolist() == [0.3, 0.3]
                and cfg.mu[1].sum() == 0.0 and cfg.v[0, 0, 0].tolist() == [0.01, 0.01] and cfg.v[1].sum() == 0.0
                and cfg.support is None and cfg.cgf is None and cfg.gamma_beta is None
                and all(getattr(cfg, k + "_trans") is None for k in ("mu", "v", "support", "cgf", "gamma_beta")))
    ok &= check("options", cfg.options == dict(
        verbose=0, mip_gap=0.0, time_limit=60.0, fast=False, gurobi_params=None, objective_mode="total",
        evaluation_horizon=None, progress_interval_seconds=5.0, relaxation_warm_start=False,
        relaxation_time_limit=60.0, validation=None))
    inf = t._cfg(repair_model="ardinf")
    ok &= check("ardinf: starting anchors NaN", np.isnan(inf.anchor_mu_0).all() and np.isnan(inf.anchor_v_0).all()
                and (inf.repair_model == "ardinf").all())
    small = t._cfg(H=[1, 1], mu=np.full((2, 1, 1, 1), 0.9), allow_replacement=False,
                   options={"gurobi_params": {"DualReductions": 0}})
    ok &= check("H = [1, 1]", (small.H1, small.H2, small.T) == (1, 1, 2) and small.v.shape == (2, 1, 1, 1)
                and small.options["gurobi_params"] == {"DualReductions": 0} and small.options["mip_gap"] == 0.0
                and small.allow_replacement is False)
    ok &= check("array change broadcast", np.array_equal(t._cfg(C_R=np.array([[2.0], [0.0]])).C_R, [[2.0], [0.0]])
                and (t._cfg(rho=0.25).rho == 0.25).all() and t._cfg(v=None).v is None)
    ok &= check("INFEASIBLE constants", t._INFEASIBLE == (3, 4))
    return ok


def step3_expected(base, t):
    """Expected arrays of the test file against hand_case.py."""
    ok = True
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        import hand_case
    run = hand_case.run
    for law in ("ard1", "ardinf"):
        for kind in ("gaussian", "wiener"):
            rows = run(law, wiener=(kind == "wiener"))
            mu = [r[0] for r in rows]
            v = [r[1] for r in rows]
            g_mu = [r[2] for r in rows]
            g_v = [r[3] for r in rows]
            z = [r[4] for r in rows]
            ok &= check(f"mu {law}", np.allclose(t._TRAIN_0[law, "mu"], mu, atol=1e-6))
            ok &= check(f"z {law}", np.allclose(t._TRAIN_0[law, "z"], z, atol=1e-6))
            ok &= check(f"v {kind}", np.allclose(t._V[kind], v, atol=1e-6))
            if law == "ard1":
                ok &= check("g_mu ard1", np.allclose(t._TRAIN_0["ard1", "g_mu"], g_mu, atol=1e-6))
                if kind == "gaussian":
                    ok &= check("g_v ard1", np.allclose(t._TRAIN_0["ard1", "g_v"], g_v, atol=1e-6))
    return ok


def step3_t2(base, t):
    """T2 on the real FleetModel.increment with the profiles of ``_cfg``."""
    ok = True
    names = ["model", "x", "m_rep", "r_rep", "mu_var", "z_var", "u_var", "model_of", "repair_of", "bound_of",
             "impl_of", "encoding_of", "tau", "eps", "rho", "tangent_ref", "initial", "replacement",
             "anchor_start", "state_range", "opts"]
    kw = {n: None for n in names}
    for transitory, expected in ((None, [1, 2, 1, 2, 3]), ([10, 20], [10, 20, 1, 2, 3])):
        operating = np.zeros((2, 1, 1, 3))
        operating[0, 0, 0] = [1, 2, 3]
        mu_trans = None
        if transitory is not None:
            mu_trans = np.zeros((2, 1, 1, 2))
            mu_trans[0, 0, 0] = transitory
        cfg = t._cfg(H=[2, 3], v=None, mu=operating, mu_trans=mu_trans)
        profiles = {"mu": (cfg.mu, cfg.mu_trans)}
        ctx = base.FleetModel(F=cfg.F, H1=cfg.H1, H2=cfg.H2, M=cfg.M, L=cfg.L, T=cfg.T,
                              allow_replacement=cfg.allow_replacement, pwl_points=cfg.pwl_points,
                              profiles=profiles, **kw)
        got = [ctx.increment("mu", 0, 1, 0, k) for k in range(5)]
        ok &= check(f"T2 increments {got}", np.allclose(got, expected) and ctx.increment("mu", 0, 0, 0, 3) == 0.0)
    try:
        ctx.increment("v", 0, 1, 0, 0)
        ok &= check("T2 missing profile raises KeyError", False)
    except KeyError as error:
        ok &= check("T2 missing profile raises KeyError", "profile 'v' is not in the input" in str(error))
    return ok


def step3_t8(base, t):
    """T8: the feasibility flags of the test against a direct evaluation of the rows."""
    ok = True
    c2g = t._GAUSSIAN_C2
    ok &= check("kappa_G^2 = 1.6423744", abs(1 / c2g - 1.6423744) < 1e-6)
    ok &= check("Gaussian cap at 0.375 = 0.2378416", abs(t._cap(c2g, 0.0, 0.375) - 0.2378416) < 1e-6)
    ok &= check("Gaussian point v = 0.2354632", abs(0.99 * t._cap(c2g, 0.0, 0.375) - 0.2354632) < 1e-6)
    ok &= check("Gaussian cap of the tangent at 0.5, at 0.375 = 0.2283280",
                abs(c2g * 0.25 + (-2 * c2g * 0.5) * (0.375 - 0.5) - 0.2283280) < 1e-6)
    row = t._BERNSTEIN_ROW
    ok &= check("Bernstein c2 = 0.2171472, c1 = -0.1, side = 0.7697415",
                abs(row["c2"] - 0.2171472) < 1e-7 and abs(row["c1"] + 0.1) < 1e-12
                and abs(row["mu_side"] - 0.7697415) < 1e-7)
    ok &= check("Bernstein range_v = 0.1171472", abs(t._BERNSTEIN_RANGE_V - 0.1171472) < 1e-7)
    ok &= check("Bernstein cap at 0.3 = 0.0364021",
                abs(t._cap(row["c2"], row["c1"], 0.3) - 0.0364021) < 1e-7)

    def row_holds(impl, ref, points, mu, v, c2, c1, side, range_v, tau=1.0):
        f = lambda m: c2 * (tau - m) ** 2 + c1 * (tau - m)
        df = lambda m: -2.0 * c2 * (tau - m) - c1
        if mu > side or v > range_v:
            return False
        if impl == "exact":
            return v <= f(mu) + 1e-9
        if impl == "tangent":
            r = ref * tau
            return v <= f(r) + df(r) * (mu - r) + 1e-9
        if points == 1:
            return v <= f(tau / 2) + df(tau / 2) * (mu - tau / 2) + 1e-9
        for p in range(points):
            lo, hi, mid = p * tau / points, (p + 1) * tau / points, (p + 0.5) * tau / points
            if lo <= mu <= hi and v <= f(mid) + df(mid) * (mu - mid) + 1e-9:
                return True
        return False

    names, values = t.test_gaussian_reliability_forms._params[0]
    v = 0.99 * t._cap(c2g, 0.0, 0.375)
    for impl, ref, points, feasible in values:
        got = row_holds(impl, ref, points, 0.375, v, c2g, 0.0, 1.0, t._CELL_VALUES["range_v"])
        ok &= check(f"T8 Gaussian {impl} ref {ref} points {points}: feasible {feasible}", got == feasible)
    names, values = t.test_bernstein_reliability_row._params[0]
    for mu, scale, feasible in values:
        v = scale * t._cap(row["c2"], row["c1"], 0.3)
        got = row_holds("exact", None, None, mu, v, row["c2"], row["c1"], row["mu_side"], t._BERNSTEIN_RANGE_V)
        ok &= check(f"T8 Bernstein mu {mu} scale {scale}: feasible {feasible}", got == feasible)
    return ok


def step4(source):
    names = sorted(set(re.findall(r"\bbase\.([A-Za-z_][A-Za-z0-9_]*)", source)))
    print("base names used:", names)
    outside = [n for n in names if n not in SECTION_2]
    print("outside section 2:", outside or "none")


if __name__ == "__main__":
    source = step1()
    step2()
    base, module = load_test_module()
    ok = step3(base, module)
    if hasattr(module, "_TRAIN_0"):
        ok &= step3_expected(base, module)
    ok &= step3_t2(base, module)
    if hasattr(module, "_BERNSTEIN_ROW"):
        ok &= step3_t8(base, module)
    step4(source)
    print("ALL PASS" if ok else "SOME FAIL")
