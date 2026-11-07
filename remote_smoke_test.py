#!/usr/bin/env python3
"""
Remote smoke test of the package fleet_management (spec/spec.tex, merged version).

It runs the pytest suite of ``tests/`` and then 12 scenarios with REAL Gurobi solves,
so run it on a machine with a Gurobi licence (on Euler: ``sbatch euler/run_smoke.sh``).
The inputs of the scenarios are in ``examples/smoke/``.

Scenarios (each solved, checked with ``validate``, and plotted):
  1  five_models         G, IG, W, Ga and RF cells; repair laws ardinf and ard1; H = [2, 4]
  2  rainflow_bounds     one rainflow component per tail bound (Mk, Ca, Ho, Be, Ch)
  3  forms               exact, tangent and pwl forms; the same fleet with encoding indicator
  4  anchors             ard1 cells with starting anchors below the initial values
  5  replacement         replacement states; replacement_as_new; allow_replacement false
  6  caps                IG and Ga cells at their mean caps (caps from the package's functions)
  7  costs               the three objective modes and both penalties; cost formulas recomputed
  8  two_phase           a transitory profile mu_trans, H = [3, 4]
  9  sweep               sweep_operating_horizons over H2 = 2, 3, 4 with warm starts, and its plot
  10 greedy_start        build_greedy_warm_start on scenario 1, then solve with that source
  11 gamma_validators    validate_gamma_schedule in the modes both and stochastic
  12 mixed_example       the YAML example of the spec

Usage
-----
    python remote_smoke_test.py [--no-pytest] [--dry-run] [--out-dir DIR]

  --no-pytest   skip the pytest suite
  --dry-run     load and check every scenario input with ``load_config``, then exit (no solve,
                no file written)
  --out-dir     output folder (default: ./smoke_test_output)

Everything is written under the output folder:
  smoke_test.log      timestamped log of every step
  summary.json        pass or fail and elapsed seconds per scenario, and the pytest result
  pytest_output.txt   output of ``python -m pytest tests -v``
  plots/*.png         one plot per solve
  results/*           result files, validation reports and the sweep report
  inputs/*.yaml       the inputs that the script derives from those of examples/smoke/

The script always exits 0; read summary.json for pass or fail.
"""

import argparse
import copy
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend, set before any plotting import

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
INPUT_DIR = REPO_ROOT / "examples" / "smoke"
sys.path.insert(0, str(REPO_ROOT / "src"))

OUT_DIR = REPO_ROOT / "smoke_test_output"
LOG_PATH = None            # None: log to the console only (dry run)
DRY_RUN = False
RESULT_CACHE = {}          # results shared between scenarios


# ======================================================================
# Inputs and variants
# ======================================================================

def _mut_forms_indicator(d):
    d["encoding"] = "indicator"


def _mut_as_new(d):
    d["replacement_as_new"] = True
    d["replacement_mu"] = 0.0
    d["replacement_v"] = 0.0


def _mut_no_replacement(d):
    # a Wiener cell cannot lower its variance without replacement: use a second Gaussian cell
    d["allow_replacement"] = False
    d["model"] = ["gaussian", "gaussian", "gamma", "rainflow"]


def _mut_operating_average(d):
    d["objective_mode"] = "operating_average"


def _mut_evaluation_total(d):
    d["objective_mode"] = "evaluation_total"
    d["evaluation_horizon"] = 12


def _mut_quadratic(d):
    d["penalty_type"] = "quadratic"
    d["evaluation_horizon"] = 12   # given in the mode total: the projected cost is reported too


# variant name -> (base input name, mutation)
VARIANTS = {
    "forms_indicator": ("forms", _mut_forms_indicator),
    "replacement_as_new": ("replacement", _mut_as_new),
    "replacement_no_replacement": ("replacement", _mut_no_replacement),
    "costs_operating_average": ("costs", _mut_operating_average),
    "costs_evaluation_total": ("costs", _mut_evaluation_total),
    "costs_quadratic": ("costs", _mut_quadratic),
}

BASE_INPUTS = ["five_models", "rainflow_bounds", "forms", "anchors", "replacement", "caps",
               "costs", "two_phase", "sweep", "mixed_example"]


def input_data(name):
    """The input mapping of a base input or of a variant (nothing is written)."""
    from fleet_management.config import read_input
    if name in VARIANTS:
        base, mutate = VARIANTS[name]
        data = copy.deepcopy(read_input(INPUT_DIR / f"{base}.yaml"))
        mutate(data)
        return data
    return read_input(INPUT_DIR / f"{name}.yaml")


def input_path(name):
    """The path of an input file; a variant is written to ``inputs/`` first."""
    if name not in VARIANTS:
        return str(INPUT_DIR / f"{name}.yaml")
    import yaml
    path = OUT_DIR / "inputs" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as f:
        yaml.safe_dump(input_data(name), f, sort_keys=False)
    return str(path)


def load_cfg(name):
    from fleet_management.config import load_config
    return load_config(input_data(name))


# ======================================================================
# Logging and running
# ======================================================================

def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG_PATH is not None:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def run_pytest():
    log("Running `python -m pytest tests -v` (the tests that need Gurobi are in the suite)...")
    proc = subprocess.run([sys.executable, "-m", "pytest", "tests", "-v"], cwd=str(REPO_ROOT),
                          capture_output=True, text=True)
    (OUT_DIR / "pytest_output.txt").write_text(
        (proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")
    ok = proc.returncode == 0
    log(f"pytest exit code {proc.returncode} ({'PASSED' if ok else 'FAILED'}); "
        "output in pytest_output.txt")
    return {"ran": True, "passed": ok, "return_code": proc.returncode}


def write_summary(summary, pytest_result):
    n_pass = sum(1 for s in summary if s["status"] == "pass")
    data = {
        "pytest": pytest_result,
        "pytest_passed": pytest_result.get("passed"),
        "scenarios_passed": n_pass,
        "scenarios_failed": len(summary) - n_pass,
        "scenarios": summary,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(data, indent=2, default=str),
                                          encoding="utf-8")


def run_scenario(name, fn, summary):
    log(f"--- scenario: {name} ---")
    t0 = time.time()
    try:
        detail = fn()
        elapsed = time.time() - t0
        log(f"[PASS] {name} ({elapsed:.2f}s) {detail if detail is not None else ''}")
        summary.append({"name": name, "status": "pass", "elapsed_s": round(elapsed, 2),
                        "detail": detail})
    except Exception as exc:  # noqa: BLE001 -- report every scenario, never abort the run
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        log(f"[FAIL] {name} ({elapsed:.2f}s): {exc}\n{tb}")
        summary.append({"name": name, "status": "fail", "elapsed_s": round(elapsed, 2),
                        "error": str(exc), "traceback": tb})


# ======================================================================
# Shared steps: solve, validate, plot
# ======================================================================

def solve_check(tag, name=None, **solve_kwargs):
    """Solve the input ``name`` (default ``tag``), check it with ``validate``, plot it.

    Returns ``(result, input_path, results_path)``. A result without a solution, a failed
    check of ``validate`` and a failed plot raise ``AssertionError`` or the error itself.
    """
    from fleet_management import plot_management, solve, validate

    path = input_path(name or tag)
    res_path = OUT_DIR / "results" / f"{tag}.yaml"
    result = solve(path, str(res_path), **solve_kwargs)
    if result.get("x") is None:
        raise AssertionError(f"{tag}: no solution (status {result.get('status')})")
    if result["status"] != "optimal":
        log(f"NOTE {tag}: status {result['status']} (a solution exists), mip_gap={result['mip_gap']}")
    report = validate(path, str(res_path),
                      str(OUT_DIR / "results" / f"{tag}_validation.yaml"))
    if not report["passed"]:
        failed = [c["name"] for c in report["checks"] if not c["passed"]]
        raise AssertionError(f"{tag}: validate failed {failed}, max_violation={report['max_violation']}")
    plot_path = OUT_DIR / "plots" / f"{tag}.png"
    plot_management(str(res_path), str(plot_path))
    if not plot_path.is_file() or plot_path.stat().st_size == 0:
        raise AssertionError(f"{tag}: plot file missing or empty")
    return result, path, res_path


def _fmt(result):
    return f"status={result['status']}, objective={result['objective']:.4f}"


def _finite(a):
    return bool(np.all(np.isfinite(np.asarray(a, dtype=float))))


def _all_nan(a):
    return bool(np.all(np.isnan(np.asarray(a, dtype=float))))


def _column(result, key, l):
    return np.asarray(result[key], dtype=float)[:, l, :]


# ======================================================================
# Scenarios
# ======================================================================

def scenario_five_models():
    result, _, _ = solve_check("five_models")
    RESULT_CACHE["five_models"] = result
    models = {m for row in result["model"] for m in row}
    assert models == {"gaussian", "inverse_gaussian", "wiener", "gamma", "rainflow"}, models
    laws = {m for row in result["repair_model"] for m in row}
    assert laws == {"ardinf", "ard1"}, laws
    assert (result["H1"], result["H2"], result["T"]) == (2, 4, 6)
    # variance is tracked at G (0), W (2) and RF Cantelli (4); not at IG (1) and Ga (3)
    for l, tracked in enumerate([True, False, True, False, True]):
        assert _finite(_column(result, "v", l)) if tracked else _all_nan(_column(result, "v", l)), \
            f"v at component {l}: expected {'finite' if tracked else 'NaN'}"
    return _fmt(result)


def scenario_rainflow_bounds():
    result, _, _ = solve_check("rainflow_bounds")
    # v: Ca (1) and Be (3) only; R: Ho (2); K: Ch (4)
    for l in range(5):
        assert (_finite(_column(result, "v", l))) == (l in (1, 3)), f"v at component {l}"
        assert (_finite(_column(result, "R", l))) == (l == 2), f"R at component {l}"
        assert (_finite(_column(result, "K", l))) == (l == 4), f"K at component {l}"
    return _fmt(result)


def scenario_forms():
    cfg = load_cfg("forms")
    assert {cfg.reliability_impl[i, 1] for i in range(cfg.F)} == {"exact", "tangent", "pwl"}
    bigm, _, _ = solve_check("forms")
    indicator, _, _ = solve_check("forms_indicator")
    a, b = bigm["objective"], indicator["objective"]
    # both solves stop within 1 % of the optimum of the same model
    assert abs(a - b) <= 0.021 * max(abs(a), abs(b)) + 1e-6, f"bigm {a} vs indicator {b}"
    return f"bigm {a:.4f}, indicator {b:.4f}"


def scenario_anchors():
    result, _, _ = solve_check("anchors")
    # every cell is ard1: g_mu everywhere; g_v at G (0); g_R at Ho (2); g_K at Ch (3)
    assert _finite(result["g_mu"])
    for l in range(4):
        assert _finite(_column(result, "g_v", l)) == (l == 0), f"g_v at component {l}"
        assert _finite(_column(result, "g_R", l)) == (l == 2), f"g_R at component {l}"
        assert _finite(_column(result, "g_K", l)) == (l == 3), f"g_K at component {l}"
    return _fmt(result)


def scenario_replacement():
    base, _, _ = solve_check("replacement")
    as_new, _, _ = solve_check("replacement_as_new")
    off, _, _ = solve_check("replacement_no_replacement")
    assert np.all(np.asarray(off["r"]) == 0), "r must be zero if allow_replacement is false"
    n_base = int(np.sum(np.asarray(base["r"]) > 0.5))
    n_new = int(np.sum(np.asarray(as_new["r"]) > 0.5))
    return (f"replacements: base {n_base}, as_new {n_new}, not allowed 0; "
            f"objectives {base['objective']:.4f}, {as_new['objective']:.4f}, {off['objective']:.4f}")


def scenario_caps():
    from fleet_management.degradation_model.gamma_utils.gamma_replay_validator import gamma_cell_caps
    from fleet_management.degradation_model.inverse_gaussian import ig_mean_cap

    cfg = load_cfg("caps")
    result, _, _ = solve_check("caps")
    mu = np.asarray(result["mu"], dtype=float)
    ga_cells = [(i, l) for i in range(cfg.F) for l in range(cfg.L) if cfg.model[i, l] == "gamma"]
    caps = dict(gamma_cell_caps(cfg, ga_cells))
    for i in range(cfg.F):
        for l in range(cfg.L):
            if cfg.model[i, l] == "inverse_gaussian":
                caps[i, l] = ig_mean_cap(float(cfg.eta[i, l]), float(cfg.tau[i, l]),
                                         float(cfg.epsilon[i, l]))
    assert len(caps) == cfg.F * cfg.L, caps
    worst = 0.0
    for (i, l), cap in caps.items():
        top = float(mu[i, l].max())
        assert top <= cap * (1 + 1e-6) + 1e-9, f"cell ({i},{l}): max mu {top} above the cap {cap}"
        worst = max(worst, top / cap)
    return f"{_fmt(result)}; largest mu/cap = {worst:.4f}"


def check_costs(result, penalty):
    """Recompute every cost key of the result from its arrays (Eqs. a_daycost, a_phases, d_projected, c_u)."""
    x = np.asarray(result["x"], dtype=float)
    z = np.nan_to_num(np.asarray(result["z"], dtype=float))
    r = np.asarray(result["r"], dtype=float)
    mu = np.asarray(result["mu"], dtype=float)
    cc = result["component_costs"]
    C_M, C_D = float(cc["C_M"]), float(cc["C_D"])
    C_R, C_rep = np.asarray(cc["C_R"], dtype=float), np.asarray(cc["C_rep"], dtype=float)
    H1, H2 = int(result["H1"]), int(result["H2"])

    def close(name, got, want):
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6, err_msg=name)

    c = C_M * x[:, 0, :].sum(axis=0) + (C_R[:, :, None] * z).sum(axis=(0, 1)) \
        + (C_rep[:, :, None] * r).sum(axis=(0, 1))                       # Eq. a_daycost
    J_init, J_op = c[:H1].sum(), c[H1:].sum()                            # Eq. a_phases
    per_day = (mu.sum(axis=1) if penalty == "inf_norm" else (mu ** 2).sum(axis=1))
    u = per_day.max()                                                    # Eq. c_u
    close("step_costs", result["step_costs"], c)
    close("J_initialization", result["J_initialization"], J_init)
    close("J_op", result["J_op"], J_op)
    close("J_op_average", result["J_op_average"], J_op / H2)
    close("u", result["u"], u)
    close("peak_damage", result["peak_damage"], u)
    close("damage_penalty", result["damage_penalty"], C_D * u)
    close("J_total", result["J_total"], J_init + J_op + C_D * u)
    close("operating_objective", result["operating_objective"], J_op / H2 + C_D * u)
    E = result.get("evaluation_horizon")
    if E is None:
        assert result.get("projected_evaluation_cost") is None
        assert result.get("evaluation_objective") is None
        evaluation = None
    else:
        P_E = J_init + (E - H1) * J_op / H2                              # Eq. d_projected
        close("projected_evaluation_cost", result["projected_evaluation_cost"], P_E)
        close("evaluation_objective", result["evaluation_objective"], P_E + C_D * u)
        evaluation = P_E + C_D * u
    key = {"total": J_init + J_op + C_D * u, "operating_average": J_op / H2 + C_D * u,
           "evaluation_total": evaluation}[result["objective_mode"]]
    # the objective is not below the key; it is equal if the variable u has no slack
    assert result["objective"] >= key - 1e-6 * max(1.0, abs(key)), (result["objective"], key)
    return float(result["objective"] - key)


def scenario_costs():
    runs = [("costs", "total", "inf_norm"),
            ("costs_operating_average", "operating_average", "inf_norm"),
            ("costs_evaluation_total", "evaluation_total", "inf_norm"),
            ("costs_quadratic", "total", "quadratic")]
    details = {}
    for tag, mode, penalty in runs:
        result, _, _ = solve_check(tag)
        assert result["objective_mode"] == mode, (tag, result["objective_mode"])
        slack = check_costs(result, penalty)
        details[tag] = f"{result['objective']:.4f} (objective - key = {slack:.2e})"
    return details


def scenario_two_phase():
    cfg = load_cfg("two_phase")
    assert cfg.mu_trans is not None and cfg.mu_trans.shape[-1] == 3
    assert not np.allclose(cfg.mu_trans.mean(), cfg.mu.mean()), "mu_trans must differ from mu"
    result, _, _ = solve_check("two_phase")
    assert (result["H1"], result["H2"], result["T"]) == (3, 4, 7)
    assert np.asarray(result["mu"]).shape[-1] == 7
    return _fmt(result)


def scenario_sweep():
    from fleet_management import plot_horizon_sweep, sweep_operating_horizons

    path = input_path("sweep")
    report_path = OUT_DIR / "results" / "sweep_report.yaml"
    report = sweep_operating_horizons(path, [2, 3, 4], output_path=str(report_path),
                                      warm_start=True)
    cases = report["cases"]
    assert [c["H2"] for c in cases] == [2, 3, 4], [c["H2"] for c in cases]
    for c in cases:
        assert not str(c["status"]).startswith("error"), c["status"]
        assert c["status"] not in ("infeasible", "unbounded", "inf_or_unbounded"), c["status"]
    for c in cases[1:]:
        assert c["warm_start"]["applied"], f"no warm start applied at H2={c['H2']}"
    assert report_path.is_file()
    assert report["best_feasible_H2"] in (2, 3, 4), report["best_feasible_H2"]
    plot_path = OUT_DIR / "plots" / "sweep.png"
    plot_horizon_sweep(str(report_path), str(plot_path))
    assert plot_path.is_file() and plot_path.stat().st_size > 0
    return {"J_op_average": {c["H2"]: c["J_op_average"] for c in cases},
            "best_feasible_H2": report["best_feasible_H2"],
            "warm_values": {c["H2"]: c["warm_start"]["values"] for c in cases}}


def scenario_greedy_start():
    from fleet_management.greedy_warm_start import build_greedy_warm_start

    cfg = load_cfg("five_models")
    source, greedy_report = build_greedy_warm_start(cfg)
    greedy = greedy_report.as_dict()
    log(f"greedy report: {json.dumps(greedy, default=str)[:600]}")
    result, _, _ = solve_check("greedy_start", name="five_models",
                               warm_start=source, warm_start_mode="start")
    ws = result["warm_start"]
    assert ws["applied"] is True, ws
    assert ws["completion"]["requested"] is True, ws["completion"]
    assert ws["completion"]["status"] != "not_requested", ws["completion"]
    return {"objective": result["objective"], "repairs": greedy["repairs"],
            "replacements": greedy["replacements"],
            "repeatability_feasible": greedy["repeatability_feasible"],
            "values": ws["values"], "completion_status": ws["completion"]["status"]}


def scenario_gamma_validators():
    from fleet_management import validate_gamma_schedule

    cfg = load_cfg("five_models")
    result = RESULT_CACHE.get("five_models")
    if result is None:
        result, _, _ = solve_check("gamma_validators_solve", name="five_models")
    both = validate_gamma_schedule(cfg, result, mode="both", repetitions=20000)
    assert both["mode"] == "both", both["mode"]
    assert both["deterministic"]["valid"] is True, "replay validator: the schedule is not valid"
    assert both["stochastic"]["valid"] is True, "Monte Carlo validator: not valid"
    assert both["valid"] is True
    # mode and repetitions from the block ``validation`` of the input: stochastic, 20000
    stochastic = validate_gamma_schedule(cfg, result)
    assert stochastic["mode"] == "stochastic", stochastic["mode"]
    with open(OUT_DIR / "results" / "gamma_validators.json", "w", encoding="utf-8") as f:
        json.dump({"both": both, "stochastic": stochastic}, f, indent=2, default=str)
    return {"both_valid": both["valid"],
            "schedule_failure_rate": stochastic["schedule"]["failure_rate"],
            "upper_bound": stochastic["schedule"]["failure_rate_upper_confidence_bound"]}


def scenario_mixed_example():
    result, _, _ = solve_check("mixed_example")
    assert result["objective_mode"] == "evaluation_total" and result["evaluation_horizon"] == 30
    return _fmt(result)


SCENARIOS = [
    ("five_models", scenario_five_models),
    ("rainflow_bounds", scenario_rainflow_bounds),
    ("forms", scenario_forms),
    ("anchors", scenario_anchors),
    ("replacement", scenario_replacement),
    ("caps", scenario_caps),
    ("costs", scenario_costs),
    ("two_phase", scenario_two_phase),
    ("sweep", scenario_sweep),
    ("greedy_start", scenario_greedy_start),
    ("gamma_validators", scenario_gamma_validators),
    ("mixed_example", scenario_mixed_example),
]


# ======================================================================
# Main
# ======================================================================

def dry_run():
    """Load every input and variant with ``load_config``; no solve, no file written."""
    n_fail = 0
    for name in BASE_INPUTS + list(VARIANTS):
        try:
            cfg = load_cfg(name)
            log(f"OK   {name}: F={cfg.F} M={cfg.M} L={cfg.L} H1={cfg.H1} H2={cfg.H2}")
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            log(f"FAIL {name}: {type(exc).__name__}: {exc}")
    log(f"dry run: {len(BASE_INPUTS) + len(VARIANTS) - n_fail} inputs loaded, {n_fail} failed")


def main():
    global OUT_DIR, LOG_PATH, DRY_RUN
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-pytest", action="store_true", help="skip the pytest suite")
    parser.add_argument("--dry-run", action="store_true",
                        help="check every input with load_config, then exit")
    parser.add_argument("--out-dir", default=None, help="output folder")
    args = parser.parse_args()

    if args.out_dir:
        OUT_DIR = Path(args.out_dir).resolve()
    if args.dry_run:
        DRY_RUN = True
        dry_run()
        return

    for sub in ("plots", "results", "inputs"):
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)
    LOG_PATH = OUT_DIR / "smoke_test.log"
    LOG_PATH.write_text("", encoding="utf-8")

    log("Fleet Management remote smoke test starting.")
    log(f"Repo root: {REPO_ROOT}")
    log(f"Python: {sys.version}")
    try:
        import gurobipy as gp
        log(f"gurobipy import OK, Gurobi version {gp.gurobi.version()}")
    except Exception as exc:  # noqa: BLE001
        log(f"WARNING: gurobipy import or version check failed: {exc}")

    pytest_result = {"ran": False, "passed": None, "return_code": None}
    summary = []
    write_summary(summary, pytest_result)
    if args.no_pytest:
        log("pytest skipped (--no-pytest).")
    else:
        try:
            pytest_result = run_pytest()
        except Exception as exc:  # noqa: BLE001
            log(f"pytest could not run: {exc}\n{traceback.format_exc()}")
            pytest_result = {"ran": False, "passed": False, "return_code": None, "error": str(exc)}
        write_summary(summary, pytest_result)

    for name, fn in SCENARIOS:
        run_scenario(name, fn, summary)
        write_summary(summary, pytest_result)

    n_pass = sum(1 for s in summary if s["status"] == "pass")
    log(f"Scenario summary: {n_pass} passed, {len(summary) - n_pass} failed (of {len(summary)}).")
    log(f"pytest suite: {pytest_result}")
    log(f"Done. Zip up {OUT_DIR} and send it back for review.")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 -- the script always exits 0
        traceback.print_exc()
    sys.exit(0)
