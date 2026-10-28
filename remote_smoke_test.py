#!/usr/bin/env python3
"""
Remote smoke test for the fleet_management spec/spec.tex v0.5 rewrite.

Covers all five degradation models (Gaussian, inverse Gaussian, Wiener,
Gamma, Rainflow), both ARD1/ARA1 maintenance, replacement, mixed-model
fleets, exact/lp formulations, and the horizon loop. `input/data_example.yaml`
and `data_example_loop.yaml` already mix all five models (and ARA1, and a
Bernstein-tail-bound rainflow component), so every scenario that loads them
exercises the full Milestone-2 surface, not just Gaussian/IG.

This script performs REAL Gurobi solves -- run it on a machine with a
working Gurobi license, not in a sandboxed/no-license environment.

Usage
-----
    cd <unzipped repo root>
    pip install -e ".[dev]"
    python remote_smoke_test.py

Everything is written under ./smoke_test_output/:
  - smoke_test.log       human-readable, timestamped log of every step
  - summary.json         structured pass/fail per scenario + pytest result
  - pytest_output.txt    full `pytest -v` output (stdout+stderr)
  - plots/*.png          one plot per successful plotting scenario
  - results/*.yaml       raw solver output per scenario

Zip up smoke_test_output/ and send it back for review. The script always
exits 0 (so a single scenario failure doesn't stop others from running);
check summary.json / the log for pass/fail.
"""

import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe backend, must be set before any plotting import

REPO_ROOT = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT / "smoke_test_output"
PLOTS_DIR = OUT_DIR / "plots"
RESULTS_DIR = OUT_DIR / "results"
LOG_PATH = OUT_DIR / "smoke_test.log"
SUMMARY_PATH = OUT_DIR / "summary.json"
PYTEST_LOG_PATH = OUT_DIR / "pytest_output.txt"

# Fallback in case the package wasn't pip-installed (editable or otherwise).
sys.path.insert(0, str(REPO_ROOT / "src"))


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def run_pytest() -> bool:
    log("Running `pytest -v test/` (full suite; needs Gurobi for most tests)...")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-v", "test/"],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    PYTEST_LOG_PATH.write_text((proc.stdout or "") + "\n" + (proc.stderr or ""))
    ok = proc.returncode == 0
    log(f"pytest exit code {proc.returncode} ({'PASSED' if ok else 'FAILED'}); "
        f"full output in {PYTEST_LOG_PATH.name}")
    return ok


def _write_yaml(path, data):
    import yaml
    with open(path, "w") as f:
        yaml.dump(data, f)


def _load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def scenario(name, fn, summary: list) -> None:
    log(f"--- scenario: {name} ---")
    t0 = time.time()
    try:
        detail = fn()
        elapsed = time.time() - t0
        log(f"[PASS] {name} ({elapsed:.2f}s) {detail if detail is not None else ''}")
        summary.append({"name": name, "status": "pass", "elapsed_s": round(elapsed, 2), "detail": detail})
    except Exception as exc:  # noqa: BLE001 -- report every scenario, don't abort the run
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        log(f"[FAIL] {name} ({elapsed:.2f}s): {exc}\n{tb}")
        summary.append({
            "name": name, "status": "fail", "elapsed_s": round(elapsed, 2),
            "error": str(exc), "traceback": tb,
        })


# ======================================================================
# Scenarios
# ======================================================================

def scenario_basic_exact_mixed_fleet():
    from fleet_management import plot_management, solve

    out = RESULTS_DIR / "basic_exact.yaml"
    result = solve(str(REPO_ROOT / "input" / "data_example.yaml"), str(out))
    assert result["status"] == "optimal", result["status"]
    plot_management(str(out), str(PLOTS_DIR / "basic_exact.png"))
    return f"objective={result['objective']:.4f}"


def scenario_lp_formulation():
    from fleet_management import plot_management, solve

    data = _load_yaml(REPO_ROOT / "input" / "data_example.yaml")
    data["formulation"] = "lp"
    in_path = OUT_DIR / "_tmp_lp.yaml"
    _write_yaml(in_path, data)
    out = RESULTS_DIR / "lp_formulation.yaml"
    result = solve(str(in_path), str(out))
    assert result["status"] == "optimal", result["status"]
    plot_management(str(out), str(PLOTS_DIR / "lp_formulation.png"))
    return f"objective={result['objective']:.4f}"


def scenario_quadratic_penalty():
    from fleet_management import plot_management, solve

    data = _load_yaml(REPO_ROOT / "input" / "data_example.yaml")
    data["penalty_type"] = "quadratic"
    data["formulation"] = "exact"
    in_path = OUT_DIR / "_tmp_quadratic.yaml"
    _write_yaml(in_path, data)
    out = RESULTS_DIR / "quadratic_penalty.yaml"
    result = solve(str(in_path), str(out))
    assert result["status"] == "optimal", result["status"]
    plot_management(str(out), str(PLOTS_DIR / "quadratic_penalty.png"))
    return f"objective={result['objective']:.4f}"


def scenario_replacement_forced():
    from fleet_management import plot_management, solve

    data = _load_yaml(REPO_ROOT / "input" / "data_example.yaml")
    # Cheap replacement, expensive repair: makes replacement the economical
    # way to keep the loop constraint feasible.
    # mu_inc=0.07 (not more) matters here: since data_example.yaml now mixes
    # all five models, the gamma component (train1/comp1, beta=0.5) has by
    # far the tightest reliability cap of the fleet -- alpha_hat*beta ~ 0.081
    # (vs. ~0.45-0.49 for this example's inverse-Gaussian eta values) -- and
    # combined with weak repair (rho=0.3, i.e. ARD1 only removes 30% of
    # accumulated mean per repair) a too-aggressive increment (e.g. the
    # Milestone-1 value of 0.15, which alone exceeds the gamma cap in a
    # single mission from mu_0=0.05) makes the hard loop constraint
    # genuinely infeasible for that component, not just expensive --
    # verified locally via Gurobi's IIS. 0.07 still comfortably clears the
    # Gaussian/Rainflow validation floor of mu_inc >= 3*sqrt(v_inc) = 0.06.
    data["C_rep"] = 0.05
    data["C_R"] = 50.0
    data["rho"] = 0.3
    for i, row in enumerate(data["mu"]):
        for j, per_l in enumerate(row):
            data["mu"][i][j] = [0.07 for _ in per_l]
    in_path = OUT_DIR / "_tmp_replacement.yaml"
    _write_yaml(in_path, data)
    out = RESULTS_DIR / "replacement_forced.yaml"
    result = solve(str(in_path), str(out))
    assert result["status"] == "optimal", result["status"]
    plot_management(str(out), str(PLOTS_DIR / "replacement_forced.png"))
    import numpy as np
    n_replacements = int(np.sum(np.array(result["x_r"]) > 0.5))
    return f"objective={result['objective']:.4f}, replacements_used={n_replacements}"


def scenario_horizon_loop_sequential_warm_start():
    from fleet_management import plot_management, solve

    out = RESULTS_DIR / "horizon_loop.yaml"
    result = solve(str(REPO_ROOT / "input" / "data_example_loop.yaml"), str(out))
    assert set(result.keys()) == {4, 5, 6}, set(result.keys())
    for h in (4, 5, 6):
        assert result[h]["status"] == "optimal", (h, result[h]["status"])
    plot_management(str(out), str(PLOTS_DIR / "horizon_loop.png"))
    return {h: result[h]["objective"] for h in (4, 5, 6)}


def scenario_horizon_loop_parallel():
    from fleet_management import solve

    data = _load_yaml(REPO_ROOT / "input" / "data_example_loop.yaml")
    data["n_workers"] = 2
    data["warm_start"] = False
    in_path = OUT_DIR / "_tmp_parallel.yaml"
    _write_yaml(in_path, data)
    result = solve(str(in_path), results_path=None)
    assert set(result.keys()) == {4, 5, 6}, set(result.keys())
    for h in (4, 5, 6):
        assert result[h]["status"] == "optimal", (h, result[h]["status"])
    return {h: result[h]["objective"] for h in (4, 5, 6)}


def scenario_consistency_checks_via_files():
    from fleet_management import solve

    data = _load_yaml(REPO_ROOT / "input" / "data_example.yaml")

    bad_epsilon = json.loads(json.dumps(data))
    bad_epsilon["epsilon"] = 0.5
    p1 = OUT_DIR / "_tmp_bad_epsilon.yaml"
    _write_yaml(p1, bad_epsilon)
    try:
        solve(str(p1), results_path=None)
        raise AssertionError("expected ValueError for epsilon=0.5, none raised")
    except ValueError:
        pass

    bad_model = json.loads(json.dumps(data))
    bad_model["model"][0][0] = "not_a_real_model"
    p2 = OUT_DIR / "_tmp_bad_model.yaml"
    _write_yaml(p2, bad_model)
    try:
        solve(str(p2), results_path=None)
        raise AssertionError("expected ValueError for model='not_a_real_model', none raised")
    except ValueError:
        pass

    bad_ara1 = json.loads(json.dumps(data))
    bad_ara1["maintenance_type"][0][0] = "ARA1"  # train0/comp0 is gaussian: ARD1-only
    p3 = OUT_DIR / "_tmp_bad_ara1.yaml"
    _write_yaml(p3, bad_ara1)
    try:
        solve(str(p3), results_path=None)
        raise AssertionError("expected ValueError for ARA1 on a gaussian component, none raised")
    except ValueError:
        pass

    return "epsilon-range, unknown-model, and ARA1-on-ARD1-only-model checks all fired correctly"


def scenario_all_five_models_ara1_bernstein():
    """Dedicated Milestone-2 coverage check: confirms the resolved output
    actually reflects all five models, both maintenance types, and the
    Bernstein tail bound -- not just that a mixed-model file happens to
    solve. data_example.yaml already mixes all five models/ARD1+ARA1/both
    tail bounds, so this reuses it rather than building a new fixture."""
    from fleet_management import plot_management, solve
    import numpy as np

    out = RESULTS_DIR / "all_five_models.yaml"
    result = solve(str(REPO_ROOT / "input" / "data_example.yaml"), str(out))
    assert result["status"] == "optimal", result["status"]

    models_seen = {m for row in result["model"] for m in row}
    assert models_seen == {"gaussian", "inverse_gaussian", "wiener", "gamma", "rainflow"}, models_seen

    tail_bounds = result["tail_bound"]
    assert tail_bounds[2][0] == "bernstein", tail_bounds[2][0]  # train2/comp0 = rainflow/bernstein

    # inverse_gaussian (train0/comp1, train3/comp0) and gamma (train1/comp1)
    # don't track variance -- NaN expected there, real values everywhere else.
    ig_gamma_positions = [(0, 1), (3, 0), (1, 1)]
    for i in range(4):
        for l in range(2):
            v_is_nan = np.all(np.isnan(result["v"][i, l]))
            expected_nan = (i, l) in ig_gamma_positions
            assert v_is_nan == expected_nan, f"train{i}/comp{l}: v NaN={v_is_nan}, expected {expected_nan}"

    plot_management(str(out), str(PLOTS_DIR / "all_five_models.png"))
    return f"objective={result['objective']:.4f}, models={sorted(models_seen)}"


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    LOG_PATH.write_text("")  # truncate/start fresh

    log("Fleet Management remote smoke test starting.")
    log(f"Repo root: {REPO_ROOT}")
    log(f"Python: {sys.version}")

    try:
        import gurobipy as gp
        log(f"gurobipy import OK, Gurobi version {gp.gurobi.version()}")
    except Exception as exc:  # noqa: BLE001
        log(f"WARNING: gurobipy import/version check failed: {exc}")

    pytest_ok = run_pytest()

    summary = []
    scenario("basic_exact_mixed_fleet", scenario_basic_exact_mixed_fleet, summary)
    scenario("all_five_models_ara1_bernstein", scenario_all_five_models_ara1_bernstein, summary)
    scenario("lp_formulation", scenario_lp_formulation, summary)
    scenario("quadratic_penalty", scenario_quadratic_penalty, summary)
    scenario("replacement_forced", scenario_replacement_forced, summary)
    scenario("horizon_loop_sequential_warm_start", scenario_horizon_loop_sequential_warm_start, summary)
    scenario("horizon_loop_parallel", scenario_horizon_loop_parallel, summary)
    scenario("consistency_checks_via_files", scenario_consistency_checks_via_files, summary)

    n_pass = sum(1 for s in summary if s["status"] == "pass")
    n_fail = sum(1 for s in summary if s["status"] == "fail")
    log(f"Scenario summary: {n_pass} passed, {n_fail} failed (of {len(summary)}).")
    log(f"pytest suite: {'PASSED' if pytest_ok else 'FAILED (see pytest_output.txt)'}")

    SUMMARY_PATH.write_text(json.dumps(
        {"pytest_passed": pytest_ok, "scenarios": summary}, indent=2, default=str,
    ))
    log(f"Done. Zip up {OUT_DIR} and send it back for review.")


if __name__ == "__main__":
    main()
