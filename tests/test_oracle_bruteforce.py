"""Brute-force comparison of the solver with the reference model (oracle), on tiny fleets.

Spec: ``spec/spec.tex``, Constraints (698-1625), Full Optimization Problem (1628-1707), Result
Dictionary and Rules for the Values (1714-1784). Each input of ``tests/data/oracle/bf_*.yaml``
has F = 2 trains, M = 1 mission, L = 1 or 2 components and T <= 4 days, so that
``reference_model.best_reference`` enumerates every schedule of the spec and gives the best
cost that the spec allows.

Per case (``test_bruteforce_case``, needs Gurobi, skipped per test):
* A to F: the checks of every solved result (status, rows, safety on repetitions, tracked arrays
  as upper bounds, z, u and the cost report, the objective);
* G1: the solver's objective is at most the best reference cost;
* G2: a solver objective below the best reference cost is allowed by the spec (the one-sided
  rows let a tracked state exceed the reference trajectory); it is printed, not failed;
* G3: the solver's objective against ``best_onesided`` (the best cost over the schedules that have a
  one-sided solution, with the least solution): equal when every cell is ``ardinf`` (then it is the
  optimum of the spec's problem), at most that cost otherwise;
* an input without a feasible schedule: no invalid ``optimal`` point.

``test_bruteforce_case_nontrivial`` needs no Gurobi: the best reference schedule of each input
has an intervention or a binding reliability row.
"""

import copy
import glob
import math
import os
import warnings

import numpy as np
import pytest

from fleet_management.config import load_config, read_input
from reference_model import (best_onesided, best_reference, cell_kind, constants, costs, initial_and_new,
                             reference_trajectory, reliability_slack, repeated_safety,
                             least_solution, row_violations, u_value)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "oracle")
CASE_PATHS = sorted(glob.glob(os.path.join(DATA, "bf_*.yaml")))
CASE_IDS = [os.path.basename(p)[3:-5] for p in CASE_PATHS]

_BEST_CACHE = {}
_ONESIDED_CACHE = {}


def _load(path):
    """``(data, cfg)``: the input mapping (a fresh copy) and its checked configuration."""
    data = read_input(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = load_config(copy.deepcopy(data))
    return data, cfg


def _best(path):
    """The best reference result of the case, computed once per run."""
    if path not in _BEST_CACHE:
        data, cfg = _load(path)
        _BEST_CACHE[path] = (cfg, best_reference(cfg))
    return _BEST_CACHE[path]


def _onesided(path):
    """The best one-sided result of the case, computed once per run."""
    if path not in _ONESIDED_CACHE:
        cfg, _ = _best(path)
        _ONESIDED_CACHE[path] = best_onesided(cfg)
    return _ONESIDED_CACHE[path]


def _close(a, b, rel=1e-8, abs_tol=1e-9):
    return abs(a - b) <= abs_tol + rel * max(abs(a), abs(b))


def _min_slack(cfg, traj):
    """Smallest reliability slack of a reference trajectory over all cells and days 1..T."""
    best = math.inf
    for i in range(cfg.F):
        for l in range(cfg.L):
            tracked = cell_kind(cfg, i, l)["tracked"]
            for k in range(1, cfg.T + 1):
                state = {q: float(traj[q][i, l, k]) for q in tracked}
                best = min(best, reliability_slack(cfg, i, l, state))
    return best


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def test_bruteforce_inputs_are_documented():
    """Every input starts with a comment line and sets the run options of the plan (spec 482-496)."""
    assert len(CASE_PATHS) >= 35
    for path in CASE_PATHS:
        with open(path, encoding="utf-8") as f:
            assert f.readline().startswith("#"), path
        data = read_input(path)
        assert data["verbose"] == 0 and data["mip_gap"] == 0 and data["time_limit"] == 60
        assert data["gurobi_params"] == {"Threads": 1}
        cfg = load_config(copy.deepcopy(data))
        assert cfg.F == 2 and cfg.M == 1 and cfg.L in (1, 2) and cfg.T <= 4


@pytest.mark.parametrize("path", CASE_PATHS, ids=CASE_IDS)
def test_bruteforce_case_nontrivial(path):
    """The case has a feasible schedule, and the best one has an intervention or a binding row.

    An input named ``bf_infeasible_*`` has no feasible schedule (loop row, spec 1606-1617).
    """
    cfg, best = _best(path)
    if os.path.basename(path).startswith("bf_infeasible"):
        assert best["n_feasible"] == 0 and best["cost"] is None
        return
    assert best["n_feasible"] >= 1, "no feasible schedule: change the numbers"
    x, m, r = best["schedule"]
    interventions = int(np.sum(m) + np.sum(r))
    traj = reference_trajectory(cfg, x, m, r)
    slack = _min_slack(cfg, traj)
    scale = max(1.0, float(np.max(cfg.tau)))
    binding = slack <= 1e-6 * scale
    assert interventions > 0 or binding, "trivial case: change the numbers"


# ---------------------------------------------------------------------------
# Checks on a solved result
# ---------------------------------------------------------------------------
def _binary(res, key, tol=1e-5):
    """A binary array of the result, within ``tol`` of {0, 1}, rounded."""
    a = np.asarray(res[key], dtype=float)
    assert np.all(np.abs(a - np.round(a)) <= tol), f"{key} is not binary"
    return np.round(a).astype(np.int8)


def _solution(res):
    """The mapping that ``row_violations`` reads: binaries rounded, the rest as reported."""
    sol = {"x": _binary(res, "x"), "m": _binary(res, "m"), "r": _binary(res, "r")}
    for key in ("mu", "v", "R", "K", "g_mu", "g_v", "g_R", "g_K", "z"):
        sol[key] = np.asarray(res[key], dtype=float)
    for key in ("y_mu", "y_v", "y_R", "y_K", "y_g_mu", "y_g_v", "y_g_R", "y_g_K"):   # start of the repetition
        sol[key] = np.asarray(res[key], dtype=float)
    sol["u"] = float(res["u"])
    return sol


def _check_rows_and_safety(cfg, res):
    """Checks B and C of the plan (spec 698-1625 and the claim of 1612-1617)."""
    sol = _solution(res)
    violations = row_violations(cfg, sol)
    assert not violations, f"rows of the spec broken: {violations[:5]}"
    # The loop rows hold up to the solver's feasibility tolerance, which adds up over the
    # repetitions: 20 cycles of 1e-6 stay below 5e-5.
    safety = repeated_safety(cfg, sol["x"], sol["m"], sol["r"], cycles=20)
    assert safety >= -5e-5, f"schedule unsafe on a repetition: slack {safety}"
    return sol


def _check_upper_bounds(cfg, res, sol):
    """Check D (spec 1761): each tracked array is at least the reference trajectory."""
    traj = reference_trajectory(cfg, sol["x"], sol["m"], sol["r"])
    const = constants(cfg)
    for name in ("mu", "v", "R", "K", "g_mu", "g_v", "g_R", "g_K"):
        ref = traj[name][..., 1:]
        got = np.asarray(res[name], dtype=float)
        for i in range(cfg.F):
            for l in range(cfg.L):
                if np.all(np.isnan(ref[i, l])):
                    continue
                Ms = const[(i, l)]["M"][name[2:] if name.startswith("g_") else name]
                tol = 1e-6 + 2e-5 * Ms
                assert np.all(got[i, l] >= ref[i, l] - tol), \
                    f"{name} of cell {(i, l)} below the reference trajectory"
                # the start of the repetition is at least the reference value on day H1, and in [0, M_s]
                y = float(np.asarray(res["y_" + name], dtype=float)[i, l])
                assert y >= traj[name][i, l, cfg.H1] - tol, f"y_{name} of cell {(i, l)} below day H1"
                assert -tol <= y <= Ms + tol, f"y_{name} of cell {(i, l)} outside [0, {Ms}]"


def _check_z_u_costs(cfg, res, sol):
    """Check E: z on the transitory days, u, and the cost report (spec 1762-1783)."""
    ini = initial_and_new(cfg)
    mode = cfg.options.get("objective_mode", "total")
    if mode == "operating_average":                   # Eq. c_zpost
        z = np.asarray(res["z"], dtype=float)
        mu = np.asarray(res["mu"], dtype=float)
        g = np.asarray(res["g_mu"], dtype=float)
        for i in range(cfg.F):
            for l in range(cfg.L):
                if float(cfg.C_R[i, l]) == 0.0:
                    continue
                ard1 = str(cfg.repair_model[i, l]) == "ard1"
                for k in range(1, cfg.H1 + 1):
                    mu_prev = ini[(i, l)]["init"]["mu"] if k == 1 else mu[i, l, k - 2]
                    g_prev = (ini[(i, l)]["anchor0"]["mu"] if k == 1 else g[i, l, k - 2]) \
                        if ard1 else 0.0
                    want = float(cfg.rho[i, l]) * (mu_prev - g_prev) if sol["m"][i, l, k - 1] else 0.0
                    assert _close(z[i, l, k - 1], want, rel=1e-8, abs_tol=1e-9), \
                        f"z of cell {(i, l)} on transitory day {k}: {z[i, l, k - 1]} != {want}"
    assert _close(float(res["u"]), u_value(cfg, res["mu"]), rel=1e-8), "u is not recomputed from mu"
    rep = costs(cfg, sol["x"], sol["r"], res["z"], float(res["u"]))
    for key in ("J_initialization", "J_op", "J_op_average", "peak_damage", "damage_penalty",
                "operating_objective", "J_total"):
        assert _close(float(res[key]), rep[key]), f"{key}: {res[key]} != {rep[key]}"
    assert np.allclose(np.asarray(res["step_costs"], dtype=float), rep["step_costs"],
                       rtol=1e-8, atol=1e-9), "step_costs"
    for key in ("projected_evaluation_cost", "evaluation_objective"):
        if rep[key] is None:
            assert res[key] is None, key
        else:
            assert _close(float(res[key]), rep[key]), f"{key}: {res[key]} != {rep[key]}"
    return rep


def _check_objective(res, rep):
    """Check F (Rules 1778-1783): the objective is the key of the mode in use."""
    assert _close(float(res["objective"]), rep["objective"], rel=1e-6, abs_tol=1e-9), \
        f"objective {res['objective']} != {rep['objective']}"


# ---------------------------------------------------------------------------
# The solver against the enumeration
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", CASE_PATHS, ids=CASE_IDS)
def test_bruteforce_case(path):
    """Spec 698-1707 and 1714-1784: checks A to F, then G1 and G2 against ``best_reference``."""
    pytest.importorskip("gurobipy")
    from fleet_management.solver import solve_mapping

    cfg, best = _best(path)
    data, _ = _load(path)
    res = solve_mapping(data, None)
    name = os.path.basename(path)[3:-5]

    if best["cost"] is None:
        # Infeasible case: no schedule satisfies the rows, so the solver must not report an
        # optimal point that breaks them.
        if res.get("x") is None:
            assert res["status"] != "optimal"
            return
        _check_rows_and_safety(cfg, res)
        return

    assert res["status"] == "optimal"                                    # A
    sol = _check_rows_and_safety(cfg, res)                               # B, C
    _check_upper_bounds(cfg, res, sol)                                   # D
    rep = _check_z_u_costs(cfg, res, sol)                                # E
    _check_objective(res, rep)                                           # F

    objective = float(res["objective"])
    ref = float(best["cost"])
    # Gurobi's FeasibilityTol is 1e-6 per row, so the optimum can differ from the reference by a few 1e-6.
    slack = 1e-5 * max(1.0, abs(ref))
    assert objective <= ref + slack, \
        f"G1: the solver's optimum {objective} is above the best reference cost {ref}"
    if objective < ref - slack:                                          # G2
        print(f"G2 {name}: solver objective {objective!r} < reference best {ref!r}")
    ons = _onesided(path)                                                # G3
    ocost = float(ons["cost"])
    oslack = 1e-5 * max(1.0, abs(ocost))                                 # same reason as for slack
    if ons["exact"]:
        assert abs(objective - ocost) <= oslack, \
            f"G3: the solver's optimum {objective} differs from the spec's optimum {ocost}"
    else:
        assert objective <= ocost + oslack, \
            f"G3: the solver's optimum {objective} is above the one-sided bound {ocost}"


@pytest.mark.parametrize("path", CASE_PATHS, ids=CASE_IDS)
def test_bruteforce_case_onesided(path):
    """No Gurobi. Spec 780-836, 1602-1617: ``best_onesided <= best_reference``, and the least solution of
    the best one-sided schedule breaks no row of the spec (exact tolerances)."""
    cfg, best = _best(path)
    ons = _onesided(path)
    if best["cost"] is None:                  # bf_infeasible_*: no one-sided solution either (checked locally)
        assert ons["cost"] is None and ons["n_feasible"] == 0
        return
    assert ons["cost"] is not None and ons["cost"] <= best["cost"] + 1e-12
    assert ons["n_feasible"] >= best["n_feasible"]
    x, m, r = ons["schedule"]
    sol = least_solution(cfg, x, m, r)
    assert sol["feasible"] and sol["row_violations"] == [], sol["row_violations"][:5]
    assert _close(sol["u"], u_value(cfg, sol["mu"]))
    rep = costs(cfg, x, r, sol["z"], sol["u"])
    assert _close(rep["objective"], ons["cost"], rel=1e-9, abs_tol=1e-9)
