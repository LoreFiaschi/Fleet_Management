"""Replay tests: solve small inputs and check every result with the reference model (``reference_model.py``).

Task O3 of WP8. Spec = ``spec/spec.tex`` (the ``lore`` copy): Decision Variables (603-650), Constraints
(698-1625), Result Dictionary and Rules for the Values (1714-1784).

Each case is an input in ``tests/data/oracle/replay_<case>.yaml``. The Gurobi test solves it with
``solve_mapping`` and checks the result:

* A  the status is ``optimal``;
* B  every row of the spec holds at the reported solution (``row_violations``), with the tolerances of
     the reference model (a big-M row: ``1e-6 + 2e-5 M_s``; a linear row: ``1e-6``; an exact quadratic
     row: ``1e-5`` relative);
* C  the schedule is safe on every repetition of the operating phase (``repeated_safety``, spec 1612-1617);
* D  each reported tracked array is at least the reference trajectory of the reported schedule
     ("Upper bounds", spec 1761);
* E  ``z`` (Eq. c_zpost on the transitory days of ``operating_average``), ``u`` (Eq. c_u) and the keys of
     the cost report equal the reference model on the reported arrays;
* F  ``objective`` equals the key of the objective mode (Rules for the Values, spec 1778-1783).

The variable domains of the table of spec 612-644 are checked as well; ``row_violations`` does not check them.

The test of the non-triviality needs no Gurobi: in every case, every schedule without intervention
breaks a row (the loop row ``s_T <= s_{H1}``, spec Eq. a_loop), because every mission is flown on every
day and every increment of the operating phase is positive.
"""

import copy
import math
from pathlib import Path

import numpy as np
import pytest

from fleet_management.config import load_config, read_input

import reference_model as rm

DATA = Path(__file__).parent / "data" / "oracle"

CASES = [
    "g_ardinf_exact", "g_ard1_tangent", "g_pwl4", "g_indicator",
    "ig_ardinf", "ig_ard1",
    "w_ardinf", "w_ard1_vmax",
    "ga_ardinf_beta0", "ga_ard1_rates",
    "rf_markov", "rf_cantelli_tangent", "rf_hoeffding_exact", "rf_bernstein_pwl", "rf_chernoff_ard1",
    "mixed",
    "mode_operating_average", "mode_evaluation_total", "penalty_quadratic",
    "no_replacement", "replacement_as_new", "transitory_profile",
]

TRACKED_ARRAYS = ("mu", "v", "R", "K")
ANCHOR_ARRAYS = ("g_mu", "g_v", "g_R", "g_K")


def _data(name):
    """The input mapping of a case, with the run options of the plan."""
    data = read_input(DATA / f"replay_{name}.yaml")
    data["verbose"] = 0
    data["mip_gap"] = 0
    data.setdefault("time_limit", 60)
    data["gurobi_params"] = {"Threads": 1}
    return data


def _config(name):
    return load_config(copy.deepcopy(_data(name)))


def _rounded_binary(a, name):
    """A binary array reported within ``1e-5`` of {0, 1}, rounded."""
    a = np.asarray(a, dtype=float)
    assert np.all(np.abs(a - np.round(a)) <= 1e-5), f"{name} is not binary"
    return np.round(a)


def _isclose(actual, expected, rel, abs_):
    return bool(np.all(np.isclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float),
                                  rtol=rel, atol=abs_, equal_nan=True)))


# ---------------------------------------------------------------------------
# The test that needs no Gurobi
# ---------------------------------------------------------------------------
def test_every_replay_file_is_listed():
    """The list of cases is the list of the files."""
    files = sorted(p.name[len("replay_"):-len(".yaml")] for p in DATA.glob("replay_*.yaml"))
    assert files == sorted(CASES)


def _schedule_in_turn(cfg, depot_repair=False):
    """Mission ``j`` is flown on day ``k`` by train ``(k + j) mod F``; no intervention, unless
    ``depot_repair``: then every other train is in the depot and repairs every component."""
    F, M, L, T = cfg.F, cfg.M, cfg.L, cfg.T
    x = np.zeros((F, M + 1, T), dtype=np.int8)
    m = np.zeros((F, L, T), dtype=np.int8)
    for k in range(T):
        flyers = [(k + j) % F for j in range(M)]
        for j, i in enumerate(flyers, start=1):
            x[i, j, k] = 1
        if depot_repair:
            for i in range(F):
                if i not in flyers:
                    x[i, 0, k] = 1
                    m[i, :, k] = 1
    return x, m, np.zeros_like(m)


def _reference_solution(cfg, x, m, r):
    """The schedule with its reference trajectory, the smallest ``z`` and the ``u`` of the reference."""
    traj = rm.reference_trajectory(cfg, x, m, r)
    sol = {"x": x, "m": m, "r": r, "z": rm.minimal_z(cfg, traj, m), "u": rm.u_value(cfg, traj["mu"])}
    for name in TRACKED_ARRAYS + ANCHOR_ARRAYS:
        sol[name] = traj[name][..., 1:]
    return sol


@pytest.mark.parametrize("name", CASES)
def test_case_is_not_trivial(name):
    """The input passes ``load_config`` and no schedule without intervention satisfies every row.

    Spec Eq. a_loop (loop row ``s_T <= s_{H1}`` for every tracked quantity). Without an intervention a
    state never falls. Every mission is flown on every day (Eq. a_coverage) and every increment of the
    operating phase is positive, so the state of some cell is larger on day T than on day H1: the loop row
    breaks, whatever the assignment. The schedule in turn is checked with the oracle as an instance.
    """
    cfg = _config(name)
    profile = rm.day_profile(cfg, "mu")
    assert profile[..., cfg.H1:].min() > 0.0
    x, m, r = _schedule_in_turn(cfg)
    violations = rm.row_violations(cfg, _reference_solution(cfg, x, m, r),
                                   tol=1e-9, bigm_tol=0.0, quad_rel=0.0)
    rows = {v.row for v in violations}
    assert "mu_loop" in rows, rows
    if all(str(cfg.model[i, l]) == "wiener" for i in range(cfg.F) for l in range(cfg.L)):
        # Wiener variance falls only by replacement (spec 1612-1617 and the Wiener section): repairs
        # in every free depot slot do not satisfy the loop row of the variance
        x, m, r = _schedule_in_turn(cfg, depot_repair=True)
        violations = rm.row_violations(cfg, _reference_solution(cfg, x, m, r),
                                       tol=1e-9, bigm_tol=0.0, quad_rel=0.0)
        assert "v_loop" in {v.row for v in violations}


# ---------------------------------------------------------------------------
# The Gurobi test: checks A to F
# ---------------------------------------------------------------------------
def _check_domains(cfg, res, const):
    """Variable domains of spec 612-644: ``mu`` in ``[0, tau]``, ``v`` in ``[0, M_v]``, ``R`` in ``[0, M_R]``,
    ``K`` in ``[0, M_K]``, anchors in the range of their quantity, ``z >= 0``, ``u >= 0``."""
    for i in range(cfg.F):
        for l in range(cfg.L):
            for q, Ms in const[(i, l)]["M"].items():
                tol = 1e-6 + 2e-5 * Ms
                for key in (q, "g_" + q):
                    values = np.asarray(res[key], dtype=float)[i, l]
                    values = values[~np.isnan(values)]
                    assert np.all(values >= -tol), f"{key} below 0 at cell {(i, l)}: {values.min()}"
                    assert np.all(values <= Ms + tol), f"{key} above {Ms} at cell {(i, l)}: {values.max()}"
    z = np.asarray(res["z"], dtype=float)
    assert np.all(z[~np.isnan(z)] >= -1e-6), "z below 0"
    assert float(res["u"]) >= -1e-6, "u below 0"


def _check_nan_pattern(res, traj):
    """``NaN`` exactly where the cell tracks no such quantity (spec 1736, 1738, 1739)."""
    for key in TRACKED_ARRAYS + ANCHOR_ARRAYS:
        assert np.array_equal(np.isnan(np.asarray(res[key], dtype=float)), np.isnan(traj[key][..., 1:])), \
            f"NaN pattern of {key}"


def _check_zpost(cfg, res):
    """Eq. c_zpost on the transitory days of the mode ``operating_average`` (spec 1762-1770)."""
    ini = rm.initial_and_new(cfg)
    z = np.asarray(res["z"], dtype=float)
    mu = np.asarray(res["mu"], dtype=float)
    g_mu = np.asarray(res["g_mu"], dtype=float)
    m = np.asarray(res["m"], dtype=float)
    for i in range(cfg.F):
        for l in range(cfg.L):
            if float(cfg.C_R[i, l]) == 0.0:
                continue
            ard1 = str(cfg.repair_model[i, l]) == "ard1"
            for k in range(1, cfg.H1 + 1):
                prev = ini[(i, l)]["init"]["mu"] if k == 1 else mu[i, l, k - 2]
                if not ard1:
                    g_prev = 0.0
                else:
                    g_prev = ini[(i, l)]["anchor0"]["mu"] if k == 1 else g_mu[i, l, k - 2]
                expected = float(cfg.rho[i, l]) * (prev - g_prev) if m[i, l, k - 1] > 0.5 else 0.0
                assert abs(z[i, l, k - 1] - expected) <= 1e-9, f"z of cell {(i, l)} on day {k}"


@pytest.mark.parametrize("name", CASES)
def test_solution_means_what_the_spec_says(name):
    """Solve the case and run the checks A to F of the plan with the reference model."""
    pytest.importorskip("gurobipy")
    from fleet_management.solver import solve_mapping

    data = _data(name)
    cfg = load_config(copy.deepcopy(data))
    res = solve_mapping(data, None)

    # A. status
    assert res["status"] == "optimal"

    x = _rounded_binary(res["x"], "x")
    m = _rounded_binary(res["m"], "m")
    r = np.zeros_like(m) if res.get("r") is None else _rounded_binary(res["r"], "r")
    if not cfg.allow_replacement:
        assert not r.any()
    idle = np.asarray(res["idle"], dtype=float)
    assert np.allclose(idle, x[:, 0, :][:, None, :] - m - r, atol=1e-9)       # spec 1734

    # B. every row of the spec at the reported solution
    sol = {key: res[key] for key in TRACKED_ARRAYS + ANCHOR_ARRAYS + ("z", "u")}
    sol.update(x=x, m=m, r=r)
    violations = rm.row_violations(cfg, sol, tol=1e-6, bigm_tol=2e-5, quad_rel=1e-5)
    assert violations == [], violations[:10]

    const = rm.constants(cfg)
    _check_domains(cfg, res, const)

    # C. the schedule is safe on every repetition of the operating phase
    assert rm.repeated_safety(cfg, x, m, r, cycles=20) >= -1e-9

    # D. reported tracked arrays are at least the reference trajectory of the reported schedule
    traj = rm.reference_trajectory(cfg, x, m, r)
    _check_nan_pattern(res, traj)
    for i in range(cfg.F):
        for l in range(cfg.L):
            for q, Ms in const[(i, l)]["M"].items():
                tol = 1e-5 + 1e-4 * Ms
                for key in (q, "g_" + q):
                    ref = traj[key][i, l, 1:]
                    if np.isnan(ref).all():
                        continue
                    reported = np.asarray(res[key], dtype=float)[i, l]
                    assert np.all(reported >= ref - tol), \
                        f"{key} of cell {(i, l)} below the reference: {reported} vs {ref}"

    # E. z, u and the cost report
    mode = cfg.options.get("objective_mode", "total")
    z = np.asarray(res["z"], dtype=float)
    assert np.array_equal(np.isnan(z)[:, :, 0], np.asarray(cfg.C_R, dtype=float) == 0.0)   # spec 1737
    if mode == "operating_average":
        _check_zpost(cfg, res)
    u = float(res["u"])
    assert math.isclose(u, rm.u_value(cfg, res["mu"]), rel_tol=1e-9, abs_tol=1e-12)        # Eq. c_u
    expected = rm.costs(cfg, x, r, z, u)
    assert _isclose(res["step_costs"], expected["step_costs"], 1e-8, 1e-9)
    for key in ("J_initialization", "J_op", "J_op_average", "peak_damage", "damage_penalty",
                "operating_objective", "J_total"):
        assert math.isclose(float(res[key]), expected[key], rel_tol=1e-8, abs_tol=1e-9), key
    for key in ("projected_evaluation_cost", "evaluation_objective"):
        if expected[key] is None:
            assert res[key] is None, key
        else:
            assert math.isclose(float(res[key]), expected[key], rel_tol=1e-8, abs_tol=1e-9), key

    # F. objective = the key of the mode (spec 1778-1783)
    key = {"total": "J_total", "operating_average": "operating_objective",
           "evaluation_total": "evaluation_objective"}[mode]
    assert math.isclose(expected["objective"], float(res[key]), rel_tol=1e-8, abs_tol=1e-9)
    objective = float(res["objective"])
    assert abs(objective - expected["objective"]) <= 1e-6 * max(1.0, abs(expected["objective"])) + 1e-4, \
        (objective, expected["objective"])
