"""Tests ported from the earlier suite of the branch ``lore`` and from ``main``'s regression checks.

Ported from ``lore``'s ``test/`` (written for its old API; the merged keys and result schema
replace the old ones):
* the inverse Gaussian cap falls when ``eta`` grows (``test_solve_mu_bar_monotone_in_eta``);
* one free solve of a Gamma-only fleet (``test_gamma_reliability_cap_and_loop_constraint_hold``);
* three solves followed by a plot, from a YAML, a JSON and an HDF5 input file
  (``test_yaml_end_to_end_mixed``, ``test_json_end_to_end_gaussian_only``,
  ``test_hdf5_end_to_end_gaussian_only``);
* one solve, ``validate`` and ``plot_management`` on a fleet with the five models and both
  repair laws (``test_all_five_models_mixed_fleet_with_ara1_and_bernstein``).

Ported from ``examples/regression/check_gamma_ard1_integration.py`` of ``main``: a complete
repair (``rho = 1``) and two consecutive repairs on a Gamma cell, each with a hand-derived
trajectory (spec, "Repair Laws" and "Gamma Model", Eq. gm_map).

The inputs are in ``tests/data/ported/``. Every test that solves needs gurobipy and takes
less than 5 s; the cap test needs neither.
"""

import json
import math
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from fleet_management.config import load_config, read_input
from fleet_management.degradation_model import gamma
from fleet_management.degradation_model.inverse_gaussian import ig_mean_cap

NAN = math.nan
DATA = Path(__file__).resolve().parent / "data" / "ported"
_NO_SOLUTION = ("infeasible", "inf_or_unbounded", "unbounded")


# ---------------------------------------------------------------------------
# The inverse Gaussian cap (no Gurobi)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("epsilon", [0.01, 0.05, 0.1])
@pytest.mark.parametrize("tau", [0.5, 1.0])
def test_ig_cap_falls_when_eta_grows(tau, epsilon):
    # spec, Inverse Gaussian Model, Eq. ig_row: mu_bar = tau / E[Z | Z > q]; a larger eta
    # (a wider law of each term) gives a smaller cap, and 0 < mu_bar < tau at every eta
    etas = [1e-3, 0.01, 0.05, 0.15, 0.5, 1.0, 2.0, 10.0, 100.0]
    caps = [ig_mean_cap(eta, tau, epsilon) for eta in etas]
    assert all(0.0 < cap < tau for cap in caps)
    assert all(a > b for a, b in zip(caps, caps[1:]))


# ---------------------------------------------------------------------------
# Helpers for the tests that solve
# ---------------------------------------------------------------------------
def _gurobi():
    pytest.importorskip("gurobipy")


def _png(path):
    path = Path(path)
    assert path.exists() and path.stat().st_size > 1000
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def _write_hdf5(path, data):
    """Write an input mapping with the HDF5 layout of the spec (Input File Schema, HDF5 Layout).

    A number, a flag or a string is an attribute; ``H`` with two entries is an integer
    attribute; a list of numbers is a ``float64`` dataset (``None`` is ``NaN``); a list of
    strings and a mapping are JSON text in an attribute.
    """
    with h5py.File(path, "w") as f:
        for key, value in data.items():
            if key == "H" and isinstance(value, list):
                f.attrs[key] = np.array(value, dtype=np.int64)
            elif isinstance(value, dict):
                f.attrs[key] = json.dumps(value)
            elif isinstance(value, list) and any(isinstance(v, str) for v in value):
                f.attrs[key] = json.dumps(value)
            elif isinstance(value, list):
                f.create_dataset(key, data=np.array(value, dtype=np.float64))
            else:
                f.attrs[key] = value


def _solve_and_plot(input_path, results_path, plot_path):
    from fleet_management import plot_management, solve
    from fleet_management.solver import read_result
    result = solve(str(input_path), str(results_path))
    assert result["status"] not in _NO_SOLUTION
    assert result["x"] is not None
    saved = read_result(results_path)
    assert saved["status"] == result["status"]
    assert np.asarray(saved["x"]).shape == np.asarray(result["x"]).shape
    plot_management(str(results_path), str(plot_path))
    _png(plot_path)
    return result


# ---------------------------------------------------------------------------
# A free solve of a Gamma-only fleet
# ---------------------------------------------------------------------------
def test_gamma_free_solve_cap_and_loop_rows(tmp_path):
    # spec, Gamma Model, Eq. gm_row: mu <= A_1(beta* tau, eps) / beta* on every day; Loop
    # Constraint, Eq. a_loop: mu and the anchor g_mu at day T are not above day H1.
    _gurobi()
    from fleet_management import solve
    path = DATA / "gamma_only.yaml"
    cfg = load_config(read_input(path))
    result = solve(str(path), None)
    assert result["status"] == "optimal"
    # every increment has the rate 20 and mu_0 is a known constant: beta* = 20
    cap = gamma.mean_cap(20.0, 1.0, 0.1)
    assert math.isclose(cap, 0.744517324566, rel_tol=1e-9)
    mu = np.asarray(result["mu"])
    assert mu.shape == (cfg.F, cfg.L, cfg.T)
    assert np.all(mu <= cap + 1e-6)
    assert np.all(np.isnan(np.asarray(result["v"])))        # Ga tracks no variance
    h1, last = cfg.H1 - 1, cfg.T - 1
    assert np.all(mu[:, :, last] <= mu[:, :, h1] + 1e-6)
    g_mu = np.asarray(result["g_mu"])
    assert np.all(g_mu[:, 0, last] <= g_mu[:, 0, h1] + 1e-6)    # ard1 component
    assert np.all(np.isnan(g_mu[:, 1, :]))                       # ardinf component: no anchor


# ---------------------------------------------------------------------------
# Solve, then plot, from YAML, JSON and HDF5 input files
# ---------------------------------------------------------------------------
def test_yaml_end_to_end_mixed(tmp_path):
    _gurobi()
    result = _solve_and_plot(DATA / "e2e_mixed.yaml", tmp_path / "result.yaml", tmp_path / "plot.png")
    assert np.asarray(result["mu"]).shape == (2, 3, 3)
    v = np.asarray(result["v"])
    assert not np.isnan(v[:, 0, :]).any() and np.isnan(v[:, 1:, :]).all()


def test_json_end_to_end_gaussian_only(tmp_path):
    _gurobi()
    data = yaml.safe_load((DATA / "e2e_gaussian.yaml").read_text(encoding="utf-8"))
    source = tmp_path / "input.json"
    source.write_text(json.dumps(data), encoding="utf-8")
    result = _solve_and_plot(source, tmp_path / "result.json", tmp_path / "plot.png")
    assert np.asarray(result["mu"]).shape == (2, 1, 3)
    assert np.all(np.isfinite(np.asarray(result["v"])))


def test_hdf5_end_to_end_gaussian_only(tmp_path):
    _gurobi()
    data = yaml.safe_load((DATA / "e2e_gaussian.yaml").read_text(encoding="utf-8"))
    source = tmp_path / "input.h5"
    _write_hdf5(source, data)
    # the reader returns the same configuration as the YAML file
    cfg_h5 = load_config(read_input(source))
    cfg_yaml = load_config(read_input(DATA / "e2e_gaussian.yaml"))
    assert (cfg_h5.F, cfg_h5.M, cfg_h5.L, cfg_h5.H1, cfg_h5.H2) == (2, 1, 1, 1, 2)
    assert cfg_h5.T == cfg_yaml.T
    result = _solve_and_plot(source, tmp_path / "result.h5", tmp_path / "plot.png")
    expected = _solve_and_plot(DATA / "e2e_gaussian.yaml", tmp_path / "expected.yaml",
                               tmp_path / "expected.png")
    gap = 0.01 * max(1.0, abs(expected["objective"]))      # mip_gap of the input
    assert abs(result["objective"] - expected["objective"]) <= 2.0 * gap


# ---------------------------------------------------------------------------
# One solve, validate and plot on a fleet with the five models and both repair laws
# ---------------------------------------------------------------------------
def test_five_models_solve_validate_and_plot(tmp_path):
    # the laws are ``ardinf`` and ``ard1``; the rainflow cell uses the default bound Cantelli
    _gurobi()
    from fleet_management import validate
    path = DATA / "five_models.yaml"
    cfg = load_config(read_input(path))
    results = tmp_path / "result.yaml"
    result = _solve_and_plot(path, results, tmp_path / "plot.png")
    assert result["status"] == "optimal"

    report = validate(str(path), str(results), str(tmp_path / "validation.yaml"))
    assert report["passed"], [c for c in report["checks"] if not c["passed"]]
    assert (tmp_path / "validation.yaml").exists()

    laws = np.asarray(yaml.safe_load(path.read_text(encoding="utf-8"))["repair_model"])
    assert set(laws.ravel()) == {"ard1", "ardinf"}
    v = np.asarray(result["v"])
    g_mu = np.asarray(result["g_mu"])
    for i in range(cfg.F):
        for l in range(cfg.L):
            assert np.isnan(v[i, l]).all() == (l in (1, 3))      # IG and Ga track no variance
            assert np.isnan(g_mu[i, l]).all() == (laws[i, l] == "ardinf")


# ---------------------------------------------------------------------------
# Gamma ARD1: complete repair and consecutive repairs (hand-derived trajectories)
# ---------------------------------------------------------------------------
def _gamma_input(**changes):
    """Two trains, one mission, one Gamma component, ard1; train 1 flies with mean 0."""
    data = {"F": 2, "M": 1, "model": "gamma", "repair_model": "ard1", "encoding": "bigm",
            "tau": 1.0, "epsilon": 0.1, "rho": 0.5, "mu_0": 0.4, "anchor_mu_0": 0.1,
            "replacement_mu": 0.1, "mu": [[[0.3]], [[0.0]]], "gamma_beta": 20.0,
            "C_M": 1.0, "C_R": 2.0, "C_rep": 3.0, "C_D": 0.0,
            "verbose": 0, "mip_gap": 0.0, "time_limit": 60,
            "gurobi_params": {"DualReductions": 0}}
    data.update(changes)
    return data


def _solve_fixed(cfg, flights, repairs, replacements):
    """Fix the schedule, minimise mu + g_mu + z and optimise; return the result dictionary.

    ``flights`` are the keys ``(i, j, k)`` of ``x`` set to 1 (``j = 0``: a depot day), with
    ``k`` the 0-based day; ``repairs`` and ``replacements`` are keys ``(i, 0, k)``.
    """
    _gurobi()
    import gurobipy as gp
    from fleet_management.degradation_model import base
    ctx = base.build_fleet(cfg, base.resolve_run_options(cfg))
    for key, var in ctx.x.items():
        var.LB = var.UB = 1.0 if key in flights else 0.0
    for key, var in ctx.m_rep.items():
        var.LB = var.UB = 1.0 if key in repairs else 0.0
    for key, var in ctx.r_rep.items():
        var.LB = var.UB = 1.0 if key in replacements else 0.0
    states = (list(ctx.mu_var.values()) + list(ctx.anchor_vars["mu"].values())
              + list(ctx.z_var.values()))
    ctx.model.setObjective(gp.quicksum(states), gp.GRB.MINIMIZE)
    ctx.model.optimize()
    return base.extract_solution(ctx, cfg)


@pytest.mark.parametrize("encoding", ["bigm", "indicator"])
def test_gamma_ard1_complete_repair(encoding):
    # rho = 1, H = [2, 2], T = 4, mission mean 0.3, mu_0 = 0.2, starting anchor 0, no replacement.
    # Train 0 flies on days 1 and 3 and is in the depot, repaired, on days 2 and 4; train 1
    # flies on days 2 and 4. ard1 repair (Eq. a_ard1): D+ = D- - rho (D- - g) with g the level
    # after the last intervention, so rho = 1 gives D+ = g (Eq. gm_map: (1 - rho) mu + rho g).
    #   day 1: fly      mu = 0.2 + 0.3 = 0.5   g = 0
    #   day 2: repair   mu = 0 * 0.5 + 1 * 0 = 0    g = 0    z = rho (0.5 - 0) = 0.5
    #   day 3: fly      mu = 0 + 0.3 = 0.3     g = 0
    #   day 4: repair   mu = 0 * 0.3 + 1 * 0 = 0    g = 0    z = rho (0.3 - 0) = 0.3
    # The loop rows hold: mu and g at day 4 (0, 0) are not above day 2 (0, 0).
    cfg = load_config(_gamma_input(H=[2, 2], rho=1.0, mu_0=0.2, anchor_mu_0=0.0,
                                   replacement_mu=0.0, encoding=encoding))
    flights = {(0, 1, 0), (1, 1, 1), (0, 0, 1), (0, 1, 2), (1, 1, 3), (0, 0, 3)}
    out = _solve_fixed(cfg, flights, {(0, 0, 1), (0, 0, 3)}, set())
    assert out["status"] == "optimal"
    np.testing.assert_allclose(out["mu"][0, 0], [0.5, 0.0, 0.3, 0.0], atol=1e-6)
    np.testing.assert_allclose(out["g_mu"][0, 0], [0.0, 0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(out["z"][0, 0], [0.0, 0.5, 0.0, 0.3], atol=1e-6)
    # train 1 never has an intervention: mu_0 and the anchor stay
    np.testing.assert_allclose(out["mu"][1, 0], [0.2] * 4, atol=1e-6)
    np.testing.assert_allclose(out["g_mu"][1, 0], [0.0] * 4, atol=1e-6)


@pytest.mark.parametrize("encoding", ["bigm", "indicator"])
def test_gamma_ard1_consecutive_repairs(encoding):
    # rho = 0.5, H = [2, 3], T = 5, mission mean 0.3, mu_0 = 0.4, starting anchor 0.1,
    # replacement_mu = 0.1. Train 0 flies on days 1 and 4; it is in the depot on days 2 and
    # 3 (two repairs in a row) and on day 5 (replacement); train 1 flies on days 2, 3 and 5.
    #   day 1: fly      mu = 0.4 + 0.3 = 0.7   g = 0.1
    #   day 2: repair   mu = 0.5 * 0.7 + 0.5 * 0.1 = 0.4   g = 0.4   z = 0.5 (0.7 - 0.1) = 0.3
    #   day 3: repair   mu = 0.5 * 0.4 + 0.5 * 0.4 = 0.4   g = 0.4   z = 0.5 (0.4 - 0.4) = 0
    #          (the anchor is the level after the last intervention, so the second repair
    #          removes nothing)
    #   day 4: fly      mu = 0.4 + 0.3 = 0.7   g = 0.4
    #   day 5: replace  mu = 0.1   g = 0.1   z = 0
    # mu = 0.7 on day 1 is below the cap A_1(20, 0.1) / 20 = 0.7445; the loop rows hold:
    # mu and g at day 5 (0.1, 0.1) are not above day 2 (0.4, 0.4).
    cfg = load_config(_gamma_input(H=[2, 3], encoding=encoding))
    flights = {(0, 1, 0), (1, 1, 1), (0, 0, 1), (1, 1, 2), (0, 0, 2), (0, 1, 3),
               (1, 1, 4), (0, 0, 4)}
    out = _solve_fixed(cfg, flights, {(0, 0, 1), (0, 0, 2)}, {(0, 0, 4)})
    assert out["status"] == "optimal"
    np.testing.assert_allclose(out["mu"][0, 0], [0.7, 0.4, 0.4, 0.7, 0.1], atol=1e-6)
    np.testing.assert_allclose(out["g_mu"][0, 0], [0.1, 0.4, 0.4, 0.4, 0.1], atol=1e-6)
    np.testing.assert_allclose(out["z"][0, 0], [0.0, 0.3, 0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(out["mu"][1, 0], [0.4] * 5, atol=1e-6)
    np.testing.assert_allclose(out["g_mu"][1, 0], [0.1] * 5, atol=1e-6)
