"""
Pytest configuration for fleet_management tests.

Adds the project root to sys.path so that 'fleet_management' can be imported
even without an editable install, and provides shared fixtures/helpers used
across the test suite.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the package source is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

INPUT_DIR = PROJECT_ROOT / "input"


def fake_result(F=2, M=1, L=2, H=3, seed=0, model=None, objective=None):
    """Hand-build a result dict shaped exactly like solve() would produce.

    Used to test I/O (save/read/plot) without invoking Gurobi at all: no
    assignment/reliability constraints are actually satisfied by this data,
    it is only shaped correctly for the round-trip and plotting code paths.
    """
    two_h = 2 * H
    rng = np.random.default_rng(seed)
    if model is None:
        model = [
            ["gaussian" if l % 2 == 0 else "inverse_gaussian" for l in range(L)]
            for _ in range(F)
        ]
    tau = np.ones((F, L))
    mu = rng.uniform(0, 0.9, size=(F, L, two_h))
    x = np.zeros((F, M + 1, two_h))
    x_m = np.zeros((F, L, two_h))
    x_r = np.zeros((F, L, two_h))
    for i in range(F):
        x[i, 1, 0] = 1
    x_m[0, 0, 1 % two_h] = 1
    x[0, 0, 1 % two_h] = 1
    x_r[0, L - 1, 2 % two_h] = 1
    x[0, 0, 2 % two_h] = 1
    return {
        "status": "optimal",
        "objective": objective if objective is not None else 12.34 + H,
        "H": H, "F": F, "M": M, "L": L,
        "model": model,
        "tail_bound": [[None] * L for _ in range(F)],
        "tau": tau.tolist(),
        "x": x, "x_m": x_m, "x_r": x_r,
        "mu": mu, "v": np.full((F, L, two_h), np.nan),
        "u": 0.5, "z": np.zeros((F, L, two_h)),
    }


def fake_shared(result, H=None):
    """Minimal 'shared' params dict accepted by solver._save_results/_write_hdf5_single."""
    F, L = result["F"], result["L"]
    return {
        "H": H if H is not None else result["H"],
        "epsilon": 0.01,
        "tau": np.ones((F, L)),
        "rho": np.full((F, L), 0.7),
        "model_assignment": np.array(result["model"], dtype=object),
    }


@pytest.fixture
def make_fake_result():
    return fake_result


@pytest.fixture
def make_fake_shared():
    return fake_shared


def tiny_problem_data(model_grid, F, M, L, H, tau=1.0, mu_inc=0.08, v_inc=0.0004,
                       eta=0.15, rho=0.7, mu_0=0.1, v_0=0.0004, C_M=1.0, C_D=1.0,
                       C_R=1.0, C_rep=5.0, sigma=0.03, beta=0.5, alpha_inc=0.03,
                       tail_bound="cantelli", b_inc=0.3, b_0=0.3, b_new=0.0,
                       maintenance_type=None, **overrides):
    """Build a minimal in-memory problem dict (the same shape solver.solve()
    would read from a YAML file) for direct use with
    ``solver._parse_and_validate`` / ``solver._solve_single_horizon`` in tests
    that need a real Gurobi solve. ``maintenance_type`` defaults to all-ARD1;
    pass a full (F, L) grid to exercise ARA1 (wiener/gamma only).
    """
    flat_models = [m for row in model_grid for m in row]
    needs_mu = any(m != "gamma" for m in flat_models)
    needs_v0 = any(m in ("gaussian", "wiener", "rainflow") for m in flat_models)
    needs_v_inc = any(m in ("gaussian", "rainflow") for m in flat_models)
    needs_eta = any(m == "inverse_gaussian" for m in flat_models)
    needs_sigma = any(m == "wiener" for m in flat_models)
    needs_gamma = any(m == "gamma" for m in flat_models)
    needs_rainflow = any(m == "rainflow" for m in flat_models)

    if maintenance_type is None:
        maintenance_type = [["ARD1"] * L for _ in range(F)]

    data = {
        "F": F, "H": H, "M": M, "L": L, "tau": tau, "epsilon": 0.01,
        "C_M": C_M, "C_D": C_D, "penalty_type": "inf_norm", "formulation": "exact",
        "n_workers": 1, "warm_start": False, "verbose": 0,
        "model": model_grid,
        "maintenance_type": maintenance_type,
        "rho": rho, "C_R": C_R, "C_rep": C_rep,
        "mu_0": [[mu_0] * L for _ in range(F)],
        "mu_new": [[0.0] * L for _ in range(F)],
    }
    if needs_mu:
        data["mu"] = [
            [[mu_inc if model_grid[i][l] != "gamma" else None for l in range(L)] for _ in range(M)]
            for i in range(F)
        ]
    if needs_v0:
        v0_models = ("gaussian", "wiener", "rainflow")
        data["v_0"] = [
            [v_0 if model_grid[i][l] in v0_models else None for l in range(L)] for i in range(F)
        ]
        data["v_new"] = [
            [0.0 if model_grid[i][l] in v0_models else None for l in range(L)] for i in range(F)
        ]
    if needs_v_inc:
        v_inc_models = ("gaussian", "rainflow")
        data["v"] = [
            [[v_inc if model_grid[i][l] in v_inc_models else None for l in range(L)] for _ in range(M)]
            for i in range(F)
        ]
    if needs_eta:
        data["eta"] = [
            [eta if model_grid[i][l] == "inverse_gaussian" else None for l in range(L)]
            for i in range(F)
        ]
    if needs_sigma:
        data["sigma"] = [
            [sigma if model_grid[i][l] == "wiener" else None for l in range(L)] for i in range(F)
        ]
    if needs_gamma:
        data["beta"] = [
            [beta if model_grid[i][l] == "gamma" else None for l in range(L)] for i in range(F)
        ]
        data["alpha_inc"] = [
            [[alpha_inc if model_grid[i][l] == "gamma" else None for l in range(L)] for _ in range(M)]
            for i in range(F)
        ]
    if needs_rainflow:
        data["tail_bound"] = tail_bound
        data["b_inc"] = [
            [[b_inc if model_grid[i][l] == "rainflow" else None for l in range(L)] for _ in range(M)]
            for i in range(F)
        ]
        data["b_0"] = [
            [b_0 if model_grid[i][l] == "rainflow" else None for l in range(L)] for i in range(F)
        ]
        data["b_new"] = [
            [b_new if model_grid[i][l] == "rainflow" else None for l in range(L)] for i in range(F)
        ]
    data.update(overrides)
    return data


@pytest.fixture
def make_tiny_problem():
    return tiny_problem_data
