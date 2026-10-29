"""Check that the test inputs of the plan load and that hand results round-trip through the writers."""
import sys
import types
import warnings
from pathlib import Path

import numpy as np

SRC = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
package = types.ModuleType("fleet_management")
package.__path__ = [str(SRC / "fleet_management")]
sys.modules["fleet_management"] = package
sys.modules["gurobipy"] = None
from fleet_management.config import load_config, read_input, _lookup  # noqa: E402
from fleet_management.solver import _save_results  # noqa: E402
from fleet_management.degradation_model.gamma import mean_cap, smallest_rate  # noqa: E402

OUT = Path(__file__).resolve().parent / "tmp"
OUT.mkdir(exist_ok=True)
warnings.simplefilter("error")


def data(**changes):
    d = {"F": 2, "M": 1, "H": [2, 2], "model": "gamma", "repair_model": "ard1", "tau": 1.0,
         "epsilon": 0.1, "rho": 0.5, "mu_0": 0.2, "replacement_mu": 0.1, "mu": [[[0.3]], [[0.0]]],
         "gamma_beta": 20.0, "C_M": 1.0, "C_R": 2.0, "C_rep": 3.0, "C_D": 0.0}
    d.update(changes)
    return d


def mc(**changes):
    d = {"F": 2, "M": 1, "H": [1, 1], "model": "gamma", "repair_model": "ardinf", "tau": 1.0,
         "epsilon": 0.1, "rho": 0.5, "mu_0": 0.0, "mu": 0.5, "gamma_beta": 4.0,
         "C_M": 1.0, "C_R": 2.0, "C_rep": 3.0, "C_D": 0.0}
    d.update(changes)
    return d


def caps(cfg):
    mu_inc = _lookup(cfg.mu, cfg.mu_trans, cfg.H1, cfg.H2)
    beta_inc = _lookup(cfg.gamma_beta, cfg.gamma_beta_trans, cfg.H1, cfg.H2)
    out = {}
    for i in range(cfg.F):
        for l in range(cfg.L):
            if str(cfg.model[i, l]) != "gamma":
                continue
            b = smallest_rate(mu_inc[i, l], beta_inc[i, l], float(cfg.mu_0[i, l]), float(cfg.gamma_beta_0[i, l]),
                              float(cfg.replacement_mu[i, l]), float(cfg.gamma_beta_new[i, l]),
                              bool(cfg.allow_replacement))
            out[i, l] = (b, mean_cap(b, float(cfg.tau[i, l]), float(cfg.epsilon[i, l])))
    return out


for name, d in [("A", data()), ("B", data(repair_model="ardinf")),
                ("C", data(gamma_beta_0=20.0, anchor_mu_0=0.05, gamma_beta_new=30.0)),
                ("A eps .01", data(epsilon=0.01)), ("B rho 1", data(repair_model="ardinf", rho=1.0)),
                ("A C_R cell", data(C_R=[[2.0], [0.0]])),
                ("A validation", data(validation={"mode": "stochastic", "repetitions": 500, "batch_size": 200})),
                ("no gamma", data(model="inverse_gaussian", eta=0.5)),
                ("MC1", mc()), ("MC3", mc(mu_0=0.6)),
                ("MC2", mc(repair_model="ard1", rho=1.0, mu_0=[[0.6], [0.0]], gamma_beta_0=4.0,
                           mu=[[[0.5]], [[0.0]]])),
                ("MC2 anchor 0", mc(repair_model="ard1", rho=1.0, mu_0=[[0.6], [0.0]], gamma_beta_0=4.0,
                                    mu=[[[0.5]], [[0.0]]], anchor_mu_0=0.0)),
                ("MC excl", mc(model=[["gamma"], ["inverse_gaussian"]], eta=0.5))]:
    cfg = load_config(d)
    text = {k: v for k, v in caps(cfg).items()} if "gamma" in np.asarray(cfg.model) else "no gamma"
    print(name, "anchor", None if cfg.anchor_mu_0 is None else cfg.anchor_mu_0.ravel().tolist(),
          "beta0", cfg.gamma_beta_0.ravel().tolist(), "opts.validation", cfg.options.get("validation"),
          "caps", text)

# hand result A through the three writers and the reader
x = np.zeros((2, 2, 4))
for i, j, k in ((0, 1, 0), (0, 0, 1), (0, 1, 2), (0, 0, 3), (1, 1, 1), (1, 1, 3)):
    x[i, j, k] = 1
m = np.zeros((2, 1, 4)); m[0, 0, 1] = 1
r = np.zeros((2, 1, 4)); r[0, 0, 3] = 1
z = np.zeros((2, 1, 4)); z[0, 0, 1] = 0.15; z[1, 0, :] = np.nan
result = {"status": "optimal", "F": 2, "M": 1, "L": 1, "H1": 2, "H2": 2, "T": 4, "H": [2, 2],
          "model": [["gamma"], ["gamma"]], "x": x, "m": m, "r": r,
          "mu": np.array([[[0.5, 0.35, 0.65, 0.1]], [[0.2] * 4]]), "z": z,
          "g_mu": np.array([[[0.2, 0.35, 0.35, 0.1]], [[0.2] * 4]])}
for suffix in (".yaml", ".json", ".h5"):
    path = OUT / f"result{suffix}"
    _save_results(result, path)
    back = read_input(path)
    zz = np.asarray(back["z"], dtype=float)
    print(suffix, sorted(back), "T", back["T"], type(back["T"]).__name__, "model", back["model"],
          "z nan count", int(np.isnan(zz).sum()), "H", back["H"])
