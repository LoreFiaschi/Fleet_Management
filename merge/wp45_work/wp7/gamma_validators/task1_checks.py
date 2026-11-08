"""Checks of tests/test_gamma_validators.py that need no Task 2 code.

1. every input the tests build loads in load_config without a warning;
2. every hand result round-trips through the three writers and solver.read_result with the
   same keys, dtypes and values (NaN equal) as the helper _result builds;
3. the expected values of the module agree with the plan's expected_values_output.txt.
"""
import importlib.util
import sys
import types
import warnings
from pathlib import Path

import numpy as np

SRC = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
TEST = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\tests\test_gamma_validators.py")
OUT = Path(__file__).resolve().parent / "tmp"
OUT.mkdir(exist_ok=True)

package = types.ModuleType("fleet_management")
package.__path__ = [str(SRC / "fleet_management")]
sys.modules["fleet_management"] = package
sys.modules["gurobipy"] = None

stub = types.ModuleType("pytest")        # no pytest on this computer; the module only needs the names
stub.mark = types.SimpleNamespace(parametrize=lambda *a, **k: (lambda f: f))
stub.raises = None
sys.modules["pytest"] = stub

spec = importlib.util.spec_from_file_location("test_gamma_validators", TEST)
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
from fleet_management.config import load_config  # noqa: E402
from fleet_management.solver import _save_results, read_result  # noqa: E402

warnings.simplefilter("error")

# 1. inputs
inputs = []
for case in "ABC":
    inputs.append((case, t._cfg(case)))
inputs += [
    ("A eps", t._cfg("A", epsilon=0.01)),
    ("B rho1", t._cfg("B", rho=1.0)),
    ("A C_R", t._cfg("A", C_R=[[2.0], [0.0]])),
    ("B C_R", t._cfg("B", C_R=[[2.0], [0.0]])),
    ("A no gamma", t._cfg("A", model="inverse_gaussian", eta=0.5)),
    ("A validation", t._cfg("A", validation={"mode": "stochastic", "repetitions": 500, "batch_size": 200})),
    ("A validation typo", t._cfg("A", validation={"repetition": 5})),
    ("data B C_R", load_config(t._data(repair_model="ardinf", C_R=[[2.0], [0.0]]))),
    ("data A", load_config(t._data())),
    ("MC1", load_config(t._mc_data())),
    ("MC3", load_config(t._mc_data(mu_0=0.6))),
    ("MC2", load_config(t._mc_data(repair_model="ard1", rho=1.0, mu_0=[[0.6], [0.0]], gamma_beta_0=4.0,
                                   mu=[[[0.5]], [[0.0]]]))),
    ("MC2 anchor 0", load_config({**t._mc_data(repair_model="ard1", rho=1.0, mu_0=[[0.6], [0.0]],
                                               gamma_beta_0=4.0, mu=[[[0.5]], [[0.0]]]), "anchor_mu_0": 0.0})),
    ("MC excl", load_config(t._mc_data(model=[["gamma"], ["inverse_gaussian"]], eta=0.5))),
]
for name, cfg in inputs:
    print("loads without warning:", name, "T =", cfg.T)

# 2. results through the writers and the reader
for case in "ABC":
    for suffix in (".yaml", ".json", ".h5"):
        path = OUT / f"check_{case}{suffix}"
        built = t._result(case)
        _save_results(built, path)
        saved = read_result(path)
        assert set(saved) == set(built), (case, suffix)
        for key in ("x", "m", "r", "mu", "z", "g_mu"):
            assert saved[key].dtype == built[key].dtype == np.float64, (case, suffix, key)
            assert saved[key].shape == built[key].shape
            assert np.array_equal(saved[key], built[key], equal_nan=True), (case, suffix, key)
        assert saved["status"] == "optimal" and int(saved["T"]) == 4
        assert [int(v) for v in saved["H"]] == [2, 2]
        assert saved["model"] == [["gamma"], ["gamma"]], saved["model"]
        print("round trip ok:", case, suffix)
# keep-NaN case (z NaN)
for suffix in (".h5", ".yaml"):
    built = t._result("B")
    built["z"][1, 0, :] = np.nan
    path = OUT / f"check_nan{suffix}"
    _save_results(built, path)
    saved = read_result(path)
    assert np.array_equal(saved["z"], built["z"], equal_nan=True) and np.isnan(saved["z"][1]).all()
    print("NaN round trip ok:", suffix)
# MC results
for suffix in (".yaml", ".h5"):
    built = t._mc_result(flights=((0, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1)), repairs=((0, 0, 0),))
    path = OUT / f"check_mc{suffix}"
    _save_results(built, path)
    saved = read_result(path)
    assert np.array_equal(saved["x"], built["x"]) and saved["mu"] is None and saved["z"] is None
    print("mc round trip ok:", suffix)

# 3. expected values
from scipy.special import gammaincc  # noqa: E402
assert abs(t._EXACT_UPPER["A"][0][0] - 0.0013837850247628773) < 1e-18
assert abs(t._EXACT_UPPER["A"][0][1] - 4.17052838324186e-09) < 1e-18
assert abs(t._EXACT_UPPER["B"][0][1] - 1.3507272658311747e-10) < 1e-20
assert abs(t._EXACT_UPPER["C"][0][0] - 0.004995412308307578) < 1e-17
assert abs(t._EXACT_UPPER["C"][0][3] - 4.50101664801213e-11) < 1e-20
assert abs(t._EXACT_UPPER["C"][1][0] - 3.203719780476998e-06) < 1e-18
assert abs(gammaincc(4, 4) - 0.43347012036670896) < 1e-15
assert abs(gammaincc(4, 1.6) - 0.9211865127702811) < 1e-15
assert abs(gammaincc(2.4, 4) - 0.1418232030653136) < 1e-15
assert abs(gammaincc(6, 20) - 7.190884052842894e-05) < 1e-17
print("expected values agree with expected_values_output.txt and plan_prototype_output.txt")
