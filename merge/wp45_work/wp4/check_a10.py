import importlib.util
import sys
import types
import warnings

import numpy as np

package = types.ModuleType("fleet_management")
package.__path__ = [r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src\fleet_management"]
sys.modules["fleet_management"] = package
sys.modules["pytest"] = types.ModuleType("pytest")
from fleet_management import config as c

spec = importlib.util.spec_from_file_location(
    "tc", r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\tests\test_config.py")
tc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tc)


def attempt(label, data):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            cfg = c.load_config(data)
            print(f"{label}: ok warnings={[str(w.message)[:70] for w in caught]}")
            return cfg
        except Exception as exc:
            print(f"{label}: {type(exc).__name__}: {exc}")


# tolerance mutation: without _le the tolerance case must fail
original = c._le
c._le = lambda a, b: np.asarray(a, dtype=float) <= np.asarray(b, dtype=float)
attempt("tolerance without _le (must fail)", tc._small("rainflow", "chernoff", mu_0=0.14, cgf_0=0.7))
c._le = original
attempt("tolerance with _le (must pass)", tc._small("rainflow", "chernoff", mu_0=0.14, cgf_0=0.7))

# L given with a string model
cfg = attempt("string model, L=2", tc._small("gaussian", L=2, rho=[0.5, 0.6], mu_0=[0.1, 0.1], v=[[0.0001], [0.0001]], mu=[[0.05], [0.05]]))
print(cfg.model.shape, cfg.rho.tolist())
# per-cell reliability_impl value at a linear-row cell
attempt("reliability_impl per cell at IG cell",
        tc._example(reliability_impl=[["tangent", "tangent"], ["pwl", None], ["exact", None]]))
# a scalar selector for a not-applicable key does not warn
attempt("bound_method scalar on example (rainflow only)", tc._example(bound_method="bernstein"))
# example with allow_replacement false
attempt("example allow_replacement false", tc._example(allow_replacement=False))
# example with all quadratic cells exact
attempt("example reliability_impl exact", tc._example(reliability_impl="exact"))
# example with penalty quadratic: needs exact
attempt("example penalty quadratic (tangent cell)", tc._example(penalty_type="quadratic"))
attempt("example penalty quadratic exact", tc._example(penalty_type="quadratic", reliability_impl="exact"))
# H scalar
attempt("H scalar", tc._example(H=3, mu_trans=None) if False else tc._small("gaussian", H=3))
# profile with None for mu entry
attempt("null entry in mu", tc._small("gaussian", mu=[[None]]) if False else tc._small("gaussian", mu=None))
# numpy arrays as input values
d = tc._small("gaussian")
d["mu"] = np.full((2, 1, 1, 2), 0.05)
attempt("numpy mu", d)
# F > M with F=3, M=2, L=2, mu (L, M, H2) etc is covered by the example
# example transitory-only gamma beta
attempt("gamma_beta_0 given", tc._small("gamma", gamma_beta_0=15.0, mu_0=0.1))
cfg = attempt("v_max_user given at G cell", tc._small("gaussian", v_max_user=0.01))
print(cfg.v_max_user.tolist())
