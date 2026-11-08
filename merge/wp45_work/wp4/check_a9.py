import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

package = types.ModuleType("fleet_management")
package.__path__ = [r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src\fleet_management"]
sys.modules["fleet_management"] = package
pytest_stub = types.ModuleType("pytest")
sys.modules["pytest"] = pytest_stub
from fleet_management import config as c

spec = importlib.util.spec_from_file_location(
    "tc", r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\tests\test_config.py")
tc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tc)


def cap_at(model, bound, mu, **changes):
    cfg = c.load_config(tc._small(model, bound, **changes))
    cfg.replacement_mu = np.full_like(cfg.tau, mu)
    m = c._masks(cfg.model, cfg.bound_method, cfg.repair_model, cfg.reliability_impl)
    chi, c2, c1, side = c._quad_coefficients(cfg, m)
    cap = c._cap(cfg.replacement_mu, cfg.tau, c2, c1, cfg.reliability_impl, cfg.tangent_ref, cfg.pwl_points)
    return cap[0, 0], side[0, 0]


print("G exact       ", cap_at("gaussian", None, 0.9), "expect 0.0060887")
print("G tangent 0.4 ", cap_at("gaussian", None, 0.9, reliability_impl="tangent", tangent_ref=0.4), "expect -0.14613")
print("W pwl 6       ", cap_at("wiener", None, 0.95, reliability_impl="pwl", pwl_points=6), "expect 0.00084566")
print("Ca            ", cap_at("rainflow", "cantelli", 0.9), "expect 0.0011111")
print("Ho            ", cap_at("rainflow", "hoeffding", 0.9), "expect 0.0086859")
print("Be side       ", cap_at("rainflow", "bernstein", 0.97), "expect side 0.96162")
print("W pwl at boundary mu=0.5, P=2", cap_at("wiener", None, 0.5, reliability_impl="pwl", pwl_points=2))
print("W pwl mu>tau", cap_at("wiener", None, 1.5, reliability_impl="pwl", pwl_points=2))
