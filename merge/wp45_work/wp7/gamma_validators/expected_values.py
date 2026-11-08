"""Expected values for tests/test_gamma_validators.py (WP7, part gamma_validators).

Independent of the validators: exact tails by scipy (one rate) and by the quadrature
reference of examples/regression/gamma_quadrature_reference.py (several rates), which
does not use the Moschopoulos code. The Moschopoulos values of the worktree module are
printed beside them for comparison only. The caps come from degradation_model/gamma.py.
"""
import importlib.util
import math
import sys
import time
import types
from pathlib import Path

import numpy as np
from scipy.special import gammaincc
from scipy.stats import beta as beta_distribution

WT = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge")
SRC = WT / "src"

# load the package without __init__.py (no gurobipy here)
package = types.ModuleType("fleet_management")
package.__path__ = [str(SRC / "fleet_management")]
sys.modules["fleet_management"] = package
sys.modules["gurobipy"] = None
from fleet_management.degradation_model import gamma  # noqa: E402
from fleet_management.degradation_model.gamma_utils.gamma_tail_bound import (  # noqa: E402
    moschopoulos_tail_probability,
)

spec = importlib.util.spec_from_file_location(
    "gamma_quadrature_reference", WT / "examples" / "regression" / "gamma_quadrature_reference.py")
quad_ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quad_ref)


def quad_tail(shapes, rates, threshold):
    tail, error = quad_ref.quadrature_gamma_sum_tail(np.asarray(shapes, float),
                                                     np.asarray(rates, float), threshold)
    return float(tail), float(error)


def mosch(shapes, rates, threshold):
    t = moschopoulos_tail_probability(shapes, rates, threshold)
    return t.estimate, t.upper_bound, t.remaining_mass, t.series_terms


print("== caps")
for b in (20.0, 10.0, 4.0):
    print(f"mean_cap({b}, tau=1, eps=0.1) = {gamma.mean_cap(b, 1.0, 0.1):.12g}")

print("\n== case A (hand case of test_gamma.py, ard1; train 0)")
print("day0 tail Q(6,16)          =", repr(float(gammaincc(6.0, 16.0))))
print("day1 tail Q(6,32)          =", repr(float(gammaincc(6.0, 32.0))))
qa = quad_tail([6.0, 6.0], [40.0, 20.0], 0.8)
print("day2 tail G(6,40)+G(6,20)>0.8 quad =", repr(qa), " mosch =", mosch([6.0, 6.0], [40.0, 20.0], 0.8))

print("\n== case B (hand case, ardinf; train 0)")
print("day1 tail Q(6,36)          =", repr(float(gammaincc(6.0, 36.0))))
qb = quad_tail([6.0, 6.0], [40.0, 20.0], 0.9)
print("day2 tail G(6,40)+G(6,20)>0.9 quad =", repr(qb), " mosch =", mosch([6.0, 6.0], [40.0, 20.0], 0.9))

print("\n== case C (ard1, gamma_beta_0=20, anchor_mu_0=0.05, gamma_beta_new=30, mu_0=0.2)")
# initial: frozen G(1,20) (mean 0.05), active G(3,20) (mean 0.15); day 0 mission G(6,20)
print("day0 G(10,20)>1 Q(10,20) =", repr(float(gammaincc(10.0, 20.0))))
# day 1 repair rho=0.5: active -> G(3,40), G(6,40), frozen; D = G(1,20) + G(9,40)
c1 = quad_tail([1.0, 9.0], [20.0, 40.0], 1.0)
print("day1 G(1,20)+G(9,40)>1 quad =", repr(c1), " mosch =", mosch([1.0, 3.0, 6.0], [20.0, 40.0, 40.0], 1.0))
# day 2 mission G(6,20): D = G(7,20) + G(9,40)
c2 = quad_tail([7.0, 9.0], [20.0, 40.0], 1.0)
print("day2 G(7,20)+G(9,40)>1 quad =", repr(c2),
      " mosch =", mosch([1.0, 3.0, 6.0, 6.0], [20.0, 40.0, 40.0, 20.0], 1.0))
print("day3 G(3,30)>1 Q(3,30) =", repr(float(gammaincc(3.0, 30.0))))
print("train1 G(4,20)>1 Q(4,20) =", repr(float(gammaincc(4.0, 20.0))))

print("\n== Monte Carlo probabilities")
print("MC1 day0 Q(2,4) =", repr(float(gammaincc(2.0, 4.0))), " day1 Q(4,4) =", repr(float(gammaincc(4.0, 4.0))))
print("MC2 Q(2.4,4)    =", repr(float(gammaincc(2.4, 4.0))))
print("MC3 day0 Q(2,1.6) =", repr(float(gammaincc(2.0, 1.6))), " day1 Q(4,1.6) =", repr(float(gammaincc(4.0, 1.6))))
print("Clopper-Pearson f=0 n=100 0.95 =", repr(float(beta_distribution.ppf(0.95, 1, 100))),
      " closed form =", repr(1 - 0.05 ** (1 / 100)))
print("Clopper-Pearson f=3 n=100 0.95 =", repr(float(beta_distribution.ppf(0.95, 4, 97))))

print("\n== cost of the Moschopoulos series after repeated ardinf repairs (rho = 0.5)")
for repairs in (2, 4, 6, 8):
    shapes = [6.0] * (repairs + 1)
    rates = [20.0 * 2.0 ** n for n in range(repairs + 1)]
    start = time.perf_counter()
    try:
        est = mosch(shapes, rates, 0.8)
        print(f"{repairs:2d} repairs: terms={est[3]:6d}  seconds={time.perf_counter() - start:.2f}")
    except RuntimeError as error:
        print(f"{repairs:2d} repairs: RuntimeError {error}  seconds={time.perf_counter() - start:.2f}")
