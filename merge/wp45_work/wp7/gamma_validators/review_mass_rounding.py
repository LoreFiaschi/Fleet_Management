"""Review of step C4: rounding error of the remaining mass of the series, float64 against 40 digits.

For one term repaired n times at rho = 0.5 beside a fresh one (the case of Q6), the
worktree's series stops at K terms with a float64 remaining mass. The same K weights are
recomputed with mpmath at 40 digits; the true remaining mass 1 - sum(w) is compared with
the float64 one. A float64 mass below the true one would make the upper bound too low.
"""
import sys
import time
import types
from pathlib import Path

import mpmath as mp
import numpy as np

SRC = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
package = types.ModuleType("fleet_management")
package.__path__ = [str(SRC / "fleet_management")]
sys.modules["fleet_management"] = package
sys.modules["gurobipy"] = None
from fleet_management.degradation_model.gamma_utils.gamma_tail_bound import (  # noqa: E402
    moschopoulos_tail_probability,
)

mp.mp.dps = 40
print("longdouble eps:", np.finfo(np.longdouble).eps, "float64 eps:", np.finfo(np.float64).eps)
for repairs in (2, 4, 6):
    shapes = [6.0] * (repairs + 1)
    rates = [20.0 * 2.0 ** n for n in range(repairs + 1)]
    tail = moschopoulos_tail_probability(shapes, rates, 0.8)
    K = tail.series_terms
    start = time.perf_counter()
    beta_ref = max(rates)
    q = [1 - mp.mpf(r) / beta_ref for r in rates]
    w = [mp.exp(sum(mp.mpf(a) * mp.log(mp.mpf(r) / beta_ref) for a, r in zip(shapes, rates)))]
    kg = [mp.mpf(0)]
    power = [mp.mpf(1)] * len(q)
    for k in range(1, K):
        power = [p * c for p, c in zip(power, q)]
        kg.append(mp.fsum(mp.mpf(a) * p for a, p in zip(shapes, power)))
        w.append(mp.fsum(kg[i] * w[k - i] for i in range(1, k + 1)) / k)
    true_remaining = 1 - mp.fsum(w)
    print(f"{repairs} repairs: K={K} float64 remaining={tail.remaining_mass:.6e} "
          f"true remaining={mp.nstr(true_remaining, 8)} "
          f"difference={mp.nstr(true_remaining - tail.remaining_mass, 4)} "
          f"({time.perf_counter() - start:.1f} s)")
