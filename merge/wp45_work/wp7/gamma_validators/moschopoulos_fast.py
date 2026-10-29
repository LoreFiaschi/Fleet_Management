"""Prototype of a vectorised Moschopoulos series (planned step A7), compared with the worktree code.

The weight recursion stays O(n^2) in the number n of series terms, but each step is one
numpy dot product instead of a Python loop. The Gamma tails of the mixture are computed
in blocks with one vectorised call of scipy.special.gammaincc.
"""
import sys
import time
import types
from pathlib import Path

import numpy as np
from scipy.special import gammaincc

SRC = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
package = types.ModuleType("fleet_management")
package.__path__ = [str(SRC / "fleet_management")]
sys.modules["fleet_management"] = package
sys.modules["gurobipy"] = None
from fleet_management.degradation_model.gamma_utils.gamma_tail_bound import (  # noqa: E402
    moschopoulos_tail_probability,
)

BLOCK = 256


def fast(shapes, rates, threshold, tolerance=1e-12, max_series_terms=100_000):
    shapes = np.asarray(shapes, dtype=float)
    rates = np.asarray(rates, dtype=float)
    total_shape = float(shapes.sum())
    if np.allclose(rates, rates[0], rtol=1e-13, atol=0.0):
        value = float(gammaincc(total_shape, rates[0] * threshold))
        return value, value, 0.0, 1
    beta_ref = float(rates.max())
    ratios = rates / beta_ref
    initial_weight = float(np.exp(np.dot(shapes, np.log(ratios))))
    if initial_weight == 0.0:
        raise RuntimeError("initial weight underflowed")
    x = beta_ref * threshold
    q = 1.0 - ratios
    weights = np.empty(max_series_terms)
    k_gamma = np.empty(max_series_terms)      # index * gamma_index, index = 0..n-1
    weights[0] = initial_weight
    k_gamma[0] = 0.0
    power = np.ones_like(q)
    mass = initial_weight
    tail = initial_weight * float(gammaincc(total_shape, x))
    done = 1
    for k in range(1, max_series_terms):
        power = power * q
        k_gamma[k] = float(np.dot(shapes, power))          # k * gamma_k
        weights[k] = float(np.dot(k_gamma[1:k + 1], weights[k - 1::-1])) / k
        mass += weights[k]
        remaining = max(0.0, 1.0 - mass)
        if k + 1 - done == BLOCK or remaining <= tolerance:
            idx = np.arange(done, k + 1)
            tail += float(np.dot(weights[done:k + 1], gammaincc(total_shape + idx, x)))
            done = k + 1
        if remaining <= tolerance:
            return tail, min(1.0, tail + remaining), remaining, k + 1
    raise RuntimeError("no convergence")


cases = [([6.0, 6.0], [40.0, 20.0], 0.8), ([1.0, 3.0, 6.0, 6.0], [20.0, 40.0, 40.0, 20.0], 1.0)]
for repairs in (2, 4, 6):
    cases.append(([6.0] * (repairs + 1), [20.0 * 2.0 ** n for n in range(repairs + 1)], 0.8))
for shapes, rates, threshold in cases:
    t0 = time.perf_counter()
    new = fast(shapes, rates, threshold)
    t1 = time.perf_counter()
    if len(shapes) <= 5:
        old = moschopoulos_tail_probability(shapes, rates, threshold)
        t2 = time.perf_counter()
        print(f"n={len(shapes)} terms={new[3]:6d} new={new[1]:.15e} old={old.upper_bound:.15e} "
              f"diff={abs(new[1] - old.upper_bound):.1e} t_new={t1 - t0:.3f}s t_old={t2 - t1:.3f}s")
    else:
        print(f"n={len(shapes)} terms={new[3]:6d} new={new[1]:.15e} t_new={t1 - t0:.3f}s (old not run)")
for repairs in (8, 10):
    shapes, rates = [6.0] * (repairs + 1), [20.0 * 2.0 ** n for n in range(repairs + 1)]
    t0 = time.perf_counter()
    try:
        new = fast(shapes, rates, 0.8)
        print(f"{repairs} repairs: terms={new[3]} upper={new[1]:.6e} t_new={time.perf_counter() - t0:.2f}s")
    except RuntimeError as error:
        print(f"{repairs} repairs: {error} t_new={time.perf_counter() - t0:.2f}s")
