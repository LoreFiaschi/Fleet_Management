"""Replica of main's repeated_increment calibration, standard library only.

Source: main:src/fleet_management/degradation_model/gamma_utils/gamma_repeated_calibration.py
(maximum_safe_repetitions 196-232, calibrate_repeated_increment_tail_bound 235-330,
calibrate_gamma_cell_tail_bound 333-495, required_shape_for_tail 498-528) and the cap
A_max = required_shape_for_tail(eps, b*, tau) at main:src/fleet_management/degradation_model/base.py:606-610.
scipy's gamma.sf(t, a=A, scale=1/b) is Q(A, b t) here; brentq is replaced by bisection.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gamma_lib import tail, shape_for_tail  # noqa: E402


def required_shape_for_tail(p, rate, tau):
    if p == 0.0:
        return 0.0
    return shape_for_tail(min(p, 1.0 - 1e-16), rate, tau)


def maximum_safe_repetitions(shape, rate, tau, eps, maximum_count, tol=1e-12):
    safe = 0
    for m in range(1, maximum_count + 1):
        if tail(m * shape, rate, tau) <= eps + tol:
            safe = m
        else:
            break
    return safe


def calibrate_repeated(shape, rate, bstar, tau, eps, maximum_count, feas_tol=1e-9):
    if bstar > rate + 1e-12:
        raise ValueError("common_rate cannot exceed the original rate.")
    safe = maximum_safe_repetitions(shape, rate, tau, eps, maximum_count)
    checked = max(1, safe)
    per = []
    for m in range(1, checked + 1):
        exact = tail(m * shape, rate, tau)
        per.append(required_shape_for_tail(exact, bstar, tau) / m)
    bounded = max(per)
    if bounded > shape + feas_tol:
        raise RuntimeError("calibration increased alpha")
    slack = eps - tail(checked * bounded, bstar, tau)
    if safe > 0 and slack < -feas_tol:
        raise ValueError("no common-rate alpha satisfies the contract")
    return bounded, safe


def calibrate_cell(means, rates, tau, eps, T, occurrences=None, common_rate=None):
    """means, rates: one entry per increment type; each type occurs `occurrences` times
    (default T: time-independent mission data). Returns (bounded shapes, safe counts, b*, A_max)."""
    bstar = min(rates) if common_rate is None else common_rate
    occ = occurrences or [T] * len(means)
    out, safes = [], []
    for m, b, o in zip(means, rates, occ):
        a_b, s = calibrate_repeated(m * b, b, bstar, tau, eps, min(T, o))
        out.append(a_b)
        safes.append(s)
    Amax = required_shape_for_tail(eps, bstar, tau)
    return out, safes, bstar, Amax
