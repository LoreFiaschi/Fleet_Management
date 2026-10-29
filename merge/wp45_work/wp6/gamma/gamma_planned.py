"""
Gamma cell builder and the sharp cap ``A_1`` (spec, section "Gamma Model").

A Gamma cell tracks the mean ``mu`` of its damage and, at an ard1 cell, the anchor
``gmu``, with the generic rows of ``base``. Its reliability row is linear: on every
day, ``mu <= A_1(beta* tau, epsilon) / beta*`` (Eq. gm_row), where ``beta*`` is the
smallest Gamma rate among the terms of the cell with a positive mean (Eq.
gm_betastar). A cell in which no term has a rate gets ``mu <= tau``.

The cap functions use numpy and scipy only, so ``config.py`` and the Gamma
validators import them without Gurobi. The builder needs ``base``, which imports
``gurobipy``; without ``gurobipy`` the module still loads and registers no builder.
"""

from __future__ import annotations

import functools
import math

import numpy as np
from scipy.special import gammaincc

try:
    from fleet_management.degradation_model import base
except ModuleNotFoundError as error:      # no gurobipy: the cap functions still work
    if error.name != "gurobipy":
        raise
    base = None


# ===========================================================================
# The sharp cap A_1 (no Gurobi object)
# ===========================================================================
_GRID_POINTS = 401           # values of c in one grid (spec: at least 400)
_GRID_EXTENSIONS = 10        # downward extensions of the grid, by a factor 100 each
_GOLDEN_STEPS = 60           # golden-section steps in log c
_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0
_SHAPE_RTOL = 1e-12          # bisection of step 1
_CAP_RTOL = 1e-9             # bisection of step 3


def one_rate_cap(t: float, eps: float) -> float:
    """``A_hat``: the root of ``Q(A_hat, t) = eps`` (step 1 of the computation of ``A_1``).

    Bisection on the first argument of ``Q`` (``gammaincinv`` solves for the second
    argument and does not apply). It returns the lower end of its last interval,
    so ``Q(A_hat, t) <= eps``.
    """
    if not (math.isfinite(t) and t > 0.0 and 0.0 < eps < 1.0):
        raise ValueError(
            f"the Gamma cap needs t > 0 and 0 < epsilon < 1 (t={t!r}, epsilon={eps!r})"
        )
    lo, hi = 0.0, max(1.0, 2.0 * t)
    while gammaincc(hi, t) <= eps:
        lo, hi = hi, 2.0 * hi
    while hi - lo > _SHAPE_RTOL * hi:
        mid = 0.5 * (lo + hi)
        if gammaincc(mid, t) <= eps:
            lo = mid
        else:
            hi = mid
    return lo


def worst_one_term_tail(A: float, t: float) -> float:
    """``w(A)``: the maximum over ``c`` in ``(0, 1]`` of ``Q(A/c, t/c)`` (step 2).

    A logarithmic grid of 401 values of ``c`` in ``[min(1e-4, t/100), 1]``, extended
    downwards while its best point is its smallest ``c``, then a golden-section
    search in ``log c`` between the neighbours of the best point.
    """
    grid = np.geomspace(min(1e-4, t / 100.0), 1.0, _GRID_POINTS)
    values = gammaincc(A / grid, t / grid)
    best = int(np.argmax(values))
    extensions = 0
    while best == 0:
        if extensions == _GRID_EXTENSIONS:
            raise RuntimeError(
                f"the maximum of Q(A/c, t/c) lies below c = {grid[0]:.3g} "
                f"(A={A!r}, t={t!r}); the Gamma cap cannot be computed"
            )
        grid = np.geomspace(grid[0] / 100.0, grid[0], _GRID_POINTS)
        values = gammaincc(A / grid, t / grid)
        best = int(np.argmax(values))
        extensions += 1

    def tail(u):
        c = math.exp(u)
        return float(gammaincc(A / c, t / c))

    lo = math.log(grid[max(best - 1, 0)])
    hi = math.log(grid[min(best + 1, _GRID_POINTS - 1)])
    u1, u2 = hi - _GOLDEN * (hi - lo), lo + _GOLDEN * (hi - lo)
    f1, f2 = tail(u1), tail(u2)
    for _ in range(_GOLDEN_STEPS):
        if f1 < f2:
            lo, u1, f1 = u1, u2, f2
            u2 = lo + _GOLDEN * (hi - lo)
            f2 = tail(u2)
        else:
            hi, u2, f2 = u2, u1, f1
            u1 = hi - _GOLDEN * (hi - lo)
            f1 = tail(u1)
    return max(float(values[best]), f1, f2)


def _within_level(A: float, t: float, eps: float) -> bool:
    """``w(A) <= eps``; true at every ``A < t`` when ``eps >= 1/2``, because ``w < 1/2``."""
    return eps >= 0.5 or worst_one_term_tail(A, t) <= eps


@functools.lru_cache(maxsize=None)
def sharp_cap(t: float, eps: float) -> float:
    """``A_1(t, eps)``, the sharp cap (step 3 of the computation of ``A_1``).

    ``A_hat`` if ``A_hat < t`` and ``w(A_hat) <= eps``. Otherwise a bisection on ``A``
    in ``[eps t, min(A_hat, t))`` to the relative tolerance 1e-9; it returns the
    lower end of its last interval, so ``w(A_1) <= eps``. One value per ``(t, eps)``
    is computed and kept.
    """
    a_hat = one_rate_cap(t, eps)
    if a_hat < t and _within_level(a_hat, t, eps):
        return a_hat
    lo, hi = eps * t, min(a_hat, t)
    while hi - lo > _CAP_RTOL * hi:
        mid = 0.5 * (lo + hi)
        if _within_level(mid, t, eps):
            lo = mid
        else:
            hi = mid
    return lo


def smallest_rate(mu_inc, beta_inc, mu_0: float, beta_0: float, mu_new: float,
                  beta_new: float, allow_replacement: bool) -> float | None:
    """``beta*`` of one Gamma cell (Eq. gm_betastar); ``None`` if no term has a rate.

    ``mu_inc`` and ``beta_inc`` hold the mean and the rate of every mission and day
    of both phases, after the phase lookup (two arrays of one shape). ``beta_0`` and
    ``beta_new`` are NaN where not given; ``beta_new`` enters only if replacement
    is allowed.
    """
    mu_inc = np.asarray(mu_inc, dtype=float)
    rates = np.asarray(beta_inc, dtype=float)[mu_inc > 0.0].tolist()
    if mu_0 > 0.0 and not math.isnan(beta_0):
        rates.append(float(beta_0))
    if allow_replacement and mu_new > 0.0 and not math.isnan(beta_new):
        rates.append(float(beta_new))
    return min(rates) if rates else None


def mean_cap(beta_star: float | None, tau: float, eps: float) -> float:
    """Right-hand side of Eq. gm_row: ``A_1(beta* tau, eps) / beta*``; ``tau`` without rates."""
    if beta_star is None:
        return float(tau)
    return sharp_cap(beta_star * tau, eps) / beta_star


# ===========================================================================
# Cell builder
# ===========================================================================
class GammaCellBuilder:
    """Rows of the Gamma cells: tracked mean and anchor, cap row, repair cost."""

    name = "gamma"

    def prepare(self, ctx: base.FleetModel, cfg, cells, opts: dict) -> None:
        """Anchors and selectors of ``cells``; ``beta*`` and the cap of each cell.

        ``ctx.extras["gamma"]`` holds the dicts ``beta_star`` and ``cap``, keyed by
        ``(i, l)``. The rows are linear, so ``NonConvex`` is not set.
        """
        base.add_state_vars(ctx, "mu", cells)
        base.add_selector_vars(ctx, cells)
        M, T = ctx.M, ctx.T
        beta_star, cap = {}, {}
        for i, l in cells:
            mu_inc = [[ctx.increment("mu", i, j, l, k) for k in range(T)]
                      for j in range(1, M + 1)]
            beta_inc = [[ctx.increment("gamma_beta", i, j, l, k) for k in range(T)]
                        for j in range(1, M + 1)]
            beta_star[i, l] = smallest_rate(
                mu_inc, beta_inc,
                float(ctx.initial["mu"][i, l]), float(cfg.gamma_beta_0[i, l]),
                float(ctx.replacement["mu"][i, l]), float(cfg.gamma_beta_new[i, l]),
                ctx.allow_replacement,
            )
            cap[i, l] = mean_cap(beta_star[i, l], float(ctx.tau[i, l]), float(ctx.eps[i, l]))
        ctx.extras["gamma"] = {"beta_star": beta_star, "cap": cap}

    def add_cell(self, ctx: base.FleetModel, i: int, l: int) -> None:
        """Selector, mean, anchor, loop, cap and repair-cost rows of cell ``(i, l)``."""
        base.add_selector_rows(ctx, i, l)
        increments = [base.day_increment(ctx, i, l, k, "mu") for k in range(ctx.T)]
        base.add_quantity_rows(ctx, i, l, "mu", increments)    # Eqs. gm_mu_carry to gm_anchor_keep
        base.add_upper_cap_rows(ctx, i, l, "mu", ctx.extras["gamma"]["cap"][i, l],
                                name="rel")                    # Eq. gm_row
        base.add_repair_cost_rows(ctx, i, l)


if base is not None:
    base.register_cell_builder("gamma", GammaCellBuilder())
