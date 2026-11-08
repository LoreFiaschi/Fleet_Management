"""
Inverse Gaussian cell builder (spec, section "Inverse Gaussian Model").

A cell tracks the mean ``mu``; under the law ``ard1`` also its anchor
``g_mu``. The day map is Eq. ig_map. The reliability row is the linear cap
Eq. ig_row, ``mu <= mu_bar = tau / E[Z | Z > q]``, with ``Z ~ IG(1, 1/eta)``
and ``q`` its ``(1 - epsilon)``-quantile; ``reliability_impl`` is ignored.

The cap ``mu_bar`` lives in this module: ``ig_mean_cap`` and the functions
below it need only numpy and scipy. ``config.py`` imports ``ig_mean_cap``
inside the check of the replacement state, so the module also loads without
gurobipy; ``base`` is then ``None`` and no builder is registered.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import log_ndtr

try:
    from fleet_management.degradation_model import base
except ModuleNotFoundError as error:      # no gurobipy: the cap functions still work
    if error.name != "gurobipy":
        raise
    base = None


# ===========================================================================
# Mean cap
# ===========================================================================
def _ig_unit_log_cdf(x: float, eta: float) -> float:
    """ln F(x) of Z ~ IG(1, 1/eta), x > 0 (spec, "Inverse Gaussian Model")."""
    r = 1.0 / math.sqrt(eta * x)
    return float(np.logaddexp(log_ndtr(r * (x - 1.0)), 2.0 / eta + log_ndtr(-r * (x + 1.0))))


def _ig_unit_log_sf(x: float, eta: float) -> float:
    """ln (1 - F(x)) of Z ~ IG(1, 1/eta), x > 0, without the cancellation of 1 - F."""
    r = 1.0 / math.sqrt(eta * x)
    first = float(log_ndtr(-r * (x - 1.0)))
    second = 2.0 / eta + float(log_ndtr(-r * (x + 1.0)))
    return first + math.log1p(-math.exp(second - first))


def _ig_unit_quantile(eta: float, epsilon: float) -> float:
    """The (1 - epsilon)-quantile q of Z ~ IG(1, 1/eta): Pr(Z > q) = epsilon.

    Root of ln Pr(Z > e^u) = ln epsilon in u = ln q, by brentq on [-50, ln b], with
    b = 1 + sqrt(eta (2/epsilon - 1)): Var Z = eta, so Pr(Z > b) <= epsilon/2 (Cantelli).
    """
    target = math.log(epsilon)

    def gap(u):
        return _ig_unit_log_sf(math.exp(u), eta) - target

    lo, hi = -50.0, math.log(1.0 + math.sqrt(eta * (2.0 / epsilon - 1.0)))
    if not (gap(lo) > 0.0 > gap(hi)):
        raise ValueError(f"no quantile bracket for eta={eta}, epsilon={epsilon}")
    return math.exp(brentq(gap, lo, hi, xtol=1e-14, rtol=1e-15, maxiter=200))


def ig_mean_cap(eta: float, tau: float, epsilon: float) -> float:
    """Mean cap mu_bar = tau / E[Z | Z > q] of an inverse Gaussian cell (spec, Eq. ig_row).

    E[Z; Z > q] = F(1/q), so mu_bar = tau * epsilon / F(1/q), computed in log space.
    It lives in this module and runs without Gurobi: ``config.py`` imports it inside the
    check of the replacement state, and the builder calls it once per cell.
    """
    q = _ig_unit_quantile(eta, epsilon)
    return tau / q


# ===========================================================================
# Cell builder
# ===========================================================================
class InverseGaussianCellBuilder:
    """Rows of the cells with ``model = inverse_gaussian``."""

    name = "inverse_gaussian"

    def prepare(self, ctx: base.FleetModel, cfg, cells, opts: dict) -> None:
        """The mean anchors and selectors; the cap ``mu_bar`` of every cell."""
        base.add_state_vars(ctx, "mu", cells)
        base.add_selector_vars(ctx, cells)
        ctx.extras[self.name] = {
            "mu_bar": {(i, l): ig_mean_cap(float(cfg.eta[i, l]), float(ctx.tau[i, l]),
                                           float(ctx.eps[i, l]))
                       for i, l in cells},
        }

    def add_cell(self, ctx: base.FleetModel, i: int, l: int) -> None:
        """Dynamics, replacement, anchor and loop rows; the cap row; repair cost."""
        T = ctx.T
        base.add_selector_rows(ctx, i, l)
        # Eqs. ig_mu_carry ... ig_anchor_keep
        base.add_quantity_rows(ctx, i, l, "mu",
                               [base.day_increment(ctx, i, l, k, "mu") for k in range(T)])
        # Eq. ig_row
        base.add_upper_cap_rows(ctx, i, l, "mu", ctx.extras[self.name]["mu_bar"][i, l],
                                name="rel")
        base.add_repair_cost_rows(ctx, i, l)


if base is not None:
    base.register_cell_builder("inverse_gaussian", InverseGaussianCellBuilder())
