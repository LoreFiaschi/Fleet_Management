"""
Rainflow cell builder (spec, "Rainflow Model").

A rainflow cell tracks the mean ``mu`` and, by its tail bound
``bound_method``, at most one more quantity: the variance ``v`` (Cantelli,
Bernstein), the Hoeffding budget ``R`` (Hoeffding) or the Chernoff quantity
``K`` (Chernoff). Markov tracks the mean only. Under the law ``ard1`` every
tracked quantity has an anchor. Both repair laws are accepted for every bound.

The dynamics, replacement, anchor, loop and repair-cost rows come from the
helpers of ``base``. This module adds the reliability row of the bound, in the
form ``reliability_impl`` for a quadratic row (Cantelli, Hoeffding, Bernstein)
and in its exact form for a linear row (Markov, Chernoff), and the optional row
``v <= v_max_user`` at Cantelli and Bernstein cells.
"""

from __future__ import annotations

import math

import numpy as np

from fleet_management.degradation_model import base


# ===========================================================================
# Tail bounds
# ===========================================================================
# bound -> (tracked quantity besides mu, increment profile, squared) or None
BOUND_QUANTITY = {
    "markov": None,
    "cantelli": ("v", "v", False),
    "hoeffding": ("R", "support", True),
    "bernstein": ("v", "v", False),
    "chernoff": ("K", "cgf", False),
}
QUADRATIC_BOUNDS = ("cantelli", "hoeffding", "bernstein")


def row_constants(bound: str, tau: float, eps: float, b: float = math.nan,
                  s: float = math.nan) -> dict:
    """Constants of the reliability row of a rainflow cell.

    Quadratic row (Eq. a_quad): ``q``, ``c2``, ``c1``, ``mu_side`` for
    ``chi <= c2 (tau - mu)^2 + c1 (tau - mu)`` and ``mu <= mu_side``, with
    ``chi = state_vars[q]``. Linear row: ``q`` and ``cap`` for
    ``state_vars[q] <= cap``. ``b`` is the Bernstein support constant and ``s``
    the Chernoff parameter; each is read only by its bound.
    """
    kappa = math.log(1.0 / eps)
    if bound == "markov":
        return {"q": "mu", "cap": eps * tau}                                   # Eq. rf_markov
    if bound == "cantelli":
        return {"q": "v", "c2": eps / (1.0 - eps), "c1": 0.0, "mu_side": tau}  # Eq. rf_cantelli
    if bound == "hoeffding":
        return {"q": "R", "c2": 2.0 / kappa, "c1": 0.0, "mu_side": tau}        # Eq. rf_hoeffding
    if bound == "bernstein":
        c = kappa * b / 3.0                                                    # Eq. rf_bsupport
        return {"q": "v", "c2": 1.0 / (2.0 * kappa), "c1": -b / 3.0,
                "mu_side": tau - c}                                            # Eq. rf_bernstein
    if bound == "chernoff":
        return {"q": "K", "cap": s * tau + math.log(eps)}                      # Eq. rf_chernoff
    raise ValueError(f"unknown bound_method {bound!r}")


# ===========================================================================
# Cell builder
# ===========================================================================
class RainflowCellBuilder:
    """Rainflow implementation of the ``base.CellBuilder`` interface."""

    name = "rainflow"

    def prepare(self, ctx: base.FleetModel, cfg, cells, opts: dict) -> None:
        """Variables of the rainflow cells, their row constants and ``NonConvex``.

        The anchors of ``mu`` (ard1 cells); ``v``, ``R`` or ``K`` with its anchor,
        by bound; the selector ``n`` (indicator cells); the segment binaries
        (quadratic rows in the form ``pwl``). ``NonConvex = 2`` if a quadratic row
        is built in the form ``exact``. The row constants of each cell go to
        ``ctx.extras["rainflow_rows"]``.
        """
        cells = list(cells)
        by_bound = {bound: [] for bound in BOUND_QUANTITY}
        for i, l in cells:
            bound = str(ctx.bound_of[i, l])
            if bound not in BOUND_QUANTITY:
                raise ValueError(f"unknown bound_method {bound!r} at cell (i={i}, l={l})")
            by_bound[bound].append((i, l))

        base.add_state_vars(ctx, "mu", cells)
        for bound, own in by_bound.items():
            if BOUND_QUANTITY[bound] is not None:
                base.add_state_vars(ctx, BOUND_QUANTITY[bound][0], own)
        base.add_selector_vars(ctx, cells)
        quadratic = [cell for bound in QUADRATIC_BOUNDS for cell in by_bound[bound]]
        base.add_pwl_vars(ctx, quadratic)
        if any(str(ctx.impl_of[i, l]) == "exact" for i, l in quadratic):
            ctx.model.Params.NonConvex = 2

        support_const = np.asarray(cfg.support_const, dtype=float)
        s_chernoff = np.asarray(cfg.s_chernoff, dtype=float)
        v_max_user = np.asarray(cfg.v_max_user, dtype=float)
        rows = {}
        for i, l in cells:
            bound = str(ctx.bound_of[i, l])
            b = float(support_const[i, l]) if bound == "bernstein" else math.nan
            s = float(s_chernoff[i, l]) if bound == "chernoff" else math.nan
            if bound == "bernstein" and not math.isfinite(b):
                raise ValueError(f"support_const is not finite at cell (i={i}, l={l})")
            if bound == "chernoff" and not math.isfinite(s):
                raise ValueError(f"s_chernoff is not finite at cell (i={i}, l={l})")
            constants = row_constants(bound, float(ctx.tau[i, l]), float(ctx.eps[i, l]), b, s)
            constants["v_max_user"] = (float(v_max_user[i, l]) if constants["q"] == "v"
                                       else math.nan)
            rows[i, l] = constants
        ctx.extras["rainflow_rows"] = rows

    def add_cell(self, ctx: base.FleetModel, i: int, l: int) -> None:
        """Rows of one rainflow cell, in the call order of ``base``."""
        T = ctx.T
        bound = str(ctx.bound_of[i, l])
        rows = ctx.extras["rainflow_rows"][i, l]

        base.add_selector_rows(ctx, i, l)
        base.add_quantity_rows(ctx, i, l, "mu",
                               [base.day_increment(ctx, i, l, k, "mu") for k in range(T)])
        if BOUND_QUANTITY[bound] is not None:
            q, key, squared = BOUND_QUANTITY[bound]
            base.add_quantity_rows(
                ctx, i, l, q,
                [base.day_increment(ctx, i, l, k, key, squared=squared) for k in range(T)])
        if bound in QUADRATIC_BOUNDS:
            base.add_quadratic_reliability_rows(ctx, i, l, rows["q"], c2=rows["c2"],
                                                c1=rows["c1"], mu_side=rows["mu_side"])
        else:
            base.add_upper_cap_rows(ctx, i, l, rows["q"], rows["cap"], name="rel")
        if math.isfinite(rows["v_max_user"]):
            base.add_upper_cap_rows(ctx, i, l, "v", rows["v_max_user"], name="v_max_user")
        base.add_repair_cost_rows(ctx, i, l)


base.register_cell_builder("rainflow", RainflowCellBuilder())
