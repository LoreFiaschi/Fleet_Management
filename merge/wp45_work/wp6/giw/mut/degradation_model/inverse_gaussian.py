"""
Inverse Gaussian cell builder (spec, section "Inverse Gaussian Model").

A cell tracks the mean ``mu``; under the law ``ard1`` also its anchor
``g_mu``. The day map is Eq. ig_map. The reliability row is the linear cap
Eq. ig_row, ``mu <= mu_bar = tau / E[Z | Z > q]``, with ``Z ~ IG(1, 1/eta)``
and ``q`` its ``(1 - epsilon)``-quantile. ``config.ig_mean_cap`` computes
``mu_bar`` once per cell; it lives in ``config.py`` because the input check of
the replacement state needs it without Gurobi. ``reliability_impl`` is ignored.
"""

from __future__ import annotations

from fleet_management.config import ig_mean_cap
from fleet_management.degradation_model import base


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
                                name="rel_ig") if False else base.add_upper_cap_rows(ctx, i, l, "mu", 1.0, name="rel_ig")
        base.add_repair_cost_rows(ctx, i, l)


base.register_cell_builder("inverse_gaussian", InverseGaussianCellBuilder())
