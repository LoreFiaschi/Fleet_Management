"""
Wiener cell builder (spec, section "Wiener Model").

A cell tracks the mean ``mu`` and the variance ``v``; under the law ``ard1``
also the mean anchor ``g_mu``, and no variance anchor. A mission day adds
``sigma^2`` to the variance (Eq. w_map_v); a repair acts on the mean only, so
the variance falls only at a replacement (Eq. w_vrow in the big-M form). The
reliability row is the Gaussian row, Eq. g_row, in the form
``reliability_impl``; the optional row ``v <= v_max_user`` holds where the
input gives it.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtri

from fleet_management.degradation_model import base


# ===========================================================================
# Cell builder
# ===========================================================================
class WienerCellBuilder:
    """Rows of the cells with ``model = wiener``."""

    name = "wiener"

    def prepare(self, ctx: base.FleetModel, cfg, cells, opts: dict) -> None:
        """Variables of ``v`` (no anchor), the mean anchors, selectors and segment binaries."""
        base.add_state_vars(ctx, "mu", cells)
        base.add_state_vars(ctx, "v", cells, anchored=False)
        base.add_selector_vars(ctx, cells)
        base.add_pwl_vars(ctx, cells)
        ctx.extras[self.name] = {
            "sigma": np.asarray(cfg.sigma, dtype=float),
            "v_max_user": np.asarray(cfg.v_max_user, dtype=float),
        }
        if any(str(ctx.impl_of[i, l]) == "exact" for i, l in cells):
            ctx.model.Params.NonConvex = 2

    def add_cell(self, ctx: base.FleetModel, i: int, l: int) -> None:
        """Dynamics, replacement, anchor and loop rows; reliability rows; repair cost."""
        T = ctx.T
        tau = float(ctx.tau[i, l])
        kappa = float(ndtri(1.0 - float(ctx.eps[i, l])))
        sigma = float(ctx.extras[self.name]["sigma"][i, l])
        base.add_selector_rows(ctx, i, l)
        # Eqs. w_mu_carry ... w_anchor_keep
        base.add_quantity_rows(ctx, i, l, "mu",
                               [base.day_increment(ctx, i, l, k, "mu") for k in range(T)])
        # Eqs. w_map_v, w_vrow, w_v_repl
        base.add_quantity_rows(ctx, i, l, "v",
                               [sigma ** 2 * base.mission_count(ctx, i, k) for k in range(T)],
                               repair_keeps_state=True)
        # Eq. g_row
        base.add_quadratic_reliability_rows(ctx, i, l, "v", c2=1.0 / kappa ** 2, c1=0.0,
                                            mu_side=tau)
        v_max_user = float(ctx.extras[self.name]["v_max_user"][i, l])
        if math.isfinite(v_max_user):
            base.add_upper_cap_rows(ctx, i, l, "v", v_max_user, name="v_max_user")
        base.add_repair_cost_rows(ctx, i, l)


base.register_cell_builder("wiener", WienerCellBuilder())
