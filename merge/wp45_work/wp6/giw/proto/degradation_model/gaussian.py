"""
Gaussian cell builder (spec, section "Gaussian Model").

A cell tracks the mean ``mu`` and the variance ``v`` of its degradation; under
the law ``ard1`` also their anchors ``g_mu`` and ``g_v``. The day maps are
Eqs. g_map_mu and g_map_v, built by the helpers of ``base`` in the cell's
encoding. The reliability row is Eq. g_row, ``kappa_G^2 v <= (tau - mu)^2``
with ``kappa_G = Phi^{-1}(1 - epsilon)``, in the form ``reliability_impl``;
the optional row ``v <= v_max_user`` holds where the input gives it.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtri

from fleet_management.degradation_model import base


# ===========================================================================
# Cell builder
# ===========================================================================
class GaussianCellBuilder:
    """Rows of the cells with ``model = gaussian``."""

    name = "gaussian"

    def prepare(self, ctx: base.FleetModel, cfg, cells, opts: dict) -> None:
        """Variables of ``v``, the anchors, selectors and segment binaries; ``NonConvex``."""
        base.add_state_vars(ctx, "mu", cells)
        base.add_state_vars(ctx, "v", cells)
        base.add_selector_vars(ctx, cells)
        base.add_pwl_vars(ctx, cells)
        ctx.extras[self.name] = {"v_max_user": np.asarray(cfg.v_max_user, dtype=float)}
        if any(str(ctx.impl_of[i, l]) == "exact" for i, l in cells):
            ctx.model.Params.NonConvex = 2

    def add_cell(self, ctx: base.FleetModel, i: int, l: int) -> None:
        """Dynamics, replacement, anchor and loop rows; reliability rows; repair cost."""
        T = ctx.T
        tau = float(ctx.tau[i, l])
        kappa = float(ndtri(1.0 - float(ctx.eps[i, l])))
        base.add_selector_rows(ctx, i, l)
        # Eqs. g_mu_carry ... g_anchor_keep
        base.add_quantity_rows(ctx, i, l, "mu",
                               [base.day_increment(ctx, i, l, k, "mu") for k in range(T)])
        base.add_quantity_rows(ctx, i, l, "v",
                               [base.day_increment(ctx, i, l, k, "v") for k in range(T)])
        # Eq. g_row
        base.add_quadratic_reliability_rows(ctx, i, l, "v", c2=1.0 / kappa ** 2, c1=0.0,
                                            mu_side=tau)
        v_max_user = float(ctx.extras[self.name]["v_max_user"][i, l])
        if math.isfinite(v_max_user):
            base.add_upper_cap_rows(ctx, i, l, "v", v_max_user, name="v_max_user")
        base.add_repair_cost_rows(ctx, i, l)


base.register_cell_builder("gaussian", GaussianCellBuilder())
