"""
Shared Gurobi model layer for fleet management (model-agnostic skeleton).

This module owns what does **not** depend on the degradation model of a cell:

* the context ``FleetModel`` (sizes, shared variables, per-cell arrays, increment
  accessor, run options) and the cell-builder registry;
* the fleet rows (assignment, depot day, damage penalty) and the objective;
* the helpers that the cell builders call: variables in ``prepare``, and in
  ``add_cell`` the dynamics, replacement, anchor, loop, repair-cost and
  reliability rows;
* the result extraction, the cost report and the reports of ``solve_mixed``.

A degradation model plugs in through a ``CellBuilder`` registered under its name,
in the module ``fleet_management.degradation_model.<name>``, with the hooks
``prepare(ctx, cfg, cells, opts)``, ``add_cell(ctx, i, l)`` and the optional
``extract(ctx, cfg, out)``. ``build_fleet`` follows the building order of the
spec: model, options and shared variables; ``prepare`` of each model in
alphabetical order; objective and the rows that do not depend on the model;
``add_cell`` of every cell, ``i`` outer and ``l`` inner.

The dynamics have two encodings, chosen per cell: ``bigm`` (one-sided big-M rows)
and ``indicator`` (a selector ``n`` and Gurobi indicator constraints). At binary
points both have the same feasible set.

The Gamma block below stays until the Gamma cell builder is rewritten (work
package WP6 of the merge). It still calls helpers that no longer exist, so a
Gamma cell fails when it is built.

`base` must not import the model modules at import time (they import `base`);
``build_fleet`` imports them lazily so that their registration happens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import math
import time
from typing import Dict, Optional, Protocol

import numpy as np
import gurobipy as gp
from gurobipy import GRB


# ===========================================================================
# Shared context
# ===========================================================================
TRACKED_QUANTITIES = ("mu", "v", "R", "K")   # mean, variance, Hoeffding R, Chernoff K


def phase_day(k: int, H1: int, H2: int, has_transitory: bool) -> tuple[bool, int]:
    """Phase lookup of day index ``k`` (spec day ``k + 1``), Eq. c_phase.

    Returns ``(use_transitory, h)``: whether the transitory profile is read, and
    the index ``h`` on the day axis of the profile that is read.
    """
    if k < H1:
        return (True, k) if has_transitory else (False, k % H2)
    return (False, (k - H1) % H2)


@dataclass
class FleetModel:
    """Everything the per-cell builders need, built once and passed around.

    The shared variables, the per-cell arrays read from ``cfg``, the increment
    accessor and the run options are filled by ``build_context``. The state
    variables, anchors, selectors and segment binaries of the model-specific
    quantities are filled by the helpers that a builder's ``prepare`` calls.
    """
    # --- structure ---
    model: gp.Model
    F: int; H1: int; H2: int; M: int; L: int; T: int
    # --- shared decision variables ---
    x: gp.tupledict                        # assignment (F, M+1, T); j = 0 depot
    m_rep: gp.tupledict                    # repair x^m (F, L, T)
    r_rep: Optional[gp.tupledict]          # replacement x^r (F, L, T) or None
    mu_var: gp.tupledict                   # shared mean (F, L, T), [0, tau]
    z_var: gp.tupledict                    # removed damage z, cells with C_R > 0
    u_var: Optional[gp.Var]                # damage penalty u, None if C_D == 0
    # --- per-cell (F, L) selectors ---
    model_of: np.ndarray                   # cfg.model
    repair_of: np.ndarray                  # cfg.repair_model
    bound_of: np.ndarray                   # cfg.bound_method
    impl_of: np.ndarray                    # cfg.reliability_impl
    encoding_of: np.ndarray                # cfg.encoding
    # --- per-cell (F, L) parameters ---
    tau: np.ndarray                        # threshold
    eps: np.ndarray                        # cfg.epsilon
    rho: np.ndarray                        # repair efficiency
    tangent_ref: np.ndarray                # tangent point, as a fraction of tau
    initial: Dict[str, np.ndarray]         # q -> mu_0, v_0, R_0, cgf_0
    replacement: Dict[str, np.ndarray]     # q -> replacement_mu, replacement_v, R_new, cgf_new
    anchor_start: Dict[str, np.ndarray]    # q -> anchor_mu_0, anchor_v_0, anchor_R_0, anchor_K_0
    state_range: Dict[str, np.ndarray]     # q -> natural range M_s (range_mu, ...)
    # --- options and profiles ---
    allow_replacement: bool
    pwl_points: int
    opts: dict                             # the run options
    profiles: dict                         # key -> (operating, transitory or None)
    # --- filled by the helpers of the builders' prepare hooks ---
    state_vars: dict = field(default_factory=dict)    # q -> {(i, l, k): Var}
    anchor_vars: dict = field(default_factory=dict)   # q -> {(i, l, k): Var}, ard1 cells
    nb: dict = field(default_factory=dict)            # (i, l, k) -> selector n
    pwl_w: dict = field(default_factory=dict)         # (i, l, k, p) -> segment binary w
    # free-form per-model storage
    extras: dict = field(default_factory=dict)

    # --- helpers ---
    def cells_of(self, model_name: str):
        """All (i, l) cells whose degradation model is ``model_name``."""
        return [(i, l) for i in range(self.F) for l in range(self.L)
                if str(self.model_of[i, l]) == model_name]

    def all_cells(self):
        return [(i, l) for i in range(self.F) for l in range(self.L)]

    def increment(self, key: str, i: int, j: int, l: int, k: int) -> float:
        """Increment of profile ``key`` for train ``i``, activity ``j``,
        component ``l`` and day index ``k``, after the phase lookup (Eq. c_phase).

        ``j = 0`` is the depot and adds no increment; mission ``j`` is read at
        position ``j - 1`` of the profile.
        """
        if j == 0:
            return 0.0
        try:
            operating, transitory = self.profiles[key]
        except KeyError:
            raise KeyError(f"profile {key!r} is not in the input") from None
        use_transitory, h = phase_day(k, self.H1, self.H2, transitory is not None)
        source = transitory if use_transitory else operating
        return float(source[i, l, j - 1, h])


# ===========================================================================
# Cell-builder registry
# ===========================================================================
class CellBuilder(Protocol):
    """Interface a degradation model implements to plug into the shared model.

    ``prepare`` and ``add_cell`` are required. A builder may also define
    ``extract(ctx, cfg, out)``, which ``solve_mixed`` calls after the shared
    extraction to add model-specific entries to the result ``out``; it must
    accept ``out["x"] is None`` (no solution).
    """

    name: str

    def prepare(self, ctx: FleetModel, cfg, cells, opts: dict) -> None:
        """Create this model's auxiliary variables / per-cell arrays for ``cells``."""
        ...

    def add_cell(self, ctx: FleetModel, i: int, l: int) -> None:
        """Add the constraints of one (vehicle, component) cell."""
        ...


CELL_BUILDERS: Dict[str, CellBuilder] = {}


def register_cell_builder(name: str, builder) -> None:
    """Register a degradation model's cell builder under ``name``."""
    CELL_BUILDERS[name] = builder


def get_cell_builder(name: str):
    try:
        return CELL_BUILDERS[name]
    except KeyError:
        raise NotImplementedError(
            f"degradation model {name!r} has no cell builder registered; "
            f"available: {tuple(CELL_BUILDERS)}."
        )


def dispatch_cell(ctx: FleetModel, i: int, l: int) -> None:
    """Route one cell to its model's builder."""
    get_cell_builder(str(ctx.model_of[i, l])).add_cell(ctx, i, l)


# ===========================================================================
# ###################  GAMMA FINITE-HORIZON TAIL BLOCK  #####################
# ===========================================================================
# The modular gamma block goes here. A gamma cell shares the fleet skeleton
# (assignment x, the safety variable u and the objective), so it MUST drive the
# shared mean state ``ctx.mu_var[i,l,k]`` — that is what the u / C_D term reads.
# Everything gamma-specific (its own
# state variables, shape/scale bookkeeping) can live in ``ctx.extras["gamma"]``.
#
# The numerical calibration is independent of Gurobi. This block consumes its
# bounded mission shapes, creates A', and keeps physical expected damage mu as a
# separate shared state. Initial and replacement distributions are calibrated as
# mutually exclusive seed histories.
#
# At repair, the common bounding rate remains fixed and the bounding shape is
# reduced consistently with the selected repair model. ARD-inf contracts the
# complete mean and shape states. ARD1 contracts only the damage accumulated
# since the previous intervention and therefore requires separate mean and shape
# latches. ARD-inf always uses direct state balances and exact binary-product
# hulls, including its replacement reset. ARD1 uses corresponding products for
# repairable damage, the state discarded by replacement, and the latch reset.
# No Gamma branch needs a no-intervention selector or conditional Big-M row.
# ---------------------------------------------------------------------------
def _add_binary_scaled_product(
    model,
    product,
    state,
    active,
    *,
    scale: float,
    state_upper: float,
    name: str,
) -> None:
    """Convex-hull linearization of ``product = scale * state * active``.

    The formulation is exact for a non-negative bounded ``state`` and binary
    ``active``.  Unlike a pair of conditional state equalities, it introduces
    only the one bound-derived coefficient that the continuous-by-binary
    product mathematically requires.  ``product`` has a non-negative lower
    bound at variable creation.
    """
    factor = float(scale)
    upper = float(state_upper)
    if not np.isfinite(factor) or factor < 0.0:
        raise ValueError(f"invalid product scale for {name}: {factor}")
    if not np.isfinite(upper) or upper < 0.0:
        raise ValueError(f"invalid state upper bound for {name}: {upper}")

    coefficient = factor * upper
    model.addConstr(product <= factor * state, name=f"{name}_state_ub")
    model.addConstr(product <= coefficient * active, name=f"{name}_binary_ub")
    model.addConstr(
        product >= factor * state - coefficient * (1 - active),
        name=f"{name}_lower",
    )

    summary = getattr(model, "_binary_product_summary", None)
    if summary is None:
        summary = {
            "products": 0,
            "linear_rows": 0,
            "maximum_bound_coefficient": 0.0,
        }
        model._binary_product_summary = summary
    summary["products"] += 1
    summary["linear_rows"] += 3
    summary["maximum_bound_coefficient"] = max(
        summary["maximum_bound_coefficient"], coefficient,
    )


def _gamma_reachable_upper_bounds(
    *,
    initial_mean: float,
    replacement_mean: float,
    mean_increments: np.ndarray,
    mean_limit: float,
    initial_shape: float,
    replacement_shape: float,
    shape_increments: np.ndarray,
    shape_limit: float,
    remaining: float,
    repair_model: str,
    allow_replacement: bool,
) -> dict[str, np.ndarray]:
    """Return safe time-indexed bounds for one finite-horizon Gamma cell.

    ``mean_increments`` and ``shape_increments`` contain one non-negative
    upper bound per time step.  Normal operation adds at most that amount;
    repair cannot increase either state; replacement resets it to the supplied
    seed.  ARD1 latches are treated separately because normal operation holds
    them and an intervention resets them to the post-intervention state.

    The recursion deliberately over-approximates reachability.  It is therefore
    safe for product-hull construction without solving the scheduling problem
    first.
    """
    mean_inc = np.asarray(mean_increments, dtype=float)
    shape_inc = np.asarray(shape_increments, dtype=float)
    if mean_inc.ndim != 1 or shape_inc.shape != mean_inc.shape:
        raise ValueError(
            "Gamma reachable-bound increments must be equal 1-D arrays."
        )
    if np.any(mean_inc < 0.0) or np.any(shape_inc < 0.0):
        raise ValueError("Gamma reachable-bound increments must be non-negative.")

    T = int(mean_inc.size)
    mean_upper = np.empty(T, dtype=float)
    shape_upper = np.empty(T, dtype=float)
    removed_upper = np.empty(T, dtype=float)
    mean_latch_upper = np.zeros(T, dtype=float)
    shape_latch_upper = np.zeros(T, dtype=float)

    previous_mean = float(initial_mean)
    previous_shape = float(initial_shape)
    previous_mean_latch = 0.0
    previous_shape_latch = 0.0
    use_latch = repair_model == "ard1"
    repaired_fraction = 1.0 - float(remaining)

    for k in range(T):
        if use_latch:
            repaired_mean = (
                remaining * previous_mean
                + repaired_fraction * previous_mean_latch
            )
            repaired_shape = (
                remaining * previous_shape
                + repaired_fraction * previous_shape_latch
            )
        else:
            repaired_mean = remaining * previous_mean
            repaired_shape = remaining * previous_shape

        mean_candidates = [previous_mean + float(mean_inc[k]), repaired_mean]
        shape_candidates = [previous_shape + float(shape_inc[k]), repaired_shape]
        if allow_replacement:
            mean_candidates.append(float(replacement_mean))
            shape_candidates.append(float(replacement_shape))

        current_mean = min(float(mean_limit), max(mean_candidates))
        current_shape = min(float(shape_limit), max(shape_candidates))
        mean_upper[k] = max(0.0, current_mean)
        shape_upper[k] = max(0.0, current_shape)

        # ARD1 removes rho * (state - latch).  The latch has lower bound zero,
        # so rho * previous_mean is the safe reachable upper bound for both
        # repair models.
        repair_removed = max(0.0, repaired_fraction * previous_mean)
        # z is the damage removed by imperfect repair. Replacement has its own
        # binary cost and therefore must not also contribute to the repair cost.
        removed_upper[k] = min(float(mean_limit), repair_removed)

        if use_latch:
            mean_latch_candidates = [previous_mean_latch, repaired_mean]
            shape_latch_candidates = [previous_shape_latch, repaired_shape]
            if allow_replacement:
                mean_latch_candidates.append(float(replacement_mean))
                shape_latch_candidates.append(float(replacement_shape))
            mean_latch_upper[k] = min(
                mean_upper[k], float(mean_limit), max(mean_latch_candidates)
            )
            shape_latch_upper[k] = min(
                shape_upper[k], float(shape_limit), max(shape_latch_candidates)
            )
            previous_mean_latch = float(mean_latch_upper[k])
            previous_shape_latch = float(shape_latch_upper[k])

        previous_mean = float(mean_upper[k])
        previous_shape = float(shape_upper[k])

    return {
        "mean": mean_upper,
        "shape": shape_upper,
        "removed_mean": removed_upper,
        "mean_latch": mean_latch_upper,
        "shape_latch": shape_latch_upper,
    }


class GammaCellBuilder:
    """Selectable common-rate Gamma surrogate for modular Gamma cells.

    The current route calibrates each repeated mission-increment type and each
    alternative seed state at one common rate. The common bounding rate remains
    fixed during repair, while the physical mean and bounding shape are
    contracted according to ARD-inf or ARD1. ARD1 uses separate mean and
    bounding-shape latches storing the state after the previous intervention.
    """

    name = "gamma"

    def prepare(self, ctx: FleetModel, cfg, cells, opts: dict) -> None:
        """Calibrate mission shapes offline and create the bounding state."""
        from fleet_management.degradation_model.gamma_utils.gamma_repeated_calibration import (
            calibrate_gamma_cell_tail_bound,
            required_shape_for_tail,
        )

        cells = list(cells)
        if not cells:
            return

        def rate_profile(values, i, l, shape, name):
            """Accept the legacy (F,L) rate or the forthcoming profile form."""
            if values is None:
                raise ValueError(f"gamma cell (i={i}, l={l}) needs '{name}'.")
            arr = np.asarray(values, dtype=float)
            if arr.ndim == 2:
                value = float(arr[i, l])
                return np.full(shape, value, dtype=float)
            if arr.ndim == 4:
                cell = np.asarray(arr[i, l], dtype=float)
                try:
                    return np.broadcast_to(cell, shape).astype(float, copy=True)
                except ValueError as error:
                    raise ValueError(
                        f"gamma cell (i={i}, l={l}) {name} profile {cell.shape} "
                        f"cannot broadcast to {shape}."
                    ) from error
            raise ValueError(
                f"'{name}' must be normalized as (F,L) or (F,L,M,H); "
                f"got shape {arr.shape}."
            )

        state_keys = [(i, l, k) for i, l in cells for k in range(ctx.T)]
        A_var = ctx.model.addVars(state_keys, lb=0.0, name="A_gamma_bound")
        ard1_cells = [
            (i, l) for i, l in cells if str(cfg.repair_model[i, l]) == "ard1"
        ]
        ard1_keys = [(i, l, k) for i, l in ard1_cells for k in range(ctx.T)]
        ardinf_product_cells = [
            (i, l) for i, l in cells
            if str(cfg.repair_model[i, l]) == "ardinf"
        ]
        ard1_product_cells = list(ard1_cells)
        ard1_no_replacement_cells = (
            [] if ctx.allow_replacement else list(ard1_cells)
        )
        ard1_replacement_cells = (
            list(ard1_cells) if ctx.allow_replacement else []
        )
        removed_shape_keys = [
            (i, l, k)
            for i, l in ardinf_product_cells
            for k in range(ctx.T)
        ]
        removed_shape = (
            ctx.model.addVars(removed_shape_keys, lb=0.0, name="zA_gamma")
            if removed_shape_keys else None
        )
        ardinf_replacement_cells = (
            ardinf_product_cells if ctx.allow_replacement else []
        )
        replacement_product_keys = [
            (i, l, k)
            for i, l in ardinf_replacement_cells
            for k in range(ctx.T)
        ]
        replaced_mean = (
            ctx.model.addVars(
                replacement_product_keys,
                lb=0.0,
                name="qRmu_gamma_ardinf",
            )
            if replacement_product_keys else None
        )
        replaced_shape = (
            ctx.model.addVars(
                replacement_product_keys,
                lb=0.0,
                name="qRA_gamma_ardinf",
            )
            if replacement_product_keys else None
        )
        repairable_keys = [
            (i, l, k)
            for i, l in ard1_product_cells
            for k in range(ctx.T)
        ]
        repairable_mean = (
            ctx.model.addVars(repairable_keys, lb=0.0, name="qmu_gamma_ard1")
            if repairable_keys else None
        )
        repairable_shape = (
            ctx.model.addVars(repairable_keys, lb=0.0, name="qA_gamma_ard1")
            if repairable_keys else None
        )
        ard1_replacement_keys = [
            (i, l, k)
            for i, l in ard1_replacement_cells
            for k in range(ctx.T)
        ]
        ard1_replaced_mean = (
            ctx.model.addVars(
                ard1_replacement_keys,
                lb=0.0,
                name="qRmu_gamma_ard1",
            )
            if ard1_replacement_keys else None
        )
        ard1_replaced_shape = (
            ctx.model.addVars(
                ard1_replacement_keys,
                lb=0.0,
                name="qRA_gamma_ard1",
            )
            if ard1_replacement_keys else None
        )
        ard1_latch_replacement_keys = [
            (i, l, k)
            for i, l in ard1_replacement_cells
            for k in range(1, ctx.T)
        ]
        ard1_replaced_mean_latch = (
            ctx.model.addVars(
                ard1_latch_replacement_keys,
                lb=0.0,
                name="qRgmu_gamma_ard1",
            )
            if ard1_latch_replacement_keys else None
        )
        ard1_replaced_shape_latch = (
            ctx.model.addVars(
                ard1_latch_replacement_keys,
                lb=0.0,
                name="qRgA_gamma_ard1",
            )
            if ard1_latch_replacement_keys else None
        )
        mean_latch = (
            ctx.model.addVars(ard1_keys, lb=0.0, name="gmu_gamma")
            if ard1_keys else None
        )

        shape_latch = (
            ctx.model.addVars(ard1_keys, lb=0.0, name="gA_gamma")
            if ard1_keys else None
        )

        common_rate = np.zeros((ctx.F, ctx.L))
        maximum_shape = np.zeros((ctx.F, ctx.L))
        bounded_trans = {}
        bounded_operating = {}
        initial_shape = np.zeros((ctx.F, ctx.L))
        replacement_shape = np.zeros((ctx.F, ctx.L))
        calibrations = {}
        calibration_seconds = {}
        reachable_upper_bounds = {}

        beta_trans_cfg = getattr(cfg, "gamma_beta_trans", None)
        beta_bound_cfg = getattr(cfg, "gamma_beta_bound", None)
        calibration_method = getattr(
            cfg, "gamma_calibration_method", "finite_count"
        )
        for i, l in cells:
            repair_model = str(cfg.repair_model[i, l])
            if repair_model not in {"ardinf", "ard1"}:
                raise ValueError(
                    f"gamma cell (i={i}, l={l}): unsupported repair_model "
                    f"{repair_model!r}; expected 'ardinf' or 'ard1'."
                )
            operating = np.asarray(cfg.mu[i, l], dtype=float)
            if operating.shape[-1] != ctx.H2:
                raise ValueError(
                    f"gamma cell (i={i}, l={l}): operating mu profile must have "
                    f"length H2={ctx.H2}, got {operating.shape[-1]}."
                )
            beta_operating = rate_profile(
                cfg.gamma_beta, i, l, operating.shape, "gamma_beta"
            )

            if cfg.mu_trans is None:
                indices = np.arange(ctx.H1) % ctx.H2
                trans = operating[..., indices]
                if beta_trans_cfg is None:
                    beta_trans = beta_operating[..., indices]
                else:
                    beta_trans = rate_profile(
                        beta_trans_cfg, i, l, trans.shape, "gamma_beta_trans"
                    )
            else:
                trans = np.asarray(cfg.mu_trans[i, l], dtype=float)
                if beta_trans_cfg is None:
                    indices = np.arange(ctx.H1) % ctx.H2
                    beta_trans = beta_operating[..., indices]
                else:
                    beta_trans = rate_profile(
                        beta_trans_cfg, i, l, trans.shape, "gamma_beta_trans"
                    )

            selected_rate = (
                None
                if beta_bound_cfg is None
                else float(np.asarray(beta_bound_cfg, dtype=float)[i, l])
            )
            combined_mu = np.concatenate((trans, operating), axis=-1)
            combined_beta = np.concatenate((beta_trans, beta_operating), axis=-1)
            beta_0_cfg = getattr(cfg, "gamma_beta_0", None)
            beta_new_cfg = getattr(cfg, "gamma_beta_new", None)
            calibration_start = time.perf_counter()
            calibration_kwargs = {
                "expected_damage": combined_mu,
                "rates": combined_beta,
                "threshold": float(ctx.tau[i, l]),
                "max_total_count": ctx.T,
                "initial_expected_damage": float(ctx.mu_0[i, l]),
                "initial_rate": (
                    None if beta_0_cfg is None else float(beta_0_cfg[i, l])
                ),
                "replacement_expected_damage": float(ctx.mu_new[i, l]),
                "replacement_rate": (
                    None if beta_new_cfg is None else float(beta_new_cfg[i, l])
                ),
                "common_rate": selected_rate,
            }
            if calibration_method == "repeated_increment":
                calibration = calibrate_gamma_cell_tail_bound(
                    epsilon=float(ctx.eps[i, l]),
                    **calibration_kwargs,
                )
                ctx.impl_of[(i, l)] = "gamma_repeated_tail"
            elif calibration_method == "finite_count":
                from fleet_management.degradation_model.gamma_utils.gamma_tail_bound import (
                    calculate_seeded_profile_tail_bound_parameters,
                )

                calibration = calculate_seeded_profile_tail_bound_parameters(
                    **calibration_kwargs,
                )
                ctx.impl_of[(i, l)] = "gamma_finite_tail"
            else:
                raise ValueError(
                    f"unsupported gamma_calibration_method={calibration_method!r}"
                )
            split = ctx.H1
            bounded_trans[i, l] = calibration.bounded_shapes[..., :split]
            bounded_operating[i, l] = calibration.bounded_shapes[..., split:]
            calibrations[i, l] = calibration
            initial_shape[i, l] = calibration.initial_bounded_shape
            replacement_shape[i, l] = calibration.replacement_bounded_shape
            common_rate[i, l] = calibration.common_rate
            maximum_shape[i, l] = required_shape_for_tail(
                float(ctx.eps[i, l]),
                calibration.common_rate,
                float(ctx.tau[i, l]),
            )
            calibration_seconds[i, l] = time.perf_counter() - calibration_start
            bounds = _gamma_reachable_upper_bounds(
                initial_mean=float(ctx.mu_0[i, l]),
                replacement_mean=float(ctx.mu_new[i, l]),
                mean_increments=np.max(combined_mu, axis=0),
                mean_limit=float(ctx.tau[i, l]),
                initial_shape=float(initial_shape[i, l]),
                replacement_shape=float(replacement_shape[i, l]),
                shape_increments=np.max(calibration.bounded_shapes, axis=0),
                shape_limit=float(maximum_shape[i, l]),
                remaining=1.0 - float(ctx.rho[i, l]),
                repair_model=repair_model,
                allow_replacement=ctx.allow_replacement,
            )
            reachable_upper_bounds[i, l] = bounds
            for k in range(ctx.T):
                A_var[i, l, k].UB = float(bounds["shape"][k])
                ctx.mu_var[i, l, k].UB = float(bounds["mean"][k])
                ctx.z_var[i, l, k].UB = float(bounds["removed_mean"][k])
                if mean_latch is not None and (i, l) in ard1_cells:
                    mean_latch[i, l, k].UB = float(bounds["mean_latch"][k])
                    shape_latch[i, l, k].UB = float(bounds["shape_latch"][k])
                if removed_shape is not None and (i, l) in ardinf_product_cells:
                    previous_shape_upper = (
                        float(initial_shape[i, l])
                        if k == 0 else float(bounds["shape"][k - 1])
                    )
                    removed_shape[i, l, k].UB = (
                        float(ctx.rho[i, l]) * previous_shape_upper
                    )
                if replaced_mean is not None and (i, l) in ardinf_replacement_cells:
                    previous_mean_upper = (
                        float(ctx.mu_0[i, l])
                        if k == 0 else float(bounds["mean"][k - 1])
                    )
                    previous_shape_upper = (
                        float(initial_shape[i, l])
                        if k == 0 else float(bounds["shape"][k - 1])
                    )
                    replaced_mean[i, l, k].UB = previous_mean_upper
                    replaced_shape[i, l, k].UB = previous_shape_upper
                if repairable_mean is not None and (i, l) in ard1_product_cells:
                    previous_mean_upper = (
                        float(ctx.mu_0[i, l])
                        if k == 0 else float(bounds["mean"][k - 1])
                    )
                    previous_shape_upper = (
                        float(initial_shape[i, l])
                        if k == 0 else float(bounds["shape"][k - 1])
                    )
                    repairable_mean[i, l, k].UB = previous_mean_upper
                    repairable_shape[i, l, k].UB = previous_shape_upper
                if ard1_replaced_mean is not None and (i, l) in ard1_replacement_cells:
                    previous_mean_upper = (
                        float(ctx.mu_0[i, l])
                        if k == 0 else float(bounds["mean"][k - 1])
                    )
                    previous_shape_upper = (
                        float(initial_shape[i, l])
                        if k == 0 else float(bounds["shape"][k - 1])
                    )
                    ard1_replaced_mean[i, l, k].UB = previous_mean_upper
                    ard1_replaced_shape[i, l, k].UB = previous_shape_upper
                    if k > 0:
                        ard1_replaced_mean_latch[i, l, k].UB = float(
                            bounds["mean_latch"][k - 1]
                        )
                        ard1_replaced_shape_latch[i, l, k].UB = float(
                            bounds["shape_latch"][k - 1]
                        )

        product_hull_cells = ardinf_product_cells + ard1_product_cells
        if len(ardinf_product_cells) == len(cells):
            dynamics_formulation = (
                "ardinf_replacement_product_hull"
                if ctx.allow_replacement else "ardinf_product_hull"
            )
        elif len(ard1_product_cells) == len(cells):
            dynamics_formulation = (
                "ard1_replacement_product_hull"
                if ctx.allow_replacement else "ard1_product_hull"
            )
        elif len(product_hull_cells) == len(cells):
            dynamics_formulation = (
                "replacement_product_hull"
                if ctx.allow_replacement else "no_replacement_product_hull"
            )
        else:
            raise AssertionError("Gamma repair model was not assigned a formulation")

        ctx.extras["gamma"] = {
            "cells": cells,
            "ard1_cells": ard1_cells,
            "ardinf_product_cells": ardinf_product_cells,
            # Compatibility metadata for existing reports.
            "ardinf_no_replacement_cells": (
                [] if ctx.allow_replacement else ardinf_product_cells
            ),
            "ardinf_replacement_cells": ardinf_replacement_cells,
            "ard1_product_cells": ard1_product_cells,
            "ard1_no_replacement_cells": ard1_no_replacement_cells,
            "ard1_replacement_cells": ard1_replacement_cells,
            "repair_model": {
                (i, l): str(cfg.repair_model[i, l]) for i, l in cells
            },
            "A_var": A_var,
            "removed_shape": removed_shape,
            "replaced_mean": replaced_mean,
            "replaced_shape": replaced_shape,
            "repairable_mean": repairable_mean,
            "repairable_shape": repairable_shape,
            "ard1_replaced_mean": ard1_replaced_mean,
            "ard1_replaced_shape": ard1_replaced_shape,
            "ard1_replaced_mean_latch": ard1_replaced_mean_latch,
            "ard1_replaced_shape_latch": ard1_replaced_shape_latch,
            "mean_latch": mean_latch,
            "shape_latch": shape_latch,
            "common_rate": common_rate,
            "maximum_shape": maximum_shape,
            "bounded_trans": bounded_trans,
            "bounded_operating": bounded_operating,
            "initial_shape": initial_shape,
            "replacement_shape": replacement_shape,
            "calibrations": calibrations,
            "calibration_seconds": calibration_seconds,
            "calibration_method": calibration_method,
            "reachable_upper_bounds": reachable_upper_bounds,
            "dynamics_formulation": dynamics_formulation,
            "product_bound_strategy": "time_dependent_reachable",
            # Backwards-compatible result field; Gamma no longer uses Big-M.
            "big_m_bound_strategy": "time_dependent_reachable",
        }

    def add_cell(self, ctx: FleetModel, i: int, l: int) -> None:
        """Add physical-mean and conservative-shape dynamics for one cell."""
        add_maintenance_gating(ctx, i, l)
        data = ctx.extras["gamma"]
        A_var = data["A_var"]
        trans = data["bounded_trans"][i, l]
        operating = data["bounded_operating"][i, l]
        initial_shape = float(data["initial_shape"][i, l])
        replacement_shape = float(data["replacement_shape"][i, l])
        maximum = float(data["maximum_shape"][i, l])
        reachable = data["reachable_upper_bounds"][i, l]
        repair_model = data["repair_model"][i, l]
        use_latch = repair_model == "ard1"
        use_product_hull = (
            (i, l) in data["ardinf_product_cells"]
        )
        use_ard1_product_hull = (
            (i, l) in data["ard1_product_cells"]
        )
        removed_shape = data["removed_shape"]
        replaced_mean = data["replaced_mean"]
        replaced_shape = data["replaced_shape"]
        repairable_mean = data["repairable_mean"]
        repairable_shape = data["repairable_shape"]
        ard1_replaced_mean = data["ard1_replaced_mean"]
        ard1_replaced_shape = data["ard1_replaced_shape"]
        ard1_replaced_mean_latch = data["ard1_replaced_mean_latch"]
        ard1_replaced_shape_latch = data["ard1_replaced_shape_latch"]
        mean_latch = data["mean_latch"]
        shape_latch = data["shape_latch"]
        md = ctx.model

        # Repair retains the common bounding rate and contracts the bounding shape.
        # ARD-inf acts on the complete state. ARD1 acts only on the part accumulated
        # since the previous intervention, represented by the mean and shape latches.
        rho = float(ctx.rho[i, l])
        remaining = 1.0 - rho
        for k in range(ctx.T):
            A_prev = initial_shape if k == 0 else A_var[i, l, k - 1]
            mu_prev = float(ctx.mu_0[i, l]) if k == 0 else ctx.mu_var[i, l, k - 1]
            A_prev_ub = initial_shape if k == 0 else None
            mu_prev_ub = float(ctx.mu_0[i, l]) if k == 0 else None
            if k > 0:
                A_prev_ub = float(reachable["shape"][k - 1])
                mu_prev_ub = float(reachable["mean"][k - 1])
            mean_latch_prev = (
                0.0 if k == 0 else mean_latch[i, l, k - 1]
            ) if use_latch else None

            shape_latch_prev = (
                0.0 if k == 0 else shape_latch[i, l, k - 1]
            ) if use_latch else None

            if k < ctx.H1:
                shape_profile = trans
                h = k
            else:
                shape_profile = operating
                h = (k - ctx.H1) % ctx.H2
            shape_inc = gp.quicksum(
                ctx.x[i, j, k] * float(shape_profile[j - 1, h])
                for j in range(1, ctx.M + 1)
            )
            mean_inc = gp.quicksum(
                ctx.x[i, j, k] * ctx.mu_inc(i, j - 1, l, k)
                for j in range(1, ctx.M + 1)
            )

            if use_product_hull:
                # Maintenance gating makes repair and replacement mutually
                # exclusive and forces every mission assignment to zero when
                # either action is selected. The abstract nonlinear ARD-inf
                # transition can therefore be written as two state balances
                # and exact binary-product hulls:
                #   mu_k = mu_prev + mean_inc - z_k,
                #   z_k  = rho * mu_prev * m_k,
                # with additional previous-state products for a replacement
                # reset. The no-intervention selector is unnecessary.
                zA = removed_shape[i, l, k]
                if k == 0:
                    md.addConstr(
                        ctx.z_var[i, l, k] == rho * float(ctx.mu_0[i, l])
                        * ctx.m_rep[i, l, k],
                        name=f"z_gamma_ardinf_seed_{i}_{l}_{k}",
                    )
                    md.addConstr(
                        zA == rho * initial_shape * ctx.m_rep[i, l, k],
                        name=f"zA_gamma_ardinf_seed_{i}_{l}_{k}",
                    )
                else:
                    _add_binary_scaled_product(
                        md,
                        ctx.z_var[i, l, k],
                        mu_prev,
                        ctx.m_rep[i, l, k],
                        scale=rho,
                        state_upper=mu_prev_ub,
                        name=f"z_gamma_ardinf_product_{i}_{l}_{k}",
                    )
                    _add_binary_scaled_product(
                        md,
                        zA,
                        A_prev,
                        ctx.m_rep[i, l, k],
                        scale=rho,
                        state_upper=A_prev_ub,
                        name=f"zA_gamma_ardinf_product_{i}_{l}_{k}",
                    )
                replacement_mean_term = 0.0
                replacement_shape_term = 0.0
                replacement_seed_mean = float(ctx.mu_new[i, l])
                if ctx.allow_replacement:
                    qRmu = replaced_mean[i, l, k]
                    qRA = replaced_shape[i, l, k]
                    if k == 0:
                        md.addConstr(
                            qRmu == float(ctx.mu_0[i, l])
                            * ctx.r_rep[i, l, k],
                            name=f"qRmu_gamma_ardinf_seed_{i}_{l}_{k}",
                        )
                        md.addConstr(
                            qRA == initial_shape * ctx.r_rep[i, l, k],
                            name=f"qRA_gamma_ardinf_seed_{i}_{l}_{k}",
                        )
                    else:
                        _add_binary_scaled_product(
                            md,
                            qRmu,
                            mu_prev,
                            ctx.r_rep[i, l, k],
                            scale=1.0,
                            state_upper=mu_prev_ub,
                            name=f"qRmu_gamma_ardinf_product_{i}_{l}_{k}",
                        )
                        _add_binary_scaled_product(
                            md,
                            qRA,
                            A_prev,
                            ctx.r_rep[i, l, k],
                            scale=1.0,
                            state_upper=A_prev_ub,
                            name=f"qRA_gamma_ardinf_product_{i}_{l}_{k}",
                        )
                    replacement_mean_term = (
                        -qRmu + replacement_seed_mean * ctx.r_rep[i, l, k]
                    )
                    replacement_shape_term = (
                        -qRA + replacement_shape * ctx.r_rep[i, l, k]
                    )
                md.addConstr(
                    ctx.mu_var[i, l, k] == mu_prev + mean_inc
                    - ctx.z_var[i, l, k] + replacement_mean_term,
                    name=f"mu_gamma_ardinf_balance_{i}_{l}_{k}",
                )
                md.addConstr(
                    A_var[i, l, k] == A_prev + shape_inc - zA
                    + replacement_shape_term,
                    name=f"A_gamma_ardinf_balance_{i}_{l}_{k}",
                )
                md.addConstr(
                    A_var[i, l, k] <= maximum,
                    name=f"rel_gamma_{i}_{l}_{k}",
                )
                continue

            if use_ard1_product_hull:
                # Project ARD1 repairs only damage accumulated since the latest
                # intervention. Replacement discards the complete previous
                # state and makes its seed the new latch. Exact products select
                # the repairable state, the replaced state, and (after k=0)
                # the replaced latch. No conditional state equality is needed.
                qmu = repairable_mean[i, l, k]
                qA = repairable_shape[i, l, k]
                active_mean = mu_prev - mean_latch_prev
                active_shape = A_prev - shape_latch_prev
                if k == 0:
                    md.addConstr(
                        qmu == float(ctx.mu_0[i, l]) * ctx.m_rep[i, l, k],
                        name=f"qmu_gamma_ard1_seed_{i}_{l}_{k}",
                    )
                    md.addConstr(
                        qA == initial_shape * ctx.m_rep[i, l, k],
                        name=f"qA_gamma_ard1_seed_{i}_{l}_{k}",
                    )
                else:
                    _add_binary_scaled_product(
                        md,
                        qmu,
                        active_mean,
                        ctx.m_rep[i, l, k],
                        scale=1.0,
                        state_upper=mu_prev_ub,
                        name=f"qmu_gamma_ard1_product_{i}_{l}_{k}",
                    )
                    _add_binary_scaled_product(
                        md,
                        qA,
                        active_shape,
                        ctx.m_rep[i, l, k],
                        scale=1.0,
                        state_upper=A_prev_ub,
                        name=f"qA_gamma_ard1_product_{i}_{l}_{k}",
                    )
                replacement_mean_term = 0.0
                replacement_shape_term = 0.0
                replacement_mean_latch_term = 0.0
                replacement_shape_latch_term = 0.0
                if ctx.allow_replacement:
                    qRmu = ard1_replaced_mean[i, l, k]
                    qRA = ard1_replaced_shape[i, l, k]
                    if k == 0:
                        md.addConstr(
                            qRmu == float(ctx.mu_0[i, l])
                            * ctx.r_rep[i, l, k],
                            name=f"qRmu_gamma_ard1_seed_{i}_{l}_{k}",
                        )
                        md.addConstr(
                            qRA == initial_shape * ctx.r_rep[i, l, k],
                            name=f"qRA_gamma_ard1_seed_{i}_{l}_{k}",
                        )
                    else:
                        _add_binary_scaled_product(
                            md,
                            qRmu,
                            mu_prev,
                            ctx.r_rep[i, l, k],
                            scale=1.0,
                            state_upper=mu_prev_ub,
                            name=f"qRmu_gamma_ard1_product_{i}_{l}_{k}",
                        )
                        _add_binary_scaled_product(
                            md,
                            qRA,
                            A_prev,
                            ctx.r_rep[i, l, k],
                            scale=1.0,
                            state_upper=A_prev_ub,
                            name=f"qRA_gamma_ard1_product_{i}_{l}_{k}",
                        )
                        qRgmu = ard1_replaced_mean_latch[i, l, k]
                        qRgA = ard1_replaced_shape_latch[i, l, k]
                        _add_binary_scaled_product(
                            md,
                            qRgmu,
                            mean_latch_prev,
                            ctx.r_rep[i, l, k],
                            scale=1.0,
                            state_upper=float(reachable["mean_latch"][k - 1]),
                            name=f"qRgmu_gamma_ard1_product_{i}_{l}_{k}",
                        )
                        _add_binary_scaled_product(
                            md,
                            qRgA,
                            shape_latch_prev,
                            ctx.r_rep[i, l, k],
                            scale=1.0,
                            state_upper=float(reachable["shape_latch"][k - 1]),
                            name=f"qRgA_gamma_ard1_product_{i}_{l}_{k}",
                        )
                        replacement_mean_latch_term = -qRgmu
                        replacement_shape_latch_term = -qRgA
                    replacement_mean_term = (
                        -qRmu + float(ctx.mu_new[i, l]) * ctx.r_rep[i, l, k]
                    )
                    replacement_shape_term = (
                        -qRA + replacement_shape * ctx.r_rep[i, l, k]
                    )
                    replacement_mean_latch_term += (
                        float(ctx.mu_new[i, l]) * ctx.r_rep[i, l, k]
                    )
                    replacement_shape_latch_term += (
                        replacement_shape * ctx.r_rep[i, l, k]
                    )
                md.addConstr(
                    ctx.z_var[i, l, k] == rho * qmu,
                    name=f"z_gamma_ard1_balance_{i}_{l}_{k}",
                )
                md.addConstr(
                    ctx.mu_var[i, l, k] == mu_prev + mean_inc - rho * qmu
                    + replacement_mean_term,
                    name=f"mu_gamma_ard1_balance_{i}_{l}_{k}",
                )
                md.addConstr(
                    A_var[i, l, k] == A_prev + shape_inc - rho * qA
                    + replacement_shape_term,
                    name=f"A_gamma_ard1_balance_{i}_{l}_{k}",
                )
                md.addConstr(
                    mean_latch[i, l, k]
                    == mean_latch_prev + remaining * qmu
                    + replacement_mean_latch_term,
                    name=f"gmu_gamma_ard1_balance_{i}_{l}_{k}",
                )
                md.addConstr(
                    shape_latch[i, l, k]
                    == shape_latch_prev + remaining * qA
                    + replacement_shape_latch_term,
                    name=f"gA_gamma_ard1_balance_{i}_{l}_{k}",
                )
                md.addConstr(
                    A_var[i, l, k] <= maximum,
                    name=f"rel_gamma_{i}_{l}_{k}",
                )
                continue

            raise AssertionError(
                f"Gamma cell {(i, l)} has no product-hull dynamics"
            )

        # The bound state and the separately tracked physical mean must both be
        # repeatable because A'/beta_bar is not assumed to equal physical mu.
        k_start, k_end = ctx.H1 - 1, ctx.T - 1
        md.addConstr(
            A_var[i, l, k_end] <= A_var[i, l, k_start],
            name=f"loop_A_gamma_{i}_{l}",
        )
        md.addConstr(
            ctx.mu_var[i, l, k_end] <= ctx.mu_var[i, l, k_start],
            name=f"loop_mu_gamma_{i}_{l}",
        )

    def extract(self, ctx: FleetModel, cfg, out: dict) -> None:
        """Add bounding shapes, rates, tails, and concise calibration metadata."""
        if out.get("x") is None or "gamma" not in ctx.extras:
            return
        from scipy.stats import gamma as gamma_distribution

        data = ctx.extras["gamma"]
        shape = np.zeros((ctx.F, ctx.L, ctx.T))
        tail = np.zeros((ctx.F, ctx.L, ctx.T))

        shape_operating = np.zeros((ctx.F, ctx.L, ctx.M, ctx.H2))
        shape_transitory = np.zeros((ctx.F, ctx.L, ctx.M, ctx.H1))
        maximum_shape = np.zeros((ctx.F, ctx.L))

        mean_latch_solution = (
            np.zeros((ctx.F, ctx.L, ctx.T))
            if data["mean_latch"] is not None else None
        )

        shape_latch_solution = (
            np.zeros((ctx.F, ctx.L, ctx.T))
            if data["shape_latch"] is not None else None
        )
        summaries = []
        for i, l in data["cells"]:
            rate = float(data["common_rate"][i, l])
            calibration = data["calibrations"][i, l]

            shape_operating[i, l] = data["bounded_operating"][i, l]
            shape_transitory[i, l] = data["bounded_trans"][i, l]
            maximum_shape[i, l] = data["maximum_shape"][i, l]

            repeated_method = data["calibration_method"] == "repeated_increment"
            constraints = (
                calibration.checks
                if repeated_method
                else calibration.compressed.constraints
            )
            for k in range(ctx.T):
                value = data["A_var"][i, l, k].X
                shape[i, l, k] = value
                tail[i, l, k] = gamma_distribution.sf(
                    float(ctx.tau[i, l]), a=value, scale=1.0 / rate
                ) if value > 0.0 else 0.0
                if mean_latch_solution is not None and (i, l) in data["ard1_cells"]:
                    mean_latch_solution[i, l, k] = data["mean_latch"][i, l, k].X
                    shape_latch_solution[i, l, k] = data["shape_latch"][i, l, k].X
            repair_model = str(cfg.repair_model[i, l])
            reachable = data["reachable_upper_bounds"][i, l]
            summary = {
                "i": i,
                "l": l,
                "method": data["calibration_method"],
                "guarantee_scope": (
                    "homogeneous_repetitions_per_increment_type"
                    if repeated_method else "finite_count_vectors"
                ),
                "common_rate": rate,
                "increment_types": int(calibration.type_max_counts.size),
                "increment_opportunities": int(calibration.original_shapes.size),
                "seed_types": int(calibration.increment_offset),
                "calibration_lp_variables": (
                    0
                    if repeated_method
                    else int(calibration.compressed.original_shapes.size)
                ),
                "tail_constraints": len(constraints),
                "calibration_seconds": float(
                    data["calibration_seconds"][i, l]
                ),
                "total_convolution_series_terms": (
                    0 if repeated_method else int(sum(
                        item.convolution_series_terms for item in constraints
                    ))
                ),
                "maximum_convolution_series_terms": (
                    0 if repeated_method else int(max(
                        (item.convolution_series_terms for item in constraints),
                        default=0,
                    ))
                ),
                "maximum_convolution_remaining_mass": (
                    0.0 if repeated_method else float(max(
                        (item.convolution_remaining_mass for item in constraints),
                        default=0.0,
                    ))
                ),
                "worst_calibration_margin": calibration.worst_tail_margin,
                "initial_bounded_shape": calibration.initial_bounded_shape,
                "replacement_bounded_shape": calibration.replacement_bounded_shape,
                "repair_bound": f"{repair_model}_fixed_rate_shape_scaling",
                "big_m_bound_strategy": data["big_m_bound_strategy"],
                "maximum_reachable_mean": float(np.max(reachable["mean"])),
                "maximum_reachable_shape": float(np.max(reachable["shape"])),
                "maximum_reachable_mean_latch": float(
                    np.max(reachable["mean_latch"])
                ),
                "maximum_reachable_shape_latch": float(
                    np.max(reachable["shape_latch"])
                ),
            }
            if repeated_method:
                safe_counts = calibration.maximum_safe_counts
                summary["maximum_safe_counts"] = safe_counts.tolist()
                summary["minimum_safe_count"] = int(np.min(safe_counts))
                summary["maximum_safe_count"] = int(np.max(safe_counts))
            summaries.append(summary)
        out["gamma_shape_bound"] = shape
        out["gamma_tail_bound"] = tail
        out["gamma_beta_bound"] = data["common_rate"]
        out["gamma_calibration"] = summaries
        out["gamma_calibration_method"] = data["calibration_method"]
        out["gamma_shape_increment"] = shape_operating
        out["gamma_shape_increment_trans"] = shape_transitory
        out["gamma_maximum_shape"] = maximum_shape
        out["gamma_dynamics_formulation"] = data["dynamics_formulation"]
        out["gamma_big_m_bound_strategy"] = data["big_m_bound_strategy"]
        if mean_latch_solution is not None:
            out["gamma_mean_latch"] = mean_latch_solution
            out["gamma_shape_latch"] = shape_latch_solution


register_cell_builder("gamma", GammaCellBuilder())


# ===========================================================================
# Run options and solver settings
# ===========================================================================
def resolve_run_options(cfg) -> dict:
    """The run options of the spec ("Run Options"), defaults filled by ``load_config``."""
    return dict(cfg.options)


def cost_days(objective_mode: str, H1: int, T: int) -> range:
    """Day indices whose cost enters the objective and carry a ``z`` variable.

    The transitory days are left out in the mode ``operating_average``.
    """
    return range(H1, T) if objective_mode == "operating_average" else range(T)


def apply_performance_params(model, time_limit, mip_gap, fast, extra) -> None:
    """Presolve / heuristics tuning; everything is overridable via gurobi_params."""
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)
    if fast:
        model.Params.MIPFocus = 1
        model.Params.Heuristics = 0.5
        model.Params.ImproveStartGap = 0.5
        if time_limit is not None:
            model.Params.NoRelHeurTime = max(2.0, 0.15 * float(time_limit))
    if extra:
        for key, val in extra.items():
            model.setParam(key, val)


def _make_progress_callback(interval_seconds: float):
    """Return a lightweight Gurobi callback recording primal/dual progress."""
    interval = float(interval_seconds)

    def callback(model, where):
        if where != GRB.Callback.MIP:
            return
        try:
            runtime = float(model.cbGet(GRB.Callback.RUNTIME))
            incumbent_raw = float(model.cbGet(GRB.Callback.MIP_OBJBST))
            bound_raw = float(model.cbGet(GRB.Callback.MIP_OBJBND))
            nodes = float(model.cbGet(GRB.Callback.MIP_NODCNT))
            solutions = int(model.cbGet(GRB.Callback.MIP_SOLCNT))
        except gp.GurobiError:
            return

        incumbent = (
            None if abs(incumbent_raw) >= 0.5 * GRB.INFINITY else incumbent_raw
        )
        bound = None if abs(bound_raw) >= 0.5 * GRB.INFINITY else bound_raw
        gap = None
        if incumbent is not None and bound is not None:
            gap = abs(incumbent - bound) / max(abs(incumbent), 1e-10)

        records = model._optimization_progress
        last_time = model._optimization_progress_last_time
        last_solutions = model._optimization_progress_last_solutions
        if (
            not records
            or runtime - last_time >= interval
            or solutions != last_solutions
        ):
            records.append({
                "runtime_seconds": runtime,
                "incumbent": incumbent,
                "best_bound": bound,
                "relative_gap": gap,
                "nodes": nodes,
                "solutions": solutions,
            })
            model._optimization_progress_last_time = runtime
            model._optimization_progress_last_solutions = solutions

    return callback


def _initialize_progress(model, interval_seconds: float):
    model._optimization_progress = []
    model._optimization_progress_last_time = -math.inf
    model._optimization_progress_last_solutions = -1
    return _make_progress_callback(interval_seconds)


def _append_final_progress(model) -> None:
    """Ensure the saved trajectory contains the final solver state."""
    incumbent = float(model.ObjVal) if int(model.SolCount) > 0 else None
    try:
        bound = float(model.ObjBound)
    except (AttributeError, gp.GurobiError):
        bound = None
    try:
        gap = float(model.MIPGap) if int(model.SolCount) > 0 else None
    except (AttributeError, gp.GurobiError):
        gap = None
    record = {
        "runtime_seconds": float(model.Runtime),
        "incumbent": incumbent,
        "best_bound": bound,
        "relative_gap": gap,
        "nodes": float(model.NodeCount),
        "solutions": int(model.SolCount),
    }
    records = model._optimization_progress
    if not records or abs(records[-1]["runtime_seconds"] - record["runtime_seconds"]) > 1e-9:
        records.append(record)
    else:
        records[-1] = record


def apply_binary_warm_start(ctx: FleetModel, result: dict | None) -> dict:
    """Map a previous schedule onto the current horizon as a Gurobi MIP start.

    Initialization steps retain their index. Operating steps repeat the old H2
    schedule cyclically, allowing a solved shorter horizon to seed a longer
    candidate. Only discrete schedule/action variables are started; Gurobi
    reconstructs all continuous degradation states from the constraints.
    """
    diagnostics = {"applied": False, "values": 0, "source_horizon": None}
    if not result or result.get("x") is None:
        return diagnostics

    old_h1 = int(result.get("H1", ctx.H1))
    old_h2 = int(result.get("H2", max(1, np.asarray(result["x"]).shape[-1] - old_h1)))
    x_old = np.asarray(result["x"], dtype=float)
    m_old = None if result.get("m") is None else np.asarray(result["m"], dtype=float)
    r_old = None if result.get("r") is None else np.asarray(result["r"], dtype=float)

    def old_step(k: int) -> int:
        if k < ctx.H1 and k < old_h1:
            return k
        return old_h1 + ((max(0, k - ctx.H1)) % old_h2)

    values = 0
    for k in range(ctx.T):
        source_k = old_step(k)
        if source_k >= x_old.shape[-1]:
            continue
        for i in range(min(ctx.F, x_old.shape[0])):
            for j in range(min(ctx.M + 1, x_old.shape[1])):
                ctx.x[i, j, k].Start = float(round(x_old[i, j, source_k]))
                values += 1
            for l in range(ctx.L):
                if m_old is not None and i < m_old.shape[0] and l < m_old.shape[1]:
                    m_value = float(round(m_old[i, l, source_k]))
                    ctx.m_rep[i, l, k].Start = m_value
                    values += 1
                else:
                    m_value = 0.0
                r_value = 0.0
                if ctx.allow_replacement and r_old is not None:
                    if i < r_old.shape[0] and l < r_old.shape[1]:
                        r_value = float(round(r_old[i, l, source_k]))
                        ctx.r_rep[i, l, k].Start = r_value
                        values += 1
                if (i, l, k) in ctx.nb:
                    ctx.nb[i, l, k].Start = max(
                        0.0, 1.0 - m_value - r_value
                    )
                    values += 1

    diagnostics.update({
        "applied": values > 0,
        "values": values,
        "source_horizon": {"H1": old_h1, "H2": old_h2},
    })
    return diagnostics


def complete_binary_warm_start(
    model: gp.Model,
    result: dict | None,
    diagnostics: dict,
) -> dict:
    """Complete a greedy discrete start by solving its fixed-binary model.

    Gurobi can attempt to complete a partial MIP start itself, but that effort
    is deliberately limited and may reject a feasible discrete schedule before
    reconstructing the many continuous degradation and product variables.  A
    greedy start may therefore request this explicit phase: copy the complete
    model, fix every initialized integer variable, solve the remaining model,
    and copy the resulting full solution back through ``Start`` attributes.

    The original model is never relaxed.  A completed start is therefore a
    feasible point of exactly the model that will subsequently be optimized.
    """
    requested = bool(result and result.get("complete_binary_start", False))
    completion = {
        "requested": requested,
        "status": "not_requested",
        "fixed_integer_variables": 0,
        "values_copied": 0,
        "runtime_seconds": None,
        "iis_constraints": [],
    }
    diagnostics["completion"] = completion
    if not requested or not diagnostics.get("applied", False):
        return diagnostics

    model.update()
    fixed = model.copy()
    original_variables = model.getVars()
    fixed_variables = fixed.getVars()
    fixed_count = 0
    for original, candidate in zip(original_variables, fixed_variables):
        if original.VType not in {GRB.BINARY, GRB.INTEGER}:
            continue
        value = float(original.Start)
        # GRB.UNDEFINED is greater than GRB.INFINITY.  Only variables actually
        # initialized by apply_binary_warm_start are fixed here.
        if not np.isfinite(value) or abs(value) >= GRB.INFINITY:
            continue
        value = float(round(value))
        candidate.LB = value
        candidate.UB = value
        fixed_count += 1

    completion["fixed_integer_variables"] = fixed_count
    fixed.Params.OutputFlag = 0
    fixed.Params.LogToConsole = 0
    fixed.Params.LogFile = ""
    fixed.Params.TimeLimit = float(
        result.get("completion_time_limit", 60.0)
    )
    fixed.optimize()
    completion["runtime_seconds"] = float(fixed.Runtime)

    if int(fixed.SolCount) > 0:
        for original, candidate in zip(original_variables, fixed_variables):
            original.Start = float(candidate.X)
        completion["status"] = "completed"
        completion["values_copied"] = len(original_variables)
    else:
        status_names = {
            GRB.INFEASIBLE: "infeasible",
            GRB.INF_OR_UNBD: "infeasible_or_unbounded",
            GRB.TIME_LIMIT: "time_limit",
        }
        completion["status"] = status_names.get(
            int(fixed.Status), f"status_{int(fixed.Status)}"
        )
        if int(fixed.Status) == GRB.INFEASIBLE:
            fixed.computeIIS()
            rows = [
                constraint.ConstrName
                for constraint in fixed.getConstrs()
                if constraint.IISConstr
            ]
            general = [
                constraint.GenConstrName
                for constraint in fixed.getGenConstrs()
                if constraint.IISGenConstr
            ]
            completion["iis_constraints"] = (rows + general)[:50]

    fixed.dispose()
    return diagnostics


def apply_relaxation_warm_start(
    model: gp.Model,
    *,
    time_limit: float,
) -> dict:
    """Solve an optional phase-I feasibility relaxation and copy a MIP start.

    Gurobi relaxes linear rows with an L1 violation objective while retaining
    integrality and the model's general constraints. The relaxed solution is
    *not* reported as feasible for the original problem: only its discrete
    values are copied as hints, and the original model must still establish a
    valid incumbent.
    """
    diagnostics = {
        "enabled": True,
        "status": "not_run",
        "relaxation_objective": None,
        "values_copied": 0,
        "runtime_seconds": None,
    }
    model.update()
    relaxed = model.copy()
    relaxed.ModelName = f"{model.ModelName}_phase1_relaxation"
    relaxed.Params.TimeLimit = max(1.0, float(time_limit))
    relaxed.Params.MIPGap = 0.05
    # type=0: sum of absolute violations; minrelax=False; variable bounds stay
    # hard; linear constraints are relaxable.
    relaxed.feasRelaxS(0, False, False, True)
    relaxed.optimize()
    diagnostics["runtime_seconds"] = float(relaxed.Runtime)
    diagnostics["status"] = status_string(relaxed.Status)
    if int(relaxed.SolCount) <= 0:
        return diagnostics

    diagnostics["relaxation_objective"] = float(relaxed.ObjVal)
    relaxed_by_name = {var.VarName: var for var in relaxed.getVars()}
    copied = 0
    for variable in model.getVars():
        if variable.VType not in {GRB.BINARY, GRB.INTEGER}:
            continue
        source = relaxed_by_name.get(variable.VarName)
        if source is None:
            continue
        variable.Start = float(round(source.X))
        copied += 1
    diagnostics["values_copied"] = copied
    return diagnostics


# ===========================================================================
# Model + shared variables
# ===========================================================================
def build_context(cfg, opts: dict, model_name: str = "fleet_management") -> FleetModel:
    """Create the Gurobi model, the shared variables, and the context object.

    Step 1 of the building order (spec, "Cell-Builder Interface"). ``z`` is built
    only at cells with ``C_R > 0``, on the days whose cost enters the objective;
    ``u`` only if ``C_D > 0``. The per-cell arrays are read from ``cfg``.
    """
    F, L, M = cfg.F, cfg.L, cfg.M
    H1, H2, T = cfg.H1, cfg.H2, cfg.T
    allow_replacement = bool(cfg.allow_replacement)

    md = gp.Model(model_name)
    md.Params.OutputFlag = int(opts["verbose"])
    apply_performance_params(md, opts["time_limit"], opts["mip_gap"],
                             opts["fast"], opts["gurobi_params"])

    x = md.addVars(F, M + 1, T, vtype=GRB.BINARY, name="x")
    m_rep = md.addVars(F, L, T, vtype=GRB.BINARY, name="m")
    r_rep = md.addVars(F, L, T, vtype=GRB.BINARY, name="r") if allow_replacement else None
    mu_var = md.addVars(F, L, T, lb=0.0, name="mu")
    for i in range(F):
        for l in range(L):
            for k in range(T):
                mu_var[i, l, k].UB = float(cfg.tau[i, l])

    C_R = np.asarray(cfg.C_R, dtype=float)
    z_keys = [(i, l, k) for i in range(F) for l in range(L) if C_R[i, l] > 0
              for k in cost_days(str(opts["objective_mode"]), H1, T)]
    z_var = md.addVars(z_keys, lb=0.0, name="z") if z_keys else gp.tupledict()
    u_var = md.addVar(lb=0.0, name="u") if float(cfg.C_D) > 0 else None

    profiles = {}
    for key in ("mu", "v", "support", "cgf", "gamma_beta"):
        operating = getattr(cfg, key, None)
        if operating is not None:
            transitory = getattr(cfg, key + "_trans", None)
            profiles[key] = (
                np.asarray(operating, dtype=float),
                None if transitory is None else np.asarray(transitory, dtype=float),
            )

    def floats(table):
        return {q: np.asarray(values, dtype=float) for q, values in table.items()}

    ctx = FleetModel(
        model=md, F=F, H1=H1, H2=H2, M=M, L=L, T=T,
        x=x, m_rep=m_rep, r_rep=r_rep, mu_var=mu_var, z_var=z_var, u_var=u_var,
        model_of=np.asarray(cfg.model, dtype=str),
        repair_of=np.asarray(cfg.repair_model, dtype=str),
        bound_of=np.asarray(cfg.bound_method, dtype=str),
        impl_of=np.asarray(cfg.reliability_impl, dtype=str),
        encoding_of=np.asarray(cfg.encoding, dtype=str),
        tau=np.asarray(cfg.tau, dtype=float), eps=np.asarray(cfg.epsilon, dtype=float),
        rho=np.asarray(cfg.rho, dtype=float),
        tangent_ref=np.asarray(cfg.tangent_ref, dtype=float),
        initial=floats({"mu": cfg.mu_0, "v": cfg.v_0, "R": cfg.R_0, "K": cfg.cgf_0}),
        replacement=floats({"mu": cfg.replacement_mu, "v": cfg.replacement_v,
                            "R": cfg.R_new, "K": cfg.cgf_new}),
        anchor_start=floats({"mu": cfg.anchor_mu_0, "v": cfg.anchor_v_0,
                             "R": cfg.anchor_R_0, "K": cfg.anchor_K_0}),
        state_range=floats({"mu": cfg.range_mu, "v": cfg.range_v,
                            "R": cfg.range_R, "K": cfg.range_K}),
        allow_replacement=allow_replacement, pwl_points=int(cfg.pwl_points),
        opts=dict(opts), profiles=profiles,
    )
    ctx.state_vars = {"mu": mu_var, "v": {}, "R": {}, "K": {}}
    ctx.anchor_vars = {q: {} for q in TRACKED_QUANTITIES}
    return ctx


# ===========================================================================
# Helpers for the cell builders: variables
# ===========================================================================
# A builder's ``prepare`` calls these for its own cells. ``q`` names a tracked
# quantity: "mu" (mean), "v" (variance), "R" (Hoeffding budget), "K" (Chernoff).
def _cell_value(values, i: int, l: int, label: str) -> float:
    """Float value of an (F, L) array at cell (i, l); a value that is not finite is an error."""
    value = float(values[i, l])
    if not math.isfinite(value):
        raise ValueError(f"{label} is not finite at cell (i={i}, l={l})")
    return value


def repair_factor(q: str, rho: float) -> float:
    """Repair factor ``a`` of the tracked quantity ``q`` (Eq. a_psi)."""
    if q in ("mu", "K"):
        return 1.0 - rho
    if q in ("v", "R"):
        return (1.0 - rho) ** 2
    raise ValueError(f"unknown tracked quantity {q!r}")


def add_state_vars(ctx: FleetModel, q: str, cells, *, anchored: bool = True) -> None:
    """State of ``q`` (not for the shared mean) and, at ard1 cells, its anchor.

    Both lie in the natural range of ``q`` (variable table; Eq. a_anchor).
    ``anchored=False`` builds no anchor (Wiener variance).
    """
    md, T = ctx.model, ctx.T
    for i, l in cells:
        upper = _cell_value(ctx.state_range[q], i, l, f"range_{q}")
        keys = [(i, l, k) for k in range(T)]
        if q != "mu":
            ctx.state_vars[q].update(md.addVars(keys, lb=0.0, ub=upper, name=q))
        if anchored and str(ctx.repair_of[i, l]) == "ard1":
            ctx.anchor_vars[q].update(md.addVars(keys, lb=0.0, ub=upper, name=f"g{q}"))


def add_selector_vars(ctx: FleetModel, cells) -> None:
    """Selector ``n`` at the cells with ``encoding = indicator`` (Eq. a_ind_sum)."""
    md, T = ctx.model, ctx.T
    for i, l in cells:
        if str(ctx.encoding_of[i, l]) == "indicator":
            keys = [(i, l, k) for k in range(T)]
            ctx.nb.update(md.addVars(keys, vtype=GRB.BINARY, name="n"))


def add_pwl_vars(ctx: FleetModel, cells) -> None:
    """Segment binaries ``w`` at the cells with ``reliability_impl = pwl`` (Eq. a_pwl_sum).

    ``cells`` must hold only cells with a quadratic reliability row. Nothing is
    built for ``pwl_points < 2``.
    """
    md, T = ctx.model, ctx.T
    P = int(ctx.pwl_points)
    if P < 2:
        return
    for i, l in cells:
        if str(ctx.impl_of[i, l]) == "pwl":
            keys = [(i, l, k, p) for k in range(T) for p in range(P)]
            ctx.pwl_w.update(md.addVars(keys, vtype=GRB.BINARY, name="w"))


# ===========================================================================
# Helpers for the cell builders: rows
# ===========================================================================
# A builder's ``add_cell`` calls these for one cell, in this order:
# ``add_selector_rows``, ``add_quantity_rows`` once per tracked quantity ("mu"
# first), the reliability rows, ``add_repair_cost_rows``.
def mission_count(ctx: FleetModel, i: int, k: int):
    """Number of missions flown by train ``i`` on day index ``k``."""
    return gp.quicksum(ctx.x[i, j, k] for j in range(1, ctx.M + 1))


def day_increment(ctx: FleetModel, i: int, l: int, k: int, key: str,
                  *, squared: bool = False):
    """Day increment of profile ``key`` of cell ``(i, l)`` on day index ``k``.

    The sum over the missions ``j`` of ``x[i, j, k]`` times the profile value
    after the phase lookup (Eq. c_phase), or its square if ``squared``. Terms
    with value zero are left out.
    """
    terms = []
    for j in range(1, ctx.M + 1):
        value = ctx.increment(key, i, j, l, k)
        if not math.isfinite(value):
            raise ValueError(
                f"increment {key!r} is not finite at cell (i={i}, l={l}), "
                f"mission {j}, day {k + 1}"
            )
        if squared:
            value = value * value
        if value != 0.0:
            terms.append(value * ctx.x[i, j, k])
    return gp.quicksum(terms)


def add_selector_rows(ctx: FleetModel, i: int, l: int) -> None:
    """``n + x^m + x^r = 1`` at an indicator cell (Eq. a_ind_sum); nothing at a big-M cell."""
    if str(ctx.encoding_of[i, l]) != "indicator":
        return
    md = ctx.model
    for k in range(ctx.T):
        xm = ctx.m_rep[i, l, k]
        if ctx.r_rep is None:
            md.addConstr(ctx.nb[i, l, k] + xm == 1, name=f"sel_{i}_{l}_{k}")
        else:
            md.addConstr(ctx.nb[i, l, k] + xm + ctx.r_rep[i, l, k] == 1,
                         name=f"sel_{i}_{l}_{k}")


def add_quantity_rows(ctx: FleetModel, i: int, l: int, q: str, increments,
                      *, repair_keeps_state: bool = False) -> None:
    """All generic rows of the tracked quantity ``q`` of cell ``(i, l)``, every day.

    The dynamics in the cell's encoding, the replacement rows (if replacement is
    allowed), the anchor rows (at an ard1 cell) and the loop rows of the state and
    of its anchor. ``increments[k]`` is the day increment of day index ``k``.
    ``repair_keeps_state`` is for the Wiener variance: repair factor 1, no anchor,
    and in the big-M form one row replaces the carry and repair rows (Eq. w_vrow).
    """
    md, T = ctx.model, ctx.T
    S, G = ctx.state_vars[q], ctx.anchor_vars[q]
    big_m = _cell_value(ctx.state_range[q], i, l, f"range_{q}")
    start = _cell_value(ctx.initial[q], i, l, f"initial {q}")
    rho = float(ctx.rho[i, l])
    a = 1.0 if repair_keeps_state else repair_factor(q, rho)
    anchored = (not repair_keeps_state) and str(ctx.repair_of[i, l]) == "ard1"
    if anchored and (i, l, 0) not in G:
        raise ValueError(
            f"anchor of {q!r} missing at cell (i={i}, l={l}); call add_state_vars in prepare"
        )
    g_start = s_new = None
    if anchored:
        g_start = _cell_value(ctx.anchor_start[q], i, l, f"starting anchor {q}")
    if ctx.allow_replacement:
        s_new = _cell_value(ctx.replacement[q], i, l, f"replacement {q}")
    indicator = str(ctx.encoding_of[i, l]) == "indicator"
    if indicator and (i, l, 0) not in ctx.nb:
        raise ValueError(
            f"selector missing at cell (i={i}, l={l}); call add_selector_vars in prepare"
        )
    if len(increments) != T:
        raise ValueError(f"increments of {q!r} must hold {T} entries, got {len(increments)}")

    for k in range(T):
        s = S[i, l, k]
        s_prev = start if k == 0 else S[i, l, k - 1]
        xm = ctx.m_rep[i, l, k]
        xr = ctx.r_rep[i, l, k] if ctx.allow_replacement else 0.0
        if anchored:
            g = G[i, l, k]
            g_prev = g_start if k == 0 else G[i, l, k - 1]
            psi = a * s_prev + (1.0 - a) * g_prev      # Eq. a_psi, ard1
        else:
            psi = a * s_prev                           # Eq. a_psi, ardinf

        if not indicator:
            if repair_keeps_state:
                # Eq. w_vrow
                md.addConstr(s >= s_prev + increments[k] - big_m * xr,
                             name=f"{q}_carry_{i}_{l}_{k}")
            else:
                # Eq. a_bigm_carry, a_bigm_repair
                md.addConstr(s >= s_prev + increments[k] - big_m * (xm + xr),
                             name=f"{q}_carry_{i}_{l}_{k}")
                md.addConstr(s >= psi - big_m * (1 - xm),
                             name=f"{q}_repair_{i}_{l}_{k}")
            if ctx.allow_replacement:
                # Eq. a_bigm_repl
                md.addConstr(s >= s_new - big_m * (1 - xr),
                             name=f"{q}_repl_lb_{i}_{l}_{k}")
                md.addConstr(s <= s_new + big_m * (1 - xr),
                             name=f"{q}_repl_ub_{i}_{l}_{k}")
            if anchored:
                # Eq. a_bigm_anchor
                e = xm + xr
                md.addConstr(g - s <= big_m * (1 - e), name=f"g{q}_set_lb_{i}_{l}_{k}")
                md.addConstr(s - g <= big_m * (1 - e), name=f"g{q}_set_ub_{i}_{l}_{k}")
                md.addConstr(g - g_prev <= big_m * e, name=f"g{q}_keep_lb_{i}_{l}_{k}")
                md.addConstr(g_prev - g <= big_m * e, name=f"g{q}_keep_ub_{i}_{l}_{k}")
        else:
            n = ctx.nb[i, l, k]
            # Eq. a_ind_state
            md.addGenConstrIndicator(n, True, s >= s_prev + increments[k],
                                     name=f"{q}_carry_{i}_{l}_{k}")
            md.addGenConstrIndicator(xm, True, s >= psi,
                                     name=f"{q}_repair_{i}_{l}_{k}")
            if ctx.allow_replacement:
                md.addGenConstrIndicator(xr, True, s == s_new,
                                         name=f"{q}_repl_{i}_{l}_{k}")
            if anchored:
                # Eq. a_ind_anchor
                md.addGenConstrIndicator(n, True, g == g_prev, name=f"g{q}_keep_{i}_{l}_{k}")
                md.addGenConstrIndicator(xm, True, g == s, name=f"g{q}_setm_{i}_{l}_{k}")
                if ctx.allow_replacement:
                    md.addGenConstrIndicator(xr, True, g == s, name=f"g{q}_setr_{i}_{l}_{k}")

    # Eq. a_loop
    md.addConstr(S[i, l, T - 1] <= S[i, l, ctx.H1 - 1], name=f"loop_{q}_{i}_{l}")
    if anchored:
        md.addConstr(G[i, l, T - 1] <= G[i, l, ctx.H1 - 1], name=f"loop_g{q}_{i}_{l}")


def add_repair_cost_rows(ctx: FleetModel, i: int, l: int) -> None:
    """Lower rows of the repair cost ``z`` at cell ``(i, l)`` (Eq. a_z).

    ``z_k >= rho (mu_{k-1} - g_{k-1}) - rho tau (1 - x^m_k)`` with ``g = 0`` at an
    ardinf cell, on every day that has a ``z``. Nothing is built at a cell without
    ``z``; there is no row on a replacement day.
    """
    md = ctx.model
    days = [k for k in range(ctx.T) if (i, l, k) in ctx.z_var]
    if not days:
        return
    ard1 = str(ctx.repair_of[i, l]) == "ard1"
    rho, tau = float(ctx.rho[i, l]), float(ctx.tau[i, l])
    G = ctx.anchor_vars["mu"]
    mu_start = _cell_value(ctx.initial["mu"], i, l, "initial mu")
    if ard1:
        if (i, l, 0) not in G:
            raise ValueError(
                f"anchor of 'mu' missing at cell (i={i}, l={l}); call add_state_vars in prepare"
            )
        g_start = _cell_value(ctx.anchor_start["mu"], i, l, "starting anchor mu")
    for k in days:
        mu_prev = mu_start if k == 0 else ctx.mu_var[i, l, k - 1]
        g_prev = (g_start if k == 0 else G[i, l, k - 1]) if ard1 else 0.0
        md.addConstr(
            ctx.z_var[i, l, k] >= rho * (mu_prev - g_prev) - rho * tau * (1 - ctx.m_rep[i, l, k]),
            name=f"z_{i}_{l}_{k}",
        )


def add_upper_cap_rows(ctx: FleetModel, i: int, l: int, q: str, cap, *, name: str) -> None:
    """``state_vars[q] <= cap`` on every day (the linear reliability rows)."""
    cap = float(cap)
    for k in range(ctx.T):
        ctx.model.addConstr(ctx.state_vars[q][i, l, k] <= cap, name=f"{name}_{i}_{l}_{k}")


def add_quadratic_reliability_rows(ctx: FleetModel, i: int, l: int, q: str,
                                   *, c2: float, c1: float, mu_side: float) -> None:
    """Reliability row ``chi <= f(mu)`` and side row ``mu <= mu_side``, every day.

    ``chi = state_vars[q]`` and ``f(mu) = c2 (tau - mu)^2 + c1 (tau - mu)``
    (Eq. a_quad), in the form ``impl_of[i, l]``: ``exact``, ``tangent`` (at
    ``tangent_ref * tau``, Eq. a_tangent) or ``pwl`` (``pwl_points`` segments,
    Eqs. a_pwl_sum, a_pwl_seg, a_pwl_cap; one tangent at ``tau / 2`` for a single
    point). The caller sets ``NonConvex`` for the exact form.
    """
    md, T = ctx.model, ctx.T
    chi, mu_var = ctx.state_vars[q], ctx.mu_var
    tau = float(ctx.tau[i, l])
    c2, c1, mu_side = float(c2), float(c1), float(mu_side)
    form = str(ctx.impl_of[i, l])
    P = int(ctx.pwl_points)
    if form not in ("exact", "tangent", "pwl"):
        raise ValueError(f"unknown reliability_impl {form!r} at cell (i={i}, l={l})")

    def f(m):
        return c2 * (tau - m) ** 2 + c1 * (tau - m)

    def df(m):
        return -2.0 * c2 * (tau - m) - c1

    for k in range(T):
        md.addConstr(mu_var[i, l, k] <= mu_side, name=f"rel_side_{i}_{l}_{k}")

    if form == "pwl" and P >= 2:
        if (i, l, 0, 0) not in ctx.pwl_w:
            raise ValueError(
                f"segment binaries missing at cell (i={i}, l={l}); call add_pwl_vars in prepare"
            )
        for k in range(T):
            mu = mu_var[i, l, k]
            md.addConstr(gp.quicksum(ctx.pwl_w[i, l, k, p] for p in range(P)) == 1,
                         name=f"rel_seg_{i}_{l}_{k}")                       # Eq. a_pwl_sum
            for p in range(P):
                lo, hi, mid = p * tau / P, (p + 1) * tau / P, (p + 0.5) * tau / P
                w = ctx.pwl_w[i, l, k, p]
                # Eq. a_pwl_seg
                md.addGenConstrIndicator(w, True, mu >= lo, name=f"rel_lo_{i}_{l}_{k}_{p}")
                md.addGenConstrIndicator(w, True, mu <= hi, name=f"rel_hi_{i}_{l}_{k}_{p}")
                # Eq. a_pwl_cap
                md.addGenConstrIndicator(w, True, chi[i, l, k] <= f(mid) + df(mid) * (mu - mid),
                                         name=f"rel_cap_{i}_{l}_{k}_{p}")
    elif form == "exact":
        for k in range(T):
            mu = mu_var[i, l, k]
            md.addQConstr(chi[i, l, k] <= c2 * (tau - mu) * (tau - mu) + c1 * (tau - mu),
                          name=f"rel_{i}_{l}_{k}")                          # Eq. a_quad
    else:
        if form == "pwl":
            mu_r = tau / 2.0                                # pwl_points == 1: one tangent
        else:
            mu_r = _cell_value(ctx.tangent_ref, i, l, "tangent_ref") * tau
        for k in range(T):
            md.addConstr(chi[i, l, k] <= f(mu_r) + df(mu_r) * (mu_var[i, l, k] - mu_r),
                         name=f"rel_{i}_{l}_{k}")                           # Eq. a_tangent


# ===========================================================================
# General constraints and problem equations (model-agnostic)
# ===========================================================================
def add_assignment_rows(ctx: FleetModel) -> None:
    """One activity per train and day, and every mission flown every day.

    Eqs. a_one_activity and a_coverage; the depot counts as activity 0.
    """
    md, F, M, T = ctx.model, ctx.F, ctx.M, ctx.T
    x = ctx.x

    for i in range(F):
        for k in range(T):
            md.addConstr(gp.quicksum(x[i, j, k] for j in range(M + 1)) <= 1,
                         name=f"assign_{i}_{k}")
    for j in range(1, M + 1):
        for k in range(T):
            md.addConstr(gp.quicksum(x[i, j, k] for i in range(F)) == 1,
                         name=f"demand_{j}_{k}")


def add_depot_rows(ctx: FleetModel) -> None:
    """A component is repaired or replaced only on a depot day, and not both.

    Eq. a_depot: ``x^m + x^r <= x[i, 0, k]``, or ``x^m <= x[i, 0, k]`` without
    replacement. Several components of one train may be served on the same day.
    """
    md = ctx.model
    for i in range(ctx.F):
        for l in range(ctx.L):
            for k in range(ctx.T):
                xm = ctx.m_rep[i, l, k]
                if ctx.r_rep is None:
                    md.addConstr(xm <= ctx.x[i, 0, k], name=f"depot_{i}_{l}_{k}")
                else:
                    md.addConstr(xm + ctx.r_rep[i, l, k] <= ctx.x[i, 0, k],
                                 name=f"depot_{i}_{l}_{k}")


def add_damage_penalty_rows(ctx: FleetModel, penalty_type: str) -> None:
    """Rows of the damage penalty ``u``, one per train and day; none if ``u`` is not built.

    Eqs. a_u_inf and a_u_quad. The quadratic row is a rotated second-order cone
    and needs no ``NonConvex``.
    """
    if ctx.u_var is None:
        return
    md, F, L, T = ctx.model, ctx.F, ctx.L, ctx.T
    u, mu = ctx.u_var, ctx.mu_var
    if penalty_type not in ("inf_norm", "quadratic"):
        raise ValueError(f"penalty_type must be 'inf_norm' or 'quadratic'; got {penalty_type!r}.")
    for i in range(F):
        for k in range(T):
            if penalty_type == "inf_norm":
                md.addConstr(u >= gp.quicksum(mu[i, l, k] for l in range(L)),
                             name=f"u_{i}_{k}")
            else:
                md.addQConstr(u >= gp.quicksum(mu[i, l, k] * mu[i, l, k] for l in range(L)),
                              name=f"u_{i}_{k}")


def build_objective(ctx: FleetModel, cfg, opts: dict) -> None:
    """Set the objective of the mode ``objective_mode`` (Eqs. a_daycost, a_phases, a_objective).

    The cost of day index ``k`` is ``C_M`` per depot day, ``C_R z`` where ``z`` is
    built and ``C_rep x^r`` where replacement is allowed. The term ``C_D u`` is
    added once, and only if ``u`` is built.
    """
    md, F, L, T = ctx.model, ctx.F, ctx.L, ctx.T
    C_M, C_D = float(cfg.C_M), float(cfg.C_D)
    C_R = np.asarray(cfg.C_R, dtype=float)
    C_rep = np.asarray(cfg.C_rep, dtype=float)

    J_init = gp.LinExpr()
    J_op = gp.LinExpr()
    for k in range(T):
        step_cost = C_M * gp.quicksum(ctx.x[i, 0, k] for i in range(F))
        for i in range(F):
            for l in range(L):
                if (i, l, k) in ctx.z_var:
                    step_cost += float(C_R[i, l]) * ctx.z_var[i, l, k]
                if ctx.r_rep is not None:
                    step_cost += float(C_rep[i, l]) * ctx.r_rep[i, l, k]
        if k < ctx.H1:
            J_init += step_cost
        else:
            J_op += step_cost

    mode = str(opts["objective_mode"])
    if mode == "total":
        objective = J_init + J_op
    elif mode == "operating_average":
        objective = (1.0 / ctx.H2) * J_op
    elif mode == "evaluation_total":
        E = int(opts["evaluation_horizon"])
        objective = J_init + ((E - ctx.H1) / ctx.H2) * J_op
    else:
        raise ValueError(
            "objective_mode must be 'total', 'operating_average' or "
            f"'evaluation_total'; got {mode!r}."
        )
    if ctx.u_var is not None:
        objective = objective + C_D * ctx.u_var
    md.setObjective(objective, GRB.MINIMIZE)


# ===========================================================================
# Results
# ===========================================================================
def status_string(code: int) -> str:
    return {
        GRB.OPTIMAL: "optimal",
        GRB.TIME_LIMIT: "time_limit",
        GRB.SUBOPTIMAL: "suboptimal",
        GRB.INTERRUPTED: "interrupted",
        GRB.SOLUTION_LIMIT: "solution_limit",
        GRB.NODE_LIMIT: "node_limit",
        GRB.ITERATION_LIMIT: "iteration_limit",
        GRB.INFEASIBLE: "infeasible",
        GRB.INF_OR_UNBD: "inf_or_unbounded",
        GRB.UNBOUNDED: "unbounded",
    }.get(code, f"gurobi_status_{code}")


def collapse_selector(arr, *, empty_as_none: bool = False):
    """A per-cell (F, L) selector array: one string if every cell agrees, else a nested list.

    The result holds Python strings only, no NumPy objects. With ``empty_as_none``
    an empty string becomes ``None`` (spec, "Result Dictionary").
    """
    values = [[str(v) for v in row] for row in np.asarray(arr)]
    if empty_as_none:
        values = [[None if v == "" else v for v in row] for row in values]
    flat = {v for row in values for v in row}
    if len(flat) == 1:
        return next(iter(flat))
    return values


def transitory_repair_cost(mu, g_mu, m, mu_0, anchor_mu_0, rho, ard1, H1: int) -> np.ndarray:
    """Repair cost ``z`` on the transitory days, computed after the solve (Eq. c_zpost).

    Returns an array ``(F, L, H1)``: ``rho (mu_{k-1} - g_{k-1})`` on a repair day,
    ``0`` otherwise, with ``g = 0`` at an ardinf cell and the initial values at
    the first day.
    """
    mu, g_mu, m = (np.asarray(a, dtype=float) for a in (mu, g_mu, m))
    mu_0, anchor_mu_0 = np.asarray(mu_0, dtype=float), np.asarray(anchor_mu_0, dtype=float)
    out = np.zeros(mu.shape[:2] + (H1,))
    for k in range(H1):
        mu_prev = mu_0 if k == 0 else mu[:, :, k - 1]
        g_prev = anchor_mu_0 if k == 0 else g_mu[:, :, k - 1]
        g_prev = np.where(ard1, g_prev, 0.0)
        out[:, :, k] = np.where(m[:, :, k] > 0.5, rho * (mu_prev - g_prev), 0.0)
    return out


def peak_damage(mu, penalty_type: str) -> float:
    """The value of ``u`` recomputed from ``mu`` (Eq. c_u)."""
    mu = np.asarray(mu, dtype=float)
    if penalty_type == "inf_norm":
        return float(np.max(np.sum(mu, axis=1)))
    if penalty_type == "quadratic":
        return float(np.max(np.sum(mu * mu, axis=1)))
    raise ValueError(f"penalty_type must be 'inf_norm' or 'quadratic'; got {penalty_type!r}.")


def cost_report(x, r, z, C_M, C_R, C_rep, C_D, u, H1: int, H2: int, evaluation_horizon) -> dict:
    """The cost report of the result, from the arrays of the solution.

    Eqs. a_daycost and a_phases on ``x``, ``r`` and ``z`` (zero where ``NaN``),
    and ``P_E`` of Eq. d_projected for ``evaluation_horizon``, if given.
    """
    C_R, C_rep = np.asarray(C_R, dtype=float), np.asarray(C_rep, dtype=float)
    z0 = np.where(np.isnan(z), 0.0, z)
    T = x.shape[2]
    step_costs = np.zeros(T)
    for k in range(T):
        step_costs[k] = (C_M * x[:, 0, k].sum()
                         + (C_R * z0[:, :, k]).sum()
                         + (C_rep * r[:, :, k]).sum())
    J_init = float(step_costs[:H1].sum())
    J_op = float(step_costs[H1:].sum())
    J_op_average = J_op / H2
    u, C_D = float(u), float(C_D)
    penalty = C_D * u
    if evaluation_horizon is None:
        projected = evaluation = None
    else:
        projected = J_init + (int(evaluation_horizon) - H1) * J_op_average
        evaluation = projected + penalty
    return {
        "step_costs": step_costs,
        "J_initialization": J_init,
        "J_op": J_op,
        "J_op_average": J_op_average,
        "peak_damage": u,
        "damage_penalty": penalty,
        "operating_objective": J_op_average + penalty,
        "J_total": J_init + J_op + penalty,
        "projected_evaluation_cost": projected,
        "evaluation_objective": evaluation,
        "component_costs": {"C_M": float(C_M), "C_D": C_D, "C_R": C_R, "C_rep": C_rep},
    }


_SOLUTION_KEYS = ("x", "m", "r", "idle", "mu", "v", "z", "R", "K",
                  "g_mu", "g_v", "g_R", "g_K", "u")
_COST_KEYS = ("step_costs", "J_initialization", "J_op", "J_op_average", "peak_damage",
              "damage_penalty", "operating_objective", "J_total",
              "projected_evaluation_cost", "evaluation_objective", "component_costs")


def _read_values(variables, shape) -> np.ndarray:
    """Values of a dict of variables keyed by index tuples; ``NaN`` where a key is absent."""
    values = np.full(shape, np.nan)
    for key, var in variables.items():
        values[key] = var.X
    return values


def extract_solution(ctx: FleetModel, cfg) -> dict:
    """The result dictionary of the spec ("Result Dictionary") without the reports.

    ``solve_mixed`` adds ``performance``, ``optimization_progress``, ``warm_start``
    and ``relaxation_warm_start``, and calls each builder's ``extract`` hook. The
    solution arrays and the cost report are ``None`` if Gurobi found no solution.
    """
    md = ctx.model
    F, M, L, T, H1, H2 = ctx.F, ctx.M, ctx.L, ctx.T, ctx.H1, ctx.H2
    model = np.asarray(cfg.model).astype(str)
    try:
        bound = float(md.ObjBound)
    except (AttributeError, gp.GurobiError):
        bound = None

    out = {
        "status": status_string(md.Status),
        "objective": None, "mip_gap": None, "bound": bound,
        "F": F, "M": M, "L": L, "H1": H1, "H2": H2, "T": T, "H": cfg.H,
        "model": model.tolist(),
        "component_names": list(cfg.component_names),
        "tau": np.asarray(cfg.tau, dtype=float),
        "mu_0": np.asarray(cfg.mu_0, dtype=float),
        "v_0": np.asarray(cfg.v_0, dtype=float),
        "bound_method": collapse_selector(
            np.where(model == "rainflow", cfg.bound_method, ""), empty_as_none=True),
        "repair_model": collapse_selector(cfg.repair_model),
        "reliability_impl": collapse_selector(cfg.reliability_impl),
        "encoding": collapse_selector(cfg.encoding),
        "objective_mode": ctx.opts["objective_mode"],
        "evaluation_horizon": ctx.opts["evaluation_horizon"],
    }
    if int(md.SolCount) == 0:
        out.update({key: None for key in _SOLUTION_KEYS + _COST_KEYS})
        return out

    out["objective"] = float(md.ObjVal)
    try:
        out["mip_gap"] = float(md.MIPGap)
    except (AttributeError, gp.GurobiError):
        out["mip_gap"] = None

    shape = (F, L, T)
    x = _read_values(ctx.x, (F, M + 1, T))
    m = _read_values(ctx.m_rep, shape)
    r = _read_values(ctx.r_rep, shape) if ctx.r_rep is not None else np.zeros(shape)
    mu = _read_values(ctx.mu_var, shape)
    z = _read_values(ctx.z_var, shape)
    g_mu = _read_values(ctx.anchor_vars["mu"], shape)
    out.update({
        "x": x, "m": m, "r": r,
        "idle": x[:, 0, :][:, None, :] - m - r,
        "mu": mu,
        "v": _read_values(ctx.state_vars["v"], shape),
        "z": z,
        "R": _read_values(ctx.state_vars["R"], shape),
        "K": _read_values(ctx.state_vars["K"], shape),
        "g_mu": g_mu,
        "g_v": _read_values(ctx.anchor_vars["v"], shape),
        "g_R": _read_values(ctx.anchor_vars["R"], shape),
        "g_K": _read_values(ctx.anchor_vars["K"], shape),
    })

    if str(ctx.opts["objective_mode"]) == "operating_average":
        # z is not built on the transitory days; Eq. c_zpost
        C_R = np.asarray(cfg.C_R, dtype=float)
        ard1 = np.asarray(cfg.repair_model).astype(str) == "ard1"
        z[:, :, :H1] = np.where(
            C_R[:, :, None] > 0,
            transitory_repair_cost(mu, g_mu, m, ctx.initial["mu"], ctx.anchor_start["mu"],
                                   ctx.rho, ard1, H1),
            np.nan,
        )

    out["u"] = peak_damage(mu, str(cfg.penalty_type))
    out.update(cost_report(x, r, z, float(cfg.C_M), cfg.C_R, cfg.C_rep, float(cfg.C_D),
                           out["u"], H1, H2, ctx.opts["evaluation_horizon"]))
    return out


# ===========================================================================
# Assembly: build a whole fleet from the registered cell builders
# ===========================================================================
def _load_builders(names) -> None:
    """Import the module of each model in ``names`` so that it registers its cell builder.

    The module is ``fleet_management.degradation_model.<name>``. Done lazily to
    avoid a circular import (the modules import this one). A model without a
    module is reported by ``get_cell_builder``.
    """
    for name in names:
        if name in CELL_BUILDERS:
            continue
        module = f"fleet_management.degradation_model.{name}"
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as error:
            if error.name != module:
                raise


def build_fleet(cfg, opts: dict, model_name: str = "fleet_management") -> FleetModel:
    """Shared skeleton + one block of rows per cell, in the building order of the spec."""
    present = sorted({str(m) for m in np.asarray(cfg.model).ravel()})
    _load_builders(present)
    for name in present:
        get_cell_builder(name)           # fails before any Gurobi object exists

    ctx = build_context(cfg, opts, model_name)

    # per-model preparation (variables of the model's own quantities, solver flags)
    for name in present:
        get_cell_builder(name).prepare(ctx, cfg, ctx.cells_of(name), opts)

    # objective and the rows that do not depend on the model
    build_objective(ctx, cfg, opts)
    add_assignment_rows(ctx)
    add_depot_rows(ctx)
    add_damage_penalty_rows(ctx, str(cfg.penalty_type))

    # per-cell blocks
    for i in range(cfg.F):
        for l in range(cfg.L):
            dispatch_cell(ctx, i, l)
    return ctx


def _model_statistics(model) -> dict:
    """Size and work entries of ``performance`` (spec, "Reports").

    An entry that Gurobi does not give is ``None``.
    """
    model.update()

    def read(getter):
        try:
            return getter()
        except (AttributeError, TypeError, gp.GurobiError):
            return None

    def indicator_count():
        return sum(int(item.GenConstrType == GRB.GENCONSTR_INDICATOR)
                   for item in model.getGenConstrs())

    return {
        "gurobi_version": read(lambda: ".".join(str(v) for v in gp.gurobi.version())),
        "variables": read(lambda: int(model.NumVars)),
        "continuous_variables": read(lambda: int(model.NumVars - model.NumIntVars)),
        "integer_variables": read(lambda: int(model.NumIntVars)),
        "binary_variables": read(lambda: int(model.NumBinVars)),
        "linear_constraints": read(lambda: int(model.NumConstrs)),
        "general_constraints": read(lambda: int(model.NumGenConstrs)),
        "indicator_constraints": read(indicator_count),
        "quadratic_constraints": read(lambda: int(model.NumQConstrs)),
        "nonzeros": read(lambda: int(model.NumNZs)),
        "solutions_found": read(lambda: int(model.SolCount)),
        "gurobi_runtime_seconds": read(lambda: float(model.Runtime)),
        "branch_and_bound_nodes": read(lambda: float(model.NodeCount)),
        "simplex_iterations": read(lambda: float(model.IterCount)),
        "barrier_iterations": read(lambda: float(model.BarIterCount)),
        "work_units": read(lambda: float(model.Work)),
    }


def solve_mixed(cfg, *, warm_start=None, warm_start_mode: str = "start") -> dict:
    """Build, warm start, optimise and extract one fleet; return the whole result.

    The cells may use different degradation models; each cell is built by the
    builder registered for its model. The run options are ``cfg.options``. The
    result is the dictionary of the spec ("Result Dictionary"), reports included.
    """
    if warm_start_mode not in ("start", "hint", "both"):
        raise ValueError(
            f"warm_start_mode must be 'start', 'hint' or 'both'; got {warm_start_mode!r}."
        )
    has_source = warm_start is not None and warm_start.get("x") is not None
    if has_source and warm_start_mode != "start":
        raise NotImplementedError(
            f"warm_start_mode {warm_start_mode!r} is not implemented yet; only 'start' is."
        )

    backend_start = time.perf_counter()
    opts = resolve_run_options(cfg)
    construction_start = time.perf_counter()
    ctx = build_fleet(cfg, opts)
    ctx.model.update()
    warm_start_diagnostics = apply_binary_warm_start(ctx, warm_start)
    warm_start_diagnostics = complete_binary_warm_start(
        ctx.model, warm_start, warm_start_diagnostics
    )
    warm_start_diagnostics["mode"] = warm_start_mode if has_source else None
    relaxation_diagnostics = {
        "enabled": False,
        "status": "disabled",
        "relaxation_objective": None,
        "values_copied": 0,
        "runtime_seconds": None,
    }
    if opts["relaxation_warm_start"]:
        if warm_start_diagnostics["applied"]:
            relaxation_diagnostics.update({"enabled": True, "status": "skipped"})
        else:
            relaxation_diagnostics = apply_relaxation_warm_start(
                ctx.model,
                time_limit=opts["relaxation_time_limit"],
            )
    construction_seconds = time.perf_counter() - construction_start

    optimizer_start = time.perf_counter()
    callback = _initialize_progress(
        ctx.model, opts["progress_interval_seconds"]
    )
    ctx.model.optimize(callback)
    _append_final_progress(ctx.model)
    optimizer_seconds = time.perf_counter() - optimizer_start

    extraction_start = time.perf_counter()
    out = extract_solution(ctx, cfg)
    out["optimization_progress"] = list(ctx.model._optimization_progress)
    out["warm_start"] = warm_start_diagnostics
    out["relaxation_warm_start"] = relaxation_diagnostics
    for name in sorted({str(m) for m in np.asarray(cfg.model).ravel()}):
        hook = getattr(get_cell_builder(name), "extract", None)
        if hook is not None:
            hook(ctx, cfg, out)
    extraction_seconds = time.perf_counter() - extraction_start

    performance = _model_statistics(ctx.model)
    performance.update({
        "model_construction_seconds": construction_seconds,
        "optimizer_call_seconds": optimizer_seconds,
        "solution_extraction_seconds": extraction_seconds,
        "backend_wall_seconds": time.perf_counter() - backend_start,
        "requested_time_limit_seconds": float(ctx.model.Params.TimeLimit),
        "requested_mip_gap": float(ctx.model.Params.MIPGap),
        "threads": int(ctx.model.Params.Threads),
        "seed": int(ctx.model.Params.Seed),
    })
    out["performance"] = performance
    # Release native resources, including an active LogFile handle on Windows,
    # after all solution and diagnostic information has been extracted.
    ctx.model.dispose()
    return out
