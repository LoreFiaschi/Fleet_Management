"""
Shared Gurobi model layer for fleet management (model-agnostic skeleton).

This module owns everything that does **not** depend on which degradation model
a cell uses:

* the shared context object ``FleetModel`` (sizes, decision variables, per-cell
  parameters, increment accessors);
* the shared variables and the general constraints / problem equations
  (assignment, mission demand, depot capacity, aggregate-damage cap, safety
  ``u``);
* the objective  ``J = C_M(x) + C_R(z) + C_rep(r) + C_D(u)``;
* solution extraction, status decoding, run-option and cost resolution.

Degradation models plug in through the **cell-builder registry**: each model
registers a ``CellBuilder`` with

    prepare(ctx, cfg, cells, opts)   create that model's auxiliary variables and
                                     per-cell arrays for *its* cells (called once
                                     per model that appears in the fleet)
    add_cell(ctx, i, l)              add the constraints of ONE (vehicle,
                                     component) cell
    extract(ctx, cfg, out)           optional: add model-specific arrays to the
                                     result dict

``solve_mixed`` then builds one program for a fleet whose cells use *different*
degradation models: shared skeleton once, then the right block per cell.

Layering
--------
    base.py       shared skeleton + registry + solve_mixed   (this file)
    rainflow.py   rainflow cell math + reliability bounds; registers "rainflow"
    gamma block   finite-horizon common-rate tail builder below; registers "gamma"

`base` must not import the model modules at import time (they import `base`);
``solve_mixed`` imports them lazily so their registration side-effect happens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable, Dict, Optional, Protocol

import numpy as np
import gurobipy as gp
from gurobipy import GRB


# ===========================================================================
# Shared context
# ===========================================================================
@dataclass
class FleetModel:
    """Everything the per-cell builders need, built once and passed around.

    Required fields are model-agnostic. The optional ones are filled in by a
    degradation model's ``prepare`` hook for the cells that need them (e.g.
    rainflow's variance / latch / descriptor variables), so a builder can read
    ``ctx.<field>`` exactly as if it owned the context.
    """
    # --- structure ---
    model: gp.Model
    F: int; H1: int; H2: int; M: int; L: int; T: int
    # --- shared decision variables / states ---
    x: gp.tupledict                        # assignment  (F, M+1, T)
    m_rep: gp.tupledict                    # imperfect repair (F, L, T)
    r_rep: Optional[gp.tupledict]          # replacement (F, L, T) or None
    nb: gp.tupledict                       # no-intervention indicator (F, L, T)
    mu_var: gp.tupledict                   # mean damage state (F, L, T)
    z_var: gp.tupledict                    # removed expected damage (F, L, T)
    u_var: gp.tupledict                    # aggregate damage per step (T,)
    # --- shared per-cell (F, L) parameters ---
    model_of: np.ndarray
    tau: np.ndarray
    eps: np.ndarray
    rho: np.ndarray
    mu_0: np.ndarray
    mu_new: np.ndarray
    allow_replacement: bool
    mu_inc: Callable[[int, int, int, int], float]   # (i, j0, l, k) -> mean increment

    # --- optional, model-specific (filled by a model's prepare hook) ---
    v_var: Optional[gp.tupledict] = None
    gmu: Optional[gp.tupledict] = None
    gv: Optional[gp.tupledict] = None
    gR: Optional[gp.tupledict] = None      # ARD1 latch for the Hoeffding R budget
    R_var: Optional[gp.tupledict] = None
    K_var: Optional[gp.tupledict] = None
    bound_of: Optional[np.ndarray] = None
    repair_of: Optional[np.ndarray] = None
    v_0: Optional[np.ndarray] = None
    v_new: Optional[np.ndarray] = None
    s_chernoff: Optional[np.ndarray] = None
    support_max_of: Optional[np.ndarray] = None
    Le: Optional[np.ndarray] = None
    ln_eps: Optional[np.ndarray] = None
    track_v_of: Optional[np.ndarray] = None
    latch_of: Optional[np.ndarray] = None
    impl_of: dict = field(default_factory=dict)
    pwl_points: int = 8
    tangent_ref: float = 0.5
    v_inc: Optional[Callable[[int, int, int, int], float]] = None
    w2_inc: Optional[Callable[[int, int, int, int], float]] = None
    cgf_inc: Optional[Callable[[int, int, int, int], float]] = None
    # free-form per-model storage (gamma state, future models, ...)
    extras: dict = field(default_factory=dict)

    # --- helpers ---
    def cells_of(self, model_name: str):
        """All (i, l) cells whose degradation model is ``model_name``."""
        return [(i, l) for i in range(self.F) for l in range(self.L)
                if str(self.model_of[i, l]) == model_name]

    def all_cells(self):
        return [(i, l) for i in range(self.F) for l in range(self.L)]

    def rainflow_cells(self):               # backwards-compatible alias
        return self.cells_of("rainflow")


# ===========================================================================
# Cell-builder registry
# ===========================================================================
class CellBuilder(Protocol):
    """Interface a degradation model implements to plug into the shared model."""

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
# (assignment x, depot capacity, the aggregate-damage cap, the safety variable u
# and the objective), so it MUST drive the shared mean state ``ctx.mu_var[i,l,k]``
# — that is what the cap / u / C_D term read. Everything gamma-specific (its own
# state variables, shape/scale bookkeeping) can live in ``ctx.extras["gamma"]``.
#
# The numerical calibration is independent of Gurobi. This block consumes its
# bounded mission shapes, creates A', and keeps physical expected damage mu as a
# separate shared state. Initial and replacement distributions are calibrated as
# mutually exclusive seed histories. ARD-inf repair receives no reduction in
# the tail-bound shape (a safe pathwise-dominance baseline) while physical mean
# damage is contracted normally.
# ---------------------------------------------------------------------------
class GammaCellBuilder:
    """Finite-horizon common-rate tail bound for Gamma cells.

    Initial and replacement distributions are jointly calibrated with all
    finite-horizon increment combinations. ARD-inf repair is conservative by
    keeping the bounding shape unchanged; ARD1 remains unsupported rather than
    silently applying the legacy constant-rate approximation.
    """

    name = "gamma"

    def prepare(self, ctx: FleetModel, cfg, cells, opts: dict) -> None:
        """Calibrate mission shapes offline and create the bounding state."""
        from fleet_management.degradation_model.gamma_utils.gamma_tail_bound import (
            calculate_seeded_profile_tail_bound_parameters,
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
        common_rate = np.zeros((ctx.F, ctx.L))
        maximum_shape = np.zeros((ctx.F, ctx.L))
        bounded_trans = {}
        bounded_operating = {}
        initial_shape = np.zeros((ctx.F, ctx.L))
        replacement_shape = np.zeros((ctx.F, ctx.L))
        calibrations = {}
        calibration_seconds = {}

        beta_trans_cfg = getattr(cfg, "gamma_beta_trans", None)
        beta_bound_cfg = getattr(cfg, "gamma_beta_bound", None)
        for i, l in cells:
            repair_model = str(cfg.repair_model[i, l])
            if repair_model != "ardinf":
                raise NotImplementedError(
                    f"gamma cell (i={i}, l={l}): modular Gamma currently "
                    "supports only repair_model='ardinf'. ARD1 needs an "
                    "additional last-maintenance state and tail certification."
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
            calibration = calculate_seeded_profile_tail_bound_parameters(
                expected_damage=combined_mu,
                rates=combined_beta,
                threshold=float(ctx.tau[i, l]),
                max_total_count=ctx.T,
                initial_expected_damage=float(ctx.mu_0[i, l]),
                initial_rate=(
                    None if beta_0_cfg is None else float(beta_0_cfg[i, l])
                ),
                replacement_expected_damage=float(ctx.mu_new[i, l]),
                replacement_rate=(
                    None if beta_new_cfg is None else float(beta_new_cfg[i, l])
                ),
                common_rate=selected_rate,
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
            ctx.impl_of[(i, l)] = "gamma_finite_tail"
            for k in range(ctx.T):
                A_var[i, l, k].UB = float(maximum_shape[i, l])

        ctx.extras["gamma"] = {
            "cells": cells,
            "A_var": A_var,
            "common_rate": common_rate,
            "maximum_shape": maximum_shape,
            "bounded_trans": bounded_trans,
            "bounded_operating": bounded_operating,
            "initial_shape": initial_shape,
            "replacement_shape": replacement_shape,
            "calibrations": calibrations,
            "calibration_seconds": calibration_seconds,
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
        md = ctx.model

        # ARD-inf repair gets no decrease in A'. This is safe because, pathwise,
        # c*D + S <= D + S for c=1-rho and every future nonnegative increment
        # sum S. The finite-history calibration already bounds D+S. Physical mu
        # still contracts exactly, so repair can help the shared capacity state.
        rho = float(ctx.rho[i, l])
        remaining = 1.0 - rho
        for k in range(ctx.T):
            A_prev = initial_shape if k == 0 else A_var[i, l, k - 1]
            mu_prev = float(ctx.mu_0[i, l]) if k == 0 else ctx.mu_var[i, l, k - 1]
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

            md.addGenConstrIndicator(
                ctx.nb[i, l, k], True,
                A_var[i, l, k] == A_prev + shape_inc,
                name=f"A_gamma_carry_{i}_{l}_{k}",
            )
            md.addGenConstrIndicator(
                ctx.nb[i, l, k], True,
                ctx.mu_var[i, l, k] == mu_prev + mean_inc,
                name=f"mu_gamma_carry_{i}_{l}_{k}",
            )
            md.addGenConstrIndicator(
                ctx.nb[i, l, k], True,
                ctx.z_var[i, l, k] == 0.0,
                name=f"z_gamma_zero_{i}_{l}_{k}",
            )
            md.addGenConstrIndicator(
                ctx.m_rep[i, l, k], True,
                A_var[i, l, k] == A_prev,
                name=f"A_gamma_ardinf_{i}_{l}_{k}",
            )
            md.addGenConstrIndicator(
                ctx.m_rep[i, l, k], True,
                ctx.mu_var[i, l, k] == remaining * mu_prev,
                name=f"mu_gamma_ardinf_{i}_{l}_{k}",
            )
            md.addGenConstrIndicator(
                ctx.m_rep[i, l, k], True,
                ctx.z_var[i, l, k] == rho * mu_prev,
                name=f"z_gamma_ardinf_{i}_{l}_{k}",
            )
            if ctx.allow_replacement:
                md.addGenConstrIndicator(
                    ctx.r_rep[i, l, k], True,
                    A_var[i, l, k] == replacement_shape,
                    name=f"A_gamma_repl_{i}_{l}_{k}",
                )
                md.addGenConstrIndicator(
                    ctx.r_rep[i, l, k], True,
                    ctx.mu_var[i, l, k] == float(ctx.mu_new[i, l]),
                    name=f"mu_gamma_repl_{i}_{l}_{k}",
                )
                md.addGenConstrIndicator(
                    ctx.r_rep[i, l, k], True,
                    ctx.z_var[i, l, k] == mu_prev - float(ctx.mu_new[i, l]),
                    name=f"z_gamma_repl_{i}_{l}_{k}",
                )
            md.addConstr(
                A_var[i, l, k] <= maximum,
                name=f"rel_gamma_{i}_{l}_{k}",
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
        summaries = []
        for i, l in data["cells"]:
            rate = float(data["common_rate"][i, l])
            calibration = data["calibrations"][i, l]
            constraints = calibration.compressed.constraints
            for k in range(ctx.T):
                value = data["A_var"][i, l, k].X
                shape[i, l, k] = value
                tail[i, l, k] = gamma_distribution.sf(
                    float(ctx.tau[i, l]), a=value, scale=1.0 / rate
                ) if value > 0.0 else 0.0
            summaries.append({
                "i": i,
                "l": l,
                "common_rate": rate,
                "increment_types": int(calibration.type_max_counts.size),
                "increment_opportunities": int(calibration.original_shapes.size),
                "seed_types": int(calibration.increment_offset),
                "calibration_lp_variables": int(
                    calibration.compressed.original_shapes.size
                ),
                "tail_constraints": len(constraints),
                "calibration_seconds": float(
                    data["calibration_seconds"][i, l]
                ),
                "total_convolution_series_terms": int(sum(
                    item.convolution_series_terms for item in constraints
                )),
                "maximum_convolution_series_terms": int(max(
                    (item.convolution_series_terms for item in constraints),
                    default=0,
                )),
                "maximum_convolution_remaining_mass": float(max(
                    (item.convolution_remaining_mass for item in constraints),
                    default=0.0,
                )),
                "worst_calibration_margin": calibration.worst_tail_margin,
                "initial_bounded_shape": calibration.initial_bounded_shape,
                "replacement_bounded_shape": calibration.replacement_bounded_shape,
                "repair_bound": "ardinf_no_tail_credit",
            })
        out["gamma_shape_bound"] = shape
        out["gamma_tail_bound"] = tail
        out["gamma_beta_bound"] = data["common_rate"]
        out["gamma_calibration"] = summaries


register_cell_builder("gamma", GammaCellBuilder())


# ===========================================================================
# Options, costs, accessors
# ===========================================================================
def pick(explicit, from_options, default):
    """First non-None of (explicit kwarg, cfg.options value, default)."""
    if explicit is not None:
        return explicit
    if from_options is not None:
        return from_options
    return default


def resolve_run_options(cfg, **overrides) -> dict:
    """Merge explicit kwargs, ``cfg.options`` and defaults into one dict."""
    o = cfg.options
    F, M = cfg.F, cfg.M
    return {
        "allow_replacement": pick(overrides.get("allow_replacement"),
                                  o.get("allow_replacement"), True),
        "depot_capacity": int(pick(overrides.get("depot_capacity"),
                                   o.get("depot_capacity"), F - M)),
        "verbose": pick(overrides.get("verbose"), o.get("verbose"), 1),
        "mip_gap": pick(overrides.get("mip_gap"), o.get("mip_gap"), 0.12),
        "time_limit": pick(overrides.get("time_limit"), o.get("time_limit"), None),
        "fast": pick(overrides.get("fast"), o.get("fast"), False),
        "gurobi_params": pick(overrides.get("gurobi_params"),
                              o.get("gurobi_params"), None),
        # reliability-constraint implementation (rainflow cells)
        "reliability_impl": pick(overrides.get("reliability_impl"),
                                 o.get("reliability_impl"), "exact"),
        "pwl_points": int(pick(overrides.get("pwl_points"), o.get("pwl_points"), 8)),
        "tangent_ref": float(pick(overrides.get("tangent_ref"),
                                  o.get("tangent_ref"), 0.5)),
        "replacement_as_new": bool(pick(overrides.get("replacement_as_new"),
                                        o.get("replacement_as_new"), True)),
    }


def resolve_costs(cfg, tau) -> dict:
    """Objective coefficients: J = C_M(x) + C_R(z) + C_rep(r) + C_D(u).

    ``C_D`` is the damage-regularisation coefficient (legacy alias ``C_S``).
    ``C_rep`` defaults to ``C_R * max(tau)`` when not supplied.
    """
    c = dict(cfg.costs)
    if "C_D" not in c or c["C_D"] is None:
        if "C_S" in c and c["C_S"] is not None:
            c["C_D"] = float(c["C_S"])
        else:
            raise KeyError("missing damage cost 'C_D' (legacy alias 'C_S').")
    for key in ("C_M", "C_R"):
        if key not in c:
            raise KeyError(f"missing required cost coefficient '{key}'.")
    if "C_rep" not in c or c["C_rep"] is None:
        c["C_rep"] = float(c["C_R"]) * float(np.max(tau))
    return c


def cell_max(op, tr, i, l) -> float:
    """Max of a profile over cell (i, l) across operating and transitory arrays,
    each shaped (F, L, M, H)."""
    m = float(op[i, l].max()) if op is not None else 0.0
    if tr is not None:
        m = max(m, float(tr[i, l].max()))
    return m


def make_accessor(op, tr, H1, H2, transform=lambda a: a):
    """Phase-aware increment accessor ``f(i, j0, l, k)`` over a (F, L, M, H)
    profile: steps k < H1 read the transitory array when present, otherwise the
    operating one; later steps cycle the operating profile."""
    if op is None and tr is None:
        return lambda *a: 0.0

    def f(i, j0, l, k):
        if k < H1:
            src = tr if tr is not None else op
            h = k % (H1 if tr is not None else H2)
            return float(transform(src[i, l, j0, h]))
        return float(transform(op[i, l, j0, (k - H1) % H2]))
    return f


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


# ===========================================================================
# Model + shared variables
# ===========================================================================
def build_context(cfg, opts: dict, model_name: str = "fleet_management") -> FleetModel:
    """Create the Gurobi model, the shared variables, and the context object."""
    F, L, M = cfg.F, cfg.L, cfg.M
    H1, H2, T = cfg.H1, cfg.H2, cfg.T
    allow_replacement = bool(opts["allow_replacement"])

    md = gp.Model(model_name)
    md.Params.OutputFlag = int(opts["verbose"])
    apply_performance_params(md, opts["time_limit"], opts["mip_gap"],
                             opts["fast"], opts["gurobi_params"])

    x = md.addVars(F, M + 1, T, vtype=GRB.BINARY, name="x")
    m_rep = md.addVars(F, L, T, vtype=GRB.BINARY, name="m")
    r_rep = md.addVars(F, L, T, vtype=GRB.BINARY, name="r") if allow_replacement else None
    nb = md.addVars(F, L, T, vtype=GRB.BINARY, name="nb")
    mu_var = md.addVars(F, L, T, lb=0.0, name="mu")
    z_var = md.addVars(F, L, T, lb=0.0, name="z")
    u_var = md.addVars(T, lb=0.0, name="u")

    ctx = FleetModel(
        model=md, F=F, H1=H1, H2=H2, M=M, L=L, T=T,
        x=x, m_rep=m_rep, r_rep=r_rep, nb=nb,
        mu_var=mu_var, z_var=z_var, u_var=u_var,
        model_of=cfg.model, tau=cfg.tau, eps=cfg.epsilon, rho=cfg.rho,
        mu_0=cfg.mu_0,
        mu_new=(cfg.replacement_mu if cfg.replacement_mu is not None
                else np.zeros((F, L))),
        allow_replacement=allow_replacement,
        mu_inc=make_accessor(cfg.mu, cfg.mu_trans, H1, H2),
        pwl_points=int(opts["pwl_points"]), tangent_ref=float(opts["tangent_ref"]),
    )

    # generically valid bounds; a model's prepare may tighten them further
    for i in range(F):
        for l in range(L):
            t = float(cfg.tau[i, l])
            for k in range(T):
                ctx.mu_var[i, l, k].UB = t
                ctx.z_var[i, l, k].UB = t
    return ctx


# ===========================================================================
# General constraints and problem equations (model-agnostic)
# ===========================================================================
def add_base_constraints(ctx: FleetModel, depot_capacity: int) -> None:
    """Assignment, mission demand, depot capacity, aggregate-damage cap, safety u.

    These couple *all* cells, so every (i, l) must have a ``mu_var`` recursion
    defined by its model's cell builder.
    """
    md, F, M, L, T = ctx.model, ctx.F, ctx.M, ctx.L, ctx.T
    x, mu_var, u_var = ctx.x, ctx.mu_var, ctx.u_var

    # one activity per vehicle and step (depot counts as activity 0)
    for i in range(F):
        for k in range(T):
            md.addConstr(gp.quicksum(x[i, j, k] for j in range(M + 1)) <= 1,
                         name=f"assign_{i}_{k}")
    # every mission is served at every step
    for j in range(1, M + 1):
        for k in range(T):
            md.addConstr(gp.quicksum(x[i, j, k] for i in range(F)) == 1,
                         name=f"demand_{j}_{k}")
    # maintenance-slot capacity
    for k in range(T):
        md.addConstr(gp.quicksum(x[i, 0, k] for i in range(F)) <= depot_capacity,
                     name=f"depot_cap_{k}")
    # aggregate damage cap and safety variable u
    for k in range(T):
        md.addConstr(gp.quicksum(mu_var[i, l, k] for i in range(F) for l in range(L))
                     <= F - M, name=f"capacity_{k}")
        for i in range(F):
            md.addConstr(u_var[k] >= gp.quicksum(mu_var[i, l, k] for l in range(L)),
                         name=f"u_{i}_{k}")


def add_maintenance_gating(ctx: FleetModel, i: int, l: int) -> None:
    """Maintenance gating (reference eq. 3), shared by every degradation model:
    repair/replacement each require a depot day, at most one of them, and ``nb``
    is the no-intervention indicator."""
    md, T = ctx.model, ctx.T
    x, m_rep, r_rep, nb = ctx.x, ctx.m_rep, ctx.r_rep, ctx.nb
    for k in range(T):
        md.addConstr(m_rep[i, l, k] <= x[i, 0, k], name=f"m_gate_{i}_{l}_{k}")
        if ctx.allow_replacement:
            md.addConstr(r_rep[i, l, k] <= x[i, 0, k], name=f"r_gate_{i}_{l}_{k}")
            md.addConstr(nb[i, l, k] == 1 - m_rep[i, l, k] - r_rep[i, l, k],
                         name=f"nb_def_{i}_{l}_{k}")
        else:
            md.addConstr(nb[i, l, k] == 1 - m_rep[i, l, k], name=f"nb_def_{i}_{l}_{k}")


def build_objective(ctx: FleetModel, costs: dict) -> None:
    """J = C_M(x) + C_R(z) + C_rep(r) + C_D(u):
    maintenance access, imperfect repair, replacement, damage regularisation."""
    md, F, L, T = ctx.model, ctx.F, ctx.L, ctx.T
    C_M, C_R, C_D, C_rep = costs["C_M"], costs["C_R"], costs["C_D"], costs["C_rep"]
    obj = gp.LinExpr()
    for k in range(T):
        obj += C_D * ctx.u_var[k]
        for i in range(F):
            obj += C_M * ctx.x[i, 0, k]
            for l in range(L):
                obj += C_R * ctx.z_var[i, l, k]
                if ctx.allow_replacement:
                    obj += C_rep * ctx.r_rep[i, l, k]
    md.setObjective(obj, GRB.MINIMIZE)


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


def collapse(arr):
    """A per-cell (F, L) selector array -> a single string when every cell agrees,
    else a plain nested list. Keeps results free of NumPy objects."""
    if arr is None:
        return None
    flat = {str(v) for v in np.asarray(arr).ravel()}
    if len(flat) == 1:
        return next(iter(flat))
    return np.asarray(arr).astype(str).tolist()


def collapse_dict(impl_of, F, L):
    """Per-cell impl names -> one string when all agree, else a nested list."""
    flat = set(impl_of.values())
    if not flat:
        return ""
    if len(flat) == 1:
        return next(iter(flat))
    grid = [["" for _ in range(L)] for _ in range(F)]
    for (i, l), name in impl_of.items():
        grid[i][l] = name
    return grid


def extract_solution(ctx: FleetModel, cfg, model) -> dict:
    """Shared result dict; each model's ``extract`` hook may add its own arrays."""
    F, L, M, T = ctx.F, ctx.L, ctx.M, ctx.T
    meta = {
        "status": status_string(model.status),
        "method": collapse(cfg.bound_method),
        "bound_method": collapse(cfg.bound_method),
        "repair_model": collapse(cfg.repair_model),
        "reliability_impl": collapse_dict(ctx.impl_of, F, L),
        "models": cfg.models,
        "F": F, "H": cfg.H, "H1": ctx.H1, "H2": ctx.H2, "T": T, "M": M, "L": L,
        "tau": cfg.tau, "mu_0": cfg.mu_0, "v_0": cfg.v_0, "model": model,
    }
    if model.SolCount == 0:
        meta.update({"objective": None, "mip_gap": None, "bound": None,
                     "x": None, "mu": None, "v": None, "z": None,
                     "m": None, "r": None, "u": None})
        return meta

    try:
        gap = float(model.MIPGap)
    except (AttributeError, gp.GurobiError):
        gap = None
    try:
        objbnd = float(model.ObjBound)
    except (AttributeError, gp.GurobiError):
        objbnd = None

    x_sol = np.zeros((F, M + 1, T)); mu_sol = np.zeros((F, L, T))
    v_sol = np.zeros((F, L, T)); z_sol = np.zeros((F, L, T))
    m_sol = np.zeros((F, L, T)); r_sol = np.zeros((F, L, T)); u_sol = np.zeros(T)
    track_v = ctx.track_v_of
    for k in range(T):
        u_sol[k] = ctx.u_var[k].X
        for i in range(F):
            for j in range(M + 1):
                x_sol[i, j, k] = ctx.x[i, j, k].X
            for l in range(L):
                mu_sol[i, l, k] = ctx.mu_var[i, l, k].X
                if ctx.v_var is not None and (track_v is None or track_v[i, l]):
                    v_sol[i, l, k] = ctx.v_var[i, l, k].X
                z_sol[i, l, k] = ctx.z_var[i, l, k].X
                m_sol[i, l, k] = ctx.m_rep[i, l, k].X
                if ctx.allow_replacement:
                    r_sol[i, l, k] = ctx.r_rep[i, l, k].X
    meta.update({"objective": model.ObjVal, "mip_gap": gap, "bound": objbnd,
                 "x": x_sol, "mu": mu_sol, "v": v_sol, "z": z_sol,
                 "m": m_sol, "r": r_sol, "u": u_sol})
    return meta


# ===========================================================================
# Assembly: build a whole fleet from the registered cell builders
# ===========================================================================
def _load_builders() -> None:
    """Import the model modules so they register their cell builders.
    Done lazily to avoid a circular import (they import this module)."""
    if "rainflow" not in CELL_BUILDERS:
        from fleet_management.degradation_model import rainflow  # noqa: F401


def build_fleet(cfg, opts: dict, model_name: str = "fleet_management_mixed") -> FleetModel:
    """Shared skeleton + one constraint block per cell, dispatched by model."""
    _load_builders()
    ctx = build_context(cfg, opts, model_name=model_name)

    # per-model preparation (auxiliary variables, per-cell arrays, solver flags)
    present = sorted({str(m) for m in np.asarray(cfg.model).ravel()})
    for name in present:
        builder = get_cell_builder(name)
        builder.prepare(ctx, cfg, ctx.cells_of(name), opts)

    # shared objective and general constraints
    build_objective(ctx, resolve_costs(cfg, cfg.tau))
    add_base_constraints(ctx, int(opts["depot_capacity"]))

    # per-cell blocks
    for i in range(cfg.F):
        for l in range(cfg.L):
            dispatch_cell(ctx, i, l)
    return ctx


def solve_mixed(cfg, **overrides) -> dict:
    """Solve a fleet whose cells may use **different** degradation models.

    Builds the shared skeleton once (``base``) and fills in each cell's
    constraints through its registered builder. An unimplemented model (today:
    gamma) raises ``NotImplementedError`` from its placeholder block.
    """
    backend_start = time.perf_counter()
    _load_builders()
    opts = resolve_run_options(cfg, **overrides)
    construction_start = time.perf_counter()
    ctx = build_fleet(cfg, opts)
    ctx.model.update()
    construction_seconds = time.perf_counter() - construction_start

    optimizer_start = time.perf_counter()
    ctx.model.optimize()
    optimizer_seconds = time.perf_counter() - optimizer_start

    extraction_start = time.perf_counter()
    out = extract_solution(ctx, cfg, ctx.model)
    for name in sorted({str(m) for m in np.asarray(cfg.model).ravel()}):
        hook = getattr(get_cell_builder(name), "extract", None)
        if hook is not None:
            hook(ctx, cfg, out)
    extraction_seconds = time.perf_counter() - extraction_start

    from fleet_management.degradation_model.gamma_utils.gamma_diagnostics import (
        collect_gurobi_model_statistics,
        compare_estimate_with_actual,
        estimate_gamma_formulation,
    )

    performance = collect_gurobi_model_statistics(ctx.model)
    performance.update({
        "model_construction_seconds": construction_seconds,
        "optimizer_call_seconds": optimizer_seconds,
        "solution_extraction_seconds": extraction_seconds,
        "backend_wall_seconds": time.perf_counter() - backend_start,
    })
    if "gamma" in ctx.extras:
        performance["gamma_calibration_seconds"] = float(sum(
            ctx.extras["gamma"]["calibration_seconds"].values()
        ))
        formulation = estimate_gamma_formulation(
            cfg, allow_replacement=ctx.allow_replacement
        )
        formulation["actual_gurobi_model"] = {
            key: performance[key]
            for key in (
                "variables",
                "continuous_variables",
                "integer_variables",
                "binary_variables",
                "linear_constraints",
                "general_constraints",
                "indicator_constraints",
                "quadratic_constraints",
                "nonzeros",
            )
        }
        formulation["comparison"] = compare_estimate_with_actual(
            formulation, formulation["actual_gurobi_model"]
        )
        out["gamma_formulation"] = formulation
    performance["backend_wall_seconds"] = time.perf_counter() - backend_start
    out["performance"] = performance
    return out


# ===========================================================================
# Mathematical interface (kept from the original base.py)
# ===========================================================================
class AccumulatedDegradationModel(Protocol):
    """Mathematical interface for accumulated-degradation models."""

    name: str

    def increment_parameter(self, expected_damage: np.ndarray) -> np.ndarray:
        """Convert expected mission damage into model parameters."""
        ...

    def expected_damage(self, state_parameter: np.ndarray) -> np.ndarray:
        """Return expected damage represented by the state."""
        ...

    def tail_probability(self, state_parameter: np.ndarray,
                         threshold: float) -> np.ndarray:
        """Return P(D > threshold)."""
        ...
