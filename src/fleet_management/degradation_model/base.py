"""
Shared Gurobi model layer for fleet management (model-agnostic skeleton).

This module owns everything that does **not** depend on which degradation model
a cell uses:

* the shared context object ``FleetModel`` (sizes, decision variables, per-cell
  parameters, increment accessors);
* the shared variables and the general constraints / problem equations
  (assignment, mission demand, depot capacity, aggregate-damage cap, safety
  ``u``, and the repeatability / loop-closure rows that make the operating
  phase a repeatable cycle);
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
    (gamma)       PLACEHOLDER below; registers "gamma"

`base` must not import the model modules at import time (they import `base`);
``solve_mixed`` imports them lazily so their registration side-effect happens.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
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
    nb: Optional[gp.tupledict]             # no-intervention indicator (F, L, T)
                                           # None when it has been substituted
                                           # out (formulation='bigm'); read it
                                           # through ``nb_of`` / ``act_of``
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
    # --- MILP encoding of the logical (on/off) constraints -----------------
    # 'indicator' : Gurobi general indicator constraints (the original model)
    # 'bigm'      : plain linear big-M rows, nb substituted out
    formulation: str = "indicator"
    bigM: float = 1.1               # fallback big-M when a state has no finite UB
    z_exact: Optional[bool] = None  # pin z with upper rows; None = auto (see rainflow_v2)
    v_inc: Optional[Callable[[int, int, int, int], float]] = None
    w2_inc: Optional[Callable[[int, int, int, int], float]] = None
    cgf_inc: Optional[Callable[[int, int, int, int], float]] = None
    # free-form per-model storage (gamma state, future models, ...)
    extras: dict = field(default_factory=dict)

    # --- helpers ---
    def act_of(self, i: int, l: int, k: int):
        """The 'an intervention happens' expression  ``m + r``  ( = 1 - nb )."""
        if self.allow_replacement:
            return self.m_rep[i, l, k] + self.r_rep[i, l, k]
        return self.m_rep[i, l, k] + 0.0

    def nb_of(self, i: int, l: int, k: int):
        """``nb`` as a variable when it exists, else the expression it equals.

        ``nb`` is fully determined by ``m`` and ``r`` (eq. 3), so the big-M
        formulation substitutes it out; every reader must go through here.
        """
        if self.nb is not None:
            return self.nb[i, l, k]
        return 1.0 - self.act_of(i, l, k)

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

    def repeatability(self, ctx: FleetModel, i: int, l: int,
                      k_ref: int, k_end: int) -> None:
        """OPTIONAL: this model's loop-closure rows for one cell.

        Called by ``add_repeatability_constraints`` once per cell, after the
        model-agnostic mean row ``mu[k_end] <= mu[k_ref]`` has been added.  A
        model implements this only for the extra descriptors it tracks (rainflow
        adds v / R / K); omitting the hook leaves the cell with the mean row.
        """
        ...


# ---------------------------------------------------------------------------
# MILP encodings of the logical constraints.  These are *assembly* choices as
# much as modelling ones:
#   'indicator'  Gurobi general indicator constraints, one addConstr per row
#                (the reference build; rainflow_v2)
#   'bigm'       the same logic as plain linear rows, nb substituted out
#                (a DIFFERENT relaxation; rainflow_v2)
#   'sparse'     row-for-row identical to 'indicator', assembled through the
#                matrix API from scipy.sparse blocks (rainflow_sparse)
# ---------------------------------------------------------------------------
FORMULATIONS = ("indicator", "bigm", "sparse", "bigm_sparse")

# The four values are a 2x2 grid of two independent choices that the historical
# single-string option flattens:
#
#                    assembly='loop'      assembly='sparse'
#   encoding='indicator'   'indicator'          'sparse'
#   encoding='bigm'        'bigm'               'bigm_sparse'
#
# The ENCODING is a modelling choice: 'indicator' and 'bigm' have the same
# integer feasible set but different LP relaxations (run_studies' lp_gap column).
# The ASSEMBLY is not a modelling choice at all: for a fixed encoding the two
# assemblies emit the identical program and differ only in how many Python-level
# API calls it takes to write it down.  Read a formulation string through
# ``encoding_of`` / ``assembly_of`` rather than comparing it to a literal.
_ENCODING = {"indicator": "indicator", "sparse": "indicator",
             "bigm": "bigm", "bigm_sparse": "bigm"}
_ASSEMBLY = {"indicator": "loop", "bigm": "loop",
             "sparse": "sparse", "bigm_sparse": "sparse"}


def encoding_of(formulation: str) -> str:
    """'indicator' or 'bigm': which MILP encoding of the logical constraints."""
    try:
        return _ENCODING[str(formulation).lower()]
    except KeyError:
        raise ValueError(f"unknown formulation {formulation!r}; "
                         f"pick from {FORMULATIONS}.") from None


def assembly_of(formulation: str) -> str:
    """'loop' (one addConstr per row) or 'sparse' (matrix API)."""
    try:
        return _ASSEMBLY[str(formulation).lower()]
    except KeyError:
        raise ValueError(f"unknown formulation {formulation!r}; "
                         f"pick from {FORMULATIONS}.") from None


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


def resolve_builder(ctx: FleetModel, name: str):
    """The builder for ``name``, honouring a per-solve override on ``ctx``.

    ``build_fleet(..., builders={...})`` lets one entry point pin a specific
    implementation without touching the global registry -- which is how the
    legacy ``rainflow`` module keeps working now that ``rainflow_v2`` owns the
    registered "rainflow" name.
    """
    override = ctx.extras.get("_builders") or {}
    return override.get(name) or get_cell_builder(name)


def dispatch_cell(ctx: FleetModel, i: int, l: int) -> None:
    """Route one cell to its model's builder."""
    resolve_builder(ctx, str(ctx.model_of[i, l])).add_cell(ctx, i, l)


# ===========================================================================
# ####################  GAMMA PLACEHOLDER (work in progress)  ###############
# ===========================================================================
# The modular gamma block goes here. A gamma cell shares the fleet skeleton
# (assignment x, depot capacity, the aggregate-damage cap, the safety variable u
# and the objective), so it MUST drive the shared mean state ``ctx.mu_var[i,l,k]``
# — that is what the cap / u / C_D term read. Everything gamma-specific (its own
# state variables, shape/scale bookkeeping) can live in ``ctx.extras["gamma"]``.
#
# To implement:
#   1. `prepare`: create gamma's auxiliary variables for its cells and stash them
#      in ctx.extras["gamma"]; read parameters from cfg (gamma_beta, rho, tau,
#      epsilon, mu_0) and the mean profile through ctx.mu_inc.
#   2. `add_cell`: maintenance gating (can reuse the shared pattern), the gamma
#      degradation recursion driving ctx.mu_var and ctx.z_var, and the gamma
#      reliability constraint.
#
# Note (open modelling question): in a MIXED fleet the profiles are normalized to
# H_prof = H2 whenever any cell is rainflow, while a gamma-only fleet uses H1.
# Decide explicitly how a gamma cell maps onto the shared T = H1 + H2 axis.
# ---------------------------------------------------------------------------
class GammaCellBuilder:
    """PLACEHOLDER builder for gamma cells (not implemented yet)."""

    name = "gamma"

    def prepare(self, ctx: FleetModel, cfg, cells, opts: dict) -> None:
        """Create gamma auxiliary variables. Placeholder: nothing is created yet.

        Intentionally does not raise, so that model *inspection* and the build of
        the rest of a mixed fleet can proceed up to the point where a gamma cell
        actually needs its constraints (`add_cell`).
        """
        ctx.extras.setdefault("gamma", {"cells": list(cells)})

    def add_cell(self, ctx: FleetModel, i: int, l: int) -> None:
        """PLACEHOLDER — the gamma constraint block for one cell."""
        raise NotImplementedError(
            f"cell (i={i}, l={l}) model='gamma': the modular gamma block is not "
            "implemented yet (work in progress). Implement "
            "base.GammaCellBuilder.add_cell — it must drive ctx.mu_var[i, l, k] "
            "(the shared aggregate-damage cap, safety u and the C_D objective "
            "term read it). For a gamma-only fleet, use the existing gamma "
            "backend through solver.solve()."
        )

    def extract(self, ctx: FleetModel, cfg, out: dict) -> None:
        """Optional hook to add gamma-specific arrays to the result."""
        return None


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
        # MILP encoding of the logical constraints (rainflow_v2)
        "formulation": str(pick(overrides.get("formulation"),
                                o.get("formulation"), "indicator")).lower(),
        "bigM": float(pick(overrides.get("bigM"), o.get("bigM"), 1.1)),
        "z_exact": pick(overrides.get("z_exact"), o.get("z_exact"), None),
        # loop-closure / repeatability of the operating phase (eq. loop_impl)
        "repeatability": bool(pick(overrides.get("repeatability"),
                                   o.get("repeatability"), True)),
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

    formulation = str(opts.get("formulation", "indicator")).lower()
    if formulation not in FORMULATIONS:
        raise ValueError(f"unknown formulation {formulation!r}; "
                         f"pick from {FORMULATIONS}.")

    x = md.addVars(F, M + 1, T, vtype=GRB.BINARY, name="x")
    m_rep = md.addVars(F, L, T, vtype=GRB.BINARY, name="m")
    r_rep = md.addVars(F, L, T, vtype=GRB.BINARY, name="r") if allow_replacement else None
    # nb = 1 - m - r is an *implied* binary: the big-M formulation substitutes it
    # out (F*L*T fewer binaries) and reads it through ctx.nb_of / ctx.act_of.
    nb = (md.addVars(F, L, T, vtype=GRB.BINARY, name="nb")
          if encoding_of(formulation) != "bigm" else None)
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
        formulation=formulation,
        bigM=float(opts.get("bigM", 1.1)),
        z_exact=opts.get("z_exact"),
    )

    # Generically valid bounds; a model's prepare may tighten them further.
    # The scalar loop is kept for the reference builds so that 'indicator' is
    # untouched by the sparse work and stays the thing the report describes;
    # the sparse path sets the same numbers with one setAttr per block, which
    # matters because this loop is itself Theta(F*L*T) Python iterations.
    if assembly_of(formulation) == "sparse":
        tau_T = np.repeat(np.asarray(cfg.tau, dtype=float).ravel(), T)
        for block in (ctx.mu_var, ctx.z_var):
            md.setAttr(GRB.Attr.UB, list(block.values()), tau_T.tolist())
    else:
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
    is the no-intervention indicator.

    When ``nb`` has been substituted out (``formulation='bigm'``) the three rows
    collapse to the single, *stronger* row ``m + r <= x_0``: it implies both
    gates and ``m + r <= 1`` at once, so the encoding is sparser AND its LP
    relaxation is tighter than the disaggregated pair."""
    md, T = ctx.model, ctx.T
    x, m_rep, r_rep, nb = ctx.x, ctx.m_rep, ctx.r_rep, ctx.nb
    for k in range(T):
        if nb is None:                                   # nb substituted out
            md.addConstr(ctx.act_of(i, l, k) <= x[i, 0, k],
                         name=f"gate_{i}_{l}_{k}")
            continue
        md.addConstr(m_rep[i, l, k] <= x[i, 0, k], name=f"m_gate_{i}_{l}_{k}")
        if ctx.allow_replacement:
            md.addConstr(r_rep[i, l, k] <= x[i, 0, k], name=f"r_gate_{i}_{l}_{k}")
            md.addConstr(nb[i, l, k] == 1 - m_rep[i, l, k] - r_rep[i, l, k],
                         name=f"nb_def_{i}_{l}_{k}")
        else:
            md.addConstr(nb[i, l, k] == 1 - m_rep[i, l, k], name=f"nb_def_{i}_{l}_{k}")


def loop_indices(ctx: FleetModel):
    """The two time indices the repeatability constraints compare, or None.

    The reference splits the horizon into a transitory phase k = 1..H1 and an
    operating phase k = H1+1..H with H = H1 + H2.  On this module's 0-based
    axis (k = 0..T-1, T = H1 + H2) the end of the transitory phase is
    ``H1 - 1`` and the end of the operating phase is ``T - 1``.

    Returns ``(k_ref, k_end)`` = (end of transitory, end of horizon), or None
    when the horizon has no operating phase to close (H1 >= T), in which case
    every row would read ``s[T-1] <= s[T-1]`` and is dropped rather than added
    as a redundant constraint.
    """
    if ctx.H1 < 1 or ctx.H1 >= ctx.T:
        return None
    return ctx.H1 - 1, ctx.T - 1


def add_repeatability_constraints(ctx: FleetModel) -> None:
    """Moment-based repeatability (loop-closure) constraints, eq. (loop_impl).

    The operating phase is meant to be a cycle the fleet can repeat
    indefinitely, so the state at the end of the horizon must be no worse than
    the state the operating phase started from::

        mu[i,l,H] <= mu[i,l,H1]     all bounds          (mean, here)
        v [i,l,H] <= v [i,l,H1]     Cantelli, Bernstein (per-model hook)
        R [i,l,H] <= R [i,l,H1]     Hoeffding           (per-model hook)

    The mean row is model-agnostic -- every cell drives ``ctx.mu_var`` whatever
    its degradation model -- so it is imposed here for every cell.  Each further
    descriptor belongs to the model that created its variable, so this function
    then calls the cell builder's optional ``repeatability(ctx, i, l, k_ref,
    k_end)`` hook; a builder without one contributes the mean row only.

    Two things to be aware of when reading a solution:

    * The rows are RELATIVE, not absolute.  A schedule can also satisfy them by
      accumulating *more* damage before ``H1``, which raises the right-hand
      side.  What keeps that in check is the ``C_D(u)`` objective term, not
      these constraints.
    * Imposing them can make an instance infeasible that was feasible without,
      whenever the operating phase cannot be closed at all -- e.g. the cheapest
      per-step increment exceeds what repair can remove given ``rho`` and the
      depot capacity.  That is a genuine statement about the instance, not a
      bug; ``repeatability=False`` recovers the open-horizon problem.
    """
    idx = loop_indices(ctx)
    if idx is None:
        return
    k_ref, k_end = idx
    md = ctx.model
    for i, l in ctx.all_cells():
        md.addConstr(ctx.mu_var[i, l, k_end] <= ctx.mu_var[i, l, k_ref],
                     name=f"rep_mu_{i}_{l}")
        hook = getattr(resolve_builder(ctx, str(ctx.model_of[i, l])),
                       "repeatability", None)
        if hook is not None:
            hook(ctx, i, l, k_ref, k_end)


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
        "repeatability": bool(ctx.extras.get("repeatability", False)),
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
    Done lazily to avoid a circular import (they import this module).

    ``rainflow_v2`` carries BOTH encodings (``formulation='indicator'`` is the
    original math, ``'bigm'`` the new one), so it is the builder installed here
    -- deterministically, because the legacy ``rainflow`` module registers under
    the same name and would otherwise win or lose by import order alone.  The
    legacy module keeps working: its ``solve`` pins itself through
    ``build_fleet(..., builders=...)`` instead of through the registry.
    """
    from fleet_management.degradation_model import rainflow_v2
    if not isinstance(CELL_BUILDERS.get("rainflow"),
                      rainflow_v2.RainflowCellBuilder):
        register_cell_builder("rainflow", rainflow_v2.RainflowCellBuilder())


def build_fleet(cfg, opts: dict, model_name: str = "fleet_management_mixed",
                builders: Optional[Dict[str, "CellBuilder"]] = None) -> FleetModel:
    """Shared skeleton + one constraint block per cell, dispatched by model.

    ``builders`` pins a specific implementation for a model name for this solve
    only (see ``resolve_builder``); everything else comes from the registry.
    """
    _t0 = time.perf_counter()
    _load_builders()
    if assembly_of(opts.get("formulation", "indicator")) == "sparse":
        # Whole-fleet array assembly instead of the per-cell dispatch loop
        # below.  Same rows, same columns, same optimum -- see rainflow_sparse.
        from fleet_management.degradation_model import rainflow_sparse
        return rainflow_sparse.build_fleet_sparse(cfg, opts, model_name=model_name)

    ctx = build_context(cfg, opts, model_name=model_name)
    if builders:
        ctx.extras["_builders"] = dict(builders)

    # per-model preparation (auxiliary variables, per-cell arrays, solver flags)
    present = sorted({str(m) for m in np.asarray(cfg.model).ravel()})
    for name in present:
        builder = resolve_builder(ctx, name)
        builder.prepare(ctx, cfg, ctx.cells_of(name), opts)

    # shared objective and general constraints
    build_objective(ctx, resolve_costs(cfg, cfg.tau))
    add_base_constraints(ctx, int(opts["depot_capacity"]))

    # per-cell blocks
    for i in range(cfg.F):
        for l in range(cfg.L):
            dispatch_cell(ctx, i, l)

    # loop closure, last: it reads the state variables the cell blocks drive
    ctx.extras["repeatability"] = bool(opts.get("repeatability", True))
    if ctx.extras["repeatability"]:
        add_repeatability_constraints(ctx)
    ctx.extras["build_s"] = time.perf_counter() - _t0
    return ctx


def solve_mixed(cfg, **overrides) -> dict:
    """Solve a fleet whose cells may use **different** degradation models.

    Builds the shared skeleton once (``base``) and fills in each cell's
    constraints through its registered builder. An unimplemented model (today:
    gamma) raises ``NotImplementedError`` from its placeholder block.
    """
    _load_builders()
    opts = resolve_run_options(cfg, **overrides)
    ctx = build_fleet(cfg, opts)
    ctx.model.optimize()
    out = extract_solution(ctx, cfg, ctx.model)
    for name in sorted({str(m) for m in np.asarray(cfg.model).ravel()}):
        hook = getattr(get_cell_builder(name), "extract", None)
        if hook is not None:
            hook(ctx, cfg, out)
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