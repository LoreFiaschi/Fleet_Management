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


def warm_start_day(k: int, H1: int, source_H1: int, source_H2: int) -> int:
    """Source day (0-based) whose values day ``k`` takes (spec, "Day Mapping", Eq. d_daymap).

    The initialization days keep their index while both horizons have them. Every
    later day repeats the operating days of the source cyclically.
    """
    if k < min(H1, source_H1):
        return k
    return source_H1 + max(0, k - H1) % source_H2


def apply_binary_warm_start(ctx: FleetModel, source: dict | None, mode: str = "start") -> dict:
    """Map a source onto the horizon of the model as values of the binary variables.

    Spec, "Interface and Reports", "Day Mapping" and "Start, Hints and Completion".
    ``source`` carries ``x`` ``(F', M'+1, T')``, the integers ``H1`` and ``H2`` with
    ``T' = H1 + H2``, and may carry ``m`` and ``r`` ``(F', L', T')``. Day ``k`` takes
    the values of the source day ``warm_start_day`` (Eq. d_daymap), rounded to the
    nearest integer, for the indices that the source and the model both have. The
    mode ``"start"`` sets ``Start``, ``"hint"`` sets ``VarHintVal`` and ``"both"``
    sets both. ``x`` takes the value of ``x'``; ``x^m`` that of ``m'`` (``0`` if the
    source has no ``m``); ``x^r`` that of ``r'``. The selector ``n`` takes
    ``max(0, 1 - x^m - x^r)``, or ``0`` if one of them is ``1``. A source without ``x``
    has no effect. Gurobi reconstructs the continuous variables.

    Returns the report ``warm_start`` without ``completion``: ``applied``, ``values``
    (the variables that got a value), ``mode`` and ``source_horizon``.
    """
    if not source or source.get("x") is None:
        return {"applied": False, "values": 0, "mode": None, "source_horizon": None}
    try:
        H1_src, H2_src = int(source["H1"]), int(source["H2"])
    except (KeyError, TypeError, ValueError):
        H1_src = H2_src = 0
    if H1_src < 1 or H2_src < 1:
        raise ValueError("a warm-start source must carry the integers 'H1' and 'H2' (both >= 1).")
    x_src = np.asarray(source["x"], dtype=float)
    if x_src.ndim != 3 or x_src.shape[2] != H1_src + H2_src:
        raise ValueError(
            f"'x' of a warm-start source must have the shape (F', M'+1, {H1_src + H2_src}); "
            f"got {x_src.shape}."
        )
    m_src = None if source.get("m") is None else np.asarray(source["m"], dtype=float)
    r_src = None if source.get("r") is None else np.asarray(source["r"], dtype=float)

    def set_value(var, value: float) -> None:
        if mode in ("start", "both"):
            var.Start = value
        if mode in ("hint", "both"):
            var.VarHintVal = value

    values = 0
    for k in range(ctx.T):
        day = warm_start_day(k, ctx.H1, H1_src, H2_src)
        for i in range(min(ctx.F, x_src.shape[0])):
            for j in range(min(ctx.M + 1, x_src.shape[1])):
                set_value(ctx.x[i, j, k], float(round(x_src[i, j, day])))
                values += 1
            for l in range(ctx.L):
                if m_src is None:
                    m_value = 0.0
                elif i < m_src.shape[0] and l < m_src.shape[1]:
                    m_value = float(round(m_src[i, l, day]))
                else:
                    m_value = None
                if ctx.r_rep is None:
                    r_value = 0.0
                elif r_src is not None and i < r_src.shape[0] and l < r_src.shape[1]:
                    r_value = float(round(r_src[i, l, day]))
                else:
                    r_value = None
                if m_value is not None:
                    set_value(ctx.m_rep[i, l, k], m_value)
                    values += 1
                if r_value is not None and ctx.r_rep is not None:
                    set_value(ctx.r_rep[i, l, k], r_value)
                    values += 1
                if (i, l, k) in ctx.nb:
                    if m_value is not None and r_value is not None:
                        n_value = max(0.0, 1.0 - m_value - r_value)
                    elif m_value == 1.0 or r_value == 1.0:
                        n_value = 0.0
                    else:
                        continue
                    set_value(ctx.nb[i, l, k], n_value)
                    values += 1

    return {"applied": values > 0, "values": values, "mode": mode,
            "source_horizon": [H1_src, H2_src]}


def complete_binary_warm_start(
    model: gp.Model,
    source: dict | None,
    report: dict,
    mode: str = "start",
) -> dict:
    """Complete a discrete start by solving its fixed-binary model (spec, "Start, Hints and Completion").

    Gurobi can attempt to complete a partial MIP start itself, but that effort
    is deliberately limited and may reject a feasible discrete schedule before
    reconstructing the many continuous degradation and product variables.  A
    source may therefore request this explicit phase (``complete_binary_start``,
    as the greedy heuristic does): copy the complete model, fix every binary
    variable that has a ``Start`` value, solve the remaining model within
    ``completion_time_limit`` of the source, and copy the resulting full solution
    back through ``Start`` attributes.

    The completion runs only in the modes ``"start"`` and ``"both"``, and only if
    the mapping set at least one value (``report["applied"]``). Otherwise the status
    stays ``"not_requested"``; ``requested`` is always the flag of the source.

    The original model is never relaxed.  A completed start is therefore a
    feasible point of exactly the model that will subsequently be optimized.
    """
    requested = bool(source and source.get("complete_binary_start", False))
    completion = {
        "requested": requested,
        "status": "not_requested",
        "fixed_integer_variables": 0,
        "values_copied": 0,
        "runtime_seconds": None,
        "iis_constraints": [],
    }
    report["completion"] = completion
    if not (requested and mode in ("start", "both") and report.get("applied", False)):
        return report

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
        source.get("completion_time_limit", 60.0)
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
    return report


def apply_relaxation_warm_start(
    model: gp.Model,
    *,
    time_limit: float,
) -> dict:
    """Solve an optional phase-I feasibility relaxation and copy a MIP start (spec, "Relaxation Start").

    Gurobi relaxes linear rows with an L1 violation objective while retaining
    integrality and the model's general constraints. The relaxed solution is
    *not* reported as feasible for the original problem: its integer values become
    ``Start`` values, and the original model must still establish a valid
    incumbent. The relaxed copy is disposed of before the function returns.
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
        relaxed.dispose()
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
    relaxed.dispose()
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

    backend_start = time.perf_counter()
    opts = resolve_run_options(cfg)
    construction_start = time.perf_counter()
    ctx = build_fleet(cfg, opts)
    ctx.model.update()
    warm_start_diagnostics = apply_binary_warm_start(ctx, warm_start, warm_start_mode)
    warm_start_diagnostics = complete_binary_warm_start(
        ctx.model, warm_start, warm_start_diagnostics, warm_start_mode
    )
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
