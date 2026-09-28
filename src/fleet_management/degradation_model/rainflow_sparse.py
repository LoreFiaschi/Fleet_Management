"""
Rainflow cells, sparse assembly:  the SAME program as ``formulation='indicator'``,
built through the matrix API instead of one ``addConstr`` per row.

Why this module exists
----------------------
The sparsity chapter measures ``nnz/(m+n) ~ 2.5`` and density well below
1 %, and concludes that the cost of a solve is the branch-and-bound tree, not
the linear algebra.  That conclusion is about the *solver*.  It says nothing
about the *builder*, and the builder is where a Python MILP model actually
becomes slow: ``base.build_fleet`` walks ``for i: for l: dispatch_cell``, and
each cell writes its rows one ``addConstr`` at a time, each with a
``gp.quicksum`` over the mission index and an f-string name.  That is
``Theta(FLH)`` Python-level API calls with a fat constant, and on the large end
of the scaling ladder it overtakes the solve it was supposed to set up.

Sparsity is what makes the alternative possible.  Because every row's support is
bounded by ``M+3`` uniformly in ``F``, ``L`` and ``H``, and because the per-cell
block is *identical* across cells up to its data, the whole linear part of the
program can be written down as COO triplets with NumPy and handed to Gurobi in a
handful of ``addMConstr`` calls -- one per constraint sense -- instead of
``Theta(FLH)`` calls.

There are two independent choices here, which the single ``formulation`` string
flattens into four values (see ``base.encoding_of`` / ``base.assembly_of``):

                        assembly='loop'        assembly='sparse'
    encoding='indicator'    'indicator'            'sparse'
    encoding='bigm'         'bigm'                 'bigm_sparse'

The ENCODING is a modelling choice, and ``rainflow_v2`` owns it: 'indicator'
and 'bigm' have the same integer feasible set but different LP relaxations.
The ASSEMBLY is not a modelling choice at all, and this module owns it: for a
fixed encoding the two assemblies emit the identical program.

What is and is not vectorised
-----------------------------
Vectorised into ``O(1)`` API calls (all of it pure NumPy / scipy.sparse):

* the shared skeleton -- assignment, mission demand, depot capacity, the
  aggregate-damage cap and the ``u`` epigraph rows;
* maintenance gating (``m <= x_0``, ``r <= x_0``, ``nb = 1 - m - r``);
* the repeatability / loop-closure rows;
* every *linear* reliability row: markov, the ``mu <= tau`` gaps, the single
  supporting tangent, and the pwl segment-selection equalities;
* the objective, built once as ``LinExpr(coeffs, vars)``;
* the variable upper bounds, set with one ``setAttr`` per block instead of one
  assignment per variable.

* under ``encoding='bigm'``, the state recursions too -- every row of
  ``_state_bigm`` / ``_latch_bigm`` / ``_z_bigm`` is a plain linear inequality,
  so there is nothing left outside ``RowSet`` and the build stops being
  ``Theta(F*L*T)`` API calls altogether.

NOT vectorised, because Gurobi has no matrix form for them:

* under ``encoding='indicator'``, the ``6*S*C`` general indicator constraints of
  the state recursions;
* the ``addQConstr`` rows of an ``exact`` quadratic reliability encoding;
* the pwl per-segment indicator rows (again, only under ``encoding='indicator'``:
  the big-M pwl encoding selects its segment with two aggregated linear rows).

Those still cost one API call each.  What this module does for them is remove
the fat: a ``gp.LinExpr(coeffs, vars)`` built from a precomputed NumPy
coefficient tensor, the 5-argument ``addGenConstrIndicator`` signature (no
``TempConstr`` from operator overloading), flat-list variable access instead of
``tupledict`` hashing, and no f-string names unless asked.

That split is the whole result, and it is worth stating plainly:

    'sparse'       ~2x faster to build, and it SATURATES there.  Once the
                   linear families are free, the indicator calls are the build.
    'bigm_sparse'  ~5-7x, and it keeps paying as the instance grows, because
                   there is no per-row API call left to dominate.

So the encoding and the assembly are independent as options but not in effect:
'bigm' is what makes 'sparse' worth having.  ``test_sparse_version.py`` measures
both, side by side, for exactly this reason.

Equivalence
-----------
This is not a reformulation.  The emitted program is row-for-row the one
``formulation='indicator'`` emits, so the objective, the LP relaxation, the
node count and the optimal schedule must all agree exactly.  Anything else is a
bug in this file, and ``test_sparse_version.py`` is written to find it: it
compares the two constraint matrices as canonicalised multisets of rows, and
the indicator constraints as canonicalised triples, not just the objectives.

Where things live
-----------------
    solve                       entry point (mirrors rainflow_v2.solve)
    build_fleet_sparse          whole-model assembly; called by base.build_fleet
    ColumnMap                   flat column index over the model's variables
    RowSet                      COO accumulator -> one addMConstr per sense
    profile_to_T                (F,L,M,H) profile -> dense (F,L,M,T) coefficients
    _base_rows                  shared skeleton, vectorised
    _gating_rows                maintenance gating, vectorised
    _repeatability_rows         loop closure, vectorised
    _reliability_rows           per (bound, impl) family, vectorised where linear
    _state_indicators             encoding='indicator': the recursions that must
                                  stay scalar
    _state_bigm                   encoding='bigm': the recursions, vectorised
      _CellSteps                    the (cell, step) index arrays, split at k=0
      _state_bigm_group             one state's four-to-six big-M rows
      _latch_bigm_group             one ARD1 latch's four rows
      _z_bigm_rows                  eq. 6 without indicators
    _rel_pwl_bigm               aggregated pwl segment selection, vectorised

Author: Johann Tschan  (sparse assembly of the indicator formulation)
"""

from __future__ import annotations

import time
from typing import Dict, List, Sequence

import numpy as np
import gurobipy as gp
from gurobipy import GRB

try:                                    # scipy is the natural home for COO->CSR
    from scipy import sparse as _sp
except ImportError:                     # pragma: no cover - checked at build time
    _sp = None

from fleet_management.degradation_model.base import (
    FleetModel as _RFModel,
    assembly_of,
    build_context,
    encoding_of,
    loop_indices,
    resolve_costs as _resolve_costs,
    resolve_run_options,
    extract_solution as _extract_solution,
)


SENSES = (GRB.LESS_EQUAL, GRB.EQUAL, GRB.GREATER_EQUAL)


# ===========================================================================
# Entry point
# ===========================================================================
def solve(cfg, **overrides) -> dict:
    """Solve a rainflow fleet with the sparse assembly of either encoding.

    Accepts the same keywords as ``rainflow_v2.solve``.  ``formulation``
    defaults to ``'sparse'`` (the indicator program) and may be given as
    ``'bigm_sparse'``; a loop-assembly value is promoted to its sparse twin, so
    ``rainflow_sparse.solve(cfg, formulation='bigm')`` does what it looks like.
    Everything else -- the reliability registry, the result dict, the config
    schema -- is shared, so this is a drop-in swap for ``rainflow_v2.solve``.
    """
    wanted = str(overrides.get("formulation") or "sparse").lower()
    overrides["formulation"] = ("bigm_sparse" if encoding_of(wanted) == "bigm"
                                else "sparse")
    opts = resolve_run_options(cfg, **overrides)
    ctx = build_fleet_sparse(cfg, opts)
    ctx.model.optimize()
    out = _extract_solution(ctx, cfg, ctx.model)
    out["formulation"] = str(ctx.formulation)
    out["sparse_cuts"] = str(ctx.sparse_cuts)
    out["build_s"] = float(ctx.extras.get("build_s", float("nan")))
    return out


# ===========================================================================
# Column index:  every variable of the model, as one flat vector
# ===========================================================================
class ColumnMap:
    """A flat column ordering over the model's variables, addressed by NumPy.

    ``addMConstr(A, xvars, sense, b)`` needs ``A``'s columns to line up with a
    list of ``Var``.  Blocks are registered in the order the model created them,
    each with its logical shape, and ``idx`` turns broadcastable index arrays
    into flat column indices with ``ravel_multi_index`` -- no Python loop and no
    ``tupledict`` hashing on the hot path.
    """

    def __init__(self) -> None:
        self._offset: Dict[str, int] = {}
        self._shape: Dict[str, tuple] = {}
        self.vars: List[gp.Var] = []

    @property
    def n(self) -> int:
        return len(self.vars)

    def add(self, name: str, block, shape: tuple) -> None:
        """Register a variable block.  ``block`` is a tupledict or a sequence.

        ``tupledict.values()`` preserves creation order, which for
        ``addVars(d1, ..., dn)`` is lexicographic in the indices -- the same
        order ``ravel_multi_index`` assumes.
        """
        if block is None:
            return
        vals = list(block.values()) if hasattr(block, "values") else list(block)
        n_expect = int(np.prod(shape)) if shape else len(vals)
        if len(vals) != n_expect:
            raise ValueError(f"block {name!r}: {len(vals)} vars for shape {shape}.")
        self._offset[name] = len(self.vars)
        self._shape[name] = tuple(shape)
        self.vars.extend(vals)

    def has(self, name: str) -> bool:
        return name in self._offset

    def idx(self, name: str, *index_arrays) -> np.ndarray:
        """Flat column indices for a broadcast index tuple into block ``name``."""
        shape = self._shape[name]
        if len(index_arrays) != len(shape):
            raise ValueError(f"block {name!r} has rank {len(shape)}, "
                             f"got {len(index_arrays)} indices.")
        arrs = np.broadcast_arrays(*[np.asarray(a) for a in index_arrays])
        return self._offset[name] + np.ravel_multi_index(arrs, shape)

    def var(self, name: str, *index) -> gp.Var:
        """One ``Var`` by logical index (scalar path, for the indicator rows)."""
        return self.vars[int(self.idx(name, *index))]

    def block(self, name: str) -> np.ndarray:
        """The block's ``Var`` objects as an object array of its logical shape.

        Object-array fancy indexing is how the scalar indicator loop gets its
        variables without paying for a dict lookup per term.
        """
        off, shape = self._offset[name], self._shape[name]
        n = int(np.prod(shape))
        out = np.empty(n, dtype=object)
        out[:] = self.vars[off:off + n]
        return out.reshape(shape)


# ===========================================================================
# Row accumulator:  COO triplets -> one addMConstr per sense
# ===========================================================================
class RowSet:
    """Accumulate linear rows as COO triplets, then emit them in three calls.

    ``add(sense, rhs, *terms)`` appends a *family* of ``N`` structurally
    identical rows at once.  Each term is ``(cols, coeffs)`` where ``cols`` has
    shape ``(N,)`` (one nonzero per row) or ``(N, p)`` (a ``p``-term sum, e.g.
    the mission index or the components of a vehicle), and ``coeffs``
    broadcasts against it.  ``N`` is inferred from the first term.

    Duplicate ``(row, col)`` pairs are summed by the COO->CSR conversion, which
    is the correct behaviour: a variable appearing twice in a row must end up
    with the sum of its coefficients, exactly as ``LinExpr`` would do.
    """

    def __init__(self, ncols: int) -> None:
        if _sp is None:
            raise ImportError(
                "formulation='sparse' needs scipy (pip install scipy); the "
                "assembly is COO -> CSR before it reaches Gurobi.")
        self.ncols = int(ncols)
        self._acc = {s: {"r": [], "c": [], "v": [], "b": [], "n": 0} for s in SENSES}

    def add(self, sense: str, rhs, *terms) -> int:
        """Append ``N`` rows; return ``N``.  ``rhs`` is a scalar or shape (N,)."""
        if sense not in self._acc:
            raise ValueError(f"sense must be one of {SENSES}, got {sense!r}.")
        first = np.asarray(terms[0][0])
        n = first.shape[0] if first.ndim else 1
        if n == 0:
            return 0
        acc = self._acc[sense]
        base = acc["n"]
        for cols, coeffs in terms:
            cols = np.asarray(cols, dtype=np.int64)
            if cols.ndim == 1:
                cols = cols[:, None]
            if cols.shape[0] != n:
                raise ValueError(f"term has {cols.shape[0]} rows, expected {n}.")
            co = np.asarray(coeffs, dtype=float)
            if co.ndim == 1 and co.shape[0] == n:      # one coefficient per row
                co = co[:, None]
            vals = np.broadcast_to(co, cols.shape).ravel()
            rows = np.repeat(base + np.arange(n, dtype=np.int64), cols.shape[1])
            acc["r"].append(rows)
            acc["c"].append(cols.ravel())
            acc["v"].append(vals)
        acc["b"].append(np.broadcast_to(np.asarray(rhs, dtype=float), (n,)).copy())
        acc["n"] = base + n
        return n

    def emit(self, model: gp.Model, xvars: Sequence[gp.Var]) -> int:
        """Hand every accumulated family to Gurobi: at most one call per sense."""
        total = 0
        for sense in SENSES:
            acc = self._acc[sense]
            if acc["n"] == 0:
                continue
            A = _sp.csr_matrix(
                (np.concatenate(acc["v"]),
                 (np.concatenate(acc["r"]), np.concatenate(acc["c"]))),
                shape=(acc["n"], self.ncols))
            A.sum_duplicates()
            model.addMConstr(A, list(xvars), sense, np.concatenate(acc["b"]))
            total += acc["n"]
        return total

    def nnz(self) -> int:
        return int(sum(sum(v.size for v in acc["v"]) for acc in self._acc.values()))


# ===========================================================================
# Profiles:  (F, L, M, H_prof) input arrays -> dense (F, L, M, T) coefficients
# ===========================================================================
def profile_to_T(op, tr, H1: int, H2: int, T: int, F: int, L: int, M: int,
                 transform=None) -> np.ndarray:
    """Vectorised twin of ``base.make_accessor``.

    ``make_accessor`` returns a closure evaluated once per ``(i, j, l, k)``;
    on the hot path that is ``F*L*M*T`` Python calls purely to look up a float.
    The mapping is a gather, so build the whole tensor once:

        k < H1 : transitory array at ``k % H1`` when present, else the
                 operating array at ``k % H2``
        k >= H1: operating array at ``(k - H1) % H2``
    """
    if op is None and tr is None:
        return np.zeros((F, L, M, T))
    src_lo = tr if tr is not None else op
    per_lo = H1 if tr is not None else H2
    k = np.arange(T)
    h_lo = k[:H1] % per_lo
    h_hi = (k[H1:] - H1) % H2
    out = np.empty((F, L, M, T), dtype=float)
    out[..., :H1] = np.asarray(src_lo, dtype=float)[..., h_lo]
    out[..., H1:] = np.asarray(op, dtype=float)[..., h_hi]
    return transform(out) if transform is not None else out


# ===========================================================================
# Assembly
# ===========================================================================
def build_fleet_sparse(cfg, opts: dict,
                       model_name: str = "fleet_management_rainflow_sparse"
                       ) -> _RFModel:
    """Build the whole program with the matrix API.  Same rows as 'indicator'.

    Mirrors ``base.build_fleet``'s sequence -- context, per-model ``prepare``,
    objective, shared constraints, per-cell blocks, loop closure -- but replaces
    the per-cell dispatch loop by whole-fleet array assembly.  ``prepare`` is
    reused unchanged: it creates the auxiliary state variables and tightens the
    bounds, and neither is a hot path (one call per *model*, not per cell).
    """
    t0 = time.perf_counter()

    formulation = str(opts.get("formulation", "sparse")).lower()
    if assembly_of(formulation) != "sparse":
        raise ValueError(f"build_fleet_sparse needs an assembly='sparse' "
                         f"formulation ('sparse' or 'bigm_sparse'); "
                         f"got {formulation!r}.")
    encoding = encoding_of(formulation)
    models = sorted({str(m) for m in np.asarray(cfg.model).ravel()})
    if models != ["rainflow"]:
        raise NotImplementedError(
            f"the sparse assembly covers a rainflow-only fleet; this one has "
            f"{models}. Use formulation='indicator' for a mixed fleet, or add "
            f"the missing model's blocks to _state_indicators.")

    from fleet_management.degradation_model import rainflow_v2

    ctx = build_context(cfg, opts, model_name=model_name)
    rainflow_v2.prepare(ctx, cfg, ctx.all_cells(), opts)
    ctx.model.update()                       # bounds must be readable as big-Ms

    t_prep = time.perf_counter()
    cols = _column_map(ctx)
    rows = RowSet(cols.n)
    costs = _resolve_costs(cfg, cfg.tau)

    _objective(ctx, cols, costs)
    _base_rows(ctx, cols, rows, int(opts["depot_capacity"]))
    _gating_rows(ctx, cols, rows)

    prof = _profiles(ctx, cfg)
    t_lin0 = time.perf_counter()
    if encoding == "bigm":
        # Every row is linear here, so the state recursions go through RowSet
        # like everything else and NOTHING is left at one API call per row.
        _state_bigm(ctx, cols, rows, prof)
    else:
        _state_indicators(ctx, cols, rows, prof)
        # The sparse strengthening only exists for the indicator encoding; the
        # big-M rows already imply every one of these.
        _sparse_cut_rows(ctx, cols, rows, prof)
    t_ind = time.perf_counter() - t_lin0

    _reliability_rows(ctx, cols, rows)
    ctx.extras["repeatability"] = bool(opts.get("repeatability", True))
    if ctx.extras["repeatability"]:
        _repeatability_rows(ctx, cols, rows)

    t_emit0 = time.perf_counter()
    n_lin = rows.emit(ctx.model, cols.vars)
    ctx.model.update()
    t_emit = time.perf_counter() - t_emit0

    total = time.perf_counter() - t0
    ctx.extras.update({
        "sparse_columns": cols,
        "sparse_linear_rows": n_lin,
        "sparse_linear_nnz": rows.nnz(),
        "build_s": total,
        # A phase breakdown, because the headline number hides the shape of the
        # result.  'state_rows' is the state recursion: under the indicator
        # encoding it is one API call per row with no matrix form and therefore
        # no asymptotic improvement, under big-M it is COO triplets like
        # everything else.  'emit' is the COO -> CSR -> addMConstr handover that
        # replaced Theta(F*L*T) addConstr calls.
        "build_phases_s": {
            "context_and_prepare": t_prep - t0,
            "state_rows": t_ind,
            "other_assembly": total - (t_prep - t0) - t_ind - t_emit,
            "emit": t_emit,
        },
    })
    return ctx


def _column_map(ctx: _RFModel) -> ColumnMap:
    """Register every variable block in the order the model created it."""
    F, L, T, M = ctx.F, ctx.L, ctx.T, ctx.M
    cols = ColumnMap()
    cols.add("x", ctx.x, (F, M + 1, T))
    cols.add("m", ctx.m_rep, (F, L, T))
    if ctx.allow_replacement:
        cols.add("r", ctx.r_rep, (F, L, T))
    if ctx.nb is not None:
        cols.add("nb", ctx.nb, (F, L, T))
    cols.add("mu", ctx.mu_var, (F, L, T))
    cols.add("z", ctx.z_var, (F, L, T))
    cols.add("u", ctx.u_var, (T,))
    for name, block in (("v", ctx.v_var), ("gmu", ctx.gmu), ("gv", ctx.gv),
                        ("R", ctx.R_var), ("gR", ctx.gR), ("K", ctx.K_var)):
        if block is not None:
            cols.add(name, block, (F, L, T))
    return cols


def _profiles(ctx: _RFModel, cfg) -> dict:
    """Every per-mission increment tensor this fleet needs, shaped (F, L, M, T)."""
    F, L, M, T, H1, H2 = ctx.F, ctx.L, ctx.M, ctx.T, ctx.H1, ctx.H2
    bounds = {str(b) for b in np.asarray(ctx.bound_of).ravel()}
    out = {"mu": profile_to_T(cfg.mu, cfg.mu_trans, H1, H2, T, F, L, M)}
    if bool(np.any(ctx.track_v_of)):
        out["v"] = profile_to_T(cfg.v, cfg.v_trans, H1, H2, T, F, L, M)
    if "hoeffding" in bounds:
        out["R"] = profile_to_T(cfg.support, cfg.support_trans, H1, H2, T, F, L, M,
                                transform=lambda a: a * a)
    if "chernoff" in bounds:
        out["K"] = profile_to_T(cfg.cgf, cfg.cgf_trans, H1, H2, T, F, L, M)
    return out


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
def _objective(ctx: _RFModel, cols: ColumnMap, costs: dict) -> None:
    """J = C_M(x_0) + C_R(z) + C_rep(r) + C_D(u), as ONE LinExpr.

    ``base.build_objective`` grows a ``LinExpr`` with ``+=`` inside a triple
    loop; every ``+=`` allocates.  The coefficient vector is dense and known in
    closed form, so build it with NumPy and construct the expression once.
    """
    F, L, T = ctx.F, ctx.L, ctx.T
    c = np.zeros(cols.n)
    k = np.arange(T)
    i = np.arange(F)[:, None]
    c[cols.idx("x", i, 0, k[None, :])] = float(costs["C_M"])
    c[cols.idx("z", np.arange(F)[:, None, None], np.arange(L)[None, :, None],
               k[None, None, :])] = float(costs["C_R"])
    c[cols.idx("u", k)] = float(costs["C_D"])
    if ctx.allow_replacement:
        c[cols.idx("r", np.arange(F)[:, None, None], np.arange(L)[None, :, None],
                   k[None, None, :])] = float(costs["C_rep"])
    nz = np.nonzero(c)[0]
    ctx.model.setObjective(
        gp.LinExpr(c[nz].tolist(), [cols.vars[j] for j in nz]), GRB.MINIMIZE)


# ---------------------------------------------------------------------------
# Shared skeleton  (twin of base.add_base_constraints)
# ---------------------------------------------------------------------------
def _base_rows(ctx: _RFModel, cols: ColumnMap, rows: RowSet,
               depot_capacity: int) -> None:
    """Assignment, mission demand, depot capacity, damage cap, safety ``u``."""
    F, L, M, T = ctx.F, ctx.L, ctx.M, ctx.T
    i = np.arange(F); j = np.arange(M + 1); k = np.arange(T)

    # sum_j x[i,j,k] <= 1                                   (F*T rows, M+1 nnz)
    ik_i, ik_k = np.meshgrid(i, k, indexing="ij")
    rows.add(GRB.LESS_EQUAL, 1.0,
             (cols.idx("x", ik_i.ravel()[:, None], j[None, :],
                       ik_k.ravel()[:, None]), 1.0))

    # sum_i x[i,j,k] == 1  for j >= 1                       (M*T rows, F nnz)
    jk_j, jk_k = np.meshgrid(np.arange(1, M + 1), k, indexing="ij")
    rows.add(GRB.EQUAL, 1.0,
             (cols.idx("x", i[None, :], jk_j.ravel()[:, None],
                       jk_k.ravel()[:, None]), 1.0))

    # sum_i x[i,0,k] <= depot_capacity                      (T rows, F nnz)
    rows.add(GRB.LESS_EQUAL, float(depot_capacity),
             (cols.idx("x", i[None, :], 0, k[:, None]), 1.0))

    # sum_{i,l} mu[i,l,k] <= F - M                          (T rows, F*L nnz)
    il_i, il_l = np.meshgrid(i, np.arange(L), indexing="ij")
    rows.add(GRB.LESS_EQUAL, float(F - M),
             (cols.idx("mu", il_i.ravel()[None, :], il_l.ravel()[None, :],
                       k[:, None]), 1.0))

    # u[k] - sum_l mu[i,l,k] >= 0                           (F*T rows, L+1 nnz)
    rows.add(GRB.GREATER_EQUAL, 0.0,
             (cols.idx("u", ik_k.ravel()), 1.0),
             (cols.idx("mu", ik_i.ravel()[:, None], np.arange(L)[None, :],
                       ik_k.ravel()[:, None]), -1.0))


# ---------------------------------------------------------------------------
# Maintenance gating  (twin of base.add_maintenance_gating, nb present)
# ---------------------------------------------------------------------------
def _gating_rows(ctx: _RFModel, cols: ColumnMap, rows: RowSet) -> None:
    """``m <= x_0``, ``r <= x_0``, ``nb = 1 - m - r`` over every cell-step."""
    F, L, T = ctx.F, ctx.L, ctx.T
    ci, cl, ck = (a.ravel() for a in np.meshgrid(np.arange(F), np.arange(L),
                                                 np.arange(T), indexing="ij"))
    x0 = cols.idx("x", ci, 0, ck)
    m = cols.idx("m", ci, cl, ck)
    r = cols.idx("r", ci, cl, ck) if ctx.allow_replacement else None

    if ctx.nb is None:
        # nb substituted out: the three rows collapse to the single, stronger
        # row  m + r <= x_0,  which implies both gates and  m + r <= 1  at once.
        terms = [(m, 1.0), (x0, -1.0)]
        if r is not None:
            terms.insert(1, (r, 1.0))
        rows.add(GRB.LESS_EQUAL, 0.0, *terms)
        return

    rows.add(GRB.LESS_EQUAL, 0.0, (m, 1.0), (x0, -1.0))
    if r is not None:
        rows.add(GRB.LESS_EQUAL, 0.0, (r, 1.0), (x0, -1.0))
        rows.add(GRB.EQUAL, 1.0, (cols.idx("nb", ci, cl, ck), 1.0),
                 (m, 1.0), (r, 1.0))
    else:
        rows.add(GRB.EQUAL, 1.0, (cols.idx("nb", ci, cl, ck), 1.0), (m, 1.0))


# ---------------------------------------------------------------------------
# Repeatability  (twin of base.add_repeatability_constraints)
# ---------------------------------------------------------------------------
def _repeatability_rows(ctx: _RFModel, cols: ColumnMap, rows: RowSet) -> None:
    """``s[T-1] <= s[H1-1]`` for the mean and every descriptor the cells track.

    ``base.add_repeatability_constraints`` imposes the mean row for every cell
    and then calls the builder's ``repeatability`` hook;
    ``rainflow_v2._repeatability`` adds one row per EXTRA descriptor the cell's
    bound reads:

        v[k_end] <= v[k_ref]    Cantelli, Bernstein
        R[k_end] <= R[k_ref]    Hoeffding
        K[k_end] <= K[k_ref]    Chernoff

    Markov reads only the mean, so the shared row already covers it.  The ARD1
    latch registers (gmu / gv / gR) are deliberately left free: they are memory
    of the last intervention, not degradation state.

    This is the sparse twin of that pair, and it must stay in step with it.  A
    descriptor added on one side and not the other is exactly the kind of
    silent divergence ``test_sparse_version.py``'s (S1) exists to catch.
    """
    idx = loop_indices(ctx)
    if idx is None:
        return
    k_ref, k_end = idx

    def _rows_for(block: str, cells) -> None:
        cells = np.asarray(cells, dtype=int).reshape(-1, 2)
        if not len(cells) or not cols.has(block):
            return
        rows.add(GRB.LESS_EQUAL, 0.0,
                 (cols.idx(block, cells[:, 0], cells[:, 1], k_end), 1.0),
                 (cols.idx(block, cells[:, 0], cells[:, 1], k_ref), -1.0))

    # the model-agnostic mean row, for every cell
    _rows_for("mu", ctx.all_cells())

    # the per-bound descriptor rows (rainflow_v2._repeatability)
    bound_of = np.asarray(ctx.bound_of).astype(str)
    track_v = np.asarray(ctx.track_v_of, dtype=bool)
    _rows_for("v", [(i, l) for i, l in ctx.all_cells() if track_v[i, l]])
    for block, bound in (("R", "hoeffding"), ("K", "chernoff")):
        _rows_for(block, [(i, l) for i, l in ctx.all_cells()
                          if bound_of[i, l] == bound])


# ---------------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------------
def _cell_groups(ctx: _RFModel) -> Dict[tuple, np.ndarray]:
    """Cells grouped by ``(bound, impl)``, as arrays of ``(i, l)`` pairs.

    The grouping loop is over ``F*L`` cells, not ``F*L*T`` cell-steps, so it is
    off the hot path; everything inside a group is then done with array ops.
    """
    groups: Dict[tuple, list] = {}
    for i, l in ctx.all_cells():
        key = (str(ctx.bound_of[i, l]), ctx.impl_of.get((i, l), "exact"))
        groups.setdefault(key, []).append((i, l))
    return {k: np.asarray(v, dtype=int) for k, v in groups.items()}


def _rel_gap_rows(ctx, cols, rows, cells, ub) -> None:
    """``mu[i,l,k] <= ub[i,l]`` for a group of cells, all steps at once."""
    T = ctx.T
    ci = np.repeat(cells[:, 0], T); cl = np.repeat(cells[:, 1], T)
    ck = np.tile(np.arange(T), len(cells))
    rows.add(GRB.LESS_EQUAL, np.repeat(ub, T), (cols.idx("mu", ci, cl, ck), 1.0))


def _rel_tangent_rows(ctx, cols, rows, cells, mu_p_frac) -> None:
    """``Q <= g(mu_p) + g'(mu_p)(mu - mu_p)`` for a group, all steps at once.

    Same halfspace as ``rainflow_v2._add_tangent_cap``, rearranged to
    ``Q - g'*mu <= g(mu_p) - g'*mu_p`` so the row is a pair of columns and a
    scalar right-hand side.
    """
    from fleet_management.degradation_model.rainflow_v2 import _cap_coeffs
    T = ctx.T
    by_q: Dict[str, list] = {}
    for i, l in cells:
        bound = str(ctx.bound_of[i, l])
        c2, c1, Qvar = _cap_coeffs(ctx, i, l, bound)
        tau = float(ctx.tau[i, l])
        mu_p = float(np.clip(mu_p_frac, 0.0, 1.0)) * tau
        d_p = tau - mu_p
        g_p = c2 * d_p * d_p + c1 * d_p
        gpp = -2.0 * c2 * d_p - c1
        qname = _q_block_name(ctx, Qvar)
        by_q.setdefault(qname, []).append((i, l, g_p, gpp, mu_p))
    for qname, items in by_q.items():
        arr = np.asarray([(i, l) for i, l, *_ in items], dtype=int)
        g_p = np.asarray([it[2] for it in items]); gpp = np.asarray([it[3] for it in items])
        mu_p = np.asarray([it[4] for it in items])
        ci = np.repeat(arr[:, 0], T); cl = np.repeat(arr[:, 1], T)
        ck = np.tile(np.arange(T), len(items))
        rows.add(GRB.LESS_EQUAL, np.repeat(g_p - gpp * mu_p, T),
                 (cols.idx(qname, ci, cl, ck), 1.0),
                 (cols.idx("mu", ci, cl, ck), np.repeat(-gpp, T)))


def _q_block_name(ctx: _RFModel, Qvar) -> str:
    for name, block in (("v", ctx.v_var), ("R", ctx.R_var), ("K", ctx.K_var)):
        if Qvar is block:
            return name
    raise ValueError("reliability cap references an unregistered state block.")


def _reliability_rows(ctx: _RFModel, cols: ColumnMap, rows: RowSet) -> None:
    """Per-step ``P(D > tau) <= eps``, dispatched by (bound, impl).

    The linear families -- markov, the ``mu <= tau`` gaps, the single tangent,
    and the pwl segment-selection equality -- go through ``RowSet``.  The
    quadratic ``exact`` encodings and the pwl per-segment indicators have no
    matrix form and stay scalar; they are written here rather than delegated so
    that the row order matches the indicator build family by family.
    """
    T = ctx.T
    for (bound, impl), cells in _cell_groups(ctx).items():
        if bound == "markov":
            ub = np.asarray([float(ctx.eps[i, l]) * float(ctx.tau[i, l])
                             for i, l in cells])
            _rel_gap_rows(ctx, cols, rows, cells, ub)
            continue

        if bound == "chernoff":
            ci = np.repeat(cells[:, 0], T); cl = np.repeat(cells[:, 1], T)
            ck = np.tile(np.arange(T), len(cells))
            s = np.asarray([float(ctx.s_chernoff[i, l]) for i, l in cells])
            tau = np.asarray([float(ctx.tau[i, l]) for i, l in cells])
            ln_eps = np.asarray([float(ctx.ln_eps[i, l]) for i, l in cells])
            rows.add(GRB.LESS_EQUAL, np.repeat(ln_eps + s * tau, T),
                     (cols.idx("K", ci, cl, ck), 1.0))
            continue

        # the quadratic family: every impl starts from the mu <= tau gap
        tau = np.asarray([float(ctx.tau[i, l]) for i, l in cells])
        _rel_gap_rows(ctx, cols, rows, cells, tau)

        if impl == "tangent":
            _rel_tangent_rows(ctx, cols, rows, cells, ctx.tangent_ref)
        elif impl == "pwl":
            if encoding_of(ctx.formulation) == "bigm":
                _rel_pwl_bigm(ctx, cols, rows, cells)
            else:
                _rel_pwl(ctx, cols, rows, cells)
        else:                                  # 'exact': nonconvex quadratic
            _rel_exact_quadratic(ctx, cols, cells, bound)


def _rel_exact_quadratic(ctx: _RFModel, cols: ColumnMap, cells, bound: str) -> None:
    """The exact quadratic caps.  No matrix form; one ``addQConstr`` per row.

    Kept byte-for-byte equivalent to ``rainflow_v2``'s ``_rel_*_exact`` bodies
    minus the ``mu <= tau`` row, which ``_reliability_rows`` has already added
    through the sparse path.
    """
    md, T = ctx.model, ctx.T
    mu_b = cols.block("mu")
    for i, l in cells:
        tau = float(ctx.tau[i, l])
        if bound == "cantelli":
            eps = float(ctx.eps[i, l]); v_b = cols.block("v")
            for k in range(T):
                mu = mu_b[i, l, k]
                md.addQConstr((1.0 - eps) * v_b[i, l, k]
                              <= eps * (tau - mu) * (tau - mu))
        elif bound == "hoeffding":
            Le = float(ctx.Le[i, l]); R_b = cols.block("R")
            for k in range(T):
                mu = mu_b[i, l, k]
                md.addQConstr((tau - mu) * (tau - mu) >= 0.5 * Le * R_b[i, l, k])
        elif bound == "bernstein":
            Le = float(ctx.Le[i, l]); b = float(ctx.support_max_of[i, l])
            v_b = cols.block("v")
            for k in range(T):
                t = tau - mu_b[i, l, k]
                md.addQConstr(0.5 * t * t - (Le * b / 3.0) * t
                              - Le * v_b[i, l, k] >= 0)
        else:
            raise ValueError(f"no exact quadratic cap for bound {bound!r}.")


def _rel_pwl(ctx: _RFModel, cols: ColumnMap, rows: RowSet, cells) -> None:
    """Piecewise-tangent cap: segment binaries + selection, indicator encoding.

    ``sum_s z_s = 1`` is linear and goes through ``RowSet``; the membership and
    tangent rows are indicators, exactly as in ``rainflow_v2``.  The segment
    binaries keep that module's names so the two models can be matched variable
    by variable.
    """
    from fleet_management.degradation_model.rainflow_v2 import _cap_coeffs
    md, T = ctx.model, ctx.T
    K = max(1, int(ctx.pwl_points))
    if K == 1:
        _rel_tangent_rows(ctx, cols, rows, cells, 0.5)
        return
    mu_b = cols.block("mu")
    for i, l in cells:
        bound = str(ctx.bound_of[i, l])
        c2, c1, Qvar = _cap_coeffs(ctx, i, l, bound)
        Q_b = cols.block(_q_block_name(ctx, Qvar))
        tau = float(ctx.tau[i, l])
        edges = np.linspace(0.0, tau, K + 1)
        mid = 0.5 * (edges[:-1] + edges[1:])
        d_p = tau - mid
        g_p = c2 * d_p * d_p + c1 * d_p
        gpp = -2.0 * c2 * d_p - c1
        for k in range(T):
            mu, Q = mu_b[i, l, k], Q_b[i, l, k]
            zs = md.addVars(K, vtype=GRB.BINARY, name=f"relseg_{i}_{l}_{k}")
            zl = list(zs.values())
            md.addConstr(gp.LinExpr([1.0] * K, zl) == 1.0,
                         name=f"relseg1_{i}_{l}_{k}")
            for s in range(K):
                md.addGenConstrIndicator(zl[s], True, gp.LinExpr(1.0, mu),
                                         GRB.GREATER_EQUAL, float(edges[s]))
                md.addGenConstrIndicator(zl[s], True, gp.LinExpr(1.0, mu),
                                         GRB.LESS_EQUAL, float(edges[s + 1]))
                md.addGenConstrIndicator(
                    zl[s], True, gp.LinExpr([1.0, -float(gpp[s])], [Q, mu]),
                    GRB.LESS_EQUAL, float(g_p[s] - gpp[s] * mid[s]))


# ---------------------------------------------------------------------------
# State recursions:  the part Gurobi has no matrix form for
# ---------------------------------------------------------------------------
def _state_indicators(ctx: _RFModel, cols: ColumnMap, rows: RowSet,
                      prof: dict) -> None:
    """The ``6*S*C`` indicator rows of the three-branch recursions.

    Identical rows to ``rainflow_v2._add_rainflow_state_ind``.  What is
    different is only how they are constructed: coefficients come from the
    precomputed ``(F, L, M, T)`` tensors in ``prof``, variables from flat
    object arrays flattened to Python lists once per cell, and each row is one
    ``LinExpr(coeffs, vars)`` plus one 5-argument ``addGenConstrIndicator``.
    No ``quicksum``, no ``TempConstr`` from operator overloading, no f-string
    names, no ``tupledict`` hashing.

    This family stays ``Theta(F*L*T)`` API calls -- Gurobi has no matrix form
    for general indicator constraints -- so the gain here is a constant factor,
    not a change of order.  ``test_sparse_version.py`` reports it separately
    from the linear part for exactly that reason.
    """
    T, M = ctx.T, ctx.M
    allow_rep = ctx.allow_replacement
    add_ind = ctx.model.addGenConstrIndicator      # bound once, called ~6*S*C times
    LinExpr = gp.LinExpr
    EQ = GRB.EQUAL

    x_b = cols.block("x")
    nb_b, m_b = cols.block("nb"), cols.block("m")
    r_b = cols.block("r") if allow_rep else None
    mu_b, z_b = cols.block("mu"), cols.block("z")
    v_b = cols.block("v") if cols.has("v") else None
    gmu_b = cols.block("gmu") if cols.has("gmu") else None
    gv_b = cols.block("gv") if cols.has("gv") else None
    R_b = cols.block("R") if cols.has("R") else None
    gR_b = cols.block("gR") if cols.has("gR") else None
    K_b = cols.block("K") if cols.has("K") else None

    def _state(i, l, state_b, s0, inc, a, latch_b, new, nb_l, m_l, r_l):
        """One state's three branches over the whole horizon of one cell.

        Everything independent of ``k`` -- the mission variables of vehicle
        ``i``, the negated increment coefficients, the state column -- is pulled
        out of the loop with one ``tolist()`` each.  What is left inside is a
        list concatenation, a ``LinExpr`` and the API call.
        """
        xrow = x_b[i, 1:M + 1, :].T.tolist()          # T lists of M Var
        negc = (-inc[i, l, :, :]).T.tolist()          # T lists of M float
        cur_l = state_b[i, l, :].tolist()
        g_l = latch_b[i, l, :].tolist() if latch_b is not None else None
        a_lat, s0, new = 1.0 - a, float(s0), float(new)

        for k in range(T):
            cur = cur_l[k]
            # --- no intervention:  s_k = s_{k-1} + sum_j c_jk x_jk
            if k:
                add_ind(nb_l[k], True,
                        LinExpr([1.0] + negc[k] + [-1.0],
                                [cur] + xrow[k] + [cur_l[k - 1]]), EQ, 0.0)
            else:
                add_ind(nb_l[k], True,
                        LinExpr([1.0] + negc[k], [cur] + xrow[k]), EQ, s0)

            # --- repair:  s_k = a*s_{k-1} + (1-a)*g_{k-1}
            if k:
                coeffs, vars_ = [1.0, -a], [cur, cur_l[k - 1]]
                if g_l is not None:
                    coeffs.append(-a_lat); vars_.append(g_l[k - 1])
                add_ind(m_l[k], True, LinExpr(coeffs, vars_), EQ, 0.0)
            else:                                     # s_{-1} = s0, g_{-1} = 0
                add_ind(m_l[k], True, LinExpr(1.0, cur), EQ, a * s0)

            # --- replacement:  s_k = new
            if allow_rep:
                add_ind(r_l[k], True, LinExpr(1.0, cur), EQ, new)

    def _latch(i, l, latch_b, state_b, nb_l, m_l, r_l):
        """``g_k = g_{k-1}`` on a no-intervention step, ``g_k = s_k`` otherwise."""
        g_l = latch_b[i, l, :].tolist()
        s_l = state_b[i, l, :].tolist()
        for k in range(T):
            g_k = g_l[k]
            if k:
                add_ind(nb_l[k], True,
                        LinExpr([1.0, -1.0], [g_k, g_l[k - 1]]), EQ, 0.0)
            else:                                     # g_{-1} = 0
                add_ind(nb_l[k], True, LinExpr(1.0, g_k), EQ, 0.0)
            add_ind(m_l[k], True,
                    LinExpr([1.0, -1.0], [g_k, s_l[k]]), EQ, 0.0)
            if allow_rep:
                add_ind(r_l[k], True,
                        LinExpr([1.0, -1.0], [g_k, s_l[k]]), EQ, 0.0)

    def _z(i, l, nb_l, m_l, r_l):
        """Removed expected damage (eq. 6), indicator form."""
        z_l = z_b[i, l, :].tolist()
        mu_l = mu_b[i, l, :].tolist()
        mu0 = float(ctx.mu_0[i, l])
        for k in range(T):
            z_k = z_l[k]
            add_ind(nb_l[k], True, LinExpr(1.0, z_k), EQ, 0.0)
            if k:
                coeffs = [1.0, 1.0, -1.0]
                vars_ = [z_k, mu_l[k], mu_l[k - 1]]
                rhs = 0.0
            else:
                coeffs, vars_, rhs = [1.0, 1.0], [z_k, mu_l[0]], mu0
            add_ind(m_l[k], True, LinExpr(coeffs, vars_), EQ, rhs)
            if allow_rep:
                add_ind(r_l[k], True, LinExpr(coeffs, vars_), EQ, rhs)

    for i, l in ctx.all_cells():
        bound = str(ctx.bound_of[i, l])
        track_v = bool(ctx.track_v_of[i, l])
        use_latch = bool(ctx.latch_of[i, l])
        k1 = 1.0 - float(ctx.rho[i, l])
        k2 = k1 * k1
        nb_l = nb_b[i, l, :].tolist()
        m_l = m_b[i, l, :].tolist()
        r_l = r_b[i, l, :].tolist() if allow_rep else None

        _state(i, l, mu_b, ctx.mu_0[i, l], prof["mu"], k1,
               gmu_b if use_latch else None, ctx.mu_new[i, l], nb_l, m_l, r_l)
        if track_v:
            _state(i, l, v_b, ctx.v_0[i, l], prof["v"], k2,
                   gv_b if use_latch else None, ctx.v_new[i, l], nb_l, m_l, r_l)
        if use_latch:
            _latch(i, l, gmu_b, mu_b, nb_l, m_l, r_l)
            if track_v:
                _latch(i, l, gv_b, v_b, nb_l, m_l, r_l)
        _z(i, l, nb_l, m_l, r_l)
        if bound == "hoeffding":
            _state(i, l, R_b, 0.0, prof["R"], k2,
                   gR_b if use_latch else None, 0.0, nb_l, m_l, r_l)
            if use_latch:
                _latch(i, l, gR_b, R_b, nb_l, m_l, r_l)
        if bound == "chernoff":
            _state(i, l, K_b, 0.0, prof["K"], k1, None, 0.0, nb_l, m_l, r_l)


# ===========================================================================
# ####################  BIG-M ENCODING, SPARSE ASSEMBLY  ####################
# ===========================================================================
# This is the combination the indicator path cannot reach.  In the big-M
# encoding every row of the state recursion is a plain linear inequality, so
# there is nothing left that Gurobi has no matrix form for: the WHOLE program
# goes through RowSet, and the build stops being Theta(F*L*T) API calls
# altogether rather than merely losing a constant factor.
#
# The rows are the ones rainflow_v2._add_rainflow_state_bigm writes, term for
# term.  Two things make the vectorisation less mechanical than the shared
# skeleton's, and both are handled by splitting the row family rather than by
# branching inside it:
#
#   * step k = 0 has no predecessor column -- the previous state is the constant
#     s0 and the previous latch is 0 -- so it carries one fewer term and a
#     different right-hand side.  Every family is therefore emitted twice, once
#     over the k = 0 rows and once over the rest.
#   * a cell's STRUCTURE (does it latch under ARD1, is its replacement state
#     nonzero) decides which terms exist at all.  Cells are grouped by that
#     structure first, so that within a group every row has the same support and
#     only the coefficients differ.
# ---------------------------------------------------------------------------


def _ub_tensor(ctx: _RFModel, cols: ColumnMap, name: str) -> np.ndarray:
    """The big-M constants of one state block, as an (F, L, T) array.

    Vectorised twin of ``rainflow_v2._ub``: the tight big-M for a state is its
    own upper bound, which ``_tighten_bounds`` has already set from the problem
    data; ``ctx.bigM`` is the fallback for a variable left free (a cell whose
    bound does not use that accumulator).
    """
    F, L, T = ctx.F, ctx.L, ctx.T
    off = cols._offset[name]
    block = cols.vars[off:off + F * L * T]
    ub = np.asarray(ctx.model.getAttr(GRB.Attr.UB, block), dtype=float)
    ub = ub.reshape(F, L, T)
    return np.where(np.isfinite(ub) & (ub < GRB.INFINITY), ub, float(ctx.bigM))


class _CellSteps:
    """The (cell, step) index arrays of one group, split at ``k = 0``.

    Every big-M family needs the same four things -- the flat column of the
    current step, the column of the previous step where one exists, the
    per-row big-M, and the mission columns -- so they are computed once here
    and shared.
    """

    def __init__(self, ctx: _RFModel, cols: ColumnMap, cells: np.ndarray):
        T, M = ctx.T, ctx.M
        n = len(cells)
        self.T, self.M, self.n = T, M, n
        self.i = np.repeat(cells[:, 0], T)
        self.l = np.repeat(cells[:, 1], T)
        self.k = np.tile(np.arange(T), n)
        self.first = self.k == 0                  # rows with no predecessor
        self.later = ~self.first
        self.cols = cols
        self.x = cols.idx("x", self.i[:, None], np.arange(1, M + 1)[None, :],
                          self.k[:, None])
        self.m = cols.idx("m", self.i, self.l, self.k)
        self.r = (cols.idx("r", self.i, self.l, self.k)
                  if cols.has("r") else None)

    def col(self, name: str, mask=None, lag: int = 0) -> np.ndarray:
        """Flat columns of block ``name`` at these steps, optionally lagged."""
        if mask is None:
            return self.cols.idx(name, self.i, self.l, self.k - lag)
        return self.cols.idx(name, self.i[mask], self.l[mask], self.k[mask] - lag)

    def per_cell(self, values: np.ndarray, mask=None) -> np.ndarray:
        """An (n,) per-cell array broadcast to one value per (cell, step) row."""
        out = np.repeat(np.asarray(values, dtype=float), self.T)
        return out if mask is None else out[mask]

    def inc(self, tensor: np.ndarray, mask=None) -> np.ndarray:
        """The (rows, M) mission-increment coefficients from an (F,L,M,T) tensor."""
        if mask is None:
            return tensor[self.i, self.l, :, self.k]
        return tensor[self.i[mask], self.l[mask], :, self.k[mask]]

    def bigM(self, tensor: np.ndarray, mask=None) -> np.ndarray:
        if mask is None:
            return tensor[self.i, self.l, self.k]
        return tensor[self.i[mask], self.l[mask], self.k[mask]]


def _act_terms(cs: _CellSteps, coef, mask=None) -> list:
    """The terms of ``coef * (m + r)`` -- the 'an intervention happens' expression."""
    m = cs.m if mask is None else cs.m[mask]
    terms = [(m, coef)]
    if cs.r is not None:
        terms.append((cs.r if mask is None else cs.r[mask], coef))
    return terms


def _state_bigm_group(ctx: _RFModel, cols: ColumnMap, rows: RowSet, *,
                      state: str, cells: np.ndarray, s0, inc: np.ndarray,
                      a, latch: str, new, Mtab: np.ndarray) -> None:
    """The four-to-six big-M rows of ONE state, for a structurally uniform group.

    Reproduces ``rainflow_v2._state_bigm``:

        (U1) s_k <= s_{k-1} + inc_k + new*r_k        no M: valid on all branches
        (L1) s_k >= s_{k-1} + inc_k - M*(m+r)        carry branch
        (L2) s_k >= a*s_{k-1} + (1-a)*g_{k-1} - M*(1-m)     repair branch
        (U2) s_k <= a*s_{k-1} + (1-a)*g_{k-1} + M*(1-m)
        (U3) s_k <= new + M*(1-r)                    replacement branch
        (L3) s_k >= new - M*(1-r)                    only when new > 0
    """
    if len(cells) == 0:
        return
    cs = _CellSteps(ctx, cols, cells)
    allow_rep = cs.r is not None
    has_latch = latch is not None
    LE, GE = GRB.LESS_EQUAL, GRB.GREATER_EQUAL

    for mask in (cs.first, cs.later):
        if not mask.any():
            continue
        cur = cs.col(state, mask)
        prev = None if mask is cs.first else cs.col(state, mask, lag=1)
        s0r = cs.per_cell(s0, mask)
        ar = cs.per_cell(a, mask)
        newr = cs.per_cell(new, mask)
        Mv = cs.bigM(Mtab, mask)
        xc, cc = cs.x[mask], -cs.inc(inc, mask)
        m_col = cs.m[mask]
        r_col = cs.r[mask] if allow_rep else None
        g_prev = (cs.col(latch, mask, lag=1)
                  if (has_latch and mask is cs.later) else None)

        # (U1) no branch overshoots the carry value -- no big-M anywhere
        t = [(cur, 1.0), (xc, cc)]
        if prev is not None:
            t.append((prev, -1.0))
        if allow_rep and np.any(newr > 0.0):
            t.append((r_col, -newr))
        rows.add(LE, 0.0 if prev is not None else s0r, *t)

        # (L1) carry branch, active when m + r = 0
        t = [(cur, 1.0), (xc, cc)] + _act_terms(cs, Mv, mask)
        if prev is not None:
            t.append((prev, -1.0))
        rows.add(GE, 0.0 if prev is not None else s0r, *t)

        # (L2, U2) repair branch, active when m = 1
        base_terms, rhs_const = [(cur, 1.0)], np.zeros(int(mask.sum()))
        if prev is not None:
            base_terms.append((prev, -ar))
        else:
            rhs_const = rhs_const + ar * s0r        # s_{-1} = s0, g_{-1} = 0
        if g_prev is not None:
            base_terms.append((g_prev, -(1.0 - ar)))
        rows.add(GE, rhs_const - Mv, *base_terms, (m_col, -Mv))
        rows.add(LE, rhs_const + Mv, *base_terms, (m_col, Mv))

        # (L3, U3) replacement branch, active when r = 1
        if allow_rep:
            rows.add(LE, newr + Mv, (cur, 1.0), (r_col, Mv))
            if np.any(newr > 0.0):        # otherwise implied by s_k >= 0
                rows.add(GE, newr - Mv, (cur, 1.0), (r_col, -Mv))


def _latch_bigm_group(ctx: _RFModel, cols: ColumnMap, rows: RowSet, *,
                      latch: str, state: str, cells: np.ndarray,
                      Mtab: np.ndarray) -> None:
    """The four big-M rows of one ARD1 latch, ``g_k = g_{k-1} if nb else s_k``.

    Reproduces ``rainflow_v2._latch_bigm``; two of the four carry no big-M
    (``g_k <= s_k`` and the hold-from-below row), which is exactly why the
    big-M encoding relaxes more tightly than the indicator one.
    """
    if len(cells) == 0:
        return
    cs = _CellSteps(ctx, cols, cells)
    allow_rep = cs.r is not None
    LE, GE = GRB.LESS_EQUAL, GRB.GREATER_EQUAL

    for mask in (cs.first, cs.later):
        if not mask.any():
            continue
        g_k = cs.col(latch, mask)
        s_k = cs.col(state, mask)
        g_prev = None if mask is cs.first else cs.col(latch, mask, lag=1)
        Mg = cs.bigM(Mtab, mask)
        r_col = cs.r[mask] if allow_rep else None

        rows.add(LE, 0.0, (g_k, 1.0), (s_k, -1.0))                    # g <= s
        rows.add(GE, -Mg, (g_k, 1.0), (s_k, -1.0),                    # set on act
                 *_act_terms(cs, -Mg, mask))
        t = [(g_k, 1.0)] + _act_terms(cs, -Mg, mask)                  # hold above
        if g_prev is not None:
            t.append((g_prev, -1.0))
        rows.add(LE, 0.0, *t)
        t = [(g_k, 1.0)]                                              # hold below
        if g_prev is not None:
            t.append((g_prev, -1.0))
        if allow_rep:
            rows.add(GE, 0.0, *t, (r_col, Mg))
        else:
            rows.add(GE, 0.0, *t)


def _z_bigm_rows(ctx: _RFModel, cols: ColumnMap, rows: RowSet,
                 cells: np.ndarray, Mtab: np.ndarray) -> None:
    """Removed expected damage (eq. 6) without indicators.

    ``z_k >= mu_{k-1} - mu_k`` alone is enough whenever ``C_R > 0`` drives z
    down; the two upper rows are added only when ``ctx.z_exact`` says nothing
    does.  Same condition, same rows, as ``rainflow_v2._z_bigm``.
    """
    cs = _CellSteps(ctx, cols, cells)
    LE, GE = GRB.LESS_EQUAL, GRB.GREATER_EQUAL
    mu0 = np.asarray([float(ctx.mu_0[i, l]) for i, l in cells])

    for mask in (cs.first, cs.later):
        if not mask.any():
            continue
        z_k = cs.col("z", mask)
        mu_k = cs.col("mu", mask)
        mu_prev = None if mask is cs.first else cs.col("mu", mask, lag=1)
        rhs = 0.0 if mu_prev is not None else cs.per_cell(mu0, mask)
        t = [(z_k, 1.0), (mu_k, 1.0)]
        if mu_prev is not None:
            t.append((mu_prev, -1.0))
        rows.add(GE, rhs, *t)
        if ctx.z_exact:
            Mz = cs.bigM(Mtab, mask)
            rows.add(LE, (rhs if mu_prev is None else 0.0) + Mz,
                     *t, *_act_terms(cs, Mz, mask))
            rows.add(LE, 0.0, (z_k, 1.0), *_act_terms(cs, -Mz, mask))


def _state_bigm(ctx: _RFModel, cols: ColumnMap, rows: RowSet, prof: dict) -> None:
    """Every state recursion of every cell, big-M encoded, as linear rows.

    Top-level twin of ``rainflow_v2._add_rainflow_state_bigm``.  Cells are first
    partitioned into structurally uniform groups -- same state block, same latch
    presence, same 'is the replacement state nonzero' answer -- so that each
    group emits row families of uniform support and the per-cell numbers
    (``rho``, ``mu_0``, the replacement state, the big-Ms) travel as coefficient
    arrays.
    """
    latch_of = np.asarray(ctx.latch_of, dtype=bool)
    track_v = np.asarray(ctx.track_v_of, dtype=bool)
    rho = np.asarray(ctx.rho, dtype=float)
    Mtab = {name: _ub_tensor(ctx, cols, name)
            for name in ("mu", "z", "v", "gmu", "gv", "R", "gR", "K")
            if cols.has(name)}

    def _groups(cells, new_arr):
        """Partition by (latches, replacement state nonzero)."""
        out = {}
        for i, l in cells:
            out.setdefault((bool(latch_of[i, l]), float(new_arr[i, l]) > 0.0),
                           []).append((i, l))
        return {k: np.asarray(v, dtype=int) for k, v in out.items()}

    def _emit(cells, *, state, s0_arr, inc, exponent, latch_block, new_arr):
        for (has_latch, _), grp in _groups(cells, new_arr).items():
            a = (1.0 - rho[grp[:, 0], grp[:, 1]]) ** exponent
            _state_bigm_group(
                ctx, cols, rows, state=state, cells=grp,
                s0=s0_arr[grp[:, 0], grp[:, 1]], inc=inc, a=a,
                latch=(latch_block if has_latch else None),
                new=new_arr[grp[:, 0], grp[:, 1]], Mtab=Mtab[state])
            if has_latch and latch_block is not None:
                _latch_bigm_group(ctx, cols, rows, latch=latch_block,
                                  state=state, cells=grp,
                                  Mtab=Mtab[latch_block])

    all_cells = np.asarray(ctx.all_cells(), dtype=int)
    bound_of = np.asarray(ctx.bound_of).astype(str)

    # ----- mean -----
    _emit(all_cells, state="mu", s0_arr=np.asarray(ctx.mu_0, dtype=float),
          inc=prof["mu"], exponent=1, latch_block="gmu",
          new_arr=np.asarray(ctx.mu_new, dtype=float))

    # ----- variance (only the bounds that track it) -----
    v_cells = np.asarray([(i, l) for i, l in ctx.all_cells() if track_v[i, l]],
                         dtype=int).reshape(-1, 2)
    if len(v_cells):
        _emit(v_cells, state="v", s0_arr=np.asarray(ctx.v_0, dtype=float),
              inc=prof["v"], exponent=2, latch_block="gv",
              new_arr=np.asarray(ctx.v_new, dtype=float))

    # ----- removed expected damage z (eq. 6) -----
    _z_bigm_rows(ctx, cols, rows, all_cells, Mtab["mu"])

    # ----- extra descriptors -----
    zero = np.zeros((ctx.F, ctx.L))
    for bound, state, exponent, latch_block in (("hoeffding", "R", 2, "gR"),
                                                ("chernoff", "K", 1, None)):
        cells = np.asarray([(i, l) for i, l in ctx.all_cells()
                            if bound_of[i, l] == bound], dtype=int).reshape(-1, 2)
        if len(cells):
            _emit(cells, state=state, s0_arr=zero, inc=prof[state],
                  exponent=exponent, latch_block=latch_block, new_arr=zero)


def _rel_pwl_bigm(ctx: _RFModel, cols: ColumnMap, rows: RowSet, cells) -> None:
    """Piecewise-tangent cap, big-M segment selection -- every row linear.

    Reproduces ``rainflow_v2._rel_quadratic_pwl``'s big-M branch: aggregated
    membership (``mu >= sum lo_s z_s``, ``mu <= sum hi_s z_s``, both in the LP,
    unlike ``2K`` indicator rows) plus one tangent cap per segment with the
    smallest M that leaves it slack.  The segment binaries keep the names the
    per-cell build gives them, so the two models still match column by column.
    """
    from fleet_management.degradation_model.rainflow_v2 import _cap_coeffs
    md, T = ctx.model, ctx.T
    K = max(1, int(ctx.pwl_points))
    if K == 1:
        _rel_tangent_rows(ctx, cols, rows, cells, 0.5)
        return

    seg_block = f"relseg_{id(cells)}"
    seg_vars, mu_cols, Q_cols, lo, hi, cap_c, cap_rhs, Ms = [], [], [], [], [], [], [], []
    for i, l in cells:
        bound = str(ctx.bound_of[i, l])
        c2, c1, Qvar = _cap_coeffs(ctx, i, l, bound)
        qname = _q_block_name(ctx, Qvar)
        Qub = _ub_tensor(ctx, cols, qname)[i, l, :]
        tau = float(ctx.tau[i, l])
        edges = np.linspace(0.0, tau, K + 1)
        mid = 0.5 * (edges[:-1] + edges[1:])
        d_p = tau - mid
        g_p = c2 * d_p * d_p + c1 * d_p
        gpp = -2.0 * c2 * d_p - c1
        # the tangent is affine in mu, so its minimum over [0, tau] is attained
        # at an endpoint; the smallest valid M is Qub minus that minimum
        lo_val = np.minimum(g_p + gpp * (0.0 - mid), g_p + gpp * (tau - mid))
        for k in range(T):
            zs = md.addVars(K, vtype=GRB.BINARY, name=f"relseg_{i}_{l}_{k}")
            seg_vars.append(list(zs.values()))
            mu_cols.append(int(cols.idx("mu", i, l, k)))
            Q_cols.append(int(cols.idx(qname, i, l, k)))
            lo.append(edges[:-1]); hi.append(edges[1:])
            cap_c.append(gpp); cap_rhs.append(g_p - gpp * mid)
            Ms.append(np.maximum(0.0, float(Qub[k]) - lo_val))
    if not seg_vars:
        return

    # the segment binaries are new columns: extend the map, and with it the
    # width every accumulated row is emitted against
    base = cols.n
    cols.add(seg_block, [v for row in seg_vars for v in row], (len(seg_vars), K))
    rows.ncols = cols.n
    zc = base + np.arange(len(seg_vars) * K).reshape(len(seg_vars), K)
    mu_cols = np.asarray(mu_cols); Q_cols = np.asarray(Q_cols)
    lo = np.asarray(lo); hi = np.asarray(hi)
    cap_c = np.asarray(cap_c); cap_rhs = np.asarray(cap_rhs); Ms = np.asarray(Ms)

    rows.add(GRB.EQUAL, 1.0, (zc, 1.0))                       # sum_s z_s = 1
    rows.add(GRB.GREATER_EQUAL, 0.0, (mu_cols, 1.0), (zc, -lo))
    rows.add(GRB.LESS_EQUAL, 0.0, (mu_cols, 1.0), (zc, -hi))
    for s in range(K):                                        # K families, not K*C rows
        rows.add(GRB.LESS_EQUAL, cap_rhs[:, s] + Ms[:, s],
                 (Q_cols, 1.0), (mu_cols, -cap_c[:, s]), (zc[:, s], Ms[:, s]))


# ===========================================================================
# #####  SPARSE STRENGTHENING OF THE INDICATOR ENCODING, VECTORISED  ########
# ===========================================================================
# The vectorised twin of ``rainflow_v2.add_sparse_cuts``.  Same rows, emitted
# through RowSet instead of one addConstr each.
#
# This pairing is the point of the whole exercise, not an implementation
# detail.  The cuts exist because the sparsity result says a row of constant
# width is cheap; they are emitted through the matrix API because a family of
# O(F*L*T) constant-width rows is exactly what COO assembly is good at.  Adding
# them the slow way would hand back part of what they buy: at F=6, M=3, L=2,
# H=10 the family is ~1000 extra rows, which is ~1000 extra addConstr calls per
# solve under a loop assembly and three addMConstr calls here.
#
# If you add a row in rainflow_v2.add_sparse_cuts, add it here as well.
# ``test_sparse_version.py --tests equivalence`` compares the two builds as
# canonicalised multisets of rows and will fail loudly if they drift.
# ---------------------------------------------------------------------------
def _sparse_cut_rows(ctx: _RFModel, cols: ColumnMap, rows: RowSet,
                     prof: dict) -> None:
    """(C1)-(C5) for every cell at once.  See rainflow_v2.add_sparse_cuts."""
    from fleet_management.degradation_model.rainflow_v2 import (
        cut_states, sparse_cut_level)

    level = sparse_cut_level(getattr(ctx, "sparse_cuts", False))
    if level == "off" or encoding_of(ctx.formulation) != "indicator":
        return

    allow_rep = ctx.allow_replacement
    all_cells = np.asarray(ctx.all_cells(), dtype=int)
    cs = _CellSteps(ctx, cols, all_cells)
    LE, EQ = GRB.LESS_EQUAL, GRB.EQUAL

    # ---- (C1) telescoping mean balance -------------------------------------
    # mu_k - mu_{k-1} - sum_j c_jk x_jk + z_k = 0, with mu_{-1} folded into the
    # right-hand side at k = 0.
    for mask in (cs.first, cs.later):
        if not mask.any():
            continue
        terms = [(cs.col("mu", mask), 1.0),
                 (cs.x[mask], -cs.inc(prof["mu"], mask)),
                 (cs.col("z", mask), 1.0)]
        if mask is cs.later:
            terms.append((cs.col("mu", mask, lag=1), -1.0))
            rhs = 0.0
        else:
            mu0 = np.asarray([float(ctx.mu_0[i, l]) for i, l in all_cells])
            rhs = cs.per_cell(mu0, mask)
        rows.add(EQ, rhs, *terms)

    # ---- (C5) z gating (tight-M: the cell's own tau) -----------------------
    if level == "full":
        tau = np.asarray([float(ctx.tau[i, l]) for i, l in all_cells])
        tau_r = cs.per_cell(tau)
        rows.add(LE, 0.0, (cs.col("z"), 1.0), *_act_terms(cs, -tau_r))

    # ---- (C2)-(C4) on the descriptor states --------------------------------
    # Cells are grouped by which descriptors they carry and by whether the
    # latch rows apply, so each group emits families of uniform support.
    groups: Dict[tuple, list] = {}
    for i, l in ctx.all_cells():
        key = (tuple(n for n, *_ in cut_states(ctx, i, l)),
               bool(ctx.latch_of[i, l]))
        groups.setdefault(key, []).append((i, l))

    for (names, use_latch), cells in groups.items():
        cells = np.asarray(cells, dtype=int)
        g = _CellSteps(ctx, cols, cells)
        for name in names:
            new = np.asarray([float(_new_of(ctx, name, i, l)) for i, l in cells])
            for mask in (g.first, g.later):
                if not mask.any():
                    continue
                terms = [(g.col(name, mask), 1.0),
                         (g.x[mask], -g.inc(prof[name], mask))]
                if mask is g.later:
                    terms.append((g.col(name, mask, lag=1), -1.0))
                    rhs = 0.0
                else:
                    s0 = np.asarray([float(_s0_of(ctx, name, i, l))
                                     for i, l in cells])
                    rhs = g.per_cell(s0, mask)
                if allow_rep and np.any(new > 0.0):
                    terms.append((g.r[mask], -g.per_cell(new, mask)))
                rows.add(LE, rhs, *terms)
            if use_latch:
                _latch_cut_rows(g, cols, rows, _latch_of(name), name, allow_rep)
        if use_latch and cols.has("gmu"):
            _latch_cut_rows(g, cols, rows, "gmu", "mu", allow_rep)


def _new_of(ctx: _RFModel, name: str, i: int, l: int) -> float:
    return {"v": ctx.v_new[i, l], "R": 0.0, "K": 0.0}[name]


def _s0_of(ctx: _RFModel, name: str, i: int, l: int) -> float:
    return {"v": ctx.v_0[i, l], "R": 0.0, "K": 0.0}[name]


def _latch_of(name: str) -> str:
    return {"v": "gv", "R": "gR"}[name]


def _latch_cut_rows(cs: _CellSteps, cols: ColumnMap, rows: RowSet,
                    latch: str, state: str, allow_rep: bool) -> None:
    """(C3) ``g_k <= s_k`` and (C4) ``g_k >= g_{k-1}`` for one latch."""
    if not cols.has(latch):
        return
    rows.add(GRB.LESS_EQUAL, 0.0,
             (cs.col(latch), 1.0), (cs.col(state), -1.0))
    if not allow_rep and cs.later.any():
        rows.add(GRB.GREATER_EQUAL, 0.0,
                 (cs.col(latch, cs.later), 1.0),
                 (cs.col(latch, cs.later, lag=1), -1.0))
