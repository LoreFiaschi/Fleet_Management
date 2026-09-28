"""
test_sparse_version.py -- is the sparse assembly the same program, and is it faster?

Purpose
-------
The four ``formulation`` values are a 2x2 grid of two independent choices:

                        assembly='loop'        assembly='sparse'
    encoding='indicator'    'indicator'            'sparse'
    encoding='bigm'         'bigm'                 'bigm_sparse'

The ENCODING is a modelling choice, and `test.py` already tests it: 'indicator'
and 'bigm' have the same integer feasible set but different LP relaxations.
The ASSEMBLY is not a modelling choice at all, and it is what this file tests.
For a fixed encoding the two assemblies emit the identical program, row for row
and column for column, and differ only in how it is handed to Gurobi:
whole-fleet NumPy / scipy.sparse assembly and a few ``addMConstr`` calls,
instead of the per-cell dispatch loop with one ``addConstr`` per row.

Every comparison this file makes is therefore WITHIN an encoding, never across
one -- 'indicator' vs 'sparse', and 'bigm' vs 'bigm_sparse'.

Two claims therefore have to be tested separately, and only one of them is a
timing question:

  (S1) **Equivalence -- structural, not statistical.**  Because the two builds
       are supposed to produce the *identical* model, it is not enough to check
       that the objectives agree to within the MIP gap (the check `test.py`
       applies to indicator-vs-bigm, where the models genuinely differ).  This
       file compares the two models as objects: the column sets, their types,
       bounds and objective coefficients; the linear rows as a canonicalised
       multiset of (sense, rhs, sorted support); the general indicator
       constraints as a canonicalised multiset of (binary, value, sense, rhs,
       sorted support); and the quadratic rows likewise.  A single differing
       coefficient fails the test, which is the point: a sparse assembler that
       is 2x faster and 1e-12 wrong is worthless.

  (S2) **The build cost.**  The sparsity chapter shows the LP relaxation is
       cheap (``nnz/(m+n) ~ 2.5``) and concludes the burden is the
       branch-and-bound tree.  That is a statement about the solver.  The
       builder is a separate cost, it is ``Theta(FLH)`` Python-level API calls,
       and on the large end of the scaling ladder it is not negligible next to
       the solve it sets up.  This file measures it directly, with a phase
       breakdown, because the headline ratio hides the shape of the result.

What to expect from (S2) -- read this before quoting a number
------------------------------------------------------------
The two encodings give very different answers, and the difference is the
result, not noise:

  * **'indicator' -> 'sparse': about 2x, and it saturates there.**  The linear
    families -- the shared skeleton, gating, loop closure, the linear
    reliability rows, the objective, the variable bounds -- collapse from
    ``Theta(F*L*T)`` API calls to a handful and their cost essentially
    vanishes.  The ``6*S*C`` general indicator constraints of the state
    recursions cannot follow, because Gurobi has no matrix form for them.
    ``rainflow_sparse`` makes each one cheaper (precomputed coefficient
    tensors, ``LinExpr`` from lists, the 5-argument signature, no f-string
    names) but there is still one API call per row, and once the linear part is
    free those calls ARE the build.

  * **'bigm' -> 'bigm_sparse': about 5-7x, and it keeps paying.**  The big-M
    encoding has no indicator constraints at all: every row is a plain linear
    inequality, so the whole program goes through the matrix API and the
    per-row API calls disappear rather than getting cheaper.

So the encoding and the assembly are independent as options but not in effect.
'bigm' is what makes 'sparse' worth having, and 'bigm_sparse' is the
combination to reach for at the large end of the scaling ladder.  Running only
one encoding will give a misleading picture of either.

Tests  (--tests equivalence,scaling,solve)
------------------------------------------
  equivalence  (S1) for each encoding, build both assemblies over a factorial
               grid of bound x implementation x repair model x replacement, and
               compare the two models object by object.  Small instances, since the point is
               coverage of the code paths, not size.  With --solve it also
               solves both and requires the objectives to match EXACTLY, not
               within the gap, and reports node counts and LP relaxation
               values, which must also coincide.
  scaling      (S2) build (do NOT solve) both assemblies of each encoding along
               a geometric ladder in F, L, H and M, several repeats per point,
               and report build time, the ratio, and the sparse phase
               breakdown.  The 'state_rows' phase is the one to watch: it is
               most of the sparse build under 'indicator' and almost nothing
               under 'bigm'.  No
               solving means no licence-size limit and no solver variance in
               the measurement.
  solve        end-to-end sanity: solve the same instance both ways and compare
               status, objective, node count and runtime.  This is the weakest
               of the three and is here only to catch a wiring mistake that the
               structural test somehow misses.

Outputs
-------
One folder per run, under --out (default ./test_results):

    test_results/<yymmdd_HHMM>_sparse/
        equivalence.csv     one row per (encoding, bound, impl, repair, repl)
        scaling.csv         one row per (encoding, parameter, value, assembly)
        summary.txt         verdicts, the failure detail, the timing table
        sparse_build.png    build time and speed-up vs each parameter

Design notes -- READ BEFORE TRUSTING A NUMBER
---------------------------------------------
1. **The scaling test does not solve.**  It measures ``build_fleet`` alone.
   Mixing in a solve would swamp the measurement with MIP variance and would
   cap the ladder at whatever the licence allows.  Build times are therefore
   comparable across the whole ladder, and a size-limited licence is fine.

2. **Import cost is warmed out.**  scipy is imported once before the clock
   starts.  Without that, the first sparse build in a process carries ~0.3 s of
   ``import scipy.sparse`` and the first ladder point reads as a *slowdown*.
   The same applies to Gurobi's environment: the harness builds one throwaway
   model before measuring.

3. **Repeats, not seeds.**  Unlike `test.py` there is nothing random here: the
   same instance builds to the same model every time.  ``--repeats`` exists only
   to average out machine noise, and the reported number is the MEDIAN, not the
   mean, so one scheduling hiccup cannot move it.

4. **Equivalence is checked on the model, not on the answer.**  A run that
   solves both models and gets the same objective proves much less than it
   looks: the objectives would still agree if the sparse build silently dropped
   a redundant row.  The canonical-multiset comparison is what actually has
   teeth, and it is the one whose failure is reported as a VIOLATION.

5. **Row ORDER is deliberately not compared.**  ``rainflow_sparse`` groups rows
   by family and by sense, so its row order differs from the per-cell build's.
   That is a presentational difference: it changes nothing about the feasible
   set, and Gurobi's presolve reorders anyway.  Node counts can in principle
   differ from row order alone through tie-breaking inside the solver; if the
   `solve` test reports identical objectives but different node counts, that is
   the reason, not a modelling error.

6. **Row NAMES are not compared either.**  The sparse build does not generate
   per-row names (an f-string per row is a real fraction of the cost it exists
   to remove).  Anything that reads a constraint back by name -- an IIS report,
   a hand-written diagnostic -- must use ``formulation='indicator'``.

Usage
-----
    python test_sparse_version.py                          # all three tests
    python test_sparse_version.py --tests equivalence --solve
    python test_sparse_version.py --tests scaling --factors 1,2,4,8
    python test_sparse_version.py --encodings bigm     # the combination that pays
    python test_sparse_version.py --quick                  # small, fast, CI-sized

Author: Johann Tschan
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


# ===========================================================================
# Lazy imports (so --help and --dry-run work without a Gurobi licence)
# ===========================================================================
def _import_build():
    from fleet_management.config import load_config
    from fleet_management.degradation_model.base import (
        build_fleet, resolve_run_options)
    return load_config, build_fleet, resolve_run_options


# The four formulation strings are a 2x2 grid of (encoding, assembly); see
# base.FORMULATIONS.  Only the ASSEMBLY axis is supposed to be a no-op, so the
# comparisons this file makes are within an encoding, never across one.
PAIRS = {"indicator": ("indicator", "sparse"),
         "bigm": ("bigm", "bigm_sparse")}
ENCODINGS = tuple(PAIRS)
BOUNDS = ("markov", "cantelli", "hoeffding", "bernstein", "chernoff")
IMPLS = ("exact", "tangent", "pwl")
REPAIRS = ("ard1", "ardinf")
# chernoff has no closed ARD1 recursion (config.py rejects the pair)
SKIP = {("chernoff", "ard1")}
SCALE_PARAMS = ("F", "L", "H", "M")


# ===========================================================================
# Instances
# ===========================================================================
@dataclass
class Case:
    """One design point.  Deliberately much simpler than `test.py`'s Scenario:
    nothing here depends on the bound being *tight*, only on it being *built*,
    so the descriptors are fixed numbers rather than a calibrated distribution.
    """
    F: int = 6
    M: int = 2
    L: int = 2
    H: int = 4                      # T = 2H
    bound: str = "cantelli"
    impl: str = "tangent"
    repair: str = "ard1"
    encoding: str = "indicator"     # which PAIRS entry this case compares
    allow_replacement: bool = False
    pwl_points: int = 3
    tangent_ref: float = 0.5
    # The sparse strengthening is a THIRD axis: it changes the program, so the
    # two assemblies must agree with it on as well as off. Off by default so the
    # baseline grid stays the reference build.
    sparse_cuts: str = "off"

    @property
    def T(self) -> int:
        return 2 * int(self.H)

    def to_input(self) -> dict:
        data = {
            "model": "rainflow", "bound_method": self.bound,
            "repair_model": self.repair,
            "F": int(self.F), "M": int(self.M), "L": int(self.L), "H": int(self.H),
            "tau": 1.0, "epsilon": 0.02, "rho": 0.9, "mu_0": 0.0, "v_0": 0.0,
            "mu": 0.02, "v": 4.0e-4, "support": 0.1, "cgf": 0.01,
            "s_chernoff": 60.0,
            "C_M": 1.0, "C_R": 0.5, "C_D": 2.0, "C_rep": 25.0,
            "reliability_impl": self.impl, "pwl_points": int(self.pwl_points),
            "tangent_ref": float(self.tangent_ref),
        }
        if self.allow_replacement:
            data["allow_replacement"] = True
        return data

    def label(self) -> str:
        return (f"{self.encoding}: {self.bound}/{self.impl}/{self.repair}"
                f"/repl={int(self.allow_replacement)}/cuts={self.sparse_cuts}  "
                f"F={self.F} M={self.M} L={self.L} H={self.H}")


def build_one(case: Case, formulation: str, **opt_overrides):
    """Build one model; return (ctx, wall seconds).  Never solves."""
    load_config, build_fleet, resolve_run_options = _import_build()
    cfg = load_config(case.to_input())
    opts = resolve_run_options(
        cfg, formulation=formulation, verbose=0,
        reliability_impl=case.impl, pwl_points=case.pwl_points,
        tangent_ref=case.tangent_ref,
        allow_replacement=case.allow_replacement,
        sparse_cuts=case.sparse_cuts, **opt_overrides)
    t0 = time.perf_counter()
    ctx = build_fleet(cfg, opts, model_name=f"cmp_{formulation}")
    wall = time.perf_counter() - t0
    ctx.model.update()
    return ctx, wall


# ===========================================================================
# Canonicalisation:  a model as order-independent, name-independent data
# ===========================================================================
# Every comparison goes through a column map keyed by VARIABLE NAME, so the two
# builds may create their columns in any order; what must agree is the content
# of each row once its support is expressed in those names.  Coefficients are
# rounded to `NDIG` digits: both builds compute the same coefficients from the
# same floats by the same arithmetic, so exact equality would in fact hold, but
# rounding keeps the test from being hostage to a future refactor that
# reassociates a product.
NDIG = 9
GRB_LE = "<"                                  # avoids importing gurobipy here


def _column_key(model) -> dict:
    return {n: i for i, n in enumerate(model.getAttr("VarName"))}


def canon_columns(model) -> dict:
    """name -> (VType, LB, UB, Obj), the whole column description."""
    names = model.getAttr("VarName")
    return {n: (t, round(float(lb), NDIG), round(float(ub), NDIG),
                round(float(ob), NDIG))
            for n, t, lb, ub, ob in zip(names, model.getAttr("VType"),
                                        model.getAttr("LB"), model.getAttr("UB"),
                                        model.getAttr("Obj"))}


def canon_linear(model, key: dict) -> list:
    """Linear rows as a sorted list of (sense, rhs, sorted support).

    Read straight off the constraint matrix rather than row by row: ``getA()``
    is one call and returns scipy CSR, which is also the shape the sparse
    builder produced in the first place.
    """
    A = model.getA().tocsr()
    names = model.getAttr("VarName")
    sense, rhs = model.getAttr("Sense"), model.getAttr("RHS")
    out = []
    for i in range(A.shape[0]):
        lo, hi = A.indptr[i], A.indptr[i + 1]
        support = tuple(sorted(
            (key[names[c]], round(float(v), NDIG))
            for c, v in zip(A.indices[lo:hi], A.data[lo:hi]) if v != 0.0))
        out.append((sense[i], round(float(rhs[i]), NDIG), support))
    return sorted(out)


def canon_indicator(model, key: dict) -> list:
    """Indicator constraints as (binary, binval, sense, rhs, sorted support)."""
    import gurobipy as gp
    out = []
    for gc in model.getGenConstrs():
        if gc.GenConstrType != gp.GRB.GENCONSTR_INDICATOR:
            out.append(("non_indicator_genconstr", int(gc.GenConstrType)))
            continue
        binvar, binval, expr, sense, rhs = model.getGenConstrIndicator(gc)
        support = tuple(sorted(
            (key[expr.getVar(t).VarName], round(expr.getCoeff(t), NDIG))
            for t in range(expr.size()) if expr.getCoeff(t) != 0.0))
        # a constant on the left is the same row as its negative on the right
        out.append((key[binvar.VarName], int(binval), sense,
                    round(float(rhs) - expr.getConstant(), NDIG), support))
    return sorted(out)


def canon_quadratic(model, key: dict) -> list:
    """Quadratic rows as (sense, rhs, sorted quadratic terms, sorted linear)."""
    out = []
    for qc in model.getQConstrs():
        qe = model.getQCRow(qc)
        le = qe.getLinExpr()
        quad = tuple(sorted(
            tuple(sorted((key[qe.getVar1(t).VarName], key[qe.getVar2(t).VarName])))
            + (round(qe.getCoeff(t), NDIG),)
            for t in range(qe.size()) if qe.getCoeff(t) != 0.0))
        lin = tuple(sorted((key[le.getVar(t).VarName], round(le.getCoeff(t), NDIG))
                           for t in range(le.size()) if le.getCoeff(t) != 0.0))
        out.append((qc.QCSense, round(float(qc.QCRHS) - le.getConstant(), NDIG),
                    quad, lin))
    return sorted(out)


def count_loop_rows(model, key: dict, H1: int, T: int) -> int:
    """How many repeatability / loop-closure rows the model actually contains.

    Detected structurally rather than by name, because the sparse assembly does
    not name its rows: a loop-closure row is the only row shaped
    ``+1 s[.,.,T-1] - 1 s[.,.,H1-1] <= 0`` -- two nonzeros of opposite unit sign
    on the SAME variable block, at the end of the horizon and the end of the
    transitory phase.

    This is checked separately from the multiset comparison for a reason.  (S1)
    would happily pass if BOTH builds forgot the descriptor rows in the same
    way, which is exactly what happened while ``rainflow_v2`` had no
    ``repeatability`` hook: the loop closed on the mean only, so a
    cantelli / hoeffding / bernstein / chernoff cell could end the horizon with
    a larger variance / range / cumulant budget than it started the operating
    phase with -- a "repeatable" cycle that is not repeatable in the quantity
    its own reliability row reads.  An absolute count catches that; a
    comparison never will.
    """
    if H1 < 1 or H1 >= T:
        return 0                                   # no operating phase to close
    A = model.getA().tocsr()
    names = model.getAttr("VarName")
    sense, rhs = model.getAttr("Sense"), model.getAttr("RHS")
    found = 0
    for i in range(A.shape[0]):
        lo, hi = A.indptr[i], A.indptr[i + 1]
        if hi - lo != 2 or sense[i] != GRB_LE or abs(float(rhs[i])) > 1e-12:
            continue
        terms = sorted(zip(A.indices[lo:hi], A.data[lo:hi]))
        if sorted(round(float(v), 9) for _, v in terms) != [-1.0, 1.0]:
            continue
        blocks, steps = set(), set()
        for c, _ in terms:
            name = names[c]
            if "[" not in name:
                break
            blocks.add(name.split("[")[0])
            steps.add(int(name.split("[")[1].rstrip("]").split(",")[-1]))
        if len(blocks) == 1 and steps == {H1 - 1, T - 1}:
            found += 1
    return found


def expected_loop_rows(case: "Case") -> int:
    """How many there SHOULD be: the mean row per cell, plus one per extra
    descriptor the cell's bound reads (rainflow_v2._repeatability).

    markov     mean only
    cantelli   + v          bernstein  + v
    hoeffding  + R          chernoff   + K
    """
    if case.H < 1:
        return 0
    extra = 1 if case.bound in ("cantelli", "bernstein", "hoeffding",
                                "chernoff") else 0
    return case.F * case.L * (1 + extra)


def diff_report(what: str, a: list, b: list, limit: int = 3) -> list:
    """Human-readable first differences between two canonical lists."""
    lines = [f"    {what}: {len(a)} (indicator) vs {len(b)} (sparse)"]
    sa, sb = set(map(repr, a)), set(map(repr, b))
    miss = sorted(sa - sb)[:limit]
    extra = sorted(sb - sa)[:limit]
    for r in miss:
        lines.append(f"      only in indicator: {r[:180]}")
    for r in extra:
        lines.append(f"      only in sparse   : {r[:180]}")
    if not miss and not extra:
        lines.append("      same multiset (multiplicities may differ)")
    return lines


# ===========================================================================
# (S1) Equivalence
# ===========================================================================
def compare_models(case: Case, do_solve: bool, opts) -> dict:
    """Build both encodings of one case and compare them object by object."""
    rec = {"encoding": case.encoding, "sparse_cuts": case.sparse_cuts,
           "bound": case.bound, "impl": case.impl, "repair": case.repair,
           "allow_replacement": int(case.allow_replacement),
           "F": case.F, "M": case.M, "L": case.L, "H": case.H, "T": case.T}
    ctxs, models, walls = {}, {}, {}
    loop_form, sparse_form = PAIRS[case.encoding]
    try:
        for form in (loop_form, sparse_form):
            ctx, wall = build_one(case, form)
            ctxs[form], models[form], walls[form] = ctx, ctx.model, wall
    except Exception as exc:
        rec.update({"verdict": f"error: {type(exc).__name__}: {exc}",
                    "n_rows": math.nan, "n_gencon": math.nan})
        for c in ctxs.values():
            c.model.dispose()
        return rec

    mi, ms = models[loop_form], models[sparse_form]
    keys = {f: _column_key(m) for f, m in models.items()}
    LOOP, SPARSE = loop_form, sparse_form
    problems = []
    detail = []

    cols_i, cols_s = canon_columns(mi), canon_columns(ms)
    if set(cols_i) != set(cols_s):
        problems.append("columns")
        missing = sorted(set(cols_i) - set(cols_s))[:5]
        extra = sorted(set(cols_s) - set(cols_i))[:5]
        detail += [f"    columns: {len(cols_i)} vs {len(cols_s)}",
                   f"      only in indicator: {missing}",
                   f"      only in sparse   : {extra}"]
    else:
        bad = [n for n in cols_i if cols_i[n] != cols_s[n]]
        if bad:
            problems.append("column_attrs")
            detail.append(f"    column attributes differ on {len(bad)} of "
                          f"{len(cols_i)} columns (vtype/lb/ub/obj)")
            for n in bad[:5]:
                detail.append(f"      {n}: {cols_i[n]} vs {cols_s[n]}")

    lin_i = canon_linear(mi, keys[LOOP])
    lin_s = canon_linear(ms, keys[SPARSE])
    if lin_i != lin_s:
        problems.append("linear_rows")
        detail += diff_report("linear rows", lin_i, lin_s)

    ind_i = canon_indicator(mi, keys[LOOP])
    ind_s = canon_indicator(ms, keys[SPARSE])
    if ind_i != ind_s:
        problems.append("indicator_rows")
        detail += diff_report("indicator constraints", ind_i, ind_s)

    q_i = canon_quadratic(mi, keys[LOOP])
    q_s = canon_quadratic(ms, keys[SPARSE])
    if q_i != q_s:
        problems.append("quadratic_rows")
        detail += diff_report("quadratic constraints", q_i, q_s)

    # An ABSOLUTE check, not a comparison: both builds agreeing on a missing
    # descriptor row would sail through everything above.
    want = expected_loop_rows(case)
    got = {f: count_loop_rows(m, keys[f], case.H, case.T) for f, m in models.items()}
    if set(got.values()) != {want}:
        problems.append("loop_closure")
        detail.append(f"    loop closure: expected {want} rows "
                      f"({case.F * case.L} cells x mean"
                      + (" + descriptor" if want > case.F * case.L else "")
                      + f"), got {got}. The repeatability rows close the "
                      f"operating phase on EVERY state the reliability "
                      f"constraint reads; see rainflow_v2._repeatability and "
                      f"rainflow_sparse._repeatability_rows, which must stay "
                      f"in step.")

    rec.update({
        "n_cols": len(cols_i), "n_rows": len(lin_i), "n_gencon": len(ind_i),
        "n_qconstr": len(q_i),
        "n_loop_rows": got[LOOP], "n_loop_rows_expected": want,
        "build_loop_s": walls[LOOP], "build_sparse_s": walls[SPARSE],
        "build_speedup": (walls[LOOP] / walls[SPARSE]
                          if walls[SPARSE] > 0 else math.nan),
    })

    if do_solve:
        rec.update({k.replace(loop_form, "loop").replace(sparse_form, "sparse"): v
                    for k, v in _solve_pair(models, opts, problems, detail,
                                            loop_form, sparse_form).items()})

    rec["verdict"] = "IDENTICAL" if not problems else "MISMATCH: " + ",".join(problems)
    rec["detail"] = "\n".join(detail)
    for m in models.values():
        m.dispose()
    return rec


def _solve_pair(models: dict, opts, problems: list, detail: list,
                loop_form: str, sparse_form: str) -> dict:
    """Solve both models, and also compare their pure LP relaxations.

    The relaxation is the sharper of the two comparisons: two models with the
    same integer optimum can still have different relaxations, and here they
    must not, because they are supposed to BE the same model.
    """
    out = {}
    for form, m in models.items():
        try:
            out.update(_optimize_one(m, form, opts))
        except Exception as exc:
            # A solve can fail for reasons that say nothing about the model --
            # a size-limited licence, no free token, an out-of-memory kill. The
            # structural comparison (S1) is the check with teeth and it has
            # already run, so record the reason and keep the row rather than
            # losing the whole grid to one exception.
            detail.append(f"    solve skipped for {form}: "
                          f"{type(exc).__name__}: {exc}")
            out.update({f"solve_{form}_s": math.nan,
                        f"status_{form}": -1,
                        f"obj_{form}": math.nan,
                        f"nodes_{form}": math.nan,
                        f"lp_{form}": math.nan,
                        f"solve_error_{form}": f"{type(exc).__name__}: {exc}"})
    if -1 in (out[f"status_{loop_form}"], out[f"status_{sparse_form}"]):
        problems.append("solve_unavailable")
        return out

    if out[f"status_{loop_form}"] != out[f"status_{sparse_form}"]:
        problems.append("status")
        detail.append(f"    status {out[f'status_{loop_form}']} vs "
                      f"{out[f'status_{sparse_form}']}")
    for what, tol in (("obj", 1e-9), ("lp", 1e-9)):
        a, b = out[f"{what}_{loop_form}"], out[f"{what}_{sparse_form}"]
        if math.isnan(a) and math.isnan(b):
            continue
        if math.isnan(a) != math.isnan(b) or abs(a - b) > tol * max(1.0, abs(a)):
            problems.append(what)
            detail.append(f"    {what}: {a!r} vs {b!r}  (must be EXACT: the two "
                          f"builds are the same program)")
    return out


def _optimize_one(m, form: str, opts) -> dict:
    """Solve one model and its relaxation; return the columns they contribute."""
    m.Params.OutputFlag = 0
    m.Params.MIPGap = 1e-9
    m.Params.Seed = 0
    if opts.threads:
        m.Params.Threads = int(opts.threads)
    if opts.time_limit:
        m.Params.TimeLimit = float(opts.time_limit)
    t0 = time.perf_counter()
    m.optimize()
    out = {f"solve_{form}_s": time.perf_counter() - t0,
           f"status_{form}": int(m.Status),
           f"obj_{form}": (float(m.ObjVal) if m.SolCount else math.nan),
           f"nodes_{form}": float(getattr(m, "NodeCount", math.nan))}
    try:
        rel = m.relax()
        rel.Params.OutputFlag = 0
        rel.optimize()
        out[f"lp_{form}"] = (float(rel.ObjVal) if rel.SolCount else math.nan)
        rel.dispose()
    except Exception:
        out[f"lp_{form}"] = math.nan
    return out


def test_equivalence(opts) -> tuple[list, list]:
    """(S1) over the factorial grid of bound x impl x repair x replacement."""
    cases = []
    base = Case(F=opts.F, M=opts.M, L=opts.L, H=opts.H)
    for encoding, bound, impl, repair, repl in itertools.product(
            opts.encodings, opts.bounds, opts.impls, opts.repairs,
            opts.replacements):
        if (bound, repair) in SKIP:
            continue
        cases.append(Case(F=base.F, M=base.M, L=base.L, H=base.H,
                          bound=bound, impl=impl, repair=repair,
                          encoding=encoding, allow_replacement=repl,
                          sparse_cuts=opts.sparse_cuts,
                          pwl_points=opts.pwl_points))
    rows, lines = [], []
    header = (f"{'encoding':<10} {'bound':<10} {'impl':<8} {'repair':<7} "
              f"{'repl':>4} {'cols':>6} {'rows':>7} {'gencon':>7} {'quad':>5} "
              f"{'loop':>5}  {'loop[s]':>8} {'sp[s]':>7} {'x':>5}  verdict")
    lines += [header, "-" * len(header)]
    for case in cases:
        print(f"  [equivalence] {case.label()} ...", flush=True)
        rec = compare_models(case, opts.solve, opts)
        rows.append(rec)
        lines.append(
            f"{case.encoding:<10} {case.bound:<10} {case.impl:<8} "
            f"{case.repair:<7} {int(case.allow_replacement):>4} "
            f"{_fmt(rec.get('n_cols'), 0):>6} {_fmt(rec.get('n_rows'), 0):>7} "
            f"{_fmt(rec.get('n_gencon'), 0):>7} {_fmt(rec.get('n_qconstr'), 0):>5} "
            f"{_fmt(rec.get('n_loop_rows'), 0):>5}  "
            f"{_fmt(rec.get('build_loop_s'), 3):>8} "
            f"{_fmt(rec.get('build_sparse_s'), 3):>7} "
            f"{_fmt(rec.get('build_speedup'), 2):>5}  {rec['verdict']}")
        if rec.get("detail"):
            lines.append(rec["detail"])
    bad = [r for r in rows if r["verdict"] != "IDENTICAL"]
    lines.append("")
    if bad:
        lines.append(f"(S1) VIOLATION {len(bad)} of {len(rows)} cases differ. "
                     f"The sparse assembly must reproduce the indicator program "
                     f"exactly; a difference here is a bug in rainflow_sparse, "
                     f"not a modelling choice. The first differing rows are "
                     f"printed above -- match them against the family that emits "
                     f"them (_base_rows, _gating_rows, _reliability_rows, "
                     f"_repeatability_rows, _state_indicators).")
    else:
        lines.append(f"(S1) PASS all {len(rows)} cases build the identical "
                     f"model: same columns with the same types, bounds and "
                     f"objective coefficients, and the same linear, indicator "
                     f"and quadratic rows up to order.")
        lines.append("     Loop closure is present on both sides in every "
                     "case (the 'loop' column): the mean row per cell plus "
                     "one row per extra descriptor the cell's bound reads. "
                     "This is an ABSOLUTE count, so it also catches both "
                     "builds dropping the same row.")
        distinct = len({(r["encoding"], r["bound"], r["impl"], r["repair"],
                         r["allow_replacement"], r["n_cols"], r["n_rows"],
                         r["n_gencon"], r["n_qconstr"]) for r in rows})
        if distinct < len(rows):
            lines.append(
                f"     Of the {len(rows)} cases, {distinct} are structurally "
                f"distinct. markov and chernoff are already linear, so "
                f"rainflow_v2._resolve_impl falls back to 'exact' for them and "
                f"their 'tangent' / 'pwl' rows duplicate the 'exact' one. They "
                f"are run anyway: that fallback is itself a code path worth "
                f"exercising, and the whole grid costs under a minute.")
    return rows, lines


# ===========================================================================
# (S2) Build scaling
# ===========================================================================
def test_scaling(opts) -> tuple[list, list]:
    """(S2) build time along a geometric ladder in F, L, H and M.  No solving."""
    rows, lines = [], []
    for encoding in opts.encodings:
        base = Case(F=opts.F, M=opts.M, L=opts.L, H=opts.H,
                    bound=opts.scale_bound, impl=opts.scale_impl,
                    repair=opts.scale_repair, encoding=encoding,
                    pwl_points=opts.pwl_points)
        r, ln = _scaling_one(base, opts)
        rows += r
        lines += ln
    lines += _scaling_verdict(rows, opts.encodings)
    return rows, lines


def _scaling_one(base: Case, opts) -> tuple[list, list]:
    """The ladder for one encoding: loop assembly vs sparse assembly."""
    loop_form, sparse_form = PAIRS[base.encoding]
    rows, lines = [], []
    lines.append("")
    lines.append(f"encoding {base.encoding!r}: "
                 f"{loop_form} (per-cell addConstr) vs {sparse_form} (matrix API)")
    lines.append(f"base case: {base.label()}")
    lines.append(f"ladder: x{opts.factors}   repeats: {opts.repeats} (median)")
    lines.append("build only -- nothing is solved, see design note 1")
    lines.append("")
    header = (f"{'param':<6}{'value':>6} {'cols':>8}{'rows':>9}{'gencon':>9} "
              f"{'ind[s]':>9}{'sparse[s]':>10}{'speedup':>9}   "
              f"sparse phases [s]")
    lines += [header, "-" * len(header)]

    for param in opts.scale_params:
        for factor in opts.factors:
            case = _scaled(base, param, factor)
            if case is None:
                continue
            per_form, sizes, phases = {}, None, None
            for form in (loop_form, sparse_form):
                ts = []
                for _ in range(opts.repeats):
                    ctx, wall = build_one(case, form)
                    ts.append(wall)
                    md = ctx.model
                    sizes = (md.NumVars, md.NumConstrs, md.NumGenConstrs, md.NumNZs)
                    if form == sparse_form:
                        phases = dict(ctx.extras.get("build_phases_s", {}))
                    md.dispose()
                per_form[form] = statistics.median(ts)
                rows.append({"encoding": base.encoding,
                             "parameter": param, "value": getattr(case, param),
                             "formulation": form,
                             "assembly": ("sparse" if form == sparse_form
                                          else "loop"),
                             "build_s": per_form[form],
                             "n_cols": sizes[0], "n_rows": sizes[1],
                             "n_gencon": sizes[2], "n_nz": sizes[3],
                             "F": case.F, "M": case.M, "L": case.L, "H": case.H,
                             **{f"phase_{k}": v for k, v in (phases or {}).items()}})
            ti, tsp = per_form[loop_form], per_form[sparse_form]
            ph = "  ".join(f"{k.split('_')[0]}={v:.3f}"
                           for k, v in (phases or {}).items())
            lines.append(
                f"{param:<6}{getattr(case, param):>6} {sizes[0]:>8}{sizes[1]:>9}"
                f"{sizes[2]:>9} {ti:>9.3f}{tsp:>10.3f}"
                f"{(ti / tsp if tsp else math.nan):>8.2f}x   {ph}")
            print(f"  [scaling] {param}={getattr(case, param)}  "
                  f"{ti:.3f}s -> {tsp:.3f}s", flush=True)

    return rows, lines


def _scaled(base: Case, param: str, factor: int):
    """base scaled along one axis, or None when the value is not admissible."""
    val = int(getattr(base, param) * factor)
    if param == "M" and val >= base.F:            # config requires M < F
        return None
    if val < 1:
        return None
    return Case(**{**base.__dict__, param: val})


def _scaling_verdict(rows: list, encodings) -> list:
    """Speed-up summary plus the fitted growth exponent of each build."""
    lines, by_encoding = [""], {}
    for encoding in encodings:
        lines.append(f"  encoding {encoding!r}:")
        ratios = _verdict_one(rows, encoding, lines)
        by_encoding[encoding] = ratios
    lines += _verdict_overall(by_encoding)
    return lines


def _verdict_one(rows: list, encoding: str, lines: list) -> list:
    ratios = []
    for param in SCALE_PARAMS:
        sub = [r for r in rows if r["parameter"] == param
               and r["encoding"] == encoding]
        if not sub:
            continue
        by_form = {a: [(r["value"], r["build_s"]) for r in sub
                       if r["assembly"] == a] for a in ("loop", "sparse")}
        exps = {}
        for f, pts in by_form.items():
            pts = sorted(pts)
            if len(pts) >= 2 and all(v > 0 and t > 0 for v, t in pts):
                x = np.log([v for v, _ in pts]); y = np.log([t for _, t in pts])
                exps[f] = float(np.polyfit(x, y, 1)[0])
        r = [ti / ts for (_, ti), (_, ts) in
             zip(sorted(by_form["loop"]), sorted(by_form["sparse"])) if ts > 0]
        ratios += r
        lines.append(
            f"    {param}: build time ~ {param}^"
            f"{exps.get('loop', float('nan')):.2f} (loop), "
            f"^{exps.get('sparse', float('nan')):.2f} (sparse); "
            f"speed-up {min(r):.2f}x - {max(r):.2f}x" if r else f"    {param}: -")
    return ratios
def _verdict_overall(by_encoding: dict) -> list:
    """The point of running both encodings: where the ceiling comes from."""
    lines = [""]
    for encoding, ratios in by_encoding.items():
        if ratios:
            lines.append(f"(S2) encoding {encoding!r}: sparse assembly is "
                         f"{statistics.median(ratios):.2f}x faster to build "
                         f"(median over the ladder), {min(ratios):.2f}x - "
                         f"{max(ratios):.2f}x.")
    if len(by_encoding) < 2:
        lines.append("     Run both encodings (--encodings indicator,bigm) to "
                     "see where the ceiling on this comes from.")
        return lines
    lines += [
        "",
        "     Read the two numbers together. The gain is bounded by whatever "
        "part of the program has no matrix form:",
        "       'indicator' keeps 6*S*C general indicator constraints, which "
        "Gurobi can only take one at a time. The sparse assembly makes each "
        "call cheaper but cannot remove it, so once the linear families are "
        "free those calls ARE the build, and the speed-up saturates at a "
        "constant around 2x.",
        "       'bigm' has no indicator constraints at all: every row is "
        "linear, so the WHOLE program goes through one addMConstr per sense "
        "and the Theta(F*L*T) API calls disappear rather than getting cheaper. "
        "That is where the assembly actually pays.",
        "     So the encoding and the assembly are not independent in effect "
        "even though they are independent as options: 'bigm' is what makes "
        "'sparse' worth having, and 'bigm_sparse' is the combination to reach "
        "for at the large end of the scaling ladder.",
    ]
    return lines


def _warm_up() -> None:
    """Pay the import and Gurobi-environment cost before the clock starts."""
    try:
        import scipy.sparse
        assert scipy.sparse is not None      # imported for its cost, not its API
        for form in ("indicator", "sparse", "bigm", "bigm_sparse"):
            build_one(Case(F=2, M=1, L=1, H=2), form)[0].model.dispose()
    except Exception as exc:                               # keep the run alive
        print(f"  [warm-up] skipped: {type(exc).__name__}: {exc}")


# ===========================================================================
# End-to-end solve parity
# ===========================================================================
def test_solve(opts) -> tuple[list, list]:
    """Solve the same instance both ways; status, cost and nodes must agree."""
    rows, lines = [], []
    for encoding in opts.encodings:
        r, ln = _solve_one(encoding, opts)
        rows += r
        lines += ln
    return rows, lines


def _solve_one(encoding: str, opts) -> tuple[list, list]:
    case = Case(F=opts.F, M=opts.M, L=opts.L, H=opts.H,
                bound=opts.scale_bound, impl=opts.scale_impl,
                repair=opts.scale_repair, encoding=encoding,
                pwl_points=opts.pwl_points)
    loop_form, sparse_form = PAIRS[encoding]
    rows, lines = [], ["", f"instance: {case.label()}", ""]
    header = (f"{'form':<13} {'status':>7} {'objective':>14} {'nodes':>9} "
              f"{'build[s]':>9} {'solve[s]':>9}")
    lines += [header, "-" * len(header)]
    out = {}
    for form in (loop_form, sparse_form):
        ctx, wall = build_one(case, form)
        md = ctx.model
        try:
            got = _optimize_one(md, form, opts)
            rec = {"formulation": form, "status": got[f"status_{form}"],
                   "objective": got[f"obj_{form}"], "nodes": got[f"nodes_{form}"],
                   "build_s": wall, "solve_s": got[f"solve_{form}_s"],
                   "solve_error": ""}
        except Exception as exc:
            # See _solve_pair: a licence or memory failure is not a statement
            # about the model, and it must not take the rest of the run with it.
            rec = {"formulation": form, "status": -1, "objective": math.nan,
                   "nodes": math.nan, "build_s": wall, "solve_s": math.nan,
                   "solve_error": f"{type(exc).__name__}: {exc}"}
        out[form] = rec
        rows.append(rec)
        lines.append(f"{form:<13} {rec['status']:>7} "
                     f"{_fmt(rec['objective'], 6):>14} {_fmt(rec['nodes'], 0):>9} "
                     f"{wall:>9.3f} {_fmt(rec['solve_s'], 3):>9}"
                     + (f"   {rec['solve_error'][:60]}" if rec["solve_error"] else ""))
        md.dispose()
    lines.append("")
    a, b = out[loop_form], out[sparse_form]
    if a["status"] == -1 or b["status"] == -1:
        lines.append("(S3) SKIPPED neither model could be solved here (see the "
                     "error above -- a size-limited licence and an unavailable "
                     "token both look like this). (S1) is the test that "
                     "matters; run it with --tests equivalence.")
        return rows, lines
    same_obj = ((math.isnan(a["objective"]) and math.isnan(b["objective"]))
                or abs(a["objective"] - b["objective"])
                <= 1e-9 * max(1.0, abs(a["objective"])))
    if a["status"] == b["status"] and same_obj:
        lines.append("(S3) PASS same status and the same optimal cost.")
    else:
        lines.append("(S3) VIOLATION the two builds disagree end to end; run "
                     "--tests equivalence for the structural diff, which "
                     "localises the offending row family.")
    if a["nodes"] != b["nodes"]:
        lines.append(f"     note: node counts differ ({a['nodes']:.0f} vs "
                     f"{b['nodes']:.0f}). Row ORDER is not preserved by the "
                     f"sparse assembly (design note 5), and order alone can "
                     f"move Gurobi's tie-breaking. It is not evidence of a "
                     f"different model -- (S1) is.")
    return rows, lines


# ===========================================================================
# Output
# ===========================================================================
def _fmt(v, nd=3) -> str:
    if v is None:
        return "-"
    if isinstance(v, str):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(f):
        return "-"
    return f"{f:.{nd}f}" if nd else f"{f:.0f}"


def write_csv(rows: list, path: Path) -> None:
    if not rows:
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_scaling(rows: list, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    params = [p for p in SCALE_PARAMS if any(r["parameter"] == p for r in rows)]
    encodings = [e for e in ENCODINGS if any(r["encoding"] == e for r in rows)]
    if not params or not encodings:
        return
    nrow = len(encodings) + 1
    fig, axes = plt.subplots(nrow, len(params),
                             figsize=(3.4 * len(params), 3.0 * nrow),
                             squeeze=False)
    for e, encoding in enumerate(encodings):
        for c, param in enumerate(params):
            sub = [r for r in rows if r["parameter"] == param
                   and r["encoding"] == encoding]
            ax = axes[e][c]
            for assembly, style in (("loop", "o-"), ("sparse", "s-")):
                pts = sorted((r["value"], r["build_s"]) for r in sub
                             if r["assembly"] == assembly)
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], style,
                            label=f"{encoding}/{assembly}")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel(param)
            ax.set_ylabel(f"build time [s]\n({encoding})")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=7)
    for c, param in enumerate(params):
        ax2 = axes[-1][c]
        for encoding, colour in zip(encodings, ("tab:green", "tab:red")):
            sub = [r for r in rows if r["parameter"] == param
                   and r["encoding"] == encoding]
            lo = dict(sorted((r["value"], r["build_s"]) for r in sub
                             if r["assembly"] == "loop"))
            sp = dict(sorted((r["value"], r["build_s"]) for r in sub
                             if r["assembly"] == "sparse"))
            xs = sorted(set(lo) & set(sp))
            if xs:
                ax2.plot(xs, [lo[x] / sp[x] for x in xs], "^-", color=colour,
                         label=encoding)
        ax2.axhline(1.0, color="k", lw=0.8, ls="--")
        ax2.set_xscale("log"); ax2.set_xlabel(param)
        ax2.set_ylabel("speed-up (loop / sparse)")
        ax2.grid(True, which="both", alpha=0.3)
        ax2.legend(fontsize=7)
    fig.suptitle("Model build cost: per-cell addConstr vs sparse matrix assembly\n"
                 "(the ceiling is whatever has no matrix form: indicator rows)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# CLI
# ===========================================================================
def _csv_list(s, allowed=None, cast=str):
    vals = [cast(v.strip()) for v in str(s).split(",") if v.strip()]
    if allowed is not None:
        bad = [v for v in vals if v not in allowed]
        if bad:
            raise argparse.ArgumentTypeError(f"{bad} not in {tuple(allowed)}")
    return vals


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Compare formulation='indicator' with formulation='sparse': "
                    "structural equivalence first, build cost second.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tests", default="equivalence,scaling,solve",
                   help="comma-separated: equivalence, scaling, solve")
    p.add_argument("--out", default="test_results", help="output root")
    p.add_argument("--name", default="sparse", help="run-folder suffix")
    p.add_argument("--run-stamp", default=None, dest="run_stamp",
                   help="pin the run folder to <stamp>_<name> instead of "
                        "minting one from the clock. The Euler job scripts use "
                        "this so a resubmission lands next to its predecessor "
                        "and so several tasks can share a folder.")
    p.add_argument("--quick", action="store_true",
                   help="small grid and a short ladder; for CI")

    g = p.add_argument_group("base case")
    g.add_argument("--F", type=int, default=6)
    g.add_argument("--M", type=int, default=2)
    g.add_argument("--L", type=int, default=2)
    g.add_argument("--H", type=int, default=4, help="T = 2H")
    g.add_argument("--pwl-points", type=int, default=3, dest="pwl_points")
    g.add_argument("--sparse-cuts", default="off", dest="sparse_cuts",
                   choices=["off", "core", "full"],
                   help="also apply the sparse strengthening of the indicator "
                        "relaxation (rainflow_v2.add_sparse_cuts). It changes "
                        "the program, so the two assemblies have to agree with "
                        "it on as well as off -- run the grid both ways.")

    g = p.add_argument_group("equivalence grid")
    g.add_argument("--encodings", default=",".join(ENCODINGS),
                   help="which MILP encodings to test the assembly on: "
                        "'indicator' compares indicator vs sparse, 'bigm' "
                        "compares bigm vs bigm_sparse")
    g.add_argument("--bounds", default=",".join(BOUNDS))
    g.add_argument("--impls", default=",".join(IMPLS))
    g.add_argument("--repairs", default=",".join(REPAIRS))
    g.add_argument("--replacements", default="0,1",
                   help="values of allow_replacement to cover")
    g.add_argument("--solve", action="store_true",
                   help="also solve both models and require an EXACT match of "
                        "objective and LP relaxation (slow; needs a licence "
                        "large enough for the instance)")

    g = p.add_argument_group("scaling ladder")
    g.add_argument("--scale-params", default=",".join(SCALE_PARAMS),
                   dest="scale_params")
    g.add_argument("--factors", default="1,2,4,8",
                   help="geometric ladder applied to the base case")
    g.add_argument("--repeats", type=int, default=3,
                   help="builds per point; the MEDIAN is reported")
    g.add_argument("--scale-bound", default="cantelli", dest="scale_bound")
    g.add_argument("--scale-impl", default="tangent", dest="scale_impl")
    g.add_argument("--scale-repair", default="ard1", dest="scale_repair")

    g = p.add_argument_group("solver")
    g.add_argument("--threads", type=int, default=1,
                   help="Gurobi Threads; fixed at 1 so a timing is a timing")
    g.add_argument("--time-limit", type=float, default=120.0, dest="time_limit")
    g.add_argument("--no-plots", action="store_true")

    args = p.parse_args(argv)
    args.tests = _csv_list(args.tests, {"equivalence", "scaling", "solve"})
    args.encodings = _csv_list(args.encodings, ENCODINGS)
    args.bounds = _csv_list(args.bounds, BOUNDS)
    args.impls = _csv_list(args.impls, IMPLS)
    args.repairs = _csv_list(args.repairs, REPAIRS)
    args.replacements = [bool(int(v)) for v in _csv_list(args.replacements)]
    args.scale_params = _csv_list(args.scale_params, SCALE_PARAMS)
    args.factors = _csv_list(args.factors, cast=int)
    if args.quick:
        args.encodings = list(ENCODINGS)
        args.bounds = ["markov", "cantelli"]
        args.impls = ["tangent"]
        args.repairs = ["ard1"]
        args.replacements = [False]
        args.factors = [1, 2]
        args.repeats = 1
        args.F, args.M, args.L, args.H = 4, 1, 1, 3
    return args


def main(argv=None) -> int:
    opts = parse_args(argv)
    stamp = opts.run_stamp or datetime.now().strftime("%y%m%d_%H%M")
    run_dir = Path(opts.out) / f"{stamp}_{opts.name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Design note 2: the first sparse build in a process otherwise carries the
    # ``import scipy.sparse`` cost and reads as a slowdown.  Pay it here, once,
    # before any clock starts -- including the equivalence table's, which also
    # reports build times.
    _warm_up()

    summary = [f"test_sparse_version.py -- {datetime.now().isoformat(timespec='seconds')}",
               f"tests: {opts.tests}",
               f"base case: F={opts.F} M={opts.M} L={opts.L} H={opts.H} "
               f"(T={2 * opts.H})",
               ""]
    failed = False

    if "equivalence" in opts.tests:
        print("\n=== (S1) structural equivalence ===")
        rows, lines = test_equivalence(opts)
        write_csv(rows, run_dir / "equivalence.csv")
        summary += ["=" * 78, "(S1) STRUCTURAL EQUIVALENCE", "=" * 78] + lines + [""]
        failed |= any(r["verdict"] != "IDENTICAL" for r in rows)

    if "scaling" in opts.tests:
        print("\n=== (S2) build scaling ===")
        rows, lines = test_scaling(opts)
        write_csv(rows, run_dir / "scaling.csv")
        if not opts.no_plots:
            plot_scaling(rows, run_dir / "sparse_build.png")
        summary += ["=" * 78, "(S2) BUILD COST", "=" * 78] + lines + [""]

    if "solve" in opts.tests:
        print("\n=== (S3) end-to-end solve parity ===")
        rows, lines = test_solve(opts)
        write_csv(rows, run_dir / "solve.csv")
        summary += ["=" * 78, "(S3) SOLVE PARITY", "=" * 78] + lines + [""]

    text = "\n".join(summary)
    (run_dir / "summary.txt").write_text(text)
    print("\n" + text)
    print(f"\nwritten to {run_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
