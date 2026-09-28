#!/usr/bin/env python3
"""Solve one VBZ instance with a warm start; results in test.py's format.

The first cluster run hit the 600 s limit with sol_count = 0 and nodes = 1:
Gurobi never finished the root node, so it never produced an incumbent. That is
a find-a-feasible-point problem, not an infeasible model -- feas_oracle.py
exhibits a feasible schedule. This script hands Gurobi that schedule as a MIP
start, and checkpoints the search so a killed or truncated task still leaves a
usable record.

Output layout is test.py's, so the same downstream code reads both:

    <out>/<stamp>_<test>/
        scenario_base.yaml                the instance + solver options
        results_<strategy>.csv            one FIELDS row (streamed, flushed)
        results_<strategy>.yaml
        summary_<strategy>.txt
        progress_<strategy>.log           flushed per event; survives SIGKILL
        checkpoints_<strategy>.csv        <-- every --checkpoint-every seconds
        runs/<bound>_<impl>__<case>.yaml  replayable solver input + result
        solver_logs/<case>_<strategy>.log
        snapshots/<strategy>_t<sec>.npz   incumbent x/mu/v/m/r/u over time
        <case>_<strategy>_schedule.npz    final solution
        <case>_<strategy>_assign.csv      vehicle x step mission matrix

Local:
    python run_year.py input/vbz_man12e_year_solve.yaml --time-limit 3600

Euler (euler/submit_year.sh), one array task per strategy:
    python run_year.py input/vbz_man12e_year_solve.yaml --strategy warm \
        --out results --name year --run-stamp 202608231913 --threads 4 \
        --gurobi-params NodefileStart=1,SoftMemLimit=12
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import pathlib
import sys
import time
from datetime import datetime

import gurobipy as gp
import numpy as np
import yaml
from gurobipy import GRB

from fleet_management.config import load_config
from fleet_management.degradation_model.base import (build_fleet,
                                                     extract_solution,
                                                     resolve_run_options)

_HERE = pathlib.Path(__file__).resolve().parent


# ===========================================================================
# Reuse test.py rather than re-implement it (same trick as run_studies.py)
# ===========================================================================
def _load_harness():
    """Import the sibling test.py BY PATH.

    A plain `import test` is a trap: CPython ships a stdlib package called
    `test`. Everything structural -- the run-folder layout, the CSV schema, the
    provenance stamps, the Gurobi attribute dump -- is taken from there so a
    run_year row and a bound-test row are the same kind of object.
    """
    path = _HERE / "test.py"
    if not path.is_file():
        raise SystemExit(
            f"run_year.py expects test.py next to it (looked in {_HERE}). It "
            f"reuses the run-folder layout and the CSV schema from the bound-"
            f"test harness so that both write the same format.")
    spec = importlib.util.spec_from_file_location("bound_test_harness", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bound_test_harness"] = mod
    spec.loader.exec_module(mod)
    return mod


H = _load_harness()
FIELDS = H.FIELDS
TestRun = H.TestRun
collect_model_metrics = H.collect_model_metrics
_to_builtin = H._to_builtin
_dump_yaml = H._dump_yaml


STRATEGIES = {
    "cold":       dict(warm=False, params={}),
    "warm":       dict(warm=True,  params={"MIPFocus": 1, "NoRelHeurTime": 120,
                                           "Symmetry": 2, "Heuristics": 0.5}),
    "warm-norel": dict(warm=True,  params={"MIPFocus": 1, "NoRelHeurTime": 600,
                                           "Symmetry": 2}),
    "warm-bound": dict(warm=True,  params={"MIPFocus": 2, "Symmetry": 2,
                                           "Cuts": 2}),
    "warm-pwl":   dict(warm=True,  params={"MIPFocus": 1, "NoRelHeurTime": 120,
                                           "Symmetry": 2},
                       overrides={"reliability_impl": "pwl"}),
}

CKPT_COLS = ["wall_s", "where", "obj_best", "obj_bound", "gap",
             "nodes", "nodes_left", "sol_count", "iterations", "cuts",
             "snapshot_file"]


class _InstanceScenario:
    """Minimal stand-in for test.py's Scenario: TestRun only calls
    to_yaml_dict() on it. The instance comes from a YAML file, not from the
    synthetic generator, so there is no real Scenario to build."""

    def __init__(self, data, path):
        self._d = dict(data)
        self._d["_source_file"] = str(path)

    def to_yaml_dict(self):
        return _to_builtin(self._d)


def round_robin(F, M, T):
    """The witness feas_oracle.py certifies: even rotation, no interventions.
    Vehicles not on a mission stay idle (x[i,0,k]=0) so they cost no C_M."""
    x = np.zeros((F, M + 1, T), dtype=int)
    for k in range(T):
        for j in range(1, M + 1):
            x[(k * M + j - 1) % F, j, k] = 1
    return x


def parse_params(s):
    """'K=V,K=V' -> {K: V}; same format as test.py's --gurobi-params."""
    out = {}
    for item in (s or "").split(","):
        item = item.strip()
        if not item:
            continue
        k, _, v = item.partition("=")
        v = v.strip()
        for cast in (int, float):
            try:
                out[k.strip()] = cast(v)
                break
            except ValueError:
                continue
        else:
            out[k.strip()] = v
    return out


# ===========================================================================
# Checkpointing callback
# ===========================================================================
class Checkpointer:
    """Writes a scalar progress row every `every` seconds, and an .npz of the
    incumbent whenever a new one arrives at least `every` seconds after the
    last snapshot.

    Why both: with nodes = 1 after 600 s the MIP callback barely fires, so the
    root-node phases (presolve, simplex, barrier) are polled too -- otherwise a
    task that spends its whole allocation in the root LP leaves no trace of
    what it was doing.
    """

    def __init__(self, ctx, run, tag, every, t0):
        self.ctx, self.every, self.t0 = ctx, every, t0
        self.snap_dir = run.dir / "snapshots"
        self.snap_dir.mkdir(exist_ok=True)
        self.path = run.dir / f"checkpoints{tag}.csv"
        self.tag = tag
        self.run = run
        self._fh = self.path.open("w", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=CKPT_COLS)
        self._w.writeheader()
        self._fh.flush()
        self.last_ckpt = 0.0
        self.last_snap = -1e18
        self.n_snap = 0
        F, M, T, L = ctx.F, ctx.M, ctx.T, ctx.L
        self.shape_x = (F, M + 1, T)
        self.shape_c = (F, L, T)
        self.xv = [ctx.x[i, j, k] for i in range(F) for j in range(M + 1)
                   for k in range(T)]
        self.muv = [ctx.mu_var[i, l, k] for i in range(F) for l in range(L)
                    for k in range(T)]
        self.mv = [ctx.m_rep[i, l, k] for i in range(F) for l in range(L)
                   for k in range(T)]
        self.rv = ([ctx.r_rep[i, l, k] for i in range(F) for l in range(L)
                    for k in range(T)] if ctx.r_rep is not None else [])
        self.uv = [ctx.u_var[k] for k in range(T)]

    def _row(self, where, snap=""):
        self._w.writerow({k: v for k, v in where.items()} | {"snapshot_file": snap})
        self._fh.flush()

    def __call__(self, md, where):
        now = time.time() - self.t0
        try:
            if where == GRB.Callback.MIPSOL and now - self.last_snap >= self.every:
                obj = md.cbGet(GRB.Callback.MIPSOL_OBJ)
                name = f"{self.tag.lstrip('_')}_t{int(now):06d}.npz"
                np.savez_compressed(
                    self.snap_dir / name,
                    t=now, objective=obj,
                    x=np.array(md.cbGetSolution(self.xv)).reshape(self.shape_x),
                    mu=np.array(md.cbGetSolution(self.muv)).reshape(self.shape_c),
                    m=np.array(md.cbGetSolution(self.mv)).reshape(self.shape_c),
                    r=(np.array(md.cbGetSolution(self.rv)).reshape(self.shape_c)
                       if self.rv else np.zeros(0)),
                    u=np.array(md.cbGetSolution(self.uv)))
                self.last_snap = now
                self.n_snap += 1
                self._row(dict(wall_s=round(now, 1), where="MIPSOL",
                               obj_best=obj,
                               obj_bound=md.cbGet(GRB.Callback.MIPSOL_OBJBND),
                               nodes=md.cbGet(GRB.Callback.MIPSOL_NODCNT),
                               sol_count=md.cbGet(GRB.Callback.MIPSOL_SOLCNT)),
                          snap=name)
                self.run.note_progress(f"SNAPSHOT t={now:.0f}s obj={obj:.4g} "
                                       f"-> snapshots/{name}")
                return
            if now - self.last_ckpt < self.every:
                return
            rec = dict(wall_s=round(now, 1))
            if where == GRB.Callback.MIP:
                best = md.cbGet(GRB.Callback.MIP_OBJBST)
                bnd = md.cbGet(GRB.Callback.MIP_OBJBND)
                rec.update(where="MIP", obj_best=best, obj_bound=bnd,
                           gap=(abs(best - bnd) / abs(best)
                                if abs(best) > 1e-10 and best < GRB.INFINITY
                                else ""),
                           nodes=md.cbGet(GRB.Callback.MIP_NODCNT),
                           nodes_left=md.cbGet(GRB.Callback.MIP_NODLFT),
                           sol_count=md.cbGet(GRB.Callback.MIP_SOLCNT),
                           iterations=md.cbGet(GRB.Callback.MIP_ITRCNT),
                           cuts=md.cbGet(GRB.Callback.MIP_CUTCNT))
            elif where == GRB.Callback.MIPNODE:
                rec.update(where="MIPNODE",
                           obj_best=md.cbGet(GRB.Callback.MIPNODE_OBJBST),
                           obj_bound=md.cbGet(GRB.Callback.MIPNODE_OBJBND),
                           nodes=md.cbGet(GRB.Callback.MIPNODE_NODCNT),
                           sol_count=md.cbGet(GRB.Callback.MIPNODE_SOLCNT))
            elif where == GRB.Callback.SIMPLEX:
                rec.update(where="SIMPLEX(root)",
                           obj_best=md.cbGet(GRB.Callback.SPX_OBJVAL),
                           iterations=md.cbGet(GRB.Callback.SPX_ITRCNT))
            elif where == GRB.Callback.BARRIER:
                rec.update(where="BARRIER(root)",
                           obj_best=md.cbGet(GRB.Callback.BARRIER_PRIMOBJ),
                           iterations=md.cbGet(GRB.Callback.BARRIER_ITRCNT))
            elif where == GRB.Callback.PRESOLVE:
                rec.update(where="PRESOLVE")
            else:
                return
            self.last_ckpt = now
            self._row(rec)
            self.run.note_progress(
                f"CKPT t={now:.0f}s {rec['where']} best={rec.get('obj_best')} "
                f"bound={rec.get('obj_bound')} nodes={rec.get('nodes')}")
        except Exception as exc:                                # noqa: BLE001
            self.run.note_progress(f"checkpoint failed: {exc}")

    def close(self):
        self._fh.close()


# ===========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml_path")
    ap.add_argument("--strategy", default="warm",
                    help=f"one of {', '.join(STRATEGIES)}, or 'all' to list")
    ap.add_argument("--time-limit", type=float, default=3600.0)
    ap.add_argument("--mip-gap", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--gurobi-params", default="", dest="gurobi_params",
                    help="extra 'K=V,K=V'; applied AFTER the strategy so the "
                         "job script's memory guard always wins")
    ap.add_argument("--checkpoint-every", type=float, default=900.0,
                    help="seconds between progress rows / incumbent snapshots "
                         "(default 900 = 15 min; 0 disables)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--name", default="vbz_case")
    ap.add_argument("--test", default="year", help="names the run folder "
                                                   "<out>/<stamp>_<test>")
    ap.add_argument("--run-stamp", default=None, dest="run_stamp")
    ap.add_argument("--solver-log", default="on", dest="solver_log",
                    choices=["on", "off", "auto"])
    ap.add_argument("--verbose", type=int, default=1)
    a = ap.parse_args(argv)
    a.dry_run = False

    if a.strategy == "all":
        for k, v in STRATEGIES.items():
            print(f"{k:<12} warm={v['warm']}  {v['params']}")
        return 0
    if a.strategy not in STRATEGIES:
        ap.error(f"unknown strategy {a.strategy!r}; pick from "
                 f"{', '.join(STRATEGIES)}")
    strat = STRATEGIES[a.strategy]

    case = pathlib.Path(a.yaml_path).stem
    data = yaml.safe_load(open(a.yaml_path))
    data.update(strat.get("overrides", {}))
    sc = _InstanceScenario(data, a.yaml_path)
    tag = f"_{a.strategy}"

    run = TestRun(pathlib.Path(a.out), a.name, a.test, sc, a, suffix=tag)
    run.note_progress(f"START case={case} strategy={a.strategy} "
                      f"tl={a.time_limit}s gap={a.mip_gap} threads={a.threads}")
    log_dir = run.dir / "solver_logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{case}{tag}.log"

    rec = {}          # NOT pre-filled: TestRun.add setdefaults verdict
    rec.update(timestamp=datetime.now().isoformat(timespec="seconds"),
               test=a.test, parameter="case", value=f"{case}:{a.strategy}",
               bound=str(data.get("bound_method", "")),
               F=data.get("F"), M=data.get("M"), L=data.get("L"),
               H=data.get("H"), tau=data.get("tau"),
               epsilon=data.get("epsilon"), rho=str(data.get("rho")),
               repair_model=str(data.get("repair_model")),
               reliability_impl=str(data.get("reliability_impl")),
               pwl_points=data.get("pwl_points"),
               tangent_ref=data.get("tangent_ref"),
               allow_replacement=data.get("allow_replacement"),
               C_M=data.get("C_M"), C_R=data.get("C_R"), C_S=data.get("C_S"),
               C_P=data.get("C_P"), model=data.get("model"),
               req_mip_gap=a.mip_gap, req_time_limit=a.time_limit,
               req_verbose=a.verbose, threads=a.threads,
               gurobi_params=a.gurobi_params, solver_log=log_path.name,
               host=H._HOSTNAME, slurm_job=H._SLURM_JOB,
               git_branch=H._GIT.get("git_branch", ""),
               git_commit=H._GIT.get("git_commit", ""))

    ck = None
    t0 = time.time()
    try:
        cfg = load_config(data)
        opts = resolve_run_options(cfg, verbose=a.verbose, mip_gap=a.mip_gap,
                                   time_limit=a.time_limit)
        ctx = build_fleet(cfg, opts)
        md = ctx.model
        F, M, T, L = ctx.F, ctx.M, ctx.T, ctx.L
        md.update()
        rec.update(T=T, load=M * T / F)

        has_cap = any(c.ConstrName.startswith("capacity_")
                      for c in md.getConstrs())
        if has_cap:
            msg = ("aggregate damage cap 'capacity_*' is in the model: "
                   "base.py/config.py are not patched, or damage_capacity is a "
                   "number. See derivation.md section 8.")
            print(f"WARNING: {msg}", file=sys.stderr)
            run.note_progress(f"WARNING {msg}")
        rec["feasible_hint"] = "cap_on" if has_cap else "cap_off"

        # analytic screen: mission-weeks a cell can absorb before the bound binds
        try:
            rec["n_max_analytic"] = _n_max(data)
        except Exception:                                       # noqa: BLE001
            pass

        if strat["warm"]:
            x0 = round_robin(F, M, T)
            for i in range(F):
                for j in range(M + 1):
                    for k in range(T):
                        ctx.x[i, j, k].Start = float(x0[i, j, k])
            for i in range(F):
                for l in range(L):
                    for k in range(T):
                        ctx.m_rep[i, l, k].Start = 0.0
                        ctx.nb[i, l, k].Start = 1.0
                        if ctx.r_rep is not None:
                            ctx.r_rep[i, l, k].Start = 0.0
            run.note_progress(f"warm start: round-robin, "
                              f"{int(x0[:, 1:, :].sum())} assignments")

        if a.solver_log != "off":
            md.Params.LogFile = str(log_path)
        md.Params.Threads = a.threads
        md.Params.TimeLimit = a.time_limit
        md.Params.MIPGap = a.mip_gap
        for k, v in strat["params"].items():
            md.setParam(k, v)
        for k, v in parse_params(a.gurobi_params).items():
            md.setParam(k, v)          # job-script guards win over the strategy

        if a.checkpoint_every > 0:
            ck = Checkpointer(ctx, run, tag, a.checkpoint_every, t0)
            md.optimize(ck)
        else:
            md.optimize()
        wall = time.time() - t0

        rec.update(collect_model_metrics(md))
        rec.update(status=H.status_string(md.Status)
                   if hasattr(H, "status_string") else
                   {2: "optimal", 3: "infeasible", 4: "inf_or_unbounded",
                    9: "time_limit"}.get(md.Status, str(md.Status)),
                   objective=(float(md.ObjVal) if md.SolCount else ""),
                   mip_gap=(float(md.MIPGap) if md.SolCount else ""),
                   obj_bound=float(md.ObjBound), wall_s=wall)

        if md.SolCount:
            res = extract_solution(ctx, cfg, md)
            _save_solution(run, case, tag, res, rec, data)
            print(f"\nstatus={rec['status']} obj={rec['objective']} "
                  f"gap={rec['mip_gap']} incumbents={md.SolCount} "
                  f"max mu/tau={rec['mu_max']}")
        else:
            print(f"\nstatus={rec['status']} NO INCUMBENT after {wall:.0f} s "
                  f"(bound {md.ObjBound:.4g}). Try --strategy warm-norel or "
                  f"raise --time-limit.", file=sys.stderr)
    except Exception as exc:                                    # noqa: BLE001
        import traceback
        rec.update(status="error", wall_s=time.time() - t0,
                   traceback=traceback.format_exc()[-3000:])
        run.note_progress(f"ERROR {exc}")
        print(f"ERROR: {exc}", file=sys.stderr)
    finally:
        if ck is not None:
            ck.close()

    run.add(rec, data, sc)
    report = [f"case          : {case}",
              f"strategy      : {a.strategy}",
              f"status        : {rec['status']}",
              f"objective     : {rec['objective']}",
              f"bound         : {rec['obj_bound']}",
              f"gap           : {rec['mip_gap']}",
              f"incumbents    : {rec.get('sol_count')}",
              f"nodes         : {rec.get('nodes')}",
              f"wall_s        : {rec['wall_s']}",
              f"checkpoints   : {ck.path.name if ck else 'disabled'}"
              f" ({ck.n_snap if ck else 0} snapshots)",
              f"damage cap on : {rec['feasible_hint'] == 'cap_on'}"]
    run.close(report, extra={"case": case, "strategy": a.strategy})
    print("\n".join(report))
    print(f"\nrun folder: {run.dir}")
    return 0 if rec["status"] != "error" else 1


def _n_max(data):
    """Mission-weeks a cell absorbs before the chance constraint binds, using
    the mean/variance/support of the WORST mission. Same role as test.py's
    n_max: a cheap screen that says whether the horizon is even reachable."""
    tau, eps = float(data["tau"]), float(data["epsilon"])
    mu = np.atleast_2d(np.array(data["mu"], dtype=float))
    v = np.atleast_2d(np.array(data["v"], dtype=float))
    b = np.atleast_2d(np.array(data["support"], dtype=float))
    mu0 = np.atleast_1d(np.array(data.get("mu_0", 0.0), dtype=float))
    Le = math.log(1.0 / eps)
    out = []
    for l in range(mu.shape[0]):
        m1, v1, b1 = mu[l].max(), v[l].max(), b[l].max()
        m0 = mu0[l] if mu0.size > l else float(mu0[0])
        bound = str(data.get("bound_method", "cantelli"))
        n = 0.0
        for n_try in range(1, 10000):
            vT = v1 * n_try
            if bound == "cantelli":
                cap = tau - math.sqrt((1 - eps) / eps * vT)
            elif bound == "hoeffding":
                cap = tau - math.sqrt(0.5 * Le * b1 * b1 * n_try)
            elif bound == "bernstein":
                c = Le * b1 / 3.0
                cap = tau - (c + math.sqrt(c * c + 2 * Le * vT))
            else:
                cap = eps * tau
            if m0 + n_try * m1 > cap:
                break
            n = n_try
        out.append(n)
    return min(out)


def _save_solution(run, case, tag, res, rec, data):
    """Everything a post-analysis could want, in three files."""
    x = np.asarray(res["x"])
    mu = np.asarray(res["mu"])
    v = np.asarray(res["v"])
    m = np.asarray(res["m"])
    r = np.asarray(res["r"]) if res.get("r") is not None else np.zeros(0)
    u = np.asarray(res["u"])
    tau = float(np.max(res["tau"]))
    rec.update(mu_max=float(mu.max() / tau), v_max=float(v.max()),
               n_repairs=int(m.sum()), n_replacements=int(r.sum()),
               n_depot=int(x[:, 0, :].sum()))

    np.savez_compressed(run.dir / f"{case}{tag}_schedule.npz",
                        x=x, mu=mu, v=v, m=m, r=r, u=u, tau=tau)

    # vehicle x step mission matrix: 0 = depot, -1 = idle, j = mission j
    F, _, T = x.shape
    assign = np.full((F, T), -1, dtype=int)
    for i in range(F):
        for k in range(T):
            j = int(np.argmax(x[i, :, k])) if x[i, :, k].sum() else -1
            assign[i, k] = j
    with (run.dir / f"{case}{tag}_assign.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["vehicle"] + [f"k{k}" for k in range(T)]
                   + ["loaded_steps", "depot_steps"])
        for i in range(F):
            w.writerow([i] + list(assign[i])
                       + [int((assign[i] > 0).sum()), int((assign[i] == 0).sum())])

    # per-cell terminal state and per-step u
    with (run.dir / f"{case}{tag}_cells.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["vehicle", "component", "mu_T", "mu_T_over_tau", "v_T",
                    "repairs", "replacements"])
        for i in range(mu.shape[0]):
            for l in range(mu.shape[1]):
                w.writerow([i, l, mu[i, l, -1], mu[i, l, -1] / tau,
                            v[i, l, -1], int(m[i, l].sum()),
                            int(r[i, l].sum()) if r.size else 0])
        w.writerow([])
        w.writerow(["step", "u_k"])
        for k, val in enumerate(u):
            w.writerow([k, val])


if __name__ == "__main__":
    sys.exit(main())
