"""
test.py -- experiment harness for the reliability *bounds* (Step-3 sanity tests).

Purpose
-------
Check whether a *tighter* probabilistic bound really buys a *cheaper* optimal
schedule, and whether a *looser* bound *fails earlier*.  The reliability
requirement  P(D > tau) <= eps  is the ONLY place a bound enters the model: the
objective  J = C_M(x) + C_R(z) + C_rep(r) + C_D(u)  is identical for all five
bounds.  So if the feasible sets are nested, the optimal cost must be ordered the
same way, and

    cost(markov) >= cost(cantelli) >= cost(hoeffding)
                 >= cost(bernstein) >= cost(chernoff)                     (H1)
    a looser bound becomes infeasible at a milder stress level than a
    tighter one                                                           (H2)

are falsifiable predictions.  All runs start from ONE base case; each test
increments a single parameter and records cost + solve time (H1) or the stress
level at which the model goes infeasible (H2).

Tests  (--tests analytic,base,sweep,failure)
--------------------------------------------
  analytic    no solver.  For the base case computes, per bound, the largest
              number of reference missions n_max a component absorbs before the
              reliability constraint is violated -- using formulas that mirror
              `rainflow.py` line by line, NOT the textbook inequalities.  n_max
              is the scalar "tightness": larger n_max = tighter bound = fewer
              interventions = lower cost, and it predicts both (H1) and (H2) for
              free.  If n_max is not ordered, no MILP result can be.
  base        solve the base case once per bound; print the comparison table.
  sweep       (H1) for each parameter in --params, solve every value x every
              bound, and plot cost + time.
  failure     (H2) walk each parameter from mild to harsh until the model goes
              infeasible, and record where each bound gives up.

Outputs
-------
One folder per test, under --out (default ./test_results):

    test_results/<yymmdd>_<name>_<test>/
        <yymmdd>_results_<name>_<test>.csv     one row per run
        <yymmdd>_results_<name>_<test>.yaml    all runs, aggregated
        <yymmdd>_summary_<name>_<test>.txt     hypothesis check / violations
        <yymmdd>_scenario_base.yaml            the base case, reusable as input
        <yymmdd>_<test>_cost_<param>.png       cost + solve time vs parameter
        <yymmdd>_<test>_failure.png            failure thresholds (failure test)
        runs/<bound>__<param><value>.yaml      per-setting input + result
                                               ('input' is a valid solver input)

Design notes -- READ BEFORE TRUSTING A RESULT
---------------------------------------------
1. **Descriptor consistency.**  Comparing bounds is only meaningful if mu, v,
   support and cgf describe the *same* random increment.  They are therefore all
   derived here from one explicit distribution (scaled Bernoulli: a mission
   causes damage b with probability p, else 0), never chosen independently:
       mu = p*b,  v = p(1-p)b^2,  support = b,  cgf = ln(1-p+p*e^{s*b}).
   Hand-picking these in a YAML is the easiest way to "disprove" (H1) by
   accident.

2. **The default MIP gap is 0.12** (`base.resolve_run_options`).  A 12 % gap is
   far larger than the cost differences between neighbouring bounds, so the
   harness forces --mip-gap 1e-4 and stores the achieved gap per run.  Any
   ordering violation smaller than the gap slack is reported as "within gap",
   not as a falsification.  Use --no-time-limit for the cleanest numbers: every
   run then solves to that gap with no wall-clock cap, so a "time_limit" status
   can never be confused with a real result.

3. **The cost balance decides whether the bound bites.**  The defaults are the
   requested C_M=1.0, C_R=0.5, C_S=2.0, C_P=1.0.  Two consequences:
     * C_S (= C_D, damage regularisation) is now the LARGEST coefficient, so the
       optimizer already has an incentive to suppress damage on its own.  If it
       suppresses enough, the reliability constraint never binds at the optimum
       and every bound returns the same schedule -- a pass that proves nothing.
       Every row records n_repairs / n_depot / mu_max, and the summary flags the
       run as uninformative when those are identical across bounds.  If that
       happens, lower C_S (--C-S 0.05) rather than doubting the bounds.
     * The *ordering* in (H1) does not depend on the coefficients at all (nested
       feasible sets, identical objective).  Only the size of the separation
       does.
   Note C_P is accepted by `config.load_config` but the current objective
   (`base.build_objective`) never reads it -- no periodicity term is imposed --
   so it is written into every input for the record and has no effect on the
   result.  C_rep is only used when --allow-replacement is set.

4. **Infeasible = infinitely expensive.**  markov allows only mu <= eps*tau, so
   it is often infeasible for a horizon in which the other four are comfortable
   (a single mission increment can exceed eps*tau).  That is consistent with
   (H1): infeasible rows are stored with objective = inf and plotted as an
   annotated marker, not dropped.  The `failure` test turns this from a nuisance
   into the measurement.

5. **(H1)/(H2) are NOT theorems.**  Only two links are guaranteed: chernoff
   (with a well-chosen tilt s) dominates hoeffding and bernstein, because both
   are derived from it by over-bounding the CGF.  The middle links depend on the
   regime:
       hoeffding beats cantelli  only if  p(1-p) > Le*eps / (2(1-eps))
       bernstein beats hoeffding only if  4p(1-p) < 1 - (2/3)*sqrt(2*Le/n)
   with Le = ln(1/eps).  Those two windows overlap only for n above roughly 5-9
   (small eps helps).  `--calibrate-n` places the design point inside the
   overlap; `analytic` prints the two window checks so a violation can be read
   as "wrong regime" rather than "wrong code".

6. **Confounds in the sweeps.**  `base.add_base_constraints` imposes
   sum_{i,l} mu[i,l,k] <= F - M  and  depot_capacity defaults to F - M.  So the
   L sweep adds cells to a cap that does not grow with L, and the F / M sweeps
   move the cap and the depot capacity as well as the fleet size.  Comparisons
   ACROSS bounds at a fixed value are clean; the shape of a curve ALONG a
   parameter mixes these effects in.  In the failure test this matters more:
   shrinking F or growing M eventually makes the model infeasible for *every*
   bound through the depot capacity, not through the reliability constraint.
   Rows carry `feasible_hint` (a capacity-only feasibility estimate) so a
   capacity failure can be told apart from a reliability failure.

7. **Repair model.**  chernoff has no closed ARD1 recursion, so all bounds run
   with repair_model='ardinf' by default (--repair to change).  Mixing repair
   models across bounds would confound the comparison.

8. **The horizon must be long enough to feel the bound.**  Every mission is
   served at every step, so a vehicle absorbs T*M/F missions on average, while a
   bound lets it absorb n_max between interventions.  If n_max >= T*M/F the
   constraint never binds and that bound ties with the next one.  `analytic`
   prints this comparison per bound; the default H=10 (T=20, load=8) sits just
   above n_max(bernstein).  Only the tightest bound is allowed to be slack: it
   then simply attains the unconstrained optimum, which is still the cheapest.

Usage
-----
    python test.py --tests analytic                      # free, no Gurobi
    python test.py --tests base
    python test.py --tests sweep --params L,H,M,F
    python test.py --tests failure --failure-params H,M,F,epsilon
    python test.py --tests analytic,base,sweep,failure --no-time-limit
    python test.py --tests sweep --dry-run               # validate, do not solve

Author: test harness for the degradation-aware EV fleet scheduler.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

# Provenance: which machine / Slurm task produced a row (empty off-cluster)
import os
import socket

_HOSTNAME = socket.gethostname()
_SLURM_JOB = os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", ""))
if os.environ.get("SLURM_ARRAY_TASK_ID"):
    _SLURM_JOB += f"_{os.environ['SLURM_ARRAY_TASK_ID']}"

# ---------------------------------------------------------------------------
# Import the package (works from the repo root or from src/)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
for _cand in (_HERE, _HERE / "src"):
    if (_cand / "fleet_management").is_dir() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))


def _git_info() -> dict:
    """Branch / commit that produced a result row.

    Thesis results are worth nothing if you cannot say which code made them, and
    on a cluster the checkout is remote and easy to move on underneath you. A
    dirty working tree is flagged, because then the commit alone does not identify
    the code. Failures are silent: not every checkout is a git repo.
    """
    try:
        import subprocess

        def q(*args) -> str:
            return subprocess.run(["git", *args], cwd=_HERE, capture_output=True,
                                  text=True, timeout=5).stdout.strip()

        commit = q("rev-parse", "--short", "HEAD")
        return {"git_branch": q("rev-parse", "--abbrev-ref", "HEAD"),
                "git_commit": (commit + "+dirty") if q("status", "--porcelain") else commit}
    except Exception:
        return {"git_branch": "", "git_commit": ""}


_GIT = _git_info()


def _import_config():
    """Config only: no gurobipy, so `--dry-run` validates inputs without a licence."""
    from fleet_management.config import load_config
    return load_config


def _import_solver():
    """Imported lazily so `analytic` / `--dry-run` work without Gurobi."""
    from fleet_management.config import load_config
    from fleet_management.degradation_model.rainflow import solve as rainflow_solve
    return load_config, rainflow_solve


# Hypothesised order: loosest (most expensive, fails first) -> tightest.
BOUNDS_ORDER = ("markov", "cantelli", "hoeffding", "bernstein", "chernoff")
# Implementations, ordered loosest (most conservative) -> exact.  tangent and pwl
# are safe INNER approximations of the exact quadratic, so their feasible set is a
# SUBSET of it: they can only cost more, never less (H3).
IMPLS_ORDER = ("tangent", "pwl", "exact")
# markov and chernoff are already linear; `rainflow._resolve_impl` silently falls
# back to "exact" for them, so a non-exact run would just duplicate the exact one.
IMPL_AWARE_BOUNDS = ("cantelli", "hoeffding", "bernstein")
SWEEP_PARAMS = ("L", "H", "M", "F")
# Failure ladders: every list runs MILD -> HARSH, so the index is the stress level.
STRESS_LADDERS = {
    "H": ("longer horizon", [4, 6, 8, 10, 12, 14, 16, 20, 24]),
    "M": ("more missions", [1, 2, 3, 4, 5, 6]),
    "L": ("more components", [1, 2, 3, 4, 5, 6]),
    "F": ("smaller fleet", [8, 7, 6, 5, 4, 3, 2]),
    "epsilon": ("stricter reliability", [0.2, 0.1, 0.05, 0.02, 0.01, 5e-3, 2e-3, 1e-3]),
    "tau": ("lower failure threshold", [2.0, 1.5, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3]),
    "rho": ("weaker repair", [1.0, 0.9, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1]),
}
_INT_PARAMS = ("F", "M", "L", "H")


def _cast(param: str, value):
    return int(value) if param in _INT_PARAMS else float(value)


class Shard:
    """Work-unit partitioner for Slurm job arrays.

    Every test enumerates its solves in a deterministic order; shard k of n takes
    the units where `index % n == k`.  Each array task therefore writes its own
    output folder and no two tasks duplicate work.  The hypothesis checks need
    *all* bounds at a design point, so a shard never evaluates them -- run
    `--merge` afterwards to combine the shards and do the checks once.
    """

    def __init__(self, k: int, n: int):
        if not (0 <= k < n):
            raise SystemExit(f"--shard k/n needs 0 <= k < n (got {k}/{n})")
        self.k, self.n, self.i = k, n, 0

    def take(self) -> bool:
        mine = (self.i % self.n) == self.k
        self.i += 1
        return mine

    def __str__(self) -> str:
        return f"{self.k}/{self.n}"


def _mine(opts, ) -> bool:
    """True when this work unit belongs to this shard (always True without one)."""
    shard = getattr(opts, "shard_obj", None)
    return True if shard is None else shard.take()


def bound_impl_combos(impls, announce: bool = False) -> list:
    """The (bound, impl) pairs actually worth solving.

    A non-exact impl on markov / chernoff is dropped: `rainflow._resolve_impl`
    falls back to "exact" for them, so the run would be a byte-identical
    duplicate that just burns solver time.
    """
    out, skipped = [], []
    for bound in BOUNDS_ORDER:
        for impl in impls:
            if impl != "exact" and bound not in IMPL_AWARE_BOUNDS:
                skipped.append(f"{bound}/{impl}")
                continue
            out.append((bound, impl))
    if announce and skipped:
        print(f"  [impl] skipping {len(skipped)} duplicate combination(s): "
              f"{', '.join(skipped)} -- these bounds are linear and fall back "
              f"to 'exact'")
    return out


def impl_of_record(rec: dict) -> str:
    return str(rec.get("reliability_impl", "exact"))


def cost_for(values: dict, bound: str, impl: str):
    """Cost of (bound, impl), falling back to the exact run for linear bounds."""
    if (bound, impl) in values:
        return values[(bound, impl)]
    return values.get((bound, "exact"), math.nan)


# ===========================================================================
# Scenario: ONE physical case, from which every bound's input is derived
# ===========================================================================
@dataclass(frozen=True)
class Scenario:
    """A single design point.

    The increment distribution is the invariant: mission j inflicts damage
    ``b_j`` with probability ``p`` and 0 otherwise (scaled Bernoulli).  Every
    descriptor a bound needs is derived from it, so all five bounds see exactly
    the same physics.  ``b_ref`` is not a free parameter: it is calibrated so
    that the hoeffding constraint binds after ``n_target`` reference missions,
    which is what puts the design point in the regime where (H1) can hold at all
    (see design note 5).
    """
    # fleet / horizon (the swept parameters)
    F: int = 5
    M: int = 2
    L: int = 1
    H: int = 10                     # T = 2H; see design note 8
    # reliability
    tau: float = 1.0
    epsilon: float = 0.02
    rho: float = 0.9
    mu_0: float = 0.0
    v_0: float = 0.0
    # increment distribution
    p: float = 0.05                 # Bernoulli success probability
    n_target: float = 6.0           # calibration target for hoeffding's n_max
    severity_spread: float = 0.25   # missions span b_ref*(1 -+ spread)
    # costs (design note 3).  C_S is the alias the objective reads as C_D;
    # C_P is inert in the current objective; C_rep only matters with replacement.
    C_M: float = 1.0
    C_R: float = 0.5
    C_S: float = 2.0
    C_P: float = 1.0
    C_rep: float = 25.0
    # model options
    repair_model: str = "ardinf"
    reliability_impl: str = "exact"
    pwl_points: int = 8             # segments used by reliability_impl='pwl'
    tangent_ref: float = 0.5        # tangent taken at tangent_ref*tau
    allow_replacement: bool = False

    # ---- derived quantities ------------------------------------------------
    @property
    def T(self) -> int:
        return 2 * int(self.H)

    @property
    def Le(self) -> float:
        return math.log(1.0 / self.epsilon)

    @property
    def load(self) -> float:
        """Missions a vehicle must absorb on average over the horizon."""
        return self.T * self.M / self.F

    @property
    def b_ref(self) -> float:
        """Scale of the reference (severity-1) mission.

        From  m1*n + b*sqrt(Le*n/2) = tau  with m1 = p*b (the hoeffding
        constraint of `_rel_hoeffding_exact`), b factors out:
            b = tau / (p*n + sqrt(Le*n/2)).
        """
        n = float(self.n_target)
        return self.tau / (self.p * n + math.sqrt(self.Le * n / 2.0))

    @property
    def severities(self) -> np.ndarray:
        """Per-mission severity factors, mean 1, so mission mix matters."""
        if self.M == 1:
            return np.ones(1)
        s = np.linspace(1.0 - self.severity_spread, 1.0 + self.severity_spread, self.M)
        return s / s.mean()

    @property
    def b(self) -> np.ndarray:
        return self.b_ref * self.severities

    @property
    def s_chernoff(self) -> float:
        """Tilt s maximising the reference-mission chernoff budget.

        n_max(s) = (s*tau + ln eps) / k1(s),  k1(s) = ln(1-p+p*e^{s*b_ref}).
        s is a fixed model parameter (the MILP cannot optimise it), so a poor
        choice makes chernoff look loose -- that would be a property of the
        parameterisation, not of the bound.  Coarse grid + local refinement.
        """
        b = self.b_ref
        s_min = self.Le / self.tau                        # config.py requires s > this
        s_hi = min(60.0 / max(b, 1e-12), 1e4)             # keep e^{s*b} finite
        best_s, best_n = s_min * 1.001, -np.inf
        for lo, hi, n_pts in ((s_min * 1.001, s_hi, 4000), (None, None, 2000)):
            if lo is None:                                # local refinement pass
                lo, hi = max(best_s * 0.9, s_min * 1.001), best_s * 1.1
            for s in np.linspace(lo, hi, n_pts):
                k1 = self._cgf(s, b)
                if k1 <= 1e-15:
                    continue
                n = (s * self.tau + math.log(self.epsilon)) / k1
                if n > best_n:
                    best_n, best_s = n, s
        return float(best_s)

    def _cgf(self, s: float, b: float) -> float:
        """ln E e^{s W} for the scaled-Bernoulli increment (numerically safe)."""
        z = s * b
        if z > 700.0:
            return math.log(self.p) + z
        return math.log1p(self.p * (math.exp(z) - 1.0))

    def descriptors(self) -> dict:
        """Per-mission (M,) descriptor arrays, all from the same distribution."""
        b = self.b
        s = self.s_chernoff
        return {
            "mu": self.p * b,                              # mean increment
            "v": self.p * (1.0 - self.p) * b ** 2,         # variance increment
            "support": b,                                  # support width
            "cgf": np.array([self._cgf(s, bj) for bj in b]),
            "s_chernoff": s,
        }

    # ---- inputs ------------------------------------------------------------
    def to_input(self, bound: str) -> dict:
        """Raw input mapping for `config.load_config` for one bound.

        Every descriptor is supplied for every bound (config validates `v` for
        all rainflow cells; unused descriptors are simply not read), so the five
        inputs differ ONLY in `bound_method`.  The returned dict is a complete,
        valid solver input: dump it to YAML and `solver.solve` will run it.
        """
        d = self.descriptors()
        data = {
            "model": "rainflow",
            "bound_method": bound,
            "repair_model": self.repair_model,
            "F": int(self.F), "M": int(self.M), "L": int(self.L), "H": int(self.H),
            "tau": float(self.tau), "epsilon": float(self.epsilon),
            "rho": float(self.rho),
            "mu_0": float(self.mu_0), "v_0": float(self.v_0),
            "mu": _floats(d["mu"]), "v": _floats(d["v"]),
            "support": _floats(d["support"]), "cgf": _floats(d["cgf"]),
            "s_chernoff": float(d["s_chernoff"]),
            "C_M": float(self.C_M), "C_R": float(self.C_R),
            "C_S": float(self.C_S), "C_P": float(self.C_P),
            # Step-3 knobs; config.load_config passes these through as options
            "reliability_impl": str(self.reliability_impl),
            "pwl_points": int(self.pwl_points),
            "tangent_ref": float(self.tangent_ref),
        }
        if self.allow_replacement:
            data["C_rep"] = float(self.C_rep)
            data["allow_replacement"] = True
        return data

    def variant(self, **overrides) -> "Scenario":
        return replace(self, **overrides)

    def label(self) -> str:
        return (f"F={self.F} M={self.M} L={self.L} H={self.H} (T={self.T}) "
                f"tau={self.tau} eps={self.epsilon} rho={self.rho} p={self.p} "
                f"b_ref={self.b_ref:.4g} impl={self.reliability_impl}"
                + (f"({self.pwl_points})" if self.reliability_impl == "pwl" else ""))

    def to_yaml_dict(self) -> dict:
        """Human-readable record of the design point (not a solver input)."""
        d = self.descriptors()
        return {
            "fleet": {"F": int(self.F), "M": int(self.M), "L": int(self.L),
                      "H": int(self.H), "T": self.T,
                      "load_missions_per_vehicle": float(self.load)},
            "reliability": {"tau": float(self.tau), "epsilon": float(self.epsilon),
                            "rho": float(self.rho), "mu_0": float(self.mu_0),
                            "v_0": float(self.v_0)},
            "increment_distribution": {
                "family": "scaled Bernoulli: damage b with probability p, else 0",
                "p": float(self.p), "n_target": float(self.n_target),
                "b_ref": float(self.b_ref),
                "severity_spread": float(self.severity_spread),
                "severities": _floats(self.severities),
                "mu": _floats(d["mu"]), "v": _floats(d["v"]),
                "support": _floats(d["support"]), "cgf": _floats(d["cgf"]),
                "s_chernoff": float(d["s_chernoff"]),
            },
            "costs": {"C_M": float(self.C_M), "C_R": float(self.C_R),
                      "C_S": float(self.C_S), "C_P": float(self.C_P),
                      "C_rep": float(self.C_rep),
                      "note": "C_S is read as C_D (damage regularisation); "
                              "C_P is inert in the current objective; C_rep is "
                              "used only with allow_replacement"},
            "options": {"repair_model": self.repair_model,
                        "reliability_impl": self.reliability_impl,
                        "pwl_points": int(self.pwl_points),
                        "tangent_ref": float(self.tangent_ref),
                        "allow_replacement": bool(self.allow_replacement),
                        "note": "pwl_points/tangent_ref only affect the "
                                "'pwl'/'tangent' implementations; markov and "
                                "chernoff are linear and ignore the impl"},
            "analytic_n_max": {b: float(n_max(self, b)) for b in BOUNDS_ORDER},
        }


def _floats(arr) -> list:
    return [float(x) for x in np.asarray(arr).ravel()]


# ===========================================================================
# Analytic tightness:  n_max per bound, mirroring rainflow.py's constraints
# ===========================================================================
def _n_from_sqrt_form(m1: float, c: float, tau: float) -> float:
    """Largest n with  m1*n + c*sqrt(n) <= tau   (m1, c, tau > 0)."""
    if tau <= 0:
        return 0.0
    y = (-c + math.sqrt(c * c + 4.0 * m1 * tau)) / (2.0 * m1)   # y = sqrt(n)
    return max(0.0, y * y)


def n_max(sc: Scenario, bound: str) -> float:
    """Reference missions a component absorbs before the bound is violated.

    Continuous n (the integer floor is what the schedule can use).  Each branch
    mirrors the corresponding builder in `rainflow.py`:
      markov     mu <= eps*tau
      cantelli   (1-eps)V <= eps*(tau-mu)^2
      hoeffding  (tau-mu)^2 >= 0.5*Le*R,          R = n*b^2
      bernstein  0.5*t^2 - (Le*b/3)*t - Le*V >= 0, t = tau-mu
      chernoff   K - s*tau <= ln eps,             K = n*k1(s)
    with mu = n*m1, V = n*v1 for the reference mission (severity 1).
    """
    tau, eps, Le = sc.tau, sc.epsilon, sc.Le
    b = sc.b_ref
    m1 = sc.p * b
    v1 = sc.p * (1.0 - sc.p) * b * b

    if bound == "markov":
        return eps * tau / m1
    if bound == "cantelli":
        return _n_from_sqrt_form(m1, math.sqrt((1.0 - eps) / eps * v1), tau)
    if bound == "hoeffding":
        return _n_from_sqrt_form(m1, math.sqrt(Le * b * b / 2.0), tau)
    if bound == "bernstein":
        # t = tau - m1*n  must satisfy  t >= A + sqrt(A^2 + 2*Le*v1*n),  A = Le*b/3
        A = Le * b / 3.0
        if tau - A <= 0:
            return 0.0
        qa = m1 * m1
        qb = -(2.0 * m1 * (tau - A) + 2.0 * Le * v1)
        qc = (tau - A) ** 2 - A * A
        disc = qb * qb - 4.0 * qa * qc
        if disc < 0:                       # never violated in the reachable range
            return math.inf
        n1 = (-qb - math.sqrt(disc)) / (2.0 * qa)
        n2 = (-qb + math.sqrt(disc)) / (2.0 * qa)
        cands = [n for n in (n1, n2) if n >= 0.0 and tau - A - m1 * n >= 0.0]
        return max(0.0, min(cands)) if cands else 0.0
    if bound == "chernoff":
        s = sc.s_chernoff
        k1 = sc._cgf(s, b)
        return (s * tau + math.log(eps)) / k1 if k1 > 0 else math.inf
    raise ValueError(f"unknown bound {bound!r}")


def regime_checks(sc: Scenario) -> dict:
    """The two conditions under which the middle links of (H1) can hold."""
    q = sc.p * (1.0 - sc.p)
    hoeff_gt_cant_rhs = sc.Le * sc.epsilon / (2.0 * (1.0 - sc.epsilon))
    n = max(n_max(sc, "hoeffding"), 1e-9)
    bern_gt_hoeff_rhs = 1.0 - (2.0 / 3.0) * math.sqrt(2.0 * sc.Le / n)
    return {
        "p(1-p)": q,
        "hoeffding>cantelli  needs p(1-p) >": hoeff_gt_cant_rhs,
        "hoeffding>cantelli": q > hoeff_gt_cant_rhs,
        "4p(1-p)": 4.0 * q,
        "bernstein>hoeffding needs 4p(1-p) <": bern_gt_hoeff_rhs,
        "bernstein>hoeffding": 4.0 * q < bern_gt_hoeff_rhs,
    }


def survival_floor(sc: Scenario) -> float:
    """Reference-mission budget a vehicle needs just to keep operating.

    Even with a repair after every single mission, an ardinf/ard1 contraction
    only removes a fraction rho, so the mean damage of a vehicle that repeatedly
    flies the *severest* mission floors at
        mu_ss = (1 - rho)*mu_ss + m_max   =>   mu_ss = m_max / rho,
    i.e. s_max/rho reference missions' worth of damage.  A bound whose n_max sits
    below that cannot admit any schedule at all.
    """
    return float(np.max(sc.severities)) / sc.rho


def feasible_hint(sc: Scenario, bound: str) -> bool:
    """Cheap necessary-condition check: can a schedule exist at all?

    Two independent arguments:
      * survival -- n_max must cover the steady-state floor s_max/rho
        (`survival_floor`); below it even repairing every step is not enough;
      * capacity -- the fleet must cover T*M mission-days.  Without repair it can
        cover F*n_max; each repair buys another n_max, and at most (F-M) vehicles
        can sit at the depot per step, i.e. at most (F-M)*T repairs.
    A True here does NOT guarantee feasibility (it ignores the aggregate damage
    cap and the exact contraction bookkeeping); a False is a strong indication
    that a `failure`-test failure is *capacity/repair* driven rather than a
    genuine difference in bound tightness (design note 6).
    """
    n = n_max(sc, bound)
    if n < survival_floor(sc):
        return False
    if math.isinf(n):
        return True
    capacity = sc.F * n + max(0, sc.F - sc.M) * sc.T * n
    return capacity >= sc.T * sc.M


# ===========================================================================
# One solve
# ===========================================================================
def run_case(sc: Scenario, bound: str, opts) -> tuple[dict, dict]:
    """Solve one (scenario, bound); return (record, input dict)."""
    data = sc.to_input(bound)
    rec = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "bound": bound, "F": sc.F, "M": sc.M, "L": sc.L, "H": sc.H, "T": sc.T,
        "tau": sc.tau, "epsilon": sc.epsilon, "rho": sc.rho, "p": sc.p,
        "b_ref": sc.b_ref, "mu_ref": sc.p * sc.b_ref,
        "s_chernoff": sc.s_chernoff, "repair_model": sc.repair_model,
        "reliability_impl": sc.reliability_impl,
        "pwl_points": sc.pwl_points, "tangent_ref": sc.tangent_ref,
        "allow_replacement": sc.allow_replacement,
        "C_M": sc.C_M, "C_R": sc.C_R, "C_S": sc.C_S, "C_P": sc.C_P,
        "threads": getattr(opts, "threads", None) or "",
        "host": _HOSTNAME, "slurm_job": _SLURM_JOB,
        "git_branch": _GIT["git_branch"], "git_commit": _GIT["git_commit"],
        "n_max_analytic": n_max(sc, bound),
        # a bound only bites when n_max < load (design note 8)
        "load": sc.load,
        "feasible_hint": feasible_hint(sc, bound),
    }

    if opts.dry_run:
        _import_config()(data)                   # still validates the input
        rec.update({"status": "dry_run", "objective": math.nan, "mip_gap": math.nan,
                    "obj_bound": math.nan, "runtime_s": math.nan, "wall_s": math.nan})
        return rec, data

    load_config, rainflow_solve = _import_solver()
    cfg = load_config(data)
    t0 = time.perf_counter()
    try:
        res = rainflow_solve(
            cfg,
            verbose=opts.verbose,
            mip_gap=opts.mip_gap,
            time_limit=opts.time_limit,          # None => no wall-clock cap
            allow_replacement=sc.allow_replacement,
            reliability_impl=sc.reliability_impl,
            pwl_points=sc.pwl_points,
            tangent_ref=sc.tangent_ref,
            # On a shared cluster node Gurobi would otherwise spawn threads for
            # every core it can see, not for the cores Slurm gave the job.
            gurobi_params=({"Threads": int(opts.threads)}
                           if getattr(opts, "threads", None) else None),
        )
    except Exception as exc:                     # keep the sweep alive
        rec.update({"status": f"error: {type(exc).__name__}: {exc}",
                    "objective": math.nan, "wall_s": time.perf_counter() - t0})
        return rec, data
    wall = time.perf_counter() - t0

    obj = res.get("objective")
    rec.update({
        "status": res.get("status"),
        # infeasible == infinitely expensive (design note 4)
        "objective": (math.inf if res.get("status") == "infeasible"
                      else (float(obj) if obj is not None else math.nan)),
        "mip_gap": _f(res.get("mip_gap")),
        "obj_bound": _f(res.get("bound")),
        "wall_s": wall,
    })

    md = res.get("model")
    if md is not None:
        for key, attr in (("runtime_s", "Runtime"), ("n_vars", "NumVars"),
                          ("n_constrs", "NumConstrs"), ("n_bin", "NumBinVars"),
                          ("nodes", "NodeCount")):
            try:
                rec[key] = float(getattr(md, attr))
            except Exception:
                rec[key] = math.nan

    # binding diagnostics: are the bounds actually doing anything? (design note 3)
    for key, arr, red in (("n_repairs", res.get("m"), np.sum),
                          ("n_replacements", res.get("r"), np.sum),
                          ("mu_max", res.get("mu"), np.max),
                          ("v_max", res.get("v"), np.max)):
        rec[key] = float(red(arr)) if arr is not None else math.nan
    x = res.get("x")
    rec["n_depot"] = float(np.sum(x[:, 0, :])) if x is not None else math.nan

    if md is not None:
        try:
            md.dispose()                         # release the Gurobi environment
        except Exception:
            pass
    return rec, data


def _f(value):
    return float(value) if value is not None else math.nan


def classify(rec: dict) -> str:
    """'feasible' | 'infeasible' | 'unknown' -- only 'infeasible' counts as failure.

    A time-limited run with no solution is *unknown*, not a failure; treating it
    as one would make the failure thresholds an artefact of the time limit
    (which is why --no-time-limit is recommended for the failure test).
    """
    status = str(rec.get("status", ""))
    if status in ("infeasible", "inf_or_unbounded"):
        return "infeasible"
    obj = rec.get("objective")
    if isinstance(obj, float) and math.isfinite(obj):
        return "feasible"
    return "unknown"


# ===========================================================================
# Bookkeeping: one folder per test; CSV + YAML, streamed
# ===========================================================================
FIELDS = ["timestamp", "test", "parameter", "value", "bound", "status",
          "objective", "mip_gap", "obj_bound", "runtime_s", "wall_s",
          "verdict", "n_max_analytic", "load", "feasible_hint",
          "n_repairs", "n_replacements", "n_depot", "mu_max", "v_max",
          "n_vars", "n_constrs", "n_bin", "nodes",
          "F", "M", "L", "H", "T", "tau", "epsilon", "rho", "p", "b_ref",
          "mu_ref", "s_chernoff", "repair_model", "reliability_impl",
          "pwl_points", "tangent_ref",
          "allow_replacement", "C_M", "C_R", "C_S", "C_P",
          "threads", "host", "slurm_job", "git_branch", "git_commit"]


class TestRun:
    """Output folder for ONE test: csv + aggregate yaml + per-run yaml + plots."""

    def __init__(self, out_root: Path, name: str, test: str, sc: Scenario, opts,
                 suffix: str = ""):
        self.stamp = f"{datetime.now():%y%m%d}"
        self.test = test
        self.dir = out_root / f"{self.stamp}_{name}_{test}{suffix}"
        self.runs_dir = self.dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.stem = f"{self.stamp}_{name}_{test}{suffix}"
        self.csv_path = self.dir / f"{self.stem.replace(f'{self.stamp}_', f'{self.stamp}_results_', 1)}.csv"
        self.yaml_path = self.csv_path.with_suffix(".yaml")
        self.summary_path = self.dir / f"{self.stamp}_summary_{name}_{test}{suffix}.txt"
        self.rows: list[dict] = []
        self._fh = self.csv_path.open("w", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=FIELDS, extrasaction="ignore")
        self._w.writeheader()
        # the base case, dumped once per test folder so each folder stands alone
        _dump_yaml(self.dir / f"{self.stamp}_scenario_base.yaml",
                   {"test": test, "created": datetime.now().isoformat(timespec="seconds"),
                    "code_version": dict(_GIT, host=_HOSTNAME, slurm_job=_SLURM_JOB),
                    "solver_options": {"mip_gap": opts.mip_gap,
                                       "time_limit": opts.time_limit,
                                       "dry_run": bool(opts.dry_run)},
                    "base_case": sc.to_yaml_dict()})

    # ---- one run -----------------------------------------------------------
    def add(self, rec: dict, data: dict, sc: Scenario) -> None:
        rec.setdefault("verdict", classify(rec))
        self.rows.append(rec)
        self._w.writerow({k: rec.get(k, "") for k in FIELDS})
        self._fh.flush()
        _dump_yaml(self.runs_dir / f"{self._run_stem(rec)}.yaml", {
            "test": rec.get("test"), "parameter": rec.get("parameter"),
            "value": rec.get("value"), "bound": rec.get("bound"),
            # 'input' is exactly what went into config.load_config: a valid,
            # self-contained solver input, so any run can be replayed with
            #     python -c "from fleet_management.solver import solve; solve('<f>')"
            "input": _to_builtin(data),
            "settings": sc.to_yaml_dict(),
            "result": _to_builtin(rec),
        })

    def _run_stem(self, rec: dict) -> str:
        """`<bound>_<impl>__<param><value>`.  The impl belongs in the name: two
        impls of the same bound at the same design point would otherwise write to
        the same file and silently overwrite each other."""
        param, value = rec.get("parameter", "-"), rec.get("value", "")
        tag = "base" if param in ("-", "", None) else f"{param}{value}"
        impl = impl_of_record(rec)
        if impl == "pwl":
            impl = f"pwl{rec.get('pwl_points', '')}"
        return (f"{rec['bound']}_{impl}__{tag}"
                .replace("/", "_").replace(" ", "").replace(".", "p"))

    # ---- finish ------------------------------------------------------------
    def close(self, report: list[str], extra: dict | None = None) -> None:
        self._fh.close()
        _dump_yaml(self.yaml_path, {
            "test": self.test,
            "created": datetime.now().isoformat(timespec="seconds"),
            "n_runs": len(self.rows),
            "summary": extra or {},
            "runs": [_to_builtin(r) for r in self.rows],
        })
        self.summary_path.write_text("\n".join(report))


def _dump_yaml(path: Path, payload: dict) -> None:
    with path.open("w") as fh:
        yaml.safe_dump(_to_builtin(payload), fh, default_flow_style=False,
                       sort_keys=False)


def _to_builtin(value):
    """NumPy / nested containers -> plain Python, so safe_dump never chokes."""
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_to_builtin(v) for v in value.tolist()]
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None                              # nan -> null reads better
    return value


# ===========================================================================
# Plots
# ===========================================================================
def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:
        print(f"  [plot] matplotlib unavailable ({exc}); skipping plots")
        return None


def _safe_log_scale(ax, values, axis: str = "y") -> None:
    """Log-scale only when every finite value is strictly positive.

    matplotlib raises "Data cannot be log-scaled because all values are <= 0"
    when nothing positive is plotted (e.g. a --dry-run, where every solve time is
    NaN), and silently drops zeros otherwise -- a 0.0 s solve would vanish. So
    fall back to a linear axis unless the data is genuinely all-positive.
    """
    finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if finite and all(v > 0 for v in finite):
        ax.set_xscale("log") if axis == "x" else ax.set_yscale("log")


def _plots_enabled(opts) -> bool:
    """No plots for a dry run: every cost and time is NaN, so the figures would
    be empty."""
    if getattr(opts, "no_plots", False):
        return False
    if getattr(opts, "dry_run", False):
        print("  [plot] skipped (--dry-run produces no costs or times)")
        return False
    return True


def plot_parameter(rows: list[dict], param: str, run: TestRun,
                   base_value=None) -> Path | None:
    """Cost (top) and solve time (bottom) vs one parameter, one line per bound."""
    plt = _pyplot()
    if plt is None:
        return None
    sub = [r for r in rows if r.get("parameter") == param]
    if not sub:
        return None

    fig, (ax_c, ax_t) = plt.subplots(2, 1, figsize=(7.5, 8.0), sharex=True,
                                     height_ratios=[2, 1])
    finite = [r["objective"] for r in sub
              if isinstance(r["objective"], float) and math.isfinite(r["objective"])]
    top = max(finite) * 1.15 if finite else 1.0
    all_times: list = []

    # colour identifies the bound, line style the implementation
    impls_seen = [im for im in IMPLS_ORDER
                  if any(impl_of_record(r) == im for r in sub)]
    styles = {"tangent": ":", "pwl": "--", "exact": "-"}
    annotated = False
    for bound in BOUNDS_ORDER:
        for impl in impls_seen:
            pts = sorted((r for r in sub if r["bound"] == bound
                          and impl_of_record(r) == impl),
                         key=lambda r: r["value"])
            if not pts:
                continue
            label = bound if len(impls_seen) == 1 else f"{bound}/{impl}"
            colour = f"C{BOUNDS_ORDER.index(bound)}"
            xs = [r["value"] for r in pts]
            ys = [r["objective"] for r in pts]
            ts = [r.get("runtime_s", math.nan) for r in pts]
            good = [(x, y) for x, y in zip(xs, ys)
                    if isinstance(y, float) and math.isfinite(y)]
            ax_c.plot([g[0] for g in good], [g[1] for g in good], marker="o",
                      label=label, color=colour, ls=styles.get(impl, "-"))
            bad_x = [x for x, y in zip(xs, ys)
                     if not (isinstance(y, float) and math.isfinite(y))]
            if bad_x:
                ax_c.plot(bad_x, [top] * len(bad_x), marker="x", linestyle="none",
                          color=colour)
                if not annotated:
                    ax_c.annotate("infeasible / no solution", (bad_x[0], top),
                                  textcoords="offset points", xytext=(4, -12),
                                  fontsize=7, color=colour)
                    annotated = True
            ax_t.plot(xs, ts, marker="s", label=label, color=colour,
                      ls=styles.get(impl, "-"))
            all_times += ts

    if base_value is not None:
        for ax in (ax_c, ax_t):
            ax.axvline(base_value, color="0.6", lw=0.8, ls=":", zorder=0)
        ax_c.annotate("base case", (base_value, top), textcoords="offset points",
                      xytext=(4, 4), fontsize=7, color="0.4")

    if not finite:                               # degenerate: nothing solved
        ax_c.text(0.5, 0.5, "no bound produced a feasible solution",
                  transform=ax_c.transAxes, ha="center", va="center",
                  fontsize=9, color="0.35")
    hypo = " $\\geq$ ".join(BOUNDS_ORDER)
    style_key = ("" if len(impls_seen) == 1 else
                 "\nstyle: " + ", ".join(f"{im} {styles[im]}" for im in impls_seen))
    ax_c.set_ylabel("optimal cost  J")
    ax_c.set_title(f"Bound tightness vs {param}   (impl: {', '.join(impls_seen)})"
                   f"\nhypothesis: {hypo}{style_key}", fontsize=9)
    ax_c.grid(alpha=0.3)
    ax_c.legend(fontsize=7 if len(impls_seen) > 1 else 8,
                ncol=2 if len(impls_seen) > 1 else 1)
    ax_t.set_xlabel(param)
    ax_t.set_ylabel("solve time [s]")
    _safe_log_scale(ax_t, all_times)
    ax_t.grid(alpha=0.3, which="both")
    ax_t.set_xticks(sorted({r["value"] for r in sub}))
    fig.tight_layout()

    path = run.dir / f"{run.stem}_cost_{param}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_analytic(sc: Scenario, run: TestRun, sweeps: dict) -> Path | None:
    """n_max per bound over each swept parameter (solver-free tightness view)."""
    plt = _pyplot()
    if plt is None or not sweeps:
        return None
    params = list(sweeps)
    fig, axes = plt.subplots(1, len(params), figsize=(4.0 * len(params), 3.8),
                             squeeze=False)
    for ax, param in zip(axes[0], params):
        values = sweeps[param]
        seen: list = []
        for bound in BOUNDS_ORDER:
            ys = [n_max(sc.variant(**{param: _cast(param, v)}), bound)
                  for v in values]
            # an unbounded n_max would blow up the axis limits; drop it instead
            ys = [math.nan if math.isinf(y) else y for y in ys]
            seen += ys
            ax.plot(values, ys, marker="o", label=bound)
        loads = [sc.variant(**{param: _cast(param, v)}).load for v in values]
        seen += loads
        ax.plot(values, loads, color="0.4", ls="--", lw=1.0, label="load T*M/F")
        ax.set_xlabel(param)
        ax.set_ylabel("$n_{max}$ (missions before violation)")
        _safe_log_scale(ax, seen)
        ax.set_xticks(values)
        ax.grid(alpha=0.3, which="both")
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Analytic tightness: larger $n_{max}$ = tighter bound; a bound "
                 "binds only where its curve is below the dashed load line",
                 fontsize=9)
    fig.tight_layout()
    path = run.dir / f"{run.stem}_nmax.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_impl(rows: list[dict], run: TestRun) -> Path | None:
    """Price of linearisation: cost and solve time per (bound, implementation)."""
    plt = _pyplot()
    if plt is None:
        return None
    sub = [r for r in rows if r.get("parameter") == "impl"]
    if not sub:
        return None
    bounds = [b for b in BOUNDS_ORDER if any(r["bound"] == b for r in sub)]
    impls = [im for im in IMPLS_ORDER if any(impl_of_record(r) == im for r in sub)]
    fig, (ax_c, ax_t) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    width = 0.8 / max(len(impls), 1)
    hatches = {"tangent": "//", "pwl": "..", "exact": ""}
    for j, impl in enumerate(impls):
        xs = [i + j * width for i in range(len(bounds))]
        costs, times = [], []
        for b in bounds:
            hit = [r for r in sub if r["bound"] == b and impl_of_record(r) == impl]
            obj = hit[0].get("objective") if hit else math.nan
            costs.append(obj if isinstance(obj, float) and math.isfinite(obj)
                         else math.nan)
            times.append(hit[0].get("runtime_s", math.nan) if hit else math.nan)
        ax_c.bar(xs, costs, width, label=impl, hatch=hatches.get(impl, ""),
                 edgecolor="white")
        ax_t.bar(xs, times, width, label=impl, hatch=hatches.get(impl, ""),
                 edgecolor="white")
    for ax, title, ylab in ((ax_c, "optimal cost (lower is better)", "cost  J"),
                            (ax_t, "solve time", "solve time [s]")):
        ax.set_xticks([i + width * (len(impls) - 1) / 2 for i in range(len(bounds))])
        ax.set_xticklabels(bounds, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    _safe_log_scale(ax_t, [r.get("runtime_s", math.nan) for r in sub])
    fig.suptitle("(H3) inner approximations cost more but should solve faster: "
                 "tangent $\\geq$ pwl $\\geq$ exact in cost", fontsize=10)
    fig.tight_layout()
    path = run.dir / f"{run.stem}_impl.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_pwl_convergence(rows: list[dict], run: TestRun) -> Path | None:
    """pwl cost vs segment count, against the exact cost as a dashed reference."""
    plt = _pyplot()
    if plt is None:
        return None
    sub = [r for r in rows if r.get("parameter") == "pwl_points"]
    if not sub:
        return None
    exact = {r["bound"]: r.get("objective") for r in rows
             if r.get("parameter") == "impl" and impl_of_record(r) == "exact"}
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for bound in BOUNDS_ORDER:
        pts = sorted((r for r in sub if r["bound"] == bound),
                     key=lambda r: r["value"])
        if not pts:
            continue
        colour = f"C{BOUNDS_ORDER.index(bound)}"
        ax.plot([r["value"] for r in pts], [r["objective"] for r in pts],
                marker="o", color=colour, label=f"{bound} pwl")
        ref = exact.get(bound)
        if isinstance(ref, float) and math.isfinite(ref):
            ax.axhline(ref, color=colour, ls="--", lw=1.0, alpha=0.7)
    ax.set_xlabel("pwl_points (segments)")
    ax.set_ylabel("optimal cost  J")
    ax.set_title("pwl tightens toward the exact encoding (dashed) as segments grow",
                 fontsize=9)
    ax.set_xticks(sorted({r["value"] for r in sub}))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = run.dir / f"{run.stem}_pwl_convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_failure(thresholds: dict, run: TestRun) -> Path | None:
    """Where each bound gives up, per stress parameter (stress level on the x axis)."""
    plt = _pyplot()
    if plt is None or not thresholds:
        return None
    params = list(thresholds)
    fig, axes = plt.subplots(1, len(params), figsize=(4.2 * len(params), 3.8),
                             squeeze=False)
    for ax, param in zip(axes[0], params):
        info = thresholds[param]
        ladder = info["ladder"]
        # colour = bound, opacity = implementation (fainter is more conservative)
        keys = sorted(info["failed_at_index"],
                      key=lambda k: (BOUNDS_ORDER.index(k.split("/")[0]),
                                     IMPLS_ORDER.index(k.split("/")[1])))
        alphas = {"tangent": 0.4, "pwl": 0.7, "exact": 1.0}
        ys = range(len(keys))
        widths, labels, colours, opacity = [], [], [], []
        for key in keys:
            bound, impl = key.split("/")
            colours.append(f"C{BOUNDS_ORDER.index(bound)}")
            opacity.append(alphas.get(impl, 1.0))
            idx = info["failed_at_index"][key]
            if idx is None:                       # survived the whole ladder
                widths.append(len(ladder))
                labels.append("no failure")
            else:
                widths.append(idx)                # rungs survived before failing
                labels.append(f"fails at {ladder[idx]:g}")
        bars = ax.barh(list(ys), widths, color=colours)
        for bar, alpha in zip(bars, opacity):
            bar.set_alpha(alpha)
        for y, w, lab in zip(ys, widths, labels):
            ax.annotate(lab, (w, y), textcoords="offset points", xytext=(4, -3),
                        fontsize=8)
        ax.set_yticks(list(ys))
        ax.set_yticklabels(keys, fontsize=7)
        ax.set_xlim(0, len(ladder) + 1.2)
        ax.set_xticks(range(len(ladder)))
        ax.set_xticklabels([f"{v:g}" for v in ladder], fontsize=7, rotation=45)
        ax.set_xlabel(f"{param}  ({info['direction']}, mild -> harsh)")
        ax.set_title("rungs survived", fontsize=9)
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle("(H2) a looser bound should give up earlier (shorter bar)",
                 fontsize=10)
    fig.tight_layout()
    path = run.dir / f"{run.stem}_thresholds.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ===========================================================================
# Hypothesis checks
# ===========================================================================
def check_order(values: dict, gaps: dict) -> list[str]:
    """(H1): verify cost(looser) >= cost(tighter) along BOUNDS_ORDER.

    A violation is only real if it exceeds the slack implied by the two runs'
    MIP gaps: with gap g the reported objective may sit up to g*|obj| above the
    true optimum, so differences below that are not evidence against (H1).
    """
    issues = []
    present = [b for b in BOUNDS_ORDER
               if isinstance(values.get(b), float) and not math.isnan(values.get(b))]
    for looser, tighter in zip(present, present[1:]):
        lo, hi = values[looser], values[tighter]
        if math.isinf(lo):                     # infeasible looser bound: consistent
            continue
        if math.isinf(hi):
            issues.append(f"VIOLATION {looser}={lo:.6g} but {tighter}=infeasible")
            continue
        g = max(gaps.get(looser) or 0.0, gaps.get(tighter) or 0.0)
        slack = max(1e-6, g * max(abs(lo), abs(hi)))
        if hi > lo + slack:
            issues.append(f"VIOLATION {looser}={lo:.6g} < {tighter}={hi:.6g} "
                          f"(excess {hi - lo:.3g} > slack {slack:.3g})")
        elif hi > lo + 1e-9:
            issues.append(f"within-gap inversion {looser}={lo:.6g} "
                          f"< {tighter}={hi:.6g} (slack {slack:.3g})")
    return issues


def check_impl_order(values: dict, gaps: dict, bound: str) -> list[str]:
    """(H3): for one bound, cost(tangent) >= cost(pwl) >= cost(exact).

    tangent and pwl are documented as safe *inner* approximations, so their
    feasible sets are subsets of the exact one and their optima can only be worse.
    tangent >= pwl is expected rather than guaranteed: with pwl_points=1 the single
    segment's midpoint coincides with the default tangent_ref=0.5 and the two
    encodings collapse onto each other.
    """
    issues = []
    present = [im for im in IMPLS_ORDER if (bound, im) in values
               and isinstance(values[(bound, im)], float)
               and not math.isnan(values[(bound, im)])]
    for looser, tighter in zip(present, present[1:]):
        lo, hi = values[(bound, looser)], values[(bound, tighter)]
        if math.isinf(lo):                      # inner approx infeasible: consistent
            continue
        if math.isinf(hi):
            issues.append(f"VIOLATION {bound}: {looser}={lo:.6g} feasible but "
                          f"{tighter}=infeasible -- an inner approximation cannot "
                          f"be feasible where the exact encoding is not")
            continue
        g = max(gaps.get((bound, looser)) or 0.0, gaps.get((bound, tighter)) or 0.0)
        slack = max(1e-6, g * max(abs(lo), abs(hi)))
        if hi > lo + slack:
            issues.append(f"VIOLATION {bound}: {looser}={lo:.6g} < "
                          f"{tighter}={hi:.6g} (excess {hi - lo:.3g} > slack "
                          f"{slack:.3g}) -- an inner approximation came out "
                          f"CHEAPER than the exact encoding, which means it is not "
                          f"actually inner")
        elif hi > lo + 1e-9:
            issues.append(f"within-gap inversion {bound}: {looser}={lo:.6g} < "
                          f"{tighter}={hi:.6g} (slack {slack:.3g})")
    return issues


def check_failure_order(failed_at: dict) -> list[str]:
    """(H2): a looser bound must fail no later than a tighter one.

    `failed_at[bound]` is the ladder index of the first infeasible rung, or None
    if the bound survived the whole ladder (treated as +inf).
    """
    issues = []
    present = [b for b in BOUNDS_ORDER if b in failed_at]
    idx = {b: (math.inf if failed_at[b] is None else failed_at[b]) for b in present}
    for looser, tighter in zip(present, present[1:]):
        if idx[looser] > idx[tighter]:
            issues.append(f"VIOLATION {looser} survived to rung {idx[looser]} but "
                          f"{tighter} already failed at rung {idx[tighter]}")
    return issues


# ===========================================================================
# Tests
# ===========================================================================
def test_analytic(sc: Scenario, opts, run: TestRun, sweeps: dict) -> tuple[list, dict]:
    """Solver-free: is the parameterisation in a regime where (H1)/(H2) can hold,
    and are the bounds ordered by n_max?"""
    lines = [f"base case: {sc.label()}", ""]
    d = sc.descriptors()
    lines.append("reference-mission descriptors (severity 1, all from one "
                 "scaled-Bernoulli increment):")
    lines.append(f"  p={sc.p}  b_ref={sc.b_ref:.6g}  mu={sc.p * sc.b_ref:.6g}  "
                 f"v={sc.p * (1 - sc.p) * sc.b_ref ** 2:.6g}  "
                 f"support={sc.b_ref:.6g}")
    lines.append(f"  s_chernoff={d['s_chernoff']:.6g}  "
                 f"(must exceed ln(1/eps)/tau = {sc.Le / sc.tau:.6g})")
    lines.append(f"  per-mission mu={np.round(d['mu'], 6).tolist()}")
    lines.append(f"  per-mission v ={np.round(d['v'], 8).tolist()}")
    lines.append(f"costs: C_M={sc.C_M} C_R={sc.C_R} C_S={sc.C_S} (read as C_D) "
                 f"C_P={sc.C_P} (inert in the current objective)")
    lines.append("")

    lines.append("regime checks (design note 5):")
    for key, val in regime_checks(sc).items():
        lines.append(f"  {key}: {val}")
    lines.append("")

    lines.append("analytic tightness n_max (larger = tighter = cheaper expected):")
    vals = {}
    for bound in BOUNDS_ORDER:
        vals[bound] = n_max(sc, bound)
        lines.append(f"  {bound:<10s} n_max = {vals[bound]:8.3f}   "
                     f"(integer missions: {math.floor(vals[bound]):d})")
    order_ok = all(vals[a] <= vals[b] + 1e-9
                   for a, b in zip(BOUNDS_ORDER, BOUNDS_ORDER[1:]))
    lines.append("")
    lines.append(f"n_max ordered as (H1) predicts: {order_ok}")
    if not order_ok:
        for a, b in zip(BOUNDS_ORDER, BOUNDS_ORDER[1:]):
            if vals[a] > vals[b] + 1e-9:
                lines.append(f"  ORDER BREAK: {a} (n_max {vals[a]:.3f}) is tighter "
                             f"than {b} (n_max {vals[b]:.3f}) -- expect the same "
                             f"inversion in the MILP costs; this is a regime "
                             f"issue, not necessarily a code bug")

    load = sc.load
    lines.append("")
    lines.append(f"mission load per vehicle over the horizon ~ T*M/F = {load:.2f}"
                 f"   (a bound bites only when n_max < load)")
    slack = [b for b in BOUNDS_ORDER if vals[b] > load]
    for bound in slack:
        if bound == BOUNDS_ORDER[-1]:
            lines.append(f"  OK {bound} (tightest): n_max {vals[bound]:.2f} > load "
                         f"{load:.2f} -> never binds, so it simply attains the "
                         f"unconstrained optimum. Still the cheapest run.")
        else:
            nxt = BOUNDS_ORDER[BOUNDS_ORDER.index(bound) + 1]
            lines.append(f"  WARNING {bound}: n_max {vals[bound]:.2f} > load "
                         f"{load:.2f} -> does not bind, so its cost will TIE with "
                         f"{nxt} and the pair carries no information. A tie still "
                         f"satisfies '>=', so this shows up as a pass, not a "
                         f"failure -- raise H (or M/F) until load exceeds "
                         f"n_max({nxt}).")
    if len(slack) <= 1:
        lines.append(f"  design point discriminates all {len(BOUNDS_ORDER)} bounds.")

    lines.append("")
    floor = survival_floor(sc)
    lines.append(f"predicted failure order (H2).  Survival floor = s_max/rho = "
                 f"{floor:.3f} reference missions: a bound with n_max below it "
                 f"admits no schedule at all, however often you repair.")
    for bound in BOUNDS_ORDER:
        margin = vals[bound] - floor
        lines.append(f"  {bound:<10s} n_max {vals[bound]:8.3f}  margin "
                     f"{margin:+8.3f}  feasible_hint={feasible_hint(sc, bound)}")

    if _plots_enabled(opts):
        path = plot_analytic(sc, run, sweeps)
        if path:
            print(f"  [plot] {path.name}")
            lines.append(f"plot: {path.name}")
    return lines, {"n_max": {b: float(vals[b]) for b in BOUNDS_ORDER},
                   "n_max_ordered": bool(order_ok),
                   "load": float(load),
                   "survival_floor": float(survival_floor(sc)),
                   "feasible_hint": {b: bool(feasible_hint(sc, b))
                                     for b in BOUNDS_ORDER},
                   "regime_checks": _to_builtin(regime_checks(sc))}


def test_base(sc: Scenario, opts, run: TestRun, impls) -> tuple[list, dict]:
    """Solve the base case for every (bound, implementation) and tabulate."""
    print(f"\n[base] {sc.label()}")
    combos = bound_impl_combos(impls, announce=True)
    vals, gaps, times, lines = {}, {}, {}, [f"base case: {sc.label()}", ""]
    header = (f"{'bound':<10s} {'impl':<9s} {'status':<12s} {'cost':>12s} "
              f"{'gap':>9s} {'time[s]':>9s} {'repairs':>8s} {'depot':>7s} "
              f"{'mu_max':>9s} {'n_max':>8s}")
    lines += [header, "-" * len(header)]
    for bound, impl in combos:
        if not _mine(opts):
            continue
        variant = sc.variant(reliability_impl=impl)
        print(f"  solving {bound}/{impl} ...", flush=True)
        rec, data = run_case(variant, bound, opts)
        rec.update({"test": "base", "parameter": "-", "value": ""})
        run.add(rec, data, variant)
        vals[(bound, impl)] = rec.get("objective", math.nan)
        gaps[(bound, impl)] = rec.get("mip_gap")
        times[(bound, impl)] = rec.get("runtime_s", math.nan)
        lines.append(f"{bound:<10s} {impl:<9s} {str(rec.get('status'))[:12]:<12s} "
                     f"{_fmt(rec.get('objective')):>12s} "
                     f"{_fmt(rec.get('mip_gap'), 3):>9s} "
                     f"{_fmt(rec.get('runtime_s'), 2):>9s} "
                     f"{_fmt(rec.get('n_repairs'), 0):>8s} "
                     f"{_fmt(rec.get('n_depot'), 0):>7s} "
                     f"{_fmt(rec.get('mu_max'), 4):>9s} "
                     f"{rec.get('n_max_analytic', float('nan')):8.2f}")
    lines.append("")

    if getattr(opts, "shard_obj", None) is not None:
        lines.append(f"shard {opts.shard_obj}: {len(run.rows)} run(s) only -- the "
                     f"hypothesis checks need every bound, so run --merge when all "
                     f"array tasks are done.")
        return lines, {"shard": str(opts.shard_obj), "n_runs": len(run.rows)}

    # (H1) per implementation: the bound ordering must hold within each encoding
    summary = {"H1": {}, "H3": {}}
    all_ok = True
    for impl in impls:
        per = {b: cost_for(vals, b, impl) for b in BOUNDS_ORDER}
        per_gaps = {b: gaps.get((b, impl), gaps.get((b, "exact"))) for b in BOUNDS_ORDER}
        issues = check_order(per, per_gaps)
        holds = not any(i.startswith("VIOLATION") for i in issues)
        all_ok = all_ok and holds
        lines.append(f"hypothesis (H1) with impl={impl}: "
                     f"{'HOLDS' if holds else 'VIOLATED'}")
        lines += [f"  {i}" for i in issues]
        summary["H1"][impl] = {"holds": bool(holds), "issues": issues,
                               "costs": {b: _to_builtin(per[b]) for b in per}}
    lines.append("")

    # (H3) per bound: inner approximations cannot be cheaper than exact
    for bound in BOUNDS_ORDER:
        have = [im for im in IMPLS_ORDER if (bound, im) in vals]
        if len(have) < 2:
            continue
        issues = check_impl_order(vals, gaps, bound)
        holds = not any(i.startswith("VIOLATION") for i in issues)
        costs = "  ".join(f"{im}={_fmt(vals[(bound, im)])}" for im in have)
        tms = "  ".join(f"{im}={_fmt(times[(bound, im)], 2)}s" for im in have)
        lines.append(f"hypothesis (H3) {bound}: {'HOLDS' if holds else 'VIOLATED'}"
                     f"   {costs}")
        lines.append(f"    price of linearisation, solve time: {tms}")
        lines += [f"    {i}" for i in issues]
        summary["H3"][bound] = {"holds": bool(holds), "issues": issues,
                                "costs": {im: _to_builtin(vals[(bound, im)])
                                          for im in have},
                                "times": {im: _to_builtin(times[(bound, im)])
                                          for im in have}}
    if _uninformative(run.rows):
        lines.append("  WARNING all runs produced the same repair/depot pattern "
                     "-- the reliability constraint is not binding, so this run "
                     "cannot discriminate between bounds or implementations. "
                     "Lower C_S (design note 3) or raise H (design note 8).")
    summary["H1_all_impls_hold"] = bool(all_ok)
    return lines, summary


def test_sweep(sc: Scenario, opts, run: TestRun, sweeps: dict,
               impls) -> tuple[list, dict]:
    """(H1): one parameter at a time, every (bound, impl), cost + time."""
    lines = [f"base case: {sc.label()}", f"implementations: {list(impls)}", ""]
    combos = bound_impl_combos(impls, announce=True)
    verdicts = {}
    for param, values in sweeps.items():
        print(f"\n[sweep] {param} over {values}")
        lines.append(f"=== {param} ===")
        for v in values:
            base_variant = sc.variant(**{param: _cast(param, v)})
            if base_variant.F <= base_variant.M:
                print(f"  skip {param}={v}: needs F > M "
                      f"(F={base_variant.F}, M={base_variant.M})")
                lines.append(f"  {param}={v}: skipped (needs F > M)")
                continue
            vals, gaps = {}, {}
            for bound, impl in combos:
                if not _mine(opts):
                    continue
                variant = base_variant.variant(reliability_impl=impl)
                print(f"  {param}={v} {bound}/{impl} ...", flush=True)
                rec, data = run_case(variant, bound, opts)
                rec.update({"test": "sweep", "parameter": param, "value": v})
                run.add(rec, data, variant)
                vals[(bound, impl)] = rec.get("objective", math.nan)
                gaps[(bound, impl)] = rec.get("mip_gap")
                print(f"      cost={_fmt(rec.get('objective'))} "
                      f"time={_fmt(rec.get('runtime_s'), 2)}s "
                      f"status={rec.get('status')}")
            if getattr(opts, "shard_obj", None) is not None:
                continue                         # checks happen in --merge
            for impl in impls:
                per = {b: cost_for(vals, b, impl) for b in BOUNDS_ORDER}
                per_gaps = {b: gaps.get((b, impl), gaps.get((b, "exact")))
                            for b in BOUNDS_ORDER}
                issues = check_order(per, per_gaps)
                holds = not any(i.startswith("VIOLATION") for i in issues)
                verdicts[f"{param}={v}/{impl}"] = {
                    "H1_holds": bool(holds), "issues": issues,
                    "costs": {b: _to_builtin(per[b]) for b in per}}
                costs = "  ".join(f"{b}={_fmt(per[b])}" for b in BOUNDS_ORDER)
                lines.append(f"  {param}={v} impl={impl}: (H1) "
                             f"{'HOLDS' if holds else 'VIOLATED'}   {costs}")
                lines += [f"      {i}" for i in issues]
            for bound in BOUNDS_ORDER:
                if sum(1 for im in IMPLS_ORDER if (bound, im) in vals) < 2:
                    continue
                issues = check_impl_order(vals, gaps, bound)
                if issues:
                    lines += [f"      (H3) {i}" for i in issues]
        lines.append("")

    if _plots_enabled(opts):
        for param in sweeps:
            path = plot_parameter(run.rows, param, run, getattr(sc, param))
            if path:
                print(f"  [plot] {path.name}")
                lines.append(f"plot: {path.name}")
    if getattr(opts, "shard_obj", None) is not None:
        lines.append(f"shard {opts.shard_obj}: {len(run.rows)} run(s); (H1) is "
                     f"evaluated by --merge once every shard has finished.")
        return lines, {"shard": str(opts.shard_obj), "n_runs": len(run.rows)}
    n_bad = sum(1 for v in verdicts.values() if not v["H1_holds"])
    lines.append(f"(H1) held at {len(verdicts) - n_bad}/{len(verdicts)} "
                 f"(design point, implementation) combinations.")
    return lines, {"H1_violations": n_bad, "implementations": list(impls),
                   "points": verdicts}


def test_failure(sc: Scenario, opts, run: TestRun, ladders: dict,
                 impls) -> tuple[list, dict]:
    """(H2): how far along each stress ladder does each bound stay feasible?

    Each ladder runs mild -> harsh.  For a bound we walk it until the model comes
    back infeasible; that rung is the bound's failure point.  A looser bound
    should fail at the same rung or earlier.  Runs are stopped after the first
    failure (--no-early-stop keeps going, which is slower but confirms that
    infeasibility is monotone in the stress -- worth doing once).
    """
    lines = [f"base case: {sc.label()}", f"implementations: {list(impls)}", "",
             "each ladder runs MILD -> HARSH; a (bound, impl) failure point is "
             "the first rung that comes back infeasible", ""]
    combos = bound_impl_combos(impls, announce=True)
    summary = {}
    for param, (direction, ladder) in ladders.items():
        print(f"\n[failure] {param} ({direction}) over {ladder}")
        lines.append(f"=== {param}  ({direction}) ===")
        lines.append(f"  ladder: {ladder}")
        failed_at, failed_val, notes = {}, {}, []
        n_runs = {key: 0 for key in combos}
        for bound, impl in combos:
            key = (bound, impl)
            failed_at[key] = None
            # A ladder walk is the shard unit: the early-stop logic needs the
            # rungs of one (bound, impl) in order, in one process.
            if not _mine(opts):
                del failed_at[key]
                continue
            for idx, v in enumerate(ladder):
                variant = sc.variant(**{param: _cast(param, v)},
                                     reliability_impl=impl)
                if variant.F <= variant.M:
                    notes.append(f"{param}={v}: skipped (needs F > M)")
                    continue
                print(f"  {bound}/{impl} {param}={v} ...", flush=True)
                rec, data = run_case(variant, bound, opts)
                rec.update({"test": "failure", "parameter": param, "value": v})
                run.add(rec, data, variant)
                n_runs[key] += 1
                verdict = rec["verdict"]
                print(f"      {verdict} (status={rec.get('status')}, "
                      f"cost={_fmt(rec.get('objective'))})")
                if verdict == "infeasible":
                    failed_at[key], failed_val[key] = idx, v
                    kind = ("capacity (feasible_hint=False)"
                            if not rec.get("feasible_hint") else "reliability")
                    notes.append(f"{bound}/{impl}: first infeasible at {param}={v} "
                                 f"(rung {idx}) -- likely {kind} failure")
                    if not opts.no_early_stop:
                        break
                elif verdict == "unknown" and not opts.dry_run:
                    tag = f"{bound}/{impl}: inconclusive"
                    if not any(n.startswith(tag) for n in notes):
                        notes.append(f"{tag} at {param}={v} "
                                     f"(status={rec.get('status')}) -- not counted "
                                     f"as a failure, so the threshold below may be "
                                     f"an underestimate; rerun with "
                                     f"--no-time-limit")
        for bound, impl in combos:
            key = (bound, impl)
            if key not in failed_at:
                continue                         # not this shard's ladder
            tag = f"{bound}/{impl}"
            if n_runs[key] == 0:
                lines.append(f"  {tag:<20s} no valid rung on this ladder "
                             f"(every value skipped)")
            elif failed_at[key] is None:
                lines.append(f"  {tag:<20s} survived the whole ladder")
            else:
                lines.append(f"  {tag:<20s} failed at {param}="
                             f"{failed_val[key]:g} (rung {failed_at[key]} "
                             f"of {len(ladder) - 1})")
        if getattr(opts, "shard_obj", None) is not None:
            summary[param] = {
                "direction": direction, "ladder": [_to_builtin(v) for v in ladder],
                "shard": str(opts.shard_obj),
                "failed_at_index": {f"{b}/{im}": failed_at[(b, im)]
                                    for (b, im) in failed_at},
                "failed_at_value": {f"{b}/{im}": _to_builtin(failed_val.get((b, im)))
                                    for (b, im) in failed_at},
                "notes": notes}
            lines.append(f"  shard {opts.shard_obj}: partial ladder -- run --merge "
                         f"for the (H2) verdict")
            lines.append("")
            continue
        all_issues, holds = [], True
        for impl in impls:                       # (H2) within each encoding
            per = {b: failed_at.get((b, impl), failed_at.get((b, "exact")))
                   for b in BOUNDS_ORDER
                   if (b, impl) in failed_at or (b, "exact") in failed_at}
            issues = check_failure_order(per)
            lines.append(f"  hypothesis (H2) on {param}, impl={impl}: "
                         f"{'HOLDS' if not issues else 'VIOLATED'}")
            lines += [f"      {i}" for i in issues]
            all_issues += issues
            holds = holds and not issues
        for bound in BOUNDS_ORDER:               # (H3) inner approx fails first
            have = [im for im in IMPLS_ORDER if (bound, im) in failed_at]
            if len(have) < 2:
                continue
            idx = {im: (math.inf if failed_at[(bound, im)] is None
                        else failed_at[(bound, im)]) for im in have}
            for looser, tighter in zip(have, have[1:]):
                if idx[looser] > idx[tighter]:
                    msg = (f"(H3) VIOLATION {bound}: {looser} survived to rung "
                           f"{idx[looser]} but {tighter} failed at rung "
                           f"{idx[tighter]} -- an inner approximation outlasted "
                           f"the exact encoding")
                    lines.append(f"      {msg}")
                    all_issues.append(msg)
                    holds = False
        lines += [f"      note: {n}" for n in notes]
        lines.append("")
        summary[param] = {
            "direction": direction, "ladder": [_to_builtin(v) for v in ladder],
            "failed_at_index": {f"{b}/{im}": failed_at[(b, im)]
                                for (b, im) in failed_at},
            "n_runs": {f"{b}/{im}": n_runs[(b, im)] for (b, im) in n_runs},
            "failed_at_value": {f"{b}/{im}": _to_builtin(failed_val.get((b, im)))
                                for (b, im) in failed_at},
            "H2_holds": bool(holds), "issues": all_issues, "notes": notes,
        }

    if _plots_enabled(opts):
        path = plot_failure(summary, run)
        if path:
            print(f"  [plot] {path.name}")
            lines.append(f"plot: {path.name}")
        for param in ladders:
            path = plot_parameter(run.rows, param, run, getattr(sc, param, None))
            if path:
                print(f"  [plot] {path.name}")
                lines.append(f"plot: {path.name}")
    if getattr(opts, "shard_obj", None) is not None:
        lines.append(f"shard {opts.shard_obj}: {len(run.rows)} run(s); (H2) is "
                     f"evaluated by --merge once every shard has finished.")
        return lines, {"shard": str(opts.shard_obj), "n_runs": len(run.rows),
                       "ladders": summary}
    n_bad = sum(1 for v in summary.values() if not v["H2_holds"])
    lines.append(f"(H2) held on {len(summary) - n_bad}/{len(summary)} ladders.")
    return lines, {"H2_violations": n_bad, "ladders": summary}


def test_impl(sc: Scenario, opts, run: TestRun, impls, pwl_ladder) -> tuple[list, dict]:
    """(H3): what does linearising the reliability constraint cost, and buy?

    `tangent` and `pwl` are documented safe *inner* approximations of the exact
    nonconvex quadratic: their feasible sets are subsets of it, so their optima
    can only be more expensive, while the model becomes a MILP instead of a
    MIQCP and should solve faster.  This test quantifies both halves of that
    trade at one design point, and (with --pwl-ladder) shows pwl converging to
    exact as the segment count grows.

    markov and chernoff are already linear and fall back to 'exact', so they
    appear once with no comparison to make.
    """
    lines = [f"base case: {sc.label()}", f"implementations: {list(impls)}",
             f"tangent_ref={sc.tangent_ref}  pwl_points={sc.pwl_points}", ""]
    combos = bound_impl_combos(impls, announce=True)
    vals, gaps, times = {}, {}, {}
    header = (f"{'bound':<10s} {'impl':<9s} {'status':<12s} {'cost':>12s} "
              f"{'gap':>9s} {'time[s]':>9s} {'vars':>8s} {'constrs':>9s}")
    lines += [header, "-" * len(header)]
    for bound, impl in combos:
        if not _mine(opts):
            continue
        variant = sc.variant(reliability_impl=impl)
        print(f"  [impl] {bound}/{impl} ...", flush=True)
        rec, data = run_case(variant, bound, opts)
        rec.update({"test": "impl", "parameter": "impl", "value": impl})
        run.add(rec, data, variant)
        vals[(bound, impl)] = rec.get("objective", math.nan)
        gaps[(bound, impl)] = rec.get("mip_gap")
        times[(bound, impl)] = rec.get("runtime_s", math.nan)
        lines.append(f"{bound:<10s} {impl:<9s} {str(rec.get('status'))[:12]:<12s} "
                     f"{_fmt(rec.get('objective')):>12s} "
                     f"{_fmt(rec.get('mip_gap'), 3):>9s} "
                     f"{_fmt(rec.get('runtime_s'), 2):>9s} "
                     f"{_fmt(rec.get('n_vars'), 0):>8s} "
                     f"{_fmt(rec.get('n_constrs'), 0):>9s}")
    lines.append("")

    summary = {"H3": {}, "pwl_convergence": {}}
    for bound in BOUNDS_ORDER:
        have = [im for im in IMPLS_ORDER if (bound, im) in vals]
        if len(have) < 2:
            if have:
                lines.append(f"{bound}: only '{have[0]}' applies "
                             f"(linear bound, impl is ignored)")
            continue
        issues = check_impl_order(vals, gaps, bound)
        holds = not any(i.startswith("VIOLATION") for i in issues)
        costs = "  ".join(f"{im}={_fmt(vals[(bound, im)])}" for im in have)
        tms = "  ".join(f"{im}={_fmt(times[(bound, im)], 2)}s" for im in have)
        lines.append(f"(H3) {bound}: {'HOLDS' if holds else 'VIOLATED'}   {costs}")
        lines.append(f"     solve time: {tms}")
        ex, tg = vals.get((bound, "exact")), vals.get((bound, "tangent"))
        if all(isinstance(q, float) and math.isfinite(q) and q != 0 for q in (ex, tg)):
            lines.append(f"     optimality price of the single tangent: "
                         f"{100.0 * (tg - ex) / abs(ex):+.2f} %")
        lines += [f"     {i}" for i in issues]
        summary["H3"][bound] = {
            "holds": bool(holds), "issues": issues,
            "costs": {im: _to_builtin(vals[(bound, im)]) for im in have},
            "times": {im: _to_builtin(times[(bound, im)]) for im in have}}
    lines.append("")

    # ---- pwl_points ladder ------------------------------------------------
    if pwl_ladder:
        lines.append(f"=== pwl_points ladder {pwl_ladder} ===")
        for bound in BOUNDS_ORDER:
            if bound not in IMPL_AWARE_BOUNDS:
                continue
            series = []
            for n in pwl_ladder:
                if not _mine(opts):
                    continue
                variant = sc.variant(reliability_impl="pwl", pwl_points=int(n))
                print(f"  [impl] {bound}/pwl({n}) ...", flush=True)
                rec, data = run_case(variant, bound, opts)
                rec.update({"test": "impl", "parameter": "pwl_points",
                            "value": int(n)})
                run.add(rec, data, variant)
                series.append((int(n), rec.get("objective", math.nan),
                               rec.get("runtime_s", math.nan)))
            txt = "  ".join(f"{n}:{_fmt(c)}" for n, c, _ in series)
            lines.append(f"  {bound}: {txt}")
            ex = vals.get((bound, "exact"), math.nan)
            finite = [(n, c) for n, c, _ in series
                      if isinstance(c, float) and math.isfinite(c)]
            mono = all(b[1] <= a[1] + 1e-6 for a, b in zip(finite, finite[1:]))
            lines.append(f"     exact={_fmt(ex)}   monotone non-increasing in "
                         f"segments: {mono}")
            if not mono:
                lines.append("     NOTE more segments should never cost more; an "
                             "increase points at MIP-gap noise (lower --mip-gap) "
                             "rather than at the encoding.")
            summary["pwl_convergence"][bound] = {
                "exact": _to_builtin(ex), "monotone": bool(mono),
                "series": [{"pwl_points": n, "cost": _to_builtin(c),
                            "runtime_s": _to_builtin(t)} for n, c, t in series]}
        lines.append("")

    if _plots_enabled(opts):
        for fn in (plot_impl, plot_pwl_convergence):
            path = fn(run.rows, run)
            if path:
                print(f"  [plot] {path.name}")
                lines.append(f"plot: {path.name}")
    if getattr(opts, "shard_obj", None) is not None:
        lines.append(f"shard {opts.shard_obj}: {len(run.rows)} run(s); (H3) is "
                     f"evaluated by --merge once every shard has finished.")
        return lines, {"shard": str(opts.shard_obj), "n_runs": len(run.rows)}
    n_bad = sum(1 for v in summary["H3"].values() if not v["holds"])
    lines.append(f"(H3) held for {len(summary['H3']) - n_bad}/"
                 f"{len(summary['H3'])} impl-aware bounds.")
    summary["H3_violations"] = n_bad
    return lines, summary


def _uninformative(rows: list[dict]) -> bool:
    """True when every bound gave the same intervention pattern."""
    sigs = {(r.get("n_repairs"), r.get("n_depot"), r.get("n_replacements"))
            for r in rows if r.get("status") == "optimal"}
    return len(sigs) == 1 and len(rows) > 1


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value):
            return "-"
        if math.isinf(value):
            return "inf"
        return f"{value:.{digits}f}"
    return str(value)


# ===========================================================================
# Merge: reduce step for Slurm job arrays
# ===========================================================================
_NUM_FIELDS = ("objective", "mip_gap", "obj_bound", "runtime_s", "wall_s",
               "n_max_analytic", "load", "n_repairs", "n_replacements",
               "n_depot", "mu_max", "v_max", "n_vars", "n_constrs", "n_bin",
               "nodes", "tau", "epsilon", "rho", "p", "b_ref", "mu_ref",
               "s_chernoff", "tangent_ref", "C_M", "C_R", "C_S", "C_P")
_INT_FIELDS = ("F", "M", "L", "H", "T", "pwl_points")


def _read_rows(csv_path: Path) -> list:
    """Read a results CSV back into records with numbers as numbers."""
    out = []
    with csv_path.open(newline="") as fh:
        for raw in csv.DictReader(fh):
            rec = dict(raw)
            for key in _NUM_FIELDS:
                rec[key] = _parse_float(rec.get(key))
            for key in _INT_FIELDS:
                try:
                    rec[key] = int(float(rec[key]))
                except (TypeError, ValueError):
                    rec[key] = None
            val = rec.get("value")
            if val not in ("", None, "-"):
                param = rec.get("parameter", "")
                try:
                    rec["value"] = _cast(param, val) if param in SWEEP_PARAMS + \
                        tuple(STRESS_LADDERS) else val
                except ValueError:
                    pass                          # impl names stay strings
            rec["feasible_hint"] = str(rec.get("feasible_hint", "")).lower() == "true"
            out.append(rec)
    return out


def _parse_float(text):
    if text in (None, "", "-"):
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def merge_shards(out_root: Path, name: str, test: str, opts) -> int:
    """Combine every shard folder of one test, then run the checks and plots once.

    A shard only ever holds a slice of the design points, so no shard can evaluate
    (H1)/(H2)/(H3) -- those need every bound at a point. This reads all shard CSVs
    back, concatenates them, and produces one merged folder with the verdicts.
    """
    pattern = f"*_{name}_{test}*"
    csvs = sorted(q for d in out_root.glob(pattern) if d.is_dir()
                  for q in d.glob("*results*.csv"))
    csvs = [q for q in csvs if "_merged" not in q.parent.name]
    if not csvs:
        print(f"[merge] no shard results under {out_root}/{pattern}")
        return 1
    rows: list = []
    for q in csvs:
        got = _read_rows(q)
        print(f"[merge] {q.parent.name}/{q.name}: {len(got)} rows")
        rows += got
    # a shard may have been resubmitted: keep the newest row per unique run
    unique = {}
    for rec in rows:
        key = (rec.get("test"), rec.get("parameter"), str(rec.get("value")),
               rec.get("bound"), rec.get("reliability_impl"),
               rec.get("pwl_points"))
        prev = unique.get(key)
        if prev is None or str(rec.get("timestamp", "")) >= str(prev.get("timestamp", "")):
            unique[key] = rec
    rows = list(unique.values())
    dropped = len(csvs) and (len(unique) != len(rows))

    stamp = f"{datetime.now():%y%m%d}"
    mdir = out_root / f"{stamp}_{name}_{test}_merged"
    mdir.mkdir(parents=True, exist_ok=True)
    stem = f"{stamp}_{name}_{test}_merged"

    # duck-types TestRun for the plot helpers (dir / stem / rows)
    run = SimpleNamespace(dir=mdir, stem=stem, rows=rows)

    csv_path = mdir / f"{stamp}_results_{name}_{test}_merged.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for rec in sorted(rows, key=lambda r: (str(r.get("parameter")),
                                               str(r.get("value")),
                                               BOUNDS_ORDER.index(r["bound"])
                                               if r.get("bound") in BOUNDS_ORDER else 99)):
            w.writerow({k: rec.get(k, "") for k in FIELDS})

    report = [f"# merged report  {datetime.now():%Y-%m-%d %H:%M:%S}",
              f"# test={test}  shards merged={len(csvs)}  rows={len(rows)}",
              f"# bounds: {list(BOUNDS_ORDER)}", ""]
    statuses = {}
    for rec in rows:
        statuses[str(rec.get("status"))] = statuses.get(str(rec.get("status")), 0) + 1
    report.append(f"status counts: {statuses}")
    versions = {(r.get("git_branch"), r.get("git_commit")) for r in rows
                if r.get("git_commit")}
    if len(versions) > 1:
        report.append(f"WARNING these shards were produced by {len(versions)} "
                      f"different code versions: "
                      f"{sorted(f'{b}@{c}' for b, c in versions)}. Comparing them "
                      f"is only valid if the change could not affect the model -- "
                      f"do not edit or 'git pull' while an array is running.")
    elif versions:
        b, c = next(iter(versions))
        report.append(f"code: branch={b} commit={c}")
    missing = [r for r in rows if str(r.get("status")).startswith("error")
               or classify(r) == "unknown"]
    if missing:
        report.append(f"WARNING {len(missing)} run(s) did not produce a solution "
                      f"(errors or time limits) -- the checks below treat those as "
                      f"missing, not as failures. Look for status=time_limit and "
                      f"resubmit with --no-time-limit.")
    report.append("")

    summary = {"n_rows": len(rows), "shards": len(csvs), "status_counts": statuses,
               "code_versions": sorted(f"{b}@{c}" for b, c in versions)}

    # ---- (H1)/(H3) per design point -------------------------------------
    impls_seen = [im for im in IMPLS_ORDER
                  if any(impl_of_record(r) == im for r in rows)]
    points = {}
    n_bad = 0
    groups = {}
    for rec in rows:
        if rec.get("test") == "failure":
            continue
        groups.setdefault((rec.get("parameter"), str(rec.get("value"))), []).append(rec)
    for (param, value), recs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        vals = {(r["bound"], impl_of_record(r)): r["objective"] for r in recs}
        gaps = {(r["bound"], impl_of_record(r)): r["mip_gap"] for r in recs}
        for impl in impls_seen:
            per = {b: cost_for(vals, b, impl) for b in BOUNDS_ORDER}
            per_gaps = {b: gaps.get((b, impl), gaps.get((b, "exact")))
                        for b in BOUNDS_ORDER}
            issues = check_order(per, per_gaps)
            holds = not any(i.startswith("VIOLATION") for i in issues)
            n_bad += 0 if holds else 1
            costs = "  ".join(f"{b}={_fmt(per[b])}" for b in BOUNDS_ORDER)
            report.append(f"{param}={value} impl={impl}: (H1) "
                          f"{'HOLDS' if holds else 'VIOLATED'}   {costs}")
            report += [f"    {i}" for i in issues]
            points[f"{param}={value}/{impl}"] = {"H1_holds": bool(holds),
                                                 "issues": issues}
        for bound in BOUNDS_ORDER:
            if sum(1 for im in IMPLS_ORDER if (bound, im) in vals) < 2:
                continue
            issues = check_impl_order(vals, gaps, bound)
            report += [f"    (H3) {i}" for i in issues]
    summary["H1_violations"] = n_bad
    summary["points"] = points
    report.append("")

    # ---- (H2) from the failure rows -------------------------------------
    fail_rows = [r for r in rows if r.get("test") == "failure"]
    if fail_rows:
        ladders = {}
        for rec in fail_rows:
            ladders.setdefault(rec["parameter"], []).append(rec)
        for param, recs in ladders.items():
            order = sorted({r["value"] for r in recs},
                           key=lambda v: [str(x) for x in
                                          (STRESS_LADDERS.get(param, ("", []))[1]
                                           or sorted({q["value"] for q in recs}))].index(str(v))
                           if str(v) in [str(x) for x in
                                         (STRESS_LADDERS.get(param, ("", []))[1] or [])]
                           else 10**6)
            failed_at = {}
            for rec in recs:
                key = (rec["bound"], impl_of_record(rec))
                failed_at.setdefault(key, None)
                if classify(rec) == "infeasible":
                    idx = order.index(rec["value"]) if rec["value"] in order else None
                    cur = failed_at[key]
                    if idx is not None and (cur is None or idx < cur):
                        failed_at[key] = idx
            report.append(f"=== (H2) {param} ladder {order} ===")
            for (bound, impl), idx in sorted(
                    failed_at.items(),
                    key=lambda kv: (BOUNDS_ORDER.index(kv[0][0]),
                                    IMPLS_ORDER.index(kv[0][1]))):
                where = "survived" if idx is None else f"failed at {order[idx]}"
                report.append(f"  {bound}/{impl:<8s} {where}")
            all_issues = []
            for impl in impls_seen:
                per = {b: failed_at.get((b, impl), failed_at.get((b, "exact")))
                       for b in BOUNDS_ORDER
                       if (b, impl) in failed_at or (b, "exact") in failed_at}
                issues = check_failure_order(per)
                report.append(f"  (H2) impl={impl}: "
                              f"{'HOLDS' if not issues else 'VIOLATED'}")
                report += [f"      {i}" for i in issues]
                all_issues += issues
            summary.setdefault("H2", {})[param] = {
                "ladder": [_to_builtin(v) for v in order],
                "failed_at_index": {f"{b}/{im}": failed_at[(b, im)]
                                    for (b, im) in failed_at},
                "issues": all_issues}
            report.append("")

    # ---- plots from the merged rows --------------------------------------
    if _plots_enabled(opts):
        for param in sorted({str(r.get("parameter")) for r in rows}
                            - {"-", "impl", "pwl_points", "None", ""}):
            path = plot_parameter(rows, param, run, None)
            if path:
                print(f"  [plot] {path.name}")
                report.append(f"plot: {path.name}")
        for fn in (plot_impl, plot_pwl_convergence):
            path = fn(rows, run)
            if path:
                print(f"  [plot] {path.name}")
                report.append(f"plot: {path.name}")

    _dump_yaml(mdir / f"{stamp}_results_{name}_{test}_merged.yaml",
               {"test": test, "merged_from": [str(q.parent.name) for q in csvs],
                "created": datetime.now().isoformat(timespec="seconds"),
                "summary": summary, "runs": [_to_builtin(r) for r in rows]})
    (mdir / f"{stamp}_summary_{name}_{test}_merged.txt").write_text("\n".join(report))
    print("\n".join(report))
    print(f"\n[merge] folder : {mdir}")
    print(f"[merge] results: {csv_path.name}")
    return 0


# ===========================================================================
# CLI
# ===========================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Bound-tightness and bound-failure tests for the reliability "
                    "constraints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Start with:  python test.py --tests analytic   (no Gurobi needed)")
    p.add_argument("--tests", default="analytic",
                   help="comma list of analytic,base,sweep,impl,failure "
                        "(default: analytic)")
    p.add_argument("--name", default="bound_tightness",
                   help="test name used in the folder and file names")
    p.add_argument("--out", default="test_results",
                   help="root output directory; each test gets its own folder")
    p.add_argument("--params", default="L,H,M,F",
                   help="parameters to sweep in the 'sweep' test (L,H,M,F)")
    p.add_argument("--values", default=None,
                   help="explicit sweep values, e.g. 'L=1,2,3;H=4,6,8,10'")
    p.add_argument("--failure-params", default="H,M,F,epsilon",
                   help=f"stress ladders for the 'failure' test, from "
                        f"{tuple(STRESS_LADDERS)}")
    p.add_argument("--failure-values", default=None,
                   help="explicit ladders, mild->harsh, e.g. 'epsilon=0.1,0.01,0.001'")
    p.add_argument("--no-early-stop", action="store_true",
                   help="in the failure test, keep going past the first "
                        "infeasible rung (confirms monotone failure)")
    p.add_argument("--bounds", default=",".join(BOUNDS_ORDER),
                   help="comma list of bounds to test, loosest first")
    # solver options
    p.add_argument("--mip-gap", type=float, default=1e-4,
                   help="MIP gap (default 1e-4; the model default 0.12 is far too "
                        "loose for this comparison)")
    p.add_argument("--time-limit", type=float, default=300.0,
                   help="per-solve time limit in seconds (default 300); "
                        "<= 0 means no limit")
    p.add_argument("--no-time-limit", action="store_true",
                   help="no wall-clock limit: every solve runs to --mip-gap. "
                        "Recommended for the failure test, so 'time_limit' can "
                        "never be mistaken for 'infeasible'.")
    p.add_argument("--threads", type=int, default=None,
                   help="Gurobi Threads. On a shared cluster node ALWAYS set this "
                        "to the cores Slurm gave you ($SLURM_CPUS_PER_TASK); "
                        "otherwise Gurobi threads for every core it can see.")
    p.add_argument("--shard", default=None, metavar="K/N",
                   help="run only work unit k of n (for Slurm job arrays, e.g. "
                        "--shard $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT). "
                        "Each shard writes its own folder; hypothesis checks are "
                        "skipped -- run --merge afterwards.")
    p.add_argument("--merge", action="store_true",
                   help="combine the shard folders of --tests under --out, run the "
                        "hypothesis checks and plots once, and write a _merged "
                        "folder. Solves nothing.")
    p.add_argument("--verbose", type=int, default=0, help="Gurobi output flag")
    p.add_argument("--dry-run", action="store_true",
                   help="build and validate every input, but do not solve")
    p.add_argument("--no-plots", action="store_true")
    # base case overrides
    p.add_argument("--F", type=int, default=None)
    p.add_argument("--M", type=int, default=None)
    p.add_argument("--L", type=int, default=None)
    p.add_argument("--H", type=int, default=None)
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--epsilon", type=float, default=None)
    p.add_argument("--rho", type=float, default=None)
    p.add_argument("--p", type=float, default=None,
                   help="Bernoulli probability of the damage increment")
    p.add_argument("--calibrate-n", type=float, default=None, dest="n_target",
                   help="place the design point where hoeffding binds after this "
                        "many reference missions (default 6)")
    p.add_argument("--C-M", type=float, default=None, dest="C_M")
    p.add_argument("--C-R", type=float, default=None, dest="C_R")
    p.add_argument("--C-S", type=float, default=None, dest="C_S",
                   help="damage regularisation (read as C_D); keep it small if the "
                        "reliability constraint stops binding")
    p.add_argument("--C-P", type=float, default=None, dest="C_P",
                   help="periodicity cost; accepted by config but inert in the "
                        "current objective")
    p.add_argument("--C-rep", type=float, default=None, dest="C_rep")
    p.add_argument("--repair", default=None, dest="repair_model",
                   choices=["ard1", "ardinf"],
                   help="ardinf (default) is the only model chernoff supports")
    p.add_argument("--impls", default=None,
                   help="implementations to test, comma separated, loosest "
                        f"first; pick from {IMPLS_ORDER} (default: exact). "
                        "Give several to add the implementation as a test "
                        "dimension, e.g. --impls tangent,pwl,exact")
    p.add_argument("--impl", default=None, dest="reliability_impl",
                   choices=["exact", "tangent", "pwl"],
                   help="shorthand for a single implementation (same as "
                        "--impls <value>)")
    p.add_argument("--pwl-points", type=int, default=None, dest="pwl_points",
                   help="segments used by reliability_impl='pwl' (default 8)")
    p.add_argument("--tangent-ref", type=float, default=None, dest="tangent_ref",
                   help="tangent taken at tangent_ref*tau (default 0.5)")
    p.add_argument("--pwl-ladder", default=None,
                   help="in the 'impl' test, also sweep pwl_points, e.g. "
                        "'2,4,8,16' -- shows pwl converging to exact")
    p.add_argument("--allow-replacement", action="store_true", default=None)

    args = p.parse_args(argv)
    if args.no_time_limit or (args.time_limit is not None and args.time_limit <= 0):
        args.time_limit = None

    # --impls is the general form; --impl is the single-value shorthand
    if args.impls:
        impls = [q.strip() for q in args.impls.split(",") if q.strip()]
    elif args.reliability_impl:
        impls = [args.reliability_impl]
    else:
        impls = ["exact"]
    for impl in impls:
        if impl not in IMPLS_ORDER:
            raise SystemExit(f"unknown implementation {impl!r}; "
                             f"pick from {IMPLS_ORDER}")
    # keep them in loosest -> exact order, which is what H3 is stated in
    args.impl_list = [im for im in IMPLS_ORDER if im in impls]
    # the scenario's own impl is the first one; each test overrides per run
    args.reliability_impl = args.impl_list[0]
    args.shard_obj = None
    if args.shard:
        try:
            k, n = (int(q) for q in args.shard.split("/"))
        except ValueError:
            raise SystemExit("--shard expects K/N, e.g. --shard 3/20")
        args.shard_obj = Shard(k, n)
    args.pwl_ladder_list = ([int(q) for q in args.pwl_ladder.split(",") if q.strip()]
                            if args.pwl_ladder else [])
    if args.pwl_ladder_list and "pwl" not in args.impl_list:
        args.impl_list = [im for im in IMPLS_ORDER
                          if im in set(args.impl_list) | {"pwl"}]
        print(f"[impl] --pwl-ladder given, so 'pwl' was added to the "
              f"implementations: {args.impl_list}")
    return args


def scenario_from_args(args) -> Scenario:
    keys = ("F", "M", "L", "H", "tau", "epsilon", "rho", "p", "n_target",
            "C_M", "C_R", "C_S", "C_P", "C_rep", "repair_model",
            "reliability_impl", "pwl_points", "tangent_ref",
            "allow_replacement")
    overrides = {k: getattr(args, k) for k in keys if getattr(args, k) is not None}
    return Scenario(**overrides)


def parse_sweeps(args, sc: Scenario) -> dict:
    """Sweep values per parameter; the base value is always included so the
    curves share one common point."""
    defaults = {"L": [1, 2, 3], "H": [4, 6, 8, 10, 12],
                "M": [1, 2, 3, 4], "F": [3, 4, 5, 6, 7]}
    if args.values:
        for block in args.values.split(";"):
            if not block.strip():
                continue
            key, _, vals = block.partition("=")
            defaults[key.strip()] = [_cast(key.strip(), v)
                                     for v in vals.split(",") if v.strip()]
    out = {}
    for param in (q.strip() for q in args.params.split(",") if q.strip()):
        if param not in SWEEP_PARAMS:
            raise SystemExit(f"unknown sweep parameter {param!r}; "
                             f"pick from {SWEEP_PARAMS}")
        out[param] = sorted(set(defaults[param]) | {_cast(param, getattr(sc, param))})
    return out


def parse_ladders(args) -> dict:
    """Stress ladders for the failure test: {param: (direction, [mild..harsh])}."""
    custom = {}
    if args.failure_values:
        for block in args.failure_values.split(";"):
            if not block.strip():
                continue
            key, _, vals = block.partition("=")
            key = key.strip()
            custom[key] = [_cast(key, v) for v in vals.split(",") if v.strip()]
    out = {}
    for param in (q.strip() for q in args.failure_params.split(",") if q.strip()):
        if param in custom:
            direction = (STRESS_LADDERS[param][0] if param in STRESS_LADDERS
                         else "custom ladder")
            out[param] = (direction, custom[param])
        elif param in STRESS_LADDERS:
            out[param] = STRESS_LADDERS[param]
        else:
            raise SystemExit(f"no stress ladder for {param!r}; known: "
                             f"{tuple(STRESS_LADDERS)} (or pass --failure-values)")
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    global BOUNDS_ORDER
    BOUNDS_ORDER = tuple(b.strip() for b in args.bounds.split(",") if b.strip())

    sc = scenario_from_args(args)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    tests = [t.strip() for t in args.tests.split(",") if t.strip()]

    if args.merge:                               # reduce step, no solving
        rc = 0
        for test in tests:
            rc |= merge_shards(out_root, args.name, test, args)
        return rc
    sweeps = parse_sweeps(args, sc)
    ladders = parse_ladders(args) if "failure" in tests else {}

    header = [f"# bound tests  {datetime.now():%Y-%m-%d %H:%M:%S}",
              f"# bounds: {list(BOUNDS_ORDER)}",
              f"# implementations: {args.impl_list}"
              + (f"  pwl_ladder={args.pwl_ladder_list}"
                 if args.pwl_ladder_list else ""),
              f"# mip_gap={args.mip_gap}  "
              f"time_limit={'none' if args.time_limit is None else args.time_limit}"
              f"  dry_run={args.dry_run}"
              + (f"  threads={args.threads}" if args.threads else "")
              + (f"  shard={args.shard_obj}" if args.shard_obj else ""),
              (f"# host={_HOSTNAME} slurm_job={_SLURM_JOB}" if _SLURM_JOB else ""),
              (f"# code: branch={_GIT['git_branch']} commit={_GIT['git_commit']}"
               if _GIT["git_commit"] else ""),
              ""]

    runners = {
        "analytic": lambda r: test_analytic(sc, args, r, sweeps),
        "base": lambda r: test_base(sc, args, r, args.impl_list),
        "sweep": lambda r: test_sweep(sc, args, r, sweeps, args.impl_list),
        "impl": lambda r: test_impl(sc, args, r, args.impl_list,
                                    args.pwl_ladder_list),
        "failure": lambda r: test_failure(sc, args, r, ladders, args.impl_list),
    }
    for test in tests:
        if test not in runners:
            raise SystemExit(f"unknown test {test!r}; pick from {tuple(runners)}")
    if "impl" in tests and len(args.impl_list) < 2 and not args.pwl_ladder_list:
        raise SystemExit("the 'impl' test compares implementations, so give at "
                         "least two: --impls tangent,pwl,exact "
                         "(optionally with --pwl-ladder 2,4,8,16)")

    suffix = "" if args.shard_obj is None else f"_shard{args.shard_obj.k}"
    for test in tests:
        if args.shard_obj is not None:
            args.shard_obj.i = 0                 # restart the unit counter per test
        run = TestRun(out_root, args.name, test, sc, args, suffix=suffix)
        report = header + [f"TEST {test}", "=" * 72]
        extra = None
        try:
            lines, extra = runners[test](run)
            report += lines
        except BaseException as exc:             # Ctrl-C included: keep the data
            report += ["", f"ABORTED: {type(exc).__name__}: {exc}"]
            run.close(report, extra)
            raise
        run.close(report, extra)
        print("\n".join(report))
        print(f"\n[{test}] folder  : {run.dir}")
        print(f"[{test}] results : {run.csv_path.name}, {run.yaml_path.name}, "
              f"{run.summary_path.name}, runs/ ({len(run.rows)} yaml)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())