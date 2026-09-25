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
       Every row records n_repairs / n_depot / n_idle / mu_max, and the summary
       flags the run as uninformative when those are identical across bounds.
       Note that n_depot counts PAID maintenance slots (x[i,0,k]) only: the
       assignment constraint is `sum_j x <= 1`, so a vehicle may also be left
       unassigned, which is free.  That case is n_idle, and `n_depot_noop`
       counts depot steps where no repair or replacement fired -- expected to
       be 0 at a closed optimum, since idling instead would be strictly
       cheaper for an identical state trajectory.  If that
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
import collections
import csv
import traceback
import textwrap
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


def _import_dispatch():
    """The project's own model dispatcher, so a case file can use any model.

    `solver._solve_mixed` routes gamma-only, rainflow-only and genuinely mixed
    fleets. Run options travel inside the config (config.load_config passes
    mip_gap / time_limit / gurobi_params / reliability_impl through), which is why
    nothing has to be threaded past it.
    """
    from fleet_management.config import load_config
    from fleet_management.solver import _solve_mixed
    try:
        from fleet_management.solver import _read_input
    except ImportError:
        _read_input = None
    return load_config, _solve_mixed, _read_input


def _import_solver():
    """Imported lazily so `analytic` / `--dry-run` work without Gurobi."""
    from fleet_management.config import load_config
    from fleet_management.degradation_model.rainflow_v2 import solve as rainflow_solve
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
# Structural parameters, plus the reliability and repair knobs. Sweeping the
# latter answers a different question from H1: not "which bound is tighter" but
# "what does reliability cost".
SWEEP_PARAMS = ("L", "H", "M", "F", "epsilon", "rho", "tau", "p", "tangent_ref")
# Sweeping any of these changes b_ref through the calibration, which would move
# the increment distribution as well as the requirement -- two things at once.
# The harness freezes the increment scale at its base value for these.
SCALE_COUPLED = ("epsilon", "tau", "p")
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


def _clean(token: str) -> str:
    """Strip whitespace and stray quotes from a CLI token.

    `EXTRA="... --values 'H=1,2;M=3'"` in a submit script survives into the job as
    a literal apostrophe on the first and last token, because $EXTRA is expanded
    unquoted: bash word-splits but does not re-process quotes. Without this, the
    last value parses as `3'` and int() raises a bare ValueError from deep inside
    argument parsing.
    """
    return token.strip().strip("\"'").strip()


def _cast(param: str, value):
    token = _clean(str(value))
    try:
        return int(token) if param in _INT_PARAMS else float(token)
    except ValueError:
        raise SystemExit(
            f"cannot read {token!r} as a value for {param!r}. Expected "
            f"{'an integer' if param in _INT_PARAMS else 'a number'}. If this came "
            f"from --values/--failure-values in a submit script, drop the inner "
            f"quotes: EXTRA=\"--values H=10,12;F=3,4\" (the string has no spaces, "
            f"so it needs none, and quotes inside EXTRA survive as literal "
            f"characters).")


def run_stamp(opts) -> str:
    """YYYYMMDDHHMM identifying the RUN (shared by every shard of an array)."""
    stamp = getattr(opts, "run_stamp", None) or os.environ.get("RUN_STAMP", "")
    stamp = str(stamp).strip()
    if stamp:
        if not (len(stamp) == 12 and stamp.isdigit()):
            raise SystemExit(f"--run-stamp must be 12 digits YYYYMMDDHHMM, "
                             f"got {stamp!r}")
        return stamp
    return f"{datetime.now():%Y%m%d%H%M}"


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

    markov and chernoff are already linear: `rainflow._resolve_impl` falls back to
    "exact" for them whatever is requested. They are therefore emitted exactly
    ONCE, as (bound, "exact"), no matter which impls were asked for.

    That "once, always" matters. Dropping them when the requested impl is not
    "exact" would silently remove two of the five bounds, and (H1) is a statement
    about all five -- `--impls tangent` would then quietly test a different, weaker
    hypothesis. `cost_for` maps them onto every impl group when the checks run.
    """
    out, folded = [], []
    for bound in BOUNDS_ORDER:
        if bound not in IMPL_AWARE_BOUNDS:
            out.append((bound, "exact"))         # linear bound: encoding is moot
            if set(impls) != {"exact"}:
                folded.append(bound)
            continue
        for impl in impls:
            out.append((bound, impl))
    if announce and folded:
        print(f"  [impl] {', '.join(folded)} are linear bounds: the implementation "
              f"does not apply, so each runs once as 'exact' and is compared "
              f"against every requested encoding.")
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
    b_ref_fixed: float | None = None  # freeze the increment scale (see SCALE_COUPLED)
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
    # MILP encoding of the logical constraints (rainflow_v2).  'indicator' is the
    # original model; 'bigm' substitutes nb out and writes linear big-M rows.
    # The two have the SAME integer feasible set, so `objective` must agree --
    # what changes is the relaxation, the size and the solve time.
    formulation: str = "indicator"
    bigM: float = 1.1               # fallback big-M; states live in [0, 1.1]

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
        if self.b_ref_fixed is not None:
            return float(self.b_ref_fixed)
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
            # `formulation` here may be a harness LABEL (e.g. 'indicator_cuts');
            # split it into the two solver options it stands for.
            **dict(zip(("formulation", "sparse_cuts"),
                       split_variant(self.formulation))),
            "bigM": float(self.bigM),
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
                + (f"({self.pwl_points})" if self.reliability_impl == "pwl" else "")
                + f" form={self.formulation}")

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
                        "formulation": self.formulation,
                        "bigM": float(self.bigM),
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
        #
        # CAREFUL: the model uses `support_max_of` -- the max support over ALL
        # permitted missions -- not the support of the mission actually flown,
        # because Bernstein's b enters non-additively and has to be a constant.
        # Using b_ref here silently overstates how permissive bernstein is: with a
        # +-25 % severity spread, b_max = 1.25*b_ref and the drift term Le*b/3
        # grows with it. That difference is large enough to reorder the bounds.
        A = Le * float(np.max(sc.b)) / 3.0
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


def tangent_cap_n_max(sc: Scenario, bound: str, ref: float) -> float:
    """Missions the SINGLE-TANGENT encoding allows, mirroring `_add_tangent_cap`.

    The model caps  Q <= g(mu_p) + g'(mu_p)(mu - mu_p)  with
    g(mu) = c2*d^2 + c1*d, d = tau - mu, taken at mu_p = ref*tau. Both sides are
    linear in the mission count n, so this is closed form.

    Returns -1.0 when the cap is already negative at mu = 0: since Q >= 0 always,
    the cell is then INFEASIBLE after a single mission -- which is exactly what
    bernstein does at the default ref = 0.5, because its c1 = -b/3 makes g(0.5*tau)
    negative.
    """
    tau, eps, Le = sc.tau, sc.epsilon, sc.Le
    m1 = sc.p * sc.b_ref
    b_max = float(np.max(sc.b))
    if bound == "cantelli":
        c2, c1 = eps / (1.0 - eps), 0.0
        q_slope = sc.p * (1.0 - sc.p) * sc.b_ref ** 2          # v per mission
    elif bound == "hoeffding":
        c2, c1 = 2.0 / Le, 0.0
        q_slope = sc.b_ref ** 2                                # R per mission
    elif bound == "bernstein":
        c2, c1 = 1.0 / (2.0 * Le), -(b_max / 3.0)
        q_slope = sc.p * (1.0 - sc.p) * sc.b_ref ** 2
    else:
        return math.inf                                        # linear bound
    mu_p = float(np.clip(ref, 0.0, 1.0)) * tau
    d_p = tau - mu_p
    g_p = c2 * d_p * d_p + c1 * d_p
    gp = -2.0 * c2 * d_p - c1
    intercept = g_p - gp * mu_p                                # cap at mu = 0
    if intercept < 0:
        return -1.0
    denom = q_slope - gp * m1
    return intercept / denom if denom > 0 else math.inf


def best_tangent_ref(sc: Scenario, bound: str) -> tuple[float, float]:
    """The tangent_ref that maximises the tangent encoding's budget."""
    best = (0.0, -math.inf)
    for ref in np.linspace(0.0, 0.95, 96):
        n = tangent_cap_n_max(sc, bound, float(ref))
        if n > best[1]:
            best = (float(ref), n)
    return best


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


def _assignment_counts(x, m, r) -> dict:
    """Depot / idle / wasted-depot counts from the assignment variable.

    ``n_depot``  paid maintenance slots, sum of x[i,0,k].
    ``n_idle``   vehicle-steps with no assignment at all (the slack in
                 ``sum_j x[i,j,k] <= 1``); free, and invisible unless counted
                 here, since the schedule dump used to label them "depot".
    ``n_depot_noop``  depot steps at which neither repair nor replacement
                 fires.  Repair is gated on a same-step depot day with no lead
                 time, so such a step buys nothing and costs C_M: idling
                 instead is strictly cheaper and leaves the state identical.
                 A nonzero count on a run reported "optimal" therefore means
                 the gap is not closed (check mip_gap) -- it is a cheap
                 canary for a schedule that looks busier than it needs to be.
    """
    if x is None:
        return {"n_depot": math.nan, "n_idle": math.nan,
                "n_depot_noop": math.nan}
    F, Jp1, T = x.shape
    depot = x[:, 0, :] > 0.5                                   # (F, T)
    assigned = np.sum(x > 0.5, axis=1) > 0                     # (F, T)
    out = {"n_depot": float(np.sum(x[:, 0, :])),
           "n_idle": float(np.sum(~assigned))}
    if m is None and r is None:
        out["n_depot_noop"] = math.nan
        return out
    acted = np.zeros((F, T), dtype=bool)
    for arr in (m, r):
        if arr is not None:
            acted |= np.any(np.asarray(arr) > 0.5, axis=1)     # any component
    out["n_depot_noop"] = float(np.sum(depot & ~acted))
    return out


# ===========================================================================
# One solve
# ===========================================================================
def run_case(sc: Scenario, bound: str, opts, log_path=None) -> tuple[dict, dict]:
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
        "formulation": sc.formulation, "bigM": sc.bigM,
        "encoding": split_variant(sc.formulation)[0],
        "sparse_cuts": split_variant(sc.formulation)[1],
        "allow_replacement": sc.allow_replacement,
        # The REQUESTED limits, as opposed to the mip_gap and runtime_s the
        # solve achieved. Only the case test recorded these before, which left
        # every other test's CSV unable to answer "did this stop because it
        # converged or because it was allowed to?" -- and left the plots with no
        # reference line to draw.
        "req_mip_gap": _f(getattr(opts, "mip_gap", None)),
        "req_time_limit": _f(getattr(opts, "time_limit", None)),
        "C_M": sc.C_M, "C_R": sc.C_R, "C_S": sc.C_S, "C_P": sc.C_P,
        "threads": getattr(opts, "threads", None) or "",
        "gurobi_params": ",".join(f"{k}={v}" for k, v in
                                  sorted((_gurobi_params(opts) or {}).items())),
        "solver_log": (log_path.name if log_path is not None else ""),
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
                    "obj_bound": math.nan, "runtime_s": math.nan, "wall_s": math.nan,
                    "build_s": math.nan})
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
            formulation=split_variant(sc.formulation)[0],
            sparse_cuts=split_variant(sc.formulation)[1],
            bigM=sc.bigM,
            # On a shared cluster node Gurobi would otherwise spawn threads for
            # every core it can see, not for the cores Slurm gave the job.
            gurobi_params=_gurobi_params(opts, log_path),
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
        # wall_s covers build + solve; runtime_s is Gurobi's solve alone. build_s
        # is what the sparse assembly changes, and without it a
        # formulation=indicator,sparse comparison has nothing to look at: the
        # two are the same program, so runtime_s and nodes must agree and only
        # the construction cost moves.
        "build_s": _f(res.get("build_s")),
    })

    md = res.get("model")
    if md is not None:
        rec.update(collect_model_metrics(md))

    # binding diagnostics: are the bounds actually doing anything? (design note 3)
    for key, arr, red in (("n_repairs", res.get("m"), np.sum),
                          ("n_replacements", res.get("r"), np.sum),
                          ("mu_max", res.get("mu"), np.max),
                          ("v_max", res.get("v"), np.max)):
        rec[key] = float(red(arr)) if arr is not None else math.nan
    x = res.get("x")
    rec.update(_assignment_counts(x, res.get("m"), res.get("r")))

    n_mc = int(getattr(opts, "verify_mc", 0) or 0)
    if n_mc > 0 and isinstance(rec.get("objective"), float) \
            and math.isfinite(rec["objective"]):
        rng = np.random.default_rng(int(getattr(opts, "mc_seed", 0)))
        rec.update(monte_carlo_check(sc, res, n_mc, rng,
                                     getattr(opts, "mc_dist", "bernoulli")))

    if md is not None:
        try:
            md.dispose()                         # release the Gurobi environment
        except Exception:
            pass
    return rec, data


def _sample_increment(sc: Scenario, b: float, n: int, rng, dist: str) -> np.ndarray:
    """Draw n damage increments for a mission of severity b.

    `bernoulli` is the distribution the model's descriptors were derived from.

    Note it is also the ONLY bounded one available here: on support [0, b] with
    mean p*b the largest achievable variance is p(1-p)b^2, which the two-point
    distribution attains exactly. So no other distribution on [0, b] can match
    both moments -- the model's descriptors already describe the extremal case,
    and any moment-matched alternative must leave the support.

    `lognormal` does exactly that: same mean and variance, unbounded above. The
    support-based bounds (hoeffding, bernstein) lose their justification under it
    while the moment-based ones (markov, cantelli) keep theirs, so the empirical
    P(D>tau) shows which guarantees actually survive misspecification.
    """
    m = sc.p * b                                   # mean the model was given
    v = sc.p * (1.0 - sc.p) * b * b                # variance the model was given
    if dist == "bernoulli":
        return (rng.random(n) < sc.p) * b
    if dist == "lognormal":
        sig2 = math.log(1.0 + v / (m * m))
        return rng.lognormal(math.log(m) - sig2 / 2.0, math.sqrt(sig2), n)
    raise SystemExit(f"unknown --mc-dist {dist!r}; pick from bernoulli, lognormal")


def monte_carlo_check(sc: Scenario, res: dict, n_samples: int,
                      rng: "np.random.Generator", dist: str = "bernoulli") -> dict:
    """Empirical P(D > tau) for the SOLVED schedule, per rainflow cell.

    The bounds only certify P(D > tau) <= eps; they say nothing about how much
    slack they leave. Simulating the optimised schedule with sampled increments
    turns that slack into a number, which is what makes "tighter bound" a
    measured claim rather than an algebraic one.

    Mirrors the pathwise recursion in `rainflow._add_cell_dynamics` exactly:
        mission step   D <- D + b_j * Bernoulli(p)
        repair step    D <- (1 - rho) * D          (ardinf; no increment, the
                                                    vehicle is at the depot)
        replacement    D <- mu_new
    and evaluates the constraint where the model imposes it: at EVERY step, so
    the reported figure is max over k, then the worst cell.
    """
    x, m, r = res.get("x"), res.get("m"), res.get("r")
    if x is None or m is None:
        return {}
    F, L, T = sc.F, sc.L, sc.T
    b = sc.b                                       # per-mission severity
    worst, worst_cell, per_cell = 0.0, "", []
    for i in range(F):
        # which mission (if any) vehicle i flies at each step; column 0 = depot
        mission_at = [next((j for j in range(1, sc.M + 1) if x[i, j, k] > 0.5), 0)
                      for k in range(T)]
        for l in range(L):
            D = np.full(n_samples, float(sc.mu_0))
            peak = 0.0
            for k in range(T):
                if r is not None and r[i, l, k] > 0.5:
                    D[:] = 0.0                     # replacement resets the cell
                elif m[i, l, k] > 0.5:
                    D *= (1.0 - sc.rho)
                elif mission_at[k]:
                    D = D + _sample_increment(sc, b[mission_at[k] - 1],
                                              n_samples, rng, dist)
                peak = max(peak, float(np.mean(D > sc.tau)))
            per_cell.append(peak)
            if peak > worst:
                worst, worst_cell = peak, f"i={i},l={l}"
    # Wilson-free normal CI is fine at these sample counts; report it so a reader
    # can see whether "0.000" means zero or just under-sampled
    se = math.sqrt(max(worst, 1e-12) * (1 - worst) / n_samples)
    return {"mc_samples": n_samples, "mc_dist": dist,
            "mc_p_max": worst,
            "mc_p_max_cell": worst_cell,
            "mc_p_mean": float(np.mean(per_cell)) if per_cell else math.nan,
            "mc_ci95": 1.96 * se,
            "mc_slack": sc.epsilon - worst,
            "mc_conservatism": (sc.epsilon / worst) if worst > 0 else math.inf}


def solver_log_path(run, label: str) -> "Path | None":
    """`<run folder>/solver_logs/<label>.log`, or None when logging is off."""
    if getattr(run, "log_mode", "off") == "off":
        return None
    d = run.dir / "solver_logs"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{_safe_filename(label)}.log"


def _gurobi_params(opts, log_path=None) -> dict | None:
    """Threads plus any --gurobi-params overrides, e.g. MIPFocus=3,Symmetry=2.

    Useful when the dual bound is the bottleneck rather than the incumbent: the
    reliability constraint makes this a nonconvex MIQCP whose LP relaxation is
    weak, and F identical vehicles make the tree highly symmetric.
    """
    params: dict = {}
    if getattr(opts, "threads", None):
        params["Threads"] = int(opts.threads)
    # accept the flag more than once and MERGE. Anything else is a trap: a job
    # script that sets NodefileStart and a user EXTRA that sets MIPFocus would
    # otherwise silently discard the first, and the tree stays in RAM until the
    # node is OOM-killed. Later occurrences win per key.
    raw = getattr(opts, "gurobi_params", None) or []
    if isinstance(raw, str):
        raw = [raw]
    for item in (q for chunk in raw for q in str(chunk).split(",")):
        item = _clean(item)
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"--gurobi-params expects KEY=VALUE, got {item!r}")
        key, _, val = item.partition("=")
        try:
            params[key.strip()] = int(val)
        except ValueError:
            try:
                params[key.strip()] = float(val)
            except ValueError:
                params[key.strip()] = val.strip()
    if log_path is not None:
        # OutputFlag gates BOTH the console and the log file, and base.py sets it
        # from `verbose` before applying these -- so enabling the log means
        # enabling OutputFlag and muting the console separately.
        params["LogFile"] = str(log_path)
        params["OutputFlag"] = 1
        params["LogToConsole"] = 1 if int(getattr(opts, "verbose", 0) or 0) else 0
    return params or None


def _f(value):
    return float(value) if value is not None else math.nan


# Every Gurobi attribute worth recording. Missing ones are skipped silently:
# availability depends on the model class (QCP attributes on a MILP, for example)
# and on whether a solution was found at all.
MODEL_ATTRS = {
    "runtime_s": "Runtime", "work": "Work",
    "n_vars": "NumVars", "n_constrs": "NumConstrs", "n_qconstrs": "NumQConstrs",
    "n_genconstrs": "NumGenConstrs", "n_sos": "NumSOS",
    "n_nz": "NumNZs", "n_qnz": "NumQNZs",
    "n_int": "NumIntVars", "n_bin": "NumBinVars",
    "nodes": "NodeCount", "iterations": "IterCount", "bar_iterations": "BarIterCount",
    "sol_count": "SolCount", "obj_val": "ObjVal", "obj_bound_c": "ObjBoundC",
    "max_vio": "MaxVio", "is_mip": "IsMIP", "is_qcp": "IsQCP",
}


def collect_model_metrics(md) -> dict:
    """Read every available attribute off a solved Gurobi model."""
    out = {}
    for key, attr in MODEL_ATTRS.items():
        try:
            out[key] = float(getattr(md, attr))
        except Exception:
            pass
    return out


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
          "objective", "mip_gap", "obj_bound", "runtime_s", "wall_s", "build_s",
          "verdict", "n_max_analytic", "load", "feasible_hint",
          "n_repairs", "n_replacements", "n_depot", "n_idle",
          "n_depot_noop", "mu_max", "v_max",
          "n_vars", "n_constrs", "n_qconstrs", "n_genconstrs", "n_sos",
          "n_nz", "n_qnz", "n_int", "n_bin", "nodes", "iterations",
          "bar_iterations", "sol_count", "work", "obj_val", "obj_bound_c",
          "max_vio", "is_mip", "is_qcp",
          "F", "M", "L", "H", "T", "tau", "epsilon", "rho", "p", "b_ref",
          "mu_ref", "s_chernoff", "repair_model", "reliability_impl",
          "pwl_points", "tangent_ref", "formulation", "encoding", "sparse_cuts",
          "bigM",
          "allow_replacement", "C_M", "C_R", "C_S", "C_P",
          "model", "req_mip_gap", "req_time_limit", "req_verbose", "traceback",
          "mc_samples", "mc_dist", "mc_p_max", "mc_p_max_cell", "mc_p_mean", "mc_ci95",
          "mc_slack", "mc_conservatism",
          "threads", "gurobi_params", "solver_log", "host", "slurm_job",
          "git_branch", "git_commit"]


class TestRun:
    """One folder per run: `<out>/YYYYMMDDHHMM_<test>/`, everything inside it.

    The stamp is the time the RUN started, not the time this process started.
    A Slurm array launches one process per shard, minutes apart, so each would
    otherwise mint its own folder and the shards would never be merged. So the
    stamp comes from --run-stamp / $RUN_STAMP (exported once by submit.sh) and
    only falls back to the local clock for a plain interactive run.

    Layout:
        <out>/202608131228_sweep/
            scenario_base.yaml          the design point
            results_shard0.csv          one per shard (or results.csv unsharded)
            summary_shard0.txt
            progress_shard0.log         flushed per solve; survives a SIGKILL
            runs/<bound>_<impl>__<param><value>.yaml
            merged_results.csv|.yaml    written by --merge
            merged_summary.txt
            merged_cost_<param>.png
    """

    def __init__(self, out_root: Path, name: str, test: str, sc: Scenario, opts,
                 suffix: str = ""):
        self.stamp = run_stamp(opts)
        self.test = test
        self.name = name
        self.dir = out_root / f"{self.stamp}_{test}"
        self.runs_dir = self.dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        # shard identity lives in FILE names; the folder is shared by the whole run
        self.tag = suffix or ""
        # 'auto' means: on for the case test (a handful of runs, each worth a log)
        # and off for the sweeps, where hundreds of logs would swamp the folder
        mode = getattr(opts, "solver_log", "auto")
        self.log_mode = (("on" if test == "case" else "off")
                         if mode == "auto" else mode)
        self.stem = test
        self.csv_path = self.dir / f"results{self.tag}.csv"
        self.yaml_path = self.dir / f"results{self.tag}.yaml"
        self.summary_path = self.dir / f"summary{self.tag}.txt"
        self.progress_path = self.dir / f"progress{self.tag}.log"
        self.rows: list[dict] = []
        self._fh = self.csv_path.open("w", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=FIELDS, extrasaction="ignore")
        self._w.writeheader()
        # Flush immediately. Without this a task killed by Slurm (OOM or timeout)
        # before the first solve completes leaves a 0-byte file, which looks like
        # "the script never ran" rather than "it was killed mid-solve".
        self._fh.flush()
        # one shared copy of the design point; shards would all write the same
        # bytes, so the first one wins and the rest skip it
        base_yaml = self.dir / "scenario_base.yaml"
        if not base_yaml.exists():
            _dump_yaml(base_yaml, {
                "test": test, "name": name, "run_stamp": self.stamp,
                "created": datetime.now().isoformat(timespec="seconds"),
                "code_version": dict(_GIT, host=_HOSTNAME, slurm_job=_SLURM_JOB),
                "solver_options": {"mip_gap": opts.mip_gap,
                                   "time_limit": opts.time_limit,
                                   "threads": getattr(opts, "threads", None),
                                   "gurobi_params": getattr(opts, "gurobi_params", None),
                                   "dry_run": bool(opts.dry_run)},
                "base_case": sc.to_yaml_dict()})

    # ---- one run -----------------------------------------------------------
    def note_progress(self, text: str) -> None:
        """Append to a progress file, flushed per line.

        The CSV only gains a row once a solve *finishes*. If a task is killed
        during a long solve there is otherwise no record of which design point it
        died on -- this file has it.
        """
        with self.progress_path.open("a") as fh:
            fh.write(f"{datetime.now():%H:%M:%S} {text}\n")

    def add(self, rec: dict, data: dict, sc: Scenario) -> None:
        rec.setdefault("verdict", classify(rec))
        self.note_progress(f"DONE  {rec.get('parameter')}={rec.get('value')} "
                           f"{rec.get('bound')}/{impl_of_record(rec)} "
                           f"status={rec.get('status')} "
                           f"cost={rec.get('objective')} "
                           f"time={rec.get('runtime_s')}")
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
        """`<bound>_<impl>__<param><value>`. The impl belongs in the name: two
        impls of the same bound at the same design point would otherwise write to
        the same file and silently overwrite each other."""
        param, value = rec.get("parameter", "-"), rec.get("value", "")
        tag = "base" if param in ("-", "", None) else f"{param}{value}"
        impl = impl_of_record(rec)
        if impl == "pwl":
            impl = f"pwl{rec.get('pwl_points', '')}"
        return _safe_filename(f"{rec.get('bound', 'unspecified')}_{impl}__{tag}")

    # ---- finish ------------------------------------------------------------
    def close(self, report: list[str], extra: dict | None = None) -> None:
        self._fh.close()
        _dump_yaml(self.yaml_path, {
            "test": self.test, "name": self.name, "run_stamp": self.stamp,
            "created": datetime.now().isoformat(timespec="seconds"),
            "n_runs": len(self.rows),
            "summary": extra or {},
            "runs": [_to_builtin(r) for r in self.rows],
        })
        self.summary_path.write_text("\n".join(report))


_BAD_FILENAME = set('<>:"/\\|?*') | {chr(c) for c in range(32)}


def _safe_filename(text: str) -> str:
    r"""Make a string usable as a filename on Windows as well as POSIX.

    Windows rejects <>:"/\|?* and control characters, and dislikes trailing dots
    or spaces. A case named after a path ("data/test/x") or a field that fell back
    to "?" would otherwise raise OSError [Errno 22] deep inside the YAML dump,
    after the solve has already been paid for.
    """
    out = "".join("_" if ch in _BAD_FILENAME else ch for ch in str(text))
    out = out.replace(".", "p").strip(" _")
    return out or "unnamed"


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


def plot_scalability(rows: list[dict], param: str, run) -> Path | None:
    """Solver effort vs one parameter: runtime, nodes and final gap per bound.

    Cost answers "which bound is better"; this answers "what does it cost to
    find out", which is the complexity story.
    """
    plt = _pyplot()
    if plt is None:
        return None
    sub = [r for r in rows if r.get("parameter") == param]
    if not sub:
        return None
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.5), sharex=True)
    styles = {"tangent": ":", "pwl": "--", "exact": "-"}
    impls_seen = [im for im in IMPLS_ORDER
                  if any(impl_of_record(r) == im for r in sub)]
    series = {"runtime_s": [], "nodes": [], "mip_gap": []}
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
            for ax, key in zip(axes, ("runtime_s", "nodes", "mip_gap")):
                ys = [r.get(key, math.nan) for r in pts]
                series[key] += ys
                ax.plot(xs, ys, marker="o", label=label, color=colour,
                        ls=styles.get(impl, "-"))
    for ax, key, lab in zip(axes, ("runtime_s", "nodes", "mip_gap"),
                            ("solve time [s]", "B&B nodes", "final MIP gap")):
        ax.set_ylabel(lab)
        _safe_log_scale(ax, series[key])
        ax.grid(alpha=0.3, which="both")
    axes[2].axhline(0.01, color="0.5", lw=0.8, ls=":")
    axes[2].annotate("1 %", (axes[2].get_xlim()[0], 0.01), fontsize=7, color="0.4")
    axes[0].set_title(f"Solver effort vs {param}", fontsize=10)
    axes[0].legend(fontsize=7, ncol=2 if len(impls_seen) > 1 else 1)
    axes[2].set_xlabel(param)
    axes[2].set_xticks(sorted({r["value"] for r in sub}))
    fig.tight_layout()
    path = run.dir / f"{run.stem}_scalability_{param}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_formulation_bars(rows: list[dict], run: TestRun) -> Path | None:
    """Grouped bars comparing the formulations of the 'formulation' test.

    Why bars and not the generic `plot_parameter` line plot: `formulation` is a
    CATEGORICAL parameter with two or four values, so a line between them
    interpolates something that does not exist, and on a two-point axis it
    renders as a flat segment that hides the very differences it is meant to
    show. One bar per (bound, formulation) is the honest shape.

    Four panels, chosen so that each one answers a different question:

        build time   the ONLY thing an encoding's sparse twin changes; the
                     per-cell addConstr loop against the matrix API
        solve time   must be ~equal within an assembly pair (same program); a
                     hatched bar marks a run that hit the time limit, where the
                     number is the limit and not a measurement
        B&B nodes    also ~equal within a pair, and NOT exactly so: the sparse
                     assembly groups rows by family, and row order alone moves
                     Gurobi's tie-breaking
        final gap    what the solve actually achieved, with the requested gap
                     drawn in; a bar at the requested line is a solve that
                     stopped because it was allowed to, not because it converged

    Infeasible runs are labelled rather than drawn as a zero-height bar, since a
    0 s solve of an infeasible model is a real result and an empty slot is not.
    """
    plt = _pyplot()
    if plt is None:
        return None
    sub = [r for r in rows if str(r.get("parameter")) == "formulation"]
    sub = [r for r in sub if str(r.get("bound")) in BOUNDS_ORDER]
    if not sub:
        return None
    forms = [f for f in FORMULATIONS_ORDER
             if any(str(r.get("value")) == f for r in sub)]
    groups = [(b, im) for b in BOUNDS_ORDER for im in IMPLS_ORDER
              if any(r["bound"] == b and impl_of_record(r) == im for r in sub)]
    if len(forms) < 2 or not groups:
        return None

    def cell(bound, impl, form, key):
        for r in sub:
            if (r["bound"] == bound and impl_of_record(r) == impl
                    and str(r.get("value")) == form):
                v = r.get(key, math.nan)
                return v if isinstance(v, (int, float)) else math.nan
        return math.nan

    def status(bound, impl, form) -> str:
        for r in sub:
            if (r["bound"] == bound and impl_of_record(r) == impl
                    and str(r.get("value")) == form):
                return str(r.get("status", ""))
        return ""

    # log only where the data spans decades. The build panel spans a factor of
    # ~3, and a log axis there compresses the one difference the figure exists
    # to show; the solve panel spans 4e-4 s to 3600 s and needs it.
    panels = (("build_s", "model build [s]", False),
              ("runtime_s", "solve time [s]", True),
              ("nodes", "B&B nodes", True),
              ("mip_gap", "final MIP gap", False))
    fig, axes = plt.subplots(len(panels), 1, figsize=(1.7 * len(groups) + 4.0,
                                                     9.0), sharex=True)
    x = np.arange(len(groups), dtype=float)
    width = 0.8 / len(forms)

    for ax, (key, label, logy) in zip(axes, panels):
        vals_all = []
        for fi, form in enumerate(forms):
            ys = [cell(b, im, form, key) for b, im in groups]
            sts = [status(b, im, form) for b, im in groups]
            vals_all += ys
            pos = x - 0.4 + width * (fi + 0.5)
            bars = ax.bar(pos, [0.0 if not math.isfinite(v) else v for v in ys],
                          width * 0.92, label=form, color=f"C{fi}",
                          edgecolor="0.25", linewidth=0.6)
            for bar, v, st in zip(bars, ys, sts):
                # A run stopped by the clock is not a measurement of anything;
                # hatch it so it cannot be read as one.
                if key in ("runtime_s", "mip_gap") and st.startswith("time_limi"):
                    bar.set_hatch("///")
                if not math.isfinite(v):
                    ax.annotate("infeasible" if st.startswith("infeas") else "n/a",
                                (bar.get_x() + bar.get_width() / 2, 0),
                                textcoords="offset points", xytext=(0, 4),
                                ha="center", fontsize=6, rotation=90,
                                color="0.35")
        ax.set_ylabel(label)
        if logy:
            _safe_log_scale(ax, vals_all)
        ax.grid(alpha=0.3, axis="y", which="both")
        if key == "build_s" and len(forms) > 1:
            # The build cost is the whole point of an assembly comparison, so
            # put the factor on the figure rather than making the reader
            # eyeball two bar heights.
            ref = [cell(b, im, forms[0], key) for b, im in groups]
            for fi, form in enumerate(forms[1:], start=1):
                ys = [cell(b, im, form, key) for b, im in groups]
                for gi, (a, v) in enumerate(zip(ref, ys)):
                    if not (math.isfinite(a) and math.isfinite(v) and v > 0):
                        continue
                    ax.annotate(f"{a / v:.1f}x",
                                (x[gi] - 0.4 + width * (fi + 0.5), v),
                                textcoords="offset points", xytext=(0, 3),
                                ha="center", fontsize=7, color="0.25")

    # The requested gap, so a bar sitting on it reads as "stopped because it was
    # allowed to" rather than "converged". req_mip_gap is missing from CSVs
    # written before it was recorded for every test, so fall back to the largest
    # gap achieved by a run that finished OPTIMAL: the solver stopped there
    # voluntarily, which puts a lower bound on what was asked for.
    req = [r.get("req_mip_gap") for r in sub]
    req = [g for g in req if isinstance(g, (int, float)) and math.isfinite(g)]
    missing_req = not req
    if req:
        axes[3].axhline(max(req), color="0.3", lw=1.0, ls="--", zorder=5)
        axes[3].annotate(f"requested gap {max(req):g}",
                         (axes[3].get_xlim()[0], max(req)),
                         textcoords="offset points", xytext=(4, 2),
                         fontsize=7, color="0.25", va="bottom", zorder=6)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([b if len({im for _, im in groups}) == 1
                              else f"{b}\n{im}" for b, im in groups])
    axes[-1].set_xlabel("bound")
    axes[0].legend(fontsize=8, ncol=len(forms), title="formulation",
                   title_fontsize=8, loc="lower center",
                   bbox_to_anchor=(0.5, 1.02), frameon=False)

    # State the invariant on the figure: within an assembly pair the model is
    # identical, so any solve-time or node difference is tie-breaking, not
    # modelling. Without this the reader has no way to know that from the plot.
    # Which formulations produce a byte-identical model. Sizes differ BETWEEN
    # bounds, so group per (bound, impl) first and then take the distinct sets
    # of formulations that agreed -- otherwise the same pair is listed once per
    # bound, which is how the first version of this note ended up repeating
    # itself.
    agreed = set()
    for b, im in groups:
        by_size = {}
        for form in forms:
            key = tuple(cell(b, im, form, k) for k in
                        ("n_vars", "n_constrs", "n_genconstrs", "n_nz"))
            by_size.setdefault(key, []).append(form)
        for grp in by_size.values():
            if len(grp) > 1:
                agreed.add(tuple(sorted(grp)))
    note = ("hatched = the solve was stopped by the time limit, so that bar is "
            "the limit and not a measurement")
    if missing_req:
        note += ("; the requested-gap line is absent because this CSV predates "
                 "req_mip_gap being recorded for every test")
    if agreed:
        note = ("identical model size (vars, rows, genconstrs, nonzeros) for " +
                " and ".join("/".join(g) for g in sorted(agreed)) +
                ", so differences in solve time and node count are Gurobi "
                "tie-breaking on row order, not modelling.  " + note)
    fig.suptitle("Formulation comparison", fontsize=11, y=0.99)
    # matplotlib's wrap=True measures against the figure edge, so a long note
    # still runs off a wide figure; wrap it by hand against the panel width.
    note = "\n".join(textwrap.wrap(note, width=max(90, 13 * len(groups) + 30)))
    fig.text(0.5, 0.006, note, ha="center", va="bottom", fontsize=7,
             color="0.3")
    fig.tight_layout(rect=(0, 0.025 + 0.013 * note.count("\n"), 1, 0.93))
    path = run.dir / f"{run.stem}_formulation_bars.png"
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
def interval_order(values: dict, bounds: dict) -> tuple[str, list[str]]:
    """(H1) by RIGOROUS interval separation -- the right test under a time limit.

    A time-limited solve is not a failed solve: it returns a feasible incumbent
    (a valid UPPER bound on the optimum) and a dual LOWER bound, so the true
    optimum satisfies z* in [LB, UB]. Therefore

        cost(looser) >= cost(tighter)  is PROVEN   iff  LB_looser >= UB_tighter
                                       is DISPROVEN iff  UB_looser <  LB_tighter

    and neither depends on how wide the gaps are. Overlapping intervals mean the
    pair is simply unresolved -- more solver time, not a different conclusion.

    Falls back to the gap heuristic (`order_verdict`) only when a dual bound is
    missing, which happens for infeasible runs and for runs that produced no
    incumbent at all.
    """
    issues, proven, overlap = [], 0, 0
    present = [b for b in BOUNDS_ORDER
               if isinstance(values.get(b), float) and not math.isnan(values.get(b))]
    disproven = False
    for looser, tighter in zip(present, present[1:]):
        ub_lo, ub_hi = values[looser], values[tighter]
        lb_lo, lb_hi = bounds.get(looser, math.nan), bounds.get(tighter, math.nan)
        if math.isinf(ub_lo):                    # infeasible looser bound
            proven += 1
            continue
        if math.isinf(ub_hi):
            disproven = True
            issues.append(f"DISPROVEN {looser}={ub_lo:.6g} feasible but "
                          f"{tighter}=infeasible")
            continue
        if math.isnan(lb_lo) or math.isnan(ub_hi):
            issues.append(f"UNRESOLVED {looser} vs {tighter}: no dual bound "
                          f"recorded, cannot certify either way")
            overlap += 1
            continue
        if lb_lo >= ub_hi - 1e-9:
            proven += 1
        elif ub_lo < lb_hi - 1e-9:
            disproven = True
            issues.append(f"DISPROVEN {looser}: UB {ub_lo:.6g} < {tighter} LB "
                          f"{lb_hi:.6g} -- the optima cannot be ordered this way")
        else:
            overlap += 1
            issues.append(f"UNRESOLVED {looser} [{lb_lo:.4g}, {ub_lo:.4g}] vs "
                          f"{tighter} [{lb_hi:.4g}, {ub_hi:.4g}]: intervals "
                          f"overlap, so more solver time is needed to rank them")
    if disproven:
        return "DISPROVEN", issues
    if overlap:
        return f"PARTIAL {proven}/{proven + overlap}", issues
    return "PROVEN", issues


def order_verdict(values: dict, gaps: dict) -> tuple[str, list[str]]:
    """(H1) verdict: HOLDS / VIOLATED / INCONCLUSIVE, plus the details.

    The third outcome matters. `check_order` only reports a violation when the
    inversion exceeds the MIP-gap slack, so with large gaps NOTHING can violate
    and the test would report HOLDS while proving nothing -- the ordering would be
    unfalsifiable, not confirmed. So if any adjacent pair's cost difference is
    SMALLER than the slack that pair carries, the design point is INCONCLUSIVE:
    the solver has not resolved the costs finely enough to rank those two bounds.
    """
    issues = check_order(values, gaps)
    if any(i.startswith("VIOLATION") for i in issues):
        return "VIOLATED", issues
    present = [b for b in BOUNDS_ORDER
               if isinstance(values.get(b), float) and not math.isnan(values.get(b))]
    unresolved = []
    for looser, tighter in zip(present, present[1:]):
        lo, hi = values[looser], values[tighter]
        if math.isinf(lo) or math.isinf(hi):
            continue                             # infeasibility is unambiguous
        g = max(gaps.get(looser) or 0.0, gaps.get(tighter) or 0.0)
        slack = max(1e-6, g * max(abs(lo), abs(hi)))
        if abs(lo - hi) < slack:
            unresolved.append(f"UNRESOLVED {looser} vs {tighter}: "
                              f"|{lo:.6g} - {hi:.6g}| = {abs(lo - hi):.3g} is below "
                              f"the gap slack {slack:.3g} -- these two cannot be "
                              f"ranked from these runs")
    if unresolved:
        return "INCONCLUSIVE", issues + unresolved
    return "HOLDS", issues


def check_order(values: dict, gaps: dict) -> list[str]:
    """(H1): verify cost(looser) >= cost(tighter) along BOUNDS_ORDER.

    A violation is only real if it exceeds the slack implied by the two runs'
    MIP gaps: with gap g the reported objective may sit up to g*|obj| above the
    true optimum, so differences below that are not evidence against (H1).
    Use `order_verdict` for the verdict -- absence of violations is NOT proof.
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
    lines.append("what each ENCODING actually allows (missions before violation):")
    lines.append(f"  {'bound':<10s} {'exact':>8s} {'tangent':>9s} "
                 f"{'best ref':>9s} {'at ref':>8s}")
    tangent_trouble = []
    for bound in BOUNDS_ORDER:
        if bound not in IMPL_AWARE_BOUNDS:
            lines.append(f"  {bound:<10s} {vals[bound]:8.2f} {'n/a':>9s}"
                         f"   (linear bound, encoding does not apply)")
            continue
        tn = tangent_cap_n_max(sc, bound, sc.tangent_ref)
        ref, best_n = best_tangent_ref(sc, bound)
        shown = "INFEAS" if tn < 0 else f"{tn:.2f}"
        lines.append(f"  {bound:<10s} {vals[bound]:8.2f} {shown:>9s} "
                     f"{ref:9.2f} {best_n:8.2f}")
        if tn < 0:
            tangent_trouble.append(bound)
    if tangent_trouble:
        lines.append(f"  ERROR with tangent_ref={sc.tangent_ref}, the tangent cap for "
                     f"{', '.join(tangent_trouble)} is NEGATIVE at mu=0. Since the "
                     f"accumulator is non-negative, every cell is infeasible after "
                     f"one mission -- the whole model will come back INFEASIBLE. "
                     f"This is a linearisation-point problem, not a property of the "
                     f"bound: bernstein's cap has c1 = -b/3, so g(0.5*tau) < 0. Use "
                     f"--tangent-ref near the operating mu (see 'best ref' above).")
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
              f"{'idle':>6s} "
              f"{'mu_max':>9s} {'n_max':>8s} {'P(D>tau)':>10s}")
    lines += [header, "-" * len(header)]
    for bound, impl in combos:
        if not _mine(opts):
            continue
        variant = sc.variant(reliability_impl=impl)
        print(f"  solving {bound}/{impl} ...", flush=True)
        run.note_progress(f"START base {bound}/{impl}")
        rec, data = run_case(variant, bound, opts,
                             solver_log_path(run, f"base_{bound}_{impl}"))
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
                     f"{_fmt(rec.get('n_idle'), 0):>6s} "
                     f"{_fmt(rec.get('mu_max'), 4):>9s} "
                     f"{rec.get('n_max_analytic', float('nan')):8.2f} "
                     f"{_fmt(rec.get('mc_p_max'), 5):>10s}")
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
        if not any((b, impl) in vals for b in IMPL_AWARE_BOUNDS):
            continue                             # no impl-aware run in this group
        per = {b: cost_for(vals, b, impl) for b in BOUNDS_ORDER}
        per_gaps = {b: gaps.get((b, impl), gaps.get((b, "exact"))) for b in BOUNDS_ORDER}
        verdict, issues = order_verdict(per, per_gaps)
        holds = verdict == "HOLDS"
        all_ok = all_ok and verdict != "VIOLATED"
        lines.append(f"hypothesis (H1) with impl={impl}: {verdict}")
        lines += [f"  {i}" for i in issues]
        summary["H1"][impl] = {"verdict": verdict, "holds": bool(holds),
                               "issues": issues,
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
    mc = [(r["bound"], r.get("mc_p_max")) for r in run.rows
          if isinstance(r.get("mc_p_max"), float)]
    if mc:
        lines.append("")
        lines.append(f"empirical P(D>tau) of the optimised schedule vs eps="
                     f"{sc.epsilon} (Monte Carlo, {run.rows[0].get('mc_samples')} "
                     f"samples). A tighter bound should sit CLOSER to eps: that is "
                     f"the same claim as the cost ordering, seen from the "
                     f"probability side.")
        for bound, pmax in mc:
            slack = sc.epsilon / pmax if pmax else math.inf
            lines.append(f"  {bound:<10s} P={pmax:.5f}   "
                         f"{'unused risk budget x%.0f' % slack if math.isfinite(slack) else 'no exceedance observed'}")
        if any(p > sc.epsilon for _, p in mc):
            lines.append("  ERROR at least one bound EXCEEDS eps -- the chance "
                         "constraint is not being honoured. That is a model bug, "
                         "not conservatism.")

    worst = max((r.get("mip_gap") or 0.0) for r in run.rows) if run.rows else 0.0
    if worst > 0.01:
        lines.append(f"  WARNING largest MIP gap in this table is {worst:.1%}. Cost "
                     f"differences smaller than that cannot be trusted; check the "
                     f"per-pair UNRESOLVED notes above.")
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
        if param in SCALE_COUPLED:
            lines.append(f"  NOTE {param} is coupled to the increment scale through "
                         f"the calibration, so b_ref is frozen at its base value "
                         f"{sc.b_ref:.6g}. Otherwise the increments would move with "
                         f"the requirement and the sweep would confound the two.")
            print(f"  [sweep] {param}: freezing b_ref at {sc.b_ref:.6g}")
        for v in values:
            base_variant = sc.variant(**{param: _cast(param, v)})
            if param in SCALE_COUPLED:
                base_variant = base_variant.variant(b_ref_fixed=sc.b_ref)
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
                run.note_progress(f"START {param}={v} {bound}/{impl}")
                rec, data = run_case(variant, bound, opts,
                                     solver_log_path(run, f"{param}{v}_{bound}_{impl}"))
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
                if not any((b, impl) in vals for b in IMPL_AWARE_BOUNDS):
                    continue                     # no impl-aware run in this group
                per = {b: cost_for(vals, b, impl) for b in BOUNDS_ORDER}
                per_gaps = {b: gaps.get((b, impl), gaps.get((b, "exact")))
                            for b in BOUNDS_ORDER}
                verdict, issues = order_verdict(per, per_gaps)
                verdicts[f"{param}={v}/{impl}"] = {
                    "H1_verdict": verdict, "H1_holds": verdict == "HOLDS",
                    "issues": issues,
                    "costs": {b: _to_builtin(per[b]) for b in per}}
                costs = "  ".join(f"{b}={_fmt(per[b])}" for b in BOUNDS_ORDER)
                lines.append(f"  {param}={v} impl={impl}: (H1) {verdict}   {costs}")
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
            for fn in (plot_parameter, plot_scalability):
                path = (fn(run.rows, param, run, getattr(sc, param))
                        if fn is plot_parameter else fn(run.rows, param, run))
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
                run.note_progress(f"START {param}={v} {bound}/{impl}")
                rec, data = run_case(variant, bound, opts,
                                     solver_log_path(run, f"{param}{v}_{bound}_{impl}"))
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
        rec, data = run_case(variant, bound, opts,
                             solver_log_path(run, f"impl_{bound}_{impl}"))
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
                rec, data = run_case(variant, bound, opts,
                                     solver_log_path(run, f"pwl{n}_{bound}"))
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


def _scalarize(value, default: str = "unspecified") -> str:
    """Collapse a per-cell selector array to one label.

    `config.load_config` normalises `model`, `bound_method` and `repair_model`
    into (F, L) string arrays -- one per cell -- because a fleet may be
    heterogeneous. Using such an array in a boolean context raises
    "truth value of an array with more than one element is ambiguous", so it has
    to be reduced explicitly. A uniform fleet gives one label; a mixed one is
    reported as such rather than silently picking a cell.
    """
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.ndim == 0:
        return str(arr.item()) or default
    vals = sorted({str(q) for q in arr.ravel().tolist() if str(q) != ""})
    if not vals:
        return default
    return vals[0] if len(vals) == 1 else "mixed:" + "+".join(vals)


def resolve_case_path(name: str, root: Path) -> Path:
    """`--case demo` -> input/demo.yaml, or an explicit path, whichever exists."""
    cand = [Path(name)]
    if not Path(name).suffix:
        cand += [Path(name).with_suffix(ext) for ext in (".yaml", ".yml", ".json")]
    cand += [root / c for c in list(cand)]
    for q in cand:
        if q.is_file():
            return q
    raise SystemExit(f"case file not found for {name!r}. Looked for: "
                     + ", ".join(str(q) for q in cand))


# Four values, but only TWO models: the string flattens a 2x2 grid of
# (encoding, assembly).  'sparse' is the 'indicator' program and 'bigm_sparse'
# the 'bigm' program, both assembled through the matrix API (rainflow_sparse).
# Within an encoding the objective, node count and LP bound must match EXACTLY,
# not merely within the gap -- test_sparse_version.py is what checks that in
# detail.  (H4) below is about the encoding axis, so read it across
# indicator/bigm and treat the sparse twins as replicates.
# The harness's `formulation` axis is WIDER than base.FORMULATIONS, which is a
# 2x2 grid of (encoding, assembly).  There is a third, independent choice --
# whether to add the sparse strengthening of the indicator relaxation
# (rainflow_v2.add_sparse_cuts) -- and rather than turn the solver option into
# a 2x2x3 string, the harness composes a LABEL and splits it before solving.
# So `--formulations indicator,indicator_cuts` is the (H5) experiment: same
# encoding, same integer optimum, different relaxation.
FORMULATIONS_ORDER = ("indicator", "indicator_cuts_core", "indicator_cuts",
                      "sparse", "sparse_cuts_core", "sparse_cuts",
                      "bigm", "bigm_sparse")

# label -> (base.formulation, sparse_cuts level)
_VARIANTS = {
    "indicator":           ("indicator",   "off"),
    "indicator_cuts_core": ("indicator",   "core"),
    "indicator_cuts":      ("indicator",   "full"),
    "sparse":              ("sparse",      "off"),
    "sparse_cuts_core":    ("sparse",      "core"),
    "sparse_cuts":         ("sparse",      "full"),
    "bigm":                ("bigm",        "off"),
    "bigm_sparse":         ("bigm_sparse", "off"),
}
# the inverse, so --sparse-cuts and a variant label are two spellings of one
# thing rather than two mutually exclusive options
_COMPOSE = {v: k for k, v in _VARIANTS.items()}


def compose_variant(formulation: str, cuts) -> str:
    """('indicator', 'full') -> 'indicator_cuts'.

    Exists so ``--formulation indicator --sparse-cuts full`` and
    ``--formulation indicator_cuts`` mean the same thing.  Reaching for a
    ``--sparse-cuts`` flag is the obvious move, and having it silently not
    exist cost a 12-shard submission.
    """
    from fleet_management.degradation_model.rainflow_v2 import sparse_cut_level
    level = sparse_cut_level(cuts)
    base_form, own = split_variant(formulation)
    if level == "off":
        return formulation
    if own not in ("off", level):
        raise SystemExit(
            f"--formulation {formulation!r} already implies sparse_cuts="
            f"{own!r}, which contradicts --sparse-cuts {level!r}. Give one or "
            f"the other.")
    key = (base_form, level)
    if key not in _COMPOSE:
        # the strengthening only exists for the indicator encoding; the big-M
        # rows already imply every one of its inequalities
        print(f"[warn] --sparse-cuts {level} has no effect under "
              f"formulation={formulation!r} (the big-M rows already imply the "
              f"cuts); ignoring it.")
        return formulation
    return _COMPOSE[key]


def split_variant(label: str) -> tuple[str, str]:
    """'indicator_cuts' -> ('indicator', 'full').

    Keeps the solver's `formulation` option meaning exactly what it means in
    `base.FORMULATIONS` (encoding x assembly) while letting the harness treat
    the strengthening as a third value on the same axis, which is what makes it
    comparable in one (H4)/(H5) table.
    """
    try:
        return _VARIANTS[str(label).strip().lower()]
    except KeyError:
        raise SystemExit(f"unknown formulation {label!r}; "
                         f"pick from {FORMULATIONS_ORDER}") from None


def test_formulation(sc: Scenario, opts, run: TestRun, impls,
                     formulations) -> tuple[list, dict]:
    """(H4): does the big-M / substituted encoding buy anything?

    `formulation='indicator'` and `formulation='bigm'` describe the SAME integer
    feasible set -- they differ only in how the logical constraints are written
    (see `rainflow_v2`).  Two things therefore have to be reported separately:

      * **equivalence** -- the optimal objectives must agree to within the MIP
        gap.  A disagreement is a bug in a big-M constant, not a result.  Run
        this with a small --mip-gap (1e-6) before trusting any timing.
      * **the trade** -- the big-M model has fewer binaries (no `nb`) but more
        rows, and its rows are *in* the LP relaxation, which the indicator ones
        are not.  So compare, per bound: model size, root/objective bound, node
        count and solve time.

    `obj_bound` at optimality is the objective itself and says nothing about the
    relaxation; the informative column when both models solve to optimality is
    `nodes`.  Use `run_studies.py --studies scaling` for the honest LP-gap
    measurement, which solves the pure relaxation on a copy of the model.
    """
    lines = [f"base case: {sc.label()}",
             f"formulations: {list(formulations)}   implementations: {list(impls)}",
             ""]
    combos = bound_impl_combos(impls, announce=True)
    vals, sizes, times, nodes, bounds_, builds = {}, {}, {}, {}, {}, {}
    header = (f"{'bound':<10s} {'impl':<9s} {'form':<12s} {'status':<12s} "
              f"{'cost':>12s} {'gap':>9s} {'build[s]':>9s} {'time[s]':>9s} "
              f"{'nodes':>9s} {'bin':>7s} {'constrs':>9s} {'gencon':>7s}")
    lines += [header, "-" * len(header)]
    for bound, impl in combos:
        for form in formulations:
            if not _mine(opts):
                continue
            variant = sc.variant(reliability_impl=impl, formulation=form)
            print(f"  [formulation] {bound}/{impl}/{form} ...", flush=True)
            rec, data = run_case(variant, bound, opts,
                                 solver_log_path(run, f"form_{bound}_{impl}_{form}"))
            rec.update({"test": "formulation", "parameter": "formulation",
                        "value": form})
            run.add(rec, data, variant)
            key = (bound, impl, form)
            vals[key] = rec.get("objective", math.nan)
            times[key] = rec.get("runtime_s", math.nan)
            nodes[key] = rec.get("nodes", math.nan)
            bounds_[key] = rec.get("obj_bound", math.nan)
            builds[key] = rec.get("build_s", math.nan)
            sizes[key] = (rec.get("n_bin"), rec.get("n_constrs"),
                          rec.get("n_genconstrs"))
            lines.append(f"{bound:<10s} {impl:<9s} {form:<12s} "
                         f"{str(rec.get('status'))[:12]:<12s} "
                         f"{_fmt(rec.get('objective')):>12s} "
                         f"{_fmt(rec.get('mip_gap'), 3):>9s} "
                         f"{_fmt(rec.get('build_s'), 3):>9s} "
                         f"{_fmt(rec.get('runtime_s'), 2):>9s} "
                         f"{_fmt(rec.get('nodes'), 0):>9s} "
                         f"{_fmt(rec.get('n_bin'), 0):>7s} "
                         f"{_fmt(rec.get('n_constrs'), 0):>9s} "
                         f"{_fmt(rec.get('n_genconstrs'), 0):>7s}")
    lines.append("")

    tol = max(float(getattr(opts, "mip_gap", 0.0) or 0.0), 1e-6)
    summary = {"H4": {}}
    for bound, impl in combos:
        have = [f for f in FORMULATIONS_ORDER if (bound, impl, f) in vals]
        if len(have) < 2:
            continue
        costs = [vals[(bound, impl, f)] for f in have]
        finite = [c for c in costs if isinstance(c, float) and math.isfinite(c)]
        if len(finite) == len(costs) and finite:
            spread = (max(finite) - min(finite)) / max(1e-12, abs(min(finite)))
            agree = spread <= 2.0 * tol + 1e-9
        else:                       # infeasible must be infeasible on both sides
            agree = len({("inf" if c == math.inf else "fin"
                          if isinstance(c, float) and math.isfinite(c) else "nan")
                         for c in costs}) == 1
            spread = math.nan
        tag = "AGREE" if agree else "MISMATCH"
        # An encoding and its sparse twin are the SAME program, so AGREE here is
        # true by construction and proves nothing. Say so, rather than letting a
        # green tag look like evidence.
        pairs = [(a, b) for a, b in (("indicator", "sparse"),
                                     ("bigm", "bigm_sparse"))
                 if a in have and b in have]
        for a, b in pairs:
            ta, tb = builds.get((bound, impl, a)), builds.get((bound, impl, b))
            speed = (f"{ta / tb:.2f}x faster to build"
                     if all(isinstance(q, float) and math.isfinite(q) and q > 0
                            for q in (ta, tb)) else "build time unavailable")
            lines.append(
                f"     note: {a} and {b} are the same program in two "
                f"assemblies, so AGREE is true by construction and says "
                f"nothing. What IS informative: {b} is {speed}, while nodes "
                f"and runtime must match. Use test_sparse_version.py for the "
                f"structural comparison.")
        cost_txt = "  ".join(f"{f}={_fmt(vals[(bound, impl, f)])}" for f in have)
        lines.append(f"(H4) {bound}/{impl}: {tag}   {cost_txt}")
        lines.append("     time: " + "  ".join(
            f"{f}={_fmt(times[(bound, impl, f)], 2)}s" for f in have)
            + "   nodes: " + "  ".join(
            f"{f}={_fmt(nodes[(bound, impl, f)], 0)}" for f in have))
        lines.append("     build: " + "  ".join(
            f"{f}={_fmt(builds[(bound, impl, f)], 3)}s" for f in have))
        lines.append("     size (bin/constrs/gencon): " + "  ".join(
            f"{f}={sizes[(bound, impl, f)]}" for f in have))
        if not agree:
            lines.append("     VIOLATION the two encodings must have the same "
                         "integer optimum. Suspect a big-M that is too small "
                         "(raise --bigM or check the state upper bounds) before "
                         "suspecting the model.")
        ti, tb = times.get((bound, impl, "indicator")), times.get((bound, impl, "bigm"))
        if all(isinstance(q, float) and math.isfinite(q) and q > 0 for q in (ti, tb)):
            lines.append(f"     speed-up of bigm over indicator: {ti / tb:.2f}x")
        summary["H4"][f"{bound}/{impl}"] = {
            "agree": bool(agree), "cost_spread_rel": _to_builtin(spread),
            "costs": {f: _to_builtin(vals[(bound, impl, f)]) for f in have},
            "times": {f: _to_builtin(times[(bound, impl, f)]) for f in have},
            "nodes": {f: _to_builtin(nodes[(bound, impl, f)]) for f in have},
            "obj_bound": {f: _to_builtin(bounds_[(bound, impl, f)]) for f in have},
            "size_bin_constr_gencon": {f: _to_builtin(sizes[(bound, impl, f)])
                                       for f in have}}
    lines.append("")
    if getattr(opts, "shard_obj", None) is not None:
        lines.append(f"shard {opts.shard_obj}: {len(run.rows)} run(s); (H4) is "
                     f"evaluated by --merge once every shard has finished.")
        return lines, {"shard": str(opts.shard_obj), "n_runs": len(run.rows)}
    n_bad = sum(1 for v in summary["H4"].values() if not v["agree"])
    lines.append(f"(H4) the two encodings agreed on "
                 f"{len(summary['H4']) - n_bad}/{len(summary['H4'])} "
                 f"(bound, impl) pair(s).")
    return lines, summary


def test_case(sc: Scenario, opts, run: TestRun, cases: list, in_root: Path
              ) -> tuple[list, dict]:
    """Solve one or more hand-written input files from `input/` as they are.

    Unlike the other tests this does not generate a scenario: the YAML is the
    experiment. It is read with the project's own reader, normalised by
    `config.load_config`, and dispatched by `solver._solve_mixed`, so gamma,
    rainflow and mixed fleets all work and the bound / encoding come from the
    file rather than from the CLI.

    Run options (`mip_gap`, `time_limit`, `gurobi_params`, ...) are injected into
    the input ONLY where the file does not already set them: the file wins, since
    the point of this test is to run the case as written.

    Note `--verify-mc` does not apply here. The Monte Carlo needs the increment
    LAW, and a case file supplies only descriptors; reconstructing a law from them
    would be an assumption the file never made.
    """
    # dry-run only needs the validator, so the solver (and gurobipy) is imported
    # lazily -- the same discipline the other tests use
    if opts.dry_run:
        load_config, solve_mixed, read_input = _import_config(), None, None
    else:
        load_config, solve_mixed, read_input = _import_dispatch()
    lines = [f"input directory: {in_root}",
             "the input file is authoritative: CLI run options fill gaps only, so "
             "the mip_gap / time_limit in the report header above are NOT "
             "necessarily what ran -- see 'options in effect' per case.", ""]
    header = (f"{'case':<22}{'model':<10}{'bound':<11}{'status':<14}"
              f"{'cost':>12}{'gap':>9}{'time[s]':>9}{'nodes':>11}")
    lines += [header, "-" * len(header)]
    summary = {}

    for name in cases:
        if not _mine(opts):
            continue
        path = resolve_case_path(name, in_root)
        data = (read_input(path) if read_input is not None
                else yaml.safe_load(path.read_text()))
        if not isinstance(data, dict):
            raise SystemExit(f"{path} did not parse to a mapping")

        # CLI run options fill gaps only; anything the file sets is preserved
        injected = {}
        for key, val in (("mip_gap", opts.mip_gap),
                         ("time_limit", opts.time_limit),
                         ("verbose", opts.verbose)):
            if key not in data and val is not None:
                data[key] = val; injected[key] = val
        log_path = solver_log_path(run, name)
        gp = _gurobi_params(opts, log_path)
        if gp and "gurobi_params" not in data:
            data["gurobi_params"] = gp; injected["gurobi_params"] = gp
        elif gp and log_path is not None:
            # the file set its own gurobi_params: merge the log keys in rather
            # than losing the log
            merged = dict(data.get("gurobi_params") or {})
            merged.update({k: gp[k] for k in
                           ("LogFile", "OutputFlag", "LogToConsole") if k in gp})
            data["gurobi_params"] = merged

        print(f"  [case] {path} ...", flush=True)
        run.note_progress(f"START case {name} ({path})")
        rec = {"timestamp": datetime.now().isoformat(timespec="seconds"),
               "test": "case", "parameter": "case", "value": name,
               "bound": str(data.get("bound_method") or "unspecified"),
               "reliability_impl": str(data.get("reliability_impl", "exact")),
               "repair_model": str(data.get("repair_model", "?")),
               "F": data.get("F"), "M": data.get("M"), "L": data.get("L", 1),
               "H": data.get("H"), "T": 2 * int(data["H"]) if "H" in data else None,
               "tau": data.get("tau"), "epsilon": data.get("epsilon"),
               "rho": data.get("rho"),
               "threads": getattr(opts, "threads", None) or "",
               "gurobi_params": ",".join(f"{k}={v}" for k, v in
                                         sorted((gp or {}).items())),
               "solver_log": (log_path.name if log_path is not None else ""),
               "host": _HOSTNAME, "slurm_job": _SLURM_JOB,
               "git_branch": _GIT["git_branch"], "git_commit": _GIT["git_commit"]}

        t0 = time.perf_counter()
        if opts.dry_run:
            load_config(data)                    # validate only
            rec.update({"status": "dry_run", "objective": math.nan})
            run.add(rec, data, sc)
            lines.append(f"{name[:21]:<22}{str(data.get('model'))[:9]:<10}"
                         f"{rec['bound'][:10]:<11}{'dry_run':<14}"
                         f"{'-':>12}{'-':>9}{'-':>9}{'-':>11}")
            continue
        try:
            cfg = load_config(data)
            # the normalised config is authoritative: a file may leave the bound
            # implicit, or name it somewhere load_config resolves
            # these are (F, L) arrays on the config, one entry per cell
            rec["bound"] = _scalarize(getattr(cfg, "bound_method", None),
                                      str(data.get("bound_method") or "unspecified"))
            rec["repair_model"] = _scalarize(getattr(cfg, "repair_model", None),
                                             str(data.get("repair_model")
                                                 or "unspecified"))
            rec["model"] = _scalarize(getattr(cfg, "model", None)
                                      if hasattr(cfg, "model")
                                      else getattr(cfg, "models", None),
                                      str(data.get("model") or "unspecified"))
            # what the solver will ACTUALLY use, after the file and the CLI merge
            eff = dict(getattr(cfg, "options", {}) or {})
            rec["req_mip_gap"] = eff.get("mip_gap")
            rec["req_time_limit"] = eff.get("time_limit")
            rec["req_verbose"] = eff.get("verbose")
            rec["reliability_impl"] = str(eff.get("reliability_impl", "exact"))
            res = solve_mixed(cfg)
        except Exception as exc:
            # Keep the traceback. A bare "TypeError: 'NoneType' object is not
            # subscriptable" says nothing about WHERE, and a case can take hours to
            # reach the failure -- the location has to survive the handler.
            tb = traceback.format_exc()
            rec.update({"status": f"error: {type(exc).__name__}: {exc}",
                        "objective": math.nan,
                        "wall_s": time.perf_counter() - t0,
                        "traceback": tb})
            run.add(rec, data, sc)
            (run.dir / f"traceback_{_safe_filename(name)}.txt").write_text(tb)
            lines.append(f"{name[:21]:<22}{'-':<10}{'-':<11}{'ERROR':<14}")
            lines.append(f"    {type(exc).__name__}: {exc}")
            for frame in traceback.extract_tb(exc.__traceback__)[-3:]:
                lines.append(f"      at {Path(frame.filename).name}:{frame.lineno} "
                             f"in {frame.name}(): {(frame.line or '').strip()[:70]}")
            lines.append(f"      full traceback: "
                         f"traceback_{_safe_filename(name)}.txt")
            continue
        wall = time.perf_counter() - t0

        obj = res.get("objective")
        rec.update({
            "status": res.get("status"),
            "objective": (math.inf if res.get("status") == "infeasible"
                          else (float(obj) if obj is not None else math.nan)),
            "mip_gap": _f(res.get("mip_gap")), "obj_bound": _f(res.get("bound")),
            "wall_s": wall, "degradation": res.get("degradation"),
            # Empty when the file named no encoding, i.e. when solver.py took
            # the legacy rainflow.py route (indicator, per-cell loop).
            "formulation": res.get("formulation") or "",
            "build_s": _f(res.get("build_s")),
        })
        md = res.get("model")
        if md is not None:
            rec.update(collect_model_metrics(md))
        for key, arr, red in (("n_repairs", res.get("m"), np.sum),
                              ("n_replacements", res.get("r"), np.sum),
                              ("mu_max", res.get("mu"), np.max),
                              ("v_max", res.get("v"), np.max)):
            rec[key] = float(red(arr)) if arr is not None else math.nan
        x = res.get("x")
        rec.update(_assignment_counts(x, res.get("m"), res.get("r")))

        _dump_schedule(run, name, res)
        if md is not None:
            try:
                md.dispose()
            except Exception:
                pass
        run.add(rec, data, sc)
        summary[name] = {"input": str(path), "injected_options": _to_builtin(injected),
                         "result": _to_builtin(rec)}
        lines.append(f"{name[:21]:<22}{rec.get('model', '-')[:9]:<10}"
                     f"{rec['bound'][:10]:<11}{str(rec['status'])[:13]:<14}"
                     f"{_fmt(rec.get('objective')):>12}{_fmt(rec.get('mip_gap'), 4):>9}"
                     f"{_fmt(rec.get('runtime_s'), 1):>9}"
                     f"{_fmt(rec.get('nodes'), 0):>11}")
        # be explicit about which options actually applied: the file wins, so the
        # header's CLI values are NOT what ran
        src = {k: ("CLI" if k in injected else "file")
               for k in ("mip_gap", "time_limit", "verbose")}
        lines.append(f"    options in effect: "
                     f"mip_gap={rec.get('req_mip_gap')} ({src['mip_gap']})  "
                     f"time_limit={rec.get('req_time_limit')} ({src['time_limit']})  "
                     f"verbose={rec.get('req_verbose')} ({src['verbose']})  "
                     f"impl={rec.get('reliability_impl')}  "
                     f"repair={rec.get('repair_model')}  "
                     # Which encoding x assembly the FILE selected. A case file
                     # with no 'formulation' key takes the legacy rainflow.py
                     # builder (indicator, per-cell loop); any of the four
                     # values routes to rainflow_v2 / rainflow_sparse instead.
                     # Worth printing, because the two legacy-vs-v2 indicator
                     # builds are the same model and would otherwise be
                     # indistinguishable in this report.
                     f"formulation={rec.get('formulation') or 'legacy (indicator)'}  "
                     f"build={_fmt(rec.get('build_s'), 3)}s")
    lines.append("")
    lines.append("full metrics are in results.csv / results.yaml; the schedule and "
                 "state trajectories are in schedule_<case>.csv")
    return lines, {"cases": summary}


def _activity_of(x, i: int, k: int, Jp1: int) -> tuple:
    """Label vehicle i's assignment at step k as (activity, mission index).

    The base assignment constraint is ``sum_j x[i,j,k] <= 1``, NOT ``== 1``, so
    an unassigned vehicle is a legal and meaningful outcome: it rests without
    accruing damage and without paying the C_M depot charge.  That is an *idle*
    step and it is not the same thing as a *depot* step (``x[i,0,k] == 1``),
    which costs C_M and is what `base.add_maintenance_gating` gates repair and
    replacement on.  Collapsing the two -- i.e. reading "no mission" as "depot"
    -- makes idling invisible in the dump and makes an optimal schedule look
    like it is burning depot days for nothing.

    A mission takes precedence if the solution somehow assigns more than one
    activity (only reachable through integrality tolerances or a model bug), so
    a corrupt solution still dumps rather than raising here.
    """
    j = next((q for q in range(1, Jp1) if x[i, q, k] > 0.5), None)
    if j is not None:
        return f"mission_{j}", j
    if x[i, 0, k] > 0.5:
        return "depot", 0
    return "idle", 0


def _dump_schedule(run: TestRun, name: str, res: dict) -> None:
    """Long-format schedule + state trajectory, one row per (vehicle, comp, step).

    This is the artefact a plot script wants: activity, repair/replace flags and
    the mean/variance trajectory, without having to reload the solver.

    ``activity`` is one of ``mission_<j>``, ``depot`` (x[i,0,k] == 1, a paid
    maintenance slot, the only step at which repair/replace may fire) or
    ``idle`` (no assignment at all).  ``mission`` is 0 for both depot and idle,
    so read ``activity`` -- not ``mission`` -- to tell them apart.
    """
    x, m, r = res.get("x"), res.get("m"), res.get("r")
    mu, v = res.get("mu"), res.get("v")
    if x is None or mu is None:
        return
    F, Jp1, T = x.shape
    L = mu.shape[1]
    path = run.dir / f"schedule_{_safe_filename(name)}.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["vehicle", "component", "step", "activity", "mission",
                    "repair", "replace", "mu", "v"])
        for i in range(F):
            for k in range(T):
                act, j = _activity_of(x, i, k, Jp1)
                for l in range(L):
                    w.writerow([i, l, k, act, j,
                                int(m[i, l, k] > 0.5) if m is not None else "",
                                int(r[i, l, k] > 0.5) if r is not None else "",
                                float(mu[i, l, k]),
                                float(v[i, l, k]) if v is not None else ""])


def _uninformative(rows: list[dict]) -> bool:
    """True when every bound gave the same intervention pattern."""
    sigs = {(r.get("n_repairs"), r.get("n_depot"), r.get("n_idle"),
             r.get("n_replacements"))
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
               "n_depot", "n_idle", "n_depot_noop",
               "mu_max", "v_max", "n_vars", "n_constrs", "n_bin",
               "nodes", "tau", "epsilon", "rho", "p", "b_ref", "mu_ref",
               "s_chernoff", "tangent_ref", "bigM", "C_M", "C_R", "C_S", "C_P")
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
    """Combine the shard files of ONE run folder, then check and plot once.

    A shard only holds a slice of the design points, so no shard can evaluate
    (H1)/(H2)/(H3) -- those need every bound at a point. This reads the shard CSVs
    back, concatenates them, and writes merged_* files into the same folder.

    Which folder: --run-stamp if given, else the newest `*_<test>` under --out.
    Working inside one folder is what stops a failed earlier attempt from being
    silently mixed into the results.
    """
    stamp = (getattr(opts, "run_stamp", None) or os.environ.get("RUN_STAMP", "")).strip()
    folders = sorted(d for d in out_root.glob(f"*_{test}") if d.is_dir())
    if stamp:
        folders = [d for d in folders if d.name.startswith(stamp)]
    if not folders:
        print(f"[merge] no run folder matching {out_root}/"
              f"{stamp or '*'}_{test}", file=sys.stderr)
        return 1
    folder = folders[-1]                          # names sort chronologically
    if len(folders) > 1 and not stamp:
        print(f"[merge] {len(folders)} run folders exist; using the newest: "
              f"{folder.name}")
        print(f"[merge] pass --run-stamp YYYYMMDDHHMM to pick another")

    csvs = sorted(folder.glob("results_shard*.csv")) or sorted(
        q for q in folder.glob("results.csv"))
    if not csvs:
        print(f"[merge] {folder.name} contains no results*.csv", file=sys.stderr)
        return 1

    rows: list = []
    empty = []
    for q in csvs:
        got = _read_rows(q)
        print(f"[merge] {folder.name}/{q.name}: {len(got)} rows")
        if not got:
            empty.append(q.name)
        rows += got
    if not rows:
        print(f"\n[merge] ERROR every one of the {len(csvs)} shard file(s) is "
              f"header-only, so there is nothing to merge.", file=sys.stderr)
        print("[merge] The solves never ran. Check, in this order:", file=sys.stderr)
        print(f"  1. sacct -j <arrayjobid> --format=JobID%18,State,ExitCode",
              file=sys.stderr)
        print(f"  2. tail -n 40 logs/bound_tests_<arrayjobid>_0.err", file=sys.stderr)
        print(f"  3. {folder}/progress_shard*.log  -- how far each task got",
              file=sys.stderr)
        return 1
    if empty:
        print(f"[merge] WARNING {len(empty)} shard file(s) produced no rows: "
              f"{', '.join(empty)}", file=sys.stderr)

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

    # duck-types TestRun for the plot helpers (dir / stem / rows)
    run = SimpleNamespace(dir=folder, stem="merged", rows=rows)

    csv_path = folder / "merged_results.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for rec in sorted(rows, key=lambda r: (str(r.get("parameter")),
                                               str(r.get("value")),
                                               BOUNDS_ORDER.index(r["bound"])
                                               if r.get("bound") in BOUNDS_ORDER else 99)):
            w.writerow({k: rec.get(k, "") for k in FIELDS})

    report = [f"# merged report  {datetime.now():%Y-%m-%d %H:%M:%S}",
              f"# run folder={folder.name}  shard files={len(csvs)}  rows={len(rows)}",
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
    gaps = [r["mip_gap"] for r in rows
            if isinstance(r.get("mip_gap"), float) and math.isfinite(r["mip_gap"])]
    if gaps:
        gaps_sorted = sorted(gaps)
        report.append(f"mip gap: median {gaps_sorted[len(gaps_sorted) // 2]:.4g}  "
                      f"max {max(gaps):.4g}  ({sum(1 for g in gaps if g > 0.01)} of "
                      f"{len(gaps)} runs above 1%)")
    missing = [r for r in rows if str(r.get("status")).startswith("error")
               or classify(r) == "unknown"]
    if missing:
        report.append(f"WARNING {len(missing)} run(s) did not produce a solution "
                      f"(errors or time limits) -- the checks below treat those as "
                      f"missing, not as failures.")
    report.append("")

    summary = {"n_rows": len(rows), "shard_files": len(csvs),
               "run_folder": folder.name, "status_counts": statuses,
               "code_versions": sorted(f"{b}@{c}" for b, c in versions)}

    # ---- case runs: no hypothesis, just a table --------------------------
    if test == "case":
        report.append("case results (each row is a hand-written input file; there "
                      "is no bound ordering to test here)")
        hdr = (f"{'case':<24}{'model':<10}{'bound':<11}{'impl':<9}{'status':<13}"
               f"{'cost':>12}{'gap':>9}{'time[s]':>9}{'nodes':>11}")
        report += [hdr, "-" * len(hdr)]
        for rec in sorted(rows, key=lambda r: str(r.get("value"))):
            report.append(
                f"{str(rec.get('value'))[:23]:<24}"
                f"{str(rec.get('model', '-'))[:9]:<10}"
                f"{str(rec.get('bound', '-'))[:10]:<11}"
                f"{impl_of_record(rec)[:8]:<9}"
                f"{str(rec.get('status'))[:12]:<13}"
                f"{_fmt(rec.get('objective')):>12}{_fmt(rec.get('mip_gap'), 4):>9}"
                f"{_fmt(rec.get('runtime_s'), 1):>9}"
                f"{_fmt(rec.get('nodes'), 0):>11}")
        summary["cases"] = {str(r.get("value")): _to_builtin(r) for r in rows}
        _dump_yaml(folder / "merged_results.yaml",
                   {"test": test, "run_folder": folder.name,
                    "merged_from": [q.name for q in csvs],
                    "created": datetime.now().isoformat(timespec="seconds"),
                    "summary": summary, "runs": [_to_builtin(r) for r in rows]})
        (folder / "merged_summary.txt").write_text("\n".join(report))
        print("\n".join(report))
        print(f"\n[merge] folder : {folder}")
        return 0

    # ---- (H1)/(H3) per design point -------------------------------------
    impls_seen = [im for im in IMPLS_ORDER
                  if any(impl_of_record(r) == im and r.get("bound") in IMPL_AWARE_BOUNDS
                         for r in rows)]
    if not impls_seen:
        impls_seen = [im for im in IMPLS_ORDER
                      if any(impl_of_record(r) == im for r in rows)]
    points = {}
    n_bad = n_incon = n_proven = 0
    pair_stats = {f"{a}>={b}": {"proven": 0, "overlap": 0, "disproven": 0}
                  for a, b in zip(BOUNDS_ORDER, BOUNDS_ORDER[1:])}
    groups = {}
    for rec in rows:
        if rec.get("test") == "failure":
            continue
        groups.setdefault((rec.get("parameter"), str(rec.get("value"))), []).append(rec)
    for (param, value), recs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        vals = {(r["bound"], impl_of_record(r)): r["objective"] for r in recs}
        gaps_d = {(r["bound"], impl_of_record(r)): r["mip_gap"] for r in recs}
        for impl in impls_seen:
            per = {b: cost_for(vals, b, impl) for b in BOUNDS_ORDER}
            per_gaps = {b: gaps_d.get((b, impl), gaps_d.get((b, "exact")))
                        for b in BOUNDS_ORDER}
            per_lb = {b: next((r["obj_bound"] for r in recs
                               if r["bound"] == b
                               and impl_of_record(r) in (impl, "exact")), math.nan)
                      for b in BOUNDS_ORDER}
            verdict, issues = interval_order(per, per_lb)
            n_bad += 1 if verdict == "DISPROVEN" else 0
            n_incon += 1 if verdict.startswith("PARTIAL") else 0
            n_proven += 1 if verdict == "PROVEN" else 0
            costs = "  ".join(f"{b}={_fmt(per[b])}" for b in BOUNDS_ORDER)
            report.append(f"{param}={value} impl={impl}: (H1) {verdict}   {costs}")
            report += [f"    {i}" for i in issues]
            points[f"{param}={value}/{impl}"] = {"H1_verdict": verdict,
                                                 "issues": issues}
        for bound in BOUNDS_ORDER:
            if sum(1 for im in IMPLS_ORDER if (bound, im) in vals) < 2:
                continue
            issues = check_impl_order(vals, gaps_d, bound)
            report += [f"    (H3) {i}" for i in issues]
    summary["H1_violations"] = n_bad
    summary["H1_inconclusive"] = n_incon
    summary["H1_proven"] = n_proven
    summary["points"] = points
    report.append("")
    report.append("(H1) by rigorous interval separation -- a pair is PROVEN when "
                  "LB(looser) >= UB(tighter), which holds however wide the MIP gaps "
                  "are. Overlap means unresolved, not refuted.")
    report.append(f"  design points: {n_proven} fully proven, {n_incon} partial, "
                  f"{n_bad} disproven")
    if n_incon:
        report.append(f"WARNING {n_incon} design point(s) are INCONCLUSIVE: the MIP "
                      f"gap is larger than the cost differences between adjacent "
                      f"bounds, so those pairs cannot be ranked. Tighten --mip-gap, "
                      f"or use a MILP encoding (--impls tangent,pwl) where the gap "
                      f"actually closes.")
    report.append("")

    # ---- (H4) formulation equivalence, across shards --------------------
    form_rows = [r for r in rows if r.get("test") == "formulation"
                 or r.get("parameter") == "formulation"]
    if form_rows:
        report.append("(H4) the two MILP encodings describe the same integer "
                      "feasible set, so their optima must agree to within the MIP "
                      "gap. A MISMATCH is a wrong big-M, not a result.")
        by_pair = {}
        for rec in form_rows:
            key = (rec.get("bound"), impl_of_record(rec))
            by_pair.setdefault(key, {})[str(rec.get("formulation")
                                            or rec.get("value"))] = rec
        n_agree = n_mismatch = 0
        for (bound, impl), per_form in sorted(by_pair.items(), key=lambda kv: str(kv[0])):
            forms = [f for f in FORMULATIONS_ORDER if f in per_form]
            if len(forms) < 2:
                continue
            costs = [per_form[f].get("objective") for f in forms]
            gaps_f = [per_form[f].get("mip_gap") for f in forms]
            tol = max([g for g in gaps_f
                       if isinstance(g, float) and math.isfinite(g)] or [0.0])
            finite = [c for c in costs
                      if isinstance(c, float) and math.isfinite(c)]
            if len(finite) == len(costs) and finite:
                spread = (max(finite) - min(finite)) / max(1e-12, abs(min(finite)))
                agree = spread <= 2.0 * tol + 1e-6
            else:                       # infeasible must be infeasible both sides
                agree = len({classify(per_form[f]) for f in forms}) == 1
                spread = math.nan
            n_agree += agree
            n_mismatch += (not agree)
            txt = "  ".join(f"{f}={_fmt(per_form[f].get('objective'))}" for f in forms)
            report.append(f"  {bound}/{impl}: {'AGREE' if agree else 'MISMATCH'}"
                          f"   {txt}")
            report.append("    time: " + "  ".join(
                f"{f}={_fmt(per_form[f].get('runtime_s'), 2)}s" for f in forms)
                + "   nodes: " + "  ".join(
                f"{f}={_fmt(per_form[f].get('nodes'), 0)}" for f in forms)
                + "   binaries: " + "  ".join(
                f"{f}={_fmt(per_form[f].get('n_bin'), 0)}" for f in forms))
            if not agree:
                report.append("    VIOLATION raise --bigM or check the state upper "
                              "bounds in rainflow_v2._tighten_bounds.")
        report.append(f"  {n_agree} pair(s) agreed, {n_mismatch} mismatched.")
        summary["H4_agree"] = n_agree
        summary["H4_mismatch"] = n_mismatch
        report.append("")

    # ---- (H2) from the failure rows -------------------------------------
    fail_rows = [r for r in rows if r.get("test") == "failure"]
    if fail_rows:
        ladders = {}
        for rec in fail_rows:
            ladders.setdefault(rec["parameter"], []).append(rec)
        for param, recs in ladders.items():
            ladder = STRESS_LADDERS.get(param, ("", []))[1]
            order = [v for v in ladder if str(v) in {str(r["value"]) for r in recs}] \
                or sorted({r["value"] for r in recs}, key=str)
            failed_at = {}
            for rec in recs:
                key = (rec["bound"], impl_of_record(rec))
                failed_at.setdefault(key, None)
                if classify(rec) == "infeasible":
                    idx = next((i for i, v in enumerate(order)
                                if str(v) == str(rec["value"])), None)
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
                            - {"-", "impl", "pwl_points", "case", "None", ""}):
            # 'formulation' is categorical, so the line plots interpolate
            # between values that have no midpoint; bars instead.
            made = ((plot_formulation_bars(rows, run),)
                    if param == "formulation"
                    else (plot_parameter(rows, param, run, None),
                          plot_scalability(rows, param, run)))
            for path in made:
                if path:
                    print(f"  [plot] {path.name}")
                    report.append(f"plot: {path.name}")
        for fn in (plot_impl, plot_pwl_convergence):
            path = fn(rows, run)
            if path:
                print(f"  [plot] {path.name}")
                report.append(f"plot: {path.name}")

    _dump_yaml(folder / "merged_results.yaml",
               {"test": test, "run_folder": folder.name,
                "merged_from": [q.name for q in csvs],
                "created": datetime.now().isoformat(timespec="seconds"),
                "summary": summary, "runs": [_to_builtin(r) for r in rows]})
    (folder / "merged_summary.txt").write_text("\n".join(report))
    print("\n".join(report))
    print(f"\n[merge] folder : {folder}")
    print(f"[merge] results: merged_results.csv, merged_results.yaml, "
          f"merged_summary.txt")
    return 0


# ===========================================================================
# Planner: decide the configuration before spending the allocation
# ===========================================================================
def plan_run(args, sc: Scenario, sweeps: dict, ladders: dict, tests: list) -> int:
    """Print the work matrix, the wall-clock arithmetic, and what is still open.

    Choosing a configuration is three questions, and all three have answers:
      1. How many solves does this configuration imply?
      2. Will they fit in the Slurm limit at this shard count and time limit?
      3. Which of them would actually tell you something you do not already know?
    (3) is the one people skip. With --plan-from pointing at a previous
    merged_results.csv, the planner replays the rigorous interval test and lists
    only the comparisons still unresolved, so the budget follows the uncertainty
    instead of re-proving what is already proven.
    """
    combos = bound_impl_combos(args.impl_list)
    units: list = []
    for test in tests:
        if test == "analytic":
            continue
        if test == "case":
            # one solve per input file; the bound and encoding come from the file,
            # so there is no bound x impl product here
            units += [("case", "case", nm, "file", "file")
                      for nm in getattr(args, "case_names", [])]
        elif test == "base":
            units += [("base", "-", "", b, im) for b, im in combos]
        elif test == "impl":
            units += [("impl", "impl", im, b, im) for b, im in combos]
            units += [("impl", "pwl_points", n, b, "pwl")
                      for n in args.pwl_ladder_list for b in IMPL_AWARE_BOUNDS]
        elif test == "sweep":
            for param, values in sweeps.items():
                for v in values:
                    if sc.variant(**{param: _cast(param, v)}).F <= \
                       sc.variant(**{param: _cast(param, v)}).M:
                        continue
                    units += [("sweep", param, v, b, im) for b, im in combos]
        elif test == "failure":
            for param, (_d, ladder) in ladders.items():
                # early-stop makes this a lower/upper bound, not a count
                units += [("failure", param, v, b, im)
                          for v in ladder for b, im in combos]

    print(f"\n=== work matrix ===")
    print(f"design points : {len({(u[1], str(u[2])) for u in units})}")
    print(f"bound x impl  : {len(combos)}  {combos}")
    print(f"total solves  : {len(units)}"
          + ("   (failure early-stops, so this is the worst case)"
             if "failure" in tests else ""))
    by_test = collections.Counter(u[0] for u in units)
    for t, n in by_test.items():
        print(f"    {t:<9} {n}")

    # ---- timing -----------------------------------------------------------
    tl = args.time_limit
    med = {}
    if args.plan_from:
        try:
            prev = _read_rows(Path(args.plan_from))
        except Exception as exc:
            print(f"[plan] could not read {args.plan_from}: {exc}")
            prev = []
        for r in prev:
            k = (r.get("bound"), impl_of_record(r))
            if isinstance(r.get("runtime_s"), float) and math.isfinite(r["runtime_s"]):
                med.setdefault(k, []).append(r["runtime_s"])
        med = {k: sorted(v)[len(v) // 2] for k, v in med.items()}
        if med:
            print(f"\n=== measured solve time (median, from {args.plan_from}) ===")
            for (b, im), t in sorted(med.items()):
                cap = "  (hit the old limit)" if t >= 0.98 * max(
                    (r.get("runtime_s") or 0) for r in prev) else ""
                print(f"    {b:<10}/{im:<8} {t:8.0f} s{cap}")

    def est(u):
        t = med.get((u[3], u[4]))
        return min(t, tl) if (t is not None and tl) else (tl or 600.0)

    total_s = sum(est(u) for u in units)
    worst_s = len(units) * (tl or 0)
    shards = args.plan_shards or 12
    wall_h = args.plan_wall
    print(f"\n=== wall clock at {shards} shards, --time-limit "
          f"{'none' if tl is None else int(tl)}s ===")
    print(f"    expected  {total_s / 3600:7.1f} core-h  ->  "
          f"{total_s / 3600 / shards:6.1f} h per shard")
    if tl:
        print(f"    worst case{worst_s / 3600:7.1f} core-h  ->  "
              f"{worst_s / 3600 / shards:6.1f} h per shard"
              f"   {'OK' if worst_s / 3600 / shards <= wall_h else 'EXCEEDS the ' + str(wall_h) + ' h limit'}")
        need = math.ceil(worst_s / 3600 / wall_h)
        print(f"    shards needed so the WORST case fits in {wall_h} h: {need}")
    else:
        print(f"    no per-solve limit: one hard instance can consume the whole "
              f"task. Set --time-limit for anything but the failure test.")
    # the planner above divides evenly; the harness actually assigns unit i to
    # shard i % n, so one shard can collect a disproportionate share of the slow
    # bounds. Compute the real per-shard load.
    per_shard = [0.0] * shards
    per_shard_worst = [0.0] * shards
    for idx, u in enumerate(units):
        per_shard[idx % shards] += est(u)
        per_shard_worst[idx % shards] += (tl or 0)
    hi = max(per_shard) / 3600.0
    hi_w = max(per_shard_worst) / 3600.0
    print(f"    actual busiest shard (index %% n): {hi:.1f} h expected, "
          f"{hi_w:.1f} h worst   "
          f"{'OK' if hi_w <= wall_h else 'EXCEEDS ' + str(wall_h) + ' h -- raise shards'}")
    if getattr(args, "plan_maxpar", 0):
        waves = math.ceil(shards / args.plan_maxpar)
        print(f"    at MAXPAR={args.plan_maxpar}: {waves} wave(s) -> up to "
              f"{waves * hi_w:.0f} h before the merge job starts")
    print(f"    solves per shard: {len(units) / shards:.1f}"
          + ("   (< 5 is mostly startup overhead; use fewer shards)"
             if len(units) / shards < 5 else ""))

    # ---- what is still open ----------------------------------------------
    if args.plan_from and prev:
        print(f"\n=== what is still unresolved in {args.plan_from} ===")
        UB = collections.defaultdict(dict); LB = collections.defaultdict(dict)
        for r in prev:
            k = (r.get("parameter"), str(r.get("value")))
            UB[k][r["bound"]] = r["objective"]; LB[k][r["bound"]] = r["obj_bound"]
        open_pairs = collections.Counter()
        open_points = collections.defaultdict(set)
        proven = 0
        for k in UB:
            for a, b in zip(BOUNDS_ORDER, BOUNDS_ORDER[1:]):
                ua, la = UB[k].get(a, math.nan), LB[k].get(a, math.nan)
                ub_, lb = UB[k].get(b, math.nan), LB[k].get(b, math.nan)
                if isinstance(ua, float) and math.isinf(ua):
                    proven += 1; continue
                if any(isinstance(x, float) and math.isnan(x) for x in (la, ub_)):
                    continue
                if la >= ub_ - 1e-9:
                    proven += 1
                else:
                    open_pairs[f"{a}>={b}"] += 1
                    open_points[f"{a}>={b}"].add(k)
        print(f"    already proven : {proven} adjacent comparisons")
        if not open_pairs:
            print("    nothing left to resolve -- a bigger run adds no evidence.")
        for pair, n in open_pairs.most_common():
            pts = sorted(open_points[pair], key=lambda k: (k[0], k[1]))
            print(f"    OPEN {pair:<22} {n:>2} point(s): "
                  + ", ".join(f"{p}={v}" for p, v in pts[:8])
                  + (" ..." if len(pts) > 8 else ""))
        bounds_needed = sorted({b for pair in open_pairs for b in pair.split(">=")},
                               key=lambda b: BOUNDS_ORDER.index(b))
        vals = collections.defaultdict(set)
        for pair in open_pairs:
            for p, v in open_points[pair]:
                vals[p].add(v)
        if bounds_needed:
            spec = ";".join(f"{p}=" + ",".join(sorted(v, key=float))
                            for p, v in sorted(vals.items()))
            n_targeted = len(bounds_needed) * sum(len(v) for v in vals.values())
            print(f"\n    targeted rerun -- {n_targeted} solves instead of "
                  f"{len(units)}:")
            print(f"      --bounds {','.join(bounds_needed)} --values {spec}")
    print()
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
                   help="comma list of analytic,base,sweep,impl,formulation,"
                        "failure,case (default: analytic)")
    p.add_argument("--case", default=None,
                   help="for --tests case: comma-separated input file names, "
                        "resolved against --input-dir (e.g. 'demo' -> "
                        "input/demo.yaml). An explicit path also works.")
    p.add_argument("--input-dir", default="input", dest="input_dir",
                   help="where --case looks for input files (default: input)")
    p.add_argument("--name", default="bound_tightness",
                   help="test name used in the folder and file names")
    p.add_argument("--out", default="results",
                   help="root output directory; each run gets its own "
                        "<YYYYMMDDHHMM>_<test> folder inside it")
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
    p.add_argument("--exclude-bounds", default="", dest="exclude_bounds",
                   help="comma list of bounds to drop from the PLOTS and the "
                        "hypothesis checks, e.g. 'chernoff'. The rows stay in "
                        "the CSV -- this only changes what is shown, so a "
                        "re-merge with a different value costs nothing and "
                        "loses nothing. Useful when one bound's scale swamps "
                        "the others, or when its tilt parameter makes it not "
                        "comparable with the rest.")
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
    p.add_argument("--gurobi-params", action="append", default=None,
                   dest="gurobi_params", metavar="K=V,...",
                   help="extra Gurobi parameters, e.g. "
                        "'MIPFocus=3,Symmetry=2,Cuts=2,Heuristics=0.05'. May be "
                        "given several times; all occurrences are MERGED (later "
                        "wins per key), so a job script's settings and yours "
                        "coexist. Use when the DUAL bound is the bottleneck. "
                        "NOTE MIPFocus=3 grows the tree, so pair it with "
                        "NodefileStart/SoftMemLimit.")
    p.add_argument("--shard", default=None, metavar="K/N",
                   help="run only work unit k of n (for Slurm job arrays, e.g. "
                        "--shard $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT). "
                        "Each shard writes its own folder; hypothesis checks are "
                        "skipped -- run --merge afterwards.")
    p.add_argument("--run-stamp", default=None, dest="run_stamp",
                   metavar="YYYYMMDDHHMM",
                   help="identifies the run folder <out>/<stamp>_<test>. Exported "
                        "as RUN_STAMP by submit.sh so every shard of an array "
                        "writes into ONE folder; also selects the folder to "
                        "--merge. Defaults to the current minute.")
    p.add_argument("--plan", action="store_true",
                   help="print the work matrix, the wall-clock arithmetic and "
                        "what is still unresolved, then exit. Solves nothing.")
    p.add_argument("--plan-from", default=None, metavar="CSV",
                   help="a previous merged_results.csv: use its measured solve "
                        "times for the estimate, and list the comparisons that are "
                        "still open so the budget can target them")
    p.add_argument("--plan-shards", type=int, default=None,
                   help="shard count to plan for (default 12)")
    p.add_argument("--plan-maxpar", type=int, default=0,
                   help="concurrent array tasks (MAXPAR in submit.sh); reports how "
                        "many waves and the elapsed time before the merge runs")
    p.add_argument("--plan-wall", type=float, default=24.0,
                   help="Slurm wall-clock limit per task, hours (default 24)")
    p.add_argument("--merge", action="store_true",
                   help="combine the shard folders of --tests under --out, run the "
                        "hypothesis checks and plots once, and write a _merged "
                        "folder. Solves nothing.")
    p.add_argument("--verify-mc", type=int, default=0, dest="verify_mc",
                   metavar="N",
                   help="after each solve, simulate the optimised schedule N times "
                        "and record the empirical P(D>tau) (max over steps, worst "
                        "cell). Verifies the chance constraint really holds and "
                        "measures how much slack each bound leaves. 20000 is "
                        "plenty; costs well under a second per solve.")
    p.add_argument("--mc-dist", default="bernoulli", dest="mc_dist",
                   choices=["bernoulli", "lognormal"],
                   help="distribution used by --verify-mc. 'bernoulli' matches the "
                        "descriptors the model was given; the others keep the same "
                        "mean and variance but is unbounded, which breaks the "
                        "support assumption hoeffding and bernstein rely on while "
                        "leaving markov's and cantelli's intact -- a direct test of "
                        "which guarantees survive misspecification.")
    p.add_argument("--mc-seed", type=int, default=0, dest="mc_seed",
                   help="seed for --verify-mc, so the check is reproducible")
    p.add_argument("--solver-log", default="auto", dest="solver_log",
                   choices=["auto", "on", "off"],
                   help="write Gurobi's log per solve to <run folder>/solver_logs/. "
                        "'auto' (default) means on for --tests case and off for the "
                        "sweeps, where hundreds of logs would swamp the folder. The "
                        "console stays quiet unless --verbose is set.")
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
    p.add_argument("--severity-spread", type=float, default=None,
                   dest="severity_spread",
                   help="missions span b_ref*(1 -+ spread) (default 0.25). Watch "
                        "this one: bernstein's drift term uses support_max, the "
                        "WORST mission, so a wide spread makes bernstein looser "
                        "than hoeffding and (H1) genuinely false.")
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
    p.add_argument("--formulation", default=None, dest="formulation",
                   choices=list(FORMULATIONS_ORDER),
                   help="encoding x assembly x strengthening. "
                        "ENCODING: 'indicator' (default, the original model) or "
                        "'bigm' (nb substituted out, linear big-M rows, tighter "
                        "relaxation) -- same integer optimum, so objectives "
                        "that disagree are a bug, not a modelling choice. "
                        "ASSEMBLY: add the '_sparse' twin ('sparse', "
                        "'bigm_sparse') to build the SAME program through the "
                        "matrix API; it must match its loop twin row for row. "
                        "STRENGTHENING: '_cuts' / '_cuts_core' add the "
                        "locally-supported valid inequalities of "
                        "rainflow_v2.add_sparse_cuts on top of the indicator "
                        "encoding -- same integer optimum, non-trivial root "
                        "bound. 'core' is the big-M-free subset.")
    p.add_argument("--formulations", default=None,
                   help="encodings to compare in the 'formulation' test, comma "
                        f"separated; pick from {FORMULATIONS_ORDER} "
                        "(e.g. --formulations indicator,bigm)")
    p.add_argument("--sparse-cuts", default=None, dest="sparse_cuts",
                   choices=["off", "core", "full"],
                   help="add the locally-supported valid inequalities of "
                        "rainflow_v2.add_sparse_cuts on top of the indicator "
                        "encoding. Same integer optimum, non-trivial root "
                        "bound. Equivalent to naming the composed variant: "
                        "'--formulation indicator --sparse-cuts full' is "
                        "'--formulation indicator_cuts'. Ignored under 'bigm'.")
    p.add_argument("--bigM", type=float, default=None, dest="bigM",
                   help="fallback big-M for a state with no finite bound "
                        "(default 1.1); bounded states use their own bound")
    p.add_argument("--pwl-ladder", default=None,
                   help="in the 'impl' test, also sweep pwl_points, e.g. "
                        "'2,4,8,16' -- shows pwl converging to exact")
    p.add_argument("--allow-replacement", action="store_true", default=None)

    args = p.parse_args(argv)
    if args.no_time_limit or (args.time_limit is not None and args.time_limit <= 0):
        args.time_limit = None

    # --impls is the general form; --impl is the single-value shorthand
    if args.impls:
        impls = [_clean(q) for q in args.impls.split(",") if _clean(q)]
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
    # --formulations is the general form; --formulation the single-value shorthand
    if args.formulations:
        forms = [_clean(q) for q in args.formulations.split(",") if _clean(q)]
    elif args.formulation:
        forms = [args.formulation]
    else:
        forms = ["indicator"]
    for form in forms:
        if form not in FORMULATIONS_ORDER:
            raise SystemExit(f"unknown formulation {form!r}; "
                             f"pick from {FORMULATIONS_ORDER}")
    # --sparse-cuts is the second spelling of the '_cuts' variants; fold it in
    # so the two options compose instead of one silently winning.
    if args.sparse_cuts:
        forms = [compose_variant(f, args.sparse_cuts) for f in forms]
    args.formulation_list = [f for f in FORMULATIONS_ORDER if f in forms]
    args.formulation = args.formulation_list[0]
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
            "formulation", "bigM",
            "severity_spread", "allow_replacement")
    overrides = {k: getattr(args, k) for k in keys if getattr(args, k) is not None}
    return Scenario(**overrides)


def parse_sweeps(args, sc: Scenario) -> dict:
    """Sweep values per parameter; the base value is always included so the
    curves share one common point."""
    defaults = {"L": [1, 2, 3], "H": [4, 6, 8, 10, 12],
                "M": [1, 2, 3, 4], "F": [3, 4, 5, 6, 7],
                "epsilon": [0.2, 0.1, 0.05, 0.02, 0.01, 0.005],
                "rho": [1.0, 0.9, 0.8, 0.6, 0.4],
                "tau": [0.6, 0.8, 1.0, 1.5, 2.0],
                "p": [0.02, 0.05, 0.1, 0.2],
                "tangent_ref": [0.0, 0.05, 0.08, 0.15, 0.3, 0.5]}
    if args.values:
        for block in args.values.split(";"):
            if not block.strip():
                continue
            key, _, vals = block.partition("=")
            key = _clean(key)
            if key not in SWEEP_PARAMS:
                raise SystemExit(f"--values: unknown parameter {key!r}; "
                                 f"pick from {SWEEP_PARAMS}")
            defaults[key] = [_cast(key, v) for v in vals.split(",") if _clean(v)]
    out = {}
    for param in (_clean(q) for q in args.params.split(",") if _clean(q)):
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
            key = _clean(key)
            custom[key] = [_cast(key, v) for v in vals.split(",") if _clean(v)]
    out = {}
    for param in (_clean(q) for q in args.failure_params.split(",") if _clean(q)):
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
    BOUNDS_ORDER = tuple(_clean(b) for b in args.bounds.split(",") if _clean(b))
    # --exclude-bounds narrows the global order, which every plot and every
    # hypothesis check iterates over, so one filter covers all of them. Applied
    # AFTER --bounds so the two compose: --bounds a,b,c --exclude-bounds b.
    _drop = {_clean(b) for b in args.exclude_bounds.split(",") if _clean(b)}
    if _drop:
        unknown = _drop - set(BOUNDS_ORDER)
        if unknown:
            raise SystemExit(f"--exclude-bounds: unknown bound(s) "
                             f"{sorted(unknown)}; pick from {BOUNDS_ORDER}")
        BOUNDS_ORDER = tuple(b for b in BOUNDS_ORDER if b not in _drop)
        if not BOUNDS_ORDER:
            raise SystemExit("--exclude-bounds removed every bound")
        print(f"[plots] excluding bound(s) {sorted(_drop)}; "
              f"showing {list(BOUNDS_ORDER)}")

    sc = scenario_from_args(args)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    tests = [_clean(t) for t in args.tests.split(",") if _clean(t)]

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
        "formulation": lambda r: test_formulation(sc, args, r, args.impl_list,
                                                  args.formulation_list),
        "case": lambda r: test_case(sc, args, r, case_names, Path(args.input_dir)),
    }
    for test in tests:
        if test not in runners:
            raise SystemExit(f"unknown test {test!r}; pick from {tuple(runners)}")
    case_names = [_clean(q) for q in (args.case or "").split(",") if _clean(q)]
    if "case" in tests and not case_names:
        raise SystemExit("--tests case needs --case NAME[,NAME...] naming input "
                         f"file(s) under {args.input_dir}/")
    if "formulation" in tests and len(args.formulation_list) < 2:
        raise SystemExit("the 'formulation' test compares encodings, so give "
                         "both: --formulations indicator,bigm")
    if "impl" in tests and len(args.impl_list) < 2 and not args.pwl_ladder_list:
        raise SystemExit("the 'impl' test compares implementations, so give at "
                         "least two: --impls tangent,pwl,exact "
                         "(optionally with --pwl-ladder 2,4,8,16)")

    if args.plan:                                # planning only, no solving
        args.case_names = case_names
        return plan_run(args, sc, sweeps, ladders, tests)

    suffix = "" if args.shard_obj is None else f"_shard{args.shard_obj.k}"
    # suffix tags FILES inside the shared run folder (see TestRun docstring)
    for test in tests:
        if args.shard_obj is not None:
            args.shard_obj.i = 0                 # restart the unit counter per test
        run = TestRun(out_root, args.name, test, sc, args, suffix=suffix)
        report = header + [f"TEST {test}", "=" * 72]
        extra = None
        try:
            lines, extra = runners[test](run)
            report += lines
            # A shard that solved nothing writes a header-only CSV, and the merge
            # then reports "0 rows" with no clue why. Fail loudly instead.
            if args.shard_obj is not None and not run.rows and test != "analytic":
                units = args.shard_obj.i
                msg = (f"ERROR shard {args.shard_obj} took 0 of {units} work "
                       f"units. Either N exceeds the number of units, or the "
                       f"shard index is wrong. Check that --shard K/N matches the "
                       f"array size (0-{args.shard_obj.n - 1}).")
                report.append(msg)
                print(msg, file=sys.stderr)
                run.close(report, extra)
                return 2
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