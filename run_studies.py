"""
run_studies.py -- scalability / solution-quality studies for the semester report.

Purpose
-------
`test.py` asks *"is a tighter bound cheaper?"* (a statement about the FEASIBLE
SET).  This file asks the two questions a MILP chapter has to answer instead:

  (S1) **How does the program scale?**  Solve time, search-tree size and the
       strength of the relaxation as F (vehicles), M (missions), L (components)
       and H (horizon) grow, for each reliability formulation.
  (S2) **How long a horizon do we actually need?**  Whether the objective and --
       more importantly -- the *decisions taken now* stop moving once H is long
       enough, and whether that H* depends on the mission/component load.

Studies (--studies scaling,horizon,heatmap,convergence)
-------------------------------------------------------
  scaling      one parameter at a time from the base case B, on a geometric
               ladder (1xB, 2xB, 4xB, 8xB), several seeds per setting.  Feeds
               plot categories 1 (time), 2 (nodes) and the LP-gap column.
  horizon      H swept on a fine ladder at a fixed (F, M, L).  Records the
               objective AND the first-period decision vector, so the harness
               can report how many period-0 assignments flip when H grows by one
               step -- the myopia measure.  Feeds section 2, plot 1.
  heatmap      H x M and H x L grids, colour = solve time.  Feeds section 2,
               plot 2: does a longer horizon cost more when the fleet is busier?
  convergence  a few representative (F, M, L, H) points solved with a Gurobi
               callback attached, recording the (time, incumbent, ObjBound)
               trajectory.  Feeds plot category 3 and section 3.

Outputs (one folder per study, mirroring `test.py`)
---------------------------------------------------
    <out>/<YYYYMMDDHHMM>_<study>/
        scenario_base.yaml              the base design point
        results[_shardK].csv            ONE ROW PER SOLVE, ~90 columns
        results[_shardK].yaml           the same rows, aggregated
        summary[_shardK].txt            human-readable report
        progress[_shardK].log           flushed per solve; survives a SIGKILL
        runs/<bound>_<impl>__<tag>.yaml per-solve input + full result
        traces/<tag>.csv                t, incumbent, ObjBound, gap, nodes
        <study>_*.png                   the figures
        merged_*.{csv,yaml,txt,png}     written by --merge

Every row carries the requested five numbers -- `objective`, `runtime_s`,
`mip_gap`, `nodes`, `lp_gap` -- plus everything needed to redo the analysis
without re-solving: model dimensions, seed, host, Slurm id, git commit, the
full Gurobi attribute set, and the period-0 decision vector.

Design notes -- READ BEFORE TRUSTING A NUMBER
---------------------------------------------
1. **What a "seed" varies here.**  MILP solve times are notoriously variable, so
   every setting is solved several times.  A seed changes two things:
     * the Gurobi random seed (`Seed`), which reshuffles tie-breaking in
       presolve, heuristics and branching -- this is *performance variability*
       on a FIXED instance, and it is the dominant effect here;
     * the instance itself, but only through the per-mission severity draw and
       (optionally, off by default) the initial damage `mu_0`.
   Consequence you must state in the report: **at M = 1 there is no severity to
   draw**, so the base case and the F / L / H ladders measure performance
   variability alone.  Genuine instance-to-instance variability only enters the
   M ladder and the heatmaps.  That is a deliberate trade -- see note 2.

2. **Why the instance jitter is so timid.**  markov is by far the loosest bound:
   with the default calibration it admits only `n_max ~ 1.5` reference missions
   between repairs, i.e. `mu <= eps*tau` is a hair away from binding at every
   step.  A perturbation of the increment scale or of `mu_0` that any other
   bound would shrug off makes *markov* infeasible, and an infeasible model is
   detected in milliseconds -- which would silently turn the markov curve into a
   plot of how fast Gurobi proves infeasibility.  So:
     * severity draws are confined to the SAME interval the deterministic
       profile uses, `[1-spread, 1+spread]` with mean exactly 1, so the worst
       mission is never worse than in the unjittered instance;
     * `--mu0-jitter` is 0 by default; when enabled, M vehicles are forced to
       start fresh so the fleet can still serve every mission at k = 0.

3. **The calibration defaults are not `test.py`'s, and the difference matters.**
   `--severity-spread` defaults to **0.10** here, against 0.25 in the Scenario.
   bernstein's drift term `Le*b/3` uses the support of the WORST mission, so a
   wide spread eats its whole budget the moment `M > 1`: at spread 0.25 and
   `--calibrate-n 4`, `n_max(bernstein)` collapses from 2.18 (M=1) to 0.00
   (M=2) and EVERY bernstein run in the M ladder, the horizon study and the
   heatmaps comes back infeasible.  `--calibrate-n` defaults to 6 for the same
   reason -- below about 5 the same collapse happens even at a narrow spread.
   These two numbers were chosen by running the feasibility screen (note 4) over
   the whole default grid until nothing failed; re-run it after changing either.

4. **The feasibility screen, and why to run --plan first.**  `feasible_hint`
   (from `test.py`) is a cheap necessary condition: `n_max` must cover the
   steady-state damage floor `s_max/rho`, and the fleet must be able to cover
   `T*M` mission-days given that at most `F - M` vehicles can be at the depot per
   step.  `--plan` evaluates it for every (configuration, bound) and prints a
   `screen` column.  A bound listed there will return `infeasible` in
   milliseconds -- a wasted rung, not a data point.  Every result row also
   carries `feasible_hint`, so an infeasible status can be attributed to
   *capacity starvation* rather than to *bound tightness*.

5. **The bounds are NOT equally tight at the base case, and cannot be.**  With
   the defaults, hoeffding binds after ~6 reference missions and markov after
   ~1.5, while the base load is `T*M/F = 2`.  So markov's constraint bites at
   the base case and the other three are slack; they start to bite further up
   the H and M ladders (load reaches 16 at H=32).  This is a property of the
   bounds, not a bug: no single design point makes all four bind at once.  Every
   row records `n_max_analytic` and `load`, and the summary prints a binds/slack
   table.  Read the four curves as "four formulations of the same scheduling
   problem", and use the table to say which ones the reliability constraint was
   actually shaping.

6. **Why the M ladder grows the fleet.**  `F > M` is required and, worse,
   `depot_capacity` defaults to `F - M`: pinning `F = 9` for an M ladder of
   1, 2, 4, 8 leaves ONE maintenance slot per step at the top rung, and markov
   then has no feasible schedule at all.  So `--m-sweep-mode fixed-depot`
   (the default) holds `F - M = 8` and lets F grow with M: 9, 10, 12, 16.  The
   number of maintenance slots per step is then constant and the rung is
   genuinely about "more missions".  The cost is that F drifts, so the M curve's
   *shape* is comparable to the others and its *level* is not.  Use
   `--m-sweep-mode fixed-fleet` for the pinned-F version and expect the top
   rungs to die.

7. **The aggregate damage cap does not grow with the problem.**
   `base.add_base_constraints` imposes `sum_{i,l} mu[i,l,k] <= F - M` for every
   step.  The L ladder adds F*L cells to a cap that does not move, and the F and
   M ladders move the cap and the depot capacity together with the dimension
   being swept.  So the F, M and L curves mix "more variables" with "a tighter
   or looser shared resource".  The H ladder is the clean one: it changes only
   the number of time steps.  Say this explicitly next to the figures.

8. **LP gap.**  All four studied formulations (markov/exact plus
   cantelli|hoeffding|bernstein with the single-tangent encoding) are LINEAR, so
   `model.relax()` is a genuine LP and `lp_gap = (z_MIP - z_LP)/|z_MIP|` is the
   textbook integrality gap.  It is measured on a *copy* of the model with its
   own time limit, so it costs extra wall clock -- `--no-lp-relax` turns it off.
   `root_gap` is the different, complementary number: the gap left after Gurobi's
   presolve and root cuts, read off the first callback.  Quote both: `lp_gap`
   says how weak the formulation is, `root_gap` says how much of that Gurobi
   repairs for free.
   Adding `exact` (quadratic) formulations to `--combos` is supported, but then
   `relax()` is a nonconvex QCP, not an LP -- the column is renamed in the
   summary and must not be called an integrality gap.

9. **Censoring.**  A run that stops at `--time-limit` is NOT a measurement of
   solve time, it is a lower bound on it.  Those rows are kept (status
   `time_limit`), flagged with `censored=True`, drawn as hollow markers with an
   up-arrow, and excluded from the median/IQR band unless `--include-censored`
   is given.  A panel where the band stops early is telling you the truth.

10. **Symmetry and the first-period flip metric.**  F vehicles with identical
   `mu_0` are interchangeable, so "vehicle 2 flew mission 1 at H=6 but vehicle 3
   flew it at H=8" may be pure relabelling.  The harness therefore reports three
   numbers: `flip_vars` (Hamming over the F x (M+1) period-0 assignment block),
   `flip_vehicles` (fraction of vehicles whose period-0 activity changed) and
   `flip_matched` (the same, minimised over vehicle permutations -- the part
   that survives relabelling).  Quote `flip_matched` as the myopia measure and
   the gap between it and `flip_vehicles` as the symmetry artefact.
   The comparison is only valid if the instance is otherwise identical across H,
   which is why the seeded draws are made from shapes that do not depend on H.

11. **Provenance.**  Every row records host, Slurm job/array id, git branch and
   commit (with a `+dirty` marker).  Do not `git pull` while an array is in
   flight; `--merge` warns when the shards disagree on the commit.

Usage
-----
    python run_studies.py --plan                          # work matrix, no solve
    python run_studies.py --studies scaling --dry-run     # validate inputs only
    python run_studies.py --studies scaling
    python run_studies.py --studies horizon,heatmap
    python run_studies.py --studies convergence --time-limit 600 --trace on
    python run_studies.py --studies scaling --shard 3/20 --run-stamp 202608211200
    python run_studies.py --studies scaling --merge --run-stamp 202608211200

Slurm (phase 2)
---------------
The sharding is already in place: `--shard K/N` takes work unit K of N in a
deterministic enumeration, every task writes its own `results_shardK.csv` into
ONE folder pinned by `--run-stamp`, and `--merge` reduces them.  A submit script
only has to export RUN_STAMP once, pass
`--shard $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT` and
`--threads $SLURM_CPUS_PER_TASK`, and run `--merge` in a dependent job.

Author: study harness for the degradation-aware EV fleet scheduler.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import math
import os
import socket
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent


# ===========================================================================
# Reuse test.py rather than re-implement it
# ===========================================================================
def _load_harness():
    """Import the sibling `test.py` BY PATH.

    A plain `import test` is a trap: CPython ships a stdlib package called
    `test`, so whichever of the two is found first on sys.path wins, and that
    depends on the working directory.  Loading by explicit path removes the
    ambiguity and also works when this file is invoked from somewhere else.

    Everything structural is taken from there on purpose -- the Scenario, the
    output folder layout, the CSV schema, the shard partitioner, the provenance
    stamps -- so a study row and a bound-test row are the same kind of object
    and can be read by the same downstream code.
    """
    path = _HERE / "test.py"
    if not path.is_file():
        raise SystemExit(
            f"run_studies.py expects test.py next to it (looked in {_HERE}). "
            f"It reuses the Scenario, the run-folder layout and the CSV schema "
            f"from the bound-test harness so that both write the same format.")
    spec = importlib.util.spec_from_file_location("bound_test_harness", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bound_test_harness"] = mod
    spec.loader.exec_module(mod)
    return mod


H = _load_harness()

Scenario = H.Scenario
Shard = H.Shard
TestRun = H.TestRun
n_max = H.n_max
feasible_hint = H.feasible_hint
survival_floor = H.survival_floor
collect_model_metrics = H.collect_model_metrics
classify = H.classify
run_stamp = H.run_stamp
_clean = H._clean
_cast = H._cast
_f = H._f
_fmt = H._fmt
_floats = H._floats
_dump_yaml = H._dump_yaml
_to_builtin = H._to_builtin
_safe_filename = H._safe_filename
_gurobi_params = H._gurobi_params
_pyplot = H._pyplot
_safe_log_scale = H._safe_log_scale
_plots_enabled = H._plots_enabled
_read_rows = H._read_rows

_HOSTNAME = socket.gethostname()
_SLURM_JOB = os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", ""))
if os.environ.get("SLURM_ARRAY_TASK_ID"):
    _SLURM_JOB += f"_{os.environ['SLURM_ARRAY_TASK_ID']}"
_GIT = H._GIT


# ---------------------------------------------------------------------------
# Extend the shared CSV schema with the study columns
# ---------------------------------------------------------------------------
# TestRun writes with `extrasaction="ignore"`, so a column that is not in FIELDS
# is silently DROPPED. Patching the harness module's globals (rather than
# subclassing TestRun) keeps the writer, the reader and --merge in agreement.
STUDY_FIELDS = [
    "study", "factor", "seed", "gurobi_seed", "config_id", "config_label",
    "combo", "censored", "skipped",
    # relaxation strength
    "lp_obj", "lp_status", "lp_gap", "lp_time_s", "lp_quadratic",
    "root_bound", "root_gap",
    # size predicted before the solve (so --plan and the CSV agree)
    "size_binaries", "size_cells",
    # first-period decisions (section 2, plot 1)
    "x0", "m0", "n_depot_0", "obj_per_step",
    # trajectory bookkeeping
    "trace_file", "trace_points", "n_incumbents", "t_first_incumbent",
    "t_last_improvement", "gap_at_first_incumbent",
    # jitter provenance
    "mu0_jitter", "severity_seeded", "severities",
    # MILP encoding x assembly x strengthening
    "formulation", "encoding", "sparse_cuts", "bigM",
]
H.FIELDS = list(H.FIELDS) + [c for c in STUDY_FIELDS if c not in H.FIELDS]
H._NUM_FIELDS = tuple(H._NUM_FIELDS) + (
    "lp_obj", "lp_gap", "lp_time_s", "root_bound", "root_gap", "obj_per_step",
    "t_first_incumbent", "t_last_improvement", "gap_at_first_incumbent",
    "n_depot_0", "mu0_jitter")
H._INT_FIELDS = tuple(H._INT_FIELDS) + (
    "seed", "gurobi_seed", "factor", "size_binaries", "size_cells",
    "trace_points", "n_incumbents")


# ===========================================================================
# What is studied
# ===========================================================================
# (bound, reliability_impl) pairs, exactly the four the report compares.
# markov is linear already, so `rainflow._resolve_impl` folds any requested
# encoding back to "exact" for it -- writing it as ("markov", "exact") makes
# that explicit instead of leaving a phantom "markov/tangent" in the legend.
DEFAULT_COMBOS = (("markov", "exact"),
                  ("cantelli", "tangent"),
                  ("bernstein", "tangent"),
                  ("hoeffding", "tangent"))

# Colour per bound, line style per implementation: two visual channels for the
# two dimensions, so adding `exact` later does not need a new palette.
BOUND_COLOUR = {"markov": "C3", "cantelli": "C0", "hoeffding": "C2",
                "bernstein": "C1", "chernoff": "C4"}
IMPL_STYLE = {"exact": "-", "tangent": "--", "pwl": ":"}
IMPL_MARKER = {"exact": "o", "tangent": "s", "pwl": "^"}

SCALE_PARAMS = ("F", "M", "L", "H")


def combo_label(bound: str, impl: str) -> str:
    return f"{bound}/{impl}"


def parse_combos(text: str) -> tuple:
    out = []
    for block in (b for b in text.split(",") if _clean(b)):
        bound, _, impl = _clean(block).partition("/")
        impl = _clean(impl) or "exact"
        if bound not in H.BOUNDS_ORDER:
            raise SystemExit(f"unknown bound {bound!r}; pick from {H.BOUNDS_ORDER}")
        if impl not in H.IMPLS_ORDER:
            raise SystemExit(f"unknown implementation {impl!r}; "
                             f"pick from {H.IMPLS_ORDER}")
        if bound not in H.IMPL_AWARE_BOUNDS and impl != "exact":
            print(f"[combos] {bound} is a linear bound: the '{impl}' encoding "
                  f"does not apply and rainflow folds it back to 'exact'. "
                  f"Recording it as {bound}/exact.")
            impl = "exact"
        if (bound, impl) not in out:
            out.append((bound, impl))
    if not out:
        raise SystemExit("--combos selected nothing")
    return tuple(out)


# ===========================================================================
# StudyScenario: a Scenario plus a seed
# ===========================================================================
@dataclass(frozen=True)
class StudyScenario(Scenario):
    """A design point plus the randomisation that turns it into an *instance*.

    `seed` drives three things, in this fixed order so that the draws are
    reproducible and, crucially, INDEPENDENT OF H (see design note 10):
        1. the per-mission severity vector, shape (M,);
        2. the per-cell initial damage, shape (F, L);
        3. Gurobi's own `Seed` parameter.
    Nothing here depends on T, so two scenarios that differ only in H see the
    same missions and the same fleet -- which is what makes the first-period
    flip metric a statement about the horizon rather than about the data.
    """
    seed: int = 0
    mu0_jitter: float = 0.0          # fraction of the markov-safe headroom
    jitter_severity: bool = True     # re-draw severities (no-op when M == 1)
    gurobi_seed_offset: int = 0      # keeps solver seeds distinct from data seeds

    # ---- randomisation -----------------------------------------------------
    def _rng(self, stream: int) -> "np.random.Generator":
        """One independent stream per purpose.

        SeedSequence with a spawn key means stream 0 and stream 1 are
        statistically independent, so adding a third draw later cannot shift the
        first two and silently change every instance already measured.
        """
        return np.random.default_rng(
            np.random.SeedSequence(int(self.seed), spawn_key=(int(stream),)))

    @property
    def severities(self) -> np.ndarray:
        """Per-mission severity factors: mean exactly 1, range [1-s, 1+s].

        The deterministic parent spreads them on a linspace; the seeded version
        draws them uniformly from the SAME interval and then affine-corrects the
        mean back to 1, clipping to keep the interval.  Keeping the interval is
        the point (design note 2): the heaviest mission is `1 + spread` in both
        cases, so a seed can never make markov infeasible when the unjittered
        instance was feasible.
        """
        if self.M == 1 or not self.jitter_severity:
            return super().severities
        spread = float(self.severity_spread)
        if spread <= 0.0:
            return np.ones(int(self.M))
        # Work with the DEVIATIONS from 1, not with the levels: centring them
        # gives mean 1 exactly, and shrinking them (rather than clipping) keeps
        # the interval without disturbing the mean again. Clipping after
        # centring, or renormalising after clipping, does disturb it -- either
        # order lets a value escape [1-spread, 1+spread] by a fraction of a
        # percent, which is exactly the margin markov has left.
        d = self._rng(0).uniform(-spread, spread, size=int(self.M))
        d -= d.mean()
        worst = float(np.max(np.abs(d)))
        if worst > spread:
            d *= spread / worst
        return 1.0 + d

    @property
    def mu0_headroom(self) -> float:
        """Largest `mu_0` a vehicle can carry and still be repairable into
        markov's feasible region at step 0: a repair maps mu_0 -> (1-rho)*mu_0,
        which must not exceed eps*tau."""
        return self.epsilon * self.tau / max(1.0 - self.rho, 1e-9)

    def mu0_grid(self) -> np.ndarray:
        """(F, L) initial damage.

        Off by default.  When on, `M` randomly chosen vehicles are reset to zero
        so the fleet can serve every mission at k = 0 without a depot day --
        otherwise `demand_j_0` and markov's `mu <= eps*tau` are jointly
        infeasible and the whole markov curve collapses to "proved infeasible in
        3 ms" (design note 2).
        """
        F, L = int(self.F), int(self.L)
        if self.mu0_jitter <= 0.0:
            return np.full((F, L), float(self.mu_0))
        hi = float(self.mu0_jitter) * self.mu0_headroom
        rng = self._rng(1)
        grid = rng.uniform(0.0, hi, size=(F, L))
        fresh = rng.choice(F, size=min(int(self.M), F), replace=False)
        grid[fresh, :] = 0.0
        return grid

    @property
    def gurobi_seed(self) -> int:
        return int(self.seed) + int(self.gurobi_seed_offset)

    # ---- inputs ------------------------------------------------------------
    def to_input(self, bound: str) -> dict:
        data = super().to_input(bound)
        if self.mu0_jitter > 0.0:
            data["mu_0"] = _to_builtin(self.mu0_grid())      # (F, L) nested list
        return data

    def to_yaml_dict(self) -> dict:
        out = super().to_yaml_dict()
        out["instance"] = {
            "seed": int(self.seed),
            "gurobi_seed": self.gurobi_seed,
            "jitter_severity": bool(self.jitter_severity),
            "severities": _floats(self.severities),
            "mu0_jitter": float(self.mu0_jitter),
            "mu0_headroom": float(self.mu0_headroom),
            "mu_0": _to_builtin(self.mu0_grid()),
            "note": "seed drives the severity draw, mu_0 and Gurobi's Seed; "
                    "none of the draws depend on H, so scenarios differing only "
                    "in H are the same instance on a longer horizon",
        }
        return out

    # ---- size, used by the budget guard and by --plan -----------------------
    @property
    def size_binaries(self) -> int:
        """Binary count BEFORE presolve: x (F x (M+1) x T), m (F x L x T), nb
        (F x L x T, only in the 'indicator' ENCODING -- 'bigm' substitutes it
        out; the '_sparse' twins change only the assembly, not the columns),
        plus r when replacement is on.  Cheap, deterministic, and the only
        size estimate available without building the model."""
        F, M, L, T = int(self.F), int(self.M), int(self.L), int(self.T)
        n = F * (M + 1) * T + F * L * T
        from fleet_management.degradation_model.base import encoding_of
        # `formulation` may be a harness LABEL ('indicator_cuts'); the binary
        # count depends only on the ENCODING it stands for.
        if encoding_of(H.split_variant(self.formulation)[0]) != "bigm":
            n += F * L * T                       # nb
        if self.allow_replacement:
            n += F * L * T
        return n

    @property
    def size_cells(self) -> int:
        return int(self.F) * int(self.L)

    def study_label(self) -> str:
        return (f"F={self.F} M={self.M} L={self.L} H={self.H} (T={self.T}) "
                f"seed={self.seed}")


def base_scenario(args) -> StudyScenario:
    """The base case B, with every CLI override applied."""
    kw = dict(F=args.F, M=args.M, L=args.L, H=args.H,
              n_target=args.n_target, epsilon=args.epsilon, tau=args.tau,
              rho=args.rho, p=args.p, severity_spread=args.severity_spread,
              C_M=args.C_M, C_R=args.C_R, C_S=args.C_S, C_P=args.C_P,
              repair_model=args.repair_model,
              tangent_ref=args.tangent_ref, pwl_points=args.pwl_points,
              # --sparse-cuts is the second spelling of the '_cuts' variants;
              # fold it into the label so both options compose.
              formulation=(H.compose_variant(args.formulation or "indicator",
                                             args.sparse_cuts)
                           if getattr(args, "sparse_cuts", None)
                           else args.formulation),
              bigM=args.bigM,
              allow_replacement=bool(args.allow_replacement),
              mu0_jitter=args.mu0_jitter,
              jitter_severity=not args.no_severity_jitter)
    return StudyScenario(**{k: v for k, v in kw.items() if v is not None})


# ===========================================================================
# Configuration ladders
# ===========================================================================
@dataclass
class Config:
    """One (parameter, value) design point, before seeds and bounds are added."""
    study: str
    param: str                       # 'F' | 'M' | 'L' | 'H' | 'HxM' | 'HxL' | '-'
    value: object                    # the swept value (or 'H=4,M=2' for a grid)
    factor: int
    scenario: StudyScenario
    label: str
    note: str = ""

    @property
    def config_id(self) -> str:
        sc = self.scenario
        return f"F{sc.F}_M{sc.M}_L{sc.L}_H{sc.H}"


def ladder(base_value: int, factors) -> list:
    """Geometric ladder `factor * base`, de-duplicated, ascending."""
    seen, out = set(), []
    for f in factors:
        v = int(round(base_value * f))
        if v >= 1 and v not in seen:
            seen.add(v)
            out.append((int(f), v))
    return out


def scaling_configs(sc: StudyScenario, args) -> list:
    """The 1x / 2x / 4x / 8x ladders, one parameter at a time.

    F, L and H are swept from the shared base B.  M is swept at `--m-sweep-F`
    instead, because `F > M` makes an 8x ladder impossible at the base fleet
    size -- see design note 6.  Configurations that would still violate `F > M`,
    or that exceed `--max-binaries`, are recorded as skipped rather than
    silently dropped: an absent point in a scalability plot must be explainable.
    """
    out = []
    for param in args.scale_params:
        anchor = sc if param != "M" else sc.variant(F=int(args.m_sweep_F))
        note = ""
        if param == "M" and args.m_sweep_mode == "fixed-depot":
            note = (f"the M ladder holds the DEPOT HEADROOM at "
                    f"F - M = {args.m_depot_headroom} (--m-sweep-mode "
                    f"fixed-depot), so F grows with M. Holding F fixed instead "
                    f"starves maintenance -- at F=9, M=8 only one vehicle a step "
                    f"can be repaired and markov has no feasible schedule at all, "
                    f"so the top rung would measure how fast Gurobi proves "
                    f"infeasibility. Its level is not comparable to the other "
                    f"ladders, only its shape.")
        elif param == "M":
            note = (f"the M ladder runs at a fixed F={anchor.F} (--m-sweep-F). "
                    f"Depot capacity is F - M, so the top rungs may be infeasible "
                    f"for the tighter bounds -- check the feasibility screen.")
        for factor, value in ladder(getattr(anchor, param), args.factors):
            variant = anchor.variant(**{param: value})
            if param == "M" and args.m_sweep_mode == "fixed-depot":
                variant = variant.variant(F=int(value) + int(args.m_depot_headroom))
            cfg = Config(study="scaling", param=param, value=value, factor=factor,
                         scenario=variant,
                         label=f"{param}={value} ({factor}xB)", note=note)
            out.append(cfg)
    return out


def horizon_configs(sc: StudyScenario, args) -> list:
    """A fine H ladder at a fixed (F, M, L): the convergence curve of section 2.

    Fine, because the question is *where* the curve flattens; a 1/2/4/8 ladder
    would place at most one point in the interesting region.  Values are sorted
    so that consecutive pairs are the natural comparison for the flip metric.
    """
    anchor = sc.variant(F=int(args.horizon_F), M=int(args.horizon_M),
                        L=int(args.horizon_L))
    out = []
    for value in sorted(set(args.h_values)):
        out.append(Config(study="horizon", param="H", value=int(value),
                          factor=0, scenario=anchor.variant(H=int(value)),
                          label=f"H={value}",
                          note="first-period decisions are compared against the "
                               "next H up the ladder"))
    return out


def heatmap_configs(sc: StudyScenario, args) -> list:
    """H x M and H x L grids: does a longer horizon cost more when busier?"""
    out = []
    for other in args.heatmap_dims:
        anchor = sc.variant(F=int(args.heatmap_F)) if other == "M" else sc
        for h in sorted(set(args.heatmap_h)):
            for v in sorted(set(args.heatmap_other)):
                variant = anchor.variant(H=int(h), **{other: int(v)})
                out.append(Config(
                    study="heatmap", param=f"Hx{other}",
                    value=f"H{h}_{other}{v}", factor=0, scenario=variant,
                    label=f"H={h} {other}={v}",
                    note=f"heatmap cell; fleet held at F={variant.F}"))
    return out


def convergence_configs(sc: StudyScenario, args) -> list:
    """Three (or however many) representative points: small / medium / large.

    Deliberately NOT the extreme of every ladder.  The point of the trajectory
    plot is to show the bound and the incumbent doing something interesting;
    a point that closes in 0.2 s and one that never finds an incumbent are both
    uninformative, so the defaults sit in between and can be overridden.
    """
    out = []
    for name, (Fv, Mv, Lv, Hv) in args.conv_points.items():
        variant = sc.variant(F=int(Fv), M=int(Mv), L=int(Lv), H=int(Hv))
        out.append(Config(study="convergence", param="point", value=name,
                          factor=0, scenario=variant,
                          label=f"{name}: F={Fv} M={Mv} L={Lv} H={Hv}",
                          note="solved with a callback attached; full "
                               "(t, incumbent, ObjBound) trajectory recorded"))
    return out


def seeds_for(cfg: Config, args) -> list:
    """Instance budget: cheap settings get more seeds than expensive ones.

    The split is by predicted binary count, not by measured time, so the plan
    and the run agree and a shard can decide its own workload without solving
    anything first.
    """
    if cfg.study == "convergence":
        n = int(args.conv_seeds)
    elif cfg.study == "heatmap":
        n = int(args.heatmap_seeds)
    else:
        n = (int(args.seeds_small) if cfg.scenario.size_binaries <= args.small_size
             else int(args.seeds_large))
    return list(range(int(args.seed0), int(args.seed0) + max(1, n)))


def skip_reason(cfg: Config, args) -> str:
    """Why a configuration is not solved -- recorded, never silently dropped."""
    sc = cfg.scenario
    if sc.F <= sc.M:
        return f"needs F > M (F={sc.F}, M={sc.M})"
    if sc.size_binaries > int(args.max_binaries):
        return (f"{sc.size_binaries} binaries exceeds --max-binaries "
                f"{args.max_binaries}")
    return ""


def build_configs(sc: StudyScenario, args, studies) -> dict:
    builders = {"scaling": scaling_configs, "horizon": horizon_configs,
                "heatmap": heatmap_configs, "convergence": convergence_configs}
    return {s: builders[s](sc, args) for s in studies}


# ===========================================================================
# Output folder: TestRun, with two study-specific fixes
# ===========================================================================
class StudyRun(TestRun):
    """`test.TestRun` plus seeds.

    Two things change once a design point is solved several times:

    * `TestRun._run_stem` names the per-run YAML `<bound>_<impl>__<param><value>`,
      which is unique in `test.py` because each design point is solved once.
      Here three seeds of the same point would all write to that one file and
      the first two would be silently overwritten -- the CSV would show three
      runs and `runs/` would hold one.  The seed goes into the name.
    * skipped configurations never reach the solver, so they never reach
      `add()`; `record()` puts them in the same CSV through the same writer, so
      the file stays the single source of truth for what was attempted.
    """

    def _run_stem(self, rec: dict) -> str:
        return _safe_filename(f"{super()._run_stem(rec)}__seed{rec.get('seed', '')}")

    def record(self, rec: dict) -> None:
        rec.setdefault("verdict", classify(rec))
        self.rows.append(rec)
        self._w.writerow({k: rec.get(k, "") for k in H.FIELDS})
        self._fh.flush()


# ===========================================================================
# One instrumented solve
# ===========================================================================
@dataclass
class Trace:
    """The (time, incumbent, bound) trajectory of one solve.

    Sampled from a Gurobi callback.  A point is kept when either value has moved
    by more than `min_rel` OR `min_dt` seconds have passed, which keeps a
    ten-minute solve to a few hundred rows instead of a few hundred thousand
    while never dropping an improvement.
    """
    t: list = field(default_factory=list)
    incumbent: list = field(default_factory=list)
    bound: list = field(default_factory=list)
    nodes: list = field(default_factory=list)
    n_incumbents: int = 0
    t_first_incumbent: float = math.nan
    t_last_improvement: float = math.nan
    gap_at_first_incumbent: float = math.nan

    def rows(self) -> list:
        out = []
        for t, inc, bnd, nod in zip(self.t, self.incumbent, self.bound, self.nodes):
            gap = (abs(inc - bnd) / max(abs(inc), 1e-10)
                   if math.isfinite(inc) and math.isfinite(bnd) else math.nan)
            out.append({"t_s": t, "incumbent": inc, "obj_bound": bnd,
                        "gap": gap, "nodes": nod})
        return out

    def write(self, path: Path) -> int:
        rows = self.rows()
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["t_s", "incumbent", "obj_bound",
                                               "gap", "nodes"])
            w.writeheader()
            for r in rows:
                w.writerow({k: ("" if isinstance(v, float) and not math.isfinite(v)
                                else v) for k, v in r.items()})
        return len(rows)


def _finite(value, sentinel=1e100):
    """Gurobi reports 'no incumbent' as +1e100 and 'no bound' as -1e100."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return math.nan
    return math.nan if abs(v) >= sentinel else v


def _make_callback(trace: Trace, min_dt: float, min_rel: float):
    """Record the incumbent/bound trajectory without slowing the solve down.

    Only reads callback scalars -- no cbGetSolution, no cuts, no lazy
    constraints -- so the search is not perturbed and the measured runtime is
    still the runtime of the model as `test.py` would have solved it.
    """
    from gurobipy import GRB

    state = {"last_t": -1e9, "last_inc": math.nan, "last_bnd": math.nan}

    def record(t, inc, bnd, nod, force=False):
        moved = False
        for prev, now in ((state["last_inc"], inc), (state["last_bnd"], bnd)):
            if math.isnan(prev) != math.isnan(now):
                moved = True
            elif math.isfinite(prev) and math.isfinite(now):
                moved = moved or abs(now - prev) > min_rel * max(abs(prev), 1e-10)
        if not (force or moved or (t - state["last_t"]) >= min_dt):
            return
        trace.t.append(float(t))
        trace.incumbent.append(float(inc))
        trace.bound.append(float(bnd))
        trace.nodes.append(float(nod))
        state.update(last_t=t, last_inc=inc, last_bnd=bnd)

    def cb(model, where):
        try:
            if where == GRB.Callback.MIPSOL:
                t = model.cbGet(GRB.Callback.RUNTIME)
                inc = _finite(model.cbGet(GRB.Callback.MIPSOL_OBJBST))
                bnd = _finite(model.cbGet(GRB.Callback.MIPSOL_OBJBND))
                nod = model.cbGet(GRB.Callback.MIPSOL_NODCNT)
                trace.n_incumbents += 1
                if math.isnan(trace.t_first_incumbent):
                    trace.t_first_incumbent = float(t)
                    if math.isfinite(inc) and math.isfinite(bnd):
                        trace.gap_at_first_incumbent = (
                            abs(inc - bnd) / max(abs(inc), 1e-10))
                trace.t_last_improvement = float(t)
                record(t, inc, bnd, nod, force=True)
            elif where == GRB.Callback.MIP:
                t = model.cbGet(GRB.Callback.RUNTIME)
                record(t,
                       _finite(model.cbGet(GRB.Callback.MIP_OBJBST)),
                       _finite(model.cbGet(GRB.Callback.MIP_OBJBND)),
                       model.cbGet(GRB.Callback.MIP_NODCNT))
        except Exception:
            # A callback exception aborts the optimisation in gurobipy. Losing
            # the trajectory is a nuisance; losing the solve is a lost run.
            pass

    return cb


def _relaxation(md, opts, quadratic: bool) -> dict:
    """Solve the continuous relaxation on a COPY of the model.

    `Model.relax()` drops integrality (and converts SOS / general constraints to
    their continuous form) but keeps everything else, so for the four linear
    formulations studied here this is exactly the LP relaxation whose optimum
    defines the integrality gap.  With an `exact` (quadratic) formulation in
    --combos it is a nonconvex QCP instead -- still a valid lower bound, but no
    longer an LP, which is why the flag is carried through to the summary.
    """
    out = {"lp_obj": math.nan, "lp_status": "", "lp_time_s": math.nan,
           "lp_quadratic": bool(quadratic)}
    t0 = time.perf_counter()
    rel = None
    try:
        rel = md.relax()
        rel.Params.OutputFlag = 0
        if opts.get("lp_time_limit"):
            rel.Params.TimeLimit = float(opts["lp_time_limit"])
        if opts.get("threads"):
            rel.Params.Threads = int(opts["threads"])
        rel.optimize()
        out["lp_status"] = _status_string(rel.status)
        if rel.SolCount > 0:
            out["lp_obj"] = float(rel.ObjVal)
    except Exception as exc:
        out["lp_status"] = f"error: {type(exc).__name__}: {exc}"
    finally:
        out["lp_time_s"] = time.perf_counter() - t0
        if rel is not None:
            try:
                rel.dispose()
            except Exception:
                pass
    return out


def _status_string(code: int) -> str:
    from fleet_management.degradation_model.base import status_string
    return status_string(code)


def solve_instrumented(sc: StudyScenario, bound: str, opts,
                       log_path=None, trace_on: bool = False) -> dict:
    """Build the model here, solve it here -- everything `rainflow.solve` does,
    plus the three things it cannot expose.

    `rainflow.solve` calls `ctx.model.optimize()` with no callback and disposes
    of nothing until `extract_solution` has run, so there is no hook for the
    incumbent/bound trajectory and no chance to touch the model before it is
    optimised.  Rebuilding the same three lines here (`resolve_run_options` ->
    `build_fleet` -> `optimize` -> `extract_solution`) buys:
        * the callback trajectory (plot category 3),
        * the continuous relaxation, solved on a copy before the MIP,
        * a per-solve Gurobi `Seed`, which is what makes the seed bands mean
          "performance variability" rather than "the same run four times".
    It uses the project's own builders, so the MODEL is identical to the one
    `test.py` solves -- only the driving is different.
    """
    from fleet_management.config import load_config
    from fleet_management.degradation_model.base import (
        build_fleet, extract_solution, resolve_run_options)

    data = sc.to_input(bound)
    cfg = load_config(data)

    gp_params = dict(_gurobi_params(opts, log_path) or {})
    gp_params.setdefault("Seed", sc.gurobi_seed)

    run_opts = resolve_run_options(
        cfg,
        allow_replacement=sc.allow_replacement,
        verbose=int(getattr(opts, "verbose", 0) or 0),
        mip_gap=opts.mip_gap,
        time_limit=opts.time_limit,
        gurobi_params=gp_params,
        reliability_impl=sc.reliability_impl,
        pwl_points=sc.pwl_points,
        tangent_ref=sc.tangent_ref,
        # A harness label ('indicator_cuts') stands for a (formulation,
        # sparse_cuts) pair; split it. See test.split_variant.
        formulation=H.split_variant(sc.formulation)[0],
        sparse_cuts=H.split_variant(sc.formulation)[1],
        bigM=sc.bigM,
    )

    ctx = build_fleet(cfg, run_opts, model_name="fleet_management_study")
    md = ctx.model
    md.update()

    extra = {"n_qconstrs_built": int(md.NumQConstrs),
             "n_genconstrs_built": int(md.NumGenConstrs)}

    # ---- relaxation first: `relax()` on an un-optimised model is cheap to
    # copy, and doing it here means the MIP's own Runtime attribute is not
    # polluted by the LP.
    if getattr(opts, "lp_relax", True):
        extra.update(_relaxation(
            md, {"lp_time_limit": getattr(opts, "lp_time_limit", None),
                 "threads": getattr(opts, "threads", None)},
            quadratic=bool(md.NumQConstrs)))

    trace = Trace()
    cb = _make_callback(trace, float(getattr(opts, "trace_min_dt", 0.25)),
                        float(getattr(opts, "trace_min_rel", 1e-9))) if trace_on else None

    t0 = time.perf_counter()
    if cb is not None:
        md.optimize(cb)
    else:
        md.optimize()
    wall = time.perf_counter() - t0

    res = extract_solution(ctx, cfg, md)
    res["wall_s"] = wall
    res["trace"] = trace
    res.update(extra)

    # A model solved in presolve never enters the callback, so close the
    # trajectory by hand: without this the "solved instantly" cases show an
    # empty panel instead of a single point at t = runtime.
    if trace_on:
        try:
            final_inc = float(md.ObjVal) if md.SolCount > 0 else math.nan
            final_bnd = float(md.ObjBound)
            trace.t.append(float(md.Runtime))
            trace.incumbent.append(final_inc)
            trace.bound.append(final_bnd)
            trace.nodes.append(float(md.NodeCount))
        except Exception:
            pass

    # Root bound = the first dual bound the callback saw, i.e. after presolve
    # and root cuts. Complementary to lp_obj, which is before both.
    res["root_bound"] = next((b for b in trace.bound if math.isfinite(b)), math.nan)
    return res


# ===========================================================================
# One row
# ===========================================================================
def decision_signature(res: dict, sc: StudyScenario) -> dict:
    """The period-0 decisions, compact enough for a CSV cell.

    `x0` is one activity index per vehicle: -1 idle, 0 depot, j>=1 mission j.
    `m0` is the per-cell repair indicator at k = 0.  Together they are the
    "what do I do tomorrow morning" answer, which is the only part of the plan
    a rolling-horizon operator ever executes -- and therefore the part whose
    stability in H is worth measuring (section 2, plot 1).
    """
    x = res.get("x")
    m = res.get("m")
    out = {"x0": "", "m0": "", "n_depot_0": math.nan}
    if x is None:
        return out
    F, Mp1, _ = np.asarray(x).shape
    acts = []
    for i in range(F):
        col = np.asarray(x)[i, :, 0]
        acts.append(int(np.argmax(col)) if col.max() > 0.5 else -1)
    out["x0"] = "|".join(str(a) for a in acts)
    out["n_depot_0"] = float(sum(1 for a in acts if a == 0))
    if m is not None:
        out["m0"] = "|".join(str(int(round(float(v))))
                             for v in np.asarray(m)[:, :, 0].ravel())
    return out


def run_case(cfg: Config, bound: str, impl: str, seed: int, opts, run,
             trace_on: bool) -> tuple:
    """Solve one (configuration, bound, implementation, seed) and build its row.

    Mirrors `test.run_case`: same provenance fields, same "infeasible == infinitely
    expensive" convention, same never-raise policy, so a single bad design point
    cannot take a twelve-hour array down with it.
    """
    sc = cfg.scenario.variant(reliability_impl=impl, seed=int(seed))
    rec = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "study": cfg.study, "test": cfg.study,
        "parameter": cfg.param, "value": cfg.value, "factor": cfg.factor,
        "config_id": cfg.config_id, "config_label": cfg.label,
        "bound": bound, "combo": combo_label(bound, impl),
        "seed": int(seed), "gurobi_seed": sc.gurobi_seed,
        "F": sc.F, "M": sc.M, "L": sc.L, "H": sc.H, "T": sc.T,
        "tau": sc.tau, "epsilon": sc.epsilon, "rho": sc.rho, "p": sc.p,
        "b_ref": sc.b_ref, "mu_ref": sc.p * sc.b_ref,
        "s_chernoff": sc.s_chernoff, "repair_model": sc.repair_model,
        "reliability_impl": impl, "pwl_points": sc.pwl_points,
        "tangent_ref": sc.tangent_ref,
        "formulation": sc.formulation, "bigM": sc.bigM,
        "encoding": H.split_variant(sc.formulation)[0],
        "sparse_cuts": H.split_variant(sc.formulation)[1],
        "allow_replacement": sc.allow_replacement,
        "C_M": sc.C_M, "C_R": sc.C_R, "C_S": sc.C_S, "C_P": sc.C_P,
        "size_binaries": sc.size_binaries, "size_cells": sc.size_cells,
        "mu0_jitter": sc.mu0_jitter,
        "severity_seeded": bool(sc.jitter_severity and sc.M > 1),
        "severities": ",".join(f"{v:.6g}" for v in sc.severities),
        "threads": getattr(opts, "threads", None) or "",
        "gurobi_params": ",".join(f"{k}={v}" for k, v in
                                  sorted((_gurobi_params(opts) or {}).items())),
        "req_mip_gap": opts.mip_gap, "req_time_limit": opts.time_limit,
        "solver_log": (log_name(opts, run, cfg, bound, impl, seed) or ""),
        "host": _HOSTNAME, "slurm_job": _SLURM_JOB,
        "git_branch": _GIT["git_branch"], "git_commit": _GIT["git_commit"],
        "n_max_analytic": n_max(sc, bound),
        "load": sc.load,
        "feasible_hint": feasible_hint(sc, bound),
        "censored": False, "skipped": "",
    }

    data = sc.to_input(bound)

    if opts.dry_run:
        from fleet_management.config import load_config
        load_config(data)                        # validate without a licence
        rec.update({"status": "dry_run", "objective": math.nan,
                    "mip_gap": math.nan, "obj_bound": math.nan,
                    "runtime_s": math.nan, "wall_s": math.nan})
        return rec, data, None

    log_path = H.solver_log_path(run, f"{cfg.config_id}_{bound}_{impl}_s{seed}")
    t0 = time.perf_counter()
    try:
        res = solve_instrumented(sc, bound, opts, log_path, trace_on)
    except BaseException as exc:                 # never let one point kill a run
        if isinstance(exc, KeyboardInterrupt):
            raise
        import traceback as tb
        rec.update({"status": f"error: {type(exc).__name__}: {exc}",
                    "objective": math.nan, "wall_s": time.perf_counter() - t0,
                    "traceback": tb.format_exc(limit=6).replace("\n", " | ")})
        return rec, data, None

    obj = res.get("objective")
    status = res.get("status")
    rec.update({
        "status": status,
        "objective": (math.inf if status == "infeasible"
                      else (float(obj) if obj is not None else math.nan)),
        "mip_gap": _f(res.get("mip_gap")),
        "obj_bound": _f(res.get("bound")),
        "wall_s": res.get("wall_s"),
        "censored": status == "time_limit",
        "lp_obj": _f(res.get("lp_obj")),
        "lp_status": res.get("lp_status", ""),
        "lp_time_s": _f(res.get("lp_time_s")),
        "root_bound": _f(res.get("root_bound")),
        "lp_quadratic": res.get("lp_quadratic", ""),
    })

    md = res.get("model")
    if md is not None:
        rec.update(collect_model_metrics(md))

    # relaxation gaps, both relative to the achieved incumbent
    z = rec.get("objective")
    if isinstance(z, float) and math.isfinite(z) and abs(z) > 1e-12:
        for src, dst in (("lp_obj", "lp_gap"), ("root_bound", "root_gap")):
            b = rec.get(src)
            if isinstance(b, float) and math.isfinite(b):
                rec[dst] = (z - b) / abs(z)
    if isinstance(z, float) and math.isfinite(z) and sc.T:
        rec["obj_per_step"] = z / float(sc.T)

    for key, arr, red in (("n_repairs", res.get("m"), np.sum),
                          ("n_replacements", res.get("r"), np.sum),
                          ("mu_max", res.get("mu"), np.max),
                          ("v_max", res.get("v"), np.max)):
        rec[key] = float(red(arr)) if arr is not None else math.nan
    x = res.get("x")
    rec["n_depot"] = float(np.sum(x[:, 0, :])) if x is not None else math.nan
    rec.update(decision_signature(res, sc))

    trace = res.get("trace")
    if trace is not None and trace.t:
        rec.update({"n_incumbents": trace.n_incumbents,
                    "t_first_incumbent": trace.t_first_incumbent,
                    "t_last_improvement": trace.t_last_improvement,
                    "gap_at_first_incumbent": trace.gap_at_first_incumbent})
        name = _safe_filename(f"{cfg.config_id}_{bound}_{impl}_s{seed}")
        path = run.dir / "traces"
        path.mkdir(parents=True, exist_ok=True)
        rec["trace_points"] = trace.write(path / f"{name}.csv")
        rec["trace_file"] = f"traces/{name}.csv"

    if md is not None:
        try:
            md.dispose()
        except Exception:
            pass
    return rec, data, trace


def log_name(opts, run, cfg, bound, impl, seed):
    p = H.solver_log_path(run, f"{cfg.config_id}_{bound}_{impl}_s{seed}")
    return p.name if p is not None else ""


def skipped_row(cfg: Config, bound: str, impl: str, reason: str) -> dict:
    """A skipped configuration still gets a row.

    An empty point in a scalability plot is a claim about the solver; a skipped
    point is a claim about the experiment.  Recording the reason keeps the two
    apart when someone reads the CSV six months from now.
    """
    sc = cfg.scenario
    return {"timestamp": datetime.now().isoformat(timespec="seconds"),
            "study": cfg.study, "test": cfg.study, "parameter": cfg.param,
            "value": cfg.value, "factor": cfg.factor,
            "config_id": cfg.config_id, "config_label": cfg.label,
            "bound": bound, "combo": combo_label(bound, impl),
            "reliability_impl": impl,
            "F": sc.F, "M": sc.M, "L": sc.L, "H": sc.H, "T": sc.T,
            "size_binaries": sc.size_binaries,
            "status": "skipped", "skipped": reason,
            "objective": math.nan, "runtime_s": math.nan,
            "verdict": "skipped",
            "host": _HOSTNAME, "slurm_job": _SLURM_JOB,
            "git_branch": _GIT["git_branch"], "git_commit": _GIT["git_commit"]}


# ===========================================================================
# The study driver
# ===========================================================================
def run_study(study: str, configs: list, sc0: StudyScenario, opts, run) -> tuple:
    """Solve every (configuration, combo, seed) that belongs to this shard.

    The enumeration order is fixed -- configuration, then combo, then seed -- so
    `--shard K/N` partitions the SAME list on every task and no two tasks
    duplicate work.  Interleaving by `index % N` (rather than by contiguous
    blocks) also spreads the expensive settings evenly, so one task does not get
    all the 8x rungs.
    """
    lines = [f"base case: {sc0.study_label()}",
             f"combos: {[combo_label(*c) for c in opts.combos]}",
             f"mip_gap={opts.mip_gap}  "
             f"time_limit={'none' if opts.time_limit is None else opts.time_limit}"
             f"  lp_relax={bool(opts.lp_relax)}", ""]
    trace_default = {"on": True, "off": False}.get(
        opts.trace, study == "convergence")

    n_solved = 0
    for cfg in configs:
        reason = skip_reason(cfg, opts)
        seeds = seeds_for(cfg, opts)
        for bound, impl in opts.combos:
            if reason:
                # Skips are enumerated as work units too, so the shard split
                # does not shift when a skip appears or disappears.
                for seed in seeds:
                    if H._mine(opts):
                        rec = skipped_row(cfg, bound, impl, reason)
                        rec["seed"] = int(seed)
                        run.record(rec)
                continue
            for seed in seeds:
                if not H._mine(opts):
                    continue
                tag = f"{cfg.label} {combo_label(bound, impl)} seed={seed}"
                print(f"  [{study}] {tag} ...", flush=True)
                run.note_progress(f"START {tag}")
                rec, data, _trace = run_case(cfg, bound, impl, seed, opts, run,
                                             trace_default)
                run.add(rec, data, cfg.scenario.variant(reliability_impl=impl,
                                                        seed=int(seed)))
                n_solved += 1
                print(f"      cost={_fmt(rec.get('objective'))} "
                      f"time={_fmt(rec.get('runtime_s'), 2)}s "
                      f"nodes={_fmt(rec.get('nodes'), 0)} "
                      f"gap={_fmt(rec.get('mip_gap'))} "
                      f"lp_gap={_fmt(rec.get('lp_gap'))} "
                      f"status={rec.get('status')}")

    lines += summarise(study, run.rows, opts)
    if _plots_enabled(opts) and getattr(opts, "shard_obj", None) is None:
        lines += make_plots(study, run.rows, run, opts)
    elif getattr(opts, "shard_obj", None) is not None:
        lines.append(f"shard {opts.shard_obj}: {n_solved} solve(s); plots and "
                     f"cross-configuration statistics are produced by --merge.")
    return lines, {"study": study, "n_rows": len(run.rows), "n_solved": n_solved}


# ===========================================================================
# Aggregation
# ===========================================================================
def _num(rec, key):
    v = rec.get(key)
    if isinstance(v, str):
        try:
            v = float(v)
        except ValueError:
            return math.nan
    return float(v) if isinstance(v, (int, float)) else math.nan


def usable(rec, include_censored: bool = False) -> bool:
    """Rows that may enter a timing statistic.

    A censored row understates the solve time by construction, so mixing it into
    a median silently biases the curve downwards exactly where the problem got
    hard -- which is the region the plot exists to show (design note 9).
    """
    if rec.get("status") in ("skipped", "dry_run"):
        return False
    if str(rec.get("status", "")).startswith("error"):
        return False
    if not include_censored and rec.get("censored") in (True, "True", "true"):
        return False
    return True


def aggregate(rows, metric: str, xkey: str = "value",
              band: str = "iqr", include_censored: bool = False) -> dict:
    """{(bound, impl): [(x, centre, lo, hi, n, n_censored, n_total)]}, x ascending.

    Median + IQR by default rather than mean + std: with two or three seeds the
    distribution of MILP solve times is skewed and a single unlucky run drags a
    mean by an order of magnitude.  With n = 2 the IQR degenerates to the range,
    which is honest -- the band is then "these were the two values".
    """
    groups = {}
    for rec in rows:
        key = (rec.get("bound"), rec.get("reliability_impl") or "exact")
        x = rec.get(xkey)
        try:
            x = float(x)
        except (TypeError, ValueError):
            continue
        groups.setdefault(key, {}).setdefault(x, []).append(rec)

    out = {}
    for key, per_x in groups.items():
        series = []
        for x in sorted(per_x):
            recs = per_x[x]
            n_cens = sum(1 for r in recs if r.get("censored") in (True, "True", "true"))
            vals = [_num(r, metric) for r in recs if usable(r, include_censored)]
            vals = [v for v in vals if math.isfinite(v)]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            if band == "std":
                centre = float(arr.mean())
                lo, hi = centre - float(arr.std()), centre + float(arr.std())
            elif band == "minmax":
                centre = float(np.median(arr))
                lo, hi = float(arr.min()), float(arr.max())
            else:
                centre = float(np.median(arr))
                lo, hi = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
            series.append((x, centre, lo, hi, len(vals), n_cens, len(recs)))
        if series:
            out[key] = series
    return out


# ===========================================================================
# Plots
# ===========================================================================
def _style(bound, impl):
    return dict(color=BOUND_COLOUR.get(bound, "C7"),
                ls=IMPL_STYLE.get(impl, "-"),
                marker=IMPL_MARKER.get(impl, "o"))


def _draw_series(ax, series, bound, impl, band_alpha=0.18):
    xs = [p[0] for p in series]
    ys = [p[1] for p in series]
    lo = [p[2] for p in series]
    hi = [p[3] for p in series]
    st = _style(bound, impl)
    ax.plot(xs, ys, label=combo_label(bound, impl), ms=4, lw=1.4, **st)
    ax.fill_between(xs, lo, hi, color=st["color"], alpha=band_alpha, lw=0)
    return ys + lo + hi


def _mark_censored(ax, rows, xkey, metric, bound, impl):
    """Hollow up-arrows where at least one seed hit the time limit."""
    pts = {}
    for r in rows:
        if r.get("bound") != bound or (r.get("reliability_impl") or "exact") != impl:
            continue
        if r.get("censored") not in (True, "True", "true"):
            continue
        try:
            x = float(r.get(xkey))
        except (TypeError, ValueError):
            continue
        y = _num(r, metric)
        if math.isfinite(y):
            pts[x] = max(pts.get(x, 0.0), y)
    if pts:
        ax.scatter(list(pts), list(pts.values()), marker="^", s=42,
                   facecolors="none", edgecolors=BOUND_COLOUR.get(bound, "C7"),
                   linewidths=1.1, zorder=5)


def plot_effort_vs_params(rows, run, opts, metric: str, ylabel: str,
                          title: str, fname: str):
    """Plot categories 1 and 2: one PANEL per swept parameter, log y.

    Separate panels, not one shared x-axis: F = 8 and H = 8 are not the same
    quantity and putting them on one axis invites a comparison that means
    nothing.  A shared *y* axis is kept so the panels can be compared by eye --
    which is the whole point, since the question is which dimension is the hard
    one.  On a log y-axis exponential growth is a straight line and polynomial
    growth bends downwards; the slope is quoted in the summary.
    """
    plt = _pyplot()
    if plt is None:
        return None
    params = [p for p in SCALE_PARAMS
              if any(r.get("parameter") == p for r in rows)]
    if not params:
        return None

    fig, axes = plt.subplots(1, len(params), figsize=(3.6 * len(params), 3.9),
                             sharey=True, squeeze=False)
    axes = axes[0]
    all_vals = []
    for ax, param in zip(axes, params):
        sub = [r for r in rows if r.get("parameter") == param]
        agg = aggregate(sub, metric, "value", opts.band, opts.include_censored)
        for (bound, impl) in opts.combos:
            series = agg.get((bound, impl))
            if not series:
                continue
            all_vals += _draw_series(ax, series, bound, impl)
            _mark_censored(ax, sub, "value", metric, bound, impl)
        ax.set_xlabel(param)
        ax.grid(alpha=0.3, which="both")
        xs = sorted({float(r["value"]) for r in sub
                     if isinstance(r.get("value"), (int, float))
                     and r.get("status") != "skipped"})
        if xs:
            # log-2 x-axis: the ladder is geometric, so equal factors get equal
            # spacing and a straight line means "time scales like a power of the
            # parameter" on both axes at once.
            ax.set_xscale("log", base=2)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{int(v)}" for v in xs])
    _safe_log_scale(axes[0], all_vals)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(fontsize=7, loc="upper left")
    band = {"iqr": "median, IQR band", "minmax": "median, min-max band",
            "std": "mean, +-1 std band"}[opts.band]
    fig.suptitle(f"{title}   ({band}; hollow triangles hit the "
                 f"{opts.time_limit}s limit)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = run.dir / f"{run.stem}_{fname}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_relaxation(rows, run, opts):
    """The LP gap next to the final MIP gap: is the formulation weak, or is the
    search slow?  A large `lp_gap` that Gurobi closes anyway is a formulation
    worth tightening; a small one that still takes minutes is a search problem."""
    plt = _pyplot()
    if plt is None:
        return None
    params = [p for p in SCALE_PARAMS
              if any(r.get("parameter") == p for r in rows)]
    if not params:
        return None
    fig, axes = plt.subplots(2, len(params), figsize=(3.6 * len(params), 6.0),
                             sharey="row", squeeze=False)
    for col, param in enumerate(params):
        sub = [r for r in rows if r.get("parameter") == param]
        for row_i, (metric, lab) in enumerate((("lp_gap", "LP relaxation gap"),
                                               ("root_gap", "root gap (after cuts)"))):
            ax = axes[row_i][col]
            agg = aggregate(sub, metric, "value", opts.band, True)
            for (bound, impl) in opts.combos:
                series = agg.get((bound, impl))
                if series:
                    _draw_series(ax, series, bound, impl)
            ax.grid(alpha=0.3, which="both")
            if row_i == 0:
                ax.set_title(param, fontsize=9)
            if col == 0:
                ax.set_ylabel(lab)
            if row_i == 1:
                ax.set_xlabel(param)
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Relaxation strength: (z_MIP - z_LP)/|z_MIP| and the gap left "
                 "after Gurobi's root cuts", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = run.dir / f"{run.stem}_relaxation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _read_trace(run_dir: Path, rel: str) -> list:
    path = run_dir / rel
    if not path.is_file():
        return []
    out = []
    with path.open(newline="") as fh:
        for raw in csv.DictReader(fh):
            def g(k):
                v = raw.get(k, "")
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return math.nan
            out.append((g("t_s"), g("incumbent"), g("obj_bound")))
    return out


def plot_convergence(rows, run, opts):
    """Plot category 3 / section 3: bound vs incumbent, small multiples.

    One panel per representative point, side by side, sharing nothing: the
    absolute objective differs by an order of magnitude between small and large,
    so a shared y-axis would flatten the small panel to a line.  The shaded
    region between the two curves IS the gap -- reading a time limit off the
    x-axis and the remaining gap off the shading is the whole purpose.
    """
    plt = _pyplot()
    if plt is None:
        return None
    traced = [r for r in rows if r.get("trace_file")]
    if not traced:
        return None
    points = sorted({str(r.get("value")) for r in traced},
                    key=lambda v: min((_num(r, "size_binaries") for r in traced
                                       if str(r.get("value")) == v), default=0))
    combos = [c for c in opts.combos
              if any(r.get("bound") == c[0] and
                     (r.get("reliability_impl") or "exact") == c[1] for r in traced)]
    if not points or not combos:
        return None

    fig, axes = plt.subplots(len(combos), len(points),
                             figsize=(3.7 * len(points), 2.7 * len(combos)),
                             squeeze=False)
    for ci, (bound, impl) in enumerate(combos):
        for pi, point in enumerate(points):
            ax = axes[ci][pi]
            sel = [r for r in traced
                   if str(r.get("value")) == point and r.get("bound") == bound
                   and (r.get("reliability_impl") or "exact") == impl]
            if not sel:
                ax.axis("off")
                continue
            rec = sel[0]
            tr = _read_trace(run.dir, str(rec.get("trace_file")))
            if not tr:
                ax.axis("off")
                continue
            ts = [p[0] for p in tr]
            inc = [p[1] for p in tr]
            bnd = [p[2] for p in tr]
            col = BOUND_COLOUR.get(bound, "C7")
            # step-post: both series are piecewise constant between events
            ax.step(ts, inc, where="post", color=col, lw=1.5, label="incumbent")
            ax.step(ts, bnd, where="post", color=col, lw=1.2, ls="--",
                    label="ObjBound")
            fin = [(t, i, b) for t, i, b in tr
                   if math.isfinite(i) and math.isfinite(b)]
            if fin:
                ax.fill_between([p[0] for p in fin], [p[1] for p in fin],
                                [p[2] for p in fin], step="post",
                                color=col, alpha=0.15, lw=0)
            tl = float(opts.time_limit) if opts.time_limit else math.nan
            if (rec.get("censored") in (True, "True", "true")
                    and math.isfinite(tl) and tl <= 1.3 * ts[-1]):
                # only when the trace really runs into it; a stale limit far to
                # the right would rescale the panel and flatten the trajectory
                ax.axvline(tl, color="0.4", lw=0.8, ls=":")
            ax.grid(alpha=0.3)
            ax.set_title(f"{point} - {combo_label(bound, impl)}\n"
                         f"gap {_fmt(rec.get('mip_gap'))}, "
                         f"{_fmt(rec.get('nodes'), 0)} nodes", fontsize=8)
            if ci == len(combos) - 1:
                ax.set_xlabel("wall-clock time [s]")
            if pi == 0:
                ax.set_ylabel("objective")
            if ci == 0 and pi == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Dual bound vs incumbent: which one is the bottleneck?",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = run.dir / f"{run.stem}_convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---- horizon study --------------------------------------------------------
def _parse_x0(text) -> list:
    if not text or not isinstance(text, str):
        return []
    try:
        return [int(v) for v in text.split("|") if v != ""]
    except ValueError:
        return []


def _matched_flip(a: list, b: list) -> float:
    """Fraction of vehicles whose period-0 activity changed, MINIMISED over
    relabellings of the fleet.

    With identical vehicles, "vehicle 2 goes to the depot" and "vehicle 3 goes
    to the depot" are the same plan.  Minimising the mismatch over permutations
    strips that artefact out, so what is left is a decision change a planner
    would actually notice.  Hungarian when SciPy is available, brute force for
    a small fleet, NaN rather than a wrong number otherwise.
    """
    if not a or len(a) != len(b):
        return math.nan
    n = len(a)
    cost = np.array([[0.0 if a[i] == b[j] else 1.0 for j in range(n)]
                     for i in range(n)])
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost)
        return float(cost[r, c].sum()) / n
    except Exception:
        pass
    if n > 8:
        return math.nan
    best = min(sum(cost[i, perm[i]] for i in range(n))
               for perm in itertools.permutations(range(n)))
    return float(best) / n


def horizon_stability(rows) -> list:
    """Per (combo, seed), compare each H with the NEXT H on the ladder.

    Compared against the next H rather than against the longest one, because the
    operational question is "would one more period of foresight have changed
    what I do tomorrow?".  A curve of these differences that decays to zero is
    the evidence that H* has been reached; one that stays flat says the horizon
    never stops mattering.
    """
    by_key = {}
    for r in rows:
        if (r.get("study") or r.get("test")) != "horizon":
            continue
        if not usable(r, include_censored=True):
            continue
        key = (r.get("bound"), r.get("reliability_impl") or "exact",
               r.get("seed"))
        try:
            h = int(float(r.get("H")))
        except (TypeError, ValueError):
            continue
        by_key.setdefault(key, {})[h] = r

    out = []
    for (bound, impl, seed), per_h in by_key.items():
        hs = sorted(per_h)
        for h, h_next in zip(hs, hs[1:]):
            a, b = per_h[h], per_h[h_next]
            xa, xb = _parse_x0(a.get("x0")), _parse_x0(b.get("x0"))
            n = len(xa)
            flip_vehicles = (sum(1 for i in range(n) if xa[i] != xb[i]) / n
                             if n and len(xb) == n else math.nan)
            # Hamming over the F x (M+1) binary block: an activity change flips
            # exactly two entries (one off, one on), an idle<->busy change one.
            flips = 0
            if n and len(xb) == n:
                for i in range(n):
                    if xa[i] != xb[i]:
                        flips += (1 if -1 in (xa[i], xb[i]) else 2)
            try:
                Mv = int(float(a.get("M")))
            except (TypeError, ValueError):
                Mv = 0
            denom = n * (Mv + 1)
            out.append({
                "bound": bound, "reliability_impl": impl, "seed": seed,
                "H": h, "H_next": h_next,
                "flip_vars": flips / denom if denom else math.nan,
                "flip_vehicles": flip_vehicles,
                "flip_matched": _matched_flip(xa, xb),
                "obj_per_step": _num(a, "obj_per_step"),
                "obj_per_step_next": _num(b, "obj_per_step"),
                "d_obj_per_step": (_num(b, "obj_per_step") - _num(a, "obj_per_step")),
            })
    return out


def plot_horizon(rows, stability, run, opts):
    """Section 2, plot 1: does the plan settle down as H grows?

    Two stacked panels sharing the H axis.  Top: the objective per time step,
    which is the only H-comparable normalisation (the raw objective grows with
    T by construction, so a raw-objective curve would show convergence that is
    purely arithmetic).  Bottom: how much of the period-0 decision survives one
    more period of foresight, with the symmetry-corrected version drawn solid.
    """
    plt = _pyplot()
    if plt is None:
        return None
    sub = [r for r in rows if (r.get("study") or r.get("test")) == "horizon"]
    if not sub:
        return None
    fig, (ax_o, ax_f) = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True,
                                     height_ratios=[1, 1])

    agg = aggregate(sub, "obj_per_step", "H", opts.band, True)
    for (bound, impl) in opts.combos:
        series = agg.get((bound, impl))
        if series:
            _draw_series(ax_o, series, bound, impl)
    ax_o.set_ylabel("objective per time step  J/T")
    ax_o.grid(alpha=0.3)
    ax_o.legend(fontsize=7)
    ax_o.set_title("Does a longer horizon change the plan?", fontsize=10)

    for (bound, impl) in opts.combos:
        pts = [s for s in stability
               if s["bound"] == bound and s["reliability_impl"] == impl]
        if not pts:
            continue
        st = _style(bound, impl)
        for metric, alpha, lw, ls in (("flip_matched", 1.0, 1.5, st["ls"]),
                                      ("flip_vehicles", 0.35, 1.0, ":")):
            by_h = {}
            for s in pts:
                if math.isfinite(s[metric]):
                    by_h.setdefault(s["H"], []).append(s[metric])
            if not by_h:
                continue
            hs = sorted(by_h)
            ys = [float(np.median(by_h[h])) for h in hs]
            ax_f.plot(hs, ys, color=st["color"], ls=ls, lw=lw, alpha=alpha,
                      marker=st["marker"] if metric == "flip_matched" else None,
                      ms=4,
                      label=(combo_label(bound, impl)
                             if metric == "flip_matched" else None))
    ax_f.set_ylabel("period-0 decisions that change\nwhen H grows one step")
    ax_f.set_xlabel("H  (horizon; T = 2H)")
    ax_f.set_ylim(bottom=0)
    ax_f.grid(alpha=0.3)
    ax_f.legend(fontsize=7, title="solid: symmetry-corrected;  dotted: raw",
                title_fontsize=7)
    fig.tight_layout()
    path = run.dir / f"{run.stem}_horizon_stability.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_heatmaps(rows, run, opts):
    """Section 2, plot 2: solve time over the (H, other) grid, one panel per combo.

    Log colour scale, because the range spans orders of magnitude and a linear
    map would render every cell but the worst one identical.  Each cell is
    annotated with the median seconds so the figure survives being printed in
    greyscale.
    """
    plt = _pyplot()
    if plt is None:
        return None
    try:
        from matplotlib.colors import LogNorm
    except Exception:
        return None
    dims = sorted({str(r.get("parameter")) for r in rows
                   if str(r.get("parameter", "")).startswith("Hx")})
    if not dims:
        return None
    paths = []
    for dim in dims:
        other = dim[2:]
        sub = [r for r in rows if r.get("parameter") == dim and usable(r, True)]
        if not sub:
            continue
        hs = sorted({int(float(r["H"])) for r in sub})
        vs = sorted({int(float(r[other])) for r in sub})
        combos = [c for c in opts.combos]
        ncol = min(2, len(combos))
        nrow = int(math.ceil(len(combos) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.8 * nrow),
                                 squeeze=False)
        grids = {}
        for (bound, impl) in combos:
            g = np.full((len(vs), len(hs)), np.nan)
            for a, v in enumerate(vs):
                for b, h in enumerate(hs):
                    vals = [_num(r, "runtime_s") for r in sub
                            if r.get("bound") == bound
                            and (r.get("reliability_impl") or "exact") == impl
                            and int(float(r["H"])) == h
                            and int(float(r[other])) == v
                            and usable(r, opts.include_censored)]
                    vals = [x for x in vals if math.isfinite(x) and x > 0]
                    if vals:
                        g[a, b] = float(np.median(vals))
            grids[(bound, impl)] = g
        finite = np.concatenate([g[np.isfinite(g)].ravel() for g in grids.values()]) \
            if grids else np.array([])
        if finite.size == 0:
            plt.close(fig)
            continue
        norm = LogNorm(vmin=max(finite.min(), 1e-3), vmax=max(finite.max(), 1e-2))
        im = None
        for idx, (bound, impl) in enumerate(combos):
            ax = axes[idx // ncol][idx % ncol]
            g = grids[(bound, impl)]
            im = ax.imshow(g, origin="lower", aspect="auto", cmap="viridis",
                           norm=norm)
            ax.set_xticks(range(len(hs)))
            ax.set_xticklabels([str(h) for h in hs])
            ax.set_yticks(range(len(vs)))
            ax.set_yticklabels([str(v) for v in vs])
            ax.set_xlabel("H")
            ax.set_ylabel(other)
            ax.set_title(combo_label(bound, impl), fontsize=9)
            for a in range(len(vs)):
                for b in range(len(hs)):
                    if math.isfinite(g[a, b]):
                        ax.text(b, a, f"{g[a, b]:.3g}", ha="center", va="center",
                                fontsize=6.5, color="w")
        for idx in range(len(combos), nrow * ncol):
            axes[idx // ncol][idx % ncol].axis("off")
        if im is not None:
            fig.colorbar(im, ax=axes.ravel().tolist(), label="median solve time [s]",
                         fraction=0.03)
        fig.suptitle(f"Solve time over H x {other}: does a longer horizon cost "
                     f"more when {other} grows?", fontsize=10)
        path = run.dir / f"{run.stem}_heatmap_H{other}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def make_plots(study: str, rows: list, run, opts) -> list:
    lines = []

    def note(p):
        if p:
            for q in (p if isinstance(p, list) else [p]):
                print(f"  [plot] {q.name}")
                lines.append(f"plot: {q.name}")

    if study == "scaling":
        note(plot_effort_vs_params(rows, run, opts, "runtime_s",
                                   "solve time [s]",
                                   "Solve time vs problem dimension",
                                   "time_vs_param"))
        note(plot_effort_vs_params(rows, run, opts, "nodes",
                                   "branch-and-bound nodes",
                                   "Search-tree size vs problem dimension",
                                   "nodes_vs_param"))
        note(plot_effort_vs_params(rows, run, opts, "objective",
                                   "objective J",
                                   "Optimal cost vs problem dimension",
                                   "cost_vs_param"))
        note(plot_relaxation(rows, run, opts))
    elif study == "horizon":
        note(plot_horizon(rows, horizon_stability(rows), run, opts))
    elif study == "heatmap":
        note(plot_heatmaps(rows, run, opts))
    elif study == "convergence":
        note(plot_convergence(rows, run, opts))
    return lines


# ===========================================================================
# Summaries
# ===========================================================================
def growth_exponent(series) -> tuple:
    """Fit log(y) = a + b*log(x) and log(y) = a + c*x, return (b, c, which).

    Both are reported because they answer different questions and the plot
    cannot distinguish them by eye at four points: `b` is the polynomial degree
    (b ~ 1 linear, b ~ 3 cubic), `c` is the exponential rate.  Whichever fits
    better (higher R^2) is named, so the text under the figure can say
    "roughly cubic in F" or "exponential in H" with a number behind it.
    """
    xs = np.array([p[0] for p in series], dtype=float)
    ys = np.array([p[1] for p in series], dtype=float)
    ok = np.isfinite(xs) & np.isfinite(ys) & (xs > 0) & (ys > 0)
    xs, ys = xs[ok], ys[ok]
    if xs.size < 3:
        return math.nan, math.nan, "too few points"
    ly = np.log(ys)

    def fit(a):
        b, c = np.polyfit(a, ly, 1)
        resid = ly - (b * a + c)
        ss = float(np.sum((ly - ly.mean()) ** 2))
        return b, (1.0 - float(np.sum(resid ** 2)) / ss if ss > 0 else 0.0)

    b_pow, r_pow = fit(np.log(xs))
    b_exp, r_exp = fit(xs)
    which = (f"power law, exponent {b_pow:.2f} (R2={r_pow:.2f})" if r_pow >= r_exp
             else f"exponential, rate {b_exp:.2f}/unit (R2={r_exp:.2f})")
    return float(b_pow), float(b_exp), which


def summarise(study: str, rows: list, opts) -> list:
    """The text a reader needs before believing the figures."""
    lines = ["", f"--- {study}: summary " + "-" * 40]
    status = {}
    for r in rows:
        status[str(r.get("status"))] = status.get(str(r.get("status")), 0) + 1
    lines.append(f"rows: {len(rows)}   status: {status}")

    skipped = [r for r in rows if r.get("status") == "skipped"]
    if skipped:
        reasons = sorted({str(r.get("skipped")) for r in skipped})
        lines.append(f"skipped {len(skipped)} row(s):")
        lines += [f"    {q}" for q in reasons]

    cens = [r for r in rows if r.get("censored") in (True, "True", "true")]
    if cens:
        lines.append(f"WARNING {len(cens)} run(s) stopped at the time limit "
                     f"({opts.time_limit}s). Their solve times are LOWER BOUNDS "
                     f"and are excluded from the bands unless --include-censored.")
        for r in sorted(cens, key=lambda q: str(q.get("config_label")))[:12]:
            lines.append(f"    {r.get('config_label')} {r.get('combo')} "
                         f"seed={r.get('seed')} gap={_fmt(r.get('mip_gap'))}")

    # does the reliability constraint actually bite here? (design note 5)
    lines.append("")
    lines.append("binding check -- a bound only shapes the schedule when "
                 "n_max < load (= T*M/F):")
    seen = set()
    hdr = f"    {'configuration':<26}{'load':>7}  " + "".join(
        f"{b[:9]:>11}" for b, _ in opts.combos)
    lines.append(hdr)
    for r in rows:
        cid = str(r.get("config_id"))
        if cid in seen or r.get("status") == "skipped":
            continue
        seen.add(cid)
        peers = {str(q.get("bound")): q for q in rows if q.get("config_id") == cid}
        cells = []
        for bound, _impl in opts.combos:
            q = peers.get(bound)
            nm = _num(q, "n_max_analytic") if q else math.nan
            ld = _num(q, "load") if q else math.nan
            flag = "bind" if (math.isfinite(nm) and math.isfinite(ld)
                              and nm < ld) else "slack"
            cells.append(f"{nm:>7.2f}{flag[0]:>4}")
        ld = _num(r, "load")
        lines.append(f"    {cid:<26}{ld:>7.2f}  " + "".join(cells))
        if len(seen) >= 24:
            lines.append("    ... (truncated; the CSV has every row)")
            break
    lines.append("    'b' = binding, 's' = slack. A slack row means that bound's "
                 "curve is the unconstrained schedule.")

    if study == "scaling":
        lines.append("")
        lines.append("growth fits (median solve time, censored rows excluded):")
        for param in SCALE_PARAMS:
            sub = [r for r in rows if r.get("parameter") == param]
            if not sub:
                continue
            agg = aggregate(sub, "runtime_s", "value", opts.band,
                            opts.include_censored)
            nodes = aggregate(sub, "nodes", "value", opts.band,
                              opts.include_censored)
            for (bound, impl) in opts.combos:
                s = agg.get((bound, impl))
                if not s:
                    continue
                _, _, which = growth_exponent(s)
                sn = nodes.get((bound, impl))
                nwhich = growth_exponent(sn)[2] if sn else "n/a"
                lines.append(f"    {param}  {combo_label(bound, impl):<20} "
                             f"time: {which:<42} nodes: {nwhich}")
        lines.append("    Read the two together: nodes growing like the time is a "
                     "search-tree explosion; nodes flat while time grows means the "
                     "per-node LP got more expensive, not the tree bigger.")

    if study == "horizon":
        st = horizon_stability(rows)
        lines.append("")
        lines.append("first-period stability -- fraction of period-0 decisions "
                     "that change when H grows one step:")
        lines.append(f"    {'combo':<22}{'H->H+':>10}{'matched':>10}"
                     f"{'raw':>8}{'dJ/T':>10}")
        for s in sorted(st, key=lambda q: (str(q["bound"]), q["H"])):
            lines.append(f"    {combo_label(s['bound'], s['reliability_impl']):<22}"
                         f"{str(s['H']) + '->' + str(s['H_next']):>10}"
                         f"{_fmt(s['flip_matched'], 3):>10}"
                         f"{_fmt(s['flip_vehicles'], 3):>8}"
                         f"{_fmt(s['d_obj_per_step'], 4):>10}")
        tail = [s for s in st if s["H"] >= max((q["H"] for q in st), default=0) * 0.6]
        vals = [s["flip_matched"] for s in tail if math.isfinite(s["flip_matched"])]
        if vals:
            lines.append(f"    over the top 40% of the ladder the matched flip "
                         f"fraction is {np.median(vals):.3f} on median: "
                         + ("the plan has settled, so H* is inside the ladder."
                            if np.median(vals) < 0.05 else
                            "the plan is STILL moving, so myopia persists and H* "
                            "is beyond this ladder -- say so rather than picking "
                            "the largest H you could afford."))

    if study == "convergence":
        lines.append("")
        lines.append("bound vs incumbent:")
        lines.append(f"    {'point':<12}{'combo':<22}{'t_1st inc':>10}"
                     f"{'gap@1st':>9}{'t_last':>9}{'final gap':>11}{'nodes':>10}")
        for r in sorted(rows, key=lambda q: str(q.get("value"))):
            if r.get("status") == "skipped":
                continue
            lines.append(f"    {str(r.get('value')):<12}{str(r.get('combo')):<22}"
                         f"{_fmt(r.get('t_first_incumbent'), 2):>10}"
                         f"{_fmt(r.get('gap_at_first_incumbent'), 3):>9}"
                         f"{_fmt(r.get('t_last_improvement'), 2):>9}"
                         f"{_fmt(r.get('mip_gap'), 5):>11}"
                         f"{_fmt(r.get('nodes'), 0):>10}")
        lines.append("    A gap that closes because the INCUMBENT drops means the "
                     "heuristics were the bottleneck; one that closes because the "
                     "BOUND rises means the relaxation was.")
    return lines


# ===========================================================================
# Planning (no solving)
# ===========================================================================
def plan(configs_by_study: dict, sc0: StudyScenario, opts) -> int:
    """Print the work matrix and the wall-clock arithmetic, then exit.

    Worth running before every array submission: it is the only cheap way to
    find out that a ladder you just widened has quadrupled the budget, or that
    an 8x rung was going to be skipped for `F > M` all along.
    """
    total_units = 0
    print(f"base case B: {sc0.study_label()}  "
          f"(T={sc0.T}, {sc0.size_binaries} binaries, load={sc0.load:.2f})")
    print(f"combos     : {[combo_label(*c) for c in opts.combos]}")
    print(f"seeds      : {opts.seeds_small} when <= {opts.small_size} binaries, "
          f"else {opts.seeds_large}")
    print(f"time limit : {opts.time_limit}s per solve   mip_gap={opts.mip_gap}")
    print()
    for study, configs in configs_by_study.items():
        print(f"=== {study} " + "=" * 56)
        hdr = (f"  {'configuration':<26}{'F':>3}{'T':>4}{'bins':>7}{'cells':>6}"
               f"{'load':>7}{'seeds':>6}{'solves':>7}  {'screen':<12}note")
        print(hdr)
        study_units = 0
        for cfg in configs:
            sc = cfg.scenario
            reason = skip_reason(cfg, opts)
            seeds = seeds_for(cfg, opts)
            units = len(seeds) * len(opts.combos)
            study_units += units
            # Pre-flight feasibility: `feasible_hint` is a cheap necessary
            # condition (survival floor + repair capacity). A bound listed here
            # will almost certainly come back 'infeasible' in milliseconds, which
            # is a wasted rung, not a data point -- catch it before the cluster
            # does. See design note 4.
            miss = sorted({b for b, _ in opts.combos
                           if not feasible_hint(sc, b)}) if not reason else []
            screen = ("no:" + ",".join(q[:4] for q in miss)) if miss else "ok"
            flag = f"SKIP: {reason}" if reason else cfg.note[:0]
            print(f"  {cfg.label:<26}{sc.F:>3}{sc.T:>4}{sc.size_binaries:>7}"
                  f"{sc.size_cells:>6}{sc.load:>7.2f}{len(seeds):>6}"
                  f"{units:>7}  {screen:<12}{flag}")
        skipped = sum(len(seeds_for(c, opts)) * len(opts.combos)
                      for c in configs if skip_reason(c, opts))
        print(f"  -> {study_units} work units ({skipped} of them skipped, "
              f"{study_units - skipped} solves)")
        total_units += study_units
        notes = sorted({c.note for c in configs if c.note})
        for n in notes:
            print(f"  NOTE {n}")
        print()

    tl = opts.time_limit or 0.0
    print(f"TOTAL {total_units} work units.")
    n_bad = sum(1 for cs in configs_by_study.values() for c in cs
                if not skip_reason(c, opts)
                and any(not feasible_hint(c.scenario, b) for b, _ in opts.combos))
    if n_bad:
        print(f"WARNING {n_bad} configuration(s) failed the feasibility screen "
              f"(the 'screen' column). Those bounds will return 'infeasible' in "
              f"milliseconds, which is not a solve-time measurement. Raise "
              f"--calibrate-n, lower --severity-spread, or give the sweep more "
              f"depot headroom before spending cluster time on them.")
    if tl:
        print(f"Worst case (every solve hits the {tl:.0f}s limit): "
              f"{total_units * tl / 3600:.1f} h of solver time.")
    n = int(opts.plan_shards)
    print(f"With --shard k/{n}: about {math.ceil(total_units / n)} units per task"
          + (f", worst case {math.ceil(total_units / n) * tl / 3600:.1f} h "
             f"per task." if tl else "."))
    print()
    print("Unresolved before you submit:")
    print("  * is the base case really easy? run --studies scaling with "
          "--factors 1 first and look at the times.")
    print("  * do the four bounds bite at the design point? check the binding "
          "table in the summary; retune with --calibrate-n if three of them are "
          "slack everywhere.")
    print("  * --lp-relax solves a second (continuous) model per row. Turn it "
          "off if the LP itself becomes the cost.")
    return 0


# ===========================================================================
# Merge (reduce step for a Slurm array)
# ===========================================================================
def merge(out_root: Path, study: str, opts) -> int:
    """Concatenate the shard CSVs of one run folder, then plot and summarise once.

    A shard sees a slice of the design points, so it can neither aggregate over
    seeds nor compare consecutive H values -- both of those live here.
    """
    stamp = (getattr(opts, "run_stamp", None) or os.environ.get("RUN_STAMP", "")).strip()
    folders = sorted(d for d in out_root.glob(f"*_{study}") if d.is_dir())
    if stamp:
        folders = [d for d in folders if d.name.startswith(stamp)]
    if not folders:
        print(f"[merge] no run folder matching {out_root}/{stamp or '*'}_{study}",
              file=sys.stderr)
        return 1
    folder = folders[-1]
    if len(folders) > 1 and not stamp:
        print(f"[merge] {len(folders)} folders match; using the newest: "
              f"{folder.name}  (pass --run-stamp to pick another)")

    csvs = sorted(folder.glob("results_shard*.csv")) or sorted(folder.glob("results.csv"))
    if not csvs:
        print(f"[merge] {folder.name} has no results*.csv", file=sys.stderr)
        return 1
    rows = []
    for q in csvs:
        got = _read_rows(q)
        print(f"[merge] {q.name}: {len(got)} rows")
        rows += got
    if not rows:
        print("[merge] every shard file is header-only: the solves never ran. "
              "Check sacct, then the .err logs, then progress_shard*.log.",
              file=sys.stderr)
        return 1

    # a resubmitted shard may have produced a row twice: newest wins
    unique = {}
    for rec in rows:
        key = (rec.get("study"), rec.get("parameter"), str(rec.get("value")),
               rec.get("bound"), rec.get("reliability_impl"), str(rec.get("seed")))
        prev = unique.get(key)
        if prev is None or str(rec.get("timestamp", "")) >= str(prev.get("timestamp", "")):
            unique[key] = rec
    rows = list(unique.values())
    for rec in rows:                              # CSV round-trip loses the bool
        rec["censored"] = str(rec.get("censored", "")).lower() == "true"

    csv_path = folder / "merged_results.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=H.FIELDS, extrasaction="ignore")
        w.writeheader()
        for rec in sorted(rows, key=lambda r: (str(r.get("parameter")),
                                               str(r.get("value")),
                                               str(r.get("bound")),
                                               str(r.get("seed")))):
            w.writerow({k: rec.get(k, "") for k in H.FIELDS})

    run = SimpleNamespace(dir=folder, stem="merged", rows=rows)
    report = [f"# merged {study} report  {datetime.now():%Y-%m-%d %H:%M:%S}",
              f"# folder={folder.name}  shard files={len(csvs)}  rows={len(rows)}",
              ""]
    versions = {(r.get("git_branch"), r.get("git_commit")) for r in rows
                if r.get("git_commit")}
    if len(versions) > 1:
        report.append(f"WARNING {len(versions)} different code versions produced "
                      f"these shards: {sorted(f'{b}@{c}' for b, c in versions)}. "
                      f"Do not edit or pull while an array is running.")
    report += summarise(study, rows, opts)
    if _plots_enabled(opts):
        report += make_plots(study, rows, run, opts)

    if study == "horizon":
        st = horizon_stability(rows)
        if st:
            path = folder / "merged_horizon_stability.csv"
            with path.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(st[0]))
                w.writeheader()
                w.writerows(st)
            report.append(f"stability table: {path.name}")

    (folder / "merged_summary.txt").write_text("\n".join(report))
    _dump_yaml(folder / "merged_results.yaml",
               {"study": study, "folder": folder.name, "n_rows": len(rows),
                "created": datetime.now().isoformat(timespec="seconds"),
                "rows": [_to_builtin(r) for r in rows]})
    print("\n".join(report))
    print(f"\n[merge] wrote {csv_path.name}, merged_results.yaml, merged_summary.txt")
    return 0


# ===========================================================================
# CLI
# ===========================================================================
def _int_list(text) -> list:
    return [int(_clean(v)) for v in str(text).split(",") if _clean(v)]


def _parse_points(text) -> dict:
    """'small=4:1:1:4;medium=8:2:2:6' -> {'small': (4,1,1,4), ...}"""
    out = {}
    for block in str(text).split(";"):
        if not _clean(block):
            continue
        name, _, spec = _clean(block).partition("=")
        parts = [int(_clean(v)) for v in spec.split(":") if _clean(v)]
        if len(parts) != 4:
            raise SystemExit(f"--conv-points: {block!r} must read NAME=F:M:L:H")
        out[_clean(name)] = tuple(parts)
    if not out:
        raise SystemExit("--conv-points selected nothing")
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Scalability and solution-quality studies for the "
                    "degradation-aware EV fleet scheduler.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Start with:  python run_studies.py --plan")
    p.add_argument("--studies", default="scaling",
                   help="comma list of scaling,horizon,heatmap,convergence "
                        "(default: scaling)")
    p.add_argument("--out", default="results",
                   help="root output directory; each study gets its own "
                        "<YYYYMMDDHHMM>_<study> folder")
    p.add_argument("--name", default="studies",
                   help="run name recorded in the YAML")
    p.add_argument("--combos", default=",".join(f"{b}/{i}" for b, i in DEFAULT_COMBOS),
                   help="bound/implementation pairs to compare, e.g. "
                        "'markov/exact,cantelli/tangent,hoeffding/tangent'. "
                        "Adding an /exact quadratic pair turns the model into a "
                        "nonconvex MIQCP and the 'LP gap' column into a QCP bound.")

    # ---- base case B ----
    g = p.add_argument_group("base case B")
    g.add_argument("--F", type=int, default=3, help="base fleet size (default 3)")
    g.add_argument("--M", type=int, default=1, help="base mission count (default 1)")
    g.add_argument("--L", type=int, default=1, help="base components (default 1)")
    g.add_argument("--H", type=int, default=4,
                   help="base horizon; T = 2H (default 4)")
    g.add_argument("--tau", type=float, default=None)
    g.add_argument("--epsilon", type=float, default=None)
    g.add_argument("--rho", type=float, default=None)
    g.add_argument("--p", type=float, default=None)
    g.add_argument("--severity-spread", type=float, default=0.0,
                   dest="severity_spread",
                   help="missions span b_ref*(1 -+ spread) (default 0.0, NOT "
                        "the Scenario default of 0.25). bernstein's drift term "
                        "uses the WORST mission's support, so a wide spread "
                        "collapses its budget to zero as soon as M > 1 and every "
                        "bernstein run in the study comes back infeasible. Widen "
                        "it only after re-running the feasibility screen.")
    g.add_argument("--calibrate-n", type=float, default=6.0, dest="n_target",
                   help="place the design point where hoeffding binds after this "
                        "many reference missions (default 6). Lower it to make "
                        "the constraint bite sooner -- but below about 5 the "
                        "bernstein budget collapses and the feasibility screen "
                        "will say so.")
    g.add_argument("--C-M", type=float, default=None, dest="C_M")
    g.add_argument("--C-R", type=float, default=None, dest="C_R")
    g.add_argument("--C-S", type=float, default=None, dest="C_S")
    g.add_argument("--C-P", type=float, default=None, dest="C_P")
    g.add_argument("--repair", default=None, dest="repair_model",
                   choices=["ard1", "ardinf"],
                   help="ardinf (default in Scenario) is the only model chernoff "
                        "supports; keep it fixed across the study")
    g.add_argument("--tangent-ref", type=float, default=None, dest="tangent_ref")
    g.add_argument("--pwl-points", type=int, default=None, dest="pwl_points")
    g.add_argument("--formulation", default=None, dest="formulation",
                   choices=list(H.FORMULATIONS_ORDER),
                   help="encoding x assembly x strengthening of the logical "
                        "constraints. ENCODING: 'indicator' (default) or "
                        "'bigm' (nb substituted out, linear big-M rows); these "
                        "share the integer optimum and differ in the "
                        "relaxation, which is exactly what the lp_gap column "
                        "measures. ASSEMBLY: the '_sparse' twins build the same "
                        "program through the matrix API -- same rows, same "
                        "relaxation, 2x faster to build under 'indicator' and "
                        "5-7x under 'bigm'. STRENGTHENING: 'indicator_cuts' / "
                        "'indicator_cuts_core' add the locally-supported valid "
                        "inequalities of rainflow_v2.add_sparse_cuts on top of "
                        "the indicator encoding -- same integer optimum, "
                        "non-trivial root bound, so this is the option lp_gap "
                        "exists to measure.")
    g.add_argument("--sparse-cuts", default=None, dest="sparse_cuts",
                   choices=["off", "core", "full"],
                   help="add the locally-supported valid inequalities of "
                        "rainflow_v2.add_sparse_cuts on top of the indicator "
                        "encoding. Same integer optimum, non-trivial root "
                        "bound -- so lp_gap is the column to read. Equivalent "
                        "to '--formulation indicator_cuts'. Ignored under "
                        "'bigm', whose rows already imply the cuts.")
    g.add_argument("--bigM", type=float, default=None, dest="bigM",
                   help="fallback big-M for a state with no finite bound "
                        "(default 1.1)")
    g.add_argument("--allow-replacement", action="store_true", default=True)

    # ---- ladders ----
    g = p.add_argument_group("ladders")
    g.add_argument("--factors", default="1,2,4,8",
                   help="geometric ladder applied to F, M, L, H (default 1,2,4,8)")
    g.add_argument("--scale-params", default="F,M,L,H",
                   help="which dimensions the scaling study sweeps")
    g.add_argument("--m-sweep-mode", default="fixed-depot", dest="m_sweep_mode",
                   choices=["fixed-depot", "fixed-fleet"],
                   help="how the M ladder keeps F > M. 'fixed-depot' (default) "
                        "sets F = M + --m-depot-headroom, so the number of "
                        "maintenance slots per step stays constant; "
                        "'fixed-fleet' pins F at --m-sweep-F and lets depot "
                        "capacity shrink (see design note 6)")
    g.add_argument("--m-depot-headroom", type=int, default=8,
                   dest="m_depot_headroom",
                   help="F - M held constant by --m-sweep-mode fixed-depot "
                        "(default 8, so the M=1 rung is F=9)")
    g.add_argument("--m-sweep-F", type=int, default=9, dest="m_sweep_F",
                   help="fleet size for the M ladder in 'fixed-fleet' mode, and "
                        "the M=1 anchor in either mode (default 9)")
    g.add_argument("--h-values", default="2,3,4,6,8,10,12,16",
                   help="H ladder for the horizon study (fine, not geometric)")
    g.add_argument("--horizon-F", type=int, default=4, dest="horizon_F")
    g.add_argument("--horizon-M", type=int, default=2, dest="horizon_M",
                   help="M > 1 in the horizon study so period-0 assignments have "
                        "something to choose between (default 2)")
    g.add_argument("--horizon-L", type=int, default=1, dest="horizon_L")
    g.add_argument("--heatmap-h", default="2,4,6,8",
                   help="H axis of the heatmaps")
    g.add_argument("--heatmap-other", default="1,2,3,4",
                   help="second axis of the heatmaps (values of M or L)")
    g.add_argument("--heatmap-dims", default="M,L",
                   help="which second dimensions to grid against H")
    g.add_argument("--heatmap-F", type=int, default=9, dest="heatmap_F",
                   help="fleet size for the H x M heatmap. Held CONSTANT across "
                        "the grid so a colour difference is about H and M, not "
                        "about the fleet -- which means it must be large enough "
                        "that the largest M still leaves depot slots free "
                        "(default 9 for M up to 4).")
    g.add_argument("--conv-points",
                   default="small=4:1:1:4;medium=8:2:2:6;large=12:3:2:10",
                   help="representative points for the convergence study, "
                        "NAME=F:M:L:H separated by ';'")

    # ---- instance budget ----
    g = p.add_argument_group("instances and seeds")
    g.add_argument("--seeds-small", type=int, default=3, dest="seeds_small",
                   help="seeds for cheap settings (default 3)")
    g.add_argument("--seeds-large", type=int, default=2, dest="seeds_large",
                   help="seeds for expensive settings (default 2)")
    g.add_argument("--small-size", type=int, default=600, dest="small_size",
                   help="binary count below which a setting counts as cheap")
    g.add_argument("--seed0", type=int, default=0, help="first seed (default 0)")
    g.add_argument("--heatmap-seeds", type=int, default=2, dest="heatmap_seeds")
    g.add_argument("--conv-seeds", type=int, default=1, dest="conv_seeds")
    g.add_argument("--mu0-jitter", type=float, default=0.0, dest="mu0_jitter",
                   help="randomise initial damage, as a fraction of the "
                        "markov-safe headroom eps*tau/(1-rho). 0 (default) keeps "
                        "the fleet uniform -- see design note 2 before raising it.")
    g.add_argument("--no-severity-jitter", action="store_true",
                   help="keep the deterministic linspace severity profile, so a "
                        "seed only changes Gurobi's random seed")
    g.add_argument("--max-binaries", type=int, default=4000, dest="max_binaries",
                   help="skip (and record) configurations larger than this")

    # ---- solver ----
    g = p.add_argument_group("solver")
    g.add_argument("--mip-gap", type=float, default=0.12,
                   help="MIP gap (default 1e-4; the model default 0.12 would make "
                        "every solve time a measurement of the heuristic, not of "
                        "the search)")
    g.add_argument("--time-limit", type=float, default=300.0,
                   help="per-solve time limit in seconds (default 300); "
                        "<= 0 means no limit")
    g.add_argument("--no-time-limit", action="store_true")
    g.add_argument("--threads", type=int, default=None,
                   help="Gurobi Threads. On a cluster node ALWAYS set this to "
                        "$SLURM_CPUS_PER_TASK, and keep it FIXED across the whole "
                        "study -- a timing curve measured on a varying thread "
                        "count is not a timing curve.")
    g.add_argument("--gurobi-params", action="append", default=None,
                   dest="gurobi_params", metavar="K=V,...",
                   help="extra Gurobi parameters, merged across occurrences")
    g.add_argument("--no-lp-relax", action="store_true",
                   help="skip the continuous relaxation (drops the lp_gap column "
                        "and roughly halves the model-building cost)")
    g.add_argument("--lp-time-limit", type=float, default=60.0,
                   dest="lp_time_limit",
                   help="time limit for the relaxation solve (default 60s)")
    g.add_argument("--trace", default="auto", choices=["auto", "on", "off"],
                   help="record the (time, incumbent, ObjBound) trajectory. "
                        "'auto' = on for the convergence study only.")
    g.add_argument("--trace-min-dt", type=float, default=0.25, dest="trace_min_dt",
                   help="minimum seconds between trajectory samples (default 0.25)")
    g.add_argument("--solver-log", default="off", dest="solver_log",
                   choices=["auto", "on", "off"])
    g.add_argument("--verbose", type=int, default=0)

    # ---- run control ----
    g = p.add_argument_group("run control")
    g.add_argument("--shard", default=None, metavar="K/N",
                   help="run only work unit k of n, for Slurm arrays: "
                        "--shard $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT")
    g.add_argument("--run-stamp", default=None, dest="run_stamp",
                   metavar="YYYYMMDDHHMM",
                   help="pins the run folder so every shard writes into ONE "
                        "folder; also selects the folder to --merge")
    g.add_argument("--merge", action="store_true",
                   help="reduce the shard CSVs, then plot and summarise once")
    g.add_argument("--plan", action="store_true",
                   help="print the work matrix and the budget, then exit")
    g.add_argument("--plan-shards", type=int, default=12, dest="plan_shards")
    g.add_argument("--dry-run", action="store_true",
                   help="build and validate every input, solve nothing")
    g.add_argument("--no-plots", action="store_true")
    g.add_argument("--band", default="iqr", choices=["iqr", "minmax", "std"],
                   help="what the shaded band shows (default iqr)")
    g.add_argument("--include-censored", action="store_true",
                   help="let time-limited runs into the medians. Off by default: "
                        "they are lower bounds, not measurements.")

    args = p.parse_args(argv)
    if args.no_time_limit or (args.time_limit is not None and args.time_limit <= 0):
        args.time_limit = None
    args.lp_relax = not args.no_lp_relax
    args.combos = parse_combos(args.combos)
    args.factors = _int_list(args.factors)
    args.scale_params = [q for q in (_clean(v) for v in args.scale_params.split(","))
                         if q]
    for q in args.scale_params:
        if q not in SCALE_PARAMS:
            raise SystemExit(f"--scale-params: unknown {q!r}; pick from {SCALE_PARAMS}")
    args.h_values = _int_list(args.h_values)
    args.heatmap_h = _int_list(args.heatmap_h)
    args.heatmap_other = _int_list(args.heatmap_other)
    args.heatmap_dims = [q for q in (_clean(v) for v in args.heatmap_dims.split(","))
                         if q]
    for q in args.heatmap_dims:
        if q not in ("M", "L"):
            raise SystemExit(f"--heatmap-dims: pick from M,L (got {q!r})")
    args.conv_points = _parse_points(args.conv_points)
    args.shard_obj = None
    if args.shard:
        try:
            k, n = (int(q) for q in args.shard.split("/"))
        except ValueError:
            raise SystemExit("--shard expects K/N, e.g. --shard 3/20")
        args.shard_obj = Shard(k, n)
    # `test.py` helpers read these names off the options object
    args.reliability_impl = args.combos[0][1]
    args.impl_list = sorted({i for _, i in args.combos})
    args.trace_min_rel = 1e-9
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    studies = [q for q in (_clean(v) for v in args.studies.split(",")) if q]
    known = ("scaling", "horizon", "heatmap", "convergence")
    for s in studies:
        if s not in known:
            raise SystemExit(f"unknown study {s!r}; pick from {known}")

    if args.merge:
        rc = 0
        for s in studies:
            rc |= merge(out_root, s, args)
        return rc

    sc0 = base_scenario(args)
    configs = build_configs(sc0, args, studies)

    if args.plan:
        return plan(configs, sc0, args)

    header = [f"# fleet scheduling studies  {datetime.now():%Y-%m-%d %H:%M:%S}",
              f"# combos: {[combo_label(*c) for c in args.combos]}",
              f"# base B: {sc0.study_label()}",
              f"# mip_gap={args.mip_gap} "
              f"time_limit={'none' if args.time_limit is None else args.time_limit} "
              f"threads={args.threads} lp_relax={args.lp_relax} "
              f"dry_run={args.dry_run}",
              (f"# host={_HOSTNAME} slurm_job={_SLURM_JOB}" if _SLURM_JOB else ""),
              (f"# code: branch={_GIT['git_branch']} commit={_GIT['git_commit']}"
               if _GIT["git_commit"] else ""),
              ""]

    suffix = "" if args.shard_obj is None else f"_shard{args.shard_obj.k}"
    for study in studies:
        if args.shard_obj is not None:
            args.shard_obj.i = 0                  # restart the unit counter
        run = StudyRun(out_root, args.name, study, sc0, args, suffix=suffix)
        report = header + [f"STUDY {study}", "=" * 72]
        extra = None
        try:
            lines, extra = run_study(study, configs[study], sc0, args, run)
            report += lines
        except BaseException as exc:              # Ctrl-C included: keep the data
            report += ["", f"ABORTED: {type(exc).__name__}: {exc}"]
            run.close(report, extra)
            raise
        run.close(report, extra)
        print("\n".join(report))
        print(f"\n[{study}] folder : {run.dir}")
        print(f"[{study}] results: {run.csv_path.name}, {run.yaml_path.name}, "
              f"{run.summary_path.name}, runs/ ({len(run.rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
