#!/usr/bin/env python3
"""fix_root_gap.py -- recompute root_bound / root_gap from the saved traces.

Why this exists
---------------
`solve_instrumented` used to record the FIRST finite dual bound the callback
reported.  `GRB.Callback.MIP` fires before Gurobi has processed the root node, so
that value is the trivial bound -- 0 for a non-negative objective -- and every
row came out with `root_bound = 0` and `root_gap = 1.0` exactly.  The correct
value is the bound Gurobi leaves the ROOT NODE with, i.e. the last callback
sample taken while `NodeCount` was still 0.

The traces already contain it (`traces/*.csv` has t_s, incumbent, obj_bound,
gap, nodes), so this is a pure post-processing fix: no model is rebuilt and no
solve is repeated.  run_studies.py has been corrected for future runs; this
script repairs results produced before the fix.

Usage
-----
    python fix_root_gap.py results/202608211813_scaling
    python fix_root_gap.py results/202608211813_scaling --apply

Without --apply it only reports what would change.  With --apply it rewrites
merged_results.csv in place (after saving merged_results.csv.orig) and you can
redraw the figures with:

    RUN_STAMP=<stamp> STUDIES=scaling sbatch euler/merge_studies.sbatch

...but note the merge REBUILDS merged_results.csv from the shard files, so run
this on the shards instead if you want the fix to survive a re-merge:

    python fix_root_gap.py results/<stamp>_scaling --shards --apply
"""
from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path


def root_bound_from_trace(path: Path):
    """Largest dual bound recorded while the search was still at the root.

    Minimisation, so the bound only rises: the last root sample is the largest,
    and taking max() is robust to a trace whose nodes column is not monotone.
    Returns (root_bound, n_root_samples, final_bound).
    """
    at_root, finite = [], []
    try:
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    b = float(row["obj_bound"])
                    n = float(row["nodes"] or 0)
                except (TypeError, ValueError, KeyError):
                    continue
                if not math.isfinite(b):
                    continue
                finite.append(b)
                if n <= 0:
                    at_root.append(b)
    except OSError:
        return math.nan, 0, math.nan
    rb = max(at_root) if at_root else (finite[0] if finite else math.nan)
    return rb, len(at_root), (finite[-1] if finite else math.nan)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="a results/<stamp>_<study> directory")
    ap.add_argument("--apply", action="store_true", help="rewrite the CSVs")
    ap.add_argument("--shards", action="store_true",
                    help="fix results_shard*.csv instead of merged_results.csv, "
                         "so the correction survives a re-merge")
    args = ap.parse_args()

    folder = Path(args.folder)
    targets = (sorted(folder.glob("results_shard*.csv")) if args.shards
               else [folder / "merged_results.csv"])
    targets = [t for t in targets if t.is_file()]
    if not targets:
        print(f"no CSV found in {folder}")
        return 1

    for target in targets:
        with target.open(newline="") as fh:
            reader = csv.DictReader(fh)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
        changed = missing = 0
        print(f"\n{target.name}: {len(rows)} rows")
        print(f"  {'config':<26}{'combo':<20}{'old rb':>9}{'new rb':>9}"
              f"{'root_gap':>10}{'root pts':>9}")
        for r in rows:
            rel = r.get("trace_file", "")
            if not rel:
                missing += 1
                continue
            rb, n_root, _final = root_bound_from_trace(folder / rel)
            if not math.isfinite(rb):
                missing += 1
                continue
            try:
                z = float(r.get("objective", ""))
            except (TypeError, ValueError):
                z = math.nan
            gap = (z - rb) / abs(z) if math.isfinite(z) and abs(z) > 1e-12 else math.nan
            old = r.get("root_bound", "")
            r["root_bound"] = f"{rb:.10g}"
            r["root_gap"] = "" if math.isnan(gap) else f"{gap:.10g}"
            changed += 1
            if changed <= 12:
                print(f"  {str(r.get('config_id')):<26}{str(r.get('combo')):<20}"
                      f"{(old or 'nan')[:9]:>9}{rb:>9.4f}{gap:>10.4f}{n_root:>9}")
        if changed > 12:
            print(f"  ... and {changed - 12} more")
        print(f"  fixed {changed}, skipped {missing} (no usable trace)")

        if args.apply and changed:
            backup = target.with_suffix(target.suffix + ".orig")
            if not backup.exists():
                shutil.copy2(target, backup)
                print(f"  backup -> {backup.name}")
            with target.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            print(f"  rewrote {target.name}")
    if not args.apply:
        print("\n(dry run -- pass --apply to rewrite)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
