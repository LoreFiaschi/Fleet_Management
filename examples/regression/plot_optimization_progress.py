#!/usr/bin/env python3
"""Plot incumbent, lower bound and relative MIP gap over optimizer time."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path, help="Solver result YAML")
    parser.add_argument("output", type=Path, help="Output PNG or PDF")
    parser.add_argument(
        "--title", default="Optimization convergence",
        help="Figure title",
    )
    args = parser.parse_args()

    progress = _read_progress(args.result)
    if not progress:
        raise ValueError(
            "Result contains no optimization_progress. Re-run with the updated solver."
        )

    time_minutes = np.asarray(
        [float(row["runtime_seconds"]) / 60.0 for row in progress]
    )
    incumbent = np.asarray([
        np.nan if row.get("incumbent") is None else float(row["incumbent"])
        for row in progress
    ])
    bound = np.asarray([
        np.nan if row.get("best_bound") is None else float(row["best_bound"])
        for row in progress
    ])
    gap_percent = np.asarray([
        np.nan if row.get("relative_gap") is None
        else 100.0 * float(row["relative_gap"])
        for row in progress
    ])

    fig, (ax_obj, ax_gap) = plt.subplots(
        2, 1, figsize=(9.0, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    ax_obj.step(time_minutes, incumbent, where="post", label="Incumbent", lw=2)
    ax_obj.step(time_minutes, bound, where="post", label="Best bound", lw=2)
    ax_obj.fill_between(
        time_minutes, bound, incumbent, step="post", alpha=0.15,
        label="Uncertified objective interval",
    )
    ax_obj.set_ylabel("Objective")
    ax_obj.grid(alpha=0.3)
    ax_obj.legend()

    ax_gap.step(time_minutes, gap_percent, where="post", color="tab:orange", lw=2)
    ax_gap.axhline(5.0, color="tab:green", ls="--", label="5% target")
    ax_gap.set_xlabel("Optimizer time [min]")
    ax_gap.set_ylabel("MIP gap [%]")
    ax_gap.set_ylim(bottom=0.0)
    ax_gap.grid(alpha=0.3)
    ax_gap.legend()

    fig.suptitle(args.title)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Wrote {args.output}")


def _read_progress(path: Path) -> list[dict]:
    """Read callback data from YAML or recover rows from a Gurobi text log."""
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data.get("optimization_progress") or []

    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        tokens = line.split()
        gap_index = next(
            (index for index, token in enumerate(tokens) if token.endswith("%")),
            None,
        )
        if gap_index is None or gap_index < 2:
            continue
        runtime_match = re.fullmatch(r"([0-9.]+)s", tokens[-1])
        if runtime_match is None:
            continue
        try:
            incumbent = float(tokens[gap_index - 2])
            bound = float(tokens[gap_index - 1])
            gap = float(tokens[gap_index][:-1]) / 100.0
        except ValueError:
            continue
        first = tokens[0].lstrip("H*")
        nodes = float(first) if first.isdigit() else np.nan
        rows.append({
            "runtime_seconds": float(runtime_match.group(1)),
            "incumbent": incumbent,
            "best_bound": bound,
            "relative_gap": gap,
            "nodes": nodes,
            "solutions": None,
        })
    return rows


if __name__ == "__main__":
    main()
