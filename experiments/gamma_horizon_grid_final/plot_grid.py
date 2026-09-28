"""Plot projected cost and final MIP gap over an H1/H2 grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


def _matrix(cases: list[dict], h1_values: list[int], h2_values: list[int], key: str):
    values = np.full((len(h1_values), len(h2_values)), np.nan)
    row_of = {value: index for index, value in enumerate(h1_values)}
    column_of = {value: index for index, value in enumerate(h2_values)}
    for case in cases:
        value = case.get(key)
        if value is not None:
            values[row_of[int(case["H1"])], column_of[int(case["H2"])]] = float(value)
    return values


def _annotate(axis, values: np.ndarray, *, percent: bool = False) -> None:
    finite = values[np.isfinite(values)]
    midpoint = np.nan if finite.size == 0 else (finite.min() + finite.max()) / 2
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if not np.isfinite(value):
                label = "—"
                color = "black"
            else:
                label = f"{100 * value:.1f}%" if percent else f"{value:.3f}"
                color = "white" if value > midpoint else "black"
            axis.text(column, row, label, ha="center", va="center", color=color, fontsize=9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    report = yaml.safe_load(args.report.read_text(encoding="utf-8"))
    cases = report.get("cases", [])
    h1_values = [int(value) for value in report["planned_transitory_horizons"]]
    h2_values = [int(value) for value in report["planned_operating_horizons"]]

    cost = _matrix(cases, h1_values, h2_values, "projected_evaluation_cost")
    gap = _matrix(cases, h1_values, h2_values, "mip_gap")

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    cost_image = axes[0].imshow(cost, aspect="auto", cmap="viridis")
    gap_image = axes[1].imshow(gap, aspect="auto", cmap="magma_r", vmin=0.0)

    for axis in axes:
        axis.set_xticks(range(len(h2_values)), h2_values)
        axis.set_yticks(range(len(h1_values)), h1_values)
        axis.set_xlabel(r"Operating horizon $H_2$")
        axis.set_ylabel(r"Transitory horizon $H_1$")

    axes[0].set_title("Projected 52-period cost")
    axes[1].set_title("Final relative MIP gap")
    _annotate(axes[0], cost)
    _annotate(axes[1], gap, percent=True)
    figure.colorbar(cost_image, ax=axes[0], shrink=0.82, label="Projected cost")
    figure.colorbar(gap_image, ax=axes[1], shrink=0.82, label="Relative gap")

    best = report.get("best_feasible")
    subtitle = "No feasible pair found"
    if best is not None:
        gap_value = best.get("mip_gap")
        gap_text = "not reported" if gap_value is None else f"{gap_value:.1%}"
        subtitle = (
            f"Best feasible: H1={best['H1']}, H2={best['H2']}, "
            f"projected cost={best['projected_evaluation_cost']:.4g}, "
            f"gap={gap_text}"
        )
    figure.suptitle(f"Horizon-grid result\n{subtitle}", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
