"""Plot H1 sensitivity or the complete H1/H2 metric grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Patch, Rectangle


METRICS = (
    ("J_initialization", r"Initialization cost $J_{init}$", False, "viridis"),
    ("J_op_average", r"Operating cost $J_{op}/H_2$", False, "viridis"),
    ("projected_evaluation_cost", "Projected 52-period cost", False, "viridis"),
    ("mip_gap", "Final relative MIP gap", True, "magma_r"),
)


def outcome(case: dict) -> str:
    if str(case.get("status", "")).lower() == "infeasible":
        return "infeasible"
    if case.get("objective") is None:
        return "no_incumbent"
    return "incumbent"


def outcome_legend():
    return [
        Patch(facecolor="#d9d9d9", edgecolor="#666666", hatch="///",
              label="INF: proven infeasible"),
        Patch(facecolor="#f4cccc", edgecolor="#b85450", hatch="..",
              label="TL no inc.: time limit, no feasible incumbent found"),
    ]


def plot_h1(report: dict, output: Path) -> None:
    cases = sorted(report.get("cases", []), key=lambda row: int(row["H1"]))
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharex=True)
    axes = axes.ravel()
    for axis, (key, title, percent, _) in zip(axes, METRICS, strict=True):
        finite_x, finite_y = [], []
        for case in cases:
            value = case.get(key)
            if value is not None:
                finite_x.append(int(case["H1"]))
                finite_y.append(float(value))
        axis.plot(finite_x, finite_y, color="#2468b4", marker="o", linewidth=2)
        for x, y in zip(finite_x, finite_y, strict=True):
            label = f"{y:.1%}" if percent else f"{y:.3f}"
            axis.annotate(label, (x, y), xytext=(0, 7), textcoords="offset points",
                          ha="center", fontsize=8)
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
        axis.set_xlabel(r"Transitory horizon $H_1$")
        if finite_y:
            low, high = axis.get_ylim()
            marker_y = low + 0.05 * (high - low)
            for case in cases:
                state = outcome(case)
                if state == "incumbent":
                    continue
                color = "#666666" if state == "infeasible" else "#b85450"
                label = "INF" if state == "infeasible" else "TL"
                x = int(case["H1"])
                axis.scatter([x], [marker_y], marker="X", s=55, color=color, zorder=3)
                axis.annotate(label, (x, marker_y), xytext=(0, 7),
                              textcoords="offset points", ha="center", fontsize=8)

    h2_values = sorted({int(case["H2"]) for case in cases})
    h2_text = h2_values[0] if len(h2_values) == 1 else h2_values
    figure.suptitle(
        f"Initial-damage sensitivity at fixed $H_2={h2_text}$\n"
        r"Operating cost is reported separately from initialization cost",
        fontsize=14,
    )
    figure.legend(handles=outcome_legend(), loc="lower center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0.08, 1, 0.90))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def matrix(cases, h1_values, h2_values, key):
    values = np.full((len(h1_values), len(h2_values)), np.nan)
    states = np.full(values.shape, "missing", dtype=object)
    row_of = {value: index for index, value in enumerate(h1_values)}
    col_of = {value: index for index, value in enumerate(h2_values)}
    for case in cases:
        row, col = row_of[int(case["H1"])], col_of[int(case["H2"])]
        states[row, col] = outcome(case)
        value = case.get(key)
        if value is not None:
            values[row, col] = float(value)
    return values, states


def decorate(axis, values, states, percent):
    finite = values[np.isfinite(values)]
    midpoint = np.nan if finite.size == 0 else (finite.min() + finite.max()) / 2
    styles = {
        "infeasible": ("#d9d9d9", "#666666", "///", "INF"),
        "no_incumbent": ("#f4cccc", "#b85450", "..", "TL\nno inc."),
        "missing": ("white", "#999999", "xx", "not run"),
    }
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value, state = values[row, col], states[row, col]
            if np.isfinite(value):
                label = f"{value:.1%}" if percent else f"{value:.3f}"
                # viridis becomes bright for large values, while magma_r
                # becomes dark. Choose annotation contrast accordingly.
                color = "white" if (
                    (not percent and value < midpoint)
                    or (percent and value > midpoint)
                ) else "black"
            else:
                face, edge, hatch, label = styles[state]
                axis.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1,
                                         facecolor=face, edgecolor=edge,
                                         hatch=hatch, linewidth=0.8, zorder=2))
                color = "black"
            axis.text(col, row, label, ha="center", va="center",
                      fontsize=7.5, color=color, zorder=3)


def plot_grid(report: dict, output: Path) -> None:
    cases = report.get("cases", [])
    h1_values = [int(v) for v in report["planned_transitory_horizons"]]
    h2_values = [int(v) for v in report["planned_operating_horizons"]]
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    for axis, (key, title, percent, cmap) in zip(axes.ravel(), METRICS, strict=True):
        values, states = matrix(cases, h1_values, h2_values, key)
        image = axis.imshow(values, aspect="auto", cmap=cmap,
                            vmin=0.0 if percent else None)
        axis.set_xticks(range(len(h2_values)), h2_values)
        axis.set_yticks(range(len(h1_values)), h1_values)
        axis.set_xlabel(r"Operating horizon $H_2$")
        axis.set_ylabel(r"Transitory horizon $H_1$")
        axis.set_title(title)
        decorate(axis, values, states, percent)
        figure.colorbar(image, ax=axis, shrink=0.78)

    best = report.get("best_feasible")
    subtitle = "No feasible incumbent found"
    if best:
        subtitle = (
            f"Best observed: H1={best['H1']}, H2={best['H2']}, "
            f"projected cost={best['projected_evaluation_cost']:.4g}, "
            f"gap={best['mip_gap']:.1%}"
        )
    figure.suptitle(f"Initial-damage H1/H2 grid\n{subtitle}", fontsize=14)
    figure.legend(handles=outcome_legend(), loc="lower center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0.06, 1, 0.92))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--kind", choices=("h1", "grid"), required=True)
    args = parser.parse_args()
    report = yaml.safe_load(args.report.read_text(encoding="utf-8"))
    if args.kind == "h1":
        plot_h1(report, args.output)
    else:
        plot_grid(report, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
