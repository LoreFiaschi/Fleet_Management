"""The schedule plot and the sweep plot (spec, "Plotting API and Layout").

``plot_management`` draws a result file of ``solve`` as a grid of cells and days with a
summary panel. ``plot_horizon_sweep`` draws one or more reports of
``sweep_operating_horizons`` over H2. Result files and reports are read with
``config.read_input``, the reader of every input path (spec, "Format Rules").
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from fleet_management.config import read_input

SUPPORTED_INPUT_EXTENSIONS = {".yaml", ".yml", ".json", ".h5", ".hdf5"}
SUPPORTED_PLOT_EXTENSIONS = {".png", ".pdf"}
ACTION_COLOURS = {
    "mission": "#173f5f", "idle": "#6b7280", "repair": "#f59e0b",
    "replacement": "#8b5cf6", "depot": "#111827",
}
REQUIRED_RESULT_KEYS = ("F", "M", "L", "H1", "H2", "tau", "mu_0", "mu", "x")


def plot_management(input_file_path: str, plot_file_path: str | None = None) -> None:
    """Plot a result file of ``solve`` (spec, "Schedule Plot").

    One row per cell (train, component), one column per day after the column
    "initial"; the colour is the tracked mean over the failure threshold, the mark of
    a day is P, R, M_j, D or I. A side panel summarises the run.
    """
    input_file = Path(input_file_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file_path}")
    if input_file.suffix.lower() not in SUPPORTED_INPUT_EXTENSIONS:
        raise ValueError(
            f"Unsupported input file type {input_file.suffix!r}; expected one of "
            f"{sorted(SUPPORTED_INPUT_EXTENSIONS)}."
        )

    plot_path = _resolve_plot_path(plot_file_path)
    if plot_path.suffix.lower() not in SUPPORTED_PLOT_EXTENSIONS:
        raise ValueError(
            f"Unsupported plot file type {plot_path.suffix!r}; expected one of "
            f"{sorted(SUPPORTED_PLOT_EXTENSIONS)}."
        )
    if not plot_path.parent.exists():
        raise FileNotFoundError(f"Plot directory does not exist: {plot_path.parent}")
    if not os.access(plot_path.parent, os.W_OK):
        raise PermissionError(f"Plot directory is not writable: {plot_path.parent}")

    view = _normalise_solution(read_input(input_file), _read_presolve_log(input_file))
    figure = _draw_solution(view)
    figure.savefig(plot_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def plot_horizon_sweep(
    sweep_report_path: (
        str | os.PathLike[str] | Sequence[str | os.PathLike[str]]
    ),
    plot_file_path: str | None = None,
) -> None:
    """Visualize one or more compatible operating-horizon sweep reports.

    The objective panel distinguishes proven-optimal cases from feasible
    cases stopped by a solver limit. The remaining panels report the MIP
    gap and deterministic formulation growth.
    """
    written_paths = (
        [sweep_report_path]
        if isinstance(sweep_report_path, (str, os.PathLike))
        else list(sweep_report_path)
    )
    if not written_paths:
        raise ValueError("At least one horizon-sweep report is required.")
    report_paths = [Path(path) for path in written_paths]
    for report_path in report_paths:
        if not report_path.exists():
            raise FileNotFoundError(f"Sweep report not found: {report_path}")
        if report_path.suffix.lower() not in {".yaml", ".yml", ".json"}:
            raise ValueError("Horizon-sweep reports must be YAML or JSON files.")

    plot_path = _resolve_plot_path(
        "horizon_sweep.png" if plot_file_path is None else plot_file_path
    )
    if plot_path.suffix.lower() not in SUPPORTED_PLOT_EXTENSIONS:
        raise ValueError(
            f"Unsupported plot file type {plot_path.suffix!r}; expected one of "
            f"{sorted(SUPPORTED_PLOT_EXTENSIONS)}."
        )
    if not plot_path.parent.exists():
        raise FileNotFoundError(f"Plot directory does not exist: {plot_path.parent}")
    if not os.access(plot_path.parent, os.W_OK):
        raise PermissionError(f"Plot directory is not writable: {plot_path.parent}")

    reports = [read_input(report_path) for report_path in report_paths]
    merged_report = _merge_horizon_sweep_reports(reports)
    figure = _draw_horizon_sweep(_normalise_horizon_sweep(merged_report))
    figure.savefig(plot_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _merge_horizon_sweep_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine compatible sweep reports and retain one case per H2 value."""
    if len(reports) == 1:
        return reports[0]

    fixed = reports[0].get("fixed_dimensions") or {}
    H1 = fixed.get("H1", reports[0].get("H1"))
    selected: dict[int, dict[str, Any]] = {}
    for report_index, report in enumerate(reports):
        candidate_fixed = report.get("fixed_dimensions") or {}
        candidate_H1 = candidate_fixed.get("H1", report.get("H1"))
        for name in ("F", "M", "L"):
            if candidate_fixed.get(name) != fixed.get(name):
                raise ValueError(
                    f"Horizon-sweep report {report_index} changes fixed {name}."
                )
        if candidate_H1 != H1:
            raise ValueError(
                f"Horizon-sweep report {report_index} changes fixed H1."
            )
        cases = report.get("cases")
        if not isinstance(cases, list):
            raise ValueError(
                f"Horizon-sweep report {report_index} has no cases list."
            )
        for case in cases:
            H2 = int(case["H2"])
            previous = selected.get(H2)
            if previous is None or _prefer_horizon_case(case, previous):
                selected[H2] = case

    return {
        "objective": reports[0].get("objective"),
        "fixed_dimensions": fixed,
        "H1": H1,
        "stopping_rule": reports[0].get("stopping_rule") or {},
        "cases": [selected[H2] for H2 in sorted(selected)],
    }


def _prefer_horizon_case(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
    """Prefer a feasible solution, then its cost, then the tighter bound."""
    candidate_cost = _optional_float(candidate.get("J_op_average"))
    current_cost = _optional_float(current.get("J_op_average"))
    if (candidate_cost is not None) != (current_cost is not None):
        return candidate_cost is not None
    if candidate_cost is not None and current_cost is not None:
        if not np.isclose(candidate_cost, current_cost):
            return candidate_cost < current_cost
    candidate_bound = _optional_float(candidate.get("objective_bound"))
    current_bound = _optional_float(current.get("objective_bound"))
    if candidate_bound is None:
        return False
    return current_bound is None or candidate_bound > current_bound


def _normalise_horizon_sweep(report: dict[str, Any]) -> dict[str, Any]:
    cases = report.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Horizon-sweep report must contain a nonempty 'cases' list.")

    stopping_rule = report.get("stopping_rule") or {}
    maximum_stopping_gap = _optional_float(
        stopping_rule.get("maximum_mip_gap_for_stopping")
    )

    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or case.get("H2") is None:
            raise ValueError(f"Horizon-sweep case {index} has no H2 value.")
        formulation = case.get("formulation") or {}
        timing = case.get("timing") or {}
        status = str(case.get("status", "unknown"))
        cost = _optional_float(case.get("J_op_average"))
        mip_gap = _optional_float(case.get("mip_gap"))
        has_feasible = (
            cost is not None
            and status not in {"infeasible", "inf_or_unbounded", "unbounded"}
        )
        rows.append({
            "H2": int(case["H2"]),
            "T": int(case.get("T", int(report.get("H1", 0)) + int(case["H2"]))),
            "status": status,
            "cost": cost,
            "objective_bound": _optional_float(case.get("objective_bound")),
            "mip_gap": mip_gap,
            "requested_mip_gap": _optional_float(case.get("requested_mip_gap")),
            "runtime": _optional_float(
                timing.get("optimizer_call_seconds", case.get("optimizer_seconds"))
            ),
            "variables": _optional_float(formulation.get("variables")),
            "continuous_variables": _optional_float(
                formulation.get("continuous_variables")
            ),
            "integer_variables": _optional_float(
                formulation.get("integer_variables")
            ),
            "linear_constraints": _optional_float(
                formulation.get("linear_constraints")
            ),
            "has_feasible": has_feasible,
            "is_proven": has_feasible and status == "optimal",
        })
    rows.sort(key=lambda row: row["H2"])

    fixed = report.get("fixed_dimensions") or {}
    feasible_rows = [row for row in rows if row["has_feasible"]]
    proven_rows = [row for row in feasible_rows if row["is_proven"]]
    best_proven_row = (
        min(proven_rows, key=lambda row: row["cost"])
        if proven_rows else None
    )
    best_feasible_row = (
        min(feasible_rows, key=lambda row: row["cost"])
        if feasible_rows else None
    )
    return {
        "rows": rows,
        "F": fixed.get("F"),
        "M": fixed.get("M"),
        "L": fixed.get("L"),
        "H1": fixed.get("H1", report.get("H1")),
        "maximum_mip_gap_for_stopping": maximum_stopping_gap,
        "best_proven_H2": (
            None if best_proven_row is None else best_proven_row["H2"]
        ),
        "best_feasible_H2": (
            None if best_feasible_row is None else best_feasible_row["H2"]
        ),
        "best_feasible_status": (
            None if best_feasible_row is None else best_feasible_row["status"]
        ),
    }


def _draw_horizon_sweep(view: dict[str, Any]):
    """Return the figure of the sweep plot: panels (a), (b), (c) of the spec."""
    rows = view["rows"]
    h2 = np.asarray([row["H2"] for row in rows], dtype=float)
    costs = np.asarray([
        np.nan if row["cost"] is None else row["cost"] for row in rows
    ])
    gaps = np.asarray([
        np.nan if row["mip_gap"] is None else row["mip_gap"] for row in rows
    ])
    continuous = np.asarray([
        np.nan if row["continuous_variables"] is None
        else row["continuous_variables"] for row in rows
    ])
    integers = np.asarray([
        np.nan if row["integer_variables"] is None
        else row["integer_variables"] for row in rows
    ])
    constraints = np.asarray([
        np.nan if row["linear_constraints"] is None
        else row["linear_constraints"] for row in rows
    ])

    figure = plt.figure(figsize=(14.8, 7.2))
    grid = figure.add_gridspec(2, 2, width_ratios=(1.12, 1.0))
    cost_ax = figure.add_subplot(grid[:, 0])
    gap_ax = figure.add_subplot(grid[0, 1])
    formulation_ax = figure.add_subplot(grid[1, 1])
    feasible = np.asarray([row["has_feasible"] for row in rows], dtype=bool)
    optimal = np.asarray([row["is_proven"] for row in rows], dtype=bool)
    limited = feasible & ~optimal

    if np.count_nonzero(feasible) > 1:
        # Plot the complete array so NaN entries break the line wherever a
        # time-limited case produced no feasible solution.
        cost_ax.plot(h2, costs, color="#9ca3af", linewidth=1.5, zorder=1)
    cost_ax.scatter(h2[optimal], costs[optimal], s=72, marker="o",
                    facecolor="#2563eb", edgecolor="white", linewidth=0.9,
                    label="proven (optimal)", zorder=3)
    cost_ax.scatter(h2[limited], costs[limited], s=82, marker="D",
                    facecolor="white", edgecolor="#d97706", linewidth=1.8,
                    label="feasible, not proven", zorder=3)
    no_feasible = ~feasible
    if np.any(no_feasible):
        cost_ax.scatter(
            h2[no_feasible], np.full(np.count_nonzero(no_feasible), 0.02),
            transform=cost_ax.get_xaxis_transform(), clip_on=False,
            s=58, marker="x", color="#dc2626", linewidth=1.6,
            label="no feasible solution", zorder=4,
        )

    for row in rows:
        if row["cost"] is None:
            continue
        if row["is_proven"] or row["H2"] == view["best_feasible_H2"]:
            cost_ax.annotate(
                f"{row['cost']:.3g}", (row["H2"], row["cost"]),
                xytext=(0, 9), textcoords="offset points", ha="center",
                fontsize=8, color="#374151",
            )

    feasible_h2 = view["best_feasible_H2"]
    if feasible_h2 is not None:
        row = _row_at_h2(rows, feasible_h2)
        if row is not None and row["cost"] is not None:
            cost_ax.scatter([feasible_h2], [row["cost"]], s=175, marker="P",
                            facecolor="#f59e0b", edgecolor="#92400e",
                            linewidth=0.8, label="best feasible", zorder=5)

    cost_ax.set_title("Operating objective", loc="left", fontweight="bold")
    cost_ax.set_xlabel("Operating horizon $H_2$")
    cost_ax.set_ylabel("$J_{op} / H_2$  (lower is better)")
    cost_ax.set_xticks(_readable_horizon_ticks(h2))
    cost_ax.grid(axis="both", color="#e5e7eb", linewidth=0.8)
    cost_ax.legend(frameon=False, loc="best", fontsize=8)

    # Reference of panel (b): the stopping gap if it is a number, otherwise the
    # requested MIP gap of each case.
    if view["maximum_mip_gap_for_stopping"] is not None:
        stopping_percent = 100.0 * view["maximum_mip_gap_for_stopping"]
        gap_ax.axhline(
            stopping_percent, color="#15803d", linestyle="--", linewidth=1.3,
            label=f"{stopping_percent:g}% stopping gap", zorder=2,
        )
    else:
        requested = [
            (row["H2"], 100.0 * row["requested_mip_gap"]) for row in rows
            if row["requested_mip_gap"] is not None
        ]
        if requested:
            gap_ax.scatter(
                [h for h, _ in requested], [g for _, g in requested],
                s=160, marker="_", color="#15803d", linewidth=1.6,
                label="requested MIP gap", zorder=2,
            )
    finite_gaps = feasible & np.isfinite(gaps)
    if np.any(finite_gaps):
        gap_percent = 100.0 * gaps
        gap_ax.plot(h2, gap_percent, color="#9ca3af", linewidth=1.2, zorder=1)
        gap_ax.scatter(
            h2[optimal], gap_percent[optimal], s=45, marker="o",
            facecolor="#2563eb", edgecolor="white", linewidth=0.7,
            label="proven (optimal)", zorder=3,
        )
        gap_ax.scatter(
            h2[limited], gap_percent[limited], s=48, marker="D",
            facecolor="white", edgecolor="#d97706", linewidth=1.5,
            label="feasible, not proven", zorder=3,
        )
        for row in rows:
            if row["mip_gap"] is None:
                continue
            gap_ax.annotate(
                f"{100.0 * row['mip_gap']:.1f}%",
                (row["H2"], 100.0 * row["mip_gap"]),
                xytext=(0, 5), textcoords="offset points", ha="center",
                fontsize=6.5, color="#4b5563", rotation=35,
            )
    if np.any(no_feasible):
        gap_ax.scatter(
            h2[no_feasible], np.full(np.count_nonzero(no_feasible), 0.96),
            transform=gap_ax.get_xaxis_transform(), clip_on=False,
            s=38, marker="x", color="#dc2626", linewidth=1.4,
            label="no feasible solution", zorder=4,
        )
    gap_ax.set_title("MIP gap", loc="left", fontweight="bold")
    gap_ax.set_ylabel("Relative MIP gap [%]")
    gap_ax.set_xticks(_readable_horizon_ticks(h2))
    gap_ax.grid(axis="both", color="#e5e7eb", linewidth=0.8)
    gap_ax.legend(frameon=False, loc="upper left", fontsize=7.5)

    if np.any(np.isfinite(continuous)):
        formulation_ax.plot(h2, continuous, marker="s", color="#2563eb",
                            linewidth=1.8, label="continuous variables")
    if np.any(np.isfinite(integers)):
        formulation_ax.plot(h2, integers, marker="D", color="#0891b2",
                            linewidth=1.8, label="integer variables")
    if np.any(np.isfinite(constraints)):
        formulation_ax.plot(h2, constraints, marker="^", color="#7c3aed",
                            linewidth=1.8, label="linear constraints")
    formulation_ax.set_xlabel("Operating horizon $H_2$")
    formulation_ax.set_ylabel("Formulation count")
    formulation_ax.set_xticks(_readable_horizon_ticks(h2))
    formulation_ax.grid(axis="both", color="#e5e7eb", linewidth=0.8)
    formulation_ax.set_title("Formulation growth", loc="left", fontweight="bold")
    formulation_ax.legend(frameon=False, loc="upper left", fontsize=7.5)

    dimensions = ", ".join(
        f"{name}={view[name]}" for name in ("F", "M", "L", "H1")
        if view[name] is not None
    )
    figure.suptitle(f"Operating-horizon sweep ({dimensions})", x=0.02, y=0.975,
                    ha="left", fontsize=16, fontweight="bold")
    figure.text(
        0.5, 0.035,
        "Lines connect sampled horizons only; they are not extrapolations. "
        "The MIP gap quantifies uncertainty relative to the solver's bound.",
        ha="center", fontsize=8.5, color="#4b5563",
    )
    figure.subplots_adjust(
        left=0.075, right=0.925, top=0.84, bottom=0.15,
        wspace=0.32, hspace=0.42
    )
    return figure


def _readable_horizon_ticks(horizons: np.ndarray) -> np.ndarray:
    if horizons.size <= 10:
        return horizons
    ticks = horizons[::2]
    if ticks[-1] != horizons[-1]:
        ticks = np.append(ticks, horizons[-1])
    return ticks


def _row_at_h2(rows: list[dict[str, Any]], h2: int) -> dict[str, Any] | None:
    return next((row for row in rows if row["H2"] == h2), None)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _normalise_solution(data: dict[str, Any], presolved: dict[str, int]) -> dict[str, Any]:
    """Check a result mapping against the spec's shapes and build the view of the plot.

    ``presolved`` holds the counts parsed from ``runner.out``, or is empty.
    """
    missing = [key for key in REQUIRED_RESULT_KEYS if data.get(key) is None]
    if missing:
        raise ValueError(
            "The result has no plottable solution; missing keys: " + ", ".join(missing)
        )

    F, M, L = int(data["F"]), int(data["M"]), int(data["L"])
    H1, H2 = int(data["H1"]), int(data["H2"])
    tau = np.asarray(data["tau"], dtype=float)
    mu_0 = np.asarray(data["mu_0"], dtype=float)
    mu = np.asarray(data["mu"], dtype=float)
    x = np.asarray(data["x"], dtype=float)
    for name, value in (("tau", tau), ("mu_0", mu_0)):
        if value.shape != (F, L):
            raise ValueError(f"{name} has shape {value.shape}; expected {(F, L)}")
    if mu.ndim != 3 or mu.shape[:2] != (F, L):
        raise ValueError(f"mu has shape {mu.shape}; expected ({F}, {L}, T)")
    T = int(mu.shape[2])
    if data.get("T") is not None and int(data["T"]) != T:
        raise ValueError(f"mu has {T} days; the key T is {data['T']}")
    if H1 + H2 != T:
        raise ValueError(f"H1 + H2 must equal T; got {H1} + {H2} != {T}")
    if x.shape != (F, M + 1, T):
        raise ValueError(f"x has shape {x.shape}; expected {(F, M + 1, T)}")
    if np.any(tau <= 0.0):
        raise ValueError("All degradation thresholds must be positive.")

    m = _optional_array(data, "m", (F, L, T))
    r = _optional_array(data, "r", (F, L, T))

    # Counts of the summary panel: the presolved counts of runner.out if that log
    # exists, otherwise the counts of the original model in ``performance``.
    if presolved:
        counts, counts_source = presolved, "runner.out (presolved)"
    else:
        counts, counts_source = data.get("performance") or {}, "performance"

    grid = np.empty((F, L, T + 1), dtype=float)
    grid[:, :, 0], grid[:, :, 1:] = mu_0, mu
    return {
        "raw": data, "F": F, "M": M, "L": L, "T": T, "H1": H1, "H2": H2,
        "x": x, "m": m, "r": r, "grid": grid, "tau": tau,
        "models": _model_grid(data, F, L),
        "component_names": _component_names(data, L),
        "variables": counts.get("variables"),
        "continuous_variables": counts.get("continuous_variables"),
        "integer_variables": counts.get("integer_variables"),
        "linear_rows": counts.get("linear_constraints"),
        "counts_source": counts_source,
        # Count component actions, not train visits: two components repaired on
        # the same depot day are two repairs.
        "repairs": int(np.count_nonzero(m > 0.5)),
        "replacements": int(np.count_nonzero(r > 0.5)),
    }


def _draw_solution(view: dict[str, Any]):
    """Return the figure of the schedule plot: the grid and the summary panel."""
    F, M, L, T = (view[key] for key in ("F", "M", "L", "T"))
    H1, H2, n_cols = view["H1"], view["H2"], T + 1
    width = max(11.5, min(24.0, 0.64 * n_cols + 6.2))
    row_count = F * L
    height = max(4.8, 0.58 * row_count + 2.1)
    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=(max(n_cols, 7), 5.6))
    ax, stats_ax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "damage_fraction", ["#2ca25f", "#fee08b", "#d73027"]
    )
    norm = mcolors.Normalize(0.0, 1.0, clip=True)
    for i in range(F):
        for l in range(L):
            row = i * L + l
            for column in range(n_cols):
                fraction = view["grid"][i, l, column] / view["tau"][i, l]
                ax.add_patch(mpatches.Rectangle(
                    (column - 0.5, row - 0.5), 1.0, 1.0,
                    facecolor=cmap(norm(fraction)), edgecolor="none",
                ))

    for i in range(F):
        for l in range(L):
            row = i * L + l
            for k in range(T):
                label, category = _component_action_label(view, i, l, k, M)
                ax.text(k + 1, row, label, ha="center", va="center", fontsize=8.5,
                        fontweight="bold", fontfamily="monospace",
                        color=ACTION_COLOURS[category],
                        bbox={"boxstyle": "round,pad=0.16", "facecolor": "white",
                              "edgecolor": ACTION_COLOURS[category], "alpha": 0.88,
                              "linewidth": 0.8})

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(row_count - 0.5, -0.5)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(["initial", *range(1, T + 1)], rotation=45 if T > 12 else 0,
                       ha="right" if T > 12 else "center")
    ax.set_yticks(range(row_count))
    ax.set_yticklabels([
        _component_row_label(view, i, l)
        for i in range(F) for l in range(L)
    ])
    ax.set_xlabel("Day")
    ax.set_ylabel("Train and component")
    ax.set_title("Schedule and tracked mean degradation", loc="left", pad=42,
                 fontweight="bold")

    for row in range(row_count + 1):
        linewidth = 0.9 if row % L == 0 else 0.4
        colour = "#111827" if row % L == 0 else "#9ca3af"
        ax.axhline(row - 0.5, color=colour, linewidth=linewidth)
    for column in range(n_cols + 1):
        ax.axvline(column - 0.5, color="#4b5563", linewidth=0.35)
    if 0 < H1 < T:
        ax.axvline(H1 + 0.5, color="#2563eb", linewidth=2.0, linestyle="--")
        ax.text(
            (H1 + 1) / 2, 1.015, "transitory",
            transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            color="#2563eb", fontsize=8,
        )
        ax.text(
            H1 + 0.5 + H2 / 2, 1.015, "operating",
            transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            color="#2563eb", fontsize=8,
        )

    scalar = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar.set_array([])
    fig.colorbar(scalar, ax=ax, label="tracked mean / failure threshold",
                 fraction=0.032, pad=0.02)
    handles = [mpatches.Patch(facecolor="white", edgecolor=colour, label=label)
               for label, colour in (("P replacement", ACTION_COLOURS["replacement"]),
                                     ("R repair", ACTION_COLOURS["repair"]),
                                     ("M$_j$ mission $j$", ACTION_COLOURS["mission"]),
                                     ("D depot, no intervention", ACTION_COLOURS["depot"]),
                                     ("I idle train", ACTION_COLOURS["idle"]))]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17),
              ncol=5, frameon=False, fontsize=8)
    _draw_statistics(stats_ax, view)
    fig.suptitle("Fleet-management result overview", x=0.02, ha="left",
                 fontsize=15, fontweight="bold")
    return fig


def _component_action_label(
    view: dict[str, Any], i: int, l: int, k: int, M: int
) -> tuple[str, str]:
    """Return the mark of cell (i, l) on day index k: the first rule that applies."""
    if view["r"][i, l, k] > 0.5:
        return "P", "replacement"
    if view["m"][i, l, k] > 0.5:
        return "R", "repair"
    assigned = np.flatnonzero(view["x"][i, 1:M + 1, k] > 0.5)
    if assigned.size:
        return f"M{int(assigned[0]) + 1}", "mission"
    if view["x"][i, 0, k] > 0.5:
        return "D", "depot"
    return "I", "idle"


def _draw_statistics(ax, view: dict[str, Any]) -> None:
    data, performance = view["raw"], view["raw"].get("performance") or {}
    status = data.get("status")
    if status is None:
        status_text, status_colour = "unknown", "#374151"
    else:
        status_text = str(status).replace("_", " ")
        status_colour = "#15803d" if status == "optimal" else "#b45309"
    lines = [
        ("Status", status_text, status_colour),
        ("Objective", _number(data.get("objective")), "#111827"),
        ("Dimensions", f"F={view['F']}, M={view['M']}, L={view['L']}", "#111827"),
        ("Horizon", f"H1={view['H1']}, H2={view['H2']}, T={view['T']}", "#111827"),
        ("Models", ", ".join(_unique_values(view["models"].tolist())), "#111827"),
        ("Repair laws", ", ".join(_unique_values(data.get("repair_model"))) or "-",
         "#111827"),
    ]
    if data.get("objective_mode") is not None:
        lines += [("Objective mode", _compact(data.get("objective_mode")), "#111827"),
                  ("J_op / H2", _number(data.get("J_op_average")), "#111827")]
    if data.get("mip_gap") is not None:
        lines.append(("MIP gap", f"{100.0 * float(data['mip_gap']):.2f}%", "#111827"))
    lines.append(("Counts from", view["counts_source"], "#111827"))
    for label, key in (
        ("Variables", "variables"),
        ("Continuous variables", "continuous_variables"),
        ("Integer variables", "integer_variables"),
        ("Linear rows", "linear_rows"),
    ):
        value = view.get(key)
        if value is not None:
            lines.append((label, f"{int(value):,}", "#111827"))
    if performance.get("optimizer_call_seconds") is not None:
        lines.append(("Optimizer", _seconds(float(performance["optimizer_call_seconds"])), "#111827"))
    if performance.get("branch_and_bound_nodes") is not None:
        lines.append(("B&B nodes", f"{float(performance['branch_and_bound_nodes']):,.0f}", "#111827"))

    lines += [
        ("Repairs", str(view["repairs"]), "#111827"),
        ("Replacements", str(view["replacements"]), "#111827"),
    ]

    ax.set_axis_off()
    ax.set_title("Run summary", loc="left", fontweight="bold", pad=12)
    y = 0.96
    for label, value, colour in lines:
        ax.text(0.0, y, label, transform=ax.transAxes, fontsize=8.5,
                color="#6b7280", va="top")
        ax.text(0.70, y, value, transform=ax.transAxes, fontsize=8.5,
                color=colour, va="top", wrap=True,
                fontweight="bold" if label == "Status" else "normal")
        y -= 0.058


def _model_grid(data: dict[str, Any], F: int, L: int) -> np.ndarray:
    """The model name of each cell, from the key ``model``; "unknown" if it is missing."""
    value = data.get("model")
    if value is None:
        return np.full((F, L), "unknown", dtype=object)
    arr = np.asarray(value, dtype=object)
    if arr.shape != (F, L):
        raise ValueError(f"model has shape {arr.shape}; expected {(F, L)}")
    return arr


def _component_names(data: dict[str, Any], L: int) -> list[str]:
    names = data.get("component_names")
    if names is None:
        return [f"Component {l + 1}" for l in range(L)]
    if isinstance(names, str):
        names = [names]
    if len(names) != L:
        raise ValueError(f"component_names has length {len(names)}; expected {L}")
    return [str(name) for name in names]


def _component_row_label(view: dict[str, Any], i: int, l: int) -> str:
    component = view["component_names"][l]
    return f"Train {i + 1} — {component}\n{view['models'][i, l]}"


def _unique_values(value: Any) -> list[str]:
    """The distinct strings of a string or a nested list, in order of first appearance."""
    if value is None:
        return []
    if not isinstance(value, list):
        return [str(value)]
    seen: list[str] = []
    for item in value:
        for text in _unique_values(item):
            if text not in seen:
                seen.append(text)
    return seen


def _optional_array(data: dict[str, Any], key: str, shape: tuple[int, ...]) -> np.ndarray:
    if data.get(key) is None:
        return np.zeros(shape, dtype=float)
    arr = np.asarray(data[key], dtype=float)
    if arr.shape != shape:
        raise ValueError(f"{key} has shape {arr.shape}; expected {shape}")
    return arr


def _resolve_plot_path(plot_file_path: str | None) -> Path:
    path = Path("output.png" if plot_file_path is None else plot_file_path)
    return path.with_suffix(".png") if path.suffix == "" else path


def _read_presolve_log(input_file: Path) -> dict[str, int]:
    """Presolved counts from the Gurobi log ``runner.out`` beside the result file.

    Empty if there is no such file or it has no presolve lines (spec, "Summary panel").
    """
    log_path = input_file.parent / "runner.out"
    if not log_path.is_file():
        return {}
    return _parse_gurobi_presolve(
        log_path.read_text(encoding="utf-8", errors="replace")
    )


def _parse_gurobi_presolve(log_text: str) -> dict[str, int]:
    """Extract the presolved model size from a Gurobi text log.

    The main solve is the last one in the log (the warm-start completion and the
    relaxed copy run before it), so the last "Presolved:" line is used.
    """
    model_matches = list(re.finditer(
        r"^Presolved:\s*([\d,]+)\s+rows,\s*([\d,]+)\s+columns(?:,.*)?$",
        log_text,
        flags=re.MULTILINE,
    ))
    if not model_matches:
        return {}
    model_match = model_matches[-1]

    following_text = log_text[model_match.end() :]
    types_match = re.search(
        r"^Variable types:\s*([\d,]+)\s+continuous,\s*"
        r"([\d,]+)\s+integer(?:\s*\(([\d,]+)\s+binary\))?",
        following_text,
        flags=re.MULTILINE,
    )
    if types_match is None:
        return {}

    def count(value: str) -> int:
        return int(value.replace(",", ""))

    result = {
        "linear_rows": count(model_match.group(1)),
        "linear_constraints": count(model_match.group(1)),
        "variables": count(model_match.group(2)),
        "continuous_variables": count(types_match.group(1)),
        "integer_variables": count(types_match.group(2)),
    }
    if types_match.group(3) is not None:
        result["binary_variables"] = count(types_match.group(3))
    return result


def _number(value: Any) -> str:
    return "-" if value is None else f"{float(value):.6g}"


def _compact(value: Any) -> str:
    if value is None:
        return "-"
    return ", ".join(str(item) for item in value) if isinstance(value, list) else str(value).replace("_", " ")


def _seconds(value: float) -> str:
    if value < 1.0:
        return f"{1000.0 * value:.1f} ms"
    if value < 60.0:
        return f"{value:.2f} s"
    return f"{value / 60.0:.1f} min"
