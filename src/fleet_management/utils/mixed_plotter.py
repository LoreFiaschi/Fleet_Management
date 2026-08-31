from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import yaml

SUPPORTED_INPUT_EXTENSIONS = {".yaml", ".yml", ".json", ".h5", ".hdf5"}
SUPPORTED_PLOT_EXTENSIONS = {".png", ".pdf"}
ACTION_COLOURS = {
    "mission": "#173f5f", "idle": "#6b7280", "repair": "#f59e0b",
    "replacement": "#8b5cf6", "mixed_intervention": "#dc2626",
    "depot": "#111827",
}


def plot_mixed_management(input_file_path: str, plot_file_path: str | None = None) -> None:
    """Plot a saved solver result as a schedule/degradation overview.

    Each vehicle/time cell is split into one strip per component, coloured by
    physical expected damage divided by the failure threshold. Mission, idle,
    repair and replacement decisions are annotated separately. A side panel
    records the dimensions, degradation models, solve outcome and useful
    formulation/performance statistics.
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

    _draw_solution(_normalise_solution(_read_input(input_file)), plot_path)


def plot_horizon_sweep(
    sweep_report_path: str,
    plot_file_path: str | None = None,
) -> None:
    """Visualize an operating-horizon sweep report.

    The left panel compares ``J_op / H2`` and distinguishes proven optimal
    cases from feasible stopped by a solver limit. The right panel
    separates optimizer time from deterministic formulation growth.
    """
    report_path = Path(sweep_report_path)
    if not report_path.exists():
        raise FileNotFoundError(f"Sweep report not found: {sweep_report_path}")
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

    _draw_horizon_sweep(_normalise_horizon_sweep(_read_input(report_path)), plot_path)


def _normalise_horizon_sweep(report: dict[str, Any]) -> dict[str, Any]:
    cases = report.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Horizon-sweep report must contain a nonempty 'cases' list.")

    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or case.get("H2") is None:
            raise ValueError(f"Horizon-sweep case {index} has no H2 value.")
        formulation = case.get("formulation") or {}
        timing = case.get("timing") or {}
        status = str(case.get("status", "unknown"))
        cost = _optional_float(case.get("J_op_average"))
        rows.append({
            "H2": int(case["H2"]),
            "T": int(case.get("T", int(report.get("H1", 0)) + int(case["H2"]))),
            "status": status,
            "cost": cost,
            "J_trans": _optional_float(case.get("J_trans")),
            "runtime": _optional_float(
                timing.get("optimizer_call_seconds", case.get("optimizer_seconds"))
            ),
            "variables": _optional_float(formulation.get("variables")),
            "linear_constraints": _optional_float(
                formulation.get("linear_constraints")
            ),
            "has_feasible": (
                cost is not None
                and status not in {"infeasible", "inf_or_unbounded", "unbounded"}
            ),
        })
    rows.sort(key=lambda row: row["H2"])

    fixed = report.get("fixed_dimensions") or {}
    budget = _optional_float(report.get("transitory_budget"))
    feasible_rows = [
        row for row in rows
        if row["has_feasible"]
        and (
            budget is None
            or row["J_trans"] is None
            or row["J_trans"] <= budget + 1e-8
        )
    ]
    proven_rows = [row for row in feasible_rows if row["status"] == "optimal"]
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
        "budget": budget,
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


def _draw_horizon_sweep(view: dict[str, Any], plot_path: Path) -> None:
    rows = view["rows"]
    h2 = np.asarray([row["H2"] for row in rows], dtype=float)
    costs = np.asarray([
        np.nan if row["cost"] is None else row["cost"] for row in rows
    ])
    runtimes = np.asarray([
        np.nan if row["runtime"] is None else row["runtime"] for row in rows
    ])
    variables = np.asarray([
        np.nan if row["variables"] is None else row["variables"] for row in rows
    ])
    constraints = np.asarray([
        np.nan if row["linear_constraints"] is None
        else row["linear_constraints"] for row in rows
    ])

    figure, (cost_ax, scale_ax) = plt.subplots(1, 2, figsize=(13.2, 6.2))
    feasible = np.asarray([row["has_feasible"] for row in rows], dtype=bool)
    optimal = np.asarray([row["status"] == "optimal" for row in rows], dtype=bool)
    limited = feasible & ~optimal

    if np.count_nonzero(feasible) > 1:
        cost_ax.plot(h2[feasible], costs[feasible], color="#9ca3af",
                     linewidth=1.5, zorder=1)
    cost_ax.scatter(h2[optimal], costs[optimal], s=72, marker="o",
                    facecolor="#2563eb", edgecolor="white", linewidth=0.9,
                    label="proven optimal", zorder=3)
    cost_ax.scatter(h2[limited], costs[limited], s=82, marker="D",
                    facecolor="white", edgecolor="#d97706", linewidth=1.8,
                    label="feasible", zorder=3)

    for row in rows:
        if row["cost"] is None:
            continue
        cost_ax.annotate(
            f"{row['cost']:.3g}", (row["H2"], row["cost"]),
            xytext=(0, 9), textcoords="offset points", ha="center",
            fontsize=8, color="#374151",
        )

    proven_h2 = view["best_proven_H2"]
    feasible_h2 = view["best_feasible_H2"]
    if proven_h2 is not None:
        row = _row_at_h2(rows, proven_h2)
        if row is not None and row["cost"] is not None:
            cost_ax.scatter([proven_h2], [row["cost"]], s=230, marker="*",
                            facecolor="#16a34a", edgecolor="#14532d",
                            linewidth=0.8, label="best proven", zorder=5)
    if feasible_h2 is not None and feasible_h2 != proven_h2:
        row = _row_at_h2(rows, feasible_h2)
        if row is not None and row["cost"] is not None:
            cost_ax.scatter([feasible_h2], [row["cost"]], s=175, marker="P",
                            facecolor="#f59e0b", edgecolor="#92400e",
                            linewidth=0.8, label="best feasible", zorder=5)

    cost_ax.set_title("Operating objective", loc="left", fontweight="bold")
    cost_ax.set_xlabel("Operating horizon $H_2$")
    cost_ax.set_ylabel("$J_{op} / H_2$  (lower is better)")
    cost_ax.set_xticks(h2)
    cost_ax.grid(axis="both", color="#e5e7eb", linewidth=0.8)
    cost_ax.legend(frameon=False, loc="best")

    runtime_mask = np.isfinite(runtimes) & (runtimes > 0.0)
    if np.any(runtime_mask):
        scale_ax.plot(h2[runtime_mask], runtimes[runtime_mask], marker="o",
                      color="#d97706", linewidth=2.0, label="optimizer time")
        scale_ax.set_yscale("log")
    scale_ax.set_xlabel("Operating horizon $H_2$")
    scale_ax.set_ylabel("Optimizer time [s] (log scale)", color="#b45309")
    scale_ax.tick_params(axis="y", labelcolor="#b45309")
    scale_ax.set_xticks(h2)
    scale_ax.grid(axis="both", color="#e5e7eb", linewidth=0.8)

    count_ax = scale_ax.twinx()
    if np.any(np.isfinite(variables)):
        count_ax.plot(h2, variables, marker="s", color="#2563eb",
                      linewidth=1.8, label="variables")
    if np.any(np.isfinite(constraints)):
        count_ax.plot(h2, constraints, marker="^", color="#7c3aed",
                      linewidth=1.8, label="linear constraints")
    count_ax.set_ylabel("Formulation count", color="#4b5563")
    count_ax.tick_params(axis="y", labelcolor="#4b5563")
    scale_ax.set_title("Computational growth", loc="left", fontweight="bold")

    handles_left, labels_left = scale_ax.get_legend_handles_labels()
    handles_right, labels_right = count_ax.get_legend_handles_labels()
    scale_ax.legend(handles_left + handles_right, labels_left + labels_right,
                    frameon=False, loc="upper left")

    for row in rows:
        colour = "#15803d" if row["status"] == "optimal" else "#b45309"
        scale_ax.annotate(row["status"].replace("_", " "),
                          (row["H2"], 0.02), xycoords=("data", "axes fraction"),
                          rotation=90, ha="center", va="bottom", fontsize=7,
                          color=colour)

    dimensions = ", ".join(
        f"{name}={view[name]}" for name in ("F", "M", "L", "H1")
        if view[name] is not None
    )
    budget = "" if view["budget"] is None else f";  $B_{{trans}}$={view['budget']:g}"
    figure.suptitle("Operating-horizon sweep", x=0.02, y=0.975, ha="left",
                    fontsize=16, fontweight="bold")
    figure.text(0.02, 0.925, dimensions + budget, fontsize=9, color="#4b5563")
    figure.text(
        0.5, 0.035,
        "A time-limit feasible may have a lower observed cost than the best "
        "proven case; it is not an optimality certificate.",
        ha="center", fontsize=8.5, color="#4b5563",
    )
    figure.subplots_adjust(
        left=0.075, right=0.925, top=0.84, bottom=0.18, wspace=0.32
    )
    figure.savefig(plot_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _row_at_h2(rows: list[dict[str, Any]], h2: int) -> dict[str, Any] | None:
    return next((row for row in rows if row["H2"] == h2), None)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _normalise_solution(data: dict[str, Any]) -> dict[str, Any]:
    required = ("F", "M", "mu_0", "mu", "x")
    missing = [key for key in required if data.get(key) is None]
    if missing:
        raise ValueError(
            "The result does not contain a plottable feasible case; missing "
            + ", ".join(missing)
        )

    F, M, L = int(data["F"]), int(data["M"]), int(data.get("L", 1))
    mu_0 = np.asarray(data["mu_0"], dtype=float)
    mu = np.asarray(data["mu"], dtype=float)
    x = np.asarray(data["x"], dtype=float)
    if mu_0.ndim == 1:
        mu_0 = mu_0[:, None]
    if mu.ndim == 2:
        mu = mu[:, None, :]
    if mu_0.shape != (F, L):
        raise ValueError(f"mu_0 has shape {mu_0.shape}; expected {(F, L)}")
    if mu.shape[:2] != (F, L):
        raise ValueError(f"mu has shape {mu.shape}; expected ({F}, {L}, T)")

    T = int(mu.shape[-1])
    if x.shape != (F, M + 1, T):
        raise ValueError(f"x has shape {x.shape}; expected {(F, M + 1, T)}")
    threshold = data.get("alpha") if data.get("alpha") is not None else data.get("tau", 1.0)
    tau = _cell_grid(threshold, F, L, "alpha/tau")
    if np.any(tau <= 0.0):
        raise ValueError("All degradation thresholds must be positive.")

    H1 = int(data.get("H1", _legacy_h1(data.get("H"), T)))
    H2 = int(data.get("H2", T - H1))
    if H1 + H2 != T:
        raise ValueError(f"H1 + H2 must equal T; got {H1} + {H2} != {T}")

    grid = np.empty((F, L, T + 1), dtype=float)
    grid[:, :, 0], grid[:, :, 1:] = mu_0, mu
    return {
        "raw": data, "F": F, "M": M, "L": L, "T": T, "H1": H1, "H2": H2,
        "x": x, "m": _optional_array(data, "m", (F, L, T)),
        "r": _optional_array(data, "r", (F, L, T)), "grid": grid, "tau": tau,
        "models": _model_grid(data, F, L),
    }


def _draw_solution(view: dict[str, Any], plot_path: Path) -> None:
    F, M, L, T = (view[key] for key in ("F", "M", "L", "T"))
    H1, H2, n_cols = view["H1"], view["H2"], T + 1
    width = max(10.5, min(22.0, 0.64 * n_cols + 4.8))
    height = max(4.8, 0.78 * F + 2.1)
    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=(max(n_cols, 7), 4.2))
    ax, stats_ax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "damage_fraction", ["#2ca25f", "#fee08b", "#d73027"]
    )
    norm, strip_h = mcolors.Normalize(0.0, 1.0, clip=True), 1.0 / L
    for i in range(F):
        for column in range(n_cols):
            for l in range(L):
                fraction = view["grid"][i, l, column] / view["tau"][i, l]
                ax.add_patch(mpatches.Rectangle(
                    (column - 0.5, i - 0.5 + l * strip_h), 1.0, strip_h,
                    facecolor=cmap(norm(fraction)), edgecolor="none",
                ))

    for i in range(F):
        ax.text(0, i, "init", ha="center", va="center", fontsize=7,
                color="#111827", fontweight="bold")
        for k in range(T):
            label, category = _action_label(view, i, k, M)
            ax.text(k + 1, i, label, ha="center", va="center", fontsize=8.5,
                    fontweight="bold", color=ACTION_COLOURS[category],
                    bbox={"boxstyle": "round,pad=0.16", "facecolor": "white",
                          "edgecolor": ACTION_COLOURS[category], "alpha": 0.88,
                          "linewidth": 0.8})

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(F - 0.5, -0.5)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(["initial", *range(T)], rotation=45 if T > 12 else 0,
                       ha="right" if T > 12 else "center")
    ax.set_yticks(range(F))
    ax.set_yticklabels([_vehicle_label(view["models"], i) for i in range(F)])
    ax.set_xlabel("Time step")
    ax.set_ylabel("Vehicle and component model")
    ax.set_title("Schedule and physical expected damage", loc="left", pad=12,
                 fontweight="bold")

    for i in range(F + 1):
        ax.axhline(i - 0.5, color="#111827", linewidth=0.65)
    for column in range(n_cols + 1):
        ax.axvline(column - 0.5, color="#4b5563", linewidth=0.35)
    if L > 1:
        for i in range(F):
            for l in range(1, L):
                ax.axhline(i - 0.5 + l * strip_h, color="white", linewidth=0.7)

    if 0 < H1 < T:
        ax.axvline(H1 + 0.5, color="#2563eb", linewidth=2.0, linestyle="--")
        ax.text((H1 + 1) / 2, -0.68, "transitory", ha="center", va="bottom",
                color="#2563eb", fontsize=8)
        ax.text(H1 + 0.5 + H2 / 2, -0.68, "operating", ha="center", va="bottom",
                color="#2563eb", fontsize=8)

    scalar = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar.set_array([])
    fig.colorbar(scalar, ax=ax, label="physical mean / failure threshold",
                 fraction=0.032, pad=0.02)
    handles = [mpatches.Patch(facecolor="white", edgecolor=colour, label=label)
               for label, colour in (("mission Mj", ACTION_COLOURS["mission"]),
                                     ("idle", ACTION_COLOURS["idle"]),
                                     ("repair R", ACTION_COLOURS["repair"]),
                                     ("replacement P", ACTION_COLOURS["replacement"]),
                                     ("depot D", ACTION_COLOURS["depot"]))]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17),
              ncol=5, frameon=False, fontsize=8)
    _draw_statistics(stats_ax, view)
    fig.suptitle("Fleet-management result overview", x=0.02, ha="left",
                 fontsize=15, fontweight="bold")
    fig.savefig(plot_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _action_label(view: dict[str, Any], i: int, k: int, M: int) -> tuple[str, str]:
    repaired = np.flatnonzero(view["m"][i, :, k] > 0.5)
    replaced = np.flatnonzero(view["r"][i, :, k] > 0.5)
    if repaired.size and replaced.size:
        return "R/P", "mixed_intervention"
    if replaced.size:
        return _component_action("P", replaced), "replacement"
    if repaired.size:
        return _component_action("R", repaired), "repair"
    assigned = np.flatnonzero(view["x"][i, 1:M + 1, k] > 0.5)
    if assigned.size:
        return f"M{int(assigned[0]) + 1}", "mission"
    if view["x"][i, 0, k] > 0.5:
        return "D", "depot"
    return "-", "idle"


def _component_action(symbol: str, components: np.ndarray) -> str:
    return f"{symbol}{int(components[0]) + 1}" if components.size == 1 else symbol


def _draw_statistics(ax, view: dict[str, Any]) -> None:
    data, performance = view["raw"], view["raw"].get("performance") or {}
    actual = (data.get("gamma_formulation") or {}).get("actual_gurobi_model") or {}
    status = str(data.get("status", "unknown"))
    status_colour = {"optimal": "#15803d", "time_limit": "#b45309",
                     "infeasible": "#b91c1c"}.get(status, "#374151")
    lines = [
        ("Status", status.replace("_", " "), status_colour),
        ("Objective", _number(data.get("objective")), "#111827"),
        ("Dimensions", f"F={view['F']}, M={view['M']}, L={view['L']}", "#111827"),
        ("Horizon", f"H1={view['H1']}, H2={view['H2']}, T={view['T']}", "#111827"),
        ("Models", ", ".join(_unique_models(view["models"])), "#111827"),
        ("Repair", _compact(data.get("repair_model")), "#111827"),
    ]
    if data.get("objective_mode") is not None:
        lines += [("Objective mode", _compact(data.get("objective_mode")), "#111827"),
                  ("J_trans", _number(data.get("J_trans")), "#111827"),
                  ("J_op / H2", _number(data.get("J_op_average")), "#111827")]
    if data.get("mip_gap") is not None:
        lines.append(("MIP gap", f"{100.0 * float(data['mip_gap']):.2f}%", "#111827"))
    for label, key in (("Variables", "variables"), ("Linear rows", "linear_constraints"),
                       ("General rows", "general_constraints")):
        value = performance.get(key, actual.get(key))
        if value is not None:
            lines.append((label, f"{int(value):,}", "#111827"))
    if performance.get("optimizer_call_seconds") is not None:
        lines.append(("Optimizer", _seconds(float(performance["optimizer_call_seconds"])), "#111827"))
    if performance.get("branch_and_bound_nodes") is not None:
        lines.append(("B&B nodes", f"{float(performance['branch_and_bound_nodes']):,.0f}", "#111827"))

    repair_visits = int(
        np.count_nonzero(np.any(view["m"] > 0.5, axis=1))
    )
    replacement_visits = int(
        np.count_nonzero(np.any(view["r"] > 0.5, axis=1))
    )

    lines += [
        ("Repair visits", str(repair_visits), "#111827"),
        ("Replacement visits", str(replacement_visits), "#111827"),
    ]

    ax.set_axis_off()
    ax.set_title("Run summary", loc="left", fontweight="bold", pad=12)
    y = 0.96
    for label, value, colour in lines:
        ax.text(0.0, y, label, transform=ax.transAxes, fontsize=8.5,
                color="#6b7280", va="top")
        ax.text(0.52, y, value, transform=ax.transAxes, fontsize=8.5,
                color=colour, va="top", wrap=True,
                fontweight="bold" if label == "Status" else "normal")
        y -= 0.055
    ax.text(0.0, max(0.02, y - 0.02),
            "Cell colour shows the physical expected-damage state.\n"
            "Gamma reliability uses a separate bounding-shape state.",
            transform=ax.transAxes, fontsize=7.7, color="#4b5563", va="top",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f3f4f6",
                  "edgecolor": "#d1d5db"})


def _model_grid(data: dict[str, Any], F: int, L: int) -> np.ndarray:
    candidate = data.get("reliability_impl")
    if candidate is None:
        candidate = data.get("models", data.get("degradation", "unknown"))
    arr = np.asarray(candidate, dtype=object)
    if arr.ndim == 0 or arr.size == 1:
        return np.full((F, L), str(arr.reshape(-1)[0]), dtype=object)
    if arr.shape == (L,):
        return np.tile(arr.reshape(1, L), (F, 1)).astype(object)
    if arr.shape == (F, L):
        return arr.astype(object)
    label = "/".join(str(value) for value in arr.ravel())
    return np.full((F, L), label, dtype=object)


def _vehicle_label(model_grid: np.ndarray, i: int) -> str:
    parts = [f"C{l + 1}:{_model_abbreviation(model_grid[i, l])}"
             for l in range(model_grid.shape[1])]
    return f"Vehicle {i + 1}\n" + "  ".join(parts)


def _model_abbreviation(value: Any) -> str:
    text = str(value).lower()

    if "gamma" in text:
        return "Gamma"

    if ("rain" in text or "remaining" in text or text == "exact"):
        return "Rainflow"

    return str(value)


def _unique_models(model_grid: np.ndarray) -> list[str]:
    seen: list[str] = []
    for value in model_grid.ravel():
        label = _model_abbreviation(value)
        if label not in seen:
            seen.append(label)
    return seen


def _cell_grid(value: Any, F: int, L: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0 or arr.size == 1:
        return np.full((F, L), float(arr.reshape(-1)[0]))
    if arr.shape == (L,):
        return np.tile(arr.reshape(1, L), (F, 1))
    if arr.shape == (F,):
        return np.tile(arr.reshape(F, 1), (1, L))
    if arr.shape == (F, L):
        return arr
    raise ValueError(f"{name} has shape {arr.shape}; expected scalar, ({L},), ({F},), or ({F}, {L})")


def _optional_array(data: dict[str, Any], key: str, shape: tuple[int, ...]) -> np.ndarray:
    if data.get(key) is None:
        return np.zeros(shape, dtype=float)
    arr = np.asarray(data[key], dtype=float)
    if arr.shape != shape:
        raise ValueError(f"{key} has shape {arr.shape}; expected {shape}")
    return arr


def _legacy_h1(value: Any, T: int) -> int:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0])
    return int(value) if value is not None else T // 2


def _resolve_plot_path(plot_file_path: str | None) -> Path:
    path = Path("output.png" if plot_file_path is None else plot_file_path)
    return path.with_suffix(".png") if path.suffix == "" else path


def _read_input(input_file: Path) -> dict[str, Any]:
    extension = input_file.suffix.lower()
    if extension in {".yaml", ".yml"}:
        return yaml.safe_load(input_file.read_text(encoding="utf-8"))
    if extension == ".json":
        return json.loads(input_file.read_text(encoding="utf-8"))
    if extension in {".h5", ".hdf5"}:
        return _read_hdf5(input_file)
    raise ValueError(f"Unsupported input file type: {extension}")


def _read_hdf5(path: Path) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "Reading HDF5 result files requires the optional 'h5py' package."
        ) from exc

    data: dict[str, Any] = {}
    with h5py.File(path, "r") as handle:
        for key, value in handle.attrs.items():
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            if isinstance(value, str) and value[:1] in {"{", "["}:
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass
            data[key] = value.tolist() if isinstance(value, np.ndarray) else value
        for key, dataset in handle.items():
            value = dataset[()]
            data[key] = value.tolist() if isinstance(value, np.ndarray) else value
    return data


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
