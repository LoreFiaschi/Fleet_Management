"""Gamma diagnostic of the dashboards (spec, section "Dashboards", mode "Gamma diagnostic").

The functions without the prefix ``render_`` need neither Streamlit nor Gurobi.
"""

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from fleet_management.validation.dashboard_inspector import load_pair, reliability_caps


# ---------------------------------------------------------------------------
# Tables and loading
# ---------------------------------------------------------------------------
def gamma_cells(cfg) -> list:
    """The Gamma cells (i, l) of the input, i outer (spec, Dashboards, Gamma diagnostic)."""
    cells = [(int(i), int(l)) for i, l in np.argwhere(cfg.model == "gamma")]
    if not cells:
        raise ValueError("the input has no Gamma cell.")
    return cells


def gamma_mean_table(cfg, result, caps, tol) -> pd.DataFrame:
    """Tracked mean and cap of each Gamma cell and day (Eq. gm_row)."""
    mu = np.asarray(result["mu"], dtype=float)
    rows = []
    for i, l in gamma_cells(cfg):
        for k in range(cfg.T):
            margin = float(caps[i, l] - mu[i, l, k])
            rows.append([i + 1, cfg.component_names[l], k + 1, float(mu[i, l, k]),
                         float(caps[i, l]), margin, margin >= -tol])
    columns = ["train", "component", "day", "mu", "cap", "margin", "holds"]
    return pd.DataFrame(rows, columns=columns)


def gamma_tail_table(cfg, exact_report, tol) -> pd.DataFrame:
    """Exact tail upper bound against epsilon, one row per step (spec, Exact validator)."""
    rows = []
    for step in exact_report["steps"]:
        i, l, k = step["i"], step["l"], step["k"]
        bound, epsilon = float(step["exact_tail_upper_bound"]), float(cfg.epsilon[i, l])
        rows.append([i + 1, cfg.component_names[l], k + 1, step.get("event"), bound, epsilon,
                     epsilon - bound, epsilon - bound >= -tol])
    return pd.DataFrame(rows, columns=["train", "component", "day", "event", "tail_upper_bound",
                                       "epsilon", "slack", "holds"])


def gamma_overview(replay) -> dict:
    """Summary of the tab "Overview" from the replay report (spec, Replay validator)."""
    return {
        "valid": bool(replay["valid"]),
        "repairs": replay["repairs"],
        "replacements": replay["replacements"],
        "global_violations": list(replay["global_violations"]),
        "violations": len(replay["violations"]),
        "maximum_errors": dict(replay["maximum_errors"]),
    }


def load_gamma_diagnostic(input_path, results_path, tol) -> dict:
    """Load the pair, run the replay and exact validators, build the tables (spec, Dashboards)."""
    cfg, result = load_pair(input_path, results_path)
    gamma_cells(cfg)
    # imported here: a module of the Gamma validators that fails to load breaks only this mode
    from fleet_management.degradation_model.gamma_utils.gamma_replay_validator import (
        validate_gamma_replay_schedule,
    )
    from fleet_management.degradation_model.gamma_utils.gamma_tail_validator import (
        validate_gamma_tail_bound_schedule,
    )
    replay = validate_gamma_replay_schedule(cfg, result, tolerance=tol)
    exact = validate_gamma_tail_bound_schedule(cfg, result, tolerance=tol, include_steps=True)
    return {
        "cfg": cfg,
        "result": result,
        "replay": replay,
        "exact": exact,
        "overview": gamma_overview(replay),
        "means": gamma_mean_table(cfg, result, reliability_caps(cfg), tol),
        "tails": gamma_tail_table(cfg, exact, tol),
        "tol": tol,
    }


# ---------------------------------------------------------------------------
# Figures (built without pyplot: no backend is needed)
# ---------------------------------------------------------------------------
def _cell_rows(cfg, table, i: int, l: int) -> pd.DataFrame:
    """The rows of one cell of a table with the columns train and component (spec, Dashboards)."""
    return table[(table["train"] == i + 1) & (table["component"] == cfg.component_names[l])]


def gamma_mean_figure(cfg, means, i: int, l: int) -> Figure:
    """Tracked mean of one Gamma cell with its cap (spec, Dashboards, Gamma diagnostic)."""
    rows = _cell_rows(cfg, means, i, l)
    fig = Figure(figsize=(max(cfg.T * 0.9, 7.0), 4.0), layout="constrained")
    ax = fig.subplots()
    ax.plot(rows["day"], rows["mu"], marker="o", label="tracked mean mu")
    ax.axhline(float(rows["cap"].iloc[0]), linestyle="--", color="red", label="cap")
    ax.set_xticks(range(1, cfg.T + 1))
    ax.set_xlabel("day")
    ax.set_ylabel("tracked mean mu")
    ax.set_title(f"Train {i + 1} - {cfg.component_names[l]}: tracked mean and cap A_1/beta*")
    ax.legend()
    return fig


def gamma_tail_figure(cfg, tails, i: int, l: int) -> Figure:
    """Exact tail upper bound of one Gamma cell against epsilon (spec, Dashboards)."""
    rows = _cell_rows(cfg, tails, i, l)
    fig = Figure(figsize=(max(cfg.T * 0.9, 7.0), 4.0), layout="constrained")
    ax = fig.subplots()
    ax.plot(rows["day"], rows["tail_upper_bound"], marker="o", label="upper bound of the tail")
    ax.axhline(float(cfg.epsilon[i, l]), linestyle="--", color="red", label="epsilon")
    ax.set_xticks(range(1, cfg.T + 1))
    ax.set_xlabel("day")
    ax.set_ylabel("P(D > tau)")
    ax.set_title(f"Train {i + 1} - {cfg.component_names[l]}: exact tail P(D > tau)")
    ax.legend()
    return fig


def gamma_tail_heatmap_figure(cfg, tails) -> Figure:
    """Exact tail upper bound over the Gamma cells and days (spec, Dashboards)."""
    cells = gamma_cells(cfg)
    fig = Figure(figsize=(max(cfg.T * 0.8, 7.0), max(len(cells) * 0.5, 3.0)),
                 layout="constrained")
    ax = fig.subplots()
    values = np.full((len(cells), cfg.T), np.nan)
    for row, (i, l) in enumerate(cells):
        cell = _cell_rows(cfg, tails, i, l)
        values[row, cell["day"].to_numpy() - 1] = cell["tail_upper_bound"].to_numpy()
        for day, holds in zip(cell["day"], cell["holds"]):
            if not holds:
                ax.text(day - 1, row, "!", ha="center", va="center", fontweight="bold")
    vmax = max(float(cfg.epsilon[i, l]) for i, l in cells)
    image = ax.imshow(values, aspect="auto", cmap="Reds", vmin=0.0, vmax=vmax)
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels([f"Train {i + 1} - {cfg.component_names[l]}" for i, l in cells])
    ax.set_xticks(range(cfg.T))
    ax.set_xticklabels([str(k) for k in range(1, cfg.T + 1)])
    ax.axvline(cfg.H1 - 0.5, color="black")
    ax.set_xlabel("day")
    ax.set_title("Exact tail upper bound (! where it exceeds epsilon)")
    fig.colorbar(image, ax=ax, label="upper bound of P(D > tau)")
    return fig


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def render_gamma_diagnostic(state: dict) -> None:
    """Draw the tabs of the mode "Gamma diagnostic" (spec, Dashboards)."""
    import streamlit as st

    cfg, summary = state["cfg"], state["overview"]
    means, tails = state["means"], state["tails"]
    cells = gamma_cells(cfg)

    def label(cell):
        return f"Train {cell[0] + 1} - {cfg.component_names[cell[1]]}"

    tab_overview, tab_mean, tab_tail, tab_raw = st.tabs(
        ["Overview", "Mean and cap", "Exact tail", "Raw data"])

    with tab_overview:
        columns = st.columns(4)
        columns[0].metric("Repairs", summary["repairs"])
        columns[1].metric("Replacements", summary["replacements"])
        columns[2].metric("Replay validator", "valid" if summary["valid"] else "not valid")
        columns[3].metric("Violations", summary["violations"])
        if summary["global_violations"]:
            st.subheader("Global violations")
            for text in summary["global_violations"]:
                st.write(f"- {text}")
        st.subheader("Largest error of each field")
        st.json(summary["maximum_errors"])

    with tab_mean:
        i, l = st.selectbox("Cell", cells, format_func=label)
        st.pyplot(gamma_mean_figure(cfg, means, i, l))
        st.dataframe(_cell_rows(cfg, means, i, l), hide_index=True)

    with tab_tail:
        i, l = st.selectbox("Cell", cells, format_func=label, key="tail_cell")
        st.pyplot(gamma_tail_figure(cfg, tails, i, l))
        st.pyplot(gamma_tail_heatmap_figure(cfg, tails))
        violated = int((~tails["holds"]).sum())
        if violated == 0:
            st.success("Every exact tail upper bound is at most epsilon.")
        else:
            st.error(f"{violated} days have an exact tail upper bound above epsilon.")

    with tab_raw:
        st.subheader("Tracked mean and cap")
        st.dataframe(means, hide_index=True)
        st.download_button("Download the means as CSV", means.to_csv(index=False).encode("utf-8"),
                           file_name="gamma_means.csv", mime="text/csv")
        st.subheader("Exact tail")
        st.dataframe(tails, hide_index=True)
        st.download_button("Download the tails as CSV", tails.to_csv(index=False).encode("utf-8"),
                           file_name="gamma_tails.csv", mime="text/csv")
