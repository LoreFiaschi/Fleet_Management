"""Result inspector of the dashboards (spec, section "Dashboards", mode "Result inspector").

The functions without the prefix ``render_`` need neither Streamlit nor Gurobi.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.special import ndtri

from fleet_management.config import _lookup, load_config, read_input
from fleet_management.degradation_model.gamma import mean_cap, smallest_rate
from fleet_management.degradation_model.inverse_gaussian import ig_mean_cap
from fleet_management.solver import read_result
from fleet_management.validation.validator import (
    LOOP_KEYS,
    check_result,
    check_sizes,
    start_values,
)


# ---------------------------------------------------------------------------
# Loading and margins
# ---------------------------------------------------------------------------
def load_pair(input_path, results_path) -> tuple:
    """Read an input file and a result file and check that they fit (spec, Dashboards, Loading)."""
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not Path(results_path).exists():
        raise FileNotFoundError(f"Result file not found: {results_path}")
    cfg = load_config(read_input(input_path))
    result = read_result(results_path)
    check_sizes(cfg, result)
    if result.get("model") is None or (
        np.asarray(result["model"], dtype=object).astype(str).tolist()
        != cfg.model.astype(str).tolist()
    ):
        raise ValueError("'model' differs between the result and the input.")
    return cfg, result


def cell_label(cfg, i: int, l: int) -> str:
    """Name of the model of cell (i, l); a rainflow cell adds its bound (spec, Dashboards)."""
    if cfg.model[i, l] == "rainflow":
        return f"rainflow ({cfg.bound_method[i, l]})"
    return str(cfg.model[i, l])


def reliability_caps(cfg) -> np.ndarray:
    """Mean caps of the inverse Gaussian and Gamma cells (Eqs. ig_row, gm_row); NaN elsewhere."""
    caps = np.full((cfg.F, cfg.L), np.nan)
    gamma = np.argwhere(cfg.model == "gamma")
    if len(gamma):
        mu_inc = _lookup(cfg.mu, cfg.mu_trans, cfg.H1, cfg.H2)
        beta_inc = _lookup(cfg.gamma_beta, cfg.gamma_beta_trans, cfg.H1, cfg.H2)
    for i, l in np.argwhere(cfg.model == "inverse_gaussian"):
        caps[i, l] = ig_mean_cap(float(cfg.eta[i, l]), float(cfg.tau[i, l]),
                                 float(cfg.epsilon[i, l]))
    for i, l in gamma:
        beta_star = smallest_rate(
            mu_inc[i, l], beta_inc[i, l], float(cfg.mu_0[i, l]), float(cfg.gamma_beta_0[i, l]),
            float(cfg.replacement_mu[i, l]), float(cfg.gamma_beta_new[i, l]),
            cfg.allow_replacement,
        )
        caps[i, l] = mean_cap(beta_star, float(cfg.tau[i, l]), float(cfg.epsilon[i, l]))
    return caps


def reliability_margins(cfg, result) -> np.ndarray:
    """Right side minus left side of the exact reliability row of each cell (spec, Dashboards)."""
    shape = (cfg.F, cfg.L, cfg.T)

    def state(key):
        value = result.get(key)
        return np.full(shape, np.nan) if value is None else np.asarray(value, dtype=float)

    mu, v, R, K = (state(key) for key in ("mu", "v", "R", "K"))
    tau, eps = cfg.tau[:, :, None], cfg.epsilon[:, :, None]
    s = cfg.s_chernoff[:, :, None]
    cap = reliability_caps(cfg)[:, :, None]
    labels = np.array([[cell_label(cfg, i, l) for l in range(cfg.L)] for i in range(cfg.F)])

    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.log(1.0 / eps)
        kg2 = (-ndtri(eps)) ** 2
        c = kappa * cfg.support_const[:, :, None] / 3.0
        gaussian = np.minimum((tau - mu) ** 2 - kg2 * v, tau - mu)           # Eq. g_row
        margins = {
            "gaussian": gaussian,
            "wiener": gaussian,
            "inverse_gaussian": cap - mu,                                    # Eq. ig_row
            "gamma": cap - mu,                                               # Eq. gm_row
            "rainflow (markov)": eps * tau - mu,                             # Eq. rf_markov
            "rainflow (cantelli)": np.minimum(                               # Eq. rf_cantelli
                (tau - mu) ** 2 - (1.0 - eps) / eps * v, tau - mu),
            "rainflow (hoeffding)": np.minimum(                              # Eq. rf_hoeffding
                (tau - mu) ** 2 - kappa / 2.0 * R, tau - mu),
            "rainflow (bernstein)": np.minimum(                              # Eq. rf_bernstein
                (tau - c - mu) ** 2 - c ** 2 - 2.0 * kappa * v, tau - c - mu),
            "rainflow (chernoff)": np.log(eps) - (K - s * tau),              # Eq. rf_chernoff
        }

    out = np.full(shape, np.nan)
    for label, margin in margins.items():
        out = np.where((labels == label)[:, :, None], margin, out)
    return out


# ---------------------------------------------------------------------------
# Tables and marks
# ---------------------------------------------------------------------------
def margin_table(cfg, result, margins, tol) -> pd.DataFrame:
    """One row per cell and day; rows without a margin are left out (spec, Dashboards)."""
    F, L, T = margins.shape
    ii, ll, kk = (a.ravel() for a in np.meshgrid(np.arange(F), np.arange(L), np.arange(T),
                                                 indexing="ij"))
    mu = np.asarray(result["mu"], dtype=float).ravel()
    tau = cfg.tau[ii, ll]
    labels = np.array([[cell_label(cfg, i, l) for l in range(L)] for i in range(F)])
    table = pd.DataFrame({
        "train": ii + 1,
        "component": np.array(cfg.component_names, dtype=object)[ll],
        "model": labels[ii, ll],
        "day": kk + 1,
        "phase": np.where(kk < cfg.H1, "transitory", "operating"),
        "mu": mu,
        "tau": tau,
        "mu_over_tau": mu / tau,
        "margin": margins.ravel(),
    })
    table["holds"] = table["margin"] >= -tol
    return table[table["margin"].notna()].reset_index(drop=True)


def overview(cfg, result, table, n: int = 10) -> dict:
    """Summary of the tab "Overview" (spec, Dashboards, mode Result inspector)."""
    # Largest mu/tau over every cell and day with a number, also where the margin is NaN.
    mu_over_tau = np.asarray(result["mu"], dtype=float) / np.asarray(cfg.tau, dtype=float)[:, :, None]
    finite = mu_over_tau[np.isfinite(mu_over_tau)]
    return {
        "status": result.get("status"),
        "objective": result.get("objective"),
        "models": sorted({cell_label(cfg, i, l) for i in range(cfg.F) for l in range(cfg.L)}),
        "H1": cfg.H1,
        "H2": cfg.H2,
        "T": cfg.T,
        "reliability_violations": int((~table["holds"]).sum()),
        "largest_mu_over_tau": float(finite.max()) if finite.size else float("nan"),
        "smallest_margins": table.nsmallest(n, "margin"),
    }


def loop_table(cfg, result, tol) -> pd.DataFrame:
    """The loop rows of each cell and saved tracked quantity (Eq. a_loop).

    With the saved start ``y_q`` of the repetitions: ``margin_T = y - q_T``, ``margin_H1 =
    y - q_H1`` and ``margin`` their minimum. Without it (a result written before ``y``
    existed): ``margin_T = q_H1 - q_T``, ``margin_H1`` and ``y`` are NaN, ``margin = margin_T``.
    A row holds if ``margin >= -tol``.
    """
    columns = ["train", "component", "quantity", "value_at_H1", "y", "value_at_T",
               "margin_T", "margin_H1", "margin", "holds"]
    nan = float("nan")
    rows = []
    for key in LOOP_KEYS:
        if result.get(key) is None:
            continue
        a = np.asarray(result[key], dtype=float)
        y = start_values(result, key)
        for i in range(cfg.F):
            for l in range(cfg.L):
                first, last = float(a[i, l, cfg.H1 - 1]), float(a[i, l, cfg.T - 1])
                if not (np.isfinite(first) and np.isfinite(last)):
                    continue
                if y is None:
                    margin_T, margin_H1, start = first - last, nan, nan
                else:
                    start = float(y[i, l])
                    if not np.isfinite(start):
                        continue                       # no start saved at this cell: skipped
                    margin_T, margin_H1 = start - last, start - first
                margin = margin_T if y is None else min(margin_T, margin_H1)
                rows.append([i + 1, cfg.component_names[l], key, first, start, last,
                             margin_T, margin_H1, margin, margin >= -tol])
    table = pd.DataFrame(rows, columns=columns)
    order = {(i + 1, name): n for i in range(cfg.F) for n, name in enumerate(cfg.component_names)}
    keys = {key: n for n, key in enumerate(LOOP_KEYS)}
    table = table.assign(_c=[order[(t, c)] for t, c in zip(table["train"], table["component"])],
                         _q=[keys[q] for q in table["quantity"]])
    table = table.sort_values(["train", "_c", "_q"], kind="stable").drop(columns=["_c", "_q"])
    return table.reset_index(drop=True)


def day_marks(result, i: int, l: int) -> list:
    """Mark of each day of cell (i, l): P, R, Mj, D or I (spec, Plotting API and Layout)."""
    x, m, r = (np.asarray(result[key], dtype=float) for key in ("x", "m", "r"))
    marks = []
    for k in range(x.shape[2]):
        missions = np.flatnonzero(x[i, 1:, k] > 0.5)
        if r[i, l, k] > 0.5:
            marks.append("P")
        elif m[i, l, k] > 0.5:
            marks.append("R")
        elif len(missions):
            marks.append(f"M{missions[0] + 1}")
        elif x[i, 0, k] > 0.5:
            marks.append("D")
        else:
            marks.append("I")
    return marks


def load_inspector(input_path, results_path, tol) -> dict:
    """Load the pair and compute everything the inspector shows (spec, Dashboards)."""
    cfg, result = load_pair(input_path, results_path)
    margins = reliability_margins(cfg, result)
    table = margin_table(cfg, result, margins, tol)
    return {
        "cfg": cfg,
        "result": result,
        "margins": margins,
        "table": table,
        "overview": overview(cfg, result, table),
        "loops": loop_table(cfg, result, tol),
        "report": check_result(cfg, result, tol),
        "tol": tol,
    }


# ---------------------------------------------------------------------------
# Figures (built without pyplot: no backend is needed)
# ---------------------------------------------------------------------------
def timeline_figure(cfg, result, i: int) -> Figure:
    """Means of train i with tau and the marks of its days (spec, Dashboards, Train timeline)."""
    fig = Figure(figsize=(max(cfg.T * 0.9, 7.0), 4.5), layout="constrained")
    ax = fig.subplots()
    mu = np.asarray(result["mu"], dtype=float)
    for l in range(cfg.L):
        name = cfg.component_names[l]
        values = np.concatenate([[cfg.mu_0[i, l]], mu[i, l, :]])
        line, = ax.plot(np.arange(cfg.T + 1), values, marker="o", label=name)
        ax.axhline(cfg.tau[i, l], linestyle="--", color=line.get_color(), label=f"tau ({name})")
        shift = 5.0 * (2 * l - cfg.L + 1)       # components side by side, as their points coincide
        for k, mark in enumerate(day_marks(result, i, l), start=1):
            ax.annotate(mark, (k, values[k]), textcoords="offset points", xytext=(shift, 6),
                        ha="center", fontsize=8, color=line.get_color())
    ax.axvline(cfg.H1 + 0.5, linestyle=":", color="gray")
    ax.set_xticks(range(cfg.T + 1))
    ax.set_xticklabels(["initial"] + [str(k) for k in range(1, cfg.T + 1)])
    ax.set_xlabel("day")
    ax.set_ylabel("tracked mean mu")
    ax.set_title(f"Train {i + 1}")
    ax.legend()
    return fig


def margin_heatmap_figure(cfg, margins) -> Figure:
    """Margins over cells and days (spec, Dashboards, Heatmap)."""
    rows = cfg.F * cfg.L
    fig = Figure(figsize=(max(cfg.T * 0.8, 7.0), max(rows * 0.5, 3.0)), layout="constrained")
    ax = fig.subplots()
    finite = np.abs(margins[np.isfinite(margins)])
    vmax = float(finite.max()) if finite.size and finite.max() > 0.0 else 1.0
    image = ax.imshow(margins.reshape(rows, cfg.T), aspect="auto", cmap="RdYlGn",
                      vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(rows))
    ax.set_yticklabels([f"Train {i + 1} - {cfg.component_names[l]}"
                        for i in range(cfg.F) for l in range(cfg.L)])
    ax.set_xticks(range(cfg.T))
    ax.set_xticklabels([str(k) for k in range(1, cfg.T + 1)])
    ax.axvline(cfg.H1 - 0.5, color="black")
    ax.set_xlabel("day")
    ax.set_title("Reliability margins (negative: the row is violated)")
    fig.colorbar(image, ax=ax, label="margin of the reliability row")
    return fig


def component_figure(cfg, result, l: int) -> Figure:
    """One component across the trains (spec, Dashboards, Component comparison)."""
    fig = Figure(figsize=(max(cfg.T * 0.9, 7.0), 4.0), layout="constrained")
    ax = fig.subplots()
    mu = np.asarray(result["mu"], dtype=float)
    days = np.arange(1, cfg.T + 1)
    for i in range(cfg.F):
        ax.plot(days, mu[i, l, :], marker="o", label=f"Train {i + 1}")
    for value in np.unique(cfg.tau[:, l]):
        ax.axhline(value, linestyle="--", color="gray", label=f"tau = {value:g}")
    ax.set_xticks(days)
    ax.set_xlabel("day")
    ax.set_ylabel("tracked mean mu")
    ax.set_title(f"{cfg.component_names[l]} across the trains")
    ax.legend()
    return fig


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def render_result_inspector(state: dict) -> None:
    """Draw the tabs of the mode "Result inspector" (spec, Dashboards)."""
    import streamlit as st

    cfg, result, table = state["cfg"], state["result"], state["table"]
    summary, loops, report = state["overview"], state["loops"], state["report"]
    names = list(cfg.component_names)
    trains = list(range(1, cfg.F + 1))

    (tab_overview, tab_reliability, tab_repeat, tab_timeline, tab_heatmap, tab_component,
     tab_validation, tab_raw) = st.tabs([
        "Overview", "Reliability", "Repeatability", "Train timeline", "Heatmap",
        "Component comparison", "Validation", "Raw data"])

    with tab_overview:
        objective = summary["objective"]
        top, bottom = st.columns(4), st.columns(3)
        top[0].metric("Status", str(summary["status"]))
        top[1].metric("Objective", "n/a" if objective is None else f"{objective:.4f}")
        top[2].metric("H1", summary["H1"])
        top[3].metric("H2", summary["H2"])
        bottom[0].metric("T", summary["T"])
        bottom[1].metric("Reliability violations", summary["reliability_violations"])
        bottom[2].metric("Largest mu/tau", f"{summary['largest_mu_over_tau']:.3f}")
        st.write("Models: " + ", ".join(summary["models"]))
        st.subheader("Cells with the smallest margin")
        st.dataframe(summary["smallest_margins"], hide_index=True)

    with tab_reliability:
        st.write(f"A row holds if its margin is at least -{state['tol']:g}.")
        left, middle, right = st.columns(3)
        with left:
            chosen_trains = st.multiselect("Train", trains, default=trains)
        with middle:
            chosen_components = st.multiselect("Component", names, default=names)
        with right:
            only_violations = st.checkbox("Violations only")
        shown = table[table["train"].isin(chosen_trains)
                      & table["component"].isin(chosen_components)]
        if only_violations:
            shown = shown[~shown["holds"]]
        st.dataframe(shown.sort_values("margin"), hide_index=True)

    with tab_repeat:
        st.write("Loop rows of each cell and saved tracked quantity: the values at days H1 and "
                 "T must not exceed the start y of the repetitions (margins y - q_T and "
                 "y - q_H1). A result without y is checked as before: the value at day T must "
                 "not exceed the value at day H1.")
        st.dataframe(loops, hide_index=True)
        violated = int((~loops["holds"]).sum())
        if violated == 0:
            st.success("Every loop row holds.")
        else:
            st.error(f"{violated} loop rows are violated.")

    with tab_timeline:
        train = st.selectbox("Train", trains)
        st.pyplot(timeline_figure(cfg, result, train - 1))
        st.caption("Marks: P replacement, R repair, Mj mission j, D depot day without "
                   "intervention, I idle day.")

    with tab_heatmap:
        st.pyplot(margin_heatmap_figure(cfg, state["margins"]))

    with tab_component:
        name = st.selectbox("Component", names)
        st.pyplot(component_figure(cfg, result, names.index(name)))
        by_train = (table[table["component"] == name].groupby("train")
                    .agg(largest_mu=("mu", "max"), smallest_margin=("margin", "min"))
                    .reset_index())
        st.dataframe(by_train, hide_index=True)

    with tab_validation:
        left, right = st.columns(2)
        left.metric("Solver objective", f"{report['solver_objective']:.4f}")
        right.metric("Recomputed objective", f"{report['recomputed_objective']:.4f}")
        st.dataframe(pd.DataFrame(report["checks"]), hide_index=True)
        failed = sum(not check["passed"] for check in report["checks"])
        if failed == 0:
            st.success("All checks passed.")
        else:
            st.error(f"{failed} checks failed.")

    with tab_raw:
        st.dataframe(table, hide_index=True)
        st.download_button("Download as CSV", table.to_csv(index=False).encode("utf-8"),
                           file_name="result_inspector.csv", mime="text/csv")
