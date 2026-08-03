"""
Streamlit dashboard: "Rainflow Inspector".

A rainflow-compatible counterpart to the baseline Gaussian validator. It reads
only an input YAML and a solver output YAML -- NO log file is required; every
quantity (structural checks, reliability margins, damage trajectories) is
recomputed on the fly from those two files.

Differences from the Gaussian validator:
  * Threshold is ``tau`` (Palmgren-Miner limit), read from the output; the code
    falls back to ``alpha`` if a file still stores the threshold under that key.
  * Reliability is the distribution-free bound  P(D > tau) <= epsilon  selected
    by ``method`` (markov / cantelli / hoeffding / bernstein / chernoff), not the
    Gaussian quantile. Cantelli/Markov need only the (mu, v) state that the
    output already carries; Hoeffding/Bernstein/Chernoff additionally use the
    input support / cgf arrays, reconstructed along the schedule.

Wire-up: add one option to the mode selectbox in ``validator_dashboard.py`` and
dispatch to ``render_rainflow_inspector_dashboard()`` (see the accompanying
snippet).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import streamlit as st
import yaml


# ===========================================================================
# Pure-numeric helpers (no streamlit / no package imports) -- unit-testable
# ===========================================================================
_METHODS = ("markov", "cantelli", "hoeffding", "bernstein", "chernoff")


def read_results_file(path: Path) -> dict:
    """Read a solver output file (YAML or JSON) into a dict."""
    ext = path.suffix.lower()
    if ext in (".yaml", ".yml"):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    if ext == ".json":
        with open(path, "r") as f:
            return json.load(f)
    raise ValueError(f"Unsupported result file type: {ext}")


def parse_results(data: dict) -> dict:
    """Extract the rainflow result arrays, tolerant of tau-vs-alpha naming."""
    F = int(data["F"]); M = int(data["M"]); H = int(data["H"])
    L = int(data.get("L", 1))
    # Two horizons: transitory H1 + operating H2 (T = H1 + H2). For a single
    # horizon (or legacy output files) H1 = H2 = H and T = 2H.
    H1 = int(data.get("H1", H))
    H2 = int(data.get("H2", H))
    T = int(data.get("T", H1 + H2))
    # threshold: prefer tau, fall back to alpha
    threshold = data.get("tau", data.get("alpha"))
    if threshold is None:
        raise KeyError("Result file has neither 'tau' nor 'alpha'.")
    out = {
        "status": str(data.get("status", "unknown")).lower(),
        "objective": None if data.get("objective") is None else float(data["objective"]),
        "degradation": str(data.get("degradation", "rainflow")).lower(),
        "method": str(data.get("method", "cantelli")).lower(),
        "F": F, "M": M, "H": H, "H1": H1, "H2": H2, "T": T, "L": L,
        "tau": float(threshold),
        "x": np.asarray(data["x"], dtype=float),
        "mu": np.asarray(data["mu"], dtype=float),
        "u": np.asarray(data["u"], dtype=float),
        "z": np.asarray(data["z"], dtype=float),
        "mu_0": _as_2d(data["mu_0"], F, L),
    }
    if "v" in data and data["v"] is not None:
        out["v"] = np.asarray(data["v"], dtype=float)
    if "v_0" in data and data["v_0"] is not None:
        out["v_0"] = _as_2d(data["v_0"], F, L)
    return out


def _as_2d(value, F: int, L: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape == (F, L):
        return arr
    if L == 1 and arr.shape == (F,):
        return arr[:, np.newaxis]
    raise ValueError(f"array shape {arr.shape} != (F={F}, L={L})")


def activity_grid(x: np.ndarray, M: int, tol: float = 1e-6) -> np.ndarray:
    """(F, 2H) int grid: -1 idle, 0 maintenance, j=1..M mission j."""
    F, _, T = x.shape
    act = np.full((F, T), -1, dtype=int)
    for i in range(F):
        for k in range(T):
            for j in range(M + 1):
                if abs(x[i, j, k] - 1.0) <= 0.5:
                    act[i, k] = j
                    break
    return act


def reconstruct_state(x: np.ndarray, incr: np.ndarray, xi: np.ndarray,
                      H1: int, H2: int, initial: np.ndarray | None,
                      incr_trans: np.ndarray | None = None,
                      tol: float = 1e-6) -> np.ndarray:
    """Roll an accumulating descriptor forward along the schedule.

    Same recursion the solver uses: mission -> += increment, maintenance ->
    *= (1 - xi), idle -> unchanged.  ``incr`` is the OPERATING profile
    (F, M, L, H2); ``incr_trans`` is the optional transitory profile
    (F, M, L, H1).  Time indexing is phase-aware, matching the two-horizon
    solver: step k < H1 uses local time in the transitory phase, k >= H1 uses
    (k - H1) in the operating phase.  With H1 == H2 and no transitory profile
    this reduces to the classic ``k % H`` indexing.  Returns (F, L, T).

    Used to rebuild the Hoeffding support-sum (incr = support**2) and the
    Chernoff CGF (incr = cgf) states, which the output file does not store.
    """
    F, M1, T = x.shape
    M = M1 - 1
    L = incr.shape[2]

    def _inc(i, j, l, k):
        if k < H1:
            if incr_trans is not None:
                return float(incr_trans[i, j, l, k % H1])
            return float(incr[i, j, l, k % H2])
        return float(incr[i, j, l, (k - H1) % H2])

    out = np.zeros((F, L, T))
    for i in range(F):
        for l in range(L):
            s = 0.0 if initial is None else float(initial[i, l])
            for k in range(T):
                if abs(x[i, 0, k] - 1.0) <= 0.5:
                    s = s * (1.0 - xi[i, l])
                else:
                    j = next((jj for jj in range(1, M + 1)
                              if abs(x[i, jj, k] - 1.0) <= 0.5), None)
                    if j is not None:
                        s = s + _inc(i, j - 1, l, k)
                out[i, l, k] = s
    return out


def reliability_margin_grid(method: str, mu: np.ndarray, v: np.ndarray,
                            tau: float, eps: float,
                            R: np.ndarray | None = None,
                            K: np.ndarray | None = None,
                            s: float | None = None,
                            b: float | None = None) -> np.ndarray:
    """Signed slack of P(D > tau) <= eps (>=0 satisfied). Vectorized over any shape.

    Same algebra as the solver's per-step constraint. Magnitudes are not
    comparable across methods; use the sign.
    """
    Le = math.log(1.0 / eps)
    if method == "markov":
        return eps * tau - mu
    t = tau - mu
    if method == "cantelli":
        m = eps * t * t - (1.0 - eps) * v
    elif method == "hoeffding":
        m = t * t - 0.5 * Le * R
    elif method == "bernstein":
        m = 0.5 * t * t - (Le * b / 3.0) * t - Le * v
    elif method == "chernoff":
        return math.log(eps) - (K - s * tau)
    else:
        m = eps * t * t - (1.0 - eps) * v  # default: cantelli
    # bounds above require mean below threshold; force a violation where mu > tau
    return np.where(mu > tau, np.minimum(m, tau - mu), m)


def build_state_dataframe(res: dict, act: np.ndarray, eps: float,
                          rel_margin: np.ndarray, tol: float) -> pd.DataFrame:
    """One row per (vehicle, component, time step) of the reported state."""
    F, L, T = res["mu"].shape
    tau = res["tau"]
    mu = res["mu"]
    v = res.get("v")
    act_name = {-1: "idle", 0: "maintenance"}
    rows = []
    for i in range(F):
        for l in range(L):
            for k in range(T):
                a = int(act[i, k])
                name = act_name.get(a, f"mission {a}")
                dmg = float(mu[i, l, k])
                rows.append({
                    "time_step": k,
                    "vehicle": i,
                    "component": l,
                    "activity": name,
                    "damage_mu": dmg,
                    "variance_v": float(v[i, l, k]) if v is not None else np.nan,
                    "threshold_tau": tau,
                    "margin_to_tau": tau - dmg,
                    "utilization_of_tau": dmg / tau if tau else np.nan,
                    "reliability_margin": float(rel_margin[i, l, k]),
                    "reliable": bool(rel_margin[i, l, k] >= -tol),
                    "horizon": "transitory" if k < res["H1"] else "operating",
                })
    return pd.DataFrame(rows)


def structural_checks(res: dict, tol: float) -> list[dict]:
    """Recompute the schedule-level feasibility checks from the output alone."""
    x, mu, u = res["x"], res["mu"], res["u"]
    F, M, L = res["F"], res["M"], res["L"]
    H1 = res.get("H1", res["H"])
    T = res.get("T", mu.shape[-1])
    checks = []

    def add(name, violation):
        checks.append({"name": name, "violation": float(violation),
                       "passed": bool(violation <= tol)})

    add("x_binary", float(np.max(np.abs(x - np.round(x)))))
    add("assignment_sum_j_x_le_1", max(0.0, float(np.max(np.sum(x, axis=1) - 1.0))))
    add("demand_sum_i_x_eq_1", float(np.max(np.abs(np.sum(x, axis=0) - 1.0))))
    add("u_ge_mu", max(0.0, float(np.max(np.max(mu, axis=(0, 1)) - u))))
    add("capacity_sum_mu_le_F_minus_M",
        max(0.0, float(np.max(np.sum(mu, axis=(0, 1)) - (F - M)))))
    # operating-horizon loop: state(T-1) <= state(H1-1)
    add("mu_periodic_operating_loop",
        max(0.0, float(np.max(mu[:, :, T - 1] - mu[:, :, H1 - 1]))))
    if res.get("v") is not None:
        v = res["v"]
        add("v_periodic_operating_loop",
            max(0.0, float(np.max(v[:, :, T - 1] - v[:, :, H1 - 1]))))
    return checks


# ===========================================================================
# Drawing helpers (mirror the Gaussian dashboard's icons)
# ===========================================================================
def _draw_gear(ax, cx, cy):
    ax.text(cx, cy, "\u2699", ha="center", va="center", fontsize=16, color="black")


def _draw_sleep_cloud(ax, cx, cy):
    cloud = mpatches.FancyBboxPatch(
        (cx - 0.3, cy - 0.2), 0.6, 0.4, boxstyle="round,pad=0.05",
        facecolor="white", edgecolor="gray", linewidth=0.8, alpha=0.85)
    ax.add_patch(cloud)
    ax.text(cx, cy, "zzz", ha="center", va="center", fontsize=8,
            fontstyle="italic", color="darkblue")


def _draw_violation(ax, cx, cy):
    ax.text(cx, cy, "\u26A0", ha="center", va="center", fontsize=11,
            color="white", fontweight="bold",
            bbox=dict(boxstyle="circle,pad=0.18", facecolor="darkred",
                      edgecolor="white", linewidth=0.8, alpha=0.95), zorder=10)


def _schedule_heatmap(res: dict, rel_margin: np.ndarray, tol: float):
    """The signature grid: damage heatmap (0..tau) with schedule icons and
    reliability-violation badges. Mirrors the Gaussian heatmap tab."""
    F, L, T = res["mu"].shape
    M, tau = res["M"], res["tau"]
    H1 = res.get("H1", res["H"])
    x, mu_0, mu = res["x"], res["mu_0"], res["mu"]

    n_cols = T + 1                       # column 0 = initial state
    grid = np.zeros((F, L, n_cols))
    grid[:, :, 0] = mu_0
    grid[:, :, 1:] = mu

    cmap = mcolors.LinearSegmentedColormap.from_list("gr", ["green", "red"])
    cnorm = mcolors.Normalize(vmin=0.0, vmax=tau)
    fig, ax = plt.subplots(figsize=(max(n_cols * 0.8, 6), max(F * 0.8, 4)))

    strip_h = 1.0 / L
    for i in range(F):
        for k in range(n_cols):
            for l in range(L):
                y0 = i - 0.5 + l * strip_h
                ax.add_patch(mpatches.Rectangle(
                    (k - 0.5, y0), 1.0, strip_h,
                    facecolor=cmap(cnorm(grid[i, l, k])), edgecolor="none"))

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(F - 0.5, -0.5)

    if L > 1:
        for i in range(F):
            for l in range(1, L):
                y = i - 0.5 + l * strip_h
                ax.hlines(y, -0.5, n_cols - 0.5, colors="gray",
                          linewidths=0.4, linestyle="--")

    # schedule icons (column k>=1 maps to solver step k-1)
    for i in range(F):
        for k in range(1, n_cols):
            kx = k - 1
            if abs(x[i, 0, kx] - 1.0) <= 0.5:
                _draw_gear(ax, k, i)
            else:
                j = next((jj for jj in range(1, M + 1)
                          if abs(x[i, jj, kx] - 1.0) <= 0.5), None)
                if j is not None:
                    ax.text(k, i, str(j), ha="center", va="center",
                            fontsize=10, fontweight="bold", color="black")
                else:
                    _draw_sleep_cloud(ax, k, i)

    # reliability-violation badges (on state columns)
    n_viol = 0
    for i in range(F):
        for l in range(L):
            for kx in range(T):
                if rel_margin[i, l, kx] < -tol:
                    y_center = i - 0.5 + l * strip_h + 0.5 * strip_h
                    _draw_violation(ax, (kx + 1) + 0.28, y_center)
                    n_viol += 1

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(range(0, n_cols))
    ax.set_yticks(range(F))
    ax.set_yticklabels(range(0, F))
    ax.set_xlabel("Time step k  (column 0 = initial)")
    ax.set_ylabel("Vehicle i")
    for i in range(F + 1):
        ax.axhline(i - 0.5, color="black", linewidth=0.5)
    for k in range(n_cols + 1):
        ax.axvline(k - 0.5, color="black", linewidth=0.5)
    # transitory | operating divider (between mu-column H1 and H1+1)
    ax.axvline(H1 + 0.5, color="black", linewidth=2.0)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=cnorm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="mean accumulated damage E[D]", shrink=0.8)
    fig.tight_layout()
    return fig, n_viol


# ===========================================================================
# Streamlit entry point
# ===========================================================================
def _path_exists(p: str) -> bool:
    return Path(p).exists()


def render_rainflow_inspector_dashboard():
    st.title("Rainflow Inspector")
    st.write("BUILD CHECK: repeatability + timeline v2")
    st.caption("Reliability & schedule inspection for rainflow (remaining-life) "
               "solver outputs -- reads input + output YAML, no log file needed.")

    # -- sidebar --------------------------------------------------------------
    with st.sidebar:
        st.header("Files")
        input_path = st.text_input("Input YAML", value="input/data_test_rainflow.yaml")
        results_path = st.text_input("Solver output YAML", value="results/output_rainflow.yaml")

        st.header("Diagnostic settings")
        tol = st.number_input("Tolerance", min_value=0.0, value=1e-6, format="%.1e")
        eps_override = None
        if st.checkbox("Override epsilon", value=False):
            eps_override = st.number_input("epsilon", min_value=1e-9, max_value=0.999,
                                           value=0.1, format="%.4f")
        method_override = st.selectbox(
            "Reliability bound (override)",
            options=["(use output/method)"] + list(_METHODS), index=0)
        load = st.button("Load data", type="primary")

    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Input file**"); st.code(input_path)
        st.write("Exists:", "✅" if _path_exists(input_path) else "❌")
    with col_b:
        st.write("**Output file**"); st.code(results_path)
        st.write("Exists:", "✅" if _path_exists(results_path) else "❌")

    if load:
        _load_into_state(input_path, results_path, tol, eps_override, method_override)

    if "rf" not in st.session_state:
        st.info("Set the paths in the sidebar and click **Load data**.")
        st.stop()

    _render_tabs(st.session_state["rf"])


def _load_into_state(input_path, results_path, tol, eps_override, method_override):
    if not _path_exists(input_path):
        st.error(f"Input file not found: {input_path}"); st.stop()
    if not _path_exists(results_path):
        st.error(f"Output file not found: {results_path}"); st.stop()
    try:
        res = parse_results(read_results_file(Path(results_path)))

        # Pull epsilon (+ optional support/cgf/xi) from the input via the
        # package's registry parser. Kept optional so cantelli/markov still work
        # if the parser or extra arrays are unavailable.
        eps, xi, support_param, cgf_param, s_chernoff = _read_input_extras(
            input_path, res)

        if eps_override is not None:
            eps = eps_override
        if eps is None:
            eps = 0.1
            st.warning("epsilon not found in input; defaulting to 0.1 "
                       "(set it via 'Override epsilon').")

        method = res["method"]
        if method_override != "(use output/method)":
            method = method_override

        # reconstruct extra descriptor states if the chosen method needs them
        R = K = None
        note = None
        H1, H2 = res["H1"], res["H2"]
        if method == "hoeffding":
            if support_param is not None and xi is not None:
                R = reconstruct_state(res["x"], support_param ** 2, xi, H1, H2, None)
            else:
                note = "Hoeffding needs input 'support'; falling back to Cantelli."
                method = "cantelli"
        elif method == "chernoff":
            if cgf_param is not None and xi is not None and s_chernoff is not None:
                K = reconstruct_state(res["x"], cgf_param, xi, H1, H2, None)
            else:
                note = "Chernoff needs input 'cgf' + 's'; falling back to Cantelli."
                method = "cantelli"
        b = float(support_param.max()) if (method == "bernstein" and support_param is not None) else None
        if method == "bernstein" and b is None:
            note = "Bernstein needs input 'support'; falling back to Cantelli."
            method = "cantelli"

        v = res.get("v")
        if v is None:
            v = np.zeros_like(res["mu"])
        rel = reliability_margin_grid(method, res["mu"], v, res["tau"], eps,
                                      R=R, K=K, s=s_chernoff, b=b)
        act = activity_grid(res["x"], res["M"])
        df = build_state_dataframe(res, act, eps, rel, tol)
        checks = structural_checks(res, tol)

        st.session_state["rf"] = {
            "res": res, "act": act, "rel": rel, "df": df, "checks": checks,
            "eps": eps, "method_used": method, "tol": tol, "note": note,
            "xi": xi,
            "input_path": input_path, "results_path": results_path,
        }
    except Exception as exc:  # noqa: BLE001
        st.error("Failed to load rainflow data."); st.exception(exc); st.stop()


def _read_input_extras(input_path, res):
    """Best-effort read of epsilon / xi / support / cgf / s from the input file.

    Tries the package registry parser first; falls back to a raw YAML read so the
    dashboard still works outside the package. Returns (eps, xi, support, cgf, s).
    """
    eps = xi = support = cgf = s = None
    F, M, H, L = res["F"], res["M"], res["H"], res["L"]
    try:
        from fleet_management.solver import _read_input
        from fleet_management.model_registry import extract_degradation_parameters
        data = _read_input(Path(input_path))
        p = extract_degradation_parameters(data, "rainflow")
        eps = p.get("epsilon")
        xi = np.asarray(p["xi"], float) if p.get("xi") is not None else None
        support = np.asarray(p["support_param"], float) if p.get("support_param") is not None else None
        cgf = np.asarray(p["cgf_param"], float) if p.get("cgf_param") is not None else None
        s = p.get("s_chernoff")
        return eps, xi, support, cgf, s
    except Exception:
        pass
    # raw fallback
    try:
        with open(input_path) as f:
            data = yaml.safe_load(f)
        eps = data.get("epsilon")
        if "xi" in data:
            xi = _as_2d(data["xi"], F, L)
    except Exception:
        pass
    return eps, xi, support, cgf, s


def _render_tabs(state: dict):
    res, df, rel = state["res"], state["df"], state["rel"]
    method_used, eps, tol = state["method_used"], state["eps"], state["tol"]
    tau = res["tau"]

    if state.get("note"):
        st.warning(state["note"])

    (tab_overview, tab_reliability, tab_repeat, tab_vehicle,
     tab_heatmap, tab_component, tab_validation, tab_raw) = st.tabs([
        "Overview", "Reliability", "Repeatability", "Vehicle timeline",
        "Heatmap", "Component comparison", "Validation", "Raw data"])

    # ---- Overview ----
    with tab_overview:
        st.header("Overview")
        c = st.columns(5)
        c[0].metric("Status", res["status"])
        c[1].metric("Objective",
                    "n/a" if res["objective"] is None else f"{res['objective']:.4f}")
        c[2].metric("Method", method_used)
        c[3].metric("tau", f"{tau:.3f}")
        c[4].metric("epsilon", f"{eps:.3f}")

        n_unreliable = int((~df["reliable"]).sum())
        worst = df.loc[df["reliability_margin"].idxmin()]
        hottest = df.loc[df["damage_mu"].idxmax()]
        c2 = st.columns(4)
        c2[0].metric("F x L x T cells", len(df))
        c2[1].metric("Reliability violations", n_unreliable)
        c2[2].metric("Max damage E[D]", f"{hottest['damage_mu']:.4f}")
        c2[3].metric("Max utilization of tau",
                     f"{100.0 * df['utilization_of_tau'].max():.1f}%")

        if n_unreliable == 0:
            st.success(f"All cells satisfy P(D>tau)<=eps under '{method_used}'.")
        else:
            st.error(f"{n_unreliable} cells violate reliability under '{method_used}'.")

        st.subheader("Tightest reliability margin")
        st.dataframe(pd.DataFrame([worst]).reset_index(drop=True),
                     width="stretch", hide_index=True)

        st.subheader("Closest cells to the reliability limit")
        n = st.slider("Rows", 5, 50, 10, 5)
        st.dataframe(df.sort_values("reliability_margin").head(n).reset_index(drop=True),
                     width="stretch", hide_index=True)

    # ---- Reliability ----
    with tab_reliability:
        st.header("Reliability")
        st.write(f"Bound in use: **{method_used}**, threshold tau=**{tau:.3f}**, "
                 f"epsilon=**{eps:.3f}**. A cell is reliable when its margin >= 0.")
        f1, f2, f3 = st.columns(3)
        with f1:
            veh = st.multiselect("Vehicle", sorted(df["vehicle"].unique()),
                                 default=sorted(df["vehicle"].unique()))
        with f2:
            comp = st.multiselect("Component", sorted(df["component"].unique()),
                                  default=sorted(df["component"].unique()))
        with f3:
            only_bad = st.checkbox("Only violations", value=False)
        sub = df[df["vehicle"].isin(veh) & df["component"].isin(comp)]
        if only_bad:
            sub = sub[~sub["reliable"]]
        st.dataframe(sub.sort_values("reliability_margin").reset_index(drop=True),
                     width="stretch", hide_index=True)

        # reliability-margin heatmap (per component, stacked)
        st.subheader("Reliability-margin heatmap (red = violation)")
        F, L, T = res["mu"].shape
        H1 = res["H1"]
        fig, axes = plt.subplots(L, 1, figsize=(max(T * 0.6, 6), max(F * 0.5 * L, 3)),
                                 squeeze=False)
        vmax = float(np.abs(rel).max()) or 1.0
        for l in range(L):
            ax = axes[l, 0]
            im = ax.imshow(rel[:, l, :], aspect="auto", cmap="RdYlGn",
                           vmin=-vmax, vmax=vmax)
            ax.set_title(f"component {l}", fontsize=10, loc="left")
            ax.set_yticks(range(F)); ax.set_yticklabels(range(F))
            ax.set_xticks(range(T)); ax.set_xticklabels(range(T), fontsize=7)
            ax.axvline(H1 - 0.5, color="k", lw=1.5)
            ax.set_ylabel("vehicle")
        axes[-1, 0].set_xlabel("time step k")
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7,
                     label="reliability margin")
        st.pyplot(fig)

    # ---- Repeatability ----
    with tab_repeat:
        st.header("Repeatability")
        st.write("Repeatability loops the **operating** horizon (slide 36): for every "
                 "vehicle and component the state at the end of the operating phase "
                 "must be no worse than the state entering it, i.e. "
                 "**state(T) <= state(H1)** for both mean and variance "
                 "(with a single horizon this is the classic mu(2H) <= mu(H)). "
                 "A margin is (operating-start - operating-end); it must be >= 0.")
        F, L, T = res["mu"].shape
        H1 = res["H1"]
        mu, v = res["mu"], res.get("v")
        rep_rows = []
        for i in range(F):
            for l in range(L):
                mu_H, mu_2H = float(mu[i, l, H1 - 1]), float(mu[i, l, T - 1])
                row = {
                    "vehicle": i, "component": l,
                    "mu(H)": mu_H, "mu(2H)": mu_2H,
                    "mu_margin": mu_H - mu_2H,
                    "mu_ok": bool(mu_2H <= mu_H + tol),
                }
                if v is not None:
                    v_H, v_2H = float(v[i, l, H1 - 1]), float(v[i, l, T - 1])
                    row.update({"v(H)": v_H, "v(2H)": v_2H,
                                "v_margin": v_H - v_2H,
                                "v_ok": bool(v_2H <= v_H + tol)})
                row["repeatable"] = bool(row["mu_ok"] and row.get("v_ok", True))
                rep_rows.append(row)
        rep_df = pd.DataFrame(rep_rows)

        n_bad = int((~rep_df["repeatable"]).sum())
        worst_mu = float(rep_df["mu_margin"].min())
        cols = st.columns(3)
        cols[0].metric("Vehicle-component pairs", len(rep_df))
        cols[1].metric("Repeatability violations", n_bad)
        cols[2].metric("Tightest mu margin", f"{worst_mu:.3e}")
        if n_bad == 0:
            st.success("Repeatability holds for every vehicle and component.")
        else:
            st.error(f"{n_bad} vehicle-component pair(s) violate repeatability.")

        st.dataframe(rep_df.sort_values(["mu_margin"]).reset_index(drop=True),
                     width="stretch", hide_index=True)

        # visual: first vs repeated end-of-horizon state, per pair
        st.subheader("First vs repeated horizon end state")
        labels = [f"V{r.vehicle}\u00b7C{r.component}" for r in rep_df.itertuples()]
        xpos = np.arange(len(rep_df))
        ncols = 2 if v is not None else 1
        fig, axes = plt.subplots(1, ncols, figsize=(max(len(rep_df) * 0.6, 6), 4),
                                 squeeze=False)
        axm = axes[0, 0]
        axm.bar(xpos - 0.2, rep_df["mu(H)"], width=0.4, label="mu(H1)  [operating start]")
        axm.bar(xpos + 0.2, rep_df["mu(2H)"], width=0.4, label="mu(T)  [operating end]")
        axm.set_xticks(xpos); axm.set_xticklabels(labels, rotation=90, fontsize=7)
        axm.set_title("mean: mu(H1) vs mu(T)"); axm.legend(); axm.grid(alpha=0.3)
        # mark violations in red
        for idx in range(len(rep_df)):
            if not bool(rep_df["mu_ok"].iloc[idx]):
                y = max(float(rep_df["mu(H)"].iloc[idx]),
                        float(rep_df["mu(2H)"].iloc[idx]))
                axm.annotate("!", (idx, y), color="red", ha="center",
                             va="bottom", fontweight="bold")
        if v is not None:
            axv = axes[0, 1]
            axv.bar(xpos - 0.2, rep_df["v(H)"], width=0.4, label="v(H1)  [operating start]")
            axv.bar(xpos + 0.2, rep_df["v(2H)"], width=0.4, label="v(T)  [operating end]")
            axv.set_xticks(xpos); axv.set_xticklabels(labels, rotation=90, fontsize=7)
            axv.set_title("variance: v(H1) vs v(T)"); axv.legend(); axv.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        st.caption("A pair is repeatable when the operating-end bar (T) does not "
                   "exceed the operating-start bar (H1) for both mean and variance.")

    # ---- Vehicle timeline ----
    with tab_vehicle:
        st.header("Vehicle damage timeline")
        F, L, T = res["mu"].shape
        H1 = res["H1"]
        mu, v = res["mu"], res.get("v")
        mu_0, v_0 = res["mu_0"], res.get("v_0")
        i = st.selectbox("Vehicle", range(F))
        act_i = state["act"][i]                       # (2H,) activity for this vehicle

        # --- mean trajectory ---
        fig_mu, ax = plt.subplots(figsize=(max(T * 0.5, 6), 3.6))
        for l in range(L):
            ax.plot(range(T), mu[i, l, :], marker="o", label=f"component {l}")
        ax.axhline(tau, linestyle="--", color="red", label=f"tau = {tau:.3f}")
        ax.axvline(H1 - 0.5, color="gray", linestyle=":", label="transitory | operating")
        # mark maintenance steps
        for k in range(T):
            if act_i[k] == 0:
                ax.axvspan(k - 0.5, k + 0.5, color="tab:blue", alpha=0.10)
        ax.set_xlabel("time step k"); ax.set_ylabel("mean E[D]")
        ax.set_title(f"Vehicle {i} - mean accumulated damage"
                     "  (shaded = maintenance step)")
        ax.grid(alpha=0.3); ax.legend(loc="best")
        st.pyplot(fig_mu)

        # --- variance trajectory (new) ---
        if v is not None:
            fig_v, axv = plt.subplots(figsize=(max(T * 0.5, 6), 3.6))
            for l in range(L):
                axv.plot(range(T), v[i, l, :], marker="s", label=f"component {l}")
            axv.axvline(H1 - 0.5, color="gray", linestyle=":", label="transitory | operating")
            for k in range(T):
                if act_i[k] == 0:
                    axv.axvspan(k - 0.5, k + 0.5, color="tab:blue", alpha=0.10)
            axv.set_xlabel("time step k"); axv.set_ylabel("variance Var[D]")
            axv.set_title(f"Vehicle {i} - variance of accumulated damage")
            axv.grid(alpha=0.3); axv.legend(loc="best")
            st.pyplot(fig_v)
        else:
            st.info("Output file has no variance array 'v'; variance plot skipped.")

        # --- exact mean/variance before & after each maintenance/repair step ---
        st.subheader("State before & after each maintenance / repair step")
        st.write("For every maintenance step (activity = 0) of this vehicle: the "
                 "component state at the previous step (**before**) and at the "
                 "maintenance step (**after**). Maintenance removes a fraction xi, "
                 "so the *intended* post-state is `before * (1 - xi)`.")
        xi = state.get("xi")
        ba_rows = []
        for k in range(T):
            if act_i[k] != 0:                          # only maintenance/repair steps
                continue
            for l in range(L):
                xil = float(xi[i, l]) if xi is not None else np.nan
                mu_before = float(mu_0[i, l]) if k == 0 else float(mu[i, l, k - 1])
                mu_after = float(mu[i, l, k])
                row = {
                    "time_step": k, "component": l,
                    "mu_before": mu_before,
                    "mu_after_reported": mu_after,
                    "mu_after_intended": mu_before * (1.0 - xil),
                    "mu_removed_intended": mu_before * xil,
                }
                if v is not None:
                    v_before = (float(v_0[i, l]) if (k == 0 and v_0 is not None)
                                else float(v[i, l, k - 1]) if k > 0 else np.nan)
                    v_after = float(v[i, l, k])
                    row.update({
                        "v_before": v_before,
                        "v_after_reported": v_after,
                        "v_after_intended": (v_before * (1.0 - xil)
                                             if not np.isnan(v_before) else np.nan),
                        "v_removed_intended": (v_before * xil
                                               if not np.isnan(v_before) else np.nan),
                    })
                ba_rows.append(row)
        if ba_rows:
            fmt = st.column_config.NumberColumn(format="%.5f")
            st.dataframe(
                pd.DataFrame(ba_rows), width="stretch", hide_index=True,
                column_config={c: fmt for c in (
                    "mu_before", "mu_after_reported", "mu_after_intended",
                    "mu_removed_intended", "v_before", "v_after_reported",
                    "v_after_intended", "v_removed_intended")})
            boundary = {res["H1"] - 1, T - 1}
            if any(r["time_step"] in boundary for r in ba_rows):
                st.caption(
                    "At the operating-loop boundary steps k = H1-1 and k = T-1, "
                    "`mu_after_reported` / `v_after_reported` can exceed the intended "
                    "post-maintenance value: the objective's periodicity term "
                    "C_P*(state(H1) - state(T)) lifts the reported state variable "
                    "there. It remains a valid upper bound; the *intended* columns "
                    "show the physical effect of maintenance (removing fraction xi).")
        else:
            st.info("This vehicle has no maintenance step in the horizon.")

    # ---- Heatmap ----
    with tab_heatmap:
        st.header("Schedule + damage heatmap")
        fig, n_viol = _schedule_heatmap(res, rel, tol)
        st.pyplot(fig)
        st.caption("Symbols: gear = maintenance, zzz = idle, number = mission j, "
                   "warning badge = reliability violation. Colour = E[D] on 0..tau. "
                   "Thick line marks the first / repeated horizon boundary.")
        if n_viol:
            st.warning(f"{n_viol} cell(s) violate P(D>tau)<=eps under '{method_used}'.")
        else:
            st.success("No reliability violations in the schedule.")

    # ---- Component comparison ----
    with tab_component:
        st.header("Component comparison")
        F, L, T = res["mu"].shape
        l = st.selectbox("Component", range(L))
        fig, ax = plt.subplots(figsize=(max(T * 0.5, 6), 4))
        for i in range(F):
            ax.plot(range(T), res["mu"][i, l, :], marker="o", label=f"vehicle {i}")
        ax.axhline(tau, linestyle="--", color="red", label=f"tau = {tau:.3f}")
        ax.set_xlabel("time step k"); ax.set_ylabel("E[D]")
        ax.set_title(f"Component {l} across vehicles"); ax.grid(alpha=0.3)
        ax.legend(loc="best")
        st.pyplot(fig)
        st.subheader("Per-vehicle summary for this component")
        rowsum = (df[df["component"] == l]
                  .groupby("vehicle")
                  .agg(mean_damage=("damage_mu", "mean"),
                       max_damage=("damage_mu", "max"),
                       min_rel_margin=("reliability_margin", "min"))
                  .reset_index())
        st.dataframe(rowsum, width="stretch", hide_index=True)

    # ---- Validation ----
    with tab_validation:
        st.header("Structural validation")
        st.write("Recomputed directly from the output (no log file required).")
        cdf = pd.DataFrame(state["checks"])
        st.dataframe(cdf, width="stretch", hide_index=True)
        if bool(cdf["passed"].all()):
            st.success("All structural checks passed.")
        else:
            st.error("Some structural checks failed (see 'violation' column).")
        st.caption("Note: 'capacity_sum_mu_le_F_minus_M' is the inherited "
                   "aggregate cap sum(mu) <= F-M; if you rescaled or removed it in "
                   "the solver, ignore this row.")

    # ---- Raw data ----
    with tab_raw:
        st.header("Raw state dataframe")
        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button("Download as CSV",
                           df.to_csv(index=False).encode("utf-8"),
                           file_name="rainflow_state.csv", mime="text/csv")


if __name__ == "__main__":
    # allow `streamlit run dashboard_rainflow.py` directly for quick testing
    render_rainflow_inspector_dashboard()