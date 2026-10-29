# Gamma diagnostic dashboard for fleet_management
# author: Christoph Langenauer
# purpose: Streamlit UI for synthetic Gamma-process diagnostics

from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from fleet_management.validation.validator import (
    build_gamma_diagnostic_dataframe,
    validate_gamma_synthetic_diagnostic,
)


def render_gamma_diagnostic_dashboard() -> None:
    """
    Render the Streamlit dashboard for the Gamma synthetic diagnostic.

    This dashboard is intentionally separated from validator_dashboard.py so that
    the main Gaussian/baseline dashboard stays clean. It reads a Gamma input YAML
    with a synthetic schedule x, propagates the Gamma shape state, checks

        P(D > tau) <= epsilon

    and checks the shared-rate Gamma loop condition

        A_2H <= A_H.
    """

    st.title("Gamma synthetic diagnostic dashboard")

    st.write(
        "This view visualizes the current Gamma-process diagnostic prototype. "
        "It reads a Gamma input YAML with a synthetic schedule `x`, propagates "
        "the Gamma shape state, checks the chance constraint "
        "`P(D > tau) <= epsilon`, and checks the loop condition `A_2H <= A_H`."
    )

    _render_gamma_sidebar()

    if not st.session_state.get("gamma_loaded", False):
        st.info("Enter a Gamma input YAML path and click **Load Gamma diagnostic**.")
        return

    gamma_input_path = st.session_state["gamma_input_path"]
    gamma_log_path = st.session_state["gamma_log_path"]
    gamma_tol = st.session_state["gamma_tol"]

    _render_gamma_configuration(
        gamma_input_path=gamma_input_path,
        gamma_log_path=gamma_log_path,
    )

    try:
        gamma_df = build_gamma_diagnostic_dataframe(
            input_path=gamma_input_path,
            tol=gamma_tol,
        )

        gamma_report = validate_gamma_synthetic_diagnostic(
            input_path=gamma_input_path,
            log_path=gamma_log_path,
            tol=gamma_tol,
        )

    except Exception as exc:
        st.error("Failed to run Gamma diagnostic.")
        st.exception(exc)
        return

    tab_overview, tab_timeline, tab_heatmap, tab_raw = st.tabs(
        [
            "Overview",
            "Shape / mean timeline",
            "Fleet heatmap",
            "Raw data",
        ]
    )

    with tab_overview:
        _render_gamma_overview(
            gamma_df=gamma_df,
            gamma_report=gamma_report,
        )

    with tab_timeline:
        _render_gamma_shape_mean_timeline(gamma_df)

    # with tab_probability:                                             # obsolete 08.07.2026
    #     _render_gamma_failure_probability(gamma_df)

    with tab_heatmap:
        _render_gamma_fleet_heatmap(gamma_df)

    with tab_raw:
        _render_gamma_raw_data(gamma_df)


def _render_gamma_sidebar() -> None:
    # Render sidebar controls for the Gamma diagnostic dashboard.

    st.sidebar.header("Gamma input")

    gamma_input_path = st.sidebar.text_input(
        "Gamma input YAML",
        value=st.session_state.get(
            "gamma_input_path",
            "input/tiny_gamma_synthetic_replacement.yaml",
        ),
    )

    gamma_log_path = st.sidebar.text_input(
        "Gamma log output path",
        value=st.session_state.get(
            "gamma_log_path",
            "results/gamma_synthetic_dashboard.log",
        ),
    )

    gamma_tol = st.sidebar.number_input(
        "Tolerance",
        min_value=0.0,
        value=float(st.session_state.get("gamma_tol", 1e-6)),
        format="%.1e",
        key="gamma_tol_input",
    )

    load_gamma = st.sidebar.button("Load Gamma diagnostic")

    if load_gamma:
        st.session_state["gamma_loaded"] = True
        st.session_state["gamma_input_path"] = gamma_input_path
        st.session_state["gamma_log_path"] = gamma_log_path
        st.session_state["gamma_tol"] = gamma_tol


def _render_gamma_configuration(
    gamma_input_path: str,
    gamma_log_path: str,
) -> None:
    # Render selected Gamma input/log paths.

    st.header("Configuration")

    col_input, col_log = st.columns(2)

    with col_input:
        st.subheader("Input file")
        st.code(gamma_input_path)
        st.write(f"Exists: {'✅' if Path(gamma_input_path).exists() else '❌'}")

    with col_log:
        st.subheader("Log file")
        st.code(gamma_log_path)
        st.write("Will be created/overwritten when the diagnostic runs.")


def _render_gamma_overview(
    gamma_df,
    gamma_report: dict,
) -> None:
    # Render overview metrics and critical rows.

    st.header("Gamma diagnostic overview")

    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)

    with metric_col_1:
        st.metric("Rows checked", gamma_report["rows_checked"])

    with metric_col_2:
        st.metric("Reliability failures", gamma_report["reliability_failures"])

    with metric_col_3:
        st.metric("Loop passed", "Yes" if gamma_report["loop_passed"] else "No")

    with metric_col_4:
        st.metric("Overall passed", "Yes" if gamma_report["passed"] else "No")

    metric_col_5, metric_col_6, metric_col_7 = st.columns(3)

    with metric_col_5:
        st.metric(
            "Max failure probability",
            f"{100.0 * gamma_report['max_failure_probability']:.4f}%",
        )

    with metric_col_6:
        st.metric(
            "Max shape after",
            f"{gamma_report['max_shape_after']:.4f}",
        )

    with metric_col_7:
        st.metric(
            "Threshold tau",
            f"{gamma_report['tau']:.4f}",
        )

    st.subheader("Worst failure-probability location")
    st.json(gamma_report["max_failure_probability_location"])

    st.subheader("Reliability interpretation")

    st.latex(
        r"""
        D_{i\ell k} \sim \mathrm{Gamma}(A_{i\ell k}, \beta_\ell),
        \qquad
        \mathbb{E}[D_{i\ell k}] = \frac{A_{i\ell k}}{\beta_\ell}
        """
    )

    st.latex(
        r"""
        \Pr(D_{i\ell k} > \tau) \leq \varepsilon
        """
    )

    st.latex(
        r"""
        A_{i\ell,2H} \leq A_{i\ell,H}
        """
    )

    st.caption(
        "The current Gamma diagnostic uses the shape-rate convention. "
        "The input YAML stores expected damage increments `mu`, which are "
        "converted to Gamma shape increments by `A_increment = beta * mu`."
    )

    st.subheader("Critical Gamma rows")

    critical_gamma_columns = [
        "time_step",
        "input_day",
        "vehicle",
        "activity",
        "mission",
        "component",
        "beta",
        "shape_after",
        "mean_after",
        "failure_probability_percent",
        "threshold_utilization_percent",
        "reliability_passed",
        "assignment_feasible",
        "status",
    ]

    available_critical_columns = [
        col for col in critical_gamma_columns if col in gamma_df.columns
    ]

    critical_gamma_df = (
        gamma_df.sort_values(
            ["failure_probability_after", "threshold_utilization_percent"],
            ascending=False,
        )
        .head(20)[available_critical_columns]
        .reset_index(drop=True)
    )

    st.dataframe(
        critical_gamma_df,
        width="stretch",
        hide_index=True,
        column_config={
            "failure_probability_percent": st.column_config.NumberColumn(
                "failure probability",
                format="%.4f%%",
            ),
            "threshold_utilization_percent": st.column_config.NumberColumn(
                "threshold utilization",
                format="%.2f%%",
            ),
        },
    )


def _render_gamma_shape_mean_timeline(gamma_df) -> None:
    # Render Gamma shape and expected damage for one vehicle/component.

    st.header("Gamma shape and expected damage timeline")

    st.write(
        "Select one vehicle and component. The plot shows the accumulated "
        "Gamma shape `A` and the corresponding expected damage `E[D] = A / beta`."
    )

    vehicles = sorted(gamma_df["vehicle"].unique().tolist())
    components = sorted(gamma_df["component"].unique().tolist())

    col_select_1, col_select_2 = st.columns(2)

    with col_select_1:
        selected_vehicle = st.selectbox(
            "Vehicle",
            options=vehicles,
            format_func=lambda i: f"Vehicle {int(i)}",
            key="gamma_timeline_vehicle",
        )

    with col_select_2:
        selected_component = st.selectbox(
            "Component",
            options=components,
            format_func=lambda l: f"Component {int(l)}",
            key="gamma_timeline_component",
        )

    selected_gamma_df = gamma_df[
        (gamma_df["vehicle"] == selected_vehicle)
        & (gamma_df["component"] == selected_component)
    ].sort_values("time_step")

    if selected_gamma_df.empty:
        st.info("No rows found for this vehicle/component selection.")
        return

    """fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        selected_gamma_df["time_step"],
        selected_gamma_df["shape_after"],
        marker="o",
        label="Gamma shape A",
    )

    ax.set_xlabel("Time step k")
    ax.set_ylabel("Gamma shape A")
    ax.set_xticks(selected_gamma_df["time_step"].astype(int).tolist())
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()

    ax2.plot(
        selected_gamma_df["time_step"],
        selected_gamma_df["mean_after"],
        marker="x",
        linestyle="--",
        label="Expected damage E[D]",
    )

    ax2.axhline(
        selected_gamma_df["threshold"].iloc[0],
        linestyle=":",
        linewidth=1.5,
        label="Threshold tau",
    )

    ax2.set_ylabel("Expected damage / threshold")

    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    ax.set_title(
        f"Gamma trajectory for vehicle {int(selected_vehicle)}, "
        f"component {int(selected_component)}"
    )

    st.pyplot(fig)"""

    fig, (ax_shape, ax_damage, ax_prob) = plt.subplots(
        3, 1, figsize=(12, 10), sharex=True
    )

    time_steps = selected_gamma_df["time_step"]

    # Shape
    ax_shape.plot(
        time_steps,
        selected_gamma_df["shape_after"],
        marker="o",
        label="Gamma shape A",
    )
    ax_shape.set_ylabel("Shape A")
    ax_shape.grid(True, alpha=0.3)
    ax_shape.legend(loc="best")

    # Expected damage
    ax_damage.plot(
        time_steps,
        selected_gamma_df["mean_after"],
        marker="x",
        linestyle="--",
        label="Expected damage E[D]",
    )
    ax_damage.axhline(
        selected_gamma_df["tau"].iloc[0] if "tau" in selected_gamma_df.columns
        else selected_gamma_df["threshold"].iloc[0],
        linestyle=":",
        linewidth=1.8,
        label="Threshold tau",
    )
    ax_damage.set_ylabel("Expected damage")
    ax_damage.grid(True, alpha=0.3)
    ax_damage.legend(loc="best")

    # Failure probability
    ax_prob.plot(
        time_steps,
        100.0 * selected_gamma_df["failure_probability_after"],
        marker="s",
        label="Failure probability",
    )
    ax_prob.axhline(
        100.0 * selected_gamma_df["epsilon"].iloc[0],
        linestyle="--",
        linewidth=1.8,
        label="Epsilon",
    )
    ax_prob.set_xlabel("Time step k")
    ax_prob.set_ylabel("P(D > tau) [%]")
    ax_prob.set_xticks(time_steps.astype(int).tolist())
    ax_prob.grid(True, alpha=0.3)
    ax_prob.legend(loc="best")

    st.pyplot(fig)

    st.subheader("Timeline rows")

    timeline_columns = [
        "time_step",
        "input_day",
        "activity",
        "mission",
        "shape_before",
        "shape_increment",
        "shape_after",
        "mean_before",
        "mean_increment",
        "mean_after",
        "failure_probability_percent",
        "status",
    ]

    st.dataframe(
        selected_gamma_df[timeline_columns],
        width="stretch",
        hide_index=True,
        column_config={
            "failure_probability_percent": st.column_config.NumberColumn(
                "failure probability",
                format="%.4f%%",
            ),
        },
    )


def _render_gamma_fleet_heatmap(gamma_df) -> None:
    # Render a fleet-level Gamma diagnostic heatmap.

    st.header("Fleet-level Gamma heatmap")

    st.write(
        "This heatmap shows one Gamma diagnostic quantity across all vehicles "
        "and time steps for a selected component."
    )

    components = sorted(gamma_df["component"].unique().tolist())

    col_1, col_2 = st.columns(2)

    with col_1:
        selected_component = st.selectbox(
            "Component",
            options=components,
            format_func=lambda l: f"Component {int(l)}",
            key="gamma_heatmap_component",
        )

    with col_2:
        selected_quantity = st.selectbox(
            "Quantity",
            options=[
                "failure_probability_percent",
                "mean_after",
                "shape_after",
                "threshold_utilization_percent",
            ],
            format_func=lambda q: {
                "failure_probability_percent": "Failure probability [%]",
                "mean_after": "Expected damage E[D]",
                "shape_after": "Gamma shape A",
                "threshold_utilization_percent": "Threshold utilization [%]",
            }[q],
            key="gamma_heatmap_quantity",
        )

    component_df = gamma_df[
        gamma_df["component"] == selected_component
    ].copy()

    if component_df.empty:
        st.info("No rows found for this component.")
        return

    heatmap_data = component_df.pivot(
        index="vehicle",
        columns="time_step",
        values=selected_quantity,
    ).sort_index()

    fig, ax = plt.subplots(figsize=(12, 4.8))

    image = ax.imshow(
        heatmap_data.values,
        aspect="auto",
        origin="upper",
    )

    ax.set_title(
        f"{selected_quantity} for component {int(selected_component)}"
    )
    ax.set_xlabel("Time step k")
    ax.set_ylabel("Vehicle i")

    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels([int(k) for k in heatmap_data.columns])

    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels([int(i) for i in heatmap_data.index])

    colorbar = fig.colorbar(image, ax=ax)

    if selected_quantity == "failure_probability_percent":
        colorbar.set_label("Failure probability [%]")
    elif selected_quantity == "threshold_utilization_percent":
        colorbar.set_label("Threshold utilization [%]")
    elif selected_quantity == "mean_after":
        colorbar.set_label("Expected damage E[D]")
    elif selected_quantity == "shape_after":
        colorbar.set_label("Gamma shape A")

    # Annotate cells with rounded values and activity labels.
    for row_idx, vehicle in enumerate(heatmap_data.index):
        for col_idx, time_step in enumerate(heatmap_data.columns):
            row = component_df[
                (component_df["vehicle"] == vehicle)
                & (component_df["time_step"] == time_step)
            ]

            if row.empty:
                continue

            row = row.iloc[0]
            value = row[selected_quantity]

            if row["activity"] == "replacement":
                activity_label = "R"
            elif row["activity"] == "maintenance_or_idle":
                activity_label = "I"
            elif row["activity"] == "mission":
                activity_label = (
                    "M?"
                    if row["mission"] is None
                    else f"M{int(row['mission'])}"
                )
            else:
                activity_label = str(row["activity"])

            if selected_quantity in [
                "failure_probability_percent",
                "threshold_utilization_percent",
            ]:
                value_label = f"{value:.2f}%"
            else:
                value_label = f"{value:.2f}"

            ax.text(
                col_idx,
                row_idx,
                f"{value_label}\n{activity_label}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    fig.tight_layout()
    st.pyplot(fig)

    if selected_quantity == "failure_probability_percent":
        epsilon_percent = 100.0 * float(component_df["epsilon"].iloc[0])
        max_probability = float(component_df[selected_quantity].max())

        if max_probability <= epsilon_percent:
            st.success(
                f"All entries for component {int(selected_component)} are below "
                f"epsilon = {epsilon_percent:.2f}%."
            )
        else:
            st.warning(
                f"At least one entry for component {int(selected_component)} exceeds "
                f"epsilon = {epsilon_percent:.2f}%."
            )

    st.caption(
        "Cell labels show the selected value and activity: "
        "`M#` = mission index, `R` = replacement, `I` = idle/no replacement."
    )


def _render_gamma_failure_probability(gamma_df) -> None:
    """Render failure probability over time for one vehicle/component."""

    st.header("Failure probability over the horizon")

    st.write(
        "This plot shows the Gamma chance-constraint quantity "
        "`P(D > tau)` over time. The horizontal line is epsilon."
    )

    vehicles = sorted(gamma_df["vehicle"].unique().tolist())
    components = sorted(gamma_df["component"].unique().tolist())

    col_prob_1, col_prob_2 = st.columns(2)

    with col_prob_1:
        selected_vehicle = st.selectbox(
            "Vehicle",
            options=vehicles,
            format_func=lambda i: f"Vehicle {int(i)}",
            key="gamma_probability_vehicle",
        )

    with col_prob_2:
        selected_component = st.selectbox(
            "Component",
            options=components,
            format_func=lambda l: f"Component {int(l)}",
            key="gamma_probability_component",
        )

    prob_df = gamma_df[
        (gamma_df["vehicle"] == selected_vehicle)
        & (gamma_df["component"] == selected_component)
    ].sort_values("time_step")

    if prob_df.empty:
        st.info("No rows found for this vehicle/component selection.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        prob_df["time_step"],
        100.0 * prob_df["failure_probability_after"],
        marker="o",
        label="Failure probability",
    )

    epsilon_percent = 100.0 * float(prob_df["epsilon"].iloc[0])

    ax.axhline(
        epsilon_percent,
        linestyle="--",
        linewidth=1.5,
        label=f"Epsilon = {epsilon_percent:.2f}%",
    )

    ax.set_title(
        f"Failure probability for vehicle {int(selected_vehicle)}, "
        f"component {int(selected_component)}"
    )
    ax.set_xlabel("Time step k")
    ax.set_ylabel("Failure probability [%]")
    ax.set_xticks(prob_df["time_step"].astype(int).tolist())
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    st.pyplot(fig)

    max_prob_row = prob_df.loc[prob_df["failure_probability_after"].idxmax()]

    st.info(
        "Maximum failure probability for this selection: "
        f"{100.0 * max_prob_row['failure_probability_after']:.4f}% "
        f"at k={int(max_prob_row['time_step'])}."
    )


def _render_gamma_raw_data(gamma_df) -> None:
    """Render the raw Gamma diagnostic dataframe."""

    st.header("Raw Gamma diagnostic dataframe")

    st.dataframe(
        gamma_df,
        width="stretch",
        hide_index=True,
        column_config={
            "failure_probability_percent": st.column_config.NumberColumn(
                "failure probability",
                format="%.4f%%",
            ),
            "threshold_utilization_percent": st.column_config.NumberColumn(
                "threshold utilization",
                format="%.2f%%",
            ),
        },
    )
