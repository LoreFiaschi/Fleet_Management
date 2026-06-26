"""
Streamlit dashboard for inspecting baseline assignment feasibility.

Run from the project root with:

    streamlit run scripts/validator_dashboard.py

The dashboard reads:
    - an input YAML file
    - a solver output YAML file

It then uses build_assignment_feasibility_dataframe(...) to create a
component-level table for interactive inspection.
"""

from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from fleet_management.validator import (
    build_assignment_feasibility_dataframe,
    validate_baseline_assignment_feasibility,
)


st.set_page_config(
    page_title="Fleet Management Validator Dashboard",
    layout="wide",
)


def _path_exists(path_str: str) -> bool:
    return Path(path_str).exists()


def _format_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


@st.cache_data
def cached_build_dataframe(
    input_path: str,
    results_path: str,
    tol: float,
    alpha_override: float | None,
    degradation_scale: float,
):
    return build_assignment_feasibility_dataframe(
        input_path=input_path,
        results_path=results_path,
        tol=tol,
        alpha_override=alpha_override,
        degradation_scale=degradation_scale,
    )

# caching data for easier handling of different plots
@st.cache_data
def cached_validate_and_write_log(
    input_path: str,
    results_path: str,
    log_path: str,
    tol: float,
    alpha_override: float | None,
    degradation_scale: float,
):
    return validate_baseline_assignment_feasibility(
        input_path=input_path,
        results_path=results_path,
        log_path=log_path,
        tol=tol,
        alpha_override=alpha_override,
        degradation_scale=degradation_scale,
    )

st.title("Fleet Management Validator Dashboard")
st.caption("Baseline assignment feasibility diagnostics for solver outputs")

with st.sidebar:
    st.header("Files")

    input_path = st.text_input(
        "Input YAML",
        value="input/data_test_baseline.yaml",
    )

    results_path = st.text_input(
        "Solver output YAML",
        value="results/output_baseline.yaml",
    )

    log_path = st.text_input(
        "Log output path",
        value="results/baseline_assignment_feasibility_dashboard.log",
    )

    st.header("Diagnostic settings")

    tol = st.number_input(
        "Tolerance",
        min_value=0.0,
        value=1e-6,
        format="%.1e",
    )

    use_alpha_override = st.checkbox("Override alpha", value=False)

    alpha_override = None
    if use_alpha_override:
        alpha_override = st.number_input(
            "Effective alpha",
            min_value=1e-12,
            value=1.0,
            format="%.6f",
        )

    degradation_scale = st.number_input(
        "Degradation scale",
        min_value=1e-12,
        value=1.0,
        format="%.6f",
    )

    load_button = st.button("Load data", type="primary")


st.subheader("Configuration")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.write("**Input file**")
    st.code(input_path)
    st.write("Exists:", "✅" if _path_exists(input_path) else "❌")

with col_b:
    st.write("**Results file**")
    st.code(results_path)
    st.write("Exists:", "✅" if _path_exists(results_path) else "❌")

with col_c:
    st.write("**Log file**")
    st.code(log_path)
    st.write("Will be created/overwritten when log is written.")


if load_button:
    if not _path_exists(input_path):
        st.error(f"Input file not found: {input_path}")
        st.stop()

    if not _path_exists(results_path):
        st.error(f"Results file not found: {results_path}")
        st.stop()

    try:
        st.session_state["df"] = cached_build_dataframe(
            input_path=input_path,
            results_path=results_path,
            tol=tol,
            alpha_override=alpha_override,
            degradation_scale=degradation_scale,
        )

        st.session_state["report"] = cached_validate_and_write_log(
            input_path=input_path,
            results_path=results_path,
            log_path=log_path,
            tol=tol,
            alpha_override=alpha_override,
            degradation_scale=degradation_scale,
        )

        st.session_state["loaded_paths"] = {
            "input_path": input_path,
            "results_path": results_path,
            "log_path": log_path,
            "tol": tol,
            "alpha_override": alpha_override,
            "degradation_scale": degradation_scale,
        }

    except Exception as exc:
        st.error("Failed to load validator data.")
        st.exception(exc)
        st.stop()

if "df" not in st.session_state:
    st.info("Set the file paths in the sidebar and click **Load data** once.")
    st.stop()

df = st.session_state["df"]
report = st.session_state.get("report")


if not _path_exists(input_path):
    st.error(f"Input file not found: {input_path}")
    st.stop()

if not _path_exists(results_path):
    st.error(f"Results file not found: {results_path}")
    st.stop()


tab_overview, tab_failed, tab_mission, tab_vehicle, tab_component, tab_data = st.tabs(
    [
        "Overview",
        "Failed / critical assignments",
        "Mission damage analysis",
        "Vehicle damage timeline",
        "Component comparison",
        "Raw data",
    ])


with tab_overview:
    st.header("Overview")

    if df.empty:
        st.warning("No active mission assignments were found.")
        st.stop()

    total_component_rows = len(df)

    assignment_df = df.drop_duplicates(
        subset=["time_step", "vehicle", "mission"]
    )

    total_assignments = len(assignment_df)
    feasible_assignments = int(assignment_df["assignment_feasible"].sum())
    infeasible_assignments = total_assignments - feasible_assignments

    failed_component_rows = int((~df["feasible"]).sum())

    maintenance_events = None
    if report is not None:
        maintenance_events = report.get("maintenance_events")

    max_damage_idx = df["damage_after"].idxmax()
    max_damage_row = df.loc[max_damage_idx]

    min_margin_idx = df["margin_to_threshold"].idxmin()
    min_margin_row = df.loc[min_margin_idx]

    average_utilization = float(df["utilization_of_threshold"].mean())
    max_utilization = float(df["utilization_of_threshold"].max())

    metric_cols = st.columns(5)

    metric_cols[0].metric(
        "Assignments checked",
        total_assignments,
    )

    metric_cols[1].metric(
        "Feasible assignments",
        feasible_assignments,
    )

    metric_cols[2].metric(
        "Infeasible assignments",
        infeasible_assignments,
    )

    metric_cols[3].metric(
        "Failed component rows",
        failed_component_rows,
    )

    metric_cols[4].metric(
        "Maintenance events",
        "n/a" if maintenance_events is None else maintenance_events,
    )

    st.divider()

    metric_cols_2 = st.columns(4)

    metric_cols_2[0].metric(
        "Max damage after assignment",
        _format_float(float(max_damage_row["damage_after"])),
    )

    metric_cols_2[1].metric(
        "Minimum margin to threshold",
        _format_float(float(min_margin_row["margin_to_threshold"])),
    )

    metric_cols_2[2].metric(
        "Average threshold utilization",
        f"{100.0 * average_utilization:.1f}%",
    )

    metric_cols_2[3].metric(
        "Max threshold utilization",
        f"{100.0 * max_utilization:.1f}%",
    )

    st.subheader("Worst damage location")

    worst_damage_table = pd.DataFrame(
        [
            {
                "time_step": int(max_damage_row["time_step"]),
                "input_day": int(max_damage_row["input_day"]),
                "vehicle": int(max_damage_row["vehicle"]),
                "mission": int(max_damage_row["mission"]),
                "component": int(max_damage_row["component"]),
                "damage_before": float(max_damage_row["damage_before"]),
                "expected_increment": float(max_damage_row["expected_increment"]),
                "damage_after": float(max_damage_row["damage_after"]),
                "threshold": float(max_damage_row["threshold"]),
                "margin_to_threshold": float(max_damage_row["margin_to_threshold"]),
                "status": str(max_damage_row["status"]),
            }
        ]
    )

    st.dataframe(
        worst_damage_table,
        width="stretch",
        hide_index=True,
    )

    st.subheader("Closest assignments to threshold")

    n_critical = st.slider(
        "Number of critical component rows to show",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
    )

    critical_columns = [
        "time_step",
        "input_day",
        "vehicle",
        "mission",
        "component",
        "damage_before",
        "expected_increment",
        "damage_after",
        "threshold",
        "margin_to_threshold",
        "threshold_utilization_percent",
        "status",
    ]

    display_df = df.copy()
    display_df["threshold_utilization_percent"] = (
        100.0 * display_df["utilization_of_threshold"]
    )

    critical_df = (
        display_df.sort_values("margin_to_threshold", ascending=True)
        .head(n_critical)[critical_columns]
        .reset_index(drop=True)
    )

    st.dataframe(
        critical_df,
        width="stretch",
        hide_index=True,
        column_config={
            "threshold_utilization_percent": st.column_config.NumberColumn(
                "threshold utilization",
                format="%.1f%%",
            ),
        },
    )

    st.subheader("Failed assignments")

    failed_df = display_df[~display_df["feasible"]].copy()

    if failed_df.empty:
        st.success("No component-level threshold violations found.")
    else:
        st.error(f"{len(failed_df)} component-level threshold violations found.")
        st.dataframe(
            failed_df[critical_columns],
            width="stretch",
            hide_index=True,
        )

    st.subheader("Log file")

    if report is not None:
        st.write(f"Log written to: `{report['log_path']}`")

        if Path(report["log_path"]).exists():
            with open(report["log_path"], "r") as f:
                log_text = f.read()

            with st.expander("Show generated log"):
                st.text(log_text)


with tab_failed:
    st.header("Failed / critical assignments")

    st.write(
        "This tab lists component-level threshold violations and near-threshold "
        "assignments. A row corresponds to one component of one active "
        "vehicle-mission-time assignment."
    )

    if df.empty:
        st.warning("No active mission assignments were found.")
        st.stop()

    # -----------------------------
    # Filters
    # -----------------------------
    st.subheader("Filters")

    filter_col_1, filter_col_2, filter_col_3, filter_col_4 = st.columns(4)

    with filter_col_1:
        vehicle_filter = st.multiselect(
            "Vehicle",
            options=sorted(df["vehicle"].unique()),
            default=sorted(df["vehicle"].unique()),
            format_func=lambda x: f"Vehicle {int(x)}",
        )

    with filter_col_2:
        mission_filter = st.multiselect(
            "Mission",
            options=sorted(df["mission"].unique()),
            default=sorted(df["mission"].unique()),
            format_func=lambda x: f"Mission {int(x)}",
        )

    with filter_col_3:
        component_filter = st.multiselect(
            "Component",
            options=sorted(df["component"].unique()),
            default=sorted(df["component"].unique()),
            format_func=lambda x: f"Component {int(x)}",
        )

    with filter_col_4:
        status_filter = st.multiselect(
            "Status",
            options=sorted(df["status"].unique()),
            default=sorted(df["status"].unique()),
        )

    filtered_df = df[
        df["vehicle"].isin(vehicle_filter)
        & df["mission"].isin(mission_filter)
        & df["component"].isin(component_filter)
        & df["status"].isin(status_filter)
    ].copy()

    if filtered_df.empty:
        st.info("No rows match the selected filters.")
        st.stop()

    # -----------------------------
    # Criticality threshold
    # -----------------------------
    st.subheader("Criticality definition")

    critical_mode = st.radio(
        "Define critical assignments by",
        options=[
            "Smallest margin to threshold",
            "Threshold utilization above limit",
        ],
        horizontal=True,
    )

    if critical_mode == "Smallest margin to threshold":
        n_critical = st.slider(
            "Number of most critical component rows",
            min_value=5,
            max_value=min(100, len(filtered_df)),
            value=min(20, len(filtered_df)),
            step=5,
        )
        critical_df = (
            filtered_df.sort_values("margin_to_threshold", ascending=True)
            .head(n_critical)
            .copy()
        )

    else:
        utilization_limit = st.slider(
            "Minimum threshold utilization",
            min_value=0.0,
            max_value=1.5,
            value=0.8,
            step=0.05,
        )
        critical_df = filtered_df[
            filtered_df["threshold_utilization_percent"] >= utilization_limit
        ].sort_values("threshold_utilization_percent", ascending=False)

    # -----------------------------
    # Summary metrics
    # -----------------------------
    failed_component_df = filtered_df[~filtered_df["feasible"]].copy()

    failed_assignment_df = (
        filtered_df[~filtered_df["assignment_feasible"]]
        .drop_duplicates(subset=["time_step", "vehicle", "mission"])
        .copy()
    )

    critical_assignment_df = (
        critical_df.drop_duplicates(subset=["time_step", "vehicle", "mission"])
        .copy()
    )

    metric_cols = st.columns(4)

    metric_cols[0].metric(
        "Filtered component rows",
        len(filtered_df),
    )

    metric_cols[1].metric(
        "Failed component rows",
        len(failed_component_df),
    )

    metric_cols[2].metric(
        "Failed assignments",
        len(failed_assignment_df),
    )

    metric_cols[3].metric(
        "Critical assignments",
        len(critical_assignment_df),
    )

    # -----------------------------
    # Failed component rows
    # -----------------------------
    st.divider()
    st.subheader("Failed component-level checks")

    display_columns = [
        "time_step",
        "input_day",
        "vehicle",
        "mission",
        "component",
        "damage_before",
        "expected_increment",
        "damage_after",
        "threshold",
        "margin_to_threshold",
        "utilization_of_threshold",
        "status",
    ]

    if failed_component_df.empty:
        st.success("No component-level threshold violations found.")
    else:
        st.error(
            f"{len(failed_component_df)} component-level threshold violations found."
        )
        st.dataframe(
            failed_component_df[display_columns]
            .sort_values("margin_to_threshold", ascending=True)
            .reset_index(drop=True),
            width="stretch",
            hide_index=True,
        )

    # -----------------------------
    # Failed assignment-level rows
    # -----------------------------
    st.subheader("Failed assignments grouped by vehicle / mission / time")

    if failed_assignment_df.empty:
        st.success("No assignment-level threshold violations found.")
    else:
        assignment_summary = (
            failed_component_df.groupby(["time_step", "input_day", "vehicle", "mission"])
            .agg(
                violating_components=("component", lambda x: list(map(int, x))),
                max_damage_after=("damage_after", "max"),
                min_margin_to_threshold=("margin_to_threshold", "min"),
                max_utilization=("utilization_of_threshold", "max"),
            )
            .reset_index()
            .sort_values("min_margin_to_threshold", ascending=True)
        )

        st.dataframe(
            assignment_summary,
            width="stretch",
            hide_index=True,
        )

    # -----------------------------
    # Critical rows
    # -----------------------------
    st.divider()
    st.subheader("Critical / near-threshold component rows")

    if critical_df.empty:
        st.info("No critical rows found for the selected definition.")
    else:
        st.dataframe(
            critical_df[display_columns]
            .sort_values("margin_to_threshold", ascending=True)
            .reset_index(drop=True),
            width="stretch",
            hide_index=True,
        )

    # -----------------------------
    # Critical assignment-level summary
    # -----------------------------
    st.subheader("Critical assignments grouped by vehicle / mission / time")

    if critical_df.empty:
        st.info("No critical assignments found.")
    else:
        critical_assignment_summary = (
            critical_df.groupby(["time_step", "input_day", "vehicle", "mission"])
            .agg(
                critical_components=("component", lambda x: list(map(int, x))),
                max_damage_after=("damage_after", "max"),
                min_margin_to_threshold=("margin_to_threshold", "min"),
                max_utilization=("utilization_of_threshold", "max"),
                total_expected_increment=("expected_increment", "sum"),
            )
            .reset_index()
            .sort_values("min_margin_to_threshold", ascending=True)
        )

        st.dataframe(
            critical_assignment_summary,
            width="stretch",
            hide_index=True,
        )

    # -----------------------------
    # Downloads
    # -----------------------------
    st.divider()
    st.subheader("Export")

    export_col_1, export_col_2 = st.columns(2)

    with export_col_1:
        failed_csv = failed_component_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download failed component rows as CSV",
            data=failed_csv,
            file_name="failed_component_rows.csv",
            mime="text/csv",
            disabled=failed_component_df.empty,
        )

    with export_col_2:
        critical_csv = critical_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download critical component rows as CSV",
            data=critical_csv,
            file_name="critical_component_rows.csv",
            mime="text/csv",
            disabled=critical_df.empty,
        )


with tab_mission:
    st.header("Mission damage analysis")

    st.write(
        "This tab shows the assignment-level dataframe by mission and component. "
        "It helps identify which missions consistently cause high degradation and "
        "which components are most affected."
    )

    if df.empty:
        st.warning("No active mission assignments were found.")
        st.stop()

    mission_display_df = df.copy()
    mission_display_df["threshold_utilization_percent"] = (
        100.0 * mission_display_df["utilization_of_threshold"]
    )

    # ---------------------------------------------------------------------
    # Mission/component summary
    # ---------------------------------------------------------------------
    st.subheader("Mission × component damage summary")

    mission_component_summary = (
        mission_display_df
        .groupby(["mission", "component"])
        .agg(
            assignments=("expected_increment", "count"),
            mean_expected_increment=("expected_increment", "mean"),
            max_expected_increment=("expected_increment", "max"),
            total_expected_increment=("expected_increment", "sum"),
            mean_damage_after=("damage_after", "mean"),
            max_damage_after=("damage_after", "max"),
            mean_threshold_utilization_percent=(
                "threshold_utilization_percent",
                "mean",
            ),
            max_threshold_utilization_percent=(
                "threshold_utilization_percent",
                "max",
            ),
            min_margin_to_threshold=("margin_to_threshold", "min"),
            failed_component_rows=("feasible", lambda x: int((~x).sum())),
        )
        .reset_index()
    )

    ranking_metric = st.selectbox(
        "Rank mission/component pairs by",
        options=[
            "mean_expected_increment",
            "max_expected_increment",
            "total_expected_increment",
            "max_damage_after",
            "max_threshold_utilization_percent",
            "min_margin_to_threshold",
            "failed_component_rows",
        ],
        index=0,
    )

    ascending = ranking_metric == "min_margin_to_threshold"

    mission_component_summary = mission_component_summary.sort_values(
        ranking_metric,
        ascending=ascending,
    ).reset_index(drop=True)

    percentage_column_config_mission = {
        "mean_threshold_utilization_percent": st.column_config.NumberColumn(
            "mean utilization",
            format="%.1f%%",
        ),
        "max_threshold_utilization_percent": st.column_config.NumberColumn(
            "max utilization",
            format="%.1f%%",
        ),
    }

    st.dataframe(
        mission_component_summary,
        width="stretch",
        hide_index=True,
        column_config=percentage_column_config_mission,
    )

    # ---------------------------------------------------------------------
    # Mission-only summary
    # ---------------------------------------------------------------------
    st.subheader("Mission-level summary")

    mission_summary = (
        mission_display_df
        .groupby("mission")
        .agg(
            component_rows=("expected_increment", "count"),
            mean_expected_increment=("expected_increment", "mean"),
            max_expected_increment=("expected_increment", "max"),
            total_expected_increment=("expected_increment", "sum"),
            mean_damage_after=("damage_after", "mean"),
            max_damage_after=("damage_after", "max"),
            mean_threshold_utilization_percent=(
                "threshold_utilization_percent",
                "mean",
            ),
            max_threshold_utilization_percent=(
                "threshold_utilization_percent",
                "max",
            ),
            min_margin_to_threshold=("margin_to_threshold", "min"),
            failed_component_rows=("feasible", lambda x: int((~x).sum())),
        )
        .reset_index()
        .sort_values("mean_expected_increment", ascending=False)
    )

    st.dataframe(
        mission_summary,
        width="stretch",
        hide_index=True,
        column_config=percentage_column_config_mission,
    )

    # ---------------------------------------------------------------------
    # Component-only summary
    # ---------------------------------------------------------------------
    st.subheader("Component-level summary")

    component_summary = (
        mission_display_df
        .groupby("component")
        .agg(
            component_rows=("expected_increment", "count"),
            mean_expected_increment=("expected_increment", "mean"),
            max_expected_increment=("expected_increment", "max"),
            total_expected_increment=("expected_increment", "sum"),
            mean_damage_after=("damage_after", "mean"),
            max_damage_after=("damage_after", "max"),
            mean_threshold_utilization_percent=(
                "threshold_utilization_percent",
                "mean",
            ),
            max_threshold_utilization_percent=(
                "threshold_utilization_percent",
                "max",
            ),
            min_margin_to_threshold=("margin_to_threshold", "min"),
            failed_component_rows=("feasible", lambda x: int((~x).sum())),
        )
        .reset_index()
        .sort_values("mean_expected_increment", ascending=False)
    )

    st.dataframe(
        component_summary,
        width="stretch",
        hide_index=True,
        column_config=percentage_column_config_mission,
    )

    # ---------------------------------------------------------------------
    # Worst single mission increments
    # ---------------------------------------------------------------------
    st.divider()
    st.subheader("Worst single mission increments")

    n_worst = st.slider(
        "Number of largest single increments to show",
        min_value=5,
        max_value=min(100, len(mission_display_df)),
        value=min(20, len(mission_display_df)),
        step=5,
    )

    worst_increment_columns = [
        "time_step",
        "input_day",
        "vehicle",
        "mission",
        "component",
        "damage_before",
        "expected_increment",
        "damage_after",
        "threshold",
        "margin_to_threshold",
        "threshold_utilization_percent",
        "status",
    ]

    worst_increments = (
        mission_display_df
        .sort_values("expected_increment", ascending=False)
        .head(n_worst)[worst_increment_columns]
        .reset_index(drop=True)
    )

    st.dataframe(
        worst_increments,
        width="stretch",
        hide_index=True,
        column_config={
            "threshold_utilization_percent": st.column_config.NumberColumn(
                "threshold utilization",
                format="%.1f%%",
            ),
        },
    )

    # ---------------------------------------------------------------------
    # Worst mission for each component
    # ---------------------------------------------------------------------
    st.subheader("Most damaging mission for each component")

    idx = (
        mission_component_summary
        .groupby("component")["mean_expected_increment"]
        .idxmax()
    )

    worst_mission_per_component = (
        mission_component_summary
        .loc[idx]
        .sort_values("component")
        .reset_index(drop=True)
    )

    st.dataframe(
        worst_mission_per_component,
        width="stretch",
        hide_index=True,
        column_config=percentage_column_config_mission,
    )

    # ---------------------------------------------------------------------
    # Simple bar plots
    # ---------------------------------------------------------------------
    st.divider()
    st.subheader("Mission damage plots")

    plot_col_1, plot_col_2 = st.columns(2)

    with plot_col_1:
        st.write("**Mean expected increment by mission/component**")

        pivot_mean = mission_component_summary.pivot(
            index="mission",
            columns="component",
            values="mean_expected_increment",
        )

        st.bar_chart(pivot_mean)

    with plot_col_2:
        st.write("**Max threshold utilization by mission/component**")

        pivot_util = mission_component_summary.pivot(
            index="mission",
            columns="component",
            values="max_threshold_utilization_percent",
        )

        st.bar_chart(pivot_util)

    # ---------------------------------------------------------------------
    # Export
    # ---------------------------------------------------------------------
    st.divider()
    st.subheader("Export")

    export_col_1, export_col_2 = st.columns(2)

    with export_col_1:
        mission_component_csv = mission_component_summary.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download mission/component summary as CSV",
            data=mission_component_csv,
            file_name="mission_component_damage_summary.csv",
            mime="text/csv",
        )

    with export_col_2:
        mission_summary_csv = mission_summary.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download mission summary as CSV",
            data=mission_summary_csv,
            file_name="mission_damage_summary.csv",
            mime="text/csv",
        )


with tab_vehicle:
    st.header("Vehicle damage timeline")

    st.write(
        "Select one vehicle to inspect the solver-reported damage trajectory "
        "`mu_result[i,l,k]` for all components over the full `2H` horizon."
    )

    # Read result data directly to access the full solver trajectory.
    try:
        from fleet_management.validator import _read_results, _extract_results_parameters

        results_data = _read_results(Path(results_path))
        results_params = _extract_results_parameters(results_data)

        mu_result = results_params["mu"]   # shape: F x L x 2H
        x = results_params["x"]            # shape: F x (M+1) x 2H
        alpha = float(results_params["alpha"])

        F = int(results_params["F"])
        L = int(results_params["L"])
        H = int(results_params["H"])
        horizon = 2 * H

    except Exception as exc:
        st.error("Could not read solver trajectory from results file.")
        st.exception(exc)
        st.stop()

    loaded_paths = st.session_state.get("loaded_paths", {})
    loaded_alpha_override = loaded_paths.get("alpha_override", None)

    effective_alpha = (
        float(loaded_alpha_override)
        if loaded_alpha_override is not None
        else alpha
    )

    # force pick the worst conditioned vehicle
    vehicle_violation_summary = (
        df[df["damage_after"] > effective_alpha + tol]
        .groupby("vehicle")
        .agg(
            violation_count=("damage_after", "count"),
            max_violation_damage=("damage_after", "max"),
        )
        .reset_index()
        .sort_values(["violation_count", "max_violation_damage"], ascending=False)
    )

    if vehicle_violation_summary.empty:
        st.info(
            f"No vehicle has validator-computed damage above effective alpha "
            f"{effective_alpha:.3f}."
        )
        default_vehicle = 0
    else:
        worst_vehicle = int(vehicle_violation_summary.iloc[0]["vehicle"])
        st.warning(
            f"Worst violating vehicle under effective alpha {effective_alpha:.3f}: "
            f"Vehicle {worst_vehicle}"
        )

        st.dataframe(
            vehicle_violation_summary,
            width="stretch",
            hide_index=True,
        )

        default_vehicle = worst_vehicle

    vehicle_options = list(range(F))

    selected_vehicle = st.selectbox(
        "Vehicle",
        options=vehicle_options,
        index=vehicle_options.index(default_vehicle),
        format_func=lambda i: f"Vehicle {i}",
    )

    show_expected_damage = st.checkbox(
        "Show validator expected post-mission damage",
        value=True,
    )

    show_only_expected_violations = st.checkbox(
        "Only show expected values above effective alpha",
        value=False,
        disabled=not show_expected_damage,
    )

    show_assignment_markers = st.checkbox(
        "Show mission / maintenance markers",
        value=True,
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    time_steps = list(range(horizon))
    plt.xticks(range(0, len(time_steps) + 1))

    for l in range(L):
        ax.plot(
            time_steps,
            mu_result[selected_vehicle, l, :],
            marker="o",
            label=f"Component {l}",
        )

    ax.axhline(
        alpha,
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold α = {alpha:.3f}",
    )

    if loaded_alpha_override is not None:
        ax.axhline(
            float(loaded_alpha_override),
            linestyle=":",
            linewidth=2.0,
            label=f"Override α = {float(loaded_alpha_override):.3f}",
        )

    if show_assignment_markers:
        y_min = float(mu_result[selected_vehicle, :, :].min())

        y_max_candidates = [
            float(mu_result[selected_vehicle, :, :].max()),
            alpha,
        ]

        if loaded_alpha_override is not None:
            y_max_candidates.append(float(loaded_alpha_override))

        y_max = max(y_max_candidates)

        y_range = max(y_max - y_min, 1e-6)

        marker_y = y_min - 0.08 * y_range

        for k in range(horizon):
            action_indices = [
                j for j in range(x.shape[1])
                if abs(x[selected_vehicle, j, k] - 1.0) <= tol
            ]

            if not action_indices:
                continue

            action = action_indices[0]

            if action == 0:
                label = "M"
            else:
                label = f"{action - 1}"

            ax.text(
                k,
                marker_y,
                label,
                ha="center",
                va="center",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", alpha=0.2),
            )

        ax.set_ylim(marker_y - 0.08 * y_range, y_max + 0.12 * y_range)

    ax.set_title(f"Damage trajectory for vehicle {selected_vehicle}")
    ax.set_xlabel("Time step k")
    ax.set_ylabel("Damage / μ value")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    st.pyplot(fig)

    caption_text = (
        "Markers: `M` = maintenance, numbers = mission index. "
        "The plotted lines are the solver-reported component damage trajectories."
    )

    if loaded_alpha_override is not None:
        caption_text += (
            " The dotted horizontal line shows the temporary override alpha used "
            "for validator stress-testing."
        )

    st.caption(caption_text)

    st.subheader("Largest jumps for selected vehicle")

    vehicle_df = df[df["vehicle"] == selected_vehicle].copy()

    if vehicle_df.empty:
        st.info("No active mission assignments found for this vehicle.")
    else:
        jump_columns = [
            "time_step",
            "input_day",
            "mission",
            "component",
            "damage_before",
            "expected_increment",
            "damage_after",
            "threshold",
            "margin_to_threshold",
            "status",
        ]

        largest_jumps = (
            vehicle_df.sort_values("expected_increment", ascending=False)
            .head(10)[jump_columns]
            .reset_index(drop=True)
        )

        st.dataframe(
            largest_jumps,
            width="stretch",
            hide_index=True,
        )


with tab_component:
    st.header("Component comparison across vehicles")

    st.write(
        "Select one component to compare the solver-reported damage trajectory "
        "across all vehicles over the full horizon. This is useful for comparing "
        "the same component type, for example all batteries of all vehicles."
    )

    try:
        from fleet_management.validator import _read_results, _extract_results_parameters

        results_data = _read_results(Path(results_path))
        results_params = _extract_results_parameters(results_data)

        mu_result = results_params["mu"]   # shape: F x L x 2H
        x = results_params["x"]            # shape: F x (M+1) x 2H
        alpha = float(results_params["alpha"])

        F = int(results_params["F"])
        L = int(results_params["L"])
        H = int(results_params["H"])
        horizon = 2 * H

    except Exception as exc:
        st.error("Could not read solver trajectory from results file.")
        st.exception(exc)
        st.stop()

    loaded_paths = st.session_state.get("loaded_paths", {})
    loaded_alpha_override = loaded_paths.get("alpha_override", None)

    effective_alpha = (
        float(loaded_alpha_override)
        if loaded_alpha_override is not None
        else alpha
    )

    st.subheader("Filters")

    filter_col_1, filter_col_2, filter_col_3 = st.columns(3)

    with filter_col_1:
        selected_component = st.selectbox(
            "Component",
            options=list(range(L)),
            format_func=lambda l: f"Component {l}",
        )

    with filter_col_2:
        vehicle_filter = st.multiselect(
            "Vehicles",
            options=list(range(F)),
            default=list(range(F)),
            format_func=lambda i: f"Vehicle {i}",
        )

    with filter_col_3:
        show_effective_alpha = st.checkbox(
            "Show effective alpha",
            value=True,
        )

    show_mean = st.checkbox(
        "Show fleet mean trajectory",
        value=True,
    )

    show_max = st.checkbox(
        "Show fleet max trajectory",
        value=False,
    )

    show_assignment_markers = st.checkbox(
        "Show maintenance markers",
        value=False,
    )

    if not vehicle_filter:
        st.info("Select at least one vehicle.")
        st.stop()

    time_steps = list(range(horizon))

    fig, ax = plt.subplots(figsize=(12, 5))

    # Individual vehicle trajectories
    for i in vehicle_filter:
        ax.plot(
            time_steps,
            mu_result[i, selected_component, :],
            marker="o",
            linewidth=1.5,
            label=f"Vehicle {i}",
        )

    selected_component_matrix = mu_result[
        vehicle_filter,
        selected_component,
        :
    ]

    if show_mean:
        mean_trajectory = selected_component_matrix.mean(axis=0)
        ax.plot(
            time_steps,
            mean_trajectory,
            linestyle="--",
            linewidth=2.5,
            label="Fleet mean",
        )

    if show_max:
        max_trajectory = selected_component_matrix.max(axis=0)
        ax.plot(
            time_steps,
            max_trajectory,
            linestyle=":",
            linewidth=2.5,
            label="Fleet max",
        )

    ax.axhline(
        alpha,
        linestyle="--",
        linewidth=1.5,
        label=f"Original threshold α = {alpha:.3f}",
    )

    if show_effective_alpha and loaded_alpha_override is not None:
        ax.axhline(
            effective_alpha,
            linestyle=":",
            linewidth=2.0,
            label=f"Override α = {effective_alpha:.3f}",
        )

    if show_assignment_markers:
        y_min = float(selected_component_matrix.min())
        y_max_candidates = [
            float(selected_component_matrix.max()),
            alpha,
            effective_alpha,
        ]
        y_max = max(y_max_candidates)
        y_range = max(y_max - y_min, 1e-6)

        marker_y_base = y_min - 0.08 * y_range

        for idx, i in enumerate(vehicle_filter):
            marker_y = marker_y_base - idx * 0.04 * y_range

            for k in range(horizon):
                if abs(x[i, 0, k] - 1.0) <= tol:
                    ax.text(
                        k,
                        marker_y,
                        "M",
                        ha="center",
                        va="center",
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", alpha=0.2),
                    )

        ax.set_ylim(
            marker_y_base - len(vehicle_filter) * 0.05 * y_range,
            y_max + 0.12 * y_range,
        )

    ax.set_title(
        f"Component {selected_component} damage trajectory across vehicles"
    )
    ax.set_xlabel("Time step k")
    ax.set_ylabel("Damage / μ value")
    ax.set_xticks(range(horizon))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    st.pyplot(fig)

    st.caption(
        "Each solid line is the solver-reported damage trajectory of the selected "
        "component for one vehicle. Optional dashed/dotted lines show fleet mean, "
        "fleet max, and threshold values."
    )

    # ------------------------------------------------------------------
    # Summary table for selected component
    # ------------------------------------------------------------------
    st.subheader("Selected component summary by vehicle")

    component_rows = []

    for i in vehicle_filter:
        trajectory = mu_result[i, selected_component, :]

        max_idx = int(np.argmax(trajectory))
        max_damage = float(trajectory[max_idx])
        final_damage = float(trajectory[-1])
        mean_damage = float(np.mean(trajectory))
        min_margin = float(effective_alpha - max_damage)
        max_utilization_percent = float(100.0 * max_damage / effective_alpha)

        component_rows.append(
            {
                "vehicle": i,
                "component": selected_component,
                "mean_damage": mean_damage,
                "max_damage": max_damage,
                "time_of_max_damage": max_idx,
                "final_damage": final_damage,
                "min_margin_to_effective_alpha": min_margin,
                "max_threshold_utilization_percent": max_utilization_percent,
                "exceeds_effective_alpha": bool(max_damage > effective_alpha + tol),
            }
        )

    component_summary = (
        pd.DataFrame(component_rows)
        .sort_values("max_damage", ascending=False)
        .reset_index(drop=True)
    )

    st.dataframe(
        component_summary,
        width="stretch",
        hide_index=True,
        column_config={
            "max_threshold_utilization_percent": st.column_config.NumberColumn(
                "max threshold utilization",
                format="%.1f%%",
            ),
        },
    )

    # ------------------------------------------------------------------
    # Validator assignment rows for selected component
    # ------------------------------------------------------------------
    st.subheader("Assignment-level rows for selected component")

    component_assignment_df = df[
        (df["component"] == selected_component)
        & (df["vehicle"].isin(vehicle_filter))
    ].copy()

    if component_assignment_df.empty:
        st.info("No active mission assignments found for this component selection.")
    else:
        component_assignment_df["threshold_utilization_percent"] = (
            100.0 * component_assignment_df["utilization_of_threshold"]
        )

        focus_mode = st.radio(
            "Rows to show",
            options=[
                "Closest to threshold",
                "Largest expected increments",
                "Above effective alpha only",
                "All selected rows",
            ],
            horizontal=True,
        )

        if focus_mode == "Closest to threshold":
            shown_df = component_assignment_df.sort_values(
                "margin_to_threshold",
                ascending=True,
            ).head(25)

        elif focus_mode == "Largest expected increments":
            shown_df = component_assignment_df.sort_values(
                "expected_increment",
                ascending=False,
            ).head(25)

        elif focus_mode == "Above effective alpha only":
            shown_df = component_assignment_df[
                component_assignment_df["damage_after"] > effective_alpha + tol
            ].sort_values("damage_after", ascending=False)

        else:
            shown_df = component_assignment_df.sort_values(
                ["vehicle", "time_step"],
                ascending=True,
            )

        display_columns = [
            "time_step",
            "input_day",
            "vehicle",
            "mission",
            "component",
            "damage_before",
            "expected_increment",
            "damage_after",
            "threshold",
            "margin_to_threshold",
            "threshold_utilization_percent",
            "status",
        ]

        st.dataframe(
            shown_df[display_columns].reset_index(drop=True),
            width="stretch",
            hide_index=True,
            column_config={
                "threshold_utilization_percent": st.column_config.NumberColumn(
                    "threshold utilization",
                    format="%.1f%%",
                ),
            },
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    st.subheader("Export")

    export_col_1, export_col_2 = st.columns(2)

    with export_col_1:
        summary_csv = component_summary.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download component summary as CSV",
            data=summary_csv,
            file_name=f"component_{selected_component}_vehicle_summary.csv",
            mime="text/csv",
        )

    with export_col_2:
        assignment_csv = component_assignment_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download component assignment rows as CSV",
            data=assignment_csv,
            file_name=f"component_{selected_component}_assignment_rows.csv",
            mime="text/csv",
            disabled=component_assignment_df.empty,
        )


with tab_data:
    st.header("Raw component-level dataframe")

    st.write(
        "Each row corresponds to one component of one active "
        "vehicle-mission-time assignment."
    )

    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
    )

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download dataframe as CSV",
        data=csv,
        file_name="assignment_feasibility_dataframe.csv",
        mime="text/csv",
    )