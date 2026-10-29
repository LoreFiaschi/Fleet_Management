"""Streamlit entry point of the dashboards (spec, section "Dashboards").

Start it from the project root with:

    python -m streamlit run src/fleet_management/validator_dashboard.py

The sidebar selects the mode, the input file, the result file and the tolerance.
The button "Load" reads the two files and runs the mode.
A failed load shows the error and no page.
"""

from fleet_management.validation.dashboard_gamma import (
    load_gamma_diagnostic,
    render_gamma_diagnostic,
)
from fleet_management.validation.dashboard_inspector import (
    load_inspector,
    render_result_inspector,
)

MODES = {
    "Result inspector": (load_inspector, render_result_inspector),
    "Gamma diagnostic": (load_gamma_diagnostic, render_gamma_diagnostic),
}


def main() -> None:
    """Draw the sidebar, load the pair on the button, show the chosen mode (spec, Dashboards)."""
    import streamlit as st  # here, so that this module loads without Streamlit

    st.set_page_config(page_title="Fleet Management dashboards", layout="wide")
    with st.sidebar:
        mode = st.selectbox("Mode", list(MODES))
        input_path = st.text_input("Input file", value="input.yaml")
        results_path = st.text_input("Result file", value="output.yaml")
        tol = st.number_input("Tolerance", min_value=1e-12, value=1e-6, step=1e-6,
                              format="%.1e")
        load = st.button("Load", type="primary")
    if load:
        try:
            st.session_state["loaded"] = (mode, MODES[mode][0](input_path, results_path, tol))
        except Exception as exc:
            st.session_state.pop("loaded", None)
            st.error(f"Could not load the files: {exc}")
            st.stop()
    loaded = st.session_state.get("loaded")
    if loaded is None or loaded[0] != mode:
        st.info("Choose the files in the sidebar and press Load.")
        st.stop()
    MODES[mode][1](loaded[1])


if __name__ == "__main__":     # ``streamlit run`` executes this file as the module __main__
    main()
