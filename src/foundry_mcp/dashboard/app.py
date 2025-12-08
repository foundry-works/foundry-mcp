"""Main Streamlit dashboard application.

This is the entry point for the Streamlit dashboard.
Run with: streamlit run src/foundry_mcp/dashboard/app.py
"""

import streamlit as st

# Page config must be first Streamlit command
st.set_page_config(
    page_title="foundry-mcp Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import pages after config
from foundry_mcp.dashboard.pages import overview, errors, metrics, providers, sdd_workflow

# Custom dark theme CSS
st.markdown(
    """
<style>
    /* Consistent dark theme styling */
    .stMetric {
        background-color: rgba(30, 30, 46, 0.8);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid rgba(100, 100, 150, 0.3);
    }
    .stMetric label {
        color: #a0a0b0 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #e0e0f0 !important;
    }
    /* Card-like containers */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
        background-color: rgba(30, 30, 46, 0.5);
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False
if "refresh_interval" not in st.session_state:
    st.session_state.refresh_interval = 5


def render_sidebar():
    """Render navigation sidebar."""
    with st.sidebar:
        st.title(":cube: foundry-mcp")
        st.caption("Observability Dashboard")

        st.divider()

        # Navigation
        page = st.radio(
            "Navigate",
            options=["Overview", "Errors", "Metrics", "Providers", "SDD Workflow"],
            label_visibility="collapsed",
        )

        st.divider()

        # Auto-refresh controls
        st.subheader("Settings")
        st.session_state.auto_refresh = st.checkbox(
            "Auto-refresh",
            value=st.session_state.auto_refresh,
        )
        if st.session_state.auto_refresh:
            st.session_state.refresh_interval = st.slider(
                "Interval (sec)",
                min_value=5,
                max_value=60,
                value=st.session_state.refresh_interval,
            )

        # Manual refresh button
        if st.button("Refresh Now", use_container_width=True):
            st.rerun()

        return page


def main():
    """Main dashboard entry point."""
    # Render sidebar and get page selection
    page = render_sidebar()

    # Route to page
    page_map = {
        "Overview": overview.render,
        "Errors": errors.render,
        "Metrics": metrics.render,
        "Providers": providers.render,
        "SDD Workflow": sdd_workflow.render,
    }

    # Render selected page
    render_func = page_map.get(page, overview.render)
    render_func()

    # Auto-refresh logic
    if st.session_state.auto_refresh:
        import time

        time.sleep(st.session_state.refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
