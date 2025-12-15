"""Errors page - filterable error list with patterns and details."""

import streamlit as st

from typing import Any

from foundry_mcp.dashboard.components.filters import (
    time_range_filter,
    text_filter,
    filter_row,
)
from foundry_mcp.dashboard.components.tables import (
    error_table_config,
    paginated_table,
)
from foundry_mcp.dashboard.components.charts import treemap_chart, pie_chart
from foundry_mcp.dashboard.data.stores import (
    get_errors,
    get_error_stats,
    get_error_patterns,
    get_error_by_id,
)

# Try importing pandas
pd: Any
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None


def render():
    """Render the Errors page."""
    st.header("Errors")

    # Check if error collection is enabled
    stats = get_error_stats()
    if not stats.get("enabled"):
        st.warning(
            "Error collection is disabled. Enable it in foundry-mcp.toml under [error_collection]"
        )
        return

    # Filters
    st.subheader("Filters")
    cols = filter_row(4)

    with cols[0]:
        hours = time_range_filter(key="error_time_range", default="24h")
    with cols[1]:
        tool_filter = text_filter(
            "Tool Name", key="error_tool", placeholder="e.g., spec"
        )
    with cols[2]:
        code_filter = text_filter(
            "Error Code", key="error_code", placeholder="e.g., VALIDATION_ERROR"
        )
    with cols[3]:
        # Show stats
        st.metric("Total Errors", stats.get("total", 0))

    st.divider()

    # Get filtered errors
    errors_df = get_errors(
        tool_name=tool_filter if tool_filter else None,
        error_code=code_filter if code_filter else None,
        since_hours=hours,
        limit=500,
    )

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["Error List", "Patterns", "Analysis"])

    with tab1:
        st.subheader("Error List")
        if errors_df is not None and not errors_df.empty:
            # Show table with selection
            st.caption(f"Showing {len(errors_df)} errors")

            # Paginated table
            paginated_table(
                errors_df,
                page_size=25,
                key="errors_page",
                columns=error_table_config(),
            )

            # Error detail expander
            st.subheader("Error Details")
            selected_id = st.text_input(
                "Enter Error ID to view details",
                key="error_detail_id",
                placeholder="Click an error ID above and paste here",
            )

            if selected_id:
                error = get_error_by_id(selected_id)
                if error:
                    with st.expander(f"Error: {selected_id}", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Tool:** " + error.get("tool_name", "N/A"))
                            st.markdown("**Code:** " + error.get("error_code", "N/A"))
                            st.markdown("**Type:** " + error.get("error_type", "N/A"))
                            st.markdown(
                                "**Time:** " + str(error.get("timestamp", "N/A"))
                            )

                        with col2:
                            st.markdown(
                                "**Fingerprint:** "
                                + error.get("fingerprint", "N/A")[:20]
                                + "..."
                            )

                        st.markdown("**Message:**")
                        st.text(error.get("message", "No message"))

                        if error.get("stack_trace"):
                            st.markdown("**Stack Trace:**")
                            st.code(error["stack_trace"], language="python")

                        if error.get("context"):
                            st.markdown("**Context:**")
                            st.json(error["context"])
                else:
                    st.warning(f"Error {selected_id} not found")
        else:
            st.info("No errors found matching the filters")

    with tab2:
        st.subheader("Error Patterns")
        patterns = get_error_patterns(min_count=2)

        if patterns and PANDAS_AVAILABLE and pd is not None:
            # Create DataFrame for visualization
            patterns_df = pd.DataFrame(patterns)

            # Treemap of patterns
            if "tool_name" in patterns_df.columns and "count" in patterns_df.columns:
                treemap_chart(
                    patterns_df,
                    path=["tool_name", "error_code"]
                    if "error_code" in patterns_df.columns
                    else ["tool_name"],
                    values="count",
                    title="Error Distribution by Tool",
                    height=400,
                )

            # Pattern details table
            st.subheader("Pattern Details")
            st.dataframe(
                patterns_df,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No recurring patterns detected (minimum 2 occurrences required)")

    with tab3:
        st.subheader("Error Analysis")

        if errors_df is not None and not errors_df.empty and PANDAS_AVAILABLE:
            col1, col2 = st.columns(2)

            with col1:
                # Errors by tool
                if "tool_name" in errors_df.columns:
                    tool_counts = errors_df["tool_name"].value_counts().reset_index()
                    tool_counts.columns = ["tool_name", "count"]
                    pie_chart(
                        tool_counts,
                        values="count",
                        names="tool_name",
                        title="Errors by Tool",
                        height=300,
                    )

            with col2:
                # Errors by code
                if "error_code" in errors_df.columns:
                    code_counts = errors_df["error_code"].value_counts().reset_index()
                    code_counts.columns = ["error_code", "count"]
                    pie_chart(
                        code_counts,
                        values="count",
                        names="error_code",
                        title="Errors by Code",
                        height=300,
                    )

            # Export button
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                csv = errors_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="errors_export.csv",
                    mime="text/csv",
                )
            with col2:
                json_data = errors_df.to_json(orient="records")
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name="errors_export.json",
                    mime="application/json",
                )
        else:
            st.info("No error data available for analysis")
