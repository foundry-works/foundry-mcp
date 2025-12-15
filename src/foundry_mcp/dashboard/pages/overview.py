"""Overview page - dashboard home with KPIs and summary charts."""

import streamlit as st

from foundry_mcp.dashboard.components.cards import kpi_row
from foundry_mcp.dashboard.components.charts import line_chart, empty_chart
from foundry_mcp.dashboard.data.stores import (
    get_overview_summary,
    get_metrics_timeseries,
    get_errors,
    get_error_patterns,
)


def render():
    """Render the Overview page."""
    st.header("Overview")

    # Get summary data
    summary = get_overview_summary()

    # KPI Cards Row
    st.subheader("Key Metrics")
    kpi_row(
        [
            {
                "label": "Tool Invocations",
                "value": summary.get("total_invocations", 0),
                "help": "Total tool invocations recorded",
            },
            {
                "label": "Active Tools",
                "value": summary.get("active_tools", 0),
                "help": "Unique tools used in the last hour",
            },
            {
                "label": "Health",
                "value": summary.get("health_status", "unknown").title(),
                "help": "Overall system health status",
            },
            {
                "label": "Avg Latency",
                "value": f"{summary.get('avg_latency_ms', 0):.0f}ms",
                "help": "Average tool execution time",
            },
            {
                "label": "Errors (24h)",
                "value": summary.get("error_count", 0),
                "help": "Errors in the last 24 hours",
            },
            {
                "label": "Providers",
                "value": summary.get("provider_count", 0),
                "help": "Available AI providers",
            },
        ]
    )

    st.divider()

    # Charts Row
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Tool Invocations")
        invocations_df = get_metrics_timeseries("tool_invocations_total", since_hours=24)
        if invocations_df is not None and not invocations_df.empty:
            line_chart(
                invocations_df,
                x="timestamp",
                y="value",
                title=None,
                height=300,
            )
        else:
            empty_chart("No invocation data available")

    with col2:
        st.subheader("Error Rate")
        errors_df = get_errors(since_hours=24, limit=500)
        if errors_df is not None and not errors_df.empty:
            # Group by hour for error rate
            try:
                errors_df["hour"] = errors_df["timestamp"].dt.floor("H")
                hourly_errors = errors_df.groupby("hour").size().reset_index(name="count")
                hourly_errors.columns = ["timestamp", "value"]
                line_chart(
                    hourly_errors,
                    x="timestamp",
                    y="value",
                    title=None,
                    height=300,
                )
            except Exception:
                empty_chart("Could not process error data")
        else:
            empty_chart("No error data available")

    st.divider()

    # Bottom Row - Patterns and Recent
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Error Patterns")
        patterns = get_error_patterns(min_count=2)
        if patterns:
            for i, p in enumerate(patterns[:5]):
                with st.container(border=True):
                    st.markdown(f"**{p.get('tool_name', 'Unknown')}**")
                    st.caption(f"Count: {p.get('count', 0)} | Code: {p.get('error_code', 'N/A')}")
                    if p.get("message"):
                        st.text(p["message"][:100] + "..." if len(p.get("message", "")) > 100 else p.get("message", ""))
        else:
            st.info("No recurring error patterns detected")

    with col2:
        st.subheader("Recent Errors")
        errors_df = get_errors(since_hours=1, limit=5)
        if errors_df is not None and not errors_df.empty:
            for _, row in errors_df.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{row.get('tool_name', 'Unknown')}**")
                    st.caption(f"{row.get('timestamp', '')} | {row.get('error_code', 'N/A')}")
                    msg = row.get("message", "")
                    st.text(msg[:80] + "..." if len(msg) > 80 else msg)
        else:
            st.info("No recent errors")
