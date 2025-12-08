"""Metrics page - time-series viewer with summaries."""

import streamlit as st

from foundry_mcp.dashboard.components.filters import time_range_filter
from foundry_mcp.dashboard.components.charts import line_chart, empty_chart
from foundry_mcp.dashboard.components.cards import kpi_row
from foundry_mcp.dashboard.data.stores import get_metrics_list, get_metrics_timeseries, get_metrics_summary

# Try importing pandas
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None


def render():
    """Render the Metrics page."""
    st.header("Metrics")

    # Get available metrics
    metrics_list = get_metrics_list()

    if not metrics_list:
        st.warning("No metrics available. Metrics persistence may be disabled.")
        st.info("Enable metrics persistence in foundry-mcp.toml under [metrics_persistence]")
        return

    # Metric selector and time range
    col1, col2 = st.columns([2, 1])

    with col1:
        metric_names = [m.get("metric_name", "unknown") for m in metrics_list]
        selected_metric = st.selectbox(
            "Select Metric",
            options=metric_names,
            key="metrics_selector",
        )

    with col2:
        hours = time_range_filter(key="metrics_time_range", default="24h")

    st.divider()

    if selected_metric:
        # Get summary statistics
        summary = get_metrics_summary(selected_metric, since_hours=hours)

        # Summary cards
        st.subheader("Summary Statistics")
        if summary.get("enabled"):
            kpi_row(
                [
                    {"label": "Count", "value": summary.get("count", 0)},
                    {"label": "Min", "value": f"{summary.get('min', 0):.2f}"},
                    {"label": "Max", "value": f"{summary.get('max', 0):.2f}"},
                    {"label": "Average", "value": f"{summary.get('avg', 0):.2f}"},
                    {"label": "Sum", "value": f"{summary.get('sum', 0):.2f}"},
                ],
                columns=5,
            )
        else:
            st.info("Summary not available")

        st.divider()

        # Time-series chart
        st.subheader(f"Time Series: {selected_metric}")
        timeseries_df = get_metrics_timeseries(selected_metric, since_hours=hours)

        if timeseries_df is not None and not timeseries_df.empty:
            line_chart(
                timeseries_df,
                x="timestamp",
                y="value",
                title=None,
                height=400,
            )

            # Data table
            with st.expander("View Raw Data"):
                st.dataframe(
                    timeseries_df,
                    use_container_width=True,
                    hide_index=True,
                )

                # Export
                col1, col2 = st.columns(2)
                with col1:
                    csv = timeseries_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{selected_metric}_export.csv",
                        mime="text/csv",
                    )
                with col2:
                    json_data = timeseries_df.to_json(orient="records")
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name=f"{selected_metric}_export.json",
                        mime="application/json",
                    )
        else:
            empty_chart(f"No data available for {selected_metric}")

    # Metrics catalog
    st.divider()
    st.subheader("Available Metrics")

    if PANDAS_AVAILABLE:
        metrics_df = pd.DataFrame(metrics_list)
        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "metric_name": st.column_config.TextColumn("Metric", width="medium"),
                "count": st.column_config.NumberColumn("Records", width="small"),
            },
        )
    else:
        for m in metrics_list:
            st.text(f"- {m.get('metric_name', 'unknown')} ({m.get('count', 0)} records)")
