"""Data table components for dashboard."""

from typing import Any, Optional

import streamlit as st

# Try importing pandas
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None


def data_table(
    df: "pd.DataFrame",
    columns: Optional[dict[str, Any]] = None,
    height: Optional[int] = None,
    selection_mode: Optional[str] = None,
    on_select: Optional[callable] = None,
) -> Optional[dict]:
    """Render an interactive data table.

    Args:
        df: DataFrame to display
        columns: Column configuration dict for st.column_config
        height: Optional fixed height
        selection_mode: "single-row", "multi-row", or None
        on_select: Callback for selection events

    Returns:
        Selection info if selection_mode is set, None otherwise
    """
    if not PANDAS_AVAILABLE:
        st.warning("Pandas not installed")
        return None

    if df is None or df.empty:
        st.info("No data to display")
        return None

    # Build kwargs
    kwargs = {
        "use_container_width": True,
        "hide_index": True,
    }

    if columns:
        kwargs["column_config"] = columns

    if height:
        kwargs["height"] = height

    if selection_mode:
        kwargs["selection_mode"] = selection_mode
        event = st.dataframe(df, **kwargs, on_select="rerun")
        return event
    else:
        st.dataframe(df, **kwargs)
        return None


def error_table_config() -> dict:
    """Get column configuration for error tables."""
    return {
        "id": st.column_config.TextColumn(
            "ID",
            width="small",
            help="Error ID",
        ),
        "timestamp": st.column_config.DatetimeColumn(
            "Time",
            format="HH:mm:ss",
            width="small",
        ),
        "tool_name": st.column_config.TextColumn(
            "Tool",
            width="medium",
        ),
        "error_code": st.column_config.TextColumn(
            "Code",
            width="small",
        ),
        "message": st.column_config.TextColumn(
            "Message",
            width="large",
        ),
        "fingerprint": st.column_config.TextColumn(
            "Pattern",
            width="small",
            help="Error fingerprint for pattern matching",
        ),
    }


def metrics_table_config() -> dict:
    """Get column configuration for metrics tables."""
    return {
        "metric_name": st.column_config.TextColumn(
            "Metric",
            width="medium",
        ),
        "count": st.column_config.NumberColumn(
            "Records",
            width="small",
        ),
        "first_seen": st.column_config.DatetimeColumn(
            "First Seen",
            format="MMM DD, HH:mm",
            width="medium",
        ),
        "last_seen": st.column_config.DatetimeColumn(
            "Last Seen",
            format="MMM DD, HH:mm",
            width="medium",
        ),
    }


def task_table_config() -> dict:
    """Get column configuration for task tables."""
    return {
        "id": st.column_config.TextColumn(
            "ID",
            width="small",
        ),
        "title": st.column_config.TextColumn(
            "Title",
            width="large",
        ),
        "status": st.column_config.TextColumn(
            "Status",
            width="small",
        ),
        "estimated_hours": st.column_config.NumberColumn(
            "Est. Hours",
            format="%.1f",
            width="small",
        ),
        "actual_hours": st.column_config.NumberColumn(
            "Act. Hours",
            format="%.1f",
            width="small",
        ),
    }


def paginated_table(
    df: "pd.DataFrame",
    page_size: int = 50,
    key: str = "table_page",
    columns: Optional[dict] = None,
) -> None:
    """Render a paginated data table.

    Args:
        df: DataFrame to display
        page_size: Rows per page
        key: Unique key for pagination state
        columns: Column configuration
    """
    if not PANDAS_AVAILABLE:
        st.warning("Pandas not installed")
        return

    if df is None or df.empty:
        st.info("No data to display")
        return

    total_rows = len(df)
    total_pages = (total_rows + page_size - 1) // page_size

    # Page selector
    col1, col2 = st.columns([3, 1])
    with col2:
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=max(1, total_pages),
            value=1,
            key=key,
        )

    with col1:
        st.caption(f"Showing {min(page_size, total_rows)} of {total_rows} rows")

    # Slice data for current page
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_df = df.iloc[start_idx:end_idx]

    # Display table
    data_table(page_df, columns=columns)
