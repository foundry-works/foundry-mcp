"""Plotly chart builders for dashboard."""

from typing import Optional

import streamlit as st

# Try importing plotly - it's an optional dependency
try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None

# Try importing pandas
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None


def _check_deps():
    """Check if required dependencies are available."""
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly not installed. Install with: pip install plotly")
        return False
    if not PANDAS_AVAILABLE:
        st.warning("Pandas not installed. Install with: pip install pandas")
        return False
    return True


def line_chart(
    df: "pd.DataFrame",
    x: str,
    y: str,
    title: Optional[str] = None,
    color: Optional[str] = None,
    height: int = 400,
) -> None:
    """Render an interactive line chart.

    Args:
        df: DataFrame with data
        x: Column name for x-axis
        y: Column name for y-axis
        title: Optional chart title
        color: Optional column for color grouping
        height: Chart height in pixels
    """
    if not _check_deps():
        return

    if df is None or df.empty:
        st.info("No data to display")
        return

    fig = px.line(
        df,
        x=x,
        y=y,
        title=title,
        color=color,
    )

    # Configure layout for dark theme
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=1, label="1h", step="hour", stepmode="backward"),
                        dict(count=6, label="6h", step="hour", stepmode="backward"),
                        dict(count=24, label="24h", step="hour", stepmode="backward"),
                        dict(step="all", label="All"),
                    ]
                )
            ),
            rangeslider=dict(visible=True),
            type="date",
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


def bar_chart(
    df: "pd.DataFrame",
    x: str,
    y: str,
    title: Optional[str] = None,
    color: Optional[str] = None,
    orientation: str = "v",
    height: int = 400,
) -> None:
    """Render an interactive bar chart.

    Args:
        df: DataFrame with data
        x: Column name for x-axis (or values if horizontal)
        y: Column name for y-axis (or categories if horizontal)
        title: Optional chart title
        color: Optional column for color grouping
        orientation: "v" for vertical, "h" for horizontal
        height: Chart height in pixels
    """
    if not _check_deps():
        return

    if df is None or df.empty:
        st.info("No data to display")
        return

    fig = px.bar(
        df,
        x=x,
        y=y,
        title=title,
        color=color,
        orientation=orientation,
    )

    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)


def pie_chart(
    df: "pd.DataFrame",
    values: str,
    names: str,
    title: Optional[str] = None,
    hole: float = 0.4,
    height: int = 400,
) -> None:
    """Render an interactive pie/donut chart.

    Args:
        df: DataFrame with data
        values: Column name for values
        names: Column name for category names
        title: Optional chart title
        hole: Hole size for donut (0 for pie, 0.4 for donut)
        height: Chart height in pixels
    """
    if not _check_deps():
        return

    if df is None or df.empty:
        st.info("No data to display")
        return

    # Custom colors for status charts
    color_map = {
        "completed": "#10b981",
        "in_progress": "#3b82f6",
        "pending": "#9ca3af",
        "blocked": "#ef4444",
    }

    fig = px.pie(
        df,
        values=values,
        names=names,
        title=title,
        hole=hole,
        color=names,
        color_discrete_map=color_map,
    )

    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)


def treemap_chart(
    df: "pd.DataFrame",
    path: list[str],
    values: str,
    title: Optional[str] = None,
    height: int = 400,
) -> None:
    """Render an interactive treemap chart.

    Args:
        df: DataFrame with data
        path: List of column names for hierarchy path
        values: Column name for values
        title: Optional chart title
        height: Chart height in pixels
    """
    if not _check_deps():
        return

    if df is None or df.empty:
        st.info("No data to display")
        return

    fig = px.treemap(
        df,
        path=path,
        values=values,
        title=title,
    )

    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)


def empty_chart(message: str = "No data available") -> None:
    """Display a placeholder for empty charts."""
    st.info(message)
