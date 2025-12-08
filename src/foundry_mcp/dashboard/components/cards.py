"""KPI card components for dashboard."""

from typing import Optional

import streamlit as st


def metric_card(
    label: str,
    value: str | int | float,
    delta: Optional[str] = None,
    delta_color: str = "normal",
    help_text: Optional[str] = None,
) -> None:
    """Render a metric card with optional delta indicator.

    Args:
        label: Card title/label
        value: Main value to display
        delta: Optional change indicator (e.g., "+5%", "-3")
        delta_color: Color for delta ("normal", "inverse", "off")
        help_text: Optional tooltip help text
    """
    st.metric(
        label=label,
        value=value,
        delta=delta,
        delta_color=delta_color,
        help=help_text,
    )


def status_badge(status: str, label: Optional[str] = None) -> None:
    """Render a status badge with color coding.

    Args:
        status: Status value ("healthy", "unhealthy", "warning", "unknown")
        label: Optional label to show before status
    """
    colors = {
        "healthy": ":green_circle:",
        "unhealthy": ":red_circle:",
        "warning": ":yellow_circle:",
        "unknown": ":white_circle:",
        "available": ":green_circle:",
        "unavailable": ":red_circle:",
    }

    emoji = colors.get(status.lower(), ":white_circle:")
    text = label if label else status.title()
    st.markdown(f"{emoji} **{text}**")


def info_card(title: str, content: str, icon: Optional[str] = None) -> None:
    """Render an info card with title and content.

    Args:
        title: Card title
        content: Card content (markdown supported)
        icon: Optional emoji icon
    """
    with st.container(border=True):
        header = f"{icon} {title}" if icon else title
        st.subheader(header)
        st.markdown(content)


def kpi_row(
    metrics: list[dict],
    columns: int = 6,
) -> None:
    """Render a row of KPI metric cards.

    Args:
        metrics: List of metric dicts with keys: label, value, delta, help
        columns: Number of columns (default 6)
    """
    cols = st.columns(columns)

    for i, m in enumerate(metrics):
        with cols[i % columns]:
            metric_card(
                label=m.get("label", ""),
                value=m.get("value", 0),
                delta=m.get("delta"),
                delta_color=m.get("delta_color", "normal"),
                help_text=m.get("help"),
            )
