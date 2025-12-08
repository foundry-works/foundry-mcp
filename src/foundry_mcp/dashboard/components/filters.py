"""Filter widget components for dashboard."""

from datetime import datetime, timedelta
from typing import Optional

import streamlit as st


def time_range_filter(
    key: str = "time_range",
    default: str = "24h",
) -> int:
    """Render a time range selector.

    Args:
        key: Unique key for the widget
        default: Default selection

    Returns:
        Number of hours for the selected range
    """
    options = {
        "1 hour": 1,
        "6 hours": 6,
        "24 hours": 24,
        "7 days": 168,
        "30 days": 720,
    }

    # Find default index
    default_hours = {"1h": 1, "6h": 6, "24h": 24, "7d": 168, "30d": 720}.get(default, 24)
    default_label = next((k for k, v in options.items() if v == default_hours), "24 hours")
    default_idx = list(options.keys()).index(default_label)

    selected = st.selectbox(
        "Time Range",
        options=list(options.keys()),
        index=default_idx,
        key=key,
    )

    return options[selected]


def date_range_filter(
    key: str = "date_range",
    default_days: int = 7,
) -> tuple[datetime, datetime]:
    """Render a date range picker.

    Args:
        key: Unique key for the widget
        default_days: Default number of days back

    Returns:
        Tuple of (start_date, end_date)
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=default_days)

    dates = st.date_input(
        "Date Range",
        value=(start_date.date(), end_date.date()),
        key=key,
    )

    if isinstance(dates, tuple) and len(dates) == 2:
        return (
            datetime.combine(dates[0], datetime.min.time()),
            datetime.combine(dates[1], datetime.max.time()),
        )
    else:
        # Single date selected
        return (
            datetime.combine(dates, datetime.min.time()),
            datetime.combine(dates, datetime.max.time()),
        )


def multi_select_filter(
    label: str,
    options: list[str],
    key: str,
    default: Optional[list[str]] = None,
) -> list[str]:
    """Render a multi-select filter.

    Args:
        label: Filter label
        options: List of options to choose from
        key: Unique key for the widget
        default: Default selected values

    Returns:
        List of selected values
    """
    return st.multiselect(
        label,
        options=options,
        default=default,
        key=key,
    )


def text_filter(
    label: str,
    key: str,
    placeholder: Optional[str] = None,
) -> str:
    """Render a text input filter.

    Args:
        label: Filter label
        key: Unique key for the widget
        placeholder: Optional placeholder text

    Returns:
        Input text value
    """
    return st.text_input(
        label,
        key=key,
        placeholder=placeholder or f"Filter by {label.lower()}...",
    )


def filter_row(num_cols: int = 4):
    """Create a row of filter columns.

    Args:
        num_cols: Number of columns

    Returns:
        Tuple of column objects
    """
    return st.columns(num_cols)
