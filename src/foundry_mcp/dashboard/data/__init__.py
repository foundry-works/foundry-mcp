"""Data access layer for dashboard.

Provides cached access to MetricsStore, ErrorStore, and other data sources,
returning pandas DataFrames for easy use with Streamlit components.
"""

from foundry_mcp.dashboard.data import stores

__all__ = [
    "stores",
]
