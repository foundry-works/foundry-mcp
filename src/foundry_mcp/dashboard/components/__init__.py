"""Reusable dashboard UI components.

Components:
    cards: KPI metric card helpers
    charts: Plotly chart builders
    filters: Time range and filter widgets
    tables: Data table configurations
"""

from foundry_mcp.dashboard.components import cards, charts, filters, tables

__all__ = [
    "cards",
    "charts",
    "filters",
    "tables",
]
