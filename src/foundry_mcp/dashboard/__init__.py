"""
Built-in web dashboard for foundry-mcp.

Provides a web UI for viewing errors, metrics, and AI provider status
without requiring external tools like Grafana.
"""

from foundry_mcp.dashboard.server import (
    start_dashboard,
    stop_dashboard,
    get_dashboard_url,
)

__all__ = [
    "start_dashboard",
    "stop_dashboard",
    "get_dashboard_url",
]
