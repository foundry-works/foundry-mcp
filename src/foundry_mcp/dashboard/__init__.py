"""Streamlit-based dashboard for foundry-mcp observability.

This module provides a web UI for viewing errors, metrics, provider status,
and SDD workflow progress.

Public API:
    launch_dashboard: Start the Streamlit dashboard server
    stop_dashboard: Stop the running dashboard server
    get_dashboard_status: Check if dashboard is running

Usage:
    from foundry_mcp.dashboard import launch_dashboard, stop_dashboard

    # Start dashboard
    result = launch_dashboard(port=8501, open_browser=True)
    print(f"Dashboard running at {result['url']}")

    # Stop dashboard
    stop_dashboard()
"""

from foundry_mcp.dashboard.launcher import (
    get_dashboard_status,
    launch_dashboard,
    stop_dashboard,
)

__all__ = [
    "launch_dashboard",
    "stop_dashboard",
    "get_dashboard_status",
]
