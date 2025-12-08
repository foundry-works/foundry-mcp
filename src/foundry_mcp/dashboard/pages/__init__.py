"""Dashboard page modules.

Each page module provides a `render()` function that renders the page content.
"""

from foundry_mcp.dashboard.pages import (
    errors,
    metrics,
    overview,
    providers,
    sdd_workflow,
)

__all__ = [
    "overview",
    "errors",
    "metrics",
    "providers",
    "sdd_workflow",
]
