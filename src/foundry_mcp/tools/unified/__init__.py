"""Unified action-based MCP tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .authoring import register_unified_authoring_tool
from .error import register_unified_error_tool
from .health import register_unified_health_tool
from .journal import register_unified_journal_tool
from .metrics import register_unified_metrics_tool
from .plan import register_unified_plan_tool
from .pr import register_unified_pr_tool
from .provider import register_unified_provider_tool
from .environment import register_unified_environment_tool
from .lifecycle import register_unified_lifecycle_tool
from .verification import register_unified_verification_tool
from .review import register_unified_review_tool
from .spec import register_unified_spec_tool
from .code import register_unified_code_tool
from .server import register_unified_server_tool
from .test import register_unified_test_tool


if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from mcp.server.fastmcp import FastMCP
    from foundry_mcp.config import ServerConfig


def register_unified_tools(mcp: "FastMCP", config: "ServerConfig") -> None:
    """Register consolidated tool families."""

    register_unified_health_tool(mcp, config)
    register_unified_plan_tool(mcp, config)
    register_unified_pr_tool(mcp, config)
    register_unified_error_tool(mcp, config)
    register_unified_metrics_tool(mcp, config)
    register_unified_journal_tool(mcp, config)
    register_unified_authoring_tool(mcp, config)
    register_unified_review_tool(mcp, config)
    register_unified_spec_tool(mcp, config)

    from importlib import import_module

    _task_router = import_module("foundry_mcp.tools.unified.task")
    _task_router.register_unified_task_tool(mcp, config)
    register_unified_provider_tool(mcp, config)
    register_unified_environment_tool(mcp, config)
    register_unified_lifecycle_tool(mcp, config)
    register_unified_verification_tool(mcp, config)
    register_unified_code_tool(mcp, config)
    register_unified_server_tool(mcp, config)
    register_unified_test_tool(mcp, config)


__all__ = [
    "register_unified_tools",
    "register_unified_health_tool",
    "register_unified_plan_tool",
    "register_unified_pr_tool",
    "register_unified_error_tool",
    "register_unified_metrics_tool",
    "register_unified_journal_tool",
    "register_unified_authoring_tool",
    "register_unified_review_tool",
    "register_unified_spec_tool",
    "register_unified_provider_tool",
    "register_unified_environment_tool",
    "register_unified_lifecycle_tool",
    "register_unified_verification_tool",
    "register_unified_code_tool",
    "register_unified_server_tool",
    "register_unified_test_tool",
]
