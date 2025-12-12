"""Legacy plan tools delegating to the unified action router."""

from __future__ import annotations

from typing import Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.observability import mcp_tool
from foundry_mcp.tools.unified.plan import legacy_plan_action


def register_plan_review_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """Register legacy plan-* tools that forward into the unified plan action router."""

    @canonical_tool(
        mcp,
        canonical_name="plan-review",
    )
    @mcp_tool(tool_name="plan-review", emit_metrics=True, audit=True)
    def plan_review(
        plan_path: str,
        review_type: str = "full",
        ai_provider: Optional[str] = None,
        ai_timeout: float = 120.0,
        consultation_cache: bool = True,
        dry_run: bool = False,
    ) -> dict:
        return legacy_plan_action(
            action="review",
            plan_path=plan_path,
            review_type=review_type,
            ai_provider=ai_provider,
            ai_timeout=ai_timeout,
            consultation_cache=consultation_cache,
            dry_run=dry_run,
        )

    @canonical_tool(
        mcp,
        canonical_name="plan-create",
    )
    @mcp_tool(tool_name="plan-create", emit_metrics=True, audit=True)
    def plan_create(
        name: str,
        template: str = "detailed",
    ) -> dict:
        return legacy_plan_action(
            action="create",
            name=name,
            template=template,
        )

    @canonical_tool(
        mcp,
        canonical_name="plan-list",
    )
    @mcp_tool(tool_name="plan-list", emit_metrics=True, audit=True)
    def plan_list() -> dict:
        return legacy_plan_action(action="list")


__all__ = ["register_plan_review_tools"]
