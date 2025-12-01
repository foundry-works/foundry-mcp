"""
Review tools for foundry-mcp.

Provides MCP tools for spec review information and configuration status.
Actual LLM-powered reviews require external AI tool integration.
"""

import logging
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.responses import success_response, error_response
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.observability import get_metrics, mcp_tool
from foundry_mcp.core.providers import (
    get_provider_statuses,
    available_providers,
)

logger = logging.getLogger(__name__)

# Metrics singleton for review tools
_metrics = get_metrics()

# Available review types
REVIEW_TYPES = ["quick", "full", "security", "feasibility"]

# Legacy list - now replaced by provider abstraction
# Use available_providers() and get_provider_statuses() instead
# REVIEW_TOOLS = ["cursor-agent", "gemini", "codex"]


def _get_llm_status() -> Dict[str, Any]:
    """Get LLM configuration status for review operations.

    Returns:
        Dict with LLM status info
    """
    try:
        from foundry_mcp.core.llm_config import get_llm_config

        config = get_llm_config()
        return {
            "configured": config.get_api_key() is not None,
            "provider": config.provider.value,
            "model": config.get_model(),
        }
    except ImportError:
        return {"configured": False, "error": "LLM config not available"}
    except Exception as e:
        return {"configured": False, "error": str(e)}


def register_review_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register review tools with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """

    @canonical_tool(
        mcp,
        canonical_name="spec-review",
    )
    @mcp_tool(tool_name="spec-review", emit_metrics=True, audit=True)
    def spec_review(
        spec_id: str,
        review_type: str = "quick",
        tools: Optional[str] = None,
        model: Optional[str] = None,
        path: Optional[str] = None,
        dry_run: bool = False,
    ) -> dict:
        """
        Run an LLM-powered review session on a specification.

        Wraps the SDD review command to perform intelligent spec analysis
        and generate improvement suggestions. Supports multiple review types
        and external AI tool integration.

        WHEN TO USE:
        - Before starting implementation to catch issues early
        - After completing a phase for quality checks
        - To get security or feasibility analysis
        - For automated spec improvement suggestions

        Args:
            spec_id: Specification ID to review
            review_type: Type of review - "quick", "full", "security", or "feasibility"
            tools: Comma-separated list of review tools (cursor-agent, gemini, codex)
            model: LLM model to use for review (default: from config)
            path: Project root path (default: current directory)
            dry_run: If True, show what would be reviewed without executing

        Returns:
            JSON object with review results:
            - spec_id: The reviewed specification ID
            - review_type: Type of review performed
            - llm_status: LLM configuration status
            - findings: List of review findings (if review executed)
            - suggestions: List of improvement suggestions
            - summary: Human-readable summary

        LIMITATIONS:
        - Requires LLM configuration for intelligent analysis
        - Falls back to basic structural review if LLM unavailable
        - External tools (cursor-agent, gemini, codex) must be installed
        """
        # LLM-powered spec review requires external AI tool integration.
        # This functionality is not available as a direct core API.
        # Use the sdd-toolkit:sdd-plan-review skill for AI-powered spec reviews.
        return asdict(
            error_response(
                "LLM-powered spec review requires external AI tool integration. "
                "Use the sdd-toolkit:sdd-plan-review skill for AI-powered spec reviews.",
                error_code="NOT_IMPLEMENTED",
                error_type="unavailable",
                data={
                    "spec_id": spec_id,
                    "review_type": review_type,
                    "tools": tools,
                    "model": model,
                    "dry_run": dry_run,
                    "alternative": "sdd-toolkit:sdd-plan-review skill",
                    "feature_status": "requires_external_integration",
                },
                remediation="Use the sdd-toolkit:sdd-plan-review skill which provides "
                "multi-model AI consultation and structured review feedback.",
            )
        )

    @canonical_tool(
        mcp,
        canonical_name="review-list-tools",
    )
    @mcp_tool(tool_name="review-list-tools", emit_metrics=True, audit=False)
    def review_list_tools() -> dict:
        """
        List available review tools and pipelines.

        Returns the set of external AI tools that can be used for spec
        reviews, along with their availability status.

        WHEN TO USE:
        - Before running a review to check which tools are available
        - To discover what review capabilities are installed
        - For debugging review tool configuration

        Returns:
            JSON object with:
            - tools: List of tool objects with name and availability
            - llm_status: Current LLM configuration status
        """
        start_time = time.perf_counter()

        try:
            llm_status = _get_llm_status()

            # Get provider statuses from the provider abstraction layer
            provider_statuses = get_provider_statuses()

            # Build tools info from provider statuses
            tools_info = []
            for provider_id, status in provider_statuses.items():
                tools_info.append({
                    "name": provider_id,
                    "available": status.status.value == "available",
                    "status": status.status.value,
                    "reason": status.reason,
                    "checked_at": (
                        status.checked_at.isoformat() if status.checked_at else None
                    ),
                })

            # Also include providers that are registered but not checked
            all_providers = available_providers()
            for provider_id in all_providers:
                if provider_id not in provider_statuses:
                    tools_info.append({
                        "name": provider_id,
                        "available": None,
                        "status": "unknown",
                        "reason": "Not yet checked",
                        "checked_at": None,
                    })

            duration_ms = (time.perf_counter() - start_time) * 1000
            _metrics.timer("review.review_list_tools.duration_ms", duration_ms)

            return asdict(
                success_response(
                    tools=tools_info,
                    llm_status=llm_status,
                    review_types=REVIEW_TYPES,
                    available_count=sum(1 for t in tools_info if t.get("available")),
                    total_count=len(tools_info),
                    duration_ms=round(duration_ms, 2),
                )
            )

        except Exception as e:
            logger.exception("Error listing review tools")
            return asdict(
                error_response(
                    f"Error listing tools: {str(e)}",
                    data={"error_type": type(e).__name__},
                )
            )

    @canonical_tool(
        mcp,
        canonical_name="review-list-plan-tools",
    )
    @mcp_tool(tool_name="review-list-plan-tools", emit_metrics=True, audit=False)
    def review_list_plan_tools() -> dict:
        """
        Enumerate review toolchains available for plan analysis.

        Returns the set of tools specifically designed for reviewing
        SDD plans, including their capabilities and recommended usage.

        WHEN TO USE:
        - When deciding how to review a new plan
        - To understand available review pipelines
        - For configuring automated review workflows

        Returns:
            JSON object with:
            - plan_tools: List of plan review toolchains
            - capabilities: What each toolchain can analyze
            - recommendations: Suggested tool combinations
        """
        start_time = time.perf_counter()

        try:
            llm_status = _get_llm_status()

            # Define plan review toolchains
            plan_tools = [
                {
                    "name": "quick-review",
                    "description": "Fast structural review for basic validation",
                    "capabilities": ["structure", "syntax", "basic_quality"],
                    "llm_required": False,
                    "estimated_time": "< 10 seconds",
                },
                {
                    "name": "full-review",
                    "description": "Comprehensive review with LLM analysis",
                    "capabilities": ["structure", "quality", "feasibility", "suggestions"],
                    "llm_required": True,
                    "estimated_time": "30-60 seconds",
                },
                {
                    "name": "security-review",
                    "description": "Security-focused analysis of plan",
                    "capabilities": ["security", "trust_boundaries", "data_flow"],
                    "llm_required": True,
                    "estimated_time": "30-60 seconds",
                },
                {
                    "name": "feasibility-review",
                    "description": "Implementation feasibility assessment",
                    "capabilities": ["complexity", "dependencies", "risk"],
                    "llm_required": True,
                    "estimated_time": "30-60 seconds",
                },
            ]

            # Filter by LLM availability
            available_tools = []
            for tool in plan_tools:
                tool_info = tool.copy()
                if tool["llm_required"] and not llm_status.get("configured"):
                    tool_info["status"] = "unavailable"
                    tool_info["reason"] = "LLM not configured"
                else:
                    tool_info["status"] = "available"
                available_tools.append(tool_info)

            # Build recommendations based on LLM status
            if llm_status.get("configured"):
                recommendations = [
                    "Use 'full-review' for comprehensive plan analysis",
                    "Run 'security-review' before implementation of sensitive features",
                    "Use 'feasibility-review' for complex or risky plans",
                ]
            else:
                recommendations = [
                    "Use 'quick-review' for basic validation (no LLM required)",
                    "Configure LLM to unlock full review capabilities",
                    "Set FOUNDRY_MCP_LLM_API_KEY or provider-specific env var",
                ]

            duration_ms = (time.perf_counter() - start_time) * 1000
            _metrics.timer("review.review_list_plan_tools.duration_ms", duration_ms)

            return asdict(
                success_response(
                    plan_tools=available_tools,
                    llm_status=llm_status,
                    recommendations=recommendations,
                    duration_ms=round(duration_ms, 2),
                )
            )

        except Exception as e:
            logger.exception("Error listing plan tools")
            return asdict(
                error_response(
                    f"Error listing plan tools: {str(e)}",
                    data={"error_type": type(e).__name__},
                )
            )
