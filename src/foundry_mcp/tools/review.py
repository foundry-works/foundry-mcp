"""
Review tools for foundry-mcp.

Provides MCP tools for LLM-powered spec review and analysis.
Wraps SDD CLI review commands with intelligent spec analysis.

Resilience features:
- Circuit breaker for SDD CLI calls (opens after 5 consecutive failures)
- Timing metrics for all tool invocations
- Graceful degradation when LLM not configured
"""

import json
import logging
import subprocess
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.responses import success_response, error_response
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.observability import get_metrics, mcp_tool
from foundry_mcp.core.resilience import (
    CircuitBreaker,
    CircuitBreakerError,
    MEDIUM_TIMEOUT,
    SLOW_TIMEOUT,
)

logger = logging.getLogger(__name__)

# Metrics singleton for review tools
_metrics = get_metrics()

# Circuit breaker for SDD CLI review operations
# Opens after 5 consecutive failures, recovers after 30 seconds
_review_breaker = CircuitBreaker(
    name="sdd_cli_review",
    failure_threshold=5,
    recovery_timeout=30.0,
    half_open_max_calls=3,
)

# Default timeout for review operations (120 seconds - reviews can be slow)
REVIEW_TIMEOUT: float = SLOW_TIMEOUT

# Available review types
REVIEW_TYPES = ["quick", "full", "security", "feasibility"]

# Available review tools/pipelines
REVIEW_TOOLS = [
    "cursor-agent",
    "gemini",
    "codex",
]


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


def _run_review_command(
    cmd: List[str],
    tool_name: str,
    timeout: float = REVIEW_TIMEOUT,
) -> subprocess.CompletedProcess:
    """
    Execute an SDD review CLI command with circuit breaker protection and timing.

    Args:
        cmd: Command list to execute
        tool_name: Name of the calling tool (for metrics)
        timeout: Timeout in seconds

    Returns:
        CompletedProcess result from subprocess.run

    Raises:
        CircuitBreakerError: If circuit breaker is open
        subprocess.TimeoutExpired: If command times out
        FileNotFoundError: If SDD CLI is not found
    """
    # Check circuit breaker
    if not _review_breaker.can_execute():
        status = _review_breaker.get_status()
        _metrics.counter(f"review.{tool_name}", labels={"status": "circuit_open"})
        raise CircuitBreakerError(
            f"SDD review circuit breaker is open (retry after {status.get('retry_after_seconds', 0):.1f}s)",
            breaker_name="sdd_cli_review",
            state=_review_breaker.state,
            retry_after=status.get("retry_after_seconds"),
        )

    start_time = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        # Record success or failure based on return code
        if result.returncode == 0:
            _review_breaker.record_success()
        else:
            # Non-zero return code counts as a failure for circuit breaker
            _review_breaker.record_failure()

        return result

    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        # These are infrastructure failures that should trip the circuit breaker
        _review_breaker.record_failure()
        raise
    finally:
        # Record timing metrics
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        _metrics.timer(f"review.{tool_name}.duration_ms", elapsed_ms)


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
        start_time = time.perf_counter()

        try:
            # Validate review type
            if review_type not in REVIEW_TYPES:
                return asdict(
                    error_response(
                        f"Invalid review type: {review_type}. Must be one of: {REVIEW_TYPES}",
                        data={"valid_types": REVIEW_TYPES},
                    )
                )

            # Check LLM status
            llm_status = _get_llm_status()

            # Build command
            cmd = ["foundry-cli", "review", spec_id, "--type", review_type, "--json"]

            if tools:
                cmd.extend(["--tools", tools])
            if model:
                cmd.extend(["--model", model])
            if path:
                cmd.extend(["--path", path])
            if dry_run:
                cmd.append("--dry-run")

            # Dry run mode - return what would be executed
            if dry_run:
                return asdict(
                    success_response(
                        spec_id=spec_id,
                        review_type=review_type,
                        llm_status=llm_status,
                        dry_run=True,
                        command=" ".join(cmd),
                        message="Dry run - no review executed",
                    )
                )

            # Execute review
            result = _run_review_command(cmd, "spec-review")

            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Review failed"
                return asdict(
                    error_response(
                        f"Review failed: {error_msg}",
                        data={
                            "spec_id": spec_id,
                            "review_type": review_type,
                            "exit_code": result.returncode,
                        },
                    )
                )

            # Parse JSON output
            try:
                review_data = json.loads(result.stdout)
            except json.JSONDecodeError:
                # If not JSON, return raw output
                review_data = {"raw_output": result.stdout}

            duration_ms = (time.perf_counter() - start_time) * 1000
            _metrics.timer(
                "review.spec_review.duration_ms",
                duration_ms,
                labels={"review_type": review_type},
            )

            # Remove keys that we're explicitly setting to avoid duplicate keyword args
            review_data.pop("spec_id", None)
            review_data.pop("review_type", None)
            return asdict(
                success_response(
                    spec_id=spec_id,
                    review_type=review_type,
                    llm_status=llm_status,
                    duration_ms=round(duration_ms, 2),
                    **review_data,
                )
            )

        except CircuitBreakerError as e:
            return asdict(
                error_response(
                    str(e),
                    data={
                        "spec_id": spec_id,
                        "retry_after_seconds": e.retry_after,
                        "breaker_state": e.state,
                    },
                )
            )
        except subprocess.TimeoutExpired:
            return asdict(
                error_response(
                    f"Review timed out after {REVIEW_TIMEOUT}s",
                    data={
                        "spec_id": spec_id,
                        "review_type": review_type,
                        "timeout_seconds": REVIEW_TIMEOUT,
                    },
                )
            )
        except FileNotFoundError:
            return asdict(
                error_response(
                    "SDD CLI not found. Ensure 'sdd' is installed and in PATH.",
                    data={"spec_id": spec_id},
                )
            )
        except Exception as e:
            logger.exception(f"Error in spec-review for {spec_id}")
            return asdict(
                error_response(
                    f"Review error: {str(e)}",
                    data={
                        "spec_id": spec_id,
                        "error_type": type(e).__name__,
                    },
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

            # Check tool availability
            tools_info = []
            for tool in REVIEW_TOOLS:
                # Check if tool is available by trying to run --version
                try:
                    result = subprocess.run(
                        [tool, "--version"],
                        capture_output=True,
                        text=True,
                        timeout=5.0,
                    )
                    available = result.returncode == 0
                    version = result.stdout.strip() if available else None
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    available = False
                    version = None

                tools_info.append({
                    "name": tool,
                    "available": available,
                    "version": version,
                })

            duration_ms = (time.perf_counter() - start_time) * 1000

            return asdict(
                success_response(
                    tools=tools_info,
                    llm_status=llm_status,
                    review_types=REVIEW_TYPES,
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
