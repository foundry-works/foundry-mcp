"""
Review tools for foundry-mcp.

Provides MCP tools for spec review including:
- Quick structural review (no LLM required)
- AI-powered full/security/feasibility reviews via ConsultationOrchestrator

AI-enhanced reviews use:
- PLAN_REVIEW_FULL_V1: Comprehensive 6-dimension review
- PLAN_REVIEW_QUICK_V1: Critical blockers and questions focus
- PLAN_REVIEW_SECURITY_V1: Security-focused review
- PLAN_REVIEW_FEASIBILITY_V1: Technical complexity assessment
"""

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.responses import success_response, error_response, sanitize_error_message
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

# Map review types to PLAN_REVIEW templates
REVIEW_TYPE_TO_TEMPLATE = {
    "full": "PLAN_REVIEW_FULL_V1",
    "security": "PLAN_REVIEW_SECURITY_V1",
    "feasibility": "PLAN_REVIEW_FEASIBILITY_V1",
}

# Default AI consultation timeout
DEFAULT_AI_TIMEOUT = 120.0

# Legacy list - used as fallback when review_provider_integration flag is disabled
LEGACY_REVIEW_TOOLS = ["cursor-agent", "gemini", "codex"]

# Feature flag name for provider integration
REVIEW_PROVIDER_FLAG = "review_provider_integration"


def _is_provider_integration_enabled() -> bool:
    """Check if provider integration feature flag is enabled.

    Returns:
        True if provider integration should be used, False for legacy behavior.
    """
    # Default to True (use provider integration)
    # This can be overridden by environment variable for rollback
    import os
    flag_override = os.environ.get("FOUNDRY_REVIEW_PROVIDER_INTEGRATION", "").lower()
    if flag_override == "false" or flag_override == "0":
        return False
    return True


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
        logger.debug(f"Failed to get LLM config: {e}")
        return {"configured": False, "error": "Failed to load LLM configuration"}


def _run_quick_review(
    spec_id: str,
    path: Optional[str],
    dry_run: bool,
    llm_status: Dict[str, Any],
    start_time: float,
) -> dict:
    """
    Run a quick (non-LLM) structural review.

    Args:
        spec_id: Specification ID to review
        path: Project root path
        dry_run: Preview without executing
        llm_status: LLM configuration status
        start_time: Start time for duration calculation

    Returns:
        Dict with review results
    """
    from foundry_mcp.core.review import quick_review, prepare_review_context
    from foundry_mcp.core.spec import find_specs_directory

    # Resolve specs directory
    specs_dir = None
    if path:
        specs_dir = Path(path) / "specs"
        if not specs_dir.exists():
            specs_dir = find_specs_directory(Path(path))
    else:
        specs_dir = find_specs_directory()

    if dry_run:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return asdict(
            success_response(
                spec_id=spec_id,
                review_type="quick",
                dry_run=True,
                llm_status=llm_status,
                message="Dry run - quick review skipped",
                duration_ms=round(duration_ms, 2),
            )
        )

    # Run quick review
    result = quick_review(spec_id=spec_id, specs_dir=specs_dir)
    duration_ms = (time.perf_counter() - start_time) * 1000

    # Convert findings to dicts
    findings = [
        {
            "code": f.code,
            "message": f.message,
            "severity": f.severity,
            "category": f.category,
            "location": f.location,
            "suggestion": f.suggestion,
        }
        for f in result.findings
    ]

    return asdict(
        success_response(
            spec_id=result.spec_id,
            title=result.title,
            review_type=result.review_type,
            is_valid=result.is_valid,
            findings=findings,
            summary=result.summary,
            error_count=result.error_count,
            warning_count=result.warning_count,
            info_count=result.info_count,
            llm_status=llm_status,
            duration_ms=round(duration_ms, 2),
        )
    )


def _run_ai_review(
    spec_id: str,
    review_type: str,
    ai_provider: Optional[str],
    ai_timeout: float,
    consultation_cache: bool,
    path: Optional[str],
    dry_run: bool,
    llm_status: Dict[str, Any],
    start_time: float,
) -> dict:
    """
    Run an AI-powered review using ConsultationOrchestrator.

    Args:
        spec_id: Specification ID to review
        review_type: Type of review (full, security, feasibility)
        ai_provider: Explicit provider selection
        ai_timeout: Consultation timeout in seconds
        consultation_cache: Whether to use consultation cache
        path: Project root path
        dry_run: Preview without executing
        llm_status: LLM configuration status
        start_time: Start time for duration calculation

    Returns:
        Dict with review results
    """
    from foundry_mcp.core.review import prepare_review_context
    from foundry_mcp.core.spec import find_specs_directory

    # Resolve specs directory
    specs_dir = None
    if path:
        specs_dir = Path(path) / "specs"
        if not specs_dir.exists():
            specs_dir = find_specs_directory(Path(path))
    else:
        specs_dir = find_specs_directory()

    # Get template for review type
    template_id = REVIEW_TYPE_TO_TEMPLATE.get(review_type)
    if template_id is None:
        return asdict(
            error_response(
                f"Unknown review type: {review_type}",
                error_code="INVALID_REVIEW_TYPE",
                error_type="validation",
                data={"review_type": review_type},
                remediation=f"Use one of: {', '.join(REVIEW_TYPE_TO_TEMPLATE.keys())}",
            )
        )

    # Prepare review context
    context = prepare_review_context(
        spec_id=spec_id,
        specs_dir=specs_dir,
        include_tasks=True,
        include_journals=True,
    )

    if context is None:
        return asdict(
            error_response(
                f"Specification '{spec_id}' not found",
                error_code="SPEC_NOT_FOUND",
                error_type="not_found",
                data={"spec_id": spec_id},
                remediation="Verify the spec ID and that the spec exists in the specs directory",
            )
        )

    # Dry run - preview what would be reviewed
    if dry_run:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return asdict(
            success_response(
                spec_id=spec_id,
                review_type=review_type,
                template_id=template_id,
                dry_run=True,
                llm_status=llm_status,
                ai_provider=ai_provider,
                consultation_cache=consultation_cache,
                message=f"Dry run - {review_type} review would use template {template_id}",
                spec_title=context.title,
                task_count=context.stats.total_tasks if context.stats else 0,
                duration_ms=round(duration_ms, 2),
            )
        )

    # Import consultation layer components
    try:
        from foundry_mcp.core.ai_consultation import (
            ConsensusResult,
            ConsultationOrchestrator,
            ConsultationRequest,
            ConsultationWorkflow,
        )
    except ImportError as exc:
        logger.debug(f"AI consultation import error: {exc}")
        return asdict(
            error_response(
                "AI consultation layer not available",
                error_code="AI_NOT_AVAILABLE",
                error_type="unavailable",
                remediation="Ensure foundry_mcp.core.ai_consultation is properly installed",
            )
        )

    # Initialize orchestrator with preferred provider if specified
    preferred_providers = [ai_provider] if ai_provider else []
    orchestrator = ConsultationOrchestrator(
        preferred_providers=preferred_providers,
        default_timeout=ai_timeout,
    )

    # Check if any providers are available
    if not orchestrator.is_available(provider_id=ai_provider):
        provider_msg = f" (requested: {ai_provider})" if ai_provider else ""
        _metrics.counter(
            "review.errors",
            labels={"tool": "spec-review", "error_type": "ai_no_provider"},
        )
        return asdict(
            error_response(
                f"AI-enhanced review requested but no providers available{provider_msg}",
                error_code="AI_NO_PROVIDER",
                error_type="unavailable",
                data={
                    "spec_id": spec_id,
                    "review_type": review_type,
                    "requested_provider": ai_provider,
                    "llm_status": llm_status,
                },
                remediation="Install and configure an AI provider (gemini, cursor-agent, codex) "
                "or use review_type='quick' for non-AI review.",
            )
        )

    # Build context for prompt template
    spec_content = json.dumps(context.spec_data, indent=2)

    # Create consultation request - orchestrator handles prompt building
    request = ConsultationRequest(
        workflow=ConsultationWorkflow.PLAN_REVIEW,
        prompt_id=template_id,
        context={
            "spec_content": spec_content,
            "spec_id": spec_id,
            "title": context.title,
            "review_type": review_type,
        },
        provider_id=ai_provider,
        timeout=ai_timeout,
    )

    # Execute consultation
    try:
        result = orchestrator.consult(request, use_cache=consultation_cache)
    except Exception as exc:
        logger.exception(f"AI consultation failed for {spec_id}")
        return asdict(
            error_response(
                "AI consultation failed",
                error_code="AI_CONSULTATION_ERROR",
                error_type="error",
                data={
                    "spec_id": spec_id,
                    "review_type": review_type,
                },
                remediation="Check provider configuration and try again",
            )
        )

    duration_ms = (time.perf_counter() - start_time) * 1000

    # Build response - handle both single-model and multi-model results
    if isinstance(result, ConsensusResult):
        # Multi-model consensus result
        responses_data = [
            {
                "provider_id": r.provider_id,
                "model_used": r.model_used,
                "content": r.content,
                "success": r.success,
                "error": r.error,
                "tokens": r.tokens,
                "duration_ms": r.duration_ms,
            }
            for r in result.responses
        ]
        agreement_data = None
        if result.agreement:
            agreement_data = {
                "total_providers": result.agreement.total_providers,
                "successful_providers": result.agreement.successful_providers,
                "failed_providers": result.agreement.failed_providers,
                "success_rate": result.agreement.success_rate,
                "has_consensus": result.agreement.has_consensus,
            }
        return asdict(
            success_response(
                spec_id=spec_id,
                title=context.title,
                review_type=review_type,
                template_id=template_id,
                llm_status=llm_status,
                mode="multi_model",
                consultation_cache=consultation_cache,
                responses=responses_data,
                agreement=agreement_data,
                primary_content=result.primary_content,
                warnings=result.warnings,
                stats={
                    "total_tasks": context.stats.total_tasks if context.stats else 0,
                    "completed_tasks": context.stats.completed_tasks if context.stats else 0,
                    "progress_percentage": context.progress.get("percentage", 0) if context.progress else 0,
                },
                duration_ms=round(duration_ms, 2),
            )
        )
    else:
        # Single-model result (ConsultationResult)
        return asdict(
            success_response(
                spec_id=spec_id,
                title=context.title,
                review_type=review_type,
                template_id=template_id,
                llm_status=llm_status,
                mode="single_model",
                ai_provider=result.provider_id if result else ai_provider,
                consultation_cache=consultation_cache,
                response=result.content if result else None,
                model=result.model_used if result else None,
                cached=result.cache_hit if result else False,
                stats={
                    "total_tasks": context.stats.total_tasks if context.stats else 0,
                    "completed_tasks": context.stats.completed_tasks if context.stats else 0,
                    "progress_percentage": context.progress.get("percentage", 0) if context.progress else 0,
                },
                duration_ms=round(duration_ms, 2),
            )
        )


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
        ai_provider: Optional[str] = None,
        ai_timeout: float = DEFAULT_AI_TIMEOUT,
        consultation_cache: bool = True,
        path: Optional[str] = None,
        dry_run: bool = False,
    ) -> dict:
        """
        Run a structural or AI-powered review session on a specification.

        Performs spec analysis using ConsultationOrchestrator for AI-powered
        reviews, or quick structural review for non-LLM analysis.

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
            ai_provider: Explicit AI provider selection (e.g., gemini, cursor-agent)
            ai_timeout: AI consultation timeout in seconds (default: 120)
            consultation_cache: Whether to use AI consultation cache (default: True)
            path: Project root path (default: current directory)
            dry_run: If True, show what would be reviewed without executing

        Returns:
            JSON object with review results:
            - spec_id: The reviewed specification ID
            - review_type: Type of review performed
            - llm_status: LLM configuration status
            - response: AI review content (for LLM reviews)
            - findings: List of review findings (for quick review)
            - summary: Human-readable summary

        LIMITATIONS:
        - Requires LLM configuration for AI-powered analysis
        - Falls back to error if AI unavailable for LLM review types
        - External tools (cursor-agent, gemini, codex) must be installed
        """
        start_time = time.perf_counter()
        llm_status = _get_llm_status()

        # Quick review - no LLM required
        if review_type == "quick":
            return _run_quick_review(
                spec_id=spec_id,
                path=path,
                dry_run=dry_run,
                llm_status=llm_status,
                start_time=start_time,
            )

        # LLM-powered review types (full, security, feasibility)
        return _run_ai_review(
            spec_id=spec_id,
            review_type=review_type,
            ai_provider=ai_provider,
            ai_timeout=ai_timeout,
            consultation_cache=consultation_cache,
            path=path,
            dry_run=dry_run,
            llm_status=llm_status,
            start_time=start_time,
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
            use_provider_integration = _is_provider_integration_enabled()

            if use_provider_integration:
                # Get provider statuses from the provider abstraction layer
                # Note: get_provider_statuses() returns Dict[str, bool]
                provider_statuses = get_provider_statuses()

                # Build tools info from provider statuses
                tools_info = []
                for provider_id, is_available in provider_statuses.items():
                    tools_info.append({
                        "name": provider_id,
                        "available": is_available,
                        "status": "available" if is_available else "unavailable",
                        "reason": None,  # Simple API doesn't provide reason
                        "checked_at": None,  # Simple API doesn't provide timestamp
                    })
            else:
                # Legacy fallback: return static tool list with placeholder availability
                tools_info = [
                    {
                        "name": tool,
                        "available": None,
                        "status": "unknown",
                        "reason": "Legacy mode - external shell check required",
                        "checked_at": None,
                    }
                    for tool in LEGACY_REVIEW_TOOLS
                ]

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
                    provider_integration=use_provider_integration,
                )
            )

        except Exception as e:
            logger.exception("Error listing review tools")
            return asdict(
                error_response(
                    sanitize_error_message(e, context="review tools"),
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
                    sanitize_error_message(e, context="plan tools"),
                )
            )
