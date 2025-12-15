"""Shared review helpers for the unified review tool.

This module centralizes the reusable building blocks used by
`foundry_mcp.tools.unified.review` so the main router stays focused on
input validation and action dispatch.

The implementation is adapted from the legacy CLI review command helpers.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    ai_no_provider_error,
    ai_provider_error,
    ai_provider_timeout_error,
    error_response,
    success_response,
)

logger = logging.getLogger(__name__)


# Default AI consultation timeout
DEFAULT_AI_TIMEOUT = 120.0

# Review types supported by the unified `review(action="spec")` entrypoint.
REVIEW_TYPES = ["quick", "full", "security", "feasibility"]

# Map review types to PLAN_REVIEW templates
_REVIEW_TYPE_TO_TEMPLATE = {
    "full": "PLAN_REVIEW_FULL_V1",
    "security": "PLAN_REVIEW_SECURITY_V1",
    "feasibility": "PLAN_REVIEW_FEASIBILITY_V1",
}


def _get_llm_status() -> Dict[str, Any]:
    """Get LLM configuration status.

    This is a lightweight wrapper around `foundry_mcp.core.review.get_llm_status`
    that normalizes exception handling for tool surfaces.
    """

    try:
        from foundry_mcp.core.review import get_llm_status

        return get_llm_status()
    except Exception as exc:
        logger.debug("Failed to get LLM status: %s", exc)
        return {"configured": False, "error": "Failed to load LLM configuration"}


def _run_quick_review(
    *,
    spec_id: str,
    specs_dir: Optional[Path],
    dry_run: bool,
    llm_status: Dict[str, Any],
    start_time: float,
) -> dict:
    if dry_run:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return asdict(
            success_response(
                spec_id=spec_id,
                review_type="quick",
                dry_run=True,
                llm_status=llm_status,
                message="Dry run - quick review skipped",
                telemetry={"duration_ms": round(duration_ms, 2)},
            )
        )

    try:
        from foundry_mcp.core.review import quick_review

        result = quick_review(spec_id=spec_id, specs_dir=specs_dir)
    except Exception as exc:
        logger.exception("Quick review failed")
        return asdict(
            error_response(
                f"Quick review failed: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check logs for details and retry.",
            )
        )

    duration_ms = (time.perf_counter() - start_time) * 1000

    payload = asdict(result)
    payload["llm_status"] = llm_status

    return asdict(
        success_response(
            **payload,
            telemetry={"duration_ms": round(duration_ms, 2)},
        )
    )


def _run_ai_review(
    *,
    spec_id: str,
    specs_dir: Optional[Path],
    review_type: str,
    ai_provider: Optional[str],
    model: Optional[str],
    ai_timeout: float,
    consultation_cache: bool,
    dry_run: bool,
    llm_status: Dict[str, Any],
    start_time: float,
) -> dict:
    template_id = _REVIEW_TYPE_TO_TEMPLATE.get(review_type)
    if template_id is None:
        return asdict(
            error_response(
                f"Unknown review type: {review_type}",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation=f"Use one of: {', '.join(_REVIEW_TYPE_TO_TEMPLATE.keys())}",
                data={"review_type": review_type},
            )
        )

    try:
        from foundry_mcp.core.review import prepare_review_context

        context = prepare_review_context(
            spec_id=spec_id,
            specs_dir=specs_dir,
            include_tasks=True,
            include_journals=True,
        )
    except Exception as exc:
        logger.exception("Failed preparing review context")
        return asdict(
            error_response(
                f"Failed preparing review context: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check logs for details and retry.",
            )
        )

    if context is None:
        return asdict(
            error_response(
                f"Specification '{spec_id}' not found",
                error_code=ErrorCode.SPEC_NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Verify the spec_id and that the spec exists.",
                data={"spec_id": spec_id},
            )
        )

    if dry_run:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return asdict(
            success_response(
                spec_id=spec_id,
                title=context.title,
                review_type=review_type,
                template_id=template_id,
                dry_run=True,
                llm_status=llm_status,
                ai_provider=ai_provider,
                model=model,
                consultation_cache=consultation_cache,
                message=f"Dry run - {review_type} review would use template {template_id}",
                stats={
                    "total_tasks": context.stats.totals.get("tasks", 0)
                    if context.stats
                    else 0,
                },
                telemetry={"duration_ms": round(duration_ms, 2)},
            )
        )

    try:
        from foundry_mcp.core.ai_consultation import (
            ConsultationOrchestrator,
            ConsultationRequest,
            ConsultationWorkflow,
        )
    except ImportError:
        return asdict(
            error_response(
                "AI consultation layer not available",
                error_code=ErrorCode.UNAVAILABLE,
                error_type=ErrorType.UNAVAILABLE,
                remediation="Ensure foundry_mcp.core.ai_consultation is installed.",
            )
        )

    orchestrator = ConsultationOrchestrator(default_timeout=ai_timeout)

    if not orchestrator.is_available(provider_id=ai_provider):
        return asdict(
            ai_no_provider_error(
                "AI-enhanced review requested but no providers available",
                required_providers=[ai_provider] if ai_provider else None,
            )
        )

    spec_content = json.dumps(context.spec_data, indent=2)

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
        model=model,
        timeout=ai_timeout,
    )

    try:
        result = orchestrator.consult(request, use_cache=consultation_cache)
    except Exception as exc:
        error_lower = str(exc).lower()
        if "timeout" in error_lower or "timed out" in error_lower:
            return asdict(
                ai_provider_timeout_error(
                    ai_provider or "unknown",
                    int(ai_timeout),
                )
            )

        return asdict(ai_provider_error(ai_provider or "unknown", str(exc)))

    from foundry_mcp.core.ai_consultation import ConsensusResult

    duration_ms = (time.perf_counter() - start_time) * 1000

    is_consensus = isinstance(result, ConsensusResult)
    if is_consensus:
        primary = (
            result.successful_responses[0] if result.successful_responses else None
        )
        provider_id = primary.provider_id if primary else None
        model_used = primary.model_used if primary else None
        cached = bool(primary.cache_hit) if primary else False
        content = result.primary_content
        consensus = {
            "mode": "multi_model",
            "total_providers": len(result.responses),
            "successful_providers": len(result.successful_responses),
            "failed_providers": len(result.failed_responses),
        }
    else:
        provider_id = getattr(result, "provider_id", ai_provider)
        model_used = getattr(result, "model_used", None)
        cached = bool(getattr(result, "cache_hit", False))
        content = getattr(result, "content", None)
        consensus = {"mode": "single_model"}

    total_tasks = context.stats.totals.get("tasks", 0) if context.stats else 0
    completed_tasks = (
        context.stats.status_counts.get("completed", 0) if context.stats else 0
    )

    return asdict(
        success_response(
            spec_id=spec_id,
            title=context.title,
            review_type=review_type,
            template_id=template_id,
            llm_status=llm_status,
            ai_provider=provider_id,
            model=model_used,
            consultation_cache=consultation_cache,
            response=content,
            cached=cached,
            consensus=consensus,
            stats={
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "progress_percentage": context.progress.get("percentage", 0)
                if context.progress
                else 0,
            },
            telemetry={"duration_ms": round(duration_ms, 2)},
        )
    )
