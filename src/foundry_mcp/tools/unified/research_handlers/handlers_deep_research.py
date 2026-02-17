"""Deep research lifecycle handlers: start, status, report, list, delete."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

from foundry_mcp.core.research.workflows import DeepResearchWorkflow
from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    error_response,
    success_response,
)

from ._helpers import _get_config, _get_memory, _validation_error


def _handle_deep_research(
    *,
    query: Optional[str] = None,
    research_id: Optional[str] = None,
    deep_research_action: str = "start",
    provider_id: Optional[str] = None,
    system_prompt: Optional[str] = None,
    max_iterations: int = 3,
    max_sub_queries: int = 5,
    max_sources_per_query: int = 5,
    follow_links: bool = True,
    timeout_per_operation: float = 120.0,
    max_concurrent: int = 3,
    task_timeout: Optional[float] = None,
    **kwargs: Any,
) -> dict:
    """Handle deep-research action with background execution.

    CRITICAL: This handler uses asyncio.create_task() via the workflow's
    background mode to start research and return immediately with the
    research_id. The workflow runs in the background and can be polled
    via deep-research-status.

    Supports:
    - start: Begin new research, returns immediately with research_id
    - continue: Resume paused research in background
    - resume: Alias for continue (for backward compatibility)
    """
    # Normalize 'resume' to 'continue' for workflow compatibility
    if deep_research_action == "resume":
        deep_research_action = "continue"

    # Validate based on action
    if deep_research_action == "start" and not query:
        return _validation_error(
            field="query",
            action="deep-research",
            message="Query is required to start deep research",
            remediation="Provide a research query to investigate",
        )

    if deep_research_action in ("continue",) and not research_id:
        return _validation_error(
            field="research_id",
            action="deep-research",
            message=f"research_id is required for '{deep_research_action}' action",
            remediation="Use deep-research-list to find existing research sessions",
        )

    config = _get_config()
    workflow = DeepResearchWorkflow(config.research, _get_memory())

    # Apply config default for task_timeout if not explicitly set
    # Precedence: explicit param > config > hardcoded fallback (600s)
    effective_timeout = task_timeout
    if effective_timeout is None:
        effective_timeout = config.research.deep_research_timeout

    # Execute with background=True for non-blocking execution
    # This uses asyncio.create_task() internally and returns immediately
    result = workflow.execute(
        query=query,
        research_id=research_id,
        action=deep_research_action,
        provider_id=provider_id,
        system_prompt=system_prompt,
        max_iterations=max_iterations,
        max_sub_queries=max_sub_queries,
        max_sources_per_query=max_sources_per_query,
        follow_links=follow_links,
        timeout_per_operation=timeout_per_operation,
        max_concurrent=max_concurrent,
        background=True,  # CRITICAL: Run in background, return immediately
        task_timeout=effective_timeout,
    )

    if result.success:
        # For background execution, return started status with research_id
        response_data = {
            "research_id": result.metadata.get("research_id"),
            "status": "started",
            "effective_timeout": effective_timeout,
            "message": (
                "Deep research started. This typically takes 3-5 minutes. "
                "IMPORTANT: Communicate progress to user before each status check. "
                "Maximum 5 status checks allowed. "
                "Do NOT use WebSearch/WebFetch while this research is running."
            ),
            "polling_guidance": {
                "max_checks": 5,
                "typical_duration_minutes": 5,
                "require_user_communication": True,
                "no_independent_research": True,
            },
        }

        # Include additional metadata if available (for continue/resume)
        if result.metadata.get("phase"):
            response_data["phase"] = result.metadata.get("phase")
        if result.metadata.get("iteration") is not None:
            response_data["iteration"] = result.metadata.get("iteration")

        return asdict(success_response(data=response_data))
    else:
        return asdict(
            error_response(
                result.error or "Deep research failed to start",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check query or research_id validity and provider availability",
                details={"action": deep_research_action},
            )
        )


def _handle_deep_research_status(
    *,
    research_id: Optional[str] = None,
    **kwargs: Any,
) -> dict:
    """Handle deep-research-status action."""
    if not research_id:
        return _validation_error(field="research_id", action="deep-research-status", message="Required")

    config = _get_config()
    workflow = DeepResearchWorkflow(config.research, _get_memory())

    result = workflow.execute(
        research_id=research_id,
        action="status",
    )

    if result.success:
        # Add next_action guidance based on check count
        status_data = dict(result.metadata) if result.metadata else {}
        check_count = status_data.get("status_check_count", 1)
        checks_remaining = max(0, 5 - check_count)

        if checks_remaining > 0:
            status_data["next_action"] = (
                f"BEFORE next check: Tell user about progress. {checks_remaining} checks remaining."
            )
        else:
            status_data["next_action"] = (
                "Max checks reached. Offer user options: wait, background, or cancel."
            )

        return asdict(success_response(data=status_data))
    else:
        return asdict(
            error_response(
                result.error or "Failed to get status",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Use deep-research-list to find valid research IDs",
            )
        )


def _handle_deep_research_report(
    *,
    research_id: Optional[str] = None,
    **kwargs: Any,
) -> dict:
    """Handle deep-research-report action."""
    if not research_id:
        return _validation_error(field="research_id", action="deep-research-report", message="Required")

    config = _get_config()
    workflow = DeepResearchWorkflow(config.research, _get_memory())

    result = workflow.execute(
        research_id=research_id,
        action="report",
    )

    if result.success:
        # Extract warnings from metadata for routing to meta.warnings
        metadata = result.metadata or {}
        warnings = metadata.pop("warnings", None)

        # Build response data with all fields
        response_data = {
            "report": result.content,
            **metadata,
        }

        return asdict(
            success_response(
                data=response_data,
                warnings=warnings,  # Route warnings to meta.warnings
            )
        )
    else:
        return asdict(
            error_response(
                result.error or "Failed to get report",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Ensure research is complete or use deep-research-status to check",
            )
        )


def _handle_deep_research_list(
    *,
    limit: int = 50,
    cursor: Optional[str] = None,
    completed_only: bool = False,
    **kwargs: Any,
) -> dict:
    """Handle deep-research-list action."""
    config = _get_config()
    workflow = DeepResearchWorkflow(config.research, _get_memory())

    sessions = workflow.list_sessions(
        limit=limit,
        cursor=cursor,
        completed_only=completed_only,
    )

    # Build response with pagination support
    response_data: dict[str, Any] = {
        "sessions": sessions,
        "count": len(sessions),
    }

    # Include next cursor if there are more results
    if sessions and len(sessions) == limit:
        # Use last session's ID as cursor for next page
        response_data["next_cursor"] = sessions[-1].get("id")

    return asdict(success_response(data=response_data))


def _handle_deep_research_delete(
    *,
    research_id: Optional[str] = None,
    **kwargs: Any,
) -> dict:
    """Handle deep-research-delete action."""
    if not research_id:
        return _validation_error(field="research_id", action="deep-research-delete", message="Required")

    config = _get_config()
    workflow = DeepResearchWorkflow(config.research, _get_memory())

    deleted = workflow.delete_session(research_id)

    if not deleted:
        return asdict(
            error_response(
                f"Research session '{research_id}' not found",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Use deep-research-list to find valid research IDs",
            )
        )

    return asdict(
        success_response(
            data={
                "deleted": True,
                "research_id": research_id,
            }
        )
    )
