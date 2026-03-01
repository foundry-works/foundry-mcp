"""Deep research lifecycle handlers: start, status, report, list, delete."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from foundry_mcp.core.research.workflows import DeepResearchWorkflow
from foundry_mcp.core.responses.builders import (
    error_response,
    success_response,
)
from foundry_mcp.core.responses.types import (
    ErrorCode,
    ErrorType,
)
from foundry_mcp.tools.unified.param_schema import Str, validate_payload

from ._helpers import _get_config, _get_memory, _validation_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Declarative validation schemas
# ---------------------------------------------------------------------------

_DR_STATUS_SCHEMA = {
    "research_id": Str(required=True),
}

_DR_REPORT_SCHEMA = {
    "research_id": Str(required=True),
}

_DR_DELETE_SCHEMA = {
    "research_id": Str(required=True),
}

_DR_EVALUATE_SCHEMA = {
    "research_id": Str(required=True),
}


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
    research_profile: Optional[str] = None,
    profile_overrides: Optional[dict] = None,
    **kwargs: Any,
) -> dict:
    """Handle deep-research action with background execution.

    Starts deep research in a background thread and returns immediately
    with a ``research_id`` for polling via ``deep-research-status``.
    When the workflow completes, use ``deep-research-report`` to retrieve
    the full report with content-fidelity metadata and allocation warnings.

    Supports:
    - start: Begin new research, return research_id immediately
    - continue: Resume paused research, return research_id immediately
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

    # PLAN-1: Resolve research profile from parameters + config
    resolved_profile = None
    if deep_research_action == "start":
        try:
            resolved_profile = config.research.resolve_profile(
                research_profile=research_profile,
                research_mode=None,  # Legacy mode comes from config.deep_research_mode
                profile_overrides=profile_overrides,
            )
        except ValueError as exc:
            return _validation_error(
                field="research_profile",
                action="deep-research",
                message=str(exc),
                remediation="Use a valid profile name: general, academic, systematic-review, bibliometric, technical",
            )

    # Apply config default for task_timeout if not explicitly set
    # Precedence: explicit param > config > hardcoded fallback
    effective_timeout = task_timeout
    if effective_timeout is None:
        effective_timeout = config.research.deep_research_timeout

    # Execute in background — returns immediately with research_id
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
        background=True,
        task_timeout=effective_timeout,
        research_profile=resolved_profile,
    )

    if result.success:
        # Background mode: return research_id for status polling
        response_data: dict[str, Any] = {
            "status": "started",
            "message": (
                "Deep research started in background. "
                "Use deep-research-status to monitor progress, "
                "then deep-research-report to retrieve results."
            ),
        }
        if result.metadata:
            response_data.update(result.metadata)
        return asdict(success_response(data=response_data))
    else:
        details: dict[str, Any] = {"action": deep_research_action}
        rid = (result.metadata or {}).get("research_id")
        if rid:
            details["research_id"] = rid
        return asdict(
            error_response(
                result.error or "Deep research failed",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check query or research_id validity and provider availability",
                details=details,
            )
        )


def _handle_deep_research_status(
    *,
    research_id: Optional[str] = None,
    wait: bool = True,
    wait_timeout: float = 90.0,
    **kwargs: Any,
) -> dict:
    """Handle deep-research-status action."""
    payload = {"research_id": research_id}
    err = validate_payload(payload, _DR_STATUS_SCHEMA, tool_name="research", action="deep-research-status")
    if err:
        return err

    config = _get_config()
    memory = _get_memory()
    workflow = DeepResearchWorkflow(config.research, memory)

    result = workflow.execute(
        research_id=research_id,
        action="status",
        wait=wait,
        wait_timeout=wait_timeout,
    )

    if result.success:
        # Add next_action guidance — research runs in the background,
        # so the caller should relay progress to the user and poll again.
        status_data = dict(result.metadata) if result.metadata else {}
        research_status = status_data.get("status", "unknown")

        if research_status in ("completed", "failed", "cancelled"):
            status_data["next_action"] = "Research finished. Use deep-research-report to retrieve results."
            # Include report file path if available
            if "report_output_path" not in status_data and research_id:
                state = memory.load_deep_research(research_id)
                if state and state.report_output_path:
                    status_data["report_output_path"] = state.report_output_path
        else:
            status_data["next_action"] = (
                "Research is running in the background. Tell the user about current progress, "
                "then call deep-research-status with wait=true to block until new progress is available."
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
    output_path: Optional[str] = None,
    **kwargs: Any,
) -> dict:
    """Handle deep-research-report action."""
    payload = {"research_id": research_id}
    err = validate_payload(payload, _DR_REPORT_SCHEMA, tool_name="research", action="deep-research-report")
    if err:
        return err

    config = _get_config()
    memory = _get_memory()
    workflow = DeepResearchWorkflow(config.research, memory)

    result = workflow.execute(
        research_id=research_id,
        action="report",
    )

    if result.success:
        # Extract warnings from metadata for routing to meta.warnings
        metadata = result.metadata or {}
        warnings = metadata.pop("warnings", None)

        # Determine report file path
        resolved_path: Optional[str] = None

        if output_path and result.content:
            # User requested a custom output path — save/override there
            try:
                p = Path(output_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(result.content, encoding="utf-8")
                resolved_path = str(p)

                # Update state so future calls reflect the new path
                assert research_id is not None
                state = memory.load_deep_research(research_id)
                if state:
                    state.report_output_path = resolved_path
                    memory.save_deep_research(state)
            except Exception:
                logger.warning("Failed to save report to %s", output_path, exc_info=True)
        else:
            # Fall back to the auto-saved path from synthesis
            resolved_path = metadata.pop("report_output_path", None)
            if not resolved_path and research_id:
                state = memory.load_deep_research(research_id)
                if state and state.report_output_path:
                    resolved_path = state.report_output_path

        # Build response data with all fields
        response_data: dict[str, Any] = {
            "report": result.content,
            **metadata,
        }
        if resolved_path:
            response_data["output_path"] = resolved_path

        # PLAN-1 Item 2: Include provenance summary in report response
        if research_id:
            state = memory.load_deep_research(research_id) if not resolved_path else None
            # Try to load state if we don't already have it
            if state is None and research_id:
                state = memory.load_deep_research(research_id)
            if state and state.provenance is not None:
                response_data["provenance_summary"] = {
                    "entry_count": len(state.provenance.entries),
                    "started_at": state.provenance.started_at,
                    "completed_at": state.provenance.completed_at,
                    "profile": state.provenance.profile,
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


def _handle_deep_research_evaluate(
    *,
    research_id: Optional[str] = None,
    **kwargs: Any,
) -> dict:
    """Handle deep-research-evaluate action (LLM-as-judge quality scoring)."""
    payload = {"research_id": research_id}
    err = validate_payload(payload, _DR_EVALUATE_SCHEMA, tool_name="research", action="deep-research-evaluate")
    if err:
        return err

    config = _get_config()
    workflow = DeepResearchWorkflow(config.research, _get_memory())

    result = workflow.execute(
        research_id=research_id,
        action="evaluate",
    )

    if result.success:
        return asdict(success_response(data=dict(result.metadata) if result.metadata else {}))
    else:
        return asdict(
            error_response(
                result.error or "Evaluation failed",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Ensure research is complete with a generated report",
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
    payload = {"research_id": research_id}
    err = validate_payload(payload, _DR_DELETE_SCHEMA, tool_name="research", action="deep-research-delete")
    if err:
        return err

    config = _get_config()
    workflow = DeepResearchWorkflow(config.research, _get_memory())

    assert research_id is not None  # validated by _DR_DELETE_SCHEMA
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


# ---------------------------------------------------------------------------
# PLAN-1 Item 2: Provenance audit trail
# ---------------------------------------------------------------------------

_DR_PROVENANCE_SCHEMA = {
    "research_id": Str(required=True),
}


def _handle_deep_research_provenance(
    *,
    research_id: Optional[str] = None,
    **kwargs: Any,
) -> dict:
    """Retrieve the provenance audit trail for a deep research session.

    Returns the full provenance log with all events, or a summary if the
    session has no provenance (pre-PLAN-1 sessions).
    """
    payload = {"research_id": research_id}
    err = validate_payload(
        payload, _DR_PROVENANCE_SCHEMA, tool_name="research", action="deep-research-provenance"
    )
    if err:
        return err

    memory = _get_memory()
    assert research_id is not None  # validated by schema
    state = memory.load_deep_research(research_id)

    if state is None:
        return asdict(
            error_response(
                f"Research session '{research_id}' not found",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Use deep-research-list to find valid research IDs",
            )
        )

    if state.provenance is None:
        return asdict(
            success_response(
                data={
                    "research_id": research_id,
                    "provenance": None,
                    "message": "No provenance data available (session predates provenance feature)",
                }
            )
        )

    return asdict(
        success_response(
            data={
                "research_id": research_id,
                "provenance": state.provenance.model_dump(mode="json"),
            }
        )
    )


# ---------------------------------------------------------------------------
# PLAN-3: BibTeX / RIS export
# ---------------------------------------------------------------------------

_DR_EXPORT_SCHEMA = {
    "research_id": Str(required=True),
}


def _handle_deep_research_export(
    *,
    research_id: Optional[str] = None,
    format: str = "bibtex",
    academic_only: bool = True,
    **kwargs: Any,
) -> dict:
    """Export bibliography from a completed research session.

    Args:
        research_id: ID of the deep research session.
        format: Export format — ``"bibtex"`` or ``"ris"``.
        academic_only: When True, only export academic sources.

    Returns:
        Response envelope with the exported bibliography string.
    """
    payload = {"research_id": research_id}
    err = validate_payload(payload, _DR_EXPORT_SCHEMA, tool_name="research", action="deep-research-export")
    if err:
        return err

    memory = _get_memory()
    assert research_id is not None  # validated by _DR_EXPORT_SCHEMA
    state = memory.load_deep_research(research_id)

    if state is None:
        return asdict(
            error_response(
                f"Research session '{research_id}' not found",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Use deep-research-list to find valid research IDs",
            )
        )

    from foundry_mcp.core.research.models.sources import SourceType

    # Filter sources
    sources = state.sources
    if academic_only:
        sources = [s for s in sources if s.source_type == SourceType.ACADEMIC]

    if not sources:
        return asdict(
            success_response(
                data={
                    "research_id": research_id,
                    "format": format,
                    "source_count": 0,
                    "content": "",
                    "message": "No sources to export"
                    + (" (try academic_only=false)" if academic_only else ""),
                }
            )
        )

    # Generate export
    if format == "ris":
        from foundry_mcp.core.research.export.ris import sources_to_ris

        content = sources_to_ris(sources)
    else:
        from foundry_mcp.core.research.export.bibtex import sources_to_bibtex

        content = sources_to_bibtex(sources)

    return asdict(
        success_response(
            data={
                "research_id": research_id,
                "format": format,
                "source_count": len(sources),
                "content": content,
            }
        )
    )
