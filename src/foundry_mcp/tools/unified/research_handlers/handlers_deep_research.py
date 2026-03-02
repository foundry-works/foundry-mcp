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

        # Load state once for path resolution, provenance, and structured output
        assert research_id is not None  # validated by _DR_REPORT_SCHEMA
        state = memory.load_deep_research(research_id)

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
                if state:
                    state.report_output_path = resolved_path
                    memory.save_deep_research(state)
            except Exception:
                logger.warning("Failed to save report to %s", output_path, exc_info=True)
        else:
            # Fall back to the auto-saved path from synthesis
            resolved_path = metadata.pop("report_output_path", None)
            if not resolved_path and state and state.report_output_path:
                resolved_path = state.report_output_path

        # Build response data with all fields
        response_data: dict[str, Any] = {
            "report": result.content,
            **metadata,
        }
        if resolved_path:
            response_data["output_path"] = resolved_path

        # PLAN-1 Items 2 & 6: Include provenance summary and structured output
        if state is not None:
            if state.provenance is not None:
                response_data["provenance_summary"] = {
                    "entry_count": len(state.provenance.entries),
                    "started_at": state.provenance.started_at,
                    "completed_at": state.provenance.completed_at,
                    "profile": state.provenance.profile,
                }
            if state.extensions.structured_output is not None:
                response_data["structured"] = state.extensions.structured_output.model_dump(
                    mode="json",
                )

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


_VALID_EXPORT_FORMATS = frozenset({"bibtex", "ris"})


def _handle_deep_research_export(
    *,
    research_id: Optional[str] = None,
    export_format: Optional[str] = None,
    academic_only: Optional[bool] = None,
    # Accept legacy name for backward compat but prefer export_format
    format: Optional[str] = None,
    **kwargs: Any,
) -> dict:
    """Export bibliography from a completed research session.

    Args:
        research_id: ID of the deep research session.
        export_format: Export format — ``"bibtex"`` or ``"ris"``.
        academic_only: When True, only export academic sources.

    Returns:
        Response envelope with the exported bibliography string.
    """
    payload = {"research_id": research_id}
    err = validate_payload(payload, _DR_EXPORT_SCHEMA, tool_name="research", action="deep-research-export")
    if err:
        return err

    # Resolve export_format: explicit param > legacy 'format' param > default
    effective_format = export_format or format or "bibtex"
    if effective_format not in _VALID_EXPORT_FORMATS:
        return asdict(
            error_response(
                f"Unknown export format '{effective_format}'. Valid formats: {', '.join(sorted(_VALID_EXPORT_FORMATS))}",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation="Use export_format='bibtex' or export_format='ris'",
            )
        )

    # Default academic_only to True when not provided
    effective_academic_only = academic_only if academic_only is not None else True

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
    if effective_academic_only:
        sources = [s for s in sources if s.source_type == SourceType.ACADEMIC]

    if not sources:
        return asdict(
            success_response(
                data={
                    "research_id": research_id,
                    "format": effective_format,
                    "source_count": 0,
                    "content": "",
                    "message": "No sources to export"
                    + (" (try academic_only=false)" if effective_academic_only else ""),
                }
            )
        )

    # Generate export
    if effective_format == "ris":
        from foundry_mcp.core.research.export.ris import sources_to_ris

        content = sources_to_ris(sources)
    else:
        from foundry_mcp.core.research.export.bibtex import sources_to_bibtex

        content = sources_to_bibtex(sources)

    return asdict(
        success_response(
            data={
                "research_id": research_id,
                "format": effective_format,
                "source_count": len(sources),
                "content": content,
            }
        )
    )


# ---------------------------------------------------------------------------
# PLAN-4 Item 2: Citation Network
# ---------------------------------------------------------------------------

_DR_NETWORK_SCHEMA = {
    "research_id": Str(required=True),
}


def _handle_deep_research_network(
    *,
    research_id: Optional[str] = None,
    max_references_per_paper: Optional[int] = None,
    max_citations_per_paper: Optional[int] = None,
    **kwargs: Any,
) -> dict:
    """Build citation network for a completed research session.

    User-triggered. Requires a completed session with 3+ academic sources
    that have paper IDs. Uses OpenAlex (primary) and Semantic Scholar
    (fallback) to fetch references and forward citations.

    Args:
        research_id: ID of the deep research session.
        max_references_per_paper: Max backward references per source (default from config).
        max_citations_per_paper: Max forward citations per source (default from config).
    """
    import asyncio

    payload = {"research_id": research_id}
    err = validate_payload(
        payload, _DR_NETWORK_SCHEMA, tool_name="research", action="deep-research-network"
    )
    if err:
        return err

    memory = _get_memory()
    config = _get_config()
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

    # Use explicit parameter if provided, otherwise fall back to config.
    # Clamp to [1, 100] to prevent excessive API calls from unbounded input.
    effective_max_refs = (
        max_references_per_paper
        if max_references_per_paper is not None
        else config.research.deep_research_citation_network_max_refs_per_paper
    )
    effective_max_cites = (
        max_citations_per_paper
        if max_citations_per_paper is not None
        else config.research.deep_research_citation_network_max_cites_per_paper
    )
    effective_max_refs = max(1, min(effective_max_refs, 100))
    effective_max_cites = max(1, min(effective_max_cites, 100))

    # Build providers
    openalex_provider = None
    semantic_scholar_provider = None

    try:
        from foundry_mcp.core.research.providers.openalex import OpenAlexProvider

        api_key = config.research.openalex_api_key or None
        if config.research.openalex_enabled:
            openalex_provider = OpenAlexProvider(api_key=api_key)
    except Exception:
        logger.debug("Could not initialize OpenAlex provider for citation network")

    try:
        from foundry_mcp.core.research.providers.semantic_scholar import (
            SemanticScholarProvider,
        )

        # S2 works without a key at lower rate limits
        api_key = config.research.semantic_scholar_api_key or None
        semantic_scholar_provider = SemanticScholarProvider(api_key=api_key)
    except Exception:
        logger.debug("Could not initialize Semantic Scholar provider for citation network")

    if openalex_provider is None and semantic_scholar_provider is None:
        return asdict(
            error_response(
                "No academic providers available for citation network construction",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Ensure OpenAlex or Semantic Scholar is configured",
            )
        )

    from foundry_mcp.core.research.workflows.deep_research.phases.citation_network import (
        CitationNetworkBuilder,
    )

    builder = CitationNetworkBuilder(
        openalex_provider=openalex_provider,
        semantic_scholar_provider=semantic_scholar_provider,
        max_references_per_paper=effective_max_refs,
        max_citations_per_paper=effective_max_cites,
    )

    # Run async builder in event loop
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        network = asyncio.run(builder.build_network(state.sources))
    else:
        # Avoid blocking a running loop by executing in a worker thread.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            network = pool.submit(
                asyncio.run, builder.build_network(state.sources)
            ).result(timeout=120)

    # Save to state
    state.extensions.citation_network = network
    memory.save_deep_research(state)

    # Check if skipped
    if network.stats.get("status") == "skipped":
        return asdict(
            success_response(
                data={
                    "research_id": research_id,
                    "status": "skipped",
                    "reason": network.stats.get("reason", "insufficient academic sources"),
                    "academic_source_count": network.stats.get("academic_source_count", 0),
                }
            )
        )

    return asdict(
        success_response(
            data={
                "research_id": research_id,
                "status": "completed",
                "network": network.model_dump(mode="json"),
            }
        )
    )


# ---------------------------------------------------------------------------
# PLAN-4 Item 3: Methodology Assessment (user-triggered)
# ---------------------------------------------------------------------------

_DR_ASSESS_SCHEMA = {
    "research_id": Str(required=True),
}


def _handle_deep_research_assess(
    *,
    research_id: Optional[str] = None,
    **kwargs: Any,
) -> dict:
    """Run methodology quality assessment on a completed research session.

    User-triggered post-hoc action. Filters to academic sources with
    sufficient content, then uses LLM extraction to assess study design,
    sample size, effect size, limitations, and biases.

    Args:
        research_id: ID of the deep research session.
    """
    import asyncio

    payload = {"research_id": research_id}
    err = validate_payload(
        payload, _DR_ASSESS_SCHEMA, tool_name="research", action="deep-research-assess"
    )
    if err:
        return err

    memory = _get_memory()
    config = _get_config()
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

    # Validate session has sources
    if not state.sources:
        return asdict(
            error_response(
                "Research session has no sources to assess",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation="Run deep-research first to gather sources",
            )
        )

    from foundry_mcp.core.research.workflows.deep_research.phases.methodology_assessment import (
        MethodologyAssessor,
    )

    # Use config for assessment parameters
    assess_provider_id = config.research.deep_research_methodology_assessment_provider
    assess_timeout = config.research.deep_research_methodology_assessment_timeout
    assessor = MethodologyAssessor(
        provider_id=assess_provider_id,
        timeout=assess_timeout,
        min_content_length=config.research.deep_research_methodology_assessment_min_content_length,
    )

    # Check eligibility before running expensive LLM calls
    eligible = assessor.filter_assessable_sources(state.sources)
    if len(eligible) < 2:
        return asdict(
            error_response(
                f"Only {len(eligible)} eligible academic source(s) found (need at least 2)",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation="Session needs 2+ academic sources with sufficient content for methodology assessment",
                details={
                    "research_id": research_id,
                    "eligible_count": len(eligible),
                    "total_sources": len(state.sources),
                },
            )
        )

    # Build standalone LLM call function using provider infrastructure
    # so assess_sources can make LLM calls without a full workflow object.
    async def _llm_call_fn(system_prompt: str, user_prompt: str) -> str | None:
        import asyncio as _aio

        from foundry_mcp.core.providers import (
            ProviderHooks,
            ProviderRequest,
            ProviderStatus,
        )
        from foundry_mcp.core.providers.registry import resolve_provider

        effective_provider_id = assess_provider_id or config.research.default_provider
        try:
            provider = resolve_provider(effective_provider_id, hooks=ProviderHooks())
        except Exception:
            logger.warning(
                "Failed to resolve provider '%s' for methodology assessment",
                effective_provider_id,
                exc_info=True,
            )
            return None

        request = ProviderRequest(
            prompt=user_prompt,
            system_prompt=system_prompt,
            timeout=assess_timeout,
            temperature=0.1,
        )
        # provider.generate is synchronous — run in executor to avoid blocking
        loop = _aio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, provider.generate, request)
        except Exception:
            logger.warning(
                "Methodology assessment provider call failed",
                exc_info=True,
            )
            return None

        if result.status == ProviderStatus.SUCCESS:
            return result.content
        logger.warning(
            "Methodology assessment provider returned status %s",
            result.status.value,
        )
        return None

    # Run async assessor
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        assessments = asyncio.run(
            assessor.assess_sources(state.sources, llm_call_fn=_llm_call_fn)
        )
    else:
        # Avoid blocking a running loop by executing in a worker thread.
        import concurrent.futures

        total_timeout = assess_timeout * len(eligible) + 30
        with concurrent.futures.ThreadPoolExecutor() as pool:
            assessments = pool.submit(
                asyncio.run,
                assessor.assess_sources(state.sources, llm_call_fn=_llm_call_fn),
            ).result(timeout=total_timeout)

    # Save assessments to state
    state.extensions.methodology_assessments = assessments
    memory.save_deep_research(state)

    # Build response
    assessment_summaries = []
    for a in assessments:
        summary: dict[str, Any] = {
            "source_id": a.source_id,
            "study_design": a.study_design.value,
            "confidence": a.confidence,
            "content_basis": a.content_basis,
        }
        if a.sample_size is not None:
            summary["sample_size"] = a.sample_size
        if a.effect_size:
            summary["effect_size"] = a.effect_size
        if a.limitations_noted:
            summary["limitations_count"] = len(a.limitations_noted)
        if a.potential_biases:
            summary["biases_count"] = len(a.potential_biases)
        assessment_summaries.append(summary)

    return asdict(
        success_response(
            data={
                "research_id": research_id,
                "status": "completed",
                "assessment_count": len(assessments),
                "eligible_sources": len(eligible),
                "total_sources": len(state.sources),
                "assessments": assessment_summaries,
            }
        )
    )
