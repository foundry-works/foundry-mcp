"""Shared LLM call lifecycle helpers for deep research phase mixins.

Extracts the common boilerplate around LLM provider calls: heartbeat updates,
audit events, ContextWindowError handling, metrics emission, token tracking,
and PhaseMetrics recording. Each phase mixin calls these helpers instead of
duplicating ~88 lines of lifecycle code.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from foundry_mcp.core.errors.provider import ContextWindowError
from foundry_mcp.core.observability import get_metrics
from foundry_mcp.core.research.models.deep_research import DeepResearchState
from foundry_mcp.core.research.models.fidelity import PhaseMetrics
from foundry_mcp.core.research.workflows.base import WorkflowResult

logger = logging.getLogger(__name__)


@dataclass
class LLMCallResult:
    """Result of a successful LLM call with provider metadata."""

    result: WorkflowResult
    llm_call_duration_ms: float


async def execute_llm_call(
    workflow: Any,
    state: DeepResearchState,
    phase_name: str,
    system_prompt: str,
    user_prompt: str,
    provider_id: Optional[str],
    model: Optional[str],
    temperature: float,
    timeout: float,
    error_metadata: Optional[dict[str, Any]] = None,
) -> LLMCallResult | WorkflowResult:
    """Execute an LLM call with full lifecycle instrumentation.

    Handles: heartbeat update, state persistence, audit events (started/completed),
    provider call with ContextWindowError handling, metrics emission, timeout/failure
    check, token tracking, and PhaseMetrics recording.

    Args:
        workflow: The DeepResearchWorkflow instance (provides config, memory, etc.)
        state: Current research state
        phase_name: Phase identifier (e.g. "planning", "analysis")
        system_prompt: System prompt for the LLM call
        user_prompt: User prompt for the LLM call
        provider_id: Explicit provider ID (may be None for phase default)
        model: Model override for the provider
        temperature: Sampling temperature
        timeout: Request timeout in seconds
        error_metadata: Extra fields to include in ContextWindowError response metadata

    Returns:
        LLMCallResult on success (caller uses .result for the WorkflowResult),
        or WorkflowResult directly on error (ContextWindowError, timeout, failure).
        Callers use ``isinstance(ret, WorkflowResult)`` to branch on error.
    """
    effective_provider = provider_id

    # Heartbeat + persist
    llm_call_start_time = time.perf_counter()
    state.last_heartbeat_at = datetime.now(timezone.utc)
    workflow.memory.save_deep_research(state)

    # Audit: llm.call.started
    workflow._write_audit_event(
        state,
        "llm.call.started",
        data={
            "provider": effective_provider,
            "task_id": state.id,
            "phase": phase_name,
        },
    )

    # Provider call with ContextWindowError handling
    try:
        result = await workflow._execute_provider_async(
            prompt=user_prompt,
            provider_id=effective_provider,
            model=model,
            system_prompt=system_prompt,
            timeout=timeout,
            temperature=temperature,
            phase=phase_name,
            fallback_providers=workflow.config.get_phase_fallback_providers(phase_name),
            max_retries=workflow.config.deep_research_max_retries,
            retry_delay=workflow.config.deep_research_retry_delay,
        )
    except ContextWindowError as e:
        llm_call_duration_ms = (time.perf_counter() - llm_call_start_time) * 1000

        # Audit + metrics for error
        workflow._write_audit_event(
            state,
            "llm.call.completed",
            data={
                "provider": effective_provider,
                "task_id": state.id,
                "duration_ms": llm_call_duration_ms,
                "status": "error",
                "error_type": "context_window_exceeded",
            },
        )
        get_metrics().histogram(
            "foundry_mcp_research_llm_call_duration_seconds",
            llm_call_duration_ms / 1000.0,
            labels={"provider": effective_provider or "unknown", "status": "error"},
        )

        logger.error(
            "%s phase context window exceeded: prompt_tokens=%s, max_tokens=%s, truncation_needed=%s, provider=%s",
            phase_name.capitalize(),
            e.prompt_tokens,
            e.max_tokens,
            e.truncation_needed,
            e.provider,
        )

        metadata: dict[str, Any] = {
            "research_id": state.id,
            "phase": phase_name,
            "error_type": "context_window_exceeded",
            "prompt_tokens": e.prompt_tokens,
            "max_tokens": e.max_tokens,
            "truncation_needed": e.truncation_needed,
        }
        if error_metadata:
            metadata.update(error_metadata)

        return WorkflowResult(
            success=False,
            content="",
            error=str(e),
            metadata=metadata,
        )

    # Audit + metrics for completion
    llm_call_duration_ms = (time.perf_counter() - llm_call_start_time) * 1000
    llm_call_status = "success" if result.success else "error"
    llm_call_provider: str = result.provider_id or effective_provider or "unknown"

    workflow._write_audit_event(
        state,
        "llm.call.completed",
        data={
            "provider": llm_call_provider,
            "task_id": state.id,
            "duration_ms": llm_call_duration_ms,
            "status": llm_call_status,
        },
    )
    get_metrics().histogram(
        "foundry_mcp_research_llm_call_duration_seconds",
        llm_call_duration_ms / 1000.0,
        labels={"provider": llm_call_provider, "status": llm_call_status},
    )

    # Failure early return
    if not result.success:
        if result.metadata and result.metadata.get("timeout"):
            logger.error(
                "%s phase timed out after exhausting all providers: %s",
                phase_name.capitalize(),
                result.metadata.get("providers_tried", []),
            )
        else:
            logger.error("%s phase LLM call failed: %s", phase_name.capitalize(), result.error)
        return result

    # Token tracking
    if result.tokens_used:
        state.total_tokens_used += result.tokens_used

    # Phase metrics
    state.phase_metrics.append(
        PhaseMetrics(
            phase=phase_name,
            duration_ms=result.duration_ms or 0.0,
            input_tokens=result.input_tokens or 0,
            output_tokens=result.output_tokens or 0,
            cached_tokens=result.cached_tokens or 0,
            provider_id=result.provider_id,
            model_used=result.model_used,
        )
    )

    return LLMCallResult(result=result, llm_call_duration_ms=llm_call_duration_ms)


def finalize_phase(
    workflow: Any,
    state: DeepResearchState,
    phase_name: str,
    phase_start_time: float,
) -> None:
    """Emit phase.completed audit event and duration metric.

    Args:
        workflow: The DeepResearchWorkflow instance
        state: Current research state
        phase_name: Phase identifier (e.g. "planning", "analysis")
        phase_start_time: Value from ``time.perf_counter()`` at phase start
    """
    phase_duration_ms = (time.perf_counter() - phase_start_time) * 1000

    workflow._write_audit_event(
        state,
        "phase.completed",
        data={
            "phase_name": phase_name,
            "iteration": state.iteration,
            "task_id": state.id,
            "duration_ms": phase_duration_ms,
        },
    )

    get_metrics().histogram(
        "foundry_mcp_research_phase_duration_seconds",
        phase_duration_ms / 1000.0,
        labels={"phase_name": phase_name, "status": "success"},
    )
