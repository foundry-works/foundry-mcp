"""Shared LLM call lifecycle helpers for deep research phase mixins.

Extracts the common boilerplate around LLM provider calls: heartbeat updates,
audit events, ContextWindowError handling, metrics emission, token tracking,
and PhaseMetrics recording. Each phase mixin calls these helpers instead of
duplicating ~88 lines of lifecycle code.
"""

from __future__ import annotations

import logging
import re
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

# ---------------------------------------------------------------------------
# Provider-specific context-window error detection
# ---------------------------------------------------------------------------

#: Regex patterns matched against exception messages to detect context-window
#: errors that providers raise as generic ``BadRequestError`` or similar.
#: Each entry is ``(provider_hint, pattern)``.  When the exception class name
#: **and** message match, the error is re-classified as a ContextWindowError.
_CONTEXT_WINDOW_ERROR_PATTERNS: list[tuple[str, str]] = [
    # OpenAI / Codex: BadRequestError with token/context/length keywords
    ("openai", r"(?i)\b(?:token|context|length|maximum.*context)\b"),
    # Anthropic: BadRequestError with "prompt is too long"
    ("anthropic", r"(?i)prompt\s+is\s+too\s+long"),
    # Google: ResourceExhausted exception type (matched by class name)
    ("google", r"(?i)\b(?:resource\s*exhausted|context\s*length|token\s*limit)\b"),
]

#: Exception class names that are known context-window indicators for
#: specific providers, regardless of message content.
_CONTEXT_WINDOW_ERROR_CLASSES: set[str] = {
    "ResourceExhausted",  # Google / gRPC
}


def _is_context_window_error(exc: Exception) -> bool:
    """Detect if a generic exception is actually a context-window overflow.

    Checks the exception's class name and message against known provider
    patterns.  Returns ``True`` if the exception should be treated as a
    ``ContextWindowError`` for progressive-truncation recovery.

    This catches errors that the provider layer raises as generic
    ``BadRequestError`` (OpenAI, Anthropic) or ``ResourceExhausted``
    (Google) instead of the canonical ``ContextWindowError``.
    """
    cls_name = type(exc).__name__

    # Fast path: known class names
    if cls_name in _CONTEXT_WINDOW_ERROR_CLASSES:
        return True

    msg = str(exc)

    for _provider_hint, pattern in _CONTEXT_WINDOW_ERROR_PATTERNS:
        if re.search(pattern, msg):
            return True

    return False


@dataclass
class LLMCallResult:
    """Result of a successful LLM call with provider metadata."""

    result: WorkflowResult
    llm_call_duration_ms: float


# Maximum number of progressive truncation retries on context-window errors.
_MAX_TOKEN_LIMIT_RETRIES: int = 3

# Each retry truncates the user prompt to this fraction of the previous size.
_TRUNCATION_FACTOR: float = 0.9  # keep 90%, remove 10%

# Fallback context-window size when neither the error nor the model registry
# provides a concrete limit.  128K tokens is a conservative default.
_FALLBACK_CONTEXT_WINDOW: int = 128_000


def _truncate_for_retry(
    user_prompt: str,
    error_max_tokens: Optional[int],
    model: Optional[str],
    retry_count: int,
    truncate_fn: Any,
    estimate_limit_fn: Any,
    token_limits: dict[str, int],
) -> str:
    """Compute a truncated user prompt for a token-limit retry.

    Determines the token budget from (in order): the error's max_tokens,
    the model registry, or the fallback default. Then applies progressive
    reduction (90% per retry) and truncates at a natural boundary.
    """
    max_tokens = error_max_tokens
    if max_tokens is None:
        max_tokens = estimate_limit_fn(model, token_limits)
    if max_tokens is None:
        max_tokens = _FALLBACK_CONTEXT_WINDOW

    reduced_budget = int(max_tokens * (_TRUNCATION_FACTOR ** retry_count))
    return truncate_fn(user_prompt, reduced_budget)


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
    role: Optional[str] = None,
) -> LLMCallResult | WorkflowResult:
    """Execute an LLM call with full lifecycle instrumentation.

    Handles: heartbeat update, state persistence, audit events (started/completed),
    provider call with ContextWindowError handling (including progressive
    truncation recovery), metrics emission, timeout/failure check, token
    tracking, and PhaseMetrics recording.

    **Progressive token-limit recovery:** When a ``ContextWindowError`` (or a
    provider-specific equivalent detected by ``_is_context_window_error``) is
    raised, the user prompt is truncated by 10% and the call is retried, up to
    3 times.  Only the user prompt is truncated — the system prompt is never
    modified.  If all retries are exhausted the original hard-error path is
    taken.

    **Role-based model resolution (Phase 6):** When *role* is provided and
    *provider_id* / *model* are ``None``, the provider and model are resolved
    from ``workflow.config.resolve_model_for_role(role)``.  Explicit
    *provider_id* / *model* values always take precedence over role-based
    resolution.

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
        role: Model role for cost-optimized routing (e.g. "research", "report",
            "reflection", "summarization", "compression", "clarification").
            When set, resolves provider/model from config if not explicitly
            provided.

    Returns:
        LLMCallResult on success (caller uses .result for the WorkflowResult),
        or WorkflowResult directly on error (ContextWindowError, timeout, failure).
        Callers use ``isinstance(ret, WorkflowResult)`` to branch on error.
    """
    from foundry_mcp.core.research.providers.base import SearchProvider
    from foundry_mcp.core.research.workflows.deep_research._helpers import (
        estimate_token_limit_for_model,
        truncate_to_token_estimate,
    )

    # Role-based model resolution (Phase 6): when role is provided and
    # provider_id / model are not explicitly set, resolve from config.
    if role and hasattr(workflow, "config") and hasattr(workflow.config, "resolve_model_for_role"):
        role_provider, role_model = workflow.config.resolve_model_for_role(role)
        if provider_id is None:
            provider_id = role_provider
        if model is None:
            model = role_model

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

    # ------------------------------------------------------------------
    # Provider call with progressive token-limit recovery
    # ------------------------------------------------------------------
    current_user_prompt = user_prompt
    token_limit_retries = 0
    result: Optional[WorkflowResult] = None
    last_context_error: ContextWindowError | Exception | None = None

    for attempt in range(_MAX_TOKEN_LIMIT_RETRIES + 1):  # 0 = initial, 1-3 = retries
        try:
            result = await workflow._execute_provider_async(
                prompt=current_user_prompt,
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
            # Success — clear any prior context error and break
            last_context_error = None
            break

        except ContextWindowError as e:
            last_context_error = e
            if attempt >= _MAX_TOKEN_LIMIT_RETRIES:
                break  # All retries exhausted

            token_limit_retries += 1
            current_user_prompt = _truncate_for_retry(
                current_user_prompt, e.max_tokens, model, token_limit_retries,
                truncate_to_token_estimate, estimate_token_limit_for_model,
                SearchProvider.TOKEN_LIMITS,
            )

            logger.warning(
                "%s phase context window exceeded (attempt %d/%d), "
                "truncating user prompt and retrying. "
                "prompt_tokens=%s, max_tokens=%s, provider=%s",
                phase_name.capitalize(),
                token_limit_retries,
                _MAX_TOKEN_LIMIT_RETRIES,
                e.prompt_tokens,
                e.max_tokens,
                e.provider,
            )

        except Exception as e:
            if _is_context_window_error(e) and attempt < _MAX_TOKEN_LIMIT_RETRIES:
                last_context_error = e
                token_limit_retries += 1
                current_user_prompt = _truncate_for_retry(
                    current_user_prompt, None, model, token_limit_retries,
                    truncate_to_token_estimate, estimate_token_limit_for_model,
                    SearchProvider.TOKEN_LIMITS,
                )

                logger.warning(
                    "%s phase detected provider-specific context window error "
                    "(attempt %d/%d), truncating user prompt. "
                    "error_class=%s, message=%s",
                    phase_name.capitalize(),
                    token_limit_retries,
                    _MAX_TOKEN_LIMIT_RETRIES,
                    type(e).__name__,
                    str(e)[:200],
                )
            else:
                raise

    # ------------------------------------------------------------------
    # All retries exhausted — emit hard error
    # ------------------------------------------------------------------
    if last_context_error is not None:
        llm_call_duration_ms = (time.perf_counter() - llm_call_start_time) * 1000

        prompt_tokens = getattr(last_context_error, "prompt_tokens", None)
        max_tokens_val = getattr(last_context_error, "max_tokens", None)
        truncation_needed = getattr(last_context_error, "truncation_needed", None)
        error_provider = getattr(last_context_error, "provider", None)

        workflow._write_audit_event(
            state,
            "llm.call.completed",
            data={
                "provider": effective_provider,
                "task_id": state.id,
                "duration_ms": llm_call_duration_ms,
                "status": "error",
                "error_type": "context_window_exceeded",
                "token_limit_retries": token_limit_retries,
            },
        )
        get_metrics().histogram(
            "foundry_mcp_research_llm_call_duration_seconds",
            llm_call_duration_ms / 1000.0,
            labels={"provider": effective_provider or "unknown", "status": "error"},
        )

        logger.error(
            "%s phase context window exceeded after %d retries: "
            "prompt_tokens=%s, max_tokens=%s, truncation_needed=%s, provider=%s",
            phase_name.capitalize(),
            token_limit_retries,
            prompt_tokens,
            max_tokens_val,
            truncation_needed,
            error_provider,
        )

        metadata: dict[str, Any] = {
            "research_id": state.id,
            "phase": phase_name,
            "error_type": "context_window_exceeded",
            "prompt_tokens": prompt_tokens,
            "max_tokens": max_tokens_val,
            "truncation_needed": truncation_needed,
            "token_limit_retries": token_limit_retries,
        }
        if error_metadata:
            metadata.update(error_metadata)

        return WorkflowResult(
            success=False,
            content="",
            error=str(last_context_error),
            metadata=metadata,
        )

    # Safety: result must be set at this point (break from loop with no error)
    assert result is not None

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

    # Phase metrics (include token_limit_retries if any occurred)
    phase_metrics_metadata: dict[str, Any] = {}
    if token_limit_retries > 0:
        phase_metrics_metadata["token_limit_retries"] = token_limit_retries
    if role:
        phase_metrics_metadata["role"] = role

    state.phase_metrics.append(
        PhaseMetrics(
            phase=phase_name,
            duration_ms=result.duration_ms or 0.0,
            input_tokens=result.input_tokens or 0,
            output_tokens=result.output_tokens or 0,
            cached_tokens=result.cached_tokens or 0,
            provider_id=result.provider_id,
            model_used=result.model_used,
            metadata=phase_metrics_metadata,
        )
    )

    return LLMCallResult(result=result, llm_call_duration_ms=llm_call_duration_ms)


# Maximum parse-validation retries for structured LLM calls.
_MAX_STRUCTURED_PARSE_RETRIES: int = 3


@dataclass
class StructuredLLMCallResult:
    """Result of a structured LLM call with parsed data.

    Attributes:
        result: The underlying WorkflowResult from the LLM call.
        llm_call_duration_ms: Total time spent across all LLM call attempts.
        parsed: The parsed structured data (output of ``parse_fn``), or
            ``None`` if parsing failed on all attempts.
        parse_retries: Number of parse-validation retries that were needed.
    """

    result: WorkflowResult
    llm_call_duration_ms: float
    parsed: Any
    parse_retries: int = 0


async def execute_structured_llm_call(
    workflow: Any,
    state: DeepResearchState,
    phase_name: str,
    system_prompt: str,
    user_prompt: str,
    provider_id: Optional[str],
    model: Optional[str],
    temperature: float,
    timeout: float,
    parse_fn: Any,
    error_metadata: Optional[dict[str, Any]] = None,
    role: Optional[str] = None,
) -> StructuredLLMCallResult | WorkflowResult:
    """Execute an LLM call expecting structured JSON output.

    Wraps :func:`execute_llm_call` with parse-validation and retry logic.
    On each attempt the LLM response content is passed through *parse_fn*.
    If *parse_fn* raises (``ValueError``, ``json.JSONDecodeError``, etc.),
    the call is retried with a reinforced JSON instruction appended to the
    user prompt, up to ``_MAX_STRUCTURED_PARSE_RETRIES`` times.

    If all parse attempts fail, returns a :class:`StructuredLLMCallResult`
    with ``parsed=None`` and the last successful LLM result — letting the
    caller fall back to unstructured handling.

    Args:
        workflow: The DeepResearchWorkflow instance
        state: Current research state
        phase_name: Phase identifier
        system_prompt: System prompt (should already request JSON output)
        user_prompt: User prompt for the LLM call
        provider_id: Explicit provider ID
        model: Model override
        temperature: Sampling temperature
        timeout: Request timeout in seconds
        parse_fn: Callable ``(content: str) -> T`` that parses the LLM
            response content.  Should raise on validation failure.
        error_metadata: Extra fields for error response metadata
        role: Model role for cost-optimized routing (passed through to
            :func:`execute_llm_call`)

    Returns:
        StructuredLLMCallResult on success or parse-exhaustion (check
        ``.parsed`` for ``None``), or WorkflowResult on LLM-level error.
    """
    import json as _json

    total_duration_ms = 0.0
    last_llm_result: Optional[LLMCallResult] = None
    parse_retries = 0

    current_user_prompt = user_prompt

    for attempt in range(_MAX_STRUCTURED_PARSE_RETRIES + 1):  # 0 = initial, 1-3 = retries
        call_result = await execute_llm_call(
            workflow=workflow,
            state=state,
            phase_name=phase_name,
            system_prompt=system_prompt,
            user_prompt=current_user_prompt,
            provider_id=provider_id,
            model=model,
            temperature=temperature,
            timeout=timeout,
            error_metadata=error_metadata,
            role=role,
        )

        # LLM-level error — propagate immediately
        if isinstance(call_result, WorkflowResult):
            return call_result

        last_llm_result = call_result
        total_duration_ms += call_result.llm_call_duration_ms

        # Try parsing the structured output
        content = call_result.result.content or ""
        try:
            parsed = parse_fn(content)
            return StructuredLLMCallResult(
                result=call_result.result,
                llm_call_duration_ms=total_duration_ms,
                parsed=parsed,
                parse_retries=parse_retries,
            )
        except (ValueError, _json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning(
                "%s phase structured parse failed (attempt %d/%d): %s",
                phase_name.capitalize(),
                attempt + 1,
                _MAX_STRUCTURED_PARSE_RETRIES + 1,
                exc,
            )

            if attempt >= _MAX_STRUCTURED_PARSE_RETRIES:
                break  # All retries exhausted

            parse_retries += 1

            # Reinforce JSON instruction for next attempt
            current_user_prompt = (
                user_prompt
                + "\n\nIMPORTANT: Your previous response could not be parsed as valid JSON. "
                "You MUST respond with ONLY a valid JSON object, no markdown formatting, "
                "no extra text before or after the JSON."
            )

    # Parse exhausted — return with parsed=None so caller can fall back
    assert last_llm_result is not None
    logger.warning(
        "%s phase structured output parsing failed after %d retries, "
        "falling back to unstructured handling",
        phase_name.capitalize(),
        parse_retries,
    )
    return StructuredLLMCallResult(
        result=last_llm_result.result,
        llm_call_duration_ms=total_duration_ms,
        parsed=None,
        parse_retries=parse_retries,
    )


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
