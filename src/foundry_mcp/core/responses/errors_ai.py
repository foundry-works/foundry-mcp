"""
AI/LLM provider error helpers for MCP tool responses.

Provides 6 helpers for AI provider errors: no provider available,
provider timeout, provider error, context too large, prompt not found,
and cache stale.
"""

from typing import Any, Dict, Optional, Sequence

from foundry_mcp.core.responses.builders import error_response
from foundry_mcp.core.responses.types import ErrorCode, ErrorType, ToolResponse


def ai_no_provider_error(
    message: str = "No AI provider available",
    *,
    required_providers: Optional[Sequence[str]] = None,
    remediation: Optional[str] = None,
    request_id: Optional[str] = None,
) -> ToolResponse:
    """Create an error response for when no AI provider is available.

    Use when an AI consultation is requested but no providers are configured
    or all configured providers have failed availability checks.

    Args:
        message: Human-readable description.
        required_providers: List of provider IDs that were checked.
        remediation: Guidance on how to configure a provider.
        request_id: Correlation identifier.

    Example:
        >>> ai_no_provider_error(
        ...     "No AI provider available for plan review",
        ...     required_providers=["gemini", "cursor-agent", "codex"],
        ... )
    """
    data: Dict[str, Any] = {}
    if required_providers:
        data["required_providers"] = list(required_providers)

    default_remediation = (
        "Configure an AI provider: set GEMINI_API_KEY, OPENAI_API_KEY, "
        "or ANTHROPIC_API_KEY environment variable, or ensure cursor-agent is available."
    )

    return error_response(
        message,
        error_code=ErrorCode.AI_NO_PROVIDER,
        error_type=ErrorType.AI_PROVIDER,
        data=data if data else None,
        remediation=remediation or default_remediation,
        request_id=request_id,
    )


def ai_provider_timeout_error(
    provider_id: str,
    timeout_seconds: int,
    *,
    message: Optional[str] = None,
    remediation: Optional[str] = None,
    request_id: Optional[str] = None,
) -> ToolResponse:
    """Create an error response for AI provider execution timeout.

    Use when an AI provider call exceeds the configured timeout limit.

    Args:
        provider_id: The provider that timed out (e.g., "gemini", "codex").
        timeout_seconds: The timeout that was exceeded.
        message: Human-readable description (auto-generated if not provided).
        remediation: Guidance on how to handle the timeout.
        request_id: Correlation identifier.

    Example:
        >>> ai_provider_timeout_error("gemini", 300)
    """
    default_message = f"AI provider '{provider_id}' timed out after {timeout_seconds}s"

    return error_response(
        message or default_message,
        error_code=ErrorCode.AI_PROVIDER_TIMEOUT,
        error_type=ErrorType.AI_PROVIDER,
        data={
            "provider_id": provider_id,
            "timeout_seconds": timeout_seconds,
        },
        remediation=remediation
        or (
            "Try again with a smaller context, increase the timeout, "
            "or use a different provider."
        ),
        request_id=request_id,
    )


def ai_provider_error(
    provider_id: str,
    error_detail: str,
    *,
    status_code: Optional[int] = None,
    remediation: Optional[str] = None,
    request_id: Optional[str] = None,
) -> ToolResponse:
    """Create an error response for when an AI provider returns an error.

    Use when an AI provider API call fails with an error response.

    Args:
        provider_id: The provider that returned the error.
        error_detail: The error message from the provider.
        status_code: HTTP status code from the provider (if applicable).
        remediation: Guidance on how to resolve the issue.
        request_id: Correlation identifier.

    Example:
        >>> ai_provider_error("gemini", "Invalid API key", status_code=401)
    """
    data: Dict[str, Any] = {
        "provider_id": provider_id,
        "error_detail": error_detail,
    }
    if status_code is not None:
        data["status_code"] = status_code

    return error_response(
        f"AI provider '{provider_id}' returned error: {error_detail}",
        error_code=ErrorCode.AI_PROVIDER_ERROR,
        error_type=ErrorType.AI_PROVIDER,
        data=data,
        remediation=remediation
        or (
            "Check provider configuration and API key validity. "
            "Consult provider documentation for error details."
        ),
        request_id=request_id,
    )


def ai_context_too_large_error(
    context_tokens: int,
    max_tokens: int,
    *,
    provider_id: Optional[str] = None,
    remediation: Optional[str] = None,
    request_id: Optional[str] = None,
) -> ToolResponse:
    """Create an error response for when context exceeds model limits.

    Use when the prompt/context size exceeds the AI model's token limit.

    Args:
        context_tokens: Number of tokens in the attempted context.
        max_tokens: Maximum tokens allowed by the model.
        provider_id: The provider that rejected the context.
        remediation: Guidance on how to reduce context size.
        request_id: Correlation identifier.

    Example:
        >>> ai_context_too_large_error(150000, 128000, provider_id="gemini")
    """
    data: Dict[str, Any] = {
        "context_tokens": context_tokens,
        "max_tokens": max_tokens,
        "overflow_tokens": context_tokens - max_tokens,
    }
    if provider_id:
        data["provider_id"] = provider_id

    return error_response(
        f"Context size ({context_tokens} tokens) exceeds limit ({max_tokens} tokens)",
        error_code=ErrorCode.AI_CONTEXT_TOO_LARGE,
        error_type=ErrorType.AI_PROVIDER,
        data=data,
        remediation=remediation
        or (
            "Reduce context size by: excluding large files, using incremental mode, "
            "or reviewing only specific tasks/phases instead of the full spec."
        ),
        request_id=request_id,
    )


def ai_prompt_not_found_error(
    prompt_id: str,
    *,
    available_prompts: Optional[Sequence[str]] = None,
    workflow: Optional[str] = None,
    remediation: Optional[str] = None,
    request_id: Optional[str] = None,
) -> ToolResponse:
    """Create an error response for when a prompt template is not found.

    Use when a requested prompt ID doesn't exist in the prompt registry.

    Args:
        prompt_id: The prompt ID that was not found.
        available_prompts: List of valid prompt IDs for the workflow.
        workflow: The workflow context (e.g., "plan_review", "fidelity_review").
        remediation: Guidance on how to find the correct prompt ID.
        request_id: Correlation identifier.

    Example:
        >>> ai_prompt_not_found_error(
        ...     "INVALID_PROMPT",
        ...     available_prompts=["PLAN_REVIEW_FULL_V1", "PLAN_REVIEW_QUICK_V1"],
        ...     workflow="plan_review",
        ... )
    """
    data: Dict[str, Any] = {"prompt_id": prompt_id}
    if available_prompts:
        data["available_prompts"] = list(available_prompts)
    if workflow:
        data["workflow"] = workflow

    available_str = ""
    if available_prompts:
        available_str = f" Available: {', '.join(available_prompts[:5])}"
        if len(available_prompts) > 5:
            available_str += f" (and {len(available_prompts) - 5} more)"

    return error_response(
        f"Prompt '{prompt_id}' not found.{available_str}",
        error_code=ErrorCode.AI_PROMPT_NOT_FOUND,
        error_type=ErrorType.NOT_FOUND,
        data=data,
        remediation=remediation
        or (
            "Use a valid prompt ID from the workflow's prompt builder. "
            "Call list_prompts() to see available templates."
        ),
        request_id=request_id,
    )


def ai_cache_stale_error(
    cache_key: str,
    cache_age_seconds: int,
    max_age_seconds: int,
    *,
    remediation: Optional[str] = None,
    request_id: Optional[str] = None,
) -> ToolResponse:
    """Create an error response for when cached AI result is stale.

    Use when a cached consultation result has expired and needs refresh.

    Args:
        cache_key: Identifier for the cached item.
        cache_age_seconds: Age of the cached result in seconds.
        max_age_seconds: Maximum allowed age for cached results.
        remediation: Guidance on how to refresh the cache.
        request_id: Correlation identifier.

    Example:
        >>> ai_cache_stale_error(
        ...     "plan_review:spec-001:full",
        ...     cache_age_seconds=7200,
        ...     max_age_seconds=3600,
        ... )
    """
    return error_response(
        f"Cached result for '{cache_key}' is stale ({cache_age_seconds}s > {max_age_seconds}s)",
        error_code=ErrorCode.AI_CACHE_STALE,
        error_type=ErrorType.AI_PROVIDER,
        data={
            "cache_key": cache_key,
            "cache_age_seconds": cache_age_seconds,
            "max_age_seconds": max_age_seconds,
        },
        remediation=remediation
        or (
            "Re-run the consultation to refresh cached results, "
            "or use --no-cache to bypass the cache."
        ),
        request_id=request_id,
    )
