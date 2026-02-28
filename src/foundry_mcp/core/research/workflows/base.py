"""Base class for research workflows.

Provides common infrastructure for provider integration, error handling,
and response normalization across all research workflow types.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional

from foundry_mcp.config.research import ResearchConfig
from foundry_mcp.core.errors.provider import ContextWindowError, ProviderTimeoutError
from foundry_mcp.core.llm_config.provider_spec import ProviderSpec
from foundry_mcp.core.research.token_management import get_effective_context, get_model_limits
from foundry_mcp.core.providers import (
    ProviderContext,
    ProviderHooks,
    ProviderRequest,
    ProviderResult,
    ProviderStatus,
    create_context_window_guidance,
    extract_token_counts,
    is_context_window_error,
)
from foundry_mcp.core.providers.registry import available_providers, resolve_provider
from foundry_mcp.core.research.memory import ResearchMemory

logger = logging.getLogger(__name__)

# Fallback max prompt length (chars) when model context window is unknown.
# ~600k chars ≈ 150k tokens at ~4 chars/token.  When the target model is
# known, _max_prompt_chars_for_model() derives a tighter limit from the
# model's actual context window in the token-management registry.
MAX_PROMPT_LENGTH = 600_000


# Approximate chars-per-token ratio for converting token budgets to char limits.
_CHARS_PER_TOKEN = 4
# Fraction of the model's effective context to use as the char-limit ceiling.
# Leaves headroom for system prompt + output tokens.
_CONTEXT_USAGE_FRACTION = 0.75


def _max_prompt_chars_for_model(
    provider_id: str | None,
    model: str | None,
) -> int:
    """Derive a max prompt character limit from the model's context window.

    Returns ``MAX_PROMPT_LENGTH`` when provider/model are unknown.
    """
    if not provider_id:
        return MAX_PROMPT_LENGTH
    try:
        limits = get_model_limits(provider_id, model)
        effective_tokens = get_effective_context(limits)
        return int(effective_tokens * _CHARS_PER_TOKEN * _CONTEXT_USAGE_FRACTION)
    except Exception:
        return MAX_PROMPT_LENGTH


def _estimate_prompt_tokens(prompt: str, system_prompt: str | None = None) -> int:
    """Estimate token count for a prompt using simple heuristic.

    Uses ~4 characters per token as a rough estimate. This is conservative
    and works reasonably well for English text.

    Args:
        prompt: User prompt
        system_prompt: Optional system prompt

    Returns:
        Estimated token count
    """
    total_chars = len(prompt)
    if system_prompt:
        total_chars += len(system_prompt)
    return total_chars // 4


@dataclass
class WorkflowResult:
    """Result of a workflow execution.

    Attributes:
        success: Whether the workflow completed successfully
        content: Main response content
        provider_id: Provider that generated the response
        model_used: Model that generated the response
        tokens_used: Total tokens consumed
        input_tokens: Tokens consumed by the prompt
        output_tokens: Tokens generated in the response
        cached_tokens: Tokens served from cache
        duration_ms: Execution duration in milliseconds
        metadata: Additional workflow-specific data
        error: Error message if success is False
    """

    success: bool
    content: str
    provider_id: Optional[str] = None
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    duration_ms: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class ResearchWorkflowBase(ABC):
    """Base class for all research workflows.

    Provides common functionality for provider resolution, request execution,
    and memory management.
    """

    def __init__(
        self,
        config: ResearchConfig,
        memory: Optional[ResearchMemory] = None,
    ) -> None:
        """Initialize workflow with configuration and memory.

        Args:
            config: Research configuration
            memory: Optional memory instance (creates default if not provided)
        """
        self.config = config
        # Memory should be provided by caller with proper research_dir from ServerConfig
        # Fallback uses ResearchMemory default (~/.foundry-mcp/research)
        self.memory = memory or ResearchMemory(ttl_hours=config.ttl_hours)
        self._provider_cache: dict[str, ProviderContext] = {}

    def _resolve_provider(
        self,
        provider_id: Optional[str] = None,
        hooks: Optional[ProviderHooks] = None,
    ) -> Optional[ProviderContext]:
        """Resolve and cache a provider instance.

        Args:
            provider_id: Provider ID or full spec to resolve (uses config default if None)
                         Supports both simple IDs ("codex") and full specs ("[cli]codex:gpt-5.2")
            hooks: Optional provider hooks

        Returns:
            ProviderContext instance or None if unavailable
        """
        provider_spec_str = provider_id or self.config.default_provider

        # Check cache first (using full spec string as key)
        if provider_spec_str in self._provider_cache:
            return self._provider_cache[provider_spec_str]

        # Parse the provider spec to extract base provider ID
        try:
            spec = ProviderSpec.parse_flexible(provider_spec_str)
        except ValueError as exc:
            logger.warning("Invalid provider spec '%s': %s", provider_spec_str, exc)
            return None

        # Check availability using base provider ID
        available = available_providers()
        if spec.provider not in available:
            logger.warning(
                "Provider %s (from spec '%s') not available. Available: %s",
                spec.provider,
                provider_spec_str,
                available,
            )
            return None

        try:
            # Resolve using base provider ID and pass model override if specified
            provider = resolve_provider(
                spec.provider,
                hooks=hooks or ProviderHooks(),
                model=spec.model,
            )
            self._provider_cache[provider_spec_str] = provider
            return provider
        except Exception as exc:
            logger.error("Failed to resolve provider %s: %s", spec.provider, exc)
            return None

    def _execute_provider(
        self,
        prompt: str,
        provider_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        hooks: Optional[ProviderHooks] = None,
    ) -> WorkflowResult:
        """Execute a single provider request.

        Args:
            prompt: User prompt
            provider_id: Provider to use (uses config default if None)
            system_prompt: Optional system prompt
            model: Optional model override
            timeout: Optional timeout in seconds
            temperature: Optional temperature setting
            max_tokens: Optional max tokens
            hooks: Optional provider hooks

        Returns:
            WorkflowResult with response or error
        """
        provider = self._resolve_provider(provider_id, hooks)
        if provider is None:
            logger.warning(
                "_execute_provider: Provider resolution failed for '%s'",
                provider_id or self.config.default_provider,
            )
            return WorkflowResult(
                success=False,
                content="",
                error=f"Provider '{provider_id or self.config.default_provider}' is not available",
            )

        request = ProviderRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            timeout=timeout or self.config.default_timeout,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Estimate prompt tokens for error reporting
        estimated_tokens = _estimate_prompt_tokens(prompt, system_prompt)

        try:
            result: ProviderResult = provider.generate(request)

            if result.status != ProviderStatus.SUCCESS:
                return WorkflowResult(
                    success=False,
                    content=result.content or "",
                    provider_id=result.provider_id,
                    model_used=result.model_used,
                    error=f"Provider returned status: {result.status.value}",
                )

            return WorkflowResult(
                success=True,
                content=result.content,
                provider_id=result.provider_id,
                model_used=result.model_used,
                tokens_used=result.tokens.total_tokens if result.tokens else None,
                input_tokens=result.tokens.input_tokens if result.tokens else None,
                output_tokens=result.tokens.output_tokens if result.tokens else None,
                cached_tokens=result.tokens.cached_input_tokens if result.tokens else None,
                duration_ms=result.duration_ms,
            )

        except ContextWindowError:
            # Re-raise context window errors directly
            raise

        except Exception as exc:
            # Check if this is a context window error
            if is_context_window_error(exc):
                # Extract token counts from error message if available
                prompt_tokens, max_context = extract_token_counts(str(exc))

                # Use estimated tokens if not extracted
                if prompt_tokens is None:
                    prompt_tokens = estimated_tokens

                # Log detailed context window error
                logger.error(
                    "Context window exceeded: prompt_tokens=%s, max_tokens=%s, "
                    "estimated_tokens=%d, provider=%s, error=%s",
                    prompt_tokens,
                    max_context,
                    estimated_tokens,
                    provider_id,
                    str(exc),
                )

                # Generate actionable guidance
                guidance = create_context_window_guidance(
                    prompt_tokens=prompt_tokens,
                    max_tokens=max_context,
                    provider_id=provider_id,
                )

                # Raise specific context window error with details
                raise ContextWindowError(
                    guidance,
                    provider=provider_id,
                    prompt_tokens=prompt_tokens,
                    max_tokens=max_context,
                ) from exc

            # Non-context-window error - log and return error result
            logger.error("Provider execution failed: %s", exc)
            return WorkflowResult(
                success=False,
                content="",
                provider_id=provider_id,
                error=str(exc),
            )

    async def _execute_provider_async(
        self,
        prompt: str,
        provider_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        hooks: Optional[ProviderHooks] = None,
        phase: Optional[str] = None,
        fallback_providers: Optional[List[str]] = None,
        max_retries: int = 2,
        retry_delay: float = 5.0,
    ) -> WorkflowResult:
        """Execute a provider request asynchronously with timeout protection.

        This method wraps the synchronous provider.generate() call in an executor
        with asyncio.wait_for() timeout protection. It also supports retry and
        fallback logic for resilience.

        Args:
            prompt: User prompt
            provider_id: Provider to use (uses config default if None)
            system_prompt: Optional system prompt
            model: Optional model override
            timeout: Optional timeout in seconds (applied to provider execution)
            temperature: Optional temperature setting
            max_tokens: Optional max tokens
            hooks: Optional provider hooks
            phase: Phase name for logging (e.g., "planning", "analysis")
            fallback_providers: List of fallback provider IDs to try on failure
            max_retries: Maximum retry attempts per provider (default: 2)
            retry_delay: Delay between retries in seconds (default: 5.0)

        Returns:
            WorkflowResult with response, error, or timeout metadata
        """
        effective_timeout = timeout or self.config.default_timeout

        # Input bounds validation: reject oversized prompts early.
        # Derive char limit from the target model's context window when
        # provider/model are known; fall back to the module constant.
        effective_max = _max_prompt_chars_for_model(
            provider_id or getattr(self.config, "default_provider", None),
            model,
        )
        if len(prompt) > effective_max:
            return WorkflowResult(
                success=False,
                content="",
                error=(f"Prompt length {len(prompt)} exceeds maximum {effective_max} characters"),
                metadata={"phase": phase, "validation_error": "prompt_too_long"},
            )

        # Track wall-clock time for observability
        method_start = time.monotonic()

        primary_provider = provider_id or self.config.default_provider
        providers_to_try = [primary_provider]

        # Add fallback providers if configured
        if fallback_providers:
            for fp in fallback_providers:
                if fp not in providers_to_try:
                    providers_to_try.append(fp)

        providers_tried: List[str] = []
        total_retries = 0
        last_error: Optional[str] = None
        saw_timeout = False
        saw_non_timeout = False

        for current_provider_id in providers_to_try:
            current_spec: Optional[ProviderSpec] = None
            try:
                current_spec = ProviderSpec.parse_flexible(current_provider_id)
            except ValueError:
                current_spec = None

            # Try this provider with retries
            for attempt in range(max_retries + 1):
                start_time = time.perf_counter()
                providers_tried.append(current_provider_id)

                try:
                    provider = self._resolve_provider(current_provider_id, hooks)
                    if provider is None:
                        last_error = f"Provider '{current_provider_id}' is not available"
                        saw_non_timeout = True
                        logger.warning(
                            "%s phase: Provider resolution failed for '%s' (attempt %d)",
                            phase or "unknown",
                            current_provider_id,
                            attempt + 1,
                        )
                        break  # Don't retry if provider can't be resolved

                    request_model = None
                    if current_spec and current_spec.model:
                        request_model = current_spec.model
                    elif model is not None and current_provider_id == primary_provider:
                        request_model = model

                    request = ProviderRequest(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        model=request_model,
                        timeout=effective_timeout,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                    # Run synchronous generate in thread pool
                    loop = asyncio.get_running_loop()
                    result: ProviderResult = await asyncio.wait_for(
                        loop.run_in_executor(None, provider.generate, request),
                        timeout=effective_timeout,
                    )

                    duration_ms = (time.perf_counter() - start_time) * 1000

                    if result.status != ProviderStatus.SUCCESS:
                        last_error = f"Provider returned status: {result.status.value}"
                        saw_non_timeout = True
                        logger.warning(
                            "%s phase: Provider %s returned %s (attempt %d)",
                            phase or "unknown",
                            current_provider_id,
                            result.status.value,
                            attempt + 1,
                        )
                        # Retry on non-success status
                        if attempt < max_retries:
                            total_retries += 1
                            await asyncio.sleep(retry_delay)
                            continue
                        # Try next provider
                        break

                    # Success!
                    total_elapsed_ms = (time.monotonic() - method_start) * 1000
                    return WorkflowResult(
                        success=True,
                        content=result.content,
                        provider_id=result.provider_id,
                        model_used=result.model_used,
                        tokens_used=result.tokens.total_tokens if result.tokens else None,
                        input_tokens=result.tokens.input_tokens if result.tokens else None,
                        output_tokens=result.tokens.output_tokens if result.tokens else None,
                        cached_tokens=result.tokens.cached_input_tokens if result.tokens else None,
                        duration_ms=duration_ms,
                        metadata={
                            "phase": phase,
                            "retries": total_retries,
                            "providers_tried": providers_tried,
                            "wall_clock_ms": total_elapsed_ms,
                            "configured_timeout_s": effective_timeout,
                        },
                    )

                except ProviderTimeoutError as exc:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    last_error = str(exc) or f"Timed out after {effective_timeout:.1f}s"
                    saw_timeout = True
                    logger.warning(
                        "%s phase: Provider %s timed out after %.1fs (attempt %d/%d)",
                        phase or "unknown",
                        current_provider_id,
                        duration_ms / 1000,
                        attempt + 1,
                        max_retries + 1,
                    )
                    # Retry on timeout
                    if attempt < max_retries:
                        total_retries += 1
                        await asyncio.sleep(retry_delay)
                        continue
                    # Try next provider
                    break

                except asyncio.TimeoutError:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    last_error = f"Timed out after {effective_timeout:.1f}s"
                    saw_timeout = True
                    logger.warning(
                        "%s phase: Provider %s timed out after %.1fs (attempt %d/%d)",
                        phase or "unknown",
                        current_provider_id,
                        duration_ms / 1000,
                        attempt + 1,
                        max_retries + 1,
                    )
                    # Retry on timeout
                    if attempt < max_retries:
                        total_retries += 1
                        await asyncio.sleep(retry_delay)
                        continue
                    # Try next provider
                    break

                except ContextWindowError:
                    # Don't retry context window errors - they'll fail everywhere
                    raise

                except Exception as exc:
                    duration_ms = (time.perf_counter() - start_time) * 1000

                    # Check if this is a context window error
                    if is_context_window_error(exc):
                        # Extract token counts and re-raise as ContextWindowError
                        prompt_tokens, max_context = extract_token_counts(str(exc))
                        estimated_tokens = _estimate_prompt_tokens(prompt, system_prompt)
                        if prompt_tokens is None:
                            prompt_tokens = estimated_tokens

                        guidance = create_context_window_guidance(
                            prompt_tokens=prompt_tokens,
                            max_tokens=max_context,
                            provider_id=current_provider_id,
                        )
                        raise ContextWindowError(
                            guidance,
                            provider=current_provider_id,
                            prompt_tokens=prompt_tokens,
                            max_tokens=max_context,
                        ) from exc

                    last_error = str(exc)
                    saw_non_timeout = True
                    logger.warning(
                        "%s phase: Provider %s failed with %s (attempt %d): %s",
                        phase or "unknown",
                        current_provider_id,
                        type(exc).__name__,
                        attempt + 1,
                        exc,
                    )
                    # Retry on other errors
                    if attempt < max_retries:
                        total_retries += 1
                        await asyncio.sleep(retry_delay)
                        continue
                    # Try next provider
                    break

        # All providers exhausted
        total_elapsed_ms = (time.monotonic() - method_start) * 1000
        logger.error(
            "%s phase: All providers exhausted after %d total attempts "
            "(%.1fs wall-clock of %.1fs budget). Providers tried: %s. Last error: %s",
            phase or "unknown",
            len(providers_tried),
            total_elapsed_ms / 1000,
            effective_timeout,
            providers_tried,
            last_error,
        )

        timed_out = saw_timeout and not saw_non_timeout
        return WorkflowResult(
            success=False,
            content="",
            error=last_error or "All providers exhausted",
            metadata={
                "phase": phase,
                "timeout": timed_out,
                "retries": total_retries,
                "providers_tried": providers_tried,
                "wall_clock_ms": total_elapsed_ms,
                "configured_timeout_s": effective_timeout,
            },
        )

    def get_available_providers(self) -> list[str]:
        """Get list of available provider IDs.

        Returns:
            List of available provider identifiers
        """
        return available_providers()

    @abstractmethod
    def execute(self, **kwargs: Any) -> WorkflowResult:
        """Execute the workflow.

        Subclasses must implement this method with their specific logic.

        Returns:
            WorkflowResult with response or error
        """
        ...
