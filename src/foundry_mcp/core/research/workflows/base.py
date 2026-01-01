"""Base class for research workflows.

Provides common infrastructure for provider integration, error handling,
and response normalization across all research workflow types.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from foundry_mcp.config import ResearchConfig
from foundry_mcp.core.providers import (
    ContextWindowError,
    ProviderContext,
    ProviderHooks,
    ProviderRequest,
    ProviderResult,
    ProviderStatus,
    is_context_window_error,
    extract_token_counts,
    create_context_window_guidance,
)
from foundry_mcp.core.providers.registry import available_providers, resolve_provider
from foundry_mcp.core.research.memory import ResearchMemory

logger = logging.getLogger(__name__)


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
    metadata: dict[str, Any] = None
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
        self.memory = memory or ResearchMemory(
            base_path=config.get_storage_path(),
            ttl_hours=config.ttl_hours,
        )
        self._provider_cache: dict[str, ProviderContext] = {}

    def _resolve_provider(
        self,
        provider_id: Optional[str] = None,
        hooks: Optional[ProviderHooks] = None,
    ) -> Optional[ProviderContext]:
        """Resolve and cache a provider instance.

        Args:
            provider_id: Provider ID to resolve (uses config default if None)
            hooks: Optional provider hooks

        Returns:
            ProviderContext instance or None if unavailable
        """
        provider_id = provider_id or self.config.default_provider

        # Check cache first
        if provider_id in self._provider_cache:
            return self._provider_cache[provider_id]

        # Check availability
        available = available_providers()
        if provider_id not in available:
            logger.warning("Provider %s not available. Available: %s", provider_id, available)
            return None

        try:
            provider = resolve_provider(provider_id, hooks=hooks or ProviderHooks())
            self._provider_cache[provider_id] = provider
            return provider
        except Exception as exc:
            logger.error("Failed to resolve provider %s: %s", provider_id, exc)
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
            return WorkflowResult(
                success=False,
                content="",
                error=f"Provider '{provider_id or self.config.default_provider}' is not available",
            )

        request = ProviderRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            timeout=timeout or 120.0,
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
