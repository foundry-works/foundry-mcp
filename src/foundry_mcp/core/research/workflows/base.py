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
    ProviderContext,
    ProviderHooks,
    ProviderRequest,
    ProviderResult,
    ProviderStatus,
)
from foundry_mcp.core.providers.registry import available_providers, resolve_provider
from foundry_mcp.core.research.memory import ResearchMemory

logger = logging.getLogger(__name__)


@dataclass
class WorkflowResult:
    """Result of a workflow execution.

    Attributes:
        success: Whether the workflow completed successfully
        content: Main response content
        provider_id: Provider that generated the response
        model_used: Model that generated the response
        tokens_used: Total tokens consumed
        duration_ms: Execution duration in milliseconds
        metadata: Additional workflow-specific data
        error: Error message if success is False
    """

    success: bool
    content: str
    provider_id: Optional[str] = None
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None
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
            timeout=timeout or 30.0,
            temperature=temperature,
            max_tokens=max_tokens,
        )

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
                duration_ms=result.duration_ms,
            )

        except Exception as exc:
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
