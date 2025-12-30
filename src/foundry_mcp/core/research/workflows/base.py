"""Base class for research workflows.

Provides common infrastructure for provider integration, error handling,
and response normalization across all research workflow types.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from foundry_mcp.config import ResearchConfig
from foundry_mcp.core.llm_config import ProviderSpec
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
