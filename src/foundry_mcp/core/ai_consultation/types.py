"""
Consultation type definitions: enums, dataclasses, and type aliases.

This module contains all data structures used across the AI consultation
layer. These are pure data types with no external dependencies beyond
the standard library and foundry_mcp.core.llm_config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union


# =============================================================================
# Workflow Types
# =============================================================================


class ConsultationWorkflow(str, Enum):
    """
    Supported AI consultation workflows.

    Each workflow corresponds to a category of prompt templates and
    determines cache partitioning and result handling.

    Values:
        PLAN_REVIEW: Review and critique SDD specifications
        FIDELITY_REVIEW: Compare implementation against specifications
        MARKDOWN_PLAN_REVIEW: Review markdown plans before spec creation
    """

    PLAN_REVIEW = "plan_review"
    FIDELITY_REVIEW = "fidelity_review"
    MARKDOWN_PLAN_REVIEW = "markdown_plan_review"


# =============================================================================
# Request/Response Dataclasses
# =============================================================================


@dataclass
class ResolvedProvider:
    """
    Resolved provider information from a ProviderSpec.

    Contains the provider ID to use for registry lookup, along with
    model and override settings from the priority configuration.

    Attributes:
        provider_id: Provider ID for registry lookup (e.g., "gemini", "opencode")
        model: Model identifier to use (may include backend routing for CLI)
        overrides: Per-provider setting overrides from config
        spec_str: Original spec string for logging/debugging
    """

    provider_id: str
    model: Optional[str] = None
    overrides: Dict[str, Any] = field(default_factory=dict)
    spec_str: str = ""


@dataclass(frozen=True)
class ConsultationRequest:
    """
    Request payload for AI consultation.

    Encapsulates all parameters needed to execute a consultation workflow,
    including prompt selection, context data, and provider preferences.

    Attributes:
        workflow: The consultation workflow type
        prompt_id: Identifier for the prompt template within the workflow
        context: Structured context data to inject into the prompt
        provider_id: Optional preferred provider (uses first available if None)
        model: Optional model override for the provider
        cache_key: Optional explicit cache key (auto-generated if None)
        timeout: Request timeout in seconds (default: 120)
        temperature: Sampling temperature (default: provider default)
        max_tokens: Maximum output tokens (default: provider default)
        system_prompt_override: Optional system prompt override
    """

    workflow: ConsultationWorkflow
    prompt_id: str
    context: Dict[str, Any] = field(default_factory=dict)
    provider_id: Optional[str] = None
    model: Optional[str] = None
    cache_key: Optional[str] = None
    timeout: float = 120.0
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    system_prompt_override: Optional[str] = None


@dataclass
class ConsultationResult:
    """
    Result of an AI consultation.

    Provides a consistent structure for consultation outcomes across all
    workflows and providers, including metadata for debugging and analytics.

    Attributes:
        workflow: The workflow that produced this result
        content: The generated content (may be empty on failure)
        provider_id: Provider that handled the request
        model_used: Fully-qualified model identifier
        tokens: Token usage if reported by provider
        duration_ms: Total consultation duration in milliseconds
        cache_hit: Whether result was served from cache
        raw_payload: Provider-specific metadata and debug info
        warnings: Non-fatal issues encountered during consultation
        error: Error message if consultation failed
    """

    workflow: ConsultationWorkflow
    content: str
    provider_id: str
    model_used: str
    tokens: Dict[str, int] = field(default_factory=dict)
    duration_ms: float = 0.0
    cache_hit: bool = False
    raw_payload: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Return True if consultation succeeded (has content, no error)."""
        return bool(self.content) and self.error is None


@dataclass
class ProviderResponse:
    """
    Response from a single provider in a multi-model consultation.

    Encapsulates the result from one provider when executing parallel
    consultations across multiple models. Used as building blocks for
    ConsensusResult aggregation.

    Attributes:
        provider_id: Identifier of the provider that handled this request
        model_used: Fully-qualified model identifier used for generation
        content: Generated content (empty string on failure)
        success: Whether this provider's request succeeded
        error: Error message if the request failed
        tokens: Total token usage (prompt + completion) if available
        duration_ms: Request duration in milliseconds
        cache_hit: Whether result was served from cache
    """

    provider_id: str
    model_used: str
    content: str
    success: bool
    error: Optional[str] = None
    tokens: Optional[int] = None
    duration_ms: Optional[int] = None
    cache_hit: bool = False

    @classmethod
    def from_result(
        cls,
        result: ConsultationResult,
    ) -> "ProviderResponse":
        """
        Create a ProviderResponse from a ConsultationResult.

        Convenience factory for converting single-provider results to the
        multi-provider response format.

        Args:
            result: ConsultationResult to convert

        Returns:
            ProviderResponse with fields mapped from the result
        """
        total_tokens = sum(result.tokens.values()) if result.tokens else None
        return cls(
            provider_id=result.provider_id,
            model_used=result.model_used,
            content=result.content,
            success=result.success,
            error=result.error,
            tokens=total_tokens,
            duration_ms=int(result.duration_ms) if result.duration_ms else None,
            cache_hit=result.cache_hit,
        )


@dataclass
class AgreementMetadata:
    """
    Metadata about provider agreement in a multi-model consultation.

    Tracks how many providers were consulted, how many succeeded, and how
    many failed. Used to assess consensus quality and reliability.

    Attributes:
        total_providers: Total number of providers that were consulted
        successful_providers: Number of providers that returned successful responses
        failed_providers: Number of providers that failed (timeout, error, etc.)
    """

    total_providers: int
    successful_providers: int
    failed_providers: int

    @property
    def success_rate(self) -> float:
        """Calculate the success rate as a percentage (0.0 - 1.0)."""
        if self.total_providers == 0:
            return 0.0
        return self.successful_providers / self.total_providers

    @property
    def has_consensus(self) -> bool:
        """Return True if at least 2 providers succeeded."""
        return self.successful_providers >= 2

    @classmethod
    def from_responses(
        cls, responses: Sequence["ProviderResponse"]
    ) -> "AgreementMetadata":
        """
        Create AgreementMetadata from a list of provider responses.

        Args:
            responses: Sequence of ProviderResponse objects

        Returns:
            AgreementMetadata with computed counts
        """
        total = len(responses)
        successful = sum(1 for r in responses if r.success)
        failed = total - successful
        return cls(
            total_providers=total,
            successful_providers=successful,
            failed_providers=failed,
        )


@dataclass
class ConsensusResult:
    """
    Aggregated result from multi-model consensus consultation.

    Collects responses from multiple providers along with metadata about
    agreement levels and overall success. Used when min_models > 1 in
    workflow configuration.

    Attributes:
        workflow: The consultation workflow that produced this result
        responses: List of individual provider responses
        agreement: Metadata about provider agreement and success rates
        duration_ms: Total consultation duration in milliseconds
        warnings: Non-fatal issues encountered during consultation

    Properties:
        success: True if at least one provider succeeded
        primary_content: Content from the first successful response (for compatibility)
    """

    workflow: ConsultationWorkflow
    responses: List[ProviderResponse] = field(default_factory=list)
    agreement: Optional[AgreementMetadata] = None
    duration_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Auto-compute agreement metadata if not provided."""
        if self.agreement is None and self.responses:
            self.agreement = AgreementMetadata.from_responses(self.responses)

    @property
    def success(self) -> bool:
        """Return True if at least one provider returned a successful response."""
        return any(r.success for r in self.responses)

    @property
    def primary_content(self) -> str:
        """
        Return content from the first successful response.

        For backward compatibility with code expecting a single response.
        Returns empty string if no successful responses.
        """
        for response in self.responses:
            if response.success and response.content:
                return response.content
        return ""

    @property
    def successful_responses(self) -> List[ProviderResponse]:
        """Return list of successful responses only."""
        return [r for r in self.responses if r.success]

    @property
    def failed_responses(self) -> List[ProviderResponse]:
        """Return list of failed responses only."""
        return [r for r in self.responses if not r.success]


# Type alias for backward-compatible result handling
ConsultationOutcome = Union[ConsultationResult, ConsensusResult]
"""
Type alias for consultation results supporting both single and multi-model modes.

When min_models == 1 (default): Returns ConsultationResult (single provider)
When min_models > 1: Returns ConsensusResult (multiple providers with agreement)

Use isinstance() to differentiate:
    if isinstance(outcome, ConsensusResult):
        # Handle multi-model result with agreement metadata
    else:
        # Handle single-model ConsultationResult
"""


__all__ = [
    "ConsultationWorkflow",
    "ResolvedProvider",
    "ConsultationRequest",
    "ConsultationResult",
    "ProviderResponse",
    "AgreementMetadata",
    "ConsensusResult",
    "ConsultationOutcome",
]
