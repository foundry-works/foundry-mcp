"""Fidelity tracking models for token budget management."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class PhaseMetrics(BaseModel):
    """Metrics for a single phase execution.

    Tracks timing, token usage, and provider information for each phase
    of the deep research workflow. Used for audit and cost tracking.
    """

    phase: str = Field(..., description="Phase name (planning, analysis, etc.)")
    duration_ms: float = Field(default=0.0, description="Phase duration in milliseconds")
    input_tokens: int = Field(default=0, description="Tokens consumed by the prompt")
    output_tokens: int = Field(default=0, description="Tokens generated in the response")
    cached_tokens: int = Field(default=0, description="Tokens served from cache")
    provider_id: Optional[str] = Field(default=None, description="Provider used for this phase")
    model_used: Optional[str] = Field(default=None, description="Model used for this phase")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible metadata (e.g. token_limit_retries, model_roles)",
    )
