"""CONSENSUS workflow models (multi-model parallel execution)."""

from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from foundry_mcp.core.research.models.enums import ConsensusStrategy


class ModelResponse(BaseModel):
    """A response from a single model in CONSENSUS workflow."""

    provider_id: str = Field(..., description="Provider that generated this response")
    model_used: Optional[str] = Field(default=None)
    content: str = Field(..., description="Response content")
    success: bool = Field(default=True)
    error_message: Optional[str] = Field(default=None)
    tokens_used: Optional[int] = Field(default=None)
    duration_ms: Optional[float] = Field(default=None)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConsensusConfig(BaseModel):
    """Configuration for a CONSENSUS workflow execution."""

    providers: list[str] = Field(..., description="List of provider IDs to consult", min_length=1)
    strategy: ConsensusStrategy = Field(default=ConsensusStrategy.SYNTHESIZE)
    synthesis_provider: Optional[str] = Field(
        default=None, description="Provider to use for synthesis (if strategy=synthesize)"
    )
    timeout_per_provider: float = Field(default=360.0, description="Timeout in seconds per provider")
    max_concurrent: int = Field(default=3, description="Maximum concurrent provider calls")
    require_all: bool = Field(default=False, description="Require all providers to succeed")
    min_responses: int = Field(default=1, description="Minimum responses needed for success")


class ConsensusState(BaseModel):
    """State for a CONSENSUS workflow execution."""

    id: str = Field(default_factory=lambda: f"consensus-{uuid4().hex[:12]}")
    prompt: str = Field(..., description="The prompt sent to all providers")
    config: ConsensusConfig = Field(..., description="Consensus configuration")
    responses: list[ModelResponse] = Field(default_factory=list)
    synthesis: Optional[str] = Field(default=None, description="Synthesized response if strategy requires it")
    completed: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(default=None)
    system_prompt: Optional[str] = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_response(self, response: ModelResponse) -> None:
        """Add a model response to the consensus."""
        self.responses.append(response)

    def successful_responses(self) -> list[ModelResponse]:
        """Get only successful responses."""
        return [r for r in self.responses if r.success]

    def failed_responses(self) -> list[ModelResponse]:
        """Get failed responses."""
        return [r for r in self.responses if not r.success]

    def is_quorum_met(self) -> bool:
        """Check if minimum response requirement is met."""
        return len(self.successful_responses()) >= self.config.min_responses

    def mark_completed(self, synthesis: Optional[str] = None) -> None:
        """Mark the consensus as completed."""
        self.completed = True
        self.completed_at = datetime.utcnow()
        if synthesis:
            self.synthesis = synthesis
