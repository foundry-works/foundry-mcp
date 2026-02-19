"""THINKDEEP workflow models (hypothesis-driven investigation)."""

from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from foundry_mcp.core.research.models.enums import ConfidenceLevel


class Hypothesis(BaseModel):
    """A hypothesis being investigated in THINKDEEP workflow."""

    id: str = Field(default_factory=lambda: f"hyp-{uuid4().hex[:8]}")
    statement: str = Field(..., description="The hypothesis statement")
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.SPECULATION)
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_evidence(self, evidence: str, supporting: bool = True) -> None:
        """Add evidence for or against this hypothesis."""
        if supporting:
            self.supporting_evidence.append(evidence)
        else:
            self.contradicting_evidence.append(evidence)
        self.updated_at = datetime.utcnow()

    def update_confidence(self, new_confidence: ConfidenceLevel) -> None:
        """Update the confidence level of this hypothesis."""
        self.confidence = new_confidence
        self.updated_at = datetime.utcnow()


class InvestigationStep(BaseModel):
    """A single step in a THINKDEEP investigation."""

    id: str = Field(default_factory=lambda: f"step-{uuid4().hex[:8]}")
    depth: int = Field(..., description="Depth level of this step (0-indexed)")
    query: str = Field(..., description="The question or query for this step")
    response: Optional[str] = Field(default=None, description="Model response")
    hypotheses_generated: list[str] = Field(
        default_factory=list, description="IDs of hypotheses generated in this step"
    )
    hypotheses_updated: list[str] = Field(default_factory=list, description="IDs of hypotheses updated in this step")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    provider_id: Optional[str] = Field(default=None)
    model_used: Optional[str] = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ThinkDeepState(BaseModel):
    """State for a THINKDEEP investigation session."""

    id: str = Field(default_factory=lambda: f"investigation-{uuid4().hex[:12]}")
    topic: str = Field(..., description="The topic being investigated")
    current_depth: int = Field(default=0, description="Current investigation depth")
    max_depth: int = Field(default=5, description="Maximum investigation depth")
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    steps: list[InvestigationStep] = Field(default_factory=list)
    converged: bool = Field(default=False, description="Whether investigation has converged")
    convergence_reason: Optional[str] = Field(default=None, description="Reason for convergence if converged")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    system_prompt: Optional[str] = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_hypothesis(self, statement: str, **kwargs: Any) -> Hypothesis:
        """Create and add a new hypothesis."""
        hypothesis = Hypothesis(statement=statement, **kwargs)
        self.hypotheses.append(hypothesis)
        self.updated_at = datetime.utcnow()
        return hypothesis

    def get_hypothesis(self, hypothesis_id: str) -> Optional[Hypothesis]:
        """Get a hypothesis by ID."""
        for h in self.hypotheses:
            if h.id == hypothesis_id:
                return h
        return None

    def add_step(self, query: str, depth: Optional[int] = None) -> InvestigationStep:
        """Create and add a new investigation step."""
        step = InvestigationStep(depth=depth if depth is not None else self.current_depth, query=query)
        self.steps.append(step)
        self.updated_at = datetime.utcnow()
        return step

    def check_convergence(self) -> bool:
        """Check if investigation should converge based on criteria."""
        # Converge if max depth reached
        if self.current_depth >= self.max_depth:
            self.converged = True
            self.convergence_reason = "Maximum depth reached"
            return True

        # Converge if all hypotheses have high confidence
        if self.hypotheses and all(
            h.confidence in (ConfidenceLevel.HIGH, ConfidenceLevel.CONFIRMED) for h in self.hypotheses
        ):
            self.converged = True
            self.convergence_reason = "All hypotheses reached high confidence"
            return True

        return False
