"""Pydantic models for research workflows.

These models define the data structures for conversation threading,
multi-model consensus, hypothesis-driven investigation, and creative
brainstorming workflows.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================


class WorkflowType(str, Enum):
    """Types of research workflows available."""

    CHAT = "chat"
    CONSENSUS = "consensus"
    THINKDEEP = "thinkdeep"
    IDEATE = "ideate"


class ConfidenceLevel(str, Enum):
    """Confidence levels for hypotheses in THINKDEEP workflow."""

    SPECULATION = "speculation"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CONFIRMED = "confirmed"


class ConsensusStrategy(str, Enum):
    """Strategies for synthesizing multi-model responses in CONSENSUS workflow."""

    ALL_RESPONSES = "all_responses"  # Return all responses without synthesis
    SYNTHESIZE = "synthesize"  # Use a model to synthesize responses
    MAJORITY = "majority"  # Use majority vote for factual questions
    FIRST_VALID = "first_valid"  # Return first successful response


class ThreadStatus(str, Enum):
    """Status of a conversation thread."""

    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class IdeationPhase(str, Enum):
    """Phases of the IDEATE workflow."""

    DIVERGENT = "divergent"  # Generate diverse ideas
    CONVERGENT = "convergent"  # Cluster and score ideas
    SELECTION = "selection"  # Select clusters for elaboration
    ELABORATION = "elaboration"  # Develop selected ideas


# =============================================================================
# Conversation Models (CHAT workflow)
# =============================================================================


class ConversationMessage(BaseModel):
    """A single message in a conversation thread."""

    id: str = Field(default_factory=lambda: f"msg-{uuid4().hex[:8]}")
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    provider_id: Optional[str] = Field(
        default=None, description="Provider that generated this message"
    )
    model_used: Optional[str] = Field(
        default=None, description="Model that generated this message"
    )
    tokens_used: Optional[int] = Field(
        default=None, description="Tokens consumed for this message"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional message metadata"
    )


class ConversationThread(BaseModel):
    """A conversation thread with message history."""

    id: str = Field(default_factory=lambda: f"thread-{uuid4().hex[:12]}")
    title: Optional[str] = Field(default=None, description="Optional thread title")
    status: ThreadStatus = Field(default=ThreadStatus.ACTIVE)
    messages: list[ConversationMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    provider_id: Optional[str] = Field(
        default=None, description="Default provider for this thread"
    )
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt for this thread"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional thread metadata"
    )

    def add_message(
        self,
        role: str,
        content: str,
        provider_id: Optional[str] = None,
        model_used: Optional[str] = None,
        tokens_used: Optional[int] = None,
        **metadata: Any,
    ) -> ConversationMessage:
        """Add a message to the thread and update timestamp."""
        message = ConversationMessage(
            role=role,
            content=content,
            provider_id=provider_id,
            model_used=model_used,
            tokens_used=tokens_used,
            metadata=metadata,
        )
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
        return message

    def get_context_messages(
        self, max_messages: Optional[int] = None
    ) -> list[ConversationMessage]:
        """Get messages for context, optionally limited to recent N messages."""
        if max_messages is None or max_messages >= len(self.messages):
            return self.messages
        return self.messages[-max_messages:]


# =============================================================================
# THINKDEEP Models (Hypothesis-driven investigation)
# =============================================================================


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
    hypotheses_updated: list[str] = Field(
        default_factory=list, description="IDs of hypotheses updated in this step"
    )
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
    converged: bool = Field(
        default=False, description="Whether investigation has converged"
    )
    convergence_reason: Optional[str] = Field(
        default=None, description="Reason for convergence if converged"
    )
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
        step = InvestigationStep(
            depth=depth if depth is not None else self.current_depth, query=query
        )
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
            h.confidence in (ConfidenceLevel.HIGH, ConfidenceLevel.CONFIRMED)
            for h in self.hypotheses
        ):
            self.converged = True
            self.convergence_reason = "All hypotheses reached high confidence"
            return True

        return False


# =============================================================================
# IDEATE Models (Creative brainstorming)
# =============================================================================


class Idea(BaseModel):
    """A single idea generated in IDEATE workflow."""

    id: str = Field(default_factory=lambda: f"idea-{uuid4().hex[:8]}")
    content: str = Field(..., description="The idea content")
    perspective: Optional[str] = Field(
        default=None, description="Perspective that generated this idea"
    )
    score: Optional[float] = Field(
        default=None, description="Score from 0-1 based on criteria"
    )
    cluster_id: Optional[str] = Field(
        default=None, description="ID of cluster this idea belongs to"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    provider_id: Optional[str] = Field(default=None)
    model_used: Optional[str] = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IdeaCluster(BaseModel):
    """A cluster of related ideas in IDEATE workflow."""

    id: str = Field(default_factory=lambda: f"cluster-{uuid4().hex[:8]}")
    name: str = Field(..., description="Cluster name/theme")
    description: Optional[str] = Field(default=None, description="Cluster description")
    idea_ids: list[str] = Field(default_factory=list, description="IDs of ideas in cluster")
    average_score: Optional[float] = Field(default=None)
    selected_for_elaboration: bool = Field(default=False)
    elaboration: Optional[str] = Field(
        default=None, description="Detailed elaboration if selected"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IdeationState(BaseModel):
    """State for an IDEATE brainstorming session."""

    id: str = Field(default_factory=lambda: f"ideation-{uuid4().hex[:12]}")
    topic: str = Field(..., description="The topic being brainstormed")
    phase: IdeationPhase = Field(default=IdeationPhase.DIVERGENT)
    perspectives: list[str] = Field(
        default_factory=lambda: ["technical", "creative", "practical", "visionary"]
    )
    ideas: list[Idea] = Field(default_factory=list)
    clusters: list[IdeaCluster] = Field(default_factory=list)
    scoring_criteria: list[str] = Field(
        default_factory=lambda: ["novelty", "feasibility", "impact"]
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    system_prompt: Optional[str] = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_idea(
        self,
        content: str,
        perspective: Optional[str] = None,
        **kwargs: Any,
    ) -> Idea:
        """Add a new idea to the session."""
        idea = Idea(content=content, perspective=perspective, **kwargs)
        self.ideas.append(idea)
        self.updated_at = datetime.utcnow()
        return idea

    def create_cluster(self, name: str, description: Optional[str] = None) -> IdeaCluster:
        """Create a new idea cluster."""
        cluster = IdeaCluster(name=name, description=description)
        self.clusters.append(cluster)
        self.updated_at = datetime.utcnow()
        return cluster

    def assign_idea_to_cluster(self, idea_id: str, cluster_id: str) -> bool:
        """Assign an idea to a cluster."""
        idea = next((i for i in self.ideas if i.id == idea_id), None)
        cluster = next((c for c in self.clusters if c.id == cluster_id), None)

        if idea and cluster:
            idea.cluster_id = cluster_id
            if idea_id not in cluster.idea_ids:
                cluster.idea_ids.append(idea_id)
            self.updated_at = datetime.utcnow()
            return True
        return False

    def advance_phase(self) -> IdeationPhase:
        """Advance to the next ideation phase."""
        phase_order = list(IdeationPhase)
        current_index = phase_order.index(self.phase)
        if current_index < len(phase_order) - 1:
            self.phase = phase_order[current_index + 1]
        self.updated_at = datetime.utcnow()
        return self.phase


# =============================================================================
# CONSENSUS Models (Multi-model parallel execution)
# =============================================================================


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

    providers: list[str] = Field(
        ..., description="List of provider IDs to consult", min_length=1
    )
    strategy: ConsensusStrategy = Field(default=ConsensusStrategy.SYNTHESIZE)
    synthesis_provider: Optional[str] = Field(
        default=None, description="Provider to use for synthesis (if strategy=synthesize)"
    )
    timeout_per_provider: float = Field(
        default=30.0, description="Timeout in seconds per provider"
    )
    max_concurrent: int = Field(
        default=3, description="Maximum concurrent provider calls"
    )
    require_all: bool = Field(
        default=False, description="Require all providers to succeed"
    )
    min_responses: int = Field(
        default=1, description="Minimum responses needed for success"
    )


class ConsensusState(BaseModel):
    """State for a CONSENSUS workflow execution."""

    id: str = Field(default_factory=lambda: f"consensus-{uuid4().hex[:12]}")
    prompt: str = Field(..., description="The prompt sent to all providers")
    config: ConsensusConfig = Field(..., description="Consensus configuration")
    responses: list[ModelResponse] = Field(default_factory=list)
    synthesis: Optional[str] = Field(
        default=None, description="Synthesized response if strategy requires it"
    )
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
