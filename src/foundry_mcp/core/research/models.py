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
# Fragment ID Utilities
# =============================================================================


def make_fragment_id(base_id: str, fragment_index: int) -> str:
    """Generate a stable fragment ID for chunked content.

    Creates a predictable ID for content fragments by appending a
    fragment index to the base item ID. This enables tracking fidelity
    at the chunk level while maintaining parent item relationships.

    Args:
        base_id: Base item ID (e.g., "src-abc123")
        fragment_index: Zero-based index of the fragment/chunk

    Returns:
        Fragment ID in format "{base_id}#fragment-{N}"

    Examples:
        >>> make_fragment_id("src-abc123", 0)
        'src-abc123#fragment-0'
        >>> make_fragment_id("src-abc123", 3)
        'src-abc123#fragment-3'
    """
    return f"{base_id}#fragment-{fragment_index}"


def parse_fragment_id(fragment_id: str) -> tuple[str, Optional[int]]:
    """Parse a fragment ID into base ID and fragment index.

    Extracts the base item ID and optional fragment index from a
    fragment ID. If the ID doesn't contain a fragment suffix, returns
    the original ID with None for the fragment index.

    Args:
        fragment_id: ID that may contain fragment suffix

    Returns:
        Tuple of (base_id, fragment_index) where fragment_index is
        None if no fragment suffix was present

    Examples:
        >>> parse_fragment_id("src-abc123#fragment-0")
        ('src-abc123', 0)
        >>> parse_fragment_id("src-abc123")
        ('src-abc123', None)
    """
    if "#fragment-" not in fragment_id:
        return fragment_id, None

    base_id, suffix = fragment_id.rsplit("#fragment-", 1)
    try:
        fragment_index = int(suffix)
        return base_id, fragment_index
    except ValueError:
        # Invalid fragment suffix, return original as-is
        return fragment_id, None


def is_fragment_id(item_id: str) -> bool:
    """Check if an ID is a fragment ID.

    Args:
        item_id: ID to check

    Returns:
        True if the ID contains a fragment suffix

    Examples:
        >>> is_fragment_id("src-abc123#fragment-0")
        True
        >>> is_fragment_id("src-abc123")
        False
    """
    _, fragment_index = parse_fragment_id(item_id)
    return fragment_index is not None


def get_base_id(item_id: str) -> str:
    """Get the base ID from a potentially fragment ID.

    Strips the fragment suffix if present, returning the original
    item ID.

    Args:
        item_id: ID that may contain fragment suffix

    Returns:
        Base item ID without fragment suffix

    Examples:
        >>> get_base_id("src-abc123#fragment-0")
        'src-abc123'
        >>> get_base_id("src-abc123")
        'src-abc123'
    """
    base_id, _ = parse_fragment_id(item_id)
    return base_id


# =============================================================================
# Enums
# =============================================================================


class WorkflowType(str, Enum):
    """Types of research workflows available."""

    CHAT = "chat"
    CONSENSUS = "consensus"
    THINKDEEP = "thinkdeep"
    IDEATE = "ideate"
    DEEP_RESEARCH = "deep_research"


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
        default=360.0, description="Timeout in seconds per provider"
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


class DeepResearchConfig(BaseModel):
    """Configuration for DEEP_RESEARCH workflow execution.

    Groups deep research parameters into a single config object to reduce
    parameter sprawl in the MCP tool interface. All fields have sensible
    defaults that can be overridden at the tool level.

    Note: Provider configuration is handled via ResearchConfig TOML settings,
    not through this config object. This is intentional - providers should be
    configured at the server level, not per-request.
    """

    max_iterations: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum refinement iterations before forced completion",
    )
    max_sub_queries: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum sub-queries for query decomposition",
    )
    max_sources_per_query: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum sources to gather per sub-query",
    )
    follow_links: bool = Field(
        default=True,
        description="Whether to follow URLs and extract full content",
    )
    timeout_per_operation: float = Field(
        default=360.0,
        ge=1.0,
        le=1800.0,
        description="Timeout in seconds for each search/fetch operation",
    )
    max_concurrent: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum concurrent operations (search, fetch)",
    )

    @classmethod
    def from_defaults(cls) -> "DeepResearchConfig":
        """Create config with all default values.

        Returns:
            DeepResearchConfig with sensible defaults
        """
        return cls()

    def merge_overrides(self, **overrides: Any) -> "DeepResearchConfig":
        """Create a new config with specified overrides applied.

        Args:
            **overrides: Field values to override (None values are ignored)

        Returns:
            New DeepResearchConfig with overrides applied
        """
        current = self.model_dump()
        for key, value in overrides.items():
            if value is not None and key in current:
                current[key] = value
        return DeepResearchConfig(**current)


# =============================================================================
# DEEP RESEARCH Models (Multi-phase iterative research)
# =============================================================================


class DeepResearchPhase(str, Enum):
    """Phases of the DEEP_RESEARCH workflow.

    The deep research workflow progresses through five sequential phases:
    1. PLANNING - Analyze the query and decompose into focused sub-queries
    2. GATHERING - Execute sub-queries in parallel and collect sources
    3. ANALYSIS - Extract findings and assess source quality
    4. SYNTHESIS - Combine findings into a comprehensive report
    5. REFINEMENT - Identify gaps and potentially loop back for more research

    The ordering of these enum values is significant - it defines the
    progression through advance_phase() method.
    """

    PLANNING = "planning"
    GATHERING = "gathering"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    REFINEMENT = "refinement"


class FidelityLevel(str, Enum):
    """Content fidelity levels for token budget management.

    Defines how much content has been preserved or compressed during
    budget allocation. Each level represents a progressively more
    aggressive compression applied to fit within token constraints.

    Levels (ordered from highest to lowest fidelity):
        FULL: Content unchanged - original content preserved
        CONDENSED: Light summarization (~50-70% of original)
        KEY_POINTS: Bullet point extraction (~20-40% of original)
        HEADLINE: Single sentence summary (~5-10% of original)
        TRUNCATED: Hard cut with marker (arbitrary %)
        DROPPED: Content completely removed (0%)
    """

    FULL = "full"
    CONDENSED = "condensed"
    KEY_POINTS = "key_points"
    HEADLINE = "headline"
    TRUNCATED = "truncated"
    DROPPED = "dropped"

    @property
    def is_degraded(self) -> bool:
        """Check if this level represents degraded content."""
        return self != FidelityLevel.FULL

    @property
    def is_available(self) -> bool:
        """Check if content is still available (not dropped)."""
        return self != FidelityLevel.DROPPED


class PhaseContentFidelityRecord(BaseModel):
    """Record of fidelity for a specific content item in a specific phase.

    Tracks when and why content was degraded during a particular
    workflow phase, along with any warnings generated.

    Attributes:
        level: Fidelity level applied in this phase
        reason: Why degradation was applied (e.g., "budget_exceeded")
        warnings: Any warnings generated during processing
        timestamp: When this fidelity was applied
        original_tokens: Token count before degradation
        final_tokens: Token count after degradation
    """

    level: FidelityLevel = Field(
        default=FidelityLevel.FULL,
        description="Fidelity level applied in this phase",
    )
    reason: str = Field(
        default="",
        description="Why degradation was applied (e.g., 'budget_exceeded', 'priority_low')",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Any warnings generated during processing",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this fidelity was applied",
    )
    original_tokens: Optional[int] = Field(
        default=None,
        description="Token count before degradation",
    )
    final_tokens: Optional[int] = Field(
        default=None,
        description="Token count after degradation",
    )


class ContentFidelityRecord(BaseModel):
    """Tracks fidelity history for a single content item across all phases.

    Maintains a per-phase record of how content fidelity changed throughout
    the workflow. This enables auditing of content degradation decisions
    and supports potential future content restoration.

    The `phases` dict is keyed by phase name (e.g., "analysis", "synthesis")
    and contains the fidelity record for that phase.

    Attributes:
        item_id: Unique identifier for the content item (source/finding/gap ID)
        item_type: Type of content ("source", "finding", "gap")
        phases: Per-phase fidelity records, keyed by phase name
        current_level: Most recent fidelity level (convenience field)
        created_at: When tracking began for this item
        updated_at: Last time any phase record was updated
    """

    item_id: str = Field(
        ...,
        description="Unique identifier for the content item",
    )
    item_type: str = Field(
        default="source",
        description="Type of content: 'source', 'finding', 'gap'",
    )
    phases: dict[str, PhaseContentFidelityRecord] = Field(
        default_factory=dict,
        description="Per-phase fidelity records, keyed by phase name",
    )
    current_level: FidelityLevel = Field(
        default=FidelityLevel.FULL,
        description="Most recent fidelity level (convenience field)",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When tracking began for this item",
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last time any phase record was updated",
    )

    def record_phase(
        self,
        phase: str,
        level: FidelityLevel,
        reason: str = "",
        warnings: Optional[list[str]] = None,
        original_tokens: Optional[int] = None,
        final_tokens: Optional[int] = None,
    ) -> None:
        """Record fidelity for a specific phase.

        Args:
            phase: Phase name (e.g., "analysis", "synthesis")
            level: Fidelity level applied
            reason: Why degradation was applied
            warnings: Any warnings generated
            original_tokens: Token count before degradation
            final_tokens: Token count after degradation
        """
        self.phases[phase] = PhaseContentFidelityRecord(
            level=level,
            reason=reason,
            warnings=warnings or [],
            original_tokens=original_tokens,
            final_tokens=final_tokens,
        )
        self.current_level = level
        self.updated_at = datetime.utcnow()

    def get_phase(self, phase: str) -> Optional[PhaseContentFidelityRecord]:
        """Get fidelity record for a specific phase.

        Args:
            phase: Phase name to look up

        Returns:
            PhaseContentFidelityRecord if exists, None otherwise
        """
        return self.phases.get(phase)

    def merge_phases_from(self, other: "ContentFidelityRecord") -> None:
        """Merge phase records from another ContentFidelityRecord.

        Implements the fidelity merge rules:
        - Latest phase overwrites same-phase entry (by timestamp)
        - Prior phases are preserved for history

        For each phase in `other`:
        - If phase doesn't exist in self, add it
        - If phase exists, keep the one with the later timestamp

        This enables reconstructing fidelity history after content
        re-processing or migration scenarios.

        Args:
            other: Another ContentFidelityRecord to merge from
        """
        for phase_name, other_record in other.phases.items():
            if phase_name not in self.phases:
                # New phase - add it
                self.phases[phase_name] = other_record
            else:
                # Existing phase - keep the latest by timestamp
                self_record = self.phases[phase_name]
                if other_record.timestamp > self_record.timestamp:
                    self.phases[phase_name] = other_record

        # Update current_level to the most recent phase's level
        if self.phases:
            latest_phase = max(
                self.phases.values(),
                key=lambda r: r.timestamp,
            )
            self.current_level = latest_phase.level

        self.updated_at = datetime.utcnow()

    def get_phases_for_item(self) -> list[str]:
        """Get all phase names recorded for this item.

        Returns:
            List of phase names in chronological order (by timestamp)
        """
        sorted_phases = sorted(
            self.phases.items(),
            key=lambda kv: kv[1].timestamp,
        )
        return [phase_name for phase_name, _ in sorted_phases]

    def get_fidelity_history(self) -> list[dict[str, Any]]:
        """Get the fidelity history across all phases.

        Returns a list of records showing how fidelity changed over time,
        ordered chronologically. Useful for debugging and auditing.

        Returns:
            List of dicts with phase, level, reason, timestamp
        """
        history = []
        for phase_name, record in sorted(
            self.phases.items(),
            key=lambda kv: kv[1].timestamp,
        ):
            history.append({
                "phase": phase_name,
                "level": record.level.value,
                "reason": record.reason,
                "timestamp": record.timestamp.isoformat(),
                "original_tokens": record.original_tokens,
                "final_tokens": record.final_tokens,
            })
        return history


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


class SourceType(str, Enum):
    """Types of research sources that can be discovered.

    V1 Implementation:
    - WEB: General web search results (via Tavily/Google)
    - ACADEMIC: Academic papers and journals (via Semantic Scholar)

    Future Extensions (placeholders):
    - EXPERT: Expert profiles and interviews (reserved)
    - CODE: Code repositories and examples (reserved for GitHub search)
    - NEWS: News articles and press releases
    - DOCUMENTATION: Technical documentation
    """

    WEB = "web"
    ACADEMIC = "academic"
    EXPERT = "expert"  # Future: expert profiles, interviews
    CODE = "code"  # Future: GitHub, code search


class SourceQuality(str, Enum):
    """Quality assessment for research sources.

    Quality levels are assigned during the ANALYSIS phase based on:
    - Source authority and credibility
    - Content recency and relevance
    - Citation count and peer review status (for academic)
    - Domain reputation (for web sources)
    """

    UNKNOWN = "unknown"  # Not yet assessed
    LOW = "low"  # Questionable reliability
    MEDIUM = "medium"  # Generally reliable
    HIGH = "high"  # Authoritative source


class ResearchMode(str, Enum):
    """Research modes that control source prioritization.

    Each mode applies different domain-based quality heuristics:
    - GENERAL: No domain preferences, balanced approach (default)
    - ACADEMIC: Prioritizes journals, publishers, preprints
    - TECHNICAL: Prioritizes official docs, arxiv, code repositories
    """

    GENERAL = "general"
    ACADEMIC = "academic"
    TECHNICAL = "technical"


# Domain tier lists for source quality assessment by research mode
# Patterns support wildcards: "*.edu" matches any .edu domain
DOMAIN_TIERS: dict[str, dict[str, list[str]]] = {
    "academic": {
        "high": [
            # Aggregators & indexes
            "scholar.google.com",
            "semanticscholar.org",
            "pubmed.gov",
            "ncbi.nlm.nih.gov",
            "jstor.org",
            # Major publishers
            "springer.com",
            "link.springer.com",
            "sciencedirect.com",
            "elsevier.com",
            "wiley.com",
            "onlinelibrary.wiley.com",
            "tandfonline.com",  # Taylor & Francis
            "sagepub.com",
            "nature.com",
            "science.org",  # AAAS/Science
            "frontiersin.org",
            "plos.org",
            "journals.plos.org",
            "mdpi.com",
            "oup.com",
            "academic.oup.com",  # Oxford
            "cambridge.org",
            # Preprints & open access
            "arxiv.org",
            "biorxiv.org",
            "medrxiv.org",
            "psyarxiv.com",
            "ssrn.com",
            # Field-specific
            "apa.org",
            "psycnet.apa.org",  # Psychology
            "aclanthology.org",  # Computational linguistics
            # CS/Tech academic
            "acm.org",
            "dl.acm.org",
            "ieee.org",
            "ieeexplore.ieee.org",
            # Institutional patterns
            "*.edu",
            "*.ac.uk",
            "*.edu.au",
        ],
        "low": [
            "reddit.com",
            "quora.com",
            "medium.com",
            "linkedin.com",
            "twitter.com",
            "x.com",
            "facebook.com",
            "pinterest.com",
            "instagram.com",
            "tiktok.com",
            "youtube.com",  # Can have good content but inconsistent
        ],
    },
    "technical": {
        "high": [
            # Preprints (technical papers)
            "arxiv.org",
            # Official documentation patterns
            "docs.*",
            "developer.*",
            "*.dev",
            "devdocs.io",
            # Code & technical resources
            "github.com",
            "stackoverflow.com",
            "stackexchange.com",
            # Language/framework official sites
            "python.org",
            "docs.python.org",
            "nodejs.org",
            "rust-lang.org",
            "doc.rust-lang.org",
            "go.dev",
            "typescriptlang.org",
            "react.dev",
            "vuejs.org",
            "angular.io",
            # Cloud providers
            "aws.amazon.com",
            "cloud.google.com",
            "docs.microsoft.com",
            "learn.microsoft.com",
            "azure.microsoft.com",
            # Tech company engineering blogs
            "engineering.fb.com",
            "netflixtechblog.com",
            "uber.com/blog/engineering",
            "blog.google",
            # Academic (relevant for technical research)
            "acm.org",
            "dl.acm.org",
            "ieee.org",
            "ieeexplore.ieee.org",
        ],
        "low": [
            "reddit.com",
            "quora.com",
            "linkedin.com",
            "twitter.com",
            "x.com",
            "facebook.com",
            "pinterest.com",
        ],
    },
    "general": {
        "high": [],  # No domain preferences
        "low": [
            # Still deprioritize social media
            "pinterest.com",
            "facebook.com",
            "instagram.com",
            "tiktok.com",
        ],
    },
}


class SubQuery(BaseModel):
    """A decomposed sub-query for focused research.

    During the PLANNING phase, the original research query is decomposed
    into multiple focused sub-queries. Each sub-query targets a specific
    aspect of the research question and can be executed independently
    during the GATHERING phase.

    Status transitions:
    - pending -> executing -> completed (success path)
    - pending -> executing -> failed (error path)
    """

    id: str = Field(default_factory=lambda: f"subq-{uuid4().hex[:8]}")
    query: str = Field(..., description="The focused sub-query text")
    rationale: Optional[str] = Field(
        default=None,
        description="Why this sub-query was generated and what aspect it covers",
    )
    priority: int = Field(
        default=1,
        description="Execution priority (1=highest, larger=lower priority)",
    )
    status: str = Field(
        default="pending",
        description="Current status: pending, executing, completed, failed",
    )
    source_ids: list[str] = Field(
        default_factory=list,
        description="IDs of ResearchSource objects found for this query",
    )
    findings_summary: Optional[str] = Field(
        default=None,
        description="Brief summary of what was found from this sub-query",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(default=None)
    error: Optional[str] = Field(
        default=None,
        description="Error message if status is 'failed'",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    def mark_completed(self, findings: Optional[str] = None) -> None:
        """Mark this sub-query as successfully completed.

        Args:
            findings: Optional summary of findings from this sub-query
        """
        self.status = "completed"
        self.completed_at = datetime.utcnow()
        if findings:
            self.findings_summary = findings

    def mark_failed(self, error: str) -> None:
        """Mark this sub-query as failed with an error message.

        Args:
            error: Description of why the sub-query failed
        """
        self.status = "failed"
        self.completed_at = datetime.utcnow()
        self.error = error


class ResearchSource(BaseModel):
    """A source discovered during research.

    Sources are collected during the GATHERING phase when sub-queries
    are executed against search providers. Each source represents a
    piece of external content (web page, paper, etc.) that may contain
    relevant information for the research query.

    Quality is assessed during the ANALYSIS phase based on source
    authority, content relevance, and other factors.
    """

    id: str = Field(default_factory=lambda: f"src-{uuid4().hex[:8]}")
    url: Optional[str] = Field(
        default=None,
        description="URL of the source (may be None for non-web sources)",
    )
    title: str = Field(..., description="Title or headline of the source")
    source_type: SourceType = Field(
        default=SourceType.WEB,
        description="Type of source (web, academic, etc.)",
    )
    quality: SourceQuality = Field(
        default=SourceQuality.UNKNOWN,
        description="Assessed quality level of this source",
    )
    snippet: Optional[str] = Field(
        default=None,
        description="Brief excerpt or description from the source",
    )
    content: Optional[str] = Field(
        default=None,
        description="Full extracted content (if follow_links enabled)",
    )
    sub_query_id: Optional[str] = Field(
        default=None,
        description="ID of the SubQuery that discovered this source",
    )
    discovered_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchFinding(BaseModel):
    """A key finding extracted from research sources.

    Findings are extracted during the ANALYSIS phase by examining
    source content and identifying key insights. Each finding has
    an associated confidence level and links back to supporting sources.

    Findings are organized by category/theme during synthesis to
    create a structured report.
    """

    id: str = Field(default_factory=lambda: f"find-{uuid4().hex[:8]}")
    content: str = Field(..., description="The key finding or insight")
    confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.MEDIUM,
        description="Confidence level in this finding",
    )
    source_ids: list[str] = Field(
        default_factory=list,
        description="IDs of ResearchSource objects supporting this finding",
    )
    sub_query_id: Optional[str] = Field(
        default=None,
        description="ID of SubQuery that produced this finding",
    )
    category: Optional[str] = Field(
        default=None,
        description="Theme or category for organizing findings",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchGap(BaseModel):
    """An identified gap in the research requiring follow-up.

    Gaps are identified during the ANALYSIS and SYNTHESIS phases when
    the research reveals missing information or unanswered questions.
    Each gap includes suggested follow-up queries that can be used
    in subsequent refinement iterations.

    Gaps drive the REFINEMENT phase: if unresolved gaps exist and
    max_iterations hasn't been reached, the workflow loops back
    to GATHERING with new sub-queries derived from gap suggestions.
    """

    id: str = Field(default_factory=lambda: f"gap-{uuid4().hex[:8]}")
    description: str = Field(
        ...,
        description="Description of the knowledge gap or missing information",
    )
    suggested_queries: list[str] = Field(
        default_factory=list,
        description="Follow-up queries that could fill this gap",
    )
    priority: int = Field(
        default=1,
        description="Priority for follow-up (1=highest, larger=lower priority)",
    )
    resolved: bool = Field(
        default=False,
        description="Whether this gap has been addressed in a refinement iteration",
    )
    resolution_notes: Optional[str] = Field(
        default=None,
        description="Notes on how the gap was resolved",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DeepResearchState(BaseModel):
    """Main state model for a deep research session.

    Manages the entire lifecycle of a multi-phase research workflow:
    - Tracks the current phase and iteration
    - Contains all sub-queries, sources, findings, and gaps
    - Provides helper methods for state manipulation
    - Handles phase advancement and refinement iteration logic

    The state is persisted to enable session resume capability.
    """

    id: str = Field(default_factory=lambda: f"deepres-{uuid4().hex[:12]}")
    original_query: str = Field(..., description="The original research query")
    research_brief: Optional[str] = Field(
        default=None,
        description="Expanded research plan generated in PLANNING phase",
    )
    phase: DeepResearchPhase = Field(
        default=DeepResearchPhase.PLANNING,
        description="Current workflow phase",
    )
    iteration: int = Field(
        default=1,
        description="Current refinement iteration (1-based)",
    )
    max_iterations: int = Field(
        default=3,
        description="Maximum refinement iterations before forced completion",
    )

    # Collections
    sub_queries: list[SubQuery] = Field(default_factory=list)
    sources: list[ResearchSource] = Field(default_factory=list)
    findings: list[ResearchFinding] = Field(default_factory=list)
    gaps: list[ResearchGap] = Field(default_factory=list)

    # Final output
    report: Optional[str] = Field(
        default=None,
        description="Final synthesized research report",
    )
    report_sections: dict[str, str] = Field(
        default_factory=dict,
        description="Named sections of the report for structured access",
    )

    # Execution tracking
    total_sources_examined: int = Field(default=0)
    total_tokens_used: int = Field(default=0)
    total_duration_ms: float = Field(default=0.0)

    # Per-phase metrics for audit
    phase_metrics: list[PhaseMetrics] = Field(
        default_factory=list,
        description="Metrics for each executed phase (timing, tokens, provider)",
    )
    # Search provider query counts (provider_name -> query_count)
    search_provider_stats: dict[str, int] = Field(
        default_factory=dict,
        description="Count of queries executed per search provider",
    )

    # Polling tracking
    status_check_count: int = Field(
        default=0,
        description="Number of status checks made",
    )
    last_status_check_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last status check",
    )

    # Heartbeat tracking for progress visibility
    last_heartbeat_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last heartbeat (updated before provider calls)",
    )

    # Content fidelity tracking (for token budget management)
    # Per-item fidelity records: content_fidelity[item_id].phases[phase] = {level, reason, warnings, timestamp}
    content_fidelity: dict[str, ContentFidelityRecord] = Field(
        default_factory=dict,
        description="Per-item fidelity records tracking degradation across phases",
    )
    dropped_content_ids: list[str] = Field(
        default_factory=list,
        description="IDs of sources dropped during budget allocation",
    )
    content_allocation_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Aggregate metadata: total_tokens_used, overall_fidelity_score, phase_budgets, warnings",
    )

    # Configuration
    source_types: list[SourceType] = Field(
        default_factory=lambda: [SourceType.WEB, SourceType.ACADEMIC],
    )
    max_sources_per_query: int = Field(default=5)
    max_sub_queries: int = Field(default=5)
    follow_links: bool = Field(
        default=True,
        description="Whether to follow URLs and extract full content",
    )
    research_mode: ResearchMode = Field(
        default=ResearchMode.GENERAL,
        description="Research mode for source prioritization",
    )

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(default=None)

    # Provider tracking (per-phase LLM provider configuration)
    # Supports ProviderSpec format: "[cli]gemini:pro" or simple names: "gemini"
    planning_provider: Optional[str] = Field(default=None)
    analysis_provider: Optional[str] = Field(default=None)
    synthesis_provider: Optional[str] = Field(default=None)
    refinement_provider: Optional[str] = Field(default=None)
    # Per-phase model overrides (from ProviderSpec parsing)
    planning_model: Optional[str] = Field(default=None)
    analysis_model: Optional[str] = Field(default=None)
    synthesis_model: Optional[str] = Field(default=None)
    refinement_model: Optional[str] = Field(default=None)

    system_prompt: Optional[str] = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # =========================================================================
    # Collection Management Methods
    # =========================================================================

    def add_sub_query(
        self,
        query: str,
        rationale: Optional[str] = None,
        priority: int = 1,
    ) -> SubQuery:
        """Add a new sub-query for research.

        Args:
            query: The focused sub-query text
            rationale: Why this sub-query was generated
            priority: Execution priority (1=highest)

        Returns:
            The created SubQuery instance
        """
        sub_query = SubQuery(query=query, rationale=rationale, priority=priority)
        self.sub_queries.append(sub_query)
        self.updated_at = datetime.utcnow()
        return sub_query

    def get_sub_query(self, sub_query_id: str) -> Optional[SubQuery]:
        """Get a sub-query by ID."""
        for sq in self.sub_queries:
            if sq.id == sub_query_id:
                return sq
        return None

    def get_source(self, source_id: str) -> Optional[ResearchSource]:
        """Get a source by ID."""
        for source in self.sources:
            if source.id == source_id:
                return source
        return None

    def get_gap(self, gap_id: str) -> Optional[ResearchGap]:
        """Get a gap by ID."""
        for gap in self.gaps:
            if gap.id == gap_id:
                return gap
        return None

    def add_source(
        self,
        title: str,
        url: Optional[str] = None,
        source_type: SourceType = SourceType.WEB,
        snippet: Optional[str] = None,
        sub_query_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ResearchSource:
        """Add a discovered source.

        Args:
            title: Source title
            url: Source URL (optional)
            source_type: Type of source
            snippet: Brief excerpt
            sub_query_id: ID of sub-query that found this
            **kwargs: Additional fields

        Returns:
            The created ResearchSource instance
        """
        source = ResearchSource(
            title=title,
            url=url,
            source_type=source_type,
            snippet=snippet,
            sub_query_id=sub_query_id,
            **kwargs,
        )
        self.sources.append(source)
        self.total_sources_examined += 1
        self.updated_at = datetime.utcnow()
        return source

    def add_finding(
        self,
        content: str,
        confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
        source_ids: Optional[list[str]] = None,
        sub_query_id: Optional[str] = None,
        category: Optional[str] = None,
    ) -> ResearchFinding:
        """Add a research finding.

        Args:
            content: The finding content
            confidence: Confidence level
            source_ids: Supporting source IDs
            sub_query_id: Originating sub-query ID
            category: Theme/category

        Returns:
            The created ResearchFinding instance
        """
        finding = ResearchFinding(
            content=content,
            confidence=confidence,
            source_ids=source_ids or [],
            sub_query_id=sub_query_id,
            category=category,
        )
        self.findings.append(finding)
        self.updated_at = datetime.utcnow()
        return finding

    def add_gap(
        self,
        description: str,
        suggested_queries: Optional[list[str]] = None,
        priority: int = 1,
    ) -> ResearchGap:
        """Add an identified research gap.

        Args:
            description: What information is missing
            suggested_queries: Follow-up queries to fill the gap
            priority: Priority for follow-up (1=highest)

        Returns:
            The created ResearchGap instance
        """
        gap = ResearchGap(
            description=description,
            suggested_queries=suggested_queries or [],
            priority=priority,
        )
        self.gaps.append(gap)
        self.updated_at = datetime.utcnow()
        return gap

    # =========================================================================
    # Query Helpers
    # =========================================================================

    def pending_sub_queries(self) -> list[SubQuery]:
        """Get sub-queries that haven't been executed yet."""
        return [sq for sq in self.sub_queries if sq.status == "pending"]

    def completed_sub_queries(self) -> list[SubQuery]:
        """Get successfully completed sub-queries."""
        return [sq for sq in self.sub_queries if sq.status == "completed"]

    def failed_sub_queries(self) -> list[SubQuery]:
        """Get sub-queries that failed during execution."""
        return [sq for sq in self.sub_queries if sq.status == "failed"]

    def unresolved_gaps(self) -> list[ResearchGap]:
        """Get gaps that haven't been resolved yet."""
        return [g for g in self.gaps if not g.resolved]

    # =========================================================================
    # Phase Management
    # =========================================================================

    def advance_phase(self) -> DeepResearchPhase:
        """Advance to the next research phase.

        Phases advance in order: PLANNING -> GATHERING -> ANALYSIS ->
        SYNTHESIS -> REFINEMENT. Does nothing if already at REFINEMENT.

        Returns:
            The new phase after advancement
        """
        phase_order = list(DeepResearchPhase)
        current_index = phase_order.index(self.phase)
        if current_index < len(phase_order) - 1:
            self.phase = phase_order[current_index + 1]
        self.updated_at = datetime.utcnow()
        return self.phase

    def should_continue_refinement(self) -> bool:
        """Check if another refinement iteration should occur.

        Returns True if:
        - Current iteration < max_iterations AND
        - There are unresolved gaps

        Returns:
            True if refinement should continue, False otherwise
        """
        if self.iteration >= self.max_iterations:
            return False
        if not self.unresolved_gaps():
            return False
        return True

    def start_new_iteration(self) -> int:
        """Start a new refinement iteration.

        Increments iteration counter and resets phase to GATHERING
        to begin collecting sources for the new sub-queries.

        Returns:
            The new iteration number
        """
        self.iteration += 1
        self.phase = DeepResearchPhase.GATHERING
        self.updated_at = datetime.utcnow()
        return self.iteration

    def mark_completed(self, report: Optional[str] = None) -> None:
        """Mark the research session as completed.

        Args:
            report: Optional final report content
        """
        self.phase = DeepResearchPhase.SYNTHESIS
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        if report:
            self.report = report

    def mark_failed(self, error: str) -> None:
        """Mark the research session as failed with an error message.

        This sets completed_at to indicate the session has ended, and stores
        the failure information in metadata for status reporting.

        Args:
            error: Description of why the research failed
        """
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.metadata["failed"] = True
        self.metadata["failure_error"] = error

    # ==========================================================================
    # Content Fidelity Tracking Methods
    # ==========================================================================

    def record_item_fidelity(
        self,
        item_id: str,
        phase: str,
        level: FidelityLevel,
        item_type: str = "source",
        reason: str = "",
        warnings: Optional[list[str]] = None,
        original_tokens: Optional[int] = None,
        final_tokens: Optional[int] = None,
    ) -> ContentFidelityRecord:
        """Record fidelity for a content item in a specific phase.

        Creates or updates the ContentFidelityRecord for the item and
        adds the phase-specific record.

        Args:
            item_id: Unique identifier for the content item
            phase: Phase name (e.g., "analysis", "synthesis")
            level: Fidelity level applied
            item_type: Type of content ("source", "finding", "gap")
            reason: Why degradation was applied
            warnings: Any warnings generated
            original_tokens: Token count before degradation
            final_tokens: Token count after degradation

        Returns:
            The ContentFidelityRecord for the item
        """
        # Create or get existing record
        if item_id not in self.content_fidelity:
            self.content_fidelity[item_id] = ContentFidelityRecord(
                item_id=item_id,
                item_type=item_type,
            )

        record = self.content_fidelity[item_id]
        record.record_phase(
            phase=phase,
            level=level,
            reason=reason,
            warnings=warnings,
            original_tokens=original_tokens,
            final_tokens=final_tokens,
        )

        # Track dropped items
        if level == FidelityLevel.DROPPED and item_id not in self.dropped_content_ids:
            self.dropped_content_ids.append(item_id)

        self.updated_at = datetime.utcnow()
        return record

    def get_item_fidelity(self, item_id: str) -> Optional[ContentFidelityRecord]:
        """Get fidelity record for a content item.

        Args:
            item_id: ID of the content item

        Returns:
            ContentFidelityRecord if exists, None otherwise
        """
        return self.content_fidelity.get(item_id)

    def get_items_at_fidelity(self, level: FidelityLevel) -> list[str]:
        """Get all item IDs currently at a specific fidelity level.

        Args:
            level: Fidelity level to filter by

        Returns:
            List of item IDs at that fidelity level
        """
        return [
            item_id
            for item_id, record in self.content_fidelity.items()
            if record.current_level == level
        ]

    def get_overall_fidelity_score(self) -> float:
        """Calculate an overall fidelity score for the session.

        Returns a value between 0.0 and 1.0 representing the average
        content preservation across all tracked items.

        Returns:
            Overall fidelity score (1.0 = all full fidelity, 0.0 = all dropped)
        """
        if not self.content_fidelity:
            return 1.0

        level_scores = {
            FidelityLevel.FULL: 1.0,
            FidelityLevel.CONDENSED: 0.7,
            FidelityLevel.KEY_POINTS: 0.4,
            FidelityLevel.HEADLINE: 0.2,
            FidelityLevel.TRUNCATED: 0.3,
            FidelityLevel.DROPPED: 0.0,
        }

        total_score = sum(
            level_scores.get(record.current_level, 0.5)
            for record in self.content_fidelity.values()
        )
        return total_score / len(self.content_fidelity)

    def has_degraded_content(self) -> bool:
        """Check if any content has been degraded from full fidelity.

        Returns:
            True if any content is below FULL fidelity
        """
        return any(
            record.current_level != FidelityLevel.FULL
            for record in self.content_fidelity.values()
        )

    def record_chunk_fidelity(
        self,
        base_id: str,
        chunk_index: int,
        phase: str,
        level: FidelityLevel,
        item_type: str = "source",
        reason: str = "",
        warnings: Optional[list[str]] = None,
        original_tokens: Optional[int] = None,
        final_tokens: Optional[int] = None,
    ) -> ContentFidelityRecord:
        """Record fidelity for a specific chunk of a content item.

        Creates a fidelity record with a stable fragment ID in the format
        "{base_id}#fragment-{N}". This allows tracking fidelity at the
        chunk level while maintaining the parent item relationship.

        Args:
            base_id: Base item ID (e.g., "src-abc123")
            chunk_index: Zero-based index of the chunk
            phase: Phase name (e.g., "analysis", "synthesis")
            level: Fidelity level applied
            item_type: Type of content ("source", "finding", "gap")
            reason: Why degradation was applied
            warnings: Any warnings generated
            original_tokens: Token count before degradation
            final_tokens: Token count after degradation

        Returns:
            The ContentFidelityRecord for the chunk
        """
        fragment_id = make_fragment_id(base_id, chunk_index)
        return self.record_item_fidelity(
            item_id=fragment_id,
            phase=phase,
            level=level,
            item_type=item_type,
            reason=reason,
            warnings=warnings,
            original_tokens=original_tokens,
            final_tokens=final_tokens,
        )

    def get_chunk_fidelity(
        self, base_id: str, chunk_index: int
    ) -> Optional[ContentFidelityRecord]:
        """Get fidelity record for a specific chunk.

        Args:
            base_id: Base item ID (e.g., "src-abc123")
            chunk_index: Zero-based index of the chunk

        Returns:
            ContentFidelityRecord if exists, None otherwise
        """
        fragment_id = make_fragment_id(base_id, chunk_index)
        return self.get_item_fidelity(fragment_id)

    def get_all_chunks_for_item(self, base_id: str) -> dict[int, ContentFidelityRecord]:
        """Get all chunk fidelity records for a base item.

        Finds all fragment IDs that derive from the given base ID and
        returns their fidelity records indexed by chunk number.

        Args:
            base_id: Base item ID (e.g., "src-abc123")

        Returns:
            Dict mapping chunk_index to ContentFidelityRecord
        """
        chunks = {}
        prefix = f"{base_id}#fragment-"
        for item_id, record in self.content_fidelity.items():
            if item_id.startswith(prefix):
                _, fragment_index = parse_fragment_id(item_id)
                if fragment_index is not None:
                    chunks[fragment_index] = record
        return chunks

    def merge_fidelity_record(
        self, item_id: str, other_record: ContentFidelityRecord
    ) -> ContentFidelityRecord:
        """Merge another fidelity record into the state.

        Implements the fidelity merge rules:
        - Latest phase overwrites same-phase entry (by timestamp)
        - Prior phases are preserved for history

        If the item doesn't exist in state, adds it directly.
        If the item exists, merges phases from the other record.

        Args:
            item_id: ID of the content item
            other_record: ContentFidelityRecord to merge

        Returns:
            The merged ContentFidelityRecord
        """
        if item_id not in self.content_fidelity:
            # New item - add directly
            self.content_fidelity[item_id] = other_record
        else:
            # Existing item - merge phases
            self.content_fidelity[item_id].merge_phases_from(other_record)

        # Track dropped items
        record = self.content_fidelity[item_id]
        if (
            record.current_level == FidelityLevel.DROPPED
            and item_id not in self.dropped_content_ids
        ):
            self.dropped_content_ids.append(item_id)

        self.updated_at = datetime.utcnow()
        return record

    def get_aggregate_chunk_fidelity(self, base_id: str) -> Optional[FidelityLevel]:
        """Get the aggregate fidelity level across all chunks of an item.

        Returns the lowest (most degraded) fidelity level among all
        chunks. This represents the "worst case" fidelity for the item.

        Args:
            base_id: Base item ID

        Returns:
            Lowest FidelityLevel among chunks, or None if no chunks exist
        """
        chunks = self.get_all_chunks_for_item(base_id)
        if not chunks:
            return None

        # Order: FULL > CONDENSED > KEY_POINTS > HEADLINE > TRUNCATED > DROPPED
        level_order = [
            FidelityLevel.FULL,
            FidelityLevel.CONDENSED,
            FidelityLevel.KEY_POINTS,
            FidelityLevel.HEADLINE,
            FidelityLevel.TRUNCATED,
            FidelityLevel.DROPPED,
        ]

        worst_level = FidelityLevel.FULL
        for record in chunks.values():
            if level_order.index(record.current_level) > level_order.index(worst_level):
                worst_level = record.current_level

        return worst_level
