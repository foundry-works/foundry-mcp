"""Deep research workflow models (multi-phase iterative research)."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from foundry_mcp.core.research.models.digest import make_fragment_id, parse_fragment_id
from foundry_mcp.core.research.models.enums import ConfidenceLevel
from foundry_mcp.core.research.models.fidelity import (
    ContentFidelityRecord,
    FidelityLevel,
    PhaseMetrics,
)
from foundry_mcp.core.research.models.sources import (
    ResearchFinding,
    ResearchGap,
    ResearchMode,
    ResearchSource,
    SourceType,
    SubQuery,
)


class TopicResearchResult(BaseModel):
    """Result of a per-topic ReAct research loop.

    Each sub-query can be investigated independently by a topic researcher
    that runs its own search → reflect → refine cycle. This model captures
    the outcome of that per-topic investigation.
    """

    sub_query_id: str = Field(..., description="ID of the SubQuery this result belongs to")
    searches_performed: int = Field(default=0, description="Number of search iterations executed")
    sources_found: int = Field(default=0, description="Total unique sources discovered for this topic")
    per_topic_summary: Optional[str] = Field(
        default=None,
        description="LLM-generated summary of findings for this specific topic",
    )
    reflection_notes: list[str] = Field(
        default_factory=list,
        description="Notes from per-topic reflection steps (e.g., identified gaps, query refinements)",
    )
    refined_queries: list[str] = Field(
        default_factory=list,
        description="Refined queries generated during the ReAct loop",
    )
    source_ids: list[str] = Field(
        default_factory=list,
        description="IDs of sources discovered by this topic researcher",
    )


class Contradiction(BaseModel):
    """A contradiction detected between research findings.

    Identified during the analysis phase when multiple sources provide
    conflicting information on the same topic. Contradictions are surfaced
    in the synthesis prompt so the final report can address them explicitly.
    """

    id: str = Field(default_factory=lambda: f"contra-{uuid4().hex[:8]}")
    finding_ids: list[str] = Field(
        ...,
        description="IDs of the conflicting ResearchFinding objects",
    )
    description: str = Field(
        ...,
        description="Description of the conflict between findings",
    )
    resolution: Optional[str] = Field(
        default=None,
        description="Suggested resolution or explanation for the contradiction",
    )
    preferred_source_id: Optional[str] = Field(
        default=None,
        description="ID of the more authoritative source, if determinable",
    )
    severity: Literal["major", "minor"] = Field(
        default="minor",
        description="Severity of the contradiction: 'major' or 'minor'",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


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


class DeepResearchPhase(str, Enum):
    """Phases of the DEEP_RESEARCH workflow.

    The deep research workflow progresses through six sequential phases:
    0. CLARIFICATION - (Optional) Analyze query specificity and ask clarifying questions
    1. PLANNING - Analyze the query and decompose into focused sub-queries
    2. GATHERING - Execute sub-queries in parallel and collect sources
    3. ANALYSIS - Extract findings and assess source quality
    4. SYNTHESIS - Combine findings into a comprehensive report
    5. REFINEMENT - Identify gaps and potentially loop back for more research

    The ordering of these enum values is significant - it defines the
    progression through advance_phase() method.
    """

    CLARIFICATION = "clarification"
    PLANNING = "planning"
    GATHERING = "gathering"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    REFINEMENT = "refinement"


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
    clarification_constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="Constraints and context inferred or provided during CLARIFICATION phase",
    )
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
    contradictions: list[Contradiction] = Field(
        default_factory=list,
        description="Contradictions detected between findings during analysis",
    )
    topic_research_results: list[TopicResearchResult] = Field(
        default_factory=list,
        description="Per-topic research results from parallel topic researcher agents",
    )

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
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
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
        self.updated_at = datetime.now(timezone.utc)
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

    def get_citation_map(self) -> dict[int, ResearchSource]:
        """Build a mapping from citation number to source.

        Returns:
            Dict mapping citation_number → ResearchSource for all sources
            that have an assigned citation number.
        """
        return {s.citation_number: s for s in self.sources if s.citation_number is not None}

    def source_id_to_citation(self) -> dict[str, int]:
        """Build a mapping from source ID to citation number.

        Returns:
            Dict mapping source.id → citation_number for all sources
            that have an assigned citation number.
        """
        return {s.id: s.citation_number for s in self.sources if s.citation_number is not None}

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
        # Assign the next citation number based on the highest existing number.
        # This is the SINGLE source of truth for citation numbering — callers
        # must NOT assign citation_number manually.
        next_citation = max((s.citation_number or 0 for s in self.sources), default=0) + 1
        source = ResearchSource(
            title=title,
            url=url,
            source_type=source_type,
            snippet=snippet,
            sub_query_id=sub_query_id,
            citation_number=next_citation,
            **kwargs,
        )
        self.sources.append(source)
        self.total_sources_examined += 1
        self.updated_at = datetime.now(timezone.utc)
        return source

    def append_source(self, source: ResearchSource) -> ResearchSource:
        """Append a pre-constructed source, assigning it the next citation number.

        Use this when the source is already constructed (e.g., from a search
        provider) but needs a stable citation number and state tracking.

        Args:
            source: Pre-constructed ResearchSource (citation_number will be overwritten)

        Returns:
            The same source instance, with citation_number set
        """
        next_citation = max((s.citation_number or 0 for s in self.sources), default=0) + 1
        source.citation_number = next_citation
        self.sources.append(source)
        self.total_sources_examined += 1
        self.updated_at = datetime.now(timezone.utc)
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
        self.updated_at = datetime.now(timezone.utc)
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
        self.updated_at = datetime.now(timezone.utc)
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

        Phases advance in order: CLARIFICATION -> PLANNING -> GATHERING ->
        ANALYSIS -> SYNTHESIS -> REFINEMENT. Does nothing if already at
        REFINEMENT. The phase order is derived from the DeepResearchPhase
        enum definition order.

        Returns:
            The new phase after advancement
        """
        phase_order = list(DeepResearchPhase)
        current_index = phase_order.index(self.phase)
        if current_index < len(phase_order) - 1:
            self.phase = phase_order[current_index + 1]
        self.updated_at = datetime.now(timezone.utc)
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

        Note: We intentionally skip CLARIFICATION and PLANNING here.
        Clarification is a one-time pre-planning step (query refinement
        is not needed once research is underway), and planning has
        already decomposed the query into sub-queries. Refinement
        iterations only need to re-gather, re-analyze, and re-synthesize.

        Returns:
            The new iteration number
        """
        self.iteration += 1
        self.phase = DeepResearchPhase.GATHERING
        self.updated_at = datetime.now(timezone.utc)
        return self.iteration

    def mark_completed(self, report: Optional[str] = None) -> None:
        """Mark the research session as completed.

        Args:
            report: Optional final report content
        """
        self.phase = DeepResearchPhase.SYNTHESIS
        self.completed_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        if report:
            self.report = report

    def mark_failed(self, error: str) -> None:
        """Mark the research session as failed with an error message.

        This sets completed_at to indicate the session has ended, and stores
        the failure information in metadata for status reporting.

        Args:
            error: Description of why the research failed
        """
        self.completed_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        self.metadata["failed"] = True
        self.metadata["failure_error"] = error

    def mark_cancelled(self, *, phase_state: Optional[str] = None) -> None:
        """Mark the research session as cancelled by user request.

        Distinct from mark_failed (error) and mark_interrupted (SIGTERM).
        Sets completed_at and stores cancellation context in metadata.

        Args:
            phase_state: Optional description of phase state at cancellation time
        """
        self.completed_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        self.metadata["cancelled"] = True
        self.metadata["terminal_status"] = "cancelled"
        if phase_state:
            self.metadata["cancelled_phase_state"] = phase_state

    def mark_interrupted(self, *, reason: str = "SIGTERM") -> None:
        """Mark the research session as interrupted by process signal.

        Distinct from mark_cancelled (user-initiated) and mark_failed (error).
        Used for SIGTERM and other process-level interruptions.

        Args:
            reason: Reason for interruption (default: "SIGTERM")
        """
        self.completed_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        self.metadata["interrupted"] = True
        self.metadata["terminal_status"] = "interrupted"
        self.metadata["interrupt_reason"] = reason
        self.metadata["interrupt_phase"] = self.phase.value
        self.metadata["interrupt_iteration"] = self.iteration

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

        self.updated_at = datetime.now(timezone.utc)
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
        return [item_id for item_id, record in self.content_fidelity.items() if record.current_level == level]

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

        total_score = sum(level_scores.get(record.current_level, 0.5) for record in self.content_fidelity.values())
        return total_score / len(self.content_fidelity)

    def has_degraded_content(self) -> bool:
        """Check if any content has been degraded from full fidelity.

        Returns:
            True if any content is below FULL fidelity
        """
        return any(record.current_level != FidelityLevel.FULL for record in self.content_fidelity.values())

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

    def get_chunk_fidelity(self, base_id: str, chunk_index: int) -> Optional[ContentFidelityRecord]:
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

    def merge_fidelity_record(self, item_id: str, other_record: ContentFidelityRecord) -> ContentFidelityRecord:
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
        if record.current_level == FidelityLevel.DROPPED and item_id not in self.dropped_content_ids:
            self.dropped_content_ids.append(item_id)

        self.updated_at = datetime.now(timezone.utc)
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
