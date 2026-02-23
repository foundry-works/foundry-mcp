"""Fidelity tracking models for token budget management."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class FidelityLevel(str, Enum):
    """Content fidelity levels for token budget management.

    Defines how much content has been preserved or compressed during
    budget allocation. Each level represents a progressively more
    aggressive compression applied to fit within token constraints.

    Levels (ordered from highest to lowest fidelity):
        FULL: Content unchanged - original content preserved
        CONDENSED: Light summarization (~50-70% of original)
        KEY_POINTS: Bullet point extraction (~20-40% of original)
        DIGEST: Structured digest with evidence snippets (~15-30% of original)
        HEADLINE: Single sentence summary (~5-10% of original)
        TRUNCATED: Hard cut with marker (arbitrary %)
        DROPPED: Content completely removed (0%)
    """

    FULL = "full"
    CONDENSED = "condensed"
    KEY_POINTS = "key_points"
    DIGEST = "digest"
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
            history.append(
                {
                    "phase": phase_name,
                    "level": record.level.value,
                    "reason": record.reason,
                    "timestamp": record.timestamp.isoformat(),
                    "original_tokens": record.original_tokens,
                    "final_tokens": record.final_tokens,
                }
            )
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
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible metadata (e.g. token_limit_retries, model_roles)",
    )
