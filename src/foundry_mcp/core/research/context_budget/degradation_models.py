"""Data models for the degradation pipeline.

Provides enums, step records, chunk tracking, and result containers
for the graceful content degradation system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .models import AllocatedItem


class DegradationLevel(str, Enum):
    """Levels of content degradation in the fallback chain.

    The degradation pipeline attempts levels in order:
    FULL -> KEY_POINTS -> HEADLINE -> TRUNCATE -> DROP

    Each level represents progressively more aggressive compression:
        FULL: No degradation, content at original fidelity
        KEY_POINTS: Summarize to key points (~30% of original)
        HEADLINE: Extreme summarization to headline (~10% of original)
        TRUNCATE: Hard truncation with warning marker (always enabled)
        DROP: Remove item entirely (only if allow_content_dropping=True)
    """

    FULL = "full"
    KEY_POINTS = "key_points"
    HEADLINE = "headline"
    TRUNCATE = "truncate"
    DROP = "drop"

    def next_level(self) -> Optional["DegradationLevel"]:
        """Get the next degradation level in the chain.

        Returns:
            Next tighter level, or None if at DROP
        """
        order = [
            DegradationLevel.FULL,
            DegradationLevel.KEY_POINTS,
            DegradationLevel.HEADLINE,
            DegradationLevel.TRUNCATE,
            DegradationLevel.DROP,
        ]
        try:
            idx = order.index(self)
            if idx < len(order) - 1:
                return order[idx + 1]
        except ValueError:
            pass
        return None


@dataclass
class DegradationStep:
    """Record of a degradation action taken on an item.

    Attributes:
        item_id: ID of the item that was degraded
        from_level: Level before degradation
        to_level: Level after degradation
        original_tokens: Token count before degradation
        result_tokens: Token count after degradation
        success: Whether degradation achieved target budget
        warning: Warning message if any issues occurred
        chunk_id: Optional chunk identifier for chunk-level tracking
    """

    item_id: str
    from_level: DegradationLevel
    to_level: DegradationLevel
    original_tokens: int
    result_tokens: int
    success: bool = True
    warning: Optional[str] = None
    chunk_id: Optional[str] = None


@dataclass
class ChunkFailure:
    """Record of a chunk-level failure during degradation.

    Attributes:
        item_id: ID of the parent item containing the chunk
        chunk_id: Identifier of the failed chunk (e.g., "chunk-0", "chunk-1")
        original_level: Degradation level at which failure occurred
        retry_level: Level used for retry attempt, if any
        error: Error message from the failure
        recovered: Whether the chunk was successfully recovered after retry
    """

    item_id: str
    chunk_id: str
    original_level: DegradationLevel
    retry_level: Optional[DegradationLevel] = None
    error: Optional[str] = None
    recovered: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "item_id": self.item_id,
            "chunk_id": self.chunk_id,
            "original_level": self.original_level.value,
            "retry_level": self.retry_level.value if self.retry_level else None,
            "error": self.error,
            "recovered": self.recovered,
        }


@dataclass
class ChunkResult:
    """Result of processing a single chunk during degradation.

    Attributes:
        item_id: ID of the parent item containing the chunk
        chunk_id: Identifier of the chunk (e.g., "chunk-0", "chunk-1")
        content: The processed chunk content (may be degraded/summarized)
        tokens: Token count of the processed content
        level: Degradation level at which content was produced
        success: Whether chunk processing succeeded
        retried: Whether the chunk was retried at a tighter level
        failures: List of failures encountered during processing
    """

    item_id: str
    chunk_id: str
    content: str
    tokens: int
    level: DegradationLevel
    success: bool = True
    retried: bool = False
    failures: list[ChunkFailure] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "item_id": self.item_id,
            "chunk_id": self.chunk_id,
            "tokens": self.tokens,
            "level": self.level.value,
            "success": self.success,
            "retried": self.retried,
            "failures": [f.to_dict() for f in self.failures],
        }


@dataclass
class DegradationResult:
    """Result of running the degradation pipeline.

    Attributes:
        items: List of allocated items after degradation
        tokens_used: Total tokens after degradation
        fidelity: Overall content fidelity (0.0-1.0)
        steps: List of degradation steps taken
        dropped_ids: IDs of items that were dropped
        warnings: List of warnings generated
        min_items_enforced: Whether min items guardrail was active
        chunk_failures: List of chunk-level failures encountered during processing
    """

    items: list[AllocatedItem] = field(default_factory=list)
    tokens_used: int = 0
    fidelity: float = 1.0
    steps: list[DegradationStep] = field(default_factory=list)
    dropped_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    min_items_enforced: bool = False
    chunk_failures: list[ChunkFailure] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "items": [
                {
                    "id": item.id,
                    "priority": item.priority,
                    "original_tokens": item.original_tokens,
                    "allocated_tokens": item.allocated_tokens,
                    "needs_summarization": item.needs_summarization,
                    "allocation_ratio": item.allocation_ratio,
                }
                for item in self.items
            ],
            "tokens_used": self.tokens_used,
            "fidelity": self.fidelity,
            "steps": [
                {
                    "item_id": step.item_id,
                    "from_level": step.from_level.value,
                    "to_level": step.to_level.value,
                    "original_tokens": step.original_tokens,
                    "result_tokens": step.result_tokens,
                    "success": step.success,
                    "warning": step.warning,
                    "chunk_id": step.chunk_id,
                }
                for step in self.steps
            ],
            "dropped_ids": self.dropped_ids,
            "warnings": self.warnings,
            "min_items_enforced": self.min_items_enforced,
            "chunk_failures": [cf.to_dict() for cf in self.chunk_failures],
        }
