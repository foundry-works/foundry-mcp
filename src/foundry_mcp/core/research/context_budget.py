"""Context budget management for deep research workflows.

Provides priority-based token budget allocation for managing content items
within token-constrained environments. The ContextBudgetManager orchestrates
allocation decisions based on item priority and available budget.

Key Components:
    - compute_priority: Score items by quality, confidence, recency, relevance
    - compute_recency_score: Convert content age to a 0-1 recency score
    - AllocationStrategy: Enum defining budget distribution strategies
    - ContentItem: Dataclass for content with priority, source_id, protected flag
    - ContentItemProtocol: Protocol for custom allocatable content items
    - AllocationResult: Dataclass with allocation outcome and metadata
    - ContextBudgetManager: Main orchestrator for budget allocation
    - DegradationLevel: Enum defining fallback compression levels
    - DegradationPipeline: Centralized fallback chain for graceful degradation
    - ChunkFailure: Record of a chunk-level failure during degradation
    - ChunkResult: Result of processing a single chunk with retry history

Usage:
    from foundry_mcp.core.research.context_budget import (
        ContextBudgetManager,
        AllocationStrategy,
        AllocationResult,
        ContentItem,
        compute_priority,
    )
    from foundry_mcp.core.research.models.enums import ConfidenceLevel
    from foundry_mcp.core.research.models.sources import SourceQuality

    # Create content items (protected items won't be dropped)
    item1 = ContentItem(id="finding-1", content="...", priority=1)
    citation = ContentItem(id="cite-1", content="...", priority=1, protected=True)

    # Compute priority for a content item
    priority = compute_priority(
        source_quality=SourceQuality.HIGH,
        confidence=ConfidenceLevel.CONFIRMED,
        recency_score=0.9,
        relevance_score=0.95,
    )

    # Create manager
    manager = ContextBudgetManager()

    # Allocate budget across items
    result = manager.allocate_budget(
        items=[item1, citation],
        budget=50_000,
        strategy=AllocationStrategy.PRIORITY_FIRST,
    )

    # Check results
    print(f"Allocated {len(result.items)} items using {result.tokens_used} tokens")
    print(f"Fidelity: {result.fidelity:.1%}")
    if result.dropped_ids:
        print(f"Dropped: {result.dropped_ids}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Protocol, Sequence, runtime_checkable

from foundry_mcp.core.research.token_management import estimate_tokens
from foundry_mcp.core.research.models.enums import ConfidenceLevel
from foundry_mcp.core.research.models.sources import ResearchSource, SourceQuality

logger = logging.getLogger(__name__)


# =============================================================================
# Degradation Constants
# =============================================================================

# Minimum items to preserve per phase (guardrail)
MIN_ITEMS_PER_PHASE = 3

# Number of top priority items to preserve at minimum condensed fidelity
TOP_PRIORITY_ITEMS = 5

# Minimum fidelity ratio for condensed level (30% of original)
CONDENSED_MIN_FIDELITY = 0.30

# Minimum fidelity ratio for headline level (10% of original)
HEADLINE_MIN_FIDELITY = 0.10

# Truncation marker for content that has been truncated
TRUNCATION_MARKER = " [... truncated]"

# Characters per token estimate for truncation calculations
CHARS_PER_TOKEN = 4

# Maximum entries in per-source token cache (FIFO eviction)
MAX_TOKEN_CACHE_ENTRIES = 50


# =============================================================================
# Priority Scoring
# =============================================================================

# Weight factors for priority scoring (must sum to 1.0)
PRIORITY_WEIGHT_SOURCE_QUALITY = 0.40
PRIORITY_WEIGHT_CONFIDENCE = 0.30
PRIORITY_WEIGHT_RECENCY = 0.15
PRIORITY_WEIGHT_RELEVANCE = 0.15

# Source quality score mapping
SOURCE_QUALITY_SCORES: dict[SourceQuality, float] = {
    SourceQuality.HIGH: 1.0,
    SourceQuality.MEDIUM: 0.7,
    SourceQuality.LOW: 0.4,
    SourceQuality.UNKNOWN: 0.5,
}

# Confidence level score mapping
CONFIDENCE_SCORES: dict[ConfidenceLevel, float] = {
    ConfidenceLevel.CONFIRMED: 1.0,
    ConfidenceLevel.HIGH: 0.9,
    ConfidenceLevel.MEDIUM: 0.7,
    ConfidenceLevel.LOW: 0.4,
    ConfidenceLevel.SPECULATION: 0.2,
}


def compute_priority(
    *,
    source_quality: Optional[SourceQuality] = None,
    confidence: Optional[ConfidenceLevel] = None,
    recency_score: float = 0.5,
    relevance_score: float = 0.5,
) -> float:
    """Compute a priority score for content prioritization.

    Calculates a weighted priority score based on multiple factors:
    - Source quality (40%): Reliability and authority of the source
    - Confidence (30%): Certainty level of findings/claims
    - Recency (15%): How recent the content is
    - Relevance (15%): How relevant to the research query

    The resulting score is used to prioritize content when allocating
    limited token budget. Higher scores = higher priority.

    Args:
        source_quality: Quality assessment of the source (HIGH/MEDIUM/LOW/UNKNOWN).
            If None, defaults to UNKNOWN (0.5 score).
        confidence: Confidence level for findings (CONFIRMED/HIGH/MEDIUM/LOW/SPECULATION).
            If None, defaults to MEDIUM (0.7 score).
        recency_score: Score from 0.0 to 1.0 indicating content freshness.
            1.0 = very recent, 0.0 = very old. Default 0.5.
        relevance_score: Score from 0.0 to 1.0 indicating query relevance.
            1.0 = highly relevant, 0.0 = not relevant. Default 0.5.

    Returns:
        Priority score between 0.0 and 1.0, where higher = higher priority.

    Raises:
        ValueError: If recency_score or relevance_score is outside [0.0, 1.0]

    Example:
        # High-quality, confirmed finding from recent relevant source
        score = compute_priority(
            source_quality=SourceQuality.HIGH,
            confidence=ConfidenceLevel.CONFIRMED,
            recency_score=0.9,
            relevance_score=0.95,
        )
        # Returns ~0.97

        # Low-quality speculation from old, marginally relevant source
        score = compute_priority(
            source_quality=SourceQuality.LOW,
            confidence=ConfidenceLevel.SPECULATION,
            recency_score=0.1,
            relevance_score=0.3,
        )
        # Returns ~0.28
    """
    # Validate input scores
    if not 0.0 <= recency_score <= 1.0:
        raise ValueError(f"recency_score must be in [0.0, 1.0], got {recency_score}")
    if not 0.0 <= relevance_score <= 1.0:
        raise ValueError(f"relevance_score must be in [0.0, 1.0], got {relevance_score}")

    # Get scores with defaults
    quality_score = SOURCE_QUALITY_SCORES.get(
        source_quality or SourceQuality.UNKNOWN, 0.5
    )
    confidence_score = CONFIDENCE_SCORES.get(
        confidence or ConfidenceLevel.MEDIUM, 0.7
    )

    # Compute weighted sum
    priority = (
        PRIORITY_WEIGHT_SOURCE_QUALITY * quality_score
        + PRIORITY_WEIGHT_CONFIDENCE * confidence_score
        + PRIORITY_WEIGHT_RECENCY * recency_score
        + PRIORITY_WEIGHT_RELEVANCE * relevance_score
    )

    # Clamp to valid range (should be 0-1 by construction, but be safe)
    return max(0.0, min(1.0, priority))


def compute_recency_score(
    age_hours: float,
    max_age_hours: float = 720.0,  # 30 days default
) -> float:
    """Compute a recency score based on content age.

    Uses linear decay from 1.0 (brand new) to 0.0 (at or beyond max age).

    Args:
        age_hours: Age of the content in hours
        max_age_hours: Age at which score becomes 0.0 (default 720 = 30 days)

    Returns:
        Recency score from 0.0 to 1.0

    Example:
        # Content from 1 hour ago
        score = compute_recency_score(1.0)  # ~0.999

        # Content from 15 days ago
        score = compute_recency_score(360.0)  # ~0.5

        # Content from 60 days ago
        score = compute_recency_score(1440.0)  # 0.0
    """
    if age_hours < 0:
        raise ValueError(f"age_hours must be non-negative, got {age_hours}")
    if max_age_hours <= 0:
        raise ValueError(f"max_age_hours must be positive, got {max_age_hours}")

    if age_hours >= max_age_hours:
        return 0.0

    return 1.0 - (age_hours / max_age_hours)


# =============================================================================
# Allocation Strategies
# =============================================================================


class AllocationStrategy(str, Enum):
    """Strategies for distributing token budget across content items.

    Strategies:
        PRIORITY_FIRST: Allocate to highest-priority items first until budget
            exhausted. Lower-priority items may be dropped entirely.
        EQUAL_SHARE: Distribute budget equally across all items. Each item
            gets budget / num_items tokens (may require summarization).
        PROPORTIONAL: Distribute budget proportional to each item's original
            size. Larger items get larger allocations.

    Example:
        # For research findings with varying importance
        strategy = AllocationStrategy.PRIORITY_FIRST

        # For balanced representation across sources
        strategy = AllocationStrategy.EQUAL_SHARE
    """

    PRIORITY_FIRST = "priority_first"
    EQUAL_SHARE = "equal_share"
    PROPORTIONAL = "proportional"


@runtime_checkable
class ContentItemProtocol(Protocol):
    """Protocol for content items that can be allocated budget.

    Any object implementing these attributes can be used with
    ContextBudgetManager. This allows flexibility in what types
    of content can be managed.

    Required Attributes:
        id: Unique identifier for the item
        content: Text content to be included
        priority: Priority level (1 = highest, higher numbers = lower priority)

    Optional Attributes:
        tokens: Pre-computed token count (if None, will be estimated)
        protected: If True, item must not be dropped during allocation

    Example:
        @dataclass
        class ResearchFinding:
            id: str
            content: str
            priority: int = 1
            tokens: Optional[int] = None
            protected: bool = False
    """

    id: str
    content: str
    priority: int


@dataclass
class ContentItem:
    """Concrete content item for budget allocation.

    Represents a piece of content with metadata for priority-based
    budget allocation. Use this class directly or implement the
    ContentItemProtocol for custom content types.

    Attributes:
        id: Stable unique identifier for fidelity tracking
        content: Text content to be included in the context
        priority: Priority level (1 = highest, higher numbers = lower priority)
        source_id: Optional identifier of the source (e.g., ResearchSource.id)
        source_ref: Optional ResearchSource object for token cache reuse
        token_count: Pre-computed token count (if None, will be estimated)
        protected: If True, item must not be dropped during allocation.
            Use for critical content like citations or key findings.

    Example:
        # Create a regular content item
        item = ContentItem(
            id="finding-123",
            content="AI models show improved performance...",
            priority=1,
            source_id="source-456",
        )

        # Create a protected citation that must be included
        citation = ContentItem(
            id="citation-789",
            content="[1] Smith et al., 2024...",
            priority=1,
            protected=True,
        )
    """

    id: str
    content: str
    priority: int = 1
    source_id: Optional[str] = None
    source_ref: Optional[ResearchSource] = None
    token_count: Optional[int] = None
    protected: bool = False

    @property
    def tokens(self) -> Optional[int]:
        """Alias for token_count for protocol compatibility."""
        return self.token_count


@dataclass
class AllocatedItem:
    """An item with its allocation details.

    Represents a content item after budget allocation, including
    whether it was allocated at full fidelity or needs compression.

    Attributes:
        id: Identifier of the original item
        content: Content text (may be original or summarized)
        priority: Original priority level
        original_tokens: Token count before allocation
        allocated_tokens: Tokens actually allocated to this item
        needs_summarization: Whether item exceeds allocation and needs compression
        allocation_ratio: Ratio of allocated to original tokens (1.0 = full fidelity)
    """

    id: str
    content: str
    priority: int
    original_tokens: int
    allocated_tokens: int
    needs_summarization: bool = False
    allocation_ratio: float = 1.0

    def __post_init__(self) -> None:
        """Calculate allocation ratio if not provided."""
        if self.original_tokens > 0:
            self.allocation_ratio = self.allocated_tokens / self.original_tokens
        else:
            self.allocation_ratio = 1.0


@dataclass
class AllocationResult:
    """Result of a budget allocation operation.

    Contains the allocated items along with aggregate metrics about
    the allocation process for monitoring and debugging.

    Attributes:
        items: List of allocated items with their budget assignments
        tokens_used: Total tokens allocated across all items
        tokens_available: Total budget that was available
        fidelity: Overall fidelity score (1.0 = all items at full fidelity)
        warnings: List of warnings generated during allocation
        dropped_ids: IDs of items that couldn't fit in the budget

    Example:
        result = manager.allocate_budget(items, budget=50_000)
        if result.fidelity < 0.8:
            print("Warning: Significant content compression occurred")
        for item_id in result.dropped_ids:
            print(f"Dropped item: {item_id}")
    """

    items: list[AllocatedItem] = field(default_factory=list)
    tokens_used: int = 0
    tokens_available: int = 0
    fidelity: float = 1.0
    warnings: list[str] = field(default_factory=list)
    dropped_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate result consistency."""
        if self.tokens_used < 0:
            raise ValueError(f"tokens_used must be non-negative, got {self.tokens_used}")
        if self.tokens_available < 0:
            raise ValueError(f"tokens_available must be non-negative, got {self.tokens_available}")
        if not 0.0 <= self.fidelity <= 1.0:
            raise ValueError(f"fidelity must be in [0.0, 1.0], got {self.fidelity}")

    @property
    def utilization(self) -> float:
        """Calculate what fraction of available budget was used.

        Returns:
            Fraction of budget utilized (0.0 to 1.0)
        """
        if self.tokens_available <= 0:
            return 0.0
        return min(1.0, self.tokens_used / self.tokens_available)

    @property
    def items_allocated(self) -> int:
        """Count of items that received allocation."""
        return len(self.items)

    @property
    def items_dropped(self) -> int:
        """Count of items that were dropped."""
        return len(self.dropped_ids)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dict representation of the result
        """
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
            "tokens_available": self.tokens_available,
            "fidelity": self.fidelity,
            "utilization": self.utilization,
            "warnings": self.warnings,
            "dropped_ids": self.dropped_ids,
            "items_allocated": self.items_allocated,
            "items_dropped": self.items_dropped,
        }


# =============================================================================
# Degradation Pipeline
# =============================================================================


class DegradationLevel(str, Enum):
    """Levels of content degradation in the fallback chain.

    The degradation pipeline attempts levels in order:
    FULL → KEY_POINTS → HEADLINE → TRUNCATE → DROP

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


# Error class (canonical definition in foundry_mcp.core.errors.research)
from foundry_mcp.core.errors.research import ProtectedContentOverflowError  # noqa: E402


class DegradationPipeline:
    """Centralized fallback chain for graceful content degradation.

    Implements the degradation chain:
    FULL → KEY_POINTS → HEADLINE → TRUNCATE → DROP

    The pipeline progressively degrades content to fit within budget:
    1. Start with full content
    2. If over budget, summarize to KEY_POINTS (~30%)
    3. If still over, summarize to HEADLINE (~10%)
    4. If still over, TRUNCATE with warning (always enabled)
    5. If still over and allow_content_dropping=True, DROP lowest priority

    Guardrails:
    - Protected items are never dropped
    - Top-5 priority items never go below condensed fidelity (~30%)
    - Min 3 items per phase preserved when possible
    - Truncation fallback is always enabled (hardcoded)

    Warning Codes:
    - PRIORITY_SUMMARIZED: A top-priority item was degraded (summarized/truncated)
    - CONTENT_DROPPED: A low-priority item was dropped
    - CONTENT_TRUNCATED: Content was truncated
    - PROTECTED_OVERFLOW: Protected item force-allocated with minimal budget
    - TOKEN_BUDGET_FLOORED: Item preserved due to min items guardrail

    Example:
        pipeline = DegradationPipeline(
            allow_content_dropping=True,
            min_items=3,
            priority_items=5,
        )
        result = pipeline.degrade(
            items=sources,
            budget=50_000,
        )
        if result.warnings:
            print(f"Degradation warnings: {result.warnings}")
    """

    def __init__(
        self,
        *,
        token_estimator: Optional[Callable[[str], int]] = None,
        allow_content_dropping: bool = False,
        min_items: int = MIN_ITEMS_PER_PHASE,
        priority_items: int = TOP_PRIORITY_ITEMS,
    ):
        """Initialize the degradation pipeline.

        Args:
            token_estimator: Custom function to estimate tokens.
                If not provided, uses heuristic (len/4).
            allow_content_dropping: If True, allows dropping lowest-priority
                items when other degradation levels fail. Default False.
            min_items: Minimum items to preserve per phase (guardrail).
                Default is MIN_ITEMS_PER_PHASE (3).
            priority_items: Number of top-priority items to preserve at
                minimum condensed fidelity. Default is TOP_PRIORITY_ITEMS (5).
        """
        self._token_estimator = token_estimator
        self._allow_content_dropping = allow_content_dropping
        self._min_items = min_items
        self._priority_items = priority_items

    def _estimate_tokens(self, content: str) -> int:
        """Estimate tokens for content."""
        if self._token_estimator:
            return self._token_estimator(content)
        return max(1, len(content) // CHARS_PER_TOKEN)

    def _truncate_content(self, content: str, target_tokens: int) -> str:
        """Truncate content to fit target token budget.

        Args:
            content: Original content
            target_tokens: Target token count

        Returns:
            Truncated content with marker
        """
        if target_tokens <= 0:
            return TRUNCATION_MARKER.strip()

        # Reserve space for truncation marker
        marker_tokens = len(TRUNCATION_MARKER) // CHARS_PER_TOKEN + 1
        content_tokens = max(1, target_tokens - marker_tokens)
        content_chars = content_tokens * CHARS_PER_TOKEN

        if len(content) <= content_chars:
            return content

        return content[:content_chars].rstrip() + TRUNCATION_MARKER

    def _is_priority_item(self, item_index: int) -> bool:
        """Check if an item is in the top priority set.

        Args:
            item_index: Zero-based index in priority-sorted list

        Returns:
            True if item is in top priority_items (default 5)
        """
        return item_index < self._priority_items

    def _get_min_priority_allocation(self, original_tokens: int) -> int:
        """Get minimum token allocation for priority items.

        Priority items must maintain at least condensed fidelity (30%).

        Args:
            original_tokens: Original token count

        Returns:
            Minimum tokens to allocate (at least 30% of original)
        """
        return max(1, int(original_tokens * CONDENSED_MIN_FIDELITY))

    def _get_headline_allocation(self, original_tokens: int) -> int:
        """Get headline-level token allocation for protected items.

        Headline is the most aggressive compression (~10% of original).
        Used as last resort for protected content overflow.

        Args:
            original_tokens: Original token count

        Returns:
            Minimum tokens for headline level (at least 10% of original)
        """
        return max(1, int(original_tokens * HEADLINE_MIN_FIDELITY))

    def _check_protected_content_budget(
        self,
        protected_items: Sequence[ContentItem],
        budget: int,
    ) -> tuple[bool, int, list[str]]:
        """Check if protected content fits within budget at headline level.

        Args:
            protected_items: List of protected content items
            budget: Available token budget

        Returns:
            Tuple of (fits, total_headline_tokens, item_ids)
        """
        total_headline_tokens = 0
        item_ids = []

        for item in protected_items:
            item_tokens = self._estimate_tokens(item.content)
            headline_tokens = self._get_headline_allocation(item_tokens)
            total_headline_tokens += headline_tokens
            item_ids.append(item.id)

        return (total_headline_tokens <= budget, total_headline_tokens, item_ids)

    def _emit_chunk_warning(
        self,
        item_id: str,
        chunk_id: str,
        message: str,
        *,
        level: Optional[DegradationLevel] = None,
        tokens: Optional[int] = None,
    ) -> str:
        """Generate a standardized chunk-level warning message.

        Creates warning messages that include both item_id and chunk_id
        for precise identification of chunk-level issues.

        Args:
            item_id: ID of the parent item
            chunk_id: ID of the specific chunk (e.g., "chunk-0")
            message: Warning message type/description
            level: Optional degradation level for context
            tokens: Optional token count for context

        Returns:
            Formatted warning string
        """
        parts = [f"CHUNK_FAILURE: {message}"]
        parts.append(f"item_id={item_id}")
        parts.append(f"chunk_id={chunk_id}")

        if level is not None:
            parts.append(f"level={level.value}")
        if tokens is not None:
            parts.append(f"tokens={tokens}")

        return " | ".join(parts)

    def _retry_chunk_at_tighter_level(
        self,
        content: str,
        item_id: str,
        chunk_id: str,
        current_level: DegradationLevel,
        target_tokens: int,
    ) -> ChunkResult:
        """Retry a failed chunk at a more aggressive summarization level.

        Attempts to process a chunk that failed at the current level by
        using a tighter degradation level. Progresses through levels until
        success or reaching TRUNCATE as a last resort.

        Args:
            content: Chunk content to process
            item_id: ID of the parent item
            chunk_id: ID of the chunk (e.g., "chunk-0")
            current_level: Level at which the chunk failed
            target_tokens: Target token count for the output

        Returns:
            ChunkResult with processed content and failure history
        """
        failures: list[ChunkFailure] = []
        level = current_level

        while True:
            next_level = level.next_level()

            if next_level is None or next_level == DegradationLevel.DROP:
                # Reached end of chain - use truncation as last resort
                truncated_content = self._truncate_content(content, target_tokens)
                truncated_tokens = self._estimate_tokens(truncated_content)

                return ChunkResult(
                    item_id=item_id,
                    chunk_id=chunk_id,
                    content=truncated_content,
                    tokens=truncated_tokens,
                    level=DegradationLevel.TRUNCATE,
                    success=True,
                    retried=True,
                    failures=failures,
                )

            level = next_level

            # Try the next level
            # For sync pipeline, we use truncation at progressively tighter ratios
            if level == DegradationLevel.KEY_POINTS:
                allocation = self._get_min_priority_allocation(len(content) // CHARS_PER_TOKEN)
            elif level == DegradationLevel.HEADLINE:
                allocation = self._get_headline_allocation(len(content) // CHARS_PER_TOKEN)
            else:
                allocation = target_tokens

            try:
                truncated_content = self._truncate_content(content, allocation)
                truncated_tokens = self._estimate_tokens(truncated_content)

                if truncated_tokens <= target_tokens:
                    return ChunkResult(
                        item_id=item_id,
                        chunk_id=chunk_id,
                        content=truncated_content,
                        tokens=truncated_tokens,
                        level=level,
                        success=True,
                        retried=True,
                        failures=failures,
                    )

                # Still too large, record failure and continue
                failures.append(ChunkFailure(
                    item_id=item_id,
                    chunk_id=chunk_id,
                    original_level=current_level,
                    retry_level=level,
                    error=f"Still exceeds target: {truncated_tokens} > {target_tokens}",
                    recovered=False,
                ))

            except Exception as e:
                # Record the failure and continue to next level
                failures.append(ChunkFailure(
                    item_id=item_id,
                    chunk_id=chunk_id,
                    original_level=current_level,
                    retry_level=level,
                    error=str(e),
                    recovered=False,
                ))

    def _process_chunk_with_retry(
        self,
        content: str,
        item_id: str,
        chunk_id: str,
        target_tokens: int,
        initial_level: DegradationLevel = DegradationLevel.FULL,
    ) -> ChunkResult:
        """Process a single chunk with automatic retry on failure.

        Attempts to process a chunk at the initial level. If processing
        fails or the result exceeds the target, retries at progressively
        tighter levels until success.

        Successful chunk summaries are preserved; only failed chunks are
        retried. This enables partial results when some chunks succeed.

        Args:
            content: Chunk content to process
            item_id: ID of the parent item
            chunk_id: ID of the chunk (e.g., "chunk-0")
            target_tokens: Target token count for the output
            initial_level: Starting degradation level

        Returns:
            ChunkResult with processed content and any failures
        """
        chunk_tokens = self._estimate_tokens(content)

        # If content already fits, return as-is
        if chunk_tokens <= target_tokens:
            return ChunkResult(
                item_id=item_id,
                chunk_id=chunk_id,
                content=content,
                tokens=chunk_tokens,
                level=initial_level,
                success=True,
                retried=False,
                failures=[],
            )

        # Content doesn't fit - try truncation at current level first
        try:
            truncated_content = self._truncate_content(content, target_tokens)
            truncated_tokens = self._estimate_tokens(truncated_content)

            if truncated_tokens <= target_tokens:
                return ChunkResult(
                    item_id=item_id,
                    chunk_id=chunk_id,
                    content=truncated_content,
                    tokens=truncated_tokens,
                    level=initial_level,
                    success=True,
                    retried=False,
                    failures=[],
                )
        except Exception as e:
            # Initial truncation failed - record and retry at tighter level
            logger.warning(
                f"Chunk truncation failed for {item_id}/{chunk_id}: {e}"
            )

        # Retry at tighter levels
        return self._retry_chunk_at_tighter_level(
            content=content,
            item_id=item_id,
            chunk_id=chunk_id,
            current_level=initial_level,
            target_tokens=target_tokens,
        )

    def process_chunked_item(
        self,
        item_id: str,
        chunks: list[str],
        target_tokens_per_chunk: int,
        initial_level: DegradationLevel = DegradationLevel.FULL,
    ) -> tuple[list[ChunkResult], list[str]]:
        """Process multiple chunks for a single item with failure handling.

        Processes each chunk with automatic retry on failure. Preserves
        successful chunk summaries and retries failed chunks at tighter
        levels. Returns warnings with item_id and chunk_id for each issue.

        Args:
            item_id: ID of the parent item
            chunks: List of chunk content strings
            target_tokens_per_chunk: Target tokens per chunk
            initial_level: Starting degradation level for all chunks

        Returns:
            Tuple of (chunk_results, warnings) where:
            - chunk_results: List of ChunkResult for each chunk
            - warnings: List of warning messages with item_id and chunk_id
        """
        results: list[ChunkResult] = []
        warnings: list[str] = []

        for i, chunk_content in enumerate(chunks):
            chunk_id = f"chunk-{i}"

            result = self._process_chunk_with_retry(
                content=chunk_content,
                item_id=item_id,
                chunk_id=chunk_id,
                target_tokens=target_tokens_per_chunk,
                initial_level=initial_level,
            )

            results.append(result)

            # Generate warnings for any failures
            if result.failures:
                for failure in result.failures:
                    warning = self._emit_chunk_warning(
                        item_id=failure.item_id,
                        chunk_id=failure.chunk_id,
                        message=f"Retry at {failure.retry_level.value if failure.retry_level else 'unknown'}: {failure.error}",
                        level=failure.original_level,
                    )
                    warnings.append(warning)

            # Warn if chunk was retried at tighter level
            if result.retried:
                warning = self._emit_chunk_warning(
                    item_id=item_id,
                    chunk_id=chunk_id,
                    message=f"Recovered at {result.level.value}",
                    level=result.level,
                    tokens=result.tokens,
                )
                warnings.append(warning)

        return results, warnings

    def degrade(
        self,
        items: Sequence[ContentItem],
        budget: int,
    ) -> DegradationResult:
        """Run the degradation pipeline on items to fit budget.

        Attempts progressive degradation to fit content within budget:
        1. Allocate items at full fidelity (priority order)
        2. For items that don't fit, try KEY_POINTS summarization
        3. If still over, try HEADLINE summarization
        4. If still over, TRUNCATE (always enabled)
        5. If still over and allow_content_dropping=True, DROP

        Protected content handling:
        - Protected items are never dropped
        - If budget is tight, protected items get headline allocation (~10%)
        - If protected content exceeds budget even at headline level,
          raises ProtectedContentOverflowError with remediation guidance

        Args:
            items: Content items to degrade (must have id, content, priority)
            budget: Total token budget available

        Returns:
            DegradationResult with degraded items and metadata

        Raises:
            ValueError: If budget is not positive
            ProtectedContentOverflowError: If protected content exceeds budget
                even at headline level
        """
        if not items:
            return DegradationResult(fidelity=1.0)

        if budget <= 0:
            raise ValueError(f"budget must be positive, got {budget}")

        # Pre-check: Verify protected content fits at headline level
        protected_items_list = [i for i in items if i.protected]
        if protected_items_list:
            fits, headline_tokens, protected_ids = self._check_protected_content_budget(
                protected_items_list, budget
            )
            if not fits:
                raise ProtectedContentOverflowError(
                    protected_tokens=headline_tokens,
                    budget=budget,
                    item_ids=protected_ids,
                )

        # Sort by priority (1 = highest, first)
        sorted_items = sorted(items, key=lambda x: x.priority)

        # Track state
        allocated: list[AllocatedItem] = []
        steps: list[DegradationStep] = []
        dropped_ids: list[str] = []
        warnings: list[str] = []
        remaining_budget = budget
        total_original_tokens = 0
        min_items_enforced = False

        # Count protected and non-protected items
        protected_items = [i for i in sorted_items if i.protected]
        droppable_items = [i for i in sorted_items if not i.protected]

        for item_index, item in enumerate(sorted_items):
            is_priority = self._is_priority_item(item_index)
            item_tokens = self._estimate_tokens(item.content)
            total_original_tokens += item_tokens

            # Check if item fits at full fidelity
            if item_tokens <= remaining_budget:
                # Full fidelity allocation
                allocated.append(AllocatedItem(
                    id=item.id,
                    content=item.content,
                    priority=item.priority,
                    original_tokens=item_tokens,
                    allocated_tokens=item_tokens,
                    needs_summarization=False,
                ))
                remaining_budget -= item_tokens
                continue

            # Item doesn't fit at full fidelity - use truncation fallback
            # Note: KEY_POINTS and HEADLINE summarization require async operations
            # and would be handled by ContentSummarizer. The sync pipeline uses
            # truncation as the fallback (always enabled per spec).

            if remaining_budget > 0:
                # For priority items, enforce minimum condensed fidelity (30%)
                if is_priority:
                    min_allocation = self._get_min_priority_allocation(item_tokens)
                    target_tokens = max(remaining_budget, min_allocation)
                else:
                    target_tokens = remaining_budget

                # Truncate to fit target budget
                truncated_content = self._truncate_content(item.content, target_tokens)
                truncated_tokens = self._estimate_tokens(truncated_content)

                # Determine the degradation level
                allocation_ratio = truncated_tokens / item_tokens if item_tokens > 0 else 1.0
                if allocation_ratio >= CONDENSED_MIN_FIDELITY:
                    to_level = DegradationLevel.KEY_POINTS
                else:
                    to_level = DegradationLevel.TRUNCATE

                steps.append(DegradationStep(
                    item_id=item.id,
                    from_level=DegradationLevel.FULL,
                    to_level=to_level,
                    original_tokens=item_tokens,
                    result_tokens=truncated_tokens,
                    success=True,
                    warning=f"Content degraded from {item_tokens} to {truncated_tokens} tokens",
                ))

                allocated.append(AllocatedItem(
                    id=item.id,
                    content=truncated_content,
                    priority=item.priority,
                    original_tokens=item_tokens,
                    allocated_tokens=truncated_tokens,
                    needs_summarization=True,  # Mark as degraded
                ))
                remaining_budget -= truncated_tokens

                # Emit appropriate warning based on priority status
                if is_priority:
                    warnings.append(
                        f"PRIORITY_SUMMARIZED: Priority item {item.id} degraded from "
                        f"{item_tokens} to {truncated_tokens} tokens "
                        f"(fidelity={allocation_ratio:.1%}, min={CONDENSED_MIN_FIDELITY:.0%})"
                    )
                else:
                    warnings.append(
                        f"CONTENT_TRUNCATED: Item {item.id} truncated from "
                        f"{item_tokens} to {truncated_tokens} tokens"
                    )
                continue

            # No budget remaining - consider dropping
            # Protected items and priority items are never dropped
            if item.protected:
                # Protected items get headline allocation (~10%) as last resort
                # (pre-check guarantees this fits within budget)
                headline_allocation = self._get_headline_allocation(item_tokens)
                headline_content = self._truncate_content(item.content, headline_allocation)
                headline_tokens = self._estimate_tokens(headline_content)

                steps.append(DegradationStep(
                    item_id=item.id,
                    from_level=DegradationLevel.FULL,
                    to_level=DegradationLevel.HEADLINE,
                    original_tokens=item_tokens,
                    result_tokens=headline_tokens,
                    success=True,
                    warning="Protected item compressed to headline level",
                ))
                allocated.append(AllocatedItem(
                    id=item.id,
                    content=headline_content,
                    priority=item.priority,
                    original_tokens=item_tokens,
                    allocated_tokens=headline_tokens,
                    needs_summarization=True,
                ))
                warnings.append(
                    f"PROTECTED_OVERFLOW: Protected item {item.id} compressed to headline "
                    f"({headline_tokens}/{item_tokens} tokens, "
                    f"fidelity={headline_tokens/item_tokens:.1%})"
                )
                continue

            # Priority items (top-5) must maintain at least condensed fidelity
            if is_priority:
                min_allocation = self._get_min_priority_allocation(item_tokens)
                minimal_content = self._truncate_content(item.content, min_allocation)
                minimal_tokens = self._estimate_tokens(minimal_content)
                steps.append(DegradationStep(
                    item_id=item.id,
                    from_level=DegradationLevel.FULL,
                    to_level=DegradationLevel.KEY_POINTS,
                    original_tokens=item_tokens,
                    result_tokens=minimal_tokens,
                    success=False,
                    warning="Priority item force-allocated at condensed fidelity",
                ))
                allocated.append(AllocatedItem(
                    id=item.id,
                    content=minimal_content,
                    priority=item.priority,
                    original_tokens=item_tokens,
                    allocated_tokens=minimal_tokens,
                    needs_summarization=True,
                ))
                warnings.append(
                    f"PRIORITY_SUMMARIZED: Priority item {item.id} force-allocated "
                    f"at condensed fidelity ({minimal_tokens}/{item_tokens} tokens)"
                )
                continue

            # Check if we can drop this low-priority item
            if self._allow_content_dropping:
                # Check min items guardrail
                current_allocated_count = len(allocated) + len(protected_items) - len([
                    a for a in allocated if any(p.id == a.id for p in protected_items)
                ])
                # Count remaining items that could still be allocated
                remaining_droppable = len([
                    d for d in droppable_items
                    if d.id not in dropped_ids and d.id != item.id
                ])
                potential_total = current_allocated_count + remaining_droppable

                if potential_total >= self._min_items:
                    # Safe to drop
                    steps.append(DegradationStep(
                        item_id=item.id,
                        from_level=DegradationLevel.TRUNCATE,
                        to_level=DegradationLevel.DROP,
                        original_tokens=item_tokens,
                        result_tokens=0,
                        success=True,
                    ))
                    dropped_ids.append(item.id)
                    warnings.append(
                        f"CONTENT_DROPPED: Item {item.id} dropped "
                        f"(priority={item.priority}, tokens={item_tokens})"
                    )
                else:
                    # Would violate min items - force allocate with truncation
                    min_items_enforced = True
                    minimal_content = self._truncate_content(item.content, 1)
                    steps.append(DegradationStep(
                        item_id=item.id,
                        from_level=DegradationLevel.DROP,
                        to_level=DegradationLevel.TRUNCATE,
                        original_tokens=item_tokens,
                        result_tokens=1,
                        success=False,
                        warning=f"Min items guardrail ({self._min_items}) prevented drop",
                    ))
                    allocated.append(AllocatedItem(
                        id=item.id,
                        content=minimal_content,
                        priority=item.priority,
                        original_tokens=item_tokens,
                        allocated_tokens=1,
                        needs_summarization=True,
                    ))
                    warnings.append(
                        f"TOKEN_BUDGET_FLOORED: Item {item.id} preserved due to "
                        f"min items guardrail ({self._min_items} items)"
                    )
            else:
                # Dropping not allowed - force allocate with minimal truncation
                minimal_content = self._truncate_content(item.content, 1)
                steps.append(DegradationStep(
                    item_id=item.id,
                    from_level=DegradationLevel.TRUNCATE,
                    to_level=DegradationLevel.TRUNCATE,
                    original_tokens=item_tokens,
                    result_tokens=1,
                    success=False,
                    warning="Content dropping disabled, forced minimal allocation",
                ))
                allocated.append(AllocatedItem(
                    id=item.id,
                    content=minimal_content,
                    priority=item.priority,
                    original_tokens=item_tokens,
                    allocated_tokens=1,
                    needs_summarization=True,
                ))
                warnings.append(
                    f"CONTENT_TRUNCATED: Item {item.id} force-allocated with "
                    f"minimal budget (content_dropping=False)"
                )

        # Calculate fidelity
        total_allocated = sum(item.allocated_tokens for item in allocated)
        fidelity = total_allocated / total_original_tokens if total_original_tokens > 0 else 1.0

        return DegradationResult(
            items=allocated,
            tokens_used=total_allocated,
            fidelity=max(0.0, min(1.0, fidelity)),
            steps=steps,
            dropped_ids=dropped_ids,
            warnings=warnings,
            min_items_enforced=min_items_enforced,
        )


class ContextBudgetManager:
    """Orchestrates priority-based token budget allocation.

    Manages the distribution of a token budget across multiple content
    items based on priority and allocation strategy. Tracks which items
    fit at full fidelity, which need compression, and which must be dropped.

    The manager does not perform actual summarization - it determines
    allocation targets. Use ContentSummarizer to compress items that
    have needs_summarization=True in the result.

    Attributes:
        token_estimator: Function to estimate tokens for content
        provider: Provider hint for token estimation accuracy

    Example:
        manager = ContextBudgetManager(provider="claude")

        # Prepare items (any objects implementing ContentItem protocol)
        items = [
            {"id": "src-1", "content": "...", "priority": 1},
            {"id": "src-2", "content": "...", "priority": 2},
        ]

        # Allocate budget
        result = manager.allocate_budget(
            items=items,
            budget=50_000,
            strategy=AllocationStrategy.PRIORITY_FIRST,
        )

        # Process results
        for item in result.items:
            if item.needs_summarization:
                # Summarize to fit allocated_tokens
                summarized = await summarizer.summarize(
                    item.content,
                    target_tokens=item.allocated_tokens,
                )
    """

    def __init__(
        self,
        *,
        token_estimator: Optional[Callable[[str], int]] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize the context budget manager.

        Args:
            token_estimator: Custom function to estimate token counts.
                If not provided, uses estimate_tokens from token_management.
            provider: Provider hint for more accurate token estimation
            model: Model hint for more accurate token estimation
        """
        self._token_estimator = token_estimator
        self._provider = provider
        self._model = model

    def _estimate_tokens(self, content: str) -> int:
        """Estimate tokens for content using configured estimator.

        Args:
            content: Text content to estimate

        Returns:
            Estimated token count
        """
        if self._token_estimator:
            return self._token_estimator(content)
        return estimate_tokens(
            content,
            provider=self._provider,
            model=self._model,
            warn_on_heuristic=False,  # Suppress repeated warnings in batch
        )

    def _get_item_tokens(self, item: Any) -> int:
        """Get or estimate token count for an item.

        For ResearchSource items, checks the internal token cache first.
        On cache miss, computes the token count and stores it with FIFO
        eviction when the cache exceeds MAX_TOKEN_CACHE_ENTRIES (50).

        Args:
            item: Content item (must have 'content' attribute)

        Returns:
            Token count (from cache, item.tokens if present, else estimated)
        """
        # Check for pre-computed tokens
        if hasattr(item, "tokens") and item.tokens is not None:
            return item.tokens

        # For ResearchSource (direct or attached to content item), check cached token count
        source_ref: Optional[ResearchSource] = None
        if isinstance(item, ResearchSource):
            source_ref = item
        else:
            candidate = getattr(item, "source_ref", None)
            if isinstance(candidate, ResearchSource):
                source_ref = candidate

        if source_ref is not None and self._provider and self._model:
            cached = source_ref._get_cached_token_count(self._provider, self._model)
            if cached is not None:
                logger.debug(
                    f"Token cache hit for {source_ref.id}: {cached} tokens "
                    f"(provider={self._provider}, model={self._model})"
                )
                return cached

        # Estimate from content
        content = getattr(item, "content", "")
        tokens = self._estimate_tokens(content)

        # For ResearchSource, store in cache with FIFO eviction
        if source_ref is not None and self._provider and self._model:
            self._store_token_count_with_eviction(source_ref, tokens)
            logger.debug(
                f"Token cache miss for {source_ref.id}: computed {tokens} tokens "
                f"(provider={self._provider}, model={self._model})"
            )

        return tokens

    def _store_token_count_with_eviction(
        self, source: ResearchSource, count: int
    ) -> None:
        """Store token count in source cache with FIFO eviction.

        If the cache exceeds MAX_TOKEN_CACHE_ENTRIES, removes the oldest
        entry before adding the new one. Dict key insertion order is
        preserved in Python 3.7+, so we remove the first key for FIFO.

        Args:
            source: ResearchSource to update
            count: Token count to store
        """
        if not self._provider or not self._model:
            return

        # Ensure cache exists
        if "_token_cache" not in source.metadata:
            source.metadata["_token_cache"] = {"v": 1, "counts": {}}

        cache = source.metadata["_token_cache"]
        if "counts" not in cache:
            cache["counts"] = {}

        counts = cache["counts"]

        # FIFO eviction if at capacity
        while len(counts) >= MAX_TOKEN_CACHE_ENTRIES:
            # Remove oldest entry (first key in insertion order)
            oldest_key = next(iter(counts))
            del counts[oldest_key]
            logger.debug(f"Token cache eviction: removed {oldest_key}")

        # Store new count
        source._set_cached_token_count(self._provider, self._model, count)

    def _sort_by_priority(self, items: Sequence[Any]) -> list[Any]:
        """Sort items by priority (1 = highest, first).

        Args:
            items: Sequence of content items

        Returns:
            List sorted by priority ascending (highest priority first)
        """
        return sorted(items, key=lambda x: getattr(x, "priority", 999))

    def allocate_budget(
        self,
        items: Sequence[Any],
        budget: int,
        strategy: AllocationStrategy = AllocationStrategy.PRIORITY_FIRST,
    ) -> AllocationResult:
        """Allocate token budget across content items.

        Distributes the available budget across items based on the specified
        strategy. Higher-priority items (priority=1) are favored when budget
        is limited.

        Args:
            items: Sequence of content items implementing ContentItem protocol.
                Each must have id, content, and priority attributes.
            budget: Total token budget available for allocation
            strategy: Strategy for distributing budget across items

        Returns:
            AllocationResult with allocated items, metrics, and dropped IDs

        Raises:
            ValueError: If budget is not positive

        Example:
            result = manager.allocate_budget(
                items=sources,
                budget=100_000,
                strategy=AllocationStrategy.PRIORITY_FIRST,
            )
        """
        if budget <= 0:
            raise ValueError(f"budget must be positive, got {budget}")

        if not items:
            return AllocationResult(
                items=[],
                tokens_used=0,
                tokens_available=budget,
                fidelity=1.0,
                warnings=[],
                dropped_ids=[],
            )

        # Sort items by priority
        sorted_items = self._sort_by_priority(items)

        # Estimate tokens for all items
        item_tokens: list[tuple[Any, int]] = []
        total_original_tokens = 0
        for item in sorted_items:
            tokens = self._get_item_tokens(item)
            item_tokens.append((item, tokens))
            total_original_tokens += tokens

        # Dispatch to strategy-specific allocation
        if strategy == AllocationStrategy.PRIORITY_FIRST:
            return self._allocate_priority_first(item_tokens, budget, total_original_tokens)
        elif strategy == AllocationStrategy.EQUAL_SHARE:
            return self._allocate_equal_share(item_tokens, budget, total_original_tokens)
        else:  # strategy == AllocationStrategy.PROPORTIONAL
            return self._allocate_proportional(item_tokens, budget, total_original_tokens)

    def _allocate_priority_first(
        self,
        item_tokens: list[tuple[Any, int]],
        budget: int,
        total_original_tokens: int,
    ) -> AllocationResult:
        """Allocate budget to highest-priority items first.

        Items are allocated in priority order. Each item gets its full
        token requirement if budget allows, otherwise it's either allocated
        remaining budget (needs_summarization=True) or dropped.

        Args:
            item_tokens: List of (item, token_count) tuples, sorted by priority
            budget: Total budget available
            total_original_tokens: Sum of all original token counts

        Returns:
            AllocationResult with allocation details
        """
        allocated_items: list[AllocatedItem] = []
        dropped_ids: list[str] = []
        warnings: list[str] = []
        remaining_budget = budget
        total_allocated_tokens = 0

        for item, tokens in item_tokens:
            item_id = getattr(item, "id", str(id(item)))
            item_priority = getattr(item, "priority", 999)
            item_content = getattr(item, "content", "")
            item_protected = getattr(item, "protected", False)

            if remaining_budget <= 0:
                if item_protected:
                    # Protected items must be allocated even without budget
                    # They will need aggressive summarization
                    allocated_items.append(
                        AllocatedItem(
                            id=item_id,
                            content=item_content,
                            priority=item_priority,
                            original_tokens=tokens,
                            allocated_tokens=1,  # Minimum allocation
                            needs_summarization=True,
                        )
                    )
                    total_allocated_tokens += 1
                    warnings.append(
                        f"Protected item {item_id} force-allocated with minimal budget: "
                        f"{tokens} tokens -> 1 allocated (needs aggressive summarization)"
                    )
                else:
                    # No budget left - drop non-protected items
                    dropped_ids.append(item_id)
                    warnings.append(
                        f"Dropped item {item_id} (priority={item_priority}): no budget remaining"
                    )
                continue

            if tokens <= remaining_budget:
                # Full allocation
                allocated_items.append(
                    AllocatedItem(
                        id=item_id,
                        content=item_content,
                        priority=item_priority,
                        original_tokens=tokens,
                        allocated_tokens=tokens,
                        needs_summarization=False,
                    )
                )
                remaining_budget -= tokens
                total_allocated_tokens += tokens
            else:
                # Partial allocation - needs summarization
                allocated_tokens = remaining_budget
                allocated_items.append(
                    AllocatedItem(
                        id=item_id,
                        content=item_content,
                        priority=item_priority,
                        original_tokens=tokens,
                        allocated_tokens=allocated_tokens,
                        needs_summarization=True,
                    )
                )
                remaining_budget = 0
                total_allocated_tokens += allocated_tokens
                warnings.append(
                    f"Item {item_id} needs summarization: "
                    f"{tokens} tokens -> {allocated_tokens} allocated"
                )

        # Calculate fidelity
        fidelity = self._calculate_fidelity(allocated_items, total_original_tokens)

        logger.debug(
            f"Priority-first allocation: {len(allocated_items)} items allocated, "
            f"{len(dropped_ids)} dropped, fidelity={fidelity:.2%}"
        )

        return AllocationResult(
            items=allocated_items,
            tokens_used=total_allocated_tokens,
            tokens_available=budget,
            fidelity=fidelity,
            warnings=warnings,
            dropped_ids=dropped_ids,
        )

    def _allocate_equal_share(
        self,
        item_tokens: list[tuple[Any, int]],
        budget: int,
        total_original_tokens: int,
    ) -> AllocationResult:
        """Allocate budget equally across all items.

        Each item receives budget / num_items tokens. Items requiring
        less than their share get their actual requirement; excess is
        redistributed to items needing more.

        Args:
            item_tokens: List of (item, token_count) tuples, sorted by priority
            budget: Total budget available
            total_original_tokens: Sum of all original token counts

        Returns:
            AllocationResult with allocation details
        """
        if not item_tokens:
            return AllocationResult(
                tokens_available=budget,
                fidelity=1.0,
            )

        num_items = len(item_tokens)
        base_share = budget // num_items

        allocated_items: list[AllocatedItem] = []
        warnings: list[str] = []
        total_allocated_tokens = 0

        # First pass: allocate base share or less
        excess_budget = 0
        items_needing_more: list[tuple[int, Any, int]] = []  # (index, item, tokens)

        for idx, (item, tokens) in enumerate(item_tokens):
            if tokens <= base_share:
                # Item fits in base share
                item_id = getattr(item, "id", str(id(item)))
                item_priority = getattr(item, "priority", 999)
                item_content = getattr(item, "content", "")

                allocated_items.append(
                    AllocatedItem(
                        id=item_id,
                        content=item_content,
                        priority=item_priority,
                        original_tokens=tokens,
                        allocated_tokens=tokens,
                        needs_summarization=False,
                    )
                )
                total_allocated_tokens += tokens
                excess_budget += base_share - tokens
            else:
                # Item needs more than base share
                items_needing_more.append((idx, item, tokens))

        # Second pass: redistribute excess to items needing more
        if items_needing_more and excess_budget > 0:
            extra_per_item = excess_budget // len(items_needing_more)
        else:
            extra_per_item = 0

        for idx, item, tokens in items_needing_more:
            item_id = getattr(item, "id", str(id(item)))
            item_priority = getattr(item, "priority", 999)
            item_content = getattr(item, "content", "")

            allocated = min(tokens, base_share + extra_per_item)
            needs_summarization = allocated < tokens

            allocated_items.append(
                AllocatedItem(
                    id=item_id,
                    content=item_content,
                    priority=item_priority,
                    original_tokens=tokens,
                    allocated_tokens=allocated,
                    needs_summarization=needs_summarization,
                )
            )
            total_allocated_tokens += allocated

            if needs_summarization:
                warnings.append(
                    f"Item {item_id} needs summarization: "
                    f"{tokens} tokens -> {allocated} allocated (equal share)"
                )

        # Re-sort by priority for consistent output
        allocated_items.sort(key=lambda x: x.priority)

        # Calculate fidelity
        fidelity = self._calculate_fidelity(allocated_items, total_original_tokens)

        logger.debug(
            f"Equal-share allocation: {len(allocated_items)} items, "
            f"base share={base_share}, fidelity={fidelity:.2%}"
        )

        return AllocationResult(
            items=allocated_items,
            tokens_used=total_allocated_tokens,
            tokens_available=budget,
            fidelity=fidelity,
            warnings=warnings,
            dropped_ids=[],  # Equal share doesn't drop items
        )

    def _allocate_proportional(
        self,
        item_tokens: list[tuple[Any, int]],
        budget: int,
        total_original_tokens: int,
    ) -> AllocationResult:
        """Allocate budget proportional to item sizes.

        Each item receives budget * (item_tokens / total_tokens).
        Larger items get proportionally larger allocations.

        Args:
            item_tokens: List of (item, token_count) tuples, sorted by priority
            budget: Total budget available
            total_original_tokens: Sum of all original token counts

        Returns:
            AllocationResult with allocation details
        """
        if not item_tokens:
            return AllocationResult(
                tokens_available=budget,
                fidelity=1.0,
            )

        # If total fits in budget, no compression needed
        if total_original_tokens <= budget:
            allocated_items: list[AllocatedItem] = []
            for item, tokens in item_tokens:
                item_id = getattr(item, "id", str(id(item)))
                item_priority = getattr(item, "priority", 999)
                item_content = getattr(item, "content", "")

                allocated_items.append(
                    AllocatedItem(
                        id=item_id,
                        content=item_content,
                        priority=item_priority,
                        original_tokens=tokens,
                        allocated_tokens=tokens,
                        needs_summarization=False,
                    )
                )

            return AllocationResult(
                items=allocated_items,
                tokens_used=total_original_tokens,
                tokens_available=budget,
                fidelity=1.0,
                warnings=[],
                dropped_ids=[],
            )

        # Proportional allocation with compression
        compression_ratio = budget / total_original_tokens
        allocated_items = []
        warnings: list[str] = []
        total_allocated_tokens = 0

        for item, tokens in item_tokens:
            item_id = getattr(item, "id", str(id(item)))
            item_priority = getattr(item, "priority", 999)
            item_content = getattr(item, "content", "")

            # Allocate proportionally, minimum 1 token
            allocated = max(1, int(tokens * compression_ratio))

            allocated_items.append(
                AllocatedItem(
                    id=item_id,
                    content=item_content,
                    priority=item_priority,
                    original_tokens=tokens,
                    allocated_tokens=allocated,
                    needs_summarization=allocated < tokens,
                )
            )
            total_allocated_tokens += allocated

            if allocated < tokens:
                warnings.append(
                    f"Item {item_id} compressed: {tokens} -> {allocated} tokens "
                    f"({compression_ratio:.1%} of original)"
                )

        # Calculate fidelity
        fidelity = self._calculate_fidelity(allocated_items, total_original_tokens)

        logger.debug(
            f"Proportional allocation: {len(allocated_items)} items, "
            f"compression={compression_ratio:.2%}, fidelity={fidelity:.2%}"
        )

        return AllocationResult(
            items=allocated_items,
            tokens_used=total_allocated_tokens,
            tokens_available=budget,
            fidelity=fidelity,
            warnings=warnings,
            dropped_ids=[],  # Proportional doesn't drop items
        )

    def _calculate_fidelity(
        self,
        allocated_items: list[AllocatedItem],
        total_original_tokens: int,
    ) -> float:
        """Calculate overall fidelity score for an allocation.

        Fidelity represents how much of the original content is preserved:
        - 1.0 = All items allocated at full fidelity
        - 0.0 = All content dropped or maximally compressed

        Dropped items are implicitly accounted for since they contribute
        0 to the allocated token total.

        Args:
            allocated_items: Items that received allocation
            total_original_tokens: Total tokens in original content

        Returns:
            Fidelity score from 0.0 to 1.0
        """
        if total_original_tokens <= 0:
            return 1.0

        # Sum of allocated tokens represents preserved content
        total_allocated = sum(item.allocated_tokens for item in allocated_items)

        # Fidelity is ratio of allocated to original
        fidelity = total_allocated / total_original_tokens

        # Clamp to valid range
        return max(0.0, min(1.0, fidelity))
