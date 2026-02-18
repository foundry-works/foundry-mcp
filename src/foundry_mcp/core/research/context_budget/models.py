"""Data models for context budget allocation.

Provides allocation strategies, content item types, and result containers
for the context budget management system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable

from foundry_mcp.core.research.models.sources import ResearchSource


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
