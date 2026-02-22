"""Context budget management sub-package.

Split from monolithic context_budget.py for maintainability.
All public symbols re-exported for backward compatibility.
"""

from foundry_mcp.core.errors.research import ProtectedContentOverflowError

from .constants import (
    CHARS_PER_TOKEN,
    CONDENSED_MIN_FIDELITY,
    HEADLINE_MIN_FIDELITY,
    MAX_TOKEN_CACHE_ENTRIES,
    MIN_ITEMS_PER_PHASE,
    PRIORITY_WEIGHT_CONFIDENCE,
    PRIORITY_WEIGHT_RECENCY,
    PRIORITY_WEIGHT_RELEVANCE,
    PRIORITY_WEIGHT_SOURCE_QUALITY,
    TOP_PRIORITY_ITEMS,
    TRUNCATION_MARKER,
)
from .degradation_models import (
    ChunkFailure,
    ChunkResult,
    DegradationLevel,
    DegradationResult,
    DegradationStep,
)
from .degradation_pipeline import DegradationPipeline
from .manager import ContextBudgetManager
from .models import (
    AllocatedItem,
    AllocationResult,
    AllocationStrategy,
    ContentItem,
    ContentItemProtocol,
)
from .priority import (
    CONFIDENCE_SCORES,
    SOURCE_QUALITY_SCORES,
    compute_priority,
    compute_recency_score,
)

__all__ = [
    # Constants
    "CHARS_PER_TOKEN",
    "CONDENSED_MIN_FIDELITY",
    "HEADLINE_MIN_FIDELITY",
    "MAX_TOKEN_CACHE_ENTRIES",
    "MIN_ITEMS_PER_PHASE",
    "PRIORITY_WEIGHT_CONFIDENCE",
    "PRIORITY_WEIGHT_RECENCY",
    "PRIORITY_WEIGHT_RELEVANCE",
    "PRIORITY_WEIGHT_SOURCE_QUALITY",
    "TOP_PRIORITY_ITEMS",
    "TRUNCATION_MARKER",
    # Priority scoring
    "CONFIDENCE_SCORES",
    "SOURCE_QUALITY_SCORES",
    "compute_priority",
    "compute_recency_score",
    # Allocation models
    "AllocationStrategy",
    "ContentItemProtocol",
    "ContentItem",
    "AllocatedItem",
    "AllocationResult",
    # Degradation models
    "DegradationLevel",
    "DegradationStep",
    "ChunkFailure",
    "ChunkResult",
    "DegradationResult",
    # Pipeline & manager
    "DegradationPipeline",
    "ContextBudgetManager",
    # Error re-export
    "ProtectedContentOverflowError",
]
