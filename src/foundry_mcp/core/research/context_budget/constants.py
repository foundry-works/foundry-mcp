"""Constants for context budget management and degradation pipeline."""

from __future__ import annotations

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
# Priority Scoring Constants
# =============================================================================

# Weight factors for priority scoring (must sum to 1.0)
PRIORITY_WEIGHT_SOURCE_QUALITY = 0.40
PRIORITY_WEIGHT_CONFIDENCE = 0.30
PRIORITY_WEIGHT_RECENCY = 0.15
PRIORITY_WEIGHT_RELEVANCE = 0.15
