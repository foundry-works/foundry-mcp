"""Priority scoring utilities for context budget allocation.

Provides functions and score mappings for computing content priority
based on source quality, confidence, recency, and relevance.
"""

from __future__ import annotations

from typing import Optional

from foundry_mcp.core.research.models.enums import ConfidenceLevel
from foundry_mcp.core.research.models.sources import SourceQuality

from .constants import (
    PRIORITY_WEIGHT_CONFIDENCE,
    PRIORITY_WEIGHT_RECENCY,
    PRIORITY_WEIGHT_RELEVANCE,
    PRIORITY_WEIGHT_SOURCE_QUALITY,
)

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
