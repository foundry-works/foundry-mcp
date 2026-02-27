"""Content similarity, deduplication, and novelty tagging."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

from foundry_mcp.core.research.workflows.deep_research.source_quality import (
    _extract_domain,
)

# N-gram size for shingling
_SHINGLE_SIZE: int = 5


def _char_ngrams(text: str, n: int = _SHINGLE_SIZE) -> set[str]:
    """Generate character n-grams (shingles) from text.

    Args:
        text: Input text (should be pre-normalized)
        n: N-gram size

    Returns:
        Set of character n-grams
    """
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _normalize_content_for_dedup(text: str) -> str:
    """Normalize text for content similarity comparison.

    Lowercases, removes extra whitespace, strips punctuation-heavy
    boilerplate (e.g. navigation, footer text) by collapsing whitespace.

    Args:
        text: Raw source content

    Returns:
        Normalized text string
    """
    if not text:
        return ""
    normalized = text.lower()
    # Remove common boilerplate markers before whitespace collapse
    normalized = re.sub(r"copyright \d{4}.*?(?:all rights reserved\.?)?", "", normalized)
    # Collapse all whitespace (including newlines) into single spaces
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def content_similarity(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity between two texts using character n-grams.

    Uses shingling (character n-grams) and Jaccard coefficient to estimate
    content overlap.  This is fast, requires no external dependencies, and
    works well for detecting mirror/syndicated content.

    Args:
        text_a: First text
        text_b: Second text

    Returns:
        Similarity score between 0.0 (no overlap) and 1.0 (identical)
    """
    norm_a = _normalize_content_for_dedup(text_a)
    norm_b = _normalize_content_for_dedup(text_b)

    if not norm_a or not norm_b:
        return 0.0

    # Quick length-ratio check: if lengths differ by more than 3x,
    # similarity will be low regardless — skip expensive shingling.
    len_ratio = min(len(norm_a), len(norm_b)) / max(len(norm_a), len(norm_b))
    if len_ratio < 0.3:
        return 0.0

    shingles_a = _char_ngrams(norm_a)
    shingles_b = _char_ngrams(norm_b)

    if not shingles_a or not shingles_b:
        return 0.0

    intersection = len(shingles_a & shingles_b)
    union = len(shingles_a | shingles_b)

    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Novelty tagging for researcher stop decisions (Phase 3)
# ---------------------------------------------------------------------------

# Thresholds for novelty classification
_NOVELTY_DUPLICATE_THRESHOLD: float = 0.7
_NOVELTY_RELATED_THRESHOLD: float = 0.3


@dataclass
class NoveltyTag:
    """Novelty classification for a single search result.

    Attributes:
        tag: Display tag — ``"[NEW]"``, ``"[RELATED: <title>]"``, or ``"[DUPLICATE]"``
        category: One of ``"new"``, ``"related"``, ``"duplicate"``
        similarity: Highest similarity score against existing sources
        matched_title: Title of the most similar existing source (if any)
    """

    tag: str
    category: str
    similarity: float
    matched_title: Optional[str] = None


def compute_novelty_tag(
    new_content: str,
    new_url: Optional[str],
    existing_sources: Sequence[tuple[str, str, Optional[str]]],
    *,
    duplicate_threshold: float = _NOVELTY_DUPLICATE_THRESHOLD,
    related_threshold: float = _NOVELTY_RELATED_THRESHOLD,
) -> NoveltyTag:
    """Classify a new search result's novelty against existing sources.

    Uses a two-pass approach: first checks URL domain overlap for a cheap
    signal, then falls back to content similarity via ``content_similarity()``.

    Args:
        new_content: Content (or summary) of the new source.
        new_url: URL of the new source (may be None).
        existing_sources: List of ``(content, title, url)`` tuples for
            sources already found for this sub-query.
        duplicate_threshold: Similarity >= this is ``[DUPLICATE]``.
        related_threshold: Similarity >= this (and < duplicate) is ``[RELATED]``.

    Returns:
        NoveltyTag with classification and metadata.
    """
    if not existing_sources:
        return NoveltyTag(tag="[NEW]", category="new", similarity=0.0)

    best_sim = 0.0
    best_title: Optional[str] = None

    for ex_content, ex_title, ex_url in existing_sources:
        # Quick URL-domain check: same domain boosts similarity estimate
        domain_boost = 0.0
        if new_url and ex_url:
            new_domain = _extract_domain(new_url)
            ex_domain = _extract_domain(ex_url)
            if new_domain and ex_domain and new_domain == ex_domain:
                domain_boost = 0.1

        sim = content_similarity(new_content, ex_content) + domain_boost
        sim = min(sim, 1.0)  # cap at 1.0

        if sim > best_sim:
            best_sim = sim
            best_title = ex_title

    if best_sim >= duplicate_threshold:
        return NoveltyTag(
            tag="[DUPLICATE]",
            category="duplicate",
            similarity=best_sim,
            matched_title=best_title,
        )
    elif best_sim >= related_threshold:
        # Truncate title for readability
        display_title = best_title[:60] + "..." if best_title and len(best_title) > 60 else best_title
        return NoveltyTag(
            tag=f"[RELATED: {display_title}]",
            category="related",
            similarity=best_sim,
            matched_title=best_title,
        )
    else:
        return NoveltyTag(tag="[NEW]", category="new", similarity=best_sim)
