"""Domain-based source quality assessment, title normalization, and relevance scoring.

Provides URL domain extraction, wildcard pattern matching,
quality tier classification, and keyword-based relevance scoring
for research sources.
"""

from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Stopwords for relevance scoring (common English words to ignore)
# ---------------------------------------------------------------------------
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "need", "must",
        "it", "its", "this", "that", "these", "those", "i", "we", "you",
        "he", "she", "they", "me", "him", "her", "us", "them", "my", "your",
        "his", "our", "their", "what", "which", "who", "whom", "how", "when",
        "where", "why", "if", "then", "so", "no", "not", "only", "very",
        "just", "about", "also", "more", "some", "any", "each", "every",
        "all", "both", "few", "most", "other", "into", "over", "such", "than",
        "too", "up", "out", "as", "well", "back", "there", "here", "after",
        "before", "between", "through", "during", "without", "again",
    }
)

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def _tokenize(text: str) -> set[str]:
    """Tokenize text into a keyword set for relevance comparison.

    Lowercases, extracts word tokens, and removes stopwords.
    """
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1}

from foundry_mcp.core.research.models.sources import (
    DOMAIN_TIERS,
    ResearchMode,
    SourceQuality,
)


def _extract_domain(url: str) -> Optional[str]:
    """Extract domain from URL.

    Args:
        url: Full URL string

    Returns:
        Domain string (e.g., "arxiv.org") or None if extraction fails
    """
    if not url:
        return None
    try:
        # Handle URLs without scheme
        if "://" not in url:
            url = "https://" + url
        # Extract domain using simple parsing
        from urllib.parse import urlparse

        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]
        return domain if domain else None
    except Exception:
        return None


def _extract_hostname(url: str) -> Optional[str]:
    """Extract full hostname from URL (preserves subdomains like www.).

    Args:
        url: Full URL string

    Returns:
        Full hostname (e.g., "www.arxiv.org", "docs.python.org") or None
    """
    if not url:
        return None
    try:
        # Handle URLs without scheme
        if "://" not in url:
            url = "https://" + url
        from urllib.parse import urlparse

        parsed = urlparse(url)
        return parsed.netloc.lower() if parsed.netloc else None
    except Exception:
        return None


def _domain_matches_pattern(domain: str, pattern: str) -> bool:
    """Check if domain matches a pattern (supports wildcards).

    Patterns:
    - "arxiv.org" - exact match
    - "*.edu" - matches stanford.edu, mit.edu, etc.
    - "docs.*" - matches docs.python.org, docs.microsoft.com, etc.

    Args:
        domain: Domain to check (e.g., "stanford.edu")
        pattern: Pattern to match (e.g., "*.edu")

    Returns:
        True if domain matches pattern
    """
    pattern = pattern.lower()
    domain = domain.lower()

    if "*" not in pattern:
        # Exact match or subdomain match
        return domain == pattern or domain.endswith("." + pattern)

    if pattern.startswith("*."):
        # Suffix pattern: *.edu matches stanford.edu
        suffix = pattern[2:]
        return domain == suffix or domain.endswith("." + suffix)

    if pattern.endswith(".*"):
        # Prefix pattern: docs.* matches docs.python.org
        prefix = pattern[:-2]
        return domain == prefix or domain.startswith(prefix + ".")

    # General wildcard (treat as contains)
    return pattern.replace("*", "") in domain


def get_domain_quality(url: str, mode: ResearchMode) -> SourceQuality:
    """Determine source quality based on domain and research mode.

    Args:
        url: Source URL
        mode: Research mode (general, academic, technical)

    Returns:
        SourceQuality based on domain tier matching
    """
    domain = _extract_domain(url)
    if not domain:
        return SourceQuality.UNKNOWN

    tiers = DOMAIN_TIERS.get(mode.value, DOMAIN_TIERS["general"])

    # Check high-priority domains first
    for pattern in tiers.get("high", []):
        if _domain_matches_pattern(domain, pattern):
            return SourceQuality.HIGH

    # Check low-priority domains
    for pattern in tiers.get("low", []):
        if _domain_matches_pattern(domain, pattern):
            return SourceQuality.LOW

    # Default to medium for unmatched domains
    return SourceQuality.MEDIUM


def _normalize_title(title: str) -> str:
    """Normalize title for deduplication matching.

    Converts to lowercase, removes punctuation, and collapses whitespace
    to enable matching the same paper from different sources (e.g., arXiv vs OpenReview).

    Args:
        title: Source title to normalize

    Returns:
        Normalized title string for comparison
    """
    if not title:
        return ""
    # Lowercase, remove punctuation, collapse whitespace
    normalized = title.lower()
    normalized = re.sub(r"[^\w\s]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


# ---------------------------------------------------------------------------
# Source relevance scoring
# ---------------------------------------------------------------------------

# Weight for title keyword overlap vs content keyword overlap.
_TITLE_WEIGHT = 0.7
_CONTENT_WEIGHT = 0.3

# Multiplier applied to academic sources to penalize tangential hits
# from broad academic search APIs.
_ACADEMIC_PENALTY = 0.7


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _overlap_coefficient(a: set[str], b: set[str]) -> float:
    """Overlap coefficient (Szymkiewicz-Simpson) between two sets.

    Measures containment: ``|A ∩ B| / min(|A|, |B|)``.
    Unlike Jaccard, this is not penalized when one set is much larger
    than the other, making it suitable for relevance scoring where
    source text may be much longer than the reference query.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def compute_source_relevance(
    source_title: str,
    source_content: str | None,
    reference_text: str,
    *,
    source_type: str = "web",
) -> float:
    """Score source relevance against reference text (brief + sub-query).

    Uses weighted keyword overlap between source title/content and the
    reference text.  Academic sources (``source_type="academic"``) receive
    a stricter scoring curve since they are more likely to be tangential
    hits from broad academic search APIs.

    Args:
        source_title: Title of the source.
        source_content: Snippet or content of the source (may be None).
        reference_text: Combined research brief + sub-query text.
        source_type: ``"web"`` or ``"academic"``.

    Returns:
        Float in ``[0.0, 1.0]`` — higher means more relevant.
    """
    ref_keywords = _tokenize(reference_text)
    if not ref_keywords:
        return 0.0

    title_keywords = _tokenize(source_title) if source_title else set()
    content_keywords = _tokenize(source_content) if source_content else set()

    title_sim = _overlap_coefficient(title_keywords, ref_keywords)
    content_sim = _overlap_coefficient(content_keywords, ref_keywords)

    # Weighted combination — title is a stronger relevance signal
    if content_keywords:
        score = _TITLE_WEIGHT * title_sim + _CONTENT_WEIGHT * content_sim
    else:
        # No content available — rely entirely on title
        score = title_sim

    # Academic sources get a stricter curve
    if source_type == "academic":
        score *= _ACADEMIC_PENALTY

    return max(0.0, min(1.0, score))
