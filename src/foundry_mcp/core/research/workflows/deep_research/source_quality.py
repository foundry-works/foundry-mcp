"""Domain-based source quality assessment and title normalization.

Provides URL domain extraction, wildcard pattern matching, and
quality tier classification for research sources.
"""

from __future__ import annotations

import re
from typing import Optional

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
