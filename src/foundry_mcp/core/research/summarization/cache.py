"""In-memory cache for summarization results."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional

from .constants import _SUMMARY_CACHE_MAX_SIZE
from .models import SummarizationLevel, SummarizationResult

logger = logging.getLogger(__name__)


class SummaryCache:
    """In-memory cache for summarization results.

    Caches summarization results using composite keys that include content hash,
    context hash, summarization level, and provider. This ensures cache
    invalidation when any relevant factor changes.

    The cache is bounded to prevent unbounded memory growth, using a simple
    half-flush eviction strategy when the limit is reached.

    Attributes:
        _cache: Internal dict mapping cache keys to SummarizationResult
        _enabled: Whether caching is enabled
        _max_size: Maximum number of entries

    Example:
        cache = SummaryCache(enabled=True)

        # Check cache before summarization
        result = cache.get(content, context, level, provider)
        if result is None:
            result = await summarizer._summarize_single(content, level, provider)
            cache.set(content, context, level, provider, result)
    """

    def __init__(
        self,
        enabled: bool = True,
        max_size: int = _SUMMARY_CACHE_MAX_SIZE,
    ):
        """Initialize the summary cache.

        Args:
            enabled: Whether caching is enabled (default True)
            max_size: Maximum cache entries before eviction
        """
        self._cache: dict[tuple[str, str, str, str], SummarizationResult] = {}
        self._enabled = enabled
        self._max_size = max_size

    @property
    def enabled(self) -> bool:
        """Check if caching is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable caching."""
        self._enabled = value

    @staticmethod
    def _content_hash(content: str) -> str:
        """Generate a hash of content for cache keying.

        Uses SHA-256 truncated to 16 characters for reasonable uniqueness
        while keeping cache keys compact.

        Args:
            content: Text content to hash

        Returns:
            Hex string hash of the content
        """
        return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:16]

    def _make_key(
        self,
        content: str,
        context: Optional[str],
        level: SummarizationLevel,
        provider_id: Optional[str],
    ) -> tuple[str, str, str, str]:
        """Create a cache key from the input parameters.

        Args:
            content: Content being summarized
            context: Optional context string
            level: Summarization level
            provider_id: Provider identifier

        Returns:
            Tuple of (content_hash, context_hash, level_value, provider_id)
        """
        content_hash = self._content_hash(content)
        context_hash = self._content_hash(context) if context else ""
        return (content_hash, context_hash, level.value, provider_id or "")

    def get(
        self,
        content: str,
        context: Optional[str],
        level: SummarizationLevel,
        provider_id: Optional[str],
    ) -> Optional[SummarizationResult]:
        """Retrieve a cached summarization result.

        Args:
            content: Content that was summarized
            context: Optional context string
            level: Summarization level
            provider_id: Provider identifier

        Returns:
            Cached SummarizationResult if found and cache enabled, None otherwise
        """
        if not self._enabled:
            return None

        key = self._make_key(content, context, level, provider_id)
        result = self._cache.get(key)

        if result is not None:
            logger.debug(f"Summary cache hit for {key[0][:8]}... at {level.value}")

        return result

    def set(
        self,
        content: str,
        context: Optional[str],
        level: SummarizationLevel,
        provider_id: Optional[str],
        result: SummarizationResult,
    ) -> None:
        """Store a summarization result in the cache.

        If the cache is full, evicts the oldest half of entries before adding.

        Args:
            content: Content that was summarized
            context: Optional context string
            level: Summarization level
            provider_id: Provider identifier
            result: The summarization result to cache
        """
        if not self._enabled:
            return

        # Evict oldest entries if at capacity (simple half-flush)
        if len(self._cache) >= self._max_size:
            keys_to_remove = list(self._cache.keys())[: self._max_size // 2]
            for key in keys_to_remove:
                del self._cache[key]
            logger.debug(f"Summary cache evicted {len(keys_to_remove)} entries")

        key = self._make_key(content, context, level, provider_id)
        self._cache[key] = result
        logger.debug(f"Summary cache stored {key[0][:8]}... at {level.value}")

    def clear(self) -> int:
        """Clear all cached entries.

        Returns:
            Number of entries that were cleared
        """
        count = len(self._cache)
        self._cache.clear()
        logger.debug(f"Summary cache cleared {count} entries")
        return count

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with size, max_size, and enabled status
        """
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "enabled": self._enabled,
        }
