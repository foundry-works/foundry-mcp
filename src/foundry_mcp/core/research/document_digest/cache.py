"""In-memory cache for digest results.

Provides DigestCache with bounded size and half-flush eviction strategy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .results import DigestResult

logger = logging.getLogger(__name__)

# Default maximum cache size
_DIGEST_CACHE_MAX_SIZE = 100


class DigestCache:
    """In-memory cache for digest results.

    Caches DigestResult objects using composite keys that include source ID,
    content hash, query hash, and config hash. This ensures cache invalidation
    when any relevant factor changes.

    The cache is bounded to prevent unbounded memory growth, using a simple
    half-flush eviction strategy when the limit is reached.

    Attributes:
        _cache: Internal dict mapping cache keys to DigestResult
        _enabled: Whether caching is enabled
        _max_size: Maximum number of entries

    Example:
        cache = DigestCache(enabled=True)

        # Check cache before digestion
        result = cache.get(cache_key)
        if result is None:
            result = await digestor._generate_digest(...)
            cache.set(cache_key, result)
    """

    def __init__(
        self,
        enabled: bool = True,
        max_size: int = _DIGEST_CACHE_MAX_SIZE,
    ):
        """Initialize the digest cache.

        Args:
            enabled: Whether caching is enabled (default True)
            max_size: Maximum cache entries before eviction
        """
        self._cache: dict[str, DigestResult] = {}
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

    def get(self, cache_key: str) -> Optional["DigestResult"]:
        """Retrieve a cached digest result.

        Args:
            cache_key: Cache key from generate_cache_key()

        Returns:
            Cached DigestResult if found and cache enabled, None otherwise
        """
        if not self._enabled:
            return None

        result = self._cache.get(cache_key)

        if result is not None:
            logger.debug(f"Digest cache hit for key {cache_key[:30]}...")

        return result

    def set(self, cache_key: str, result: "DigestResult") -> None:
        """Store a digest result in the cache.

        If the cache is full, performs half-flush eviction (removes oldest
        half of entries) before storing the new result.

        Args:
            cache_key: Cache key from generate_cache_key()
            result: DigestResult to cache
        """
        if not self._enabled:
            return

        # Evict if at capacity (half-flush strategy)
        if len(self._cache) >= self._max_size:
            keys = list(self._cache.keys())
            for key in keys[:len(keys) // 2]:
                del self._cache[key]
            logger.debug(f"Digest cache eviction: removed {len(keys) // 2} entries")

        self._cache[cache_key] = result
        logger.debug(f"Digest cached for key {cache_key[:30]}...")

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)
