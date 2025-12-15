"""Cache management for AI consultation results.

Provides a simple file-based cache for storing AI consultation results
(plan reviews, fidelity reviews, etc.) to avoid redundant API calls.
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class CacheStats:
    """Cache statistics."""

    cache_dir: str
    total_entries: int
    active_entries: int
    expired_entries: int
    total_size_bytes: int
    total_size_mb: float


def get_cache_dir() -> Path:
    """Get the cache directory path.

    Resolution order:
    1. FOUNDRY_MCP_CACHE_DIR environment variable
    2. ~/.foundry-mcp/cache

    Returns:
        Path to the cache directory.
    """
    if cache_dir := os.environ.get("FOUNDRY_MCP_CACHE_DIR"):
        return Path(cache_dir)

    return Path.home() / ".foundry-mcp" / "cache"


def is_cache_enabled() -> bool:
    """Check if caching is enabled.

    Returns:
        True if caching is enabled (default), False if disabled.
    """
    disabled = os.environ.get("FOUNDRY_MCP_CACHE_DISABLED", "").lower()
    return disabled not in ("true", "1", "yes")


class CacheManager:
    """Manages the AI consultation cache."""

    # Default TTL: 7 days in seconds
    DEFAULT_TTL = 7 * 24 * 60 * 60

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the cache manager.

        Args:
            cache_dir: Optional override for cache directory.
        """
        self.cache_dir = cache_dir or get_cache_dir()

    def ensure_dir(self) -> None:
        """Ensure the cache directory exists."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache statistics.
        """
        if not self.cache_dir.exists():
            return {
                "cache_dir": str(self.cache_dir),
                "total_entries": 0,
                "active_entries": 0,
                "expired_entries": 0,
                "total_size_bytes": 0,
                "total_size_mb": 0.0,
            }

        now = time.time()
        total_entries = 0
        active_entries = 0
        expired_entries = 0
        total_size = 0

        for entry_file in self.cache_dir.glob("*.json"):
            total_entries += 1
            total_size += entry_file.stat().st_size

            try:
                with open(entry_file, "r") as f:
                    entry = json.load(f)
                    expires_at = entry.get("expires_at", 0)
                    if expires_at > now:
                        active_entries += 1
                    else:
                        expired_entries += 1
            except (json.JSONDecodeError, KeyError):
                # Treat malformed entries as expired
                expired_entries += 1

        return {
            "cache_dir": str(self.cache_dir),
            "total_entries": total_entries,
            "active_entries": active_entries,
            "expired_entries": expired_entries,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }

    def clear(
        self,
        spec_id: Optional[str] = None,
        review_type: Optional[str] = None,
    ) -> int:
        """Clear cache entries with optional filters.

        Args:
            spec_id: Only clear entries for this spec ID.
            review_type: Only clear entries of this type (fidelity, plan).

        Returns:
            Number of entries deleted.
        """
        if not self.cache_dir.exists():
            return 0

        deleted = 0

        for entry_file in self.cache_dir.glob("*.json"):
            should_delete = True

            # Apply filters if specified
            if spec_id or review_type:
                try:
                    with open(entry_file, "r") as f:
                        entry = json.load(f)

                    if spec_id and entry.get("spec_id") != spec_id:
                        should_delete = False

                    if review_type and entry.get("review_type") != review_type:
                        should_delete = False

                except (json.JSONDecodeError, KeyError):
                    # Delete malformed entries
                    pass

            if should_delete:
                try:
                    entry_file.unlink()
                    deleted += 1
                except OSError:
                    pass

        return deleted

    def cleanup_expired(self) -> int:
        """Remove expired cache entries.

        Returns:
            Number of entries removed.
        """
        if not self.cache_dir.exists():
            return 0

        now = time.time()
        removed = 0

        for entry_file in self.cache_dir.glob("*.json"):
            try:
                with open(entry_file, "r") as f:
                    entry = json.load(f)

                expires_at = entry.get("expires_at", 0)
                if expires_at <= now:
                    entry_file.unlink()
                    removed += 1

            except (json.JSONDecodeError, KeyError, OSError):
                # Remove malformed entries
                try:
                    entry_file.unlink()
                    removed += 1
                except OSError:
                    pass

        return removed
