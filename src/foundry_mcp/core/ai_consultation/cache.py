"""
Filesystem-based cache for AI consultation results.

Provides persistent caching of consultation results to reduce redundant
API calls and improve response times for repeated queries.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from foundry_mcp.core.ai_consultation.types import (
    ConsultationResult,
    ConsultationWorkflow,
)

logger = logging.getLogger(__name__)


class ResultCache:
    """
    Filesystem-based cache for consultation results.

    Provides persistent caching of AI consultation results to reduce
    redundant API calls and improve response times for repeated queries.

    Cache Structure:
        .cache/foundry-mcp/consultations/{workflow}/{key}.json

    Each cached entry contains:
        - content: The consultation result
        - provider_id: Provider that generated the result
        - model_used: Model identifier
        - tokens: Token usage
        - timestamp: Cache entry creation time
        - ttl: Time-to-live in seconds

    Attributes:
        base_dir: Root directory for cache storage
        default_ttl: Default time-to-live in seconds (default: 3600 = 1 hour)
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        default_ttl: int = 3600,
    ):
        """
        Initialize the result cache.

        Args:
            base_dir: Root directory for cache (default: .cache/foundry-mcp/consultations)
            default_ttl: Default TTL in seconds (default: 3600)
        """
        if base_dir is None:
            base_dir = Path.cwd() / ".cache" / "foundry-mcp" / "consultations"
        self.base_dir = base_dir
        self.default_ttl = default_ttl

    def _get_cache_path(self, workflow: ConsultationWorkflow, key: str) -> Path:
        """Return the cache file path for a workflow and key."""
        # Sanitize key to be filesystem-safe
        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
        return self.base_dir / workflow.value / f"{safe_key}.json"

    def get(
        self,
        workflow: ConsultationWorkflow,
        key: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached result.

        Args:
            workflow: The consultation workflow
            key: The cache key

        Returns:
            Cached data dict if found and not expired, None otherwise
        """
        cache_path = self._get_cache_path(workflow, key)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check TTL
            timestamp = data.get("timestamp", 0)
            ttl = data.get("ttl", self.default_ttl)
            if time.time() - timestamp > ttl:
                # Expired - remove file
                cache_path.unlink(missing_ok=True)
                return None

            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read cache entry %s: %s", cache_path, exc)
            return None

    def set(
        self,
        workflow: ConsultationWorkflow,
        key: str,
        result: ConsultationResult,
        ttl: Optional[int] = None,
    ) -> None:
        """
        Store a consultation result in the cache.

        Args:
            workflow: The consultation workflow
            key: The cache key
            result: The consultation result to cache
            ttl: Time-to-live in seconds (default: default_ttl)
        """
        cache_path = self._get_cache_path(workflow, key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "content": result.content,
            "provider_id": result.provider_id,
            "model_used": result.model_used,
            "tokens": result.tokens,
            "timestamp": time.time(),
            "ttl": ttl if ttl is not None else self.default_ttl,
        }

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError as exc:
            logger.warning("Failed to write cache entry %s: %s", cache_path, exc)

    def invalidate(
        self,
        workflow: Optional[ConsultationWorkflow] = None,
        key: Optional[str] = None,
    ) -> int:
        """
        Invalidate cache entries.

        Args:
            workflow: If provided, only invalidate entries for this workflow
            key: If provided (with workflow), only invalidate this specific entry

        Returns:
            Number of entries invalidated
        """
        count = 0

        if workflow is not None and key is not None:
            # Invalidate specific entry
            cache_path = self._get_cache_path(workflow, key)
            if cache_path.exists():
                cache_path.unlink()
                count = 1
        elif workflow is not None:
            # Invalidate all entries for workflow
            workflow_dir = self.base_dir / workflow.value
            if workflow_dir.exists():
                for cache_file in workflow_dir.glob("*.json"):
                    cache_file.unlink()
                    count += 1
        else:
            # Invalidate all entries
            for workflow_enum in ConsultationWorkflow:
                workflow_dir = self.base_dir / workflow_enum.value
                if workflow_dir.exists():
                    for cache_file in workflow_dir.glob("*.json"):
                        cache_file.unlink()
                        count += 1

        return count

    def stats(self) -> Dict[str, Any]:
        """
        Return cache statistics.

        Returns:
            Dict with entry counts per workflow and total size
        """
        stats: Dict[str, Any] = {
            "total_entries": 0,
            "total_size_bytes": 0,
            "by_workflow": {},
        }

        for workflow in ConsultationWorkflow:
            workflow_dir = self.base_dir / workflow.value
            if workflow_dir.exists():
                entries = list(workflow_dir.glob("*.json"))
                size = sum(f.stat().st_size for f in entries if f.exists())
                stats["by_workflow"][workflow.value] = {
                    "entries": len(entries),
                    "size_bytes": size,
                }
                stats["total_entries"] += len(entries)
                stats["total_size_bytes"] += size
            else:
                stats["by_workflow"][workflow.value] = {
                    "entries": 0,
                    "size_bytes": 0,
                }

        return stats


__all__ = ["ResultCache"]
