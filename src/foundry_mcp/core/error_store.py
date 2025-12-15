"""
Error storage backends for the error collection infrastructure.

Provides abstract base class and concrete implementations for persisting
error records collected by ErrorCollector.
"""

from __future__ import annotations

import fcntl
import json
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from .error_collection import ErrorRecord

logger = logging.getLogger(__name__)


class ErrorStore(ABC):
    """Abstract base class for error storage backends."""

    @abstractmethod
    def append(self, record: ErrorRecord) -> None:
        """
        Append an error record to storage.

        Args:
            record: The error record to store
        """
        pass

    @abstractmethod
    def get(self, error_id: str) -> Optional[ErrorRecord]:
        """
        Retrieve an error record by ID.

        Args:
            error_id: The error ID to look up

        Returns:
            ErrorRecord if found, None otherwise
        """
        pass

    @abstractmethod
    def query(
        self,
        *,
        tool_name: Optional[str] = None,
        error_code: Optional[str] = None,
        error_type: Optional[str] = None,
        fingerprint: Optional[str] = None,
        provider_id: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ErrorRecord]:
        """
        Query error records with filtering.

        Args:
            tool_name: Filter by tool name
            error_code: Filter by error code
            error_type: Filter by error type
            fingerprint: Filter by fingerprint
            provider_id: Filter by provider ID
            since: ISO 8601 timestamp - include records after this time
            until: ISO 8601 timestamp - include records before this time
            limit: Maximum number of records to return
            offset: Number of records to skip

        Returns:
            List of matching ErrorRecords
        """
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """
        Get aggregated error statistics.

        Returns:
            Dictionary with statistics grouped by various dimensions
        """
        pass

    @abstractmethod
    def get_patterns(self, min_count: int = 3) -> list[dict[str, Any]]:
        """
        Get recurring error patterns.

        Args:
            min_count: Minimum occurrence count to include

        Returns:
            List of patterns with fingerprint, count, and metadata
        """
        pass

    @abstractmethod
    def cleanup(self, retention_days: int, max_errors: int) -> int:
        """
        Clean up old records based on retention policy.

        Args:
            retention_days: Delete records older than this many days
            max_errors: Maximum number of errors to keep

        Returns:
            Number of records deleted
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """
        Get total count of error records.

        Returns:
            Total number of stored error records
        """
        pass


class FileErrorStore(ErrorStore):
    """
    JSONL-based error storage implementation.

    Stores errors in append-only JSONL format with separate index and stats files
    for efficient querying. Thread-safe with file locking for concurrent access.

    Directory structure:
        .cache/foundry-mcp/errors/
            errors.jsonl     - Append-only error log
            index.json       - Fingerprint -> metadata mapping
            stats.json       - Pre-computed statistics (updated periodically)
    """

    def __init__(self, storage_path: str | Path):
        """
        Initialize the file-based error store.

        Args:
            storage_path: Directory path for error storage
        """
        self.storage_path = Path(storage_path).expanduser()
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.errors_file = self.storage_path / "errors.jsonl"
        self.index_file = self.storage_path / "index.json"
        self.stats_file = self.storage_path / "stats.json"

        self._lock = threading.Lock()
        self._index: dict[str, dict[str, Any]] = {}
        self._id_index: dict[str, int] = {}  # error_id -> line number
        self._stats_dirty = False
        self._last_stats_update: Optional[datetime] = None

        # Load index on initialization
        self._load_index()

    def _load_index(self) -> None:
        """Load the index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, "r") as f:
                    data = json.load(f)
                    self._index = data.get("fingerprints", {})
                    self._id_index = data.get("ids", {})
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load error index, rebuilding: {e}")
                self._rebuild_index()
        else:
            # First run or index deleted - rebuild from errors file
            self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild index from the errors JSONL file."""
        self._index = {}
        self._id_index = {}

        if not self.errors_file.exists():
            self._save_index()
            return

        line_num = 0
        try:
            with open(self.errors_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        line_num += 1
                        continue

                    try:
                        record_dict = json.loads(line)
                        error_id = record_dict.get("id", "")
                        fingerprint = record_dict.get("fingerprint", "")
                        timestamp = record_dict.get("timestamp", "")

                        if error_id:
                            self._id_index[error_id] = line_num

                        if fingerprint:
                            if fingerprint not in self._index:
                                self._index[fingerprint] = {
                                    "count": 0,
                                    "first_seen": timestamp,
                                    "last_seen": timestamp,
                                    "error_ids": [],
                                    "tool_name": record_dict.get("tool_name"),
                                    "error_code": record_dict.get("error_code"),
                                }

                            self._index[fingerprint]["count"] += 1
                            self._index[fingerprint]["last_seen"] = timestamp
                            # Keep last 10 error IDs per fingerprint
                            ids = self._index[fingerprint]["error_ids"]
                            if len(ids) >= 10:
                                ids.pop(0)
                            ids.append(error_id)

                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON at line {line_num}")

                    line_num += 1

        except OSError as e:
            logger.error(f"Failed to rebuild index: {e}")

        self._save_index()
        logger.info(f"Rebuilt error index: {len(self._index)} patterns, {len(self._id_index)} records")

    def _save_index(self) -> None:
        """Save the index to disk."""
        try:
            data = {
                "fingerprints": self._index,
                "ids": self._id_index,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            # Atomic write via temp file
            temp_file = self.index_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2)
            temp_file.rename(self.index_file)
        except OSError as e:
            logger.error(f"Failed to save error index: {e}")

    def append(self, record: ErrorRecord) -> None:
        """Append an error record to storage."""
        with self._lock:
            record_dict = asdict(record)

            # Append to JSONL file with file locking
            try:
                with open(self.errors_file, "a") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        # Get current line number before writing
                        line_num = sum(1 for _ in open(self.errors_file, "r")) if self.errors_file.exists() else 0
                        f.write(json.dumps(record_dict, default=str) + "\n")
                        f.flush()
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            except OSError as e:
                logger.error(f"Failed to append error record: {e}")
                return

            # Update index
            self._id_index[record.id] = line_num

            if record.fingerprint not in self._index:
                self._index[record.fingerprint] = {
                    "count": 0,
                    "first_seen": record.timestamp,
                    "last_seen": record.timestamp,
                    "error_ids": [],
                    "tool_name": record.tool_name,
                    "error_code": record.error_code,
                }

            fp_data = self._index[record.fingerprint]
            fp_data["count"] += 1
            fp_data["last_seen"] = record.timestamp
            if len(fp_data["error_ids"]) >= 10:
                fp_data["error_ids"].pop(0)
            fp_data["error_ids"].append(record.id)

            self._save_index()
            self._stats_dirty = True

    def get(self, error_id: str) -> Optional[ErrorRecord]:
        """Retrieve an error record by ID."""
        with self._lock:
            line_num = self._id_index.get(error_id)
            if line_num is None:
                return None

            try:
                with open(self.errors_file, "r") as f:
                    for i, line in enumerate(f):
                        if i == line_num:
                            record_dict = json.loads(line.strip())
                            return ErrorRecord(**record_dict)
            except (OSError, json.JSONDecodeError) as e:
                logger.error(f"Failed to retrieve error {error_id}: {e}")

            return None

    def query(
        self,
        *,
        tool_name: Optional[str] = None,
        error_code: Optional[str] = None,
        error_type: Optional[str] = None,
        fingerprint: Optional[str] = None,
        provider_id: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ErrorRecord]:
        """Query error records with filtering."""
        results: list[ErrorRecord] = []
        skipped = 0

        # Parse time filters
        since_dt = datetime.fromisoformat(since.replace("Z", "+00:00")) if since else None
        until_dt = datetime.fromisoformat(until.replace("Z", "+00:00")) if until else None

        with self._lock:
            if not self.errors_file.exists():
                return []

            try:
                with open(self.errors_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            record_dict = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        # Apply filters
                        if tool_name and record_dict.get("tool_name") != tool_name:
                            continue
                        if error_code and record_dict.get("error_code") != error_code:
                            continue
                        if error_type and record_dict.get("error_type") != error_type:
                            continue
                        if fingerprint and record_dict.get("fingerprint") != fingerprint:
                            continue
                        if provider_id and record_dict.get("provider_id") != provider_id:
                            continue

                        # Time filters
                        if since_dt or until_dt:
                            try:
                                ts = record_dict.get("timestamp", "")
                                record_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                                if since_dt and record_dt < since_dt:
                                    continue
                                if until_dt and record_dt > until_dt:
                                    continue
                            except (ValueError, TypeError):
                                continue

                        # Apply offset
                        if skipped < offset:
                            skipped += 1
                            continue

                        # Check limit
                        if len(results) >= limit:
                            break

                        results.append(ErrorRecord(**record_dict))

            except OSError as e:
                logger.error(f"Failed to query errors: {e}")

        return results

    def get_stats(self) -> dict[str, Any]:
        """Get aggregated error statistics."""
        # Check if we have recent cached stats
        if self.stats_file.exists() and not self._stats_dirty:
            try:
                with open(self.stats_file, "r") as f:
                    cached = json.load(f)
                    updated_at = cached.get("updated_at", "")
                    if updated_at:
                        updated_dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                        # Use cached stats if less than 5 minutes old
                        if datetime.now(timezone.utc) - updated_dt < timedelta(minutes=5):
                            return cached
            except (OSError, json.JSONDecodeError):
                pass

        # Compute fresh stats
        stats = self._compute_stats()

        # Cache stats
        try:
            with open(self.stats_file, "w") as f:
                json.dump(stats, f, indent=2)
            self._stats_dirty = False
        except OSError as e:
            logger.warning(f"Failed to cache stats: {e}")

        return stats

    def _compute_stats(self) -> dict[str, Any]:
        """Compute error statistics from index."""
        with self._lock:
            total_errors = sum(fp["count"] for fp in self._index.values())
            unique_patterns = len(self._index)

            # Group by tool
            by_tool: dict[str, int] = {}
            by_error_code: dict[str, int] = {}

            for fp_data in self._index.values():
                tool = fp_data.get("tool_name", "unknown")
                code = fp_data.get("error_code", "unknown")
                count = fp_data.get("count", 0)

                by_tool[tool] = by_tool.get(tool, 0) + count
                by_error_code[code] = by_error_code.get(code, 0) + count

            # Top patterns
            top_patterns = sorted(
                [
                    {
                        "fingerprint": fp,
                        "count": data["count"],
                        "tool_name": data.get("tool_name"),
                        "error_code": data.get("error_code"),
                        "first_seen": data.get("first_seen"),
                        "last_seen": data.get("last_seen"),
                    }
                    for fp, data in self._index.items()
                ],
                key=lambda x: x["count"],
                reverse=True,
            )[:20]

            return {
                "total_errors": total_errors,
                "unique_patterns": unique_patterns,
                "by_tool": by_tool,
                "by_error_code": by_error_code,
                "top_patterns": top_patterns,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

    def get_patterns(self, min_count: int = 3) -> list[dict[str, Any]]:
        """Get recurring error patterns."""
        with self._lock:
            patterns = []

            for fp, data in self._index.items():
                if data.get("count", 0) >= min_count:
                    patterns.append(
                        {
                            "fingerprint": fp,
                            "count": data["count"],
                            "tool_name": data.get("tool_name"),
                            "error_code": data.get("error_code"),
                            "first_seen": data.get("first_seen"),
                            "last_seen": data.get("last_seen"),
                            "sample_ids": data.get("error_ids", [])[-5:],
                        }
                    )

            # Sort by count descending
            patterns.sort(key=lambda x: x["count"], reverse=True)
            return patterns

    def cleanup(self, retention_days: int, max_errors: int) -> int:
        """Clean up old records based on retention policy."""
        with self._lock:
            if not self.errors_file.exists():
                return 0

            cutoff_dt = datetime.now(timezone.utc) - timedelta(days=retention_days)
            kept_records: list[str] = []
            deleted_count = 0

            try:
                # Read all records
                with open(self.errors_file, "r") as f:
                    lines = f.readlines()

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record_dict = json.loads(line)
                        ts = record_dict.get("timestamp", "")
                        record_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))

                        # Keep if within retention period
                        if record_dt >= cutoff_dt:
                            kept_records.append(line)
                        else:
                            deleted_count += 1

                    except (json.JSONDecodeError, ValueError):
                        # Keep malformed records to avoid data loss
                        kept_records.append(line)

                # Enforce max_errors limit (keep most recent)
                if len(kept_records) > max_errors:
                    deleted_count += len(kept_records) - max_errors
                    kept_records = kept_records[-max_errors:]

                # Write back
                temp_file = self.errors_file.with_suffix(".tmp")
                with open(temp_file, "w") as f:
                    for line in kept_records:
                        f.write(line + "\n")
                temp_file.rename(self.errors_file)

                # Rebuild index after cleanup
                self._rebuild_index()

                logger.info(f"Cleaned up {deleted_count} error records")
                return deleted_count

            except OSError as e:
                logger.error(f"Failed to cleanup errors: {e}")
                return 0

    def count(self) -> int:
        """Get total count of error records."""
        with self._lock:
            return len(self._id_index)


# Global store instance
_error_store: Optional[ErrorStore] = None
_store_lock = threading.Lock()


def get_error_store(storage_path: Optional[str | Path] = None) -> ErrorStore:
    """
    Get the global error store instance.

    Args:
        storage_path: Optional path to initialize the store. If not provided
                     on first call, uses default path.

    Returns:
        The ErrorStore instance
    """
    global _error_store

    with _store_lock:
        if _error_store is None:
            if storage_path is None:
                # Default path
                storage_path = Path.home() / ".cache" / "foundry-mcp" / "errors"
            _error_store = FileErrorStore(storage_path)

        return _error_store


def reset_error_store() -> None:
    """Reset the global error store (for testing)."""
    global _error_store
    with _store_lock:
        _error_store = None
