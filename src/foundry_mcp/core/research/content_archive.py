"""File-based content archive for research workflows.

Provides archival storage for content that has been dropped or compressed
during token budget management. This enables potential future restoration
of original content and maintains an audit trail of content transformations.

Content is stored with SHA256 hash-based filenames for deduplication and
efficient retrieval. Each record includes the original content, metadata
about the archival reason, and TTL for automatic cleanup.
"""

import hashlib
import json
import logging
import os
import stat
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from filelock import FileLock

logger = logging.getLogger(__name__)


# Default TTL for archived content (7 days)
DEFAULT_ARCHIVE_TTL_HOURS = 168

# Warning codes for archive failures
ARCHIVE_WRITE_FAILED = "ARCHIVE_WRITE_FAILED"
ARCHIVE_DISABLED = "ARCHIVE_DISABLED"
ARCHIVE_READ_CORRUPT = "ARCHIVE_READ_CORRUPT"


@dataclass
class ArchivedContent:
    """Record of archived content.

    Stores the original content along with metadata about when and why
    it was archived. Supports JSON serialization for file storage.

    Attributes:
        content_hash: SHA256 hash of the content (also used as filename)
        content: The original content text
        item_id: ID of the item this content belongs to
        item_type: Type of content ("source", "finding", "gap")
        archived_at: UTC timestamp when content was archived
        archive_reason: Why the content was archived (e.g., "dropped", "compressed")
        original_tokens: Token count of the original content
        metadata: Additional metadata about the content
    """

    content_hash: str
    content: str
    item_id: str
    item_type: str = "source"
    archived_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    archive_reason: str = ""
    original_tokens: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "content_hash": self.content_hash,
            "content": self.content,
            "item_id": self.item_id,
            "item_type": self.item_type,
            "archived_at": self.archived_at.isoformat(),
            "archive_reason": self.archive_reason,
            "original_tokens": self.original_tokens,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArchivedContent":
        """Create from dictionary (deserialization)."""
        archived_at = data.get("archived_at")
        if isinstance(archived_at, str):
            # Parse ISO format timestamp
            archived_at = datetime.fromisoformat(archived_at.replace("Z", "+00:00"))
        elif archived_at is None:
            archived_at = datetime.now(timezone.utc)

        return cls(
            content_hash=data["content_hash"],
            content=data["content"],
            item_id=data["item_id"],
            item_type=data.get("item_type", "source"),
            archived_at=archived_at,
            archive_reason=data.get("archive_reason", ""),
            original_tokens=data.get("original_tokens"),
            metadata=data.get("metadata", {}),
        )


def compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of content for deduplication.

    Args:
        content: Text content to hash

    Returns:
        Hex-encoded SHA256 hash (64 characters)
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class ContentArchive:
    """File-based archive for dropped or compressed content.

    Stores content using SHA256 hash as filename for deduplication.
    Provides automatic TTL-based cleanup and thread-safe operations.

    Directory permissions are set to 0o700 (owner read/write/execute only)
    for security, as archived content may contain sensitive research data.

    Example usage:
        archive = ContentArchive(storage_path=Path("./archive"))

        # Archive some content
        record = archive.archive(
            content="Full article text...",
            item_id="src-abc123",
            reason="budget_exceeded",
        )

        # Retrieve later
        retrieved = archive.retrieve(record.content_hash)
        if retrieved:
            print(retrieved.content)

        # Cleanup old entries
        removed = archive.cleanup_expired()
    """

    def __init__(
        self,
        storage_path: Path,
        ttl_hours: int = DEFAULT_ARCHIVE_TTL_HOURS,
        enabled: bool = False,
    ) -> None:
        """Initialize content archive.

        Archive is disabled by default. When disabled, archive operations
        are no-ops that return None. Enable explicitly when archival is
        needed for a session.

        Args:
            storage_path: Directory to store archived content
            ttl_hours: Time-to-live in hours (default: 168 = 7 days)
            enabled: Whether archival is enabled (default: False)
        """
        self.storage_path = storage_path
        self.ttl_hours = ttl_hours
        self._enabled = enabled
        self._writable: Optional[bool] = None  # Cached writability check
        self._warnings: list[str] = []  # Collected warnings

        if self._enabled:
            self._ensure_directory()
            self._check_writable()

    @property
    def enabled(self) -> bool:
        """Check if archive is enabled and writable.

        Returns False if:
        - Archive was initialized with enabled=False
        - Storage path is not writable (cached after first check)
        """
        if not self._enabled:
            return False
        if self._writable is False:
            return False
        return True

    @property
    def warnings(self) -> list[str]:
        """Get warnings collected during archive operations."""
        return self._warnings.copy()

    def clear_warnings(self) -> None:
        """Clear collected warnings."""
        self._warnings.clear()

    def enable(self) -> bool:
        """Enable archival and check writability.

        Performs startup capability check. If storage path is not
        writable, caches disabled state and emits warning.

        Returns:
            True if archive is now enabled and writable
        """
        self._enabled = True
        self._ensure_directory()
        return self._check_writable()

    def disable(self) -> None:
        """Disable archival."""
        self._enabled = False

    def _check_writable(self) -> bool:
        """Check if storage path is writable.

        Performs a test write to verify the archive directory is
        accessible. Caches the result to avoid repeated checks.

        Returns:
            True if writable, False otherwise
        """
        if self._writable is not None:
            return self._writable

        test_file = self.storage_path / ".write_test"
        try:
            test_file.write_text("test")
            test_file.unlink()
            self._writable = True
            logger.debug("Archive storage is writable: %s", self.storage_path)
            return True
        except OSError as e:
            self._writable = False
            warning = f"{ARCHIVE_WRITE_FAILED}: Storage path not writable: {self.storage_path} ({e})"
            self._warnings.append(warning)
            logger.warning(warning)
            return False

    def _ensure_directory(self) -> None:
        """Create storage directory with private permissions if needed.

        On failure, disables archival and emits ARCHIVE_WRITE_FAILED warning.
        """
        try:
            if not self.storage_path.exists():
                self.storage_path.mkdir(parents=True, exist_ok=True)
                # Set directory to owner-only access (0o700)
                try:
                    os.chmod(self.storage_path, stat.S_IRWXU)
                    logger.debug(
                        "Created archive directory with private permissions: %s",
                        self.storage_path,
                    )
                except OSError as e:
                    logger.warning(
                        "Could not set directory permissions on %s: %s",
                        self.storage_path,
                        e,
                    )
        except OSError as e:
            self._writable = False
            warning = f"{ARCHIVE_WRITE_FAILED}: Could not create archive directory: {self.storage_path} ({e})"
            self._warnings.append(warning)
            logger.warning(warning)

    def _get_file_path(self, content_hash: str) -> Path:
        """Get file path for a content hash.

        Args:
            content_hash: SHA256 hash of the content

        Returns:
            Path to the archive file
        """
        # Validate hash format (hex string, 64 chars for SHA256)
        if not (len(content_hash) == 64 and all(c in "0123456789abcdef" for c in content_hash.lower())):
            # Sanitize invalid hashes to prevent path traversal
            safe_hash = "".join(c for c in content_hash if c.isalnum())[:64]
            content_hash = safe_hash or "invalid"

        return self.storage_path / f"{content_hash}.json"

    def _get_lock_path(self, content_hash: str) -> Path:
        """Get lock file path for a content hash."""
        return self._get_file_path(content_hash).with_suffix(".lock")

    def _is_expired(self, file_path: Path) -> bool:
        """Check if an archive file has expired based on TTL.

        Args:
            file_path: Path to check

        Returns:
            True if expired, False otherwise
        """
        try:
            mtime = datetime.fromtimestamp(
                file_path.stat().st_mtime,
                tz=timezone.utc,
            )
            expiry = mtime + timedelta(hours=self.ttl_hours)
            return datetime.now(timezone.utc) > expiry
        except OSError:
            return True

    def archive(
        self,
        content: str,
        item_id: str,
        reason: str = "",
        item_type: str = "source",
        original_tokens: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[ArchivedContent]:
        """Archive content to file storage.

        Uses SHA256 hash of content as filename for deduplication.
        If content already exists, updates metadata but preserves content.

        If archival is disabled or storage is not writable, returns None
        and collects an ARCHIVE_WRITE_FAILED warning.

        Args:
            content: Text content to archive
            item_id: ID of the item this content belongs to
            reason: Why the content is being archived
            item_type: Type of content ("source", "finding", "gap")
            original_tokens: Token count of the content
            metadata: Additional metadata to store

        Returns:
            ArchivedContent record with the content hash, or None if disabled/failed
        """
        # Check if archival is enabled
        if not self.enabled:
            logger.debug(
                "Archive disabled, skipping content %s for item %s",
                compute_content_hash(content)[:12],
                item_id,
            )
            return None

        content_hash = compute_content_hash(content)
        file_path = self._get_file_path(content_hash)
        lock_path = self._get_lock_path(content_hash)

        record = ArchivedContent(
            content_hash=content_hash,
            content=content,
            item_id=item_id,
            item_type=item_type,
            archive_reason=reason,
            original_tokens=original_tokens,
            metadata=metadata or {},
        )

        try:
            with FileLock(lock_path, timeout=10):
                # Check for existing record (deduplication)
                if file_path.exists():
                    try:
                        existing_data = json.loads(file_path.read_text())
                        existing = ArchivedContent.from_dict(existing_data)
                        # Preserve original archived_at, update metadata
                        record.archived_at = existing.archived_at
                        logger.debug(
                            "Content %s already archived, updating metadata",
                            content_hash[:12],
                        )
                    except (json.JSONDecodeError, KeyError):
                        # Overwrite corrupt file - log with warning code
                        logger.warning(
                            "%s: Corrupt archive file %s, overwriting",
                            ARCHIVE_READ_CORRUPT,
                            content_hash[:12],
                        )

                # Atomic write: temp file + rename
                # Write to temp file in same directory (ensures same filesystem)
                fd, temp_path = tempfile.mkstemp(
                    suffix=".tmp",
                    prefix=f".{content_hash[:12]}_",
                    dir=self.storage_path,
                )
                try:
                    # Write content to temp file
                    with os.fdopen(fd, "w") as f:
                        json.dump(record.to_dict(), f, indent=2, default=str)

                    # Set file permissions before rename (0o600)
                    try:
                        os.chmod(temp_path, stat.S_IRUSR | stat.S_IWUSR)
                    except OSError:
                        pass  # Best effort on permissions

                    # Atomic rename to target path
                    os.rename(temp_path, file_path)
                except BaseException:
                    # Clean up temp file on any failure
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass
                    raise

                logger.debug(
                    "Archived content %s for item %s (%s)",
                    content_hash[:12],
                    item_id,
                    reason,
                )

            return record

        except OSError as e:
            # Write failure - emit warning and cache disabled state
            self._writable = False
            warning = f"{ARCHIVE_WRITE_FAILED}: Failed to archive content {content_hash[:12]} for item {item_id}: {e}"
            self._warnings.append(warning)
            logger.warning(warning)
            return None

    def retrieve(self, content_hash: str) -> Optional[ArchivedContent]:
        """Retrieve archived content by hash.

        Args:
            content_hash: SHA256 hash of the content

        Returns:
            ArchivedContent if found and not expired, None otherwise
        """
        file_path = self._get_file_path(content_hash)
        lock_path = self._get_lock_path(content_hash)

        if not file_path.exists():
            return None

        if self._is_expired(file_path):
            logger.debug("Archive %s has expired, removing", content_hash[:12])
            self._delete_file(content_hash)
            return None

        with FileLock(lock_path, timeout=10):
            try:
                data = json.loads(file_path.read_text())
                return ArchivedContent.from_dict(data)
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                # Skip-on-corruption policy: log warning and return None
                logger.warning(
                    "%s: Failed to load archived content %s, skipping: %s",
                    ARCHIVE_READ_CORRUPT,
                    content_hash[:12],
                    exc,
                )
                return None

    def retrieve_by_item_id(self, item_id: str) -> list[ArchivedContent]:
        """Retrieve all archived content for a specific item.

        Scans all archive files to find content matching the item ID.
        Less efficient than hash-based retrieval, use sparingly.

        Args:
            item_id: ID of the item to find

        Returns:
            List of matching ArchivedContent records
        """
        results = []

        for file_path in self.storage_path.glob("*.json"):
            if self._is_expired(file_path):
                continue

            try:
                with FileLock(file_path.with_suffix(".lock"), timeout=5):
                    data = json.loads(file_path.read_text())
                    if data.get("item_id") == item_id:
                        results.append(ArchivedContent.from_dict(data))
            except (json.JSONDecodeError, KeyError) as exc:
                # Skip-on-corruption policy: log warning and continue
                logger.warning(
                    "%s: Corrupt archive file %s, skipping: %s",
                    ARCHIVE_READ_CORRUPT,
                    file_path.stem[:12],
                    exc,
                )
                continue
            except TimeoutError:
                continue

        return results

    def _delete_file(self, content_hash: str) -> bool:
        """Delete an archive file and its lock.

        Args:
            content_hash: Hash of the content to delete

        Returns:
            True if deleted, False otherwise
        """
        file_path = self._get_file_path(content_hash)
        lock_path = self._get_lock_path(content_hash)

        if not file_path.exists():
            return False

        with FileLock(lock_path, timeout=10):
            try:
                file_path.unlink()
                if lock_path.exists():
                    lock_path.unlink()
                return True
            except OSError as exc:
                logger.warning(
                    "Failed to delete archive %s: %s",
                    content_hash[:12],
                    exc,
                )
                return False

    def delete(self, content_hash: str) -> bool:
        """Delete archived content by hash.

        Args:
            content_hash: SHA256 hash of the content to delete

        Returns:
            True if deleted, False if not found
        """
        return self._delete_file(content_hash)

    def cleanup_expired(self) -> int:
        """Remove all expired archive entries.

        Scans the archive directory and removes files older than TTL.

        Returns:
            Number of entries removed
        """
        removed = 0

        for file_path in self.storage_path.glob("*.json"):
            if self._is_expired(file_path):
                content_hash = file_path.stem
                if self._delete_file(content_hash):
                    removed += 1
                    logger.debug("Cleaned up expired archive: %s", content_hash[:12])

        if removed > 0:
            logger.info("Cleaned up %d expired archive entries", removed)

        return removed

    def list_hashes(self) -> list[str]:
        """List all content hashes in the archive.

        Returns:
            List of content hashes (excluding expired entries)
        """
        hashes = []

        for file_path in self.storage_path.glob("*.json"):
            if not self._is_expired(file_path):
                hashes.append(file_path.stem)

        return sorted(hashes)

    def get_stats(self) -> dict[str, Any]:
        """Get archive statistics.

        Returns:
            Dict with count, total_size, oldest, newest timestamps
        """
        count = 0
        total_size = 0
        oldest: Optional[datetime] = None
        newest: Optional[datetime] = None

        for file_path in self.storage_path.glob("*.json"):
            if self._is_expired(file_path):
                continue

            count += 1
            total_size += file_path.stat().st_size

            mtime = datetime.fromtimestamp(
                file_path.stat().st_mtime,
                tz=timezone.utc,
            )
            if oldest is None or mtime < oldest:
                oldest = mtime
            if newest is None or mtime > newest:
                newest = mtime

        return {
            "enabled": self.enabled,
            "writable": self._writable,
            "count": count,
            "total_size_bytes": total_size,
            "oldest": oldest.isoformat() if oldest else None,
            "newest": newest.isoformat() if newest else None,
            "ttl_hours": self.ttl_hours,
            "storage_path": str(self.storage_path),
            "warnings": self._warnings.copy(),
        }
