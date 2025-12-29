"""
Intake storage backend for the bikelane fast-capture system.

Provides thread-safe JSONL-based storage for intake items with file locking
for concurrent access. Items are stored in specs/.bikelane/intake.jsonl.

Storage Structure:
    specs/.bikelane/
        intake.jsonl      - Append-only intake log
        .intake.lock      - Lock file for cross-process synchronization
        intake.YYYY-MM.jsonl  - Archived intake files (after rotation)
"""

from __future__ import annotations

import base64
import fcntl
import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from foundry_mcp.core.security import is_prompt_injection

logger = logging.getLogger(__name__)

# Security constants
LOG_DESCRIPTION_MAX_LENGTH = 100  # Max chars for descriptions in logs

# Constants
LOCK_TIMEOUT_SECONDS = 5.0
ROTATION_ITEM_THRESHOLD = 1000
ROTATION_SIZE_THRESHOLD = 1024 * 1024  # 1MB
IDEMPOTENCY_SCAN_LIMIT = 100
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200

# Validation patterns
INTAKE_ID_PATTERN = re.compile(r"^intake-[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$")
TAG_PATTERN = re.compile(r"^[a-z0-9_-]+$")


@dataclass
class IntakeItem:
    """
    Represents a single intake item in the bikelane queue.

    Attributes:
        schema_version: Fixed value 'intake-v1'
        id: Unique identifier in format 'intake-<uuid4>'
        title: Brief title (1-140 chars)
        status: 'new' or 'dismissed'
        created_at: ISO 8601 UTC timestamp
        updated_at: ISO 8601 UTC timestamp
        description: Optional detailed description (max 2000 chars)
        priority: Priority level p0-p4 (default p2)
        tags: List of lowercase tags (max 20, each 1-32 chars)
        source: Origin of the intake item (max 100 chars)
        requester: Person/entity who requested (max 100 chars)
        idempotency_key: Optional client key for deduplication (max 64 chars)
    """
    schema_version: str = "intake-v1"
    id: str = ""
    title: str = ""
    status: str = "new"
    created_at: str = ""
    updated_at: str = ""
    description: Optional[str] = None
    priority: str = "p2"
    tags: list[str] = field(default_factory=list)
    source: Optional[str] = None
    requester: Optional[str] = None
    idempotency_key: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values for optional fields."""
        result = {
            "schema_version": self.schema_version,
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "priority": self.priority,
            "tags": self.tags,
        }
        if self.description is not None:
            result["description"] = self.description
        if self.source is not None:
            result["source"] = self.source
        if self.requester is not None:
            result["requester"] = self.requester
        if self.idempotency_key is not None:
            result["idempotency_key"] = self.idempotency_key
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IntakeItem":
        """Create an IntakeItem from a dictionary."""
        return cls(
            schema_version=data.get("schema_version", "intake-v1"),
            id=data.get("id", ""),
            title=data.get("title", ""),
            status=data.get("status", "new"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            description=data.get("description"),
            priority=data.get("priority", "p2"),
            tags=data.get("tags", []),
            source=data.get("source"),
            requester=data.get("requester"),
            idempotency_key=data.get("idempotency_key"),
        )


@dataclass
class PaginationCursor:
    """Cursor for pagination through intake items."""
    last_id: str
    line_hint: int
    version: int = 1

    def encode(self) -> str:
        """Encode cursor to base64 string."""
        payload = {
            "last_id": self.last_id,
            "line_hint": self.line_hint,
            "version": self.version,
        }
        json_bytes = json.dumps(payload).encode("utf-8")
        return base64.b64encode(json_bytes).decode("ascii")

    @classmethod
    def decode(cls, encoded: str) -> Optional["PaginationCursor"]:
        """Decode cursor from base64 string. Returns None if invalid."""
        try:
            json_bytes = base64.b64decode(encoded.encode("ascii"))
            payload = json.loads(json_bytes.decode("utf-8"))
            return cls(
                last_id=payload.get("last_id", ""),
                line_hint=payload.get("line_hint", 0),
                version=payload.get("version", 1),
            )
        except (ValueError, json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to decode pagination cursor: {e}")
            return None


class LockAcquisitionError(Exception):
    """Raised when file lock cannot be acquired within timeout."""
    pass


class IntakeStore:
    """
    JSONL-based intake storage with thread-safe file locking.

    Provides append-only storage for intake items with:
    - fcntl file locking for cross-process safety
    - threading.Lock for in-memory thread safety
    - Atomic writes via temp file rename pattern
    - Cursor-based pagination with line hints
    - File rotation when thresholds are exceeded

    Directory structure:
        specs/.bikelane/
            intake.jsonl       - Active intake items
            .intake.lock       - Lock file for synchronization
            intake.YYYY-MM.jsonl - Archived files (rotated)
    """

    def __init__(
        self,
        specs_dir: str | Path,
        workspace_root: Optional[str | Path] = None,
        bikelane_dir: Optional[str | Path] = None,
    ):
        """
        Initialize the intake store.

        Args:
            specs_dir: Path to the specs directory (e.g., /workspace/specs)
            workspace_root: Optional workspace root for path validation.
                           Defaults to specs_dir parent.
            bikelane_dir: Optional custom path for bikelane storage.
                         Defaults to specs_dir/.bikelane if not provided.

        Raises:
            ValueError: If specs_dir or bikelane_dir is outside the workspace root
                       (path traversal attempt)
        """
        self.specs_dir = Path(specs_dir).resolve()
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else self.specs_dir.parent

        # Validate specs_dir is within workspace (prevent path traversal)
        if not self._validate_path_in_workspace(self.specs_dir):
            raise ValueError(
                f"specs_dir '{specs_dir}' is outside workspace root '{self.workspace_root}'"
            )

        # Use custom bikelane_dir if provided, otherwise default to specs/.bikelane
        if bikelane_dir is not None:
            self.bikelane_dir = Path(bikelane_dir).resolve()
            # Validate bikelane_dir is within workspace
            if not self._validate_path_in_workspace(self.bikelane_dir):
                raise ValueError(
                    f"bikelane_dir '{bikelane_dir}' is outside workspace root '{self.workspace_root}'"
                )
        else:
            self.bikelane_dir = self.specs_dir / ".bikelane"

        self.intake_file = self.bikelane_dir / "intake.jsonl"
        self.lock_file = self.bikelane_dir / ".intake.lock"

        self._thread_lock = threading.Lock()
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Ensure the bikelane directory exists."""
        self.bikelane_dir.mkdir(parents=True, exist_ok=True)

    def _validate_path_in_workspace(self, path: Path) -> bool:
        """
        Validate that a path is within the workspace sandbox.

        Uses resolve() + relative_to() pattern to prevent path traversal attacks
        via symlinks or .. components.

        Args:
            path: Path to validate (will be resolved)

        Returns:
            True if path is within workspace_root, False otherwise
        """
        try:
            resolved = path.resolve()
            resolved.relative_to(self.workspace_root)
            return True
        except ValueError:
            logger.warning(
                f"Path traversal attempt blocked: '{path}' is outside workspace"
            )
            return False

    def _acquire_lock(self, exclusive: bool = True, timeout: float = LOCK_TIMEOUT_SECONDS) -> int:
        """
        Acquire file lock with timeout.

        Args:
            exclusive: If True, acquire exclusive lock (for writes).
                      If False, acquire shared lock (for reads).
            timeout: Maximum time to wait for lock in seconds.

        Returns:
            File descriptor for the lock file.

        Raises:
            LockAcquisitionError: If lock cannot be acquired within timeout.
        """
        lock_type = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH

        # Ensure lock file exists
        self.lock_file.touch(exist_ok=True)

        fd = os.open(str(self.lock_file), os.O_RDWR | os.O_CREAT)

        start_time = time.monotonic()
        while True:
            try:
                fcntl.flock(fd, lock_type | fcntl.LOCK_NB)
                return fd
            except (IOError, OSError):
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    os.close(fd)
                    raise LockAcquisitionError(
                        f"Failed to acquire {'exclusive' if exclusive else 'shared'} lock "
                        f"within {timeout} seconds"
                    )
                time.sleep(0.01)  # 10ms between retries

    def _release_lock(self, fd: int) -> None:
        """Release file lock and close file descriptor."""
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def _generate_id(self) -> str:
        """Generate a new intake ID."""
        return f"intake-{uuid.uuid4()}"

    def _now_utc(self) -> str:
        """Get current UTC timestamp in ISO 8601 format with Z suffix."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    def _strip_control_chars(self, text: str) -> str:
        """Strip control characters from text for security."""
        if not text:
            return text
        # Remove ASCII control characters (0x00-0x1F) except common whitespace
        return "".join(
            char for char in text
            if ord(char) >= 32 or char in "\n\r\t"
        )

    def _truncate_for_log(self, text: str, max_length: int = LOG_DESCRIPTION_MAX_LENGTH) -> str:
        """
        Truncate text for safe logging.

        Args:
            text: Text to truncate
            max_length: Maximum length (default: LOG_DESCRIPTION_MAX_LENGTH)

        Returns:
            Truncated text with ellipsis if needed
        """
        if not text or len(text) <= max_length:
            return text or ""
        return text[:max_length] + "..."

    def _sanitize_for_injection(self, text: str, field_name: str) -> str:
        """
        Check text for prompt injection patterns and sanitize if found.

        Args:
            text: Text to check
            field_name: Name of field for logging

        Returns:
            Original text if clean, sanitized text if injection detected
        """
        if not text:
            return text

        if is_prompt_injection(text):
            logger.warning(
                f"Prompt injection pattern detected in {field_name}, "
                f"input sanitized (preview: {self._truncate_for_log(text, 50)})"
            )
            # Replace known injection patterns with safe placeholder
            # We strip the offending patterns rather than rejecting entirely
            # to maintain usability while preventing the attack
            sanitized = re.sub(
                r"ignore\s+(all\s+)?(previous|prior)\s+(instructions?|prompts?)",
                "[SANITIZED]",
                text,
                flags=re.IGNORECASE,
            )
            sanitized = re.sub(
                r"disregard\s+(all\s+)?(previous|prior|above)",
                "[SANITIZED]",
                sanitized,
                flags=re.IGNORECASE,
            )
            sanitized = re.sub(
                r"system\s*:\s*",
                "[SANITIZED]",
                sanitized,
                flags=re.IGNORECASE,
            )
            sanitized = re.sub(
                r"<\s*system\s*>",
                "[SANITIZED]",
                sanitized,
                flags=re.IGNORECASE,
            )
            return sanitized

        return text

    def _normalize_tags(self, tags: list[str]) -> list[str]:
        """Normalize tags to lowercase and validate format."""
        normalized = []
        for tag in tags:
            tag_lower = tag.lower().strip()
            if tag_lower and TAG_PATTERN.match(tag_lower):
                normalized.append(tag_lower)
        return normalized

    def _count_items(self) -> int:
        """Count total items in the intake file."""
        if not self.intake_file.exists():
            return 0
        try:
            with open(self.intake_file, "r") as f:
                return sum(1 for line in f if line.strip())
        except OSError:
            return 0

    def _get_file_size(self) -> int:
        """Get size of the intake file in bytes."""
        if not self.intake_file.exists():
            return 0
        try:
            return self.intake_file.stat().st_size
        except OSError:
            return 0

    def _should_rotate(self) -> bool:
        """Check if file rotation is needed."""
        item_count = self._count_items()
        file_size = self._get_file_size()
        return item_count >= ROTATION_ITEM_THRESHOLD or file_size >= ROTATION_SIZE_THRESHOLD

    def _get_oldest_item_date(self) -> Optional[str]:
        """Get the created_at date of the oldest item for archive naming."""
        if not self.intake_file.exists():
            return None
        try:
            with open(self.intake_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            return item.get("created_at", "")[:7]  # YYYY-MM
                        except json.JSONDecodeError:
                            continue
        except OSError:
            pass
        return None

    def rotate_if_needed(self) -> Optional[str]:
        """
        Rotate the intake file if thresholds are exceeded.

        Returns:
            Path to archive file if rotation occurred, None otherwise.
        """
        with self._thread_lock:
            if not self._should_rotate():
                return None

            # Get date for archive naming
            date_str = self._get_oldest_item_date()
            if not date_str:
                date_str = datetime.now(timezone.utc).strftime("%Y-%m")

            # Generate unique archive name
            archive_name = f"intake.{date_str}.jsonl"
            archive_path = self.bikelane_dir / archive_name

            # If archive already exists, add a suffix
            counter = 1
            while archive_path.exists():
                archive_name = f"intake.{date_str}.{counter}.jsonl"
                archive_path = self.bikelane_dir / archive_name
                counter += 1

            fd = None
            try:
                fd = self._acquire_lock(exclusive=True)

                # Rename current file to archive
                if self.intake_file.exists():
                    self.intake_file.rename(archive_path)
                    logger.info(f"Rotated intake file to {archive_path}")

                    # Create fresh intake file
                    self.intake_file.touch()

                    return str(archive_path)

            except LockAcquisitionError:
                logger.error("Failed to acquire lock for file rotation")
                raise
            finally:
                if fd is not None:
                    self._release_lock(fd)

            return None

    def add(
        self,
        *,
        title: str,
        description: Optional[str] = None,
        priority: str = "p2",
        tags: Optional[list[str]] = None,
        source: Optional[str] = None,
        requester: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        dry_run: bool = False,
    ) -> tuple[IntakeItem, bool, float]:
        """
        Add a new intake item.

        Args:
            title: Item title (1-140 chars, required)
            description: Detailed description (max 2000 chars)
            priority: Priority level p0-p4 (default p2)
            tags: List of tags (max 20, normalized to lowercase)
            source: Origin of the item (max 100 chars)
            requester: Who requested the work (max 100 chars)
            idempotency_key: Key for deduplication (max 64 chars)
            dry_run: If True, validate but don't persist

        Returns:
            Tuple of (created IntakeItem, was_duplicate, lock_wait_ms)

        Raises:
            LockAcquisitionError: If lock cannot be acquired within timeout.
        """
        now = self._now_utc()
        lock_wait_ms = 0.0

        # Sanitize inputs - strip control characters
        title = self._strip_control_chars(title)
        if description:
            description = self._strip_control_chars(description)
        if source:
            source = self._strip_control_chars(source)
        if requester:
            requester = self._strip_control_chars(requester)

        # Sanitize title and description for prompt injection patterns
        title = self._sanitize_for_injection(title, "title")
        if description:
            description = self._sanitize_for_injection(description, "description")

        # Normalize tags
        normalized_tags = self._normalize_tags(tags or [])

        # Create item
        item = IntakeItem(
            id=self._generate_id(),
            title=title,
            description=description,
            status="new",
            priority=priority,
            tags=normalized_tags,
            source=source,
            requester=requester,
            idempotency_key=idempotency_key,
            created_at=now,
            updated_at=now,
        )

        if dry_run:
            return item, False, lock_wait_ms

        with self._thread_lock:
            fd = None
            lock_start = time.monotonic()

            try:
                fd = self._acquire_lock(exclusive=True)
                lock_wait_ms = (time.monotonic() - lock_start) * 1000

                # Check for idempotency key duplicate
                if idempotency_key:
                    existing = self._find_by_idempotency_key(idempotency_key)
                    if existing:
                        return existing, True, lock_wait_ms

                # Check rotation threshold
                if self._should_rotate():
                    self._release_lock(fd)
                    fd = None
                    self.rotate_if_needed()
                    fd = self._acquire_lock(exclusive=True)

                # Append to file
                with open(self.intake_file, "a") as f:
                    f.write(json.dumps(item.to_dict()) + "\n")
                    f.flush()

                logger.info(
                    f"Added intake item: {item.id} "
                    f"(title: {self._truncate_for_log(item.title)})"
                )
                return item, False, lock_wait_ms

            finally:
                if fd is not None:
                    self._release_lock(fd)

    def _find_by_idempotency_key(self, key: str) -> Optional[IntakeItem]:
        """
        Search for an item by idempotency key in the last N items.

        Note: This should be called while holding the lock.
        """
        if not self.intake_file.exists():
            return None

        # Read last IDEMPOTENCY_SCAN_LIMIT lines
        try:
            with open(self.intake_file, "r") as f:
                lines = f.readlines()

            # Check last N items in reverse order
            for line in reversed(lines[-IDEMPOTENCY_SCAN_LIMIT:]):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("idempotency_key") == key:
                        return IntakeItem.from_dict(data)
                except json.JSONDecodeError:
                    continue

        except OSError as e:
            logger.warning(f"Error reading intake file for idempotency check: {e}")

        return None

    def list_new(
        self,
        *,
        cursor: Optional[str] = None,
        limit: int = DEFAULT_PAGE_SIZE,
    ) -> tuple[list[IntakeItem], int, Optional[str], bool, float]:
        """
        List intake items with status='new' in FIFO order.

        Args:
            cursor: Pagination cursor from previous call
            limit: Maximum items to return (1-200)

        Returns:
            Tuple of (items, total_count, next_cursor, has_more, lock_wait_ms)

        Note: This operation is O(n) where n = total items in file.
              File rotation bounds n to ~1000 items.
        """
        limit = max(1, min(limit, MAX_PAGE_SIZE))
        lock_wait_ms = 0.0

        with self._thread_lock:
            fd = None
            lock_start = time.monotonic()

            try:
                fd = self._acquire_lock(exclusive=False)  # Shared lock for reads
                lock_wait_ms = (time.monotonic() - lock_start) * 1000

                if not self.intake_file.exists():
                    return [], 0, None, False, lock_wait_ms

                # Parse cursor if provided
                parsed_cursor: Optional[PaginationCursor] = None
                if cursor:
                    parsed_cursor = PaginationCursor.decode(cursor)
                    if parsed_cursor is None:
                        logger.warning("Invalid cursor provided, starting from beginning")

                items: list[IntakeItem] = []
                total_count = 0
                found_cursor_position = parsed_cursor is None  # True if no cursor
                line_num = 0
                cursor_fallback = False

                with open(self.intake_file, "r") as f:
                    # Try to seek to line hint if cursor provided
                    if parsed_cursor and parsed_cursor.line_hint > 0:
                        # Read up to line_hint to find position
                        for i in range(parsed_cursor.line_hint):
                            line = f.readline()
                            if not line:
                                break
                            line_num += 1
                            # Still need to count new items for total
                            line = line.strip()
                            if line:
                                try:
                                    data = json.loads(line)
                                    if data.get("status") == "new":
                                        total_count += 1
                                except json.JSONDecodeError:
                                    pass

                        # Check if we found the cursor position
                        line = f.readline()
                        if line:
                            try:
                                data = json.loads(line.strip())
                                if data.get("id") != parsed_cursor.last_id:
                                    # Line hint didn't match, need full scan
                                    cursor_fallback = True
                                    logger.warning(
                                        f"Cursor line hint mismatch, falling back to full scan"
                                    )
                                    f.seek(0)
                                    line_num = 0
                                    total_count = 0
                                else:
                                    # Found cursor position, count this item
                                    if data.get("status") == "new":
                                        total_count += 1
                                    found_cursor_position = True
                                    line_num += 1
                            except json.JSONDecodeError:
                                cursor_fallback = True
                                f.seek(0)
                                line_num = 0
                                total_count = 0

                    # Read remaining lines
                    for line in f:
                        current_line = line_num
                        line_num += 1
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        # Only count 'new' items
                        if data.get("status") != "new":
                            continue

                        total_count += 1

                        # Handle cursor position finding
                        if not found_cursor_position:
                            if data.get("id") == parsed_cursor.last_id:
                                found_cursor_position = True
                            continue

                        # Collect items up to limit
                        if len(items) < limit:
                            items.append(IntakeItem.from_dict(data))

                # Build next cursor
                next_cursor: Optional[str] = None
                has_more = len(items) == limit and total_count > len(items)

                if items and has_more:
                    last_item = items[-1]
                    # Estimate line position (not perfect but helpful)
                    next_cursor = PaginationCursor(
                        last_id=last_item.id,
                        line_hint=line_num - 1,
                    ).encode()

                if cursor_fallback and lock_wait_ms > 1000:
                    logger.warning(f"Cursor fallback with long lock wait: {lock_wait_ms:.2f}ms")

                return items, total_count, next_cursor, has_more, lock_wait_ms

            finally:
                if fd is not None:
                    self._release_lock(fd)

    def dismiss(
        self,
        intake_id: str,
        *,
        reason: Optional[str] = None,
        dry_run: bool = False,
    ) -> tuple[Optional[IntakeItem], float]:
        """
        Dismiss an intake item by changing its status to 'dismissed'.

        Args:
            intake_id: The intake item ID to dismiss
            reason: Optional reason for dismissal (max 200 chars)
            dry_run: If True, find item but don't modify

        Returns:
            Tuple of (updated IntakeItem or None if not found, lock_wait_ms)

        Raises:
            LockAcquisitionError: If lock cannot be acquired within timeout.
        """
        lock_wait_ms = 0.0
        now = self._now_utc()

        # Sanitize reason
        if reason:
            reason = self._strip_control_chars(reason)

        with self._thread_lock:
            fd = None
            lock_start = time.monotonic()

            try:
                fd = self._acquire_lock(exclusive=True)
                lock_wait_ms = (time.monotonic() - lock_start) * 1000

                if not self.intake_file.exists():
                    return None, lock_wait_ms

                # Read all lines
                with open(self.intake_file, "r") as f:
                    lines = f.readlines()

                # Find and update the item
                found_item: Optional[IntakeItem] = None
                updated_lines: list[str] = []

                for line in lines:
                    stripped = line.strip()
                    if not stripped:
                        updated_lines.append(line)
                        continue

                    try:
                        data = json.loads(stripped)
                    except json.JSONDecodeError:
                        updated_lines.append(line)
                        continue

                    if data.get("id") == intake_id:
                        # Found the item
                        data["status"] = "dismissed"
                        data["updated_at"] = now
                        found_item = IntakeItem.from_dict(data)
                        updated_lines.append(json.dumps(data) + "\n")
                    else:
                        updated_lines.append(line)

                if found_item is None:
                    return None, lock_wait_ms

                if dry_run:
                    return found_item, lock_wait_ms

                # Atomic write via temp file
                temp_file = self.intake_file.with_suffix(".tmp")
                with open(temp_file, "w") as f:
                    f.writelines(updated_lines)
                    f.flush()
                    os.fsync(f.fileno())

                temp_file.rename(self.intake_file)

                logger.info(f"Dismissed intake item: {intake_id}")
                return found_item, lock_wait_ms

            finally:
                if fd is not None:
                    self._release_lock(fd)

    def get(self, intake_id: str) -> tuple[Optional[IntakeItem], float]:
        """
        Get a single intake item by ID.

        Args:
            intake_id: The intake item ID to retrieve

        Returns:
            Tuple of (IntakeItem or None if not found, lock_wait_ms)
        """
        lock_wait_ms = 0.0

        with self._thread_lock:
            fd = None
            lock_start = time.monotonic()

            try:
                fd = self._acquire_lock(exclusive=False)
                lock_wait_ms = (time.monotonic() - lock_start) * 1000

                if not self.intake_file.exists():
                    return None, lock_wait_ms

                with open(self.intake_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            if data.get("id") == intake_id:
                                return IntakeItem.from_dict(data), lock_wait_ms
                        except json.JSONDecodeError:
                            continue

                return None, lock_wait_ms

            finally:
                if fd is not None:
                    self._release_lock(fd)

    @property
    def intake_path(self) -> str:
        """Get the absolute path to the intake file."""
        return str(self.intake_file.absolute())


# Global store instance
_intake_store: Optional[IntakeStore] = None
_store_lock = threading.Lock()


def get_intake_store(
    specs_dir: Optional[str | Path] = None,
    bikelane_dir: Optional[str | Path] = None,
) -> IntakeStore:
    """
    Get the global intake store instance.

    Args:
        specs_dir: Path to specs directory. Required on first call.
        bikelane_dir: Optional custom path for bikelane storage.
                     Defaults to specs_dir/.bikelane if not provided.

    Returns:
        The IntakeStore instance.

    Raises:
        ValueError: If specs_dir not provided on first call.
    """
    global _intake_store

    with _store_lock:
        if _intake_store is None:
            if specs_dir is None:
                raise ValueError("specs_dir required for first IntakeStore initialization")
            _intake_store = IntakeStore(specs_dir, bikelane_dir=bikelane_dir)

        return _intake_store


def reset_intake_store() -> None:
    """Reset the global intake store (for testing)."""
    global _intake_store
    with _store_lock:
        _intake_store = None
