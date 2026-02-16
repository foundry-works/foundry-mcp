"""File-based storage backend for autonomous session state.

Provides thread-safe persistence for autonomous execution sessions with:
- Atomic writes (temp+fsync+rename)
- File locking with timeout
- Per-spec pointer files for active session lookup
- Cursor-based pagination
- Garbage collection with TTL
"""

import base64
import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from filelock import FileLock, Timeout

from .models import (
    AutonomousSessionState,
    SessionStatus,
    SessionSummary,
    TERMINAL_STATUSES,
)
from .state_migrations import (
    CURRENT_SCHEMA_VERSION,
    migrate_state,
)

logger = logging.getLogger(__name__)

# Default storage paths
DEFAULT_WORKSPACE_SESSIONS_PATH = Path("specs") / ".autonomy" / "sessions"
DEFAULT_FALLBACK_PATH = Path.home() / ".foundry-mcp" / "autonomy" / "sessions"

# Lock acquisition timeout (seconds)
LOCK_ACQUISITION_TIMEOUT = 5

# GC TTLs by session status (days)
GC_TTL_DAYS: Dict[SessionStatus, int] = {
    SessionStatus.COMPLETED: 7,
    SessionStatus.ENDED: 7,
    SessionStatus.FAILED: 30,
}

class ActiveSessionLookupResult(Enum):
    """Result of active session lookup."""

    FOUND = "found"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"


@dataclass
class ListSessionsResult:
    """Result of listing sessions with pagination."""

    sessions: List[SessionSummary]
    cursor: Optional[str]
    has_more: bool
    total_count: Optional[int]


def sanitize_id(item_id: str) -> str:
    """Sanitize ID to prevent path traversal attacks.

    Args:
        item_id: Raw identifier

    Returns:
        Sanitized identifier safe for filesystem use
    """
    # Only allow alphanumeric, hyphens, underscores
    return "".join(c for c in item_id if c.isalnum() or c in "-_")


def encode_cursor(
    schema_version: int,
    last_updated_at: str,
    last_session_id: str,
    filters_hash: str,
) -> str:
    """Encode pagination cursor.

    Args:
        schema_version: Cursor schema version
        last_updated_at: ISO timestamp of last item
        last_session_id: Session ID of last item
        filters_hash: Hash of applied filters

    Returns:
        Base64-encoded opaque cursor
    """
    cursor_data = json.dumps({
        "v": schema_version,
        "u": last_updated_at,
        "s": last_session_id,
        "f": filters_hash,
    }, separators=(",", ":"))
    return base64.urlsafe_b64encode(cursor_data.encode()).decode()


def decode_cursor(cursor: str) -> Tuple[int, str, str, str]:
    """Decode pagination cursor.

    Args:
        cursor: Base64-encoded cursor

    Returns:
        Tuple of (schema_version, last_updated_at, last_session_id, filters_hash)

    Raises:
        ValueError: If cursor is invalid or malformed
    """
    try:
        cursor_data = json.loads(base64.urlsafe_b64decode(cursor.encode()))
        return (
            cursor_data["v"],
            cursor_data["u"],
            cursor_data["s"],
            cursor_data["f"],
        )
    except (json.JSONDecodeError, KeyError, base64.binascii.Error) as e:
        raise ValueError(f"Invalid cursor: {e}") from e


def compute_filters_hash(
    status_filter: Optional[str] = None,
    spec_id: Optional[str] = None,
) -> str:
    """Compute hash of filter parameters for cursor validation.

    Args:
        status_filter: Status filter value
        spec_id: Spec ID filter value

    Returns:
        Short hash of filters
    """
    filter_data = json.dumps({
        "status": status_filter,
        "spec_id": spec_id,
    }, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(filter_data.encode()).hexdigest()[:8]


class AutonomyStorage:
    """File-based storage for autonomous session states.

    Provides CRUD operations with atomic writes, file locking, and GC.
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        workspace_path: Optional[Path] = None,
    ) -> None:
        """Initialize storage backend.

        Args:
            storage_path: Directory to store session files (default: workspace or fallback)
            workspace_path: Workspace root for relative paths
        """
        self.workspace_path = workspace_path or Path.cwd()

        if storage_path is not None:
            self.storage_path = storage_path
        else:
            # Try workspace path first, fall back to home directory
            workspace_sessions = self.workspace_path / DEFAULT_WORKSPACE_SESSIONS_PATH
            if self._is_valid_storage_path(workspace_sessions):
                self.storage_path = workspace_sessions
            else:
                self.storage_path = DEFAULT_FALLBACK_PATH

        self.index_path = self.storage_path.parent / "index"
        self.locks_path = self.storage_path.parent / "locks"

        self._ensure_directories()

    def _is_valid_storage_path(self, path: Path) -> bool:
        """Check if a storage path is valid (exists or parent is writable).

        Args:
            path: Path to check

        Returns:
            True if path is usable for storage
        """
        if path.exists():
            return path.is_dir()
        parent = path.parent
        while not parent.exists():
            parent = parent.parent
        return os.access(parent, os.W_OK)

    def _ensure_directories(self) -> None:
        """Create storage directories if they don't exist."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.locks_path.mkdir(parents=True, exist_ok=True)

    def _get_session_path(self, session_id: str) -> Path:
        """Get file path for a session.

        Args:
            session_id: Session identifier

        Returns:
            Path to session file
        """
        safe_id = sanitize_id(session_id)
        return self.storage_path / f"{safe_id}.json"

    def _get_lock_path(self, session_id: str) -> Path:
        """Get lock file path for a session.

        Args:
            session_id: Session identifier

        Returns:
            Path to lock file
        """
        safe_id = sanitize_id(session_id)
        return self.locks_path / f"{safe_id}.lock"

    def _get_spec_lock_path(self, spec_id: str) -> Path:
        """Get per-spec lock file path.

        Args:
            spec_id: Spec identifier

        Returns:
            Path to spec lock file
        """
        safe_id = sanitize_id(spec_id)
        return self.locks_path / f"spec_{safe_id}.lock"

    def _get_pointer_path(self, spec_id: str) -> Path:
        """Get active session pointer file path for a spec.

        Args:
            spec_id: Spec identifier

        Returns:
            Path to pointer file
        """
        safe_id = sanitize_id(spec_id)
        return self.index_path / f"{safe_id}.active"

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    def save(self, session: AutonomousSessionState) -> None:
        """Save a session with atomic write and locking.

        Args:
            session: Session state to save

        Raises:
            Timeout: If lock acquisition times out
        """
        session_path = self._get_session_path(session.id)
        lock_path = self._get_lock_path(session.id)

        with FileLock(lock_path, timeout=LOCK_ACQUISITION_TIMEOUT):
            # Atomic write: temp file + fsync + rename
            data = session.model_dump(mode="json", by_alias=True)

            # Write to temp file
            fd, temp_path = tempfile.mkstemp(
                dir=self.storage_path,
                prefix=f".{session.id}.",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(data, f, indent=2, default=str)
                    f.flush()
                    os.fsync(f.fileno())

                # Atomic rename
                os.replace(temp_path, session_path)

                logger.debug("Saved session %s to %s", session.id, session_path)

            except Exception:
                # Clean up temp file on error
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise

    def load(
        self,
        session_id: str,
        apply_migrations: bool = True,
    ) -> Optional[AutonomousSessionState]:
        """Load a session with locking and optional migration.

        Args:
            session_id: Session identifier
            apply_migrations: Whether to apply schema migrations

        Returns:
            Session state or None if not found/expired

        Raises:
            Timeout: If lock acquisition times out
        """
        session_path = self._get_session_path(session_id)
        lock_path = self._get_lock_path(session_id)

        # Quick existence check (non-atomic, but avoids lock contention)
        if not session_path.exists():
            return None

        with FileLock(lock_path, timeout=LOCK_ACQUISITION_TIMEOUT):
            # Re-check existence inside lock
            if not session_path.exists():
                return None

            # Check expiry
            if self._is_expired(session_path):
                logger.debug("Session %s has expired, removing", session_id)
                self._delete_session_files(session_id)
                return None

            try:
                data = json.loads(session_path.read_text())

                # Apply migrations if needed
                if apply_migrations:
                    data, warnings = migrate_state(data)
                    for warning in warnings:
                        logger.info(
                            "Migration warning for %s: %s",
                            session_id,
                            warning.message,
                        )

                return AutonomousSessionState.model_validate(data)

            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning("Failed to load session %s: %s", session_id, exc)
                return None

    def delete(self, session_id: str) -> bool:
        """Delete a session and its pointer file.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found

        Raises:
            Timeout: If lock acquisition times out
        """
        session_path = self._get_session_path(session_id)
        lock_path = self._get_lock_path(session_id)

        # Quick existence check
        if not session_path.exists():
            self._cleanup_orphaned_lock(session_id)
            return False

        with FileLock(lock_path, timeout=LOCK_ACQUISITION_TIMEOUT):
            deleted = self._delete_session_files(session_id)

        # Clean up lock file after releasing it
        if deleted:
            self._cleanup_lock_file(lock_path)

        return deleted

    def _delete_session_files(self, session_id: str) -> bool:
        """Delete session file and pointer (assumes lock is held).

        Args:
            session_id: Session identifier

        Returns:
            True if deleted
        """
        session_path = self._get_session_path(session_id)

        try:
            # Load session to get spec_id for pointer cleanup
            try:
                data = json.loads(session_path.read_text())
                spec_id = data.get("spec_id")
                if spec_id:
                    self._remove_pointer(spec_id)
            except (json.JSONDecodeError, OSError):
                pass  # Best effort

            session_path.unlink()
            logger.debug("Deleted session %s", session_id)
            return True

        except FileNotFoundError:
            return False
        except OSError as exc:
            logger.warning("Failed to delete session %s: %s", session_id, exc)
            return False

    def _cleanup_orphaned_lock(self, session_id: str) -> None:
        """Clean up orphaned lock file if session doesn't exist.

        Args:
            session_id: Session identifier
        """
        lock_path = self._get_lock_path(session_id)
        if lock_path.exists():
            try:
                lock_path.unlink()
            except OSError:
                pass

    def _cleanup_lock_file(self, lock_path: Path) -> None:
        """Clean up lock file after releasing it.

        Args:
            lock_path: Path to lock file
        """
        try:
            lock_path.unlink()
        except OSError:
            pass  # May still be in use or already gone

    # =========================================================================
    # List Operations with Pagination
    # =========================================================================

    def list_sessions(
        self,
        status_filter: Optional[str] = None,
        spec_id: Optional[str] = None,
        limit: int = 20,
        cursor: Optional[str] = None,
        include_total: bool = False,
    ) -> ListSessionsResult:
        """List sessions with filtering and cursor pagination.

        Args:
            status_filter: Filter by session status
            spec_id: Filter by spec ID
            limit: Maximum sessions to return (1-100)
            cursor: Pagination cursor
            include_total: Whether to include total count

        Returns:
            ListSessionsResult with sessions and pagination info
        """
        limit = max(1, min(limit, 100))
        filters_hash = compute_filters_hash(status_filter, spec_id)

        # Decode cursor if provided
        cursor_after_time: Optional[str] = None
        cursor_after_id: Optional[str] = None

        if cursor is not None:
            try:
                _, cursor_time, cursor_id, cursor_filters = decode_cursor(cursor)
                if cursor_filters != filters_hash:
                    raise ValueError("Cursor filters don't match current filters")
                cursor_after_time = cursor_time
                cursor_after_id = cursor_id
            except ValueError as e:
                logger.warning("Invalid cursor: %s", e)
                raise ValueError(f"INVALID_CURSOR: {e}") from e

        # Collect matching sessions
        sessions: List[Tuple[datetime, str, SessionSummary]] = []

        for session_path in self.storage_path.glob("*.json"):
            if self._is_expired(session_path):
                continue

            try:
                data = json.loads(session_path.read_text())
                session = AutonomousSessionState.model_validate(data)

                # Apply filters
                if status_filter and session.status.value != status_filter:
                    continue
                if spec_id and session.spec_id != spec_id:
                    continue

                # Compute effective status for stale sessions
                effective_status = self._compute_effective_status(session)

                summary = SessionSummary(
                    session_id=session.id,
                    spec_id=session.spec_id,
                    status=session.status,
                    effective_status=effective_status,
                    pause_reason=session.pause_reason,
                    created_at=session.created_at,
                    updated_at=session.updated_at,
                    active_phase_id=session.active_phase_id,
                    tasks_completed=session.counters.tasks_completed,
                )

                sessions.append((session.updated_at, session.id, summary))

            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning("Failed to load session from %s: %s", session_path, exc)
                continue

        # Sort by updated_at DESC, session_id DESC (deterministic)
        sessions.sort(key=lambda x: (x[0], x[1]), reverse=True)

        # Apply cursor filtering
        if cursor_after_time is not None and cursor_after_id is not None:
            cursor_time = datetime.fromisoformat(cursor_after_time.replace("Z", "+00:00"))
            filtered = []
            past_cursor = False
            for updated_at, session_id, summary in sessions:
                if past_cursor:
                    filtered.append((updated_at, session_id, summary))
                elif (updated_at, session_id) < (cursor_time, cursor_after_id):
                    # Cursor not found - it may have been deleted
                    filtered.append((updated_at, session_id, summary))
                elif updated_at == cursor_time and session_id == cursor_after_id:
                    past_cursor = True
            sessions = filtered

        # Compute total before limiting
        total_count = len(sessions) if include_total else None

        # Apply limit
        has_more = len(sessions) > limit
        sessions = sessions[:limit]

        # Build result
        result_sessions = [s[2] for s in sessions]

        # Build next cursor
        next_cursor = None
        if has_more and sessions:
            last_updated_at, last_session_id, _ = sessions[-1]
            next_cursor = encode_cursor(
                schema_version=1,
                last_updated_at=last_updated_at.isoformat(),
                last_session_id=last_session_id,
                filters_hash=filters_hash,
            )

        return ListSessionsResult(
            sessions=result_sessions,
            cursor=next_cursor,
            has_more=has_more,
            total_count=total_count,
        )

    def _compute_effective_status(
        self,
        session: AutonomousSessionState,
    ) -> Optional[SessionStatus]:
        """Compute effective status considering staleness.

        Delegates to the shared implementation in models.py to avoid duplication.
        """
        from foundry_mcp.core.autonomy.models import compute_effective_status
        return compute_effective_status(session)

    # =========================================================================
    # Active Session Pointer Management
    #
    # NOTE: ADR-002 (line 734) specifies spec_active_index.json, but this
    # implementation uses per-spec pointer files (index/{spec_id}.active)
    # instead.  Per-spec files are better for concurrency because they avoid
    # contention on a single shared index file under concurrent session
    # operations for different specs.
    # =========================================================================

    def get_active_session(self, spec_id: str) -> Optional[str]:
        """Get active session ID for a spec from pointer file.

        Args:
            spec_id: Spec identifier

        Returns:
            Active session ID or None
        """
        pointer_path = self._get_pointer_path(spec_id)

        if not pointer_path.exists():
            return None

        try:
            data = json.loads(pointer_path.read_text())
            session_id = data.get("session_id")

            # Verify session still exists and is non-terminal
            if session_id:
                session = self.load(session_id)
                if session and session.status not in TERMINAL_STATUSES:
                    return session_id

            # Pointer is stale, clean it up
            self._remove_pointer(spec_id)
            return None

        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read pointer for %s: %s", spec_id, exc)
            return None

    def set_active_session(self, spec_id: str, session_id: str) -> None:
        """Set active session pointer for a spec.

        Args:
            spec_id: Spec identifier
            session_id: Active session ID
        """
        pointer_path = self._get_pointer_path(spec_id)

        data = {
            "session_id": session_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        pointer_path.write_text(json.dumps(data, indent=2))

    def remove_active_session(self, spec_id: str) -> bool:
        """Remove active session pointer for a spec.

        Args:
            spec_id: Spec identifier

        Returns:
            True if pointer was removed
        """
        return self._remove_pointer(spec_id)

    def _remove_pointer(self, spec_id: str) -> bool:
        """Remove pointer file (internal).

        Args:
            spec_id: Spec identifier

        Returns:
            True if removed
        """
        pointer_path = self._get_pointer_path(spec_id)

        try:
            pointer_path.unlink()
            return True
        except FileNotFoundError:
            return False
        except OSError as exc:
            logger.warning("Failed to remove pointer for %s: %s", spec_id, exc)
            return False

    # =========================================================================
    # Active Session Lookup
    # =========================================================================

    def lookup_active_session(
        self,
        workspace_path: Optional[Path] = None,
    ) -> Tuple[ActiveSessionLookupResult, Optional[str]]:
        """Find the single active session in the workspace.

        Scans pointer files to find non-terminal sessions.

        Args:
            workspace_path: Workspace path (default: storage workspace)

        Returns:
            Tuple of (result, session_id):
            - (FOUND, session_id): Single active session found
            - (NOT_FOUND, None): No active sessions
            - (AMBIGUOUS, None): Multiple active sessions
        """
        active_sessions: List[str] = []

        for pointer_path in self.index_path.glob("*.active"):
            try:
                data = json.loads(pointer_path.read_text())
                session_id = data.get("session_id")

                if session_id:
                    # Verify session is non-terminal
                    session = self.load(session_id)
                    if session and session.status not in TERMINAL_STATUSES:
                        active_sessions.append(session_id)

            except (json.JSONDecodeError, OSError):
                continue

        if len(active_sessions) == 0:
            return ActiveSessionLookupResult.NOT_FOUND, None
        elif len(active_sessions) == 1:
            return ActiveSessionLookupResult.FOUND, active_sessions[0]
        else:
            return ActiveSessionLookupResult.AMBIGUOUS, None

    # =========================================================================
    # Per-Spec Locking
    # =========================================================================

    def acquire_spec_lock(
        self,
        spec_id: str,
        timeout: float = LOCK_ACQUISITION_TIMEOUT,
    ) -> FileLock:
        """Acquire per-spec lock for atomic operations.

        Args:
            spec_id: Spec identifier
            timeout: Lock acquisition timeout

        Returns:
            FileLock instance (caller must use as context manager)

        Raises:
            Timeout: If lock cannot be acquired
        """
        lock_path = self._get_spec_lock_path(spec_id)
        lock = FileLock(lock_path, timeout=timeout)
        return lock

    # =========================================================================
    # Garbage Collection
    # =========================================================================

    def _is_expired(self, session_path: Path) -> bool:
        """Check if a session file has expired based on TTL.

        Args:
            session_path: Path to session file

        Returns:
            True if expired
        """
        try:
            # Load session to get status
            data = json.loads(session_path.read_text())
            status_str = data.get("status")
            status = SessionStatus(status_str) if status_str else None

            if status is None or status not in GC_TTL_DAYS:
                return False

            ttl_days = GC_TTL_DAYS[status]
            mtime = datetime.fromtimestamp(session_path.stat().st_mtime, tz=timezone.utc)
            expiry = mtime + timedelta(days=ttl_days)

            return datetime.now(timezone.utc) > expiry

        except (json.JSONDecodeError, ValueError, OSError):
            return False

    def cleanup_expired(self) -> Dict[str, int]:
        """Remove expired sessions and orphaned files.

        Returns:
            Dict with counts of removed items
        """
        removed = {
            "sessions": 0,
            "pointers": 0,
            "locks": 0,
        }

        # Clean expired sessions
        for session_path in self.storage_path.glob("*.json"):
            if self._is_expired(session_path):
                session_id = session_path.stem
                if self.delete(session_id):
                    removed["sessions"] += 1

        # Clean orphaned pointer files
        for pointer_path in self.index_path.glob("*.active"):
            try:
                data = json.loads(pointer_path.read_text())
                session_id = data.get("session_id")

                if session_id:
                    session_path = self._get_session_path(session_id)
                    if not session_path.exists():
                        pointer_path.unlink()
                        removed["pointers"] += 1

            except (json.JSONDecodeError, OSError):
                continue

        # Clean orphaned lock files
        for lock_path in self.locks_path.glob("*.lock"):
            if lock_path.name.startswith("spec_"):
                # Spec locks: orphaned if no active pointer references this spec
                # Extract spec_id from lock filename (spec_{spec_id}.lock)
                spec_id = lock_path.stem[len("spec_"):]
                if spec_id:
                    pointer_path = self.index_path / f"{spec_id}.active"
                    if not pointer_path.exists():
                        try:
                            lock_path.unlink()
                            removed["locks"] += 1
                        except OSError:
                            pass
            else:
                # Session locks: check if session exists
                session_id = lock_path.stem.replace(".lock", "")
                session_path = self._get_session_path(session_id)
                if not session_path.exists():
                    try:
                        lock_path.unlink()
                        removed["locks"] += 1
                    except OSError:
                        pass

        logger.info(
            "GC completed: removed %d sessions, %d pointers, %d locks",
            removed["sessions"],
            removed["pointers"],
            removed["locks"],
        )

        return removed

    # =========================================================================
    # Stats
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics.

        Returns:
            Dict with storage stats
        """
        sessions_by_status: Dict[str, int] = {}
        total = 0

        for session_path in self.storage_path.glob("*.json"):
            if self._is_expired(session_path):
                continue
            try:
                data = json.loads(session_path.read_text())
                status = data.get("status", "unknown")
                sessions_by_status[status] = sessions_by_status.get(status, 0) + 1
                total += 1
            except (json.JSONDecodeError, OSError):
                continue

        return {
            "total_sessions": total,
            "by_status": sessions_by_status,
            "storage_path": str(self.storage_path),
        }

    # =========================================================================
    # Step Proof Record Management (P1.1)
    # =========================================================================

    def _get_proof_path(self, session_id: str) -> Path:
        """Get proof records directory path for a session."""
        safe_id = sanitize_id(session_id)
        proofs_dir = self.storage_path.parent / "proofs"
        proofs_dir.mkdir(parents=True, exist_ok=True)
        return proofs_dir / f"{safe_id}_proofs.json"

    def _get_proof_lock_path(self, session_id: str) -> Path:
        """Get lock file path for proof operations."""
        safe_id = sanitize_id(session_id)
        return self.locks_path / f"proof_{safe_id}.lock"

    def load_proof_records(self, session_id: str) -> Dict[str, Any]:
        """Load all proof records for a session.

        Args:
            session_id: Session identifier

        Returns:
            Dict mapping step_proof to StepProofRecord data
        """
        proof_path = self._get_proof_path(session_id)
        if not proof_path.exists():
            return {}
        try:
            data = json.loads(proof_path.read_text())
            return data.get("records", {})
        except (json.JSONDecodeError, OSError):
            return {}

    def save_proof_record(
        self,
        session_id: str,
        record: "StepProofRecord",
    ) -> None:
        """Save a proof record atomically.

        Args:
            session_id: Session identifier
            record: StepProofRecord to save

        Raises:
            Timeout: If lock acquisition times out
        """
        from .models import StepProofRecord

        proof_path = self._get_proof_path(session_id)
        lock_path = self._get_proof_lock_path(session_id)

        with FileLock(lock_path, timeout=LOCK_ACQUISITION_TIMEOUT):
            # Load existing records
            records = {}
            if proof_path.exists():
                try:
                    data = json.loads(proof_path.read_text())
                    records = data.get("records", {})
                except (json.JSONDecodeError, OSError):
                    pass

            # Add new record
            records[record.step_proof] = record.model_dump(mode="json")

            # Atomic write
            fd, temp_path = tempfile.mkstemp(
                dir=proof_path.parent,
                prefix=f".{session_id}_proofs.",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump({"records": records}, f, indent=2, default=str)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp_path, proof_path)
            except Exception:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise

    def get_proof_record(
        self,
        session_id: str,
        step_proof: str,
        *,
        include_expired: bool = False,
    ) -> Optional["StepProofRecord"]:
        """Get a specific proof record.

        Args:
            session_id: Session identifier
            step_proof: Proof token to look up
            include_expired: Include records outside replay grace window

        Returns:
            StepProofRecord if found and not expired, None otherwise
        """
        from .models import StepProofRecord

        records = self.load_proof_records(session_id)
        record_data = records.get(step_proof)
        if not record_data:
            return None

        try:
            record = StepProofRecord.model_validate(record_data)
        except (ValueError, KeyError):
            return None
        if not include_expired and record.grace_expires_at <= datetime.now(timezone.utc):
            return None
        return record

    def update_proof_record_response(
        self,
        session_id: str,
        step_proof: str,
        *,
        step_id: str,
        response: Dict[str, Any],
    ) -> bool:
        """Attach response payload/hash to a previously-consumed proof record."""
        from .models import StepProofRecord

        lock_path = self._get_proof_lock_path(session_id)
        proof_path = self._get_proof_path(session_id)
        with FileLock(lock_path, timeout=LOCK_ACQUISITION_TIMEOUT):
            records = self.load_proof_records(session_id)
            record_data = records.get(step_proof)
            if not isinstance(record_data, dict):
                return False
            try:
                record = StepProofRecord.model_validate(record_data)
            except (ValueError, KeyError):
                return False

            response_json = json.dumps(
                response, sort_keys=True, separators=(",", ":"), default=str
            )
            record.step_id = step_id
            record.cached_response = response
            record.response_hash = hashlib.sha256(response_json.encode()).hexdigest()

            records[step_proof] = record.model_dump(mode="json")

            fd, temp_path = tempfile.mkstemp(
                dir=proof_path.parent,
                prefix=f".{session_id}_proofs.",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump({"records": records}, f, indent=2, default=str)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp_path, proof_path)
            except Exception:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise
            return True

    def consume_proof_with_lock(
        self,
        session_id: str,
        step_proof: str,
        payload_hash: str,
        grace_window_seconds: int = 30,
        step_id: str = "",
    ) -> Tuple[bool, Optional["StepProofRecord"], str]:
        """Atomically consume a proof token with per-session locking.

        This method handles the complete proof consumption flow:
        1. Check if proof already consumed (return cached response if in grace window)
        2. Check for PROOF_CONFLICT (same proof, different payload)
        3. Consume proof and return success

        Args:
            session_id: Session identifier
            step_proof: Proof token to consume
            payload_hash: SHA-256 hash of request payload
            grace_window_seconds: Grace window for replay (default 30s)
            step_id: Step identifier bound to this proof token

        Returns:
            Tuple of (success, existing_record_or_none, error_code_or_empty)
            - (True, None, ""): Proof consumed successfully, proceed with execution
            - (True, record, ""): Idempotent replay within grace window, use cached response
            - (False, None, "PROOF_CONFLICT"): Same proof with different payload
            - (False, record, "PROOF_EXPIRED"): Proof consumed but grace window expired
        """
        from .models import StepProofRecord

        lock_path = self._get_proof_lock_path(session_id)

        with FileLock(lock_path, timeout=LOCK_ACQUISITION_TIMEOUT):
            now = datetime.now(timezone.utc)
            existing = self.get_proof_record(
                session_id,
                step_proof,
                include_expired=True,
            )

            if existing is not None:
                # Proof was already consumed
                if existing.payload_hash != payload_hash:
                    # Different payload - conflict
                    return (False, None, "PROOF_CONFLICT")
                if existing.grace_expires_at > now:
                    # Same payload - idempotent replay within grace window
                    return (True, existing, "")
                return (False, existing, "PROOF_EXPIRED")

            # Proof not yet consumed - consume it now
            grace_expires = now + timedelta(seconds=grace_window_seconds)

            record = StepProofRecord(
                step_proof=step_proof,
                step_id=step_id,
                payload_hash=payload_hash,
                consumed_at=now,
                grace_expires_at=grace_expires,
            )
            proof_path = self._get_proof_path(session_id)
            records = self.load_proof_records(session_id)
            records[step_proof] = record.model_dump(mode="json")

            fd, temp_path = tempfile.mkstemp(
                dir=proof_path.parent,
                prefix=f".{session_id}_proofs.",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump({"records": records}, f, indent=2, default=str)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp_path, proof_path)
            except Exception:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise
            return (True, None, "")
