"""Tests for AutonomyStorage file-based session persistence.

Covers:
- Atomic save/load round-trip
- Cursor encode/decode
- GC by TTL
- Active session pointer CRUD
- sanitize_id path traversal prevention
- Concurrent access with file locks
"""

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Thread
from typing import List

import pytest

from foundry_mcp.core.autonomy.memory import (
    ActiveSessionLookupResult,
    AutonomyStorage,
    GC_TTL_DAYS,
    ListSessionsResult,
    compute_filters_hash,
    decode_cursor,
    encode_cursor,
    sanitize_id,
)
from foundry_mcp.core.autonomy.models import SessionStatus

from .conftest import make_session


# =============================================================================
# sanitize_id (Path Traversal Prevention)
# =============================================================================


class TestSanitizeId:
    """Test sanitize_id prevents path traversal and special chars."""

    def test_alphanumeric_preserved(self):
        assert sanitize_id("abc123") == "abc123"

    def test_hyphens_preserved(self):
        assert sanitize_id("my-session-id") == "my-session-id"

    def test_underscores_preserved(self):
        assert sanitize_id("my_session_id") == "my_session_id"

    def test_path_traversal_stripped(self):
        assert sanitize_id("../../etc/passwd") == "etcpasswd"

    def test_dots_stripped(self):
        assert sanitize_id("file.json") == "filejson"

    def test_slashes_stripped(self):
        assert sanitize_id("path/to/file") == "pathtofile"
        assert sanitize_id("path\\to\\file") == "pathtofile"

    def test_spaces_stripped(self):
        assert sanitize_id("my session") == "mysession"

    def test_special_chars_stripped(self):
        assert sanitize_id("id@#$%^&*()") == "id"

    def test_empty_string(self):
        assert sanitize_id("") == ""

    def test_ulid_format_preserved(self):
        assert sanitize_id("01HXYZ1234567890ABCDEFGHIJ") == "01HXYZ1234567890ABCDEFGHIJ"


# =============================================================================
# Cursor Encode/Decode
# =============================================================================


class TestCursorEncodeDecode:
    """Test pagination cursor encoding and decoding."""

    def test_round_trip(self):
        cursor = encode_cursor(
            schema_version=1,
            last_updated_at="2024-01-01T00:00:00+00:00",
            last_session_id="session-1",
            filters_hash="abcd1234",
        )
        version, updated_at, session_id, filters_hash = decode_cursor(cursor)
        assert version == 1
        assert updated_at == "2024-01-01T00:00:00+00:00"
        assert session_id == "session-1"
        assert filters_hash == "abcd1234"

    def test_decode_invalid_base64(self):
        with pytest.raises((ValueError, UnicodeDecodeError)):
            decode_cursor("not-valid-base64!!!")

    def test_decode_invalid_json(self):
        import base64

        bad = base64.urlsafe_b64encode(b"not json").decode()
        with pytest.raises(ValueError, match="Invalid cursor"):
            decode_cursor(bad)

    def test_decode_missing_keys(self):
        import base64

        bad = base64.urlsafe_b64encode(json.dumps({"v": 1}).encode()).decode()
        with pytest.raises(ValueError, match="Invalid cursor"):
            decode_cursor(bad)


class TestComputeFiltersHash:
    """Test filter hash computation for cursor validation."""

    def test_same_filters_same_hash(self):
        h1 = compute_filters_hash(status_filter="running", spec_id="spec-1")
        h2 = compute_filters_hash(status_filter="running", spec_id="spec-1")
        assert h1 == h2

    def test_different_filters_different_hash(self):
        h1 = compute_filters_hash(status_filter="running")
        h2 = compute_filters_hash(status_filter="paused")
        assert h1 != h2

    def test_none_filters_stable(self):
        h1 = compute_filters_hash()
        h2 = compute_filters_hash()
        assert h1 == h2


# =============================================================================
# Save/Load Round-Trip
# =============================================================================


class TestSaveLoadRoundTrip:
    """Test atomic save and load operations."""

    def test_save_and_load(self, storage):
        session = make_session(session_id="sess-1")
        storage.save(session)
        loaded = storage.load("sess-1")

        assert loaded is not None
        assert loaded.id == "sess-1"
        assert loaded.spec_id == session.spec_id
        assert loaded.status == session.status

    def test_load_nonexistent_returns_none(self, storage):
        assert storage.load("nonexistent") is None

    def test_save_overwrites(self, storage):
        session = make_session(session_id="sess-1", status=SessionStatus.RUNNING)
        storage.save(session)

        session.status = SessionStatus.PAUSED
        storage.save(session)

        loaded = storage.load("sess-1")
        assert loaded.status == SessionStatus.PAUSED

    def test_schema_version_preserved(self, storage):
        session = make_session(session_id="sess-1")
        storage.save(session)

        # Read raw JSON to verify _schema_version key
        path = storage._get_session_path("sess-1")
        raw = json.loads(path.read_text())
        assert raw.get("_schema_version") == 3

    def test_atomic_write_no_partial_files(self, storage):
        """Verify no temp files remain after successful save."""
        session = make_session(session_id="sess-1")
        storage.save(session)

        tmp_files = list(storage.storage_path.glob(".*tmp"))
        assert len(tmp_files) == 0

    def test_model_dump_by_alias(self, storage):
        """Verify session serializes with by_alias=True (for _schema_version)."""
        session = make_session(session_id="sess-1")
        data = session.model_dump(mode="json", by_alias=True)
        assert "_schema_version" in data
        assert "schema_version" not in data


# =============================================================================
# Delete
# =============================================================================


class TestDelete:
    """Test session deletion."""

    def test_delete_existing(self, storage):
        session = make_session(session_id="sess-1")
        storage.save(session)
        assert storage.delete("sess-1") is True
        assert storage.load("sess-1") is None

    def test_delete_nonexistent(self, storage):
        assert storage.delete("nonexistent") is False


# =============================================================================
# Active Session Pointer CRUD
# =============================================================================


class TestActiveSessionPointers:
    """Test per-spec active session pointer management."""

    def test_set_and_get_pointer(self, storage):
        session = make_session(session_id="sess-1", spec_id="spec-1")
        storage.save(session)
        storage.set_active_session("spec-1", "sess-1")

        active_id = storage.get_active_session("spec-1")
        assert active_id == "sess-1"

    def test_get_returns_none_when_no_pointer(self, storage):
        assert storage.get_active_session("nonexistent-spec") is None

    def test_stale_pointer_cleaned_up(self, storage):
        """Pointer to a non-existent session is cleaned up on get."""
        storage.set_active_session("spec-1", "deleted-session")
        active_id = storage.get_active_session("spec-1")
        assert active_id is None

    def test_pointer_to_terminal_session_cleaned_up(self, storage):
        """Pointer to a completed session is cleaned up."""
        session = make_session(
            session_id="sess-1", spec_id="spec-1", status=SessionStatus.COMPLETED
        )
        storage.save(session)
        storage.set_active_session("spec-1", "sess-1")

        active_id = storage.get_active_session("spec-1")
        assert active_id is None

    def test_remove_pointer(self, storage):
        session = make_session(session_id="sess-1", spec_id="spec-1")
        storage.save(session)
        storage.set_active_session("spec-1", "sess-1")
        assert storage.remove_active_session("spec-1") is True
        assert storage.get_active_session("spec-1") is None

    def test_remove_nonexistent_pointer(self, storage):
        assert storage.remove_active_session("nonexistent") is False


# =============================================================================
# Active Session Lookup (Workspace Scan)
# =============================================================================


class TestLookupActiveSession:
    """Test workspace-wide active session lookup."""

    def test_no_sessions_returns_not_found(self, storage):
        result, session_id = storage.lookup_active_session()
        assert result == ActiveSessionLookupResult.NOT_FOUND
        assert session_id is None

    def test_single_active_returns_found(self, storage):
        session = make_session(session_id="sess-1", spec_id="spec-1")
        storage.save(session)
        storage.set_active_session("spec-1", "sess-1")

        result, session_id = storage.lookup_active_session()
        assert result == ActiveSessionLookupResult.FOUND
        assert session_id == "sess-1"

    def test_multiple_active_returns_ambiguous(self, storage):
        for i in range(2):
            session = make_session(
                session_id=f"sess-{i}", spec_id=f"spec-{i}"
            )
            storage.save(session)
            storage.set_active_session(f"spec-{i}", f"sess-{i}")

        result, session_id = storage.lookup_active_session()
        assert result == ActiveSessionLookupResult.AMBIGUOUS
        assert session_id is None

    def test_terminal_sessions_ignored(self, storage):
        session = make_session(
            session_id="sess-1", spec_id="spec-1", status=SessionStatus.COMPLETED
        )
        storage.save(session)
        storage.set_active_session("spec-1", "sess-1")

        result, _ = storage.lookup_active_session()
        assert result == ActiveSessionLookupResult.NOT_FOUND


# =============================================================================
# List Sessions with Pagination
# =============================================================================


class TestListSessions:
    """Test session listing with filtering and pagination."""

    def test_list_empty(self, storage):
        result = storage.list_sessions()
        assert result.sessions == []
        assert result.has_more is False

    def test_list_all_sessions(self, storage):
        for i in range(3):
            storage.save(make_session(session_id=f"sess-{i}", spec_id=f"spec-{i}"))

        result = storage.list_sessions()
        assert len(result.sessions) == 3

    def test_filter_by_status(self, storage):
        storage.save(make_session(session_id="s1", status=SessionStatus.RUNNING))
        storage.save(make_session(session_id="s2", status=SessionStatus.PAUSED))
        storage.save(make_session(session_id="s3", status=SessionStatus.RUNNING))

        result = storage.list_sessions(status_filter="running")
        assert len(result.sessions) == 2
        assert all(s.status == SessionStatus.RUNNING for s in result.sessions)

    def test_filter_by_spec_id(self, storage):
        storage.save(make_session(session_id="s1", spec_id="spec-a"))
        storage.save(make_session(session_id="s2", spec_id="spec-b"))

        result = storage.list_sessions(spec_id="spec-a")
        assert len(result.sessions) == 1
        assert result.sessions[0].spec_id == "spec-a"

    def test_pagination_limit(self, storage):
        for i in range(5):
            storage.save(make_session(session_id=f"sess-{i:02d}", spec_id=f"spec-{i}"))

        result = storage.list_sessions(limit=2)
        assert len(result.sessions) == 2
        assert result.has_more is True
        assert result.cursor is not None

    def test_pagination_cursor_continuation(self, storage):
        for i in range(5):
            session = make_session(session_id=f"sess-{i:02d}", spec_id=f"spec-{i}")
            storage.save(session)

        # Get first page
        page1 = storage.list_sessions(limit=2)
        assert len(page1.sessions) == 2
        assert page1.has_more is True

        # Get second page
        page2 = storage.list_sessions(limit=2, cursor=page1.cursor)
        assert len(page2.sessions) == 2

        # Verify no overlap
        ids1 = {s.session_id for s in page1.sessions}
        ids2 = {s.session_id for s in page2.sessions}
        assert ids1 & ids2 == set()

    def test_invalid_cursor_raises(self, storage):
        with pytest.raises(ValueError, match="INVALID_CURSOR"):
            storage.list_sessions(cursor="bad-cursor")

    def test_include_total(self, storage):
        for i in range(3):
            storage.save(make_session(session_id=f"sess-{i}"))
        result = storage.list_sessions(include_total=True)
        assert result.total_count == 3


# =============================================================================
# GC by TTL
# =============================================================================


class TestGarbageCollection:
    """Test TTL-based garbage collection."""

    def test_running_session_not_expired(self, storage):
        session = make_session(session_id="s1", status=SessionStatus.RUNNING)
        storage.save(session)
        assert storage.load("s1") is not None

    def test_completed_session_expired_after_ttl(self, storage):
        session = make_session(session_id="s1", status=SessionStatus.COMPLETED)
        storage.save(session)

        # Manually backdate the file's mtime beyond TTL
        path = storage._get_session_path("s1")
        ttl_days = GC_TTL_DAYS[SessionStatus.COMPLETED]
        old_mtime = (datetime.now(timezone.utc) - timedelta(days=ttl_days + 1)).timestamp()
        import os

        os.utime(path, (old_mtime, old_mtime))

        # Load should return None (expired)
        assert storage.load("s1") is None

    def test_cleanup_expired_removes_old_sessions(self, storage):
        session = make_session(session_id="s1", status=SessionStatus.ENDED)
        storage.save(session)

        path = storage._get_session_path("s1")
        ttl_days = GC_TTL_DAYS[SessionStatus.ENDED]
        old_mtime = (datetime.now(timezone.utc) - timedelta(days=ttl_days + 1)).timestamp()
        import os

        os.utime(path, (old_mtime, old_mtime))

        removed = storage.cleanup_expired()
        assert removed["sessions"] >= 1

    def test_failed_session_longer_ttl(self, storage):
        """Failed sessions have longer TTL (30 days vs 7 days)."""
        assert GC_TTL_DAYS[SessionStatus.FAILED] > GC_TTL_DAYS[SessionStatus.COMPLETED]


# =============================================================================
# Concurrent Access
# =============================================================================


class TestConcurrentAccess:
    """Test thread-safe operations with file locks."""

    def test_concurrent_saves_do_not_corrupt(self, storage):
        """Multiple threads saving to the same session produce valid state."""
        errors: List[Exception] = []

        def save_session(idx: int) -> None:
            try:
                session = make_session(session_id="shared", spec_id=f"spec-{idx}")
                storage.save(session)
            except Exception as e:
                errors.append(e)

        threads = [Thread(target=save_session, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Session should be loadable
        loaded = storage.load("shared")
        assert loaded is not None
        assert loaded.id == "shared"

    def test_concurrent_save_and_load(self, storage):
        """Concurrent read while writing does not crash."""
        session = make_session(session_id="concurrent-1")
        storage.save(session)

        errors: List[Exception] = []

        def reader() -> None:
            try:
                for _ in range(20):
                    storage.load("concurrent-1")
            except Exception as e:
                errors.append(e)

        def writer() -> None:
            try:
                for i in range(20):
                    s = make_session(session_id="concurrent-1", spec_id=f"spec-{i}")
                    storage.save(s)
            except Exception as e:
                errors.append(e)

        t1 = Thread(target=reader)
        t2 = Thread(target=writer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0
