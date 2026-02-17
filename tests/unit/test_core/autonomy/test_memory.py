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
import os
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
    MAX_PROOF_RECORDS,
    PROOF_TTL_SECONDS,
    VersionConflictError,
    compute_filters_hash,
    decode_cursor,
    encode_cursor,
    sanitize_id,
)
from foundry_mcp.core.autonomy.models.enums import SessionStatus
from foundry_mcp.core.autonomy.models.steps import StepProofRecord

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
# Step Proof Storage
# =============================================================================


class TestStepProofStorage:
    """Test one-time proof persistence and replay semantics."""

    def test_consume_proof_first_time(self, storage):
        success, existing, error = storage.consume_proof_with_lock(
            "sess-proof-1",
            "proof-token-1",
            "hash-1",
            grace_window_seconds=30,
            step_id="step-1",
        )

        assert success is True
        assert existing is None
        assert error == ""

    def test_consume_same_proof_payload_replays(self, storage):
        storage.consume_proof_with_lock(
            "sess-proof-2",
            "proof-token-2",
            "hash-2",
            grace_window_seconds=30,
            step_id="step-2",
        )

        success, existing, error = storage.consume_proof_with_lock(
            "sess-proof-2",
            "proof-token-2",
            "hash-2",
            grace_window_seconds=30,
            step_id="step-2",
        )

        assert success is True
        assert existing is not None
        assert existing.step_proof == "proof-token-2"
        assert error == ""

    def test_consume_same_proof_different_payload_conflicts(self, storage):
        storage.consume_proof_with_lock(
            "sess-proof-3",
            "proof-token-3",
            "hash-3",
            grace_window_seconds=30,
            step_id="step-3",
        )

        success, existing, error = storage.consume_proof_with_lock(
            "sess-proof-3",
            "proof-token-3",
            "hash-3-different",
            grace_window_seconds=30,
            step_id="step-3",
        )

        assert success is False
        assert existing is None
        assert error == "PROOF_CONFLICT"

    def test_consume_proof_after_grace_window_expires(self, storage):
        storage.consume_proof_with_lock(
            "sess-proof-4",
            "proof-token-4",
            "hash-4",
            grace_window_seconds=1,
            step_id="step-4",
        )
        time.sleep(1.1)

        success, existing, error = storage.consume_proof_with_lock(
            "sess-proof-4",
            "proof-token-4",
            "hash-4",
            grace_window_seconds=1,
            step_id="step-4",
        )

        assert success is False
        assert existing is not None
        assert error == "PROOF_EXPIRED"

    def test_update_proof_record_response_persists_cached_response(self, storage):
        storage.consume_proof_with_lock(
            "sess-proof-5",
            "proof-token-5",
            "hash-5",
            grace_window_seconds=30,
            step_id="step-5",
        )

        response = {
            "success": True,
            "data": {"session_id": "sess-proof-5", "status": "running"},
            "error": None,
            "meta": {"version": "response-v2"},
        }
        updated = storage.update_proof_record_response(
            "sess-proof-5",
            "proof-token-5",
            step_id="step-5",
            response=response,
        )

        assert updated is True
        record = storage.get_proof_record(
            "sess-proof-5",
            "proof-token-5",
            include_expired=True,
        )
        assert isinstance(record, StepProofRecord)
        assert record.cached_response is not None
        assert record.cached_response["data"]["session_id"] == "sess-proof-5"
        assert isinstance(record.response_hash, str)
        assert len(record.response_hash) == 64


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


# =============================================================================
# Optimistic Version Check (C1)
# =============================================================================


class TestOptimisticVersionCheck:
    """Test save() with expected_version for optimistic locking."""

    def test_save_with_correct_expected_version_succeeds(self, storage):
        """Save succeeds when expected_version matches on-disk version."""
        session = make_session(session_id="ver-1")
        session.state_version = 5
        storage.save(session)

        # Mutate and save with correct expected_version
        session.state_version = 6
        storage.save(session, expected_version=5)

        loaded = storage.load("ver-1")
        assert loaded is not None
        assert loaded.state_version == 6

    def test_save_with_wrong_expected_version_raises(self, storage):
        """Save raises VersionConflictError when version doesn't match."""
        session = make_session(session_id="ver-2")
        session.state_version = 5
        storage.save(session)

        session.state_version = 6
        with pytest.raises(VersionConflictError) as exc_info:
            storage.save(session, expected_version=3)

        assert exc_info.value.session_id == "ver-2"
        assert exc_info.value.expected_version == 3
        assert exc_info.value.actual_version == 5

    def test_save_without_expected_version_skips_check(self, storage):
        """Backward compat: no expected_version param always succeeds."""
        session = make_session(session_id="ver-3")
        session.state_version = 5
        storage.save(session)

        # Save again without expected_version — should always work
        session.state_version = 99
        storage.save(session)  # no expected_version, no error

        loaded = storage.load("ver-3")
        assert loaded is not None
        assert loaded.state_version == 99

    def test_concurrent_version_conflict(self, storage):
        """Two threads racing: one should succeed, one should get VersionConflictError."""
        session = make_session(session_id="ver-race")
        session.state_version = 1
        storage.save(session)

        results: List[str] = []

        def writer(label: str, delay: float) -> None:
            try:
                time.sleep(delay)
                s = make_session(session_id="ver-race")
                s.state_version = 2
                storage.save(s, expected_version=1)
                results.append(f"{label}:ok")
            except VersionConflictError:
                results.append(f"{label}:conflict")
            except Exception as e:
                results.append(f"{label}:error:{e}")

        t1 = Thread(target=writer, args=("A", 0))
        t2 = Thread(target=writer, args=("B", 0.05))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # First thread should succeed, second should see conflict
        ok_count = sum(1 for r in results if r.endswith(":ok"))
        conflict_count = sum(1 for r in results if r.endswith(":conflict"))
        assert ok_count == 1
        assert conflict_count == 1


# =============================================================================
# Proof Store Bounds (C3)
# =============================================================================


class TestProofStoreBounds:
    """Test TTL cleanup and max record enforcement for proof records."""

    def _make_proof_record_data(
        self,
        step_proof: str,
        consumed_at: datetime,
        grace_expires_at: datetime,
    ) -> dict:
        """Create a raw proof record dict for testing."""
        return {
            "step_proof": step_proof,
            "step_id": f"step-{step_proof}",
            "payload_hash": f"hash-{step_proof}",
            "consumed_at": consumed_at.isoformat(),
            "grace_expires_at": grace_expires_at.isoformat(),
            "response_hash": None,
            "cached_response": None,
        }

    def test_ttl_cleanup_removes_expired_records(self, storage):
        """Records whose grace_expires_at is older than PROOF_TTL_SECONDS are removed."""
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(seconds=PROOF_TTL_SECONDS + 600)

        records = {
            "expired-1": self._make_proof_record_data("expired-1", old_time, old_time),
            "expired-2": self._make_proof_record_data("expired-2", old_time, old_time),
        }

        cleaned, evicted = storage._cleanup_proof_records(records)
        assert evicted == 2
        assert len(cleaned) == 0

    def test_ttl_preserves_fresh_records(self, storage):
        """Records still within grace window survive cleanup."""
        now = datetime.now(timezone.utc)
        fresh_grace = now + timedelta(seconds=30)

        records = {
            "fresh-1": self._make_proof_record_data("fresh-1", now, fresh_grace),
            "fresh-2": self._make_proof_record_data("fresh-2", now, fresh_grace),
        }

        cleaned, evicted = storage._cleanup_proof_records(records)
        assert evicted == 0
        assert len(cleaned) == 2

    def test_max_records_eviction(self, storage):
        """When records exceed MAX_PROOF_RECORDS, oldest by consumed_at are evicted."""
        now = datetime.now(timezone.utc)
        future_grace = now + timedelta(seconds=300)

        # Create MAX_PROOF_RECORDS + 50 records
        records = {}
        for i in range(MAX_PROOF_RECORDS + 50):
            consumed = now - timedelta(seconds=MAX_PROOF_RECORDS + 50 - i)
            key = f"proof-{i:04d}"
            records[key] = self._make_proof_record_data(key, consumed, future_grace)

        cleaned, evicted = storage._cleanup_proof_records(records)
        assert len(cleaned) == MAX_PROOF_RECORDS
        assert evicted == 50

    def test_proof_store_bounded_under_load(self, storage):
        """After 600 inserts via save_proof_record, verify <= MAX_PROOF_RECORDS remain."""
        from foundry_mcp.core.autonomy.models.steps import StepProofRecord as SPR

        session_id = "bounded-session"
        now = datetime.now(timezone.utc)

        for i in range(600):
            record = SPR(
                step_proof=f"proof-{i:04d}",
                step_id=f"step-{i}",
                payload_hash=f"hash-{i}",
                consumed_at=now + timedelta(milliseconds=i),
                grace_expires_at=now + timedelta(hours=2),
            )
            storage.save_proof_record(session_id, record)

        final_records = storage.load_proof_records(session_id)
        assert len(final_records) <= MAX_PROOF_RECORDS + 1  # +1 for the just-added record


# =============================================================================
# T2: Step Proof Expiration with Time Advancement
# =============================================================================


class TestStepProofExpiration:
    """Verify proof replay behavior at grace window boundaries."""

    def test_replay_within_grace_window_succeeds(self, tmp_path):
        """Proof at t0, replay at t0+15s (within 30s grace) — idempotent replay."""
        from unittest.mock import MagicMock, patch

        storage = AutonomyStorage(
            storage_path=tmp_path / "sessions", workspace_path=tmp_path
        )
        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Consume at t0
        mock_dt = MagicMock(wraps=datetime)
        mock_dt.now.return_value = t0
        with patch("foundry_mcp.core.autonomy.memory.datetime", mock_dt):
            ok, rec, err = storage.consume_proof_with_lock(
                "sess-1", "proof-1", "hash-a", grace_window_seconds=30
            )
        assert ok is True
        assert rec is None
        assert err == ""

        # Replay at t0+15s (within grace)
        mock_dt.now.return_value = t0 + timedelta(seconds=15)
        with patch("foundry_mcp.core.autonomy.memory.datetime", mock_dt):
            ok, rec, err = storage.consume_proof_with_lock(
                "sess-1", "proof-1", "hash-a"
            )
        assert ok is True
        assert rec is not None  # Returns cached record
        assert err == ""

    def test_replay_past_grace_window_fails(self, tmp_path):
        """Proof at t0, replay at t0+31s (past 30s grace) — PROOF_EXPIRED."""
        from unittest.mock import MagicMock, patch

        storage = AutonomyStorage(
            storage_path=tmp_path / "sessions", workspace_path=tmp_path
        )
        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Consume at t0
        mock_dt = MagicMock(wraps=datetime)
        mock_dt.now.return_value = t0
        with patch("foundry_mcp.core.autonomy.memory.datetime", mock_dt):
            ok, rec, err = storage.consume_proof_with_lock(
                "sess-1", "proof-1", "hash-a", grace_window_seconds=30
            )
        assert ok is True

        # Replay at t0+31s (past grace)
        mock_dt.now.return_value = t0 + timedelta(seconds=31)
        with patch("foundry_mcp.core.autonomy.memory.datetime", mock_dt):
            ok, rec, err = storage.consume_proof_with_lock(
                "sess-1", "proof-1", "hash-a"
            )
        assert ok is False
        assert err == "PROOF_EXPIRED"

    def test_replay_at_exact_grace_boundary_fails(self, tmp_path):
        """Proof at t0, replay at exactly t0+30s — grace_expires_at == now is expired."""
        from unittest.mock import MagicMock, patch

        storage = AutonomyStorage(
            storage_path=tmp_path / "sessions", workspace_path=tmp_path
        )
        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Consume at t0
        mock_dt = MagicMock(wraps=datetime)
        mock_dt.now.return_value = t0
        with patch("foundry_mcp.core.autonomy.memory.datetime", mock_dt):
            ok, rec, err = storage.consume_proof_with_lock(
                "sess-1", "proof-1", "hash-a", grace_window_seconds=30
            )
        assert ok is True

        # Replay at exactly t0+30s (grace_expires_at == now, NOT greater-than)
        mock_dt.now.return_value = t0 + timedelta(seconds=30)
        with patch("foundry_mcp.core.autonomy.memory.datetime", mock_dt):
            ok, rec, err = storage.consume_proof_with_lock(
                "sess-1", "proof-1", "hash-a"
            )
        # grace_expires_at > now check: 12:00:30 > 12:00:30 is False → PROOF_EXPIRED
        assert ok is False
        assert err == "PROOF_EXPIRED"


# =============================================================================
# T4: GC-by-TTL Verification
# =============================================================================


class TestGarbageCollectionByTTL:
    """Verify GC removes sessions based on TTL per terminal status."""

    @pytest.mark.parametrize(
        "status,ttl_days",
        [
            (SessionStatus.COMPLETED, 7),
            (SessionStatus.ENDED, 7),
            (SessionStatus.FAILED, 30),
        ],
    )
    def test_session_within_ttl_survives(self, tmp_path, status, ttl_days):
        """Session at TTL-1d is not expired."""
        storage = AutonomyStorage(
            storage_path=tmp_path / "sessions", workspace_path=tmp_path
        )
        session = make_session(session_id="sess-ttl", status=status)
        storage.save(session)
        session_path = storage._get_session_path("sess-ttl")
        age_seconds = (ttl_days - 1) * 86400
        old_time = time.time() - age_seconds
        os.utime(session_path, (old_time, old_time))
        assert not storage._is_expired(session_path)

    @pytest.mark.parametrize(
        "status,ttl_days",
        [
            (SessionStatus.COMPLETED, 7),
            (SessionStatus.ENDED, 7),
            (SessionStatus.FAILED, 30),
        ],
    )
    def test_session_past_ttl_is_expired(self, tmp_path, status, ttl_days):
        """Session at TTL+1d is expired."""
        storage = AutonomyStorage(
            storage_path=tmp_path / "sessions", workspace_path=tmp_path
        )
        session = make_session(session_id="sess-ttl", status=status)
        storage.save(session)
        session_path = storage._get_session_path("sess-ttl")
        age_seconds = (ttl_days + 1) * 86400
        old_time = time.time() - age_seconds
        os.utime(session_path, (old_time, old_time))
        assert storage._is_expired(session_path)

    @pytest.mark.parametrize("status", [SessionStatus.RUNNING, SessionStatus.PAUSED])
    def test_non_terminal_never_expires(self, tmp_path, status):
        """Non-terminal sessions are never GC-expired regardless of age."""
        storage = AutonomyStorage(
            storage_path=tmp_path / "sessions", workspace_path=tmp_path
        )
        session = make_session(session_id="sess-alive", status=status)
        storage.save(session)
        session_path = storage._get_session_path("sess-alive")
        old_time = time.time() - (365 * 86400)  # 1 year old
        os.utime(session_path, (old_time, old_time))
        assert not storage._is_expired(session_path)

    def test_cleanup_expired_removes_all_terminal(self, tmp_path):
        """Bulk cleanup: all terminal sessions past TTL are removed."""
        storage = AutonomyStorage(
            storage_path=tmp_path / "sessions", workspace_path=tmp_path
        )
        statuses = [
            (SessionStatus.COMPLETED, 7),
            (SessionStatus.ENDED, 7),
            (SessionStatus.FAILED, 30),
        ]
        for i, (status, ttl_days) in enumerate(statuses):
            session = make_session(
                session_id=f"sess-gc-{i}",
                status=status,
                spec_id=f"spec-{i}",
            )
            storage.save(session)
            path = storage._get_session_path(f"sess-gc-{i}")
            old_time = time.time() - (ttl_days + 2) * 86400
            os.utime(path, (old_time, old_time))

        result = storage.cleanup_expired()
        assert result["sessions"] == 3
