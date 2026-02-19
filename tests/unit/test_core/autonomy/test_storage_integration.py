"""Integration tests for AutonomyStorage real file IO paths.

Validates actual JSON persistence, recovery after restart, atomic write
correctness, concurrent writes with FileLock, corrupted state file handling,
and GC cleanup of expired files — all without mocking storage.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from threading import Thread
from typing import List

import pytest

from foundry_mcp.core.autonomy.memory import (
    GC_TTL_DAYS,
    AutonomyStorage,
    SessionCorrupted,
)
from foundry_mcp.core.autonomy.models.enums import SessionStatus

from .conftest import make_session

# =============================================================================
# JSON Persistence Round-Trip
# =============================================================================


class TestJsonPersistenceRoundTrip:
    """Verify sessions survive full write-to-disk then read-from-disk cycle."""

    def test_create_save_reload_validates_all_fields(self, tmp_path):
        """Create a session, write to disk, create fresh storage, load back."""
        storage_path = tmp_path / "sessions"
        workspace_path = tmp_path

        # Create and save
        storage = AutonomyStorage(storage_path=storage_path, workspace_path=workspace_path)
        session = make_session(
            session_id="integ-001",
            spec_id="spec-round-trip",
            status=SessionStatus.RUNNING,
            active_phase_id="phase-1",
        )
        session.counters.tasks_completed = 3
        session.counters.consecutive_errors = 1
        storage.save(session)
        storage.set_active_session("spec-round-trip", "integ-001")

        # Create a completely new storage instance (simulates restart)
        storage2 = AutonomyStorage(storage_path=storage_path, workspace_path=workspace_path)

        loaded = storage2.load("integ-001")
        assert loaded is not None
        assert loaded.id == "integ-001"
        assert loaded.spec_id == "spec-round-trip"
        assert loaded.status == SessionStatus.RUNNING
        assert loaded.active_phase_id == "phase-1"
        assert loaded.counters.tasks_completed == 3
        assert loaded.counters.consecutive_errors == 1

        # Active pointer survives restart
        active_id = storage2.get_active_session("spec-round-trip")
        assert active_id == "integ-001"

    def test_save_produces_valid_json_on_disk(self, tmp_path):
        """Verify the file on disk is valid JSON with expected schema keys."""
        storage = AutonomyStorage(storage_path=tmp_path / "sessions", workspace_path=tmp_path)
        session = make_session(session_id="integ-json-check")
        storage.save(session)

        session_path = storage._get_session_path("integ-json-check")
        assert session_path.exists()

        raw = json.loads(session_path.read_text())
        assert raw["id"] == "integ-json-check"
        assert raw["_schema_version"] == 3
        assert "status" in raw
        assert "counters" in raw
        assert "limits" in raw

    def test_multiple_saves_overwrite_cleanly(self, tmp_path):
        """Multiple saves to same session_id produce final state, no leftovers."""
        storage = AutonomyStorage(storage_path=tmp_path / "sessions", workspace_path=tmp_path)

        for i in range(5):
            session = make_session(
                session_id="integ-overwrite",
                spec_id=f"spec-{i}",
            )
            session.counters.tasks_completed = i
            storage.save(session)

        loaded = storage.load("integ-overwrite")
        assert loaded is not None
        assert loaded.counters.tasks_completed == 4
        assert loaded.spec_id == "spec-4"

        # No temp files left behind
        tmp_files = list(storage.storage_path.glob(".*tmp"))
        assert len(tmp_files) == 0


# =============================================================================
# Concurrent Writes with FileLock
# =============================================================================


class TestConcurrentWritesWithFileLock:
    """Test that concurrent file operations are safe with real locks."""

    def test_two_threads_writing_same_session(self, tmp_path):
        """Two threads racing to save the same session produce valid state."""
        storage = AutonomyStorage(storage_path=tmp_path / "sessions", workspace_path=tmp_path)

        errors: List[Exception] = []
        written_specs: List[str] = []

        def writer(thread_id: int) -> None:
            try:
                for i in range(20):
                    s = make_session(
                        session_id="concurrent-target",
                        spec_id=f"spec-t{thread_id}-i{i}",
                    )
                    s.counters.tasks_completed = thread_id * 100 + i
                    storage.save(s)
                    written_specs.append(s.spec_id)
            except Exception as e:
                errors.append(e)

        t1 = Thread(target=writer, args=(1,))
        t2 = Thread(target=writer, args=(2,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0, f"Errors during concurrent writes: {errors}"

        # Session must be loadable and valid
        loaded = storage.load("concurrent-target")
        assert loaded is not None
        assert loaded.id == "concurrent-target"

        # File must be valid JSON
        path = storage._get_session_path("concurrent-target")
        raw = json.loads(path.read_text())
        assert raw["id"] == "concurrent-target"

    def test_concurrent_read_write_no_corruption(self, tmp_path):
        """Concurrent reads while writing never return corrupted data."""
        storage = AutonomyStorage(storage_path=tmp_path / "sessions", workspace_path=tmp_path)

        # Seed initial session
        session = make_session(session_id="rw-target")
        storage.save(session)

        errors: List[Exception] = []
        load_results: List[bool] = []

        def reader() -> None:
            try:
                for _ in range(30):
                    loaded = storage.load("rw-target")
                    # Must be either valid session or None (never partial)
                    if loaded is not None:
                        load_results.append(True)
                        assert loaded.id == "rw-target"
                    else:
                        load_results.append(False)
            except Exception as e:
                errors.append(e)

        def writer() -> None:
            try:
                for i in range(30):
                    s = make_session(session_id="rw-target", spec_id=f"spec-{i}")
                    s.counters.tasks_completed = i
                    storage.save(s)
            except Exception as e:
                errors.append(e)

        t1 = Thread(target=reader)
        t2 = Thread(target=writer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0, f"Errors during concurrent read/write: {errors}"
        # At least some reads should succeed
        assert any(load_results)


# =============================================================================
# Corrupted State File Recovery
# =============================================================================


class TestCorruptedStateFileRecovery:
    """Validate graceful handling of corrupted state files."""

    def test_invalid_json_returns_none(self, tmp_path):
        """A file with invalid JSON returns None on load."""
        storage = AutonomyStorage(storage_path=tmp_path / "sessions", workspace_path=tmp_path)

        # Write garbage to a session file
        session_path = storage._get_session_path("corrupted-1")
        session_path.write_text("{{not valid json")

        loaded = storage.load("corrupted-1")
        assert loaded is None

    def test_invalid_json_raises_when_requested(self, tmp_path):
        """Invalid JSON raises SessionCorrupted with raise_on_corrupted=True."""
        storage = AutonomyStorage(storage_path=tmp_path / "sessions", workspace_path=tmp_path)

        session_path = storage._get_session_path("corrupted-2")
        session_path.write_text("not json at all")

        with pytest.raises(SessionCorrupted) as exc_info:
            storage.load("corrupted-2", raise_on_corrupted=True)

        assert exc_info.value.session_id == "corrupted-2"
        assert "Invalid JSON" in exc_info.value.reason

    def test_schema_validation_failure_returns_none(self, tmp_path):
        """Valid JSON but invalid schema returns None."""
        storage = AutonomyStorage(storage_path=tmp_path / "sessions", workspace_path=tmp_path)

        session_path = storage._get_session_path("bad-schema")
        # Valid JSON but missing required fields
        session_path.write_text(json.dumps({"some_key": "some_value"}))

        loaded = storage.load("bad-schema")
        assert loaded is None

    def test_schema_validation_failure_raises_when_requested(self, tmp_path):
        """Invalid schema raises SessionCorrupted with raise_on_corrupted=True."""
        storage = AutonomyStorage(storage_path=tmp_path / "sessions", workspace_path=tmp_path)

        session_path = storage._get_session_path("bad-schema-2")
        session_path.write_text(json.dumps({"incomplete": True}))

        with pytest.raises(SessionCorrupted) as exc_info:
            storage.load("bad-schema-2", raise_on_corrupted=True)

        assert exc_info.value.session_id == "bad-schema-2"
        assert "Schema validation failed" in exc_info.value.reason

    def test_corrupted_file_does_not_affect_other_sessions(self, tmp_path):
        """A corrupted file for one session doesn't affect loading others."""
        storage = AutonomyStorage(storage_path=tmp_path / "sessions", workspace_path=tmp_path)

        # Save a valid session
        valid_session = make_session(session_id="valid-sess")
        storage.save(valid_session)

        # Corrupt a different session file
        bad_path = storage._get_session_path("bad-sess")
        bad_path.write_text("corrupted!!!")

        # Valid session still loads fine
        loaded = storage.load("valid-sess")
        assert loaded is not None
        assert loaded.id == "valid-sess"

        # Corrupted one returns None
        assert storage.load("bad-sess") is None

    def test_empty_file_returns_none(self, tmp_path):
        """An empty file returns None."""
        storage = AutonomyStorage(storage_path=tmp_path / "sessions", workspace_path=tmp_path)

        session_path = storage._get_session_path("empty-sess")
        session_path.write_text("")

        loaded = storage.load("empty-sess")
        assert loaded is None


# =============================================================================
# GC Cleanup of Expired Files
# =============================================================================


class TestGCCleanupExpiredFiles:
    """Verify GC actually removes expired sessions and cleans up pointers/locks."""

    def test_gc_removes_expired_completed_session(self, tmp_path):
        """Completed session past TTL is removed by GC."""
        storage = AutonomyStorage(storage_path=tmp_path / "sessions", workspace_path=tmp_path)

        session = make_session(
            session_id="gc-completed",
            spec_id="spec-gc-1",
            status=SessionStatus.COMPLETED,
        )
        storage.save(session)
        storage.set_active_session("spec-gc-1", "gc-completed")

        # Backdate the file past TTL
        path = storage._get_session_path("gc-completed")
        ttl_days = GC_TTL_DAYS[SessionStatus.COMPLETED]
        old_mtime = (datetime.now(timezone.utc) - timedelta(days=ttl_days + 1)).timestamp()
        os.utime(path, (old_mtime, old_mtime))

        removed = storage.cleanup_expired()
        assert removed["sessions"] >= 1

        # Session file gone
        assert not path.exists()
        # Load returns None
        assert storage.load("gc-completed") is None

    def test_gc_removes_expired_ended_session(self, tmp_path):
        """Ended session past TTL is cleaned up."""
        storage = AutonomyStorage(storage_path=tmp_path / "sessions", workspace_path=tmp_path)

        session = make_session(session_id="gc-ended", status=SessionStatus.ENDED)
        storage.save(session)

        path = storage._get_session_path("gc-ended")
        ttl_days = GC_TTL_DAYS[SessionStatus.ENDED]
        old_mtime = (datetime.now(timezone.utc) - timedelta(days=ttl_days + 1)).timestamp()
        os.utime(path, (old_mtime, old_mtime))

        removed = storage.cleanup_expired()
        assert removed["sessions"] >= 1
        assert storage.load("gc-ended") is None

    def test_gc_preserves_running_sessions(self, tmp_path):
        """Running sessions are never expired regardless of age."""
        storage = AutonomyStorage(storage_path=tmp_path / "sessions", workspace_path=tmp_path)

        session = make_session(session_id="gc-running", status=SessionStatus.RUNNING)
        storage.save(session)

        # Backdate heavily
        path = storage._get_session_path("gc-running")
        old_mtime = (datetime.now(timezone.utc) - timedelta(days=365)).timestamp()
        os.utime(path, (old_mtime, old_mtime))

        removed = storage.cleanup_expired()
        assert removed["sessions"] == 0
        assert storage.load("gc-running") is not None

    def test_gc_cleans_orphaned_pointer_files(self, tmp_path):
        """Pointer files referencing deleted sessions are cleaned up."""
        storage = AutonomyStorage(storage_path=tmp_path / "sessions", workspace_path=tmp_path)

        # Create pointer for a session that doesn't exist
        storage.set_active_session("orphan-spec", "nonexistent-session")
        pointer_path = storage._get_pointer_path("orphan-spec")
        assert pointer_path.exists()

        removed = storage.cleanup_expired()
        assert removed["pointers"] >= 1
        assert not pointer_path.exists()

    def test_gc_cleans_orphaned_lock_files(self, tmp_path):
        """Lock files for deleted sessions are cleaned up."""
        storage = AutonomyStorage(storage_path=tmp_path / "sessions", workspace_path=tmp_path)

        # Create an orphaned lock file (no corresponding session)
        lock_path = storage._get_lock_path("orphan-sess")
        lock_path.write_text("")

        removed = storage.cleanup_expired()
        assert removed["locks"] >= 1
        assert not lock_path.exists()

    def test_gc_failed_session_has_longer_ttl(self, tmp_path):
        """Failed sessions have 30-day TTL, not removed at 7 days."""
        storage = AutonomyStorage(storage_path=tmp_path / "sessions", workspace_path=tmp_path)

        session = make_session(session_id="gc-failed", status=SessionStatus.FAILED)
        storage.save(session)

        path = storage._get_session_path("gc-failed")

        # Backdate to 8 days (past completed TTL but within failed TTL)
        mtime_8d = (datetime.now(timezone.utc) - timedelta(days=8)).timestamp()
        os.utime(path, (mtime_8d, mtime_8d))

        removed = storage.cleanup_expired()
        assert removed["sessions"] == 0  # Not yet expired
        assert storage.load("gc-failed") is not None

        # Now backdate past failed TTL (31 days)
        mtime_31d = (datetime.now(timezone.utc) - timedelta(days=31)).timestamp()
        os.utime(path, (mtime_31d, mtime_31d))

        removed = storage.cleanup_expired()
        assert removed["sessions"] >= 1
        assert storage.load("gc-failed") is None
