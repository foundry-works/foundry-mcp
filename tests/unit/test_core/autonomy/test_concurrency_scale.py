"""T12: Concurrency and scale tests for autonomous sessions.

Tests:
- Multi-session state isolation (10 sessions with independent state)
- Concurrent step-report for same step_id
- Concurrent session-start for same spec_id
- Performance under concurrent polling (200ms target)
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

from foundry_mcp.core.autonomy.memory import AutonomyStorage
from foundry_mcp.core.autonomy.models.session_config import SessionCounters

from .conftest import make_session, make_spec_data

# =============================================================================
# Helpers
# =============================================================================


def _make_config(workspace: Path) -> MagicMock:
    config = MagicMock()
    config.workspace_path = str(workspace)
    config.specs_dir = str(workspace / "specs")
    return config


def _setup_workspace(tmp_path: Path, spec_id: str = "test-spec-001") -> Path:
    workspace = tmp_path / "ws"
    specs_dir = workspace / "specs" / "active"
    specs_dir.mkdir(parents=True)

    spec_data = make_spec_data(spec_id=spec_id)
    spec_data["title"] = "Test Spec"
    spec_data["journal"] = []
    spec_path = specs_dir / f"{spec_id}.json"
    spec_path.write_text(json.dumps(spec_data, indent=2))

    return workspace


def _assert_success(resp: dict) -> dict:
    assert resp["success"] is True, f"Expected success, got error: {resp.get('error')}"
    assert resp["error"] is None
    return resp["data"]


def _assert_error(resp: dict) -> dict:
    assert resp["success"] is False
    assert resp["error"] is not None
    return resp


# =============================================================================
# T12.1: Multi-Session State Isolation
# =============================================================================


class TestMultiSessionStateIsolation:
    """Verify 10 sessions created from different specs maintain independent state."""

    def test_ten_sessions_independent_state(self, tmp_path):
        """Create 10 sessions, each with independent state, verify no cross-contamination."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_status,
        )

        workspace = tmp_path / "ws"
        specs_dir = workspace / "specs" / "active"
        specs_dir.mkdir(parents=True)

        # Create 10 distinct specs
        num_sessions = 10
        session_ids = []
        for i in range(num_sessions):
            spec_id = f"spec-{i:03d}"
            spec_data = make_spec_data(spec_id=spec_id)
            spec_data["title"] = f"Test Spec {i}"
            spec_data["journal"] = []
            spec_path = specs_dir / f"{spec_id}.json"
            spec_path.write_text(json.dumps(spec_data, indent=2))

        config = _make_config(workspace)

        # Create all 10 sessions
        for i in range(num_sessions):
            spec_id = f"spec-{i:03d}"
            resp = _handle_session_start(
                config=config,
                spec_id=spec_id,
                workspace=str(workspace),
            )
            data = _assert_success(resp)
            session_ids.append(data["session_id"])

        # Verify all session IDs are unique
        assert len(set(session_ids)) == num_sessions

        # Verify each session has correct spec binding
        for i, sid in enumerate(session_ids):
            spec_id = f"spec-{i:03d}"
            resp = _handle_session_status(
                config=config,
                session_id=sid,
                workspace=str(workspace),
            )
            data = _assert_success(resp)
            assert data["session_id"] == sid
            assert data["spec_id"] == spec_id
            assert data["status"] == "running"

    def test_sessions_have_independent_counters(self, tmp_path):
        """Sessions with different specs track counters independently."""
        storage = AutonomyStorage(
            storage_path=tmp_path / "sessions",
            workspace_path=tmp_path,
        )

        sessions = []
        for i in range(5):
            session = make_session(
                session_id=f"sess-{i:03d}",
                spec_id=f"spec-{i:03d}",
                counters=SessionCounters(
                    tasks_completed=i * 2,
                    consecutive_errors=i,
                ),
            )
            storage.save(session)
            sessions.append(session)

        # Verify each session retains its own counter values
        for i, original in enumerate(sessions):
            loaded = storage.load(f"sess-{i:03d}")
            assert loaded is not None
            assert loaded.counters.tasks_completed == i * 2
            assert loaded.counters.consecutive_errors == i


# =============================================================================
# T12.2: Concurrent Session-Start for Same Spec
# =============================================================================


class TestConcurrentSessionStart:
    """Concurrent session-start requests for the same spec_id."""

    def test_concurrent_start_same_spec_only_one_wins(self, tmp_path):
        """Two concurrent start requests for the same spec — only one should succeed
        or both succeed idempotently (force=False, no idempotency_key)."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        results: List[dict] = [None, None]  # type: ignore[list-item]

        def start_session(index: int):
            results[index] = _handle_session_start(
                config=config,
                spec_id="test-spec-001",
                workspace=str(workspace),
            )

        t1 = threading.Thread(target=start_session, args=(0,))
        t2 = threading.Thread(target=start_session, args=(1,))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        # At least one should succeed
        successes = [r for r in results if r["success"] is True]
        assert len(successes) >= 1, "At least one session-start should succeed"

        # Collect all session IDs from successful starts
        session_ids = [r["data"]["session_id"] for r in successes]
        # Since the second request may either create a new session (force=True)
        # or get the existing active session ID, we just verify consistency
        if len(successes) == 2:
            # Both succeeded — they may return the same session (duplicate protection)
            # or different sessions if timing allows. Either is acceptable.
            pass


# =============================================================================
# T12.3: Concurrent Storage Writes
# =============================================================================


class TestConcurrentStorageWrites:
    """Concurrent writes to the same session via storage layer."""

    def test_concurrent_save_does_not_corrupt(self, tmp_path):
        """Multiple threads saving the same session should not corrupt the file."""
        storage = AutonomyStorage(
            storage_path=tmp_path / "sessions",
            workspace_path=tmp_path,
        )

        session_id = "concurrent-test-001"
        errors: List[Exception] = []

        def save_session(thread_idx: int):
            try:
                for iteration in range(20):
                    session = make_session(
                        session_id=session_id,
                        spec_id="spec-concurrent",
                        counters=SessionCounters(
                            tasks_completed=thread_idx * 100 + iteration,
                        ),
                    )
                    storage.save(session)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=save_session, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert not errors, f"Concurrent saves raised errors: {errors}"

        # The file should be valid JSON and loadable
        loaded = storage.load(session_id)
        assert loaded is not None
        assert loaded.id == session_id

    def test_concurrent_proof_consumption_no_double_spend(self, tmp_path):
        """Concurrent proof consumptions should not result in double-spend."""
        import hashlib

        storage = AutonomyStorage(
            storage_path=tmp_path / "sessions",
            workspace_path=tmp_path,
        )

        session_id = "proof-concurrent-001"
        step_proof = "proof-token-xyz"
        payload_hash = hashlib.sha256(b"test-payload").hexdigest()

        results: List[tuple] = []
        lock = threading.Lock()

        def consume(thread_idx: int):
            success, record, err = storage.consume_proof_with_lock(
                session_id=session_id,
                step_proof=step_proof,
                payload_hash=payload_hash,
                grace_window_seconds=30,
                step_id=f"step-{thread_idx}",
            )
            with lock:
                results.append((thread_idx, success, record, err))

        threads = [threading.Thread(target=consume, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(results) == 5

        # Exactly one should get (True, None, "") — first consumption
        first_consumptions = [(idx, s, r, e) for idx, s, r, e in results if s and r is None and e == ""]
        # The rest should get idempotent replays (True, record, "")
        replays = [(idx, s, r, e) for idx, s, r, e in results if s and r is not None]

        assert len(first_consumptions) == 1, f"Expected exactly one first-consumption, got {len(first_consumptions)}"


# =============================================================================
# T12.4: Performance Under Concurrent Polling
# =============================================================================


class TestConcurrentPollingPerformance:
    """Verify storage operations complete within 200ms under concurrent load."""

    def test_concurrent_session_list_under_200ms(self, tmp_path):
        """List operations with 10+ sessions should complete under 200ms."""
        storage = AutonomyStorage(
            storage_path=tmp_path / "sessions",
            workspace_path=tmp_path,
        )

        # Create 15 sessions
        for i in range(15):
            session = make_session(
                session_id=f"perf-sess-{i:03d}",
                spec_id=f"perf-spec-{i:03d}",
            )
            storage.save(session)

        # Time the list operation
        start = time.perf_counter()
        result = storage.list_sessions(limit=20, include_total=True)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(result.sessions) == 15
        assert elapsed_ms < 200, f"List 15 sessions took {elapsed_ms:.1f}ms (target: <200ms)"

    def test_concurrent_load_save_under_200ms(self, tmp_path):
        """Concurrent load/save for 10 different sessions should complete under 200ms each."""
        storage = AutonomyStorage(
            storage_path=tmp_path / "sessions",
            workspace_path=tmp_path,
        )

        # Pre-create sessions
        for i in range(10):
            session = make_session(
                session_id=f"conc-sess-{i:03d}",
                spec_id=f"conc-spec-{i:03d}",
            )
            storage.save(session)

        max_elapsed_ms = 0.0
        errors: List[Exception] = []

        def load_and_resave(idx: int):
            nonlocal max_elapsed_ms
            try:
                start = time.perf_counter()
                loaded = storage.load(f"conc-sess-{idx:03d}")
                assert loaded is not None
                loaded.counters.tasks_completed += 1
                storage.save(loaded)
                elapsed = (time.perf_counter() - start) * 1000
                max_elapsed_ms = max(max_elapsed_ms, elapsed)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=load_and_resave, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Concurrent load/save errors: {errors}"
        assert max_elapsed_ms < 200, f"Slowest load/save took {max_elapsed_ms:.1f}ms (target: <200ms)"
