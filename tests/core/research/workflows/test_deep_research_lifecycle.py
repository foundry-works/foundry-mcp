"""Tests for deep research thread safety and shutdown (Phase 3).

Tests cancellation event flags, graceful SIGTERM shutdown, and
status distinction between CANCELLED, INTERRUPTED, and FAILED.
"""

from __future__ import annotations

import signal
import threading
from unittest.mock import MagicMock, patch

import pytest

from foundry_mcp.core.background_task import BackgroundTask, TaskStatus
from foundry_mcp.core.research.models.deep_research import (
    DeepResearchPhase,
    DeepResearchState,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_state():
    """Create a minimal DeepResearchState for lifecycle tests."""
    return DeepResearchState(
        id="lifecycle-test-001",
        original_query="What is quantum computing?",
        phase=DeepResearchPhase.GATHERING,
        iteration=1,
        max_iterations=3,
    )


@pytest.fixture
def sample_state_planning():
    """Create a state in BRIEF phase."""
    return DeepResearchState(
        id="lifecycle-test-002",
        original_query="Explain reinforcement learning",
        phase=DeepResearchPhase.BRIEF,
        iteration=1,
        max_iterations=3,
    )


@pytest.fixture
def background_task():
    """Create a BackgroundTask with a mock thread."""
    task = BackgroundTask(research_id="lifecycle-test-001")
    mock_thread = MagicMock(spec=threading.Thread)
    mock_thread.is_alive.return_value = True
    mock_thread.name = "deep-research-lifecycle"
    task.thread = mock_thread
    return task


# =============================================================================
# 3a. Cancellation Event Flag Tests
# =============================================================================


class TestCancellationEventFlag:
    """Tests for per-session cancellation via BackgroundTask._cancel_event."""

    def test_cancel_event_set_on_cancel(self, background_task: BackgroundTask):
        """Cancel should set the threading Event so phase checks detect it."""
        assert not background_task._cancel_event.is_set()
        background_task.cancel(timeout=0)
        assert background_task._cancel_event.is_set()

    def test_is_cancelled_reflects_event(self, background_task: BackgroundTask):
        """is_cancelled property should reflect the cancel event state."""
        assert not background_task.is_cancelled
        background_task._cancel_event.set()
        assert background_task.is_cancelled

    def test_cancel_sets_cancelled_status(self, background_task: BackgroundTask):
        """cancel() should transition status to CANCELLED."""
        background_task.cancel(timeout=0)
        assert background_task.status == TaskStatus.CANCELLED
        assert background_task.completed_at is not None

    def test_cancel_on_completed_task_returns_false(self):
        """cancel() on an already-done task should return False."""
        task = BackgroundTask(research_id="done-task")
        mock_thread = MagicMock(spec=threading.Thread)
        mock_thread.is_alive.return_value = False
        task.thread = mock_thread
        assert task.cancel() is False

    def test_mark_cancelled_on_state(self, sample_state: DeepResearchState):
        """mark_cancelled should set terminal_status and metadata."""
        assert sample_state.completed_at is None
        sample_state.mark_cancelled(phase_state="phase=gathering, iteration=1")

        assert sample_state.completed_at is not None
        assert sample_state.metadata["cancelled"] is True
        assert sample_state.metadata["terminal_status"] == "cancelled"
        assert sample_state.metadata["cancelled_phase_state"] == "phase=gathering, iteration=1"

    def test_mark_cancelled_without_phase_state(self, sample_state: DeepResearchState):
        """mark_cancelled without phase_state should still work."""
        sample_state.mark_cancelled()
        assert sample_state.metadata["cancelled"] is True
        assert sample_state.metadata["terminal_status"] == "cancelled"
        assert "cancelled_phase_state" not in sample_state.metadata


# =============================================================================
# 3a. Cancel Detection at Phase Boundaries
# =============================================================================


class TestCancelDetectionAtPhaseBoundaries:
    """Tests that _check_cancellation raises CancelledError when event is set."""

    def test_check_cancellation_raises_when_cancelled(self):
        """_check_cancellation should raise CancelledError when task is cancelled."""
        import asyncio

        from foundry_mcp.core.research.workflows.deep_research.workflow_execution import (
            WorkflowExecutionMixin,
        )

        # Create a minimal mixin instance with required attributes
        mixin = WorkflowExecutionMixin()
        mixin._tasks = {}
        mixin._tasks_lock = threading.Lock()

        state = DeepResearchState(
            id="cancel-check-test",
            original_query="test",
            phase=DeepResearchPhase.GATHERING,
        )

        # Create a cancelled background task
        bg_task = BackgroundTask(research_id="cancel-check-test")
        bg_task._cancel_event.set()
        mixin._tasks["cancel-check-test"] = bg_task

        with pytest.raises(asyncio.CancelledError):
            mixin._check_cancellation(state)

    def test_check_cancellation_passes_when_not_cancelled(self):
        """_check_cancellation should not raise when no cancellation."""
        from foundry_mcp.core.research.workflows.deep_research.workflow_execution import (
            WorkflowExecutionMixin,
        )

        mixin = WorkflowExecutionMixin()
        mixin._tasks = {}
        mixin._tasks_lock = threading.Lock()

        state = DeepResearchState(
            id="no-cancel-test",
            original_query="test",
            phase=DeepResearchPhase.GATHERING,
        )

        # Create a non-cancelled background task
        bg_task = BackgroundTask(research_id="no-cancel-test")
        mixin._tasks["no-cancel-test"] = bg_task

        # Should not raise
        mixin._check_cancellation(state)

    def test_check_cancellation_passes_when_no_task(self):
        """_check_cancellation should not raise when no background task exists."""
        from foundry_mcp.core.research.workflows.deep_research.workflow_execution import (
            WorkflowExecutionMixin,
        )

        mixin = WorkflowExecutionMixin()
        mixin._tasks = {}
        mixin._tasks_lock = threading.Lock()

        state = DeepResearchState(
            id="orphan-test",
            original_query="test",
            phase=DeepResearchPhase.BRIEF,
        )

        # No task registered — should not raise
        mixin._check_cancellation(state)


# =============================================================================
# 3a. Cancel on Already-Completed Session
# =============================================================================


class TestCancelOnCompletedSession:
    """Tests that cancel on an already-completed session is a no-op."""

    def test_cancel_research_completed_returns_error(self):
        """_cancel_research on a completed task returns an appropriate error."""
        from foundry_mcp.core.research.workflows.deep_research.action_handlers import (
            ActionHandlersMixin,
        )

        mixin = ActionHandlersMixin()
        mixin.memory = MagicMock()

        # Mock get_background_task to return None (no running task)
        mixin.get_background_task = MagicMock(return_value=None)

        result = mixin._cancel_research(research_id="completed-session")
        assert not result.success
        assert result.error and "No running task found" in result.error

    def test_cancel_research_already_done(self):
        """cancel() on an already-completed BackgroundTask returns False."""
        task = BackgroundTask(research_id="already-done")
        task.mark_completed(result="done")
        # No thread/task — should return False
        assert task.cancel() is False


# =============================================================================
# 3b. SIGTERM Handler Tests
# =============================================================================


class TestSigtermHandler:
    """Tests for the SIGTERM graceful shutdown handler."""

    def test_sigterm_handler_sets_cancel_events(self, sample_state: DeepResearchState):
        """SIGTERM handler should set cancel events on all active tasks."""
        from foundry_mcp.core import task_registry
        from foundry_mcp.core.research.workflows.deep_research.infrastructure import (
            _active_research_sessions,
            _active_sessions_lock,
            _sigterm_handler,
        )

        # Register an active session
        bg_task = BackgroundTask(research_id=sample_state.id)
        task_registry.register(bg_task)

        with _active_sessions_lock:
            _active_research_sessions[sample_state.id] = sample_state

        try:
            # Invoke the SIGTERM handler directly
            _sigterm_handler(signal.SIGTERM, None)

            # Verify cancel event was set
            assert bg_task._cancel_event.is_set()
        finally:
            # Cleanup
            with _active_sessions_lock:
                _active_research_sessions.pop(sample_state.id, None)
            task_registry.remove(sample_state.id)

    def test_sigterm_handler_marks_sessions_interrupted(self, sample_state: DeepResearchState):
        """SIGTERM handler should mark active sessions as INTERRUPTED."""
        from foundry_mcp.core.research.workflows.deep_research.infrastructure import (
            _active_research_sessions,
            _active_sessions_lock,
            _sigterm_handler,
        )

        assert sample_state.completed_at is None

        with _active_sessions_lock:
            _active_research_sessions[sample_state.id] = sample_state

        try:
            with patch("foundry_mcp.core.research.workflows.deep_research.infrastructure._persist_active_sessions"):
                _sigterm_handler(signal.SIGTERM, None)

            assert sample_state.metadata["interrupted"] is True
            assert sample_state.metadata["terminal_status"] == "interrupted"
            assert sample_state.metadata["interrupt_reason"] == "SIGTERM"
            assert sample_state.metadata["interrupt_phase"] == "gathering"
            assert sample_state.metadata["interrupt_iteration"] == 1
            assert sample_state.completed_at is not None
        finally:
            with _active_sessions_lock:
                _active_research_sessions.pop(sample_state.id, None)

    def test_sigterm_handler_skips_completed_sessions(self):
        """SIGTERM handler should skip already-completed sessions."""
        from foundry_mcp.core.research.workflows.deep_research.infrastructure import (
            _active_research_sessions,
            _active_sessions_lock,
            _sigterm_handler,
        )

        state = DeepResearchState(
            id="completed-sigterm-test",
            original_query="test",
            phase=DeepResearchPhase.SYNTHESIS,
        )
        state.mark_completed(report="Final report")
        original_completed_at = state.completed_at

        with _active_sessions_lock:
            _active_research_sessions[state.id] = state

        try:
            with patch("foundry_mcp.core.research.workflows.deep_research.infrastructure._persist_active_sessions"):
                _sigterm_handler(signal.SIGTERM, None)

            # Should NOT have been modified (completed_at was already set)
            assert state.completed_at == original_completed_at
            assert "interrupted" not in state.metadata or not state.metadata.get("interrupted")
        finally:
            with _active_sessions_lock:
                _active_research_sessions.pop(state.id, None)

    def test_sigterm_handler_multiple_sessions(self):
        """SIGTERM handler should cancel all active sessions."""
        from foundry_mcp.core import task_registry
        from foundry_mcp.core.research.workflows.deep_research.infrastructure import (
            _active_research_sessions,
            _active_sessions_lock,
            _sigterm_handler,
        )

        states = []
        tasks = []
        for i in range(3):
            state = DeepResearchState(
                id=f"multi-sigterm-{i}",
                original_query=f"query {i}",
                phase=DeepResearchPhase.GATHERING,
            )
            bg_task = BackgroundTask(research_id=state.id)
            task_registry.register(bg_task)
            states.append(state)
            tasks.append(bg_task)

        with _active_sessions_lock:
            for s in states:
                _active_research_sessions[s.id] = s

        try:
            with patch("foundry_mcp.core.research.workflows.deep_research.infrastructure._persist_active_sessions"):
                _sigterm_handler(signal.SIGTERM, None)

            for bg_task in tasks:
                assert bg_task._cancel_event.is_set()
            for state in states:
                assert state.metadata["interrupted"] is True
                assert state.metadata["terminal_status"] == "interrupted"
        finally:
            with _active_sessions_lock:
                for s in states:
                    _active_research_sessions.pop(s.id, None)
            for s in states:
                task_registry.remove(s.id)

    def test_sigterm_handler_no_active_sessions(self):
        """SIGTERM handler with no sessions should be a safe no-op."""
        from foundry_mcp.core.research.workflows.deep_research.infrastructure import (
            _active_research_sessions,
            _active_sessions_lock,
            _sigterm_handler,
        )

        # Ensure no sessions
        with _active_sessions_lock:
            _active_research_sessions.clear()

        # Should not raise
        with patch("foundry_mcp.core.research.workflows.deep_research.infrastructure._persist_active_sessions"):
            _sigterm_handler(signal.SIGTERM, None)

    def test_sigterm_chains_previous_handler(self):
        """SIGTERM handler should chain to previous handler if callable."""
        from foundry_mcp.core.research.workflows.deep_research import infrastructure

        previous_called = []
        original_previous = infrastructure._previous_sigterm_handler

        def mock_previous(signum, frame):
            previous_called.append(signum)

        infrastructure._previous_sigterm_handler = mock_previous

        try:
            with patch("foundry_mcp.core.research.workflows.deep_research.infrastructure._persist_active_sessions"):
                infrastructure._sigterm_handler(signal.SIGTERM, None)

            assert signal.SIGTERM in previous_called
        finally:
            infrastructure._previous_sigterm_handler = original_previous


# =============================================================================
# 3b. INTERRUPTED vs CANCELLED vs FAILED Status Distinction
# =============================================================================


class TestStatusDistinction:
    """Tests that INTERRUPTED, CANCELLED, and FAILED are distinguishable."""

    def test_interrupted_status_metadata(self):
        """INTERRUPTED state should have distinct metadata markers."""
        state = DeepResearchState(
            id="status-interrupted",
            original_query="test",
            phase=DeepResearchPhase.SYNTHESIS,
            iteration=2,
        )
        state.mark_interrupted(reason="SIGTERM")

        assert state.metadata["interrupted"] is True
        assert state.metadata["terminal_status"] == "interrupted"
        assert state.metadata["interrupt_reason"] == "SIGTERM"
        assert state.metadata["interrupt_phase"] == "synthesis"
        assert state.metadata["interrupt_iteration"] == 2
        assert state.completed_at is not None
        # Should NOT have cancelled or failed markers
        assert "cancelled" not in state.metadata
        assert "failed" not in state.metadata

    def test_cancelled_status_metadata(self):
        """CANCELLED state should have distinct metadata markers."""
        state = DeepResearchState(
            id="status-cancelled",
            original_query="test",
            phase=DeepResearchPhase.GATHERING,
            iteration=1,
        )
        state.mark_cancelled(phase_state="phase=gathering, iteration=1")

        assert state.metadata["cancelled"] is True
        assert state.metadata["terminal_status"] == "cancelled"
        assert state.completed_at is not None
        # Should NOT have interrupted or failed markers
        assert "interrupted" not in state.metadata
        assert "failed" not in state.metadata

    def test_failed_status_metadata(self):
        """FAILED state should have distinct metadata markers."""
        state = DeepResearchState(
            id="status-failed",
            original_query="test",
            phase=DeepResearchPhase.SYNTHESIS,
        )
        state.mark_failed("Provider connection error")

        assert state.metadata["failed"] is True
        assert state.metadata["failure_error"] == "Provider connection error"
        assert state.completed_at is not None
        # Should NOT have interrupted or cancelled markers
        assert "interrupted" not in state.metadata
        assert "cancelled" not in state.metadata

    def test_all_three_statuses_distinguishable(self):
        """All three terminal statuses should be mutually distinguishable."""
        interrupted = DeepResearchState(id="s1", original_query="q1")
        cancelled = DeepResearchState(id="s2", original_query="q2")
        failed = DeepResearchState(id="s3", original_query="q3")

        interrupted.mark_interrupted(reason="SIGTERM")
        cancelled.mark_cancelled()
        failed.mark_failed("error")

        # Each should have a unique terminal_status
        assert interrupted.metadata.get("terminal_status") == "interrupted"
        assert cancelled.metadata.get("terminal_status") == "cancelled"
        assert failed.metadata.get("terminal_status") is None  # mark_failed doesn't set terminal_status

        # Each should have unique boolean flags
        assert interrupted.metadata.get("interrupted") is True
        assert interrupted.metadata.get("cancelled") is None
        assert interrupted.metadata.get("failed") is None

        assert cancelled.metadata.get("cancelled") is True
        assert cancelled.metadata.get("interrupted") is None
        assert cancelled.metadata.get("failed") is None

        assert failed.metadata.get("failed") is True
        assert failed.metadata.get("cancelled") is None
        assert failed.metadata.get("interrupted") is None


# =============================================================================
# 3b. Cleanup on Exit
# =============================================================================


class TestCleanupOnExit:
    """Tests for atexit cleanup handler."""

    def test_cleanup_marks_active_sessions_interrupted(self):
        """_cleanup_on_exit should mark active sessions as interrupted."""
        from foundry_mcp.core.research.workflows.deep_research.infrastructure import (
            _active_research_sessions,
            _active_sessions_lock,
            _cleanup_on_exit,
        )

        state = DeepResearchState(
            id="cleanup-test",
            original_query="test",
            phase=DeepResearchPhase.GATHERING,
        )

        with _active_sessions_lock:
            _active_research_sessions[state.id] = state

        try:
            with patch("foundry_mcp.core.research.workflows.deep_research.infrastructure._persist_active_sessions"):
                _cleanup_on_exit()

            assert state.metadata["interrupted"] is True
            assert state.metadata["terminal_status"] == "interrupted"
            assert state.metadata["interrupt_reason"] == "process_exit"
        finally:
            with _active_sessions_lock:
                _active_research_sessions.pop(state.id, None)


# =============================================================================
# Install Handler Tests
# =============================================================================


class TestInstallCrashHandler:
    """Tests for install_crash_handler idempotency and SIGTERM registration."""

    def test_install_is_idempotent(self):
        """install_crash_handler should only install once."""
        from foundry_mcp.core.research.workflows.deep_research import infrastructure

        # It's already installed (module-level side effect in core.py)
        assert infrastructure._crash_handler_installed is True
        # Calling again should be a no-op
        infrastructure.install_crash_handler()
        assert infrastructure._crash_handler_installed is True
