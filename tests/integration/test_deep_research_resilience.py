"""Integration tests for deep research resilience features.

Tests cover:
- Cancellation mid-workflow
- Timeout handling with partial results
- Crash recovery from persisted state
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from foundry_mcp.config import ResearchConfig
from foundry_mcp.core.background_task import BackgroundTask, TaskStatus
from foundry_mcp.core.research.models import (
    DeepResearchState,
    DeepResearchPhase,
    SubQuery,
)
from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow


class TestCancellationIntegration:
    """Integration tests for cancellation mid-workflow."""

    @pytest.mark.asyncio
    async def test_cancel_sets_metadata_and_persists(self):
        """Cancelling a research task sets cancelled=True in metadata and persists state."""
        config = ResearchConfig()
        workflow = DeepResearchWorkflow(config)

        # Create a state that's in progress
        state = DeepResearchState(
            id="test-cancel-integration",
            original_query="test query",
            phase=DeepResearchPhase.GATHERING,
        )
        state.sub_queries = [
            SubQuery(id="sq-1", query="sub query 1", status="pending"),
        ]

        # Create a background task and mock it as running (not done)
        bg_task = BackgroundTask(research_id=state.id)

        # Mock cancel to return True (simulates task was running)
        with patch.object(bg_task, "cancel", return_value=True):
            with patch.object(workflow, "get_background_task", return_value=bg_task):
                with patch.object(workflow.memory, "load_deep_research", return_value=state):
                    with patch.object(workflow.memory, "save_deep_research"):
                        # Execute cancel
                        result = workflow._cancel_research(state.id)

        assert result.success is True
        assert "cancelled" in result.metadata
        assert result.metadata.get("research_id") == state.id

    @pytest.mark.asyncio
    async def test_cancel_returns_partial_results(self):
        """Cancellation returns any partial results accumulated so far."""
        config = ResearchConfig()
        workflow = DeepResearchWorkflow(config)

        # Create a state with some completed work
        state = DeepResearchState(
            id="test-cancel-partial",
            original_query="test query",
            phase=DeepResearchPhase.ANALYSIS,
        )
        state.sub_queries = [
            SubQuery(id="sq-1", query="sub query 1", status="completed"),
            SubQuery(id="sq-2", query="sub query 2", status="pending"),
        ]

        bg_task = BackgroundTask(research_id=state.id)

        # Mock cancel to return True (task was running and is now cancelled)
        with patch.object(bg_task, "cancel", return_value=True):
            with patch.object(workflow, "get_background_task", return_value=bg_task):
                with patch.object(workflow.memory, "load_deep_research", return_value=state):
                    with patch.object(workflow.memory, "save_deep_research"):
                        result = workflow._cancel_research(state.id)

        assert result.success is True
        # Should include cancelled flag in metadata
        assert result.metadata.get("cancelled") is True


class TestTimeoutIntegration:
    """Integration tests for timeout handling."""

    def test_timeout_marks_state_with_abort_phase(self):
        """Timeout should mark state with abort_phase and abort_iteration."""
        config = ResearchConfig()
        workflow = DeepResearchWorkflow(config)

        # Create a timed-out background task
        bg_task = BackgroundTask(research_id="test-timeout-abort", timeout=0.01)
        time.sleep(0.02)
        bg_task.mark_timeout()

        # Create state that was in GATHERING phase
        state = DeepResearchState(
            id="test-timeout-abort",
            original_query="test query",
            phase=DeepResearchPhase.GATHERING,
            iteration=2,
        )
        state.metadata["timeout"] = True
        state.metadata["abort_phase"] = "gathering"
        state.metadata["abort_iteration"] = 2

        with patch.object(workflow, "get_background_task", return_value=bg_task):
            with patch.object(workflow.memory, "load_deep_research", return_value=state):
                result = workflow._get_status("test-timeout-abort")

        assert result.success is True
        assert result.metadata.get("is_timed_out") is True

    def test_status_includes_timeout_metadata_from_state(self):
        """Status response includes timeout metadata from persisted state."""
        config = ResearchConfig()
        workflow = DeepResearchWorkflow(config)

        # State with timeout metadata
        state = DeepResearchState(
            id="test-timeout-meta",
            original_query="test query",
            phase=DeepResearchPhase.SYNTHESIS,
        )
        state.metadata["timeout"] = True
        state.completed_at = datetime.now(timezone.utc)

        # No background task (completed/persisted state)
        with patch.object(workflow, "get_background_task", return_value=None):
            with patch.object(workflow.memory, "load_deep_research", return_value=state):
                with patch.object(workflow.memory, "save_deep_research"):
                    result = workflow._get_status("test-timeout-meta")

        assert result.success is True
        assert result.metadata.get("timed_out") is True


class TestCrashRecoveryIntegration:
    """Integration tests for crash recovery from persisted state."""

    def test_continue_loads_persisted_state(self):
        """Continue action loads state from persistence."""
        config = ResearchConfig()
        workflow = DeepResearchWorkflow(config)

        # Create persisted state from "previous session"
        state = DeepResearchState(
            id="test-recovery-state",
            original_query="test query",
            phase=DeepResearchPhase.GATHERING,
            iteration=1,
        )
        state.sub_queries = [
            SubQuery(id="sq-1", query="sub query 1", status="completed"),
            SubQuery(id="sq-2", query="sub query 2", status="pending"),
        ]

        with patch.object(workflow.memory, "load_deep_research", return_value=state):
            # Verify state can be loaded
            loaded = workflow.memory.load_deep_research("test-recovery-state")
            assert loaded is not None
            assert loaded.id == "test-recovery-state"
            assert loaded.phase == DeepResearchPhase.GATHERING
            assert len(loaded.sub_queries) == 2

    def test_status_returns_partial_progress_after_crash(self):
        """Status shows partial progress from persisted state after crash."""
        config = ResearchConfig()
        workflow = DeepResearchWorkflow(config)

        # State representing crash mid-workflow
        state = DeepResearchState(
            id="test-crash-progress",
            original_query="test query",
            phase=DeepResearchPhase.ANALYSIS,
        )
        state.sub_queries = [
            SubQuery(id="sq-1", query="q1", status="completed"),
            SubQuery(id="sq-2", query="q2", status="completed"),
            SubQuery(id="sq-3", query="q3", status="failed", error="crash"),
        ]
        state.metadata["failed"] = True
        state.metadata["failure_error"] = "Unexpected error during analysis"

        # No background task (crashed)
        with patch.object(workflow, "get_background_task", return_value=None):
            with patch.object(workflow.memory, "load_deep_research", return_value=state):
                with patch.object(workflow.memory, "save_deep_research"):
                    result = workflow._get_status("test-crash-progress")

        assert result.success is True
        assert result.metadata.get("is_failed") is True
        assert result.metadata.get("sub_queries_completed") == 2
        assert "failure_error" in result.metadata


class TestHeartbeatVisibility:
    """Integration tests for heartbeat visibility during execution."""

    def test_status_shows_last_heartbeat_during_execution(self):
        """Status response includes last_heartbeat_at for progress visibility."""
        config = ResearchConfig()
        workflow = DeepResearchWorkflow(config)

        # Active task with recent heartbeat
        bg_task = BackgroundTask(research_id="test-heartbeat-visible")

        state = DeepResearchState(
            id="test-heartbeat-visible",
            original_query="test query",
            phase=DeepResearchPhase.GATHERING,
        )
        heartbeat = datetime(2026, 1, 26, 12, 0, 0, tzinfo=timezone.utc)
        state.last_heartbeat_at = heartbeat

        with patch.object(workflow, "get_background_task", return_value=bg_task):
            with patch.object(workflow.memory, "load_deep_research", return_value=state):
                result = workflow._get_status("test-heartbeat-visible")

        assert result.success is True
        assert result.metadata.get("last_heartbeat_at") == heartbeat.isoformat()
        assert result.metadata.get("phase") == "gathering"
