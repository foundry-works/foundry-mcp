"""Tests for status response timeout metadata.

Verifies that the deep-research-status response includes timeout/staleness
metadata when a task is timed out or stale.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from foundry_mcp.config import ResearchConfig
from foundry_mcp.core.background_task import BackgroundTask, TaskStatus
from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow


class TestStatusResponseTimeoutMetadata:
    """Tests for timeout metadata in status response."""

    def test_status_includes_timeout_metadata_when_timed_out(self):
        """Status response includes is_timed_out and related metadata when task times out."""
        config = ResearchConfig()
        workflow = DeepResearchWorkflow(config)

        # Create a timed-out background task
        bg_task = BackgroundTask(research_id="test-timeout-status", timeout=0.01)
        # Wait for timeout
        time.sleep(0.02)
        # Mark timeout (this sets timed_out_at and timeout_elapsed_seconds)
        bg_task.mark_timeout()

        # Mock the workflow methods to return our task
        with patch.object(workflow, "get_background_task", return_value=bg_task):
            with patch.object(workflow.memory, "load_deep_research", return_value=None):
                result = workflow._get_status("test-timeout-status")

        assert result.success is True
        assert result.metadata["is_timed_out"] is True
        assert result.metadata["timeout_configured"] == 0.01
        assert "timed_out_at" in result.metadata
        assert "timeout_elapsed_seconds" in result.metadata
        assert result.metadata["timeout_elapsed_seconds"] >= 0.01

    def test_status_includes_timeout_metadata_when_status_is_timeout(self):
        """Status response includes is_timed_out when task status is TIMEOUT."""
        config = ResearchConfig()
        workflow = DeepResearchWorkflow(config)

        # Create a task with TIMEOUT status
        bg_task = BackgroundTask(research_id="test-status-timeout", timeout=1.0)
        bg_task.mark_timeout()
        assert bg_task.status == TaskStatus.TIMEOUT

        with patch.object(workflow, "get_background_task", return_value=bg_task):
            with patch.object(workflow.memory, "load_deep_research", return_value=None):
                result = workflow._get_status("test-status-timeout")

        assert result.success is True
        assert result.metadata["is_timed_out"] is True
        assert result.metadata["task_status"] == "timeout"

    def test_status_no_timeout_metadata_when_not_timed_out(self):
        """Status response does not include is_timed_out when task is running normally."""
        config = ResearchConfig()
        workflow = DeepResearchWorkflow(config)

        # Create a running task that hasn't timed out
        bg_task = BackgroundTask(research_id="test-normal-status", timeout=60.0)
        assert not bg_task.is_timed_out

        with patch.object(workflow, "get_background_task", return_value=bg_task):
            with patch.object(workflow.memory, "load_deep_research", return_value=None):
                result = workflow._get_status("test-normal-status")

        assert result.success is True
        assert "is_timed_out" not in result.metadata
        assert "timeout_configured" not in result.metadata


class TestStatusResponseStalenessMetadata:
    """Tests for staleness metadata in status response."""

    def test_status_includes_staleness_metadata_when_stale(self):
        """Status response includes is_stale when task is stale."""
        config = ResearchConfig()
        workflow = DeepResearchWorkflow(config)

        # Create a task and make it stale by backdating last_activity
        bg_task = BackgroundTask(research_id="test-stale-status")
        # Backdate last_activity to make it stale (more than 300s ago)
        bg_task.last_activity = time.time() - 400

        assert bg_task.is_stale(300.0)

        with patch.object(workflow, "get_background_task", return_value=bg_task):
            with patch.object(workflow.memory, "load_deep_research", return_value=None):
                result = workflow._get_status("test-stale-status")

        assert result.success is True
        assert result.metadata["is_stale"] is True
        assert "last_activity" in result.metadata

    def test_status_no_staleness_metadata_when_active(self):
        """Status response does not include is_stale when task is active."""
        config = ResearchConfig()
        workflow = DeepResearchWorkflow(config)

        # Create an active task (just started)
        bg_task = BackgroundTask(research_id="test-active-status")
        assert not bg_task.is_stale(300.0)

        with patch.object(workflow, "get_background_task", return_value=bg_task):
            with patch.object(workflow.memory, "load_deep_research", return_value=None):
                result = workflow._get_status("test-active-status")

        assert result.success is True
        assert "is_stale" not in result.metadata


class TestStatusResponseBasicMetadata:
    """Tests for basic metadata always present in status response."""

    def test_status_always_includes_basic_metadata(self):
        """Status response always includes research_id, task_status, elapsed_ms, is_complete."""
        config = ResearchConfig()
        workflow = DeepResearchWorkflow(config)

        bg_task = BackgroundTask(research_id="test-basic-metadata")

        with patch.object(workflow, "get_background_task", return_value=bg_task):
            with patch.object(workflow.memory, "load_deep_research", return_value=None):
                result = workflow._get_status("test-basic-metadata")

        assert result.success is True
        assert result.metadata["research_id"] == "test-basic-metadata"
        assert result.metadata["task_status"] == "running"
        assert "elapsed_ms" in result.metadata
        assert isinstance(result.metadata["elapsed_ms"], (int, float))
        assert "is_complete" in result.metadata

    def test_status_returns_error_without_research_id(self):
        """Status request without research_id returns error."""
        config = ResearchConfig()
        workflow = DeepResearchWorkflow(config)

        result = workflow._get_status(None)

        assert result.success is False
        assert result.error == "research_id is required"


class TestStatusResponseHeartbeat:
    """Tests for heartbeat metadata in status response."""

    def test_status_includes_last_heartbeat_at_when_state_available(self):
        """Status response includes last_heartbeat_at when state has heartbeat."""
        from datetime import datetime, timezone
        from foundry_mcp.core.research.models import DeepResearchState, DeepResearchPhase

        config = ResearchConfig()
        workflow = DeepResearchWorkflow(config)

        # Create a background task
        bg_task = BackgroundTask(research_id="test-heartbeat-status")

        # Create a state with heartbeat set
        state = DeepResearchState(
            id="test-heartbeat-status",
            original_query="test query",
            phase=DeepResearchPhase.GATHERING,
        )
        heartbeat_time = datetime(2026, 1, 26, 12, 0, 0, tzinfo=timezone.utc)
        state.last_heartbeat_at = heartbeat_time

        with patch.object(workflow, "get_background_task", return_value=bg_task):
            with patch.object(workflow.memory, "load_deep_research", return_value=state):
                result = workflow._get_status("test-heartbeat-status")

        assert result.success is True
        assert "last_heartbeat_at" in result.metadata
        assert result.metadata["last_heartbeat_at"] == heartbeat_time.isoformat()

    def test_status_includes_null_heartbeat_when_not_set(self):
        """Status response includes last_heartbeat_at as None when not set."""
        from foundry_mcp.core.research.models import DeepResearchState, DeepResearchPhase

        config = ResearchConfig()
        workflow = DeepResearchWorkflow(config)

        # Create a background task
        bg_task = BackgroundTask(research_id="test-no-heartbeat")

        # Create a state without heartbeat
        state = DeepResearchState(
            id="test-no-heartbeat",
            original_query="test query",
            phase=DeepResearchPhase.PLANNING,
        )
        assert state.last_heartbeat_at is None

        with patch.object(workflow, "get_background_task", return_value=bg_task):
            with patch.object(workflow.memory, "load_deep_research", return_value=state):
                result = workflow._get_status("test-no-heartbeat")

        assert result.success is True
        assert "last_heartbeat_at" in result.metadata
        assert result.metadata["last_heartbeat_at"] is None

    def test_persisted_status_includes_last_heartbeat_at(self):
        """Persisted state status response includes last_heartbeat_at."""
        from datetime import datetime, timezone
        from foundry_mcp.core.research.models import DeepResearchState, DeepResearchPhase

        config = ResearchConfig()
        workflow = DeepResearchWorkflow(config)

        # Create a completed state with heartbeat
        state = DeepResearchState(
            id="test-persisted-heartbeat",
            original_query="test query",
            phase=DeepResearchPhase.SYNTHESIS,
        )
        heartbeat_time = datetime(2026, 1, 26, 15, 30, 0, tzinfo=timezone.utc)
        state.last_heartbeat_at = heartbeat_time
        state.completed_at = datetime.now(timezone.utc)

        # No background task (persisted state path)
        with patch.object(workflow, "get_background_task", return_value=None):
            with patch.object(workflow.memory, "load_deep_research", return_value=state):
                with patch.object(workflow.memory, "save_deep_research"):
                    result = workflow._get_status("test-persisted-heartbeat")

        assert result.success is True
        assert "last_heartbeat_at" in result.metadata
        assert result.metadata["last_heartbeat_at"] == heartbeat_time.isoformat()
