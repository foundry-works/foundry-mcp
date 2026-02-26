"""Tests for cancellation rollback metadata in workflow_execution.py.

Phase 2C: Verifies that the rollback_note metadata flag is set when
cancellation occurs during an in-progress iteration, so resume logic
can detect that partial iteration data is retained in state.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from foundry_mcp.core.research.models.deep_research import (
    DeepResearchPhase,
    DeepResearchState,
)
from foundry_mcp.core.research.workflows.deep_research.workflow_execution import (
    WorkflowExecutionMixin,
)


class StubWorkflowExecution(WorkflowExecutionMixin):
    """Minimal concrete class for testing WorkflowExecutionMixin."""

    def __init__(self) -> None:
        self.config = MagicMock()
        self.config.get_phase_timeout.return_value = 60.0
        self.memory = MagicMock()
        self.hooks = MagicMock()
        self.orchestrator = MagicMock()
        self._tasks: dict[str, Any] = {}
        self._tasks_lock = threading.Lock()
        self._search_providers: dict[str, Any] = {}
        self._audit_events: list[tuple[str, dict]] = []

    def _write_audit_event(self, state: Any, event: str, **kwargs: Any) -> None:
        self._audit_events.append((event, kwargs))

    def _flush_state(self, state: Any) -> None:
        pass

    def _record_workflow_error(self, exc: Any, state: Any, context: str) -> None:
        pass

    def _safe_orchestrator_transition(self, state: Any, phase: Any) -> None:
        pass


class TestCancellationRollbackMetadata:
    """2C: Verify rollback_note metadata is set on cancellation rollback."""

    @pytest.mark.asyncio
    async def test_rollback_sets_metadata_on_incomplete_iteration(self):
        """When cancelled during in-progress iteration with prior checkpoint,
        rollback_note='partial_iteration_data_retained' is set."""
        stub = StubWorkflowExecution()
        state = DeepResearchState(
            original_query="test query",
            phase=DeepResearchPhase.SUPERVISION,
            iteration=2,
        )
        state.metadata["iteration_in_progress"] = True
        state.metadata["last_completed_iteration"] = 1

        # Make the supervision phase raise CancelledError
        async def raise_cancelled(**kwargs):
            raise asyncio.CancelledError("Cancellation requested")

        stub._execute_supervision_async = raise_cancelled

        result = await stub._execute_workflow_async(
            state=state,
            provider_id=None,
            timeout_per_operation=60.0,
            max_concurrent=3,
        )

        assert result.success is False
        assert state.metadata.get("rollback_note") == "partial_iteration_data_retained"
        assert state.metadata.get("discarded_iteration") == 2
        assert state.iteration == 1

    @pytest.mark.asyncio
    async def test_rollback_sets_metadata_on_first_iteration_incomplete(self):
        """When first iteration is incomplete at cancellation,
        rollback_note is still set."""
        stub = StubWorkflowExecution()
        state = DeepResearchState(
            original_query="test query",
            phase=DeepResearchPhase.SUPERVISION,
            iteration=0,
        )
        state.metadata["iteration_in_progress"] = True
        # No last_completed_iteration — first iteration

        async def raise_cancelled(**kwargs):
            raise asyncio.CancelledError("Cancellation requested")

        stub._execute_supervision_async = raise_cancelled

        result = await stub._execute_workflow_async(
            state=state,
            provider_id=None,
            timeout_per_operation=60.0,
            max_concurrent=3,
        )

        assert result.success is False
        assert state.metadata.get("rollback_note") == "partial_iteration_data_retained"
        assert state.metadata.get("discarded_iteration") == 0

    @pytest.mark.asyncio
    async def test_no_rollback_note_when_iteration_not_in_progress(self):
        """When cancelled without iteration_in_progress, no rollback_note is set."""
        stub = StubWorkflowExecution()
        state = DeepResearchState(
            original_query="test query",
            phase=DeepResearchPhase.BRIEF,
            iteration=1,
        )
        # iteration_in_progress is not set — cancel during BRIEF phase

        async def raise_cancelled(**kwargs):
            raise asyncio.CancelledError("Cancellation requested")

        stub._execute_brief_async = raise_cancelled

        result = await stub._execute_workflow_async(
            state=state,
            provider_id=None,
            timeout_per_operation=60.0,
            max_concurrent=3,
        )

        assert result.success is False
        assert "rollback_note" not in state.metadata
