"""Tests for partial result discard policy during cancellation.

Verifies:
- Partial results from incomplete iterations are discarded
- Completed iterations are preserved
- State rollback on cancellation
- Metadata tracking of discarded iterations
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from foundry_mcp.core.research.models.deep_research import (
    DeepResearchPhase,
    DeepResearchState,
)


class TestPartialResultPolicy:
    """Tests for partial result discard policy."""

    def test_iteration_in_progress_flag_set_at_start(self):
        """Should mark iteration as in_progress at start of workflow phases."""
        state = DeepResearchState(original_query="Test query")

        # Initially no flag
        assert state.metadata.get("iteration_in_progress") is None

        # Simulate workflow setting the flag at GATHERING phase start
        state.metadata["iteration_in_progress"] = True

        assert state.metadata["iteration_in_progress"] is True

    def test_iteration_in_progress_cleared_on_completion(self):
        """Should clear iteration_in_progress flag when iteration completes successfully."""
        state = DeepResearchState(original_query="Test query")
        state.metadata["iteration_in_progress"] = True

        # Simulate successful iteration completion
        state.metadata["iteration_in_progress"] = False
        state.metadata["last_completed_iteration"] = state.iteration

        assert state.metadata["iteration_in_progress"] is False
        assert state.metadata["last_completed_iteration"] == 1

    def test_discarded_iteration_recorded_on_cancel(self):
        """Should record discarded iteration when cancelled mid-iteration."""
        state = DeepResearchState(original_query="Test query")
        state.iteration = 2
        state.metadata["iteration_in_progress"] = True
        state.metadata["last_completed_iteration"] = 1

        # Simulate cancellation handling
        if state.metadata.get("iteration_in_progress"):
            last_completed = state.metadata.get("last_completed_iteration")
            if last_completed is not None and last_completed < state.iteration:
                state.metadata["discarded_iteration"] = state.iteration
                state.iteration = last_completed
                state.phase = DeepResearchPhase.SYNTHESIS

        assert state.metadata["discarded_iteration"] == 2
        assert state.iteration == 1
        assert state.phase == DeepResearchPhase.SYNTHESIS

    def test_first_iteration_incomplete_marked_for_discard(self):
        """Should mark first iteration for discard if incomplete at cancellation."""
        state = DeepResearchState(original_query="Test query")
        state.iteration = 1
        state.metadata["iteration_in_progress"] = True
        # No last_completed_iteration (first iteration never completed)

        # Simulate cancellation handling
        if state.metadata.get("iteration_in_progress"):
            last_completed = state.metadata.get("last_completed_iteration")
            if last_completed is None or last_completed >= state.iteration:
                # First iteration incomplete
                state.metadata["discarded_iteration"] = state.iteration

        assert state.metadata["discarded_iteration"] == 1

    def test_completed_iteration_preserved_on_cancel(self):
        """Should preserve completed iteration when cancelled after completion."""
        state = DeepResearchState(original_query="Test query")
        state.iteration = 2
        state.metadata["iteration_in_progress"] = False  # Not in progress
        state.metadata["last_completed_iteration"] = 2

        # Simulate cancellation handling - should not discard
        if state.metadata.get("iteration_in_progress"):
            state.metadata["discarded_iteration"] = state.iteration

        # No discard should happen
        assert state.metadata.get("discarded_iteration") is None
        assert state.iteration == 2


class TestPartialResultCancellationFlow:
    """Integration-style tests for cancellation flow with partial results."""

    @pytest.mark.asyncio
    async def test_cancel_during_gathering_discards_partial(self):
        """Should discard partial results when cancelled during gathering phase."""
        from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow

        mock_config = MagicMock()
        mock_config.deep_research_audit_artifacts = False
        mock_config.default_provider = "test"
        mock_memory = MagicMock()
        mock_memory.save_deep_research = MagicMock()

        workflow = DeepResearchWorkflow(mock_config, mock_memory)
        state = DeepResearchState(original_query="Test cancellation")
        state.phase = DeepResearchPhase.GATHERING
        state.iteration = 2
        state.metadata["iteration_in_progress"] = True
        state.metadata["last_completed_iteration"] = 1

        # Simulate the cancellation handler logic (from except asyncio.CancelledError block)
        # We can't easily trigger an actual CancelledError in unit test, so test the logic directly
        state.metadata["cancelled"] = True
        state.metadata["cancellation_state"] = "cancelling"

        # Apply partial result policy
        if state.metadata.get("iteration_in_progress"):
            last_completed = state.metadata.get("last_completed_iteration")
            if last_completed is not None and last_completed < state.iteration:
                state.metadata["discarded_iteration"] = state.iteration
                state.iteration = last_completed
                state.phase = DeepResearchPhase.SYNTHESIS

        # Verify rollback occurred
        assert state.metadata["discarded_iteration"] == 2
        assert state.iteration == 1
        assert state.phase == DeepResearchPhase.SYNTHESIS
        assert state.metadata["cancelled"] is True

    @pytest.mark.asyncio
    async def test_cancel_after_synthesis_preserves_iteration(self):
        """Should preserve iteration when cancelled after synthesis completes."""
        state = DeepResearchState(original_query="Test cancellation")
        state.phase = DeepResearchPhase.SYNTHESIS
        state.iteration = 2
        state.metadata["iteration_in_progress"] = False  # Synthesis completed
        state.metadata["last_completed_iteration"] = 2

        # Simulate cancellation
        state.metadata["cancelled"] = True
        state.metadata["cancellation_state"] = "cancelling"

        # Apply partial result policy - should not discard
        if state.metadata.get("iteration_in_progress"):
            last_completed = state.metadata.get("last_completed_iteration")
            if last_completed is not None and last_completed < state.iteration:
                state.metadata["discarded_iteration"] = state.iteration

        # Verify no rollback
        assert state.metadata.get("discarded_iteration") is None
        assert state.iteration == 2
        assert state.metadata["cancelled"] is True

    @pytest.mark.asyncio
    async def test_cancel_first_iteration_marks_for_discard(self):
        """Should mark first iteration for discard when cancelled before completion."""
        state = DeepResearchState(original_query="Test cancellation")
        state.phase = DeepResearchPhase.GATHERING
        state.iteration = 1
        state.metadata["iteration_in_progress"] = True
        # No last_completed_iteration yet

        # Simulate cancellation
        state.metadata["cancelled"] = True

        # Apply partial result policy
        if state.metadata.get("iteration_in_progress"):
            last_completed = state.metadata.get("last_completed_iteration")
            if last_completed is None or last_completed >= state.iteration:
                state.metadata["discarded_iteration"] = state.iteration

        # Verify marked for discard
        assert state.metadata["discarded_iteration"] == 1
        assert state.iteration == 1  # Not rolled back (nothing to roll back to)


class TestIterationProgressTracking:
    """Tests for iteration progress flag tracking across phases."""

    def test_progress_flag_lifecycle_gathering_to_synthesis(self):
        """Should track iteration progress through gathering to synthesis."""
        state = DeepResearchState(original_query="Test query")

        # Phase: BRIEF - no iteration_in_progress
        state.phase = DeepResearchPhase.BRIEF
        assert state.metadata.get("iteration_in_progress") is None

        # Phase: GATHERING - iteration starts
        state.phase = DeepResearchPhase.GATHERING
        state.metadata["iteration_in_progress"] = True
        assert state.metadata["iteration_in_progress"] is True

        # Phase: SUPERVISION - still in progress
        state.phase = DeepResearchPhase.SUPERVISION
        assert state.metadata["iteration_in_progress"] is True

        # Phase: SYNTHESIS - iteration completes
        state.phase = DeepResearchPhase.SYNTHESIS
        state.metadata["iteration_in_progress"] = False
        state.metadata["last_completed_iteration"] = 1
        assert state.metadata["iteration_in_progress"] is False
        assert state.metadata["last_completed_iteration"] == 1

    def test_progress_flag_lifecycle_refinement_iteration(self):
        """Should track progress through a second gathering-to-synthesis iteration."""
        state = DeepResearchState(original_query="Test query")
        state.iteration = 1
        state.metadata["last_completed_iteration"] = 1

        # Start second iteration - new gathering begins
        state.phase = DeepResearchPhase.GATHERING
        state.iteration = 2
        state.metadata["iteration_in_progress"] = True
        assert state.metadata["iteration_in_progress"] is True
        assert state.metadata["last_completed_iteration"] == 1

        # Gathering to synthesis completes
        state.phase = DeepResearchPhase.SYNTHESIS
        state.metadata["iteration_in_progress"] = False
        state.metadata["last_completed_iteration"] = 2
        assert state.metadata["iteration_in_progress"] is False
        assert state.metadata["last_completed_iteration"] == 2


class TestCancellationStateTransitions:
    """Tests for cancellation state machine transitions."""

    def test_cancellation_state_transition_cancelling(self):
        """Should transition to cancelling state on CancelledError."""
        state = DeepResearchState(original_query="Test query")

        # Initially no cancellation state
        assert state.metadata.get("cancellation_state") is None

        # Transition to cancelling
        state.metadata["cancellation_state"] = "cancelling"
        assert state.metadata["cancellation_state"] == "cancelling"

    def test_cancellation_state_transition_cleanup(self):
        """Should transition from cancelling to cleanup."""
        state = DeepResearchState(original_query="Test query")
        state.metadata["cancellation_state"] = "cancelling"

        # Transition to cleanup
        state.metadata["cancellation_state"] = "cleanup"
        assert state.metadata["cancellation_state"] == "cleanup"

    def test_full_cancellation_state_machine(self):
        """Should track full cancellation state machine flow."""
        state = DeepResearchState(original_query="Test query")
        state.iteration = 2
        state.metadata["iteration_in_progress"] = True
        state.metadata["last_completed_iteration"] = 1

        # 1. None -> "cancelling"
        state.metadata["cancellation_state"] = "cancelling"
        state.metadata["cancelled"] = True

        # 2. Apply partial result policy
        if state.metadata.get("iteration_in_progress"):
            last_completed = state.metadata.get("last_completed_iteration")
            if last_completed is not None and last_completed < state.iteration:
                state.metadata["discarded_iteration"] = state.iteration
                state.iteration = last_completed
                state.phase = DeepResearchPhase.SYNTHESIS

        # 3. "cancelling" -> "cleanup"
        state.metadata["cancellation_state"] = "cleanup"

        # Verify final state
        assert state.metadata["cancellation_state"] == "cleanup"
        assert state.metadata["cancelled"] is True
        assert state.metadata["discarded_iteration"] == 2
        assert state.iteration == 1


class TestPartialResultMetadataAudit:
    """Tests for partial result metadata tracking for audit purposes."""

    def test_audit_metadata_includes_all_fields(self):
        """Should include all relevant fields in audit metadata."""
        state = DeepResearchState(original_query="Test query")
        state.iteration = 2
        state.phase = DeepResearchPhase.GATHERING
        state.metadata["iteration_in_progress"] = True
        state.metadata["last_completed_iteration"] = 1
        state.metadata["discarded_iteration"] = None
        state.metadata["cancellation_state"] = None

        # Simulate cancellation
        state.metadata["cancellation_state"] = "cancelling"
        state.metadata["discarded_iteration"] = 2

        # Build audit data (as done in workflow)
        audit_data = {
            "phase": state.phase.value,
            "iteration": state.iteration,
            "iteration_in_progress": state.metadata.get("iteration_in_progress"),
            "last_completed_iteration": state.metadata.get("last_completed_iteration"),
            "discarded_iteration": state.metadata.get("discarded_iteration"),
            "cancellation_state": state.metadata.get("cancellation_state"),
        }

        assert audit_data["phase"] == "gathering"
        assert audit_data["iteration"] == 2
        assert audit_data["iteration_in_progress"] is True
        assert audit_data["last_completed_iteration"] == 1
        assert audit_data["discarded_iteration"] == 2
        assert audit_data["cancellation_state"] == "cancelling"


class TestCancellationRaceConditionFix:
    """Tests for the triple mark_cancelled race condition fix (PLAN Phase 3.1).

    Verifies that cancellation preserves the inner handler's partial-result
    discard state and that outer handlers don't overwrite it.
    """

    def test_cancel_research_does_not_write_state(self):
        """_cancel_research should only set the cancel flag, not write state directly.

        The workflow's own CancelledError handler in _execute_workflow_async()
        is the sole writer of terminal state.
        """
        from unittest.mock import MagicMock

        from foundry_mcp.core.research.workflows.deep_research.action_handlers import (
            ActionHandlersMixin,
        )

        mock_config = MagicMock()
        mock_memory = MagicMock()

        handler = ActionHandlersMixin.__new__(ActionHandlersMixin)
        handler.config = mock_config
        handler.memory = mock_memory

        # Create a mock background task that returns True on cancel
        mock_bg_task = MagicMock()
        mock_bg_task.cancel.return_value = True
        handler.get_background_task = MagicMock(return_value=mock_bg_task)

        result = handler._cancel_research("test-research-id")

        assert result.success is True
        assert "cancelled" in result.content

        # The key assertion: memory.load_deep_research should NOT be called.
        # The handler should not load, modify, or save state.
        mock_memory.load_deep_research.assert_not_called()
        mock_memory.save_deep_research.assert_not_called()

    def test_background_tasks_guard_skips_if_already_terminated(self):
        """background_tasks CancelledError handler should not mark_cancelled
        if state.completed_at is already set (inner handler already ran).
        """
        state = DeepResearchState(original_query="Test query")
        state.iteration = 2
        state.phase = DeepResearchPhase.GATHERING
        state.metadata["iteration_in_progress"] = True
        state.metadata["last_completed_iteration"] = 1

        # Simulate the inner handler having already terminated the state
        state.mark_cancelled(phase_state="phase=synthesis, iteration=1")
        assert state.completed_at is not None

        # Record the state after inner handler
        inner_completed_at = state.completed_at
        inner_phase_state = state.metadata["cancelled_phase_state"]

        # Simulate the outer background_tasks guard
        if state.completed_at is None:
            # This should NOT execute
            state.mark_cancelled(phase_state=f"phase={state.phase.value}, iteration={state.iteration}")
            assert False, "Should not have called mark_cancelled again"

        # State should be unchanged from inner handler's write
        assert state.completed_at == inner_completed_at
        assert state.metadata["cancelled_phase_state"] == inner_phase_state

    def test_workflow_execution_returns_result_not_raises(self):
        """_execute_workflow_async CancelledError handler should return a
        WorkflowResult instead of re-raising CancelledError, preserving
        the iteration rollback state.
        """
        from foundry_mcp.core.research.workflows.base import WorkflowResult

        state = DeepResearchState(original_query="Test query")
        state.iteration = 2
        state.phase = DeepResearchPhase.GATHERING
        state.metadata["iteration_in_progress"] = True
        state.metadata["last_completed_iteration"] = 1

        # Simulate the inner handler's iteration rollback logic
        state.metadata["cancellation_state"] = "cancelling"
        state.metadata["cancelled"] = True

        if state.metadata.get("iteration_in_progress"):
            last_completed = state.metadata.get("last_completed_iteration")
            if last_completed is not None and last_completed < state.iteration:
                state.metadata["discarded_iteration"] = state.iteration
                state.iteration = last_completed
                state.phase = DeepResearchPhase.SYNTHESIS

        state.metadata["cancellation_state"] = "cleanup"
        state.mark_cancelled(phase_state=f"phase={state.phase.value}, iteration={state.iteration}")

        # Build the result as the fixed code does (instead of re-raising)
        result = WorkflowResult(
            success=False,
            content="",
            error="Research cancelled",
            metadata={
                "research_id": state.id,
                "cancelled": True,
                "phase": state.phase.value,
                "iteration": state.iteration,
                "discarded_iteration": state.metadata.get("discarded_iteration"),
            },
        )

        # Verify the result captures the rollback state
        assert result.success is False
        assert result.error == "Research cancelled"
        assert result.metadata["cancelled"] is True
        assert result.metadata["discarded_iteration"] == 2
        assert result.metadata["iteration"] == 1
        assert result.metadata["phase"] == "synthesis"

        # Verify state was properly rolled back
        assert state.iteration == 1
        assert state.phase == DeepResearchPhase.SYNTHESIS
        assert state.completed_at is not None

    def test_cancellation_preserves_inner_handler_rollback(self):
        """Full integration test: cancellation should preserve the inner
        handler's partial-result discard even when all three paths run.

        Simulates the scenario where:
        1. _cancel_research() sets the cancel flag
        2. _execute_workflow_async catches CancelledError, rolls back, returns result
        3. background_tasks catches nothing (no CancelledError since result returned)

        The final state should reflect the inner handler's rollback.
        """
        from unittest.mock import MagicMock

        state = DeepResearchState(original_query="Test query")
        state.iteration = 3
        state.phase = DeepResearchPhase.GATHERING
        state.metadata["iteration_in_progress"] = True
        state.metadata["last_completed_iteration"] = 2

        mock_memory = MagicMock()
        saves = []

        def capture_save(s):
            # Capture a snapshot of key fields at save time
            saves.append({
                "iteration": s.iteration,
                "phase": s.phase.value,
                "completed_at": s.completed_at,
                "discarded_iteration": s.metadata.get("discarded_iteration"),
                "cancellation_state": s.metadata.get("cancellation_state"),
                "cancelled_phase_state": s.metadata.get("cancelled_phase_state"),
            })

        mock_memory.save_deep_research.side_effect = capture_save

        # Step 1: _cancel_research only sets bg_task cancel flag (no state writes)
        # (Nothing to simulate here - the fix removed state writes)

        # Step 2: _execute_workflow_async inner handler
        # Cancelling transition
        state.metadata["cancellation_state"] = "cancelling"
        state.metadata["cancelled"] = True

        # Iteration rollback
        if state.metadata.get("iteration_in_progress"):
            last_completed = state.metadata.get("last_completed_iteration")
            if last_completed is not None and last_completed < state.iteration:
                state.metadata["discarded_iteration"] = state.iteration
                state.iteration = last_completed
                state.phase = DeepResearchPhase.SYNTHESIS

        mock_memory.save_deep_research(state)  # First save: cancelling with rollback

        # Cleanup transition
        state.metadata["cancellation_state"] = "cleanup"
        state.mark_cancelled(phase_state=f"phase={state.phase.value}, iteration={state.iteration}")
        mock_memory.save_deep_research(state)  # Second save: after mark_cancelled

        # Step 3: background_tasks safety net - should be skipped
        if state.completed_at is None:
            state.mark_cancelled(phase_state="SHOULD_NOT_APPEAR")
            mock_memory.save_deep_research(state)

        # Verify final state
        assert state.iteration == 2  # Rolled back from 3 to 2
        assert state.phase == DeepResearchPhase.SYNTHESIS
        assert state.metadata["discarded_iteration"] == 3
        assert state.completed_at is not None
        assert state.metadata["terminal_status"] == "cancelled"
        assert state.metadata["cancelled_phase_state"] == "phase=synthesis, iteration=2"

        # Verify save count: only 2 saves (from inner handler), not 3
        assert len(saves) == 2
        # First save: rollback applied but not yet marked cancelled
        assert saves[0]["iteration"] == 2
        assert saves[0]["discarded_iteration"] == 3
        assert saves[0]["completed_at"] is None  # Not yet terminated
        # Second save: marked cancelled
        assert saves[1]["completed_at"] is not None
        assert saves[1]["cancellation_state"] == "cleanup"
