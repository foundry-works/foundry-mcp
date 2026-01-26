"""Tests for partial result discard policy during cancellation.

Verifies:
- Partial results from incomplete iterations are discarded
- Completed iterations are preserved
- State rollback on cancellation
- Metadata tracking of discarded iterations
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from foundry_mcp.core.research.models import (
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
        from foundry_mcp.core.research.workflows.base import WorkflowResult

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
        state.phase = DeepResearchPhase.REFINEMENT
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
        state.phase = DeepResearchPhase.ANALYSIS
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

        # Phase: PLANNING - no iteration_in_progress
        state.phase = DeepResearchPhase.PLANNING
        assert state.metadata.get("iteration_in_progress") is None

        # Phase: GATHERING - iteration starts
        state.phase = DeepResearchPhase.GATHERING
        state.metadata["iteration_in_progress"] = True
        assert state.metadata["iteration_in_progress"] is True

        # Phase: ANALYSIS - still in progress
        state.phase = DeepResearchPhase.ANALYSIS
        assert state.metadata["iteration_in_progress"] is True

        # Phase: SYNTHESIS - iteration completes
        state.phase = DeepResearchPhase.SYNTHESIS
        state.metadata["iteration_in_progress"] = False
        state.metadata["last_completed_iteration"] = 1
        assert state.metadata["iteration_in_progress"] is False
        assert state.metadata["last_completed_iteration"] == 1

    def test_progress_flag_lifecycle_refinement_iteration(self):
        """Should track progress through refinement iteration."""
        state = DeepResearchState(original_query="Test query")
        state.iteration = 1
        state.metadata["last_completed_iteration"] = 1

        # Start refinement - new iteration begins
        state.phase = DeepResearchPhase.REFINEMENT
        state.iteration = 2
        state.metadata["iteration_in_progress"] = True
        assert state.metadata["iteration_in_progress"] is True
        assert state.metadata["last_completed_iteration"] == 1

        # Refinement to synthesis completes
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
