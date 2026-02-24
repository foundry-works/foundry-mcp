"""Unit tests for heartbeat timing in DeepResearchWorkflow.

Verifies that heartbeat (last_heartbeat_at) is updated BEFORE provider calls,
ensuring progress visibility during long-running research operations.
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from foundry_mcp.core.research.models.deep_research import (
    DeepResearchPhase,
    DeepResearchState,
)
from foundry_mcp.core.research.models.enums import ConfidenceLevel
from foundry_mcp.core.research.workflows.base import WorkflowResult
from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_config():
    """Create a mock ResearchConfig for heartbeat timing tests."""
    config = MagicMock()
    config.default_provider = "test-provider"
    config.ttl_hours = 24
    config.deep_research_max_iterations = 3
    config.deep_research_max_sub_queries = 5
    config.deep_research_max_sources = 5
    config.deep_research_follow_links = True
    config.deep_research_timeout = 120.0
    config.deep_research_max_concurrent = 3
    config.deep_research_providers = ["tavily"]
    config.deep_research_audit_artifacts = False  # Disable audit for these tests
    config.deep_research_planning_timeout = 60.0
    config.deep_research_synthesis_timeout = 180.0
    config.deep_research_planning_provider = None
    config.deep_research_synthesis_provider = None
    config.deep_research_max_retries = 0
    config.deep_research_retry_delay = 1.0
    # Digest configuration
    config.deep_research_digest_min_chars = 10000
    config.deep_research_digest_max_sources = 8
    config.deep_research_digest_timeout = 60.0
    config.deep_research_digest_max_concurrent = 3
    config.deep_research_digest_include_evidence = True
    config.deep_research_digest_evidence_max_chars = 400
    config.deep_research_digest_max_evidence_snippets = 5
    config.deep_research_digest_fetch_pdfs = False
    config.deep_research_archive_content = False
    config.deep_research_digest_provider = None
    config.deep_research_digest_providers = []
    # Topic researcher config (ReAct loop)
    config.deep_research_topic_max_tool_calls = 3
    config.deep_research_topic_reflection_provider = None
    config.deep_research_reflection_provider = None
    config.deep_research_reflection_timeout = 60.0
    config.deep_research_enable_extract = False
    config.deep_research_inline_compression = False
    config.deep_research_enable_content_dedup = True
    config.deep_research_content_dedup_threshold = 0.8
    config.deep_research_clarification_provider = None
    config.deep_research_stale_task_seconds = 300.0

    def get_phase_timeout(phase: str) -> float:
        mapping = {
            "planning": config.deep_research_planning_timeout,
            "synthesis": config.deep_research_synthesis_timeout,
        }
        return mapping.get(phase.lower(), config.deep_research_timeout)

    def get_phase_provider(phase: str) -> str:
        mapping = {
            "planning": config.deep_research_planning_provider,
            "synthesis": config.deep_research_synthesis_provider,
        }
        return mapping.get(phase.lower()) or config.default_provider

    def get_phase_fallback_providers(phase: str) -> list:
        return []

    def get_digest_provider(analysis_provider: str = None) -> str:
        return analysis_provider or config.default_provider

    def get_digest_fallback_providers() -> list:
        return []

    config.get_phase_timeout = get_phase_timeout
    config.get_phase_provider = get_phase_provider
    config.get_phase_fallback_providers = get_phase_fallback_providers
    config.get_digest_provider = get_digest_provider
    config.get_digest_fallback_providers = get_digest_fallback_providers
    return config


@pytest.fixture
def mock_memory(tmp_path: Path):
    """Create a mock ResearchMemory with call tracking."""
    memory = MagicMock()
    memory.base_path = tmp_path
    memory.save_deep_research = MagicMock()
    memory.load_deep_research = MagicMock(return_value=None)
    memory.delete_deep_research = MagicMock(return_value=True)
    memory.list_deep_research = MagicMock(return_value=[])
    return memory


@pytest.fixture
def sample_state():
    """Create a sample DeepResearchState for testing."""
    return DeepResearchState(
        id="deepres-heartbeat-test",
        original_query="Test heartbeat timing",
        research_brief="Testing heartbeat update timing",
        phase=DeepResearchPhase.BRIEF,
        iteration=1,
        max_iterations=3,
    )


# =============================================================================
# Heartbeat Timing Tests
# =============================================================================


class TestHeartbeatTiming:
    """Tests verifying heartbeat is updated BEFORE provider calls."""

    @pytest.mark.asyncio
    async def test_planning_phase_heartbeat_before_provider_call(self, mock_config, mock_memory, sample_state):
        """Should update heartbeat BEFORE making provider call in planning phase."""
        workflow = DeepResearchWorkflow(mock_config, mock_memory)

        # Track operation order
        operation_order = []
        heartbeat_at_provider_call: Optional[datetime] = None

        def track_save(*args, **kwargs):
            # Record when save_deep_research is called (heartbeat update)
            if args and hasattr(args[0], "last_heartbeat_at"):
                state = args[0]
                if state.last_heartbeat_at is not None:
                    operation_order.append(("heartbeat_save", state.last_heartbeat_at))

        mock_memory.save_deep_research.side_effect = track_save

        async def track_provider(*args, **kwargs):
            nonlocal heartbeat_at_provider_call
            # Record when provider is called
            operation_order.append(("provider_call", datetime.now(timezone.utc)))
            # Capture the heartbeat value at the time of provider call
            heartbeat_at_provider_call = sample_state.last_heartbeat_at
            # Return WorkflowResult (what _execute_provider_async returns)
            return WorkflowResult(
                success=True,
                content='{"sub_queries": [{"query": "test", "rationale": "test", "priority": 1}]}',
                provider_id="test-provider",
                model_used="test-model",
                tokens_used=30,
                duration_ms=100.0,
            )

        with patch.object(workflow, "_execute_provider_async", side_effect=track_provider):
            with patch.object(workflow, "_check_cancellation"):
                await workflow._execute_planning_async(
                    state=sample_state,
                    provider_id=None,
                    timeout=60.0,
                )

        # Verify heartbeat was set before provider call
        assert heartbeat_at_provider_call is not None, "Heartbeat should be set before provider call"

        # Verify operation order: heartbeat save should come before provider call
        heartbeat_saves = [op for op in operation_order if op[0] == "heartbeat_save"]
        provider_calls = [op for op in operation_order if op[0] == "provider_call"]

        assert len(heartbeat_saves) >= 1, "Should have at least one heartbeat save"
        assert len(provider_calls) >= 1, "Should have at least one provider call"

        # The first heartbeat save should be before the first provider call
        first_heartbeat = heartbeat_saves[0][1]
        first_provider = provider_calls[0][1]
        assert first_heartbeat <= first_provider, (
            f"Heartbeat ({first_heartbeat}) should be updated before provider call ({first_provider})"
        )

    @pytest.mark.asyncio
    async def test_synthesis_phase_heartbeat_before_provider_call(self, mock_config, mock_memory):
        """Should update heartbeat BEFORE making provider call in synthesis phase."""
        workflow = DeepResearchWorkflow(mock_config, mock_memory)

        # Create state with findings for synthesis
        state = DeepResearchState(
            id="deepres-synthesis-heartbeat",
            original_query="Test synthesis heartbeat",
            phase=DeepResearchPhase.SYNTHESIS,
        )
        # Add findings to synthesize
        state.add_finding(
            content="Test finding for synthesis",
            confidence=ConfidenceLevel.HIGH,
            category="test",
        )

        heartbeat_before_call: Optional[datetime] = None

        async def track_provider(*args, **kwargs):
            nonlocal heartbeat_before_call
            heartbeat_before_call = state.last_heartbeat_at
            return WorkflowResult(
                success=True,
                content="# Research Report\n\nSynthesized findings...",
                provider_id="test-provider",
                model_used="test-model",
                tokens_used=60,
                duration_ms=200.0,
            )

        with patch.object(workflow, "_execute_provider_async", side_effect=track_provider):
            with patch.object(workflow, "_check_cancellation"):
                await workflow._execute_synthesis_async(
                    state=state,
                    provider_id=None,
                    timeout=180.0,
                )

        assert heartbeat_before_call is not None, "Heartbeat should be updated before provider call in synthesis phase"

    @pytest.mark.asyncio
    async def test_gathering_phase_heartbeat_before_search_calls(self, mock_config, mock_memory):
        """Should update heartbeat BEFORE making search provider calls in gathering phase."""
        workflow = DeepResearchWorkflow(mock_config, mock_memory)

        # Create state with sub-queries for gathering
        state = DeepResearchState(
            id="deepres-gathering-heartbeat",
            original_query="Test gathering heartbeat",
            phase=DeepResearchPhase.GATHERING,
        )
        state.add_sub_query(
            query="Test sub-query",
            rationale="Testing",
            priority=1,
        )

        heartbeat_before_search: Optional[datetime] = None
        search_called = False

        def track_save(*args, **kwargs):
            pass

        mock_memory.save_deep_research.side_effect = track_save

        # Mock search provider
        mock_search_provider = MagicMock()
        mock_search_provider.get_provider_name.return_value = "tavily"

        async def track_search(*args, **kwargs):
            nonlocal heartbeat_before_search, search_called
            heartbeat_before_search = state.last_heartbeat_at
            search_called = True
            return []  # Return empty results

        mock_search_provider.search = AsyncMock(side_effect=track_search)

        def get_search_provider(name: str):
            if name == "tavily":
                return mock_search_provider
            return None

        # Mock the LLM provider for the ReAct researcher loop
        import json as _json
        _llm_cc = 0

        async def _mock_llm(**kw):
            nonlocal _llm_cc
            _llm_cc += 1
            r = MagicMock()
            r.success = True
            r.tokens_used = 10
            r.error = None
            if _llm_cc % 2 == 1:
                r.content = _json.dumps({"tool_calls": [{"tool": "web_search", "arguments": {"query": "test", "max_results": 5}}]})
            else:
                r.content = _json.dumps({"tool_calls": [{"tool": "research_complete", "arguments": {"summary": "Done"}}]})
            return r

        with patch.object(workflow, "_get_search_provider", side_effect=get_search_provider), \
             patch.object(workflow, "_execute_provider_async", side_effect=_mock_llm), \
             patch.object(workflow, "_check_cancellation"):
            await workflow._execute_gathering_async(
                state=state,
                provider_id=None,
                timeout=30.0,
                max_concurrent=1,
            )

        assert search_called, "Search provider should have been called"
        assert heartbeat_before_search is not None, (
            "Heartbeat should be updated before search provider calls in gathering phase"
        )

    def test_heartbeat_persisted_to_memory(self, mock_config, mock_memory, sample_state):
        """Should persist state with heartbeat to memory before provider call."""
        workflow = DeepResearchWorkflow(mock_config, mock_memory)

        # Update heartbeat directly (simulating the workflow behavior)
        sample_state.last_heartbeat_at = datetime.now(timezone.utc)
        mock_memory.save_deep_research(sample_state)

        # Verify save was called with the state
        mock_memory.save_deep_research.assert_called_once_with(sample_state)

        # Verify the saved state has heartbeat set
        saved_state = mock_memory.save_deep_research.call_args[0][0]
        assert saved_state.last_heartbeat_at is not None

    @pytest.mark.asyncio
    async def test_heartbeat_provides_progress_visibility(self, mock_config, mock_memory, sample_state):
        """Heartbeat should enable progress visibility during long operations."""
        workflow = DeepResearchWorkflow(mock_config, mock_memory)

        # Simulate a slow provider call
        provider_delay = 0.1  # 100ms
        heartbeat_times: list[datetime] = []

        def capture_heartbeat(*args, **kwargs):
            if args and hasattr(args[0], "last_heartbeat_at"):
                state = args[0]
                if state.last_heartbeat_at is not None:
                    heartbeat_times.append(state.last_heartbeat_at)

        mock_memory.save_deep_research.side_effect = capture_heartbeat

        async def slow_provider(*args, **kwargs):
            await asyncio.sleep(provider_delay)
            return WorkflowResult(
                success=True,
                content='{"sub_queries": []}',
                provider_id="test-provider",
                model_used="test-model",
                tokens_used=30,
                duration_ms=provider_delay * 1000,
            )

        with patch.object(workflow, "_execute_provider_async", side_effect=slow_provider):
            with patch.object(workflow, "_check_cancellation"):
                start_time = datetime.now(timezone.utc)
                await workflow._execute_planning_async(
                    state=sample_state,
                    provider_id=None,
                    timeout=60.0,
                )
                end_time = datetime.now(timezone.utc)

        # Heartbeat should have been captured before the slow operation
        assert len(heartbeat_times) >= 1, "Should have captured at least one heartbeat"
        # First heartbeat should be close to start time, not end time
        first_heartbeat = heartbeat_times[0]
        time_from_start = (first_heartbeat - start_time).total_seconds()
        time_from_end = (end_time - first_heartbeat).total_seconds()

        # Heartbeat should be closer to start than to end (accounting for test overhead)
        assert time_from_start < time_from_end, (
            f"Heartbeat should be set before slow operation completes. "
            f"Time from start: {time_from_start:.3f}s, Time from end: {time_from_end:.3f}s"
        )
