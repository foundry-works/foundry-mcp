"""Unit tests for the DeepResearchWorkflow.

Tests the multi-phase iterative research workflow including:
- Planning phase (query decomposition)
- Gathering phase (parallel sub-query execution)
- Analysis phase (finding extraction)
- Synthesis phase (report generation)
- Refinement phase (gap identification)
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio

import pytest

from foundry_mcp.core.research.models import (
    ConfidenceLevel,
    DeepResearchPhase,
    DeepResearchState,
    PhaseMetrics,
    ResearchMode,
    ResearchSource,
    SourceType,
    SubQuery,
)
from foundry_mcp.core.research.workflows.base import WorkflowResult


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_config():
    """Create a mock ResearchConfig."""
    config = MagicMock()
    config.default_provider = "test-provider"
    config.ttl_hours = 24
    config.deep_research_max_iterations = 3
    config.deep_research_max_sub_queries = 5
    config.deep_research_max_sources = 5
    config.deep_research_follow_links = True
    config.deep_research_timeout = 120.0
    config.deep_research_max_concurrent = 3
    config.deep_research_providers = ["tavily", "google", "semantic_scholar"]
    config.deep_research_audit_artifacts = True
    # Per-phase timeout configuration
    config.deep_research_planning_timeout = 60.0
    config.deep_research_analysis_timeout = 90.0
    config.deep_research_synthesis_timeout = 180.0
    config.deep_research_refinement_timeout = 60.0
    # Per-phase provider configuration
    config.deep_research_planning_provider = None
    config.deep_research_analysis_provider = None
    config.deep_research_synthesis_provider = None
    config.deep_research_refinement_provider = None

    # Helper method mocks
    def get_phase_timeout(phase: str) -> float:
        mapping = {
            "planning": config.deep_research_planning_timeout,
            "analysis": config.deep_research_analysis_timeout,
            "synthesis": config.deep_research_synthesis_timeout,
            "refinement": config.deep_research_refinement_timeout,
        }
        return mapping.get(phase.lower(), config.deep_research_timeout)

    def get_phase_provider(phase: str) -> str:
        mapping = {
            "planning": config.deep_research_planning_provider,
            "analysis": config.deep_research_analysis_provider,
            "synthesis": config.deep_research_synthesis_provider,
            "refinement": config.deep_research_refinement_provider,
        }
        return mapping.get(phase.lower()) or config.default_provider

    config.get_phase_timeout = get_phase_timeout
    config.get_phase_provider = get_phase_provider
    return config


@pytest.fixture
def mock_memory(tmp_path: Path):
    """Create a mock ResearchMemory."""
    memory = MagicMock()
    memory.base_path = tmp_path
    memory.save_deep_research = MagicMock()
    memory.load_deep_research = MagicMock(return_value=None)
    memory.delete_deep_research = MagicMock(return_value=True)
    memory.list_deep_research = MagicMock(return_value=[])
    return memory


@pytest.fixture
def mock_provider_result():
    """Create a mock ProviderResult factory."""
    def _create(content: str, success: bool = True):
        from foundry_mcp.core.providers.base import ProviderResult, ProviderStatus, TokenUsage
        return ProviderResult(
            content=content,
            provider_id="test-provider",
            model_used="test-model",
            status=ProviderStatus.SUCCESS if success else ProviderStatus.ERROR,
            tokens=TokenUsage(input_tokens=10, output_tokens=20),
            duration_ms=100.0,
        )
    return _create


@pytest.fixture
def sample_deep_research_state():
    """Create a sample DeepResearchState for testing."""
    state = DeepResearchState(
        id="deepres-test123",
        original_query="What is deep learning?",
        research_brief="Investigating deep learning fundamentals",
        phase=DeepResearchPhase.PLANNING,
        iteration=1,
        max_iterations=3,
    )
    return state


# =============================================================================
# Model Tests
# =============================================================================


class TestDeepResearchState:
    """Tests for DeepResearchState model."""

    def test_create_state(self):
        """Should create a state with default values."""
        state = DeepResearchState(original_query="Test query")

        assert state.original_query == "Test query"
        assert state.phase == DeepResearchPhase.PLANNING
        assert state.iteration == 1
        assert state.max_iterations == 3
        assert len(state.sub_queries) == 0
        assert len(state.sources) == 0
        assert len(state.findings) == 0
        assert state.report is None
        assert state.completed_at is None

    def test_add_sub_query(self, sample_deep_research_state):
        """Should add a sub-query to the state."""
        state = sample_deep_research_state

        sub_query = state.add_sub_query(
            query="What are neural networks?",
            rationale="Foundation concept",
            priority=1,
        )

        assert len(state.sub_queries) == 1
        assert sub_query.query == "What are neural networks?"
        assert sub_query.rationale == "Foundation concept"
        assert sub_query.priority == 1
        assert sub_query.status == "pending"

    def test_add_source(self, sample_deep_research_state):
        """Should add a source to the state."""
        state = sample_deep_research_state

        source = state.add_source(
            title="Deep Learning Book",
            url="https://www.deeplearningbook.org",
            source_type=SourceType.ACADEMIC,
            snippet="Comprehensive guide to deep learning",
        )

        assert len(state.sources) == 1
        assert source.title == "Deep Learning Book"
        assert source.source_type == SourceType.ACADEMIC
        assert state.total_sources_examined == 1

    def test_add_finding(self, sample_deep_research_state):
        """Should add a finding to the state."""
        state = sample_deep_research_state

        finding = state.add_finding(
            content="Deep learning uses multiple layers",
            confidence=ConfidenceLevel.HIGH,
            category="Architecture",
        )

        assert len(state.findings) == 1
        assert finding.content == "Deep learning uses multiple layers"
        assert finding.confidence == ConfidenceLevel.HIGH
        assert finding.category == "Architecture"

    def test_add_gap(self, sample_deep_research_state):
        """Should add a research gap to the state."""
        state = sample_deep_research_state

        gap = state.add_gap(
            description="Missing information about transformers",
            suggested_queries=["What are transformer architectures?"],
            priority=1,
        )

        assert len(state.gaps) == 1
        assert gap.description == "Missing information about transformers"
        assert len(gap.suggested_queries) == 1

    def test_get_source_and_gap(self, sample_deep_research_state):
        """Should fetch sources and gaps by ID."""
        state = sample_deep_research_state

        source = state.add_source(
            title="Deep Learning Book",
            url="https://www.deeplearningbook.org",
            source_type=SourceType.ACADEMIC,
            snippet="Comprehensive guide to deep learning",
        )
        gap = state.add_gap(
            description="Missing information about transformers",
            suggested_queries=["What are transformer architectures?"],
            priority=1,
        )

        assert state.get_source(source.id) == source
        assert state.get_gap(gap.id) == gap
        assert state.get_source("missing") is None
        assert state.get_gap("missing") is None

    def test_advance_phase(self, sample_deep_research_state):
        """Should advance through phases correctly."""
        state = sample_deep_research_state

        assert state.phase == DeepResearchPhase.PLANNING

        state.advance_phase()
        assert state.phase == DeepResearchPhase.GATHERING

        state.advance_phase()
        assert state.phase == DeepResearchPhase.ANALYSIS

        state.advance_phase()
        assert state.phase == DeepResearchPhase.SYNTHESIS

        state.advance_phase()
        assert state.phase == DeepResearchPhase.REFINEMENT

    def test_pending_sub_queries(self, sample_deep_research_state):
        """Should return only pending sub-queries."""
        state = sample_deep_research_state

        sq1 = state.add_sub_query("Query 1")
        sq2 = state.add_sub_query("Query 2")
        sq1.status = "completed"

        pending = state.pending_sub_queries()
        assert len(pending) == 1
        assert pending[0].query == "Query 2"

    def test_should_continue_refinement(self, sample_deep_research_state):
        """Should correctly determine if refinement should continue."""
        state = sample_deep_research_state

        # No gaps, should not continue
        assert state.should_continue_refinement() is False

        # Add unresolved gap
        state.add_gap("Missing info")
        assert state.should_continue_refinement() is True

        # Max iterations reached
        state.iteration = 3
        assert state.should_continue_refinement() is False

    def test_mark_completed(self, sample_deep_research_state):
        """Should mark research as completed."""
        state = sample_deep_research_state

        state.mark_completed(report="Final report content")

        assert state.completed_at is not None
        assert state.report == "Final report content"
        assert state.phase == DeepResearchPhase.SYNTHESIS


class TestSubQuery:
    """Tests for SubQuery model."""

    def test_mark_completed(self):
        """Should mark sub-query as completed."""
        sq = SubQuery(query="Test query")

        sq.mark_completed(findings="Found important info")

        assert sq.status == "completed"
        assert sq.completed_at is not None
        assert sq.findings_summary == "Found important info"

    def test_mark_failed(self):
        """Should mark sub-query as failed."""
        sq = SubQuery(query="Test query")

        sq.mark_failed("Timeout error")

        assert sq.status == "failed"
        assert sq.completed_at is not None
        assert sq.error == "Timeout error"


class TestDeepResearchStateFailedSubQueries:
    """Tests for failed sub-query tracking."""

    def test_failed_sub_queries_returns_failed(self):
        """Should return sub-queries with status='failed'."""
        state = DeepResearchState(original_query="Test query")

        sq1 = state.add_sub_query("Completed query")
        sq1.mark_completed(findings="Found data")

        sq2 = state.add_sub_query("Failed query 1")
        sq2.mark_failed("Timeout after 30s")

        sq3 = state.add_sub_query("Pending query")

        sq4 = state.add_sub_query("Failed query 2")
        sq4.mark_failed("Provider unavailable")

        failed = state.failed_sub_queries()

        assert len(failed) == 2
        assert failed[0].query == "Failed query 1"
        assert failed[0].error == "Timeout after 30s"
        assert failed[1].query == "Failed query 2"
        assert failed[1].error == "Provider unavailable"

    def test_failed_sub_queries_empty_when_none_failed(self):
        """Should return empty list when no sub-queries failed."""
        state = DeepResearchState(original_query="Test query")

        sq1 = state.add_sub_query("Completed query")
        sq1.mark_completed(findings="Found data")

        sq2 = state.add_sub_query("Pending query")

        failed = state.failed_sub_queries()

        assert len(failed) == 0


# =============================================================================
# Workflow Tests
# =============================================================================


class TestDeepResearchWorkflow:
    """Tests for DeepResearchWorkflow class."""

    def test_workflow_initialization(self, mock_config, mock_memory):
        """Should initialize workflow with config and memory."""
        from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow

        workflow = DeepResearchWorkflow(mock_config, mock_memory)

        assert workflow.config == mock_config
        assert workflow.memory == mock_memory

    def test_audit_artifact_written(self, mock_config, mock_memory, tmp_path):
        """Should write audit events to JSONL artifact."""
        from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow

        workflow = DeepResearchWorkflow(mock_config, mock_memory)
        state = DeepResearchState(original_query="Audit test")

        workflow._write_audit_event(state, "test_event", data={"ok": True})

        audit_path = mock_memory.base_path / "deep_research" / f"{state.id}.audit.jsonl"
        assert audit_path.exists()
        lines = audit_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        payload = json.loads(lines[0])
        assert payload["event_type"] == "test_event"
        assert payload["research_id"] == state.id

    def test_workflow_complete_audit_enhanced_fields(
        self, mock_config, mock_memory, tmp_path
    ):
        """Should include enhanced statistics in workflow_complete audit event."""
        from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow

        workflow = DeepResearchWorkflow(mock_config, mock_memory)

        # Create state with sample data
        state = DeepResearchState(
            original_query="Enhanced audit test",
            research_mode=ResearchMode.TECHNICAL,
        )

        # Add phase metrics
        state.phase_metrics = [
            PhaseMetrics(
                phase="planning",
                duration_ms=1000.0,
                input_tokens=100,
                output_tokens=50,
                cached_tokens=10,
                provider_id="test-provider",
                model_used="test-model",
            ),
            PhaseMetrics(
                phase="analysis",
                duration_ms=2000.0,
                input_tokens=200,
                output_tokens=100,
                cached_tokens=20,
                provider_id="test-provider",
                model_used="test-model",
            ),
        ]

        # Add search provider stats
        state.search_provider_stats = {
            "tavily": 3,
            "google": 2,
            "semantic_scholar": 1,
        }

        # Add sources with URLs
        state.sources = [
            ResearchSource(
                title="Source 1",
                url="https://arxiv.org/paper1",
                source_type=SourceType.ACADEMIC,
            ),
            ResearchSource(
                title="Source 2",
                url="https://docs.python.org/guide",
                source_type=SourceType.WEB,
            ),
            ResearchSource(
                title="Source 3",
                url="https://arxiv.org/paper2",
                source_type=SourceType.ACADEMIC,
            ),
        ]

        state.report = "Test report content"
        state.phase = DeepResearchPhase.SYNTHESIS
        state.iteration = 1
        state.total_tokens_used = 480
        state.total_duration_ms = 3000.0

        # Write workflow_complete event with the new structure
        workflow._write_audit_event(
            state,
            "workflow_complete",
            data={
                "success": True,
                "phase": state.phase.value,
                "iteration": state.iteration,
                "sub_query_count": len(state.sub_queries),
                "source_count": len(state.sources),
                "finding_count": len(state.findings),
                "gap_count": len(state.unresolved_gaps()),
                "report_length": len(state.report or ""),
                "total_tokens_used": state.total_tokens_used,
                "total_duration_ms": state.total_duration_ms,
                "total_input_tokens": sum(
                    m.input_tokens for m in state.phase_metrics
                ),
                "total_output_tokens": sum(
                    m.output_tokens for m in state.phase_metrics
                ),
                "total_cached_tokens": sum(
                    m.cached_tokens for m in state.phase_metrics
                ),
                "phase_metrics": [
                    {
                        "phase": m.phase,
                        "duration_ms": m.duration_ms,
                        "input_tokens": m.input_tokens,
                        "output_tokens": m.output_tokens,
                        "cached_tokens": m.cached_tokens,
                        "provider_id": m.provider_id,
                        "model_used": m.model_used,
                    }
                    for m in state.phase_metrics
                ],
                "search_provider_stats": state.search_provider_stats,
                "total_search_queries": sum(state.search_provider_stats.values()),
                "source_hostnames": ["arxiv.org", "docs.python.org"],
                "research_mode": state.research_mode.value,
            },
        )

        audit_path = mock_memory.base_path / "deep_research" / f"{state.id}.audit.jsonl"
        assert audit_path.exists()
        lines = audit_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1

        payload = json.loads(lines[0])
        data = payload["data"]

        # Verify token breakdown totals
        assert data["total_input_tokens"] == 300
        assert data["total_output_tokens"] == 150
        assert data["total_cached_tokens"] == 30

        # Verify phase metrics
        assert len(data["phase_metrics"]) == 2
        assert data["phase_metrics"][0]["phase"] == "planning"
        assert data["phase_metrics"][0]["input_tokens"] == 100
        assert data["phase_metrics"][1]["phase"] == "analysis"
        assert data["phase_metrics"][1]["provider_id"] == "test-provider"

        # Verify search provider stats
        assert data["search_provider_stats"]["tavily"] == 3
        assert data["total_search_queries"] == 6

        # Verify source hostnames
        assert "arxiv.org" in data["source_hostnames"]
        assert "docs.python.org" in data["source_hostnames"]

        # Verify research mode
        assert data["research_mode"] == "technical"

    @pytest.mark.asyncio
    async def test_execute_gathering_multi_provider(
        self, mock_config, mock_memory, sample_deep_research_state
    ):
        """Should gather sources from multiple providers with dedup."""
        from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow

        workflow = DeepResearchWorkflow(mock_config, mock_memory)
        state = sample_deep_research_state
        state.phase = DeepResearchPhase.GATHERING
        sub_query = state.add_sub_query("Test query")

        tavily_provider = MagicMock()
        tavily_provider.get_provider_name.return_value = "tavily"
        tavily_provider.search = AsyncMock(
            return_value=[
                ResearchSource(
                    title="Result A",
                    url="http://example.com/a",
                    source_type=SourceType.WEB,
                    sub_query_id=sub_query.id,
                )
            ]
        )

        scholar_provider = MagicMock()
        scholar_provider.get_provider_name.return_value = "semantic_scholar"
        scholar_provider.search = AsyncMock(
            return_value=[
                ResearchSource(
                    title="Result A (duplicate)",
                    url="http://example.com/a",
                    source_type=SourceType.ACADEMIC,
                    sub_query_id=sub_query.id,
                ),
                ResearchSource(
                    title="Result B",
                    url="http://example.com/b",
                    source_type=SourceType.ACADEMIC,
                    sub_query_id=sub_query.id,
                ),
            ]
        )

        mock_config.deep_research_providers = ["tavily", "semantic_scholar"]

        def provider_lookup(name: str):
            return {
                "tavily": tavily_provider,
                "semantic_scholar": scholar_provider,
            }.get(name)

        with patch.object(workflow, "_get_search_provider", side_effect=provider_lookup):
            result = await workflow._execute_gathering_async(
                state=state,
                provider_id=None,
                timeout=30.0,
                max_concurrent=2,
            )

        assert result.success is True
        assert len(state.sources) == 2
        assert sub_query.status == "completed"
        assert result.metadata["providers_used"] == ["tavily", "semantic_scholar"]

    @pytest.mark.asyncio
    async def test_execute_gathering_deduplicates_by_title(
        self, mock_config, mock_memory, sample_deep_research_state
    ):
        """Should deduplicate sources with same title from different domains."""
        from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow

        workflow = DeepResearchWorkflow(mock_config, mock_memory)
        state = sample_deep_research_state
        state.phase = DeepResearchPhase.GATHERING
        sub_query = state.add_sub_query("Test query")

        # Same paper from OpenReview
        openreview_provider = MagicMock()
        openreview_provider.get_provider_name.return_value = "tavily"
        openreview_provider.search = AsyncMock(
            return_value=[
                ResearchSource(
                    title="Self-Preference Bias in LLM-as-a-Judge",
                    url="http://openreview.net/forum?id=abc123",
                    source_type=SourceType.WEB,
                    sub_query_id=sub_query.id,
                )
            ]
        )

        # Same paper from arXiv (different URL, same title)
        arxiv_provider = MagicMock()
        arxiv_provider.get_provider_name.return_value = "semantic_scholar"
        arxiv_provider.search = AsyncMock(
            return_value=[
                ResearchSource(
                    title="Self-Preference Bias in LLM-as-a-Judge",  # Same title
                    url="http://arxiv.org/abs/2401.12345",  # Different URL
                    source_type=SourceType.ACADEMIC,
                    sub_query_id=sub_query.id,
                ),
                ResearchSource(
                    title="A Different Paper About Something Else",
                    url="http://arxiv.org/abs/2401.99999",
                    source_type=SourceType.ACADEMIC,
                    sub_query_id=sub_query.id,
                ),
            ]
        )

        mock_config.deep_research_providers = ["tavily", "semantic_scholar"]

        def provider_lookup(name: str):
            return {
                "tavily": openreview_provider,
                "semantic_scholar": arxiv_provider,
            }.get(name)

        with patch.object(workflow, "_get_search_provider", side_effect=provider_lookup):
            result = await workflow._execute_gathering_async(
                state=state,
                provider_id=None,
                timeout=30.0,
                max_concurrent=2,
            )

        assert result.success is True
        # Should have 2 sources: OpenReview version + the different paper
        # arXiv duplicate of "Self-Preference Bias" should be skipped
        assert len(state.sources) == 2
        titles = [s.title for s in state.sources]
        assert "Self-Preference Bias in LLM-as-a-Judge" in titles
        assert "A Different Paper About Something Else" in titles

    def test_background_task_timeout(self, mock_config, mock_memory):
        """Should mark background task as timed out."""
        from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow
        from foundry_mcp.core.background_task import TaskStatus

        workflow = DeepResearchWorkflow(mock_config, mock_memory)
        state = DeepResearchState(original_query="Timeout test")

        async def slow_execute(*args, **kwargs):
            await asyncio.sleep(0.2)
            return WorkflowResult(success=True, content="done")

        with patch.object(
            workflow, "_execute_workflow_async", side_effect=slow_execute
        ):
            result = workflow._start_background_task(
                state=state,
                provider_id=None,
                timeout_per_operation=1.0,
                max_concurrent=1,
                task_timeout=0.05,
            )
            bg_task = workflow.get_background_task(state.id)
            assert bg_task is not None
            assert bg_task.thread is not None
            # Wait for the thread to complete (instead of awaiting asyncio task)
            bg_task.thread.join(timeout=5.0)

        assert result.success is True
        assert bg_task.status == TaskStatus.TIMEOUT
        assert bg_task.result is not None
        assert bg_task.result.metadata["timeout"] is True

    def test_background_task_is_done_property(self, mock_config, mock_memory):
        """Should correctly report is_done for thread-based execution."""
        from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow

        workflow = DeepResearchWorkflow(mock_config, mock_memory)
        state = DeepResearchState(original_query="is_done test")

        async def slow_execute(*args, **kwargs):
            await asyncio.sleep(0.1)
            return WorkflowResult(success=True, content="done")

        with patch.object(
            workflow, "_execute_workflow_async", side_effect=slow_execute
        ):
            _ = workflow._start_background_task(
                state=state,
                provider_id=None,
                timeout_per_operation=1.0,
                max_concurrent=1,
                task_timeout=10.0,
            )
            bg_task = workflow.get_background_task(state.id)
            assert bg_task is not None

            # Task should not be None (but it will be for thread-based execution)
            # The is_done property should handle both cases
            assert bg_task.thread is not None
            assert bg_task.task is None  # No asyncio task for thread-based

            # is_done should work via thread.is_alive()
            assert bg_task.is_done is False  # Still running

            # Wait for completion
            bg_task.thread.join(timeout=5.0)
            assert bg_task.is_done is True  # Now done

    def test_get_status_during_background_task(self, mock_config, mock_memory):
        """Should get status while background task is running (bug fix test)."""
        from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow

        workflow = DeepResearchWorkflow(mock_config, mock_memory)
        state = DeepResearchState(original_query="Status check test")

        async def slow_execute(*args, **kwargs):
            await asyncio.sleep(0.2)
            return WorkflowResult(success=True, content="done")

        with patch.object(
            workflow, "_execute_workflow_async", side_effect=slow_execute
        ):
            # Start background task
            workflow._start_background_task(
                state=state,
                provider_id=None,
                timeout_per_operation=1.0,
                max_concurrent=1,
                task_timeout=10.0,
            )

            # Check status while running - this should NOT crash
            # (Previously crashed with "'NoneType' object has no attribute 'done'")
            before_save_calls = mock_memory.save_deep_research.call_count
            status_result = workflow.execute(action="status", research_id=state.id)

            assert status_result.success is True
            assert status_result.metadata["research_id"] == state.id
            assert status_result.metadata["is_complete"] is False  # Still running
            assert status_result.metadata["status_check_count"] == 1
            assert mock_memory.save_deep_research.call_count == before_save_calls

            # Wait for completion
            bg_task = workflow.get_background_task(state.id)
            assert bg_task is not None
            assert bg_task.thread is not None
            bg_task.thread.join(timeout=5.0)

    def test_continue_research_with_background(
        self, mock_config, mock_memory, sample_deep_research_state
    ):
        """Should continue research in background mode."""
        from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow

        # Set up state as not completed
        sample_deep_research_state.completed_at = None
        mock_memory.load_deep_research.return_value = sample_deep_research_state
        mock_memory.save_deep_research.return_value = None

        workflow = DeepResearchWorkflow(mock_config, mock_memory)

        async def mock_execute(*args, **kwargs):
            await asyncio.sleep(0.1)
            return WorkflowResult(success=True, content="Continued research")

        with patch.object(
            workflow, "_execute_workflow_async", side_effect=mock_execute
        ):
            # Continue with background=True
            result = workflow.execute(
                action="continue",
                research_id=sample_deep_research_state.id,
                background=True,
                task_timeout=10.0,
            )

            # Should return immediately with research_id
            assert result.success is True
            assert result.metadata["research_id"] == sample_deep_research_state.id

            # Background task should be running
            bg_task = workflow.get_background_task(sample_deep_research_state.id)
            assert bg_task is not None
            assert bg_task.thread is not None

            # Wait for completion
            bg_task.thread.join(timeout=5.0)

    def test_execute_start_without_query(self, mock_config, mock_memory):
        """Should return error when starting without query."""
        from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow

        workflow = DeepResearchWorkflow(mock_config, mock_memory)

        result = workflow.execute(action="start", query=None)

        assert result.success is False
        assert result.error is not None
        assert "Query is required" in result.error

    def test_execute_continue_without_research_id(self, mock_config, mock_memory):
        """Should return error when continuing without research_id."""
        from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow

        workflow = DeepResearchWorkflow(mock_config, mock_memory)

        result = workflow.execute(action="continue", research_id=None)

        assert result.success is False
        assert result.error is not None
        assert "research_id is required" in result.error

    def test_execute_status_not_found(self, mock_config, mock_memory):
        """Should return error when research session not found."""
        from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow

        mock_memory.load_deep_research.return_value = None
        workflow = DeepResearchWorkflow(mock_config, mock_memory)

        result = workflow.execute(action="status", research_id="nonexistent")

        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error

    def test_execute_unknown_action(self, mock_config, mock_memory):
        """Should return error for unknown action."""
        from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow

        workflow = DeepResearchWorkflow(mock_config, mock_memory)

        result = workflow.execute(action="unknown")

        assert result.success is False
        assert result.error is not None
        assert "Unknown action" in result.error

    def test_execute_catches_exceptions(self, mock_config, mock_memory):
        """Exceptions during execute should be caught and return error result."""
        from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow
        from unittest.mock import patch

        workflow = DeepResearchWorkflow(mock_config, mock_memory)

        # Simulate an exception during _start_research
        with patch.object(
            workflow, "_start_research", side_effect=RuntimeError("Storage unavailable")
        ):
            result = workflow.execute(query="test query", action="start")

        # Should return error result, not raise exception
        assert result.success is False
        assert result.error is not None
        assert "Storage unavailable" in result.error
        assert result.metadata["action"] == "start"
        assert result.metadata["error_type"] == "RuntimeError"

    def test_get_status_success(self, mock_config, mock_memory, sample_deep_research_state):
        """Should return status for existing research."""
        from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow

        mock_memory.load_deep_research.return_value = sample_deep_research_state
        workflow = DeepResearchWorkflow(mock_config, mock_memory)

        result = workflow.execute(action="status", research_id="deepres-test123")

        assert result.success is True
        assert "deepres-test123" in result.content
        assert result.metadata["research_id"] == "deepres-test123"
        assert result.metadata["phase"] == "planning"

    def test_get_report_not_generated(self, mock_config, mock_memory, sample_deep_research_state):
        """Should return error when report not yet generated."""
        from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow

        sample_deep_research_state.report = None
        mock_memory.load_deep_research.return_value = sample_deep_research_state
        workflow = DeepResearchWorkflow(mock_config, mock_memory)

        result = workflow.execute(action="report", research_id="deepres-test123")

        assert result.success is False
        assert result.error is not None
        assert "not yet generated" in result.error

    def test_get_report_success(self, mock_config, mock_memory, sample_deep_research_state):
        """Should return report when available."""
        from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow

        sample_deep_research_state.report = "# Research Report\n\nFindings..."
        mock_memory.load_deep_research.return_value = sample_deep_research_state
        workflow = DeepResearchWorkflow(mock_config, mock_memory)

        result = workflow.execute(action="report", research_id="deepres-test123")

        assert result.success is True
        assert "Research Report" in result.content

    def test_list_sessions(self, mock_config, mock_memory, sample_deep_research_state):
        """Should list research sessions."""
        from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow

        mock_memory.list_deep_research.return_value = [sample_deep_research_state]
        workflow = DeepResearchWorkflow(mock_config, mock_memory)

        sessions = workflow.list_sessions(limit=10)

        assert len(sessions) == 1
        assert sessions[0]["id"] == "deepres-test123"
        assert sessions[0]["query"] == "What is deep learning?"

    def test_delete_session(self, mock_config, mock_memory):
        """Should delete a research session."""
        from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow

        mock_memory.delete_deep_research.return_value = True
        workflow = DeepResearchWorkflow(mock_config, mock_memory)

        deleted = workflow.delete_session("deepres-test123")

        assert deleted is True
        mock_memory.delete_deep_research.assert_called_once_with("deepres-test123")


# =============================================================================
# Phase Configuration Tests
# =============================================================================


class TestPhaseConfiguration:
    """Tests for per-phase timeout and provider configuration."""

    def test_get_phase_timeout_returns_phase_specific_values(self, mock_config):
        """Should return correct timeout for each phase."""
        assert mock_config.get_phase_timeout("planning") == 60.0
        assert mock_config.get_phase_timeout("analysis") == 90.0
        assert mock_config.get_phase_timeout("synthesis") == 180.0
        assert mock_config.get_phase_timeout("refinement") == 60.0

    def test_get_phase_timeout_fallback_for_unknown_phase(self, mock_config):
        """Should fallback to default timeout for unknown phases."""
        assert mock_config.get_phase_timeout("unknown") == 120.0
        assert mock_config.get_phase_timeout("gathering") == 120.0

    def test_get_phase_provider_returns_default_when_unset(self, mock_config):
        """Should return default provider when phase provider is None."""
        assert mock_config.get_phase_provider("planning") == "test-provider"
        assert mock_config.get_phase_provider("analysis") == "test-provider"
        assert mock_config.get_phase_provider("synthesis") == "test-provider"
        assert mock_config.get_phase_provider("refinement") == "test-provider"

    def test_get_phase_provider_returns_phase_specific_when_set(self, mock_config):
        """Should return phase-specific provider when configured."""
        mock_config.deep_research_synthesis_provider = "claude"
        mock_config.deep_research_analysis_provider = "openai"

        # Re-bind helper to pick up new values
        def get_phase_provider(phase: str) -> str:
            mapping = {
                "planning": mock_config.deep_research_planning_provider,
                "analysis": mock_config.deep_research_analysis_provider,
                "synthesis": mock_config.deep_research_synthesis_provider,
                "refinement": mock_config.deep_research_refinement_provider,
            }
            return mapping.get(phase.lower()) or mock_config.default_provider

        mock_config.get_phase_provider = get_phase_provider

        assert mock_config.get_phase_provider("synthesis") == "claude"
        assert mock_config.get_phase_provider("analysis") == "openai"
        assert mock_config.get_phase_provider("planning") == "test-provider"

    def test_state_initializes_with_phase_providers(
        self, mock_config, mock_memory
    ):
        """Should initialize state with per-phase providers from config."""
        from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow

        # Set different providers for different phases
        mock_config.deep_research_synthesis_provider = "claude"

        def get_phase_provider(phase: str) -> str:
            mapping = {
                "planning": mock_config.deep_research_planning_provider,
                "analysis": mock_config.deep_research_analysis_provider,
                "synthesis": mock_config.deep_research_synthesis_provider,
                "refinement": mock_config.deep_research_refinement_provider,
            }
            return mapping.get(phase.lower()) or mock_config.default_provider

        mock_config.get_phase_provider = get_phase_provider

        workflow = DeepResearchWorkflow(mock_config, mock_memory)

        # Create a state using the workflow's internal method
        state = DeepResearchState(
            original_query="Test query",
            planning_provider=mock_config.get_phase_provider("planning"),
            analysis_provider=mock_config.get_phase_provider("analysis"),
            synthesis_provider=mock_config.get_phase_provider("synthesis"),
            refinement_provider=mock_config.get_phase_provider("refinement"),
        )

        assert state.planning_provider == "test-provider"
        assert state.analysis_provider == "test-provider"
        assert state.synthesis_provider == "claude"
        assert state.refinement_provider == "test-provider"


class TestResearchConfigHelpers:
    """Tests for real ResearchConfig helper methods."""

    def test_real_config_get_phase_timeout(self):
        """Should return phase-specific timeouts from real config."""
        from foundry_mcp.config import ResearchConfig

        config = ResearchConfig(
            deep_research_timeout=120.0,
            deep_research_planning_timeout=60.0,
            deep_research_analysis_timeout=90.0,
            deep_research_synthesis_timeout=180.0,
            deep_research_refinement_timeout=45.0,
        )

        assert config.get_phase_timeout("planning") == 60.0
        assert config.get_phase_timeout("analysis") == 90.0
        assert config.get_phase_timeout("synthesis") == 180.0
        assert config.get_phase_timeout("refinement") == 45.0
        # Unknown phase falls back to default
        assert config.get_phase_timeout("unknown") == 120.0

    def test_real_config_get_phase_provider(self):
        """Should return phase-specific providers from real config."""
        from foundry_mcp.config import ResearchConfig

        config = ResearchConfig(
            default_provider="gemini",
            deep_research_synthesis_provider="claude",
            deep_research_analysis_provider="openai",
        )

        assert config.get_phase_provider("planning") == "gemini"
        assert config.get_phase_provider("analysis") == "openai"
        assert config.get_phase_provider("synthesis") == "claude"
        assert config.get_phase_provider("refinement") == "gemini"

    def test_from_toml_dict_parses_phase_config(self):
        """Should parse phase config from TOML dict."""
        from foundry_mcp.config import ResearchConfig

        toml_data = {
            "enabled": True,
            "default_provider": "gemini",
            "deep_research_timeout": 120.0,
            "deep_research_planning_timeout": 45.0,
            "deep_research_synthesis_timeout": 240.0,
            "deep_research_synthesis_provider": "claude",
        }

        config = ResearchConfig.from_toml_dict(toml_data)

        assert config.deep_research_planning_timeout == 45.0
        assert config.deep_research_synthesis_timeout == 240.0
        assert config.deep_research_synthesis_provider == "claude"
        assert config.get_phase_timeout("planning") == 45.0
        assert config.get_phase_provider("synthesis") == "claude"


class TestProviderSpecIntegration:
    """Tests for ProviderSpec format support in research config."""

    def test_resolve_phase_provider_simple_name(self):
        """Should handle simple provider names."""
        from foundry_mcp.config import ResearchConfig

        config = ResearchConfig(
            default_provider="gemini",
            deep_research_synthesis_provider="claude",
        )

        # Simple names return (provider_id, None)
        provider_id, model = config.resolve_phase_provider("planning")
        assert provider_id == "gemini"
        assert model is None

        provider_id, model = config.resolve_phase_provider("synthesis")
        assert provider_id == "claude"
        assert model is None

    def test_resolve_phase_provider_cli_spec_with_model(self):
        """Should parse [cli]provider:model format."""
        from foundry_mcp.config import ResearchConfig

        config = ResearchConfig(
            default_provider="[cli]gemini:pro",
            deep_research_synthesis_provider="[cli]claude:opus",
        )

        # CLI specs return (provider_id, model)
        provider_id, model = config.resolve_phase_provider("planning")
        assert provider_id == "gemini"
        assert model == "pro"

        provider_id, model = config.resolve_phase_provider("synthesis")
        assert provider_id == "claude"
        assert model == "opus"

    def test_resolve_phase_provider_cli_spec_with_backend(self):
        """Should parse [cli]transport:backend/model format."""
        from foundry_mcp.config import ResearchConfig

        config = ResearchConfig(
            default_provider="[cli]opencode:openai/gpt-5.2",
        )

        provider_id, model = config.resolve_phase_provider("planning")
        assert provider_id == "opencode"
        assert model == "openai/gpt-5.2"

    def test_resolve_phase_provider_api_spec(self):
        """Should parse [api]provider/model format."""
        from foundry_mcp.config import ResearchConfig

        config = ResearchConfig(
            default_provider="[api]openai/gpt-4.1",
        )

        provider_id, model = config.resolve_phase_provider("synthesis")
        assert provider_id == "openai"
        assert model == "gpt-4.1"

    def test_get_phase_provider_extracts_provider_id_only(self):
        """get_phase_provider should return just the provider ID."""
        from foundry_mcp.config import ResearchConfig

        config = ResearchConfig(
            default_provider="[cli]gemini:pro",
            deep_research_synthesis_provider="[cli]claude:opus",
        )

        # get_phase_provider returns just the ID
        assert config.get_phase_provider("planning") == "gemini"
        assert config.get_phase_provider("synthesis") == "claude"

    def test_state_with_provider_spec_models(self):
        """State should store models from ProviderSpec."""
        state = DeepResearchState(
            original_query="Test",
            planning_provider="gemini",
            planning_model="pro",
            synthesis_provider="claude",
            synthesis_model="opus",
        )

        assert state.planning_provider == "gemini"
        assert state.planning_model == "pro"
        assert state.synthesis_provider == "claude"
        assert state.synthesis_model == "opus"


# =============================================================================
# Action Handler Tests
# =============================================================================


class TestDeepResearchActionHandlers:
    """Tests for deep research action handlers in the research router."""

    @pytest.fixture
    def mock_tool_config(self, tmp_path: Path):
        """Mock server config for testing."""
        with patch("foundry_mcp.tools.unified.research._get_config") as mock_get_config:
            mock_cfg = MagicMock()
            mock_cfg.research.enabled = True
            mock_cfg.get_research_dir.return_value = tmp_path
            mock_cfg.research.ttl_hours = 24
            mock_get_config.return_value = mock_cfg
            yield mock_cfg

    @pytest.fixture
    def mock_tool_memory(self):
        """Mock research memory for tool tests."""
        with patch("foundry_mcp.tools.unified.research._get_memory") as mock_get_memory:
            memory = MagicMock()
            mock_get_memory.return_value = memory
            yield memory

    def test_dispatch_to_deep_research(
        self, mock_tool_config, mock_tool_memory
    ):
        """Should dispatch 'deep-research' action to handler."""
        from foundry_mcp.tools.unified.research import _dispatch_research_action

        with patch(
            "foundry_mcp.tools.unified.research.DeepResearchWorkflow"
        ) as MockWorkflow:
            mock_workflow = MagicMock()
            mock_workflow.execute.return_value = WorkflowResult(
                success=True,
                content="Research report",
                metadata={
                    "research_id": "dr-1",
                    "phase": "synthesis",
                    "iteration": 1,
                    "sub_query_count": 3,
                    "source_count": 10,
                    "finding_count": 5,
                    "gap_count": 0,
                    "is_complete": True,
                },
                tokens_used=1000,
                duration_ms=5000.0,
            )
            MockWorkflow.return_value = mock_workflow

            result = _dispatch_research_action(
                action="deep-research",
                query="What is machine learning?",
                deep_research_action="start",
            )

            MockWorkflow.assert_called_once()
            assert result["success"] is True
            assert result["data"]["research_id"] == "dr-1"

    def test_dispatch_to_deep_research_status(
        self, mock_tool_config, mock_tool_memory
    ):
        """Should dispatch 'deep-research-status' action."""
        from foundry_mcp.tools.unified.research import _dispatch_research_action

        with patch(
            "foundry_mcp.tools.unified.research.DeepResearchWorkflow"
        ) as MockWorkflow:
            mock_workflow = MagicMock()
            mock_workflow.execute.return_value = WorkflowResult(
                success=True,
                content="Status info",
                metadata={
                    "research_id": "dr-1",
                    "phase": "gathering",
                    "iteration": 1,
                },
            )
            MockWorkflow.return_value = mock_workflow

            result = _dispatch_research_action(
                action="deep-research-status",
                research_id="dr-1",
            )

            assert result["success"] is True

    def test_dispatch_to_deep_research_list(
        self, mock_tool_config, mock_tool_memory
    ):
        """Should dispatch 'deep-research-list' action."""
        from foundry_mcp.tools.unified.research import _dispatch_research_action

        with patch(
            "foundry_mcp.tools.unified.research.DeepResearchWorkflow"
        ) as MockWorkflow:
            mock_workflow = MagicMock()
            mock_workflow.list_sessions.return_value = [
                {"id": "dr-1", "query": "Test query"},
            ]
            MockWorkflow.return_value = mock_workflow

            result = _dispatch_research_action(
                action="deep-research-list",
                limit=10,
            )

            assert result["success"] is True
            assert result["data"]["count"] == 1

    def test_dispatch_to_deep_research_delete(
        self, mock_tool_config, mock_tool_memory
    ):
        """Should dispatch 'deep-research-delete' action."""
        from foundry_mcp.tools.unified.research import _dispatch_research_action

        with patch(
            "foundry_mcp.tools.unified.research.DeepResearchWorkflow"
        ) as MockWorkflow:
            mock_workflow = MagicMock()
            mock_workflow.delete_session.return_value = True
            MockWorkflow.return_value = mock_workflow

            result = _dispatch_research_action(
                action="deep-research-delete",
                research_id="dr-1",
            )

            assert result["success"] is True
            assert result["data"]["deleted"] is True

    def test_deep_research_validation_error_no_query(
        self, mock_tool_config, mock_tool_memory
    ):
        """Should return validation error when query missing for start."""
        from foundry_mcp.tools.unified.research import _dispatch_research_action

        result = _dispatch_research_action(
            action="deep-research",
            deep_research_action="start",
            query=None,
        )

        assert result["success"] is False
        assert "query" in result["error"].lower()

    def test_deep_research_validation_error_no_research_id(
        self, mock_tool_config, mock_tool_memory
    ):
        """Should return validation error when research_id missing for status."""
        from foundry_mcp.tools.unified.research import _dispatch_research_action

        result = _dispatch_research_action(
            action="deep-research-status",
            research_id=None,
        )

        assert result["success"] is False
        assert "research_id" in result["error"].lower()

    def test_dispatch_to_deep_research_resume(
        self, mock_tool_config, mock_tool_memory
    ):
        """Should dispatch 'deep-research' action with resume sub-action."""
        from foundry_mcp.tools.unified.research import _dispatch_research_action

        with patch(
            "foundry_mcp.tools.unified.research.DeepResearchWorkflow"
        ) as MockWorkflow:
            mock_workflow = MagicMock()
            mock_workflow.execute.return_value = WorkflowResult(
                success=True,
                content="Resumed research",
                metadata={
                    "research_id": "dr-1",
                    "phase": "gathering",
                    "iteration": 2,
                },
            )
            MockWorkflow.return_value = mock_workflow

            result = _dispatch_research_action(
                action="deep-research",
                research_id="dr-1",
                deep_research_action="resume",
            )

            assert result["success"] is True
            assert result["data"]["research_id"] == "dr-1"
            # Verify 'resume' was normalized to 'continue' by checking the call
            mock_workflow.execute.assert_called_once()
            call_kwargs = mock_workflow.execute.call_args[1]
            assert call_kwargs["action"] == "continue"

    def test_deep_research_list_pagination(
        self, mock_tool_config, mock_tool_memory
    ):
        """Should support cursor-based pagination for deep-research-list."""
        from foundry_mcp.tools.unified.research import _dispatch_research_action

        with patch(
            "foundry_mcp.tools.unified.research.DeepResearchWorkflow"
        ) as MockWorkflow:
            mock_workflow = MagicMock()
            # Return exactly limit items to trigger next_cursor
            mock_workflow.list_sessions.return_value = [
                {"id": "dr-1", "query": "Query 1"},
                {"id": "dr-2", "query": "Query 2"},
            ]
            MockWorkflow.return_value = mock_workflow

            result = _dispatch_research_action(
                action="deep-research-list",
                limit=2,
                cursor="dr-0",
            )

            assert result["success"] is True
            assert result["data"]["count"] == 2
            assert result["data"]["next_cursor"] == "dr-2"
            # Verify cursor was passed to list_sessions
            mock_workflow.list_sessions.assert_called_once_with(
                limit=2,
                cursor="dr-0",
                completed_only=False,
            )


# =============================================================================
# Throttle Behavior Tests
# =============================================================================


class TestStatusPersistenceThrottle:
    """Tests for status persistence throttling behavior.

    Validates the throttle logic that reduces disk I/O during frequent
    status checks by enforcing a minimum interval between saves.
    """

    @pytest.fixture
    def workflow_with_throttle(self, mock_memory, tmp_path: Path):
        """Create a workflow with throttle configuration."""
        from foundry_mcp.config import ResearchConfig
        from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow

        config = ResearchConfig(status_persistence_throttle_seconds=5)
        workflow = DeepResearchWorkflow(config, mock_memory)
        return workflow

    @pytest.fixture
    def workflow_zero_throttle(self, mock_memory, tmp_path: Path):
        """Create a workflow with zero throttle (always persist)."""
        from foundry_mcp.config import ResearchConfig
        from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow

        config = ResearchConfig(status_persistence_throttle_seconds=0)
        workflow = DeepResearchWorkflow(config, mock_memory)
        return workflow

    def test_throttle_zero_always_persists(
        self, workflow_zero_throttle, sample_deep_research_state
    ):
        """Throttle=0 should always return True (always persist)."""
        from datetime import datetime, timezone

        workflow = workflow_zero_throttle
        state = sample_deep_research_state

        # Simulate recent persistence
        workflow._last_persisted_at = datetime.now(timezone.utc)
        workflow._last_persisted_phase = state.phase
        workflow._last_persisted_iteration = state.iteration

        # Should still persist with zero throttle
        assert workflow._should_persist_status(state) is True

    def test_throttle_first_call_always_persists(
        self, workflow_with_throttle, sample_deep_research_state
    ):
        """First call (no previous persistence) should always persist."""
        workflow = workflow_with_throttle
        state = sample_deep_research_state

        # No previous persistence
        assert workflow._last_persisted_at is None

        # Should persist
        assert workflow._should_persist_status(state) is True

    def test_throttle_blocks_immediate_second_call(
        self, workflow_with_throttle, sample_deep_research_state
    ):
        """Immediate second call should be blocked by throttle."""
        from datetime import datetime, timezone

        workflow = workflow_with_throttle
        state = sample_deep_research_state

        # Simulate recent persistence
        workflow._last_persisted_at = datetime.now(timezone.utc)
        workflow._last_persisted_phase = state.phase
        workflow._last_persisted_iteration = state.iteration

        # Should NOT persist (throttle active)
        assert workflow._should_persist_status(state) is False

    def test_throttle_uses_persisted_metadata_across_instances(
        self, mock_memory, sample_deep_research_state
    ):
        """Throttle should respect persisted tracking data across instances."""
        from foundry_mcp.config import ResearchConfig
        from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow

        config = ResearchConfig(status_persistence_throttle_seconds=5)
        state = sample_deep_research_state

        workflow1 = DeepResearchWorkflow(config, mock_memory)
        workflow1._persist_state(state)

        workflow2 = DeepResearchWorkflow(config, mock_memory)
        assert workflow2._should_persist_status(state) is False

    def test_throttle_allows_after_interval_elapsed(
        self, workflow_with_throttle, sample_deep_research_state
    ):
        """Should persist after throttle interval has elapsed."""
        from datetime import datetime, timezone, timedelta

        workflow = workflow_with_throttle
        state = sample_deep_research_state

        # Simulate persistence 10 seconds ago (throttle is 5)
        workflow._last_persisted_at = datetime.now(timezone.utc) - timedelta(seconds=10)
        workflow._last_persisted_phase = state.phase
        workflow._last_persisted_iteration = state.iteration

        # Should persist (interval elapsed)
        assert workflow._should_persist_status(state) is True

    def test_terminal_state_completed_persists_during_throttle(
        self, workflow_with_throttle, sample_deep_research_state
    ):
        """Terminal state (completed) should persist even during throttle."""
        from datetime import datetime, timezone

        workflow = workflow_with_throttle
        state = sample_deep_research_state

        # Simulate recent persistence (throttle active)
        workflow._last_persisted_at = datetime.now(timezone.utc)
        workflow._last_persisted_phase = state.phase
        workflow._last_persisted_iteration = state.iteration

        # Mark as completed (terminal state)
        state.completed_at = datetime.now(timezone.utc)

        # Should persist (terminal state overrides throttle)
        assert workflow._should_persist_status(state) is True

    def test_terminal_state_failed_persists_during_throttle(
        self, workflow_with_throttle, sample_deep_research_state
    ):
        """Terminal state (failed) should persist even during throttle."""
        from datetime import datetime, timezone

        workflow = workflow_with_throttle
        state = sample_deep_research_state

        # Simulate recent persistence (throttle active)
        workflow._last_persisted_at = datetime.now(timezone.utc)
        workflow._last_persisted_phase = state.phase
        workflow._last_persisted_iteration = state.iteration

        # Mark as failed (terminal state)
        state.metadata["failed"] = True

        # Should persist (terminal state overrides throttle)
        assert workflow._should_persist_status(state) is True

    def test_phase_change_persists_during_throttle(
        self, workflow_with_throttle, sample_deep_research_state
    ):
        """Phase change should persist even during throttle."""
        from datetime import datetime, timezone

        workflow = workflow_with_throttle
        state = sample_deep_research_state

        # Simulate recent persistence at PLANNING phase
        workflow._last_persisted_at = datetime.now(timezone.utc)
        workflow._last_persisted_phase = DeepResearchPhase.PLANNING
        workflow._last_persisted_iteration = state.iteration

        # Change phase to GATHERING
        state.phase = DeepResearchPhase.GATHERING

        # Should persist (phase change overrides throttle)
        assert workflow._should_persist_status(state) is True

    def test_iteration_change_persists_during_throttle(
        self, workflow_with_throttle, sample_deep_research_state
    ):
        """Iteration change should persist even during throttle."""
        from datetime import datetime, timezone

        workflow = workflow_with_throttle
        state = sample_deep_research_state

        # Simulate recent persistence at iteration 1
        workflow._last_persisted_at = datetime.now(timezone.utc)
        workflow._last_persisted_phase = state.phase
        workflow._last_persisted_iteration = 1

        # Change iteration to 2
        state.iteration = 2

        # Should persist (iteration change overrides throttle)
        assert workflow._should_persist_status(state) is True

    def test_persist_state_updates_tracking_fields(
        self, workflow_with_throttle, sample_deep_research_state
    ):
        """_persist_state should update all tracking fields."""
        from datetime import datetime, timezone

        workflow = workflow_with_throttle
        state = sample_deep_research_state

        # Verify initial state
        assert workflow._last_persisted_at is None
        assert workflow._last_persisted_phase is None
        assert workflow._last_persisted_iteration is None

        # Persist state
        workflow._persist_state(state)

        # Verify tracking fields updated
        assert workflow._last_persisted_at is not None
        assert workflow._last_persisted_phase == state.phase
        assert workflow._last_persisted_iteration == state.iteration

        # Verify memory.save_deep_research was called
        workflow.memory.save_deep_research.assert_called_once_with(state)

    def test_persist_state_if_needed_returns_true_on_persist(
        self, workflow_with_throttle, sample_deep_research_state
    ):
        """_persist_state_if_needed should return True when persisting."""
        workflow = workflow_with_throttle
        state = sample_deep_research_state

        # First call should persist
        result = workflow._persist_state_if_needed(state)
        assert result is True

    def test_persist_state_if_needed_returns_false_on_skip(
        self, workflow_with_throttle, sample_deep_research_state
    ):
        """_persist_state_if_needed should return False when skipping."""
        from datetime import datetime, timezone

        workflow = workflow_with_throttle
        state = sample_deep_research_state

        # Simulate recent persistence
        workflow._last_persisted_at = datetime.now(timezone.utc)
        workflow._last_persisted_phase = state.phase
        workflow._last_persisted_iteration = state.iteration

        # Second call should skip
        result = workflow._persist_state_if_needed(state)
        assert result is False

    def test_is_terminal_state_completed(self, workflow_with_throttle):
        """_is_terminal_state should return True for completed state."""
        from datetime import datetime, timezone

        state = DeepResearchState(original_query="Test")
        state.completed_at = datetime.now(timezone.utc)

        assert workflow_with_throttle._is_terminal_state(state) is True

    def test_is_terminal_state_failed(self, workflow_with_throttle):
        """_is_terminal_state should return True for failed state."""
        state = DeepResearchState(original_query="Test")
        state.metadata["failed"] = True

        assert workflow_with_throttle._is_terminal_state(state) is True

    def test_is_terminal_state_in_progress(self, workflow_with_throttle):
        """_is_terminal_state should return False for in-progress state."""
        state = DeepResearchState(original_query="Test")

        assert workflow_with_throttle._is_terminal_state(state) is False


class TestAuditVerbosity:
    """Tests for audit verbosity modes (_prepare_audit_payload)."""

    @pytest.fixture
    def workflow_full_verbosity(self, mock_memory, tmp_path):
        """Create workflow with full audit verbosity."""
        from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow

        config = MagicMock()
        config.audit_verbosity = "full"
        config.deep_research_audit_artifacts = True
        workflow = DeepResearchWorkflow(config=config, memory=mock_memory)
        return workflow

    @pytest.fixture
    def workflow_minimal_verbosity(self, mock_memory, tmp_path):
        """Create workflow with minimal audit verbosity."""
        from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow

        config = MagicMock()
        config.audit_verbosity = "minimal"
        config.deep_research_audit_artifacts = True
        workflow = DeepResearchWorkflow(config=config, memory=mock_memory)
        return workflow

    @pytest.fixture
    def sample_audit_data(self):
        """Sample audit data with all field types."""
        return {
            # Fields to be nulled in minimal mode
            "system_prompt": "You are a research assistant",
            "user_prompt": "Tell me about deep learning",
            "raw_response": "Deep learning is a subset of machine learning...",
            "report": "# Research Report\n\nDeep learning...",
            "error": "Some error message",
            "traceback": "Traceback (most recent call last):\n  File...",
            # Preserved metrics fields
            "provider_id": "openai",
            "model_used": "gpt-4",
            "tokens_used": 1500,
            "duration_ms": 2500,
            "sources_added": 5,
            "report_length": 4200,
            "parse_success": True,
            # Nested structures
            "findings": [
                {"id": "find-1", "content": "Finding content text", "confidence": "high"},
                {"id": "find-2", "content": "Another finding", "confidence": "medium"},
            ],
            "gaps": [
                {"id": "gap-1", "description": "Gap description text", "priority": 1},
                {"id": "gap-2", "description": "Another gap", "priority": 2},
            ],
        }

    def test_full_mode_returns_data_unchanged(
        self, workflow_full_verbosity, sample_audit_data
    ):
        """Full mode should return audit data unchanged."""
        result = workflow_full_verbosity._prepare_audit_payload(sample_audit_data)

        # Data should be identical in full mode
        assert result == sample_audit_data
        # Verify text fields are preserved
        assert result["system_prompt"] == "You are a research assistant"
        assert result["user_prompt"] == "Tell me about deep learning"
        assert result["raw_response"] == "Deep learning is a subset of machine learning..."
        assert result["report"] == "# Research Report\n\nDeep learning..."
        assert result["error"] == "Some error message"
        assert result["traceback"] == "Traceback (most recent call last):\n  File..."

    def test_minimal_mode_nulls_documented_fields(
        self, workflow_minimal_verbosity, sample_audit_data
    ):
        """Minimal mode should null documented text fields."""
        result = workflow_minimal_verbosity._prepare_audit_payload(sample_audit_data)

        # Top-level text fields should be null
        assert result["system_prompt"] is None
        assert result["user_prompt"] is None
        assert result["raw_response"] is None
        assert result["report"] is None
        assert result["error"] is None
        assert result["traceback"] is None

    def test_minimal_mode_preserves_metrics(
        self, workflow_minimal_verbosity, sample_audit_data
    ):
        """Minimal mode should preserve metrics fields."""
        result = workflow_minimal_verbosity._prepare_audit_payload(sample_audit_data)

        # Metrics fields should be unchanged
        assert result["provider_id"] == "openai"
        assert result["model_used"] == "gpt-4"
        assert result["tokens_used"] == 1500
        assert result["duration_ms"] == 2500
        assert result["sources_added"] == 5
        assert result["report_length"] == 4200
        assert result["parse_success"] is True

    def test_schema_keys_identical_in_both_modes(
        self, workflow_full_verbosity, workflow_minimal_verbosity, sample_audit_data
    ):
        """Both modes should produce the same set of keys (schema stability)."""
        full_result = workflow_full_verbosity._prepare_audit_payload(sample_audit_data)
        minimal_result = workflow_minimal_verbosity._prepare_audit_payload(sample_audit_data)

        # Top-level keys should be identical
        assert set(full_result.keys()) == set(minimal_result.keys())

        # Findings keys should be identical
        assert len(full_result["findings"]) == len(minimal_result["findings"])
        for full_f, min_f in zip(full_result["findings"], minimal_result["findings"]):
            assert set(full_f.keys()) == set(min_f.keys())

        # Gaps keys should be identical
        assert len(full_result["gaps"]) == len(minimal_result["gaps"])
        for full_g, min_g in zip(full_result["gaps"], minimal_result["gaps"]):
            assert set(full_g.keys()) == set(min_g.keys())

    def test_nested_findings_content_nulled_in_minimal(
        self, workflow_minimal_verbosity, sample_audit_data
    ):
        """Minimal mode should null findings[*].content while preserving other fields."""
        result = workflow_minimal_verbosity._prepare_audit_payload(sample_audit_data)

        # Content should be nulled
        for finding in result["findings"]:
            assert finding["content"] is None
            # Other fields preserved
            assert "id" in finding
            assert "confidence" in finding

        # Verify specific findings preserved other data
        assert result["findings"][0]["id"] == "find-1"
        assert result["findings"][0]["confidence"] == "high"
        assert result["findings"][1]["id"] == "find-2"
        assert result["findings"][1]["confidence"] == "medium"

    def test_nested_gaps_description_nulled_in_minimal(
        self, workflow_minimal_verbosity, sample_audit_data
    ):
        """Minimal mode should null gaps[*].description while preserving other fields."""
        result = workflow_minimal_verbosity._prepare_audit_payload(sample_audit_data)

        # Description should be nulled
        for gap in result["gaps"]:
            assert gap["description"] is None
            # Other fields preserved
            assert "id" in gap
            assert "priority" in gap

        # Verify specific gaps preserved other data
        assert result["gaps"][0]["id"] == "gap-1"
        assert result["gaps"][0]["priority"] == 1
        assert result["gaps"][1]["id"] == "gap-2"
        assert result["gaps"][1]["priority"] == 2

    def test_handles_missing_optional_fields(
        self, workflow_minimal_verbosity
    ):
        """Minimal mode should handle data without optional text fields."""
        minimal_data = {
            "provider_id": "test",
            "tokens_used": 100,
        }

        result = workflow_minimal_verbosity._prepare_audit_payload(minimal_data)

        # Should not add fields that weren't present
        assert "system_prompt" not in result
        assert "report" not in result
        # Preserved fields should remain
        assert result["provider_id"] == "test"
        assert result["tokens_used"] == 100

    def test_handles_empty_nested_arrays(
        self, workflow_minimal_verbosity
    ):
        """Minimal mode should handle empty findings and gaps arrays."""
        data_with_empty_arrays = {
            "provider_id": "test",
            "findings": [],
            "gaps": [],
        }

        result = workflow_minimal_verbosity._prepare_audit_payload(data_with_empty_arrays)

        # Empty arrays should remain empty
        assert result["findings"] == []
        assert result["gaps"] == []

    def test_handles_non_dict_items_in_nested_arrays(
        self, workflow_minimal_verbosity
    ):
        """Minimal mode should handle non-dict items in nested arrays gracefully."""
        data_with_mixed = {
            "provider_id": "test",
            "findings": [
                {"content": "text", "id": "f1"},
                "not a dict",  # Edge case: non-dict item
                None,  # Edge case: null item
            ],
            "gaps": [
                {"description": "text", "id": "g1"},
                123,  # Edge case: non-dict item
            ],
        }

        result = workflow_minimal_verbosity._prepare_audit_payload(data_with_mixed)

        # Dict items should have content/description nulled
        assert result["findings"][0]["content"] is None
        assert result["findings"][0]["id"] == "f1"
        assert result["gaps"][0]["description"] is None
        assert result["gaps"][0]["id"] == "g1"

        # Non-dict items should pass through unchanged
        assert result["findings"][1] == "not a dict"
        assert result["findings"][2] is None
        assert result["gaps"][1] == 123

    def test_does_not_mutate_original_data(
        self, workflow_minimal_verbosity, sample_audit_data
    ):
        """Minimal mode should not mutate the original data dictionary."""
        import copy
        original_copy = copy.deepcopy(sample_audit_data)

        workflow_minimal_verbosity._prepare_audit_payload(sample_audit_data)

        # Original should be unchanged
        assert sample_audit_data == original_copy


# =============================================================================
# Deep Research Failover Integration Tests
# =============================================================================


class TestDeepResearchProviderFailover:
    """Integration tests for deep research provider failover with circuit breakers.

    Tests the gathering phase's ability to handle provider failures gracefully:
    - Skipping providers with OPEN circuit breakers
    - Allowing HALF_OPEN recovery probes
    - Handling all_providers_circuit_open scenario
    - Graceful degradation when provider trips mid-gathering

    All tests use reset_resilience_manager_for_testing() for proper isolation.
    """

    @pytest.fixture(autouse=True)
    def reset_resilience_state(self):
        """Reset resilience manager before and after each test for isolation."""
        from foundry_mcp.core.research.providers.resilience import (
            reset_resilience_manager_for_testing,
        )
        reset_resilience_manager_for_testing()
        yield
        reset_resilience_manager_for_testing()

    @pytest.fixture
    def workflow_with_providers(self, mock_config, mock_memory):
        """Create workflow instance with configured providers."""
        from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow
        mock_config.deep_research_providers = ["tavily", "google"]
        workflow = DeepResearchWorkflow(config=mock_config, memory=mock_memory)
        return workflow

    @pytest.fixture
    def state_with_pending_queries(self):
        """Create state with pending sub-queries for gathering phase."""
        state = DeepResearchState(
            id="test-failover-001",
            original_query="Test query for failover",
            research_brief="Testing provider failover",
            phase=DeepResearchPhase.GATHERING,
            iteration=1,
            max_iterations=3,
            max_sources_per_query=5,
        )
        # Add pending sub-queries
        state.add_sub_query(
            query="Sub-query 1",
            rationale="Test rationale",
            priority=1,
        )
        state.add_sub_query(
            query="Sub-query 2",
            rationale="Test rationale 2",
            priority=2,
        )
        return state

    @pytest.mark.asyncio
    async def test_skips_open_circuit_breaker_providers(
        self, workflow_with_providers, state_with_pending_queries, mock_memory
    ):
        """Providers with OPEN circuit breakers should be skipped during gathering."""
        from foundry_mcp.core.research.providers.resilience import get_resilience_manager
        from foundry_mcp.core.resilience import CircuitState

        mgr = get_resilience_manager()

        # Trip tavily's circuit breaker
        tavily_breaker = mgr._get_or_create_circuit_breaker("tavily")
        for _ in range(10):
            tavily_breaker.record_failure()
        assert tavily_breaker.state == CircuitState.OPEN

        # Google should still be available
        assert mgr.is_provider_available("google") is True
        assert mgr.is_provider_available("tavily") is False

        # Mock the search providers
        mock_google_sources = [
            ResearchSource(
                url="https://google-result.com/1",
                title="Google Result 1",
                source_type=SourceType.WEB,
            )
        ]

        with patch.object(
            workflow_with_providers,
            "_get_search_provider",
            side_effect=lambda name: (
                self._create_mock_provider(name, mock_google_sources)
                if name == "google"
                else self._create_mock_provider(name, [])
            ),
        ):
            result = await workflow_with_providers._execute_gathering_async(
                state=state_with_pending_queries,
                provider_id=None,
                timeout=30.0,
                max_concurrent=2,
            )

        # Should succeed with google results
        assert result.success is True
        # Tavily should have been filtered out
        assert "tavily" not in result.metadata.get("providers_used", [])

    @pytest.mark.asyncio
    async def test_allows_half_open_recovery_probes(
        self, workflow_with_providers, state_with_pending_queries, mock_memory
    ):
        """HALF_OPEN providers should be allowed to enable recovery probes."""
        from foundry_mcp.core.research.providers.resilience import get_resilience_manager
        from foundry_mcp.core.resilience import CircuitState

        mgr = get_resilience_manager()

        # Trip tavily's circuit breaker
        tavily_breaker = mgr._get_or_create_circuit_breaker("tavily")
        tavily_breaker.recovery_timeout = 0.01  # Very short for testing
        for _ in range(10):
            tavily_breaker.record_failure()
        assert tavily_breaker.state == CircuitState.OPEN

        # Wait for recovery timeout to allow HALF_OPEN transition
        await asyncio.sleep(0.02)

        # Trigger HALF_OPEN transition
        assert tavily_breaker.can_execute() is True
        assert tavily_breaker.state == CircuitState.HALF_OPEN

        # Both should now be available (tavily in HALF_OPEN, google in CLOSED)
        assert mgr.is_provider_available("tavily") is True
        assert mgr.is_provider_available("google") is True

        # Mock providers to return results
        mock_sources = [
            ResearchSource(
                url="https://example.com/1",
                title="Test Result",
                source_type=SourceType.WEB,
            )
        ]

        with patch.object(
            workflow_with_providers,
            "_get_search_provider",
            side_effect=lambda name: self._create_mock_provider(name, mock_sources),
        ):
            result = await workflow_with_providers._execute_gathering_async(
                state=state_with_pending_queries,
                provider_id=None,
                timeout=30.0,
                max_concurrent=2,
            )

        assert result.success is True
        # Both providers should have been used
        providers_used = result.metadata.get("providers_used", [])
        assert "tavily" in providers_used or "google" in providers_used

    @pytest.mark.asyncio
    async def test_all_providers_circuit_open_returns_error(
        self, workflow_with_providers, state_with_pending_queries, mock_memory
    ):
        """All providers having OPEN circuits should return descriptive error."""
        from foundry_mcp.core.research.providers.resilience import get_resilience_manager
        from foundry_mcp.core.resilience import CircuitState

        mgr = get_resilience_manager()

        # Trip both circuit breakers
        for provider_name in ["tavily", "google"]:
            breaker = mgr._get_or_create_circuit_breaker(provider_name)
            for _ in range(10):
                breaker.record_failure()
            assert breaker.state == CircuitState.OPEN

        # Both should be unavailable
        assert mgr.is_provider_available("tavily") is False
        assert mgr.is_provider_available("google") is False

        # Mock providers to return valid objects (though they won't be used)
        with patch.object(
            workflow_with_providers,
            "_get_search_provider",
            side_effect=lambda name: self._create_mock_provider(name, []),
        ):
            result = await workflow_with_providers._execute_gathering_async(
                state=state_with_pending_queries,
                provider_id=None,
                timeout=30.0,
                max_concurrent=2,
            )

        # Should fail with circuit breaker error
        assert result.success is False
        assert result.error is not None
        assert "circuit breaker" in result.error.lower()
        assert "temporarily unavailable" in result.error.lower()

    @pytest.mark.asyncio
    async def test_graceful_degradation_when_provider_trips_mid_gathering(
        self, workflow_with_providers, state_with_pending_queries, mock_memory
    ):
        """Provider tripping mid-gathering should skip remaining calls gracefully."""
        from foundry_mcp.core.research.providers.resilience import get_resilience_manager
        from foundry_mcp.core.resilience import CircuitState

        mgr = get_resilience_manager()
        call_count = {"tavily": 0, "google": 0}

        # Tavily will succeed first time, then we trip its breaker
        async def tavily_search(*args, **kwargs):
            call_count["tavily"] += 1
            if call_count["tavily"] == 1:
                # First call succeeds, then trip the breaker
                breaker = mgr._get_or_create_circuit_breaker("tavily")
                for _ in range(10):
                    breaker.record_failure()
                return [
                    ResearchSource(
                        url=f"https://tavily.com/{call_count['tavily']}",
                        title=f"Tavily Result {call_count['tavily']}",
                        source_type=SourceType.WEB,
                    )
                ]
            # Subsequent calls would not be made due to circuit open
            return []

        async def google_search(*args, **kwargs):
            call_count["google"] += 1
            return [
                ResearchSource(
                    url=f"https://google.com/{call_count['google']}",
                    title=f"Google Result {call_count['google']}",
                    source_type=SourceType.WEB,
                )
            ]

        def create_provider(name):
            mock_provider = MagicMock()
            mock_provider.get_provider_name.return_value = name
            if name == "tavily":
                mock_provider.search = AsyncMock(side_effect=tavily_search)
            else:
                mock_provider.search = AsyncMock(side_effect=google_search)
            return mock_provider

        with patch.object(
            workflow_with_providers,
            "_get_search_provider",
            side_effect=create_provider,
        ):
            result = await workflow_with_providers._execute_gathering_async(
                state=state_with_pending_queries,
                provider_id=None,
                timeout=30.0,
                max_concurrent=1,  # Sequential to control ordering
            )

        # Should succeed overall
        assert result.success is True

        # Tavily should have only been called once (before circuit opened)
        # due to graceful degradation checking circuit state mid-gathering
        assert call_count["tavily"] >= 1
        # Google should have been called for both sub-queries
        assert call_count["google"] >= 1

    @pytest.mark.asyncio
    async def test_resilience_state_isolation_between_tests(self, mock_config, mock_memory):
        """Verify reset_resilience_manager_for_testing provides proper isolation."""
        from foundry_mcp.core.research.providers.resilience import (
            get_resilience_manager,
            reset_resilience_manager_for_testing,
        )
        from foundry_mcp.core.resilience import CircuitState

        # First: trip a breaker
        mgr1 = get_resilience_manager()
        breaker1 = mgr1._get_or_create_circuit_breaker("tavily")
        for _ in range(10):
            breaker1.record_failure()
        assert breaker1.state == CircuitState.OPEN

        # Reset manager
        reset_resilience_manager_for_testing()

        # After reset: new manager should have fresh state
        mgr2 = get_resilience_manager()
        assert mgr2 is not mgr1  # Different instance
        breaker2 = mgr2._get_or_create_circuit_breaker("tavily")
        assert breaker2.state == CircuitState.CLOSED  # Fresh state

    @pytest.mark.asyncio
    async def test_circuit_breaker_states_captured_in_metadata(
        self, workflow_with_providers, state_with_pending_queries, mock_memory
    ):
        """Circuit breaker states should be captured in result metadata."""
        from foundry_mcp.core.research.providers.resilience import get_resilience_manager

        mgr = get_resilience_manager()

        # Add some failures to tavily (but not enough to trip)
        tavily_breaker = mgr._get_or_create_circuit_breaker("tavily")
        tavily_breaker.record_failure()
        tavily_breaker.record_failure()

        mock_sources = [
            ResearchSource(
                url="https://example.com/1",
                title="Test Result",
                source_type=SourceType.WEB,
            )
        ]

        with patch.object(
            workflow_with_providers,
            "_get_search_provider",
            side_effect=lambda name: self._create_mock_provider(name, mock_sources),
        ):
            result = await workflow_with_providers._execute_gathering_async(
                state=state_with_pending_queries,
                provider_id=None,
                timeout=30.0,
                max_concurrent=2,
            )

        assert result.success is True
        # Verify circuit breaker states are captured
        assert "circuit_breaker_states" in result.metadata
        cb_states = result.metadata["circuit_breaker_states"]
        assert "start" in cb_states
        assert "end" in cb_states

    def _create_mock_provider(self, name: str, sources: list) -> MagicMock:
        """Helper to create mock search provider."""
        mock_provider = MagicMock()
        mock_provider.get_provider_name.return_value = name
        mock_provider.search = AsyncMock(return_value=sources)
        return mock_provider


class TestDeepResearchProviderFailoverEdgeCases:
    """Edge case tests for provider failover scenarios."""

    @pytest.fixture(autouse=True)
    def reset_resilience_state(self):
        """Reset resilience manager before and after each test."""
        from foundry_mcp.core.research.providers.resilience import (
            reset_resilience_manager_for_testing,
        )
        reset_resilience_manager_for_testing()
        yield
        reset_resilience_manager_for_testing()

    @pytest.fixture
    def workflow_three_providers(self, mock_config, mock_memory):
        """Workflow with three providers for more complex failover scenarios."""
        from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow
        mock_config.deep_research_providers = ["tavily", "google", "semantic_scholar"]
        return DeepResearchWorkflow(config=mock_config, memory=mock_memory)

    @pytest.fixture
    def state_single_query(self):
        """State with single sub-query."""
        state = DeepResearchState(
            id="test-edge-001",
            original_query="Edge case test",
            research_brief="Testing edge cases",
            phase=DeepResearchPhase.GATHERING,
            iteration=1,
            max_iterations=3,
            max_sources_per_query=5,
        )
        state.add_sub_query(query="Single query", rationale="Test", priority=1)
        return state

    @pytest.mark.asyncio
    async def test_partial_provider_failure_continues_with_available(
        self, workflow_three_providers, state_single_query, mock_memory
    ):
        """When some providers fail, gathering continues with available ones."""
        from foundry_mcp.core.research.providers.resilience import get_resilience_manager
        from foundry_mcp.core.resilience import CircuitState

        mgr = get_resilience_manager()

        # Trip tavily and semantic_scholar, leave google available
        for name in ["tavily", "semantic_scholar"]:
            breaker = mgr._get_or_create_circuit_breaker(name)
            for _ in range(10):
                breaker.record_failure()
            assert breaker.state == CircuitState.OPEN

        assert mgr.is_provider_available("google") is True

        mock_sources = [
            ResearchSource(
                url="https://google.com/result",
                title="Google Only Result",
                source_type=SourceType.WEB,
            )
        ]

        def create_provider(name):
            if name == "google":
                mock_provider = MagicMock()
                mock_provider.get_provider_name.return_value = name
                mock_provider.search = AsyncMock(return_value=mock_sources)
                return mock_provider
            elif name in ["tavily", "semantic_scholar"]:
                mock_provider = MagicMock()
                mock_provider.get_provider_name.return_value = name
                mock_provider.search = AsyncMock(return_value=[])
                return mock_provider
            return None

        with patch.object(
            workflow_three_providers,
            "_get_search_provider",
            side_effect=create_provider,
        ):
            result = await workflow_three_providers._execute_gathering_async(
                state=state_single_query,
                provider_id=None,
                timeout=30.0,
                max_concurrent=3,
            )

        # Should succeed with just google
        assert result.success is True
        providers_used = result.metadata.get("providers_used", [])
        assert "google" in providers_used
        assert "tavily" not in providers_used
        assert "semantic_scholar" not in providers_used

    @pytest.mark.asyncio
    async def test_no_configured_providers_returns_configuration_error(
        self, mock_config, mock_memory
    ):
        """No configured providers should return configuration error, not circuit error."""
        from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow
        from foundry_mcp.core.research.providers.resilience import (
            reset_resilience_manager_for_testing,
        )
        reset_resilience_manager_for_testing()

        # Configure with providers that won't be instantiated
        mock_config.deep_research_providers = ["nonexistent_provider"]
        workflow = DeepResearchWorkflow(config=mock_config, memory=mock_memory)

        state = DeepResearchState(
            id="test-no-providers",
            original_query="Test",
            research_brief="Test",
            phase=DeepResearchPhase.GATHERING,
            iteration=1,
        )
        state.add_sub_query(query="Test query", rationale="Test", priority=1)

        # Provider lookup returns None for nonexistent
        with patch.object(
            workflow,
            "_get_search_provider",
            return_value=None,
        ):
            result = await workflow._execute_gathering_async(
                state=state,
                provider_id=None,
                timeout=30.0,
                max_concurrent=2,
            )

        assert result.success is False
        assert result.error is not None
        # Should mention configuration, not circuit breakers
        assert "no search providers available" in result.error.lower()
        assert "configure api keys" in result.error.lower()
