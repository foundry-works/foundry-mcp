"""Tests for the shared LLM call lifecycle helpers.

Covers execute_llm_call and finalize_phase used by all deep research
phase mixins (planning, analysis, synthesis, refinement).
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from foundry_mcp.core.errors.provider import ContextWindowError
from foundry_mcp.core.research.models.deep_research import (
    DeepResearchPhase,
    DeepResearchState,
)
from foundry_mcp.core.research.workflows.base import WorkflowResult
from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
    LLMCallResult,
    execute_llm_call,
    finalize_phase,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_workflow():
    """Create a mock DeepResearchWorkflow with all required methods/attrs."""
    wf = MagicMock()
    wf._execute_provider_async = AsyncMock()
    wf._write_audit_event = MagicMock()
    wf.memory.save_deep_research = MagicMock()
    wf.config.get_phase_fallback_providers = MagicMock(return_value=[])
    wf.config.deep_research_max_retries = 2
    wf.config.deep_research_retry_delay = 1.0
    return wf


@pytest.fixture
def sample_state():
    """Create a minimal DeepResearchState."""
    return DeepResearchState(
        id="deepres-lifecycle-test",
        original_query="lifecycle test query",
        phase=DeepResearchPhase.PLANNING,
        iteration=1,
        max_iterations=3,
    )


def _make_success_result(**overrides):
    """Build a successful WorkflowResult with sensible defaults."""
    defaults = dict(
        success=True,
        content="test response content",
        provider_id="test-provider",
        model_used="test-model",
        tokens_used=30,
        input_tokens=10,
        output_tokens=20,
        cached_tokens=0,
        duration_ms=150.0,
        metadata={},
    )
    defaults.update(overrides)
    return WorkflowResult(**defaults)


def _make_failure_result(**overrides):
    """Build a failed WorkflowResult."""
    defaults = dict(
        success=False,
        content="",
        error="provider error",
        metadata={},
    )
    defaults.update(overrides)
    return WorkflowResult(**defaults)


# ---------------------------------------------------------------------------
# execute_llm_call — success path
# ---------------------------------------------------------------------------


class TestExecuteLLMCallSuccess:
    """Tests for the happy-path through execute_llm_call."""

    @pytest.mark.asyncio
    async def test_returns_llm_call_result(self, mock_workflow, sample_state):
        """Should return an LLMCallResult on successful provider call."""
        mock_workflow._execute_provider_async.return_value = _make_success_result()

        ret = await execute_llm_call(
            workflow=mock_workflow,
            state=sample_state,
            phase_name="planning",
            system_prompt="sys",
            user_prompt="usr",
            provider_id="test-provider",
            model="test-model",
            temperature=0.7,
            timeout=60.0,
        )

        assert isinstance(ret, LLMCallResult)
        assert ret.result.success is True
        assert ret.result.content == "test response content"
        assert ret.llm_call_duration_ms > 0

    @pytest.mark.asyncio
    async def test_updates_heartbeat(self, mock_workflow, sample_state):
        """Should update heartbeat and persist state before the call."""
        mock_workflow._execute_provider_async.return_value = _make_success_result()

        await execute_llm_call(
            workflow=mock_workflow,
            state=sample_state,
            phase_name="planning",
            system_prompt="sys",
            user_prompt="usr",
            provider_id="p",
            model="m",
            temperature=0.5,
            timeout=30.0,
        )

        assert sample_state.last_heartbeat_at is not None
        mock_workflow.memory.save_deep_research.assert_called_once_with(sample_state)

    @pytest.mark.asyncio
    async def test_emits_audit_events(self, mock_workflow, sample_state):
        """Should emit llm.call.started and llm.call.completed audit events."""
        mock_workflow._execute_provider_async.return_value = _make_success_result()

        await execute_llm_call(
            workflow=mock_workflow,
            state=sample_state,
            phase_name="analysis",
            system_prompt="sys",
            user_prompt="usr",
            provider_id="p",
            model="m",
            temperature=0.3,
            timeout=90.0,
        )

        event_types = [call.args[1] for call in mock_workflow._write_audit_event.call_args_list]
        assert "llm.call.started" in event_types
        assert "llm.call.completed" in event_types

    @pytest.mark.asyncio
    async def test_tracks_tokens(self, mock_workflow, sample_state):
        """Should add tokens_used to state.total_tokens_used."""
        sample_state.total_tokens_used = 100
        mock_workflow._execute_provider_async.return_value = _make_success_result(tokens_used=50)

        await execute_llm_call(
            workflow=mock_workflow,
            state=sample_state,
            phase_name="synthesis",
            system_prompt="sys",
            user_prompt="usr",
            provider_id="p",
            model="m",
            temperature=0.5,
            timeout=60.0,
        )

        assert sample_state.total_tokens_used == 150

    @pytest.mark.asyncio
    async def test_appends_phase_metrics(self, mock_workflow, sample_state):
        """Should append a PhaseMetrics entry to state."""
        mock_workflow._execute_provider_async.return_value = _make_success_result(
            duration_ms=200.0, input_tokens=15, output_tokens=25, cached_tokens=5
        )

        await execute_llm_call(
            workflow=mock_workflow,
            state=sample_state,
            phase_name="refinement",
            system_prompt="sys",
            user_prompt="usr",
            provider_id="p",
            model="m",
            temperature=0.4,
            timeout=60.0,
        )

        assert len(sample_state.phase_metrics) == 1
        pm = sample_state.phase_metrics[0]
        assert pm.phase == "refinement"
        assert pm.duration_ms == 200.0
        assert pm.input_tokens == 15
        assert pm.output_tokens == 25
        assert pm.cached_tokens == 5

    @pytest.mark.asyncio
    async def test_passes_correct_params_to_provider(self, mock_workflow, sample_state):
        """Should forward all parameters to _execute_provider_async."""
        mock_workflow._execute_provider_async.return_value = _make_success_result()

        await execute_llm_call(
            workflow=mock_workflow,
            state=sample_state,
            phase_name="planning",
            system_prompt="my_sys",
            user_prompt="my_usr",
            provider_id="my-provider",
            model="my-model",
            temperature=0.7,
            timeout=120.0,
        )

        call_kwargs = mock_workflow._execute_provider_async.call_args.kwargs
        assert call_kwargs["prompt"] == "my_usr"
        assert call_kwargs["system_prompt"] == "my_sys"
        assert call_kwargs["provider_id"] == "my-provider"
        assert call_kwargs["model"] == "my-model"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["timeout"] == 120.0
        assert call_kwargs["phase"] == "planning"


# ---------------------------------------------------------------------------
# execute_llm_call — ContextWindowError
# ---------------------------------------------------------------------------


class TestExecuteLLMCallContextWindowError:
    """Tests for ContextWindowError handling in execute_llm_call."""

    @pytest.mark.asyncio
    async def test_returns_error_workflow_result(self, mock_workflow, sample_state):
        """Should return a failed WorkflowResult on ContextWindowError."""
        mock_workflow._execute_provider_async.side_effect = ContextWindowError(
            "Context window exceeded",
            prompt_tokens=5000,
            max_tokens=4096,
            provider="test-provider",
        )

        ret = await execute_llm_call(
            workflow=mock_workflow,
            state=sample_state,
            phase_name="analysis",
            system_prompt="sys",
            user_prompt="usr",
            provider_id="test-provider",
            model="m",
            temperature=0.3,
            timeout=60.0,
        )

        assert isinstance(ret, WorkflowResult)
        assert ret.success is False
        assert ret.metadata["error_type"] == "context_window_exceeded"
        assert ret.metadata["prompt_tokens"] == 5000
        assert ret.metadata["max_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_includes_error_metadata(self, mock_workflow, sample_state):
        """Should merge error_metadata into the returned metadata."""
        mock_workflow._execute_provider_async.side_effect = ContextWindowError(
            "Context window exceeded",
            prompt_tokens=5000,
            max_tokens=4096,
        )

        ret = await execute_llm_call(
            workflow=mock_workflow,
            state=sample_state,
            phase_name="synthesis",
            system_prompt="sys",
            user_prompt="usr",
            provider_id="p",
            model="m",
            temperature=0.5,
            timeout=60.0,
            error_metadata={"finding_count": 42, "guidance": "reduce findings"},
        )

        assert isinstance(ret, WorkflowResult)
        assert ret.metadata["finding_count"] == 42
        assert ret.metadata["guidance"] == "reduce findings"

    @pytest.mark.asyncio
    async def test_emits_error_audit_and_metrics(self, mock_workflow, sample_state):
        """Should emit llm.call.completed with error status on ContextWindowError."""
        mock_workflow._execute_provider_async.side_effect = ContextWindowError(
            "Context window exceeded",
            prompt_tokens=5000,
            max_tokens=4096,
        )

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.get_metrics"
        ) as mock_get_metrics:
            mock_metrics = MagicMock()
            mock_get_metrics.return_value = mock_metrics

            await execute_llm_call(
                workflow=mock_workflow,
                state=sample_state,
                phase_name="analysis",
                system_prompt="sys",
                user_prompt="usr",
                provider_id="p",
                model="m",
                temperature=0.3,
                timeout=60.0,
            )

            # Verify error audit event
            completed_calls = [
                c for c in mock_workflow._write_audit_event.call_args_list if c.args[1] == "llm.call.completed"
            ]
            assert len(completed_calls) == 1
            assert completed_calls[0].kwargs["data"]["status"] == "error"

            # Verify metrics
            mock_metrics.histogram.assert_called_once()


# ---------------------------------------------------------------------------
# execute_llm_call — provider failure (non-exception)
# ---------------------------------------------------------------------------


class TestExecuteLLMCallProviderFailure:
    """Tests for non-exception provider failures (timeout, generic error)."""

    @pytest.mark.asyncio
    async def test_returns_failed_result_directly(self, mock_workflow, sample_state):
        """Should return the failed WorkflowResult from the provider."""
        failed = _make_failure_result(error="provider unavailable", metadata={"timeout": False})
        mock_workflow._execute_provider_async.return_value = failed

        ret = await execute_llm_call(
            workflow=mock_workflow,
            state=sample_state,
            phase_name="planning",
            system_prompt="sys",
            user_prompt="usr",
            provider_id="p",
            model="m",
            temperature=0.7,
            timeout=60.0,
        )

        assert isinstance(ret, WorkflowResult)
        assert ret.success is False
        assert ret.error == "provider unavailable"

    @pytest.mark.asyncio
    async def test_timeout_failure(self, mock_workflow, sample_state):
        """Should handle timeout metadata correctly."""
        failed = _make_failure_result(error="timeout", metadata={"timeout": True, "providers_tried": ["a", "b"]})
        mock_workflow._execute_provider_async.return_value = failed

        ret = await execute_llm_call(
            workflow=mock_workflow,
            state=sample_state,
            phase_name="refinement",
            system_prompt="sys",
            user_prompt="usr",
            provider_id="p",
            model="m",
            temperature=0.4,
            timeout=60.0,
        )

        assert isinstance(ret, WorkflowResult)
        assert ret.success is False

    @pytest.mark.asyncio
    async def test_no_token_tracking_on_failure(self, mock_workflow, sample_state):
        """Should not modify state tokens or metrics on failure."""
        sample_state.total_tokens_used = 100
        initial_metrics_count = len(sample_state.phase_metrics)

        mock_workflow._execute_provider_async.return_value = _make_failure_result()

        await execute_llm_call(
            workflow=mock_workflow,
            state=sample_state,
            phase_name="planning",
            system_prompt="sys",
            user_prompt="usr",
            provider_id="p",
            model="m",
            temperature=0.7,
            timeout=60.0,
        )

        assert sample_state.total_tokens_used == 100
        assert len(sample_state.phase_metrics) == initial_metrics_count


# ---------------------------------------------------------------------------
# finalize_phase
# ---------------------------------------------------------------------------


class TestFinalizePhase:
    """Tests for finalize_phase helper."""

    def test_emits_phase_completed_audit(self, mock_workflow, sample_state):
        """Should emit a phase.completed audit event."""
        phase_start = time.perf_counter() - 0.5  # 500ms ago

        finalize_phase(mock_workflow, sample_state, "planning", phase_start)

        calls = mock_workflow._write_audit_event.call_args_list
        assert len(calls) == 1
        assert calls[0].args[1] == "phase.completed"
        data = calls[0].kwargs["data"]
        assert data["phase_name"] == "planning"
        assert data["iteration"] == sample_state.iteration
        assert data["task_id"] == sample_state.id
        assert data["duration_ms"] > 0

    def test_emits_phase_duration_metric(self, mock_workflow, sample_state):
        """Should emit a duration histogram metric."""
        phase_start = time.perf_counter() - 0.1

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.get_metrics"
        ) as mock_get_metrics:
            mock_metrics = MagicMock()
            mock_get_metrics.return_value = mock_metrics

            finalize_phase(mock_workflow, sample_state, "synthesis", phase_start)

            mock_metrics.histogram.assert_called_once()
            call_args = mock_metrics.histogram.call_args
            assert call_args.args[0] == "foundry_mcp_research_phase_duration_seconds"
            assert call_args.kwargs["labels"]["phase_name"] == "synthesis"
            assert call_args.kwargs["labels"]["status"] == "success"

    def test_works_for_all_phase_names(self, mock_workflow, sample_state):
        """Should work for any phase name string."""
        phase_start = time.perf_counter()

        for phase in ("planning", "analysis", "synthesis", "refinement"):
            mock_workflow._write_audit_event.reset_mock()
            finalize_phase(mock_workflow, sample_state, phase, phase_start)
            data = mock_workflow._write_audit_event.call_args.kwargs["data"]
            assert data["phase_name"] == phase


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case and regression tests."""

    @pytest.mark.asyncio
    async def test_zero_tokens_used_not_tracked(self, mock_workflow, sample_state):
        """Should not add to total_tokens_used when tokens_used is 0/None."""
        sample_state.total_tokens_used = 100
        mock_workflow._execute_provider_async.return_value = _make_success_result(tokens_used=0)

        await execute_llm_call(
            workflow=mock_workflow,
            state=sample_state,
            phase_name="planning",
            system_prompt="sys",
            user_prompt="usr",
            provider_id="p",
            model="m",
            temperature=0.7,
            timeout=60.0,
        )

        # 0 is falsy, so tokens should not be added
        assert sample_state.total_tokens_used == 100

    @pytest.mark.asyncio
    async def test_none_provider_id_handled(self, mock_workflow, sample_state):
        """Should handle None provider_id gracefully."""
        mock_workflow._execute_provider_async.return_value = _make_success_result()

        ret = await execute_llm_call(
            workflow=mock_workflow,
            state=sample_state,
            phase_name="planning",
            system_prompt="sys",
            user_prompt="usr",
            provider_id=None,
            model=None,
            temperature=0.7,
            timeout=60.0,
        )

        assert isinstance(ret, LLMCallResult)

    @pytest.mark.asyncio
    async def test_no_error_metadata_when_none(self, mock_workflow, sample_state):
        """Should not include error_metadata keys when not provided."""
        mock_workflow._execute_provider_async.side_effect = ContextWindowError(
            "Context window exceeded",
            prompt_tokens=5000,
            max_tokens=4096,
        )

        ret = await execute_llm_call(
            workflow=mock_workflow,
            state=sample_state,
            phase_name="planning",
            system_prompt="sys",
            user_prompt="usr",
            provider_id="p",
            model="m",
            temperature=0.7,
            timeout=60.0,
            error_metadata=None,
        )

        assert isinstance(ret, WorkflowResult)
        # Should have base metadata but no extra keys
        assert "finding_count" not in ret.metadata
        assert "research_id" in ret.metadata
