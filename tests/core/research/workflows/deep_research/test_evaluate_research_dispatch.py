"""Tests for _evaluate_research async dispatch (Phase 1a deadlock fix).

Verifies that _evaluate_research uses ThreadPoolExecutor dispatch
(same pattern as _run_sync) instead of run_coroutine_threadsafe,
which avoids deadlock when called from the event loop thread.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from foundry_mcp.core.research.models.deep_research import (
    DeepResearchPhase,
    DeepResearchState,
)
from foundry_mcp.core.research.workflows.base import WorkflowResult
from foundry_mcp.core.research.workflows.deep_research.action_handlers import (
    ActionHandlersMixin,
)


class StubActionHandler(ActionHandlersMixin):
    """Minimal concrete class for testing ActionHandlersMixin."""

    def __init__(self) -> None:
        self.config = MagicMock()
        self.config.deep_research_evaluation_provider = None
        self.config.deep_research_evaluation_model = None
        self.config.deep_research_evaluation_timeout = 10.0
        self.memory = MagicMock()
        self._tasks: dict[str, Any] = {}
        self._tasks_lock = threading.Lock()

    def _write_audit_event(self, state: Any, event: str, **kwargs: Any) -> None:
        pass


def _make_completed_state() -> DeepResearchState:
    state = DeepResearchState(
        id="test-eval",
        original_query="test",
        phase=DeepResearchPhase.SYNTHESIS,
    )
    state.report = "A complete research report."
    return state


class TestEvaluateResearchAsyncDispatch:
    """Phase 1a: Verify _evaluate_research doesn't deadlock in async context."""

    def test_evaluate_from_sync_context_uses_asyncio_run(self):
        """When no event loop is running, asyncio.run() is used directly."""
        stub = StubActionHandler()
        state = _make_completed_state()
        stub.memory.load_deep_research.return_value = state

        mock_result = MagicMock()
        mock_result.composite_score = 0.85
        mock_result.dimension_scores = []
        mock_result.to_dict.return_value = {}

        with patch(
            "foundry_mcp.core.research.evaluation.evaluator.evaluate_report",
            return_value=mock_result,
        ) as mock_eval:
            # This is a sync call with no event loop — should use asyncio.run()
            result = stub._evaluate_research("test-eval")

        assert result.success is True
        assert mock_eval.called

    def test_evaluate_from_async_context_completes_without_deadlock(self):
        """When called from within an async context, uses ThreadPoolExecutor
        to avoid deadlocking on run_coroutine_threadsafe."""
        stub = StubActionHandler()
        state = _make_completed_state()
        stub.memory.load_deep_research.return_value = state

        mock_result = MagicMock()
        mock_result.composite_score = 0.85
        mock_result.dimension_scores = []
        mock_result.to_dict.return_value = {}

        async def fake_evaluate(**kwargs: Any) -> Any:
            return mock_result

        with patch(
            "foundry_mcp.core.research.evaluation.evaluator.evaluate_report",
            side_effect=fake_evaluate,
        ):
            # Run _evaluate_research from within an async context
            # (simulates MCP handler calling it).
            # With the old run_coroutine_threadsafe pattern, this would deadlock.
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(
                    asyncio.wait_for(
                        asyncio.to_thread(stub._evaluate_research, "test-eval"),
                        timeout=10.0,
                    )
                )
            finally:
                loop.close()

        assert result.success is True
        assert "0.850" in result.content
