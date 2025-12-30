"""Integration tests for the unified research router.

Tests dispatch logic, action handlers, error conditions, and response envelopes
for all research tool actions: chat, consensus, thinkdeep, ideate, route,
thread-list, thread-get, thread-delete.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from foundry_mcp.core.research.models import (
    ConversationThread,
    ThreadStatus,
)
from foundry_mcp.core.research.workflows.base import WorkflowResult


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockWorkflowResult:
    """Mock WorkflowResult for testing."""

    success: bool
    content: str
    provider_id: Optional[str] = None
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None
    duration_ms: Optional[float] = None
    metadata: Optional[dict[str, Any]] = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


@pytest.fixture
def mock_feature_flag():
    """Mock feature flag service to enable research tools."""
    with patch(
        "foundry_mcp.tools.unified.research.get_flag_service"
    ) as mock_get_flag:
        mock_service = MagicMock()
        mock_service.is_enabled.return_value = True
        mock_get_flag.return_value = mock_service
        yield mock_service


@pytest.fixture
def mock_feature_flag_disabled():
    """Mock feature flag service with research tools disabled."""
    with patch(
        "foundry_mcp.tools.unified.research.get_flag_service"
    ) as mock_get_flag:
        mock_service = MagicMock()
        mock_service.is_enabled.return_value = False
        mock_get_flag.return_value = mock_service
        yield mock_service


@pytest.fixture
def mock_config(tmp_path: Path):
    """Mock server config for testing."""
    with patch("foundry_mcp.tools.unified.research._get_config") as mock_get_config:
        mock_cfg = MagicMock()
        mock_cfg.research.enabled = True
        mock_cfg.research.get_storage_path.return_value = tmp_path
        mock_cfg.research.ttl_hours = 24
        mock_get_config.return_value = mock_cfg
        yield mock_cfg


@pytest.fixture
def mock_memory():
    """Mock research memory instance."""
    with patch("foundry_mcp.tools.unified.research._get_memory") as mock_get_memory:
        memory = MagicMock()
        mock_get_memory.return_value = memory
        yield memory


# =============================================================================
# Dispatch Tests
# =============================================================================


class TestResearchDispatch:
    """Tests for action dispatch logic."""

    def test_dispatch_to_chat(self, mock_feature_flag, mock_config, mock_memory):
        """Should dispatch 'chat' action and call chat workflow."""
        from foundry_mcp.tools.unified.research import _dispatch_research_action

        with patch(
            "foundry_mcp.tools.unified.research.ChatWorkflow"
        ) as MockWorkflow:
            mock_workflow = MagicMock()
            mock_workflow.execute.return_value = WorkflowResult(
                success=True,
                content="Response",
                metadata={"thread_id": "t-1", "message_count": 1},
            )
            MockWorkflow.return_value = mock_workflow

            result = _dispatch_research_action(action="chat", prompt="Hello")

            MockWorkflow.assert_called_once()
            mock_workflow.execute.assert_called_once()
            assert result["success"] is True

    def test_dispatch_to_consensus(self, mock_feature_flag, mock_config, mock_memory):
        """Should dispatch 'consensus' action and call consensus workflow."""
        from foundry_mcp.tools.unified.research import _dispatch_research_action

        with patch(
            "foundry_mcp.tools.unified.research.ConsensusWorkflow"
        ) as MockWorkflow:
            mock_workflow = MagicMock()
            mock_workflow.execute.return_value = WorkflowResult(
                success=True,
                content="Consensus",
                metadata={
                    "consensus_id": "c-1",
                    "providers_consulted": ["openai"],
                    "strategy": "synthesize",
                    "response_count": 1,
                },
            )
            MockWorkflow.return_value = mock_workflow

            result = _dispatch_research_action(action="consensus", prompt="Test")

            MockWorkflow.assert_called_once()
            assert result["success"] is True

    def test_dispatch_to_thinkdeep(self, mock_feature_flag, mock_config, mock_memory):
        """Should dispatch 'thinkdeep' action and call thinkdeep workflow."""
        from foundry_mcp.tools.unified.research import _dispatch_research_action

        with patch(
            "foundry_mcp.tools.unified.research.ThinkDeepWorkflow"
        ) as MockWorkflow:
            mock_workflow = MagicMock()
            mock_workflow.execute.return_value = WorkflowResult(
                success=True,
                content="Findings",
                metadata={
                    "investigation_id": "inv-1",
                    "current_depth": 1,
                    "max_depth": 5,
                    "converged": False,
                    "hypothesis_count": 1,
                    "step_count": 1,
                },
            )
            MockWorkflow.return_value = mock_workflow

            result = _dispatch_research_action(action="thinkdeep", topic="Test topic")

            MockWorkflow.assert_called_once()
            assert result["success"] is True

    def test_dispatch_to_ideate(self, mock_feature_flag, mock_config, mock_memory):
        """Should dispatch 'ideate' action and call ideate workflow."""
        from foundry_mcp.tools.unified.research import _dispatch_research_action

        with patch(
            "foundry_mcp.tools.unified.research.IdeateWorkflow"
        ) as MockWorkflow:
            mock_workflow = MagicMock()
            mock_workflow.execute.return_value = WorkflowResult(
                success=True,
                content="Ideas",
                metadata={
                    "ideation_id": "ide-1",
                    "phase": "divergent",
                    "idea_count": 5,
                    "cluster_count": 0,
                },
            )
            MockWorkflow.return_value = mock_workflow

            result = _dispatch_research_action(action="ideate", topic="Ideas")

            MockWorkflow.assert_called_once()
            assert result["success"] is True

    def test_dispatch_invalid_action(self, mock_feature_flag, mock_config, mock_memory):
        """Should return error for invalid action."""
        from foundry_mcp.tools.unified.research import _dispatch_research_action

        result = _dispatch_research_action(action="invalid_action")

        assert result["success"] is False
        assert "invalid_action" in result["error"].lower()
        assert "data" in result
        assert result["data"]["error_code"] == "VALIDATION_ERROR"

# =============================================================================
# Chat Handler Tests
# =============================================================================


class TestChatHandler:
    """Tests for chat action handler."""

    def test_chat_requires_prompt(self, mock_feature_flag, mock_config, mock_memory):
        """Should return validation error when prompt is missing."""
        from foundry_mcp.tools.unified.research import _handle_chat

        result = _handle_chat()

        assert result["success"] is False
        assert "prompt" in result["error"].lower()
        assert result["data"]["error_type"] == "validation"

    def test_chat_success(self, mock_feature_flag, mock_config, mock_memory):
        """Should return success response from chat workflow."""
        from foundry_mcp.tools.unified.research import _handle_chat

        with patch(
            "foundry_mcp.tools.unified.research.ChatWorkflow"
        ) as MockWorkflow:
            mock_workflow = MagicMock()
            mock_workflow.execute.return_value = WorkflowResult(
                success=True,
                content="Hello! How can I help?",
                provider_id="openai",
                model_used="gpt-4",
                tokens_used=50,
                metadata={"thread_id": "thread-123", "message_count": 2},
            )
            MockWorkflow.return_value = mock_workflow

            result = _handle_chat(prompt="Hello")

            assert result["success"] is True
            assert result["data"]["content"] == "Hello! How can I help?"
            assert result["data"]["thread_id"] == "thread-123"
            assert result["data"]["provider_id"] == "openai"
            assert result["meta"]["version"] == "response-v2"

    def test_chat_failure(self, mock_feature_flag, mock_config, mock_memory):
        """Should return error response on chat workflow failure."""
        from foundry_mcp.tools.unified.research import _handle_chat

        with patch(
            "foundry_mcp.tools.unified.research.ChatWorkflow"
        ) as MockWorkflow:
            mock_workflow = MagicMock()
            mock_workflow.execute.return_value = WorkflowResult(
                success=False,
                content="",
                error="Provider unavailable",
            )
            MockWorkflow.return_value = mock_workflow

            result = _handle_chat(prompt="Hello")

            assert result["success"] is False
            assert "unavailable" in result["error"].lower()


# =============================================================================
# Consensus Handler Tests
# =============================================================================


class TestConsensusHandler:
    """Tests for consensus action handler."""

    def test_consensus_requires_prompt(
        self, mock_feature_flag, mock_config, mock_memory
    ):
        """Should return validation error when prompt is missing."""
        from foundry_mcp.tools.unified.research import _handle_consensus

        result = _handle_consensus()

        assert result["success"] is False
        assert "prompt" in result["error"].lower()

    def test_consensus_invalid_strategy(
        self, mock_feature_flag, mock_config, mock_memory
    ):
        """Should return validation error for invalid strategy."""
        from foundry_mcp.tools.unified.research import _handle_consensus

        result = _handle_consensus(prompt="Test", strategy="invalid_strategy")

        assert result["success"] is False
        assert "strategy" in result["error"].lower()
        assert "invalid" in result["error"].lower()

    def test_consensus_success(self, mock_feature_flag, mock_config, mock_memory):
        """Should return success response from consensus workflow."""
        from foundry_mcp.tools.unified.research import _handle_consensus

        with patch(
            "foundry_mcp.tools.unified.research.ConsensusWorkflow"
        ) as MockWorkflow:
            mock_workflow = MagicMock()
            mock_workflow.execute.return_value = WorkflowResult(
                success=True,
                content="Synthesized response from multiple models",
                metadata={
                    "consensus_id": "cons-123",
                    "providers_consulted": ["openai", "anthropic"],
                    "strategy": "synthesize",
                    "response_count": 2,
                },
            )
            MockWorkflow.return_value = mock_workflow

            result = _handle_consensus(prompt="Compare perspectives")

            assert result["success"] is True
            assert "Synthesized" in result["data"]["content"]
            assert result["data"]["consensus_id"] == "cons-123"
            assert len(result["data"]["providers_consulted"]) == 2


# =============================================================================
# ThinkDeep Handler Tests
# =============================================================================


class TestThinkDeepHandler:
    """Tests for thinkdeep action handler."""

    def test_thinkdeep_requires_topic_or_id(
        self, mock_feature_flag, mock_config, mock_memory
    ):
        """Should return validation error when neither topic nor ID provided."""
        from foundry_mcp.tools.unified.research import _handle_thinkdeep

        result = _handle_thinkdeep()

        assert result["success"] is False
        assert "topic" in result["error"].lower() or "investigation_id" in result["error"].lower()

    def test_thinkdeep_with_topic(self, mock_feature_flag, mock_config, mock_memory):
        """Should start new investigation with topic."""
        from foundry_mcp.tools.unified.research import _handle_thinkdeep

        with patch(
            "foundry_mcp.tools.unified.research.ThinkDeepWorkflow"
        ) as MockWorkflow:
            mock_workflow = MagicMock()
            mock_workflow.execute.return_value = WorkflowResult(
                success=True,
                content="Investigation findings...",
                metadata={
                    "investigation_id": "inv-123",
                    "current_depth": 1,
                    "max_depth": 5,
                    "converged": False,
                    "hypothesis_count": 2,
                    "step_count": 1,
                },
            )
            MockWorkflow.return_value = mock_workflow

            result = _handle_thinkdeep(topic="Why does X happen?")

            assert result["success"] is True
            assert result["data"]["investigation_id"] == "inv-123"
            assert result["data"]["converged"] is False

    def test_thinkdeep_with_investigation_id(
        self, mock_feature_flag, mock_config, mock_memory
    ):
        """Should continue existing investigation with ID."""
        from foundry_mcp.tools.unified.research import _handle_thinkdeep

        with patch(
            "foundry_mcp.tools.unified.research.ThinkDeepWorkflow"
        ) as MockWorkflow:
            mock_workflow = MagicMock()
            mock_workflow.execute.return_value = WorkflowResult(
                success=True,
                content="Continued findings...",
                metadata={
                    "investigation_id": "inv-123",
                    "current_depth": 3,
                    "max_depth": 5,
                    "converged": True,
                    "hypothesis_count": 4,
                    "step_count": 3,
                },
            )
            MockWorkflow.return_value = mock_workflow

            result = _handle_thinkdeep(investigation_id="inv-123", query="Why else?")

            assert result["success"] is True
            assert result["data"]["converged"] is True
            assert result["data"]["current_depth"] == 3


# =============================================================================
# Ideate Handler Tests
# =============================================================================


class TestIdeateHandler:
    """Tests for ideate action handler."""

    def test_ideate_requires_topic_or_id(
        self, mock_feature_flag, mock_config, mock_memory
    ):
        """Should return validation error when neither topic nor ID provided."""
        from foundry_mcp.tools.unified.research import _handle_ideate

        result = _handle_ideate()

        assert result["success"] is False
        assert "topic" in result["error"].lower() or "ideation_id" in result["error"].lower()

    def test_ideate_with_topic(self, mock_feature_flag, mock_config, mock_memory):
        """Should start new ideation with topic."""
        from foundry_mcp.tools.unified.research import _handle_ideate

        with patch(
            "foundry_mcp.tools.unified.research.IdeateWorkflow"
        ) as MockWorkflow:
            mock_workflow = MagicMock()
            mock_workflow.execute.return_value = WorkflowResult(
                success=True,
                content="Generated ideas...",
                metadata={
                    "ideation_id": "ide-123",
                    "phase": "divergent",
                    "idea_count": 10,
                    "cluster_count": 0,
                },
            )
            MockWorkflow.return_value = mock_workflow

            result = _handle_ideate(topic="New feature ideas")

            assert result["success"] is True
            assert result["data"]["ideation_id"] == "ide-123"
            assert result["data"]["phase"] == "divergent"
            assert result["data"]["idea_count"] == 10


# =============================================================================
# Thread Management Handler Tests
# =============================================================================


class TestThreadListHandler:
    """Tests for thread-list action handler."""

    def test_thread_list_returns_threads(
        self, mock_feature_flag, mock_config, mock_memory
    ):
        """Should return list of threads."""
        from foundry_mcp.tools.unified.research import _handle_thread_list

        with patch(
            "foundry_mcp.tools.unified.research.ChatWorkflow"
        ) as MockWorkflow:
            mock_workflow = MagicMock()
            mock_workflow.list_threads.return_value = [
                {"id": "thread-1", "title": "Thread 1", "status": "active"},
                {"id": "thread-2", "title": "Thread 2", "status": "completed"},
            ]
            MockWorkflow.return_value = mock_workflow

            result = _handle_thread_list()

            assert result["success"] is True
            assert result["data"]["count"] == 2
            assert len(result["data"]["threads"]) == 2

    def test_thread_list_with_status_filter(
        self, mock_feature_flag, mock_config, mock_memory
    ):
        """Should filter threads by status."""
        from foundry_mcp.tools.unified.research import _handle_thread_list

        with patch(
            "foundry_mcp.tools.unified.research.ChatWorkflow"
        ) as MockWorkflow:
            mock_workflow = MagicMock()
            mock_workflow.list_threads.return_value = [
                {"id": "thread-1", "title": "Thread 1", "status": "active"},
            ]
            MockWorkflow.return_value = mock_workflow

            result = _handle_thread_list(status="active")

            assert result["success"] is True
            mock_workflow.list_threads.assert_called_once()
            call_kwargs = mock_workflow.list_threads.call_args.kwargs
            assert call_kwargs["status"] == ThreadStatus.ACTIVE

    def test_thread_list_invalid_status(
        self, mock_feature_flag, mock_config, mock_memory
    ):
        """Should return validation error for invalid status."""
        from foundry_mcp.tools.unified.research import _handle_thread_list

        result = _handle_thread_list(status="invalid_status")

        assert result["success"] is False
        assert "status" in result["error"].lower()


class TestThreadGetHandler:
    """Tests for thread-get action handler."""

    def test_thread_get_requires_id(self, mock_feature_flag, mock_config, mock_memory):
        """Should return validation error when thread_id is missing."""
        from foundry_mcp.tools.unified.research import _handle_thread_get

        result = _handle_thread_get()

        assert result["success"] is False
        assert "thread_id" in result["error"].lower()

    def test_thread_get_found(self, mock_feature_flag, mock_config, mock_memory):
        """Should return thread details when found."""
        from foundry_mcp.tools.unified.research import _handle_thread_get

        with patch(
            "foundry_mcp.tools.unified.research.ChatWorkflow"
        ) as MockWorkflow:
            mock_workflow = MagicMock()
            mock_workflow.get_thread.return_value = {
                "id": "thread-123",
                "title": "Test Thread",
                "messages": [{"role": "user", "content": "Hello"}],
            }
            MockWorkflow.return_value = mock_workflow

            result = _handle_thread_get(thread_id="thread-123")

            assert result["success"] is True
            assert result["data"]["id"] == "thread-123"

    def test_thread_get_not_found(self, mock_feature_flag, mock_config, mock_memory):
        """Should return not found error when thread doesn't exist."""
        from foundry_mcp.tools.unified.research import _handle_thread_get

        with patch(
            "foundry_mcp.tools.unified.research.ChatWorkflow"
        ) as MockWorkflow:
            mock_workflow = MagicMock()
            mock_workflow.get_thread.return_value = None
            MockWorkflow.return_value = mock_workflow

            result = _handle_thread_get(thread_id="nonexistent")

            assert result["success"] is False
            assert result["data"]["error_code"] == "NOT_FOUND"


class TestThreadDeleteHandler:
    """Tests for thread-delete action handler."""

    def test_thread_delete_requires_id(
        self, mock_feature_flag, mock_config, mock_memory
    ):
        """Should return validation error when thread_id is missing."""
        from foundry_mcp.tools.unified.research import _handle_thread_delete

        result = _handle_thread_delete()

        assert result["success"] is False
        assert "thread_id" in result["error"].lower()

    def test_thread_delete_success(self, mock_feature_flag, mock_config, mock_memory):
        """Should return success when thread deleted."""
        from foundry_mcp.tools.unified.research import _handle_thread_delete

        with patch(
            "foundry_mcp.tools.unified.research.ChatWorkflow"
        ) as MockWorkflow:
            mock_workflow = MagicMock()
            mock_workflow.delete_thread.return_value = True
            MockWorkflow.return_value = mock_workflow

            result = _handle_thread_delete(thread_id="thread-123")

            assert result["success"] is True
            assert result["data"]["deleted"] is True
            assert result["data"]["thread_id"] == "thread-123"

    def test_thread_delete_not_found(self, mock_feature_flag, mock_config, mock_memory):
        """Should return not found error when thread doesn't exist."""
        from foundry_mcp.tools.unified.research import _handle_thread_delete

        with patch(
            "foundry_mcp.tools.unified.research.ChatWorkflow"
        ) as MockWorkflow:
            mock_workflow = MagicMock()
            mock_workflow.delete_thread.return_value = False
            MockWorkflow.return_value = mock_workflow

            result = _handle_thread_delete(thread_id="nonexistent")

            assert result["success"] is False
            assert result["data"]["error_code"] == "NOT_FOUND"


# =============================================================================
# Response Envelope Tests
# =============================================================================


class TestResponseEnvelope:
    """Tests for response envelope structure (meta.version=response-v2)."""

    def test_success_response_has_version(
        self, mock_feature_flag, mock_config, mock_memory
    ):
        """Success responses should have meta.version=response-v2."""
        from foundry_mcp.tools.unified.research import _handle_thread_list

        result = _handle_thread_list()  # Simplest handler

        assert result["success"] is True
        assert "meta" in result
        assert result["meta"]["version"] == "response-v2"

    def test_error_response_has_version(
        self, mock_feature_flag, mock_config, mock_memory
    ):
        """Error responses should have meta.version=response-v2."""
        from foundry_mcp.tools.unified.research import _handle_chat

        result = _handle_chat()  # Missing prompt

        assert result["success"] is False
        assert "meta" in result
        assert result["meta"]["version"] == "response-v2"

    def test_error_response_has_error_code(
        self, mock_feature_flag, mock_config, mock_memory
    ):
        """Error responses should include error_code in data."""
        from foundry_mcp.tools.unified.research import _handle_chat

        result = _handle_chat()  # Missing prompt

        assert result["success"] is False
        assert "data" in result
        assert "error_code" in result["data"]
        assert result["data"]["error_code"] == "VALIDATION_ERROR"

    def test_error_response_has_error_type(
        self, mock_feature_flag, mock_config, mock_memory
    ):
        """Error responses should include error_type in data."""
        from foundry_mcp.tools.unified.research import _handle_chat

        result = _handle_chat()  # Missing prompt

        assert result["success"] is False
        assert "data" in result
        assert "error_type" in result["data"]
        assert result["data"]["error_type"] == "validation"

    def test_error_response_has_remediation(
        self, mock_feature_flag, mock_config, mock_memory
    ):
        """Error responses should include remediation guidance."""
        from foundry_mcp.tools.unified.research import _handle_chat

        result = _handle_chat()  # Missing prompt

        assert result["success"] is False
        assert "data" in result
        assert "remediation" in result["data"]


# =============================================================================
# Feature Flag Tests
# =============================================================================


class TestFeatureFlag:
    """Tests for feature flag handling."""

    def test_feature_flag_error_response_format(
        self, mock_config, mock_memory
    ):
        """Feature flag error response should follow response-v2 format."""
        from dataclasses import asdict

        from foundry_mcp.core.responses import ErrorCode, ErrorType, error_response

        # The research tool returns this error when feature flag is disabled
        response = error_response(
            "Research tools are not enabled",
            error_code=ErrorCode.FEATURE_DISABLED,
            error_type=ErrorType.UNAVAILABLE,
            remediation="Enable 'research_tools' feature flag in configuration",
        )
        result = asdict(response)

        assert result["success"] is False
        assert "not enabled" in result["error"]
        assert result["data"]["error_code"] == "FEATURE_DISABLED"
        assert result["data"]["error_type"] == "unavailable"
        assert result["meta"]["version"] == "response-v2"

    def test_dispatch_without_feature_flag_check(
        self, mock_feature_flag, mock_config, mock_memory
    ):
        """Dispatch should work when called directly (feature flag in wrapper)."""
        from foundry_mcp.tools.unified.research import _dispatch_research_action

        # _dispatch_research_action doesn't check feature flag
        # The feature flag is checked in the registered tool function wrapper
        result = _dispatch_research_action(action="thread-list")

        assert result["success"] is True


# =============================================================================
# Error Condition Tests
# =============================================================================


class TestErrorConditions:
    """Tests for error handling."""

    def test_workflow_exception_handled(
        self, mock_feature_flag, mock_config, mock_memory
    ):
        """Should handle exceptions from workflow gracefully."""
        from foundry_mcp.tools.unified.research import _handle_chat

        with patch(
            "foundry_mcp.tools.unified.research.ChatWorkflow"
        ) as MockWorkflow:
            mock_workflow = MagicMock()
            mock_workflow.execute.return_value = WorkflowResult(
                success=False,
                content="",
                error="Connection timeout",
            )
            MockWorkflow.return_value = mock_workflow

            result = _handle_chat(prompt="Hello")

            assert result["success"] is False
            assert "timeout" in result["error"].lower()

    def test_empty_prompt_rejected(self, mock_feature_flag, mock_config, mock_memory):
        """Should reject empty prompts."""
        from foundry_mcp.tools.unified.research import _handle_chat

        result = _handle_chat(prompt="")

        assert result["success"] is False
        assert "prompt" in result["error"].lower()

    def test_empty_topic_rejected(self, mock_feature_flag, mock_config, mock_memory):
        """Should reject empty topics for thinkdeep."""
        from foundry_mcp.tools.unified.research import _handle_thinkdeep

        result = _handle_thinkdeep(topic="")

        assert result["success"] is False
        # Empty string is falsy, so neither topic nor investigation_id provided
        assert "topic" in result["error"].lower() or "investigation_id" in result["error"].lower()


# =============================================================================
# ActionRouter Unit Tests
# =============================================================================


class TestActionRouter:
    """Tests for the ActionRouter class used by research tool."""

    def test_router_requires_actions(self):
        """Should raise error when no actions provided."""
        from foundry_mcp.tools.unified.router import ActionRouter

        with pytest.raises(ValueError, match="at least one action"):
            ActionRouter(tool_name="test", actions=[])

    def test_router_duplicate_action_rejected(self):
        """Should reject duplicate action names."""
        from foundry_mcp.tools.unified.router import (
            ActionDefinition,
            ActionRouter,
        )

        with pytest.raises(ValueError, match="Duplicate action"):
            ActionRouter(
                tool_name="test",
                actions=[
                    ActionDefinition(name="action", handler=lambda: {}),
                    ActionDefinition(name="action", handler=lambda: {}),
                ],
            )

    def test_router_allows_actions(self):
        """Should return list of allowed actions."""
        from foundry_mcp.tools.unified.router import (
            ActionDefinition,
            ActionRouter,
        )

        router = ActionRouter(
            tool_name="test",
            actions=[
                ActionDefinition(name="a", handler=lambda: {}),
                ActionDefinition(name="b", handler=lambda: {}),
            ],
        )

        allowed = router.allowed_actions()
        assert "a" in allowed
        assert "b" in allowed

    def test_router_dispatch_none_action(self):
        """Should raise error when action is None."""
        from foundry_mcp.tools.unified.router import (
            ActionDefinition,
            ActionRouter,
            ActionRouterError,
        )

        router = ActionRouter(
            tool_name="test",
            actions=[ActionDefinition(name="action", handler=lambda: {})],
        )

        with pytest.raises(ActionRouterError, match="requires an action"):
            router.dispatch(action=None)

    def test_router_describe(self):
        """Should return action summaries."""
        from foundry_mcp.tools.unified.router import (
            ActionDefinition,
            ActionRouter,
        )

        router = ActionRouter(
            tool_name="test",
            actions=[
                ActionDefinition(
                    name="action1", handler=lambda: {}, summary="First action"
                ),
                ActionDefinition(
                    name="action2", handler=lambda: {}, summary="Second action"
                ),
            ],
        )

        description = router.describe()
        assert description["action1"] == "First action"
        assert description["action2"] == "Second action"
