"""Unit tests for research workflow Pydantic models.

Tests validation, serialization/deserialization, and enum behavior
for all models defined in foundry_mcp.core.research.models.
"""

import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from foundry_mcp.core.research.models import (
    ConfidenceLevel,
    ConsensusConfig,
    ConsensusState,
    ConsensusStrategy,
    ConversationMessage,
    ConversationThread,
    Hypothesis,
    Idea,
    IdeaCluster,
    IdeationPhase,
    IdeationState,
    InvestigationStep,
    ModelResponse,
    ThinkDeepState,
    ThreadStatus,
    WorkflowType,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestWorkflowTypeEnum:
    """Tests for WorkflowType enum."""

    def test_workflow_type_values(self):
        """All workflow types should have expected values."""
        assert WorkflowType.CHAT.value == "chat"
        assert WorkflowType.CONSENSUS.value == "consensus"
        assert WorkflowType.THINKDEEP.value == "thinkdeep"
        assert WorkflowType.IDEATE.value == "ideate"

    def test_workflow_type_from_string(self):
        """Should be able to create enum from string value."""
        assert WorkflowType("chat") == WorkflowType.CHAT
        assert WorkflowType("consensus") == WorkflowType.CONSENSUS

    def test_workflow_type_invalid_value(self):
        """Invalid value should raise ValueError."""
        with pytest.raises(ValueError):
            WorkflowType("invalid")


class TestConfidenceLevelEnum:
    """Tests for ConfidenceLevel enum."""

    def test_confidence_level_values(self):
        """All confidence levels should have expected values."""
        assert ConfidenceLevel.SPECULATION.value == "speculation"
        assert ConfidenceLevel.LOW.value == "low"
        assert ConfidenceLevel.MEDIUM.value == "medium"
        assert ConfidenceLevel.HIGH.value == "high"
        assert ConfidenceLevel.CONFIRMED.value == "confirmed"

    def test_confidence_level_ordering(self):
        """Confidence levels should be in logical order."""
        levels = list(ConfidenceLevel)
        assert levels[0] == ConfidenceLevel.SPECULATION
        assert levels[-1] == ConfidenceLevel.CONFIRMED


class TestConsensusStrategyEnum:
    """Tests for ConsensusStrategy enum."""

    def test_strategy_values(self):
        """All strategies should have expected values."""
        assert ConsensusStrategy.ALL_RESPONSES.value == "all_responses"
        assert ConsensusStrategy.SYNTHESIZE.value == "synthesize"
        assert ConsensusStrategy.MAJORITY.value == "majority"
        assert ConsensusStrategy.FIRST_VALID.value == "first_valid"


class TestThreadStatusEnum:
    """Tests for ThreadStatus enum."""

    def test_status_values(self):
        """All statuses should have expected values."""
        assert ThreadStatus.ACTIVE.value == "active"
        assert ThreadStatus.COMPLETED.value == "completed"
        assert ThreadStatus.ARCHIVED.value == "archived"


class TestIdeationPhaseEnum:
    """Tests for IdeationPhase enum."""

    def test_phase_values(self):
        """All phases should have expected values."""
        assert IdeationPhase.DIVERGENT.value == "divergent"
        assert IdeationPhase.CONVERGENT.value == "convergent"
        assert IdeationPhase.SELECTION.value == "selection"
        assert IdeationPhase.ELABORATION.value == "elaboration"


# =============================================================================
# Conversation Models Tests
# =============================================================================


class TestConversationMessage:
    """Tests for ConversationMessage model."""

    def test_create_minimal_message(self):
        """Should create message with minimal required fields."""
        msg = ConversationMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.id.startswith("msg-")
        assert msg.timestamp is not None

    def test_create_full_message(self):
        """Should create message with all fields."""
        msg = ConversationMessage(
            role="assistant",
            content="Response",
            provider_id="openai",
            model_used="gpt-4",
            tokens_used=100,
            metadata={"key": "value"},
        )
        assert msg.provider_id == "openai"
        assert msg.model_used == "gpt-4"
        assert msg.tokens_used == 100
        assert msg.metadata["key"] == "value"

    def test_message_serialization(self):
        """Should serialize and deserialize correctly."""
        msg = ConversationMessage(role="user", content="Test")
        data = msg.model_dump(mode="json")
        restored = ConversationMessage.model_validate(data)
        assert restored.role == msg.role
        assert restored.content == msg.content

    def test_invalid_role_type(self):
        """Should reject invalid role type."""
        with pytest.raises(ValidationError):
            ConversationMessage(role=123, content="Test")


class TestConversationThread:
    """Tests for ConversationThread model."""

    def test_create_thread(self):
        """Should create thread with defaults."""
        thread = ConversationThread()
        assert thread.id.startswith("thread-")
        assert thread.status == ThreadStatus.ACTIVE
        assert len(thread.messages) == 0

    def test_add_message(self):
        """Should add messages correctly."""
        thread = ConversationThread()
        msg = thread.add_message(role="user", content="Hello")
        assert len(thread.messages) == 1
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_add_message_with_metadata(self):
        """Should add messages with metadata."""
        thread = ConversationThread()
        msg = thread.add_message(
            role="assistant",
            content="Response",
            provider_id="openai",
            model_used="gpt-4",
            tokens_used=50,
            custom_key="custom_value",
        )
        assert msg.provider_id == "openai"
        assert msg.metadata["custom_key"] == "custom_value"

    def test_get_context_messages(self):
        """Should return context messages with limit."""
        thread = ConversationThread()
        for i in range(10):
            thread.add_message(role="user", content=f"Message {i}")

        all_msgs = thread.get_context_messages()
        assert len(all_msgs) == 10

        limited = thread.get_context_messages(max_messages=3)
        assert len(limited) == 3
        assert limited[0].content == "Message 7"

    def test_thread_serialization(self):
        """Should serialize and deserialize correctly."""
        thread = ConversationThread(title="Test Thread")
        thread.add_message(role="user", content="Hello")

        data = thread.model_dump(mode="json")
        restored = ConversationThread.model_validate(data)

        assert restored.title == thread.title
        assert len(restored.messages) == 1


# =============================================================================
# ThinkDeep Models Tests
# =============================================================================


class TestHypothesis:
    """Tests for Hypothesis model."""

    def test_create_hypothesis(self):
        """Should create hypothesis with defaults."""
        hyp = Hypothesis(statement="Test hypothesis")
        assert hyp.statement == "Test hypothesis"
        assert hyp.confidence == ConfidenceLevel.SPECULATION
        assert len(hyp.supporting_evidence) == 0
        assert len(hyp.contradicting_evidence) == 0

    def test_add_supporting_evidence(self):
        """Should add supporting evidence."""
        hyp = Hypothesis(statement="Test")
        hyp.add_evidence("Evidence 1", supporting=True)
        hyp.add_evidence("Evidence 2", supporting=True)

        assert len(hyp.supporting_evidence) == 2
        assert "Evidence 1" in hyp.supporting_evidence

    def test_add_contradicting_evidence(self):
        """Should add contradicting evidence."""
        hyp = Hypothesis(statement="Test")
        hyp.add_evidence("Counter 1", supporting=False)

        assert len(hyp.contradicting_evidence) == 1
        assert "Counter 1" in hyp.contradicting_evidence

    def test_update_confidence(self):
        """Should update confidence level."""
        hyp = Hypothesis(statement="Test")
        assert hyp.confidence == ConfidenceLevel.SPECULATION

        hyp.update_confidence(ConfidenceLevel.HIGH)
        assert hyp.confidence == ConfidenceLevel.HIGH

    def test_hypothesis_serialization(self):
        """Should serialize and deserialize correctly."""
        hyp = Hypothesis(statement="Test", confidence=ConfidenceLevel.MEDIUM)
        hyp.add_evidence("Evidence", supporting=True)

        data = hyp.model_dump(mode="json")
        restored = Hypothesis.model_validate(data)

        assert restored.statement == hyp.statement
        assert restored.confidence == hyp.confidence
        assert len(restored.supporting_evidence) == 1


class TestInvestigationStep:
    """Tests for InvestigationStep model."""

    def test_create_step(self):
        """Should create step with required fields."""
        step = InvestigationStep(depth=0, query="Initial query")
        assert step.depth == 0
        assert step.query == "Initial query"
        assert step.response is None
        assert step.id.startswith("step-")

    def test_step_with_response(self):
        """Should store response and provider info."""
        step = InvestigationStep(
            depth=1,
            query="Follow up",
            response="Provider response",
            provider_id="anthropic",
            model_used="claude-3",
        )
        assert step.response == "Provider response"
        assert step.provider_id == "anthropic"


class TestThinkDeepState:
    """Tests for ThinkDeepState model."""

    def test_create_state(self):
        """Should create state with defaults."""
        state = ThinkDeepState(topic="Test topic")
        assert state.topic == "Test topic"
        assert state.current_depth == 0
        assert state.max_depth == 5
        assert state.converged is False
        assert len(state.hypotheses) == 0
        assert len(state.steps) == 0

    def test_add_hypothesis(self):
        """Should add hypotheses correctly."""
        state = ThinkDeepState(topic="Test")
        hyp = state.add_hypothesis(
            "Hypothesis 1", confidence=ConfidenceLevel.LOW
        )

        assert len(state.hypotheses) == 1
        assert hyp.statement == "Hypothesis 1"
        assert hyp.confidence == ConfidenceLevel.LOW

    def test_get_hypothesis(self):
        """Should get hypothesis by ID."""
        state = ThinkDeepState(topic="Test")
        hyp = state.add_hypothesis("Test hyp")

        found = state.get_hypothesis(hyp.id)
        assert found is not None
        assert found.statement == "Test hyp"

        not_found = state.get_hypothesis("nonexistent")
        assert not_found is None

    def test_add_step(self):
        """Should add steps correctly."""
        state = ThinkDeepState(topic="Test")
        step = state.add_step("Query 1", depth=0)

        assert len(state.steps) == 1
        assert step.query == "Query 1"
        assert step.depth == 0

    def test_check_convergence_max_depth(self):
        """Should converge when max depth reached."""
        state = ThinkDeepState(topic="Test", max_depth=3)
        state.current_depth = 3

        converged = state.check_convergence()

        assert converged is True
        assert state.converged is True
        assert "depth" in state.convergence_reason.lower()

    def test_check_convergence_high_confidence(self):
        """Should converge when all hypotheses high confidence."""
        state = ThinkDeepState(topic="Test", max_depth=10)
        state.add_hypothesis("H1", confidence=ConfidenceLevel.HIGH)
        state.add_hypothesis("H2", confidence=ConfidenceLevel.CONFIRMED)

        converged = state.check_convergence()

        assert converged is True
        assert "confidence" in state.convergence_reason.lower()

    def test_no_convergence_mixed_confidence(self):
        """Should not converge with mixed confidence."""
        state = ThinkDeepState(topic="Test", max_depth=10)
        state.add_hypothesis("H1", confidence=ConfidenceLevel.HIGH)
        state.add_hypothesis("H2", confidence=ConfidenceLevel.LOW)

        converged = state.check_convergence()

        assert converged is False
        assert state.converged is False

    def test_state_serialization(self):
        """Should serialize and deserialize correctly."""
        state = ThinkDeepState(topic="Test", max_depth=3)
        state.add_hypothesis("Hypothesis")
        state.add_step("Query")

        data = state.model_dump(mode="json")
        restored = ThinkDeepState.model_validate(data)

        assert restored.topic == state.topic
        assert restored.max_depth == state.max_depth
        assert len(restored.hypotheses) == 1
        assert len(restored.steps) == 1


# =============================================================================
# Ideation Models Tests
# =============================================================================


class TestIdea:
    """Tests for Idea model."""

    def test_create_idea(self):
        """Should create idea with required fields."""
        idea = Idea(content="New feature idea")
        assert idea.content == "New feature idea"
        assert idea.id.startswith("idea-")
        assert idea.score is None
        assert idea.cluster_id is None

    def test_idea_with_perspective(self):
        """Should store perspective."""
        idea = Idea(content="Technical solution", perspective="technical")
        assert idea.perspective == "technical"


class TestIdeaCluster:
    """Tests for IdeaCluster model."""

    def test_create_cluster(self):
        """Should create cluster with name."""
        cluster = IdeaCluster(name="Automation Ideas")
        assert cluster.name == "Automation Ideas"
        assert cluster.id.startswith("cluster-")
        assert len(cluster.idea_ids) == 0
        assert cluster.selected_for_elaboration is False


class TestIdeationState:
    """Tests for IdeationState model."""

    def test_create_state(self):
        """Should create state with defaults."""
        state = IdeationState(topic="New product")
        assert state.topic == "New product"
        assert state.phase == IdeationPhase.DIVERGENT
        assert len(state.perspectives) == 4  # default perspectives
        assert len(state.scoring_criteria) == 3  # default criteria

    def test_add_idea(self):
        """Should add ideas correctly."""
        state = IdeationState(topic="Test")
        idea = state.add_idea("Great idea", perspective="creative")

        assert len(state.ideas) == 1
        assert idea.content == "Great idea"
        assert idea.perspective == "creative"

    def test_create_cluster(self):
        """Should create clusters correctly."""
        state = IdeationState(topic="Test")
        cluster = state.create_cluster("Tech Solutions", "Technical approaches")

        assert len(state.clusters) == 1
        assert cluster.name == "Tech Solutions"
        assert cluster.description == "Technical approaches"

    def test_assign_idea_to_cluster(self):
        """Should assign ideas to clusters."""
        state = IdeationState(topic="Test")
        idea = state.add_idea("Test idea")
        cluster = state.create_cluster("Test cluster")

        result = state.assign_idea_to_cluster(idea.id, cluster.id)

        assert result is True
        assert idea.cluster_id == cluster.id
        assert idea.id in cluster.idea_ids

    def test_assign_idea_invalid_ids(self):
        """Should return False for invalid IDs."""
        state = IdeationState(topic="Test")
        idea = state.add_idea("Test idea")

        result = state.assign_idea_to_cluster(idea.id, "nonexistent")
        assert result is False

        result = state.assign_idea_to_cluster("nonexistent", "nonexistent")
        assert result is False

    def test_advance_phase(self):
        """Should advance through phases correctly."""
        state = IdeationState(topic="Test")

        assert state.phase == IdeationPhase.DIVERGENT

        state.advance_phase()
        assert state.phase == IdeationPhase.CONVERGENT

        state.advance_phase()
        assert state.phase == IdeationPhase.SELECTION

        state.advance_phase()
        assert state.phase == IdeationPhase.ELABORATION

        # Should not advance past last phase
        state.advance_phase()
        assert state.phase == IdeationPhase.ELABORATION

    def test_state_serialization(self):
        """Should serialize and deserialize correctly."""
        state = IdeationState(
            topic="Test",
            perspectives=["a", "b"],
            scoring_criteria=["x", "y"],
        )
        state.add_idea("Idea 1")
        state.create_cluster("Cluster 1")

        data = state.model_dump(mode="json")
        restored = IdeationState.model_validate(data)

        assert restored.topic == state.topic
        assert restored.perspectives == ["a", "b"]
        assert len(restored.ideas) == 1
        assert len(restored.clusters) == 1


# =============================================================================
# Consensus Models Tests
# =============================================================================


class TestModelResponse:
    """Tests for ModelResponse model."""

    def test_create_response(self):
        """Should create response with required fields."""
        resp = ModelResponse(provider_id="openai", content="Response text")
        assert resp.provider_id == "openai"
        assert resp.content == "Response text"
        assert resp.success is True

    def test_failed_response(self):
        """Should store failure info."""
        resp = ModelResponse(
            provider_id="openai",
            content="",
            success=False,
            error_message="Rate limited",
        )
        assert resp.success is False
        assert resp.error_message == "Rate limited"


class TestConsensusConfig:
    """Tests for ConsensusConfig model."""

    def test_create_config(self):
        """Should create config with providers."""
        config = ConsensusConfig(providers=["openai", "anthropic"])
        assert len(config.providers) == 2
        assert config.strategy == ConsensusStrategy.SYNTHESIZE

    def test_config_validation_min_providers(self):
        """Should require at least one provider."""
        with pytest.raises(ValidationError):
            ConsensusConfig(providers=[])

    def test_config_with_options(self):
        """Should accept all options."""
        config = ConsensusConfig(
            providers=["openai"],
            strategy=ConsensusStrategy.MAJORITY,
            synthesis_provider="anthropic",
            timeout_per_provider=60.0,
            max_concurrent=5,
            require_all=True,
            min_responses=2,
        )
        assert config.strategy == ConsensusStrategy.MAJORITY
        assert config.timeout_per_provider == 60.0
        assert config.require_all is True


class TestConsensusState:
    """Tests for ConsensusState model."""

    def test_create_state(self):
        """Should create state with required fields."""
        config = ConsensusConfig(providers=["openai"])
        state = ConsensusState(prompt="Test prompt", config=config)

        assert state.prompt == "Test prompt"
        assert state.completed is False
        assert len(state.responses) == 0

    def test_add_response(self):
        """Should add responses."""
        config = ConsensusConfig(providers=["openai"])
        state = ConsensusState(prompt="Test", config=config)

        resp = ModelResponse(provider_id="openai", content="Response")
        state.add_response(resp)

        assert len(state.responses) == 1

    def test_successful_responses(self):
        """Should filter successful responses."""
        config = ConsensusConfig(providers=["a", "b"])
        state = ConsensusState(prompt="Test", config=config)

        state.add_response(ModelResponse(provider_id="a", content="OK", success=True))
        state.add_response(ModelResponse(provider_id="b", content="", success=False))

        successful = state.successful_responses()
        failed = state.failed_responses()

        assert len(successful) == 1
        assert len(failed) == 1

    def test_is_quorum_met(self):
        """Should check quorum correctly."""
        config = ConsensusConfig(providers=["a", "b"], min_responses=2)
        state = ConsensusState(prompt="Test", config=config)

        assert state.is_quorum_met() is False

        state.add_response(ModelResponse(provider_id="a", content="OK"))
        assert state.is_quorum_met() is False

        state.add_response(ModelResponse(provider_id="b", content="OK"))
        assert state.is_quorum_met() is True

    def test_mark_completed(self):
        """Should mark as completed with synthesis."""
        config = ConsensusConfig(providers=["a"])
        state = ConsensusState(prompt="Test", config=config)

        state.mark_completed(synthesis="Combined response")

        assert state.completed is True
        assert state.completed_at is not None
        assert state.synthesis == "Combined response"

    def test_state_serialization(self):
        """Should serialize and deserialize correctly."""
        config = ConsensusConfig(providers=["openai"])
        state = ConsensusState(prompt="Test", config=config)
        state.add_response(ModelResponse(provider_id="openai", content="Response"))

        data = state.model_dump(mode="json")
        restored = ConsensusState.model_validate(data)

        assert restored.prompt == state.prompt
        assert len(restored.responses) == 1
