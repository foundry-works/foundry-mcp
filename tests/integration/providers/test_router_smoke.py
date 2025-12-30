"""
Router-level smoke tests with real providers.

Tests the full flow through ConsultationOrchestrator and Research Router
with actual provider calls to verify end-to-end integration.

Models used:
- Primary: gemini:gemini-2.5-flash
- Secondary (for consensus): codex:gpt-5.1-codex-mini

Run with: pytest tests/integration/providers/test_router_smoke.py -m router_smoke
Enable: FOUNDRY_ENABLE_LIVE_PROVIDER_TESTS=1

Note: These tests use longer timeouts (180-300s) since real provider calls
can be slow, especially for complex workflows like thinkdeep/consensus.
"""

# Default timeouts for live provider tests
DEFAULT_TIMEOUT = 180.0  # 3 minutes for simple requests
COMPLEX_TIMEOUT = 300.0  # 5 minutes for consensus/thinkdeep

import pytest

from foundry_mcp.core.providers import detect_provider_availability


# =============================================================================
# Skip conditions
# =============================================================================

requires_gemini = pytest.mark.skipif(
    not detect_provider_availability("gemini"),
    reason="gemini CLI not available",
)

requires_codex = pytest.mark.skipif(
    not detect_provider_availability("codex"),
    reason="codex CLI not available",
)


# =============================================================================
# AI Consultation Router Smoke Tests
# =============================================================================


@pytest.mark.live_providers
@pytest.mark.router_smoke
@pytest.mark.gemini
@requires_gemini
class TestConsultationOrchestratorSmoke:
    """Smoke tests for ConsultationOrchestrator with real providers."""

    def test_plan_review_single_provider(self):
        """Plan review through orchestrator with gemini-2.5-flash."""
        from foundry_mcp.core.ai_consultation import (
            ConsultationOrchestrator,
            ConsultationRequest,
            ConsultationResult,
            ConsultationWorkflow,
        )
        from foundry_mcp.core.llm_config import ConsultationConfig

        config = ConsultationConfig(
            priority=["[cli]gemini:gemini-2.5-flash"],
            default_timeout=DEFAULT_TIMEOUT,
            fallback_enabled=False,
        )
        orchestrator = ConsultationOrchestrator(config=config)

        request = ConsultationRequest(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            prompt_id="PLAN_REVIEW_FULL_V1",
            context={
                "spec_id": "smoke-test-001",
                "title": "Smoke Test Spec",
                "version": "1.0",
                "spec_content": """# Add greeting function
## Tasks
1. Create greet(name) function that returns "Hello, {name}!"
2. Add unit tests
""",
            },
            timeout=DEFAULT_TIMEOUT,
        )

        result = orchestrator.consult(request, use_cache=False)

        assert isinstance(result, ConsultationResult)
        assert result.error is None, f"Consultation failed: {result.error}"
        assert result.content, "Expected non-empty response"
        assert result.provider_id is not None
        assert result.duration_ms > 0

    def test_fidelity_review_single_provider(self):
        """Fidelity review through orchestrator with gemini-2.5-flash."""
        from foundry_mcp.core.ai_consultation import (
            ConsultationOrchestrator,
            ConsultationRequest,
            ConsultationResult,
            ConsultationWorkflow,
        )
        from foundry_mcp.core.llm_config import ConsultationConfig

        config = ConsultationConfig(
            priority=["[cli]gemini:gemini-2.5-flash"],
            default_timeout=DEFAULT_TIMEOUT,
            fallback_enabled=False,
        )
        orchestrator = ConsultationOrchestrator(config=config)

        request = ConsultationRequest(
            workflow=ConsultationWorkflow.FIDELITY_REVIEW,
            prompt_id="FIDELITY_REVIEW_V1",
            context={
                "spec_id": "smoke-test-002",
                "spec_title": "Greeting Function",
                "review_scope": "task-1",
                "spec_requirements": "Create greet(name) that returns 'Hello, {name}!'",
                "implementation_artifacts": """def greet(name):
    return f"Hello, {name}!"
""",
            },
            timeout=DEFAULT_TIMEOUT,
        )

        result = orchestrator.consult(request, use_cache=False)

        assert isinstance(result, ConsultationResult)
        assert result.error is None, f"Consultation failed: {result.error}"
        assert result.content, "Expected non-empty response"


@pytest.mark.live_providers
@pytest.mark.router_smoke
@pytest.mark.gemini
@pytest.mark.codex
@requires_gemini
@requires_codex
class TestConsultationOrchestratorMultiModelSmoke:
    """Smoke tests for multi-model consensus with real providers."""

    def test_plan_review_multi_model_consensus(self):
        """Plan review with 2 providers for consensus."""
        from foundry_mcp.core.ai_consultation import (
            ConsensusResult,
            ConsultationOrchestrator,
            ConsultationRequest,
            ConsultationWorkflow,
        )
        from foundry_mcp.core.llm_config import (
            ConsultationConfig,
            WorkflowConsultationConfig,
        )

        config = ConsultationConfig(
            priority=[
                "[cli]gemini:gemini-2.5-flash",
                "[cli]codex:gpt-5.1-codex-mini",
            ],
            default_timeout=COMPLEX_TIMEOUT,
            fallback_enabled=True,
            workflows={
                "plan_review": WorkflowConsultationConfig(min_models=2),
            },
        )
        orchestrator = ConsultationOrchestrator(config=config)

        request = ConsultationRequest(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            prompt_id="PLAN_REVIEW_FULL_V1",
            context={
                "spec_id": "consensus-test-001",
                "title": "Consensus Test Spec",
                "version": "1.0",
                "spec_content": """# Implement calculator
## Tasks
1. Create add(a, b) function
2. Create subtract(a, b) function
3. Add tests
""",
            },
            timeout=COMPLEX_TIMEOUT,
        )

        result = orchestrator.consult(request, use_cache=False)

        assert isinstance(result, ConsensusResult)
        assert result.success, f"Consensus failed: {result.warnings}"
        assert len(result.responses) >= 2, "Expected responses from 2 providers"
        assert result.agreement.successful_providers >= 2


# =============================================================================
# Research Router Smoke Tests
# =============================================================================


@pytest.mark.live_providers
@pytest.mark.router_smoke
@pytest.mark.gemini
@requires_gemini
class TestResearchRouterSmoke:
    """Smoke tests for Research Router with real providers."""

    @pytest.fixture(autouse=True)
    def setup_config(self, tmp_path):
        """Configure research with gemini provider."""
        from unittest.mock import patch

        from foundry_mcp.config import ResearchConfig

        research_cfg = ResearchConfig(
            enabled=True,
            storage_path=str(tmp_path),
            ttl_hours=24,
            default_provider="gemini",
            consensus_providers=["gemini"],
            thinkdeep_max_depth=2,
            ideate_perspectives=["technical"],
        )

        from unittest.mock import MagicMock
        mock_server_cfg = MagicMock()
        mock_server_cfg.research = research_cfg

        with patch("foundry_mcp.tools.unified.research._get_config", return_value=mock_server_cfg):
            with patch("foundry_mcp.tools.unified.research.get_flag_service") as mock_flag:
                mock_flag.return_value.is_enabled.return_value = True
                yield

    def test_chat_action(self):
        """Chat action through research router."""
        from foundry_mcp.tools.unified.research import _dispatch_research_action

        result = _dispatch_research_action(
            action="chat",
            prompt="What is 2 + 2? Reply with just the number.",
            provider="gemini",
        )

        assert result["success"] is True, f"Chat failed: {result.get('error')}"
        assert result["data"]["content"], "Expected non-empty response"
        assert "thread_id" in result["data"]

    def test_thinkdeep_action(self):
        """ThinkDeep action starts investigation."""
        from foundry_mcp.tools.unified.research import _dispatch_research_action

        result = _dispatch_research_action(
            action="thinkdeep",
            topic="Why is the sky blue?",
            provider="gemini",
        )

        assert result["success"] is True, f"ThinkDeep failed: {result.get('error')}"
        assert result["data"]["content"], "Expected non-empty response"
        assert "investigation_id" in result["data"]

    def test_ideate_action(self):
        """Ideate action generates ideas."""
        from foundry_mcp.tools.unified.research import _dispatch_research_action

        result = _dispatch_research_action(
            action="ideate",
            topic="Ways to improve code review process",
            ideation_action="generate",
            provider="gemini",
        )

        assert result["success"] is True, f"Ideate failed: {result.get('error')}"
        assert result["data"]["content"], "Expected non-empty response"
        assert "ideation_id" in result["data"]


@pytest.mark.live_providers
@pytest.mark.router_smoke
@pytest.mark.gemini
@pytest.mark.codex
@requires_gemini
@requires_codex
class TestResearchRouterConsensusSmoke:
    """Smoke tests for Research Router consensus with multiple providers."""

    @pytest.fixture(autouse=True)
    def setup_config(self, tmp_path):
        """Configure research with multiple providers."""
        from unittest.mock import patch

        from foundry_mcp.config import ResearchConfig

        research_cfg = ResearchConfig(
            enabled=True,
            storage_path=str(tmp_path),
            ttl_hours=24,
            default_provider="gemini",
            consensus_providers=["gemini", "codex"],
            thinkdeep_max_depth=2,
            ideate_perspectives=["technical"],
        )

        from unittest.mock import MagicMock
        mock_server_cfg = MagicMock()
        mock_server_cfg.research = research_cfg

        with patch("foundry_mcp.tools.unified.research._get_config", return_value=mock_server_cfg):
            with patch("foundry_mcp.tools.unified.research.get_flag_service") as mock_flag:
                mock_flag.return_value.is_enabled.return_value = True
                yield

    def test_consensus_action(self):
        """Consensus action queries multiple providers."""
        from foundry_mcp.tools.unified.research import _dispatch_research_action

        result = _dispatch_research_action(
            action="consensus",
            prompt="What is the capital of France? Reply with just the city name.",
            providers=["gemini", "codex"],
            strategy="all_responses",
        )

        assert result["success"] is True, f"Consensus failed: {result.get('error')}"
        assert result["data"]["content"], "Expected non-empty response"
        assert "consensus_id" in result["data"]
        assert len(result["data"]["providers_consulted"]) >= 1
