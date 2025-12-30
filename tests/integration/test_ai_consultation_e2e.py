"""
End-to-end tests for AI Consultation with mocked providers.

Tests the full consultation flow from request to response,
including cache interactions, workflow selection, and error handling.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from foundry_mcp.core.ai_consultation import (
    ConsensusResult,
    ConsultationOrchestrator,
    ConsultationRequest,
    ConsultationResult,
    ConsultationWorkflow,
    ResultCache,
)
from foundry_mcp.core.llm_config import ConsultationConfig, WorkflowConsultationConfig
from foundry_mcp.core.providers import ProviderResult, ProviderStatus, TokenUsage


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Create a temporary cache directory."""
    cache = tmp_path / "consultation_cache"
    cache.mkdir(parents=True)
    return cache


@pytest.fixture
def result_cache(cache_dir: Path) -> ResultCache:
    """Create a ResultCache for testing."""
    return ResultCache(base_dir=cache_dir, default_ttl=3600)


@pytest.fixture
def single_model_config() -> ConsultationConfig:
    """Configuration for single-model mode (default)."""
    return ConsultationConfig(
        priority=["gemini", "claude"],
        default_timeout=60.0,
        fallback_enabled=True,
        max_retries=1,
        retry_delay=0.1,
        cache_ttl=3600,
    )


@pytest.fixture
def multi_model_config() -> ConsultationConfig:
    """Configuration for multi-model consensus mode."""
    return ConsultationConfig(
        priority=["gemini", "claude", "openai"],
        default_timeout=60.0,
        fallback_enabled=True,
        max_retries=1,
        retry_delay=0.1,
        cache_ttl=3600,
        workflows={
            "plan_review": WorkflowConsultationConfig(min_models=2),
            "fidelity_review": WorkflowConsultationConfig(min_models=2),
        },
    )


@pytest.fixture
def mock_provider_result():
    """Factory for creating mock ProviderResult objects."""

    def _create(
        content: str = "## AI Review\n\nThis is a generated review response.",
        success: bool = True,
        provider_id: str = "gemini",
        model: str = "gemini-2.0-flash",
    ):
        return ProviderResult(
            content=content,
            status=ProviderStatus.SUCCESS if success else ProviderStatus.ERROR,
            provider_id=provider_id,
            model_used=model,
            tokens=TokenUsage(input_tokens=100, output_tokens=200, total_tokens=300),
            duration_ms=750.0,
        )

    return _create


@pytest.fixture
def mock_provider(mock_provider_result):
    """Create a mock provider that returns successful results."""
    provider = MagicMock()
    provider.generate.return_value = mock_provider_result()
    return provider


# =============================================================================
# Plan Review E2E Tests
# =============================================================================


class TestPlanReviewE2E:
    """End-to-end tests for plan review workflow."""

    def test_plan_review_full_flow(
        self, result_cache, single_model_config, mock_provider
    ):
        """Full plan review flow from request to response."""
        orchestrator = ConsultationOrchestrator(
            cache=result_cache, config=single_model_config
        )

        with patch(
            "foundry_mcp.core.ai_consultation.check_provider_available",
            return_value=True,
        ):
            with patch(
                "foundry_mcp.core.ai_consultation.available_providers",
                return_value=["gemini"],
            ):
                with patch(
                    "foundry_mcp.core.ai_consultation.resolve_provider",
                    return_value=mock_provider,
                ):
                    request = ConsultationRequest(
                        workflow=ConsultationWorkflow.PLAN_REVIEW,
                        prompt_id="PLAN_REVIEW_FULL_V1",
                        context={
                            "spec_id": "test-spec-001",
                            "title": "Test Specification",
                            "version": "1.0",
                            "spec_content": "This is the specification content...",
                        },
                    )
                    result = orchestrator.consult(request, use_cache=False)

        assert isinstance(result, ConsultationResult)
        assert result.content != ""
        assert result.error is None
        assert result.workflow == ConsultationWorkflow.PLAN_REVIEW
        assert result.duration_ms > 0

    def test_plan_review_cached_response(
        self, result_cache, single_model_config, mock_provider
    ):
        """Plan review returns cached response on second call."""
        orchestrator = ConsultationOrchestrator(
            cache=result_cache, config=single_model_config
        )

        with patch(
            "foundry_mcp.core.ai_consultation.check_provider_available",
            return_value=True,
        ):
            with patch(
                "foundry_mcp.core.ai_consultation.available_providers",
                return_value=["gemini"],
            ):
                with patch(
                    "foundry_mcp.core.ai_consultation.resolve_provider",
                    return_value=mock_provider,
                ):
                    request = ConsultationRequest(
                        workflow=ConsultationWorkflow.PLAN_REVIEW,
                        prompt_id="PLAN_REVIEW_FULL_V1",
                        context={
                            "spec_id": "cache-test-001",
                            "title": "Cache Test",
                            "version": "1.0",
                            "spec_content": "Content for cache test",
                        },
                    )

                    # First call - should hit provider
                    result1 = orchestrator.consult(request, use_cache=True)
                    assert result1.cache_hit is False

                    # Second call - should return cached
                    result2 = orchestrator.consult(request, use_cache=True)
                    assert result2.cache_hit is True
                    assert result2.content == result1.content

        # Verify provider was only called once
        assert mock_provider.generate.call_count == 1

    def test_plan_review_bypass_cache(
        self, result_cache, single_model_config, mock_provider
    ):
        """Plan review can bypass cache when use_cache=False."""
        orchestrator = ConsultationOrchestrator(
            cache=result_cache, config=single_model_config
        )

        with patch(
            "foundry_mcp.core.ai_consultation.check_provider_available",
            return_value=True,
        ):
            with patch(
                "foundry_mcp.core.ai_consultation.available_providers",
                return_value=["gemini"],
            ):
                with patch(
                    "foundry_mcp.core.ai_consultation.resolve_provider",
                    return_value=mock_provider,
                ):
                    request = ConsultationRequest(
                        workflow=ConsultationWorkflow.PLAN_REVIEW,
                        prompt_id="PLAN_REVIEW_FULL_V1",
                        context={
                            "spec_id": "bypass-test-001",
                            "title": "Bypass Test",
                            "version": "1.0",
                            "spec_content": "Content for bypass test",
                        },
                    )

                    # First call
                    orchestrator.consult(request, use_cache=True)

                    # Second call with cache bypass
                    result = orchestrator.consult(request, use_cache=False)
                    assert result.cache_hit is False

        # Verify provider was called twice
        assert mock_provider.generate.call_count == 2


# =============================================================================
# Fidelity Review E2E Tests
# =============================================================================


class TestFidelityReviewE2E:
    """End-to-end tests for fidelity review workflow."""

    def test_fidelity_review_full_flow(
        self, result_cache, single_model_config, mock_provider_result
    ):
        """Full fidelity review flow from request to response."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = mock_provider_result(
            content='{"verdict": "compliant", "deviations": [], "confidence": 0.95}'
        )

        orchestrator = ConsultationOrchestrator(
            cache=result_cache, config=single_model_config
        )

        with patch(
            "foundry_mcp.core.ai_consultation.check_provider_available",
            return_value=True,
        ):
            with patch(
                "foundry_mcp.core.ai_consultation.available_providers",
                return_value=["gemini"],
            ):
                with patch(
                    "foundry_mcp.core.ai_consultation.resolve_provider",
                    return_value=mock_provider,
                ):
                    request = ConsultationRequest(
                        workflow=ConsultationWorkflow.FIDELITY_REVIEW,
                        prompt_id="FIDELITY_REVIEW_V1",
                        context={
                            "spec_id": "fidelity-001",
                            "spec_title": "Test Spec",
                            "review_scope": "task-1-1",
                            "spec_requirements": "Implement feature X",
                            "implementation_artifacts": "git diff content here",
                        },
                    )
                    result = orchestrator.consult(request, use_cache=False)

        assert isinstance(result, ConsultationResult)
        assert result.content != ""
        assert result.workflow == ConsultationWorkflow.FIDELITY_REVIEW


# =============================================================================
# Multi-Model Consensus E2E Tests
# =============================================================================


class TestMultiModelConsensusE2E:
    """End-to-end tests for multi-model consensus mode."""

    def test_multi_model_plan_review(
        self, result_cache, multi_model_config, mock_provider_result
    ):
        """Multi-model plan review returns ConsensusResult."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = mock_provider_result()

        orchestrator = ConsultationOrchestrator(
            cache=result_cache, config=multi_model_config
        )

        with patch(
            "foundry_mcp.core.ai_consultation.check_provider_available",
            return_value=True,
        ):
            with patch(
                "foundry_mcp.core.ai_consultation.available_providers",
                return_value=["gemini", "claude"],
            ):
                with patch(
                    "foundry_mcp.core.ai_consultation.resolve_provider",
                    return_value=mock_provider,
                ):
                    request = ConsultationRequest(
                        workflow=ConsultationWorkflow.PLAN_REVIEW,
                        prompt_id="PLAN_REVIEW_FULL_V1",
                        context={
                            "spec_id": "multi-001",
                            "title": "Multi-Model Test",
                            "version": "1.0",
                            "spec_content": "Content for multi-model test",
                        },
                    )
                    result = orchestrator.consult(request, use_cache=False)

        # Multi-model workflow should return ConsensusResult
        assert isinstance(result, ConsensusResult)
        assert len(result.responses) >= 1
        assert result.agreement is not None
        assert result.success is True

    def test_multi_model_partial_failure_with_fallback(
        self, result_cache, multi_model_config, mock_provider_result
    ):
        """Multi-model mode handles partial failures with fallback."""
        call_count = [0]

        def mock_generate(request):
            call_count[0] += 1
            if call_count[0] == 1:
                # First provider fails
                return ProviderResult(
                    content="",
                    status=ProviderStatus.ERROR,
                    provider_id="gemini",
                    model_used="gemini-2.0-flash",
                    stderr="Rate limit exceeded",
                )
            else:
                # Other providers succeed
                return mock_provider_result(provider_id=f"provider-{call_count[0]}")

        mock_provider = MagicMock()
        mock_provider.generate.side_effect = mock_generate

        orchestrator = ConsultationOrchestrator(
            cache=result_cache, config=multi_model_config
        )

        with patch(
            "foundry_mcp.core.ai_consultation.check_provider_available",
            return_value=True,
        ):
            with patch(
                "foundry_mcp.core.ai_consultation.available_providers",
                return_value=["gemini", "claude", "openai"],
            ):
                with patch(
                    "foundry_mcp.core.ai_consultation.resolve_provider",
                    return_value=mock_provider,
                ):
                    request = ConsultationRequest(
                        workflow=ConsultationWorkflow.PLAN_REVIEW,
                        prompt_id="PLAN_REVIEW_FULL_V1",
                        context={
                            "spec_id": "fallback-001",
                            "title": "Fallback Test",
                            "version": "1.0",
                            "spec_content": "Content for fallback test",
                        },
                    )
                    result = orchestrator.consult(request, use_cache=False)

        assert isinstance(result, ConsensusResult)
        # Should have succeeded - fallback achieved minimum required providers
        assert result.success is True
        # Should have at least 2 successful providers (min_models=2)
        assert result.agreement.successful_providers >= 2

    def test_multi_model_all_providers_fail(
        self, result_cache, multi_model_config
    ):
        """Multi-model mode handles complete failure gracefully."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = ProviderResult(
            content="",
            status=ProviderStatus.ERROR,
            provider_id="test",
            model_used="test-model",
            stderr="Service unavailable",
        )

        orchestrator = ConsultationOrchestrator(
            cache=result_cache, config=multi_model_config
        )

        with patch(
            "foundry_mcp.core.ai_consultation.check_provider_available",
            return_value=True,
        ):
            with patch(
                "foundry_mcp.core.ai_consultation.available_providers",
                return_value=["gemini", "claude"],
            ):
                with patch(
                    "foundry_mcp.core.ai_consultation.resolve_provider",
                    return_value=mock_provider,
                ):
                    request = ConsultationRequest(
                        workflow=ConsultationWorkflow.PLAN_REVIEW,
                        prompt_id="PLAN_REVIEW_FULL_V1",
                        context={
                            "spec_id": "all-fail-001",
                            "title": "All Fail Test",
                            "version": "1.0",
                            "spec_content": "Content for all-fail test",
                        },
                    )
                    result = orchestrator.consult(request, use_cache=False)

        assert isinstance(result, ConsensusResult)
        assert result.success is False
        assert result.agreement.successful_providers == 0


# =============================================================================
# Provider Fallback E2E Tests
# =============================================================================


class TestProviderFallbackE2E:
    """End-to-end tests for provider fallback behavior."""

    def test_fallback_to_second_provider(
        self, result_cache, single_model_config, mock_provider_result
    ):
        """System falls back to second provider when first fails."""
        call_count = [0]

        def mock_generate(request):
            call_count[0] += 1
            if call_count[0] == 1:
                return ProviderResult(
                    content="",
                    status=ProviderStatus.ERROR,
                    provider_id="gemini",
                    model_used="gemini-2.0-flash",
                    stderr="Connection timeout",
                )
            else:
                return mock_provider_result(
                    content="Fallback response",
                    provider_id="claude",
                )

        mock_provider = MagicMock()
        mock_provider.generate.side_effect = mock_generate

        orchestrator = ConsultationOrchestrator(
            cache=result_cache, config=single_model_config
        )

        with patch(
            "foundry_mcp.core.ai_consultation.check_provider_available",
            return_value=True,
        ):
            with patch(
                "foundry_mcp.core.ai_consultation.available_providers",
                return_value=["gemini", "claude"],
            ):
                with patch(
                    "foundry_mcp.core.ai_consultation.resolve_provider",
                    return_value=mock_provider,
                ):
                    request = ConsultationRequest(
                        workflow=ConsultationWorkflow.PLAN_REVIEW,
                        prompt_id="PLAN_REVIEW_FULL_V1",
                        context={
                            "spec_id": "fallback-single-001",
                            "title": "Fallback Test",
                            "version": "1.0",
                            "spec_content": "Content for fallback test",
                        },
                    )
                    result = orchestrator.consult(request, use_cache=False)

        assert isinstance(result, ConsultationResult)
        assert result.content == "Fallback response"
        assert result.error is None

    def test_no_fallback_when_disabled(
        self, result_cache, mock_provider_result
    ):
        """System does not fallback when fallback_enabled=False."""
        config = ConsultationConfig(
            priority=["gemini", "claude"],
            fallback_enabled=False,  # Disabled
        )

        mock_provider = MagicMock()
        mock_provider.generate.return_value = ProviderResult(
            content="",
            status=ProviderStatus.ERROR,
            provider_id="gemini",
            model_used="gemini-2.0-flash",
            stderr="First provider failed",
        )

        orchestrator = ConsultationOrchestrator(
            cache=result_cache, config=config
        )

        with patch(
            "foundry_mcp.core.ai_consultation.check_provider_available",
            return_value=True,
        ):
            with patch(
                "foundry_mcp.core.ai_consultation.available_providers",
                return_value=["gemini", "claude"],
            ):
                with patch(
                    "foundry_mcp.core.ai_consultation.resolve_provider",
                    return_value=mock_provider,
                ):
                    request = ConsultationRequest(
                        workflow=ConsultationWorkflow.PLAN_REVIEW,
                        prompt_id="PLAN_REVIEW_FULL_V1",
                        context={
                            "spec_id": "no-fallback-001",
                            "title": "No Fallback Test",
                            "version": "1.0",
                            "spec_content": "Content",
                        },
                    )
                    result = orchestrator.consult(request, use_cache=False)

        # Should fail without trying second provider
        assert result.error is not None
        # Should only have been called once
        assert mock_provider.generate.call_count == 1


# =============================================================================
# Error Handling E2E Tests
# =============================================================================


class TestErrorHandlingE2E:
    """End-to-end tests for error handling."""

    def test_no_providers_available_error(self, result_cache, single_model_config):
        """Appropriate error when no providers are available."""
        orchestrator = ConsultationOrchestrator(
            cache=result_cache, config=single_model_config
        )

        with patch(
            "foundry_mcp.core.ai_consultation.check_provider_available",
            return_value=False,
        ):
            with patch(
                "foundry_mcp.core.ai_consultation.available_providers",
                return_value=[],
            ):
                request = ConsultationRequest(
                    workflow=ConsultationWorkflow.PLAN_REVIEW,
                    prompt_id="PLAN_REVIEW_FULL_V1",
                    context={
                        "spec_id": "error-001",
                        "title": "Error Test",
                        "version": "1.0",
                        "spec_content": "Content",
                    },
                )
                result = orchestrator.consult(request, use_cache=False)

        assert result.error is not None
        assert "provider" in result.error.lower()

    def test_invalid_prompt_id_error(
        self, result_cache, single_model_config, mock_provider
    ):
        """Appropriate error when prompt ID is invalid."""
        orchestrator = ConsultationOrchestrator(
            cache=result_cache, config=single_model_config
        )

        with patch(
            "foundry_mcp.core.ai_consultation.check_provider_available",
            return_value=True,
        ):
            with patch(
                "foundry_mcp.core.ai_consultation.available_providers",
                return_value=["gemini"],
            ):
                request = ConsultationRequest(
                    workflow=ConsultationWorkflow.PLAN_REVIEW,
                    prompt_id="NONEXISTENT_PROMPT_ID",
                    context={},
                )
                result = orchestrator.consult(request, use_cache=False)

        assert result.error is not None
        # Should mention prompt or failed
        assert "prompt" in result.error.lower() or "failed" in result.error.lower()


# =============================================================================
# Response Structure E2E Tests
# =============================================================================


class TestResponseStructureE2E:
    """End-to-end tests verifying response structure."""

    def test_single_model_response_structure(
        self, result_cache, single_model_config, mock_provider
    ):
        """Single-model response has expected structure."""
        orchestrator = ConsultationOrchestrator(
            cache=result_cache, config=single_model_config
        )

        with patch(
            "foundry_mcp.core.ai_consultation.check_provider_available",
            return_value=True,
        ):
            with patch(
                "foundry_mcp.core.ai_consultation.available_providers",
                return_value=["gemini"],
            ):
                with patch(
                    "foundry_mcp.core.ai_consultation.resolve_provider",
                    return_value=mock_provider,
                ):
                    request = ConsultationRequest(
                        workflow=ConsultationWorkflow.PLAN_REVIEW,
                        prompt_id="PLAN_REVIEW_FULL_V1",
                        context={
                            "spec_id": "struct-001",
                            "title": "Structure Test",
                            "version": "1.0",
                            "spec_content": "Content",
                        },
                    )
                    result = orchestrator.consult(request, use_cache=False)

        # Verify structure
        assert isinstance(result, ConsultationResult)
        assert hasattr(result, "workflow")
        assert hasattr(result, "content")
        assert hasattr(result, "provider_id")
        assert hasattr(result, "model_used")
        assert hasattr(result, "tokens")
        assert hasattr(result, "duration_ms")
        assert hasattr(result, "cache_hit")
        assert hasattr(result, "error")

    def test_multi_model_response_structure(
        self, result_cache, multi_model_config, mock_provider
    ):
        """Multi-model response has expected structure."""
        orchestrator = ConsultationOrchestrator(
            cache=result_cache, config=multi_model_config
        )

        with patch(
            "foundry_mcp.core.ai_consultation.check_provider_available",
            return_value=True,
        ):
            with patch(
                "foundry_mcp.core.ai_consultation.available_providers",
                return_value=["gemini", "claude"],
            ):
                with patch(
                    "foundry_mcp.core.ai_consultation.resolve_provider",
                    return_value=mock_provider,
                ):
                    request = ConsultationRequest(
                        workflow=ConsultationWorkflow.PLAN_REVIEW,
                        prompt_id="PLAN_REVIEW_FULL_V1",
                        context={
                            "spec_id": "multi-struct-001",
                            "title": "Multi Structure Test",
                            "version": "1.0",
                            "spec_content": "Content",
                        },
                    )
                    result = orchestrator.consult(request, use_cache=False)

        # Verify structure
        assert isinstance(result, ConsensusResult)
        assert hasattr(result, "workflow")
        assert hasattr(result, "responses")
        assert hasattr(result, "agreement")
        assert hasattr(result, "duration_ms")
        assert hasattr(result, "warnings")

        # Verify agreement structure
        assert hasattr(result.agreement, "total_providers")
        assert hasattr(result.agreement, "successful_providers")
        assert hasattr(result.agreement, "failed_providers")
        assert hasattr(result.agreement, "success_rate")
        assert hasattr(result.agreement, "has_consensus")
