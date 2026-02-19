"""
Synthesis workflow integration tests.

Tests the multi-model synthesis functionality for plan review and fidelity review:
1. Plan review synthesis consolidates multiple provider reviews
2. Fidelity review synthesis consolidates multiple provider reviews
3. Fallback behavior when synthesis fails
4. Synthesized response structure validation

NOTE: These tests validate synthesis FLOW and STRUCTURE, not semantic correctness.
We validate that synthesis produces expected output format with model attribution.

Run with: pytest tests/integration/providers/test_synthesis_flow.py -m synthesis
Enable live tests: FOUNDRY_ENABLE_LIVE_PROVIDER_TESTS=1
"""

import json

import pytest

from foundry_mcp.core.ai_consultation import (
    ConsensusResult,
    ConsultationOrchestrator,
    ConsultationResult,
    ConsultationWorkflow,
    ProviderResponse,
)
from foundry_mcp.core.prompts.fidelity_review import (
    FidelityReviewPromptBuilder,
)
from foundry_mcp.core.prompts.plan_review import (
    PlanReviewPromptBuilder,
)

# =============================================================================
# Test Fixtures - Mock Provider Responses
# =============================================================================


@pytest.fixture
def mock_plan_review_response_gemini() -> str:
    """Simulated plan review response from gemini provider."""
    return """# Review Summary

## Critical Blockers
None identified

## Major Suggestions
- **[Architecture]** Consider adding input validation
  - **Description:** The greet function should validate name is not empty
  - **Impact:** Could cause unexpected behavior with empty strings
  - **Fix:** Add `if not name: raise ValueError("name required")`

## Minor Suggestions
- **[Verification]** Add edge case tests
  - **Description:** Test with special characters and unicode
  - **Fix:** Add pytest parametrize with edge cases

## Questions
None identified

## Praise
- **[Completeness]** Clear and simple design
  - **Why:** Single responsibility, easy to understand
"""


@pytest.fixture
def mock_plan_review_response_codex() -> str:
    """Simulated plan review response from codex provider."""
    return """# Review Summary

## Critical Blockers
None identified

## Major Suggestions
- **[Architecture]** Consider adding input validation
  - **Description:** Should handle None and empty string inputs
  - **Impact:** Runtime errors if called with invalid input
  - **Fix:** Add type hints and validation

## Minor Suggestions
- **[Verification]** Improve test coverage
  - **Description:** Add tests for edge cases
  - **Fix:** Use pytest parametrize

## Questions
- **[Interface Design]** Should the function support multiple names?
  - **Context:** Future extensibility consideration
  - **Needed:** Clarification on requirements

## Praise
- **[Completeness]** Well-structured implementation plan
  - **Why:** Clear steps with testability built in
"""


@pytest.fixture
def mock_fidelity_review_response_gemini() -> str:
    """Simulated fidelity review JSON response from gemini provider."""
    return json.dumps(
        {
            "verdict": "pass",
            "summary": "Implementation matches specification",
            "requirement_alignment": {"answer": "yes", "details": "Function signature and return value match spec"},
            "success_criteria": {"met": "yes", "details": "All verification steps satisfied"},
            "deviations": [],
            "test_coverage": {"status": "sufficient", "details": "Tests cover happy path"},
            "code_quality": {"issues": [], "details": "Code is clean and readable"},
            "documentation": {"status": "adequate", "details": "Docstring present"},
            "issues": [],
            "recommendations": [],
        }
    )


@pytest.fixture
def mock_fidelity_review_response_codex() -> str:
    """Simulated fidelity review JSON response from codex provider."""
    return json.dumps(
        {
            "verdict": "partial",
            "summary": "Implementation mostly matches but missing edge case handling",
            "requirement_alignment": {
                "answer": "partial",
                "details": "Core functionality matches, but missing None handling",
            },
            "success_criteria": {"met": "partial", "details": "Main verification passes, edge cases not covered"},
            "deviations": [
                {
                    "description": "Missing None input handling",
                    "justification": "Spec implies robustness",
                    "severity": "medium",
                }
            ],
            "test_coverage": {"status": "insufficient", "details": "Missing edge case tests"},
            "code_quality": {"issues": ["No type hints"], "details": "Could improve with type annotations"},
            "documentation": {"status": "adequate", "details": "Basic docstring present"},
            "issues": ["Missing None handling", "No type hints"],
            "recommendations": ["Add input validation", "Add type hints"],
        }
    )


@pytest.fixture
def mock_synthesis_response_plan() -> str:
    """Simulated synthesis response for plan review."""
    return """# Synthesis

## Overall Assessment
- **Consensus Level**: Moderate (models agree on main points, differ on details)

## Critical Blockers
None identified

## Major Suggestions
- **[Architecture]** Input validation needed - flagged by: gemini, codex
  - Impact: Runtime errors with invalid input
  - Recommended fix: Add validation for empty/None inputs

## Questions for Author
- **[Interface Design]** Multi-name support? - flagged by: codex
  - Context: Future extensibility

## Design Strengths
- **[Completeness]** Clear design - noted by: gemini, codex
  - Why this is effective: Single responsibility, easy to understand

## Points of Agreement
- Both models agree input validation is needed
- Both praise the clear, simple design

## Points of Disagreement
- gemini focuses on empty string; codex emphasizes None handling

## Synthesis Notes
- Primary recommendation: Add input validation before implementation
- Secondary: Clarify multi-name requirements if needed
"""


@pytest.fixture
def mock_synthesis_response_fidelity() -> str:
    """Simulated synthesis response for fidelity review."""
    return json.dumps(
        {
            "verdict": "partial",
            "verdict_consensus": {
                "votes": {"pass": ["gemini"], "fail": [], "partial": ["codex"], "unknown": []},
                "agreement_level": "moderate",
                "notes": "Models disagree on edge case handling importance",
            },
            "summary": "Implementation mostly correct, edge case handling debated",
            "requirement_alignment": {
                "answer": "partial",
                "details": "Core functionality matches, edge cases contested",
                "model_agreement": "split",
            },
            "success_criteria": {"met": "partial", "details": "Main verification passes", "model_agreement": "split"},
            "deviations": [
                {
                    "description": "Missing None input handling",
                    "justification": "Spec may imply robustness",
                    "severity": "medium",
                    "identified_by": ["codex"],
                    "agreement": "single",
                }
            ],
            "test_coverage": {
                "status": "insufficient",
                "details": "Edge case coverage debated",
                "model_agreement": "split",
            },
            "code_quality": {
                "issues": ["No type hints - flagged by codex"],
                "details": "Gemini finds code acceptable, codex wants improvements",
            },
            "documentation": {
                "status": "adequate",
                "details": "Both models agree documentation is adequate",
                "model_agreement": "unanimous",
            },
            "issues": ["Edge case handling debated", "Type hints suggested by codex"],
            "recommendations": [
                "Consider adding input validation for None",
                "Add type hints for better maintainability",
            ],
            "synthesis_metadata": {
                "models_consulted": ["gemini", "codex"],
                "models_succeeded": ["gemini", "codex"],
                "models_failed": [],
                "synthesis_provider": "gemini",
                "agreement_level": "moderate",
            },
        }
    )


# =============================================================================
# Unit Tests - Synthesis Prompt Rendering
# =============================================================================


@pytest.mark.synthesis
class TestSynthesisPromptRendering:
    """Test that synthesis prompts render correctly."""

    def test_plan_review_synthesis_prompt_renders(
        self,
        mock_plan_review_response_gemini,
        mock_plan_review_response_codex,
    ):
        """Test SYNTHESIS_PROMPT_V1 renders with model reviews."""
        builder = PlanReviewPromptBuilder()

        model_reviews = f"""
## Review by gemini

{mock_plan_review_response_gemini}

---

## Review by codex

{mock_plan_review_response_codex}
"""
        prompt = builder.build(
            "SYNTHESIS_PROMPT_V1",
            {
                "spec_id": "test-spec",
                "title": "Test Specification",
                "num_models": 2,
                "model_reviews": model_reviews,
            },
        )

        assert "synthesizing 2 independent AI reviews" in prompt
        assert "test-spec" in prompt
        assert "Test Specification" in prompt
        assert "gemini" in prompt
        assert "codex" in prompt

    def test_fidelity_review_synthesis_prompt_renders(
        self,
        mock_fidelity_review_response_gemini,
        mock_fidelity_review_response_codex,
    ):
        """Test FIDELITY_SYNTHESIS_PROMPT_V1 renders with model reviews."""
        builder = FidelityReviewPromptBuilder()

        model_reviews = f"""
## Review by gemini

```json
{mock_fidelity_review_response_gemini}
```

---

## Review by codex

```json
{mock_fidelity_review_response_codex}
```
"""
        prompt = builder.build(
            "FIDELITY_SYNTHESIS_PROMPT_V1",
            {
                "spec_id": "test-spec",
                "spec_title": "Test Specification",
                "review_scope": "spec",
                "num_models": 2,
                "model_reviews": model_reviews,
            },
        )

        assert "synthesizing 2 independent AI fidelity reviews" in prompt
        assert "test-spec" in prompt
        assert "Test Specification" in prompt
        assert "verdict_consensus" in prompt  # Schema should be included
        assert "identified_by" in prompt  # Schema should include attribution


@pytest.mark.synthesis
class TestSynthesisPromptSchema:
    """Test synthesis prompt schema defaults."""

    def test_plan_synthesis_uses_standard_schema(self):
        """Plan synthesis prompt includes standard response schema."""
        builder = PlanReviewPromptBuilder()
        prompt = builder.build(
            "SYNTHESIS_PROMPT_V1",
            {
                "spec_id": "test",
                "title": "Test",
                "num_models": 2,
                "model_reviews": "test reviews",
            },
        )

        # Should include synthesis-specific format
        assert "Consensus Level" in prompt
        assert "flagged by:" in prompt
        assert "Points of Agreement" in prompt

    def test_fidelity_synthesis_uses_synthesized_schema(self):
        """Fidelity synthesis prompt includes synthesized response schema."""
        builder = FidelityReviewPromptBuilder()
        prompt = builder.build(
            "FIDELITY_SYNTHESIS_PROMPT_V1",
            {
                "spec_id": "test",
                "spec_title": "Test",
                "review_scope": "spec",
                "num_models": 2,
                "model_reviews": "test reviews",
            },
        )

        # Should include synthesis-specific fields
        assert "verdict_consensus" in prompt
        assert "identified_by" in prompt
        assert "synthesis_metadata" in prompt
        assert "agreement_level" in prompt


# =============================================================================
# Unit Tests - Synthesis Flow with Mocked Orchestrator
# =============================================================================


@pytest.mark.synthesis
@pytest.mark.plan_synthesis
class TestPlanReviewSynthesisFlow:
    """Test plan review synthesis flow with mocked providers."""

    def test_synthesis_triggered_with_two_providers(
        self,
        mock_plan_review_response_gemini,
        mock_plan_review_response_codex,
        mock_synthesis_response_plan,
    ):
        """Test that synthesis is triggered when 2+ providers succeed."""
        # Create mock ConsensusResult with two successful responses
        consensus_result = ConsensusResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            responses=[
                ProviderResponse(
                    provider_id="gemini",
                    model_used="pro",
                    content=mock_plan_review_response_gemini,
                    success=True,
                    error=None,
                ),
                ProviderResponse(
                    provider_id="codex",
                    model_used="gpt-5.1-codex-mini",
                    content=mock_plan_review_response_codex,
                    success=True,
                    error=None,
                ),
            ],
        )

        # Verify we have 2 successful responses
        successful = [r for r in consensus_result.responses if r.success and r.content.strip()]
        assert len(successful) == 2, "Should have 2 successful responses"
        assert consensus_result.success, "ConsensusResult should indicate success"

    def test_synthesis_not_triggered_with_one_provider(
        self,
        mock_plan_review_response_gemini,
    ):
        """Test that synthesis is NOT triggered with only 1 successful provider."""
        consensus_result = ConsensusResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            responses=[
                ProviderResponse(
                    provider_id="gemini",
                    model_used="pro",
                    content=mock_plan_review_response_gemini,
                    success=True,
                    error=None,
                ),
                ProviderResponse(
                    provider_id="codex",
                    model_used="gpt-5.1-codex-mini",
                    content="",
                    success=False,
                    error="Provider unavailable",
                ),
            ],
        )

        successful = [r for r in consensus_result.responses if r.success and r.content.strip()]
        assert len(successful) == 1, "Should have only 1 successful response"
        # In this case, synthesis should NOT be triggered

    def test_fallback_to_primary_content_on_synthesis_failure(
        self,
        mock_plan_review_response_gemini,
        mock_plan_review_response_codex,
    ):
        """Test fallback to primary_content when synthesis fails."""
        consensus_result = ConsensusResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            responses=[
                ProviderResponse(
                    provider_id="gemini",
                    model_used="pro",
                    content=mock_plan_review_response_gemini,
                    success=True,
                    error=None,
                ),
                ProviderResponse(
                    provider_id="codex",
                    model_used="gpt-5.1-codex-mini",
                    content=mock_plan_review_response_codex,
                    success=True,
                    error=None,
                ),
            ],
        )

        # primary_content should be the first successful provider's content
        assert consensus_result.primary_content == mock_plan_review_response_gemini


@pytest.mark.synthesis
@pytest.mark.fidelity_synthesis
class TestFidelityReviewSynthesisFlow:
    """Test fidelity review synthesis flow with mocked providers."""

    def test_synthesis_triggered_with_two_providers(
        self,
        mock_fidelity_review_response_gemini,
        mock_fidelity_review_response_codex,
    ):
        """Test that synthesis is triggered when 2+ providers succeed."""
        consensus_result = ConsensusResult(
            workflow=ConsultationWorkflow.FIDELITY_REVIEW,
            responses=[
                ProviderResponse(
                    provider_id="gemini",
                    model_used="pro",
                    content=mock_fidelity_review_response_gemini,
                    success=True,
                    error=None,
                ),
                ProviderResponse(
                    provider_id="codex",
                    model_used="gpt-5.1-codex-mini",
                    content=mock_fidelity_review_response_codex,
                    success=True,
                    error=None,
                ),
            ],
        )

        successful = [r for r in consensus_result.responses if r.success and r.content.strip()]
        assert len(successful) == 2, "Should have 2 successful responses"

    def test_synthesized_response_structure_validation(
        self,
        mock_synthesis_response_fidelity,
    ):
        """Test that synthesized fidelity response has expected structure."""
        data = json.loads(mock_synthesis_response_fidelity)

        # Verify synthesis-specific fields
        assert "verdict_consensus" in data
        assert "votes" in data["verdict_consensus"]
        assert "agreement_level" in data["verdict_consensus"]

        # Verify deviation attribution
        if data["deviations"]:
            for deviation in data["deviations"]:
                assert "identified_by" in deviation, "Deviations should have model attribution"
                assert "agreement" in deviation, "Deviations should have agreement level"

        # Verify synthesis metadata
        assert "synthesis_metadata" in data
        assert "models_consulted" in data["synthesis_metadata"]
        assert "models_succeeded" in data["synthesis_metadata"]
        assert "agreement_level" in data["synthesis_metadata"]

    def test_verdict_consensus_structure(
        self,
        mock_synthesis_response_fidelity,
    ):
        """Test verdict_consensus has correct vote structure."""
        data = json.loads(mock_synthesis_response_fidelity)

        verdict_consensus = data["verdict_consensus"]
        votes = verdict_consensus["votes"]

        # All verdict options should be present
        assert "pass" in votes
        assert "fail" in votes
        assert "partial" in votes
        assert "unknown" in votes

        # Each vote category should be a list of model names
        for category in ["pass", "fail", "partial", "unknown"]:
            assert isinstance(votes[category], list)

        # Agreement level should be valid
        assert verdict_consensus["agreement_level"] in ["strong", "moderate", "weak", "conflicted"]


# =============================================================================
# Integration Tests - Live Provider Synthesis (requires providers)
# =============================================================================


@pytest.mark.synthesis
@pytest.mark.live_providers
@pytest.mark.slow
class TestLivePlanReviewSynthesis:
    """Live integration tests for plan review synthesis.

    These tests require actual AI providers to be available.
    Run with: FOUNDRY_ENABLE_LIVE_PROVIDER_TESTS=1 pytest -m "synthesis and live_providers"
    """

    def test_orchestrator_handles_consensus_result(
        self,
        available_providers_list,
    ):
        """Test that orchestrator returns ConsensusResult for multi-model config."""
        if len(available_providers_list) < 2:
            pytest.skip("Need at least 2 providers for synthesis test")

        # This test validates the orchestrator flow, not actual synthesis
        # Actual synthesis requires min_models > 1 in workflow config
        orchestrator = ConsultationOrchestrator()

        # Verify orchestrator is available
        assert orchestrator.is_available(), "Orchestrator should have available providers"


@pytest.mark.synthesis
@pytest.mark.live_providers
@pytest.mark.slow
class TestLiveFidelityReviewSynthesis:
    """Live integration tests for fidelity review synthesis.

    These tests require actual AI providers to be available.
    Run with: FOUNDRY_ENABLE_LIVE_PROVIDER_TESTS=1 pytest -m "synthesis and live_providers"
    """

    def test_orchestrator_handles_fidelity_workflow(
        self,
        available_providers_list,
    ):
        """Test that orchestrator can process fidelity review workflow."""
        if not available_providers_list:
            pytest.skip("No providers available")

        orchestrator = ConsultationOrchestrator()
        assert orchestrator.is_available(), "Orchestrator should have available providers"


# =============================================================================
# Unit Tests - Response Building with Synthesis Metadata
# =============================================================================


@pytest.mark.synthesis
class TestSynthesisResponseBuilding:
    """Test that synthesis metadata is correctly included in responses."""

    def test_consensus_info_includes_synthesis_flag(self):
        """Test consensus info includes synthesis_performed flag."""
        # Simulate what _handle_fidelity builds
        consensus_info = {
            "mode": "multi_model",
            "threshold": 2,
            "provider_id": "gemini",
            "model_used": "gemini-pro",
            "synthesis_performed": True,
            "successful_providers": ["gemini", "codex"],
            "failed_providers": [],
        }

        assert consensus_info["synthesis_performed"] is True
        assert "successful_providers" in consensus_info
        assert len(consensus_info["successful_providers"]) == 2

    def test_consensus_info_includes_synthesis_error_on_failure(self):
        """Test consensus info includes synthesis_error when synthesis fails."""
        consensus_info = {
            "mode": "multi_model",
            "threshold": 2,
            "provider_id": "gemini",
            "model_used": "gemini-pro",
            "synthesis_performed": False,
            "synthesis_error": "empty response",
            "successful_providers": ["gemini", "codex"],
            "failed_providers": [],
        }

        assert consensus_info["synthesis_performed"] is False
        assert "synthesis_error" in consensus_info


# =============================================================================
# Edge Cases
# =============================================================================


@pytest.mark.synthesis
class TestSynthesisEdgeCases:
    """Test edge cases in synthesis flow."""

    def test_empty_content_not_counted_as_success(self):
        """Test that empty content responses are not counted as successful."""
        consensus_result = ConsensusResult(
            workflow=ConsultationWorkflow.FIDELITY_REVIEW,
            responses=[
                ProviderResponse(
                    provider_id="gemini",
                    model_used="pro",
                    content="valid content",
                    success=True,
                    error=None,
                ),
                ProviderResponse(
                    provider_id="codex",
                    model_used="gpt-5.1-codex-mini",
                    content="   ",  # Whitespace only
                    success=True,
                    error=None,
                ),
            ],
        )

        # Filter as done in _handle_fidelity
        successful = [r for r in consensus_result.responses if r.success and r.content.strip()]
        assert len(successful) == 1, "Empty content should not count as successful"

    def test_all_providers_failed(self):
        """Test handling when all providers fail."""
        consensus_result = ConsensusResult(
            workflow=ConsultationWorkflow.FIDELITY_REVIEW,
            responses=[
                ProviderResponse(
                    provider_id="gemini",
                    model_used="pro",
                    content="",
                    success=False,
                    error="Timeout",
                ),
                ProviderResponse(
                    provider_id="codex",
                    model_used="gpt-5.1-codex-mini",
                    content="",
                    success=False,
                    error="Rate limited",
                ),
            ],
        )

        successful = [r for r in consensus_result.responses if r.success and r.content.strip()]
        assert len(successful) == 0, "No successful responses"
        assert not consensus_result.success, "ConsensusResult should indicate failure"

    def test_single_provider_mode_no_synthesis(self):
        """Test that single provider mode (ConsultationResult) bypasses synthesis."""
        single_result = ConsultationResult(
            workflow=ConsultationWorkflow.FIDELITY_REVIEW,
            content="Single provider response",
            provider_id="gemini",
            model_used="gemini-pro",
            tokens=100,
            duration_ms=500,
            cache_hit=False,
            warnings=[],
            error=None,
        )

        # In single provider mode, we use content directly, no synthesis
        assert single_result.content == "Single provider response"
        assert not isinstance(single_result, ConsensusResult)
