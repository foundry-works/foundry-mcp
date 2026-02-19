"""
Plan review workflow tests across providers.

Tests the plan_review consultation workflow with each provider to verify:
1. Provider can process a plan review prompt
2. Response contains expected structure (feasibility, issues, recommendation)
3. Response is parseable JSON

NOTE: These tests validate response STRUCTURE only, not semantic AI correctness.
We do not assert whether the AI's plan review judgment is correct.

Run with: pytest tests/integration/providers/test_plan_review_flow.py -m plan_review
Enable live tests: FOUNDRY_ENABLE_LIVE_PROVIDER_TESTS=1
"""

import pytest

from foundry_mcp.core.providers import (
    ProviderHooks,
    resolve_provider,
)

# =============================================================================
# Per-Provider Plan Review Tests (Structure Validation Only)
# =============================================================================


@pytest.mark.plan_review
@pytest.mark.live_providers
@pytest.mark.gemini
class TestGeminiPlanReview:
    """Plan review structure tests for Gemini provider."""

    def test_plan_review_response_structure(
        self,
        simple_plan_review_prompt,
        provider_request_factory,
        validate_provider_result,
        validate_json_response,
    ):
        """Test gemini returns valid plan review response structure."""
        provider = resolve_provider("gemini", hooks=ProviderHooks())
        request = provider_request_factory(
            simple_plan_review_prompt,
            timeout=60.0,
            temperature=0.3,
        )

        result = provider.generate(request)

        validated = validate_provider_result(result)
        data = validate_json_response(
            validated.content,
            required_keys=["feasibility", "recommendation"],
        )

        # Structure validation only - no semantic correctness assertions
        assert isinstance(data["feasibility"], str), "feasibility must be string"
        assert isinstance(data["recommendation"], str), "recommendation must be string"
        assert isinstance(data.get("issues", []), list), "issues must be list"


@pytest.mark.plan_review
@pytest.mark.live_providers
@pytest.mark.codex
class TestCodexPlanReview:
    """Plan review structure tests for Codex provider."""

    def test_plan_review_response_structure(
        self,
        simple_plan_review_prompt,
        provider_request_factory,
        validate_provider_result,
        validate_json_response,
    ):
        """Test codex returns valid plan review response structure."""
        provider = resolve_provider("codex", hooks=ProviderHooks())
        request = provider_request_factory(
            simple_plan_review_prompt,
            timeout=60.0,
            temperature=0.3,
        )

        result = provider.generate(request)

        validated = validate_provider_result(result)
        data = validate_json_response(
            validated.content,
            required_keys=["feasibility", "recommendation"],
        )

        assert isinstance(data["feasibility"], str), "feasibility must be string"
        assert isinstance(data["recommendation"], str), "recommendation must be string"
        assert isinstance(data.get("issues", []), list), "issues must be list"


@pytest.mark.plan_review
@pytest.mark.live_providers
@pytest.mark.claude
class TestClaudePlanReview:
    """Plan review structure tests for Claude provider."""

    def test_plan_review_response_structure(
        self,
        simple_plan_review_prompt,
        provider_request_factory,
        validate_provider_result,
        validate_json_response,
    ):
        """Test claude returns valid plan review response structure."""
        provider = resolve_provider("claude", hooks=ProviderHooks())
        request = provider_request_factory(
            simple_plan_review_prompt,
            model="haiku",
            timeout=60.0,
            temperature=0.3,
        )

        result = provider.generate(request)

        validated = validate_provider_result(result)
        data = validate_json_response(
            validated.content,
            required_keys=["feasibility", "recommendation"],
        )

        assert isinstance(data["feasibility"], str), "feasibility must be string"
        assert isinstance(data["recommendation"], str), "recommendation must be string"
        assert isinstance(data.get("issues", []), list), "issues must be list"


@pytest.mark.plan_review
@pytest.mark.live_providers
@pytest.mark.cursor_agent
class TestCursorAgentPlanReview:
    """Plan review structure tests for Cursor Agent provider."""

    def test_plan_review_response_structure(
        self,
        simple_plan_review_prompt,
        provider_request_factory,
        validate_provider_result,
        validate_json_response,
    ):
        """Test cursor-agent returns valid plan review response structure."""
        provider = resolve_provider("cursor-agent", hooks=ProviderHooks())
        request = provider_request_factory(
            simple_plan_review_prompt,
            timeout=60.0,
            temperature=0.3,
        )

        result = provider.generate(request)

        validated = validate_provider_result(result)
        data = validate_json_response(
            validated.content,
            required_keys=["feasibility", "recommendation"],
        )

        assert isinstance(data["feasibility"], str), "feasibility must be string"
        assert isinstance(data["recommendation"], str), "recommendation must be string"
        assert isinstance(data.get("issues", []), list), "issues must be list"


@pytest.mark.plan_review
@pytest.mark.live_providers
@pytest.mark.opencode
class TestOpenCodePlanReview:
    """Plan review structure tests for OpenCode provider."""

    def test_plan_review_response_structure(
        self,
        simple_plan_review_prompt,
        provider_request_factory,
        validate_provider_result,
        validate_json_response,
    ):
        """Test opencode returns valid plan review response structure."""
        provider = resolve_provider("opencode", hooks=ProviderHooks())
        request = provider_request_factory(
            simple_plan_review_prompt,
            timeout=60.0,
            temperature=0.3,
        )

        result = provider.generate(request)

        validated = validate_provider_result(result)
        data = validate_json_response(
            validated.content,
            required_keys=["feasibility", "recommendation"],
        )

        assert isinstance(data["feasibility"], str), "feasibility must be string"
        assert isinstance(data["recommendation"], str), "recommendation must be string"
        assert isinstance(data.get("issues", []), list), "issues must be list"


# =============================================================================
# Cross-Provider Plan Review Comparison
# =============================================================================


@pytest.mark.plan_review
@pytest.mark.live_providers
@pytest.mark.slow
class TestCrossProviderPlanReview:
    """Compare plan review response structure across providers."""

    def test_all_providers_return_valid_structure(
        self,
        simple_plan_review_prompt,
        available_providers_list,
        provider_request_factory,
        validate_provider_result,
        validate_json_response,
    ):
        """Test all providers return valid plan review response structure."""
        if not available_providers_list:
            pytest.skip("No providers available")

        results = {}
        failures = {}

        for provider_id in available_providers_list:
            try:
                provider = resolve_provider(provider_id, hooks=ProviderHooks())
                request = provider_request_factory(
                    simple_plan_review_prompt,
                    timeout=60.0,
                    temperature=0.3,
                )
                result = provider.generate(request)
                validated = validate_provider_result(result)
                data = validate_json_response(
                    validated.content,
                    required_keys=["feasibility", "recommendation"],
                )
                results[provider_id] = data
            except Exception as e:
                failures[provider_id] = str(e)

        # Report results
        print("\nPlan Review Structure Results:")
        for provider_id, data in results.items():
            feasibility_type = type(data["feasibility"]).__name__
            recommendation_type = type(data["recommendation"]).__name__
            issues_count = len(data.get("issues", []))
            print(
                f"  {provider_id}: feasibility={feasibility_type}, recommendation={recommendation_type}, issues={issues_count}"
            )

        if failures:
            print("\nProvider Failures:")
            for provider_id, error in failures.items():
                print(f"  {provider_id}: {error}")

        # Validate structure for all successful responses
        for provider_id, data in results.items():
            assert isinstance(data["feasibility"], str), f"{provider_id}: feasibility must be string"
            assert isinstance(data["recommendation"], str), f"{provider_id}: recommendation must be string"
            assert isinstance(data.get("issues", []), list), f"{provider_id}: issues must be list"

        # At least one provider should succeed
        assert results, f"All providers failed: {failures}"
