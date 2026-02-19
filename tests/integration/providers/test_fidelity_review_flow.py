"""
Fidelity review workflow tests across providers.

Tests the fidelity_review consultation workflow with each provider to verify:
1. Provider can process a fidelity review prompt
2. Response contains expected structure (compliant, deviations)
3. Response is parseable JSON

NOTE: These tests validate response STRUCTURE only, not semantic AI correctness.
We do not assert whether the AI's compliance judgment is correct.

Run with: pytest tests/integration/providers/test_fidelity_review_flow.py -m fidelity_review
Enable live tests: FOUNDRY_ENABLE_LIVE_PROVIDER_TESTS=1
"""

import pytest

from foundry_mcp.core.providers import (
    ProviderHooks,
    resolve_provider,
)

# =============================================================================
# Test Fixtures
# =============================================================================

FIDELITY_REVIEW_PROMPT = """Check if this implementation matches the spec:

SPEC: Function greet(name: str) -> str that returns "Hello, {name}!"
IMPLEMENTATION:
```python
def greet(name: str) -> str:
    return f"Hello, {name}!"
```

Respond with a JSON object containing:
- "compliant": true or false
- "deviations": list of strings describing any deviations (empty if compliant)
"""


@pytest.fixture
def fidelity_review_prompt() -> str:
    """Standard fidelity review prompt for structure validation."""
    return FIDELITY_REVIEW_PROMPT


# =============================================================================
# Per-Provider Fidelity Review Tests (Structure Validation Only)
# =============================================================================


@pytest.mark.fidelity_review
@pytest.mark.live_providers
@pytest.mark.gemini
class TestGeminiFidelityReview:
    """Fidelity review structure tests for Gemini provider."""

    def test_fidelity_review_response_structure(
        self,
        fidelity_review_prompt,
        provider_request_factory,
        validate_provider_result,
        validate_json_response,
    ):
        """Test gemini returns valid fidelity review response structure."""
        provider = resolve_provider("gemini", hooks=ProviderHooks())
        request = provider_request_factory(
            fidelity_review_prompt,
            timeout=60.0,
            temperature=0.1,
        )

        result = provider.generate(request)

        validated = validate_provider_result(result)
        data = validate_json_response(validated.content, required_keys=["compliant"])

        # Structure validation only - no semantic correctness assertions
        assert isinstance(data["compliant"], bool), "compliant must be boolean"
        assert isinstance(data.get("deviations", []), list), "deviations must be list"


@pytest.mark.fidelity_review
@pytest.mark.live_providers
@pytest.mark.codex
class TestCodexFidelityReview:
    """Fidelity review structure tests for Codex provider."""

    def test_fidelity_review_response_structure(
        self,
        fidelity_review_prompt,
        provider_request_factory,
        validate_provider_result,
        validate_json_response,
    ):
        """Test codex returns valid fidelity review response structure."""
        provider = resolve_provider("codex", hooks=ProviderHooks())
        request = provider_request_factory(
            fidelity_review_prompt,
            timeout=60.0,
            temperature=0.1,
        )

        result = provider.generate(request)

        validated = validate_provider_result(result)
        data = validate_json_response(validated.content, required_keys=["compliant"])

        assert isinstance(data["compliant"], bool), "compliant must be boolean"
        assert isinstance(data.get("deviations", []), list), "deviations must be list"


@pytest.mark.fidelity_review
@pytest.mark.live_providers
@pytest.mark.claude
class TestClaudeFidelityReview:
    """Fidelity review structure tests for Claude provider."""

    def test_fidelity_review_response_structure(
        self,
        fidelity_review_prompt,
        provider_request_factory,
        validate_provider_result,
        validate_json_response,
    ):
        """Test claude returns valid fidelity review response structure."""
        provider = resolve_provider("claude", hooks=ProviderHooks())
        request = provider_request_factory(
            fidelity_review_prompt,
            timeout=60.0,
            temperature=0.1,
        )

        result = provider.generate(request)

        validated = validate_provider_result(result)
        data = validate_json_response(validated.content, required_keys=["compliant"])

        assert isinstance(data["compliant"], bool), "compliant must be boolean"
        assert isinstance(data.get("deviations", []), list), "deviations must be list"


@pytest.mark.fidelity_review
@pytest.mark.live_providers
@pytest.mark.cursor_agent
class TestCursorAgentFidelityReview:
    """Fidelity review structure tests for Cursor Agent provider."""

    def test_fidelity_review_response_structure(
        self,
        fidelity_review_prompt,
        provider_request_factory,
        validate_provider_result,
        validate_json_response,
    ):
        """Test cursor-agent returns valid fidelity review response structure."""
        provider = resolve_provider("cursor-agent", hooks=ProviderHooks())
        request = provider_request_factory(
            fidelity_review_prompt,
            timeout=60.0,
            temperature=0.1,
        )

        result = provider.generate(request)

        validated = validate_provider_result(result)
        data = validate_json_response(validated.content, required_keys=["compliant"])

        assert isinstance(data["compliant"], bool), "compliant must be boolean"
        assert isinstance(data.get("deviations", []), list), "deviations must be list"


@pytest.mark.fidelity_review
@pytest.mark.live_providers
@pytest.mark.opencode
class TestOpenCodeFidelityReview:
    """Fidelity review structure tests for OpenCode provider."""

    def test_fidelity_review_response_structure(
        self,
        fidelity_review_prompt,
        provider_request_factory,
        validate_provider_result,
        validate_json_response,
    ):
        """Test opencode returns valid fidelity review response structure."""
        provider = resolve_provider("opencode", hooks=ProviderHooks())
        request = provider_request_factory(
            fidelity_review_prompt,
            timeout=60.0,
            temperature=0.1,
        )

        result = provider.generate(request)

        validated = validate_provider_result(result)
        data = validate_json_response(validated.content, required_keys=["compliant"])

        assert isinstance(data["compliant"], bool), "compliant must be boolean"
        assert isinstance(data.get("deviations", []), list), "deviations must be list"


# =============================================================================
# Cross-Provider Fidelity Review Comparison
# =============================================================================


@pytest.mark.fidelity_review
@pytest.mark.live_providers
@pytest.mark.slow
class TestCrossProviderFidelityReview:
    """Compare fidelity review response structure across providers."""

    def test_all_providers_return_valid_structure(
        self,
        fidelity_review_prompt,
        available_providers_list,
        provider_request_factory,
        validate_provider_result,
        validate_json_response,
    ):
        """Test all providers return valid fidelity review response structure."""
        if not available_providers_list:
            pytest.skip("No providers available")

        results = {}
        failures = {}

        for provider_id in available_providers_list:
            try:
                provider = resolve_provider(provider_id, hooks=ProviderHooks())
                request = provider_request_factory(
                    fidelity_review_prompt,
                    timeout=60.0,
                    temperature=0.1,
                )
                result = provider.generate(request)
                validated = validate_provider_result(result)
                data = validate_json_response(validated.content, required_keys=["compliant"])
                results[provider_id] = data
            except Exception as e:
                failures[provider_id] = str(e)

        # Report results
        print("\nFidelity Review Structure Results:")
        for provider_id, data in results.items():
            compliant_type = type(data["compliant"]).__name__
            deviations_type = type(data.get("deviations", [])).__name__
            print(f"  {provider_id}: compliant={compliant_type}, deviations={deviations_type}")

        if failures:
            print("\nProvider Failures:")
            for provider_id, error in failures.items():
                print(f"  {provider_id}: {error}")

        # Validate structure for all successful responses
        for provider_id, data in results.items():
            assert isinstance(data["compliant"], bool), f"{provider_id}: compliant must be boolean"
            assert isinstance(data.get("deviations", []), list), f"{provider_id}: deviations must be list"

        # At least one provider should succeed
        assert results, f"All providers failed: {failures}"
