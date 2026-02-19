"""
Provider smoke tests - basic connectivity and response validation.

These tests verify that each provider:
1. Is available (CLI installed and accessible)
2. Can accept a simple prompt
3. Returns a valid response

Run with: pytest tests/integration/providers/test_provider_smoke.py -m smoke
Enable live tests: FOUNDRY_ENABLE_LIVE_PROVIDER_TESTS=1
"""

import pytest

from foundry_mcp.core.providers import (
    ProviderHooks,
    detect_provider_availability,
    resolve_provider,
)

# =============================================================================
# Provider Availability Tests (no API calls)
# =============================================================================


class TestProviderAvailability:
    """Test provider detection without making API calls."""

    def test_gemini_detection(self):
        """Check if gemini CLI is detected."""
        available = detect_provider_availability("gemini")
        # Just report - don't fail if not available
        print(f"gemini available: {available}")

    def test_codex_detection(self):
        """Check if codex CLI is detected."""
        available = detect_provider_availability("codex")
        print(f"codex available: {available}")

    def test_claude_detection(self):
        """Check if claude CLI is detected."""
        available = detect_provider_availability("claude")
        print(f"claude available: {available}")

    def test_cursor_agent_detection(self):
        """Check if cursor-agent CLI is detected."""
        available = detect_provider_availability("cursor-agent")
        print(f"cursor-agent available: {available}")

    def test_opencode_detection(self):
        """Check if opencode CLI is detected."""
        available = detect_provider_availability("opencode")
        print(f"opencode available: {available}")

    def test_list_available_providers(self, available_providers_list):
        """List all currently available providers."""
        print(f"Available providers: {available_providers_list}")
        # At least one provider should be available for meaningful tests
        # This is informational, not a hard requirement
        if not available_providers_list:
            pytest.skip("No providers available - informational only")


# =============================================================================
# Live Provider Smoke Tests
# =============================================================================


@pytest.mark.smoke
@pytest.mark.live_providers
@pytest.mark.gemini
class TestGeminiSmoke:
    """Smoke tests for Gemini provider."""

    def test_gemini_simple_response(self, simple_prompt, provider_request_factory, validate_provider_result):
        """Test gemini responds to a simple prompt."""
        provider = resolve_provider("gemini", hooks=ProviderHooks())
        request = provider_request_factory(simple_prompt, timeout=30.0)

        result = provider.generate(request)

        validated = validate_provider_result(result)
        assert "PONG" in validated.content.upper(), f"Expected PONG in response: {validated.content}"

    def test_gemini_with_model_override(self, simple_prompt, provider_request_factory, validate_provider_result):
        """Test gemini with explicit model selection."""
        provider = resolve_provider("gemini", hooks=ProviderHooks())
        # Use gemini-2.5-flash model explicitly
        request = provider_request_factory(simple_prompt, model="gemini-2.5-flash", timeout=30.0)

        result = provider.generate(request)

        validate_provider_result(result)


@pytest.mark.smoke
@pytest.mark.live_providers
@pytest.mark.codex
class TestCodexSmoke:
    """Smoke tests for Codex provider."""

    def test_codex_simple_response(self, simple_prompt, provider_request_factory, validate_provider_result):
        """Test codex responds to a simple prompt."""
        provider = resolve_provider("codex", hooks=ProviderHooks())
        request = provider_request_factory(simple_prompt, timeout=30.0)

        result = provider.generate(request)

        validated = validate_provider_result(result)
        assert "PONG" in validated.content.upper(), f"Expected PONG in response: {validated.content}"


@pytest.mark.smoke
@pytest.mark.live_providers
@pytest.mark.claude
class TestClaudeSmoke:
    """Smoke tests for Claude provider."""

    def test_claude_simple_response(self, simple_prompt, provider_request_factory, validate_provider_result):
        """Test claude responds to a simple prompt."""
        provider = resolve_provider("claude", hooks=ProviderHooks())
        request = provider_request_factory(simple_prompt, model="haiku", timeout=30.0)

        result = provider.generate(request)

        validated = validate_provider_result(result)
        assert "PONG" in validated.content.upper(), f"Expected PONG in response: {validated.content}"

    def test_claude_with_haiku_model(self, simple_prompt, provider_request_factory, validate_provider_result):
        """Test claude with haiku model."""
        provider = resolve_provider("claude", hooks=ProviderHooks())
        request = provider_request_factory(simple_prompt, model="haiku", timeout=30.0)

        result = provider.generate(request)

        validate_provider_result(result)


@pytest.mark.smoke
@pytest.mark.live_providers
@pytest.mark.cursor_agent
class TestCursorAgentSmoke:
    """Smoke tests for Cursor Agent provider."""

    def test_cursor_agent_simple_response(self, simple_prompt, provider_request_factory, validate_provider_result):
        """Test cursor-agent responds to a simple prompt."""
        provider = resolve_provider("cursor-agent", hooks=ProviderHooks())
        request = provider_request_factory(simple_prompt, timeout=30.0)

        result = provider.generate(request)

        validated = validate_provider_result(result)
        assert "PONG" in validated.content.upper(), f"Expected PONG in response: {validated.content}"


@pytest.mark.smoke
@pytest.mark.live_providers
@pytest.mark.opencode
class TestOpenCodeSmoke:
    """Smoke tests for OpenCode provider."""

    def test_opencode_simple_response(self, simple_prompt, provider_request_factory, validate_provider_result):
        """Test opencode responds to a simple prompt."""
        provider = resolve_provider("opencode", hooks=ProviderHooks())
        request = provider_request_factory(simple_prompt, timeout=30.0)

        result = provider.generate(request)

        validated = validate_provider_result(result)
        assert "PONG" in validated.content.upper(), f"Expected PONG in response: {validated.content}"

    def test_opencode_with_backend_routing(self, simple_prompt, provider_request_factory, validate_provider_result):
        """Test opencode with backend/model routing."""
        provider = resolve_provider("opencode", hooks=ProviderHooks())
        # Route through openai backend
        request = provider_request_factory(simple_prompt, model="openai/gpt-5.1-mini", timeout=60.0)

        result = provider.generate(request)

        validate_provider_result(result)


# =============================================================================
# Cross-Provider Comparison Tests
# =============================================================================


@pytest.mark.smoke
@pytest.mark.live_providers
@pytest.mark.slow
class TestCrossProviderComparison:
    """Tests that run the same prompt across multiple providers."""

    def test_all_available_providers_respond(
        self,
        simple_prompt,
        available_providers_list,
        provider_request_factory,
        validate_provider_result,
    ):
        """Test that all available providers can respond to the same prompt."""
        if not available_providers_list:
            pytest.skip("No providers available")

        results = {}
        failures = {}

        for provider_id in available_providers_list:
            try:
                provider = resolve_provider(provider_id, hooks=ProviderHooks())
                request = provider_request_factory(simple_prompt, timeout=30.0)
                result = provider.generate(request)
                validated = validate_provider_result(result)
                results[provider_id] = validated.content
            except Exception as e:
                failures[provider_id] = str(e)

        # Report results
        print("\nProvider Results:")
        for provider_id, content in results.items():
            status = "PASS" if "PONG" in content.upper() else "FAIL"
            print(f"  {provider_id}: {status} - {content[:50]}...")

        if failures:
            print("\nProvider Failures:")
            for provider_id, error in failures.items():
                print(f"  {provider_id}: {error}")

        # At least one provider should succeed
        assert results, f"All providers failed: {failures}"
