"""
Provider integration test configuration.

Provides fixtures and markers for testing real AI provider invocations.
Tests are skipped by default unless explicitly enabled via markers or environment.

Usage:
    # Run all provider tests (requires all providers available)
    pytest tests/integration/providers/ -m live_providers

    # Run specific provider tests
    pytest tests/integration/providers/ -m gemini
    pytest tests/integration/providers/ -m codex
    pytest tests/integration/providers/ -m claude

    # Run smoke tests only (quick availability check)
    pytest tests/integration/providers/ -m smoke

    # Run workflow tests only
    pytest tests/integration/providers/ -m plan_review
    pytest tests/integration/providers/ -m fidelity_review
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from foundry_mcp.core.providers import (
    ProviderRequest,
    check_provider_available,
    detect_provider_availability,
)


# =============================================================================
# Marker Registration
# =============================================================================


def pytest_configure(config):
    """Register custom markers for provider tests."""
    # Provider-specific markers
    config.addinivalue_line("markers", "live_providers: tests that invoke real AI providers")
    config.addinivalue_line("markers", "gemini: tests requiring gemini CLI")
    config.addinivalue_line("markers", "codex: tests requiring codex CLI")
    config.addinivalue_line("markers", "claude: tests requiring claude CLI")
    config.addinivalue_line("markers", "cursor_agent: tests requiring cursor-agent CLI")
    config.addinivalue_line("markers", "opencode: tests requiring opencode CLI")

    # Test category markers
    config.addinivalue_line("markers", "smoke: quick provider availability tests")
    config.addinivalue_line("markers", "plan_review: plan review workflow tests")
    config.addinivalue_line("markers", "fidelity_review: fidelity review workflow tests")
    config.addinivalue_line("markers", "slow: tests that may take >30 seconds")
    config.addinivalue_line("markers", "synthesis: multi-model synthesis workflow tests")
    config.addinivalue_line("markers", "plan_synthesis: plan review synthesis tests")
    config.addinivalue_line("markers", "fidelity_synthesis: fidelity review synthesis tests")


# =============================================================================
# Skip Logic
# =============================================================================


def pytest_collection_modifyitems(config, items):
    """Skip provider tests if the provider CLI is not available or not configured."""
    import os

    provider_markers = {"gemini", "codex", "claude", "cursor_agent", "opencode"}

    # Check if live provider tests are explicitly enabled
    live_tests_enabled = os.environ.get("FOUNDRY_ENABLE_LIVE_PROVIDER_TESTS", "").lower() in ("1", "true", "yes")

    for item in items:
        item_markers = {m.name for m in item.iter_markers()}

        # Skip all live_providers tests unless explicitly enabled
        if "live_providers" in item_markers and not live_tests_enabled:
            item.add_marker(
                pytest.mark.skip(reason="Live provider tests disabled (set FOUNDRY_ENABLE_LIVE_PROVIDER_TESTS=1 to enable)")
            )
            continue

        # Check provider availability for specific provider tests
        for provider in provider_markers:
            if provider in item_markers:
                provider_id = provider.replace("_", "-")  # cursor_agent -> cursor-agent
                if not detect_provider_availability(provider_id):
                    item.add_marker(
                        pytest.mark.skip(reason=f"Provider '{provider_id}' not available")
                    )


# =============================================================================
# Fixtures - Simple Test Prompts
# =============================================================================

SIMPLE_PROMPT = "Reply with exactly: PONG"

SIMPLE_PLAN_REVIEW_PROMPT = """Review this simple plan and provide brief feedback:

# Plan: Add greeting function
1. Create greet(name) function
2. Return "Hello, {name}!"
3. Add tests

Respond with a JSON object containing:
- "feasibility": "high" or "medium" or "low"
- "issues": list of strings (can be empty)
- "recommendation": "approve" or "revise"
"""

SIMPLE_FIDELITY_PROMPT = """Check if this implementation matches the spec:

SPEC: Function greet(name) returns "Hello, {name}!"
IMPLEMENTATION: def greet(name): return f"Hello, {name}!"

Respond with a JSON object containing:
- "compliant": true or false
- "deviations": list of strings (can be empty)
"""


@pytest.fixture
def simple_prompt() -> str:
    """Minimal prompt for smoke testing - expects 'PONG' response."""
    return SIMPLE_PROMPT


@pytest.fixture
def simple_plan_review_prompt() -> str:
    """Simple plan review prompt with expected JSON response structure."""
    return SIMPLE_PLAN_REVIEW_PROMPT


@pytest.fixture
def simple_fidelity_prompt() -> str:
    """Simple fidelity check prompt with expected JSON response structure."""
    return SIMPLE_FIDELITY_PROMPT


# =============================================================================
# Fixtures - File-based Fixtures
# =============================================================================


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to the fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def simple_plan_md(fixtures_dir: Path) -> str:
    """Load simple_plan.md fixture content."""
    return (fixtures_dir / "simple_plan.md").read_text()


@pytest.fixture
def simple_spec_json(fixtures_dir: Path) -> Dict[str, Any]:
    """Load simple_spec.json fixture as dict."""
    return json.loads((fixtures_dir / "simple_spec.json").read_text())


# =============================================================================
# Fixtures - Provider Helpers
# =============================================================================


@pytest.fixture
def provider_request_factory():
    """Factory for creating ProviderRequest objects.

    Note: temperature and max_tokens default to None to avoid issues
    with providers that don't support these parameters (e.g., codex, claude CLI).
    If max_tokens is needed, use 4096 as a reasonable default.
    """

    def _create(
        prompt: str,
        model: Optional[str] = None,
        timeout: float = 60.0,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ProviderRequest:
        return ProviderRequest(
            prompt=prompt,
            model=model,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    return _create


@pytest.fixture
def available_providers_list() -> list:
    """List of currently available providers."""
    providers = ["gemini", "codex", "claude", "cursor-agent", "opencode"]
    return [p for p in providers if detect_provider_availability(p)]


# =============================================================================
# Fixtures - Result Validation
# =============================================================================


@pytest.fixture
def validate_provider_result():
    """Validator for ProviderResult objects."""

    def _validate(result, expect_content: bool = True):
        from foundry_mcp.core.providers import ProviderResult, ProviderStatus

        assert isinstance(result, ProviderResult), f"Expected ProviderResult, got {type(result)}"
        assert result.status in ProviderStatus, f"Invalid status: {result.status}"

        if expect_content:
            assert result.status == ProviderStatus.SUCCESS, f"Expected SUCCESS, got {result.status}"
            assert result.content, "Expected non-empty content"
            assert isinstance(result.content, str), f"Content should be str, got {type(result.content)}"

        return result

    return _validate


@pytest.fixture
def validate_json_response():
    """Validator for JSON responses from providers."""

    def _validate(content: str, required_keys: Optional[list] = None) -> Dict[str, Any]:
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            # Try to extract JSON from markdown code blocks
            import re
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
            else:
                raise AssertionError(f"Response is not valid JSON: {e}\nContent: {content[:500]}")

        if required_keys:
            missing = set(required_keys) - set(data.keys())
            assert not missing, f"Response missing required keys: {missing}"

        return data

    return _validate
