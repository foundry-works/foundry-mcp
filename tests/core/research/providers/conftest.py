"""Shared test fixtures for research provider tests.

Provides parametrized provider factories and mock response builders
used across characterization, shared, and individual provider test files.
"""

from unittest.mock import MagicMock

import httpx
import pytest

# ---------------------------------------------------------------------------
# Provider factories — create each provider with a test API key
# ---------------------------------------------------------------------------

PROVIDERS = ["tavily", "perplexity", "google", "semantic_scholar", "tavily_extract"]


def make_tavily(**kwargs):
    from foundry_mcp.core.research.providers.tavily import TavilySearchProvider

    return TavilySearchProvider(api_key="tvly-test-key", **kwargs)


def make_perplexity(**kwargs):
    from foundry_mcp.core.research.providers.perplexity import PerplexitySearchProvider

    return PerplexitySearchProvider(api_key="pplx-test-key", **kwargs)


def make_google(**kwargs):
    from foundry_mcp.core.research.providers.google import GoogleSearchProvider

    return GoogleSearchProvider(api_key="google-test-key", cx="cse-test", **kwargs)


def make_semantic_scholar(**kwargs):
    from foundry_mcp.core.research.providers.semantic_scholar import (
        SemanticScholarProvider,
    )

    return SemanticScholarProvider(api_key="s2-test-key", **kwargs)


def make_tavily_extract(**kwargs):
    from foundry_mcp.core.research.providers.tavily_extract import (
        TavilyExtractProvider,
    )

    return TavilyExtractProvider(api_key="tvly-test-key", **kwargs)


FACTORY_MAP = {
    "tavily": make_tavily,
    "perplexity": make_perplexity,
    "google": make_google,
    "semantic_scholar": make_semantic_scholar,
    "tavily_extract": make_tavily_extract,
}


# ---------------------------------------------------------------------------
# Parametrized fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=PROVIDERS)
def provider(request):
    """Parametrized fixture yielding each provider instance."""
    return FACTORY_MAP[request.param]()


@pytest.fixture(params=PROVIDERS)
def provider_name(request):
    """Parametrized fixture yielding provider name strings."""
    return request.param


# ---------------------------------------------------------------------------
# Mock response builder
# ---------------------------------------------------------------------------


def make_mock_response(
    *,
    status_code: int = 200,
    headers: dict | None = None,
    json_data: dict | None = None,
    text: str = "",
    raise_json: bool = False,
) -> MagicMock:
    """Build a mock httpx.Response for provider tests.

    Args:
        status_code: HTTP status code.
        headers: Response headers dict.
        json_data: JSON body (returned by response.json()).
        text: Plain text body.
        raise_json: If True, response.json() raises ValueError.

    Returns:
        MagicMock configured as an httpx.Response.
    """
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.headers = headers or {}
    response.text = text

    if raise_json:
        response.json.side_effect = ValueError("No JSON")
    elif json_data is not None:
        response.json.return_value = json_data
    else:
        response.json.return_value = {}

    return response
