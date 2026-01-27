"""Tests for PerplexitySearchProvider.

Tests cover:
1. Provider initialization (with/without API key)
2. Response parsing
3. Error handling (401, 429, 5xx)
4. Kwargs mapping (recency_filter, domain_filter)
"""

import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from foundry_mcp.core.research.models import SourceType
from foundry_mcp.core.research.providers.perplexity import (
    DEFAULT_RATE_LIMIT,
    DEFAULT_TIMEOUT,
    PERPLEXITY_API_BASE_URL,
    PerplexitySearchProvider,
)
from foundry_mcp.core.research.providers.base import (
    AuthenticationError,
    RateLimitError,
    SearchProviderError,
)


class TestPerplexitySearchProviderInit:
    """Tests for provider initialization."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        provider = PerplexitySearchProvider(api_key="pplx-test-key")
        assert provider._api_key == "pplx-test-key"
        assert provider._base_url == PERPLEXITY_API_BASE_URL
        assert provider._timeout == DEFAULT_TIMEOUT
        assert provider._max_retries == 3

    def test_init_with_env_var(self, monkeypatch):
        """Test initialization reads from PERPLEXITY_API_KEY env var."""
        monkeypatch.setenv("PERPLEXITY_API_KEY", "pplx-env-key")
        provider = PerplexitySearchProvider()
        assert provider._api_key == "pplx-env-key"

    def test_init_without_api_key_raises(self, monkeypatch):
        """Test initialization without API key raises ValueError."""
        monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Perplexity API key required"):
            PerplexitySearchProvider()

    def test_init_custom_settings(self):
        """Test initialization with custom settings."""
        provider = PerplexitySearchProvider(
            api_key="pplx-test",
            base_url="https://custom.api.com",
            timeout=60.0,
            max_retries=5,
        )
        assert provider._base_url == "https://custom.api.com"
        assert provider._timeout == 60.0
        assert provider._max_retries == 5


class TestPerplexitySearchProviderBasics:
    """Tests for basic provider methods."""

    @pytest.fixture
    def provider(self):
        """Create provider instance for tests."""
        return PerplexitySearchProvider(api_key="pplx-test-key")

    def test_get_provider_name(self, provider):
        """Test provider name is 'perplexity'."""
        assert provider.get_provider_name() == "perplexity"

    def test_rate_limit(self, provider):
        """Test rate limit property."""
        assert provider.rate_limit == DEFAULT_RATE_LIMIT


class TestPerplexitySearchProviderSearch:
    """Tests for search functionality."""

    @pytest.fixture
    def provider(self):
        """Create provider instance for tests."""
        return PerplexitySearchProvider(api_key="pplx-test-key")

    @pytest.fixture
    def mock_response_data(self):
        """Sample successful response data."""
        return {
            "results": [
                {
                    "title": "Test Result 1",
                    "url": "https://example.com/1",
                    "snippet": "This is a test snippet for result 1.",
                    "date": "2024-01-15T10:30:00Z",
                },
                {
                    "title": "Test Result 2",
                    "url": "https://example.com/2",
                    "snippet": "This is a test snippet for result 2.",
                    "last_updated": "2024-01-10",
                },
            ]
        }

    @pytest.mark.asyncio
    async def test_search_success(self, provider, mock_response_data):
        """Test successful search execution."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            sources = await provider.search("test query", max_results=10)

            assert len(sources) == 2
            assert sources[0].title == "Test Result 1"
            assert sources[0].url == "https://example.com/1"
            assert sources[0].snippet == "This is a test snippet for result 1."
            assert sources[0].source_type == SourceType.WEB

    @pytest.mark.asyncio
    async def test_search_with_recency_filter(self, provider, mock_response_data):
        """Test search with recency_filter parameter."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await provider.search("test query", recency_filter="week")

            # Check that recency_filter was included in payload
            call_args = mock_post.call_args
            payload = call_args.kwargs.get("json", call_args.args[1] if len(call_args.args) > 1 else {})
            assert payload.get("search_recency_filter") == "week"

    @pytest.mark.asyncio
    async def test_search_with_domain_filter(self, provider, mock_response_data):
        """Test search with domain_filter parameter."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await provider.search(
                "test query", domain_filter=["example.com", "test.org"]
            )

            # Check that domain_filter was included in payload
            call_args = mock_post.call_args
            payload = call_args.kwargs.get("json", call_args.args[1] if len(call_args.args) > 1 else {})
            assert payload.get("search_domain_filter") == ["example.com", "test.org"]

    @pytest.mark.asyncio
    async def test_search_with_country(self, provider, mock_response_data):
        """Test search with country parameter."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await provider.search("test query", country="US")

            # Check that country was included in payload
            call_args = mock_post.call_args
            payload = call_args.kwargs.get("json", call_args.args[1] if len(call_args.args) > 1 else {})
            assert payload.get("country") == "US"

    @pytest.mark.asyncio
    async def test_search_max_results_clamped(self, provider, mock_response_data):
        """Test that max_results is clamped to 20."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await provider.search("test query", max_results=50)

            # Check that max_results was clamped to 20
            call_args = mock_post.call_args
            payload = call_args.kwargs.get("json", call_args.args[1] if len(call_args.args) > 1 else {})
            assert payload.get("max_results") == 20

    @pytest.mark.asyncio
    async def test_search_with_sub_query_id(self, provider, mock_response_data):
        """Test that sub_query_id is passed to results."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            sources = await provider.search(
                "test query", sub_query_id="sq-123"
            )

            assert all(s.sub_query_id == "sq-123" for s in sources)


class TestPerplexitySearchProviderErrorHandling:
    """Tests for error handling."""

    @pytest.fixture
    def provider(self):
        """Create provider instance for tests."""
        return PerplexitySearchProvider(api_key="pplx-test-key", max_retries=1)

    @pytest.mark.asyncio
    async def test_authentication_error_401(self, provider):
        """Test 401 response raises AuthenticationError."""
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(AuthenticationError) as exc_info:
                await provider.search("test query")

            assert exc_info.value.provider == "perplexity"
            assert "Invalid API key" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_rate_limit_error_429(self, provider):
        """Test 429 response raises RateLimitError."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "30"}

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(RateLimitError) as exc_info:
                await provider.search("test query")

            assert exc_info.value.provider == "perplexity"
            assert exc_info.value.retry_after == 30.0

    @pytest.mark.asyncio
    async def test_server_error_5xx(self, provider):
        """Test 5xx response raises SearchProviderError with retryable=True."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.text = "Service Unavailable"
        mock_response.json.side_effect = Exception("No JSON")

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(SearchProviderError) as exc_info:
                await provider.search("test query")

            assert exc_info.value.provider == "perplexity"
            assert exc_info.value.retryable is True

    @pytest.mark.asyncio
    async def test_client_error_4xx(self, provider):
        """Test 4xx response (non-401, non-429) raises SearchProviderError."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_response.json.return_value = {"error": "Invalid query"}

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(SearchProviderError) as exc_info:
                await provider.search("test query")

            assert exc_info.value.provider == "perplexity"
            assert exc_info.value.retryable is False

    @pytest.mark.asyncio
    async def test_timeout_error(self, provider):
        """Test timeout raises SearchProviderError after retries."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.TimeoutException("Timeout")
            )

            with pytest.raises(SearchProviderError) as exc_info:
                await provider.search("test query")

            assert exc_info.value.provider == "perplexity"
            assert "failed after" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_request_error(self, provider):
        """Test request error raises SearchProviderError after retries."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.RequestError("Connection failed")
            )

            with pytest.raises(SearchProviderError) as exc_info:
                await provider.search("test query")

            assert exc_info.value.provider == "perplexity"


class TestPerplexitySearchProviderResponseParsing:
    """Tests for response parsing."""

    @pytest.fixture
    def provider(self):
        """Create provider instance for tests."""
        return PerplexitySearchProvider(api_key="pplx-test-key")

    def test_parse_response_empty_results(self, provider):
        """Test parsing response with empty results."""
        data = {"results": []}
        sources = provider._parse_response(data)
        assert sources == []

    def test_parse_response_missing_results_key(self, provider):
        """Test parsing response without results key."""
        data = {}
        sources = provider._parse_response(data)
        assert sources == []

    def test_parse_response_with_date(self, provider):
        """Test parsing response with date field."""
        data = {
            "results": [
                {
                    "title": "Test",
                    "url": "https://example.com",
                    "snippet": "Test snippet",
                    "date": "2024-01-15T10:30:00Z",
                }
            ]
        }
        sources = provider._parse_response(data)
        assert len(sources) == 1
        # Check metadata includes date
        assert sources[0].metadata.get("perplexity_date") == "2024-01-15T10:30:00Z"

    def test_parse_response_with_last_updated(self, provider):
        """Test parsing response with last_updated instead of date."""
        data = {
            "results": [
                {
                    "title": "Test",
                    "url": "https://example.com",
                    "snippet": "Test snippet",
                    "last_updated": "2024-01-10",
                }
            ]
        }
        sources = provider._parse_response(data)
        assert len(sources) == 1
        assert sources[0].metadata.get("perplexity_last_updated") == "2024-01-10"


class TestPerplexitySearchProviderDateParsing:
    """Tests for date parsing functionality."""

    @pytest.fixture
    def provider(self):
        """Create provider instance for tests."""
        return PerplexitySearchProvider(api_key="pplx-test-key")

    def test_parse_date_iso_format(self, provider):
        """Test parsing ISO format date."""
        result = provider._parse_date("2024-01-15T10:30:00Z")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_parse_date_simple_format(self, provider):
        """Test parsing simple date format."""
        result = provider._parse_date("2024-01-15")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_parse_date_none(self, provider):
        """Test parsing None returns None."""
        assert provider._parse_date(None) is None

    def test_parse_date_empty_string(self, provider):
        """Test parsing empty string returns None."""
        assert provider._parse_date("") is None

    def test_parse_date_invalid_format(self, provider):
        """Test parsing invalid format returns None."""
        assert provider._parse_date("not-a-date") is None


class TestPerplexitySearchProviderDomainExtraction:
    """Tests for domain extraction functionality."""

    @pytest.fixture
    def provider(self):
        """Create provider instance for tests."""
        return PerplexitySearchProvider(api_key="pplx-test-key")

    def test_extract_domain_simple(self, provider):
        """Test extracting domain from simple URL."""
        assert provider._extract_domain("https://example.com/page") == "example.com"

    def test_extract_domain_with_subdomain(self, provider):
        """Test extracting domain with subdomain."""
        assert provider._extract_domain("https://www.example.com/page") == "www.example.com"

    def test_extract_domain_with_port(self, provider):
        """Test extracting domain with port."""
        assert provider._extract_domain("https://example.com:8080/page") == "example.com:8080"

    def test_extract_domain_empty_url(self, provider):
        """Test extracting domain from empty URL returns None."""
        assert provider._extract_domain("") is None


class TestPerplexitySearchProviderHealthCheck:
    """Tests for health check functionality."""

    @pytest.fixture
    def provider(self):
        """Create provider instance for tests."""
        return PerplexitySearchProvider(api_key="pplx-test-key")

    @pytest.mark.asyncio
    async def test_health_check_success(self, provider):
        """Test successful health check."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": [{"title": "Test", "url": "http://test.com", "snippet": "test"}]}

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await provider.health_check()
            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_auth_failure(self, provider):
        """Test health check returns False on auth failure."""
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await provider.health_check()
            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_other_failure(self, provider):
        """Test health check returns False on other failures."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.RequestError("Connection failed")
            )

            result = await provider.health_check()
            assert result is False


class TestPerplexitySearchContextSize:
    """Tests for search_context_size parameter."""

    @pytest.fixture
    def provider(self):
        """Create provider instance for tests."""
        return PerplexitySearchProvider(api_key="pplx-test-key")

    @pytest.fixture
    def mock_response_data(self):
        """Sample successful response data."""
        return {
            "results": [
                {
                    "title": "Test Result",
                    "url": "https://example.com/1",
                    "snippet": "Test snippet",
                }
            ]
        }

    @pytest.mark.asyncio
    async def test_search_context_size_default_medium(self, provider, mock_response_data):
        """Test default search_context_size is 'medium'."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await provider.search("test query")

            call_args = mock_post.call_args
            payload = call_args.kwargs.get("json", call_args.args[1] if len(call_args.args) > 1 else {})
            assert payload.get("search_context_size") == "medium"

    @pytest.mark.asyncio
    async def test_search_context_size_low(self, provider, mock_response_data):
        """Test search_context_size='low' is passed correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await provider.search("test query", search_context_size="low")

            call_args = mock_post.call_args
            payload = call_args.kwargs.get("json", call_args.args[1] if len(call_args.args) > 1 else {})
            assert payload.get("search_context_size") == "low"

    @pytest.mark.asyncio
    async def test_search_context_size_medium(self, provider, mock_response_data):
        """Test search_context_size='medium' is passed correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await provider.search("test query", search_context_size="medium")

            call_args = mock_post.call_args
            payload = call_args.kwargs.get("json", call_args.args[1] if len(call_args.args) > 1 else {})
            assert payload.get("search_context_size") == "medium"

    @pytest.mark.asyncio
    async def test_search_context_size_high(self, provider, mock_response_data):
        """Test search_context_size='high' is passed correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await provider.search("test query", search_context_size="high")

            call_args = mock_post.call_args
            payload = call_args.kwargs.get("json", call_args.args[1] if len(call_args.args) > 1 else {})
            assert payload.get("search_context_size") == "high"

    @pytest.mark.asyncio
    async def test_search_context_size_invalid_raises_error(self, provider):
        """Test invalid search_context_size raises ValueError."""
        with pytest.raises(ValueError, match="Invalid search_context_size"):
            await provider.search("test query", search_context_size="invalid")


class TestPerplexityMaxTokens:
    """Tests for max_tokens and max_tokens_per_page parameters."""

    @pytest.fixture
    def provider(self):
        """Create provider instance for tests."""
        return PerplexitySearchProvider(api_key="pplx-test-key")

    @pytest.fixture
    def mock_response_data(self):
        """Sample successful response data."""
        return {
            "results": [
                {
                    "title": "Test Result",
                    "url": "https://example.com/1",
                    "snippet": "Test snippet",
                }
            ]
        }

    @pytest.mark.asyncio
    async def test_max_tokens_default(self, provider, mock_response_data):
        """Test default max_tokens is 50000."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await provider.search("test query")

            call_args = mock_post.call_args
            payload = call_args.kwargs.get("json", call_args.args[1] if len(call_args.args) > 1 else {})
            assert payload.get("max_tokens") == 50000

    @pytest.mark.asyncio
    async def test_max_tokens_custom(self, provider, mock_response_data):
        """Test custom max_tokens is passed correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await provider.search("test query", max_tokens=100000)

            call_args = mock_post.call_args
            payload = call_args.kwargs.get("json", call_args.args[1] if len(call_args.args) > 1 else {})
            assert payload.get("max_tokens") == 100000

    @pytest.mark.asyncio
    async def test_max_tokens_per_page_default(self, provider, mock_response_data):
        """Test default max_tokens_per_page is 2048."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await provider.search("test query")

            call_args = mock_post.call_args
            payload = call_args.kwargs.get("json", call_args.args[1] if len(call_args.args) > 1 else {})
            assert payload.get("max_tokens_per_page") == 2048

    @pytest.mark.asyncio
    async def test_max_tokens_per_page_custom(self, provider, mock_response_data):
        """Test custom max_tokens_per_page is passed correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await provider.search("test query", max_tokens_per_page=4096)

            call_args = mock_post.call_args
            payload = call_args.kwargs.get("json", call_args.args[1] if len(call_args.args) > 1 else {})
            assert payload.get("max_tokens_per_page") == 4096

    @pytest.mark.asyncio
    async def test_max_tokens_invalid_raises_error(self, provider):
        """Test invalid max_tokens (non-positive) raises ValueError."""
        with pytest.raises(ValueError, match="Invalid max_tokens"):
            await provider.search("test query", max_tokens=0)

    @pytest.mark.asyncio
    async def test_max_tokens_per_page_invalid_raises_error(self, provider):
        """Test invalid max_tokens_per_page (non-positive) raises ValueError."""
        with pytest.raises(ValueError, match="Invalid max_tokens_per_page"):
            await provider.search("test query", max_tokens_per_page=0)


class TestPerplexityDateFilters:
    """Tests for date filter parameters (search_after_date, search_before_date)."""

    @pytest.fixture
    def provider(self):
        """Create provider instance for tests."""
        return PerplexitySearchProvider(api_key="pplx-test-key")

    @pytest.fixture
    def mock_response_data(self):
        """Sample successful response data."""
        return {
            "results": [
                {
                    "title": "Test Result",
                    "url": "https://example.com/1",
                    "snippet": "Test snippet",
                }
            ]
        }

    @pytest.mark.asyncio
    async def test_search_after_date_valid(self, provider, mock_response_data):
        """Test valid search_after_date is passed correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await provider.search("test query", search_after_date="01/01/2024")

            call_args = mock_post.call_args
            payload = call_args.kwargs.get("json", call_args.args[1] if len(call_args.args) > 1 else {})
            assert payload.get("search_after_date") == "01/01/2024"

    @pytest.mark.asyncio
    async def test_search_before_date_valid(self, provider, mock_response_data):
        """Test valid search_before_date is passed correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await provider.search("test query", search_before_date="12/31/2024")

            call_args = mock_post.call_args
            payload = call_args.kwargs.get("json", call_args.args[1] if len(call_args.args) > 1 else {})
            assert payload.get("search_before_date") == "12/31/2024"

    @pytest.mark.asyncio
    async def test_search_after_date_invalid_format(self, provider):
        """Test invalid date format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid search_after_date"):
            await provider.search("test query", search_after_date="2024-01-01")

    @pytest.mark.asyncio
    async def test_search_before_date_invalid_format(self, provider):
        """Test invalid date format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid search_before_date"):
            await provider.search("test query", search_before_date="invalid-date")

    @pytest.mark.asyncio
    async def test_date_range_validation(self, provider):
        """Test that after_date must be before before_date."""
        with pytest.raises(ValueError, match="must be before"):
            await provider.search(
                "test query",
                search_after_date="12/31/2024",
                search_before_date="01/01/2024"
            )

    @pytest.mark.asyncio
    async def test_recency_filter_exclusivity_with_after_date(self, provider):
        """Test recency_filter cannot be combined with search_after_date."""
        with pytest.raises(ValueError, match="Cannot use recency_filter"):
            await provider.search(
                "test query",
                recency_filter="week",
                search_after_date="01/01/2024"
            )

    @pytest.mark.asyncio
    async def test_recency_filter_exclusivity_with_before_date(self, provider):
        """Test recency_filter cannot be combined with search_before_date."""
        with pytest.raises(ValueError, match="Cannot use recency_filter"):
            await provider.search(
                "test query",
                recency_filter="month",
                search_before_date="12/31/2024"
            )

    @pytest.mark.asyncio
    async def test_recency_filter_invalid_raises_error(self, provider):
        """Test invalid recency_filter raises ValueError."""
        with pytest.raises(ValueError, match="Invalid recency_filter"):
            await provider.search("test query", recency_filter="invalid")

    @pytest.mark.asyncio
    async def test_date_range_both_dates(self, provider, mock_response_data):
        """Test both date filters work together."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await provider.search(
                "test query",
                search_after_date="01/01/2024",
                search_before_date="12/31/2024"
            )

            call_args = mock_post.call_args
            payload = call_args.kwargs.get("json", call_args.args[1] if len(call_args.args) > 1 else {})
            assert payload.get("search_after_date") == "01/01/2024"
            assert payload.get("search_before_date") == "12/31/2024"


class TestPerplexityLastUpdatedFilters:
    """Tests for last_updated_after_filter and last_updated_before_filter parameters."""

    @pytest.fixture
    def provider(self):
        """Create provider instance for tests."""
        return PerplexitySearchProvider(api_key="pplx-test-key")

    @pytest.fixture
    def mock_response_data(self):
        """Sample successful response data."""
        return {
            "results": [
                {
                    "title": "Test Result",
                    "url": "https://example.com/1",
                    "snippet": "Test snippet",
                }
            ]
        }

    @pytest.mark.asyncio
    async def test_last_updated_after_filter_valid(self, provider, mock_response_data):
        """Test valid last_updated_after_filter is passed correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await provider.search("test query", last_updated_after_filter="01/01/2024")

            call_args = mock_post.call_args
            payload = call_args.kwargs.get("json", call_args.args[1] if len(call_args.args) > 1 else {})
            assert payload.get("last_updated_after_filter") == "01/01/2024"

    @pytest.mark.asyncio
    async def test_last_updated_before_filter_valid(self, provider, mock_response_data):
        """Test valid last_updated_before_filter is passed correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await provider.search("test query", last_updated_before_filter="12/31/2024")

            call_args = mock_post.call_args
            payload = call_args.kwargs.get("json", call_args.args[1] if len(call_args.args) > 1 else {})
            assert payload.get("last_updated_before_filter") == "12/31/2024"

    @pytest.mark.asyncio
    async def test_last_updated_after_filter_invalid_format(self, provider):
        """Test invalid last_updated_after_filter format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid last_updated_after_filter"):
            await provider.search("test query", last_updated_after_filter="2024-01-01")

    @pytest.mark.asyncio
    async def test_last_updated_before_filter_invalid_format(self, provider):
        """Test invalid last_updated_before_filter format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid last_updated_before_filter"):
            await provider.search("test query", last_updated_before_filter="invalid-date")

    @pytest.mark.asyncio
    async def test_last_updated_date_range_validation(self, provider):
        """Test that last_updated_after must be before last_updated_before."""
        with pytest.raises(ValueError, match="must be before"):
            await provider.search(
                "test query",
                last_updated_after_filter="12/31/2024",
                last_updated_before_filter="01/01/2024"
            )

    @pytest.mark.asyncio
    async def test_last_updated_both_filters(self, provider, mock_response_data):
        """Test both last_updated filters work together."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await provider.search(
                "test query",
                last_updated_after_filter="01/01/2024",
                last_updated_before_filter="12/31/2024"
            )

            call_args = mock_post.call_args
            payload = call_args.kwargs.get("json", call_args.args[1] if len(call_args.args) > 1 else {})
            assert payload.get("last_updated_after_filter") == "01/01/2024"
            assert payload.get("last_updated_before_filter") == "12/31/2024"

    @pytest.mark.asyncio
    async def test_last_updated_can_combine_with_recency_filter(self, provider, mock_response_data):
        """Test last_updated filters CAN be combined with recency_filter (different semantics)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            # This should NOT raise - last_updated filters have different semantics than date filters
            await provider.search(
                "test query",
                recency_filter="week",
                last_updated_after_filter="01/01/2024"
            )

            call_args = mock_post.call_args
            payload = call_args.kwargs.get("json", call_args.args[1] if len(call_args.args) > 1 else {})
            assert payload.get("search_recency_filter") == "week"
            assert payload.get("last_updated_after_filter") == "01/01/2024"
