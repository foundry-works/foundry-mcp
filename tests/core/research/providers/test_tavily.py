"""Tests for TavilySearchProvider.

Tests cover:
1. Provider initialization (with/without API key)
2. Parameter validation (search_depth, topic, days, country, chunks_per_source)
3. Payload building (parameters included when set)
4. Default values preserved
5. Invalid value rejection with clear error messages
6. Response parsing
7. Error handling (401, 429, 5xx)
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from foundry_mcp.core.research.models import ResearchSource, SourceType
from foundry_mcp.core.research.providers.tavily import (
    DEFAULT_RATE_LIMIT,
    DEFAULT_TIMEOUT,
    TAVILY_API_BASE_URL,
    VALID_SEARCH_DEPTHS,
    VALID_TOPICS,
    TavilySearchProvider,
    _normalize_include_raw_content,
    _validate_search_params,
)
from foundry_mcp.core.research.providers.base import (
    AuthenticationError,
    RateLimitError,
    SearchProviderError,
)


class TestTavilySearchProviderInit:
    """Tests for provider initialization."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        provider = TavilySearchProvider(api_key="tvly-test-key")
        assert provider._api_key == "tvly-test-key"
        assert provider._base_url == TAVILY_API_BASE_URL
        assert provider._timeout == DEFAULT_TIMEOUT
        assert provider._max_retries == 3

    def test_init_with_env_var(self, monkeypatch):
        """Test initialization reads from TAVILY_API_KEY env var."""
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-env-key")
        provider = TavilySearchProvider()
        assert provider._api_key == "tvly-env-key"

    def test_init_without_api_key_raises(self, monkeypatch):
        """Test initialization without API key raises ValueError."""
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Tavily API key required"):
            TavilySearchProvider()

    def test_init_custom_settings(self):
        """Test initialization with custom settings."""
        provider = TavilySearchProvider(
            api_key="tvly-test",
            base_url="https://custom.api.com",
            timeout=60.0,
            max_retries=5,
        )
        assert provider._base_url == "https://custom.api.com"
        assert provider._timeout == 60.0
        assert provider._max_retries == 5


class TestTavilySearchProviderBasics:
    """Tests for basic provider methods."""

    @pytest.fixture
    def provider(self):
        """Create provider instance for tests."""
        return TavilySearchProvider(api_key="tvly-test-key")

    def test_get_provider_name(self, provider):
        """Test provider name is 'tavily'."""
        assert provider.get_provider_name() == "tavily"

    def test_rate_limit(self, provider):
        """Test rate limit property."""
        assert provider.rate_limit == DEFAULT_RATE_LIMIT


class TestParameterValidation:
    """Tests for parameter validation functions."""

    def test_validate_search_depth_valid(self):
        """Test all valid search depths are accepted."""
        for depth in VALID_SEARCH_DEPTHS:
            _validate_search_params(
                search_depth=depth,
                topic="general",
                days=None,
                country=None,
                chunks_per_source=None,
            )

    def test_validate_search_depth_invalid(self):
        """Test invalid search depth raises ValueError."""
        with pytest.raises(ValueError, match="Invalid search_depth"):
            _validate_search_params(
                search_depth="invalid",
                topic="general",
                days=None,
                country=None,
                chunks_per_source=None,
            )

    def test_validate_topic_valid(self):
        """Test all valid topics are accepted."""
        for topic in VALID_TOPICS:
            _validate_search_params(
                search_depth="basic",
                topic=topic,
                days=None,
                country=None,
                chunks_per_source=None,
            )

    def test_validate_topic_invalid(self):
        """Test invalid topic raises ValueError."""
        with pytest.raises(ValueError, match="Invalid topic"):
            _validate_search_params(
                search_depth="basic",
                topic="invalid",
                days=None,
                country=None,
                chunks_per_source=None,
            )

    def test_validate_days_valid_range(self):
        """Test valid days values (1-365) are accepted."""
        for days in [1, 7, 30, 365]:
            _validate_search_params(
                search_depth="basic",
                topic="news",
                days=days,
                country=None,
                chunks_per_source=None,
            )

    def test_validate_days_invalid_zero(self):
        """Test days=0 raises ValueError."""
        with pytest.raises(ValueError, match="Invalid days"):
            _validate_search_params(
                search_depth="basic",
                topic="news",
                days=0,
                country=None,
                chunks_per_source=None,
            )

    def test_validate_days_invalid_negative(self):
        """Test negative days raises ValueError."""
        with pytest.raises(ValueError, match="Invalid days"):
            _validate_search_params(
                search_depth="basic",
                topic="news",
                days=-1,
                country=None,
                chunks_per_source=None,
            )

    def test_validate_days_invalid_over_limit(self):
        """Test days>365 raises ValueError."""
        with pytest.raises(ValueError, match="Invalid days"):
            _validate_search_params(
                search_depth="basic",
                topic="news",
                days=366,
                country=None,
                chunks_per_source=None,
            )

    def test_validate_country_valid(self):
        """Test valid country codes are accepted."""
        for country in ["US", "GB", "DE", "FR", "JP"]:
            _validate_search_params(
                search_depth="basic",
                topic="general",
                days=None,
                country=country,
                chunks_per_source=None,
            )

    def test_validate_country_invalid_lowercase(self):
        """Test lowercase country code raises ValueError."""
        with pytest.raises(ValueError, match="Invalid country"):
            _validate_search_params(
                search_depth="basic",
                topic="general",
                days=None,
                country="us",
                chunks_per_source=None,
            )

    def test_validate_country_invalid_length(self):
        """Test 3-letter country code raises ValueError."""
        with pytest.raises(ValueError, match="Invalid country"):
            _validate_search_params(
                search_depth="basic",
                topic="general",
                days=None,
                country="USA",
                chunks_per_source=None,
            )

    def test_validate_chunks_per_source_valid_range(self):
        """Test valid chunks_per_source values (1-5) are accepted."""
        for chunks in [1, 2, 3, 4, 5]:
            _validate_search_params(
                search_depth="advanced",
                topic="general",
                days=None,
                country=None,
                chunks_per_source=chunks,
            )

    def test_validate_chunks_per_source_invalid_zero(self):
        """Test chunks_per_source=0 raises ValueError."""
        with pytest.raises(ValueError, match="Invalid chunks_per_source"):
            _validate_search_params(
                search_depth="advanced",
                topic="general",
                days=None,
                country=None,
                chunks_per_source=0,
            )

    def test_validate_chunks_per_source_invalid_over_limit(self):
        """Test chunks_per_source>5 raises ValueError."""
        with pytest.raises(ValueError, match="Invalid chunks_per_source"):
            _validate_search_params(
                search_depth="advanced",
                topic="general",
                days=None,
                country=None,
                chunks_per_source=6,
            )


class TestNormalizeIncludeRawContent:
    """Tests for include_raw_content normalization."""

    def test_normalize_false(self):
        """Test False stays False."""
        assert _normalize_include_raw_content(False) is False

    def test_normalize_true_to_markdown(self):
        """Test True converts to 'markdown'."""
        assert _normalize_include_raw_content(True) == "markdown"

    def test_normalize_markdown_string(self):
        """Test 'markdown' stays 'markdown'."""
        assert _normalize_include_raw_content("markdown") == "markdown"

    def test_normalize_text_string(self):
        """Test 'text' stays 'text'."""
        assert _normalize_include_raw_content("text") == "text"

    def test_normalize_invalid_string(self):
        """Test invalid string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid include_raw_content"):
            _normalize_include_raw_content("invalid")


class TestPayloadBuilding:
    """Tests for search payload construction."""

    @pytest.fixture
    def provider(self):
        """Create provider instance for tests."""
        return TavilySearchProvider(api_key="tvly-test-key")

    @pytest.mark.asyncio
    async def test_payload_includes_required_params(self, provider):
        """Test payload includes all required parameters."""
        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"results": []}
            await provider.search("test query")

            mock_exec.assert_called_once()
            payload = mock_exec.call_args[0][0]

            # Required parameters
            assert payload["api_key"] == "tvly-test-key"
            assert payload["query"] == "test query"
            assert payload["max_results"] == 10
            assert payload["search_depth"] == "basic"
            assert payload["topic"] == "general"
            assert payload["include_answer"] is False
            assert payload["include_raw_content"] is False
            assert payload["include_images"] is False
            assert payload["include_favicon"] is False

    @pytest.mark.asyncio
    async def test_payload_excludes_optional_params_when_none(self, provider):
        """Test optional parameters not included when None."""
        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"results": []}
            await provider.search("test query")

            payload = mock_exec.call_args[0][0]

            # Optional parameters should not be in payload when None
            assert "include_domains" not in payload
            assert "exclude_domains" not in payload
            assert "days" not in payload
            assert "country" not in payload
            assert "chunks_per_source" not in payload
            assert "auto_parameters" not in payload

    @pytest.mark.asyncio
    async def test_payload_includes_days_when_set(self, provider):
        """Test days parameter included when set."""
        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"results": []}
            await provider.search("test query", topic="news", days=7)

            payload = mock_exec.call_args[0][0]
            assert payload["days"] == 7
            assert payload["topic"] == "news"

    @pytest.mark.asyncio
    async def test_payload_includes_country_when_set(self, provider):
        """Test country parameter included when set."""
        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"results": []}
            await provider.search("test query", country="US")

            payload = mock_exec.call_args[0][0]
            assert payload["country"] == "US"

    @pytest.mark.asyncio
    async def test_payload_includes_chunks_per_source_when_set(self, provider):
        """Test chunks_per_source parameter included when set."""
        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"results": []}
            await provider.search("test query", search_depth="advanced", chunks_per_source=3)

            payload = mock_exec.call_args[0][0]
            assert payload["chunks_per_source"] == 3

    @pytest.mark.asyncio
    async def test_payload_includes_domain_filters(self, provider):
        """Test domain filter parameters included when set."""
        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"results": []}
            await provider.search(
                "test query",
                include_domains=["arxiv.org", "github.com"],
                exclude_domains=["pinterest.com"],
            )

            payload = mock_exec.call_args[0][0]
            assert payload["include_domains"] == ["arxiv.org", "github.com"]
            assert payload["exclude_domains"] == ["pinterest.com"]

    @pytest.mark.asyncio
    async def test_payload_includes_auto_parameters_when_true(self, provider):
        """Test auto_parameters included when True."""
        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"results": []}
            await provider.search("test query", auto_parameters=True)

            payload = mock_exec.call_args[0][0]
            assert payload["auto_parameters"] is True

    @pytest.mark.asyncio
    async def test_payload_normalizes_include_raw_content_true(self, provider):
        """Test include_raw_content=True becomes 'markdown' in payload."""
        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"results": []}
            await provider.search("test query", include_raw_content=True)

            payload = mock_exec.call_args[0][0]
            assert payload["include_raw_content"] == "markdown"

    @pytest.mark.asyncio
    async def test_max_results_clamped_to_20(self, provider):
        """Test max_results is clamped to Tavily's limit of 20."""
        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"results": []}
            await provider.search("test query", max_results=100)

            payload = mock_exec.call_args[0][0]
            assert payload["max_results"] == 20


class TestDefaultValues:
    """Tests for default parameter values."""

    @pytest.fixture
    def provider(self):
        """Create provider instance for tests."""
        return TavilySearchProvider(api_key="tvly-test-key")

    @pytest.mark.asyncio
    async def test_default_search_depth(self, provider):
        """Test default search_depth is 'basic'."""
        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"results": []}
            await provider.search("test query")

            payload = mock_exec.call_args[0][0]
            assert payload["search_depth"] == "basic"

    @pytest.mark.asyncio
    async def test_default_topic(self, provider):
        """Test default topic is 'general'."""
        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"results": []}
            await provider.search("test query")

            payload = mock_exec.call_args[0][0]
            assert payload["topic"] == "general"

    @pytest.mark.asyncio
    async def test_default_max_results(self, provider):
        """Test default max_results is 10."""
        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"results": []}
            await provider.search("test query")

            payload = mock_exec.call_args[0][0]
            assert payload["max_results"] == 10

    @pytest.mark.asyncio
    async def test_default_include_flags(self, provider):
        """Test default include flags are False."""
        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"results": []}
            await provider.search("test query")

            payload = mock_exec.call_args[0][0]
            assert payload["include_answer"] is False
            assert payload["include_raw_content"] is False
            assert payload["include_images"] is False
            assert payload["include_favicon"] is False


class TestResponseParsing:
    """Tests for response parsing."""

    @pytest.fixture
    def provider(self):
        """Create provider instance for tests."""
        return TavilySearchProvider(api_key="tvly-test-key")

    @pytest.fixture
    def mock_response_data(self):
        """Sample successful response data."""
        return {
            "results": [
                {
                    "title": "Test Result 1",
                    "url": "https://example.com/1",
                    "content": "This is the content for result 1.",
                    "score": 0.95,
                },
                {
                    "title": "Test Result 2",
                    "url": "https://example.com/2",
                    "content": "This is the content for result 2.",
                    "score": 0.85,
                },
            ]
        }

    @pytest.mark.asyncio
    async def test_parse_response_returns_research_sources(self, provider, mock_response_data):
        """Test response parsing returns list of ResearchSource."""
        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_response_data
            results = await provider.search("test query")

            assert len(results) == 2
            assert all(isinstance(r, ResearchSource) for r in results)

    @pytest.mark.asyncio
    async def test_parse_response_maps_fields(self, provider, mock_response_data):
        """Test response fields are correctly mapped to ResearchSource."""
        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_response_data
            results = await provider.search("test query")

            assert results[0].title == "Test Result 1"
            assert results[0].url == "https://example.com/1"
            assert results[0].snippet == "This is the content for result 1."
            assert results[0].source_type == SourceType.WEB

    @pytest.mark.asyncio
    async def test_parse_response_empty_results(self, provider):
        """Test empty results returns empty list."""
        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"results": []}
            results = await provider.search("test query")

            assert results == []


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.fixture
    def provider(self):
        """Create provider instance for tests."""
        return TavilySearchProvider(api_key="tvly-test-key")

    @pytest.mark.asyncio
    async def test_authentication_error_on_401(self, provider):
        """Test 401 response raises AuthenticationError."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": "Invalid API key"}

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            with pytest.raises(AuthenticationError):
                await provider.search("test query")

    @pytest.mark.asyncio
    async def test_rate_limit_error_on_429(self, provider):
        """Test 429 response raises RateLimitError after retries."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "60"}

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            with pytest.raises(RateLimitError):
                await provider.search("test query")

    @pytest.mark.asyncio
    async def test_provider_error_on_5xx(self, provider):
        """Test 5xx response raises SearchProviderError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.json.side_effect = Exception("Not JSON")

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            with pytest.raises(SearchProviderError):
                await provider.search("test query")


# =============================================================================
# Contract Compatibility Tests
# =============================================================================


class TestTavilyAPIContractCompatibility:
    """Tests to verify compatibility with Tavily API response contracts.

    These tests use realistic fixtures matching the Tavily API documentation
    to ensure the provider correctly parses all response fields.
    """

    @pytest.fixture
    def provider(self):
        """Create provider instance for tests."""
        return TavilySearchProvider(api_key="tvly-test-key")

    @pytest.mark.asyncio
    async def test_basic_search_response_contract(self, provider):
        """Test parsing of basic Tavily search response matches API contract."""
        from tests.fixtures.tavily_responses import tavily_search_response_basic

        response = tavily_search_response_basic()

        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = response
            results = await provider.search("machine learning trends")

        # Verify correct number of results
        assert len(results) == len(response["results"])

        # Verify first result mapping
        first_result = results[0]
        assert first_result.title == response["results"][0]["title"]
        assert first_result.url == response["results"][0]["url"]
        assert first_result.snippet == response["results"][0]["content"]
        assert first_result.source_type == SourceType.WEB

    @pytest.mark.asyncio
    async def test_advanced_search_response_contract(self, provider):
        """Test parsing of advanced Tavily search response with raw_content."""
        from tests.fixtures.tavily_responses import tavily_search_response_advanced

        response = tavily_search_response_advanced()

        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = response
            results = await provider.search(
                "deep learning architectures",
                search_depth="advanced",
                include_raw_content=True,
            )

        assert len(results) == len(response["results"])

        # Advanced responses include raw_content
        first_result = results[0]
        assert first_result.title == response["results"][0]["title"]
        assert first_result.url == response["results"][0]["url"]
        # raw_content should be in content field when include_raw_content=True
        assert first_result.content == response["results"][0]["raw_content"]

    @pytest.mark.asyncio
    async def test_search_with_images_response_contract(self, provider):
        """Test parsing of search response with images."""
        from tests.fixtures.tavily_responses import tavily_search_response_with_images

        response = tavily_search_response_with_images()

        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = response
            results = await provider.search(
                "neural network diagrams",
                include_images=True,
            )

        # Should have results even with image focus
        assert len(results) >= 1
        assert results[0].source_type == SourceType.WEB

    @pytest.mark.asyncio
    async def test_news_search_response_contract(self, provider):
        """Test parsing of news-focused search response."""
        from tests.fixtures.tavily_responses import tavily_search_response_news

        response = tavily_search_response_news()

        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = response
            results = await provider.search(
                "AI regulations",
                topic="news",
                days=7,
            )

        assert len(results) == len(response["results"])
        # News results should have titles and URLs
        for result in results:
            assert result.title is not None
            assert result.url is not None

    @pytest.mark.asyncio
    async def test_empty_search_response_contract(self, provider):
        """Test parsing of empty search response."""
        from tests.fixtures.tavily_responses import tavily_search_response_empty

        response = tavily_search_response_empty()

        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = response
            results = await provider.search("very obscure query")

        assert results == []

    @pytest.mark.asyncio
    async def test_search_with_answer_response_contract(self, provider):
        """Test parsing of search response with AI-generated answer."""
        from tests.fixtures.tavily_responses import tavily_search_response_with_answer

        response = tavily_search_response_with_answer()

        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = response
            results = await provider.search(
                "What is the capital of France?",
                include_answer=True,
            )

        # Results should still be parsed correctly even when answer is present
        assert len(results) >= 1
        assert results[0].title is not None

    @pytest.mark.asyncio
    async def test_response_with_missing_optional_fields(self, provider):
        """Test parsing handles missing optional fields gracefully."""
        # Minimal response with only required fields
        minimal_response = {
            "results": [
                {
                    "title": "Minimal Result",
                    "url": "https://example.com/minimal",
                    "content": "Minimal content.",
                    # No score, no published_date, no raw_content
                }
            ]
        }

        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = minimal_response
            results = await provider.search("test")

        assert len(results) == 1
        assert results[0].title == "Minimal Result"
        assert results[0].url == "https://example.com/minimal"
        assert results[0].snippet == "Minimal content."

    @pytest.mark.asyncio
    async def test_response_with_extra_fields_ignored(self, provider):
        """Test parsing ignores unknown fields from API evolution."""
        response_with_future_fields = {
            "results": [
                {
                    "title": "Result",
                    "url": "https://example.com/page",
                    "content": "Content.",
                    "future_field_v2": "some new data",  # Unknown field
                    "another_new_field": {"nested": "data"},  # Unknown nested field
                }
            ],
            "new_api_metadata": "v2.5",  # Unknown top-level field
        }

        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = response_with_future_fields
            results = await provider.search("test")

        # Should still parse successfully
        assert len(results) == 1
        assert results[0].title == "Result"

    @pytest.mark.asyncio
    async def test_unicode_content_in_response(self, provider):
        """Test parsing handles unicode content correctly."""
        unicode_response = {
            "results": [
                {
                    "title": "中文标题 - Chinese Title",
                    "url": "https://example.com/文档",
                    "content": "日本語テスト 한국어 테스트 Ελληνικά 🚀",
                }
            ]
        }

        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = unicode_response
            results = await provider.search("test")

        assert len(results) == 1
        assert "中文" in results[0].title
        assert "🚀" in results[0].snippet

    @pytest.mark.asyncio
    async def test_very_long_content_in_response(self, provider):
        """Test parsing handles very long content without errors."""
        long_content = "A" * 100000  # 100KB of content
        response = {
            "results": [
                {
                    "title": "Long Content Article",
                    "url": "https://example.com/long",
                    "content": long_content,
                }
            ]
        }

        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = response
            results = await provider.search("test")

        assert len(results) == 1
        assert results[0].snippet is not None
        # Snippet should be truncated version of content
        assert len(results[0].snippet) <= len(long_content)