"""Tests for SemanticScholarProvider.

Tests cover:
1. Provider initialization (with/without API key)
2. Parameter validation (publication_types, sort_by, sort_order)
3. Extended fields parsing (TLDR, venue, influential citations)
4. Parameter building (publicationTypes, sort)
5. Backward compatibility
"""

import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from foundry_mcp.core.research.models import SourceType
from foundry_mcp.core.research.providers.semantic_scholar import (
    DEFAULT_FIELDS,
    DEFAULT_RATE_LIMIT,
    DEFAULT_TIMEOUT,
    DEFAULT_SORT_BY,
    EXTENDED_FIELDS,
    PAPER_SEARCH_ENDPOINT,
    SEMANTIC_SCHOLAR_BASE_URL,
    VALID_PUBLICATION_TYPES,
    VALID_SORT_FIELDS,
    SemanticScholarProvider,
    _validate_search_params,
)
from foundry_mcp.core.research.providers.base import (
    AuthenticationError,
    RateLimitError,
    SearchProviderError,
)


class TestSemanticScholarProviderInit:
    """Tests for provider initialization."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        provider = SemanticScholarProvider(api_key="test-key")
        assert provider._api_key == "test-key"
        assert provider._base_url == SEMANTIC_SCHOLAR_BASE_URL
        assert provider._timeout == DEFAULT_TIMEOUT
        assert provider._max_retries == 3

    def test_init_with_env_var(self, monkeypatch):
        """Test initialization reads from SEMANTIC_SCHOLAR_API_KEY env var."""
        monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "env-test-key")
        provider = SemanticScholarProvider()
        assert provider._api_key == "env-test-key"

    def test_init_without_api_key_works(self, monkeypatch):
        """Test initialization without API key works (optional for Semantic Scholar)."""
        monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
        provider = SemanticScholarProvider()
        assert provider._api_key is None

    def test_init_custom_settings(self):
        """Test initialization with custom settings."""
        provider = SemanticScholarProvider(
            api_key="test-key",
            base_url="https://custom.api.com",
            timeout=60.0,
            max_retries=5,
        )
        assert provider._base_url == "https://custom.api.com"
        assert provider._timeout == 60.0
        assert provider._max_retries == 5


class TestSemanticScholarProviderBasics:
    """Tests for basic provider methods."""

    @pytest.fixture
    def provider(self):
        """Create provider instance for tests."""
        return SemanticScholarProvider(api_key="test-key")

    def test_get_provider_name(self, provider):
        """Test provider name is 'semantic_scholar'."""
        assert provider.get_provider_name() == "semantic_scholar"

    def test_rate_limit(self, provider):
        """Test rate limit property."""
        assert provider.rate_limit == DEFAULT_RATE_LIMIT


class TestValidateSearchParams:
    """Tests for parameter validation."""

    def test_valid_publication_types(self):
        """Test validation passes with valid publication types."""
        _validate_search_params(["JournalArticle", "Conference"], None, None)

    def test_invalid_publication_types(self):
        """Test validation rejects invalid publication types."""
        with pytest.raises(ValueError, match="Invalid publication_types"):
            _validate_search_params(["InvalidType"], None, None)

    def test_valid_sort_by(self):
        """Test validation passes with valid sort_by."""
        _validate_search_params(None, "citationCount", "desc")

    def test_invalid_sort_by(self):
        """Test validation rejects invalid sort_by."""
        with pytest.raises(ValueError, match="Invalid sort_by"):
            _validate_search_params(None, "invalidField", None)

    def test_valid_sort_order(self):
        """Test validation passes with valid sort_order."""
        _validate_search_params(None, "citationCount", "asc")
        _validate_search_params(None, "citationCount", "desc")

    def test_invalid_sort_order(self):
        """Test validation rejects invalid sort_order."""
        with pytest.raises(ValueError, match="Invalid sort_order"):
            _validate_search_params(None, "citationCount", "invalid")

    def test_sort_order_without_sort_by_allowed(self):
        """Test sort_order without sort_by passes validation."""
        _validate_search_params(None, None, "asc")


class TestExtendedFieldsParsing:
    """Tests for extended fields parsing."""

    @pytest.fixture
    def provider(self):
        """Create provider instance for tests."""
        return SemanticScholarProvider(api_key="test-key")

    @pytest.fixture
    def mock_response_with_tldr(self):
        """Sample response with TLDR and extended fields."""
        return {
            "total": 1,
            "data": [
                {
                    "paperId": "paper123",
                    "title": "Test Paper",
                    "abstract": "This is the full abstract text.",
                    "authors": [{"name": "John Doe"}],
                    "citationCount": 100,
                    "year": 2024,
                    "externalIds": {"DOI": "10.1234/test"},
                    "url": "https://semanticscholar.org/paper/123",
                    "openAccessPdf": {"url": "https://pdf.example.com/test"},
                    "publicationDate": "2024-01-15",
                    "tldr": {"text": "This is the TLDR summary."},
                    "venue": "NeurIPS",
                    "influentialCitationCount": 25,
                    "referenceCount": 50,
                    "fieldsOfStudy": ["Computer Science", "Machine Learning"],
                }
            ],
        }

    @pytest.fixture
    def mock_response_without_tldr(self):
        """Sample response without TLDR."""
        return {
            "total": 1,
            "data": [
                {
                    "paperId": "paper456",
                    "title": "Test Paper 2",
                    "abstract": "Short abstract.",
                    "authors": [{"name": "Jane Smith"}],
                    "citationCount": 50,
                    "year": 2023,
                    "externalIds": {},
                    "url": "https://semanticscholar.org/paper/456",
                    "openAccessPdf": None,
                    "publicationDate": None,
                    "tldr": None,
                    "venue": None,
                    "influentialCitationCount": None,
                    "referenceCount": None,
                    "fieldsOfStudy": None,
                }
            ],
        }

    def test_tldr_used_as_snippet(self, provider, mock_response_with_tldr):
        """Test TLDR is used as snippet when available."""
        sources = provider._parse_response(mock_response_with_tldr)
        assert sources[0].snippet == "This is the TLDR summary."

    def test_abstract_fallback_when_no_tldr(self, provider, mock_response_without_tldr):
        """Test abstract is used as snippet when no TLDR."""
        sources = provider._parse_response(mock_response_without_tldr)
        assert sources[0].snippet == "Short abstract."

    def test_extended_metadata_fields(self, provider, mock_response_with_tldr):
        """Test extended metadata fields are extracted."""
        sources = provider._parse_response(mock_response_with_tldr)
        metadata = sources[0].metadata
        assert metadata["venue"] == "NeurIPS"
        assert metadata["influential_citation_count"] == 25
        assert metadata["reference_count"] == 50
        assert metadata["fields_of_study"] == ["Computer Science", "Machine Learning"]
        assert metadata["tldr"] == "This is the TLDR summary."

    def test_none_metadata_handling(self, provider, mock_response_without_tldr):
        """Test None values in metadata are handled gracefully."""
        sources = provider._parse_response(mock_response_without_tldr)
        metadata = sources[0].metadata
        assert metadata["venue"] is None
        assert metadata["influential_citation_count"] is None
        assert metadata["reference_count"] is None
        assert metadata["fields_of_study"] is None
        assert metadata["tldr"] is None


class TestParameterBuilding:
    """Tests for search parameter building."""

    @pytest.fixture
    def provider(self):
        """Create provider instance for tests."""
        return SemanticScholarProvider(api_key="test-key")

    @pytest.fixture
    def mock_http_response(self):
        """Create mock HTTP response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [], "total": 0}
        return mock_response

    @pytest.mark.asyncio
    async def test_use_extended_fields_default(self, provider, mock_http_response):
        """Test extended fields are used by default."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_http_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await provider.search("test query")
            params = mock_client.get.call_args.kwargs["params"]
            assert params["fields"] == EXTENDED_FIELDS

    @pytest.mark.asyncio
    async def test_use_default_fields(self, provider, mock_http_response):
        """Test default fields can be used explicitly."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_http_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await provider.search("test query", use_extended_fields=False)
            params = mock_client.get.call_args.kwargs["params"]
            assert params["fields"] == DEFAULT_FIELDS

    @pytest.mark.asyncio
    async def test_publication_types_parameter(self, provider, mock_http_response):
        """Test publication types are comma-joined."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_http_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await provider.search("test", publication_types=["JournalArticle", "Conference"])
            params = mock_client.get.call_args.kwargs["params"]
            assert params["publicationTypes"] == "JournalArticle,Conference"

    @pytest.mark.asyncio
    async def test_sort_parameter(self, provider, mock_http_response):
        """Test sort parameter is correctly formatted."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_http_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await provider.search("test", sort_by="citationCount", sort_order="desc")
            params = mock_client.get.call_args.kwargs["params"]
            assert params["sort"] == "citationCount:desc"

    @pytest.mark.asyncio
    async def test_sort_default_order(self, provider, mock_http_response):
        """Test sort_order defaults to desc."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_http_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await provider.search("test", sort_by="publicationDate")
            params = mock_client.get.call_args.kwargs["params"]
            assert params["sort"] == "publicationDate:desc"

    @pytest.mark.asyncio
    async def test_sort_order_default_sort_by(self, provider, mock_http_response):
        """Test sort_by defaults when only sort_order is provided."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_http_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await provider.search("test", sort_order="asc")
            params = mock_client.get.call_args.kwargs["params"]
            assert params["sort"] == f"{DEFAULT_SORT_BY}:asc"


class TestBackwardCompatibility:
    """Tests for backward compatibility."""

    @pytest.fixture
    def provider(self):
        """Create provider instance for tests."""
        return SemanticScholarProvider(api_key="test-key")

    @pytest.fixture
    def mock_http_response(self):
        """Create mock HTTP response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [], "total": 0}
        return mock_response

    @pytest.mark.asyncio
    async def test_existing_kwargs_still_work(self, provider, mock_http_response):
        """Test existing kwargs (year, fields_of_study, etc.) still work."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_http_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await provider.search(
                "test query",
                year="2020-2024",
                fields_of_study=["Computer Science"],
                open_access_pdf=True,
                min_citation_count=10,
            )
            params = mock_client.get.call_args.kwargs["params"]
            assert params["year"] == "2020-2024"
            assert params["fieldsOfStudy"] == "Computer Science"
            assert params["openAccessPdf"] == ""
            assert params["minCitationCount"] == 10

    def test_endpoint_constant(self):
        """Test endpoint is /paper/search (not bulk)."""
        assert PAPER_SEARCH_ENDPOINT == "/paper/search"

    @pytest.mark.asyncio
    async def test_max_results_capped_at_100(self, provider):
        """Test max_results is capped at 100 for new endpoint."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [], "total": 0}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await provider.search("test query", max_results=250)
            params = mock_client.get.call_args.kwargs["params"]
            assert params["limit"] == 100
