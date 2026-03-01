"""Tests for OpenAlexProvider.

Tests cover:
1. Provider initialization (with/without API key)
2. Abstract reconstruction from inverted index
3. Search with mocked response and metadata mapping
4. get_work with DOI input format
5. get_citations and get_references with mocked responses
6. classify_text with mocked response
7. search_by_topic with mocked response
8. Error handling (401, 429, 5xx)
9. Empty results handling
10. Rate limiting configuration
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from foundry_mcp.core.research.providers.openalex import (
    DEFAULT_RATE_LIMIT,
    DEFAULT_TIMEOUT,
    MAX_PER_PAGE,
    OPENALEX_BASE_URL,
    OpenAlexProvider,
    _reconstruct_abstract,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


MOCK_WORK = {
    "id": "https://openalex.org/W1234567890",
    "ids": {
        "openalex": "https://openalex.org/W1234567890",
        "doi": "https://doi.org/10.1234/test.2024",
    },
    "doi": "https://doi.org/10.1234/test.2024",
    "title": "Attention Is All You Need",
    "display_name": "Attention Is All You Need",
    "publication_year": 2024,
    "publication_date": "2024-06-15",
    "type": "article",
    "cited_by_count": 150,
    "abstract_inverted_index": {
        "This": [0],
        "paper": [1, 5],
        "proposes": [2],
        "a": [3],
        "new": [4],
        "architecture.": [6],
    },
    "authorships": [
        {
            "author": {"display_name": "Alice Smith", "id": "A1"},
            "institutions": [{"display_name": "MIT"}],
        },
        {
            "author": {"display_name": "Bob Jones", "id": "A2"},
            "institutions": [{"display_name": "Stanford"}],
        },
    ],
    "topics": [
        {
            "id": "T12345",
            "display_name": "Transformer Models",
            "score": 0.95,
            "subfield": {"display_name": "Natural Language Processing"},
            "field": {"display_name": "Computer Science"},
            "domain": {"display_name": "Physical Sciences"},
        },
        {
            "id": "T67890",
            "display_name": "Deep Learning",
            "score": 0.80,
        },
    ],
    "open_access": {
        "is_oa": True,
        "oa_url": "https://arxiv.org/pdf/1706.03762",
    },
    "primary_location": {
        "source": {"display_name": "Nature Machine Intelligence"},
    },
    "referenced_works": [
        "https://openalex.org/W111",
        "https://openalex.org/W222",
    ],
    "related_works": [
        "https://openalex.org/W333",
    ],
    "referenced_works_count": 35,
    "awards": [
        {
            "funder": {"display_name": "NSF"},
            "award_id": "1234567",
        }
    ],
}


MOCK_SEARCH_RESPONSE = {
    "meta": {"count": 1, "page": 1, "per_page": 10},
    "results": [MOCK_WORK],
}

MOCK_EMPTY_RESPONSE = {
    "meta": {"count": 0, "page": 1, "per_page": 10},
    "results": [],
}


MOCK_CLASSIFY_RESPONSE = {
    "topics": [
        {
            "id": "T12345",
            "display_name": "Transformer Models",
            "score": 0.95,
            "subfield": {"display_name": "NLP"},
            "field": {"display_name": "Computer Science"},
            "domain": {"display_name": "Physical Sciences"},
        },
    ]
}


# ---------------------------------------------------------------------------
# Provider initialization
# ---------------------------------------------------------------------------


class TestOpenAlexProviderInit:
    """Tests for provider initialization."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        provider = OpenAlexProvider(api_key="test-key")
        assert provider._api_key == "test-key"
        assert provider._base_url == OPENALEX_BASE_URL
        assert provider._timeout == DEFAULT_TIMEOUT
        assert provider._max_retries == 3

    def test_init_with_env_var(self, monkeypatch):
        """Test initialization reads from OPENALEX_API_KEY env var."""
        monkeypatch.setenv("OPENALEX_API_KEY", "env-test-key")
        provider = OpenAlexProvider()
        assert provider._api_key == "env-test-key"

    def test_init_without_api_key(self, monkeypatch):
        """Test initialization without API key (None)."""
        monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
        provider = OpenAlexProvider()
        assert provider._api_key is None

    def test_init_custom_settings(self):
        """Test initialization with custom settings."""
        provider = OpenAlexProvider(
            api_key="test-key",
            base_url="https://custom.api.com",
            timeout=60.0,
            max_retries=5,
        )
        assert provider._base_url == "https://custom.api.com"
        assert provider._timeout == 60.0
        assert provider._max_retries == 5

    def test_base_url_trailing_slash_stripped(self):
        """Test trailing slash is stripped from base URL."""
        provider = OpenAlexProvider(api_key="k", base_url="https://api.openalex.org/")
        assert provider._base_url == "https://api.openalex.org"


class TestOpenAlexProviderBasics:
    """Tests for basic provider methods."""

    @pytest.fixture
    def provider(self):
        return OpenAlexProvider(api_key="test-key")

    def test_get_provider_name(self, provider):
        assert provider.get_provider_name() == "openalex"

    def test_rate_limit(self, provider):
        assert DEFAULT_RATE_LIMIT == 50.0
        assert provider.rate_limit == DEFAULT_RATE_LIMIT

    def test_resilience_config_loaded(self, provider):
        config = provider.resilience_config
        assert config.requests_per_second == 50.0
        assert config.burst_limit == 10


# ---------------------------------------------------------------------------
# Abstract reconstruction
# ---------------------------------------------------------------------------


class TestAbstractReconstruction:
    """Tests for _reconstruct_abstract helper."""

    def test_basic_reconstruction(self):
        """Test reconstructing a simple inverted index."""
        index = {"Hello": [0], "world": [1]}
        assert _reconstruct_abstract(index) == "Hello world"

    def test_repeated_words(self):
        """Test reconstruction with words appearing at multiple positions."""
        index = {"This": [0], "paper": [1, 5], "proposes": [2], "a": [3], "new": [4], "architecture.": [6]}
        result = _reconstruct_abstract(index)
        assert result == "This paper proposes a new paper architecture."

    def test_none_input(self):
        """Test None input returns None."""
        assert _reconstruct_abstract(None) is None

    def test_empty_dict(self):
        """Test empty dict returns None."""
        assert _reconstruct_abstract({}) is None

    def test_empty_positions(self):
        """Test dict with empty position lists returns None."""
        assert _reconstruct_abstract({"word": []}) is None

    def test_single_word(self):
        """Test single word reconstruction."""
        assert _reconstruct_abstract({"Hello": [0]}) == "Hello"

    def test_non_contiguous_positions(self):
        """Test with gaps in positions (shouldn't happen in practice)."""
        index = {"A": [0], "B": [2]}
        result = _reconstruct_abstract(index)
        # Position 1 is empty string, filtered by join
        assert result == "A B"


# ---------------------------------------------------------------------------
# Search method
# ---------------------------------------------------------------------------


class TestOpenAlexSearch:
    """Tests for search method."""

    @pytest.fixture
    def provider(self):
        return OpenAlexProvider(api_key="test-key")

    @pytest.fixture
    def mock_http_response(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_SEARCH_RESPONSE
        return mock_response

    @pytest.mark.asyncio
    async def test_search_returns_research_sources(self, provider, mock_http_response):
        """Test search returns correctly mapped ResearchSource objects."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_http_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            sources = await provider.search("transformer architecture", max_results=10)

        assert len(sources) == 1
        source = sources[0]
        assert source.title == "Attention Is All You Need"
        assert source.source_type.value == "academic"
        assert source.content == "This paper proposes a new paper architecture."

    @pytest.mark.asyncio
    async def test_search_metadata_mapping(self, provider, mock_http_response):
        """Test OpenAlex response fields are correctly mapped to metadata."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_http_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            sources = await provider.search("test")

        meta = sources[0].metadata
        assert meta["openalex_id"] == "https://openalex.org/W1234567890"
        assert meta["doi"] == "10.1234/test.2024"
        assert meta["authors"] == "Alice Smith, Bob Jones"
        assert meta["citation_count"] == 150
        assert meta["year"] == 2024
        assert meta["primary_topic"] == "Transformer Models"
        assert meta["is_oa"] is True
        assert meta["venue"] == "Nature Machine Intelligence"
        assert meta["type"] == "article"

    @pytest.mark.asyncio
    async def test_search_with_filters(self, provider, mock_http_response):
        """Test search passes filters correctly."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_http_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await provider.search(
                "test",
                filters={
                    "publication_year": "2020-2024",
                    "open_access.is_oa": True,
                    "cited_by_count": ">100",
                },
            )

            params = mock_client.get.call_args.kwargs["params"]
            assert "filter" in params
            filter_str = params["filter"]
            assert "publication_year:2020-2024" in filter_str
            assert "open_access.is_oa:true" in filter_str
            assert "cited_by_count:>100" in filter_str

    @pytest.mark.asyncio
    async def test_search_with_sort(self, provider, mock_http_response):
        """Test search passes sort parameter."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_http_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await provider.search("test", sort="cited_by_count:desc")

            params = mock_client.get.call_args.kwargs["params"]
            assert params["sort"] == "cited_by_count:desc"

    @pytest.mark.asyncio
    async def test_search_max_results_capped(self, provider, mock_http_response):
        """Test max_results is capped at MAX_PER_PAGE."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_http_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await provider.search("test", max_results=500)
            params = mock_client.get.call_args.kwargs["params"]
            assert params["per_page"] == MAX_PER_PAGE

    @pytest.mark.asyncio
    async def test_search_api_key_in_headers(self, provider, mock_http_response):
        """Test API key is passed via x-api-key header, not query params."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_http_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await provider.search("test")
            headers = mock_client.get.call_args.kwargs["headers"]
            params = mock_client.get.call_args.kwargs["params"]
            assert headers["x-api-key"] == "test-key"
            assert "api_key" not in params

    @pytest.mark.asyncio
    async def test_search_sub_query_id(self, provider, mock_http_response):
        """Test sub_query_id is passed through to ResearchSource."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_http_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            sources = await provider.search("test", sub_query_id="sq-123")
            assert sources[0].sub_query_id == "sq-123"


# ---------------------------------------------------------------------------
# Empty results
# ---------------------------------------------------------------------------


class TestOpenAlexEmptyResults:
    """Tests for empty results handling."""

    @pytest.fixture
    def provider(self):
        return OpenAlexProvider(api_key="test-key")

    @pytest.mark.asyncio
    async def test_search_empty_results(self, provider):
        """Test search returns empty list for no results."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_EMPTY_RESPONSE

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            sources = await provider.search("nonexistent query")
            assert sources == []


# ---------------------------------------------------------------------------
# get_work
# ---------------------------------------------------------------------------


class TestOpenAlexGetWork:
    """Tests for get_work method."""

    @pytest.fixture
    def provider(self):
        return OpenAlexProvider(api_key="test-key")

    @pytest.mark.asyncio
    async def test_get_work_by_openalex_id(self, provider):
        """Test get_work with OpenAlex ID."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_WORK

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            source = await provider.get_work("W1234567890")

        assert source is not None
        assert source.title == "Attention Is All You Need"

    @pytest.mark.asyncio
    async def test_get_work_by_doi(self, provider):
        """Test get_work with DOI input (auto-prefixes https://doi.org/)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_WORK

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            source = await provider.get_work("10.1234/test.2024")

        assert source is not None
        # Verify the DOI was prefixed in the URL
        call_args = mock_client.get.call_args
        assert "https://doi.org/10.1234/test.2024" in call_args.args[0]

    @pytest.mark.asyncio
    async def test_get_work_not_found(self, provider):
        """Test get_work returns None for 404."""
        from foundry_mcp.core.errors.search import SearchProviderError

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not found"
        mock_response.json.return_value = {"error": "Not found"}
        mock_response.headers = {}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            source = await provider.get_work("W9999999999")
            assert source is None


# ---------------------------------------------------------------------------
# get_citations
# ---------------------------------------------------------------------------


class TestOpenAlexGetCitations:
    """Tests for get_citations method."""

    @pytest.fixture
    def provider(self):
        return OpenAlexProvider(api_key="test-key")

    @pytest.mark.asyncio
    async def test_get_citations(self, provider):
        """Test get_citations returns citing works."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_SEARCH_RESPONSE

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            sources = await provider.get_citations("W1234567890", max_results=5)

        assert len(sources) == 1
        # Verify the filter was set correctly
        params = mock_client.get.call_args.kwargs["params"]
        assert "cites:W1234567890" in params["filter"]


# ---------------------------------------------------------------------------
# get_references
# ---------------------------------------------------------------------------


class TestOpenAlexGetReferences:
    """Tests for get_references method."""

    @pytest.fixture
    def provider(self):
        return OpenAlexProvider(api_key="test-key")

    @pytest.mark.asyncio
    async def test_get_references(self, provider):
        """Test get_references fetches referenced works."""
        # First call returns the work with referenced_works
        work_response = MagicMock()
        work_response.status_code = 200
        work_response.json.return_value = MOCK_WORK

        # Second call returns the referenced works
        refs_response = MagicMock()
        refs_response.status_code = 200
        refs_response.json.return_value = MOCK_SEARCH_RESPONSE

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=[work_response, refs_response])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            sources = await provider.get_references("W1234567890")

        assert len(sources) == 1

    @pytest.mark.asyncio
    async def test_get_references_empty(self, provider):
        """Test get_references with no referenced works."""
        work_data = {**MOCK_WORK, "referenced_works": []}
        work_response = MagicMock()
        work_response.status_code = 200
        work_response.json.return_value = work_data

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=work_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            sources = await provider.get_references("W1234567890")
            assert sources == []


# ---------------------------------------------------------------------------
# classify_text
# ---------------------------------------------------------------------------


class TestOpenAlexClassifyText:
    """Tests for classify_text method."""

    @pytest.fixture
    def provider(self):
        return OpenAlexProvider(api_key="test-key")

    @pytest.mark.asyncio
    async def test_classify_text(self, provider):
        """Test classify_text returns topic classifications."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_CLASSIFY_RESPONSE

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            topics = await provider.classify_text("transformer attention mechanism")

        assert len(topics) == 1
        assert topics[0]["display_name"] == "Transformer Models"
        assert topics[0]["score"] == 0.95
        assert topics[0]["field"] == "Computer Science"


# ---------------------------------------------------------------------------
# search_by_topic
# ---------------------------------------------------------------------------


class TestOpenAlexSearchByTopic:
    """Tests for search_by_topic method."""

    @pytest.fixture
    def provider(self):
        return OpenAlexProvider(api_key="test-key")

    @pytest.mark.asyncio
    async def test_search_by_topic(self, provider):
        """Test search_by_topic filters by topic ID."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_SEARCH_RESPONSE

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            sources = await provider.search_by_topic("T12345", max_results=5)

        assert len(sources) == 1
        params = mock_client.get.call_args.kwargs["params"]
        assert params["filter"] == "topics.id:T12345"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestOpenAlexErrorHandling:
    """Tests for error handling."""

    @pytest.fixture
    def provider(self):
        return OpenAlexProvider(api_key="test-key")

    @pytest.mark.asyncio
    async def test_401_raises_authentication_error(self, provider):
        """Test 401 raises AuthenticationError."""
        from foundry_mcp.core.errors.search import AuthenticationError

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_response.headers = {}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with pytest.raises(AuthenticationError):
                await provider.search("test")

    @pytest.mark.asyncio
    async def test_429_raises_rate_limit_error(self, provider):
        """Test 429 raises RateLimitError."""
        from foundry_mcp.core.errors.search import RateLimitError

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"
        mock_response.headers = {}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with pytest.raises(RateLimitError):
                await provider.search("test")

    @pytest.mark.asyncio
    async def test_500_raises_provider_error(self, provider):
        """Test 5xx raises SearchProviderError with retryable=True."""
        from foundry_mcp.core.errors.search import SearchProviderError

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.json.return_value = {"error": "Internal Server Error"}
        mock_response.headers = {}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with pytest.raises(SearchProviderError):
                await provider.search("test")

    def test_error_classifiers(self, provider):
        """Test ERROR_CLASSIFIERS maps 429 and 503 correctly."""
        from foundry_mcp.core.research.providers.resilience import ErrorType

        assert OpenAlexProvider.ERROR_CLASSIFIERS[429] == ErrorType.RATE_LIMIT
        assert OpenAlexProvider.ERROR_CLASSIFIERS[503] == ErrorType.SERVER_ERROR


# ---------------------------------------------------------------------------
# Response parsing edge cases
# ---------------------------------------------------------------------------


class TestOpenAlexResponseParsing:
    """Tests for response parsing edge cases."""

    @pytest.fixture
    def provider(self):
        return OpenAlexProvider(api_key="test-key")

    def test_work_without_abstract(self, provider):
        """Test parsing work with no abstract_inverted_index."""
        work = {**MOCK_WORK, "abstract_inverted_index": None}
        sources = provider._parse_works_response({"results": [work]})
        assert sources[0].content is None
        assert sources[0].snippet is None

    def test_work_without_topics(self, provider):
        """Test parsing work with no topics."""
        work = {**MOCK_WORK, "topics": []}
        sources = provider._parse_works_response({"results": [work]})
        assert sources[0].metadata["primary_topic"] is None
        assert sources[0].metadata["topics"] == []

    def test_work_without_doi(self, provider):
        """Test parsing work with no DOI uses OA URL."""
        work = {**MOCK_WORK, "doi": None}
        sources = provider._parse_works_response({"results": [work]})
        assert sources[0].url == "https://arxiv.org/pdf/1706.03762"

    def test_work_without_open_access(self, provider):
        """Test parsing work with no open access info."""
        work = {**MOCK_WORK, "doi": None, "open_access": None}
        sources = provider._parse_works_response({"results": [work]})
        assert sources[0].url == "https://openalex.org/W1234567890"

    def test_many_authors_truncated(self, provider):
        """Test more than 5 authors get 'et al.' suffix."""
        authors = [
            {"author": {"display_name": f"Author {i}"}}
            for i in range(8)
        ]
        work = {**MOCK_WORK, "authorships": authors}
        sources = provider._parse_works_response({"results": [work]})
        assert sources[0].metadata["authors"].endswith("et al.")
        assert sources[0].metadata["authors"].count(",") == 4  # 5 names, 4 commas

    def test_funders_extracted(self, provider):
        """Test awards/funders are extracted correctly."""
        sources = provider._parse_works_response({"results": [MOCK_WORK]})
        funders = sources[0].metadata["funders"]
        assert len(funders) == 1
        assert funders[0]["funder"] == "NSF"
        assert funders[0]["award_id"] == "1234567"

    def test_venue_from_primary_location(self, provider):
        """Test venue is extracted from primary_location."""
        sources = provider._parse_works_response({"results": [MOCK_WORK]})
        assert sources[0].metadata["venue"] == "Nature Machine Intelligence"

    def test_abstract_truncation_for_snippet(self, provider):
        """Test long abstract is truncated for snippet."""
        long_abstract_index = {f"word{i}": [i] for i in range(200)}
        work = {**MOCK_WORK, "abstract_inverted_index": long_abstract_index}
        sources = provider._parse_works_response({"results": [work]})
        if sources[0].snippet:
            assert len(sources[0].snippet) <= 504  # 500 + "..."


# ---------------------------------------------------------------------------
# Rate limiting config
# ---------------------------------------------------------------------------


class TestOpenAlexRateLimitConfig:
    """Tests for rate limiting configuration."""

    def test_resilience_config_defaults(self):
        """Test default resilience config for openalex."""
        from foundry_mcp.core.research.providers.resilience import get_provider_config

        config = get_provider_config("openalex")
        assert config.requests_per_second == 50.0
        assert config.burst_limit == 10
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.circuit_failure_threshold == 5
        assert config.circuit_recovery_timeout == 30.0
