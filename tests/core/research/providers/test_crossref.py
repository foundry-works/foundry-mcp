"""Tests for CrossrefProvider.

Tests cover:
1. Provider initialization (with/without mailto)
2. JATS XML tag stripping
3. Date-parts parsing
4. get_work with mocked response and metadata normalization
5. enrich_source fills missing fields without overwriting existing
6. Graceful handling of DOIs not in Crossref (404)
7. Error handling (429, 5xx)
8. Rate limiting configuration
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from foundry_mcp.core.research.models.sources import (
    ResearchSource,
    SourceQuality,
    SourceType,
)
from foundry_mcp.core.research.providers.crossref import (
    CROSSREF_BASE_URL,
    DEFAULT_RATE_LIMIT,
    DEFAULT_TIMEOUT,
    CrossrefProvider,
    _parse_date_parts,
    _strip_jats,
)


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------


MOCK_CROSSREF_WORK = {
    "DOI": "10.1234/test.2024",
    "title": ["Attention Is All You Need"],
    "author": [
        {"given": "Alice", "family": "Smith", "affiliation": []},
        {"given": "Bob", "family": "Jones", "affiliation": []},
    ],
    "published-print": {"date-parts": [[2024, 6, 15]]},
    "container-title": ["Nature Machine Intelligence"],
    "type": "journal-article",
    "is-referenced-by-count": 150,
    "abstract": "<jats:p>This paper proposes a new architecture.</jats:p>",
    "URL": "https://doi.org/10.1234/test.2024",
    "publisher": "Springer Nature",
    "ISSN": ["1234-5678"],
    "volume": "5",
    "issue": "3",
    "page": "100-115",
    "subject": ["Computer Science (miscellaneous)"],
    "license": [{"URL": "https://creativecommons.org/licenses/by/4.0/"}],
    "funder": [{"name": "NSF", "award": ["1234567"]}],
    "link": [
        {"URL": "https://example.com/pdf", "content-type": "application/pdf"},
    ],
}

MOCK_WORK_RESPONSE = {
    "status": "ok",
    "message-type": "work",
    "message": MOCK_CROSSREF_WORK,
}

MOCK_WORK_NO_ABSTRACT = {
    **MOCK_CROSSREF_WORK,
    "abstract": None,
}

MOCK_WORK_MINIMAL = {
    "DOI": "10.9999/minimal",
    "title": ["Minimal Work"],
    "type": "journal-article",
    "is-referenced-by-count": 0,
    "URL": "https://doi.org/10.9999/minimal",
}


# ---------------------------------------------------------------------------
# Provider initialization
# ---------------------------------------------------------------------------


class TestCrossrefProviderInit:
    """Tests for provider initialization."""

    def test_init_with_mailto(self):
        """Test initialization with explicit mailto."""
        provider = CrossrefProvider(mailto="user@example.com")
        assert provider._mailto == "user@example.com"
        assert provider._base_url == CROSSREF_BASE_URL
        assert provider._timeout == DEFAULT_TIMEOUT
        assert provider._max_retries == 3

    def test_init_with_env_var(self, monkeypatch):
        """Test initialization reads from CROSSREF_MAILTO env var."""
        monkeypatch.setenv("CROSSREF_MAILTO", "env@example.com")
        provider = CrossrefProvider()
        assert provider._mailto == "env@example.com"

    def test_init_without_mailto(self, monkeypatch):
        """Test initialization without mailto (None)."""
        monkeypatch.delenv("CROSSREF_MAILTO", raising=False)
        provider = CrossrefProvider()
        assert provider._mailto is None

    def test_init_custom_settings(self):
        """Test initialization with custom settings."""
        provider = CrossrefProvider(
            mailto="user@example.com",
            base_url="https://custom.api.com",
            timeout=60.0,
            max_retries=5,
        )
        assert provider._base_url == "https://custom.api.com"
        assert provider._timeout == 60.0
        assert provider._max_retries == 5

    def test_base_url_trailing_slash_stripped(self):
        """Test trailing slash is stripped from base URL."""
        provider = CrossrefProvider(mailto="x@y.com", base_url="https://api.crossref.org/")
        assert provider._base_url == "https://api.crossref.org"


class TestCrossrefProviderBasics:
    """Tests for basic provider methods."""

    @pytest.fixture
    def provider(self):
        return CrossrefProvider(mailto="test@example.com")

    def test_get_provider_name(self, provider):
        assert provider.get_provider_name() == "crossref"

    def test_rate_limit(self, provider):
        assert DEFAULT_RATE_LIMIT == 10.0
        assert provider.rate_limit == DEFAULT_RATE_LIMIT

    def test_resilience_config_loaded(self, provider):
        config = provider.resilience_config
        assert config.requests_per_second == 10.0
        assert config.burst_limit == 5


# ---------------------------------------------------------------------------
# JATS stripping
# ---------------------------------------------------------------------------


class TestJatsStripping:
    """Tests for _strip_jats helper."""

    def test_strip_simple_jats_p(self):
        """Test stripping <jats:p> tags."""
        text = "<jats:p>This is an abstract.</jats:p>"
        assert _strip_jats(text) == "This is an abstract."

    def test_strip_nested_jats(self):
        """Test stripping nested JATS tags."""
        text = "<jats:sec><jats:title>Introduction</jats:title><jats:p>Content here.</jats:p></jats:sec>"
        assert _strip_jats(text) == "IntroductionContent here."

    def test_strip_multiple_paragraphs(self):
        """Test stripping multiple JATS paragraphs."""
        text = "<jats:p>First paragraph.</jats:p><jats:p>Second paragraph.</jats:p>"
        assert _strip_jats(text) == "First paragraph.Second paragraph."

    def test_none_input(self):
        assert _strip_jats(None) is None

    def test_empty_string(self):
        assert _strip_jats("") is None

    def test_no_tags(self):
        """Test text without any tags passes through."""
        assert _strip_jats("Plain text abstract.") == "Plain text abstract."

    def test_only_tags(self):
        """Test text with only empty tags returns None."""
        assert _strip_jats("<jats:p></jats:p>") is None


# ---------------------------------------------------------------------------
# Date parts parsing
# ---------------------------------------------------------------------------


class TestDatePartsParsing:
    """Tests for _parse_date_parts helper."""

    def test_full_date(self):
        assert _parse_date_parts({"date-parts": [[2024, 6, 15]]}) == 2024

    def test_year_month_only(self):
        assert _parse_date_parts({"date-parts": [[2023, 3]]}) == 2023

    def test_year_only(self):
        assert _parse_date_parts({"date-parts": [[2022]]}) == 2022

    def test_none_input(self):
        assert _parse_date_parts(None) is None

    def test_empty_date_parts(self):
        assert _parse_date_parts({"date-parts": []}) is None

    def test_empty_inner_list(self):
        assert _parse_date_parts({"date-parts": [[]]}) is None

    def test_missing_date_parts_key(self):
        assert _parse_date_parts({}) is None


# ---------------------------------------------------------------------------
# get_work
# ---------------------------------------------------------------------------


class TestCrossrefGetWork:
    """Tests for get_work method."""

    @pytest.fixture
    def provider(self):
        return CrossrefProvider(mailto="test@example.com")

    @pytest.fixture
    def mock_http_response(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_WORK_RESPONSE
        return mock_response

    @pytest.mark.asyncio
    async def test_get_work_returns_normalized_metadata(self, provider, mock_http_response):
        """Test get_work returns correctly normalized metadata dict."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_http_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            work = await provider.get_work("10.1234/test.2024")

        assert work is not None
        assert work["title"] == "Attention Is All You Need"
        assert work["authors"] == "Alice Smith, Bob Jones"
        assert work["venue"] == "Nature Machine Intelligence"
        assert work["volume"] == "5"
        assert work["issue"] == "3"
        assert work["page"] == "100-115"
        assert work["publisher"] == "Springer Nature"
        assert work["type"] == "journal-article"
        assert work["year"] == 2024
        assert work["citation_count"] == 150
        assert work["doi"] == "10.1234/test.2024"
        assert work["abstract"] == "This paper proposes a new architecture."
        assert work["issn"] == ["1234-5678"]
        assert work["subjects"] == ["Computer Science (miscellaneous)"]
        assert work["license_url"] == "https://creativecommons.org/licenses/by/4.0/"
        assert work["pdf_url"] == "https://example.com/pdf"
        assert work["funder"] == ["NSF"]

    @pytest.mark.asyncio
    async def test_get_work_strips_doi_url_prefix(self, provider, mock_http_response):
        """Test DOI URL prefix is stripped before API call."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_http_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await provider.get_work("https://doi.org/10.1234/test.2024")

        call_url = mock_client.get.call_args.args[0]
        # DOI should be URL-encoded in the path
        assert "/works/10.1234%2Ftest.2024" in call_url
        assert "https://doi.org/" not in call_url

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

            work = await provider.get_work("10.9999/nonexistent")
            assert work is None

    @pytest.mark.asyncio
    async def test_get_work_mailto_in_user_agent(self, provider, mock_http_response):
        """Test mailto is included in User-Agent header."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_http_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await provider.get_work("10.1234/test.2024")

        headers = mock_client.get.call_args.kwargs["headers"]
        assert "mailto:test@example.com" in headers.get("User-Agent", "")

    @pytest.mark.asyncio
    async def test_get_work_minimal_response(self, provider):
        """Test get_work handles work with minimal fields."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok", "message": MOCK_WORK_MINIMAL}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            work = await provider.get_work("10.9999/minimal")

        assert work is not None
        assert work["title"] == "Minimal Work"
        assert work["authors"] == ""
        assert work["venue"] is None
        assert work["volume"] is None
        assert work["abstract"] is None


# ---------------------------------------------------------------------------
# enrich_source
# ---------------------------------------------------------------------------


class TestCrossrefEnrichSource:
    """Tests for enrich_source method."""

    @pytest.fixture
    def provider(self):
        return CrossrefProvider(mailto="test@example.com")

    @pytest.fixture
    def base_source(self):
        """A ResearchSource with minimal metadata (missing venue, volume, etc.)."""
        return ResearchSource(
            id="src-test001",
            url="https://doi.org/10.1234/test.2024",
            title="Attention Is All You Need",
            source_type=SourceType.ACADEMIC,
            quality=SourceQuality.UNKNOWN,
            snippet=None,
            content=None,
            metadata={
                "doi": "10.1234/test.2024",
                "authors": "Existing Authors",
            },
        )

    @pytest.fixture
    def mock_http_response(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_WORK_RESPONSE
        return mock_response

    @pytest.mark.asyncio
    async def test_enrich_fills_missing_fields(self, provider, base_source, mock_http_response):
        """Test enrich_source fills missing venue, volume, issue, pages."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_http_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            enriched = await provider.enrich_source(base_source)

        assert enriched.metadata["venue"] == "Nature Machine Intelligence"
        assert enriched.metadata["volume"] == "5"
        assert enriched.metadata["issue"] == "3"
        assert enriched.metadata["page"] == "100-115"
        assert enriched.metadata["publisher"] == "Springer Nature"

    @pytest.mark.asyncio
    async def test_enrich_does_not_overwrite_existing(self, provider, base_source, mock_http_response):
        """Test enrich_source never overwrites existing metadata values."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_http_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            enriched = await provider.enrich_source(base_source)

        # Authors were already set on the source — must not be overwritten
        assert enriched.metadata["authors"] == "Existing Authors"

    @pytest.mark.asyncio
    async def test_enrich_fills_missing_content(self, provider, base_source, mock_http_response):
        """Test enrich_source fills missing content from Crossref abstract."""
        assert base_source.content is None

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_http_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            enriched = await provider.enrich_source(base_source)

        assert enriched.content == "This paper proposes a new architecture."
        assert enriched.snippet is not None

    @pytest.mark.asyncio
    async def test_enrich_preserves_existing_content(self, provider, mock_http_response):
        """Test enrich_source does not overwrite existing content."""
        source = ResearchSource(
            id="src-test002",
            url="https://doi.org/10.1234/test.2024",
            title="Test",
            source_type=SourceType.ACADEMIC,
            content="Existing abstract that should not change.",
            metadata={"doi": "10.1234/test.2024"},
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_http_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            enriched = await provider.enrich_source(source)

        assert enriched.content == "Existing abstract that should not change."

    @pytest.mark.asyncio
    async def test_enrich_no_doi_returns_unchanged(self, provider):
        """Test enrich_source returns source unchanged if no DOI."""
        source = ResearchSource(
            id="src-nodoi",
            url="https://example.com",
            title="No DOI Source",
            source_type=SourceType.WEB,
            metadata={},
        )
        enriched = await provider.enrich_source(source)
        assert enriched is source  # Same object, not a copy

    @pytest.mark.asyncio
    async def test_enrich_doi_not_in_crossref(self, provider):
        """Test enrich_source returns source unchanged if DOI not found."""
        source = ResearchSource(
            id="src-notfound",
            url="https://doi.org/10.9999/unknown",
            title="Unknown DOI",
            source_type=SourceType.ACADEMIC,
            metadata={"doi": "10.9999/unknown"},
        )

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

            enriched = await provider.enrich_source(source)

        # Source should be returned unchanged
        assert enriched is source

    @pytest.mark.asyncio
    async def test_enrich_preserves_identity_fields(self, provider, base_source, mock_http_response):
        """Test enrich_source preserves id, url, title, source_type, quality."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_http_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            enriched = await provider.enrich_source(base_source)

        assert enriched.id == base_source.id
        assert enriched.url == base_source.url
        assert enriched.title == base_source.title
        assert enriched.source_type == base_source.source_type
        assert enriched.quality == base_source.quality


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestCrossrefErrorHandling:
    """Tests for error handling."""

    @pytest.fixture
    def provider(self):
        return CrossrefProvider(mailto="test@example.com")

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
                await provider.get_work("10.1234/test.2024")

    @pytest.mark.asyncio
    async def test_500_raises_provider_error(self, provider):
        """Test 5xx raises SearchProviderError."""
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
                await provider.get_work("10.1234/test.2024")

    def test_error_classifiers(self, provider):
        """Test ERROR_CLASSIFIERS maps 429 and 503 correctly."""
        from foundry_mcp.core.research.providers.resilience import ErrorType

        assert CrossrefProvider.ERROR_CLASSIFIERS[429] == ErrorType.RATE_LIMIT
        assert CrossrefProvider.ERROR_CLASSIFIERS[503] == ErrorType.SERVER_ERROR


# ---------------------------------------------------------------------------
# Response parsing edge cases
# ---------------------------------------------------------------------------


class TestCrossrefResponseParsing:
    """Tests for response parsing edge cases."""

    @pytest.fixture
    def provider(self):
        return CrossrefProvider(mailto="test@example.com")

    def test_work_without_abstract(self, provider):
        """Test normalization with no abstract."""
        result = provider._normalize_work(MOCK_WORK_NO_ABSTRACT)
        assert result["abstract"] is None

    def test_work_without_authors(self, provider):
        """Test normalization with no authors."""
        work = {**MOCK_CROSSREF_WORK, "author": []}
        result = provider._normalize_work(work)
        assert result["authors"] == ""

    def test_many_authors_truncated(self, provider):
        """Test more than 5 authors get 'et al.' suffix."""
        authors = [
            {"given": f"Author{i}", "family": f"Last{i}"}
            for i in range(8)
        ]
        work = {**MOCK_CROSSREF_WORK, "author": authors}
        result = provider._normalize_work(work)
        assert result["authors"].endswith("et al.")
        assert result["authors"].count(",") == 4

    def test_abstract_jats_stripping(self, provider):
        """Test JATS tags are stripped from abstracts."""
        result = provider._normalize_work(MOCK_CROSSREF_WORK)
        assert result["abstract"] == "This paper proposes a new architecture."
        assert "<jats:" not in (result["abstract"] or "")

    def test_publication_date_from_published_online_fallback(self, provider):
        """Test year extraction falls back to published-online."""
        work = {**MOCK_CROSSREF_WORK}
        del work["published-print"]
        work["published-online"] = {"date-parts": [[2023, 1]]}
        result = provider._normalize_work(work)
        assert result["year"] == 2023

    def test_publication_date_from_issued_fallback(self, provider):
        """Test year extraction falls back to issued."""
        work = {**MOCK_CROSSREF_WORK}
        del work["published-print"]
        work["issued"] = {"date-parts": [[2021]]}
        result = provider._normalize_work(work)
        assert result["year"] == 2021

    def test_minimal_work_normalization(self, provider):
        """Test normalization of work with only required fields."""
        result = provider._normalize_work(MOCK_WORK_MINIMAL)
        assert result["title"] == "Minimal Work"
        assert result["authors"] == ""
        assert result["venue"] is None
        assert result["volume"] is None
        assert result["issue"] is None
        assert result["page"] is None
        assert result["abstract"] is None
        assert result["issn"] is None
        assert result["subjects"] is None
        assert result["license_url"] is None
        assert result["pdf_url"] is None
        assert result["funder"] is None


# ---------------------------------------------------------------------------
# Rate limiting config
# ---------------------------------------------------------------------------


class TestCrossrefRateLimitConfig:
    """Tests for rate limiting configuration."""

    def test_resilience_config_defaults(self):
        """Test default resilience config for crossref."""
        from foundry_mcp.core.research.providers.resilience import get_provider_config

        config = get_provider_config("crossref")
        assert config.requests_per_second == 10.0
        assert config.burst_limit == 5
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.circuit_failure_threshold == 5
        assert config.circuit_recovery_timeout == 30.0
