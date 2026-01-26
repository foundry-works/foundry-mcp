"""Tests for TavilyExtractProvider.

Tests cover:
1. Provider initialization (with/without API key)
2. Extract method with various kwargs
3. Retry logic with mock 429 responses
4. Response parsing and ResearchSource mapping
5. Error handling (auth, rate limit, network)
6. URL validation and SSRF protection
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from foundry_mcp.core.research.models import ResearchSource, SourceType
from foundry_mcp.core.research.providers.tavily_extract import (
    DEFAULT_RATE_LIMIT,
    DEFAULT_TIMEOUT,
    TAVILY_API_BASE_URL,
    VALID_EXTRACT_DEPTHS,
    VALID_FORMATS,
    TavilyExtractProvider,
    UrlValidationError,
    _is_private_ip,
    _validate_extract_params,
    validate_extract_url,
)
from foundry_mcp.core.research.providers.base import (
    AuthenticationError,
    RateLimitError,
    SearchProviderError,
)


class TestTavilyExtractProviderInit:
    """Tests for provider initialization."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        provider = TavilyExtractProvider(api_key="tvly-test-key")
        assert provider._api_key == "tvly-test-key"
        assert provider._base_url == TAVILY_API_BASE_URL
        assert provider._timeout == DEFAULT_TIMEOUT
        assert provider._max_retries == 3

    def test_init_with_env_var(self, monkeypatch):
        """Test initialization reads from TAVILY_API_KEY env var."""
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-env-key")
        provider = TavilyExtractProvider()
        assert provider._api_key == "tvly-env-key"

    def test_init_without_api_key_raises(self, monkeypatch):
        """Test initialization without API key raises ValueError."""
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Tavily API key required"):
            TavilyExtractProvider()

    def test_init_custom_settings(self):
        """Test initialization with custom settings."""
        provider = TavilyExtractProvider(
            api_key="tvly-test",
            base_url="https://custom.api.com",
            timeout=60.0,
            max_retries=5,
        )
        assert provider._base_url == "https://custom.api.com"
        assert provider._timeout == 60.0
        assert provider._max_retries == 5


class TestTavilyExtractProviderBasics:
    """Tests for basic provider methods."""

    @pytest.fixture
    def provider(self):
        """Create provider instance for tests."""
        return TavilyExtractProvider(api_key="tvly-test-key")

    def test_get_provider_name(self, provider):
        """Test provider name is 'tavily_extract'."""
        assert provider.get_provider_name() == "tavily_extract"

    def test_rate_limit(self, provider):
        """Test rate limit property."""
        assert provider.rate_limit == DEFAULT_RATE_LIMIT


class TestExtractParamValidation:
    """Tests for extract parameter validation."""

    def test_validate_extract_depth_valid(self):
        """Test all valid extract depths are accepted."""
        for depth in VALID_EXTRACT_DEPTHS:
            _validate_extract_params(
                extract_depth=depth,
                format="markdown",
                chunks_per_source=None,
            )

    def test_validate_extract_depth_invalid(self):
        """Test invalid extract depth raises ValueError."""
        with pytest.raises(ValueError, match="Invalid extract_depth"):
            _validate_extract_params(
                extract_depth="invalid",
                format="markdown",
                chunks_per_source=None,
            )

    def test_validate_format_valid(self):
        """Test all valid formats are accepted."""
        for fmt in VALID_FORMATS:
            _validate_extract_params(
                extract_depth="basic",
                format=fmt,
                chunks_per_source=None,
            )

    def test_validate_format_invalid(self):
        """Test invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid format"):
            _validate_extract_params(
                extract_depth="basic",
                format="invalid",
                chunks_per_source=None,
            )

    def test_validate_chunks_per_source_valid_range(self):
        """Test valid chunks_per_source values (1-5) are accepted."""
        for chunks in [1, 2, 3, 4, 5]:
            _validate_extract_params(
                extract_depth="basic",
                format="markdown",
                chunks_per_source=chunks,
            )

    def test_validate_chunks_per_source_invalid_zero(self):
        """Test chunks_per_source=0 raises ValueError."""
        with pytest.raises(ValueError, match="Invalid chunks_per_source"):
            _validate_extract_params(
                extract_depth="basic",
                format="markdown",
                chunks_per_source=0,
            )

    def test_validate_chunks_per_source_invalid_over_limit(self):
        """Test chunks_per_source>5 raises ValueError."""
        with pytest.raises(ValueError, match="Invalid chunks_per_source"):
            _validate_extract_params(
                extract_depth="basic",
                format="markdown",
                chunks_per_source=6,
            )


class TestUrlValidation:
    """Tests for URL validation and SSRF protection."""

    def test_validate_url_https_valid(self):
        """Test valid HTTPS URLs pass validation."""
        validate_extract_url("https://example.com/page", resolve_dns=False)

    def test_validate_url_http_valid(self):
        """Test valid HTTP URLs pass validation."""
        validate_extract_url("http://example.com/page", resolve_dns=False)

    def test_validate_url_invalid_scheme_ftp(self):
        """Test FTP scheme is rejected."""
        with pytest.raises(UrlValidationError, match="Invalid scheme"):
            validate_extract_url("ftp://example.com/file", resolve_dns=False)

    def test_validate_url_invalid_scheme_file(self):
        """Test file:// scheme is rejected."""
        with pytest.raises(UrlValidationError, match="Invalid scheme"):
            validate_extract_url("file:///etc/passwd", resolve_dns=False)

    def test_validate_url_invalid_scheme_javascript(self):
        """Test javascript: scheme is rejected."""
        with pytest.raises(UrlValidationError, match="Invalid scheme"):
            validate_extract_url("javascript:alert(1)", resolve_dns=False)

    def test_validate_url_blocked_localhost(self):
        """Test localhost is blocked."""
        with pytest.raises(UrlValidationError, match="Blocked host"):
            validate_extract_url("http://localhost/admin", resolve_dns=False)

    def test_validate_url_blocked_127_0_0_1(self):
        """Test 127.0.0.1 is blocked."""
        with pytest.raises(UrlValidationError, match="Blocked"):
            validate_extract_url("http://127.0.0.1/admin", resolve_dns=False)

    def test_validate_url_blocked_0_0_0_0(self):
        """Test 0.0.0.0 is blocked."""
        with pytest.raises(UrlValidationError, match="Blocked"):
            validate_extract_url("http://0.0.0.0/admin", resolve_dns=False)

    def test_validate_url_blocked_local_domain(self):
        """Test .local domains are blocked."""
        with pytest.raises(UrlValidationError, match="Blocked internal domain"):
            validate_extract_url("http://myserver.local/admin", resolve_dns=False)

    def test_validate_url_blocked_internal_domain(self):
        """Test .internal domains are blocked."""
        with pytest.raises(UrlValidationError, match="Blocked internal domain"):
            validate_extract_url("http://app.internal/api", resolve_dns=False)

    def test_validate_url_too_long(self):
        """Test URL length limit is enforced."""
        long_url = "https://example.com/" + "a" * 2500
        with pytest.raises(UrlValidationError, match="URL too long"):
            validate_extract_url(long_url, resolve_dns=False)

    def test_validate_url_no_hostname(self):
        """Test URL without hostname is rejected."""
        with pytest.raises(UrlValidationError, match="No hostname"):
            validate_extract_url("https:///path/only", resolve_dns=False)


class TestPrivateIpDetection:
    """Tests for private IP detection."""

    def test_is_private_ip_10_range(self):
        """Test 10.x.x.x is detected as private."""
        assert _is_private_ip("10.0.0.1") is True
        assert _is_private_ip("10.255.255.255") is True

    def test_is_private_ip_172_range(self):
        """Test 172.16-31.x.x is detected as private."""
        assert _is_private_ip("172.16.0.1") is True
        assert _is_private_ip("172.31.255.255") is True

    def test_is_private_ip_192_168_range(self):
        """Test 192.168.x.x is detected as private."""
        assert _is_private_ip("192.168.0.1") is True
        assert _is_private_ip("192.168.255.255") is True

    def test_is_private_ip_loopback(self):
        """Test loopback addresses are detected as private."""
        assert _is_private_ip("127.0.0.1") is True
        assert _is_private_ip("127.255.255.255") is True
        assert _is_private_ip("::1") is True

    def test_is_private_ip_link_local(self):
        """Test link-local addresses are detected as private."""
        assert _is_private_ip("169.254.0.1") is True
        assert _is_private_ip("169.254.255.255") is True

    def test_is_private_ip_public(self):
        """Test public IPs are not flagged as private."""
        assert _is_private_ip("8.8.8.8") is False
        assert _is_private_ip("1.1.1.1") is False
        assert _is_private_ip("93.184.216.34") is False  # example.com

    def test_is_private_ip_invalid_returns_true(self):
        """Test invalid IP format returns True (safe default)."""
        assert _is_private_ip("not-an-ip") is True
        assert _is_private_ip("") is True


class TestExtractMethod:
    """Tests for extract method with various kwargs."""

    @pytest.fixture
    def provider(self):
        """Create provider instance for tests."""
        return TavilyExtractProvider(api_key="tvly-test-key")

    @pytest.mark.asyncio
    async def test_extract_with_default_params(self, provider):
        """Test extract with default parameters."""
        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"results": []}
            with patch("foundry_mcp.core.research.providers.tavily_extract.validate_extract_url"):
                await provider.extract(["https://example.com"])

            mock_exec.assert_called_once()
            payload = mock_exec.call_args[0][0]

            assert payload["api_key"] == "tvly-test-key"
            assert payload["urls"] == ["https://example.com"]
            assert payload["extract_depth"] == "basic"
            assert payload["include_images"] is False

    @pytest.mark.asyncio
    async def test_extract_with_advanced_depth(self, provider):
        """Test extract with advanced depth."""
        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"results": []}
            with patch("foundry_mcp.core.research.providers.tavily_extract.validate_extract_url"):
                await provider.extract(
                    ["https://example.com"],
                    extract_depth="advanced",
                )

            payload = mock_exec.call_args[0][0]
            assert payload["extract_depth"] == "advanced"

    @pytest.mark.asyncio
    async def test_extract_with_format(self, provider):
        """Test extract with format parameter."""
        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"results": []}
            with patch("foundry_mcp.core.research.providers.tavily_extract.validate_extract_url"):
                await provider.extract(
                    ["https://example.com"],
                    format="text",
                )

            payload = mock_exec.call_args[0][0]
            assert payload["format"] == "text"

    @pytest.mark.asyncio
    async def test_extract_with_include_images(self, provider):
        """Test extract with include_images parameter."""
        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"results": []}
            with patch("foundry_mcp.core.research.providers.tavily_extract.validate_extract_url"):
                await provider.extract(
                    ["https://example.com"],
                    include_images=True,
                )

            payload = mock_exec.call_args[0][0]
            assert payload["include_images"] is True

    @pytest.mark.asyncio
    async def test_extract_with_query(self, provider):
        """Test extract with query parameter for chunk reranking."""
        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"results": []}
            with patch("foundry_mcp.core.research.providers.tavily_extract.validate_extract_url"):
                await provider.extract(
                    ["https://example.com"],
                    query="important topic",
                )

            payload = mock_exec.call_args[0][0]
            assert payload["query"] == "important topic"

    @pytest.mark.asyncio
    async def test_extract_with_chunks_per_source(self, provider):
        """Test extract with chunks_per_source parameter."""
        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"results": []}
            with patch("foundry_mcp.core.research.providers.tavily_extract.validate_extract_url"):
                await provider.extract(
                    ["https://example.com"],
                    chunks_per_source=3,
                )

            payload = mock_exec.call_args[0][0]
            assert payload["chunks_per_source"] == 3

    @pytest.mark.asyncio
    async def test_extract_multiple_urls(self, provider):
        """Test extract with multiple URLs."""
        urls = [
            "https://example.com/page1",
            "https://example.com/page2",
            "https://example.org/article",
        ]
        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"results": []}
            with patch("foundry_mcp.core.research.providers.tavily_extract.validate_extract_url"):
                await provider.extract(urls)

            payload = mock_exec.call_args[0][0]
            assert payload["urls"] == urls

    @pytest.mark.asyncio
    async def test_extract_url_limit_enforced(self, provider):
        """Test extract enforces max 10 URLs per request."""
        urls = [f"https://example.com/page{i}" for i in range(15)]
        with pytest.raises(ValueError, match="Too many URLs.*Maximum is 10"):
            with patch("foundry_mcp.core.research.providers.tavily_extract.validate_extract_url"):
                await provider.extract(urls)


class TestResponseParsing:
    """Tests for response parsing and ResearchSource mapping."""

    @pytest.fixture
    def provider(self):
        """Create provider instance for tests."""
        return TavilyExtractProvider(api_key="tvly-test-key")

    @pytest.fixture
    def mock_response_data(self):
        """Sample successful response data."""
        return {
            "results": [
                {
                    "url": "https://example.com/page1",
                    "title": "Test Article 1",
                    "raw_content": "This is the extracted content for page 1.",
                    "images": ["https://example.com/img1.png"],
                    "favicon": "https://example.com/favicon.ico",
                },
                {
                    "url": "https://example.com/page2",
                    "title": "Test Article 2",
                    "raw_content": "This is the extracted content for page 2.",
                    "chunks": ["Chunk 1 content", "Chunk 2 content"],
                },
            ]
        }

    @pytest.mark.asyncio
    async def test_parse_response_returns_research_sources(self, provider, mock_response_data):
        """Test response parsing returns list of ResearchSource."""
        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_response_data
            with patch("foundry_mcp.core.research.providers.tavily_extract.validate_extract_url"):
                results = await provider.extract(["https://example.com/page1", "https://example.com/page2"])

            assert len(results) == 2
            assert all(isinstance(r, ResearchSource) for r in results)

    @pytest.mark.asyncio
    async def test_parse_response_maps_basic_fields(self, provider, mock_response_data):
        """Test response fields are correctly mapped to ResearchSource."""
        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_response_data
            with patch("foundry_mcp.core.research.providers.tavily_extract.validate_extract_url"):
                results = await provider.extract(["https://example.com/page1"])

            assert results[0].url == "https://example.com/page1"
            assert results[0].title == "Test Article 1"
            assert results[0].source_type == SourceType.WEB

    @pytest.mark.asyncio
    async def test_parse_response_snippet_from_content(self, provider, mock_response_data):
        """Test snippet is first 500 chars of content."""
        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_response_data
            with patch("foundry_mcp.core.research.providers.tavily_extract.validate_extract_url"):
                results = await provider.extract(["https://example.com/page1"])

            # Snippet should be first 500 chars of raw_content
            expected_snippet = "This is the extracted content for page 1."[:500]
            assert results[0].snippet == expected_snippet

    @pytest.mark.asyncio
    async def test_parse_response_chunks_joined(self, provider, mock_response_data):
        """Test chunks are joined for content."""
        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_response_data
            with patch("foundry_mcp.core.research.providers.tavily_extract.validate_extract_url"):
                results = await provider.extract(["https://example.com/page2"])

            # Content should be chunks joined
            assert "Chunk 1 content" in results[1].content
            assert "Chunk 2 content" in results[1].content

    @pytest.mark.asyncio
    async def test_parse_response_metadata_includes_required_fields(self, provider, mock_response_data):
        """Test metadata includes extract_depth, chunk_count, format, images, favicon."""
        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_response_data
            with patch("foundry_mcp.core.research.providers.tavily_extract.validate_extract_url"):
                results = await provider.extract(["https://example.com/page1"])

            metadata = results[0].metadata
            assert "extract_depth" in metadata
            assert "chunk_count" in metadata
            assert "format" in metadata
            assert "images" in metadata
            assert "favicon" in metadata

    @pytest.mark.asyncio
    async def test_parse_response_empty_results(self, provider):
        """Test empty results returns empty list."""
        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"results": []}
            with patch("foundry_mcp.core.research.providers.tavily_extract.validate_extract_url"):
                results = await provider.extract(["https://example.com"])

            assert results == []


class TestRetryLogic:
    """Tests for retry logic with rate limiting."""

    @pytest.fixture
    def provider(self):
        """Create provider instance for tests."""
        return TavilyExtractProvider(api_key="tvly-test-key", max_retries=3)

    @pytest.mark.asyncio
    async def test_retry_on_429_response(self, provider):
        """Test retry logic on 429 rate limit response."""
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        mock_response_429.headers = {"Retry-After": "1"}

        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.json.return_value = {"results": []}

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return mock_response_429
            return mock_response_200

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = mock_post
            with patch("foundry_mcp.core.research.providers.tavily_extract.validate_extract_url"):
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    results = await provider.extract(["https://example.com"])

            assert call_count == 3
            assert results == []

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises_rate_limit_error(self, provider):
        """Test RateLimitError raised when all retries exhausted."""
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        mock_response_429.headers = {"Retry-After": "60"}

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response_429
            )
            with patch("foundry_mcp.core.research.providers.tavily_extract.validate_extract_url"):
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    with pytest.raises(RateLimitError):
                        await provider.extract(["https://example.com"])


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.fixture
    def provider(self):
        """Create provider instance for tests."""
        return TavilyExtractProvider(api_key="tvly-test-key")

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
            with patch("foundry_mcp.core.research.providers.tavily_extract.validate_extract_url"):
                with pytest.raises(AuthenticationError):
                    await provider.extract(["https://example.com"])

    @pytest.mark.asyncio
    async def test_provider_error_on_500(self, provider):
        """Test 500 response raises SearchProviderError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.json.side_effect = Exception("Not JSON")

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            with patch("foundry_mcp.core.research.providers.tavily_extract.validate_extract_url"):
                with pytest.raises(SearchProviderError):
                    await provider.extract(["https://example.com"])

    @pytest.mark.asyncio
    async def test_url_validation_error_propagates(self, provider):
        """Test URL validation errors propagate correctly."""
        with pytest.raises(UrlValidationError, match="Blocked host"):
            await provider.extract(["http://localhost/admin"])

    @pytest.mark.asyncio
    async def test_empty_urls_raises_value_error(self, provider):
        """Test empty URL list raises ValueError."""
        with pytest.raises(ValueError, match="At least one URL"):
            await provider.extract([])


# =============================================================================
# Security-Focused Tests for SSRF Protection
# =============================================================================


class TestSSRFProtection:
    """Comprehensive security tests for SSRF (Server-Side Request Forgery) protection."""

    def test_blocked_ipv6_loopback(self):
        """Test IPv6 loopback ::1 is blocked."""
        with pytest.raises(UrlValidationError, match="Blocked"):
            validate_extract_url("http://[::1]/admin", resolve_dns=False)

    def test_blocked_ipv6_localhost_expanded(self):
        """Test expanded IPv6 localhost is blocked."""
        with pytest.raises(UrlValidationError, match="Blocked"):
            validate_extract_url("http://[0:0:0:0:0:0:0:1]/admin", resolve_dns=False)

    def test_blocked_private_ip_10_network(self):
        """Test 10.x.x.x private network is blocked."""
        with pytest.raises(UrlValidationError, match="Blocked private IP"):
            validate_extract_url("http://10.0.0.1/internal", resolve_dns=False)

    def test_blocked_private_ip_172_network(self):
        """Test 172.16.x.x private network is blocked."""
        with pytest.raises(UrlValidationError, match="Blocked private IP"):
            validate_extract_url("http://172.16.0.1/internal", resolve_dns=False)

    def test_blocked_private_ip_192_168_network(self):
        """Test 192.168.x.x private network is blocked."""
        with pytest.raises(UrlValidationError, match="Blocked private IP"):
            validate_extract_url("http://192.168.1.1/router", resolve_dns=False)

    def test_blocked_link_local_169_254(self):
        """Test 169.254.x.x link-local addresses are blocked."""
        with pytest.raises(UrlValidationError, match="Blocked private IP"):
            validate_extract_url("http://169.254.169.254/metadata", resolve_dns=False)

    def test_blocked_aws_metadata_endpoint(self):
        """Test AWS metadata endpoint (169.254.169.254) is blocked."""
        with pytest.raises(UrlValidationError, match="Blocked"):
            validate_extract_url("http://169.254.169.254/latest/meta-data/", resolve_dns=False)

    def test_blocked_localhost_subdomain(self):
        """Test .localhost subdomain is blocked."""
        with pytest.raises(UrlValidationError, match="Blocked internal domain"):
            validate_extract_url("http://evil.localhost/", resolve_dns=False)

    def test_blocked_data_scheme(self):
        """Test data: URI scheme is rejected."""
        with pytest.raises(UrlValidationError, match="Invalid scheme"):
            validate_extract_url("data:text/html,<script>alert(1)</script>", resolve_dns=False)

    def test_blocked_gopher_scheme(self):
        """Test gopher: scheme is rejected."""
        with pytest.raises(UrlValidationError, match="Invalid scheme"):
            validate_extract_url("gopher://localhost:25/", resolve_dns=False)

    def test_blocked_dict_scheme(self):
        """Test dict: scheme is rejected."""
        with pytest.raises(UrlValidationError, match="Invalid scheme"):
            validate_extract_url("dict://localhost:11211/", resolve_dns=False)

    def test_allowed_public_ip(self):
        """Test public IP addresses are allowed."""
        validate_extract_url("http://8.8.8.8/", resolve_dns=False)
        validate_extract_url("http://1.1.1.1/", resolve_dns=False)

    def test_allowed_normal_domain(self):
        """Test normal public domains are allowed."""
        validate_extract_url("https://example.com/page", resolve_dns=False)
        validate_extract_url("https://github.com/repo", resolve_dns=False)

    def test_url_with_credentials_parsed(self):
        """Test URL with embedded credentials still validates host."""
        # URL with credentials in userinfo section
        validate_extract_url("https://user:pass@example.com/page", resolve_dns=False)

    def test_url_with_port_validates_host(self):
        """Test URL with port number still validates host."""
        validate_extract_url("https://example.com:8443/page", resolve_dns=False)
        with pytest.raises(UrlValidationError, match="Blocked"):
            validate_extract_url("http://localhost:8080/admin", resolve_dns=False)

    def test_idn_domain_normalization(self):
        """Test IDN (internationalized domain name) is normalized."""
        # IDN domains should be normalized to punycode
        validate_extract_url("https://münchen.example.com/", resolve_dns=False)

    def test_url_path_traversal_still_validates(self):
        """Test URL with path traversal still validates the host."""
        # Path traversal doesn't affect host validation
        validate_extract_url("https://example.com/../../../etc/passwd", resolve_dns=False)
        with pytest.raises(UrlValidationError, match="Blocked"):
            validate_extract_url("http://localhost/../../../etc/passwd", resolve_dns=False)


class TestURLEdgeCases:
    """Test edge cases in URL validation."""

    def test_empty_url_rejected(self):
        """Test empty URL is rejected."""
        with pytest.raises(UrlValidationError):
            validate_extract_url("", resolve_dns=False)

    def test_whitespace_url_rejected(self):
        """Test whitespace-only URL is rejected."""
        with pytest.raises(UrlValidationError):
            validate_extract_url("   ", resolve_dns=False)

    def test_url_with_fragment(self):
        """Test URL with fragment is accepted."""
        validate_extract_url("https://example.com/page#section", resolve_dns=False)

    def test_url_with_query_string(self):
        """Test URL with query string is accepted."""
        validate_extract_url("https://example.com/search?q=test&page=1", resolve_dns=False)

    def test_url_with_unicode_path(self):
        """Test URL with unicode in path is accepted."""
        validate_extract_url("https://example.com/文档/page", resolve_dns=False)

    def test_url_maximum_length_boundary(self):
        """Test URL at exactly maximum length."""
        # Create URL at exactly 2048 chars (MAX_URL_LENGTH)
        base = "https://example.com/"
        padding = "a" * (2048 - len(base))
        url = base + padding
        assert len(url) == 2048
        validate_extract_url(url, resolve_dns=False)

    def test_url_one_over_maximum_length(self):
        """Test URL one character over maximum length is rejected."""
        base = "https://example.com/"
        padding = "a" * (2049 - len(base))
        url = base + padding
        assert len(url) == 2049
        with pytest.raises(UrlValidationError, match="URL too long"):
            validate_extract_url(url, resolve_dns=False)


# =============================================================================
# Partial Failure Handling Tests
# =============================================================================


class TestPartialFailureHandling:
    """Tests for partial failure handling in extract operations."""

    @pytest.fixture
    def provider(self):
        """Create provider instance for tests."""
        return TavilyExtractProvider(api_key="tvly-test-key")

    @pytest.mark.asyncio
    async def test_partial_success_returns_successful_sources(self, provider):
        """When some URLs succeed and some fail, successful sources are returned."""
        # Response with 2 successes and 1 failure (implicit - not in results)
        mock_response = {
            "results": [
                {
                    "url": "https://example.com/page1",
                    "title": "Page 1",
                    "raw_content": "Content 1",
                },
                {
                    "url": "https://example.com/page2",
                    "title": "Page 2",
                    "raw_content": "Content 2",
                },
            ]
        }

        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_response
            with patch("foundry_mcp.core.research.providers.tavily_extract.validate_extract_url"):
                results = await provider.extract([
                    "https://example.com/page1",
                    "https://example.com/page2",
                    "https://example.com/page3",  # This one "fails"
                ])

        # Should return 2 successful sources
        assert len(results) == 2
        assert results[0].url == "https://example.com/page1"
        assert results[1].url == "https://example.com/page2"

    @pytest.mark.asyncio
    async def test_all_urls_fail_returns_empty_list(self, provider):
        """When all URLs fail extraction, empty list is returned."""
        mock_response = {"results": []}

        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_response
            with patch("foundry_mcp.core.research.providers.tavily_extract.validate_extract_url"):
                results = await provider.extract(["https://example.com/page1"])

        assert results == []

    @pytest.mark.asyncio
    async def test_validation_failures_tracked_separately(self, provider):
        """URLs that fail validation don't get sent to API."""
        mock_response = {
            "results": [
                {
                    "url": "https://example.com/valid",
                    "title": "Valid Page",
                    "raw_content": "Content",
                },
            ]
        }

        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_response
            # One valid URL and one invalid (localhost)
            # The localhost URL will fail validation before API call
            with pytest.raises(UrlValidationError):
                await provider.extract([
                    "https://example.com/valid",
                    "http://localhost/invalid",
                ])

    @pytest.mark.asyncio
    async def test_mixed_validation_and_api_failures(self, provider):
        """Test handling when validation fails for some URLs."""
        # This tests the pre-validation step
        with pytest.raises(UrlValidationError, match="Blocked"):
            await provider.extract([
                "http://localhost/admin",  # Fails validation
            ])

    @pytest.mark.asyncio
    async def test_successful_extraction_preserves_all_fields(self, provider):
        """Successful extraction should preserve all response fields."""
        mock_response = {
            "results": [
                {
                    "url": "https://example.com/article",
                    "title": "Test Article",
                    "raw_content": "Full article content here...",
                    "chunks": ["Chunk 1", "Chunk 2"],
                    "images": ["https://example.com/img1.png"],
                    "favicon": "https://example.com/favicon.ico",
                },
            ]
        }

        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_response
            with patch("foundry_mcp.core.research.providers.tavily_extract.validate_extract_url"):
                results = await provider.extract(["https://example.com/article"])

        assert len(results) == 1
        source = results[0]
        assert source.url == "https://example.com/article"
        assert source.title == "Test Article"
        assert source.content is not None
        assert "Chunk 1" in source.content
        assert source.metadata["images"] == ["https://example.com/img1.png"]
        assert source.metadata["favicon"] == "https://example.com/favicon.ico"


class TestExtractHandlerPartialFailure:
    """Tests for _handle_extract partial failure response envelope."""

    def test_full_success_response_format(self):
        """Full success returns success=True with no warnings."""
        from foundry_mcp.tools.unified.research import _handle_extract

        with patch("foundry_mcp.tools.unified.research._get_config") as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.research.tavily_api_key = "tvly-test"
            mock_config.return_value = mock_cfg

            with patch("foundry_mcp.core.research.providers.tavily_extract.TavilyExtractProvider") as MockProvider:
                mock_provider = MagicMock()
                mock_provider.extract = AsyncMock(return_value=[
                    MagicMock(
                        url="https://example.com",
                        title="Test",
                        source_type=MagicMock(value="web"),
                        snippet="Test snippet",
                        content="Test content",
                        metadata={},
                    )
                ])
                MockProvider.return_value = mock_provider

                with patch("foundry_mcp.core.research.providers.tavily_extract.validate_extract_url"):
                    result = _handle_extract(urls=["https://example.com"])

        assert result["success"] is True
        assert result["error"] is None
        assert "sources" in result["data"]
        assert result["data"]["stats"]["succeeded"] == 1
        assert result["data"]["stats"]["failed"] == 0
        # No warnings for full success
        assert result["meta"].get("warnings") is None

    def test_total_failure_response_format(self):
        """Total failure returns success=False with error details."""
        from foundry_mcp.tools.unified.research import _handle_extract

        with patch("foundry_mcp.tools.unified.research._get_config") as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.research.tavily_api_key = "tvly-test"
            mock_config.return_value = mock_cfg

            with patch("foundry_mcp.core.research.providers.tavily_extract.TavilyExtractProvider") as MockProvider:
                mock_provider = MagicMock()
                mock_provider.extract = AsyncMock(return_value=[])  # No results
                MockProvider.return_value = mock_provider

                with patch("foundry_mcp.core.research.providers.tavily_extract.validate_extract_url"):
                    result = _handle_extract(urls=["https://example.com"])

        assert result["success"] is False
        assert result["error"] is not None
        assert "Extract failed" in result["error"]
        assert "failed_urls" in result["data"]["details"]
        assert "error_details" in result["data"]["details"]

    def test_partial_success_response_format(self):
        """Partial success returns success=True with warnings."""
        from foundry_mcp.tools.unified.research import _handle_extract

        with patch("foundry_mcp.tools.unified.research._get_config") as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.research.tavily_api_key = "tvly-test"
            mock_config.return_value = mock_cfg

            with patch("foundry_mcp.core.research.providers.tavily_extract.TavilyExtractProvider") as MockProvider:
                mock_provider = MagicMock()
                # Only 1 of 2 URLs succeeds
                mock_provider.extract = AsyncMock(return_value=[
                    MagicMock(
                        url="https://example.com/page1",
                        title="Test",
                        source_type=MagicMock(value="web"),
                        snippet="Test snippet",
                        content="Test content",
                        metadata={},
                    )
                ])
                MockProvider.return_value = mock_provider

                with patch("foundry_mcp.core.research.providers.tavily_extract.validate_extract_url"):
                    result = _handle_extract(urls=[
                        "https://example.com/page1",
                        "https://example.com/page2",  # This one "fails"
                    ])

        assert result["success"] is True
        assert result["error"] is None
        assert result["data"]["stats"]["succeeded"] == 1
        assert result["data"]["stats"]["failed"] == 1
        assert "failed_urls" in result["data"]
        assert "https://example.com/page2" in result["data"]["failed_urls"]
        # Partial success has warnings
        assert result["meta"].get("warnings") is not None
        assert len(result["meta"]["warnings"]) > 0

    def test_validation_failure_returns_error_with_details(self):
        """URL validation failure returns error with failed_urls and error_details."""
        from foundry_mcp.tools.unified.research import _handle_extract

        with patch("foundry_mcp.tools.unified.research._get_config") as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.research.tavily_api_key = "tvly-test"
            mock_config.return_value = mock_cfg

            # All URLs fail validation
            result = _handle_extract(urls=["http://localhost/admin"])

        assert result["success"] is False
        # failed_urls is in details for error responses
        assert "failed_urls" in result["data"]["details"]
        assert "error_details" in result["data"]["details"]
        assert result["data"]["details"]["failed_urls"] == ["http://localhost/admin"]

    def test_error_response_includes_error_code(self):
        """Error responses include error_code field."""
        from foundry_mcp.tools.unified.research import _handle_extract

        with patch("foundry_mcp.tools.unified.research._get_config") as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.research.tavily_api_key = "tvly-test"
            mock_config.return_value = mock_cfg

            result = _handle_extract(urls=["http://localhost/admin"])

        assert result["success"] is False
        assert "error_code" in result["data"]

    def test_response_always_has_meta_version(self):
        """All responses have meta.version='response-v2'."""
        from foundry_mcp.tools.unified.research import _handle_extract

        with patch("foundry_mcp.tools.unified.research._get_config") as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.research.tavily_api_key = "tvly-test"
            mock_config.return_value = mock_cfg

            # Success case
            with patch("foundry_mcp.core.research.providers.tavily_extract.TavilyExtractProvider") as MockProvider:
                mock_provider = MagicMock()
                mock_provider.extract = AsyncMock(return_value=[
                    MagicMock(
                        url="https://example.com",
                        title="Test",
                        source_type=MagicMock(value="web"),
                        snippet="Test",
                        content="Test",
                        metadata={},
                    )
                ])
                MockProvider.return_value = mock_provider

                with patch("foundry_mcp.core.research.providers.tavily_extract.validate_extract_url"):
                    success_result = _handle_extract(urls=["https://example.com"])

            # Error case
            error_result = _handle_extract(urls=["http://localhost/admin"])

        assert success_result["meta"]["version"] == "response-v2"
        assert error_result["meta"]["version"] == "response-v2"
