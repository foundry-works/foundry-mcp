"""Tests for shared provider utility edge cases (Phase 4c).

Covers edge cases for shared utilities and the ERROR_CLASSIFIERS
registry pattern introduced in Phase 4b.
"""

import httpx

from foundry_mcp.core.research.providers.base import (
    SearchProviderError,
)
from foundry_mcp.core.research.providers.resilience import (
    ErrorType,
)
from foundry_mcp.core.research.providers.shared import (
    extract_domain,
    extract_status_code,
    parse_iso_date,
    parse_retry_after,
)
from tests.core.research.providers.conftest import (
    FACTORY_MAP,
)

# ===========================================================================
# extract_status_code edge cases
# ===========================================================================


class TestExtractStatusCode:
    """Test extract_status_code() — new helper for ERROR_CLASSIFIERS registry."""

    def test_standard_http_error_format(self):
        assert extract_status_code("HTTP 503 Service Unavailable") == 503

    def test_api_error_format(self):
        assert extract_status_code("API error 429: Rate limited") == 429

    def test_bare_status_code(self):
        assert extract_status_code("500 Internal Server Error") == 500

    def test_embedded_in_message(self):
        # Unanchored numbers like "Got 403" are no longer matched to avoid
        # false positives (e.g. "Found 200 results").  Use keyword-anchored
        # patterns like "error 403" or "HTTP 403" instead.
        assert extract_status_code("Got 403 from server") is None
        assert extract_status_code("HTTP error 403 from server") == 403

    def test_no_status_code(self):
        assert extract_status_code("Connection refused") is None

    def test_empty_string(self):
        assert extract_status_code("") is None

    def test_none_string(self):
        # Explicitly test with empty-like input
        assert extract_status_code("") is None

    def test_multiple_codes_returns_first(self):
        result = extract_status_code("Error 502 after retry, then 504")
        assert result == 502

    def test_non_http_numbers_ignored(self):
        # Numbers outside 100-599 range should not match
        assert extract_status_code("port 8080 is open") is None

    def test_boundary_100(self):
        assert extract_status_code("Status 100 Continue") == 100

    def test_boundary_599(self):
        assert extract_status_code("Error 599") == 599

    def test_boundary_600_ignored(self):
        assert extract_status_code("Error 600") is None


# ===========================================================================
# parse_retry_after edge cases
# ===========================================================================


class TestParseRetryAfterEdgeCases:
    """Additional edge cases for parse_retry_after beyond test_shared.py."""

    def _make_response(self, retry_after=None):
        from unittest.mock import MagicMock

        response = MagicMock(spec=httpx.Response)
        headers = {}
        if retry_after is not None:
            headers["Retry-After"] = retry_after
        response.headers = headers
        return response

    def test_zero_value(self):
        resp = self._make_response("0")
        assert parse_retry_after(resp) == 0.0

    def test_very_large_value(self):
        resp = self._make_response("86400")
        assert parse_retry_after(resp) == 86400.0

    def test_negative_value(self):
        resp = self._make_response("-1")
        assert parse_retry_after(resp) == -1.0

    def test_rfc7231_date_returns_none(self):
        """RFC 7231 date-based Retry-After is not supported."""
        resp = self._make_response("Sun, 06 Nov 1994 08:49:37 GMT")
        assert parse_retry_after(resp) is None

    def test_whitespace_only_header(self):
        resp = self._make_response("   ")
        assert parse_retry_after(resp) is None


# ===========================================================================
# extract_domain edge cases
# ===========================================================================


class TestExtractDomainEdgeCases:
    """Additional edge cases for extract_domain beyond test_shared.py."""

    def test_unicode_domain(self):
        result = extract_domain("https://münchen.de/path")
        assert result is not None

    def test_ip_address_url(self):
        result = extract_domain("http://192.168.1.1:8080/path")
        assert result == "192.168.1.1:8080"

    def test_scheme_only(self):
        result = extract_domain("https://")
        assert result is None

    def test_ftp_scheme(self):
        result = extract_domain("ftp://files.example.com/data")
        assert result == "files.example.com"

    def test_deeply_nested_path(self):
        result = extract_domain("https://api.v2.example.com/a/b/c/d?q=1")
        assert result == "api.v2.example.com"


# ===========================================================================
# parse_iso_date edge cases
# ===========================================================================


class TestParseIsoDateEdgeCases:
    """Additional edge cases for parse_iso_date beyond test_shared.py."""

    def test_timezone_aware_positive_offset(self):
        result = parse_iso_date("2024-01-15T10:30:00+05:30")
        assert result is not None
        assert result.tzinfo is not None

    def test_timezone_aware_negative_offset(self):
        result = parse_iso_date("2024-01-15T10:30:00-08:00")
        assert result is not None
        assert result.tzinfo is not None

    def test_year_only_with_extra_formats(self):
        result = parse_iso_date("2024", extra_formats=("%Y",))
        assert result is not None
        assert result.year == 2024

    def test_malformed_partial_date(self):
        assert parse_iso_date("2024-13") is None  # month 13 invalid

    def test_none_returns_none(self):
        assert parse_iso_date(None) is None


# ===========================================================================
# ERROR_CLASSIFIERS registry tests
# ===========================================================================


class TestErrorClassifiersRegistry:
    """Test that the ERROR_CLASSIFIERS registry works via base classify_error."""

    def test_google_has_403_classifier(self):
        provider = FACTORY_MAP["google"]()
        assert 403 in provider.ERROR_CLASSIFIERS
        assert provider.ERROR_CLASSIFIERS[403] == ErrorType.QUOTA_EXCEEDED

    def test_google_has_429_classifier(self):
        provider = FACTORY_MAP["google"]()
        assert 429 in provider.ERROR_CLASSIFIERS
        assert provider.ERROR_CLASSIFIERS[429] == ErrorType.RATE_LIMIT

    def test_perplexity_has_429_classifier(self):
        provider = FACTORY_MAP["perplexity"]()
        assert 429 in provider.ERROR_CLASSIFIERS
        assert provider.ERROR_CLASSIFIERS[429] == ErrorType.RATE_LIMIT

    def test_semantic_scholar_has_504_classifier(self):
        provider = FACTORY_MAP["semantic_scholar"]()
        assert 504 in provider.ERROR_CLASSIFIERS
        assert provider.ERROR_CLASSIFIERS[504] == ErrorType.SERVER_ERROR

    def test_tavily_uses_defaults(self):
        """Tavily has no custom classifiers — uses base defaults."""
        provider = FACTORY_MAP["tavily"]()
        assert provider.ERROR_CLASSIFIERS == {}

    def test_tavily_extract_has_no_registry(self):
        """TavilyExtract is standalone — doesn't inherit ERROR_CLASSIFIERS."""
        provider = FACTORY_MAP["tavily_extract"]()
        assert not hasattr(provider, "ERROR_CLASSIFIERS") or not provider.ERROR_CLASSIFIERS

    def test_registry_classifies_matching_search_provider_error(self):
        """ERROR_CLASSIFIERS registry matches status code in error message."""
        provider = FACTORY_MAP["perplexity"]()
        error = SearchProviderError(
            provider="perplexity",
            message="API error 429: Rate limited",
            retryable=True,
        )
        classification = provider.classify_error(error)
        assert classification.error_type == ErrorType.RATE_LIMIT
        assert classification.retryable is True
        assert classification.trips_breaker is False

    def test_registry_falls_through_for_unregistered_code(self):
        """Unregistered status codes fall through to generic classification."""
        provider = FACTORY_MAP["perplexity"]()
        error = SearchProviderError(
            provider="perplexity",
            message="API error 500: Internal Server Error",
            retryable=True,
        )
        classification = provider.classify_error(error)
        # 500 not in registry → falls through to classify_http_error → SERVER_ERROR
        assert classification.error_type == ErrorType.SERVER_ERROR

    def test_providers_without_registry_use_defaults(self):
        """Providers with empty ERROR_CLASSIFIERS still classify correctly."""
        provider = FACTORY_MAP["tavily"]()
        error = SearchProviderError(
            provider="tavily",
            message="API error 500: Server Error",
            retryable=True,
        )
        classification = provider.classify_error(error)
        assert classification.error_type == ErrorType.SERVER_ERROR
        assert classification.retryable is True

    def test_all_providers_classify_auth_error_consistently(self):
        """All providers agree on AuthenticationError classification."""
        from foundry_mcp.core.research.providers.base import AuthenticationError

        for name, factory in FACTORY_MAP.items():
            provider = factory()
            error = AuthenticationError(provider=name)
            classification = provider.classify_error(error)
            assert classification.error_type == ErrorType.AUTHENTICATION, (
                f"{name}: expected AUTHENTICATION, got {classification.error_type}"
            )
            assert classification.retryable is False

    def test_all_providers_classify_timeout_consistently(self):
        """All providers agree on timeout classification."""
        for name, factory in FACTORY_MAP.items():
            provider = factory()
            error = httpx.ReadTimeout("Connection timed out")
            classification = provider.classify_error(error)
            assert classification.error_type == ErrorType.TIMEOUT, (
                f"{name}: expected TIMEOUT, got {classification.error_type}"
            )
            assert classification.retryable is True
