"""Tests for shared provider utilities module.

Tests cover all 8 utility functions plus secret redaction helpers.
Each acceptance criterion from the spec is explicitly verified.
"""

import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from foundry_mcp.core.research.providers.shared import (
    _redact_value,
    check_provider_health,
    classify_http_error,
    create_resilience_executor,
    extract_domain,
    extract_error_message,
    parse_iso_date,
    parse_retry_after,
    redact_headers,
    redact_secrets,
    resolve_provider_settings,
)


# ===========================================================================
# Redaction helpers
# ===========================================================================


class TestRedactValue:
    def test_short_value(self):
        assert _redact_value("abc") == "****"

    def test_exactly_four_chars(self):
        assert _redact_value("abcd") == "****"

    def test_longer_value(self):
        assert _redact_value("tvly-abc123") == "tvly****"

    def test_empty(self):
        assert _redact_value("") == "****"


class TestRedactSecrets:
    def test_api_key_in_text(self):
        text = "Failed with api_key=tvly-secret-value-12345"
        result = redact_secrets(text)
        assert "tvly-secret-value-12345" not in result
        assert "tvly****" in result

    def test_bearer_token(self):
        text = "Authorization: Bearer sk-long-secret-token-value"
        result = redact_secrets(text)
        assert "sk-long-secret-token-value" not in result

    def test_no_secrets(self):
        text = "Normal error message without secrets"
        assert redact_secrets(text) == text

    def test_empty_string(self):
        assert redact_secrets("") == ""

    def test_none_passthrough(self):
        # empty string returns empty
        assert redact_secrets("") == ""

    def test_token_equals(self):
        text = "token=mysecrettoken123 in request"
        result = redact_secrets(text)
        assert "mysecrettoken123" not in result

    def test_password_colon(self):
        text = "password: supersecretpassword"
        result = redact_secrets(text)
        assert "supersecretpassword" not in result


class TestRedactHeaders:
    def test_redacts_authorization(self):
        headers = {"Authorization": "Bearer sk-long-token-value", "Content-Type": "application/json"}
        result = redact_headers(headers)
        assert "sk-long-token-value" not in result["Authorization"]
        assert result["Content-Type"] == "application/json"

    def test_redacts_x_api_key(self):
        headers = {"X-API-Key": "tvly-secret-12345"}
        result = redact_headers(headers)
        assert "tvly-secret-12345" not in result["X-API-Key"]
        assert result["X-API-Key"] == "tvly****"

    def test_redacts_cookie(self):
        headers = {"Cookie": "session=abc12345"}
        result = redact_headers(headers)
        assert "abc12345" not in result["Cookie"]

    def test_case_insensitive(self):
        headers = {"AUTHORIZATION": "Bearer secret-token-value"}
        result = redact_headers(headers)
        assert "secret-token-value" not in result["AUTHORIZATION"]

    def test_preserves_non_sensitive(self):
        headers = {"Content-Type": "application/json", "Accept": "text/html"}
        result = redact_headers(headers)
        assert result == headers

    def test_returns_new_dict(self):
        headers = {"Authorization": "secret"}
        result = redact_headers(headers)
        assert result is not headers


# ===========================================================================
# parse_retry_after
# ===========================================================================


class TestParseRetryAfter:
    def _make_response(self, retry_after=None):
        response = MagicMock(spec=httpx.Response)
        headers = {}
        if retry_after is not None:
            headers["Retry-After"] = retry_after
        response.headers = headers
        return response

    def test_integer_value(self):
        resp = self._make_response("30")
        assert parse_retry_after(resp) == 30.0

    def test_float_value(self):
        resp = self._make_response("1.5")
        assert parse_retry_after(resp) == 1.5

    def test_missing_header(self):
        resp = self._make_response()
        assert parse_retry_after(resp) is None

    def test_invalid_value(self):
        resp = self._make_response("not-a-number")
        assert parse_retry_after(resp) is None

    def test_empty_string(self):
        resp = self._make_response("")
        assert parse_retry_after(resp) is None


# ===========================================================================
# extract_error_message
# ===========================================================================


class TestExtractErrorMessage:
    def _make_response(self, json_data=None, text="", raise_json=False):
        response = MagicMock(spec=httpx.Response)
        if raise_json:
            response.json.side_effect = ValueError("No JSON")
        else:
            response.json.return_value = json_data
        response.text = text
        return response

    def test_error_field_string(self):
        resp = self._make_response({"error": "Something went wrong"})
        assert extract_error_message(resp) == "Something went wrong"

    def test_message_field(self):
        resp = self._make_response({"message": "Rate limit exceeded"})
        assert extract_error_message(resp) == "Rate limit exceeded"

    def test_error_field_takes_priority(self):
        resp = self._make_response({"error": "Primary", "message": "Secondary"})
        assert extract_error_message(resp) == "Primary"

    def test_nested_error_dict(self):
        resp = self._make_response({"error": {"code": 403, "message": "Quota exceeded"}})
        assert extract_error_message(resp) == "Quota exceeded"

    def test_fallback_to_text(self):
        resp = self._make_response(json_data={}, text="Raw error text")
        assert extract_error_message(resp) == "Raw error text"

    def test_json_parse_failure(self):
        resp = self._make_response(raise_json=True, text="Server error")
        assert extract_error_message(resp) == "Server error"

    def test_json_parse_failure_no_text(self):
        resp = self._make_response(raise_json=True, text="")
        assert extract_error_message(resp) == "Unknown error"

    def test_text_truncated(self):
        resp = self._make_response(raise_json=True, text="x" * 500)
        result = extract_error_message(resp)
        assert len(result) <= 200

    def test_provider_format_used(self):
        def google_format(data):
            error = data.get("error", {})
            if isinstance(error, dict):
                return error.get("message", "")
            return ""

        resp = self._make_response({"error": {"code": 403, "message": "Daily Limit Exceeded"}})
        assert extract_error_message(resp, provider_format=google_format) == "Daily Limit Exceeded"

    def test_provider_format_returns_empty_falls_through(self):
        resp = self._make_response({"error": "Fallback error"})
        result = extract_error_message(resp, provider_format=lambda d: "")
        assert result == "Fallback error"

    def test_redacts_api_key_in_error(self):
        resp = self._make_response({"error": "Invalid api_key=tvly-secret-real-key-12345"})
        result = extract_error_message(resp)
        assert "tvly-secret-real-key-12345" not in result
        assert "tvly****" in result


# ===========================================================================
# parse_iso_date
# ===========================================================================


class TestParseIsoDate:
    def test_iso_format(self):
        result = parse_iso_date("2024-01-15T10:30:00")
        assert result == datetime(2024, 1, 15, 10, 30, 0)

    def test_iso_with_z(self):
        result = parse_iso_date("2024-01-15T10:30:00Z")
        assert result is not None
        assert result.tzinfo is not None

    def test_date_only(self):
        result = parse_iso_date("2024-01-15")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_slash_format(self):
        result = parse_iso_date("2024/01/15")
        assert result is not None
        assert result.year == 2024

    def test_day_first_dash(self):
        result = parse_iso_date("15-01-2024")
        assert result is not None
        assert result.day == 15

    def test_day_first_slash(self):
        result = parse_iso_date("15/01/2024")
        assert result is not None
        assert result.day == 15

    def test_full_month_name(self):
        result = parse_iso_date("January 15, 2024")
        assert result is not None
        assert result.month == 1

    def test_abbreviated_month(self):
        result = parse_iso_date("Jan 15, 2024")
        assert result is not None
        assert result.month == 1

    def test_none_input(self):
        assert parse_iso_date(None) is None

    def test_empty_string(self):
        assert parse_iso_date("") is None

    def test_unparseable(self):
        assert parse_iso_date("not-a-date") is None

    def test_extra_formats(self):
        result = parse_iso_date("15.01.2024", extra_formats=("%d.%m.%Y",))
        assert result is not None
        assert result.day == 15


# ===========================================================================
# extract_domain
# ===========================================================================


class TestExtractDomain:
    def test_simple_url(self):
        assert extract_domain("https://example.com/path") == "example.com"

    def test_with_port(self):
        assert extract_domain("https://example.com:8080/path") == "example.com:8080"

    def test_empty_string(self):
        assert extract_domain("") is None

    def test_none_like(self):
        # Empty string returns None
        assert extract_domain("") is None

    def test_invalid_url(self):
        # urlparse handles most strings without raising
        result = extract_domain("not-a-url")
        # urlparse("not-a-url") has empty netloc
        assert result is None

    def test_subdomain(self):
        assert extract_domain("https://api.example.com/v1") == "api.example.com"


# ===========================================================================
# classify_http_error
# ===========================================================================


class TestClassifyHttpError:
    def test_authentication_error(self):
        from foundry_mcp.core.research.providers.base import AuthenticationError
        from foundry_mcp.core.research.providers.resilience import ErrorType

        error = AuthenticationError(provider="test", message="Bad key")
        result = classify_http_error(error, "test")
        assert result.retryable is False
        assert result.trips_breaker is False
        assert result.error_type == ErrorType.AUTHENTICATION

    def test_rate_limit_error(self):
        from foundry_mcp.core.research.providers.base import RateLimitError
        from foundry_mcp.core.research.providers.resilience import ErrorType

        error = RateLimitError(provider="test", retry_after=5.0)
        result = classify_http_error(error, "test")
        assert result.retryable is True
        assert result.trips_breaker is False
        assert result.backoff_seconds == 5.0
        assert result.error_type == ErrorType.RATE_LIMIT

    def test_server_error_500(self):
        from foundry_mcp.core.research.providers.base import SearchProviderError
        from foundry_mcp.core.research.providers.resilience import ErrorType

        error = SearchProviderError(provider="test", message="API error 500: Internal Server Error", retryable=True)
        result = classify_http_error(error, "test")
        assert result.retryable is True
        assert result.trips_breaker is True
        assert result.error_type == ErrorType.SERVER_ERROR

    def test_server_error_502(self):
        from foundry_mcp.core.research.providers.base import SearchProviderError
        from foundry_mcp.core.research.providers.resilience import ErrorType

        error = SearchProviderError(provider="test", message="API error 502: Bad Gateway", retryable=True)
        result = classify_http_error(error, "test")
        assert result.error_type == ErrorType.SERVER_ERROR

    def test_bad_request_400(self):
        from foundry_mcp.core.research.providers.base import SearchProviderError
        from foundry_mcp.core.research.providers.resilience import ErrorType

        error = SearchProviderError(provider="test", message="API error 400: Bad Request", retryable=False)
        result = classify_http_error(error, "test")
        assert result.retryable is False
        assert result.trips_breaker is False
        assert result.error_type == ErrorType.INVALID_REQUEST

    def test_timeout_exception(self):
        from foundry_mcp.core.research.providers.resilience import ErrorType

        error = httpx.TimeoutException("Timed out")
        result = classify_http_error(error, "test")
        assert result.retryable is True
        assert result.trips_breaker is True
        assert result.error_type == ErrorType.TIMEOUT

    def test_connect_error(self):
        from foundry_mcp.core.research.providers.resilience import ErrorType

        error = httpx.ConnectError("Connection refused")
        result = classify_http_error(error, "test")
        assert result.retryable is True
        assert result.trips_breaker is True
        assert result.error_type == ErrorType.NETWORK

    def test_unknown_error(self):
        from foundry_mcp.core.research.providers.resilience import ErrorType

        error = RuntimeError("Something unexpected")
        result = classify_http_error(error, "test")
        assert result.retryable is False
        assert result.trips_breaker is True
        assert result.error_type == ErrorType.UNKNOWN

    def test_custom_classifier_takes_priority(self):
        from foundry_mcp.core.research.providers.base import RateLimitError
        from foundry_mcp.core.research.providers.resilience import (
            ErrorClassification,
            ErrorType,
        )

        custom_result = ErrorClassification(
            retryable=True,
            trips_breaker=False,
            error_type=ErrorType.QUOTA_EXCEEDED,
        )
        error = RateLimitError(provider="google", retry_after=60.0, reason="quota")
        result = classify_http_error(error, "google", custom_classifier=lambda e: custom_result)
        assert result.error_type == ErrorType.QUOTA_EXCEEDED

    def test_custom_classifier_returns_none_falls_through(self):
        from foundry_mcp.core.research.providers.base import AuthenticationError
        from foundry_mcp.core.research.providers.resilience import ErrorType

        error = AuthenticationError(provider="test")
        result = classify_http_error(error, "test", custom_classifier=lambda e: None)
        assert result.error_type == ErrorType.AUTHENTICATION

    def test_search_provider_error_unknown_uses_retryable_flag(self):
        from foundry_mcp.core.research.providers.base import SearchProviderError
        from foundry_mcp.core.research.providers.resilience import ErrorType

        error = SearchProviderError(provider="test", message="Some error", retryable=True)
        result = classify_http_error(error, "test")
        assert result.retryable is True
        assert result.trips_breaker is True
        assert result.error_type == ErrorType.UNKNOWN


# ===========================================================================
# create_resilience_executor
# ===========================================================================


class TestCreateResilienceExecutor:
    @pytest.mark.asyncio
    async def test_successful_execution(self):
        from foundry_mcp.core.research.providers.resilience import (
            ProviderResilienceConfig,
            get_resilience_manager,
            reset_resilience_manager_for_testing,
        )

        reset_resilience_manager_for_testing()
        config = ProviderResilienceConfig(max_retries=1)

        def classifier(e):
            from foundry_mcp.core.research.providers.resilience import (
                ErrorClassification,
                ErrorType,
            )

            return ErrorClassification(retryable=False, trips_breaker=False, error_type=ErrorType.UNKNOWN)

        executor = create_resilience_executor("test_provider", config, classifier)

        async def success_func():
            return {"result": "ok"}

        result = await executor(success_func, timeout=5.0)
        assert result == {"result": "ok"}

    @pytest.mark.asyncio
    async def test_circuit_breaker_error_translation(self):
        from foundry_mcp.core.research.providers.base import SearchProviderError
        from foundry_mcp.core.research.providers.resilience import (
            ProviderResilienceConfig,
            reset_resilience_manager_for_testing,
        )
        from foundry_mcp.core.resilience import CircuitBreakerError

        reset_resilience_manager_for_testing()
        config = ProviderResilienceConfig(max_retries=0)

        executor = create_resilience_executor("test_provider", config, lambda e: None)

        with patch(
            "foundry_mcp.core.research.providers.resilience.execute_with_resilience",
            side_effect=CircuitBreakerError("test_provider", "open"),
        ):
            with pytest.raises(SearchProviderError, match="Circuit breaker open"):
                await executor(AsyncMock(), timeout=5.0)

    @pytest.mark.asyncio
    async def test_rate_limit_wait_error_translation(self):
        from foundry_mcp.core.research.providers.base import RateLimitError
        from foundry_mcp.core.research.providers.resilience import (
            ProviderResilienceConfig,
            RateLimitWaitError,
            reset_resilience_manager_for_testing,
        )

        reset_resilience_manager_for_testing()
        config = ProviderResilienceConfig(max_retries=0)

        executor = create_resilience_executor("test_provider", config, lambda e: None)

        with patch(
            "foundry_mcp.core.research.providers.resilience.execute_with_resilience",
            side_effect=RateLimitWaitError("Wait too long", wait_needed=10.0, max_wait=5.0),
        ):
            with pytest.raises(RateLimitError) as exc_info:
                await executor(AsyncMock(), timeout=5.0)
            assert exc_info.value.retry_after == 10.0

    @pytest.mark.asyncio
    async def test_time_budget_exceeded_translation(self):
        from foundry_mcp.core.research.providers.base import SearchProviderError
        from foundry_mcp.core.research.providers.resilience import (
            ProviderResilienceConfig,
            TimeBudgetExceededError,
            reset_resilience_manager_for_testing,
        )

        reset_resilience_manager_for_testing()
        config = ProviderResilienceConfig(max_retries=0)

        executor = create_resilience_executor("test_provider", config, lambda e: None)

        with patch(
            "foundry_mcp.core.research.providers.resilience.execute_with_resilience",
            side_effect=TimeBudgetExceededError("Budget exceeded"),
        ):
            with pytest.raises(SearchProviderError, match="Request timed out"):
                await executor(AsyncMock(), timeout=5.0)

    @pytest.mark.asyncio
    async def test_generic_exception_redacts_secrets(self):
        from foundry_mcp.core.research.providers.base import SearchProviderError
        from foundry_mcp.core.research.providers.resilience import (
            ErrorClassification,
            ErrorType,
            ProviderResilienceConfig,
            reset_resilience_manager_for_testing,
        )

        reset_resilience_manager_for_testing()
        config = ProviderResilienceConfig(max_retries=0)

        def classifier(e):
            return ErrorClassification(retryable=False, trips_breaker=True, error_type=ErrorType.UNKNOWN)

        executor = create_resilience_executor("test_provider", config, classifier)

        with patch(
            "foundry_mcp.core.research.providers.resilience.execute_with_resilience",
            side_effect=RuntimeError("Failed with api_key=tvly-real-secret-key-123"),
        ):
            with pytest.raises(SearchProviderError) as exc_info:
                await executor(AsyncMock(), timeout=5.0)
            assert "tvly-real-secret-key-123" not in str(exc_info.value)


# ===========================================================================
# check_provider_health
# ===========================================================================


class TestCheckProviderHealth:
    @pytest.mark.asyncio
    async def test_no_api_key(self):
        result = await check_provider_health("test", None, "https://api.test.com")
        assert result is False

    @pytest.mark.asyncio
    async def test_no_test_func(self):
        result = await check_provider_health("test", "key-123", "https://api.test.com")
        assert result is True

    @pytest.mark.asyncio
    async def test_successful_probe(self):
        probe = AsyncMock(return_value=None)
        result = await check_provider_health("test", "key-123", "https://api.test.com", test_func=probe)
        assert result is True
        probe.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_auth_error_probe(self):
        from foundry_mcp.core.research.providers.base import AuthenticationError

        probe = AsyncMock(side_effect=AuthenticationError(provider="test"))
        result = await check_provider_health("test", "key-123", "https://api.test.com", test_func=probe)
        assert result is False

    @pytest.mark.asyncio
    async def test_generic_error_probe(self):
        probe = AsyncMock(side_effect=RuntimeError("connection refused"))
        result = await check_provider_health("test", "key-123", "https://api.test.com", test_func=probe)
        assert result is False

    @pytest.mark.asyncio
    async def test_error_message_redacted(self, caplog):
        """Ensure API keys in error messages are redacted in logs."""
        probe = AsyncMock(side_effect=RuntimeError("Failed api_key=tvly-real-secret-12345"))
        with caplog.at_level("WARNING"):
            await check_provider_health("test", "key-123", "https://api.test.com", test_func=probe)
        # Check that the secret was redacted in the log output
        for record in caplog.records:
            assert "tvly-real-secret-12345" not in record.getMessage()


# ===========================================================================
# resolve_provider_settings
# ===========================================================================


class TestResolveProviderSettings:
    def test_explicit_api_key(self):
        result = resolve_provider_settings("tavily", "TAVILY_API_KEY", api_key="explicit-key")
        assert result["api_key"] == "explicit-key"
        assert result["api_key_source"] == "explicit"

    def test_env_var_fallback(self):
        with patch.dict(os.environ, {"TAVILY_API_KEY": "env-key"}, clear=False):
            result = resolve_provider_settings("tavily", "TAVILY_API_KEY")
            assert result["api_key"] == "env-key"
            assert result["api_key_source"] == "environment"

    def test_explicit_takes_priority(self):
        with patch.dict(os.environ, {"TAVILY_API_KEY": "env-key"}, clear=False):
            result = resolve_provider_settings("tavily", "TAVILY_API_KEY", api_key="explicit-key")
            assert result["api_key"] == "explicit-key"
            assert result["api_key_source"] == "explicit"

    def test_missing_required_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key required"):
                resolve_provider_settings("tavily", "TAVILY_API_KEY")

    def test_missing_optional_ok(self):
        with patch.dict(os.environ, {}, clear=True):
            result = resolve_provider_settings("semantic_scholar", "SEMANTIC_SCHOLAR_API_KEY", required=False)
            assert result["api_key"] is None
            assert result["api_key_source"] is None

    def test_base_url_defaults(self):
        result = resolve_provider_settings(
            "tavily",
            "TAVILY_API_KEY",
            api_key="key",
            default_base_url="https://api.tavily.com/",
        )
        assert result["base_url"] == "https://api.tavily.com"  # trailing slash stripped

    def test_base_url_explicit(self):
        result = resolve_provider_settings(
            "tavily",
            "TAVILY_API_KEY",
            api_key="key",
            base_url="https://custom.api.com/",
            default_base_url="https://api.tavily.com/",
        )
        assert result["base_url"] == "https://custom.api.com"

    def test_defaults(self):
        result = resolve_provider_settings("tavily", "TAVILY_API_KEY", api_key="key")
        assert result["timeout"] == 30.0
        assert result["max_retries"] == 3
        assert result["rate_limit"] == 1.0

    def test_custom_values(self):
        result = resolve_provider_settings(
            "tavily",
            "TAVILY_API_KEY",
            api_key="key",
            timeout=60.0,
            max_retries=5,
            rate_limit=0.5,
        )
        assert result["timeout"] == 60.0
        assert result["max_retries"] == 5
        assert result["rate_limit"] == 0.5

    def test_extra_env(self):
        with patch.dict(os.environ, {"GOOGLE_CSE_ID": "cse-123"}, clear=False):
            result = resolve_provider_settings(
                "google",
                "GOOGLE_API_KEY",
                api_key="key",
                extra_env={"cx": "GOOGLE_CSE_ID"},
            )
            assert result["cx"] == "cse-123"

    def test_extra_env_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            result = resolve_provider_settings(
                "google",
                "GOOGLE_API_KEY",
                api_key="key",
                extra_env={"cx": "GOOGLE_CSE_ID"},
            )
            assert result["cx"] is None

    def test_error_message_format(self):
        """Verify the error message mentions both param and env var."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="TAVILY_API_KEY"):
                resolve_provider_settings("tavily", "TAVILY_API_KEY")


# ===========================================================================
# Integration: redaction in error paths
# ===========================================================================


class TestRedactionIntegration:
    """End-to-end tests verifying that API keys never leak through any path."""

    def test_extract_error_message_redacts(self):
        response = MagicMock(spec=httpx.Response)
        response.json.return_value = {
            "error": "Auth failed for token=sk-secret-key-that-is-very-long"
        }
        result = extract_error_message(response)
        assert "sk-secret-key-that-is-very-long" not in result

    def test_extract_error_message_text_fallback_redacts(self):
        response = MagicMock(spec=httpx.Response)
        response.json.side_effect = ValueError()
        response.text = "Error: api_key=tvly-super-secret-key-value is invalid"
        result = extract_error_message(response)
        assert "tvly-super-secret-key-value" not in result

    @pytest.mark.asyncio
    async def test_health_check_never_logs_key(self, caplog):
        probe = AsyncMock(side_effect=RuntimeError("secret=my-very-long-api-key-value"))
        with caplog.at_level("WARNING"):
            await check_provider_health("test", "my-very-long-api-key-value", "https://test.com", test_func=probe)
        log_text = " ".join(r.getMessage() for r in caplog.records)
        assert "my-very-long-api-key-value" not in log_text
