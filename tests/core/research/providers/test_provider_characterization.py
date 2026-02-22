"""Provider characterization tests — behavioral baseline before extraction.

Captures the current error classification, Retry-After parsing, timeout/cancellation
propagation, API key resolution, and client lifecycle invariants across all 5
HTTP-backed research providers. These snapshots ensure behavioral parity after
the shared-utility extraction in Phase 5.
"""

import httpx
import pytest

from foundry_mcp.core.research.providers.base import (
    AuthenticationError,
    RateLimitError,
    SearchProviderError,
)
from foundry_mcp.core.research.providers.resilience import (
    ErrorType,
)

# Re-export from conftest for use in this file's test methods
from tests.core.research.providers.conftest import (
    FACTORY_MAP,
    PROVIDERS,
    make_mock_response,
)
from tests.core.research.providers.conftest import (
    make_google as _make_google,
)
from tests.core.research.providers.conftest import (
    make_perplexity as _make_perplexity,
)
from tests.core.research.providers.conftest import (
    make_semantic_scholar as _make_semantic_scholar,
)
from tests.core.research.providers.conftest import (
    make_tavily as _make_tavily,
)
from tests.core.research.providers.conftest import (
    make_tavily_extract as _make_tavily_extract,
)

# ===================================================================
# 1. Error Classification Snapshots
# ===================================================================


class TestErrorClassificationSnapshots:
    """Verify every provider classifies the same error types identically.

    These are the behavioral invariants that MUST be preserved after
    the shared-utility extraction.
    """

    def test_authentication_error_not_retryable(self, provider):
        """AuthenticationError → not retryable, no breaker trip."""
        error = AuthenticationError(provider=provider.get_provider_name())
        classification = provider.classify_error(error)

        assert classification.retryable is False
        assert classification.trips_breaker is False
        assert classification.error_type == ErrorType.AUTHENTICATION

    def test_rate_limit_error_retryable_no_breaker(self, provider):
        """RateLimitError → retryable, no breaker trip."""
        error = RateLimitError(provider=provider.get_provider_name(), retry_after=5.0)
        classification = provider.classify_error(error)

        assert classification.retryable is True
        assert classification.trips_breaker is False
        # Google classifies RateLimitError as QUOTA_EXCEEDED; others as RATE_LIMIT
        assert classification.error_type in (ErrorType.RATE_LIMIT, ErrorType.QUOTA_EXCEEDED)

    def test_rate_limit_preserves_retry_after(self, provider):
        """RateLimitError backoff_seconds reflects retry_after value."""
        error = RateLimitError(provider=provider.get_provider_name(), retry_after=42.0)
        classification = provider.classify_error(error)

        assert classification.backoff_seconds == 42.0

    def test_rate_limit_none_retry_after(self, provider):
        """RateLimitError with no retry_after → backoff_seconds is None."""
        error = RateLimitError(provider=provider.get_provider_name())
        classification = provider.classify_error(error)

        assert classification.retryable is True
        assert classification.backoff_seconds is None

    @pytest.mark.parametrize("status_code", ["500", "502", "503", "504"])
    def test_server_error_retryable_trips_breaker(self, provider, status_code):
        """5xx SearchProviderError → retryable, trips breaker."""
        error = SearchProviderError(
            provider=provider.get_provider_name(),
            message=f"HTTP {status_code} Server Error",
            retryable=True,
        )
        classification = provider.classify_error(error)

        assert classification.retryable is True
        assert classification.trips_breaker is True
        assert classification.error_type == ErrorType.SERVER_ERROR

    def test_bad_request_not_retryable(self, provider):
        """400 SearchProviderError → not retryable, no breaker trip."""
        error = SearchProviderError(
            provider=provider.get_provider_name(),
            message="HTTP 400 Bad Request",
            retryable=False,
        )
        classification = provider.classify_error(error)

        assert classification.retryable is False
        assert classification.trips_breaker is False
        assert classification.error_type == ErrorType.INVALID_REQUEST

    def test_timeout_exception_retryable_trips_breaker(self, provider):
        """httpx.TimeoutException → retryable, trips breaker."""
        error = httpx.ReadTimeout("Connection timed out")
        classification = provider.classify_error(error)

        assert classification.retryable is True
        assert classification.trips_breaker is True
        assert classification.error_type == ErrorType.TIMEOUT

    def test_connect_timeout_retryable_trips_breaker(self, provider):
        """httpx.ConnectTimeout → retryable, trips breaker (subclass of TimeoutException)."""
        error = httpx.ConnectTimeout("Connect timed out")
        classification = provider.classify_error(error)

        assert classification.retryable is True
        assert classification.trips_breaker is True
        assert classification.error_type == ErrorType.TIMEOUT

    def test_network_error_retryable_trips_breaker(self, provider):
        """httpx.RequestError (network) → retryable, trips breaker."""
        error = httpx.ConnectError("Connection refused")
        classification = provider.classify_error(error)

        assert classification.retryable is True
        assert classification.trips_breaker is True
        assert classification.error_type == ErrorType.NETWORK

    def test_unknown_error_trips_breaker(self, provider):
        """Unknown Exception → not retryable, trips breaker."""
        error = RuntimeError("Something unexpected")
        classification = provider.classify_error(error)

        assert classification.retryable is False
        assert classification.trips_breaker is True
        assert classification.error_type == ErrorType.UNKNOWN


class TestGoogleSpecificErrorClassification:
    """Google provider has special 403 quota detection."""

    def test_google_403_quota_is_rate_limit(self):
        """Google 403 with 'quota' in message → RateLimitError classification."""
        provider = _make_google()
        error = SearchProviderError(
            provider="google",
            message="HTTP 403: Daily Limit / quota exceeded",
            retryable=True,
        )
        classification = provider.classify_error(error)

        # Google treats quota-related 403 as retryable
        assert classification.retryable is True


# ===================================================================
# 2. Retry-After Header Parsing
# ===================================================================


class TestRetryAfterParsing:
    """Verify Retry-After header is parsed uniformly across providers."""

    @pytest.mark.parametrize(
        "provider_name",
        PROVIDERS,
    )
    def test_numeric_retry_after_parsed(self, provider_name):
        """Numeric Retry-After header is parsed as float seconds."""
        provider = FACTORY_MAP[provider_name]()
        response = make_mock_response(
            status_code=429,
            headers={"Retry-After": "30"},
            text="Rate limited",
            json_data={"error": "rate limited"},
        )
        result = provider._parse_retry_after(response)

        assert result == 30.0

    @pytest.mark.parametrize(
        "provider_name",
        PROVIDERS,
    )
    def test_float_retry_after_parsed(self, provider_name):
        """Float Retry-After header is parsed correctly."""
        provider = FACTORY_MAP[provider_name]()
        response = make_mock_response(
            status_code=429,
            headers={"Retry-After": "1.5"},
            text="Rate limited",
            json_data={"error": "rate limited"},
        )
        result = provider._parse_retry_after(response)

        assert result == 1.5

    @pytest.mark.parametrize(
        "provider_name",
        PROVIDERS,
    )
    def test_missing_retry_after_returns_none(self, provider_name):
        """Missing Retry-After header returns None."""
        provider = FACTORY_MAP[provider_name]()
        response = make_mock_response(
            status_code=429,
            text="Rate limited",
            json_data={"error": "rate limited"},
        )
        result = provider._parse_retry_after(response)

        assert result is None

    @pytest.mark.parametrize(
        "provider_name",
        PROVIDERS,
    )
    def test_invalid_retry_after_returns_none(self, provider_name):
        """Non-numeric Retry-After header returns None (silent failure)."""
        provider = FACTORY_MAP[provider_name]()
        response = make_mock_response(
            status_code=429,
            headers={"Retry-After": "not-a-number"},
            text="Rate limited",
            json_data={"error": "rate limited"},
        )
        result = provider._parse_retry_after(response)

        assert result is None


# ===================================================================
# 3. Timeout / Cancellation Propagation
# ===================================================================


class TestTimeoutCancellationPropagation:
    """Verify timeout and time budget behavior across providers."""

    def test_default_timeout_is_30_seconds(self, provider):
        """All providers default to 30s timeout."""
        assert provider._timeout == 30.0

    def test_custom_timeout_respected(self, provider_name):
        """Custom timeout is stored correctly."""
        provider = FACTORY_MAP[provider_name](timeout=120.0)
        assert provider._timeout == 120.0

    def test_time_budget_calculation(self, provider_name):
        """Time budget = timeout × (max_retries + 1)."""
        provider = FACTORY_MAP[provider_name](timeout=30.0, max_retries=3)
        expected_budget = 30.0 * (3 + 1)  # 120 seconds

        # Verify the provider stores the values needed for budget calculation
        assert provider._timeout == 30.0
        assert provider._max_retries == 3
        # Budget = timeout * (retries + 1)
        assert provider._timeout * (provider._max_retries + 1) == expected_budget


# ===================================================================
# 4. API Key Resolution
# ===================================================================


class TestAPIKeyResolution:
    """Verify API key resolution: explicit param > env var > error."""

    def test_tavily_explicit_key(self):
        """Tavily: explicit API key takes priority."""
        p = _make_tavily()
        assert p._api_key == "tvly-test-key"

    def test_tavily_env_var(self, monkeypatch):
        """Tavily: falls back to TAVILY_API_KEY env var."""
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-env-key")
        from foundry_mcp.core.research.providers.tavily import TavilySearchProvider

        p = TavilySearchProvider()
        assert p._api_key == "tvly-env-key"

    def test_tavily_missing_key_raises(self, monkeypatch):
        """Tavily: missing key raises ValueError."""
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        from foundry_mcp.core.research.providers.tavily import TavilySearchProvider

        with pytest.raises(ValueError, match="Tavily API key"):
            TavilySearchProvider()

    def test_perplexity_explicit_key(self):
        """Perplexity: explicit API key takes priority."""
        p = _make_perplexity()
        assert p._api_key == "pplx-test-key"

    def test_perplexity_env_var(self, monkeypatch):
        """Perplexity: falls back to PERPLEXITY_API_KEY env var."""
        monkeypatch.setenv("PERPLEXITY_API_KEY", "pplx-env-key")
        from foundry_mcp.core.research.providers.perplexity import (
            PerplexitySearchProvider,
        )

        p = PerplexitySearchProvider()
        assert p._api_key == "pplx-env-key"

    def test_perplexity_missing_key_raises(self, monkeypatch):
        """Perplexity: missing key raises ValueError."""
        monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
        from foundry_mcp.core.research.providers.perplexity import (
            PerplexitySearchProvider,
        )

        with pytest.raises(ValueError, match="Perplexity API key"):
            PerplexitySearchProvider()

    def test_google_explicit_keys(self):
        """Google: explicit API key + CSE ID takes priority."""
        p = _make_google()
        assert p._api_key == "google-test-key"
        assert p._cx == "cse-test"

    def test_google_env_vars(self, monkeypatch):
        """Google: falls back to GOOGLE_API_KEY + GOOGLE_CSE_ID env vars."""
        monkeypatch.setenv("GOOGLE_API_KEY", "google-env-key")
        monkeypatch.setenv("GOOGLE_CSE_ID", "cse-env")
        from foundry_mcp.core.research.providers.google import GoogleSearchProvider

        p = GoogleSearchProvider()
        assert p._api_key == "google-env-key"
        assert p._cx == "cse-env"

    def test_google_missing_cse_id_raises(self, monkeypatch):
        """Google: missing CSE ID raises ValueError."""
        monkeypatch.setenv("GOOGLE_API_KEY", "google-env-key")
        monkeypatch.delenv("GOOGLE_CSE_ID", raising=False)
        from foundry_mcp.core.research.providers.google import GoogleSearchProvider

        with pytest.raises(ValueError):
            GoogleSearchProvider()

    def test_google_missing_api_key_raises(self, monkeypatch):
        """Google: missing API key raises ValueError."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_CSE_ID", raising=False)
        from foundry_mcp.core.research.providers.google import GoogleSearchProvider

        with pytest.raises(ValueError, match="Google API key"):
            GoogleSearchProvider()

    def test_semantic_scholar_explicit_key(self):
        """Semantic Scholar: explicit API key takes priority."""
        p = _make_semantic_scholar()
        assert p._api_key == "s2-test-key"

    def test_semantic_scholar_optional_key(self, monkeypatch):
        """Semantic Scholar: API key is optional (works without it)."""
        monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
        from foundry_mcp.core.research.providers.semantic_scholar import (
            SemanticScholarProvider,
        )

        p = SemanticScholarProvider()
        assert p._api_key is None

    def test_semantic_scholar_env_var(self, monkeypatch):
        """Semantic Scholar: falls back to SEMANTIC_SCHOLAR_API_KEY env var."""
        monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "s2-env-key")
        from foundry_mcp.core.research.providers.semantic_scholar import (
            SemanticScholarProvider,
        )

        p = SemanticScholarProvider()
        assert p._api_key == "s2-env-key"

    def test_tavily_extract_explicit_key(self):
        """Tavily Extract: explicit API key takes priority."""
        p = _make_tavily_extract()
        assert p._api_key == "tvly-test-key"

    def test_tavily_extract_env_var(self, monkeypatch):
        """Tavily Extract: falls back to TAVILY_API_KEY env var."""
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-env-key")
        from foundry_mcp.core.research.providers.tavily_extract import (
            TavilyExtractProvider,
        )

        p = TavilyExtractProvider()
        assert p._api_key == "tvly-env-key"

    def test_tavily_extract_missing_key_raises(self, monkeypatch):
        """Tavily Extract: missing key raises ValueError."""
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        from foundry_mcp.core.research.providers.tavily_extract import (
            TavilyExtractProvider,
        )

        with pytest.raises(ValueError, match="Tavily API key"):
            TavilyExtractProvider()


# ===================================================================
# 5. Client Lifecycle Invariants
# ===================================================================


class TestClientLifecycleInvariants:
    """Verify client creation patterns are consistent across providers."""

    def test_provider_name_matches_expected(self):
        """Each provider returns the correct name string."""
        expected = {
            "tavily": "tavily",
            "perplexity": "perplexity",
            "google": "google",
            "semantic_scholar": "semantic_scholar",
            "tavily_extract": "tavily_extract",
        }
        for name, factory in FACTORY_MAP.items():
            provider = factory()
            assert provider.get_provider_name() == expected[name], f"{name} provider name mismatch"

    def test_default_max_retries(self, provider):
        """All providers default to 3 max retries."""
        assert provider._max_retries == 3

    def test_custom_max_retries(self, provider_name):
        """Custom max_retries is stored correctly."""
        provider = FACTORY_MAP[provider_name](max_retries=5)
        assert provider._max_retries == 5

    def test_resilience_config_present(self, provider):
        """All providers have a resilience config."""
        assert hasattr(provider, "_resilience_config") or hasattr(provider, "resilience_config")

    def test_rate_limit_property(self, provider):
        """All providers expose a rate_limit property."""
        rate_limit = provider.rate_limit
        assert rate_limit is None or isinstance(rate_limit, (int, float))

    def test_has_classify_error_method(self, provider):
        """All providers implement classify_error."""
        assert callable(getattr(provider, "classify_error", None))

    def test_has_parse_retry_after_method(self, provider):
        """All providers implement _parse_retry_after."""
        assert callable(getattr(provider, "_parse_retry_after", None))


# ===================================================================
# 6. Cross-Provider Classification Consistency
# ===================================================================


class TestCrossProviderConsistency:
    """Verify all providers agree on classification for the same error inputs.

    This ensures the extraction to shared utilities preserves identical behavior.
    """

    @pytest.mark.parametrize(
        "error_factory,expected_type",
        [
            (
                lambda name: AuthenticationError(provider=name),
                ErrorType.AUTHENTICATION,
            ),
            (
                lambda name: RateLimitError(provider=name, retry_after=10.0),
                # Google returns QUOTA_EXCEEDED; others RATE_LIMIT — both valid
                {ErrorType.RATE_LIMIT, ErrorType.QUOTA_EXCEEDED},
            ),
            (
                lambda name: SearchProviderError(
                    provider=name, message="HTTP 500 Internal Server Error", retryable=True
                ),
                ErrorType.SERVER_ERROR,
            ),
            (
                lambda name: SearchProviderError(provider=name, message="HTTP 400 Bad Request", retryable=False),
                ErrorType.INVALID_REQUEST,
            ),
            (
                lambda _: httpx.ReadTimeout("timeout"),
                ErrorType.TIMEOUT,
            ),
            (
                lambda _: httpx.ConnectError("connection refused"),
                ErrorType.NETWORK,
            ),
            (
                lambda _: RuntimeError("unexpected"),
                ErrorType.UNKNOWN,
            ),
        ],
        ids=[
            "auth_error",
            "rate_limit",
            "server_500",
            "bad_request_400",
            "timeout",
            "network",
            "unknown",
        ],
    )
    def test_all_providers_agree_on_error_type(self, error_factory, expected_type):
        """All providers classify the same error to the expected ErrorType(s)."""
        # expected_type may be a single ErrorType or a set of acceptable types
        acceptable = expected_type if isinstance(expected_type, set) else {expected_type}

        for name, factory in FACTORY_MAP.items():
            provider = factory()
            error = error_factory(name)
            classification = provider.classify_error(error)
            assert classification.error_type in acceptable, (
                f"{name} classified as {classification.error_type}, expected one of {acceptable}"
            )

    @pytest.mark.parametrize(
        "error_factory,expected_retryable",
        [
            (lambda name: AuthenticationError(provider=name), False),
            (lambda name: RateLimitError(provider=name), True),
            (
                lambda name: SearchProviderError(provider=name, message="HTTP 503", retryable=True),
                True,
            ),
            (
                lambda name: SearchProviderError(provider=name, message="HTTP 400", retryable=False),
                False,
            ),
            (lambda _: httpx.ReadTimeout("timeout"), True),
            (lambda _: httpx.ConnectError("refused"), True),
            (lambda _: RuntimeError("unexpected"), False),
        ],
        ids=[
            "auth_not_retryable",
            "rate_limit_retryable",
            "server_503_retryable",
            "bad_request_not_retryable",
            "timeout_retryable",
            "network_retryable",
            "unknown_not_retryable",
        ],
    )
    def test_all_providers_agree_on_retryable(self, error_factory, expected_retryable):
        """All providers agree on retryable flag for the same error."""
        for name, factory in FACTORY_MAP.items():
            provider = factory()
            error = error_factory(name)
            classification = provider.classify_error(error)
            assert classification.retryable is expected_retryable, (
                f"{name}: retryable={classification.retryable}, expected {expected_retryable}"
            )
