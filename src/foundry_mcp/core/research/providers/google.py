"""Google Custom Search provider for web search.

This module implements GoogleSearchProvider, which wraps the Google Custom Search
JSON API to provide web search capabilities for the deep research workflow.

Google Custom Search API documentation:
https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list

Resilience Configuration:
    - Rate Limit: 1 RPS with burst limit of 3 (Google CSE has daily quota)
    - Circuit Breaker: Opens after 5 failures, 30s recovery timeout
    - Retry: Up to 3 retries with exponential backoff (1-60s)
    - Error Handling:
        - 429: Retryable, does NOT trip circuit breaker
        - 401/403: Not retryable, does NOT trip circuit breaker
        - 5xx: Retryable, trips circuit breaker
        - Timeouts: Retryable, trips circuit breaker

Example usage:
    provider = GoogleSearchProvider(
        api_key="AIza...",
        cx="017576662512468239146:omuauf_lfve",
    )
    sources = await provider.search("machine learning trends", max_results=5)
"""

import logging
import os
from datetime import datetime
from dataclasses import replace
from typing import Any, Optional

import httpx

from foundry_mcp.core.research.models import ResearchSource, SourceType
from foundry_mcp.core.research.providers.base import (
    AuthenticationError,
    RateLimitError,
    SearchProvider,
    SearchProviderError,
    SearchResult,
)
from foundry_mcp.core.research.providers.resilience import (
    ErrorClassification,
    ErrorType,
    ProviderResilienceConfig,
    execute_with_resilience,
    get_provider_config,
    get_resilience_manager,
    RateLimitWaitError,
    TimeBudgetExceededError,
)
from foundry_mcp.core.resilience import CircuitBreakerError

logger = logging.getLogger(__name__)

# Google Custom Search API constants
GOOGLE_API_BASE_URL = "https://www.googleapis.com/customsearch/v1"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RATE_LIMIT = 1.0  # requests per second (Google CSE has daily quota limits)


class GoogleSearchProvider(SearchProvider):
    """Google Custom Search API provider for web search.

    Wraps the Google Custom Search JSON API to provide web search capabilities.
    Requires a Google API key and a Custom Search Engine (CSE) ID.

    To set up:
    1. Create a project in Google Cloud Console
    2. Enable the Custom Search API
    3. Create an API key
    4. Create a Custom Search Engine at https://cse.google.com/
    5. Get the Search Engine ID (cx parameter)

    Attributes:
        api_key: Google API key (required)
        cx: Custom Search Engine ID (required)
        base_url: API base URL (default: https://www.googleapis.com/customsearch/v1)
        timeout: Request timeout in seconds (default: 30.0)
        max_retries: Maximum retry attempts for rate limits (default: 3)

    Example:
        provider = GoogleSearchProvider(
            api_key="AIza...",
            cx="017576662512468239146:omuauf_lfve",
        )
        sources = await provider.search(
            "AI trends 2024",
            max_results=5,
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cx: Optional[str] = None,
        base_url: str = GOOGLE_API_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        resilience_config: Optional[ProviderResilienceConfig] = None,
    ):
        """Initialize Google Custom Search provider.

        Args:
            api_key: Google API key. If not provided, reads from GOOGLE_API_KEY env var.
            cx: Custom Search Engine ID. If not provided, reads from GOOGLE_CSE_ID env var.
            base_url: API base URL (default: https://www.googleapis.com/customsearch/v1)
            timeout: Request timeout in seconds (default: 30.0)
            max_retries: Maximum retry attempts for rate limits (default: 3)
            resilience_config: Custom resilience configuration. If None, uses
                defaults from PROVIDER_CONFIGS["google"].

        Raises:
            ValueError: If API key or CSE ID is not provided or found in environment
        """
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Google API key required. Provide via api_key parameter "
                "or GOOGLE_API_KEY environment variable."
            )

        self._cx = cx or os.environ.get("GOOGLE_CSE_ID")
        if not self._cx:
            raise ValueError(
                "Google Custom Search Engine ID required. Provide via cx parameter "
                "or GOOGLE_CSE_ID environment variable."
            )

        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._rate_limit_value = DEFAULT_RATE_LIMIT
        if resilience_config is None:
            self._resilience_config = replace(
                get_provider_config("google"),
                max_retries=max_retries,
            )
        else:
            self._resilience_config = resilience_config

    def get_provider_name(self) -> str:
        """Return the provider identifier.

        Returns:
            "google"
        """
        return "google"

    @property
    def rate_limit(self) -> Optional[float]:
        """Return the rate limit in requests per second.

        Returns:
            1.0 (one request per second)
        """
        return self._rate_limit_value

    @property
    def resilience_config(self) -> ProviderResilienceConfig:
        """Return the resilience configuration for this provider.

        Returns ProviderResilienceConfig for Google with settings for:
        - Rate limiting (requests per second, burst limit)
        - Retry behavior (max retries, delays, jitter)
        - Circuit breaker (failure threshold, recovery timeout)

        If a custom config was provided via constructor, returns that.
        Otherwise, returns defaults from PROVIDER_CONFIGS["google"].

        Returns:
            ProviderResilienceConfig for this provider
        """
        if self._resilience_config is not None:
            return self._resilience_config
        return get_provider_config("google")

    async def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any,
    ) -> list[ResearchSource]:
        """Execute a web search via Google Custom Search API.

        Args:
            query: The search query string
            max_results: Maximum number of results to return (default: 10, max: 10 per request)
            **kwargs: Additional Google CSE options:
                - site_search: Restrict results to a specific site
                - date_restrict: Restrict by date (e.g., "d7" for past week, "m1" for past month)
                - file_type: Restrict to specific file types (e.g., "pdf")
                - safe: Safe search level ("off", "medium", "high")
                - sub_query_id: SubQuery ID for source tracking

        Returns:
            List of ResearchSource objects

        Raises:
            AuthenticationError: If API key is invalid
            RateLimitError: If rate limit/quota exceeded after all retries
            SearchProviderError: For other API errors
        """
        # Extract Google-specific options
        site_search = kwargs.get("site_search")
        date_restrict = kwargs.get("date_restrict")
        file_type = kwargs.get("file_type")
        safe = kwargs.get("safe", "off")
        sub_query_id = kwargs.get("sub_query_id")

        # Google CSE returns max 10 results per request
        # For more results, pagination with 'start' parameter would be needed
        max_results = min(max_results, 10)

        # Build query parameters
        params: dict[str, Any] = {
            "key": self._api_key,
            "cx": self._cx,
            "q": query,
            "num": max_results,
            "safe": safe,
        }

        if site_search:
            params["siteSearch"] = site_search
        if date_restrict:
            params["dateRestrict"] = date_restrict
        if file_type:
            params["fileType"] = file_type

        # Execute with retry logic
        response_data = await self._execute_with_retry(params)

        # Parse results
        return self._parse_response(response_data, sub_query_id)

    async def _execute_with_retry(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute API request with resilience stack.

        Uses execute_with_resilience for circuit breaker, rate limiting,
        and retry logic.

        Args:
            params: Query parameters

        Returns:
            Parsed JSON response

        Raises:
            AuthenticationError: If API key is invalid
            RateLimitError: If rate limit exceeded after all retries
            SearchProviderError: For other API errors
        """

        async def make_request() -> dict[str, Any]:
            """Inner function that makes the actual HTTP request."""
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(self._base_url, params=params)

                # Handle authentication errors (not retryable)
                if response.status_code == 401:
                    raise AuthenticationError(
                        provider="google",
                        message="Invalid API key",
                    )

                # Handle forbidden (invalid CSE ID or API not enabled)
                if response.status_code == 403:
                    error_data = self._parse_error_response(response)
                    # Check if it's a quota error (retryable) vs auth error (not retryable)
                    if "quota" in error_data.lower() or "limit" in error_data.lower():
                        retry_after = self._parse_retry_after(response)
                        raise RateLimitError(
                            provider="google",
                            retry_after=retry_after,
                            reason="quota",
                        )
                    # Non-quota 403 errors (bad CSE ID, API not enabled)
                    raise AuthenticationError(
                        provider="google",
                        message=f"Access denied: {error_data}",
                    )

                # Handle rate limiting (429)
                if response.status_code == 429:
                    retry_after = self._parse_retry_after(response)
                    raise RateLimitError(
                        provider="google",
                        retry_after=retry_after,
                    )

                # Handle other errors
                if response.status_code >= 400:
                    error_msg = self._parse_error_response(response)
                    raise SearchProviderError(
                        provider="google",
                        message=f"API error {response.status_code}: {error_msg}",
                        retryable=response.status_code >= 500,
                    )

                return response.json()

        try:
            time_budget = self._timeout * (self.resilience_config.max_retries + 1)
            return await execute_with_resilience(
                make_request,
                provider_name="google",
                time_budget=time_budget,
                classify_error=self.classify_error,
                manager=get_resilience_manager(),
                resilience_config=self.resilience_config,
            )
        except CircuitBreakerError as e:
            raise SearchProviderError(
                provider="google",
                message=f"Circuit breaker open: {e}",
                retryable=False,
            )
        except RateLimitWaitError as e:
            raise RateLimitError(
                provider="google",
                retry_after=e.wait_needed,
            )
        except TimeBudgetExceededError as e:
            raise SearchProviderError(
                provider="google",
                message=f"Request timed out: {e}",
                retryable=True,
            )
        except SearchProviderError:
            raise
        except Exception as e:
            classification = self.classify_error(e)
            raise SearchProviderError(
                provider="google",
                message=f"Request failed after retries: {e}",
                retryable=classification.retryable,
                original_error=e,
            )

    def _parse_retry_after(self, response: httpx.Response) -> Optional[float]:
        """Parse Retry-After header from response.

        Args:
            response: HTTP response

        Returns:
            Seconds to wait, or None if not provided
        """
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        return None

    def _parse_error_response(self, response: httpx.Response) -> str:
        """Extract error message from Google API error response.

        Google API returns errors in format:
        {
            "error": {
                "code": 403,
                "message": "...",
                "errors": [...]
            }
        }

        Args:
            response: HTTP response

        Returns:
            Error message string
        """
        try:
            data = response.json()
            error = data.get("error", {})
            if isinstance(error, dict):
                return error.get("message", str(error))
            return str(error)
        except Exception:
            return response.text[:200] if response.text else "Unknown error"

    def _parse_response(
        self,
        data: dict[str, Any],
        sub_query_id: Optional[str] = None,
    ) -> list[ResearchSource]:
        """Parse Google Custom Search API response into ResearchSource objects.

        Google CSE response structure:
        {
            "items": [
                {
                    "title": "...",
                    "link": "...",
                    "snippet": "...",
                    "displayLink": "example.com",
                    "pagemap": {
                        "metatags": [{"og:description": "...", "article:published_time": "..."}]
                    }
                }
            ],
            "searchInformation": {
                "totalResults": "123456"
            }
        }

        Args:
            data: Google CSE API response JSON
            sub_query_id: SubQuery ID for source tracking

        Returns:
            List of ResearchSource objects
        """
        sources: list[ResearchSource] = []
        items = data.get("items", [])

        for item in items:
            # Extract published date from pagemap metatags if available
            published_date = self._extract_published_date(item)

            # Create SearchResult from Google response
            search_result = SearchResult(
                url=item.get("link", ""),
                title=item.get("title", "Untitled"),
                snippet=item.get("snippet"),
                content=None,  # Google CSE doesn't provide full content
                score=None,  # Google CSE doesn't provide relevance scores
                published_date=published_date,
                source=item.get("displayLink"),
                metadata={
                    "google_cache_id": item.get("cacheId"),
                    "mime_type": item.get("mime"),
                    "file_format": item.get("fileFormat"),
                },
            )

            # Convert to ResearchSource
            research_source = search_result.to_research_source(
                source_type=SourceType.WEB,
                sub_query_id=sub_query_id,
            )
            sources.append(research_source)

        return sources

    def _extract_published_date(self, item: dict[str, Any]) -> Optional[datetime]:
        """Extract published date from Google CSE item pagemap.

        Looks for common metatag fields that contain publication dates:
        - article:published_time
        - datePublished
        - og:published_time
        - article:modified_time (fallback)

        Args:
            item: Single item from Google CSE response

        Returns:
            Parsed datetime or None
        """
        pagemap = item.get("pagemap", {})
        metatags = pagemap.get("metatags", [])

        if not metatags:
            return None

        # Metatags is a list, typically with one element
        tags = metatags[0] if metatags else {}

        # Try various date fields in order of preference
        date_fields = [
            "article:published_time",
            "datepublished",
            "og:published_time",
            "article:modified_time",
            "datemodified",
        ]

        for field in date_fields:
            date_str = tags.get(field)
            if date_str:
                parsed = self._parse_date(date_str)
                if parsed:
                    return parsed

        return None

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string from various formats.

        Args:
            date_str: Date string (ISO format or other common formats)

        Returns:
            Parsed datetime or None
        """
        if not date_str:
            return None

        # Try ISO format first
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except ValueError:
            pass

        # Try common date formats
        formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%B %d, %Y",
            "%b %d, %Y",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        return None

    async def health_check(self) -> bool:
        """Check if Google Custom Search API is accessible.

        Performs a lightweight search to verify API key, CSE ID, and connectivity.

        Returns:
            True if provider is healthy, False otherwise
        """
        try:
            # Perform minimal search to verify connectivity
            await self.search("test", max_results=1)
            return True
        except AuthenticationError:
            logger.error("Google CSE health check failed: invalid API key or CSE ID")
            return False
        except Exception as e:
            logger.warning(f"Google CSE health check failed: {e}")
            return False

    def classify_error(self, error: Exception) -> ErrorClassification:
        """Classify an error for resilience decisions.

        Maps Google-specific errors to ErrorClassification for unified
        retry and circuit breaker behavior. Special handling for 403 errors
        that may indicate quota exhaustion vs authentication issues.

        Args:
            error: The exception to classify

        Returns:
            ErrorClassification with retryable, trips_breaker, and error_type

        Classification rules:
            - 401 (AuthenticationError): not retryable, no breaker trip
            - 403 with 'quota' or 'limit' in error: retryable as QUOTA_EXCEEDED, no breaker trip
            - 403 without quota keywords: not retryable as AUTHENTICATION, no breaker trip
            - 429 (RateLimitError): retryable, no breaker trip, uses Retry-After
            - 5xx (server errors): retryable, trips breaker
            - 400 (bad request): not retryable, no breaker trip
            - Timeout/connection: retryable, trips breaker
        """
        # Handle our custom exception types
        if isinstance(error, AuthenticationError):
            return ErrorClassification(
                retryable=False,
                trips_breaker=False,
                error_type=ErrorType.AUTHENTICATION,
            )

        if isinstance(error, RateLimitError):
            # RateLimitError is used for both 429 and 403 quota errors
            # Prefer explicit reason when available, fallback to message parsing
            if getattr(error, "reason", None) == "quota":
                return ErrorClassification(
                    retryable=True,
                    trips_breaker=False,
                    backoff_seconds=error.retry_after,
                    error_type=ErrorType.QUOTA_EXCEEDED,
                )
            error_str = str(error).lower()
            if "quota" in error_str or "limit" in error_str:
                return ErrorClassification(
                    retryable=True,
                    trips_breaker=False,
                    backoff_seconds=error.retry_after,
                    error_type=ErrorType.QUOTA_EXCEEDED,
                )
            return ErrorClassification(
                retryable=True,
                trips_breaker=False,
                backoff_seconds=error.retry_after,
                error_type=ErrorType.RATE_LIMIT,
            )

        if isinstance(error, SearchProviderError):
            # Check if it's a server error (5xx) based on message
            error_str = str(error).lower()
            if any(code in error_str for code in ["500", "502", "503", "504"]):
                return ErrorClassification(
                    retryable=True,
                    trips_breaker=True,
                    error_type=ErrorType.SERVER_ERROR,
                )
            # 400 bad request - not retryable, don't trip breaker
            if "400" in error_str:
                return ErrorClassification(
                    retryable=False,
                    trips_breaker=False,
                    error_type=ErrorType.INVALID_REQUEST,
                )
            # Other SearchProviderError - use its retryable flag
            return ErrorClassification(
                retryable=error.retryable,
                trips_breaker=error.retryable,  # Trip breaker if retryable (transient)
                error_type=ErrorType.UNKNOWN,
            )

        # Handle httpx exceptions
        if isinstance(error, httpx.TimeoutException):
            return ErrorClassification(
                retryable=True,
                trips_breaker=True,
                error_type=ErrorType.TIMEOUT,
            )

        if isinstance(error, httpx.RequestError):
            # Network/connection errors
            return ErrorClassification(
                retryable=True,
                trips_breaker=True,
                error_type=ErrorType.NETWORK,
            )

        # Default: unknown error - not retryable, trips breaker
        return ErrorClassification(
            retryable=False,
            trips_breaker=True,
            error_type=ErrorType.UNKNOWN,
        )
