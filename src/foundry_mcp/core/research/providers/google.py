"""Google Custom Search provider for web search.

This module implements GoogleSearchProvider, which wraps the Google Custom Search
JSON API to provide web search capabilities for the deep research workflow.

Google Custom Search API documentation:
https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list

Example usage:
    provider = GoogleSearchProvider(
        api_key="AIza...",
        cx="017576662512468239146:omuauf_lfve",
    )
    sources = await provider.search("machine learning trends", max_results=5)
"""

import asyncio
import logging
import os
from datetime import datetime
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
    ):
        """Initialize Google Custom Search provider.

        Args:
            api_key: Google API key. If not provided, reads from GOOGLE_API_KEY env var.
            cx: Custom Search Engine ID. If not provided, reads from GOOGLE_CSE_ID env var.
            base_url: API base URL (default: https://www.googleapis.com/customsearch/v1)
            timeout: Request timeout in seconds (default: 30.0)
            max_retries: Maximum retry attempts for rate limits (default: 3)

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
        """Execute API request with exponential backoff retry.

        Args:
            params: Query parameters

        Returns:
            Parsed JSON response

        Raises:
            AuthenticationError: If API key is invalid
            RateLimitError: If rate limit exceeded after all retries
            SearchProviderError: For other API errors
        """
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries):
            try:
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
                            if attempt < self._max_retries - 1:
                                wait_time = retry_after or (2**attempt)
                                logger.warning(
                                    f"Google CSE quota limit hit, waiting {wait_time}s "
                                    f"(attempt {attempt + 1}/{self._max_retries})"
                                )
                                await asyncio.sleep(wait_time)
                                continue
                            raise RateLimitError(
                                provider="google",
                                retry_after=retry_after,
                            )
                        # Non-quota 403 errors (bad CSE ID, API not enabled)
                        raise AuthenticationError(
                            provider="google",
                            message=f"Access denied: {error_data}",
                        )

                    # Handle rate limiting (429)
                    if response.status_code == 429:
                        retry_after = self._parse_retry_after(response)
                        if attempt < self._max_retries - 1:
                            wait_time = retry_after or (2**attempt)
                            logger.warning(
                                f"Google CSE rate limit hit, waiting {wait_time}s "
                                f"(attempt {attempt + 1}/{self._max_retries})"
                            )
                            await asyncio.sleep(wait_time)
                            continue
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

            except httpx.TimeoutException as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Google CSE request timeout, retrying in {wait_time}s "
                        f"(attempt {attempt + 1}/{self._max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                    continue

            except httpx.RequestError as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Google CSE request error: {e}, retrying in {wait_time}s "
                        f"(attempt {attempt + 1}/{self._max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                    continue

            except (AuthenticationError, RateLimitError, SearchProviderError):
                raise

        # All retries exhausted
        raise SearchProviderError(
            provider="google",
            message=f"Request failed after {self._max_retries} attempts",
            retryable=False,
            original_error=last_error,
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
