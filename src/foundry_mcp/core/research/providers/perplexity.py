"""Perplexity Search API provider for web search.

This module implements PerplexitySearchProvider, which wraps the Perplexity Search API
to provide web search capabilities for the deep research workflow.

Perplexity Search API documentation: https://docs.perplexity.ai/api-reference/search-post

Example usage:
    provider = PerplexitySearchProvider(api_key="pplx-...")
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

# Perplexity API constants
PERPLEXITY_API_BASE_URL = "https://api.perplexity.ai"
PERPLEXITY_SEARCH_ENDPOINT = "/search"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RATE_LIMIT = 1.0  # requests per second

# Valid search_context_size values for search API
VALID_SEARCH_CONTEXT_SIZES = frozenset(["low", "medium", "high"])


def _validate_search_params(
    search_context_size: str | None,
    max_tokens: int | None,
    max_tokens_per_page: int | None,
    search_after_date: str | None,
    search_before_date: str | None,
    recency_filter: str | None,
    last_updated_after_filter: str | None = None,
    last_updated_before_filter: str | None = None,
) -> None:
    """Validate Perplexity search parameters.

    Args:
        search_context_size: Context size for search ('low', 'medium', 'high').
        max_tokens: Maximum tokens for response content.
        max_tokens_per_page: Maximum tokens per page.
        search_after_date: Filter results after this date (MM/DD/YYYY).
        search_before_date: Filter results before this date (MM/DD/YYYY).
        recency_filter: Time filter ('day', 'week', 'month', 'year').
        last_updated_after_filter: Filter by content modified after this date (MM/DD/YYYY).
        last_updated_before_filter: Filter by content modified before this date (MM/DD/YYYY).

    Raises:
        ValueError: If any parameter is invalid.
    """
    if search_context_size is not None:
        if search_context_size not in VALID_SEARCH_CONTEXT_SIZES:
            raise ValueError(
                f"Invalid search_context_size: {search_context_size!r}. "
                f"Must be one of: {sorted(VALID_SEARCH_CONTEXT_SIZES)}"
            )

    if max_tokens is not None:
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError(
                f"Invalid max_tokens: {max_tokens!r}. Must be a positive integer."
            )

    if max_tokens_per_page is not None:
        if not isinstance(max_tokens_per_page, int) or max_tokens_per_page < 1:
            raise ValueError(
                f"Invalid max_tokens_per_page: {max_tokens_per_page!r}. "
                "Must be a positive integer."
            )

    # Parse and validate dates
    parsed_after = None
    parsed_before = None

    if search_after_date is not None:
        try:
            parsed_after = datetime.strptime(search_after_date, "%m/%d/%Y")
        except ValueError:
            raise ValueError(
                f"Invalid search_after_date: {search_after_date!r}. "
                "Must be in MM/DD/YYYY format."
            )

    if search_before_date is not None:
        try:
            parsed_before = datetime.strptime(search_before_date, "%m/%d/%Y")
        except ValueError:
            raise ValueError(
                f"Invalid search_before_date: {search_before_date!r}. "
                "Must be in MM/DD/YYYY format."
            )

    # Validate date range logic
    if parsed_after is not None and parsed_before is not None:
        if parsed_after >= parsed_before:
            raise ValueError(
                f"search_after_date ({search_after_date}) must be before "
                f"search_before_date ({search_before_date})."
            )

    # Validate last_updated date filters
    parsed_last_updated_after = None
    parsed_last_updated_before = None

    if last_updated_after_filter is not None:
        try:
            parsed_last_updated_after = datetime.strptime(last_updated_after_filter, "%m/%d/%Y")
        except ValueError:
            raise ValueError(
                f"Invalid last_updated_after_filter: {last_updated_after_filter!r}. "
                "Must be in MM/DD/YYYY format."
            )

    if last_updated_before_filter is not None:
        try:
            parsed_last_updated_before = datetime.strptime(last_updated_before_filter, "%m/%d/%Y")
        except ValueError:
            raise ValueError(
                f"Invalid last_updated_before_filter: {last_updated_before_filter!r}. "
                "Must be in MM/DD/YYYY format."
            )

    # Validate last_updated date range logic
    if parsed_last_updated_after is not None and parsed_last_updated_before is not None:
        if parsed_last_updated_after >= parsed_last_updated_before:
            raise ValueError(
                f"last_updated_after_filter ({last_updated_after_filter}) must be before "
                f"last_updated_before_filter ({last_updated_before_filter})."
            )

    # Validate recency_filter exclusivity with date filters
    if recency_filter is not None:
        valid_recency_filters = {"day", "week", "month", "year"}
        if recency_filter not in valid_recency_filters:
            raise ValueError(
                f"Invalid recency_filter: {recency_filter!r}. "
                f"Must be one of: {sorted(valid_recency_filters)}."
            )
        if search_after_date is not None or search_before_date is not None:
            raise ValueError(
                "Cannot use recency_filter with search_after_date or search_before_date. "
                "Use either recency_filter OR date filters, not both."
            )


class PerplexitySearchProvider(SearchProvider):
    """Perplexity Search API provider for web search.

    Wraps the Perplexity Search API to provide web search capabilities.
    Supports domain filtering, recency filtering, and geographic targeting.

    Pricing: $5 per 1,000 requests

    Attributes:
        api_key: Perplexity API key (required)
        base_url: API base URL (default: https://api.perplexity.ai)
        timeout: Request timeout in seconds (default: 30.0)
        max_retries: Maximum retry attempts for rate limits (default: 3)

    Example:
        provider = PerplexitySearchProvider(api_key="pplx-...")
        sources = await provider.search(
            "AI trends 2024",
            max_results=10,
            recency_filter="week",
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = PERPLEXITY_API_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        """Initialize Perplexity search provider.

        Args:
            api_key: Perplexity API key. If not provided, reads from PERPLEXITY_API_KEY env var.
            base_url: API base URL (default: https://api.perplexity.ai)
            timeout: Request timeout in seconds (default: 30.0)
            max_retries: Maximum retry attempts for rate limits (default: 3)

        Raises:
            ValueError: If no API key is provided or found in environment
        """
        self._api_key = api_key or os.environ.get("PERPLEXITY_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Perplexity API key required. Provide via api_key parameter "
                "or PERPLEXITY_API_KEY environment variable."
            )

        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._rate_limit_value = DEFAULT_RATE_LIMIT

    def get_provider_name(self) -> str:
        """Return the provider identifier.

        Returns:
            "perplexity"
        """
        return "perplexity"

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
        """Execute a web search via Perplexity Search API.

        Args:
            query: The search query string
            max_results: Maximum number of results to return (default: 10, max: 20)
            **kwargs: Additional Perplexity options:
                - recency_filter: Time filter ('day', 'week', 'month', 'year').
                    Cannot be combined with date filters.
                - domain_filter: List of domains to include (max 20). Prefix with
                    '-' to exclude (e.g., ['-example.com'] excludes example.com).
                - country: Geographic filter ('US', 'GB', etc.)
                - sub_query_id: SubQuery ID for source tracking
                - include_raw_content: If True, map snippet to content field
                - search_context_size: Context size for search results
                    ('low', 'medium', 'high'). Default: 'medium'
                - max_tokens: Maximum total tokens for response (default: 50000)
                - max_tokens_per_page: Maximum tokens per page (default: 2048)
                - search_after_date: Filter results after this date (MM/DD/YYYY format)
                - search_before_date: Filter results before this date (MM/DD/YYYY format)
                - last_updated_after_filter: Filter by content modified after this date
                    (MM/DD/YYYY format). Filters by modification date, not publication.
                - last_updated_before_filter: Filter by content modified before this date
                    (MM/DD/YYYY format). Filters by modification date, not publication.

        Returns:
            List of ResearchSource objects

        Raises:
            AuthenticationError: If API key is invalid
            RateLimitError: If rate limit exceeded after all retries
            SearchProviderError: For other API errors
            ValueError: If parameter validation fails (invalid search_context_size,
                non-positive max_tokens/max_tokens_per_page, invalid recency_filter,
                invalid date format, or conflicting filters)
        """
        # Extract Perplexity-specific options
        recency_filter = kwargs.get("recency_filter")
        domain_filter = kwargs.get("domain_filter", [])
        country = kwargs.get("country")
        sub_query_id = kwargs.get("sub_query_id")
        include_raw_content = kwargs.get("include_raw_content", False)

        # Extract new configurable parameters with defaults
        search_context_size = kwargs.get("search_context_size", "medium")
        max_tokens = kwargs.get("max_tokens", 50000)
        max_tokens_per_page = kwargs.get("max_tokens_per_page", 2048)
        search_after_date = kwargs.get("search_after_date")
        search_before_date = kwargs.get("search_before_date")
        last_updated_after_filter = kwargs.get("last_updated_after_filter")
        last_updated_before_filter = kwargs.get("last_updated_before_filter")

        # Validate parameters
        _validate_search_params(
            search_context_size=search_context_size,
            max_tokens=max_tokens,
            max_tokens_per_page=max_tokens_per_page,
            search_after_date=search_after_date,
            search_before_date=search_before_date,
            recency_filter=recency_filter,
            last_updated_after_filter=last_updated_after_filter,
            last_updated_before_filter=last_updated_before_filter,
        )

        # Clamp max_results to Perplexity's limit (1-20)
        max_results = max(1, min(max_results, 20))

        # Build request payload
        payload: dict[str, Any] = {
            "query": query,
            "max_results": max_results,
            "max_tokens": max_tokens,
            "max_tokens_per_page": max_tokens_per_page,
            "search_context_size": search_context_size,
        }

        if recency_filter and recency_filter in ("day", "week", "month", "year"):
            payload["search_recency_filter"] = recency_filter
        if domain_filter:
            # Perplexity allows max 20 domains
            payload["search_domain_filter"] = domain_filter[:20]
        if country:
            payload["country"] = country
        if search_after_date:
            payload["search_after_date"] = search_after_date
        if search_before_date:
            payload["search_before_date"] = search_before_date
        if last_updated_after_filter:
            payload["last_updated_after_filter"] = last_updated_after_filter
        if last_updated_before_filter:
            payload["last_updated_before_filter"] = last_updated_before_filter

        # Execute with retry logic
        response_data = await self._execute_with_retry(payload)

        # Parse results
        return self._parse_response(response_data, sub_query_id, include_raw_content)

    async def _execute_with_retry(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute API request with exponential backoff retry.

        Args:
            payload: Request payload

        Returns:
            Parsed JSON response

        Raises:
            AuthenticationError: If API key is invalid
            RateLimitError: If rate limit exceeded after all retries
            SearchProviderError: For other API errors
        """
        url = f"{self._base_url}{PERPLEXITY_SEARCH_ENDPOINT}"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(url, json=payload, headers=headers)

                    # Handle authentication errors (not retryable)
                    if response.status_code == 401:
                        raise AuthenticationError(
                            provider="perplexity",
                            message="Invalid API key",
                        )

                    # Handle rate limiting
                    if response.status_code == 429:
                        retry_after = self._parse_retry_after(response)
                        if attempt < self._max_retries - 1:
                            wait_time = retry_after or (2**attempt)
                            logger.warning(
                                f"Perplexity rate limit hit, waiting {wait_time}s "
                                f"(attempt {attempt + 1}/{self._max_retries})"
                            )
                            await asyncio.sleep(wait_time)
                            continue
                        raise RateLimitError(
                            provider="perplexity",
                            retry_after=retry_after,
                        )

                    # Handle server errors (retryable)
                    if response.status_code >= 500:
                        if attempt < self._max_retries - 1:
                            wait_time = 2**attempt
                            logger.warning(
                                f"Perplexity server error {response.status_code}, "
                                f"retrying in {wait_time}s "
                                f"(attempt {attempt + 1}/{self._max_retries})"
                            )
                            await asyncio.sleep(wait_time)
                            continue
                        error_msg = self._extract_error_message(response)
                        raise SearchProviderError(
                            provider="perplexity",
                            message=f"API error {response.status_code}: {error_msg}",
                            retryable=True,
                        )

                    # Handle other errors
                    if response.status_code >= 400:
                        error_msg = self._extract_error_message(response)
                        raise SearchProviderError(
                            provider="perplexity",
                            message=f"API error {response.status_code}: {error_msg}",
                            retryable=False,
                        )

                    return response.json()

            except httpx.TimeoutException as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Perplexity request timeout, retrying in {wait_time}s "
                        f"(attempt {attempt + 1}/{self._max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                    continue

            except httpx.RequestError as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Perplexity request error: {e}, retrying in {wait_time}s "
                        f"(attempt {attempt + 1}/{self._max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                    continue

            except (AuthenticationError, RateLimitError, SearchProviderError):
                raise

        # All retries exhausted
        raise SearchProviderError(
            provider="perplexity",
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

    def _extract_error_message(self, response: httpx.Response) -> str:
        """Extract error message from response.

        Args:
            response: HTTP response

        Returns:
            Error message string
        """
        try:
            data = response.json()
            return data.get("error", data.get("message", response.text[:200]))
        except Exception:
            return response.text[:200] if response.text else "Unknown error"

    def _parse_response(
        self,
        data: dict[str, Any],
        sub_query_id: Optional[str] = None,
        include_raw_content: bool = False,
    ) -> list[ResearchSource]:
        """Parse Perplexity API response into ResearchSource objects.

        Perplexity Search API response structure:
        {
            "results": [
                {
                    "title": "...",
                    "url": "...",
                    "snippet": "...",
                    "date": "...",
                    "last_updated": "..."
                }
            ]
        }

        Args:
            data: Perplexity API response JSON
            sub_query_id: SubQuery ID for source tracking
            include_raw_content: If True, map snippet to content field

        Returns:
            List of ResearchSource objects
        """
        sources: list[ResearchSource] = []
        results = data.get("results", [])

        for result in results:
            # Parse date - try both 'date' and 'last_updated' fields
            published_date = self._parse_date(
                result.get("date") or result.get("last_updated")
            )

            # Create SearchResult from Perplexity response
            # Map snippet to content when include_raw_content is requested
            search_result = SearchResult(
                url=result.get("url", ""),
                title=result.get("title", "Untitled"),
                snippet=result.get("snippet"),
                content=result.get("snippet") if include_raw_content else None,
                score=None,  # Perplexity doesn't provide relevance scores
                published_date=published_date,
                source=self._extract_domain(result.get("url", "")),
                metadata={
                    "perplexity_date": result.get("date"),
                    "perplexity_last_updated": result.get("last_updated"),
                },
            )

            # Convert to ResearchSource
            research_source = search_result.to_research_source(
                source_type=SourceType.WEB,
                sub_query_id=sub_query_id,
            )
            sources.append(research_source)

        return sources

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string from Perplexity response.

        Args:
            date_str: ISO format date string or other common formats

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

    def _extract_domain(self, url: str) -> Optional[str]:
        """Extract domain from URL.

        Args:
            url: Full URL

        Returns:
            Domain name or None
        """
        if not url:
            return None
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            return parsed.netloc or None
        except Exception:
            return None

    async def health_check(self) -> bool:
        """Check if Perplexity API is accessible.

        Performs a lightweight search to verify API key and connectivity.

        Returns:
            True if provider is healthy, False otherwise
        """
        try:
            # Perform minimal search to verify connectivity
            await self.search("test", max_results=1)
            return True
        except AuthenticationError:
            logger.error("Perplexity health check failed: invalid API key")
            return False
        except Exception as e:
            logger.warning(f"Perplexity health check failed: {e}")
            return False
