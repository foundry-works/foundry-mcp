"""Tavily search provider for web search.

This module implements TavilySearchProvider, which wraps the Tavily Search API
to provide web search capabilities for the deep research workflow.

Tavily API documentation: https://docs.tavily.com/

Example usage:
    provider = TavilySearchProvider(api_key="tvly-...")
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

# Tavily API constants
TAVILY_API_BASE_URL = "https://api.tavily.com"
TAVILY_SEARCH_ENDPOINT = "/search"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RATE_LIMIT = 1.0  # requests per second


class TavilySearchProvider(SearchProvider):
    """Tavily Search API provider for web search.

    Wraps the Tavily Search API to provide web search capabilities.
    Supports basic and advanced search depths, domain filtering,
    and automatic content extraction.

    Attributes:
        api_key: Tavily API key (required)
        base_url: API base URL (default: https://api.tavily.com)
        timeout: Request timeout in seconds (default: 30.0)
        max_retries: Maximum retry attempts for rate limits (default: 3)

    Example:
        provider = TavilySearchProvider(api_key="tvly-...")
        sources = await provider.search(
            "AI trends 2024",
            max_results=5,
            search_depth="advanced",
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = TAVILY_API_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        """Initialize Tavily search provider.

        Args:
            api_key: Tavily API key. If not provided, reads from TAVILY_API_KEY env var.
            base_url: API base URL (default: https://api.tavily.com)
            timeout: Request timeout in seconds (default: 30.0)
            max_retries: Maximum retry attempts for rate limits (default: 3)

        Raises:
            ValueError: If no API key is provided or found in environment
        """
        self._api_key = api_key or os.environ.get("TAVILY_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Tavily API key required. Provide via api_key parameter "
                "or TAVILY_API_KEY environment variable."
            )

        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._rate_limit_value = DEFAULT_RATE_LIMIT

    def get_provider_name(self) -> str:
        """Return the provider identifier.

        Returns:
            "tavily"
        """
        return "tavily"

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
        """Execute a web search via Tavily API.

        Args:
            query: The search query string
            max_results: Maximum number of results to return (default: 10, max: 20)
            **kwargs: Additional Tavily options:
                - search_depth: "basic" or "advanced" (default: "basic")
                - include_domains: List of domains to include
                - exclude_domains: List of domains to exclude
                - include_answer: Whether to include AI answer (default: False)
                - include_raw_content: Whether to include raw HTML (default: False)
                - sub_query_id: SubQuery ID for source tracking

        Returns:
            List of ResearchSource objects

        Raises:
            AuthenticationError: If API key is invalid
            RateLimitError: If rate limit exceeded after all retries
            SearchProviderError: For other API errors
        """
        # Extract Tavily-specific options
        search_depth = kwargs.get("search_depth", "basic")
        include_domains = kwargs.get("include_domains", [])
        exclude_domains = kwargs.get("exclude_domains", [])
        include_answer = kwargs.get("include_answer", False)
        include_raw_content = kwargs.get("include_raw_content", False)
        sub_query_id = kwargs.get("sub_query_id")

        # Clamp max_results to Tavily's limit
        max_results = min(max_results, 20)

        # Build request payload
        payload = {
            "api_key": self._api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
        }

        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains

        # Execute with retry logic
        response_data = await self._execute_with_retry(payload)

        # Parse results
        return self._parse_response(response_data, sub_query_id)

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
        url = f"{self._base_url}{TAVILY_SEARCH_ENDPOINT}"
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(url, json=payload)

                    # Handle authentication errors (not retryable)
                    if response.status_code == 401:
                        raise AuthenticationError(
                            provider="tavily",
                            message="Invalid API key",
                        )

                    # Handle rate limiting
                    if response.status_code == 429:
                        retry_after = self._parse_retry_after(response)
                        if attempt < self._max_retries - 1:
                            wait_time = retry_after or (2**attempt)
                            logger.warning(
                                f"Tavily rate limit hit, waiting {wait_time}s "
                                f"(attempt {attempt + 1}/{self._max_retries})"
                            )
                            await asyncio.sleep(wait_time)
                            continue
                        raise RateLimitError(
                            provider="tavily",
                            retry_after=retry_after,
                        )

                    # Handle other errors
                    if response.status_code >= 400:
                        error_msg = self._extract_error_message(response)
                        raise SearchProviderError(
                            provider="tavily",
                            message=f"API error {response.status_code}: {error_msg}",
                            retryable=response.status_code >= 500,
                        )

                    return response.json()

            except httpx.TimeoutException as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Tavily request timeout, retrying in {wait_time}s "
                        f"(attempt {attempt + 1}/{self._max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                    continue

            except httpx.RequestError as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Tavily request error: {e}, retrying in {wait_time}s "
                        f"(attempt {attempt + 1}/{self._max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                    continue

            except (AuthenticationError, RateLimitError, SearchProviderError):
                raise

        # All retries exhausted
        raise SearchProviderError(
            provider="tavily",
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
    ) -> list[ResearchSource]:
        """Parse Tavily API response into ResearchSource objects.

        Args:
            data: Tavily API response JSON
            sub_query_id: SubQuery ID for source tracking

        Returns:
            List of ResearchSource objects
        """
        sources: list[ResearchSource] = []
        results = data.get("results", [])

        for result in results:
            # Create SearchResult from Tavily response
            search_result = SearchResult(
                url=result.get("url", ""),
                title=result.get("title", "Untitled"),
                snippet=result.get("content"),  # Tavily uses "content" for snippet
                content=result.get("raw_content"),  # Full content if requested
                score=result.get("score"),
                published_date=self._parse_date(result.get("published_date")),
                source=self._extract_domain(result.get("url", "")),
                metadata={
                    "tavily_score": result.get("score"),
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
        """Parse date string from Tavily response.

        Args:
            date_str: ISO format date string

        Returns:
            Parsed datetime or None
        """
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except ValueError:
            return None

    def _extract_domain(self, url: str) -> Optional[str]:
        """Extract domain from URL.

        Args:
            url: Full URL

        Returns:
            Domain name or None
        """
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            return parsed.netloc
        except Exception:
            return None

    async def health_check(self) -> bool:
        """Check if Tavily API is accessible.

        Performs a lightweight search to verify API key and connectivity.

        Returns:
            True if provider is healthy, False otherwise
        """
        try:
            # Perform minimal search to verify connectivity
            await self.search("test", max_results=1)
            return True
        except AuthenticationError:
            logger.error("Tavily health check failed: invalid API key")
            return False
        except Exception as e:
            logger.warning(f"Tavily health check failed: {e}")
            return False
