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
import re
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

# Valid parameter values
VALID_SEARCH_DEPTHS = frozenset(["basic", "advanced", "fast", "ultra_fast"])
VALID_TOPICS = frozenset(["general", "news"])


def _normalize_include_raw_content(value: bool | str) -> bool | str:
    """Normalize include_raw_content parameter for Tavily API.

    Args:
        value: The input value (bool or string).

    Returns:
        Normalized value for API: False, "markdown", or "text".

    Raises:
        ValueError: If value is not a valid option.
    """
    if value is True:
        return "markdown"  # True maps to markdown format
    if value is False:
        return False
    if isinstance(value, str) and value in ("markdown", "text"):
        return value
    raise ValueError(
        f"Invalid include_raw_content: {value!r}. "
        "Use bool or 'markdown'/'text'."
    )


def _validate_search_params(
    search_depth: str,
    topic: str,
    days: int | None,
    country: str | None,
    chunks_per_source: int | None,
) -> None:
    """Validate Tavily search parameters.

    Args:
        search_depth: Search depth level.
        topic: Search topic category.
        days: Days limit for news search.
        country: ISO country code.
        chunks_per_source: Chunks per source limit.

    Raises:
        ValueError: If any parameter is invalid.
    """
    if search_depth not in VALID_SEARCH_DEPTHS:
        raise ValueError(
            f"Invalid search_depth: {search_depth!r}. "
            f"Must be one of: {sorted(VALID_SEARCH_DEPTHS)}"
        )

    if topic not in VALID_TOPICS:
        raise ValueError(
            f"Invalid topic: {topic!r}. "
            f"Must be one of: {sorted(VALID_TOPICS)}"
        )

    if days is not None:
        if not isinstance(days, int) or days < 1 or days > 365:
            raise ValueError(
                f"Invalid days: {days!r}. Must be an integer between 1 and 365."
            )

    if country is not None:
        if not isinstance(country, str) or not re.match(r"^[A-Z]{2}$", country):
            raise ValueError(
                f"Invalid country: {country!r}. "
                "Must be a 2-letter uppercase ISO 3166-1 alpha-2 code (e.g., 'US', 'GB')."
            )

    if chunks_per_source is not None:
        if not isinstance(chunks_per_source, int) or chunks_per_source < 1 or chunks_per_source > 5:
            raise ValueError(
                f"Invalid chunks_per_source: {chunks_per_source!r}. "
                "Must be an integer between 1 and 5."
            )


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
        *,
        search_depth: str = "basic",
        topic: str = "general",
        days: int | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        include_answer: bool | str = False,
        include_raw_content: bool | str = False,
        include_images: bool = False,
        include_favicon: bool = False,
        country: str | None = None,
        chunks_per_source: int | None = None,
        auto_parameters: bool = False,
        sub_query_id: str | None = None,
        **kwargs: Any,
    ) -> list[ResearchSource]:
        """Execute a web search via Tavily API.

        Args:
            query: The search query string (max 400 characters).
            max_results: Maximum number of results to return (default: 10, max: 20).
            search_depth: Search depth level. Options:
                - "basic": Standard search (1 credit)
                - "advanced": Deeper search with better relevance (2 credits)
                - "fast": Quick search with reduced depth
                - "ultra_fast": Fastest search option
                Default: "basic"
            topic: Search topic category. Options:
                - "general": General web search (default)
                - "news": News-focused search (use with `days` parameter)
            days: Limit results to the last N days (1-365). Only applicable when
                topic="news". Default: None (no time limit).
            include_domains: List of domains to restrict search to (max 300).
                Example: ["arxiv.org", "github.com"]
            exclude_domains: List of domains to exclude from results (max 150).
                Example: ["pinterest.com", "facebook.com"]
            include_answer: Whether to include an AI-generated answer. Options:
                - False: No answer (default)
                - True or "basic": Include basic AI answer
                - "advanced": Include detailed AI answer
            include_raw_content: Whether to include full page content. Options:
                - False: No raw content (default)
                - True or "markdown": Include content as markdown
                - "text": Include content as plain text
            include_images: Whether to include image results (default: False).
            include_favicon: Whether to include favicon URLs for each result
                (default: False).
            country: ISO 3166-1 alpha-2 country code to boost results from
                (e.g., "US", "GB", "DE"). Default: None (no country boost).
            chunks_per_source: Number of content chunks per source (1-5).
                Only applicable with search_depth="advanced". Default: 3.
            auto_parameters: Let Tavily auto-configure parameters based on
                query intent (default: False). Explicit parameters override
                auto-configured values.
            sub_query_id: SubQuery ID for source tracking in deep research
                workflows. Used internally to associate results with sub-queries.
            **kwargs: Additional parameters for forward compatibility.

        Returns:
            List of ResearchSource objects containing search results.

        Raises:
            AuthenticationError: If API key is invalid.
            RateLimitError: If rate limit exceeded after all retries.
            SearchProviderError: For other API errors.

        Example:
            # Basic search
            results = await provider.search("python tutorials", max_results=5)

            # Advanced search with domain filtering
            results = await provider.search(
                "machine learning papers",
                max_results=10,
                search_depth="advanced",
                include_domains=["arxiv.org", "paperswithcode.com"],
                include_raw_content="markdown",
            )

            # News search with time limit
            results = await provider.search(
                "AI regulations",
                topic="news",
                days=7,
                country="US",
            )
        """
        # Validate parameters
        _validate_search_params(
            search_depth=search_depth,
            topic=topic,
            days=days,
            country=country,
            chunks_per_source=chunks_per_source,
        )

        # Clamp max_results to Tavily's limit
        max_results = min(max_results, 20)

        # Normalize include_raw_content (True -> "markdown")
        normalized_raw_content = _normalize_include_raw_content(include_raw_content)

        # Build request payload with required parameters
        payload: dict[str, Any] = {
            "api_key": self._api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "topic": topic,
            "include_answer": include_answer,
            "include_raw_content": normalized_raw_content,
            "include_images": include_images,
            "include_favicon": include_favicon,
        }

        # Conditionally include optional parameters only when set
        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains
        if days is not None:
            payload["days"] = days
        if country is not None:
            payload["country"] = country
        if chunks_per_source is not None:
            payload["chunks_per_source"] = chunks_per_source
        if auto_parameters:
            payload["auto_parameters"] = auto_parameters

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
