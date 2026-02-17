"""Tavily search provider for web search.

This module implements TavilySearchProvider, which wraps the Tavily Search API
to provide web search capabilities for the deep research workflow.

Tavily API documentation: https://docs.tavily.com/

Resilience Configuration:
    - Rate Limit: 1 RPS with burst limit of 3
    - Circuit Breaker: Opens after 5 failures, 30s recovery timeout
    - Retry: Up to 3 retries with exponential backoff (1-60s)
    - Error Handling:
        - 429: Retryable, does NOT trip circuit breaker
        - 401: Not retryable, does NOT trip circuit breaker
        - 5xx: Retryable, trips circuit breaker
        - Timeouts: Retryable, trips circuit breaker

Example usage:
    provider = TavilySearchProvider(api_key="tvly-...")
    sources = await provider.search("machine learning trends", max_results=5)
"""

import logging
import os
import re
from dataclasses import replace
from typing import Any, Optional

import httpx

from foundry_mcp.core.research.models.sources import ResearchSource, SourceType
from foundry_mcp.core.errors.search import (
    AuthenticationError,
    RateLimitError,
    SearchProviderError,
)
from foundry_mcp.core.research.providers.base import (
    SearchProvider,
    SearchResult,
)
from foundry_mcp.core.research.providers.resilience import (
    ErrorClassification,
    ProviderResilienceConfig,
    get_provider_config,
)
from foundry_mcp.core.research.providers.shared import (
    check_provider_health,
    classify_http_error,
    create_resilience_executor,
    extract_domain,
    extract_error_message,
    parse_iso_date,
    parse_retry_after,
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
        resilience_config: Optional[ProviderResilienceConfig] = None,
    ):
        """Initialize Tavily search provider.

        Args:
            api_key: Tavily API key. If not provided, reads from TAVILY_API_KEY env var.
            base_url: API base URL (default: https://api.tavily.com)
            timeout: Request timeout in seconds (default: 30.0)
            max_retries: Maximum retry attempts for rate limits (default: 3)
            resilience_config: Custom resilience configuration. If None, uses
                defaults from PROVIDER_CONFIGS["tavily"].

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
        if resilience_config is None:
            self._resilience_config = replace(
                get_provider_config("tavily"),
                max_retries=max_retries,
            )
        else:
            self._resilience_config = resilience_config

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

    @property
    def resilience_config(self) -> ProviderResilienceConfig:
        """Return the resilience configuration for this provider.

        Returns ProviderResilienceConfig for Tavily with settings for:
        - Rate limiting (requests per second, burst limit)
        - Retry behavior (max retries, delays, jitter)
        - Circuit breaker (failure threshold, recovery timeout)

        If a custom config was provided via constructor, returns that.
        Otherwise, returns defaults from PROVIDER_CONFIGS["tavily"].

        Returns:
            ProviderResilienceConfig for this provider
        """
        if self._resilience_config is not None:
            return self._resilience_config
        return get_provider_config("tavily")

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
        """Execute API request with resilience stack.

        Uses shared resilience executor for circuit breaker, rate limiting,
        and retry logic.

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

        async def make_request() -> dict[str, Any]:
            """Inner function that makes the actual HTTP request."""
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
                    retry_after = parse_retry_after(response)
                    raise RateLimitError(
                        provider="tavily",
                        retry_after=retry_after,
                    )

                # Handle other errors
                if response.status_code >= 400:
                    error_msg = extract_error_message(response)
                    raise SearchProviderError(
                        provider="tavily",
                        message=f"API error {response.status_code}: {error_msg}",
                        retryable=response.status_code >= 500,
                    )

                return response.json()

        executor = create_resilience_executor(
            "tavily", self.resilience_config, self.classify_error,
        )
        return await executor(make_request, timeout=self._timeout)

    def _parse_retry_after(self, response: httpx.Response) -> Optional[float]:
        """Parse Retry-After header from response.

        Delegates to shared utility. Retained for interface compatibility.
        """
        return parse_retry_after(response)

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
                published_date=parse_iso_date(result.get("published_date")),
                source=extract_domain(result.get("url", "")),
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

    async def health_check(self) -> bool:
        """Check if Tavily API is accessible.

        Performs a lightweight search to verify API key and connectivity.

        Returns:
            True if provider is healthy, False otherwise
        """
        return await check_provider_health(
            "tavily",
            self._api_key,
            self._base_url,
            test_func=lambda: self.search("test", max_results=1),
        )

    def classify_error(self, error: Exception) -> ErrorClassification:
        """Classify an error for resilience decisions.

        Maps Tavily-specific errors to ErrorClassification for unified
        retry and circuit breaker behavior.

        Args:
            error: The exception to classify

        Returns:
            ErrorClassification with retryable, trips_breaker, and error_type
        """
        return classify_http_error(error, "tavily")
