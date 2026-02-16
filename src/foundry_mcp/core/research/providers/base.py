"""Abstract base class for search providers.

This module defines the SearchProvider interface that all concrete
search providers must implement. The interface enables dependency
injection and easy mocking for testing.

Resilience Features:
    All search providers integrate with the resilience layer which provides:
    - **Rate Limiting**: Per-provider token bucket rate limiting with
      configurable requests per second and burst limits
    - **Circuit Breaker**: Automatic failure detection with CLOSED -> OPEN ->
      HALF_OPEN state transitions for graceful degradation
    - **Retry with Backoff**: Exponential backoff with jitter for transient
      failures (429s, 5xx errors, timeouts)
    - **Error Classification**: The `classify_error()` hook enables
      provider-specific error handling decisions

    See `foundry_mcp.core.research.providers.resilience` for configuration.

Example usage:
    class TavilySearchProvider(SearchProvider):
        def get_provider_name(self) -> str:
            return "tavily"

        async def search(
            self,
            query: str,
            max_results: int = 10,
            **kwargs: Any,
        ) -> list[ResearchSource]:
            # Implementation...
            pass
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from foundry_mcp.core.research.models import (
    ResearchSource,
    SourceQuality,
    SourceType,
)

if TYPE_CHECKING:
    from foundry_mcp.core.research.providers.resilience import ErrorClassification


@dataclass(frozen=True)
class SearchResult:
    """Normalized search result from any provider.

    This dataclass provides a common structure for raw search results
    before they are converted to ResearchSource objects. It captures
    the essential fields returned by search APIs.

    Attributes:
        url: URL of the search result
        title: Title or headline of the result
        snippet: Brief excerpt or description
        content: Full content if available (e.g., from Tavily's extract)
        score: Relevance score from the search provider (0.0-1.0)
        published_date: Publication date if available
        source: Source domain or publication name
        metadata: Additional provider-specific metadata
    """

    url: str
    title: str
    snippet: Optional[str] = None
    content: Optional[str] = None
    score: Optional[float] = None
    published_date: Optional[datetime] = None
    source: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_research_source(
        self,
        source_type: SourceType = SourceType.WEB,
        sub_query_id: Optional[str] = None,
    ) -> ResearchSource:
        """Convert this search result to a ResearchSource.

        Args:
            source_type: Type of source (WEB, ACADEMIC, etc.)
            sub_query_id: ID of the SubQuery that initiated this search

        Returns:
            ResearchSource object with quality set to UNKNOWN (to be assessed later)
        """
        return ResearchSource(
            url=self.url,
            title=self.title,
            source_type=source_type,
            quality=SourceQuality.UNKNOWN,
            snippet=self.snippet,
            content=self.content,
            sub_query_id=sub_query_id,
            metadata={
                **self.metadata,
                "score": self.score,
                "published_date": (
                    self.published_date.isoformat() if self.published_date else None
                ),
                "source": self.source,
            },
        )


class SearchProvider(ABC):
    """Abstract base class for search providers.

    All concrete search providers (Tavily, Google, SemanticScholar) must
    implement this interface. This enables:
    - Dependency injection for flexible provider selection
    - Easy mocking for unit testing
    - Consistent API across different search backends

    Subclasses should:
    - Implement get_provider_name() to return a unique identifier
    - Implement search() to execute queries against the provider
    - Optionally override rate_limit property for rate limiting config
    - Optionally override classify_error() for provider-specific error handling

    Resilience Integration:
        Providers are wrapped by `execute_with_resilience()` when called from
        the deep research workflow. This provides automatic:
        - Circuit breaker protection (opens after 5 consecutive failures)
        - Rate limiting (per-provider token bucket, default 1 RPS)
        - Retry with exponential backoff and jitter for transient errors
        - Time budget enforcement with cancellation support

        The `classify_error()` method determines how errors are handled:
        - `retryable=True`: Error will trigger retry with backoff
        - `trips_breaker=True`: Error counts toward circuit breaker threshold
        - `error_type`: Categorizes error for metrics and logging

        Override `classify_error()` in subclasses to customize error handling
        based on provider-specific HTTP status codes or error responses.

    Example:
        provider = TavilySearchProvider(api_key="...")
        sources = await provider.search("machine learning trends", max_results=5)
    """

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the unique identifier for this provider.

        Returns:
            Provider name (e.g., "tavily", "google", "semantic_scholar")
        """
        ...

    @abstractmethod
    async def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any,
    ) -> list[ResearchSource]:
        """Execute a search query and return research sources.

        This method should:
        1. Make the API call to the search provider
        2. Parse the response into SearchResult objects
        3. Convert SearchResults to ResearchSource objects
        4. Handle rate limiting and retries internally

        Args:
            query: The search query string
            max_results: Maximum number of results to return (default: 10)
            **kwargs: Provider-specific options (e.g., search_depth for Tavily)

        Returns:
            List of ResearchSource objects with quality set to UNKNOWN

        Raises:
            SearchProviderError: If the search fails after retries
        """
        ...

    @property
    def rate_limit(self) -> Optional[float]:
        """Return the rate limit in requests per second.

        Override this property to specify rate limiting behavior.
        Return None to disable rate limiting (default).

        Returns:
            Requests per second limit, or None if unlimited
        """
        return None

    async def health_check(self) -> bool:
        """Check if the provider is available and properly configured.

        Default implementation returns True. Override to add actual
        health checks (e.g., API key validation, connectivity test).

        Returns:
            True if provider is healthy, False otherwise
        """
        return True

    def classify_error(self, error: Exception) -> "ErrorClassification":
        """Classify an error for resilience decisions.

        This hook is called by `execute_with_resilience()` to determine how
        to handle provider errors. The classification drives:
        - Retry behavior: `retryable=True` triggers exponential backoff retry
        - Circuit breaker: `trips_breaker=True` increments failure count
        - Metrics: `error_type` is recorded for observability

        Default implementation handles common patterns:
        - AuthenticationError: Not retryable, doesn't trip breaker
        - RateLimitError: Retryable with backoff_seconds, doesn't trip breaker
        - 5xx errors: Retryable, trips breaker
        - Timeouts: Retryable, trips breaker
        - Network errors: Retryable, trips breaker

        Override in subclasses for provider-specific error classification,
        e.g., to handle custom error codes or parse API error responses.

        Args:
            error: The exception to classify

        Returns:
            ErrorClassification with retryable, trips_breaker, and error_type
        """
        # Import here to avoid circular imports
        from foundry_mcp.core.research.providers.resilience import (
            ErrorClassification,
            ErrorType,
        )

        # Check for our custom exception types first
        if isinstance(error, AuthenticationError):
            return ErrorClassification(
                retryable=False, trips_breaker=False, error_type=ErrorType.AUTHENTICATION
            )
        if isinstance(error, RateLimitError):
            return ErrorClassification(
                retryable=True,
                trips_breaker=False,
                backoff_seconds=error.retry_after,
                error_type=ErrorType.RATE_LIMIT,
            )
        if isinstance(error, SearchProviderError):
            error_str = str(error).lower()
            if any(code in error_str for code in ["500", "502", "503", "504"]):
                return ErrorClassification(
                    retryable=True, trips_breaker=True, error_type=ErrorType.SERVER_ERROR
                )
            if "400" in error_str:
                return ErrorClassification(
                    retryable=False, trips_breaker=False, error_type=ErrorType.INVALID_REQUEST
                )
            return ErrorClassification(
                retryable=error.retryable,
                trips_breaker=error.retryable,
                error_type=ErrorType.UNKNOWN,
            )

        # Check for httpx exceptions (common in HTTP providers)
        error_type_name = type(error).__name__.lower()
        if "timeout" in error_type_name:
            return ErrorClassification(
                retryable=True, trips_breaker=True, error_type=ErrorType.TIMEOUT
            )
        if "request" in error_type_name or "connection" in error_type_name:
            return ErrorClassification(
                retryable=True, trips_breaker=True, error_type=ErrorType.NETWORK
            )

        # Default: not retryable, trips breaker
        return ErrorClassification(
            retryable=False, trips_breaker=True, error_type=ErrorType.UNKNOWN
        )


# Error classes (canonical definitions in foundry_mcp.core.errors.search)
from foundry_mcp.core.errors.search import (  # noqa: E402
    AuthenticationError,
    RateLimitError,
    SearchProviderError,
)
