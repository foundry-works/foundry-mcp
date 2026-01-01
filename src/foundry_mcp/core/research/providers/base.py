"""Abstract base class for search providers.

This module defines the SearchProvider interface that all concrete
search providers must implement. The interface enables dependency
injection and easy mocking for testing.

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
from typing import Any, Optional

from foundry_mcp.core.research.models import (
    ResearchSource,
    SourceQuality,
    SourceType,
)


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


class SearchProviderError(Exception):
    """Base exception for search provider errors.

    Attributes:
        provider: Name of the provider that raised the error
        message: Human-readable error description
        retryable: Whether the error is potentially transient
        original_error: The underlying exception if available
    """

    def __init__(
        self,
        provider: str,
        message: str,
        retryable: bool = False,
        original_error: Optional[Exception] = None,
    ):
        self.provider = provider
        self.message = message
        self.retryable = retryable
        self.original_error = original_error
        super().__init__(f"[{provider}] {message}")


class RateLimitError(SearchProviderError):
    """Raised when a provider's rate limit is exceeded.

    This error is always retryable. The retry_after field indicates
    how long to wait before retrying (if provided by the API).
    """

    def __init__(
        self,
        provider: str,
        retry_after: Optional[float] = None,
        original_error: Optional[Exception] = None,
    ):
        self.retry_after = retry_after
        message = "Rate limit exceeded"
        if retry_after:
            message += f" (retry after {retry_after}s)"
        super().__init__(
            provider=provider,
            message=message,
            retryable=True,
            original_error=original_error,
        )


class AuthenticationError(SearchProviderError):
    """Raised when API authentication fails.

    This error is NOT retryable - the API key or credentials
    need to be fixed before retrying.
    """

    def __init__(
        self,
        provider: str,
        message: str = "Authentication failed",
        original_error: Optional[Exception] = None,
    ):
        super().__init__(
            provider=provider,
            message=message,
            retryable=False,
            original_error=original_error,
        )
