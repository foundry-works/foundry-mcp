"""Search providers for deep research workflow.

This package provides abstract base classes and concrete implementations
for search providers used during the GATHERING phase of deep research.

Supported providers:
- TavilySearchProvider: Web search via Tavily API
- PerplexitySearchProvider: Web search via Perplexity Search API
- GoogleSearchProvider: Web search via Google Custom Search API
- SemanticScholarProvider: Academic paper search via Semantic Scholar API
"""

from foundry_mcp.core.research.providers.base import (
    AuthenticationError,
    RateLimitError,
    SearchProvider,
    SearchProviderError,
    SearchResult,
)
from foundry_mcp.core.research.providers.google import GoogleSearchProvider
from foundry_mcp.core.research.providers.perplexity import PerplexitySearchProvider
from foundry_mcp.core.research.providers.resilience import (
    ErrorClassification,
    ErrorType,
    ProviderResilienceConfig,
    ProviderResilienceManager,
    ProviderStatus,
    RateLimitWaitError,
    TimeBudgetExceededError,
    async_retry_with_backoff,
    execute_with_resilience,
    get_provider_config,
    get_resilience_manager,
    reset_resilience_manager_for_testing,
)
from foundry_mcp.core.research.providers.semantic_scholar import (
    SemanticScholarProvider,
)
from foundry_mcp.core.research.providers.tavily import TavilySearchProvider
from foundry_mcp.core.research.providers.tavily_extract import (
    TavilyExtractProvider,
    UrlValidationError,
)

__all__ = [
    # Abstract base
    "SearchProvider",
    "SearchResult",
    # Concrete providers
    "TavilySearchProvider",
    "TavilyExtractProvider",
    "PerplexitySearchProvider",
    "GoogleSearchProvider",
    "SemanticScholarProvider",
    # Errors
    "SearchProviderError",
    "RateLimitError",
    "AuthenticationError",
    "UrlValidationError",
    "RateLimitWaitError",
    "TimeBudgetExceededError",
    # Resilience
    "ErrorClassification",
    "ErrorType",
    "ProviderResilienceConfig",
    "ProviderResilienceManager",
    "ProviderStatus",
    "async_retry_with_backoff",
    "execute_with_resilience",
    "get_provider_config",
    "get_resilience_manager",
    "reset_resilience_manager_for_testing",
]
