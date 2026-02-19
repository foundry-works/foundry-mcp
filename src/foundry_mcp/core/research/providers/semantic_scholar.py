"""Semantic Scholar provider for academic paper search.

This module implements SemanticScholarProvider, which wraps the Semantic Scholar
Academic Graph API to provide academic paper search capabilities for the deep
research workflow.

Semantic Scholar API documentation:
https://api.semanticscholar.org/api-docs/

Resilience Configuration:
    - Rate Limit: 0.9 RPS with burst limit of 2 (conservative for academic API)
    - Circuit Breaker: Opens after 5 failures, 30s recovery timeout
    - Retry: Up to 3 retries with exponential backoff (1.5-60s base delay)
    - Error Handling:
        - 429: Retryable, does NOT trip circuit breaker
        - 401: Not retryable, does NOT trip circuit breaker
        - 5xx: Retryable, trips circuit breaker
        - Timeouts: Retryable, trips circuit breaker

    Note: Rate limit is set slightly under 1 RPS to stay within Semantic Scholar's
    documented rate limits across all endpoints.

Example usage:
    provider = SemanticScholarProvider(api_key="optional-key")
    sources = await provider.search("transformer architecture", max_results=10)
"""

import logging
import os
from dataclasses import replace
from typing import Any, Optional

import httpx

from foundry_mcp.core.errors.search import (
    AuthenticationError,
    RateLimitError,
    SearchProviderError,
)
from foundry_mcp.core.research.models.sources import ResearchSource, SourceType
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
    extract_error_message,
    parse_iso_date,
    parse_retry_after,
)

logger = logging.getLogger(__name__)

# Semantic Scholar API constants
SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1"
PAPER_SEARCH_ENDPOINT = "/paper/search"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RATE_LIMIT = 0.9  # requests per second (slightly under 1 RPS across endpoints)

# Fields to request from the API
# See: https://api.semanticscholar.org/api-docs/graph#tag/Paper-Data/operation/get_graph_paper_relevance_search
DEFAULT_FIELDS = "paperId,title,abstract,authors,citationCount,year,externalIds,url,openAccessPdf,publicationDate"

# Extended fields including TLDR and additional metadata
EXTENDED_FIELDS = (
    "paperId,title,abstract,authors,citationCount,year,"
    "externalIds,url,openAccessPdf,publicationDate,"
    "tldr,influentialCitationCount,referenceCount,venue,fieldsOfStudy"
)

# Valid publication types for filtering
# See: https://api.semanticscholar.org/api-docs/graph#tag/Paper-Data
VALID_PUBLICATION_TYPES = frozenset(
    {
        "Review",
        "JournalArticle",
        "Conference",
        "CaseReport",
        "ClinicalTrial",
        "Dataset",
        "Editorial",
        "LettersAndComments",
        "MetaAnalysis",
        "News",
        "Study",
        "Book",
        "BookSection",
    }
)

# Valid sort fields for search results
VALID_SORT_FIELDS = frozenset(
    {
        "paperId",
        "publicationDate",
        "citationCount",
    }
)

# Default sorting when sort_order is provided without sort_by
DEFAULT_SORT_BY = "publicationDate"
DEFAULT_SORT_ORDER = "desc"


def _validate_search_params(
    publication_types: list[str] | None,
    sort_by: str | None,
    sort_order: str | None,
) -> None:
    """Validate Semantic Scholar search parameters.

    Args:
        publication_types: Filter by publication types.
        sort_by: Field to sort results by.
        sort_order: Sort direction ('asc' or 'desc').

    Raises:
        ValueError: If any parameter is invalid.
    """
    if publication_types is not None:
        invalid_types = set(publication_types) - VALID_PUBLICATION_TYPES
        if invalid_types:
            raise ValueError(
                f"Invalid publication_types: {sorted(invalid_types)}. Must be from: {sorted(VALID_PUBLICATION_TYPES)}"
            )

    if sort_by is not None:
        if sort_by not in VALID_SORT_FIELDS:
            raise ValueError(f"Invalid sort_by: {sort_by!r}. Must be one of: {sorted(VALID_SORT_FIELDS)}")

    if sort_order is not None and sort_order not in ("asc", "desc"):
        raise ValueError(f"Invalid sort_order: {sort_order!r}. Must be 'asc' or 'desc'")


class SemanticScholarProvider(SearchProvider):
    """Semantic Scholar Academic Graph API provider for paper search.

    Wraps the Semantic Scholar API to provide academic paper search capabilities.
    Uses the /paper/search endpoint (relevance search) which supports TLDR summaries
    and extended metadata fields.

    API keys are optional but recommended for higher rate limits.

    Without API key: Shared rate limit among all unauthenticated users
    With API key: up to 1 request per second (provider enforces 0.9 RPS across endpoints)

    Features:
        - TLDR summaries (auto-generated paper summaries, used as snippet when available)
        - Extended metadata: venue, influential citations, reference count, fields of study
        - Publication type filtering (JournalArticle, Conference, Review, etc.)
        - Sorting by citation count, publication date, or paper ID
        - Max 100 results per query (API limit for /paper/search endpoint)

    Attributes:
        api_key: Semantic Scholar API key (optional)
        base_url: API base URL (default: https://api.semanticscholar.org/graph/v1)
        timeout: Request timeout in seconds (default: 30.0)
        max_retries: Maximum retry attempts for rate limits (default: 3)

    Example:
        provider = SemanticScholarProvider(api_key="your-key")
        sources = await provider.search(
            "deep learning for NLP",
            max_results=10,
            year="2020-2024",
            publication_types=["JournalArticle", "Conference"],
            sort_by="citationCount",
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = SEMANTIC_SCHOLAR_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        resilience_config: Optional[ProviderResilienceConfig] = None,
    ):
        """Initialize Semantic Scholar search provider.

        Args:
            api_key: Semantic Scholar API key. If not provided, reads from
                SEMANTIC_SCHOLAR_API_KEY env var. API key is optional but
                recommended for higher rate limits.
            base_url: API base URL (default: https://api.semanticscholar.org/graph/v1)
            timeout: Request timeout in seconds (default: 30.0)
            max_retries: Maximum retry attempts for rate limits (default: 3)
            resilience_config: Custom resilience configuration. If None, uses
                defaults from PROVIDER_CONFIGS["semantic_scholar"].
        """
        self._api_key = api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._rate_limit_value = DEFAULT_RATE_LIMIT
        if resilience_config is None:
            self._resilience_config = replace(
                get_provider_config("semantic_scholar"),
                max_retries=max_retries,
            )
        else:
            self._resilience_config = resilience_config

    def get_provider_name(self) -> str:
        """Return the provider identifier.

        Returns:
            "semantic_scholar"
        """
        return "semantic_scholar"

    @property
    def rate_limit(self) -> Optional[float]:
        """Return the rate limit in requests per second.

        Returns:
            0.9 (slightly under one request per second across endpoints)
        """
        return self._rate_limit_value

    @property
    def resilience_config(self) -> ProviderResilienceConfig:
        """Return the resilience configuration for this provider."""
        if self._resilience_config is not None:
            return self._resilience_config
        return get_provider_config("semantic_scholar")

    async def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any,
    ) -> list[ResearchSource]:
        """Execute an academic paper search via Semantic Scholar API.

        Args:
            query: The search query string. Supports quoted phrases for exact match.
            max_results: Maximum number of results to return (default: 10, max: 100)
            **kwargs: Additional Semantic Scholar options:
                - year: Filter by year range (e.g., "2020-2024", "2020-", "-2024")
                - fields_of_study: Filter by fields (e.g., ["Computer Science", "Medicine"])
                - open_access_pdf: Only include papers with free PDFs (bool)
                - min_citation_count: Minimum citation count filter
                - sub_query_id: SubQuery ID for source tracking
                - publication_types: Filter by publication types (e.g., ["JournalArticle", "Conference"]).
                    Valid types: Review, JournalArticle, Conference, CaseReport, ClinicalTrial,
                    Dataset, Editorial, LettersAndComments, MetaAnalysis, News, Study, Book, BookSection
                - sort_by: Sort results by field. Valid fields: paperId, publicationDate, citationCount
                - sort_order: Sort direction, 'asc' or 'desc' (default: 'desc').
                    If provided without sort_by, defaults to publicationDate.
                - use_extended_fields: Include TLDR and additional metadata (default: True)

        Returns:
            List of ResearchSource objects with source_type='academic'

        Raises:
            AuthenticationError: If API key is invalid
            RateLimitError: If rate limit exceeded after all retries
            SearchProviderError: For other API errors
        """
        # Extract Semantic Scholar-specific options
        year = kwargs.get("year")
        fields_of_study = kwargs.get("fields_of_study")
        open_access_pdf = kwargs.get("open_access_pdf")
        min_citation_count = kwargs.get("min_citation_count")
        sub_query_id = kwargs.get("sub_query_id")

        # New search parameters
        publication_types = kwargs.get("publication_types")
        sort_by = kwargs.get("sort_by")
        sort_order = kwargs.get("sort_order")
        if sort_by is None and sort_order is not None:
            sort_by = DEFAULT_SORT_BY
        if sort_by and sort_order is None:
            sort_order = DEFAULT_SORT_ORDER
        use_extended_fields = kwargs.get("use_extended_fields", True)

        # Validate new parameters
        _validate_search_params(publication_types, sort_by, sort_order)

        # Select fields based on use_extended_fields
        fields = EXTENDED_FIELDS if use_extended_fields else DEFAULT_FIELDS

        # Build query parameters
        params: dict[str, Any] = {
            "query": query,
            "limit": min(max_results, 100),  # API max is 100 for /paper/search
            "fields": fields,
        }

        if year:
            params["year"] = year
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)
        if open_access_pdf:
            params["openAccessPdf"] = ""  # Empty string means filter to only open access
        if min_citation_count:
            params["minCitationCount"] = min_citation_count
        if publication_types:
            params["publicationTypes"] = ",".join(publication_types)
        if sort_by:
            params["sort"] = f"{sort_by}:{sort_order}"

        # Execute with retry logic
        response_data = await self._execute_with_retry(params)

        # Parse results
        return self._parse_response(response_data, sub_query_id)

    async def _execute_with_retry(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute API request with shared resilience executor."""
        url = f"{self._base_url}{PAPER_SEARCH_ENDPOINT}"
        headers: dict[str, str] = {}
        if self._api_key:
            headers["x-api-key"] = self._api_key

        async def make_request() -> dict[str, Any]:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(url, params=params, headers=headers)
                if response.status_code == 401:
                    raise AuthenticationError(provider="semantic_scholar", message="Invalid API key")
                if response.status_code == 403:
                    raise AuthenticationError(provider="semantic_scholar", message="Access forbidden - check API key")
                if response.status_code == 429:
                    raise RateLimitError(provider="semantic_scholar", retry_after=parse_retry_after(response))
                if response.status_code >= 400:
                    error_msg = extract_error_message(response)
                    raise SearchProviderError(
                        provider="semantic_scholar",
                        message=f"API error {response.status_code}: {error_msg}",
                        retryable=response.status_code >= 500,
                    )
                return response.json()

        executor = create_resilience_executor(
            "semantic_scholar",
            self.resilience_config,
            self.classify_error,
        )
        return await executor(make_request, timeout=self._timeout)

    def _parse_retry_after(self, response: httpx.Response) -> Optional[float]:
        """Parse Retry-After header. Delegates to shared utility."""
        return parse_retry_after(response)

    def _parse_response(
        self,
        data: dict[str, Any],
        sub_query_id: Optional[str] = None,
    ) -> list[ResearchSource]:
        """Parse Semantic Scholar API response into ResearchSource objects.

        Semantic Scholar /paper/search response structure:
        {
            "total": 12345,
            "offset": 0,  # current offset for pagination
            "next": 10,   # next offset (absent on last page)
            "data": [
                {
                    "paperId": "abc123",
                    "title": "...",
                    "abstract": "...",
                    "authors": [{"authorId": "...", "name": "John Doe"}],
                    "citationCount": 42,
                    "year": 2023,
                    "externalIds": {"DOI": "10.1234/...", "ArXiv": "2301.12345"},
                    "url": "https://www.semanticscholar.org/paper/...",
                    "openAccessPdf": {"url": "https://..."},
                    "publicationDate": "2023-01-15"
                }
            ]
        }

        Args:
            data: Semantic Scholar API response JSON
            sub_query_id: SubQuery ID for source tracking

        Returns:
            List of ResearchSource objects with source_type='academic'
        """
        sources: list[ResearchSource] = []
        papers = data.get("data", [])

        for paper in papers:
            # Extract external IDs (DOI, arXiv, etc.)
            external_ids = self._extract_external_ids(paper.get("externalIds", {}))

            # Format authors as comma-separated names
            authors = self._format_authors(paper.get("authors", []))

            # Extract open access PDF URL if available
            open_access = paper.get("openAccessPdf")
            pdf_url = open_access.get("url") if isinstance(open_access, dict) else None

            # Parse publication date
            pub_date = self._parse_date(paper.get("publicationDate"))

            # Extract TLDR text if available
            tldr_obj = paper.get("tldr")
            tldr_text = tldr_obj.get("text") if isinstance(tldr_obj, dict) else None

            # Build the primary URL (prefer DOI link if available)
            primary_url = self._get_primary_url(paper, external_ids)

            # Create SearchResult from Semantic Scholar response
            # Use TLDR for snippet if available, fallback to truncated abstract
            snippet = tldr_text if tldr_text else self._truncate_abstract(paper.get("abstract"))
            search_result = SearchResult(
                url=primary_url,
                title=paper.get("title", "Untitled"),
                snippet=snippet,
                content=paper.get("abstract"),  # Full abstract as content
                score=None,  # Results are relevance-ranked but no numeric score provided
                published_date=pub_date,
                source="Semantic Scholar",
                metadata={
                    "paper_id": paper.get("paperId"),
                    "authors": authors,
                    "citation_count": paper.get("citationCount"),
                    "year": paper.get("year"),
                    "doi": external_ids.get("doi"),
                    "arxiv_id": external_ids.get("arxiv"),
                    "pdf_url": pdf_url,
                    "semantic_scholar_url": paper.get("url"),
                    "venue": paper.get("venue"),
                    "influential_citation_count": paper.get("influentialCitationCount"),
                    "reference_count": paper.get("referenceCount"),
                    "fields_of_study": paper.get("fieldsOfStudy"),
                    "tldr": tldr_text,
                    **{k: v for k, v in external_ids.items() if k not in ("doi", "arxiv")},
                },
            )

            # Convert to ResearchSource with ACADEMIC type
            research_source = search_result.to_research_source(
                source_type=SourceType.ACADEMIC,
                sub_query_id=sub_query_id,
            )
            sources.append(research_source)

        return sources

    def _extract_external_ids(
        self,
        external_ids: dict[str, Any],
    ) -> dict[str, str]:
        """Extract and normalize external IDs from Semantic Scholar response.

        Args:
            external_ids: Raw externalIds object from API response

        Returns:
            Dict with normalized keys (doi, arxiv, pubmed, etc.)
        """
        result: dict[str, str] = {}

        # Map common ID types to normalized keys
        id_mapping = {
            "DOI": "doi",
            "ArXiv": "arxiv",
            "PubMed": "pubmed",
            "PubMedCentral": "pmc",
            "MAG": "mag",  # Microsoft Academic Graph
            "CorpusId": "corpus_id",
            "DBLP": "dblp",
            "ACL": "acl",
        }

        for api_key, normalized_key in id_mapping.items():
            if api_key in external_ids and external_ids[api_key]:
                result[normalized_key] = str(external_ids[api_key])

        return result

    def _format_authors(self, authors: list[dict[str, Any]]) -> str:
        """Format author list as comma-separated names.

        Args:
            authors: List of author objects from API response

        Returns:
            Comma-separated author names (e.g., "John Doe, Jane Smith")
        """
        if not authors:
            return ""

        names = [a.get("name", "") for a in authors if a.get("name")]

        # Limit to first 5 authors with "et al." if more
        if len(names) > 5:
            return ", ".join(names[:5]) + " et al."

        return ", ".join(names)

    def _get_primary_url(
        self,
        paper: dict[str, Any],
        external_ids: dict[str, str],
    ) -> str:
        """Get the best primary URL for the paper.

        Priority:
        1. DOI link (most stable)
        2. arXiv link (commonly used in ML/AI)
        3. Semantic Scholar URL (always available)

        Args:
            paper: Paper object from API response
            external_ids: Extracted external IDs

        Returns:
            Best available URL for the paper
        """
        # DOI link
        if external_ids.get("doi"):
            return f"https://doi.org/{external_ids['doi']}"

        # arXiv link
        if external_ids.get("arxiv"):
            return f"https://arxiv.org/abs/{external_ids['arxiv']}"

        # Fall back to Semantic Scholar URL
        return paper.get("url", "")

    def _truncate_abstract(
        self,
        abstract: Optional[str],
        max_length: int = 500,
    ) -> Optional[str]:
        """Truncate abstract for snippet field.

        Args:
            abstract: Full abstract text
            max_length: Maximum snippet length

        Returns:
            Truncated abstract or None
        """
        if not abstract:
            return None

        if len(abstract) <= max_length:
            return abstract

        # Truncate at word boundary
        truncated = abstract[:max_length]
        last_space = truncated.rfind(" ")
        if last_space > max_length * 0.8:
            truncated = truncated[:last_space]

        return truncated + "..."

    def _parse_date(self, date_str: Optional[str]) -> Optional[Any]:
        """Parse date string. Delegates to shared utility with extra year-only format."""
        return parse_iso_date(date_str, extra_formats=("%Y",))

    async def health_check(self) -> bool:
        """Check if Semantic Scholar API is accessible."""
        return await check_provider_health(
            "semantic_scholar",
            self._api_key or "no-key-required",
            self._base_url,
            test_func=lambda: self.search("test", max_results=1),
        )

    def classify_error(self, error: Exception) -> ErrorClassification:
        """Classify an error for resilience decisions."""
        return classify_http_error(error, "semantic_scholar")
