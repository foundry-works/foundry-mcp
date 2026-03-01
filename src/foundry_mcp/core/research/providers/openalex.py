"""OpenAlex provider for comprehensive academic search.

This module implements OpenAlexProvider, which wraps the OpenAlex API
to provide academic paper search, citation graph traversal, topic
classification, and single-work lookup capabilities.

OpenAlex API documentation:
https://docs.openalex.org/

Resilience Configuration:
    - Rate Limit: 50 RPS with burst limit of 10 (conservative; 100 hard cap)
    - Circuit Breaker: Opens after 5 failures, 30s recovery timeout
    - Retry: Up to 3 retries with exponential backoff (1.0-60s base delay)
    - Error Handling:
        - 429: Retryable, does NOT trip circuit breaker
        - 401/403: Not retryable, does NOT trip circuit breaker
        - 5xx: Retryable, trips circuit breaker
        - Timeouts: Retryable, trips circuit breaker

    Note: API key required since Feb 2026 (free to create at openalex.org).

Example usage:
    provider = OpenAlexProvider(api_key="your-key")
    sources = await provider.search("transformer architecture", max_results=10)
"""

import logging
import os
from dataclasses import replace
from typing import Any, ClassVar, Optional

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
    ErrorType,
    ProviderResilienceConfig,
    get_provider_config,
)
from foundry_mcp.core.research.providers.shared import (
    check_provider_health,
    create_resilience_executor,
    extract_error_message,
    parse_iso_date,
    parse_retry_after,
)

logger = logging.getLogger(__name__)

# OpenAlex API constants
OPENALEX_BASE_URL = "https://api.openalex.org"
WORKS_ENDPOINT = "/works"
TEXT_ENDPOINT = "/text"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RATE_LIMIT = 50.0  # requests per second (conservative; 100 hard cap)

# Maximum results per API page
MAX_PER_PAGE = 200


def _reconstruct_abstract(abstract_inverted_index: Optional[dict[str, list[int]]]) -> Optional[str]:
    """Reconstruct plaintext abstract from OpenAlex inverted index format.

    OpenAlex stores abstracts as inverted indices mapping words to their
    position(s) in the text. This reconstructs the original text.

    Args:
        abstract_inverted_index: Mapping of word -> list of positions,
            e.g. {"This": [0], "is": [1, 3], "a": [2], "test": [4]}

    Returns:
        Reconstructed abstract text, or None if input is None/empty.
    """
    if not abstract_inverted_index:
        return None

    # Build position -> word mapping
    position_map: dict[int, str] = {}
    for word, positions in abstract_inverted_index.items():
        for pos in positions:
            position_map[pos] = word

    if not position_map:
        return None

    # Reconstruct text in order
    max_pos = max(position_map.keys())
    words = [position_map.get(i, "") for i in range(max_pos + 1)]
    return " ".join(w for w in words if w)


class OpenAlexProvider(SearchProvider):
    """OpenAlex API provider for comprehensive academic search.

    Wraps the OpenAlex API (477M+ works, CC0 license) to provide academic
    paper search, citation graph traversal, topic classification, and
    single-work lookup.

    API key required since Feb 2026 (free to create).

    Features:
        - Full-text search with filters (year, type, open access, topics, etc.)
        - Citation graph traversal (forward citations, backward references, related)
        - Topic classification via concept tagging
        - Single-work lookup by OpenAlex ID, DOI, or PMID
        - Abstract reconstruction from inverted index format
        - Max 200 results per page (API limit)

    Attributes:
        api_key: OpenAlex API key (required since Feb 2026)
        base_url: API base URL (default: https://api.openalex.org)
        timeout: Request timeout in seconds (default: 30.0)
        max_retries: Maximum retry attempts (default: 3)
    """

    ERROR_CLASSIFIERS: ClassVar[dict[int, ErrorType]] = {
        429: ErrorType.RATE_LIMIT,
        503: ErrorType.SERVER_ERROR,
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = OPENALEX_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        resilience_config: Optional[ProviderResilienceConfig] = None,
    ):
        """Initialize OpenAlex provider.

        Args:
            api_key: OpenAlex API key. If not provided, reads from
                OPENALEX_API_KEY env var.
            base_url: API base URL (default: https://api.openalex.org)
            timeout: Request timeout in seconds (default: 30.0)
            max_retries: Maximum retry attempts (default: 3)
            resilience_config: Custom resilience configuration. If None, uses
                defaults from PROVIDER_CONFIGS["openalex"].
        """
        self._api_key = api_key or os.environ.get("OPENALEX_API_KEY")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._rate_limit_value = DEFAULT_RATE_LIMIT
        if resilience_config is None:
            self._resilience_config = replace(
                get_provider_config("openalex"),
                max_retries=max_retries,
            )
        else:
            self._resilience_config = resilience_config

    def get_provider_name(self) -> str:
        """Return the provider identifier.

        Returns:
            "openalex"
        """
        return "openalex"

    @property
    def rate_limit(self) -> Optional[float]:
        """Return the rate limit in requests per second.

        Returns:
            50.0 (conservative; 100 hard cap)
        """
        return self._rate_limit_value

    @property
    def resilience_config(self) -> ProviderResilienceConfig:
        """Return the resilience configuration for this provider."""
        if self._resilience_config is not None:
            return self._resilience_config
        return get_provider_config("openalex")

    async def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any,
    ) -> list[ResearchSource]:
        """Execute an academic paper search via OpenAlex API.

        Args:
            query: The search query string.
            max_results: Maximum number of results to return (default: 10, max: 200)
            **kwargs: Additional OpenAlex options:
                - filters: dict of OpenAlex filter expressions, e.g.
                    {"publication_year": "2020-2024", "type": "article",
                     "open_access.is_oa": True, "cited_by_count": ">100",
                     "topics.id": "T12345",
                     "authorships.institutions.id": "I123"}
                - sort: Sort field, e.g. "cited_by_count:desc", "publication_date:asc"
                - sub_query_id: SubQuery ID for source tracking

        Returns:
            List of ResearchSource objects with source_type='academic'

        Raises:
            AuthenticationError: If API key is invalid or missing
            RateLimitError: If rate limit exceeded after all retries
            SearchProviderError: For other API errors
        """
        filters = kwargs.get("filters") or {}
        sort = kwargs.get("sort")
        sub_query_id = kwargs.get("sub_query_id")

        params: dict[str, Any] = {
            "search": query,
            "per_page": min(max_results, MAX_PER_PAGE),
        }

        # Build filter string
        filter_parts = []
        for key, value in filters.items():
            if isinstance(value, bool):
                filter_parts.append(f"{key}:{str(value).lower()}")
            else:
                filter_parts.append(f"{key}:{value}")
        if filter_parts:
            params["filter"] = ",".join(filter_parts)

        if sort:
            params["sort"] = sort

        response_data = await self._execute_request("GET", WORKS_ENDPOINT, params=params)
        return self._parse_works_response(response_data, sub_query_id)

    async def get_work(self, work_id: str) -> Optional[ResearchSource]:
        """Look up a single work by OpenAlex ID, DOI, or PMID.

        Args:
            work_id: Work identifier. Supports formats:
                - OpenAlex ID: "W1234567890"
                - DOI: "10.1234/example" or "https://doi.org/10.1234/example"
                - PMID: "pmid:12345678"

        Returns:
            ResearchSource for the work, or None if not found.
        """
        # Normalize DOI format
        if work_id.startswith("10."):
            work_id = f"https://doi.org/{work_id}"

        endpoint = f"{WORKS_ENDPOINT}/{work_id}"
        try:
            response_data = await self._execute_request("GET", endpoint)
        except SearchProviderError as e:
            if "404" in str(e):
                return None
            raise
        sources = self._parse_works_response({"results": [response_data]})
        return sources[0] if sources else None

    async def get_citations(self, work_id: str, max_results: int = 20) -> list[ResearchSource]:
        """Get works that cite the given work (forward citations).

        Args:
            work_id: OpenAlex work ID (e.g. "W1234567890")
            max_results: Maximum citations to return (default: 20)

        Returns:
            List of ResearchSource objects for citing works.
        """
        params: dict[str, Any] = {
            "filter": f"cites:{work_id}",
            "per_page": min(max_results, MAX_PER_PAGE),
            "sort": "cited_by_count:desc",
        }
        response_data = await self._execute_request("GET", WORKS_ENDPOINT, params=params)
        return self._parse_works_response(response_data)

    async def get_references(self, work_id: str, max_results: int = 20) -> list[ResearchSource]:
        """Get works referenced by the given work (backward references).

        Uses the referenced_works field from the single work endpoint,
        then fetches full metadata for each.

        Args:
            work_id: OpenAlex work ID (e.g. "W1234567890")
            max_results: Maximum references to return (default: 20)

        Returns:
            List of ResearchSource objects for referenced works.
        """
        # First get the work to extract referenced_works
        endpoint = f"{WORKS_ENDPOINT}/{work_id}"
        try:
            work_data = await self._execute_request("GET", endpoint)
        except SearchProviderError as e:
            if "404" in str(e):
                return []
            raise

        referenced_ids = work_data.get("referenced_works", [])
        if not referenced_ids:
            return []

        # Limit and fetch via filter
        referenced_ids = referenced_ids[:max_results]
        openalex_filter = "|".join(referenced_ids)
        params: dict[str, Any] = {
            "filter": f"openalex:{openalex_filter}",
            "per_page": min(len(referenced_ids), MAX_PER_PAGE),
        }
        response_data = await self._execute_request("GET", WORKS_ENDPOINT, params=params)
        return self._parse_works_response(response_data)

    async def get_related(self, work_id: str, max_results: int = 10) -> list[ResearchSource]:
        """Get works related to the given work.

        Uses the related_works field from the single work endpoint,
        then fetches full metadata for each.

        Args:
            work_id: OpenAlex work ID (e.g. "W1234567890")
            max_results: Maximum related works to return (default: 10)

        Returns:
            List of ResearchSource objects for related works.
        """
        endpoint = f"{WORKS_ENDPOINT}/{work_id}"
        try:
            work_data = await self._execute_request("GET", endpoint)
        except SearchProviderError as e:
            if "404" in str(e):
                return []
            raise

        related_ids = work_data.get("related_works", [])
        if not related_ids:
            return []

        related_ids = related_ids[:max_results]
        openalex_filter = "|".join(related_ids)
        params: dict[str, Any] = {
            "filter": f"openalex:{openalex_filter}",
            "per_page": min(len(related_ids), MAX_PER_PAGE),
        }
        response_data = await self._execute_request("GET", WORKS_ENDPOINT, params=params)
        return self._parse_works_response(response_data)

    async def classify_text(self, text: str) -> list[dict]:
        """Classify text into OpenAlex topics/concepts.

        Args:
            text: Text to classify (e.g. a paper title + abstract).

        Returns:
            List of topic dicts with keys: id, display_name, score, subfield, field, domain.
        """
        response_data = await self._execute_request(
            "POST",
            TEXT_ENDPOINT,
            json_body={"text": text},
        )
        topics = response_data.get("topics", [])
        return [
            {
                "id": t.get("id"),
                "display_name": t.get("display_name"),
                "score": t.get("score"),
                "subfield": (t.get("subfield") or {}).get("display_name"),
                "field": (t.get("field") or {}).get("display_name"),
                "domain": (t.get("domain") or {}).get("display_name"),
            }
            for t in topics
        ]

    async def search_by_topic(
        self,
        topic_id: str,
        max_results: int = 20,
        **kwargs: Any,
    ) -> list[ResearchSource]:
        """Search for works by OpenAlex topic ID.

        Args:
            topic_id: OpenAlex topic ID (e.g. "T12345")
            max_results: Maximum results to return (default: 20)
            **kwargs: Additional options:
                - sort: Sort field (default: "cited_by_count:desc")
                - sub_query_id: SubQuery ID for source tracking

        Returns:
            List of ResearchSource objects.
        """
        sort = kwargs.get("sort", "cited_by_count:desc")
        sub_query_id = kwargs.get("sub_query_id")

        params: dict[str, Any] = {
            "filter": f"topics.id:{topic_id}",
            "per_page": min(max_results, MAX_PER_PAGE),
            "sort": sort,
        }
        response_data = await self._execute_request("GET", WORKS_ENDPOINT, params=params)
        return self._parse_works_response(response_data, sub_query_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _execute_request(
        self,
        method: str,
        endpoint: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Execute API request with resilience executor."""
        url = f"{self._base_url}{endpoint}"

        # Auth: api_key param or x-api-key header
        headers: dict[str, str] = {}
        if params is None:
            params = {}
        if self._api_key:
            params["api_key"] = self._api_key

        async def make_request() -> dict[str, Any]:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                if method == "POST":
                    response = await client.post(url, params=params, json=json_body, headers=headers)
                else:
                    response = await client.get(url, params=params, headers=headers)

                if response.status_code == 401:
                    raise AuthenticationError(
                        provider="openalex",
                        message="Invalid or missing API key",
                    )
                if response.status_code == 403:
                    raise AuthenticationError(
                        provider="openalex",
                        message="Access forbidden - check API key",
                    )
                if response.status_code == 429:
                    raise RateLimitError(
                        provider="openalex",
                        retry_after=parse_retry_after(response),
                    )
                if response.status_code >= 400:
                    error_msg = extract_error_message(response)
                    raise SearchProviderError(
                        provider="openalex",
                        message=f"API error {response.status_code}: {error_msg}",
                        retryable=response.status_code >= 500,
                    )
                return response.json()

        executor = create_resilience_executor(
            "openalex",
            self.resilience_config,
            self.classify_error,
        )
        return await executor(make_request, timeout=self._timeout)

    def _parse_works_response(
        self,
        data: dict[str, Any],
        sub_query_id: Optional[str] = None,
    ) -> list[ResearchSource]:
        """Parse OpenAlex works response into ResearchSource objects.

        Handles both list responses (with "results" key) and single
        work responses wrapped in {"results": [work]}.

        Args:
            data: OpenAlex API response JSON
            sub_query_id: SubQuery ID for source tracking

        Returns:
            List of ResearchSource objects with source_type='academic'
        """
        sources: list[ResearchSource] = []
        works = data.get("results", [])

        for work in works:
            # Reconstruct abstract from inverted index
            abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))

            # Extract authorship info
            authors = self._format_authors(work.get("authorships", []))

            # Get primary topic (first topic in list)
            topics = work.get("topics", [])
            primary_topic = topics[0].get("display_name") if topics else None

            # Extract open access URL
            open_access = work.get("open_access") or {}
            oa_url = open_access.get("oa_url")

            # Extract DOI
            doi = work.get("doi")
            if doi and doi.startswith("https://doi.org/"):
                doi_short = doi[len("https://doi.org/"):]
            else:
                doi_short = doi

            # Extract IDs
            ids = work.get("ids") or {}
            openalex_id = ids.get("openalex") or work.get("id", "")

            # Parse publication date
            pub_date = parse_iso_date(work.get("publication_date"))

            # Build primary URL (prefer DOI)
            primary_url = doi if doi else (oa_url or openalex_id)

            # Extract awards/funders
            awards = work.get("awards") or work.get("grants") or []
            funders = [
                {
                    "funder": (a.get("funder") or {}).get("display_name"),
                    "award_id": a.get("award_id"),
                }
                for a in awards
                if a.get("funder")
            ]

            # Snippet: use abstract truncated
            snippet = self._truncate_abstract(abstract)

            search_result = SearchResult(
                url=primary_url or "",
                title=work.get("title") or work.get("display_name") or "Untitled",
                snippet=snippet,
                content=abstract,
                score=None,
                published_date=pub_date,
                source="OpenAlex",
                metadata={
                    "openalex_id": openalex_id,
                    "doi": doi_short,
                    "authors": authors,
                    "citation_count": work.get("cited_by_count"),
                    "year": work.get("publication_year"),
                    "primary_topic": primary_topic,
                    "topics": [t.get("display_name") for t in topics[:5]],
                    "pdf_url": oa_url,
                    "type": work.get("type"),
                    "is_oa": open_access.get("is_oa"),
                    "venue": self._extract_venue(work),
                    "funders": funders if funders else None,
                    "referenced_works_count": work.get("referenced_works_count"),
                    "cited_by_count": work.get("cited_by_count"),
                },
            )

            research_source = search_result.to_research_source(
                source_type=SourceType.ACADEMIC,
                sub_query_id=sub_query_id,
            )
            sources.append(research_source)

        return sources

    def _format_authors(self, authorships: list[dict[str, Any]]) -> str:
        """Format authorship list as comma-separated names.

        Args:
            authorships: List of authorship objects from API response

        Returns:
            Comma-separated author names (e.g., "John Doe, Jane Smith")
        """
        if not authorships:
            return ""

        names = []
        for a in authorships:
            author = a.get("author") or {}
            name = author.get("display_name")
            if name:
                names.append(name)

        if len(names) > 5:
            return ", ".join(names[:5]) + " et al."

        return ", ".join(names)

    def _extract_venue(self, work: dict[str, Any]) -> Optional[str]:
        """Extract venue/journal name from work data.

        Args:
            work: Work object from API response

        Returns:
            Venue name or None
        """
        # Try primary_location first
        primary_location = work.get("primary_location") or {}
        source = primary_location.get("source") or {}
        venue = source.get("display_name")
        if venue:
            return venue

        # Fallback to best_oa_location
        best_oa = work.get("best_oa_location") or {}
        source = best_oa.get("source") or {}
        return source.get("display_name")

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

        truncated = abstract[:max_length]
        last_space = truncated.rfind(" ")
        if last_space > max_length * 0.8:
            truncated = truncated[:last_space]

        return truncated + "..."

    async def health_check(self) -> bool:
        """Check if OpenAlex API is accessible."""
        return await check_provider_health(
            "openalex",
            self._api_key or "no-key",
            self._base_url,
            test_func=lambda: self.search("test", max_results=1),
        )
