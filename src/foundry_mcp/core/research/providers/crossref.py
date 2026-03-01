"""Crossref provider for authoritative bibliographic metadata enrichment.

This module implements CrossrefProvider, which wraps the Crossref REST API
to provide DOI-based work lookup and metadata enrichment for ResearchSource
objects. This is an enrichment provider, not a search provider — it fills
missing bibliographic fields (venue, volume, issue, pages, publisher) that
other providers may not include.

Crossref REST API documentation:
https://api.crossref.org/swagger-ui/index.html

Resilience Configuration:
    - Rate Limit: 10 RPS with burst limit of 5 (polite pool with mailto)
    - Circuit Breaker: Opens after 5 failures, 30s recovery timeout
    - Retry: Up to 3 retries with exponential backoff (1.0-60s base delay)
    - Error Handling:
        - 429: Retryable, does NOT trip circuit breaker
        - 401/403: Not retryable, does NOT trip circuit breaker
        - 5xx: Retryable, trips circuit breaker
        - Timeouts: Retryable, trips circuit breaker

    Note: No API key required. Polite pool access via mailto in User-Agent.

Example usage:
    provider = CrossrefProvider(mailto="user@example.com")
    work = await provider.get_work("10.1038/s41586-020-2649-2")
    enriched = await provider.enrich_source(some_research_source)
"""

import logging
import os
import re
from dataclasses import replace
from typing import Any, ClassVar, Optional
from urllib.parse import quote as _url_quote

import httpx

from foundry_mcp.core.errors.search import (
    AuthenticationError,
    RateLimitError,
    SearchProviderError,
)
from foundry_mcp.core.research.models.sources import ResearchSource
from foundry_mcp.core.research.providers.resilience import (
    ErrorClassification,
    ErrorType,
    ProviderResilienceConfig,
    get_provider_config,
)
from foundry_mcp.core.research.providers.shared import (
    _ERROR_TYPE_DEFAULTS,
    check_provider_health,
    classify_http_error,
    create_resilience_executor,
    extract_error_message,
    extract_status_code,
    parse_retry_after,
    truncate_abstract,
)

logger = logging.getLogger(__name__)

# Crossref API constants
CROSSREF_BASE_URL = "https://api.crossref.org"
WORKS_ENDPOINT = "/works"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RATE_LIMIT = 10.0  # Polite pool with mailto; conservative

# Regex to strip JATS XML tags from abstracts
_JATS_TAG_RE = re.compile(r"<[^>]+>")


def _strip_jats(text: Optional[str]) -> Optional[str]:
    """Strip JATS XML tags from Crossref abstract text.

    Crossref abstracts are often wrapped in JATS XML tags like
    ``<jats:p>text</jats:p>``, ``<jats:sec>``, ``<jats:title>``.

    Args:
        text: Abstract text potentially containing JATS tags.

    Returns:
        Plain text with tags removed, or None if input is None/empty.
    """
    if not text:
        return None
    stripped = _JATS_TAG_RE.sub("", text).strip()
    return stripped if stripped else None


def _parse_date_parts(date_obj: Optional[dict[str, Any]]) -> Optional[int]:
    """Extract publication year from Crossref date-parts format.

    Crossref dates use ``{"date-parts": [[year, month, day]]}`` where
    month and day may be absent.

    Args:
        date_obj: Crossref date object, e.g. ``{"date-parts": [[2024, 6, 15]]}``

    Returns:
        Publication year as int, or None if not parseable.
    """
    if not date_obj:
        return None
    parts = date_obj.get("date-parts")
    if not parts or not parts[0]:
        return None
    try:
        return int(parts[0][0])
    except (ValueError, IndexError, TypeError):
        return None


class CrossrefProvider:
    """Crossref API provider for authoritative bibliographic metadata.

    This is an enrichment provider, not a SearchProvider. Use it to fill
    missing bibliographic fields (venue, volume, issue, pages, publisher)
    on ResearchSource objects obtained from other providers.

    Crossref REST API serves 150M+ DOI records. No API key required;
    polite pool access (faster rate limits) is granted by including a
    ``mailto:`` in the User-Agent header.

    Attributes:
        mailto: Email for polite pool access (recommended).
        base_url: API base URL (default: https://api.crossref.org)
        timeout: Request timeout in seconds (default: 30.0)
        max_retries: Maximum retry attempts (default: 3)
    """

    ERROR_CLASSIFIERS: ClassVar[dict[int, ErrorType]] = {
        429: ErrorType.RATE_LIMIT,
        503: ErrorType.SERVER_ERROR,
    }

    def __init__(
        self,
        mailto: Optional[str] = None,
        base_url: str = CROSSREF_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        resilience_config: Optional[ProviderResilienceConfig] = None,
    ):
        """Initialize Crossref provider.

        Args:
            mailto: Email address for polite pool access. If not provided,
                reads from CROSSREF_MAILTO env var.
            base_url: API base URL (default: https://api.crossref.org)
            timeout: Request timeout in seconds (default: 30.0)
            max_retries: Maximum retry attempts (default: 3)
            resilience_config: Custom resilience configuration. If None, uses
                defaults from PROVIDER_CONFIGS["crossref"].
        """
        self._mailto = mailto or os.environ.get("CROSSREF_MAILTO")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._rate_limit_value = DEFAULT_RATE_LIMIT
        if resilience_config is None:
            self._resilience_config = replace(
                get_provider_config("crossref"),
                max_retries=max_retries,
            )
        else:
            self._resilience_config = resilience_config

    def get_provider_name(self) -> str:
        """Return the provider identifier.

        Returns:
            "crossref"
        """
        return "crossref"

    @property
    def rate_limit(self) -> Optional[float]:
        """Return the rate limit in requests per second.

        Returns:
            10.0 (conservative polite pool rate)
        """
        return self._rate_limit_value

    @property
    def resilience_config(self) -> ProviderResilienceConfig:
        """Return the resilience configuration for this provider."""
        if self._resilience_config is not None:
            return self._resilience_config
        return get_provider_config("crossref")

    async def get_work(self, doi: str) -> Optional[dict[str, Any]]:
        """Look up a single work by DOI and return normalized metadata.

        Args:
            doi: DOI string, e.g. "10.1038/s41586-020-2649-2".
                The ``https://doi.org/`` prefix is stripped if present.

        Returns:
            Normalized metadata dict with keys: title, authors, venue,
            volume, issue, page, publisher, type, year, citation_count,
            doi, abstract, issn, subjects, license_url, pdf_url, funder.
            Returns None if the DOI is not found in Crossref.
        """
        # Strip DOI URL prefix if present
        if doi.startswith("https://doi.org/"):
            doi = doi[len("https://doi.org/"):]
        elif doi.startswith("http://doi.org/"):
            doi = doi[len("http://doi.org/"):]

        endpoint = f"{WORKS_ENDPOINT}/{_url_quote(doi, safe='')}"
        try:
            response_data = await self._execute_request(endpoint)
        except SearchProviderError as e:
            if "404" in str(e):
                return None
            raise

        message = response_data.get("message", {})
        return self._normalize_work(message)

    async def enrich_source(self, source: ResearchSource) -> ResearchSource:
        """Enrich a ResearchSource with missing bibliographic metadata.

        Looks up the source's DOI in Crossref and fills missing metadata
        fields. Never overwrites existing values.

        Args:
            source: ResearchSource to enrich. Must have a DOI in
                ``source.metadata["doi"]`` for enrichment to proceed.

        Returns:
            Enriched ResearchSource (same object if no DOI or lookup fails).
        """
        doi = (source.metadata or {}).get("doi")
        if not doi:
            return source

        work = await self.get_work(doi)
        if not work:
            return source

        # Build enriched metadata — only fill missing fields
        enriched_meta = dict(source.metadata)
        _ENRICHMENT_FIELDS = [
            "venue",
            "volume",
            "issue",
            "page",
            "publisher",
            "type",
            "year",
            "citation_count",
            "issn",
            "subjects",
            "license_url",
            "pdf_url",
            "funder",
        ]
        for field in _ENRICHMENT_FIELDS:
            if not enriched_meta.get(field) and work.get(field) is not None:
                enriched_meta[field] = work[field]

        # Fill missing authors
        if not enriched_meta.get("authors") and work.get("authors"):
            enriched_meta["authors"] = work["authors"]

        # Fill missing abstract/content
        enriched_content = source.content
        if not enriched_content and work.get("abstract"):
            enriched_content = work["abstract"]

        enriched_snippet = source.snippet
        if not enriched_snippet and enriched_content:
            enriched_snippet = truncate_abstract(enriched_content)

        return ResearchSource(
            id=source.id,
            url=source.url,
            title=source.title,
            source_type=source.source_type,
            quality=source.quality,
            snippet=enriched_snippet,
            content=enriched_content,
            raw_content=source.raw_content,
            content_type=source.content_type,
            sub_query_id=source.sub_query_id,
            citation_number=source.citation_number,
            discovered_at=source.discovered_at,
            metadata=enriched_meta,
        )

    async def health_check(self) -> bool:
        """Check if Crossref API is accessible."""
        return await check_provider_health(
            "crossref",
            self._mailto or "public",
            self._base_url,
            test_func=lambda: self.get_work("10.1000/xyz123"),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def classify_error(self, error: Exception) -> "ErrorClassification":
        """Classify an error for resilience decisions.

        Follows the same pattern as SearchProvider.classify_error but
        without requiring the base class.
        """
        if self.ERROR_CLASSIFIERS and isinstance(error, SearchProviderError):
            code = extract_status_code(str(error))
            if code is not None and code in self.ERROR_CLASSIFIERS:
                error_type = self.ERROR_CLASSIFIERS[code]
                retryable, trips_breaker = _ERROR_TYPE_DEFAULTS.get(
                    error_type.value, (False, True)
                )
                return ErrorClassification(
                    retryable=retryable,
                    trips_breaker=trips_breaker,
                    error_type=error_type,
                )
        return classify_http_error(error, self.get_provider_name())

    async def _execute_request(
        self,
        endpoint: str,
        *,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Execute API request with resilience executor."""
        url = f"{self._base_url}{endpoint}"

        # Build User-Agent with mailto for polite pool
        headers: dict[str, str] = {}
        if self._mailto:
            headers["User-Agent"] = f"foundry-mcp/1.0 (mailto:{self._mailto})"

        if params is None:
            params = {}

        async def make_request() -> dict[str, Any]:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(url, params=params, headers=headers)

                if response.status_code == 401:
                    raise AuthenticationError(
                        provider="crossref",
                        message="Authentication failed",
                    )
                if response.status_code == 403:
                    raise AuthenticationError(
                        provider="crossref",
                        message="Access forbidden",
                    )
                if response.status_code == 429:
                    raise RateLimitError(
                        provider="crossref",
                        retry_after=parse_retry_after(response),
                    )
                if response.status_code >= 400:
                    error_msg = extract_error_message(response)
                    raise SearchProviderError(
                        provider="crossref",
                        message=f"API error {response.status_code}: {error_msg}",
                        retryable=response.status_code >= 500,
                    )
                return response.json()

        executor = create_resilience_executor(
            "crossref",
            self.resilience_config,
            self.classify_error,
        )
        return await executor(make_request, timeout=self._timeout)

    def _normalize_work(self, work: dict[str, Any]) -> dict[str, Any]:
        """Normalize a Crossref work object into a flat metadata dict.

        Args:
            work: Raw Crossref work object from API response.

        Returns:
            Normalized dict with consistent field names.
        """
        # Extract title (Crossref returns titles as arrays)
        titles = work.get("title", [])
        title = titles[0] if titles else None

        # Extract authors
        authors = self._format_authors(work.get("author", []))

        # Extract venue (container-title is the journal/conference)
        containers = work.get("container-title", [])
        venue = containers[0] if containers else None

        # Extract abstract (strip JATS tags)
        abstract = _strip_jats(work.get("abstract"))

        # Extract publication year
        year = _parse_date_parts(
            work.get("published-print") or work.get("published-online") or work.get("issued")
        )

        # Extract DOI
        doi = work.get("DOI")

        # Extract license URL (first license)
        licenses = work.get("license", [])
        license_url = licenses[0].get("URL") if licenses else None

        # Extract PDF link
        links = work.get("link", [])
        pdf_url = None
        for link in links:
            if link.get("content-type") == "application/pdf":
                pdf_url = link.get("URL")
                break

        # Extract ISSN
        issn = work.get("ISSN", [])

        # Extract subjects
        subjects = work.get("subject", [])

        # Extract funders
        funders = work.get("funder", [])
        funder_names = [f.get("name") for f in funders if f.get("name")] if funders else None

        return {
            "title": title,
            "authors": authors,
            "venue": venue,
            "volume": work.get("volume"),
            "issue": work.get("issue"),
            "page": work.get("page"),
            "publisher": work.get("publisher"),
            "type": work.get("type"),
            "year": year,
            "citation_count": work.get("is-referenced-by-count"),
            "doi": doi,
            "abstract": abstract,
            "issn": issn if issn else None,
            "subjects": subjects if subjects else None,
            "license_url": license_url,
            "pdf_url": pdf_url,
            "funder": funder_names,
        }

    def _format_authors(self, authors: list[dict[str, Any]]) -> str:
        """Format author list as comma-separated names.

        Args:
            authors: List of Crossref author objects with ``given``/``family`` keys.

        Returns:
            Comma-separated author names, e.g. "Alice Smith, Bob Jones".
        """
        if not authors:
            return ""

        names = []
        for a in authors:
            given = a.get("given", "")
            family = a.get("family", "")
            name = f"{given} {family}".strip() if (given or family) else a.get("name", "")
            if name:
                names.append(name)

        if len(names) > 5:
            return ", ".join(names[:5]) + " et al."
        return ", ".join(names)

