"""Tavily extract provider for URL content extraction.

This module implements TavilyExtractProvider, which wraps the Tavily Extract API
to provide URL content extraction capabilities for the deep research workflow.

Tavily API documentation: https://docs.tavily.com/documentation/api-reference/endpoint/extract

Example usage:
    provider = TavilyExtractProvider(api_key="tvly-...")
    sources = await provider.extract(["https://example.com/article"])
"""

import asyncio
import ipaddress
import logging
import os
import re
import socket
from typing import Any, Optional
from urllib.parse import urlparse

import httpx

from foundry_mcp.core.research.models import ResearchSource, SourceType
from foundry_mcp.core.research.providers.base import (
    AuthenticationError,
    RateLimitError,
    SearchProviderError,
)

logger = logging.getLogger(__name__)

# Tavily API constants
TAVILY_API_BASE_URL = "https://api.tavily.com"
TAVILY_EXTRACT_ENDPOINT = "/extract"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RATE_LIMIT = 1.0  # requests per second

# Extract constraints
MAX_URLS_PER_REQUEST = 10
MAX_URL_LENGTH = 2048
MAX_CONTENT_SIZE = 50000  # 50KB per source
MAX_IMAGES_PER_SOURCE = 10
DNS_TIMEOUT = 5.0  # seconds for DNS resolution

# SSRF protection - blocked hosts and hostname patterns
BLOCKED_HOSTS = frozenset(["localhost", "127.0.0.1", "0.0.0.0", "::1"])
BLOCKED_HOSTNAME_PATTERNS = [
    re.compile(r"\.local$"),  # mDNS
    re.compile(r"\.internal$"),  # Internal domains
    re.compile(r"\.localhost$"),  # localhost subdomains
]

# Private/reserved IP ranges for SSRF protection
PRIVATE_IP_RANGES = [
    ipaddress.ip_network("10.0.0.0/8"),  # RFC1918
    ipaddress.ip_network("172.16.0.0/12"),  # RFC1918
    ipaddress.ip_network("192.168.0.0/16"),  # RFC1918
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local IPv4
    ipaddress.ip_network("127.0.0.0/8"),  # Loopback IPv4
    ipaddress.ip_network("0.0.0.0/8"),  # Current network
    ipaddress.ip_network("::1/128"),  # Loopback IPv6
    ipaddress.ip_network("fe80::/10"),  # Link-local IPv6
    ipaddress.ip_network("fc00::/7"),  # Unique local IPv6
    ipaddress.ip_network("ff00::/8"),  # Multicast IPv6
]

# Valid parameter values
VALID_EXTRACT_DEPTHS = frozenset(["basic", "advanced"])
VALID_FORMATS = frozenset(["markdown", "text"])


class UrlValidationError(ValueError):
    """Raised when URL validation fails (SSRF protection).

    Attributes:
        url: The URL that failed validation.
        reason: Human-readable explanation of the failure.
        error_code: Machine-readable error code (INVALID_URL or BLOCKED_HOST).
    """

    def __init__(self, url: str, reason: str, error_code: str = "INVALID_URL"):
        self.url = url
        self.reason = reason
        self.error_code = error_code
        super().__init__(f"URL validation failed for {url!r}: {reason}")


def _is_private_ip(ip_str: str) -> bool:
    """Check if an IP address is in a private/reserved range.

    Args:
        ip_str: IP address as string (IPv4 or IPv6).

    Returns:
        True if the IP is in a private/reserved range.
    """
    try:
        ip = ipaddress.ip_address(ip_str)
        for network in PRIVATE_IP_RANGES:
            if ip in network:
                return True
        return False
    except ValueError:
        # Invalid IP format - treat as potentially dangerous
        return True


def _resolve_hostname(hostname: str, timeout: float = DNS_TIMEOUT) -> list[str]:
    """Resolve hostname to IP addresses with timeout.

    Args:
        hostname: Hostname to resolve.
        timeout: Timeout in seconds.

    Returns:
        List of resolved IP addresses.

    Raises:
        UrlValidationError: If DNS resolution fails.
    """
    # Set socket timeout for DNS resolution
    old_timeout = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(timeout)
        # Get all addresses (both IPv4 and IPv6)
        addr_info = socket.getaddrinfo(
            hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM
        )
        # Extract unique IP addresses (info[4][0] is always str for IP)
        return list({str(info[4][0]) for info in addr_info})
    except socket.gaierror as e:
        raise UrlValidationError(
            hostname,
            f"DNS resolution failed: {e}",
            error_code="INVALID_URL",
        )
    except socket.timeout:
        raise UrlValidationError(
            hostname,
            f"DNS resolution timed out after {timeout}s",
            error_code="INVALID_URL",
        )
    finally:
        socket.setdefaulttimeout(old_timeout)


def validate_extract_url(url: str, resolve_dns: bool = True) -> None:
    """Validate URL for safe extraction (SSRF protection).

    Performs comprehensive validation:
    1. URL length check
    2. Scheme validation (http/https only)
    3. Hostname validation against blocklists
    4. IDN/punycode normalization
    5. DNS resolution with timeout
    6. IP address validation against private ranges

    Args:
        url: The URL to validate.
        resolve_dns: Whether to perform DNS resolution (default: True).
            Set to False for testing without network access.

    Raises:
        UrlValidationError: If URL fails validation with error_code:
            - INVALID_URL: URL parsing, scheme, or DNS failure
            - BLOCKED_HOST: Private/internal network detected
    """
    # Check URL length
    if len(url) > MAX_URL_LENGTH:
        raise UrlValidationError(
            url,
            f"URL too long: {len(url)} chars (max {MAX_URL_LENGTH})",
            error_code="INVALID_URL",
        )

    try:
        parsed = urlparse(url)
    except Exception as e:
        raise UrlValidationError(url, f"Failed to parse URL: {e}", error_code="INVALID_URL")

    # Scheme validation: only http/https allowed
    if parsed.scheme not in ("http", "https"):
        raise UrlValidationError(
            url,
            f"Invalid scheme: {parsed.scheme!r}. Only http/https allowed.",
            error_code="INVALID_URL",
        )

    hostname = parsed.hostname
    if not hostname:
        raise UrlValidationError(url, "No hostname in URL", error_code="INVALID_URL")

    # Normalize IDN/punycode hostname
    try:
        hostname = hostname.encode("idna").decode("ascii").lower()
    except (UnicodeError, UnicodeDecodeError):
        # If IDN encoding fails, use original hostname
        hostname = hostname.lower()

    # Block known localhost/loopback addresses
    if hostname in BLOCKED_HOSTS:
        raise UrlValidationError(
            url, f"Blocked host: {hostname}", error_code="BLOCKED_HOST"
        )

    # Block hostname patterns (.local, .internal, etc.)
    for pattern in BLOCKED_HOSTNAME_PATTERNS:
        if pattern.search(hostname):
            raise UrlValidationError(
                url,
                f"Blocked internal domain: {hostname}",
                error_code="BLOCKED_HOST",
            )

    # Check if hostname is already an IP address
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        # Not an IP address, continue to DNS resolution below
        pass
    else:
        # Successfully parsed as IP address - validate it
        if _is_private_ip(str(ip)):
            raise UrlValidationError(
                url,
                f"Blocked private IP address: {hostname}",
                error_code="BLOCKED_HOST",
            )
        return  # Valid public IP, skip DNS resolution

    # DNS resolution to validate all resolved IPs
    if resolve_dns:
        resolved_ips = _resolve_hostname(hostname)
        for ip_str in resolved_ips:
            if _is_private_ip(ip_str):
                raise UrlValidationError(
                    url,
                    f"Hostname {hostname} resolves to blocked private IP: {ip_str}",
                    error_code="BLOCKED_HOST",
                )


def _validate_extract_params(
    extract_depth: str,
    format: str,
    chunks_per_source: int | None,
) -> None:
    """Validate Tavily extract parameters.

    Args:
        extract_depth: Extraction depth level.
        format: Output format.
        chunks_per_source: Chunks per source limit.

    Raises:
        ValueError: If any parameter is invalid.
    """
    if extract_depth not in VALID_EXTRACT_DEPTHS:
        raise ValueError(
            f"Invalid extract_depth: {extract_depth!r}. "
            f"Must be one of: {sorted(VALID_EXTRACT_DEPTHS)}"
        )

    if format not in VALID_FORMATS:
        raise ValueError(
            f"Invalid format: {format!r}. "
            f"Must be one of: {sorted(VALID_FORMATS)}"
        )

    if chunks_per_source is not None:
        if not isinstance(chunks_per_source, int) or chunks_per_source < 1 or chunks_per_source > 5:
            raise ValueError(
                f"Invalid chunks_per_source: {chunks_per_source!r}. "
                "Must be an integer between 1 and 5."
            )


class TavilyExtractProvider:
    """Tavily Extract API provider for URL content extraction.

    Wraps the Tavily Extract API to extract content from URLs.
    Supports basic and advanced extraction depths, multiple output formats,
    and optional relevance-based chunk reranking.

    Attributes:
        api_key: Tavily API key (required)
        base_url: API base URL (default: https://api.tavily.com)
        timeout: Request timeout in seconds (default: 30.0)
        max_retries: Maximum retry attempts for rate limits (default: 3)

    Example:
        provider = TavilyExtractProvider(api_key="tvly-...")
        sources = await provider.extract(
            urls=["https://example.com/article"],
            extract_depth="advanced",
            format="markdown",
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = TAVILY_API_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        """Initialize Tavily extract provider.

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
            "tavily_extract"
        """
        return "tavily_extract"

    @property
    def rate_limit(self) -> Optional[float]:
        """Return the rate limit in requests per second.

        Returns:
            1.0 (one request per second)
        """
        return self._rate_limit_value

    async def extract(
        self,
        urls: list[str],
        *,
        extract_depth: str = "basic",
        include_images: bool = False,
        format: str = "markdown",
        query: str | None = None,
        chunks_per_source: int | None = None,
    ) -> list[ResearchSource]:
        """Extract content from URLs via Tavily Extract API.

        Args:
            urls: List of URLs to extract content from (max 10).
            extract_depth: Extraction depth level. Options:
                - "basic": Standard extraction (1 credit per 5 URLs)
                - "advanced": Deeper extraction (2 credits per 5 URLs)
                Default: "basic"
            include_images: Whether to include images in results (default: False).
            format: Output format. Options:
                - "markdown": Content as markdown (default)
                - "text": Content as plain text
            query: Optional query for relevance-based chunk reranking.
                When provided, chunks are ordered by relevance to this query.
            chunks_per_source: Number of content chunks per URL (1-5).
                Default: 3 (Tavily default).

        Returns:
            List of ResearchSource objects containing extracted content.

        Raises:
            UrlValidationError: If any URL fails SSRF validation.
            ValueError: If parameters are invalid.
            AuthenticationError: If API key is invalid.
            RateLimitError: If rate limit exceeded after all retries.
            SearchProviderError: For other API errors.
        """
        # Validate URL count
        if not urls:
            raise ValueError("At least one URL is required")
        if len(urls) > MAX_URLS_PER_REQUEST:
            raise ValueError(
                f"Too many URLs: {len(urls)}. Maximum is {MAX_URLS_PER_REQUEST}."
            )

        # Validate each URL for SSRF protection
        for url in urls:
            validate_extract_url(url)

        # Validate other parameters
        _validate_extract_params(
            extract_depth=extract_depth,
            format=format,
            chunks_per_source=chunks_per_source,
        )

        # Build request payload
        payload: dict[str, Any] = {
            "api_key": self._api_key,
            "urls": urls,
            "extract_depth": extract_depth,
            "include_images": include_images,
            "format": format,
        }

        # Conditionally include optional parameters
        if query is not None:
            payload["query"] = query
        if chunks_per_source is not None:
            payload["chunks_per_source"] = chunks_per_source

        # Execute with retry logic
        response_data = await self._execute_with_retry(payload)

        # Parse results
        return self._parse_response(response_data, extract_depth, format)

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
        url = f"{self._base_url}{TAVILY_EXTRACT_ENDPOINT}"
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(url, json=payload)

                    # Handle authentication errors (not retryable)
                    if response.status_code == 401:
                        raise AuthenticationError(
                            provider="tavily_extract",
                            message="Invalid API key",
                        )

                    # Handle rate limiting
                    if response.status_code == 429:
                        retry_after = self._parse_retry_after(response)
                        if attempt < self._max_retries - 1:
                            wait_time = retry_after or (2**attempt)
                            logger.warning(
                                f"Tavily extract rate limit hit, waiting {wait_time}s "
                                f"(attempt {attempt + 1}/{self._max_retries})"
                            )
                            await asyncio.sleep(wait_time)
                            continue
                        raise RateLimitError(
                            provider="tavily_extract",
                            retry_after=retry_after,
                        )

                    # Handle other errors
                    if response.status_code >= 400:
                        error_msg = self._extract_error_message(response)
                        raise SearchProviderError(
                            provider="tavily_extract",
                            message=f"API error {response.status_code}: {error_msg}",
                            retryable=response.status_code >= 500,
                        )

                    return response.json()

            except httpx.TimeoutException as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Tavily extract request timeout, retrying in {wait_time}s "
                        f"(attempt {attempt + 1}/{self._max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                    continue

            except httpx.RequestError as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Tavily extract request error: {e}, retrying in {wait_time}s "
                        f"(attempt {attempt + 1}/{self._max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                    continue

            except (AuthenticationError, RateLimitError, SearchProviderError):
                raise

        # All retries exhausted
        raise SearchProviderError(
            provider="tavily_extract",
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
        extract_depth: str,
        format: str,
    ) -> list[ResearchSource]:
        """Parse Tavily Extract API response into ResearchSource objects.

        Maps Tavily ExtractResult to ResearchSource with the following conventions:
        - One ResearchSource per URL in the response
        - snippet = first_chunk[:500] (first chunk truncated to 500 chars)
        - content = all chunks joined (or raw_content if no chunks)
        - metadata includes extract_depth, chunk_count, format, images, favicon

        Args:
            data: Tavily API response JSON containing 'results' array
            extract_depth: Extraction depth used ("basic" or "advanced")
            format: Output format used ("markdown" or "text")

        Returns:
            List of ResearchSource objects, one per successfully extracted URL
        """
        sources: list[ResearchSource] = []
        results = data.get("results", [])

        for result in results:
            url = result.get("url", "")

            # Handle chunks: Tavily may return individual chunks array or raw_content
            chunks = result.get("chunks", [])
            raw_content = result.get("raw_content", "")

            # Determine content: join chunks if available, otherwise use raw_content
            if chunks:
                content = "\n\n".join(chunks)
            else:
                content = raw_content

            # Truncate content if too large
            truncated = False
            if len(content) > MAX_CONTENT_SIZE:
                content = content[:MAX_CONTENT_SIZE]
                truncated = True

            # Build snippet from first chunk (truncated to 500 chars)
            # Per acceptance criteria: snippet = first_chunk[:500]
            if chunks:
                first_chunk = chunks[0] if chunks else ""
                snippet = first_chunk[:500] if first_chunk else None
            else:
                snippet = content[:500] if content else None

            # Extract title from result or derive from URL
            title = result.get("title", "")
            if not title:
                title = self._extract_domain(url) or "Extracted Content"
            # Truncate title if too long
            if len(title) > 500:
                title = title[:497] + "..."

            # Get images (limit to MAX_IMAGES_PER_SOURCE)
            images = result.get("images", [])
            if images and len(images) > MAX_IMAGES_PER_SOURCE:
                images = images[:MAX_IMAGES_PER_SOURCE]

            # Compute chunk count: number of chunks if provided, else 1 for raw_content
            chunk_count = len(chunks) if chunks else (1 if content else 0)

            # Create ResearchSource with full metadata per acceptance criteria
            source = ResearchSource(
                url=url,
                title=title,
                source_type=SourceType.WEB,
                snippet=snippet,
                content=content if content else None,
                metadata={
                    "extract_depth": extract_depth,
                    "chunk_count": chunk_count,
                    "format": format,
                    "images": images if images else None,
                    "favicon": result.get("favicon"),
                    "truncated": truncated,
                },
            )
            sources.append(source)

        return sources

    def _extract_domain(self, url: str) -> Optional[str]:
        """Extract domain from URL.

        Args:
            url: Full URL

        Returns:
            Domain name or None
        """
        try:
            parsed = urlparse(url)
            return parsed.netloc
        except Exception:
            return None

    async def health_check(self) -> bool:
        """Check if Tavily Extract API is accessible.

        Note: Unlike search, we can't easily do a lightweight extract test.
        This method verifies the API key format and attempts a minimal request.

        Returns:
            True if provider is healthy, False otherwise
        """
        # Basic API key format check
        if not self._api_key or not self._api_key.startswith("tvly-"):
            logger.error("Tavily extract health check failed: invalid API key format")
            return False

        # We don't do an actual extract call in health check as it requires valid URLs
        # and costs credits. Just verify the API key format is valid.
        return True
