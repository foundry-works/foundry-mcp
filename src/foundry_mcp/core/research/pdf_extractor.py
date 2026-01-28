"""PDF text extraction for deep research workflows.

Provides secure PDF text extraction with page boundary tracking for
evidence snippet locators. Uses pypdf as the primary extraction engine.

Security Features:
    - SSRF protection: Blocks internal IPs, localhost, and private networks
    - Magic byte validation: Verifies %PDF- header before parsing
    - Content-type validation: Checks HTTP response content-type
    - Size limits: Configurable maximum PDF size

Key Components:
    - PDFExtractionResult: Dataclass containing extracted text and metadata
    - PDFExtractor: Main class for extracting text from PDF files/bytes

Usage:
    from foundry_mcp.core.research.pdf_extractor import (
        PDFExtractor,
        PDFExtractionResult,
    )

    # Create extractor
    extractor = PDFExtractor()

    # Extract from bytes
    result = await extractor.extract(pdf_bytes)

    # Extract from URL (with SSRF protection)
    result = await extractor.extract_from_url("https://example.com/doc.pdf")

    # Access results
    print(result.text)
    print(result.page_offsets)  # [(0, 1500), (1500, 3200), ...]
"""

from __future__ import annotations

import asyncio
import io
import ipaddress
import logging
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Union
from urllib.parse import urljoin, urlparse

from pypdf import PdfReader

logger = logging.getLogger(__name__)

# =============================================================================
# Metrics (Optional - graceful degradation if prometheus_client not installed)
# =============================================================================

try:
    from prometheus_client import Counter, Histogram

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    Counter: Any = None
    Histogram: Any = None

# Metrics instances (lazily initialized)
_pdf_extraction_duration: Optional[Any] = None
_pdf_extraction_pages: Optional[Any] = None
_metrics_initialized: bool = False


def _init_metrics() -> None:
    """Initialize PDF extraction metrics (thread-safe, idempotent)."""
    global _pdf_extraction_duration, _pdf_extraction_pages, _metrics_initialized

    if _metrics_initialized or not _PROMETHEUS_AVAILABLE:
        return

    _metrics_initialized = True

    _pdf_extraction_duration = Histogram(
        "foundry_mcp_pdf_extraction_duration_seconds",
        "PDF extraction duration in seconds",
        ["status"],
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    )

    _pdf_extraction_pages = Counter(
        "foundry_mcp_pdf_extraction_pages_total",
        "Total number of pages extracted from PDFs",
        ["status"],
    )

    logger.debug("PDF extraction metrics initialized")


def _record_extraction_metrics(
    duration_seconds: float,
    pages_extracted: int,
    status: str,
) -> None:
    """Record PDF extraction metrics.

    Args:
        duration_seconds: Extraction duration in seconds.
        pages_extracted: Number of pages successfully extracted.
        status: Extraction status - "success", "partial", or "failure".
    """
    if not _PROMETHEUS_AVAILABLE:
        return

    _init_metrics()

    if _pdf_extraction_duration is not None:
        _pdf_extraction_duration.labels(status=status).observe(duration_seconds)

    if _pdf_extraction_pages is not None and pages_extracted > 0:
        _pdf_extraction_pages.labels(status=status).inc(pages_extracted)

# =============================================================================
# Lazy Import for pdfminer.six (Optional Fallback)
# =============================================================================

_pdfminer_module: Optional[object] = None
_pdfminer_checked: bool = False


def _get_pdfminer():
    """Lazy import for pdfminer.six.

    Returns the pdfminer.high_level module if available, None otherwise.
    The import is cached after first call to avoid repeated import attempts.

    Returns:
        pdfminer.high_level module or None if not installed.
    """
    global _pdfminer_module, _pdfminer_checked

    if _pdfminer_checked:
        return _pdfminer_module

    _pdfminer_checked = True
    try:
        from pdfminer import high_level as pdfminer_hl
        _pdfminer_module = pdfminer_hl
        logger.debug("pdfminer.six available for fallback extraction")
    except ImportError:
        _pdfminer_module = None
        logger.debug("pdfminer.six not installed, fallback unavailable")

    return _pdfminer_module

# =============================================================================
# Security Constants
# =============================================================================

PDF_MAGIC_BYTES = b"%PDF-"
"""PDF files must start with this magic byte sequence."""

VALID_PDF_CONTENT_TYPES = frozenset([
    "application/pdf",
    "application/x-pdf",
    "application/octet-stream",  # Some servers serve PDFs with this
])
"""Content-types that are acceptable for PDF responses."""

DEFAULT_MAX_PDF_SIZE = 10 * 1024 * 1024  # 10 MB
"""Default maximum PDF file size in bytes."""

DEFAULT_MAX_PAGES = 500
"""Default maximum number of pages to extract."""

DEFAULT_FETCH_TIMEOUT = 30.0
"""Default timeout for URL fetches in seconds."""

MAX_PDF_REDIRECTS = 5
"""Maximum number of redirects to follow when fetching PDFs."""


# =============================================================================
# Security Exceptions
# =============================================================================


class PDFSecurityError(Exception):
    """Base exception for PDF security violations."""
    pass


class SSRFError(PDFSecurityError):
    """Raised when SSRF protection blocks a request."""
    pass


class InvalidPDFError(PDFSecurityError):
    """Raised when PDF validation fails (magic bytes, content-type)."""
    pass


class PDFSizeError(PDFSecurityError):
    """Raised when PDF exceeds size limits."""
    pass


# =============================================================================
# SSRF Protection
# =============================================================================


def is_internal_ip(ip: str) -> bool:
    """Check if an IP address is internal/private.

    Blocks:
        - Private ranges: 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
        - Loopback: 127.0.0.0/8
        - Link-local: 169.254.0.0/16
        - IPv6 equivalents

    Args:
        ip: IP address string to check.

    Returns:
        True if the IP is internal/private, False otherwise.
    """
    try:
        addr = ipaddress.ip_address(ip)
        return (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_reserved
            or addr.is_multicast
        )
    except ValueError:
        # Invalid IP format - treat as unsafe
        return True


def validate_url_for_ssrf(url: str) -> None:
    """Validate a URL is safe from SSRF attacks.

    Checks:
        - URL scheme is http or https
        - Host is not localhost or internal IP
        - DNS resolution doesn't point to internal IP

    Args:
        url: URL to validate.

    Raises:
        SSRFError: If the URL fails SSRF validation.
    """
    parsed = urlparse(url)

    # Check scheme
    if parsed.scheme not in ("http", "https"):
        raise SSRFError(f"Invalid URL scheme: {parsed.scheme}. Only http/https allowed.")

    # Check for empty host
    if not parsed.hostname:
        raise SSRFError("URL has no hostname")

    hostname = parsed.hostname.lower()

    # Block localhost variants
    if hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        raise SSRFError(f"Blocked localhost URL: {hostname}")

    # Block common internal hostnames
    internal_patterns = [
        "internal", "intranet", "corp", "private",
        "metadata", "169.254.169.254",  # Cloud metadata endpoints
    ]
    for pattern in internal_patterns:
        if pattern in hostname:
            raise SSRFError(f"Blocked internal hostname pattern: {hostname}")

    # Block internal IP literals directly (IPv4 or IPv6)
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        ip_literal = None
    else:
        ip_literal = hostname

    if ip_literal is not None:
        if is_internal_ip(ip_literal):
            raise SSRFError(f"Blocked internal IP literal: {ip_literal}")
        return

    # Resolve hostname (IPv4 + IPv6) and block if any internal IPs found
    try:
        addrinfo = socket.getaddrinfo(hostname, None)
        for _, _, _, _, sockaddr in addrinfo:
            ip = sockaddr[0]
            if is_internal_ip(ip):
                raise SSRFError(
                    f"Hostname {hostname} resolves to internal IP: {ip}"
                )
    except socket.gaierror:
        # DNS resolution failed - allow the request to fail naturally later
        logger.debug(f"DNS resolution failed for {hostname}, allowing request")


def validate_pdf_magic_bytes(data: bytes) -> None:
    """Validate PDF magic bytes.

    Args:
        data: PDF file data (at least first 5 bytes needed).

    Raises:
        InvalidPDFError: If magic bytes don't match %PDF-.
    """
    if len(data) < len(PDF_MAGIC_BYTES):
        raise InvalidPDFError(
            f"Data too short to be a PDF ({len(data)} bytes)"
        )
    if not data.startswith(PDF_MAGIC_BYTES):
        # Show first few bytes in hex for debugging
        preview = data[:20].hex()
        raise InvalidPDFError(
            f"Invalid PDF: missing %PDF- header. Got: {preview}..."
        )


def validate_content_type(content_type: Optional[str]) -> None:
    """Validate HTTP content-type for PDF responses.

    Args:
        content_type: Content-Type header value.

    Raises:
        InvalidPDFError: If content-type is not acceptable for PDF.
    """
    if not content_type:
        logger.warning("No Content-Type header, proceeding with magic byte validation")
        return

    # Extract base content type (ignore parameters like charset)
    base_type = content_type.split(";")[0].strip().lower()

    if base_type not in VALID_PDF_CONTENT_TYPES:
        raise InvalidPDFError(
            f"Invalid Content-Type for PDF: {content_type}. "
            f"Expected one of: {', '.join(VALID_PDF_CONTENT_TYPES)}"
        )


@dataclass
class PDFExtractionResult:
    """Result of PDF text extraction.

    Contains the extracted text, page boundary offsets for locator generation,
    and any warnings encountered during extraction.

    Attributes:
        text: Concatenated text from all pages, with page breaks as double newlines.
        page_offsets: List of (start, end) character offsets for each page.
            Offsets are 0-based and reference positions in the `text` field.
            Page numbers are 1-based (page_offsets[0] is page 1).
        warnings: List of warning messages from extraction (e.g., encryption notices,
            missing fonts, extraction failures for specific pages).
        page_count: Total number of pages in the PDF.
        extracted_page_count: Number of pages successfully extracted.
    """

    text: str
    page_offsets: list[tuple[int, int]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    page_count: int = 0
    extracted_page_count: int = 0

    @property
    def has_warnings(self) -> bool:
        """Check if extraction produced any warnings."""
        return len(self.warnings) > 0

    @property
    def success(self) -> bool:
        """Check if extraction produced any text."""
        return self.extracted_page_count > 0

    @property
    def is_complete(self) -> bool:
        """Check if all pages were successfully extracted."""
        return self.extracted_page_count == self.page_count

    def get_page_for_offset(self, char_offset: int) -> int | None:
        """Get the 1-based page number for a character offset.

        Args:
            char_offset: 0-based character position in the text.

        Returns:
            1-based page number, or None if offset is out of range.
        """
        for i, (start, end) in enumerate(self.page_offsets):
            if start <= char_offset < end:
                return i + 1  # 1-based page number
        return None


class PDFExtractor:
    """Extracts text from PDF files with page boundary tracking and security hardening.

    Uses pypdf for text extraction, tracking page boundaries to enable
    accurate evidence snippet locators in the format "page:N:char:S-E".

    Security Features:
        - SSRF protection for URL fetching (blocks internal IPs/localhost)
        - Magic byte validation (verifies %PDF- header)
        - Content-type validation for HTTP responses
        - Configurable size limits

    The extractor is designed for async usage in research workflows,
    running CPU-bound pypdf operations in a thread pool to avoid
    blocking the event loop.

    Attributes:
        max_size: Maximum PDF file size in bytes (default: 10MB).
        max_pages: Maximum number of pages to extract (default: 500).
        timeout: Timeout for URL fetches in seconds (default: 30s).

    Example:
        extractor = PDFExtractor()

        # Extract from bytes (validates magic bytes)
        result = await extractor.extract(pdf_bytes)

        # Extract from URL (with SSRF protection)
        result = await extractor.extract_from_url("https://example.com/doc.pdf")

        # Extract with custom limits
        extractor = PDFExtractor(max_size=5*1024*1024, max_pages=100)

        # Generate locator for a text snippet
        offset = result.text.find("important quote")
        page = result.get_page_for_offset(offset)
        locator = f"page:{page}:char:{offset}-{offset + len('important quote')}"
    """

    def __init__(
        self,
        max_size: int = DEFAULT_MAX_PDF_SIZE,
        max_pages: int = DEFAULT_MAX_PAGES,
        timeout: float = DEFAULT_FETCH_TIMEOUT,
    ):
        """Initialize PDFExtractor with resource limits.

        Args:
            max_size: Maximum PDF file size in bytes (default: 10MB).
            max_pages: Maximum number of pages to extract (default: 500).
            timeout: Timeout for URL fetches in seconds (default: 30s).
        """
        self.max_size = max_size
        self.max_pages = max_pages
        self.timeout = timeout

    async def extract(
        self,
        source: Union[bytes, io.BytesIO],
        *,
        validate_magic: bool = True,
    ) -> PDFExtractionResult:
        """Extract text from a PDF source.

        Validates PDF magic bytes before parsing and runs extraction in a
        thread pool to avoid blocking the event loop.

        Args:
            source: PDF content as bytes or BytesIO stream.
            validate_magic: Whether to validate %PDF- magic bytes (default: True).

        Returns:
            PDFExtractionResult with extracted text, page offsets, and warnings.

        Raises:
            ValueError: If source is not bytes or BytesIO.
            InvalidPDFError: If magic byte validation fails.
            PDFSizeError: If PDF exceeds max_size.
        """
        if isinstance(source, bytes):
            pdf_bytes = source
            source = io.BytesIO(source)
        elif isinstance(source, io.BytesIO):
            # Read bytes for validation, then reset
            pdf_bytes = source.getvalue()
            source.seek(0)
        else:
            raise ValueError(
                f"source must be bytes or BytesIO, got {type(source).__name__}"
            )

        # Check size limit
        if len(pdf_bytes) > self.max_size:
            raise PDFSizeError(
                f"PDF size ({len(pdf_bytes)} bytes) exceeds limit ({self.max_size} bytes)"
            )

        # Validate magic bytes
        if validate_magic:
            validate_pdf_magic_bytes(pdf_bytes)

        # Run CPU-bound extraction in thread pool with timeout
        loop = asyncio.get_event_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, self._extract_sync, source),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            raise PDFSecurityError(
                f"PDF extraction timed out after {self.timeout}s"
            )

    def _extract_page_with_pdfminer(
        self, pdf_bytes: bytes, page_num: int
    ) -> Optional[str]:
        """Extract a single page using pdfminer.six as fallback.

        Args:
            pdf_bytes: Raw PDF bytes.
            page_num: 1-based page number to extract.

        Returns:
            Extracted text for the page, or None if pdfminer.six is unavailable
            or extraction fails.
        """
        pdfminer_hl = _get_pdfminer()
        if pdfminer_hl is None:
            return None

        try:
            output = io.StringIO()
            # Extract single page (page_numbers uses 0-based indices)
            pdfminer_hl.extract_text_to_fp(
                io.BytesIO(pdf_bytes),
                output,
                page_numbers=[page_num - 1],  # 0-based index
            )
            return output.getvalue()
        except Exception as e:
            logger.debug(f"pdfminer.six fallback failed for page {page_num}: {e}")
            return None

    def _extract_full_with_pdfminer_fallback(
        self, pdf_bytes: bytes, original_error: str
    ) -> PDFExtractionResult:
        """Extract PDF using pdfminer.six when pypdf completely fails.

        This is used when PdfReader() fails to parse the PDF at all.
        Extracts page-by-page up to max_pages limit, preserving page boundaries.

        Args:
            pdf_bytes: Raw PDF bytes.
            original_error: Error message from pypdf failure.

        Returns:
            PDFExtractionResult with extracted text and page boundaries.
        """
        pdfminer_hl = _get_pdfminer()
        if pdfminer_hl is None:
            return PDFExtractionResult(
                text="",
                page_offsets=[],
                warnings=[
                    f"pypdf failed: {original_error}",
                    "pdfminer.six fallback unavailable (not installed)",
                ],
                page_count=0,
                extracted_page_count=0,
            )

        try:
            # Extract pages one at a time up to max_pages to preserve boundaries
            page_texts: list[str] = []
            page_offsets: list[tuple[int, int]] = []
            current_offset = 0
            warnings: list[str] = [f"pypdf failed: {original_error}"]

            # Try extracting pages up to max_pages
            for page_num in range(self.max_pages):
                try:
                    output = io.StringIO()
                    pdfminer_hl.extract_text_to_fp(
                        io.BytesIO(pdf_bytes),
                        output,
                        page_numbers=[page_num],  # 0-based index
                    )
                    page_text = output.getvalue()

                    # If page is empty and we have at least one page, we've likely
                    # reached the end of the document
                    if not page_text.strip() and page_num > 0:
                        # Check if this is truly empty or end of document
                        # by trying to get any content
                        if not page_text:
                            break  # Likely end of document

                    page_texts.append(page_text)

                    # Track page boundaries
                    text_len = len(page_text)
                    if page_num > 0:
                        current_offset += 2  # Account for "\n\n" separator
                    page_offsets.append((current_offset, current_offset + text_len))
                    current_offset += text_len

                except Exception as page_error:
                    # If first page fails, the PDF is likely unreadable
                    if page_num == 0:
                        raise page_error
                    # Otherwise, we've reached the end or hit a bad page
                    logger.debug(
                        f"pdfminer.six stopped at page {page_num}: {page_error}"
                    )
                    break

            if page_texts:
                full_text = "\n\n".join(page_texts)
                extracted_count = sum(1 for t in page_texts if t.strip())

                logger.info(
                    f"pdfminer.six fallback succeeded after pypdf failure, "
                    f"extracted {extracted_count} pages, {len(full_text)} chars"
                )

                warnings.append(
                    f"Extracted {extracted_count} pages using pdfminer.six fallback"
                )
                if len(page_texts) >= self.max_pages:
                    warnings.append(
                        f"Extraction stopped at max_pages limit ({self.max_pages})"
                    )

                return PDFExtractionResult(
                    text=full_text,
                    page_offsets=page_offsets,
                    warnings=warnings,
                    page_count=len(page_texts),
                    extracted_page_count=extracted_count,
                )
            else:
                return PDFExtractionResult(
                    text="",
                    page_offsets=[],
                    warnings=[
                        f"pypdf failed: {original_error}",
                        "pdfminer.six fallback returned no text",
                    ],
                    page_count=0,
                    extracted_page_count=0,
                )
        except Exception as e:
            logger.warning(f"pdfminer.six full document fallback failed: {e}")
            return PDFExtractionResult(
                text="",
                page_offsets=[],
                warnings=[
                    f"pypdf failed: {original_error}",
                    f"pdfminer.six fallback also failed: {e}",
                ],
                page_count=0,
                extracted_page_count=0,
            )

    def _extract_sync(self, stream: io.BytesIO) -> PDFExtractionResult:
        """Synchronous extraction implementation with page limits.

        Extracts pages incrementally up to max_pages limit. Each page is
        processed individually to avoid loading the entire document into
        memory at once. Falls back to pdfminer.six when pypdf fails or
        returns empty text for a page.

        Args:
            stream: BytesIO stream containing PDF data.

        Returns:
            PDFExtractionResult with extracted content.
        """
        start_time = time.perf_counter()
        warnings: list[str] = []
        page_texts: list[str] = []
        page_offsets: list[tuple[int, int]] = []

        # Keep the raw bytes for pdfminer fallback
        pdf_bytes = stream.getvalue()

        try:
            reader = PdfReader(stream)
        except Exception as e:
            logger.warning(f"Failed to read PDF with pypdf: {e}")
            # Try pdfminer.six for entire document as fallback
            result = self._extract_full_with_pdfminer_fallback(pdf_bytes, str(e))
            # Record metrics for fallback extraction
            duration = time.perf_counter() - start_time
            status = "success" if result.extracted_page_count > 0 else "failure"
            _record_extraction_metrics(duration, result.extracted_page_count, status)
            return result

        total_page_count = len(reader.pages)
        pages_to_extract = min(total_page_count, self.max_pages)
        current_offset = 0

        # Warn if truncating
        if total_page_count > self.max_pages:
            warnings.append(
                f"PDF has {total_page_count} pages, extracting only first {self.max_pages}"
            )
            logger.warning(
                f"PDF truncated: {total_page_count} pages, limit is {self.max_pages}"
            )

        # Extract pages incrementally (page-by-page for memory efficiency)
        for page_num in range(1, pages_to_extract + 1):
            page_text = ""
            used_fallback = False

            try:
                # Try pypdf first
                page = reader.pages[page_num - 1]
                page_text = page.extract_text() or ""
            except Exception as e:
                logger.warning(f"pypdf failed to extract page {page_num}: {e}")
                # pypdf failed, will try fallback below
                page_text = ""

            # Try pdfminer.six fallback if pypdf returned empty or failed
            if not page_text.strip():
                fallback_text = self._extract_page_with_pdfminer(pdf_bytes, page_num)
                if fallback_text and fallback_text.strip():
                    page_text = fallback_text
                    used_fallback = True
                    logger.debug(f"Page {page_num}: pdfminer.six fallback succeeded")

            # Record result and any warnings
            if not page_text.strip():
                warnings.append(
                    f"Page {page_num}: No text extracted (may be image-based)"
                )
            elif used_fallback:
                warnings.append(
                    f"Page {page_num}: Extracted using pdfminer.six fallback"
                )

            page_texts.append(page_text)

            # Track page boundaries
            text_len = len(page_text)
            # Add separator between pages (double newline)
            if page_num > 1:
                current_offset += 2  # Account for "\n\n" separator
            page_offsets.append((current_offset, current_offset + text_len))
            current_offset += text_len

        # Join pages with double newlines
        full_text = "\n\n".join(page_texts)
        extracted_count = sum(1 for t in page_texts if t.strip())

        logger.debug(
            f"Extracted {extracted_count}/{pages_to_extract} pages "
            f"(total in PDF: {total_page_count}), "
            f"{len(full_text)} chars, {len(warnings)} warnings"
        )

        # Record extraction metrics
        duration = time.perf_counter() - start_time
        if extracted_count == 0:
            status = "failure"
        elif extracted_count < pages_to_extract or warnings:
            status = "partial"
        else:
            status = "success"
        _record_extraction_metrics(duration, extracted_count, status)

        return PDFExtractionResult(
            text=full_text,
            page_offsets=page_offsets,
            warnings=warnings,
            page_count=total_page_count,
            extracted_page_count=extracted_count,
        )

    async def extract_from_url(self, url: str) -> PDFExtractionResult:
        """Extract text from a PDF at a URL with SSRF protection.

        Validates the URL against SSRF attacks before fetching, then
        validates content-type and magic bytes before extraction.

        Security features:
            - SSRF validation on initial URL
            - SSRF re-validation after redirects (validates final destination)
            - Streaming download with early abort at size limit
            - Content-type and magic byte validation

        Args:
            url: URL to fetch the PDF from. Must be http or https.

        Returns:
            PDFExtractionResult with extracted text, page offsets, and warnings.

        Raises:
            SSRFError: If URL fails SSRF validation (including after redirects).
            InvalidPDFError: If content-type or magic bytes are invalid.
            PDFSizeError: If PDF exceeds max_size.
        """
        # Validate initial URL for SSRF before any network request
        validate_url_for_ssrf(url)

        # Import httpx here to avoid import at module level if not needed
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for URL fetching. Install with: pip install httpx"
            )

        logger.debug(f"Fetching PDF from URL: {url}")

        current_url = url
        visited: set[str] = set()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for redirect_index in range(MAX_PDF_REDIRECTS + 1):
                if current_url in visited:
                    raise SSRFError(f"Redirect loop detected for {current_url}")
                visited.add(current_url)

                # Validate URL for SSRF before any network request
                validate_url_for_ssrf(current_url)

                async with client.stream(
                    "GET",
                    current_url,
                    follow_redirects=False,
                    headers={"User-Agent": "foundry-mcp/1.0 PDFExtractor"},
                ) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise InvalidPDFError(
                                f"Redirect response missing Location header: {current_url}"
                            )
                        next_url = urljoin(current_url, location)
                        logger.debug("Redirect detected: %s -> %s", current_url, next_url)
                        current_url = next_url
                        continue

                    response.raise_for_status()

                    # Validate content-type
                    content_type = response.headers.get("content-type")
                    validate_content_type(content_type)

                    # Stream content with size limit enforcement
                    chunks: list[bytes] = []
                    total_size = 0

                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        total_size += len(chunk)
                        if total_size > self.max_size:
                            raise PDFSizeError(
                                f"PDF size exceeds limit ({self.max_size} bytes), "
                                f"download aborted at {total_size} bytes"
                            )
                        chunks.append(chunk)

                    pdf_bytes = b"".join(chunks)

                # Validate magic bytes
                validate_pdf_magic_bytes(pdf_bytes)

                logger.debug(f"Downloaded {len(pdf_bytes)} bytes from {current_url}")

                # Extract text
                return await self.extract(pdf_bytes, validate_magic=False)  # Already validated

        raise InvalidPDFError(
            f"Too many redirects while fetching PDF (max {MAX_PDF_REDIRECTS})"
        )
