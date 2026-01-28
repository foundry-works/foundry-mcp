"""Tests for PDF extractor module.

Tests cover:
1. Valid PDF extraction - successful extraction from bytes
2. SSRF protection - blocking internal IPs, localhost, private networks
3. Magic bytes validation - rejecting invalid PDF headers
4. Size limits enforcement - enforcing configurable max size
5. Page offsets tracking - verifying page boundary calculation
"""

import io

import pytest

from foundry_mcp.core.research.pdf_extractor import (
    DEFAULT_MAX_PDF_SIZE,
    DEFAULT_MAX_PAGES,
    InvalidPDFError,
    PDFExtractionResult,
    PDFExtractor,
    PDFSecurityError,
    PDFSizeError,
    SSRFError,
    is_internal_ip,
    validate_content_type,
    validate_pdf_magic_bytes,
    validate_url_for_ssrf,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_pdf_bytes() -> bytes:
    """Create minimal valid PDF bytes for testing.

    This is a minimal valid PDF that pypdf can parse.
    """
    # Minimal valid PDF structure
    return b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Hello World) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000206 00000 n
trailer
<< /Size 5 /Root 1 0 R >>
startxref
300
%%EOF"""


@pytest.fixture
def multi_page_pdf_bytes() -> bytes:
    """Create a multi-page PDF for page offset testing.

    This creates a minimal 2-page PDF structure.
    """
    return b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Page One) Tj
ET
endstream
endobj
5 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 6 0 R >>
endobj
6 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Page Two) Tj
ET
endstream
endobj
xref
0 7
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000123 00000 n
0000000214 00000 n
0000000308 00000 n
0000000399 00000 n
trailer
<< /Size 7 /Root 1 0 R >>
startxref
493
%%EOF"""


@pytest.fixture
def extractor() -> PDFExtractor:
    """Create a PDFExtractor instance with default settings."""
    return PDFExtractor()


@pytest.fixture
def small_limit_extractor() -> PDFExtractor:
    """Create a PDFExtractor with small size limit for testing."""
    return PDFExtractor(max_size=1024)  # 1KB limit


# =============================================================================
# Test: Magic Bytes Validation
# =============================================================================


class TestMagicBytesValidation:
    """Tests for PDF magic bytes validation."""

    def test_valid_magic_bytes_accepted(self):
        """Test valid %PDF- header is accepted."""
        valid_data = b"%PDF-1.4\n..."
        # Should not raise
        validate_pdf_magic_bytes(valid_data)

    def test_pdf_1_0_magic_accepted(self):
        """Test PDF 1.0 magic bytes are accepted."""
        validate_pdf_magic_bytes(b"%PDF-1.0\nrest of file")

    def test_pdf_1_7_magic_accepted(self):
        """Test PDF 1.7 magic bytes are accepted."""
        validate_pdf_magic_bytes(b"%PDF-1.7\nrest of file")

    def test_pdf_2_0_magic_accepted(self):
        """Test PDF 2.0 magic bytes are accepted."""
        validate_pdf_magic_bytes(b"%PDF-2.0\nrest of file")

    def test_invalid_magic_bytes_rejected(self):
        """Test non-PDF data is rejected."""
        invalid_data = b"This is not a PDF file"
        with pytest.raises(InvalidPDFError) as exc_info:
            validate_pdf_magic_bytes(invalid_data)
        assert "Invalid PDF" in str(exc_info.value)
        assert "%PDF-" in str(exc_info.value)

    def test_empty_data_rejected(self):
        """Test empty data is rejected as too short."""
        with pytest.raises(InvalidPDFError) as exc_info:
            validate_pdf_magic_bytes(b"")
        assert "too short" in str(exc_info.value).lower()

    def test_short_data_rejected(self):
        """Test data shorter than magic bytes is rejected."""
        with pytest.raises(InvalidPDFError) as exc_info:
            validate_pdf_magic_bytes(b"%PDF")  # 4 bytes, need 5
        assert "too short" in str(exc_info.value).lower()

    def test_html_data_rejected(self):
        """Test HTML content is rejected."""
        html_data = b"<!DOCTYPE html><html><body>Not a PDF</body></html>"
        with pytest.raises(InvalidPDFError):
            validate_pdf_magic_bytes(html_data)

    def test_jpeg_magic_rejected(self):
        """Test JPEG magic bytes are rejected."""
        jpeg_magic = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        with pytest.raises(InvalidPDFError):
            validate_pdf_magic_bytes(jpeg_magic)

    def test_png_magic_rejected(self):
        """Test PNG magic bytes are rejected."""
        png_magic = b"\x89PNG\r\n\x1a\n"
        with pytest.raises(InvalidPDFError):
            validate_pdf_magic_bytes(png_magic)

    def test_zip_magic_rejected(self):
        """Test ZIP magic bytes are rejected."""
        zip_magic = b"PK\x03\x04"
        with pytest.raises(InvalidPDFError):
            validate_pdf_magic_bytes(zip_magic)

    def test_exactly_5_bytes_valid(self):
        """Test exactly 5 valid bytes are accepted."""
        validate_pdf_magic_bytes(b"%PDF-")  # Exactly the magic bytes

    def test_error_shows_hex_preview(self):
        """Test error message includes hex preview of invalid data."""
        invalid_data = b"\x00\x01\x02\x03\x04\x05\x06\x07"
        with pytest.raises(InvalidPDFError) as exc_info:
            validate_pdf_magic_bytes(invalid_data)
        # Should include hex representation
        assert "00010203" in str(exc_info.value).lower()


# =============================================================================
# Test: SSRF Protection
# =============================================================================


class TestSSRFProtection:
    """Tests for SSRF protection in URL validation."""

    def test_public_url_accepted(self):
        """Test public HTTPS URL is accepted."""
        # Should not raise
        validate_url_for_ssrf("https://example.com/document.pdf")

    def test_http_url_accepted(self):
        """Test HTTP URL is accepted (not just HTTPS)."""
        validate_url_for_ssrf("http://example.com/document.pdf")

    def test_localhost_blocked(self):
        """Test localhost URL is blocked."""
        with pytest.raises(SSRFError) as exc_info:
            validate_url_for_ssrf("http://localhost/admin")
        assert "localhost" in str(exc_info.value).lower()

    def test_127_0_0_1_blocked(self):
        """Test 127.0.0.1 is blocked."""
        with pytest.raises(SSRFError) as exc_info:
            validate_url_for_ssrf("http://127.0.0.1/admin")
        assert "localhost" in str(exc_info.value).lower() or "127.0.0.1" in str(exc_info.value)

    def test_ipv6_localhost_blocked(self):
        """Test IPv6 localhost (::1) is blocked."""
        with pytest.raises(SSRFError) as exc_info:
            validate_url_for_ssrf("http://[::1]/admin")
        assert "localhost" in str(exc_info.value).lower() or "::1" in str(exc_info.value)

    def test_ipv6_private_literal_blocked(self):
        """Test IPv6 private literal is blocked."""
        with pytest.raises(SSRFError) as exc_info:
            validate_url_for_ssrf("http://[fc00::1]/admin")
        assert "fc00" in str(exc_info.value).lower()

    def test_0_0_0_0_blocked(self):
        """Test 0.0.0.0 is blocked."""
        with pytest.raises(SSRFError) as exc_info:
            validate_url_for_ssrf("http://0.0.0.0/admin")
        assert "localhost" in str(exc_info.value).lower() or "0.0.0.0" in str(exc_info.value)

    def test_internal_hostname_patterns_blocked(self):
        """Test internal hostname patterns are blocked."""
        internal_patterns = [
            "http://internal.company.com/doc.pdf",
            "http://intranet.corp.local/doc.pdf",
            "http://corp.internal/doc.pdf",
            "http://private.network/doc.pdf",
        ]
        for url in internal_patterns:
            with pytest.raises(SSRFError):
                validate_url_for_ssrf(url)

    def test_metadata_endpoint_blocked(self):
        """Test cloud metadata endpoints are blocked."""
        with pytest.raises(SSRFError) as exc_info:
            validate_url_for_ssrf("http://169.254.169.254/latest/meta-data")
        assert "internal" in str(exc_info.value).lower() or "169.254" in str(exc_info.value)

    def test_invalid_scheme_blocked(self):
        """Test non-HTTP schemes are blocked."""
        invalid_schemes = [
            "ftp://example.com/file.pdf",
            "file:///etc/passwd",
            "gopher://example.com/",
            "data:application/pdf;base64,xxx",
        ]
        for url in invalid_schemes:
            with pytest.raises(SSRFError) as exc_info:
                validate_url_for_ssrf(url)
            assert "scheme" in str(exc_info.value).lower()

    def test_empty_hostname_blocked(self):
        """Test URL with no hostname is blocked."""
        with pytest.raises(SSRFError):
            validate_url_for_ssrf("http:///path/to/file")

    def test_url_with_port_accepted(self):
        """Test public URL with custom port is accepted."""
        validate_url_for_ssrf("https://example.com:8443/document.pdf")

    def test_url_with_path_accepted(self):
        """Test URL with complex path is accepted."""
        validate_url_for_ssrf("https://example.com/path/to/document.pdf")

    def test_url_with_query_params_accepted(self):
        """Test URL with query parameters is accepted."""
        validate_url_for_ssrf("https://example.com/doc.pdf?token=abc&version=1")


class TestRedirectSSRFProtection:
    """Tests for SSRF protection across redirect chains."""

    # Minimal valid PDF for testing
    MINIMAL_PDF = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >> endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
trailer << /Size 4 /Root 1 0 R >>
startxref
196
%%EOF
"""

    @pytest.mark.asyncio
    async def test_redirect_to_internal_blocked(self, monkeypatch):
        """Redirects to internal hosts should be blocked."""
        httpx = pytest.importorskip("httpx")
        extractor = PDFExtractor()

        def handler(request):
            if str(request.url) == "https://example.com/doc.pdf":
                return httpx.Response(
                    302, headers={"Location": "http://127.0.0.1/internal.pdf"}
                )
            return httpx.Response(
                200,
                content=b"%PDF-1.4\nx",
                headers={"Content-Type": "application/pdf"},
            )

        transport = httpx.MockTransport(handler)
        real_async_client = httpx.AsyncClient

        def _client_factory(**kwargs):
            return real_async_client(transport=transport, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", _client_factory)

        with pytest.raises(SSRFError):
            await extractor.extract_from_url("https://example.com/doc.pdf")

    @pytest.mark.asyncio
    async def test_redirect_to_private_ip_blocked(self, monkeypatch):
        """Redirects to private IP ranges (10.x, 192.168.x) should be blocked."""
        httpx = pytest.importorskip("httpx")
        extractor = PDFExtractor()

        def handler(request):
            if str(request.url) == "https://example.com/doc.pdf":
                return httpx.Response(
                    302, headers={"Location": "http://10.0.0.1/internal.pdf"}
                )
            return httpx.Response(
                200,
                content=self.MINIMAL_PDF,
                headers={"Content-Type": "application/pdf"},
            )

        transport = httpx.MockTransport(handler)
        real_async_client = httpx.AsyncClient
        monkeypatch.setattr(
            httpx, "AsyncClient",
            lambda **kwargs: real_async_client(transport=transport, **kwargs)
        )

        with pytest.raises(SSRFError) as exc_info:
            await extractor.extract_from_url("https://example.com/doc.pdf")
        assert "10.0.0.1" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_redirect_to_localhost_blocked(self, monkeypatch):
        """Redirects to localhost hostname should be blocked."""
        httpx = pytest.importorskip("httpx")
        extractor = PDFExtractor()

        def handler(request):
            if str(request.url) == "https://example.com/doc.pdf":
                return httpx.Response(
                    302, headers={"Location": "http://localhost/internal.pdf"}
                )
            return httpx.Response(
                200,
                content=self.MINIMAL_PDF,
                headers={"Content-Type": "application/pdf"},
            )

        transport = httpx.MockTransport(handler)
        real_async_client = httpx.AsyncClient
        monkeypatch.setattr(
            httpx, "AsyncClient",
            lambda **kwargs: real_async_client(transport=transport, **kwargs)
        )

        with pytest.raises(SSRFError) as exc_info:
            await extractor.extract_from_url("https://example.com/doc.pdf")
        assert "localhost" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_valid_redirect_succeeds(self, monkeypatch):
        """Redirects to valid external hosts should succeed."""
        httpx = pytest.importorskip("httpx")
        extractor = PDFExtractor()

        def handler(request):
            url = str(request.url)
            if url == "https://example.com/doc.pdf":
                return httpx.Response(
                    302, headers={"Location": "https://cdn.example.com/actual.pdf"}
                )
            if "cdn.example.com" in url:
                return httpx.Response(
                    200,
                    content=self.MINIMAL_PDF,
                    headers={"Content-Type": "application/pdf"},
                )
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        real_async_client = httpx.AsyncClient
        monkeypatch.setattr(
            httpx, "AsyncClient",
            lambda **kwargs: real_async_client(transport=transport, **kwargs)
        )

        result = await extractor.extract_from_url("https://example.com/doc.pdf")
        assert result.page_count == 1

    @pytest.mark.asyncio
    async def test_redirect_loop_detected(self, monkeypatch):
        """Redirect loops should be detected and blocked."""
        httpx = pytest.importorskip("httpx")
        extractor = PDFExtractor()

        def handler(request):
            url = str(request.url)
            if "page1" in url:
                return httpx.Response(
                    302, headers={"Location": "https://example.com/page2.pdf"}
                )
            if "page2" in url:
                return httpx.Response(
                    302, headers={"Location": "https://example.com/page1.pdf"}
                )
            return httpx.Response(
                302, headers={"Location": "https://example.com/page1.pdf"}
            )

        transport = httpx.MockTransport(handler)
        real_async_client = httpx.AsyncClient
        monkeypatch.setattr(
            httpx, "AsyncClient",
            lambda **kwargs: real_async_client(transport=transport, **kwargs)
        )

        with pytest.raises(SSRFError) as exc_info:
            await extractor.extract_from_url("https://example.com/doc.pdf")
        assert "loop" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_too_many_redirects_blocked(self, monkeypatch):
        """More than MAX_PDF_REDIRECTS should be blocked."""
        httpx = pytest.importorskip("httpx")
        extractor = PDFExtractor()

        redirect_count = [0]

        def handler(_request):
            redirect_count[0] += 1
            if redirect_count[0] <= 10:
                return httpx.Response(
                    302,
                    headers={"Location": f"https://example.com/r{redirect_count[0]}.pdf"}
                )
            return httpx.Response(
                200,
                content=self.MINIMAL_PDF,
                headers={"Content-Type": "application/pdf"},
            )

        transport = httpx.MockTransport(handler)
        real_async_client = httpx.AsyncClient
        monkeypatch.setattr(
            httpx, "AsyncClient",
            lambda **kwargs: real_async_client(transport=transport, **kwargs)
        )

        with pytest.raises(InvalidPDFError) as exc_info:
            await extractor.extract_from_url("https://example.com/doc.pdf")
        assert "redirect" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_redirect_to_ipv6_loopback_blocked(self, monkeypatch):
        """Redirects to IPv6 loopback should be blocked."""
        httpx = pytest.importorskip("httpx")
        extractor = PDFExtractor()

        def handler(request):
            if str(request.url) == "https://example.com/doc.pdf":
                return httpx.Response(
                    302, headers={"Location": "http://[::1]/internal.pdf"}
                )
            return httpx.Response(
                200,
                content=self.MINIMAL_PDF,
                headers={"Content-Type": "application/pdf"},
            )

        transport = httpx.MockTransport(handler)
        real_async_client = httpx.AsyncClient
        monkeypatch.setattr(
            httpx, "AsyncClient",
            lambda **kwargs: real_async_client(transport=transport, **kwargs)
        )

        with pytest.raises(SSRFError) as exc_info:
            await extractor.extract_from_url("https://example.com/doc.pdf")
        assert "::1" in str(exc_info.value)


class TestIsInternalIP:
    """Tests for internal IP detection."""

    def test_private_10_range(self):
        """Test 10.x.x.x private range is detected."""
        assert is_internal_ip("10.0.0.1") is True
        assert is_internal_ip("10.255.255.255") is True

    def test_private_172_range(self):
        """Test 172.16-31.x.x private range is detected."""
        assert is_internal_ip("172.16.0.1") is True
        assert is_internal_ip("172.31.255.255") is True

    def test_private_192_range(self):
        """Test 192.168.x.x private range is detected."""
        assert is_internal_ip("192.168.0.1") is True
        assert is_internal_ip("192.168.255.255") is True

    def test_loopback_range(self):
        """Test 127.x.x.x loopback range is detected."""
        assert is_internal_ip("127.0.0.1") is True
        assert is_internal_ip("127.255.255.255") is True

    def test_link_local_range(self):
        """Test 169.254.x.x link-local range is detected."""
        assert is_internal_ip("169.254.0.1") is True
        assert is_internal_ip("169.254.169.254") is True  # Metadata endpoint

    def test_public_ip_not_internal(self):
        """Test public IPs are not flagged as internal."""
        assert is_internal_ip("8.8.8.8") is False  # Google DNS
        assert is_internal_ip("1.1.1.1") is False  # Cloudflare
        assert is_internal_ip("93.184.216.34") is False  # example.com

    def test_ipv6_loopback(self):
        """Test IPv6 loopback is detected."""
        assert is_internal_ip("::1") is True

    def test_ipv6_private(self):
        """Test IPv6 private addresses are detected."""
        assert is_internal_ip("fc00::1") is True  # Unique local
        assert is_internal_ip("fd00::1") is True  # Unique local

    def test_ipv6_link_local(self):
        """Test IPv6 link-local is detected."""
        assert is_internal_ip("fe80::1") is True

    def test_invalid_ip_treated_as_internal(self):
        """Test invalid IP format is treated as internal (fail-safe)."""
        assert is_internal_ip("not.an.ip") is True
        assert is_internal_ip("") is True


# =============================================================================
# Test: Content Type Validation
# =============================================================================


class TestContentTypeValidation:
    """Tests for HTTP content-type validation."""

    def test_application_pdf_accepted(self):
        """Test application/pdf content-type is accepted."""
        validate_content_type("application/pdf")

    def test_application_x_pdf_accepted(self):
        """Test application/x-pdf content-type is accepted."""
        validate_content_type("application/x-pdf")

    def test_octet_stream_accepted(self):
        """Test application/octet-stream is accepted (common for downloads)."""
        validate_content_type("application/octet-stream")

    def test_content_type_with_charset_accepted(self):
        """Test content-type with charset parameter is accepted."""
        validate_content_type("application/pdf; charset=utf-8")

    def test_content_type_case_insensitive(self):
        """Test content-type matching is case-insensitive."""
        validate_content_type("Application/PDF")
        validate_content_type("APPLICATION/PDF")

    def test_none_content_type_accepted_with_warning(self):
        """Test None content-type is accepted (relies on magic bytes)."""
        # Should not raise - will use magic bytes validation
        validate_content_type(None)

    def test_empty_content_type_accepted(self):
        """Test empty content-type is accepted (relies on magic bytes)."""
        validate_content_type("")

    def test_html_content_type_rejected(self):
        """Test text/html content-type is rejected."""
        with pytest.raises(InvalidPDFError) as exc_info:
            validate_content_type("text/html")
        assert "text/html" in str(exc_info.value).lower()

    def test_json_content_type_rejected(self):
        """Test application/json content-type is rejected."""
        with pytest.raises(InvalidPDFError):
            validate_content_type("application/json")

    def test_image_content_type_rejected(self):
        """Test image content-types are rejected."""
        with pytest.raises(InvalidPDFError):
            validate_content_type("image/png")
        with pytest.raises(InvalidPDFError):
            validate_content_type("image/jpeg")


# =============================================================================
# Test: Size Limits Enforcement
# =============================================================================


class TestSizeLimitsEnforcement:
    """Tests for PDF size limit enforcement."""

    @pytest.mark.asyncio
    async def test_small_pdf_accepted(self, small_limit_extractor):
        """Test PDF under size limit is accepted."""
        # Create a PDF smaller than 1KB limit
        small_pdf = b"%PDF-1.4\n" + b"x" * 100
        # Will fail on parsing but should pass size check
        try:
            await small_limit_extractor.extract(small_pdf)
        except (InvalidPDFError, PDFSecurityError):
            pass  # Expected - PDF structure is invalid, but size was OK

    @pytest.mark.asyncio
    async def test_oversized_pdf_rejected(self, small_limit_extractor):
        """Test PDF over size limit is rejected."""
        # Create PDF larger than 1KB limit
        oversized_pdf = b"%PDF-1.4\n" + b"x" * 2000
        with pytest.raises(PDFSizeError) as exc_info:
            await small_limit_extractor.extract(oversized_pdf)
        assert "exceeds limit" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_exactly_at_limit_accepted(self):
        """Test PDF exactly at size limit is accepted."""
        limit = 500
        extractor = PDFExtractor(max_size=limit)
        # Create PDF exactly at limit
        pdf_content = b"%PDF-1.4\n"
        padding = limit - len(pdf_content)
        exact_pdf = pdf_content + b"x" * padding

        assert len(exact_pdf) == limit
        # Should pass size check (may fail on parse, but that's OK)
        try:
            await extractor.extract(exact_pdf)
        except (InvalidPDFError, PDFSecurityError):
            pass  # Size check passed

    @pytest.mark.asyncio
    async def test_one_byte_over_limit_rejected(self):
        """Test PDF one byte over limit is rejected."""
        limit = 500
        extractor = PDFExtractor(max_size=limit)
        oversized_pdf = b"%PDF-1.4\n" + b"x" * (limit - 8)  # 1 byte over

        assert len(oversized_pdf) == limit + 1
        with pytest.raises(PDFSizeError):
            await extractor.extract(oversized_pdf)

    def test_default_max_size_is_10mb(self):
        """Test default max size is 10MB."""
        assert DEFAULT_MAX_PDF_SIZE == 10 * 1024 * 1024

    def test_custom_max_size_respected(self):
        """Test custom max_size is stored correctly."""
        custom_size = 5 * 1024 * 1024  # 5MB
        extractor = PDFExtractor(max_size=custom_size)
        assert extractor.max_size == custom_size


# =============================================================================
# Test: Page Offsets Tracking
# =============================================================================


class TestPageOffsetsTracking:
    """Tests for page boundary offset calculation."""

    def test_get_page_for_offset_in_first_page(self):
        """Test offset lookup returns page 1 for first page content."""
        result = PDFExtractionResult(
            text="Page 1 content\n\nPage 2 content",
            page_offsets=[(0, 14), (16, 30)],  # Account for \n\n separator
            page_count=2,
            extracted_page_count=2,
        )
        assert result.get_page_for_offset(0) == 1
        assert result.get_page_for_offset(5) == 1
        assert result.get_page_for_offset(13) == 1

    def test_get_page_for_offset_in_second_page(self):
        """Test offset lookup returns page 2 for second page content."""
        result = PDFExtractionResult(
            text="Page 1 content\n\nPage 2 content",
            page_offsets=[(0, 14), (16, 30)],
            page_count=2,
            extracted_page_count=2,
        )
        assert result.get_page_for_offset(16) == 2
        assert result.get_page_for_offset(20) == 2
        assert result.get_page_for_offset(29) == 2

    def test_get_page_for_offset_out_of_range(self):
        """Test offset lookup returns None for out-of-range offset."""
        result = PDFExtractionResult(
            text="Page 1 content",
            page_offsets=[(0, 14)],
            page_count=1,
            extracted_page_count=1,
        )
        assert result.get_page_for_offset(100) is None
        assert result.get_page_for_offset(-1) is None

    def test_get_page_for_offset_in_separator(self):
        """Test offset in separator region returns None."""
        result = PDFExtractionResult(
            text="Page 1\n\nPage 2",
            page_offsets=[(0, 6), (8, 14)],
            page_count=2,
            extracted_page_count=2,
        )
        # Offset 6-7 is the \n\n separator
        assert result.get_page_for_offset(6) is None
        assert result.get_page_for_offset(7) is None

    def test_page_offsets_are_zero_based(self):
        """Test page offsets use 0-based indexing."""
        result = PDFExtractionResult(
            text="ABC",
            page_offsets=[(0, 3)],
            page_count=1,
            extracted_page_count=1,
        )
        # Offset 0 should be valid
        assert result.get_page_for_offset(0) == 1

    def test_page_numbers_are_one_based(self):
        """Test returned page numbers are 1-based."""
        result = PDFExtractionResult(
            text="Content",
            page_offsets=[(0, 7)],
            page_count=1,
            extracted_page_count=1,
        )
        # First page is page 1, not page 0
        assert result.get_page_for_offset(0) == 1


# =============================================================================
# Test: PDFExtractionResult Properties
# =============================================================================


class TestPDFExtractionResultProperties:
    """Tests for PDFExtractionResult dataclass properties."""

    def test_has_warnings_true(self):
        """Test has_warnings returns True when warnings present."""
        result = PDFExtractionResult(
            text="Content",
            warnings=["Some warning"],
            page_count=1,
            extracted_page_count=1,
        )
        assert result.has_warnings is True

    def test_has_warnings_false(self):
        """Test has_warnings returns False when no warnings."""
        result = PDFExtractionResult(
            text="Content",
            warnings=[],
            page_count=1,
            extracted_page_count=1,
        )
        assert result.has_warnings is False

    def test_is_complete_true(self):
        """Test is_complete returns True when all pages extracted."""
        result = PDFExtractionResult(
            text="Content",
            page_count=5,
            extracted_page_count=5,
        )
        assert result.is_complete is True

    def test_is_complete_false(self):
        """Test is_complete returns False when pages missing."""
        result = PDFExtractionResult(
            text="Content",
            page_count=5,
            extracted_page_count=3,
        )
        assert result.is_complete is False

    def test_default_values(self):
        """Test default values for optional fields."""
        result = PDFExtractionResult(text="Content")
        assert result.page_offsets == []
        assert result.warnings == []
        assert result.page_count == 0
        assert result.extracted_page_count == 0


# =============================================================================
# Test: PDFExtractor Configuration
# =============================================================================


class TestPDFExtractorConfiguration:
    """Tests for PDFExtractor initialization and configuration."""

    def test_default_max_size(self):
        """Test default max_size is 10MB."""
        extractor = PDFExtractor()
        assert extractor.max_size == DEFAULT_MAX_PDF_SIZE

    def test_default_max_pages(self):
        """Test default max_pages is 500."""
        extractor = PDFExtractor()
        assert extractor.max_pages == DEFAULT_MAX_PAGES

    def test_custom_max_size(self):
        """Test custom max_size is respected."""
        extractor = PDFExtractor(max_size=5000)
        assert extractor.max_size == 5000

    def test_custom_max_pages(self):
        """Test custom max_pages is respected."""
        extractor = PDFExtractor(max_pages=100)
        assert extractor.max_pages == 100

    def test_custom_timeout(self):
        """Test custom timeout is respected."""
        extractor = PDFExtractor(timeout=60.0)
        assert extractor.timeout == 60.0


# =============================================================================
# Test: PDF Extraction with Bytes and BytesIO
# =============================================================================


class TestPDFExtractionInput:
    """Tests for PDF extraction with different input types."""

    @pytest.mark.asyncio
    async def test_extract_from_bytes(self, extractor, simple_pdf_bytes):
        """Test extraction from bytes input."""
        result = await extractor.extract(simple_pdf_bytes)
        assert isinstance(result, PDFExtractionResult)
        assert result.page_count >= 0  # May be 0 for minimal PDF

    @pytest.mark.asyncio
    async def test_extract_from_bytesio(self, extractor, simple_pdf_bytes):
        """Test extraction from BytesIO input."""
        stream = io.BytesIO(simple_pdf_bytes)
        result = await extractor.extract(stream)
        assert isinstance(result, PDFExtractionResult)

    @pytest.mark.asyncio
    async def test_invalid_input_type_rejected(self, extractor):
        """Test invalid input type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            await extractor.extract("not bytes or BytesIO")
        assert "bytes" in str(exc_info.value).lower() or "bytesio" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_extract_with_magic_validation_enabled(self, extractor):
        """Test extraction with magic validation enabled (default)."""
        invalid_data = b"This is not a PDF"
        with pytest.raises(InvalidPDFError):
            await extractor.extract(invalid_data, validate_magic=True)

    @pytest.mark.asyncio
    async def test_extract_with_magic_validation_disabled(self, extractor):
        """Test extraction with magic validation disabled returns empty result.

        When magic validation is disabled and content is invalid, the extractor
        attempts pdfminer.six fallback which gracefully returns empty result.
        """
        invalid_data = b"This is not a PDF but magic validation is off"
        # pdfminer.six fallback handles gracefully, returning empty result
        result = await extractor.extract(invalid_data, validate_magic=False)
        # Should succeed but return empty text with warnings
        assert result.extracted_page_count == 0
        assert result.has_warnings is True
