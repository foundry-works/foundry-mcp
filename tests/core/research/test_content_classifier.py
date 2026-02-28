"""Tests for content classifier module.

Tests cover:
1. Magic bytes detection - PDF header, DOCX ZIP, generic ZIP
2. Content-Type header parsing - DOCX and PDF MIME types
3. URL extension fallback - .docx, .pdf, query params
4. Binary heuristic - non-printable character ratio detection
5. HTML detection - common HTML tags
6. Plain text default - clean string content
7. is_binary_content() fast check - garbled vs normal strings
8. Edge cases - empty content, None, conflicting signals
"""

from __future__ import annotations

import io
import zipfile

import pytest

from foundry_mcp.core.research.content_classifier import (
    ContentType,
    classify_content,
    is_binary_content,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def pdf_bytes() -> bytes:
    """Minimal bytes starting with PDF magic header."""
    return b"%PDF-1.4 fake pdf content follows"


@pytest.fixture
def docx_bytes() -> bytes:
    """Create minimal valid DOCX-like ZIP with word/document.xml entry."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", "<w:document>Hello</w:document>")
        zf.writestr("[Content_Types].xml", "<Types/>")
    return buf.getvalue()


@pytest.fixture
def generic_zip_bytes() -> bytes:
    """Create a ZIP file without word/ entries (not a DOCX)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data/file.txt", "just some data")
    return buf.getvalue()


@pytest.fixture
def html_content() -> str:
    """Sample HTML content."""
    return "<html><head><title>Test</title></head><body><p>Hello</p></body></html>"


@pytest.fixture
def plain_text() -> str:
    """Normal plain text content."""
    return "This is a perfectly normal piece of text with no binary content at all."


@pytest.fixture
def binary_garbled_string() -> str:
    """String that looks like decoded binary data (high non-printable ratio)."""
    # Mix of non-printable characters that exceeds the threshold
    non_printable = "".join(chr(i) for i in range(0, 20)) * 5
    printable = "abc"
    return (non_printable + printable) * 3


# =============================================================================
# Test: Magic Bytes Detection
# =============================================================================


class TestMagicBytesDetection:
    """Tests for magic byte signature detection (highest confidence)."""

    def test_pdf_magic_bytes(self, pdf_bytes: bytes):
        """PDF magic bytes (%PDF-) should classify as PDF."""
        result = classify_content(pdf_bytes)
        assert result == ContentType.PDF

    def test_docx_magic_bytes(self, docx_bytes: bytes):
        """DOCX (ZIP with word/document.xml) should classify as DOCX."""
        result = classify_content(docx_bytes)
        assert result == ContentType.DOCX

    def test_generic_zip_magic_bytes(self, generic_zip_bytes: bytes):
        """Generic ZIP (no word/ entries) should classify as BINARY_UNKNOWN."""
        result = classify_content(generic_zip_bytes)
        assert result == ContentType.BINARY_UNKNOWN

    def test_pdf_magic_as_string(self):
        """PDF magic bytes surviving string decode should still be detected."""
        content = "%PDF-1.4 some decoded pdf content here"
        result = classify_content(content)
        assert result == ContentType.PDF

    def test_docx_magic_as_string(self, docx_bytes: bytes):
        """DOCX bytes decoded as latin-1 string should still be detected."""
        content = docx_bytes.decode("latin-1")
        result = classify_content(content)
        assert result == ContentType.DOCX

    def test_corrupted_zip_magic(self):
        """Corrupted ZIP (valid header but bad data) → BINARY_UNKNOWN."""
        # PK header but corrupted body
        content = b"PK\x03\x04" + b"\x00" * 100
        result = classify_content(content)
        assert result == ContentType.BINARY_UNKNOWN

    def test_magic_bytes_take_priority_over_url(self, pdf_bytes: bytes):
        """Magic bytes should win over conflicting URL extension."""
        result = classify_content(pdf_bytes, url="https://example.com/file.docx")
        assert result == ContentType.PDF

    def test_magic_bytes_take_priority_over_header(self, pdf_bytes: bytes):
        """Magic bytes should win over conflicting Content-Type header."""
        result = classify_content(
            pdf_bytes,
            content_type_header="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        assert result == ContentType.PDF


# =============================================================================
# Test: Content-Type Header Parsing
# =============================================================================


class TestContentTypeHeader:
    """Tests for Content-Type header classification."""

    def test_docx_mime_type(self):
        """DOCX MIME type should classify as DOCX."""
        result = classify_content(
            "some content",
            content_type_header="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        assert result == ContentType.DOCX

    def test_pdf_mime_type(self):
        """PDF MIME type should classify as PDF."""
        result = classify_content(
            "some content",
            content_type_header="application/pdf",
        )
        assert result == ContentType.PDF

    def test_mime_type_with_charset(self):
        """MIME type with charset parameter should still be recognized."""
        result = classify_content(
            "some content",
            content_type_header="application/pdf; charset=utf-8",
        )
        assert result == ContentType.PDF

    def test_mime_type_case_insensitive(self):
        """MIME type matching should be case-insensitive."""
        result = classify_content(
            "some content",
            content_type_header="Application/PDF",
        )
        assert result == ContentType.PDF

    def test_unknown_mime_type(self):
        """Unknown MIME type should not force a classification."""
        result = classify_content(
            "Hello world, this is plain text.",
            content_type_header="application/octet-stream",
        )
        assert result == ContentType.TEXT

    def test_header_takes_priority_over_url(self):
        """Content-Type header should win over conflicting URL extension."""
        result = classify_content(
            "some content",
            url="https://example.com/file.docx",
            content_type_header="application/pdf",
        )
        assert result == ContentType.PDF


# =============================================================================
# Test: URL Extension Fallback
# =============================================================================


class TestUrlExtension:
    """Tests for URL file extension classification."""

    def test_docx_extension(self):
        """URL with .docx extension should classify as DOCX."""
        result = classify_content(
            "some text content",
            url="https://example.com/report.docx",
        )
        assert result == ContentType.DOCX

    def test_pdf_extension(self):
        """URL with .pdf extension should classify as PDF."""
        result = classify_content(
            "some text content",
            url="https://example.com/paper.pdf",
        )
        assert result == ContentType.PDF

    def test_docx_extension_with_query_params(self):
        """URL with .docx and query params should still match."""
        result = classify_content(
            "some text content",
            url="https://example.com/report.docx?v=1&token=abc",
        )
        assert result == ContentType.DOCX

    def test_no_matching_extension(self):
        """URL with unrecognized extension should not force classification."""
        result = classify_content(
            "Hello, this is plain text.",
            url="https://example.com/page.html",
        )
        # Should fall through to HTML/TEXT heuristics
        assert result in (ContentType.TEXT, ContentType.HTML)

    def test_no_extension(self):
        """URL without extension should not force classification."""
        result = classify_content(
            "Just some plain text content here.",
            url="https://example.com/api/content",
        )
        assert result == ContentType.TEXT

    def test_case_insensitive_extension(self):
        """URL extension matching should handle uppercase."""
        result = classify_content(
            "some text content",
            url="https://example.com/REPORT.DOCX",
        )
        assert result == ContentType.DOCX


# =============================================================================
# Test: Binary Heuristic
# =============================================================================


class TestBinaryHeuristic:
    """Tests for non-printable character ratio binary detection."""

    def test_high_non_printable_ratio(self, binary_garbled_string: str):
        """String with high non-printable ratio should be BINARY_UNKNOWN."""
        result = classify_content(binary_garbled_string)
        assert result == ContentType.BINARY_UNKNOWN

    def test_normal_text_passes(self, plain_text: str):
        """Normal text should not trigger binary detection."""
        result = classify_content(plain_text)
        assert result == ContentType.TEXT

    def test_raw_bytes_without_magic(self):
        """Raw bytes without recognized magic should be BINARY_UNKNOWN."""
        result = classify_content(b"\x00\x01\x02\x03\x04\x05" * 20)
        assert result == ContentType.BINARY_UNKNOWN

    def test_short_content_skips_heuristic(self):
        """Content shorter than minimum length should skip binary heuristic."""
        # Short string with non-printable chars
        content = "\x00\x01\x02"
        result = classify_content(content)
        # Should default to TEXT since it's too short for heuristic
        assert result == ContentType.TEXT

    def test_text_with_some_control_chars(self):
        """Text with a few control characters should not trigger binary detection."""
        # Below threshold — mostly printable with a few control chars
        content = "Normal text content\x00 with occasional\x01 control chars." * 5
        result = classify_content(content)
        assert result == ContentType.TEXT


# =============================================================================
# Test: HTML Detection
# =============================================================================


class TestHtmlDetection:
    """Tests for HTML tag detection."""

    def test_html_tag_detection(self, html_content: str):
        """Content with <html> tag should classify as HTML."""
        result = classify_content(html_content)
        assert result == ContentType.HTML

    def test_body_tag_detection(self):
        """Content with <body> tag should classify as HTML."""
        result = classify_content("<body><p>Hello</p></body>")
        assert result == ContentType.HTML

    def test_doctype_detection(self):
        """Content with <!DOCTYPE> should classify as HTML."""
        result = classify_content("<!DOCTYPE html><html><body>Hi</body></html>")
        assert result == ContentType.HTML

    def test_div_tag_detection(self):
        """Content with <div> should classify as HTML."""
        content = "<div class='container'><p>Content here</p></div>"
        result = classify_content(content)
        assert result == ContentType.HTML

    def test_no_html_tags(self, plain_text: str):
        """Content without HTML tags should classify as TEXT."""
        result = classify_content(plain_text)
        assert result == ContentType.TEXT


# =============================================================================
# Test: Plain Text Default
# =============================================================================


class TestPlainText:
    """Tests for default TEXT classification."""

    def test_plain_text(self, plain_text: str):
        """Clean string content should default to TEXT."""
        result = classify_content(plain_text)
        assert result == ContentType.TEXT

    def test_multiline_text(self):
        """Multiline plain text should classify as TEXT."""
        content = "Line 1\nLine 2\nLine 3\n\nParagraph two."
        result = classify_content(content)
        assert result == ContentType.TEXT


# =============================================================================
# Test: is_binary_content() Fast Check
# =============================================================================


class TestIsBinaryContent:
    """Tests for the is_binary_content() fast binary guard."""

    def test_garbled_string_is_binary(self, binary_garbled_string: str):
        """String with garbled binary data should return True."""
        assert is_binary_content(binary_garbled_string) is True

    def test_normal_text_is_not_binary(self, plain_text: str):
        """Normal text should return False."""
        assert is_binary_content(plain_text) is False

    def test_html_is_not_binary(self, html_content: str):
        """HTML content should return False."""
        assert is_binary_content(html_content) is False

    def test_empty_string(self):
        """Empty string should return False."""
        assert is_binary_content("") is False

    def test_short_string(self):
        """String shorter than minimum length should return False."""
        assert is_binary_content("hi") is False

    def test_none_input(self):
        """None input should return False (type guard)."""
        assert is_binary_content(None) is False  # type: ignore[arg-type]


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_string(self):
        """Empty string should classify as TEXT."""
        result = classify_content("")
        assert result == ContentType.TEXT

    def test_empty_bytes(self):
        """Empty bytes should classify as TEXT."""
        result = classify_content(b"")
        assert result == ContentType.TEXT

    def test_none_content(self):
        """None content should classify as TEXT."""
        result = classify_content(None)  # type: ignore[arg-type]
        assert result == ContentType.TEXT

    def test_conflicting_url_and_content(self, html_content: str):
        """HTML content with .docx URL — magic bytes check finds no magic,
        Content-Type not provided, URL extension says DOCX, but since URL
        extension has lower priority, the .docx extension wins here because
        it's checked before HTML heuristic."""
        result = classify_content(html_content, url="https://example.com/file.docx")
        # URL extension check happens before HTML heuristic
        assert result == ContentType.DOCX

    def test_conflicting_magic_wins_over_all(self, pdf_bytes: bytes):
        """When magic bytes conflict with URL and header, magic bytes win."""
        result = classify_content(
            pdf_bytes,
            url="https://example.com/file.docx",
            content_type_header="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        assert result == ContentType.PDF

    def test_content_type_enum_values(self):
        """Verify ContentType enum string values for serialization."""
        assert ContentType.TEXT.value == "text"
        assert ContentType.HTML.value == "html"
        assert ContentType.PDF.value == "pdf"
        assert ContentType.DOCX.value == "docx"
        assert ContentType.BINARY_UNKNOWN.value == "binary_unknown"

    def test_whitespace_only_content(self):
        """Whitespace-only content should classify as TEXT."""
        result = classify_content("   \n\t\n   ")
        assert result == ContentType.TEXT

    def test_url_with_fragment(self):
        """URL with fragment should still match extension."""
        result = classify_content(
            "some text content",
            url="https://example.com/report.pdf#page=5",
        )
        # Fragment is part of path in urlparse, but .pdf is before #
        # urlparse separates fragment, so path = /report.pdf
        assert result == ContentType.PDF
