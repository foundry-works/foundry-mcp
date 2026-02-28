"""Content type classification for research content processing.

Detects content type via magic bytes, URL extension, Content-Type header,
and binary heuristics. Used to route content through the appropriate
extraction pipeline (PDF extractor, DOCX extractor, or text pass-through)
and to guard the summarization pipeline from garbled binary data.

Detection Strategy (ordered by confidence):
    1. Magic bytes (highest confidence):
       - ``%PDF-`` → PDF
       - ``PK\\x03\\x04`` + ``word/document.xml`` → DOCX
       - ``PK\\x03\\x04`` without ``word/`` → BINARY_UNKNOWN
    2. Content-Type header (if provided):
       - ``application/pdf`` → PDF
       - DOCX MIME type → DOCX
    3. URL extension (fallback):
       - ``.docx`` → DOCX, ``.pdf`` → PDF
    4. Binary heuristic — high ratio of non-printable characters → BINARY_UNKNOWN
    5. Default — check for HTML tags → HTML, else TEXT

Key Components:
    - ContentType: Enum of supported content types
    - classify_content: Main classification function
    - is_binary_content: Fast binary detection for summarizer guard

Usage:
    from foundry_mcp.core.research.content_classifier import (
        ContentType,
        classify_content,
        is_binary_content,
    )

    # Classify content for routing
    content_type = classify_content(raw_content, url="https://example.com/doc.docx")

    # Fast binary check for summarizer guard
    if is_binary_content(raw_content):
        # Skip summarization
        ...
"""

from __future__ import annotations

import logging
import re
import zipfile
from enum import Enum
from io import BytesIO
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# =============================================================================
# Content Type Enum
# =============================================================================

# Magic byte signatures
PDF_MAGIC = b"%PDF-"
ZIP_MAGIC = b"PK\x03\x04"

# DOCX MIME types
DOCX_MIME_TYPES = frozenset(
    {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
)

# PDF MIME types
PDF_MIME_TYPES = frozenset(
    {
        "application/pdf",
    }
)

# Binary heuristic threshold — fraction of non-printable characters above
# which string content is considered binary. Set conservatively high to
# avoid false positives on content with some control characters.
BINARY_CHAR_THRESHOLD = 0.10

# Minimum content length for binary heuristic (very short strings are noisy)
BINARY_HEURISTIC_MIN_LENGTH = 32

# HTML detection pattern — matches common HTML tags
_HTML_PATTERN = re.compile(
    r"<(?:html|head|body|div|p|span|a|table|script|style|meta|link|!doctype)\b",
    re.IGNORECASE,
)

# File extensions mapped to content types
_EXTENSION_MAP: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
}


class ContentType(Enum):
    """Content type classification result.

    Values:
        TEXT: Plain text content (default for clean strings)
        HTML: HTML markup content
        PDF: PDF document (detected via magic bytes or metadata)
        DOCX: Microsoft Word OOXML document
        BINARY_UNKNOWN: Unrecognized binary format (should be skipped)
    """

    TEXT = "text"
    HTML = "html"
    PDF = "pdf"
    DOCX = "docx"
    BINARY_UNKNOWN = "binary_unknown"


# =============================================================================
# Classification Functions
# =============================================================================


def classify_content(
    content: str | bytes,
    *,
    url: str | None = None,
    content_type_header: str | None = None,
) -> ContentType:
    """Classify content type using magic bytes, URL extension, and Content-Type header.

    Detection is performed in order of confidence: magic bytes first, then
    Content-Type header, then URL extension, then heuristics.

    Args:
        content: The content to classify (string or bytes).
        url: Optional source URL for extension-based fallback.
        content_type_header: Optional HTTP Content-Type header value.

    Returns:
        ContentType enum value indicating the detected type.
    """
    if content is None or (isinstance(content, (str, bytes)) and len(content) == 0):
        return ContentType.TEXT

    # --- 1. Magic bytes detection (highest confidence) ---
    raw_bytes = content if isinstance(content, bytes) else None

    # For string content, check if it starts with known magic byte patterns
    # (binary data decoded as string often preserves the initial bytes)
    if raw_bytes is None and isinstance(content, str):
        try:
            # Try to get raw bytes from the string for magic byte check
            raw_bytes_candidate = content[:16].encode("latin-1")
        except (UnicodeEncodeError, UnicodeDecodeError):
            raw_bytes_candidate = None

        if raw_bytes_candidate is not None:
            magic_result = _check_magic_bytes(raw_bytes_candidate, full_content=content)
            if magic_result is not None:
                return magic_result
    elif raw_bytes is not None:
        magic_result = _check_magic_bytes(raw_bytes[:16], full_content=raw_bytes)
        if magic_result is not None:
            return magic_result

    # --- 2. Content-Type header ---
    if content_type_header:
        header_result = _check_content_type_header(content_type_header)
        if header_result is not None:
            return header_result

    # --- 3. URL extension fallback ---
    if url:
        ext_result = _check_url_extension(url)
        if ext_result is not None:
            return ext_result

    # --- 4. Binary heuristic (for string content) ---
    if isinstance(content, bytes):
        # Raw bytes with no recognized magic → binary unknown
        return ContentType.BINARY_UNKNOWN

    if isinstance(content, str) and len(content) >= BINARY_HEURISTIC_MIN_LENGTH:
        if _is_binary_string(content):
            return ContentType.BINARY_UNKNOWN

    # --- 5. HTML detection ---
    if isinstance(content, str) and _HTML_PATTERN.search(content[:1024]):
        return ContentType.HTML

    return ContentType.TEXT


def is_binary_content(content: str) -> bool:
    """Fast check for binary content in string form.

    Detects non-printable byte sequences commonly seen when binary data
    (e.g., .docx ZIP content) is decoded as a string. Use this as a guard
    before sending content to an LLM for summarization.

    Args:
        content: String content to check.

    Returns:
        True if the content appears to be binary data decoded as text.
    """
    if not content or not isinstance(content, str):
        return False

    if len(content) < BINARY_HEURISTIC_MIN_LENGTH:
        return False

    return _is_binary_string(content)


# =============================================================================
# Internal Helpers
# =============================================================================


def _check_magic_bytes(
    header: bytes,
    full_content: str | bytes,
) -> ContentType | None:
    """Check magic bytes for known binary formats.

    Args:
        header: First 4-16 bytes of content.
        full_content: Complete content for ZIP probing.

    Returns:
        ContentType if recognized, None otherwise.
    """
    if header.startswith(PDF_MAGIC):
        return ContentType.PDF

    if header.startswith(ZIP_MAGIC):
        return _probe_zip_for_docx(full_content)

    return None


def _probe_zip_for_docx(content: str | bytes) -> ContentType:
    """Probe a ZIP file to check if it's a DOCX.

    Looks for ``word/document.xml`` entry which is the signature of
    an OOXML Word document.

    Args:
        content: Full content (bytes or string with latin-1 encoding).

    Returns:
        DOCX if word/document.xml found, BINARY_UNKNOWN otherwise.
    """
    try:
        if isinstance(content, str):
            raw = content.encode("latin-1")
        else:
            raw = content

        with zipfile.ZipFile(BytesIO(raw), "r") as zf:
            names = zf.namelist()
            if any(name.startswith("word/") for name in names):
                logger.debug("ZIP contains word/ entries — classified as DOCX")
                return ContentType.DOCX
    except (zipfile.BadZipFile, OSError, UnicodeEncodeError, ValueError):
        logger.debug("ZIP probe failed — classifying as BINARY_UNKNOWN")

    return ContentType.BINARY_UNKNOWN


def _check_content_type_header(header: str) -> ContentType | None:
    """Parse Content-Type header for known MIME types.

    Args:
        header: HTTP Content-Type header value (may include charset params).

    Returns:
        ContentType if recognized, None otherwise.
    """
    # Normalize: lowercase, strip whitespace, take only the MIME part
    mime = header.strip().lower().split(";")[0].strip()

    if mime in PDF_MIME_TYPES:
        return ContentType.PDF

    if mime in DOCX_MIME_TYPES:
        return ContentType.DOCX

    return None


def _check_url_extension(url: str) -> ContentType | None:
    """Extract file extension from URL path and map to content type.

    Handles query parameters and fragments correctly.

    Args:
        url: Source URL.

    Returns:
        ContentType if extension matches, None otherwise.
    """
    try:
        parsed = urlparse(url)
        path = parsed.path.lower()

        for ext, type_name in _EXTENSION_MAP.items():
            if path.endswith(ext):
                return ContentType(type_name)
    except (ValueError, AttributeError):
        pass

    return None


def _is_binary_string(content: str) -> bool:
    """Detect if a string contains a high ratio of non-printable characters.

    Samples the content (first 512 characters) to avoid scanning huge
    strings. Characters that are not printable ASCII (0x20-0x7E),
    newlines, tabs, or carriage returns are counted as non-printable.

    Args:
        content: String to check.

    Returns:
        True if non-printable character ratio exceeds threshold.
    """
    sample = content[:512]
    if not sample:
        return False

    non_printable = sum(
        1
        for ch in sample
        if not (0x20 <= ord(ch) <= 0x7E or ch in "\n\r\t")
    )

    ratio = non_printable / len(sample)
    return ratio > BINARY_CHAR_THRESHOLD
