"""Tests for Phase 3: Binary content guard and extraction wiring.

Tests cover:
1. Binary content guard in SourceSummarizer.summarize_source()
2. Normal text content passes through to summarization
3. DOCX content detection in Tavily search path → extraction before summarization
4. BINARY_UNKNOWN content in Tavily search path → source.content set to None
5. HTML content passes through unchanged
6. python-docx not installed → graceful degradation
7. Content type detection in Tavily extract path
"""

from __future__ import annotations

import io
import zipfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from foundry_mcp.core.research.models.sources import ResearchSource, SourceType
from foundry_mcp.core.research.providers.shared import (
    SourceSummarizationResult,
    SourceSummarizer,
)

# =============================================================================
# Fixtures
# =============================================================================


def _make_docx_bytes() -> bytes:
    """Create a minimal valid DOCX (ZIP with word/document.xml entry)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "word/document.xml",
            '<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body><w:p><w:r><w:t>Hello DOCX</w:t></w:r></w:p></w:body></w:document>",
        )
        zf.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types/>')
    return buf.getvalue()


def _make_binary_string(length: int = 512) -> str:
    """Create a string that looks like binary data (high non-printable ratio)."""
    # Mix of non-printable characters to exceed the 10% threshold
    chars = []
    for i in range(length):
        if i % 3 == 0:
            chars.append(chr(0x01))  # non-printable
        elif i % 3 == 1:
            chars.append(chr(0x80))  # non-printable high byte
        else:
            chars.append("A")  # printable
    return "".join(chars)


@pytest.fixture
def mock_summarizer() -> SourceSummarizer:
    """Create a SourceSummarizer with a mock provider."""
    return SourceSummarizer(
        provider_id="test-provider",
        model="test-model",
        timeout=10.0,
        max_concurrent=2,
    )


# =============================================================================
# Tests: Binary content guard in SourceSummarizer
# =============================================================================


class TestBinaryContentGuard:
    """Tests for the binary content guard in SourceSummarizer.summarize_source()."""

    @pytest.mark.asyncio
    async def test_binary_content_returns_skip_result(self, mock_summarizer):
        """Binary content should be detected and return a skip result."""
        binary_content = _make_binary_string(512)

        result = await mock_summarizer.summarize_source(binary_content)

        assert isinstance(result, SourceSummarizationResult)
        assert "[Content skipped: binary/non-text document detected]" in result.executive_summary
        assert result.key_excerpts == []
        assert result.input_tokens == 0
        assert result.output_tokens == 0

    @pytest.mark.asyncio
    async def test_normal_text_passes_through(self, mock_summarizer):
        """Normal text content should NOT trigger the binary guard."""
        normal_text = "This is a normal article about machine learning. " * 20

        # The summarizer will try to call the LLM provider, which should fail
        # because we haven't set up a real provider. The key assertion is that
        # it does NOT return the binary skip result.
        with pytest.raises(Exception):
            # Should raise because the provider isn't set up, proving
            # the binary guard did NOT intercept normal text
            await mock_summarizer.summarize_source(normal_text)

    @pytest.mark.asyncio
    async def test_html_content_passes_through(self, mock_summarizer):
        """HTML content should NOT trigger the binary guard."""
        html_content = "<html><body><p>This is a test</p></body></html>" + " content" * 100

        with pytest.raises(Exception):
            # Should raise because provider isn't set up, proving
            # the binary guard did NOT intercept HTML
            await mock_summarizer.summarize_source(html_content)

    @pytest.mark.asyncio
    async def test_short_content_passes_through(self, mock_summarizer):
        """Very short content should not be treated as binary."""
        short_content = "Hello"

        with pytest.raises(Exception):
            # Should try to summarize, not skip
            await mock_summarizer.summarize_source(short_content)

    @pytest.mark.asyncio
    async def test_empty_content_passes_through(self, mock_summarizer):
        """Empty content should not trigger binary guard."""
        with pytest.raises(Exception):
            await mock_summarizer.summarize_source("")


# =============================================================================
# Tests: Tavily search path content type detection
# =============================================================================


class TestTavilySearchContentDetection:
    """Tests for content type detection in TavilySearchProvider._apply_source_summarization()."""

    @pytest.mark.asyncio
    async def test_binary_content_sets_source_none(self):
        """BINARY_UNKNOWN content should set source.content to None."""
        from foundry_mcp.core.research.providers.tavily import TavilySearchProvider

        binary_content = _make_binary_string(512)

        provider = TavilySearchProvider.__new__(TavilySearchProvider)
        provider._source_summarizer = MagicMock()
        provider._source_summarizer.summarize_sources = AsyncMock(return_value={})

        sources = [
            ResearchSource(
                id="src-binary",
                title="Binary Source",
                url="https://example.com/file.bin",
                content=binary_content,
                source_type=SourceType.WEB,
            ),
        ]

        result = await provider._apply_source_summarization(sources)

        # Binary content should have been set to None
        assert result[0].content is None

    @pytest.mark.asyncio
    async def test_docx_content_extracts_text(self):
        """DOCX content should be extracted before summarization."""
        from foundry_mcp.core.research.providers.tavily import TavilySearchProvider

        docx_bytes = _make_docx_bytes()
        docx_as_string = docx_bytes.decode("latin-1")

        provider = TavilySearchProvider.__new__(TavilySearchProvider)
        provider._source_summarizer = MagicMock()
        provider._source_summarizer.summarize_sources = AsyncMock(return_value={})

        sources = [
            ResearchSource(
                id="src-docx",
                title="DOCX Source",
                url="https://example.com/doc.docx",
                content=docx_as_string,
                source_type=SourceType.WEB,
            ),
        ]

        with patch(
            "foundry_mcp.core.research.providers.tavily.TavilySearchProvider._extract_docx_content",
            new_callable=AsyncMock,
            return_value="Extracted DOCX text content",
        ):
            result = await provider._apply_source_summarization(sources)

        # Content should be the extracted text
        assert result[0].content == "Extracted DOCX text content"

    @pytest.mark.asyncio
    async def test_docx_extraction_failure_sets_none(self):
        """Failed DOCX extraction should set content to None."""
        from foundry_mcp.core.research.providers.tavily import TavilySearchProvider

        docx_bytes = _make_docx_bytes()
        docx_as_string = docx_bytes.decode("latin-1")

        provider = TavilySearchProvider.__new__(TavilySearchProvider)
        provider._source_summarizer = MagicMock()
        provider._source_summarizer.summarize_sources = AsyncMock(return_value={})

        sources = [
            ResearchSource(
                id="src-docx-fail",
                title="DOCX Source",
                url="https://example.com/doc.docx",
                content=docx_as_string,
                source_type=SourceType.WEB,
            ),
        ]

        with patch(
            "foundry_mcp.core.research.providers.tavily.TavilySearchProvider._extract_docx_content",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await provider._apply_source_summarization(sources)

        assert result[0].content is None

    @pytest.mark.asyncio
    async def test_text_content_passes_through(self):
        """Normal text content should pass through unchanged."""
        from foundry_mcp.core.research.providers.tavily import TavilySearchProvider

        normal_text = "This is a normal article about machine learning. " * 20

        provider = TavilySearchProvider.__new__(TavilySearchProvider)
        provider._source_summarizer = MagicMock()
        provider._source_summarizer.summarize_sources = AsyncMock(return_value={})

        sources = [
            ResearchSource(
                id="src-text",
                title="Text Source",
                url="https://example.com/article",
                content=normal_text,
                source_type=SourceType.WEB,
            ),
        ]

        result = await provider._apply_source_summarization(sources)

        # Content should be unchanged (not set to None by detection)
        assert result[0].content == normal_text

    @pytest.mark.asyncio
    async def test_html_content_passes_through(self):
        """HTML content should pass through unchanged."""
        from foundry_mcp.core.research.providers.tavily import TavilySearchProvider

        html_content = "<html><body><p>Article content here</p></body></html>" + " text" * 100

        provider = TavilySearchProvider.__new__(TavilySearchProvider)
        provider._source_summarizer = MagicMock()
        provider._source_summarizer.summarize_sources = AsyncMock(return_value={})

        sources = [
            ResearchSource(
                id="src-html",
                title="HTML Source",
                url="https://example.com/page.html",
                content=html_content,
                source_type=SourceType.WEB,
            ),
        ]

        result = await provider._apply_source_summarization(sources)

        assert result[0].content == html_content


# =============================================================================
# Tests: DOCX extraction helper
# =============================================================================


class TestDocxExtractionHelper:
    """Tests for TavilySearchProvider._extract_docx_content()."""

    @pytest.mark.asyncio
    async def test_extract_docx_success(self):
        """Successful DOCX extraction returns text."""
        from foundry_mcp.core.research.docx_extractor import DocxExtractionResult
        from foundry_mcp.core.research.providers.tavily import TavilySearchProvider

        provider = TavilySearchProvider.__new__(TavilySearchProvider)

        mock_result = DocxExtractionResult(
            text="Extracted paragraph text",
            warnings=[],
            paragraph_count=1,
            table_count=0,
        )

        with patch(
            "foundry_mcp.core.research.docx_extractor.DocxExtractor",
        ) as MockExtractor:
            instance = MockExtractor.return_value
            instance.extract = AsyncMock(return_value=mock_result)

            result = await provider._extract_docx_content("PK\x03\x04fake docx")

        assert result == "Extracted paragraph text"

    @pytest.mark.asyncio
    async def test_extract_docx_not_installed(self):
        """python-docx not installed returns None."""
        from foundry_mcp.core.research.providers.tavily import TavilySearchProvider

        provider = TavilySearchProvider.__new__(TavilySearchProvider)

        with patch(
            "foundry_mcp.core.research.docx_extractor.DocxExtractor",
        ) as MockExtractor:
            instance = MockExtractor.return_value
            instance.extract = AsyncMock(
                side_effect=RuntimeError("python-docx is required")
            )

            result = await provider._extract_docx_content("PK\x03\x04fake docx")

        assert result is None

    @pytest.mark.asyncio
    async def test_extract_docx_failure(self):
        """DOCX extraction failure returns None."""
        from foundry_mcp.core.research.providers.tavily import TavilySearchProvider

        provider = TavilySearchProvider.__new__(TavilySearchProvider)

        with patch(
            "foundry_mcp.core.research.docx_extractor.DocxExtractor",
        ) as MockExtractor:
            instance = MockExtractor.return_value
            instance.extract = AsyncMock(
                side_effect=Exception("Corrupt DOCX")
            )

            result = await provider._extract_docx_content("PK\x03\x04fake docx")

        assert result is None


# =============================================================================
# Tests: Tavily extract path content type detection
# =============================================================================


class TestTavilyExtractContentDetection:
    """Tests for content type detection in TavilyExtractProvider._parse_response()."""

    def test_binary_content_cleared(self):
        """Binary content in extract results should be cleared."""
        from foundry_mcp.core.research.providers.tavily_extract import (
            TavilyExtractProvider,
        )

        binary_content = _make_binary_string(512)

        provider = TavilyExtractProvider.__new__(TavilyExtractProvider)
        data = {
            "results": [
                {
                    "url": "https://example.com/file.bin",
                    "raw_content": binary_content,
                }
            ]
        }

        sources = provider._parse_response(data, "basic", "markdown")

        assert len(sources) == 1
        assert sources[0].content is None

    def test_text_content_preserved(self):
        """Normal text content should be preserved."""
        from foundry_mcp.core.research.providers.tavily_extract import (
            TavilyExtractProvider,
        )

        text_content = "This is a normal article about research. " * 20

        provider = TavilyExtractProvider.__new__(TavilyExtractProvider)
        data = {
            "results": [
                {
                    "url": "https://example.com/article",
                    "raw_content": text_content,
                }
            ]
        }

        sources = provider._parse_response(data, "basic", "markdown")

        assert len(sources) == 1
        assert sources[0].content is not None
        assert "normal article" in sources[0].content

    def test_docx_content_cleared(self):
        """DOCX content in extract results should be cleared."""
        from foundry_mcp.core.research.providers.tavily_extract import (
            TavilyExtractProvider,
        )

        docx_bytes = _make_docx_bytes()
        docx_as_string = docx_bytes.decode("latin-1")

        provider = TavilyExtractProvider.__new__(TavilyExtractProvider)
        data = {
            "results": [
                {
                    "url": "https://example.com/doc.docx",
                    "raw_content": docx_as_string,
                }
            ]
        }

        sources = provider._parse_response(data, "basic", "markdown")

        assert len(sources) == 1
        # DOCX content should be cleared (not useful as text)
        assert sources[0].content is None
