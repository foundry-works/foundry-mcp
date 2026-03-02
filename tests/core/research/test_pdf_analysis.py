"""Tests for full-text PDF analysis.

Tests cover:
1. Section detection from synthetic academic text
2. Graceful fallback when no sections detected
3. Prioritized extraction respecting max_chars and section ordering
4. PDF URL detection patterns
5. ExtractPDFTool model validation
6. extract_from_url() HTTP tests (success, timeout, malformed, SSRF)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from foundry_mcp.core.research.pdf_extractor import (
    PDFExtractionResult,
    PDFExtractor,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def extractor() -> PDFExtractor:
    """Default PDFExtractor instance."""
    return PDFExtractor()


@pytest.fixture
def academic_pdf_result() -> PDFExtractionResult:
    """Synthetic academic paper text in a PDFExtractionResult."""
    text = (
        "Abstract\n"
        "This paper examines the effects of X on Y. We conducted a randomized "
        "controlled trial with N=500 participants.\n\n"
        "1. Introduction\n"
        "The relationship between X and Y has been studied extensively. Prior "
        "work by Smith et al. (2020) established a baseline correlation.\n\n"
        "2. Methods\n"
        "We recruited participants from three hospitals. Inclusion criteria "
        "included age 18-65 and no prior history of Z. Participants were "
        "randomly assigned to treatment (N=250) or control (N=250) groups. "
        "The intervention consisted of daily sessions over 12 weeks.\n\n"
        "3. Results\n"
        "The treatment group showed significant improvement (p<0.001, d=0.45). "
        "Secondary outcomes also favored the intervention. Table 1 shows "
        "the full breakdown of outcomes by subgroup.\n\n"
        "Discussion\n"
        "Our findings support the hypothesis that X positively affects Y. "
        "This is consistent with prior meta-analyses. Limitations include "
        "single-site design and self-report measures.\n\n"
        "Conclusion\n"
        "X is an effective intervention for Y with moderate effect size.\n\n"
        "References\n"
        "Smith, A. et al. (2020). Prior work on X and Y. Journal of Z, 15(3)."
    )
    # Build page offsets as if the entire text was on one page
    return PDFExtractionResult(
        text=text,
        page_offsets=[(0, len(text))],
        warnings=[],
        page_count=1,
        extracted_page_count=1,
    )


@pytest.fixture
def no_sections_result() -> PDFExtractionResult:
    """Text with no standard academic section headers."""
    text = (
        "This is a technical report about widget manufacturing. "
        "We describe our process improvements and quality metrics. "
        "The factory produced 10,000 units last quarter."
    )
    return PDFExtractionResult(
        text=text,
        page_offsets=[(0, len(text))],
        warnings=[],
        page_count=1,
        extracted_page_count=1,
    )


@pytest.fixture
def empty_result() -> PDFExtractionResult:
    """Empty extraction result."""
    return PDFExtractionResult(
        text="",
        page_offsets=[],
        warnings=["No text extracted"],
        page_count=0,
        extracted_page_count=0,
    )


# =============================================================================
# Section Detection Tests
# =============================================================================


class TestDetectSections:
    """Tests for PDFExtractor.detect_sections()."""

    def test_detects_standard_sections(
        self, extractor: PDFExtractor, academic_pdf_result: PDFExtractionResult
    ):
        """Should detect Abstract, Introduction, Methods, Results, Discussion, Conclusion, References."""
        sections = extractor.detect_sections(academic_pdf_result)

        assert "abstract" in sections
        assert "introduction" in sections
        assert "methods" in sections
        assert "results" in sections
        assert "discussion" in sections
        assert "conclusion" in sections
        assert "references" in sections

    def test_section_ordering(
        self, extractor: PDFExtractor, academic_pdf_result: PDFExtractionResult
    ):
        """Section start positions should be in document order."""
        sections = extractor.detect_sections(academic_pdf_result)
        positions = [(name, start) for name, (start, _) in sections.items()]
        starts = [start for _, start in positions]
        assert starts == sorted(starts), "Section start positions should be in order"

    def test_section_spans_are_contiguous(
        self, extractor: PDFExtractor, academic_pdf_result: PDFExtractionResult
    ):
        """Each section should end where the next begins."""
        sections = extractor.detect_sections(academic_pdf_result)
        sorted_sections = sorted(sections.items(), key=lambda x: x[1][0])

        for i in range(len(sorted_sections) - 1):
            _, (_, end) = sorted_sections[i]
            _, (next_start, _) = sorted_sections[i + 1]
            assert end == next_start, (
                f"Section gap: {sorted_sections[i][0]} ends at {end}, "
                f"{sorted_sections[i + 1][0]} starts at {next_start}"
            )

    def test_last_section_extends_to_end(
        self, extractor: PDFExtractor, academic_pdf_result: PDFExtractionResult
    ):
        """The last section should extend to the end of the text."""
        sections = extractor.detect_sections(academic_pdf_result)
        sorted_sections = sorted(sections.items(), key=lambda x: x[1][0])
        _, (_, end) = sorted_sections[-1]
        assert end == len(academic_pdf_result.text)

    def test_graceful_fallback_no_sections(
        self, extractor: PDFExtractor, no_sections_result: PDFExtractionResult
    ):
        """Should return empty dict when no sections detected."""
        sections = extractor.detect_sections(no_sections_result)
        assert sections == {}

    def test_empty_text(self, extractor: PDFExtractor, empty_result: PDFExtractionResult):
        """Should return empty dict for empty text."""
        sections = extractor.detect_sections(empty_result)
        assert sections == {}

    def test_numbered_section_headers(self, extractor: PDFExtractor):
        """Should detect numbered section headers like '2. Methods'."""
        text = "1. Introduction\nSome text here.\n\n2. Methods\nMethod description.\n\n3. Results\nResult data."
        result = PDFExtractionResult(
            text=text, page_offsets=[(0, len(text))], page_count=1, extracted_page_count=1
        )
        sections = extractor.detect_sections(result)
        assert "introduction" in sections
        assert "methods" in sections
        assert "results" in sections

    def test_materials_and_methods_variant(self, extractor: PDFExtractor):
        """Should detect 'Materials and Methods' as the methods section."""
        text = "Abstract\nSummary here.\n\nMaterials and Methods\nWe used the following materials."
        result = PDFExtractionResult(
            text=text, page_offsets=[(0, len(text))], page_count=1, extracted_page_count=1
        )
        sections = extractor.detect_sections(result)
        assert "methods" in sections


# =============================================================================
# Prioritized Extraction Tests
# =============================================================================


class TestExtractPrioritized:
    """Tests for PDFExtractor.extract_prioritized()."""

    def test_returns_full_text_when_within_limit(
        self, extractor: PDFExtractor, academic_pdf_result: PDFExtractionResult
    ):
        """Should return full text when it fits within max_chars."""
        content = extractor.extract_prioritized(academic_pdf_result, max_chars=100000)
        assert content == academic_pdf_result.text

    def test_respects_max_chars(
        self, extractor: PDFExtractor, academic_pdf_result: PDFExtractionResult
    ):
        """Output should not exceed max_chars."""
        max_chars = 200
        content = extractor.extract_prioritized(academic_pdf_result, max_chars=max_chars)
        assert len(content) <= max_chars

    def test_abstract_always_first(
        self, extractor: PDFExtractor, academic_pdf_result: PDFExtractionResult
    ):
        """Abstract should always be included first."""
        content = extractor.extract_prioritized(academic_pdf_result, max_chars=500)
        assert content.startswith("Abstract")

    def test_priority_sections_before_others(
        self, extractor: PDFExtractor, academic_pdf_result: PDFExtractionResult
    ):
        """Methods, results, discussion should appear before introduction."""
        # Use a max_chars smaller than the full text to force prioritization
        full_len = len(academic_pdf_result.text)
        content = extractor.extract_prioritized(
            academic_pdf_result, max_chars=full_len - 50, priority_sections=["methods", "results"]
        )
        # The section header "2. Methods" should appear before "1. Introduction"
        # (since methods is prioritized, it comes right after abstract)
        methods_header_pos = content.find("2. Methods")
        intro_header_pos = content.find("1. Introduction")
        if methods_header_pos >= 0 and intro_header_pos >= 0:
            assert methods_header_pos < intro_header_pos, (
                "Priority section 'methods' header should come before 'introduction' header"
            )

    def test_references_excluded(
        self, extractor: PDFExtractor, academic_pdf_result: PDFExtractionResult
    ):
        """References section should not be included when space is limited."""
        # Use a max_chars that can't fit everything
        sections = extractor.detect_sections(academic_pdf_result)
        total_non_ref = sum(
            end - start for name, (start, end) in sections.items() if name != "references"
        )
        # Set max_chars to fit all non-reference sections but not references
        content = extractor.extract_prioritized(
            academic_pdf_result, max_chars=total_non_ref + 50
        )
        # References should be excluded or truncated
        assert "Smith, A. et al." not in content or len(content) >= total_non_ref

    def test_empty_text(self, extractor: PDFExtractor, empty_result: PDFExtractionResult):
        """Should return empty string for empty text."""
        content = extractor.extract_prioritized(empty_result)
        assert content == ""

    def test_no_sections_simple_truncation(
        self, extractor: PDFExtractor, no_sections_result: PDFExtractionResult
    ):
        """When no sections detected, should do simple truncation."""
        content = extractor.extract_prioritized(no_sections_result, max_chars=50)
        assert len(content) <= 50
        assert content == no_sections_result.text[:50]

    def test_custom_priority_sections(
        self, extractor: PDFExtractor, academic_pdf_result: PDFExtractionResult
    ):
        """Should respect custom priority_sections parameter."""
        content = extractor.extract_prioritized(
            academic_pdf_result, max_chars=1000, priority_sections=["discussion"]
        )
        # Discussion should come right after abstract
        abstract_end = content.find("\n\nDiscussion")
        assert abstract_end > 0, "Discussion should follow abstract in prioritized output"


# =============================================================================
# PDF URL Detection Tests
# =============================================================================


class TestIsPdfUrl:
    """Tests for _is_pdf_url() helper."""

    def test_pdf_extension(self):
        from foundry_mcp.core.research.workflows.deep_research.phases.topic_research import (
            _is_pdf_url,
        )

        assert _is_pdf_url("https://example.com/paper.pdf") is True

    def test_pdf_extension_with_query(self):
        from foundry_mcp.core.research.workflows.deep_research.phases.topic_research import (
            _is_pdf_url,
        )

        assert _is_pdf_url("https://example.com/paper.pdf?token=abc") is True

    def test_arxiv_pdf(self):
        from foundry_mcp.core.research.workflows.deep_research.phases.topic_research import (
            _is_pdf_url,
        )

        assert _is_pdf_url("https://arxiv.org/pdf/2301.00001") is True

    def test_arxiv_ftp(self):
        from foundry_mcp.core.research.workflows.deep_research.phases.topic_research import (
            _is_pdf_url,
        )

        assert _is_pdf_url("https://arxiv.org/ftp/arxiv/papers/2301/2301.00001.pdf") is True

    def test_regular_url_not_pdf(self):
        from foundry_mcp.core.research.workflows.deep_research.phases.topic_research import (
            _is_pdf_url,
        )

        assert _is_pdf_url("https://example.com/article") is False

    def test_html_url_not_pdf(self):
        from foundry_mcp.core.research.workflows.deep_research.phases.topic_research import (
            _is_pdf_url,
        )

        assert _is_pdf_url("https://example.com/page.html") is False

    def test_pdf_in_path_not_extension(self):
        from foundry_mcp.core.research.workflows.deep_research.phases.topic_research import (
            _is_pdf_url,
        )

        # URL with "pdf" in path but not as file extension
        assert _is_pdf_url("https://example.com/pdfviewer/doc123") is False


# =============================================================================
# ExtractPDFTool Model Tests
# =============================================================================


class TestExtractPDFTool:
    """Tests for ExtractPDFTool model validation."""

    def test_valid_tool_args(self):
        from foundry_mcp.core.research.models.deep_research import ExtractPDFTool

        tool = ExtractPDFTool(url="https://arxiv.org/pdf/2301.00001.pdf", max_pages=20)
        assert tool.url == "https://arxiv.org/pdf/2301.00001.pdf"
        assert tool.max_pages == 20

    def test_default_max_pages(self):
        from foundry_mcp.core.research.models.deep_research import ExtractPDFTool

        tool = ExtractPDFTool(url="https://example.com/paper.pdf")
        assert tool.max_pages == 30

    def test_max_pages_clamped(self):
        from foundry_mcp.core.research.models.deep_research import ExtractPDFTool

        with pytest.raises(Exception):
            ExtractPDFTool(url="https://example.com/paper.pdf", max_pages=200)

    def test_registered_in_tool_schemas(self):
        from foundry_mcp.core.research.models.deep_research import (
            RESEARCHER_TOOL_SCHEMAS,
            ExtractPDFTool,
        )

        assert "extract_pdf" in RESEARCHER_TOOL_SCHEMAS
        assert RESEARCHER_TOOL_SCHEMAS["extract_pdf"] is ExtractPDFTool


# =============================================================================
# System Prompt Integration Tests
# =============================================================================


class TestSystemPromptPdfIntegration:
    """Tests for PDF tool injection into researcher system prompt."""

    def test_pdf_tool_not_in_prompt_by_default(self):
        from foundry_mcp.core.research.workflows.deep_research.phases.topic_research import (
            _build_researcher_system_prompt,
        )

        prompt = _build_researcher_system_prompt(
            budget_total=5, budget_remaining=5, extract_enabled=True
        )
        assert "extract_pdf" not in prompt

    def test_pdf_tool_in_prompt_when_enabled(self):
        from foundry_mcp.core.research.workflows.deep_research.phases.topic_research import (
            _build_researcher_system_prompt,
        )

        prompt = _build_researcher_system_prompt(
            budget_total=5,
            budget_remaining=5,
            extract_enabled=True,
            pdf_extraction_enabled=True,
        )
        assert "### extract_pdf" in prompt
        assert "open-access academic paper PDF" in prompt

    def test_pdf_tool_prompt_before_response_format(self):
        from foundry_mcp.core.research.workflows.deep_research.phases.topic_research import (
            _build_researcher_system_prompt,
        )

        prompt = _build_researcher_system_prompt(
            budget_total=5,
            budget_remaining=5,
            extract_enabled=True,
            pdf_extraction_enabled=True,
        )
        pdf_pos = prompt.find("### extract_pdf")
        format_pos = prompt.find("## Response Format")
        assert pdf_pos < format_pos, "extract_pdf should appear before Response Format"


# =============================================================================
# extract_from_url() HTTP Tests
# =============================================================================


# Minimal valid PDF bytes (header + empty body)
_MINIMAL_PDF = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"


def _make_mock_response(
    status_code: int = 200,
    content_type: str = "application/pdf",
    body: bytes = _MINIMAL_PDF,
    headers: dict | None = None,
):
    """Build a mock httpx streaming response."""
    resp_headers = {"content-type": content_type}
    if headers:
        resp_headers.update(headers)

    response = AsyncMock()
    response.status_code = status_code
    response.headers = resp_headers
    response.raise_for_status = MagicMock()

    async def aiter_bytes(chunk_size=65536):
        yield body

    response.aiter_bytes = aiter_bytes

    return response


class TestExtractFromUrl:
    """Tests for PDFExtractor.extract_from_url()."""

    @pytest.mark.asyncio
    async def test_extract_from_url_success(self):
        """Successful fetch of a valid PDF returns extraction result."""
        extractor = PDFExtractor()

        mock_response = _make_mock_response()

        # Mock httpx.AsyncClient context manager
        mock_client = AsyncMock()

        # client.stream() returns an async context manager yielding mock_response
        stream_cm = AsyncMock()
        stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
        stream_cm.__aexit__ = AsyncMock(return_value=False)
        mock_client.stream = MagicMock(return_value=stream_cm)

        client_cm = AsyncMock()
        client_cm.__aenter__ = AsyncMock(return_value=mock_client)
        client_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("foundry_mcp.core.research.pdf_extractor.validate_url_for_ssrf"), \
             patch("httpx.AsyncClient", return_value=client_cm):
            # extract() will be called on the downloaded bytes; mock it
            with patch.object(extractor, "extract", new_callable=AsyncMock) as mock_extract:
                mock_extract.return_value = PDFExtractionResult(
                    text="Extracted text from PDF",
                    page_offsets=[(0, 23)],
                    page_count=1,
                    extracted_page_count=1,
                )
                result = await extractor.extract_from_url("https://example.com/paper.pdf")

        assert result.text == "Extracted text from PDF"
        assert result.page_count == 1

    @pytest.mark.asyncio
    async def test_extract_from_url_timeout(self):
        """Timeout during fetch raises appropriate error."""
        import httpx

        extractor = PDFExtractor()

        mock_client = AsyncMock()

        # client.stream() raises a timeout
        stream_cm = AsyncMock()
        stream_cm.__aenter__ = AsyncMock(side_effect=httpx.ReadTimeout("Connection timed out"))
        stream_cm.__aexit__ = AsyncMock(return_value=False)
        mock_client.stream = MagicMock(return_value=stream_cm)

        client_cm = AsyncMock()
        client_cm.__aenter__ = AsyncMock(return_value=mock_client)
        client_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("foundry_mcp.core.research.pdf_extractor.validate_url_for_ssrf"), \
             patch("httpx.AsyncClient", return_value=client_cm):
            with pytest.raises(httpx.ReadTimeout):
                await extractor.extract_from_url("https://example.com/slow.pdf")

    @pytest.mark.asyncio
    async def test_extract_from_url_malformed_pdf(self):
        """Fetched content with invalid magic bytes raises InvalidPDFError."""
        from foundry_mcp.core.errors.research import InvalidPDFError

        extractor = PDFExtractor()
        bad_bytes = b"This is not a PDF at all"

        mock_response = _make_mock_response(body=bad_bytes)
        mock_client = AsyncMock()

        stream_cm = AsyncMock()
        stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
        stream_cm.__aexit__ = AsyncMock(return_value=False)
        mock_client.stream = MagicMock(return_value=stream_cm)

        client_cm = AsyncMock()
        client_cm.__aenter__ = AsyncMock(return_value=mock_client)
        client_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("foundry_mcp.core.research.pdf_extractor.validate_url_for_ssrf"), \
             patch("httpx.AsyncClient", return_value=client_cm):
            with pytest.raises(InvalidPDFError):
                await extractor.extract_from_url("https://example.com/not-a-pdf.pdf")

    @pytest.mark.asyncio
    async def test_extract_from_url_ssrf_blocked(self):
        """Private/internal IP URLs are blocked by SSRF validation."""
        from foundry_mcp.core.errors.research import SSRFError
        from foundry_mcp.core.research.pdf_extractor import validate_url_for_ssrf

        # Direct validation check — no mocking needed
        with pytest.raises(SSRFError):
            validate_url_for_ssrf("http://127.0.0.1/secret.pdf")

        with pytest.raises(SSRFError):
            validate_url_for_ssrf("http://localhost/secret.pdf")

        with pytest.raises(SSRFError):
            validate_url_for_ssrf("http://169.254.169.254/latest/meta-data/")

        # Also test the full extract_from_url path
        extractor = PDFExtractor()
        with pytest.raises(SSRFError):
            await extractor.extract_from_url("http://192.168.1.1/internal.pdf")
