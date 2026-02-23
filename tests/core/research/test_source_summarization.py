"""Tests for Phase 1: Fetch-time source summarization.

Tests cover:
- SourceSummarizer with mock LLM responses
- Timeout fallback to original content
- Parallel summarization with multiple sources
- Opt-out via config flag
- raw_content preservation on ResearchSource
- Backward-compat: existing ResearchSource without raw_content deserializes
- SourceSummarizationResult parsing
- max_content_length truncation before summarization (Phase 2 alignment)
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from foundry_mcp.core.research.models.sources import ResearchSource, SourceType
from foundry_mcp.core.research.providers.shared import (
    SourceSummarizationResult,
    SourceSummarizer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_sources() -> list[ResearchSource]:
    """Create a list of sample ResearchSource objects with content."""
    return [
        ResearchSource(
            id="src-001",
            title="Source One",
            url="https://example.com/1",
            content="This is a long article about machine learning. " * 50,
            source_type=SourceType.WEB,
        ),
        ResearchSource(
            id="src-002",
            title="Source Two",
            url="https://example.com/2",
            content="Deep learning has transformed computer vision. " * 40,
            source_type=SourceType.WEB,
        ),
        ResearchSource(
            id="src-003",
            title="Source Three (no content)",
            url="https://example.com/3",
            content=None,
            source_type=SourceType.WEB,
        ),
    ]


@pytest.fixture
def mock_summarizer() -> SourceSummarizer:
    """Create a SourceSummarizer with a mock provider."""
    return SourceSummarizer(
        provider_id="test-provider",
        model="test-model",
        timeout=10.0,
        max_concurrent=2,
    )


# ---------------------------------------------------------------------------
# ResearchSource model tests
# ---------------------------------------------------------------------------


class TestResearchSourceRawContent:
    """Tests for raw_content field on ResearchSource."""

    def test_raw_content_default_none(self):
        """raw_content defaults to None for backward compatibility."""
        source = ResearchSource(title="Test")
        assert source.raw_content is None

    def test_raw_content_set_and_get(self):
        """raw_content can be set and retrieved."""
        source = ResearchSource(title="Test", raw_content="original text")
        assert source.raw_content == "original text"

    def test_raw_content_excluded_from_to_dict(self):
        """raw_content is excluded from API-facing to_dict() output."""
        source = ResearchSource(
            title="Test",
            content="summarized",
            raw_content="original very long content",
        )
        data = source.to_dict()
        assert "raw_content" not in data
        assert data["content"] == "summarized"

    def test_raw_content_included_in_model_dump(self):
        """raw_content is preserved in full model_dump() for persistence."""
        source = ResearchSource(
            title="Test",
            content="summarized",
            raw_content="original content",
        )
        data = source.model_dump()
        assert data["raw_content"] == "original content"

    def test_backward_compat_deserialize_without_raw_content(self):
        """Existing serialized data without raw_content deserializes cleanly."""
        data = {
            "id": "src-old",
            "title": "Old Source",
            "content": "some content",
            "source_type": "web",
            "quality": "unknown",
        }
        source = ResearchSource.model_validate(data)
        assert source.raw_content is None
        assert source.content == "some content"

    def test_round_trip_with_raw_content(self):
        """raw_content survives model_dump() -> reconstruction round-trip."""
        source = ResearchSource(
            title="Test",
            content="summary",
            raw_content="original",
        )
        data = source.model_dump()
        reconstructed = ResearchSource(**data)
        assert reconstructed.raw_content == "original"
        assert reconstructed.content == "summary"


# ---------------------------------------------------------------------------
# SourceSummarizer._parse_summary_response tests
# ---------------------------------------------------------------------------


class TestParseSummaryResponse:
    """Tests for SourceSummarizer response parsing."""

    def test_parse_well_formatted_response(self):
        """Parse a response with both Executive Summary and Key Excerpts sections."""
        response = (
            "## Executive Summary\n"
            "This article discusses the impact of AI.\n\n"
            "## Key Excerpts\n"
            '- "AI is transforming healthcare"\n'
            '- "Machine learning models can predict outcomes"\n'
            '- "Deep learning enables new applications"\n'
        )
        summary, excerpts = SourceSummarizer._parse_summary_response(response)
        assert "impact of AI" in summary
        assert len(excerpts) == 3
        assert "AI is transforming healthcare" in excerpts[0]

    def test_parse_max_five_excerpts(self):
        """Parser limits to 5 excerpts even if more are provided."""
        lines = [f'- "Excerpt number {i}"' for i in range(10)]
        response = "## Executive Summary\nSummary text.\n\n## Key Excerpts\n" + "\n".join(lines)
        _, excerpts = SourceSummarizer._parse_summary_response(response)
        assert len(excerpts) == 5

    def test_parse_no_sections(self):
        """Falls back to entire response as summary when no sections found."""
        response = "Just a plain text summary without any headers."
        summary, excerpts = SourceSummarizer._parse_summary_response(response)
        assert summary == response
        assert excerpts == []

    def test_parse_only_excerpts_section(self):
        """Handles response with only Key Excerpts section."""
        response = (
            "Some introductory text.\n\n"
            "## Key Excerpts\n"
            '- "First quote"\n'
            '- "Second quote"\n'
        )
        summary, excerpts = SourceSummarizer._parse_summary_response(response)
        assert "introductory text" in summary
        assert len(excerpts) == 2

    def test_parse_bullet_variants(self):
        """Handles different bullet styles (-, *, bullet char)."""
        response = (
            "## Executive Summary\nSummary.\n\n"
            "## Key Excerpts\n"
            '- "Dash bullet"\n'
            '* "Star bullet"\n'
            '\u2022 "Unicode bullet"\n'
        )
        _, excerpts = SourceSummarizer._parse_summary_response(response)
        assert len(excerpts) == 3


class TestFormatSummarizedContent:
    """Tests for SourceSummarizer.format_summarized_content."""

    def test_format_with_excerpts(self):
        """Formats summary + excerpts into structured content."""
        result = SourceSummarizer.format_summarized_content(
            "This is the summary.",
            ["Quote one", "Quote two"],
        )
        assert "This is the summary." in result
        assert "**Key Excerpts:**" in result
        assert '"Quote one"' in result
        assert '"Quote two"' in result

    def test_format_without_excerpts(self):
        """Formats summary without excerpts section when list is empty."""
        result = SourceSummarizer.format_summarized_content("Just a summary.", [])
        assert result == "Just a summary."
        assert "Key Excerpts" not in result


# ---------------------------------------------------------------------------
# SourceSummarizer.summarize_source tests
# ---------------------------------------------------------------------------


class TestSummarizeSource:
    """Tests for SourceSummarizer.summarize_source with mocked provider."""

    @pytest.mark.asyncio
    async def test_summarize_source_success(self, mock_summarizer):
        """Successful summarization returns parsed result."""
        mock_response = (
            "## Executive Summary\nAI is changing the world.\n\n"
            "## Key Excerpts\n"
            '- "AI models can predict"\n'
            '- "Deep learning advances"\n'
        )
        mock_result = MagicMock()
        mock_result.status = MagicMock()
        mock_result.status.name = "SUCCESS"
        mock_result.content = mock_response
        mock_result.tokens = MagicMock(input_tokens=100, output_tokens=50)
        mock_result.stderr = None

        # Mock the provider system
        mock_provider = MagicMock()
        mock_provider.generate = MagicMock(return_value=mock_result)

        with patch(
            "foundry_mcp.core.providers.registry.resolve_provider",
            return_value=mock_provider,
        ), patch(
            "foundry_mcp.core.providers.ProviderStatus"
        ) as mock_status:
            mock_status.SUCCESS = mock_result.status
            result = await mock_summarizer.summarize_source("Long content here...")

        assert isinstance(result, SourceSummarizationResult)
        assert "AI is changing the world" in result.executive_summary
        assert len(result.key_excerpts) == 2
        assert result.input_tokens == 100
        assert result.output_tokens == 50

    @pytest.mark.asyncio
    async def test_summarize_source_provider_failure(self, mock_summarizer):
        """Provider failure raises RuntimeError."""
        mock_result = MagicMock()
        mock_result.status = MagicMock()
        mock_result.status.name = "ERROR"
        mock_result.content = ""
        mock_result.stderr = "Provider unavailable"

        mock_provider = MagicMock()
        mock_provider.generate = MagicMock(return_value=mock_result)

        with patch(
            "foundry_mcp.core.providers.registry.resolve_provider",
            return_value=mock_provider,
        ), patch(
            "foundry_mcp.core.providers.ProviderStatus"
        ) as mock_status:
            # Make the status check fail (result.status != SUCCESS)
            mock_status.SUCCESS = MagicMock()  # Different from mock_result.status
            with pytest.raises(RuntimeError, match="Summarization failed"):
                await mock_summarizer.summarize_source("Content")


# ---------------------------------------------------------------------------
# SourceSummarizer.summarize_sources tests
# ---------------------------------------------------------------------------


class TestSummarizeSources:
    """Tests for parallel source summarization."""

    @pytest.mark.asyncio
    async def test_summarize_sources_parallel(self, mock_summarizer, sample_sources):
        """Successfully summarizes multiple sources in parallel."""
        mock_result = SourceSummarizationResult(
            executive_summary="Summary text",
            key_excerpts=["Quote one"],
            input_tokens=50,
            output_tokens=25,
        )

        with patch.object(
            mock_summarizer,
            "summarize_source",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            results = await mock_summarizer.summarize_sources(sample_sources)

        # src-001 and src-002 have content, src-003 does not
        assert "src-001" in results
        assert "src-002" in results
        assert "src-003" not in results
        assert results["src-001"].executive_summary == "Summary text"

    @pytest.mark.asyncio
    async def test_summarize_sources_skips_empty_content(self, mock_summarizer, sample_sources):
        """Sources without content are skipped."""
        call_count = 0

        async def mock_summarize(content):
            nonlocal call_count
            call_count += 1
            return SourceSummarizationResult(executive_summary="S", key_excerpts=[])

        with patch.object(mock_summarizer, "summarize_source", side_effect=mock_summarize):
            await mock_summarizer.summarize_sources(sample_sources)

        # Only src-001 and src-002 have content
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_summarize_sources_timeout_fallback(self, mock_summarizer, sample_sources):
        """Timed-out sources are omitted from results (original content preserved)."""

        async def slow_summarize(content):
            await asyncio.sleep(100)  # Will be cancelled by timeout
            return SourceSummarizationResult(executive_summary="S", key_excerpts=[])

        # Use a very short timeout
        mock_summarizer._timeout = 0.01

        with patch.object(mock_summarizer, "summarize_source", side_effect=slow_summarize):
            results = await mock_summarizer.summarize_sources(sample_sources)

        # All should have timed out
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_summarize_sources_partial_failure(self, mock_summarizer, sample_sources):
        """One source failing doesn't prevent others from succeeding."""
        call_count = 0

        async def sometimes_fail(content):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Provider error")
            return SourceSummarizationResult(
                executive_summary="Success summary",
                key_excerpts=["quote"],
            )

        with patch.object(mock_summarizer, "summarize_source", side_effect=sometimes_fail):
            results = await mock_summarizer.summarize_sources(sample_sources)

        # One succeeded, one failed, one skipped (no content)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Config flag tests
# ---------------------------------------------------------------------------


class TestFetchTimeSummarizationConfig:
    """Tests for fetch_time_summarization config flag."""

    def test_config_default_enabled(self):
        """fetch_time_summarization defaults to True."""
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig()
        assert config.deep_research_fetch_time_summarization is True

    def test_config_opt_out(self):
        """fetch_time_summarization can be disabled."""
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig(deep_research_fetch_time_summarization=False)
        assert config.deep_research_fetch_time_summarization is False

    def test_config_from_toml_enabled(self):
        """TOML parsing correctly reads fetch_time_summarization."""
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig.from_toml_dict({"deep_research_fetch_time_summarization": True})
        assert config.deep_research_fetch_time_summarization is True

    def test_config_from_toml_disabled(self):
        """TOML parsing correctly reads fetch_time_summarization=false."""
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig.from_toml_dict({"deep_research_fetch_time_summarization": False})
        assert config.deep_research_fetch_time_summarization is False

    def test_config_summarization_provider(self):
        """Summarization provider config fields are accessible."""
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig(
            deep_research_summarization_provider="gemini",
            deep_research_summarization_model="flash",
        )
        assert config.get_summarization_provider() == "gemini"
        assert config.get_summarization_model() == "flash"

    def test_config_summarization_provider_fallback(self):
        """Summarization provider falls back to default_provider when not set."""
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig(default_provider="claude")
        assert config.get_summarization_provider() == "claude"
        assert config.get_summarization_model() is None


# ---------------------------------------------------------------------------
# max_content_length truncation tests (Phase 2 alignment)
# ---------------------------------------------------------------------------


class TestMaxContentLengthTruncation:
    """Tests for SourceSummarizer max_content_length input truncation.

    Verifies that content exceeding max_content_length is truncated before
    being sent to the LLM, matching open_deep_research's pattern of capping
    raw_content at max_content_length before summarization.
    """

    @pytest.mark.asyncio
    async def test_content_exceeding_limit_is_truncated(self):
        """Content longer than max_content_length is truncated before the LLM call."""
        small_limit = 100
        summarizer = SourceSummarizer(
            provider_id="test-provider",
            model="test-model",
            max_content_length=small_limit,
        )

        long_content = "A" * 200  # Twice the limit

        captured_prompts: list[str] = []
        mock_result = MagicMock()
        mock_result.status = MagicMock()
        mock_result.status.name = "SUCCESS"
        mock_result.content = "## Executive Summary\nSummary.\n\n## Key Excerpts\n- \"Quote\""
        mock_result.tokens = MagicMock(input_tokens=50, output_tokens=25)
        mock_result.stderr = None

        mock_provider = MagicMock()

        def capture_generate(request):
            captured_prompts.append(request.prompt)
            return mock_result

        mock_provider.generate = capture_generate

        with patch(
            "foundry_mcp.core.providers.registry.resolve_provider",
            return_value=mock_provider,
        ), patch(
            "foundry_mcp.core.providers.ProviderStatus"
        ) as mock_status:
            mock_status.SUCCESS = mock_result.status
            await summarizer.summarize_source(long_content)

        assert len(captured_prompts) == 1
        # The full 200-char string should NOT appear in the prompt
        assert long_content not in captured_prompts[0]
        # The truncated 100-char string SHOULD appear
        assert "A" * small_limit in captured_prompts[0]

    @pytest.mark.asyncio
    async def test_content_under_limit_is_not_truncated(self):
        """Content shorter than max_content_length passes through unchanged."""
        summarizer = SourceSummarizer(
            provider_id="test-provider",
            model="test-model",
            max_content_length=50_000,
        )

        short_content = "B" * 3000  # Well under default limit

        captured_prompts: list[str] = []
        mock_result = MagicMock()
        mock_result.status = MagicMock()
        mock_result.status.name = "SUCCESS"
        mock_result.content = "## Executive Summary\nSummary."
        mock_result.tokens = MagicMock(input_tokens=50, output_tokens=25)
        mock_result.stderr = None

        mock_provider = MagicMock()

        def capture_generate(request):
            captured_prompts.append(request.prompt)
            return mock_result

        mock_provider.generate = capture_generate

        with patch(
            "foundry_mcp.core.providers.registry.resolve_provider",
            return_value=mock_provider,
        ), patch(
            "foundry_mcp.core.providers.ProviderStatus"
        ) as mock_status:
            mock_status.SUCCESS = mock_result.status
            await summarizer.summarize_source(short_content)

        assert len(captured_prompts) == 1
        # Full content should appear in the prompt
        assert short_content in captured_prompts[0]

    @pytest.mark.asyncio
    async def test_custom_limit_is_respected(self):
        """Different max_content_length values produce different truncation points."""
        for limit in (50, 500, 5000):
            summarizer = SourceSummarizer(
                provider_id="test-provider",
                model="test-model",
                max_content_length=limit,
            )

            content = "C" * (limit + 100)  # Always exceeds the limit

            captured_prompts: list[str] = []
            mock_result = MagicMock()
            mock_result.status = MagicMock()
            mock_result.status.name = "SUCCESS"
            mock_result.content = "## Executive Summary\nSummary."
            mock_result.tokens = MagicMock(input_tokens=50, output_tokens=25)
            mock_result.stderr = None

            mock_provider = MagicMock()

            def capture_generate(request, _prompts=captured_prompts):
                _prompts.append(request.prompt)
                return mock_result

            mock_provider.generate = capture_generate

            with patch(
                "foundry_mcp.core.providers.registry.resolve_provider",
                return_value=mock_provider,
            ), patch(
                "foundry_mcp.core.providers.ProviderStatus"
            ) as mock_status:
                mock_status.SUCCESS = mock_result.status
                await summarizer.summarize_source(content)

            prompt = captured_prompts[0]
            # Should contain exactly `limit` C's, not the full content
            assert "C" * limit in prompt
            assert "C" * (limit + 100) not in prompt

    def test_default_max_content_length_config_field(self):
        """deep_research_max_content_length defaults to 50,000."""
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig()
        assert config.deep_research_max_content_length == 50_000

    def test_explicit_max_content_length_config_field(self):
        """deep_research_max_content_length can be set explicitly."""
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig(deep_research_max_content_length=25_000)
        assert config.deep_research_max_content_length == 25_000

    def test_max_content_length_from_toml(self):
        """deep_research_max_content_length is parsed from TOML config."""
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig.from_toml_dict(
            {"deep_research_max_content_length": 75000}
        )
        assert config.deep_research_max_content_length == 75_000

    def test_max_content_length_toml_default(self):
        """deep_research_max_content_length uses default when absent from TOML."""
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig.from_toml_dict({})
        assert config.deep_research_max_content_length == 50_000


# ---------------------------------------------------------------------------
# Tavily integration tests
# ---------------------------------------------------------------------------


class TestTavilySourceSummarization:
    """Tests for summarization wired into TavilySearchProvider."""

    @pytest.mark.asyncio
    async def test_tavily_applies_summarization_when_configured(self):
        """TavilySearchProvider applies summarizer to results."""
        from foundry_mcp.core.research.providers.tavily import TavilySearchProvider

        provider = TavilySearchProvider(api_key="tvly-test-key")

        # Set up a mock summarizer
        mock_summarizer = MagicMock(spec=SourceSummarizer)
        mock_summarizer.summarize_sources = AsyncMock(
            return_value={
                "src-001": SourceSummarizationResult(
                    executive_summary="AI summary",
                    key_excerpts=["Key quote"],
                    input_tokens=100,
                    output_tokens=50,
                ),
            }
        )
        provider._source_summarizer = mock_summarizer

        # Create a source to be summarized
        source = ResearchSource(
            id="src-001",
            title="Test",
            content="Original long content",
        )

        # Call the internal method
        result = await provider._apply_source_summarization([source])

        assert result[0].raw_content == "Original long content"
        assert result[0].content is not None
        assert "AI summary" in result[0].content
        assert result[0].metadata["excerpts"] == ["Key quote"]
        assert result[0].metadata["summarized"] is True

    @pytest.mark.asyncio
    async def test_tavily_skips_summarization_when_not_configured(self):
        """TavilySearchProvider skips summarization when no summarizer set."""
        from foundry_mcp.core.research.providers.tavily import TavilySearchProvider

        provider = TavilySearchProvider(api_key="tvly-test-key")
        assert provider._source_summarizer is None

        # Mock the search execution to return sources directly
        mock_response = {
            "results": [
                {
                    "url": "https://example.com",
                    "title": "Test Result",
                    "content": "Snippet text",
                    "score": 0.9,
                }
            ]
        }

        with patch.object(provider, "_execute_with_retry", new_callable=AsyncMock, return_value=mock_response):
            sources = await provider.search("test query", max_results=1)

        assert len(sources) == 1
        assert sources[0].raw_content is None  # No summarization applied
