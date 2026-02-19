"""Tests for content summarization utilities.

Tests cover:
1. SummaryCache - cache keys, hit/miss, eviction, enabled toggle
2. SummarizationLevel - enum properties, level progression
3. SummarizationResult - validation, key point extraction, serialization
4. SummarizationConfig - cache_enabled and provider chain
5. ContentSummarizer - chunking, map-reduce, level stepping, truncation, cache
"""

import pytest

from foundry_mcp.core.research.summarization import (
    CHARS_PER_TOKEN,
    DEFAULT_CHUNK_SIZE,
    ContentSummarizer,
    ProviderExhaustedError,
    SummarizationConfig,
    SummarizationError,
    SummarizationLevel,
    SummarizationResult,
    SummarizationValidationError,
    SummaryCache,
)

# =============================================================================
# Test: SummaryCache
# =============================================================================


class TestSummaryCacheKeyComposition:
    """Tests for cache key composition with all factors."""

    def test_same_inputs_same_key(self):
        """Test identical inputs produce cache hit."""
        cache = SummaryCache(enabled=True)
        result = SummarizationResult(
            content="Summary",
            level=SummarizationLevel.KEY_POINTS,
        )
        cache.set("content", "context", SummarizationLevel.KEY_POINTS, "claude", result)
        cached = cache.get("content", "context", SummarizationLevel.KEY_POINTS, "claude")
        assert cached is not None
        assert cached.content == result.content

    def test_different_content_different_key(self):
        """Test different content produces cache miss."""
        cache = SummaryCache(enabled=True)
        result = SummarizationResult(
            content="Summary",
            level=SummarizationLevel.KEY_POINTS,
        )
        cache.set("content1", "context", SummarizationLevel.KEY_POINTS, "claude", result)
        cached = cache.get("content2", "context", SummarizationLevel.KEY_POINTS, "claude")
        assert cached is None

    def test_different_context_different_key(self):
        """Test different context produces cache miss."""
        cache = SummaryCache(enabled=True)
        result = SummarizationResult(
            content="Summary",
            level=SummarizationLevel.KEY_POINTS,
        )
        cache.set("content", "context1", SummarizationLevel.KEY_POINTS, "claude", result)
        cached = cache.get("content", "context2", SummarizationLevel.KEY_POINTS, "claude")
        assert cached is None

    def test_different_level_different_key(self):
        """Test different level produces cache miss."""
        cache = SummaryCache(enabled=True)
        result = SummarizationResult(
            content="Summary",
            level=SummarizationLevel.KEY_POINTS,
        )
        cache.set("content", "context", SummarizationLevel.KEY_POINTS, "claude", result)
        cached = cache.get("content", "context", SummarizationLevel.HEADLINE, "claude")
        assert cached is None

    def test_different_provider_different_key(self):
        """Test different provider produces cache miss."""
        cache = SummaryCache(enabled=True)
        result = SummarizationResult(
            content="Summary",
            level=SummarizationLevel.KEY_POINTS,
        )
        cache.set("content", "context", SummarizationLevel.KEY_POINTS, "claude", result)
        cached = cache.get("content", "context", SummarizationLevel.KEY_POINTS, "gemini")
        assert cached is None

    def test_none_context_handled(self):
        """Test None context is handled correctly."""
        cache = SummaryCache(enabled=True)
        result = SummarizationResult(
            content="Summary",
            level=SummarizationLevel.KEY_POINTS,
        )
        cache.set("content", None, SummarizationLevel.KEY_POINTS, "claude", result)
        cached = cache.get("content", None, SummarizationLevel.KEY_POINTS, "claude")
        assert cached is not None

    def test_none_provider_handled(self):
        """Test None provider is handled correctly."""
        cache = SummaryCache(enabled=True)
        result = SummarizationResult(
            content="Summary",
            level=SummarizationLevel.KEY_POINTS,
        )
        cache.set("content", "context", SummarizationLevel.KEY_POINTS, None, result)
        cached = cache.get("content", "context", SummarizationLevel.KEY_POINTS, None)
        assert cached is not None


class TestSummaryCacheEnabledToggle:
    """Tests for cache enabled/disabled behavior."""

    def test_disabled_cache_returns_none_on_get(self):
        """Test disabled cache always returns None."""
        cache = SummaryCache(enabled=False)
        result = SummarizationResult(
            content="Summary",
            level=SummarizationLevel.KEY_POINTS,
        )
        # Try to set while disabled - should be no-op
        cache.set("content", "context", SummarizationLevel.KEY_POINTS, "claude", result)
        cached = cache.get("content", "context", SummarizationLevel.KEY_POINTS, "claude")
        assert cached is None

    def test_enabled_toggle_affects_behavior(self):
        """Test toggling enabled affects get/set behavior."""
        cache = SummaryCache(enabled=True)
        result = SummarizationResult(
            content="Summary",
            level=SummarizationLevel.KEY_POINTS,
        )
        cache.set("content", "context", SummarizationLevel.KEY_POINTS, "claude", result)
        assert cache.get("content", "context", SummarizationLevel.KEY_POINTS, "claude") is not None

        # Disable - should return None
        cache.enabled = False
        assert cache.get("content", "context", SummarizationLevel.KEY_POINTS, "claude") is None

        # Re-enable - entry should still be there
        cache.enabled = True
        assert cache.get("content", "context", SummarizationLevel.KEY_POINTS, "claude") is not None


class TestSummaryCacheEviction:
    """Tests for cache eviction behavior."""

    def test_eviction_at_max_size(self):
        """Test eviction occurs when max size is reached."""
        cache = SummaryCache(enabled=True, max_size=10)
        result = SummarizationResult(
            content="Summary",
            level=SummarizationLevel.KEY_POINTS,
        )
        # Fill cache to max
        for i in range(10):
            cache.set(f"content{i}", None, SummarizationLevel.KEY_POINTS, "claude", result)
        assert cache.get_stats()["size"] == 10

        # Add one more - should trigger eviction
        cache.set("content_new", None, SummarizationLevel.KEY_POINTS, "claude", result)
        # Should have evicted half (5) and added 1, so 6 entries
        assert cache.get_stats()["size"] == 6


class TestSummaryCacheStatsAndClear:
    """Tests for cache statistics and clear operations."""

    def test_get_stats_returns_correct_values(self):
        """Test get_stats returns accurate information."""
        cache = SummaryCache(enabled=True, max_size=100)
        stats = cache.get_stats()
        assert stats["size"] == 0
        assert stats["max_size"] == 100
        assert stats["enabled"] is True

    def test_clear_removes_all_entries(self):
        """Test clear removes all entries and returns count."""
        cache = SummaryCache(enabled=True)
        result = SummarizationResult(
            content="Summary",
            level=SummarizationLevel.KEY_POINTS,
        )
        for i in range(5):
            cache.set(f"content{i}", None, SummarizationLevel.KEY_POINTS, "claude", result)
        assert cache.get_stats()["size"] == 5

        cleared = cache.clear()
        assert cleared == 5
        assert cache.get_stats()["size"] == 0


# =============================================================================
# Test: SummarizationLevel
# =============================================================================


class TestSummarizationLevelProperties:
    """Tests for SummarizationLevel enum properties."""

    def test_level_values(self):
        """Test level string values."""
        assert SummarizationLevel.RAW.value == "raw"
        assert SummarizationLevel.CONDENSED.value == "condensed"
        assert SummarizationLevel.KEY_POINTS.value == "key_points"
        assert SummarizationLevel.HEADLINE.value == "headline"

    def test_target_compression_ratio(self):
        """Test compression ratios are correct."""
        assert SummarizationLevel.RAW.target_compression_ratio == 1.0
        assert SummarizationLevel.CONDENSED.target_compression_ratio == 0.6
        assert SummarizationLevel.KEY_POINTS.target_compression_ratio == 0.3
        assert SummarizationLevel.HEADLINE.target_compression_ratio == 0.1

    def test_max_output_tokens(self):
        """Test max output tokens are reasonable."""
        assert SummarizationLevel.RAW.max_output_tokens == 0
        assert SummarizationLevel.CONDENSED.max_output_tokens == 2000
        assert SummarizationLevel.KEY_POINTS.max_output_tokens == 500
        assert SummarizationLevel.HEADLINE.max_output_tokens == 100


class TestSummarizationLevelProgression:
    """Tests for level stepping progression."""

    def test_next_tighter_level_progression(self):
        """Test progression through tighter levels."""
        assert SummarizationLevel.RAW.next_tighter_level() == SummarizationLevel.CONDENSED
        assert SummarizationLevel.CONDENSED.next_tighter_level() == SummarizationLevel.KEY_POINTS
        assert SummarizationLevel.KEY_POINTS.next_tighter_level() == SummarizationLevel.HEADLINE
        assert SummarizationLevel.HEADLINE.next_tighter_level() is None


# =============================================================================
# Test: SummarizationResult
# =============================================================================


class TestSummarizationResultValidation:
    """Tests for SummarizationResult validation."""

    def test_validate_requires_content(self):
        """Test validation fails without content."""
        result = SummarizationResult(content="", level=SummarizationLevel.CONDENSED)
        with pytest.raises(SummarizationValidationError) as exc_info:
            result.validate()
        assert "content" in exc_info.value.missing_fields

    def test_validate_key_points_requires_key_points(self):
        """Test KEY_POINTS level requires key_points list."""
        result = SummarizationResult(
            content="Some content",
            level=SummarizationLevel.KEY_POINTS,
            key_points=[],  # Empty list should fail
        )
        with pytest.raises(SummarizationValidationError) as exc_info:
            result.validate()
        assert "key_points" in exc_info.value.missing_fields

    def test_validate_key_points_success(self):
        """Test KEY_POINTS level validates with key_points."""
        result = SummarizationResult(
            content="Some content",
            level=SummarizationLevel.KEY_POINTS,
            key_points=["point 1", "point 2"],
        )
        assert result.validate() is True

    def test_validate_headline_only_needs_content(self):
        """Test HEADLINE level only needs content."""
        result = SummarizationResult(
            content="A single headline",
            level=SummarizationLevel.HEADLINE,
        )
        assert result.validate() is True

    def test_is_valid_returns_false_instead_of_raising(self):
        """Test is_valid returns False without raising."""
        result = SummarizationResult(content="", level=SummarizationLevel.CONDENSED)
        assert result.is_valid() is False


class TestSummarizationResultKeyPointExtraction:
    """Tests for from_raw_output key point extraction."""

    def test_extract_bullet_points_with_dash(self):
        """Test extraction of dash bullet points."""
        raw = "- Point one\n- Point two\n- Point three"
        result = SummarizationResult.from_raw_output(raw, SummarizationLevel.KEY_POINTS)
        assert len(result.key_points) == 3
        assert "Point one" in result.key_points

    def test_extract_bullet_points_with_asterisk(self):
        """Test extraction of asterisk bullet points."""
        raw = "* First\n* Second"
        result = SummarizationResult.from_raw_output(raw, SummarizationLevel.KEY_POINTS)
        assert len(result.key_points) == 2

    def test_extract_numbered_list(self):
        """Test extraction of numbered list items."""
        raw = "1. First point\n2. Second point\n3. Third point"
        result = SummarizationResult.from_raw_output(raw, SummarizationLevel.KEY_POINTS)
        assert len(result.key_points) == 3

    def test_non_key_points_level_no_extraction(self):
        """Test non-KEY_POINTS levels don't extract key_points."""
        raw = "- Point one\n- Point two"
        result = SummarizationResult.from_raw_output(raw, SummarizationLevel.CONDENSED)
        assert len(result.key_points) == 0

    def test_source_ids_passed_through(self):
        """Test source_ids are passed through correctly."""
        result = SummarizationResult.from_raw_output(
            "Summary text",
            SummarizationLevel.KEY_POINTS,
            source_ids=["src-1", "src-2"],
        )
        assert result.source_ids == ["src-1", "src-2"]


class TestSummarizationResultSerialization:
    """Tests for SummarizationResult serialization."""

    def test_to_dict_includes_all_fields(self):
        """Test to_dict includes all expected fields."""
        result = SummarizationResult(
            content="Test summary",
            level=SummarizationLevel.KEY_POINTS,
            key_points=["point 1"],
            source_ids=["src-1"],
            original_tokens=100,
            summary_tokens=20,
            provider_id="claude",
            truncated=False,
            warnings=["test warning"],
        )
        d = result.to_dict()
        assert d["content"] == "Test summary"
        assert d["level"] == "key_points"
        assert d["key_points"] == ["point 1"]
        assert d["source_ids"] == ["src-1"]
        assert d["original_tokens"] == 100
        assert d["summary_tokens"] == 20
        assert d["provider_id"] == "claude"
        assert d["truncated"] is False
        assert d["warnings"] == ["test warning"]
        assert d["compression_ratio"] == 0.2

    def test_compression_ratio_calculation(self):
        """Test compression ratio is calculated correctly."""
        result = SummarizationResult(
            content="Short",
            level=SummarizationLevel.KEY_POINTS,
            original_tokens=100,
            summary_tokens=25,
        )
        assert result.compression_ratio == 0.25

    def test_compression_ratio_with_zero_original(self):
        """Test compression ratio with zero original tokens."""
        result = SummarizationResult(
            content="Short",
            level=SummarizationLevel.KEY_POINTS,
            original_tokens=0,
            summary_tokens=25,
        )
        assert result.compression_ratio == 1.0


# =============================================================================
# Test: SummarizationConfig
# =============================================================================


class TestSummarizationConfig:
    """Tests for SummarizationConfig."""

    def test_default_cache_enabled(self):
        """Test cache is enabled by default."""
        config = SummarizationConfig()
        assert config.cache_enabled is True

    def test_cache_can_be_disabled(self):
        """Test cache can be disabled via config."""
        config = SummarizationConfig(cache_enabled=False)
        assert config.cache_enabled is False

    def test_provider_chain_primary_first(self):
        """Test provider chain puts primary first."""
        config = SummarizationConfig(
            summarization_provider="claude",
            summarization_providers=["gemini", "codex"],
        )
        chain = config.get_provider_chain()
        assert chain == ["claude", "gemini", "codex"]

    def test_provider_chain_deduplicates(self):
        """Test provider chain deduplicates."""
        config = SummarizationConfig(
            summarization_provider="claude",
            summarization_providers=["claude", "gemini"],
        )
        chain = config.get_provider_chain()
        assert chain == ["claude", "gemini"]

    def test_provider_chain_empty_when_none(self):
        """Test provider chain is empty when no providers set."""
        config = SummarizationConfig()
        assert config.get_provider_chain() == []


# =============================================================================
# Test: ContentSummarizer - Chunking
# =============================================================================


class TestContentSummarizerChunking:
    """Tests for ContentSummarizer chunking logic."""

    def test_needs_chunking_small_content(self):
        """Test small content doesn't need chunking."""
        summarizer = ContentSummarizer(summarization_provider="claude")
        small_content = "a" * 1000  # Well under chunk size
        assert summarizer._needs_chunking(small_content) is False

    def test_needs_chunking_large_content(self):
        """Test large content needs chunking."""
        summarizer = ContentSummarizer(summarization_provider="claude")
        # Create content larger than chunk size in chars
        large_content = "a" * (DEFAULT_CHUNK_SIZE * CHARS_PER_TOKEN + 1000)
        assert summarizer._needs_chunking(large_content) is True

    def test_chunk_content_returns_list(self):
        """Test chunk_content returns list of chunks."""
        summarizer = ContentSummarizer(
            summarization_provider="claude",
            chunk_size=100,  # Small for testing
        )
        # Content that needs chunking
        content = "a" * 1000  # 250 tokens at 4 chars/token
        chunks = summarizer._chunk_content(content)
        assert isinstance(chunks, list)
        assert len(chunks) > 1

    def test_chunk_content_small_returns_single(self):
        """Test small content returns single chunk."""
        summarizer = ContentSummarizer(summarization_provider="claude")
        small_content = "Small content"
        chunks = summarizer._chunk_content(small_content)
        assert len(chunks) == 1
        assert chunks[0] == small_content


# =============================================================================
# Test: ContentSummarizer - Truncation
# =============================================================================


class TestContentSummarizerTruncation:
    """Tests for ContentSummarizer truncation fallback."""

    def test_truncate_with_warning_adds_marker(self):
        """Test truncation adds truncation marker."""
        summarizer = ContentSummarizer(summarization_provider="claude")
        content = "a" * 1000
        truncated = summarizer._truncate_with_warning(content, 50)  # 50 tokens = 200 chars
        assert "[... truncated]" in truncated
        assert len(truncated) <= 200

    def test_truncate_small_content_unchanged(self):
        """Test small content is not truncated."""
        summarizer = ContentSummarizer(summarization_provider="claude")
        content = "Small content"
        result = summarizer._truncate_with_warning(content, 1000)
        assert result == content


# =============================================================================
# Test: ContentSummarizer - Cache Integration
# =============================================================================


class TestContentSummarizerCacheIntegration:
    """Tests for ContentSummarizer cache integration."""

    def test_cache_enabled_by_default(self):
        """Test cache is enabled by default."""
        summarizer = ContentSummarizer(summarization_provider="claude")
        assert summarizer.cache_enabled is True
        assert summarizer.config.cache_enabled is True

    def test_cache_disabled_via_constructor(self):
        """Test cache can be disabled via constructor."""
        summarizer = ContentSummarizer(
            summarization_provider="claude",
            cache_enabled=False,
        )
        assert summarizer.cache_enabled is False

    def test_cache_disabled_via_property(self):
        """Test cache can be disabled via property."""
        summarizer = ContentSummarizer(summarization_provider="claude")
        summarizer.cache_enabled = False
        assert summarizer.cache_enabled is False
        assert summarizer.config.cache_enabled is False

    def test_from_config_passes_cache_enabled(self):
        """Test from_config passes cache_enabled correctly."""
        config = SummarizationConfig(
            summarization_provider="claude",
            cache_enabled=False,
        )
        summarizer = ContentSummarizer.from_config(config)
        assert summarizer.cache_enabled is False

    def test_get_cache_stats(self):
        """Test get_cache_stats returns valid stats."""
        summarizer = ContentSummarizer(summarization_provider="claude")
        stats = summarizer.get_cache_stats()
        assert "size" in stats
        assert "max_size" in stats
        assert "enabled" in stats

    def test_clear_cache(self):
        """Test clear_cache returns count and clears."""
        summarizer = ContentSummarizer(summarization_provider="claude")
        # Manually add to cache
        result = SummarizationResult(
            content="Test",
            level=SummarizationLevel.KEY_POINTS,
        )
        summarizer._cache.set("content", None, SummarizationLevel.KEY_POINTS, "claude", result)
        assert summarizer.get_cache_stats()["size"] == 1

        cleared = summarizer.clear_cache()
        assert cleared == 1
        assert summarizer.get_cache_stats()["size"] == 0


# =============================================================================
# Test: ContentSummarizer - Provider Chain
# =============================================================================


class TestContentSummarizerProviderChain:
    """Tests for ContentSummarizer provider chain."""

    def test_get_provider_chain(self):
        """Test get_provider_chain returns configured chain."""
        summarizer = ContentSummarizer(
            summarization_provider="claude",
            summarization_providers=["gemini", "codex"],
        )
        chain = summarizer.get_provider_chain()
        assert chain == ["claude", "gemini", "codex"]

    def test_is_available_with_provider(self):
        """Test is_available returns True with provider."""
        summarizer = ContentSummarizer(summarization_provider="claude")
        assert summarizer.is_available() is True

    def test_is_available_without_provider(self):
        """Test is_available returns False without provider."""
        summarizer = ContentSummarizer()
        assert summarizer.is_available() is False


# =============================================================================
# Test: ContentSummarizer - Async Operations with Mock
# =============================================================================


class TestContentSummarizerAsyncWithMock:
    """Tests for ContentSummarizer async operations using mock provider."""

    @pytest.fixture
    def mock_provider_func(self):
        """Create a mock provider function."""

        def provider(content: str, level: SummarizationLevel, provider_id: str) -> str:
            if level == SummarizationLevel.KEY_POINTS:
                return "- Key point 1\n- Key point 2\n- Key point 3"
            elif level == SummarizationLevel.HEADLINE:
                return "Brief headline summary"
            elif level == SummarizationLevel.CONDENSED:
                return content[: len(content) // 2]
            return content

        return provider

    @pytest.mark.asyncio
    async def test_summarize_raw_passthrough(self, mock_provider_func):
        """Test RAW level passes content through unchanged."""
        summarizer = ContentSummarizer(
            summarization_provider="claude",
            provider_func=mock_provider_func,
        )
        result = await summarizer.summarize("Original content", SummarizationLevel.RAW)
        assert result == "Original content"

    @pytest.mark.asyncio
    async def test_summarize_key_points(self, mock_provider_func):
        """Test KEY_POINTS level summarization."""
        summarizer = ContentSummarizer(
            summarization_provider="claude",
            provider_func=mock_provider_func,
        )
        result = await summarizer.summarize("Content to summarize", SummarizationLevel.KEY_POINTS)
        assert "Key point" in result

    @pytest.mark.asyncio
    async def test_summarize_with_result_caches(self, mock_provider_func):
        """Test summarize_with_result uses cache."""
        summarizer = ContentSummarizer(
            summarization_provider="claude",
            provider_func=mock_provider_func,
        )
        # First call - should cache
        result1 = await summarizer.summarize_with_result(
            "Content",
            SummarizationLevel.KEY_POINTS,
        )
        assert summarizer.get_cache_stats()["size"] == 1

        # Second call - should hit cache
        result2 = await summarizer.summarize_with_result(
            "Content",
            SummarizationLevel.KEY_POINTS,
        )
        assert result2.content == result1.content

    @pytest.mark.asyncio
    async def test_summarize_with_result_bypasses_cache_when_disabled(self, mock_provider_func):
        """Test cache bypass when use_cache=False."""
        summarizer = ContentSummarizer(
            summarization_provider="claude",
            provider_func=mock_provider_func,
        )
        await summarizer.summarize_with_result(
            "Content",
            SummarizationLevel.KEY_POINTS,
            use_cache=False,
        )
        assert summarizer.get_cache_stats()["size"] == 0

    @pytest.mark.asyncio
    async def test_summarize_no_providers_raises(self):
        """Test error when no providers configured."""
        summarizer = ContentSummarizer()
        with pytest.raises(SummarizationError):
            await summarizer.summarize("Content", SummarizationLevel.KEY_POINTS)


class TestContentSummarizerProviderFailure:
    """Tests for provider failure and exhaustion."""

    @pytest.fixture
    def failing_provider_func(self):
        """Create a provider that always fails."""

        def provider(content: str, level: SummarizationLevel, provider_id: str) -> str:
            raise Exception(f"Provider {provider_id} failed")

        return provider

    @pytest.mark.asyncio
    async def test_all_providers_fail_raises_exhausted(self, failing_provider_func):
        """Test ProviderExhaustedError when all providers fail."""
        summarizer = ContentSummarizer(
            summarization_provider="claude",
            summarization_providers=["gemini"],
            max_retries=0,  # No retries for faster test
            provider_func=failing_provider_func,
        )
        with pytest.raises(ProviderExhaustedError) as exc_info:
            await summarizer.summarize("Content", SummarizationLevel.KEY_POINTS)
        assert len(exc_info.value.errors) == 2  # Both providers failed


# =============================================================================
# Test: ContentSummarizer - Budget Enforcement
# =============================================================================


class TestContentSummarizerBudgetEnforcement:
    """Tests for budget enforcement and level stepping."""

    @pytest.fixture
    def verbose_provider_func(self):
        """Create a provider that returns verbose output."""

        def provider(content: str, level: SummarizationLevel, provider_id: str) -> str:
            # Return progressively shorter content for tighter levels
            if level == SummarizationLevel.HEADLINE:
                return "Short headline."
            elif level == SummarizationLevel.KEY_POINTS:
                return "- Point 1\n- Point 2\n" + "x" * 500  # ~125 tokens
            elif level == SummarizationLevel.CONDENSED:
                return "x" * 2000  # ~500 tokens
            return content

        return provider

    @pytest.mark.asyncio
    async def test_budget_enforcement_triggers_tighter_level(self, verbose_provider_func):
        """Test budget enforcement steps to tighter levels."""
        summarizer = ContentSummarizer(
            summarization_provider="claude",
            provider_func=verbose_provider_func,
        )
        # Request KEY_POINTS with tiny budget - should step to HEADLINE
        result = await summarizer.summarize(
            "Long content here",
            SummarizationLevel.KEY_POINTS,
            target_budget=20,  # Very small budget
        )
        # Should either get truncated or stepped down
        assert len(result) < 500 or "[... truncated]" in result

    @pytest.mark.asyncio
    async def test_summarize_with_result_includes_metadata(self, verbose_provider_func):
        """Test summarize_with_result includes token metadata."""
        summarizer = ContentSummarizer(
            summarization_provider="claude",
            provider_func=verbose_provider_func,
        )
        result = await summarizer.summarize_with_result(
            "x" * 1000,  # ~250 tokens
            SummarizationLevel.KEY_POINTS,
        )
        assert result.original_tokens > 0
        assert result.summary_tokens > 0
        assert result.level == SummarizationLevel.KEY_POINTS


# =============================================================================
# Test: Error Classes
# =============================================================================


class TestErrorClasses:
    """Tests for error class behavior."""

    def test_summarization_validation_error_fields(self):
        """Test SummarizationValidationError includes level and fields."""
        error = SummarizationValidationError(
            "Validation failed",
            SummarizationLevel.KEY_POINTS,
            ["content", "key_points"],
        )
        assert error.level == SummarizationLevel.KEY_POINTS
        assert error.missing_fields == ["content", "key_points"]
        assert "key_points" in str(error)

    def test_provider_exhausted_error_records_all_errors(self):
        """Test ProviderExhaustedError records all provider errors."""
        errors = [
            ("claude", Exception("Claude failed")),
            ("gemini", Exception("Gemini failed")),
        ]
        error = ProviderExhaustedError(errors)
        assert len(error.errors) == 2
        assert "claude" in str(error)
        assert "gemini" in str(error)
