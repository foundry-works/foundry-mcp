"""Tests for token management utilities.

Tests cover:
1. Model limits resolution order (get_model_limits)
2. Budget allocation with safety margin (TokenBudget)
3. Token estimation fallback chain (estimate_tokens)
4. Preflight validation scenarios (preflight_count)
"""

import warnings
from unittest.mock import patch

import pytest

from foundry_mcp.core.research.token_management import (
    _PROVIDER_TOKENIZERS,
    _TIKTOKEN_AVAILABLE,
    DEFAULT_MODEL_LIMITS,
    BudgetingMode,
    ModelContextLimits,
    PreflightResult,
    TokenBudget,
    TokenCountEstimateWarning,
    _get_cached_encoding,
    clear_token_cache,
    estimate_tokens,
    get_cache_stats,
    get_effective_context,
    get_model_limits,
    get_provider_model_from_spec,
    preflight_count,
    preflight_count_multiple,
    register_provider_tokenizer,
)

# =============================================================================
# Test: Model Limits Resolution (get_model_limits)
# =============================================================================


class TestGetModelLimitsResolution:
    """Tests for get_model_limits resolution order."""

    def test_exact_model_match(self):
        """Test resolution finds exact model match first."""
        limits = get_model_limits("claude", "opus")
        assert limits.context_window == 200_000
        assert limits.max_output_tokens == 32_000
        assert limits.budgeting_mode == BudgetingMode.INPUT_ONLY

    def test_provider_default_fallback(self):
        """Test fallback to provider's _default when model not found."""
        limits = get_model_limits("claude", "unknown-model")
        # Should get claude's _default
        assert limits.context_window == 200_000
        assert limits.max_output_tokens == 16_000

    def test_global_fallback_unknown_provider(self):
        """Test fallback to global default for unknown provider."""
        limits = get_model_limits("unknown-provider", "some-model")
        # Should get global fallback
        assert limits.context_window == 128_000
        assert limits.max_output_tokens == 8_000

    def test_provider_without_model(self):
        """Test resolution with provider only (no model specified)."""
        limits = get_model_limits("gemini")
        # Should get gemini's _default
        assert limits.context_window == 1_000_000
        assert limits.max_output_tokens == 8_192

    def test_case_insensitive_matching(self):
        """Test provider and model matching is case-insensitive."""
        limits1 = get_model_limits("CLAUDE", "OPUS")
        limits2 = get_model_limits("claude", "opus")
        assert limits1 == limits2

    def test_config_overrides_take_precedence(self):
        """Test config overrides override resolved limits."""
        limits = get_model_limits(
            "claude",
            "opus",
            config_overrides={
                "context_window": 50_000,
                "max_output_tokens": 4_000,
            },
        )
        assert limits.context_window == 50_000
        assert limits.max_output_tokens == 4_000

    def test_config_overrides_budgeting_mode_as_string(self):
        """Test budgeting_mode can be passed as string in config."""
        limits = get_model_limits(
            "claude",
            "opus",
            config_overrides={"budgeting_mode": "combined"},
        )
        assert limits.budgeting_mode == BudgetingMode.COMBINED

    def test_all_providers_have_defaults(self):
        """Test all registered providers have _default entries."""
        for provider in DEFAULT_MODEL_LIMITS:
            limits = get_model_limits(provider)
            assert limits is not None
            assert limits.context_window > 0
            assert limits.max_output_tokens > 0


class TestModelContextLimitsValidation:
    """Tests for ModelContextLimits dataclass validation."""

    def test_valid_limits(self):
        """Test valid limits creation."""
        limits = ModelContextLimits(
            context_window=100_000,
            max_output_tokens=8_000,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        )
        assert limits.context_window == 100_000

    def test_invalid_context_window(self):
        """Test negative context_window raises ValueError."""
        with pytest.raises(ValueError, match="context_window must be positive"):
            ModelContextLimits(context_window=-1, max_output_tokens=8_000)

    def test_invalid_max_output_tokens(self):
        """Test zero max_output_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_output_tokens must be positive"):
            ModelContextLimits(context_window=100_000, max_output_tokens=0)

    def test_combined_mode_output_reserved_validation(self):
        """Test COMBINED mode validates output_reserved."""
        with pytest.raises(ValueError, match="output_reserved.*cannot exceed"):
            ModelContextLimits(
                context_window=100_000,
                max_output_tokens=8_000,
                budgeting_mode=BudgetingMode.COMBINED,
                output_reserved=150_000,  # Exceeds context_window
            )


class TestGetEffectiveContext:
    """Tests for get_effective_context calculations."""

    def test_input_only_mode(self):
        """Test INPUT_ONLY mode returns full context_window."""
        limits = ModelContextLimits(
            context_window=200_000,
            max_output_tokens=32_000,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        )
        effective = get_effective_context(limits)
        assert effective == 200_000

    def test_combined_mode_with_output_reserved(self):
        """Test COMBINED mode subtracts output_reserved."""
        limits = ModelContextLimits(
            context_window=100_000,
            max_output_tokens=8_000,
            budgeting_mode=BudgetingMode.COMBINED,
            output_reserved=10_000,
        )
        effective = get_effective_context(limits)
        assert effective == 90_000

    def test_combined_mode_explicit_output_budget(self):
        """Test COMBINED mode with explicit output_budget."""
        limits = ModelContextLimits(
            context_window=100_000,
            max_output_tokens=8_000,
            budgeting_mode=BudgetingMode.COMBINED,
        )
        effective = get_effective_context(limits, output_budget=20_000)
        assert effective == 80_000


# =============================================================================
# Test: Budget Allocation with Safety Margin (TokenBudget)
# =============================================================================


class TestTokenBudgetAllocation:
    """Tests for TokenBudget allocation with safety margin."""

    def test_effective_budget_with_safety_margin(self):
        """Test effective_budget applies safety margin correctly."""
        budget = TokenBudget(
            total_budget=100_000,
            reserved_output=10_000,
            safety_margin=0.1,
        )
        # (100_000 - 10_000) * (1 - 0.1) = 81_000
        assert budget.effective_budget() == 81_000

    def test_effective_budget_no_safety_margin(self):
        """Test effective_budget with zero safety margin."""
        budget = TokenBudget(
            total_budget=100_000,
            reserved_output=10_000,
            safety_margin=0.0,
        )
        assert budget.effective_budget() == 90_000

    def test_can_fit_within_budget(self):
        """Test can_fit returns True for tokens within budget."""
        budget = TokenBudget(total_budget=10_000, safety_margin=0.0)
        assert budget.can_fit(5_000)
        assert budget.can_fit(10_000)  # Exactly at limit

    def test_can_fit_exceeds_budget(self):
        """Test can_fit returns False when exceeding budget."""
        budget = TokenBudget(total_budget=10_000, safety_margin=0.0)
        assert not budget.can_fit(10_001)

    def test_allocate_success(self):
        """Test allocate returns True and updates used_tokens."""
        budget = TokenBudget(total_budget=10_000, safety_margin=0.0)
        result = budget.allocate(5_000)
        assert result is True
        assert budget.used_tokens == 5_000
        assert budget.remaining() == 5_000

    def test_allocate_failure_insufficient_budget(self):
        """Test allocate returns False without modifying state."""
        budget = TokenBudget(total_budget=10_000, safety_margin=0.0)
        budget.allocate(5_000)  # Use half
        result = budget.allocate(6_000)  # Try to exceed
        assert result is False
        assert budget.used_tokens == 5_000  # Unchanged

    def test_remaining_after_allocations(self):
        """Test remaining() tracks correctly after allocations."""
        budget = TokenBudget(total_budget=10_000, safety_margin=0.0)
        assert budget.remaining() == 10_000
        budget.allocate(3_000)
        assert budget.remaining() == 7_000
        budget.allocate(7_000)
        assert budget.remaining() == 0

    def test_usage_fraction(self):
        """Test usage_fraction calculates correctly."""
        budget = TokenBudget(total_budget=10_000, safety_margin=0.0)
        assert budget.usage_fraction() == 0.0
        budget.allocate(5_000)
        assert budget.usage_fraction() == 0.5
        budget.allocate(5_000)
        assert budget.usage_fraction() == 1.0


class TestTokenBudgetValidation:
    """Tests for TokenBudget validation."""

    def test_negative_total_budget(self):
        """Test negative total_budget raises ValueError."""
        with pytest.raises(ValueError, match="total_budget must be positive"):
            TokenBudget(total_budget=-1)

    def test_reserved_output_exceeds_total(self):
        """Test reserved_output >= total_budget raises ValueError."""
        with pytest.raises(ValueError, match="reserved_output.*must be less than"):
            TokenBudget(total_budget=100, reserved_output=100)

    def test_invalid_safety_margin(self):
        """Test safety_margin >= 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="safety_margin must be in"):
            TokenBudget(total_budget=100, safety_margin=1.0)

    def test_can_fit_negative_tokens(self):
        """Test can_fit raises ValueError for negative tokens."""
        budget = TokenBudget(total_budget=100)
        with pytest.raises(ValueError, match="tokens must be non-negative"):
            budget.can_fit(-1)

    def test_allocate_negative_tokens(self):
        """Test allocate raises ValueError for negative tokens."""
        budget = TokenBudget(total_budget=100)
        with pytest.raises(ValueError, match="tokens must be non-negative"):
            budget.allocate(-1)


# =============================================================================
# Test: Token Estimation Fallback Chain (estimate_tokens)
# =============================================================================


class TestEstimateTokensFallbackChain:
    """Tests for estimate_tokens fallback chain."""

    def setup_method(self):
        """Clear cache and provider tokenizers before each test."""
        clear_token_cache()
        _PROVIDER_TOKENIZERS.clear()

    def test_heuristic_fallback_without_tiktoken(self):
        """Test heuristic is used when tiktoken unavailable."""
        # 13 chars -> 13 // 4 = 3 tokens (heuristic)
        # Patch tiktoken away so the heuristic path is exercised.
        with patch("foundry_mcp.core.research.token_management.estimation._TIKTOKEN_AVAILABLE", False):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                tokens = estimate_tokens("Hello, world!")
                assert tokens == 3
                assert len(w) == 1
                assert issubclass(w[0].category, TokenCountEstimateWarning)

    def test_provider_native_tokenizer(self):
        """Test provider-native tokenizer takes precedence."""

        def word_counter(content: str) -> int:
            return len(content.split())

        register_provider_tokenizer("test-provider", word_counter)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tokens = estimate_tokens("one two three four", provider="test-provider")
            assert tokens == 4  # 4 words
            assert len(w) == 0  # No heuristic warning

    def test_provider_tokenizer_failure_falls_back(self):
        """Test failure in provider tokenizer falls back gracefully."""

        def failing_tokenizer(_content: str) -> int:
            raise RuntimeError("Tokenizer failed")

        register_provider_tokenizer("failing", failing_tokenizer)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tokens = estimate_tokens("test content", provider="failing")
            assert tokens >= 1  # Got some estimate
            # If tiktoken is available it picks up after provider failure
            # (no heuristic warning). If not, heuristic emits a warning.
            assert len(w) <= 1

    def test_empty_content_returns_zero(self):
        """Test empty string returns 0 tokens."""
        tokens = estimate_tokens("")
        assert tokens == 0

    def test_warn_on_heuristic_can_be_disabled(self):
        """Test warn_on_heuristic=False suppresses warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            estimate_tokens("test", warn_on_heuristic=False)
            assert len(w) == 0


class TestEstimateTokensCache:
    """Tests for estimate_tokens caching behavior."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_token_cache()
        _PROVIDER_TOKENIZERS.clear()

    def test_cache_stores_by_content_hash_and_provider(self):
        """Test results are cached by content hash + provider."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tokens1 = estimate_tokens("test content", provider="claude")
            tokens2 = estimate_tokens("test content", provider="claude")
            assert tokens1 == tokens2

        stats = get_cache_stats()
        assert stats["size"] == 1

    def test_different_provider_different_cache_entry(self):
        """Test different providers create different cache entries."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimate_tokens("test", provider="claude")
            estimate_tokens("test", provider="gemini")

        stats = get_cache_stats()
        assert stats["size"] == 2

    def test_cached_result_no_warning(self):
        """Test cached results don't emit additional warnings."""
        # Force heuristic path so warnings are predictable.
        with patch("foundry_mcp.core.research.token_management.estimation._TIKTOKEN_AVAILABLE", False):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                estimate_tokens("test")  # First call - warning
                estimate_tokens("test")  # Cached - no warning
                assert len(w) == 1

    def test_use_cache_false_bypasses_cache(self):
        """Test use_cache=False bypasses cache."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimate_tokens("test", use_cache=False)

        stats = get_cache_stats()
        assert stats["size"] == 0

    def test_clear_token_cache(self):
        """Test clear_token_cache empties the cache."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimate_tokens("test1")
            estimate_tokens("test2")

        assert get_cache_stats()["size"] == 2
        cleared = clear_token_cache()
        assert cleared == 2
        assert get_cache_stats()["size"] == 0


# =============================================================================
# Test: Encoding Cache (_get_cached_encoding)
# =============================================================================


class TestEncodingCache:
    """Tests for _get_cached_encoding lru_cache behavior."""

    def setup_method(self):
        """Clear encoding cache before each test."""
        _get_cached_encoding.cache_clear()

    @pytest.mark.skipif(not _TIKTOKEN_AVAILABLE, reason="tiktoken not installed")
    def test_cache_reuses_encoding_objects(self):
        """Test cache returns the same encoding object for repeated calls."""
        # First call
        encoding1 = _get_cached_encoding("")
        # Second call - should return cached object
        encoding2 = _get_cached_encoding("")

        # Same object identity (not just equality)
        assert encoding1 is encoding2

        # Verify cache was hit
        cache_info = _get_cached_encoding.cache_info()
        assert cache_info.hits >= 1
        assert cache_info.misses == 1

    @pytest.mark.skipif(not _TIKTOKEN_AVAILABLE, reason="tiktoken not installed")
    def test_cache_reuses_for_same_model(self):
        """Test cache returns same encoding for identical model names."""
        encoding1 = _get_cached_encoding("gpt-4")
        encoding2 = _get_cached_encoding("gpt-4")

        assert encoding1 is encoding2

        cache_info = _get_cached_encoding.cache_info()
        assert cache_info.hits >= 1

    @pytest.mark.skipif(not _TIKTOKEN_AVAILABLE, reason="tiktoken not installed")
    def test_different_models_different_cache_entries(self):
        """Test different model names create different cache entries."""
        # Empty string gets default encoding
        encoding_default = _get_cached_encoding("")
        # Unknown model falls back to cl100k_base (same encoding but different cache key)
        encoding_unknown = _get_cached_encoding("unknown-model-xyz")

        cache_info = _get_cached_encoding.cache_info()
        # Both should be misses (different keys)
        assert cache_info.misses == 2

    @pytest.mark.skipif(not _TIKTOKEN_AVAILABLE, reason="tiktoken not installed")
    def test_token_counts_identical_with_cache(self):
        """Test token counts are identical whether from cache or fresh."""
        test_content = "Hello, this is a test of token counting!"

        # Clear cache and get fresh encoding
        _get_cached_encoding.cache_clear()
        encoding_fresh = _get_cached_encoding("")
        tokens_fresh = len(encoding_fresh.encode(test_content))

        # Get cached encoding
        encoding_cached = _get_cached_encoding("")
        tokens_cached = len(encoding_cached.encode(test_content))

        assert tokens_fresh == tokens_cached

    @pytest.mark.skipif(not _TIKTOKEN_AVAILABLE, reason="tiktoken not installed")
    def test_unknown_model_falls_back_to_cl100k_base(self):
        """Test unknown model names fall back to cl100k_base encoding."""
        # Get encoding for unknown model
        encoding = _get_cached_encoding("definitely-not-a-real-model")

        # Verify it can encode content (cl100k_base fallback works)
        tokens = encoding.encode("test content")
        assert len(tokens) > 0

    @pytest.mark.skipif(not _TIKTOKEN_AVAILABLE, reason="tiktoken not installed")
    def test_cache_maxsize_bound(self):
        """Test cache respects maxsize=32 bound."""
        # Fill cache with 32 different keys
        for i in range(32):
            _get_cached_encoding(f"model-{i}")

        cache_info = _get_cached_encoding.cache_info()
        assert cache_info.currsize <= 32

        # Add one more - should evict oldest
        _get_cached_encoding("model-overflow")
        cache_info = _get_cached_encoding.cache_info()
        assert cache_info.currsize <= 32

    def test_graceful_error_when_tiktoken_unavailable(self):
        """Test RuntimeError raised when tiktoken not available."""
        if _TIKTOKEN_AVAILABLE:
            pytest.skip("Test only runs when tiktoken is NOT installed")

        with pytest.raises(RuntimeError, match="tiktoken is not available"):
            _get_cached_encoding("")


# =============================================================================
# Test: Preflight Validation Scenarios (preflight_count)
# =============================================================================


class TestPreflightCount:
    """Tests for preflight_count validation."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_token_cache()

    def test_valid_payload_returns_valid_result(self):
        """Test payload within budget returns valid=True."""
        budget = TokenBudget(total_budget=10_000, safety_margin=0.0)
        content = "x" * 1000

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = preflight_count(content, budget)

        assert result.valid is True
        assert result.estimated_tokens > 0
        assert result.overflow_tokens == 0
        assert result.remaining_tokens == 10_000 - result.estimated_tokens

    def test_oversized_payload_returns_invalid_result(self):
        """Test payload exceeding budget returns valid=False."""
        budget = TokenBudget(total_budget=100, safety_margin=0.0)
        content = "x" * 8_000  # Well over 100 tokens regardless of estimator

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = preflight_count(content, budget)

        assert result.valid is False
        assert result.overflow_tokens > 0
        assert result.remaining_tokens == 0

    def test_effective_budget_in_result(self):
        """Test effective_budget is correctly set in result."""
        budget = TokenBudget(
            total_budget=10_000,
            reserved_output=2_000,
            safety_margin=0.1,
        )
        # Effective: (10_000 - 2_000) * 0.9 = 7_200

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = preflight_count("test", budget)

        assert result.effective_budget == 7_200

    def test_is_final_fit_flag(self):
        """Test is_final_fit flag is set correctly."""
        budget = TokenBudget(total_budget=10_000)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result1 = preflight_count("test", budget, is_final_fit=False)
            result2 = preflight_count("test", budget, is_final_fit=True)

        assert result1.is_final_fit is False
        assert result2.is_final_fit is True

    def test_usage_fraction_property(self):
        """Test usage_fraction property calculates correctly."""
        budget = TokenBudget(total_budget=10_000, safety_margin=0.0)
        content = "x" * 4000

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = preflight_count(content, budget)

        expected_fraction = result.estimated_tokens / 10_000
        assert result.usage_fraction == expected_fraction

    def test_to_dict_serialization(self):
        """Test to_dict includes all fields."""
        budget = TokenBudget(total_budget=10_000)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = preflight_count("test", budget)

        d = result.to_dict()
        assert "valid" in d
        assert "estimated_tokens" in d
        assert "effective_budget" in d
        assert "remaining_tokens" in d
        assert "overflow_tokens" in d
        assert "is_final_fit" in d
        assert "usage_fraction" in d


class TestPreflightCountMultiple:
    """Tests for preflight_count_multiple batch validation."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_token_cache()

    def test_multiple_valid_payloads(self):
        """Test multiple payloads that fit."""
        budget = TokenBudget(total_budget=10_000, safety_margin=0.0)
        items = ["short", "medium text", "longer content here"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            valid, counts, total = preflight_count_multiple(items, budget)

        assert valid is True
        assert len(counts) == 3
        assert total == sum(counts)

    def test_multiple_exceeds_budget(self):
        """Test multiple payloads that exceed budget."""
        budget = TokenBudget(total_budget=10, safety_margin=0.0)
        items = ["x" * 200, "x" * 200, "x" * 200]  # Well over 10 tokens total

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            valid, _counts, total = preflight_count_multiple(items, budget)

        assert valid is False
        assert total > 10

    def test_empty_list(self):
        """Test empty list returns valid with zero tokens."""
        budget = TokenBudget(total_budget=10_000)
        valid, counts, total = preflight_count_multiple([], budget)
        assert valid is True
        assert counts == []
        assert total == 0


class TestPreflightResultValidation:
    """Tests for PreflightResult validation."""

    def test_negative_estimated_tokens(self):
        """Test negative estimated_tokens raises ValueError."""
        with pytest.raises(ValueError, match="estimated_tokens must be non-negative"):
            PreflightResult(
                valid=True,
                estimated_tokens=-1,
                effective_budget=1000,
                remaining_tokens=1000,
                overflow_tokens=0,
            )

    def test_negative_effective_budget(self):
        """Test negative effective_budget raises ValueError."""
        with pytest.raises(ValueError, match="effective_budget must be non-negative"):
            PreflightResult(
                valid=True,
                estimated_tokens=100,
                effective_budget=-1,
                remaining_tokens=1000,
                overflow_tokens=0,
            )


# =============================================================================
# Test: Provider Spec Parsing
# =============================================================================


class TestGetProviderModelFromSpec:
    """Tests for get_provider_model_from_spec parsing."""

    def test_provider_only(self):
        """Test parsing provider-only spec."""
        provider, model = get_provider_model_from_spec("claude")
        assert provider == "claude"
        assert model is None

    def test_provider_and_model(self):
        """Test parsing provider:model spec."""
        provider, model = get_provider_model_from_spec("gemini:flash")
        assert provider == "gemini"
        assert model == "flash"

    def test_cli_prefix_stripped(self):
        """Test [cli] prefix is stripped."""
        provider, model = get_provider_model_from_spec("[cli]claude:opus")
        assert provider == "claude"
        assert model == "opus"

    def test_whitespace_trimmed(self):
        """Test whitespace is trimmed."""
        provider, model = get_provider_model_from_spec("  claude : opus  ")
        assert provider == "claude"
        assert model == "opus"
