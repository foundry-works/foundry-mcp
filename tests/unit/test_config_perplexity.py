"""Tests for Perplexity configuration fields in ResearchConfig.

Tests cover:
1. TOML parsing for all Perplexity search fields
2. Validation errors for invalid values
3. Default values preserved when not set
4. Precedence rules (explicit values override defaults)
"""

import pytest

from foundry_mcp.config import ResearchConfig


class TestPerplexityConfigParsing:
    """Tests for Perplexity configuration TOML parsing."""

    def test_parse_perplexity_search_context_size_low(self):
        """Test perplexity_search_context_size='low' is parsed correctly."""
        config = ResearchConfig.from_toml_dict({
            "perplexity_search_context_size": "low"
        })
        assert config.perplexity_search_context_size == "low"

    def test_parse_perplexity_search_context_size_medium(self):
        """Test perplexity_search_context_size='medium' is parsed correctly."""
        config = ResearchConfig.from_toml_dict({
            "perplexity_search_context_size": "medium"
        })
        assert config.perplexity_search_context_size == "medium"

    def test_parse_perplexity_search_context_size_high(self):
        """Test perplexity_search_context_size='high' is parsed correctly."""
        config = ResearchConfig.from_toml_dict({
            "perplexity_search_context_size": "high"
        })
        assert config.perplexity_search_context_size == "high"

    def test_parse_perplexity_max_tokens(self):
        """Test perplexity_max_tokens is parsed correctly."""
        config = ResearchConfig.from_toml_dict({
            "perplexity_max_tokens": 100000
        })
        assert config.perplexity_max_tokens == 100000

    def test_parse_perplexity_max_tokens_per_page(self):
        """Test perplexity_max_tokens_per_page is parsed correctly."""
        config = ResearchConfig.from_toml_dict({
            "perplexity_max_tokens_per_page": 4096
        })
        assert config.perplexity_max_tokens_per_page == 4096

    def test_parse_perplexity_recency_filter_day(self):
        """Test perplexity_recency_filter='day' is parsed correctly."""
        config = ResearchConfig.from_toml_dict({
            "perplexity_recency_filter": "day"
        })
        assert config.perplexity_recency_filter == "day"

    def test_parse_perplexity_recency_filter_week(self):
        """Test perplexity_recency_filter='week' is parsed correctly."""
        config = ResearchConfig.from_toml_dict({
            "perplexity_recency_filter": "week"
        })
        assert config.perplexity_recency_filter == "week"

    def test_parse_perplexity_recency_filter_month(self):
        """Test perplexity_recency_filter='month' is parsed correctly."""
        config = ResearchConfig.from_toml_dict({
            "perplexity_recency_filter": "month"
        })
        assert config.perplexity_recency_filter == "month"

    def test_parse_perplexity_recency_filter_year(self):
        """Test perplexity_recency_filter='year' is parsed correctly."""
        config = ResearchConfig.from_toml_dict({
            "perplexity_recency_filter": "year"
        })
        assert config.perplexity_recency_filter == "year"

    def test_parse_perplexity_country(self):
        """Test perplexity_country is parsed correctly."""
        config = ResearchConfig.from_toml_dict({
            "perplexity_country": "US"
        })
        assert config.perplexity_country == "US"


class TestPerplexityConfigDefaults:
    """Tests for Perplexity configuration default values."""

    def test_default_perplexity_search_context_size(self):
        """Test default perplexity_search_context_size is 'medium'."""
        config = ResearchConfig()
        assert config.perplexity_search_context_size == "medium"

    def test_default_perplexity_max_tokens(self):
        """Test default perplexity_max_tokens is 50000."""
        config = ResearchConfig()
        assert config.perplexity_max_tokens == 50000

    def test_default_perplexity_max_tokens_per_page(self):
        """Test default perplexity_max_tokens_per_page is 2048."""
        config = ResearchConfig()
        assert config.perplexity_max_tokens_per_page == 2048

    def test_default_perplexity_recency_filter_is_none(self):
        """Test default perplexity_recency_filter is None."""
        config = ResearchConfig()
        assert config.perplexity_recency_filter is None

    def test_default_perplexity_country_is_none(self):
        """Test default perplexity_country is None."""
        config = ResearchConfig()
        assert config.perplexity_country is None


class TestPerplexityConfigValidation:
    """Tests for Perplexity configuration validation."""

    def test_validate_perplexity_search_context_size_invalid(self):
        """Test invalid perplexity_search_context_size raises ValueError."""
        with pytest.raises(ValueError, match="Invalid perplexity_search_context_size"):
            ResearchConfig(perplexity_search_context_size="invalid")

    def test_validate_perplexity_max_tokens_zero(self):
        """Test perplexity_max_tokens=0 raises ValueError."""
        with pytest.raises(ValueError, match="Invalid perplexity_max_tokens"):
            ResearchConfig(perplexity_max_tokens=0)

    def test_validate_perplexity_max_tokens_negative(self):
        """Test negative perplexity_max_tokens raises ValueError."""
        with pytest.raises(ValueError, match="Invalid perplexity_max_tokens"):
            ResearchConfig(perplexity_max_tokens=-1)

    def test_validate_perplexity_max_tokens_per_page_zero(self):
        """Test perplexity_max_tokens_per_page=0 raises ValueError."""
        with pytest.raises(ValueError, match="Invalid perplexity_max_tokens_per_page"):
            ResearchConfig(perplexity_max_tokens_per_page=0)

    def test_validate_perplexity_max_tokens_per_page_negative(self):
        """Test negative perplexity_max_tokens_per_page raises ValueError."""
        with pytest.raises(ValueError, match="Invalid perplexity_max_tokens_per_page"):
            ResearchConfig(perplexity_max_tokens_per_page=-1)

    def test_validate_perplexity_recency_filter_invalid(self):
        """Test invalid perplexity_recency_filter raises ValueError."""
        with pytest.raises(ValueError, match="Invalid perplexity_recency_filter"):
            ResearchConfig(perplexity_recency_filter="invalid")

    def test_validate_perplexity_country_lowercase(self):
        """Test lowercase perplexity_country raises ValueError."""
        with pytest.raises(ValueError, match="Invalid perplexity_country"):
            ResearchConfig(perplexity_country="us")

    def test_validate_perplexity_country_too_long(self):
        """Test 3-letter perplexity_country raises ValueError."""
        with pytest.raises(ValueError, match="Invalid perplexity_country"):
            ResearchConfig(perplexity_country="USA")


class TestPerplexityConfigPrecedence:
    """Tests for configuration precedence (explicit values override defaults)."""

    def test_explicit_value_overrides_default(self):
        """Test explicitly set values override defaults."""
        config = ResearchConfig.from_toml_dict({
            "perplexity_search_context_size": "high",
            "perplexity_max_tokens": 100000,
            "perplexity_recency_filter": "week",
        })

        assert config.perplexity_search_context_size == "high"  # overridden
        assert config.perplexity_max_tokens == 100000  # overridden
        assert config.perplexity_recency_filter == "week"  # overridden
        assert config.perplexity_max_tokens_per_page == 2048  # default preserved
        assert config.perplexity_country is None  # default preserved

    def test_partial_override_preserves_other_defaults(self):
        """Test partial override preserves other default values."""
        config = ResearchConfig.from_toml_dict({
            "perplexity_country": "GB",
        })

        assert config.perplexity_country == "GB"  # overridden
        assert config.perplexity_search_context_size == "medium"  # default preserved
        assert config.perplexity_max_tokens == 50000  # default preserved
        assert config.perplexity_max_tokens_per_page == 2048  # default preserved
        assert config.perplexity_recency_filter is None  # default preserved

    def test_all_perplexity_fields_combined(self):
        """Test all Perplexity fields can be set together."""
        config = ResearchConfig.from_toml_dict({
            "perplexity_search_context_size": "high",
            "perplexity_max_tokens": 75000,
            "perplexity_max_tokens_per_page": 4096,
            "perplexity_recency_filter": "month",
            "perplexity_country": "US",
        })

        assert config.perplexity_search_context_size == "high"
        assert config.perplexity_max_tokens == 75000
        assert config.perplexity_max_tokens_per_page == 4096
        assert config.perplexity_recency_filter == "month"
        assert config.perplexity_country == "US"
