"""Tests for Tavily configuration fields in ResearchConfig.

Tests cover:
1. TOML parsing for all Tavily search and extract fields
2. Validation errors for invalid values
3. Default values preserved when not set
4. Precedence rules (explicit values override defaults)
"""

import pytest

from foundry_mcp.config import ResearchConfig


class TestTavilySearchConfigParsing:
    """Tests for Tavily search configuration TOML parsing."""

    def test_parse_tavily_search_depth_basic(self):
        """Test tavily_search_depth='basic' is parsed correctly."""
        config = ResearchConfig.from_toml_dict({
            "tavily_search_depth": "basic"
        })
        assert config.tavily_search_depth == "basic"

    def test_parse_tavily_search_depth_advanced(self):
        """Test tavily_search_depth='advanced' is parsed correctly."""
        config = ResearchConfig.from_toml_dict({
            "tavily_search_depth": "advanced"
        })
        assert config.tavily_search_depth == "advanced"

    def test_parse_tavily_search_depth_fast(self):
        """Test tavily_search_depth='fast' is parsed correctly."""
        config = ResearchConfig.from_toml_dict({
            "tavily_search_depth": "fast"
        })
        assert config.tavily_search_depth == "fast"

    def test_parse_tavily_search_depth_ultra_fast(self):
        """Test tavily_search_depth='ultra_fast' is parsed correctly."""
        config = ResearchConfig.from_toml_dict({
            "tavily_search_depth": "ultra_fast"
        })
        assert config.tavily_search_depth == "ultra_fast"

    def test_parse_tavily_topic_general(self):
        """Test tavily_topic='general' is parsed correctly."""
        config = ResearchConfig.from_toml_dict({
            "tavily_topic": "general"
        })
        assert config.tavily_topic == "general"

    def test_parse_tavily_topic_news(self):
        """Test tavily_topic='news' is parsed correctly."""
        config = ResearchConfig.from_toml_dict({
            "tavily_topic": "news"
        })
        assert config.tavily_topic == "news"

    def test_parse_tavily_news_days(self):
        """Test tavily_news_days is parsed correctly."""
        config = ResearchConfig.from_toml_dict({
            "tavily_news_days": 7
        })
        assert config.tavily_news_days == 7

    def test_parse_tavily_include_images(self):
        """Test tavily_include_images is parsed correctly."""
        config = ResearchConfig.from_toml_dict({
            "tavily_include_images": True
        })
        assert config.tavily_include_images is True

    def test_parse_tavily_country(self):
        """Test tavily_country is parsed correctly."""
        config = ResearchConfig.from_toml_dict({
            "tavily_country": "US"
        })
        assert config.tavily_country == "US"

    def test_parse_tavily_chunks_per_source(self):
        """Test tavily_chunks_per_source is parsed correctly."""
        config = ResearchConfig.from_toml_dict({
            "tavily_chunks_per_source": 3
        })
        assert config.tavily_chunks_per_source == 3

    def test_parse_tavily_auto_parameters(self):
        """Test tavily_auto_parameters is parsed correctly."""
        config = ResearchConfig.from_toml_dict({
            "tavily_auto_parameters": True
        })
        assert config.tavily_auto_parameters is True


class TestTavilyExtractConfigParsing:
    """Tests for Tavily extract configuration TOML parsing."""

    def test_parse_tavily_extract_depth_basic(self):
        """Test tavily_extract_depth='basic' is parsed correctly."""
        config = ResearchConfig.from_toml_dict({
            "tavily_extract_depth": "basic"
        })
        assert config.tavily_extract_depth == "basic"

    def test_parse_tavily_extract_depth_advanced(self):
        """Test tavily_extract_depth='advanced' is parsed correctly."""
        config = ResearchConfig.from_toml_dict({
            "tavily_extract_depth": "advanced"
        })
        assert config.tavily_extract_depth == "advanced"

    def test_parse_tavily_extract_include_images(self):
        """Test tavily_extract_include_images is parsed correctly."""
        config = ResearchConfig.from_toml_dict({
            "tavily_extract_include_images": True
        })
        assert config.tavily_extract_include_images is True

    def test_parse_tavily_extract_in_deep_research(self):
        """Test tavily_extract_in_deep_research is parsed correctly."""
        config = ResearchConfig.from_toml_dict({
            "tavily_extract_in_deep_research": True
        })
        assert config.tavily_extract_in_deep_research is True

    def test_parse_tavily_extract_max_urls(self):
        """Test tavily_extract_max_urls is parsed correctly."""
        config = ResearchConfig.from_toml_dict({
            "tavily_extract_max_urls": 10
        })
        assert config.tavily_extract_max_urls == 10


class TestTavilyConfigDefaults:
    """Tests for Tavily configuration default values."""

    def test_default_tavily_search_depth(self):
        """Test default tavily_search_depth is 'basic'."""
        config = ResearchConfig()
        assert config.tavily_search_depth == "basic"

    def test_default_tavily_topic(self):
        """Test default tavily_topic is 'general'."""
        config = ResearchConfig()
        assert config.tavily_topic == "general"

    def test_default_tavily_news_days_is_none(self):
        """Test default tavily_news_days is None."""
        config = ResearchConfig()
        assert config.tavily_news_days is None

    def test_default_tavily_include_images_is_false(self):
        """Test default tavily_include_images is False."""
        config = ResearchConfig()
        assert config.tavily_include_images is False

    def test_default_tavily_country_is_none(self):
        """Test default tavily_country is None."""
        config = ResearchConfig()
        assert config.tavily_country is None

    def test_default_tavily_chunks_per_source(self):
        """Test default tavily_chunks_per_source is 3."""
        config = ResearchConfig()
        assert config.tavily_chunks_per_source == 3

    def test_default_tavily_auto_parameters_is_false(self):
        """Test default tavily_auto_parameters is False."""
        config = ResearchConfig()
        assert config.tavily_auto_parameters is False

    def test_default_tavily_extract_depth(self):
        """Test default tavily_extract_depth is 'basic'."""
        config = ResearchConfig()
        assert config.tavily_extract_depth == "basic"

    def test_default_tavily_extract_include_images_is_false(self):
        """Test default tavily_extract_include_images is False."""
        config = ResearchConfig()
        assert config.tavily_extract_include_images is False

    def test_default_tavily_extract_in_deep_research_is_false(self):
        """Test default tavily_extract_in_deep_research is False."""
        config = ResearchConfig()
        assert config.tavily_extract_in_deep_research is False

    def test_default_tavily_extract_max_urls(self):
        """Test default tavily_extract_max_urls is 5."""
        config = ResearchConfig()
        assert config.tavily_extract_max_urls == 5


class TestTavilyConfigValidation:
    """Tests for Tavily configuration validation."""

    def test_validate_tavily_search_depth_invalid(self):
        """Test invalid tavily_search_depth raises ValueError."""
        with pytest.raises(ValueError, match="Invalid tavily_search_depth"):
            ResearchConfig(tavily_search_depth="invalid")

    def test_validate_tavily_topic_invalid(self):
        """Test invalid tavily_topic raises ValueError."""
        with pytest.raises(ValueError, match="Invalid tavily_topic"):
            ResearchConfig(tavily_topic="invalid")

    def test_validate_tavily_news_days_zero(self):
        """Test tavily_news_days=0 raises ValueError."""
        with pytest.raises(ValueError, match="Invalid tavily_news_days"):
            ResearchConfig(tavily_news_days=0)

    def test_validate_tavily_news_days_negative(self):
        """Test negative tavily_news_days raises ValueError."""
        with pytest.raises(ValueError, match="Invalid tavily_news_days"):
            ResearchConfig(tavily_news_days=-1)

    def test_validate_tavily_news_days_over_limit(self):
        """Test tavily_news_days>365 raises ValueError."""
        with pytest.raises(ValueError, match="Invalid tavily_news_days"):
            ResearchConfig(tavily_news_days=366)

    def test_validate_tavily_country_lowercase(self):
        """Test lowercase tavily_country raises ValueError."""
        with pytest.raises(ValueError, match="Invalid tavily_country"):
            ResearchConfig(tavily_country="us")

    def test_validate_tavily_country_too_long(self):
        """Test 3-letter tavily_country raises ValueError."""
        with pytest.raises(ValueError, match="Invalid tavily_country"):
            ResearchConfig(tavily_country="USA")

    def test_validate_tavily_chunks_per_source_zero(self):
        """Test tavily_chunks_per_source=0 raises ValueError."""
        with pytest.raises(ValueError, match="Invalid tavily_chunks_per_source"):
            ResearchConfig(tavily_chunks_per_source=0)

    def test_validate_tavily_chunks_per_source_over_limit(self):
        """Test tavily_chunks_per_source>5 raises ValueError."""
        with pytest.raises(ValueError, match="Invalid tavily_chunks_per_source"):
            ResearchConfig(tavily_chunks_per_source=6)

    def test_validate_tavily_extract_depth_invalid(self):
        """Test invalid tavily_extract_depth raises ValueError."""
        with pytest.raises(ValueError, match="Invalid tavily_extract_depth"):
            ResearchConfig(tavily_extract_depth="invalid")


class TestTavilyConfigPrecedence:
    """Tests for configuration precedence (explicit values override defaults)."""

    def test_explicit_value_overrides_default(self):
        """Test explicitly set values override defaults."""
        config = ResearchConfig.from_toml_dict({
            "tavily_search_depth": "advanced",
            "tavily_topic": "news",
            "tavily_news_days": 30,
        })

        assert config.tavily_search_depth == "advanced"  # overridden
        assert config.tavily_topic == "news"  # overridden
        assert config.tavily_news_days == 30  # overridden
        assert config.tavily_include_images is False  # default preserved

    def test_partial_override_preserves_other_defaults(self):
        """Test partial override preserves other default values."""
        config = ResearchConfig.from_toml_dict({
            "tavily_extract_depth": "advanced",
        })

        assert config.tavily_extract_depth == "advanced"  # overridden
        assert config.tavily_extract_include_images is False  # default preserved
        assert config.tavily_extract_in_deep_research is False  # default preserved
        assert config.tavily_extract_max_urls == 5  # default preserved

    def test_all_tavily_fields_combined(self):
        """Test all Tavily fields can be set together."""
        config = ResearchConfig.from_toml_dict({
            # Search fields
            "tavily_search_depth": "advanced",
            "tavily_topic": "news",
            "tavily_news_days": 7,
            "tavily_include_images": True,
            "tavily_country": "US",
            "tavily_chunks_per_source": 5,
            "tavily_auto_parameters": True,
            # Extract fields
            "tavily_extract_depth": "advanced",
            "tavily_extract_include_images": True,
            "tavily_extract_in_deep_research": True,
            "tavily_extract_max_urls": 10,
        })

        # Search fields
        assert config.tavily_search_depth == "advanced"
        assert config.tavily_topic == "news"
        assert config.tavily_news_days == 7
        assert config.tavily_include_images is True
        assert config.tavily_country == "US"
        assert config.tavily_chunks_per_source == 5
        assert config.tavily_auto_parameters is True

        # Extract fields
        assert config.tavily_extract_depth == "advanced"
        assert config.tavily_extract_include_images is True
        assert config.tavily_extract_in_deep_research is True
        assert config.tavily_extract_max_urls == 10
