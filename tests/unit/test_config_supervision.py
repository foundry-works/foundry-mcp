"""Tests for supervision config validation and deprecation warnings.

Covers PLAN Phase 2 / Phase 5B:
- 2A: _validate_supervision_config() clamps and raises correctly
- 2B: Deprecated fields produce DeprecationWarning with migration hints
- 2C: _COST_TIER_MODEL_DEFAULTS uses full model names
"""

import warnings

import pytest

from foundry_mcp.config.research import ResearchConfig

# ---------------------------------------------------------------------------
# 2A / 5B.1 — Supervision config validation
# ---------------------------------------------------------------------------


class TestSupervisionConfigValidation:
    """Tests for _validate_supervision_config() method."""

    def test_max_supervision_rounds_clamped_to_20(self):
        """max_supervision_rounds > 20 is clamped with warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = ResearchConfig(deep_research_max_supervision_rounds=100)

        assert config.deep_research_max_supervision_rounds == 20
        assert len(w) == 1
        assert "clamping to 20" in str(w[0].message)

    def test_max_supervision_rounds_at_boundary(self):
        """max_supervision_rounds == 20 is accepted without warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = ResearchConfig(deep_research_max_supervision_rounds=20)

        assert config.deep_research_max_supervision_rounds == 20
        supervision_warnings = [x for x in w if "max_supervision_rounds" in str(x.message)]
        assert len(supervision_warnings) == 0

    def test_max_supervision_rounds_below_minimum_raises(self):
        """max_supervision_rounds < 1 raises ValueError."""
        with pytest.raises(ValueError, match="Must be >= 1"):
            ResearchConfig(deep_research_max_supervision_rounds=0)

    def test_max_supervision_rounds_default_accepted(self):
        """Default value (6) passes validation."""
        config = ResearchConfig()
        assert config.deep_research_max_supervision_rounds == 6

    # --- coverage_confidence_threshold ---

    def test_coverage_confidence_threshold_valid(self):
        """Threshold in [0.0, 1.0] is accepted."""
        config = ResearchConfig(deep_research_coverage_confidence_threshold=0.5)
        assert config.deep_research_coverage_confidence_threshold == 0.5

    def test_coverage_confidence_threshold_boundary_zero(self):
        """Threshold == 0.0 is accepted."""
        config = ResearchConfig(deep_research_coverage_confidence_threshold=0.0)
        assert config.deep_research_coverage_confidence_threshold == 0.0

    def test_coverage_confidence_threshold_boundary_one(self):
        """Threshold == 1.0 is accepted."""
        config = ResearchConfig(deep_research_coverage_confidence_threshold=1.0)
        assert config.deep_research_coverage_confidence_threshold == 1.0

    def test_coverage_confidence_threshold_above_one_raises(self):
        """Threshold > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match=r"Must be in \[0\.0, 1\.0\]"):
            ResearchConfig(deep_research_coverage_confidence_threshold=1.5)

    def test_coverage_confidence_threshold_negative_raises(self):
        """Threshold < 0.0 raises ValueError."""
        with pytest.raises(ValueError, match=r"Must be in \[0\.0, 1\.0\]"):
            ResearchConfig(deep_research_coverage_confidence_threshold=-0.1)

    # --- max_concurrent_research_units ---

    def test_max_concurrent_research_units_clamped(self):
        """max_concurrent_research_units > 20 is clamped with warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = ResearchConfig(deep_research_max_concurrent_research_units=50)

        assert config.deep_research_max_concurrent_research_units == 20
        assert any("clamping to 20" in str(x.message) for x in w)

    def test_max_concurrent_research_units_below_minimum_raises(self):
        """max_concurrent_research_units < 1 raises ValueError."""
        with pytest.raises(ValueError, match="Must be >= 1"):
            ResearchConfig(deep_research_max_concurrent_research_units=0)

    def test_from_toml_dict_clamps_supervision_rounds(self):
        """from_toml_dict also triggers validation (via __post_init__)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = ResearchConfig.from_toml_dict({"deep_research_max_supervision_rounds": 100})

        assert config.deep_research_max_supervision_rounds == 20
        assert any("clamping to 20" in str(x.message) for x in w)


# ---------------------------------------------------------------------------
# 2.3 — Deep research bounds validation
# ---------------------------------------------------------------------------


class TestDeepResearchBoundsValidation:
    """Tests for _validate_deep_research_bounds() method."""

    # --- deep_research_max_iterations ---

    def test_max_iterations_clamped(self):
        """max_iterations > 20 is clamped with warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = ResearchConfig(deep_research_max_iterations=100)

        assert config.deep_research_max_iterations == 20
        assert any("deep_research_max_iterations" in str(x.message) for x in w)

    def test_max_iterations_at_boundary(self):
        """max_iterations == 20 is accepted without warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = ResearchConfig(deep_research_max_iterations=20)

        assert config.deep_research_max_iterations == 20
        iteration_warnings = [x for x in w if "deep_research_max_iterations" in str(x.message)]
        assert len(iteration_warnings) == 0

    def test_max_iterations_below_minimum_raises(self):
        """max_iterations < 1 raises ValueError."""
        with pytest.raises(ValueError, match="Must be >= 1"):
            ResearchConfig(deep_research_max_iterations=0)

    # --- deep_research_max_sub_queries ---

    def test_max_sub_queries_clamped(self):
        """max_sub_queries > 50 is clamped with warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = ResearchConfig(deep_research_max_sub_queries=200)

        assert config.deep_research_max_sub_queries == 50
        assert any("deep_research_max_sub_queries" in str(x.message) for x in w)

    def test_max_sub_queries_below_minimum_raises(self):
        """max_sub_queries < 1 raises ValueError."""
        with pytest.raises(ValueError, match="Must be >= 1"):
            ResearchConfig(deep_research_max_sub_queries=0)

    # --- deep_research_max_sources ---

    def test_max_sources_clamped(self):
        """max_sources > 100 is clamped with warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = ResearchConfig(deep_research_max_sources=500)

        assert config.deep_research_max_sources == 100
        assert any("deep_research_max_sources" in str(x.message) for x in w)

    def test_max_sources_below_minimum_raises(self):
        """max_sources < 1 raises ValueError."""
        with pytest.raises(ValueError, match="Must be >= 1"):
            ResearchConfig(deep_research_max_sources=0)

    # --- deep_research_max_concurrent ---

    def test_max_concurrent_clamped(self):
        """max_concurrent > 20 is clamped with warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = ResearchConfig(deep_research_max_concurrent=50)

        assert config.deep_research_max_concurrent == 20
        assert any("deep_research_max_concurrent" in str(x.message) for x in w)

    def test_max_concurrent_below_minimum_raises(self):
        """max_concurrent < 1 raises ValueError."""
        with pytest.raises(ValueError, match="Must be >= 1"):
            ResearchConfig(deep_research_max_concurrent=0)

    # --- default_timeout ---

    def test_default_timeout_clamped(self):
        """default_timeout > 3600 is clamped with warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = ResearchConfig(default_timeout=10000)

        assert config.default_timeout == 3600.0
        assert any("default_timeout" in str(x.message) for x in w)

    def test_default_timeout_below_minimum_raises(self):
        """default_timeout < 1 raises ValueError."""
        with pytest.raises(ValueError, match="Must be >= 1"):
            ResearchConfig(default_timeout=0)

    # --- token_safety_margin ---

    def test_token_safety_margin_above_one_raises(self):
        """token_safety_margin > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match=r"Must be in \[0\.0, 1\.0\]"):
            ResearchConfig(token_safety_margin=1.5)

    def test_token_safety_margin_negative_raises(self):
        """token_safety_margin < 0.0 raises ValueError."""
        with pytest.raises(ValueError, match=r"Must be in \[0\.0, 1\.0\]"):
            ResearchConfig(token_safety_margin=-0.1)

    def test_token_safety_margin_boundary_zero(self):
        """token_safety_margin == 0.0 is accepted."""
        config = ResearchConfig(token_safety_margin=0.0)
        assert config.token_safety_margin == 0.0

    def test_token_safety_margin_boundary_one(self):
        """token_safety_margin == 1.0 is accepted."""
        config = ResearchConfig(token_safety_margin=1.0)
        assert config.token_safety_margin == 1.0

    def test_defaults_pass_validation(self):
        """Default values pass all bounds validation."""
        config = ResearchConfig()
        assert config.deep_research_max_iterations == 3
        assert config.deep_research_max_sub_queries == 5
        assert config.deep_research_max_sources == 5
        assert config.deep_research_max_concurrent == 3
        assert config.default_timeout == 360.0
        assert config.token_safety_margin == 0.15


# ---------------------------------------------------------------------------
# 4A — Phase-specific timeout, threshold, and content length validation
# ---------------------------------------------------------------------------


class TestPhaseSpecificTimeoutValidation:
    """Tests for phase-specific timeout and threshold validation in _validate_deep_research_bounds()."""

    @pytest.mark.parametrize(
        "field_name,default_val",
        [
            ("deep_research_planning_timeout", 360.0),
            ("deep_research_synthesis_timeout", 600.0),
            ("deep_research_supervision_wall_clock_timeout", 1800.0),
            ("deep_research_reflection_timeout", 60.0),
            ("deep_research_evaluation_timeout", 360.0),
            ("deep_research_digest_timeout", 120.0),
            ("deep_research_summarization_timeout", 60),
            ("deep_research_retry_delay", 5.0),
        ],
    )
    def test_negative_timeout_reset_to_default(self, field_name: str, default_val, caplog):
        """Negative timeout values are reset to defaults with a warning."""
        import logging

        with caplog.at_level(logging.WARNING, logger="foundry_mcp.config.research"):
            config = ResearchConfig(**{field_name: -10})

        assert getattr(config, field_name) == default_val
        assert any(f"{field_name}=-10" in rec.message for rec in caplog.records)

    @pytest.mark.parametrize(
        "field_name,default_val",
        [
            ("deep_research_planning_timeout", 360.0),
            ("deep_research_synthesis_timeout", 600.0),
            ("deep_research_supervision_wall_clock_timeout", 1800.0),
            ("deep_research_reflection_timeout", 60.0),
            ("deep_research_evaluation_timeout", 360.0),
            ("deep_research_digest_timeout", 120.0),
            ("deep_research_summarization_timeout", 60),
            ("deep_research_retry_delay", 5.0),
        ],
    )
    def test_zero_timeout_reset_to_default(self, field_name: str, default_val, caplog):
        """Zero timeout values are reset to defaults with a warning."""
        import logging

        with caplog.at_level(logging.WARNING, logger="foundry_mcp.config.research"):
            config = ResearchConfig(**{field_name: 0})

        assert getattr(config, field_name) == default_val
        assert any("invalid (must be > 0)" in rec.message for rec in caplog.records)

    @pytest.mark.parametrize(
        "field_name",
        [
            "deep_research_planning_timeout",
            "deep_research_synthesis_timeout",
            "deep_research_supervision_wall_clock_timeout",
            "deep_research_reflection_timeout",
            "deep_research_evaluation_timeout",
            "deep_research_digest_timeout",
            "deep_research_summarization_timeout",
            "deep_research_retry_delay",
        ],
    )
    def test_positive_timeout_accepted(self, field_name: str, caplog):
        """Positive timeout values are accepted without warning."""
        import logging

        with caplog.at_level(logging.WARNING, logger="foundry_mcp.config.research"):
            config = ResearchConfig(**{field_name: 42.0})

        assert getattr(config, field_name) == 42.0
        timeout_warnings = [r for r in caplog.records if field_name in r.message]
        assert len(timeout_warnings) == 0

    # --- content_dedup_threshold ---

    def test_content_dedup_threshold_above_one_reset(self, caplog):
        """content_dedup_threshold > 1.0 is reset to 0.8."""
        import logging

        with caplog.at_level(logging.WARNING, logger="foundry_mcp.config.research"):
            config = ResearchConfig(deep_research_content_dedup_threshold=1.5)

        assert config.deep_research_content_dedup_threshold == 0.8
        assert any("content_dedup_threshold" in rec.message for rec in caplog.records)

    def test_content_dedup_threshold_negative_reset(self, caplog):
        """content_dedup_threshold < 0.0 is reset to 0.8."""
        import logging

        with caplog.at_level(logging.WARNING, logger="foundry_mcp.config.research"):
            config = ResearchConfig(deep_research_content_dedup_threshold=-0.1)

        assert config.deep_research_content_dedup_threshold == 0.8
        assert any("content_dedup_threshold" in rec.message for rec in caplog.records)

    def test_content_dedup_threshold_valid_boundaries(self):
        """content_dedup_threshold at 0.0 and 1.0 boundaries accepted."""
        config_zero = ResearchConfig(deep_research_content_dedup_threshold=0.0)
        assert config_zero.deep_research_content_dedup_threshold == 0.0

        config_one = ResearchConfig(deep_research_content_dedup_threshold=1.0)
        assert config_one.deep_research_content_dedup_threshold == 1.0

    # --- compression_max_content_length ---

    def test_compression_max_content_length_zero_reset(self, caplog):
        """compression_max_content_length == 0 is reset to 50000."""
        import logging

        with caplog.at_level(logging.WARNING, logger="foundry_mcp.config.research"):
            config = ResearchConfig(deep_research_compression_max_content_length=0)

        assert config.deep_research_compression_max_content_length == 50000
        assert any("compression_max_content_length" in rec.message for rec in caplog.records)

    def test_compression_max_content_length_negative_reset(self, caplog):
        """compression_max_content_length < 0 is reset to 50000."""
        import logging

        with caplog.at_level(logging.WARNING, logger="foundry_mcp.config.research"):
            config = ResearchConfig(deep_research_compression_max_content_length=-100)

        assert config.deep_research_compression_max_content_length == 50000
        assert any("compression_max_content_length" in rec.message for rec in caplog.records)

    def test_compression_max_content_length_positive_accepted(self):
        """Positive compression_max_content_length is accepted."""
        config = ResearchConfig(deep_research_compression_max_content_length=75000)
        assert config.deep_research_compression_max_content_length == 75000


# ---------------------------------------------------------------------------
# 2B / 5B.3 — Deprecated field warnings
# ---------------------------------------------------------------------------


class TestDeprecatedFieldWarnings:
    """Tests for deprecated config field detection in from_toml_dict."""

    @pytest.mark.parametrize(
        "field_name",
        [
            "deep_research_enable_reflection",
            "deep_research_enable_contradiction_detection",
            "deep_research_enable_topic_agents",
            "deep_research_analysis_timeout",
            "deep_research_refinement_timeout",
            "deep_research_analysis_provider",
            "deep_research_refinement_provider",
            "deep_research_analysis_providers",
            "deep_research_refinement_providers",
            "tavily_extract_in_deep_research",
            "tavily_extract_max_urls",
            "deep_research_digest_policy",
        ],
    )
    def test_deprecated_field_emits_warning(self, field_name: str):
        """Each deprecated field emits a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ResearchConfig.from_toml_dict({field_name: "some_value"})

        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1, (
            f"Expected DeprecationWarning for '{field_name}', got: {[str(x.message) for x in w]}"
        )
        msg = str(deprecation_warnings[0].message)
        assert field_name in msg
        assert "has been removed" in msg

    def test_deprecated_field_includes_migration_hint(self):
        """Warning message includes migration guidance."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ResearchConfig.from_toml_dict({"deep_research_enable_reflection": True})

        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        msg = str(deprecation_warnings[0].message)
        assert "always-on" in msg.lower() or "supervision" in msg.lower()

    def test_multiple_deprecated_fields_emit_multiple_warnings(self):
        """Multiple deprecated fields each get their own warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ResearchConfig.from_toml_dict(
                {
                    "deep_research_enable_reflection": True,
                    "deep_research_digest_policy": "auto",
                    "tavily_extract_in_deep_research": False,
                }
            )

        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 3

    def test_valid_field_no_deprecation_warning(self):
        """Known active fields do not trigger deprecation warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ResearchConfig.from_toml_dict({"deep_research_max_supervision_rounds": 6})

        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 0


# ---------------------------------------------------------------------------
# 2C — Cost tier model defaults use full model names
# ---------------------------------------------------------------------------


class TestCostTierModelDefaults:
    """Tests for _COST_TIER_MODEL_DEFAULTS using full model names."""

    def test_cost_tier_defaults_match_token_limits(self):
        """Cost tier model names should match model_token_limits.json entries."""
        import json
        from pathlib import Path

        limits_path = Path(__file__).resolve().parents[2] / "src" / "foundry_mcp" / "config" / "model_token_limits.json"
        with open(limits_path) as f:
            token_limits = json.load(f)["limits"]

        for role, model_name in ResearchConfig._COST_TIER_MODEL_DEFAULTS.items():
            # The model name should appear as a key in model_token_limits.json
            assert model_name in token_limits, (
                f"Cost tier default for '{role}' role is '{model_name}', "
                f"but it's not in model_token_limits.json. "
                f"Available: {list(token_limits.keys())}"
            )

    def test_cost_tier_defaults_not_ambiguous_short_names(self):
        """Cost tier model names should be full names, not short like '2.0-flash'."""
        for role, model_name in ResearchConfig._COST_TIER_MODEL_DEFAULTS.items():
            # Full names include the vendor prefix (e.g. "gemini-")
            assert not model_name[0].isdigit(), (
                f"Cost tier default for '{role}' role is '{model_name}', "
                f"which looks like a short name. Use full model names "
                f"(e.g., 'gemini-2.5-flash') for reliable token limit estimation."
            )
