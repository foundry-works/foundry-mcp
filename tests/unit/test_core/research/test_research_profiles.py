"""Tests for research profiles.

Covers:
- ResearchProfile model creation and validation
- Built-in profile registry
- Profile resolution via ResearchConfig.resolve_profile()
- Legacy research_mode backward compatibility
- Per-request overrides
- Config-defined custom profiles
- State serialization roundtrip
"""

from __future__ import annotations

import warnings

import pytest

from foundry_mcp.config.research import ResearchConfig
from foundry_mcp.core.research.models.deep_research import (
    BUILTIN_PROFILES,
    PROFILE_ACADEMIC,
    PROFILE_BIBLIOMETRIC,
    PROFILE_GENERAL,
    PROFILE_SYSTEMATIC_REVIEW,
    PROFILE_TECHNICAL,
    DeepResearchState,
    ResearchExtensions,
    ResearchProfile,
    _RESEARCH_MODE_TO_PROFILE,
)
from foundry_mcp.core.research.models.sources import ResearchMode


# =========================================================================
# Model tests
# =========================================================================


class TestResearchProfileModel:
    """Test ResearchProfile model creation and defaults."""

    def test_default_profile(self):
        p = ResearchProfile()
        assert p.name == "general"
        assert p.source_quality_mode == ResearchMode.GENERAL
        assert p.citation_style == "default"
        assert p.synthesis_template is None
        assert p.enable_citation_tools is False
        assert p.providers == ["tavily", "semantic_scholar"]

    def test_custom_profile(self):
        p = ResearchProfile(
            name="custom",
            providers=["google"],
            citation_style="apa",
            disciplinary_scope=["psychology", "education"],
            time_period="2020-2024",
        )
        assert p.name == "custom"
        assert p.providers == ["google"]
        assert p.citation_style == "apa"
        assert p.disciplinary_scope == ["psychology", "education"]
        assert p.time_period == "2020-2024"

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):  # Pydantic ValidationError
            ResearchProfile(name="bad", unknown_field="value")

    def test_model_copy_with_overrides(self):
        p = PROFILE_ACADEMIC.model_copy(
            update={"citation_style": "chicago", "time_period": "2015-2024"}
        )
        assert p.name == "academic"
        assert p.citation_style == "chicago"
        assert p.time_period == "2015-2024"
        # Other fields preserved
        assert p.source_quality_mode == ResearchMode.ACADEMIC
        assert p.enable_citation_tools is True


# =========================================================================
# Built-in profiles
# =========================================================================


class TestBuiltinProfiles:
    """Test built-in profile registry."""

    def test_all_profiles_registered(self):
        assert set(BUILTIN_PROFILES.keys()) == {
            "general",
            "academic",
            "systematic-review",
            "bibliometric",
            "technical",
        }

    def test_general_profile(self):
        p = PROFILE_GENERAL
        assert p.name == "general"
        assert p.source_quality_mode == ResearchMode.GENERAL
        assert p.citation_style == "default"
        assert p.enable_citation_tools is False

    def test_academic_profile(self):
        p = PROFILE_ACADEMIC
        assert p.name == "academic"
        assert p.source_quality_mode == ResearchMode.ACADEMIC
        assert p.citation_style == "apa"
        assert p.enable_citation_tools is True
        assert "semantic_scholar" in p.providers

    def test_systematic_review_profile(self):
        p = PROFILE_SYSTEMATIC_REVIEW
        assert p.name == "systematic-review"
        assert p.source_quality_mode == ResearchMode.ACADEMIC
        assert p.synthesis_template == "literature_review"
        assert p.enable_methodology_assessment is True
        assert p.enable_pdf_extraction is True

    def test_bibliometric_profile(self):
        p = PROFILE_BIBLIOMETRIC
        assert p.name == "bibliometric"
        assert p.enable_citation_network is True
        assert p.enable_citation_tools is True

    def test_technical_profile(self):
        p = PROFILE_TECHNICAL
        assert p.name == "technical"
        assert p.source_quality_mode == ResearchMode.TECHNICAL
        assert "google" in p.providers

    def test_legacy_mode_mapping_covers_all_modes(self):
        for mode in ResearchMode:
            assert mode.value in _RESEARCH_MODE_TO_PROFILE


# =========================================================================
# Profile resolution
# =========================================================================


class TestProfileResolution:
    """Test ResearchConfig.resolve_profile() resolution logic."""

    def test_default_resolution(self):
        config = ResearchConfig()
        p = config.resolve_profile()
        assert p.name == "general"

    @pytest.mark.parametrize(
        "profile_name",
        ["general", "academic", "systematic-review", "bibliometric", "technical"],
    )
    def test_builtin_profile_by_name(self, profile_name: str):
        config = ResearchConfig()
        p = config.resolve_profile(research_profile=profile_name)
        assert p.name == profile_name

    def test_legacy_research_mode_general(self):
        config = ResearchConfig()
        p = config.resolve_profile(research_mode="general")
        assert p.name == "general"

    def test_legacy_research_mode_academic(self):
        config = ResearchConfig()
        p = config.resolve_profile(research_mode="academic")
        assert p.name == "academic"
        assert p.citation_style == "apa"

    def test_legacy_research_mode_technical(self):
        config = ResearchConfig()
        p = config.resolve_profile(research_mode="technical")
        assert p.name == "technical"

    def test_unknown_profile_raises(self):
        config = ResearchConfig()
        with pytest.raises(ValueError, match="Unknown research profile 'nonexistent'"):
            config.resolve_profile(research_profile="nonexistent")

    def test_unknown_research_mode_raises(self):
        config = ResearchConfig()
        with pytest.raises(ValueError, match="Unknown research_mode 'invalid'"):
            config.resolve_profile(research_mode="invalid")

    def test_per_request_overrides(self):
        config = ResearchConfig()
        p = config.resolve_profile(
            research_profile="academic",
            profile_overrides={"citation_style": "ieee", "time_period": "2020-2024"},
        )
        assert p.name == "academic"
        assert p.citation_style == "ieee"
        assert p.time_period == "2020-2024"
        # Original unchanged
        assert BUILTIN_PROFILES["academic"].citation_style == "apa"

    def test_config_default_profile(self):
        config = ResearchConfig(deep_research_default_profile="technical")
        p = config.resolve_profile()
        assert p.name == "technical"

    def test_config_custom_profile(self):
        config = ResearchConfig(
            deep_research_profiles={
                "my-custom": {
                    "name": "my-custom",
                    "citation_style": "ieee",
                    "providers": ["google"],
                }
            }
        )
        p = config.resolve_profile(research_profile="my-custom")
        assert p.name == "my-custom"
        assert p.citation_style == "ieee"
        assert p.providers == ["google"]

    def test_research_profile_wins_over_research_mode(self):
        config = ResearchConfig()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            p = config.resolve_profile(
                research_profile="academic", research_mode="technical"
            )
            assert p.name == "academic"
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "research_profile takes precedence" in str(w[0].message)

    def test_backward_compat_mode_equals_profile(self):
        """research_mode='academic' should produce identical profile as research_profile='academic'."""
        config = ResearchConfig()
        from_mode = config.resolve_profile(research_mode="academic")
        from_profile = config.resolve_profile(research_profile="academic")
        assert from_mode.model_dump() == from_profile.model_dump()


# =========================================================================
# Extensions and state wiring
# =========================================================================


class TestProfileWiring:
    """Test ResearchProfile wiring into ResearchExtensions and DeepResearchState."""

    def test_extensions_default_none(self):
        ext = ResearchExtensions()
        assert ext.research_profile is None

    def test_extensions_with_profile(self):
        ext = ResearchExtensions(research_profile=PROFILE_ACADEMIC)
        assert ext.research_profile is not None
        assert ext.research_profile.name == "academic"

    def test_state_default_profile(self):
        state = DeepResearchState(original_query="test")
        # Convenience accessor returns default when not set
        assert state.research_profile.name == "general"

    def test_state_with_profile(self):
        state = DeepResearchState(original_query="test")
        state.extensions.research_profile = PROFILE_ACADEMIC
        assert state.research_profile.name == "academic"
        assert state.research_profile.citation_style == "apa"

    def test_serialization_roundtrip_without_profile(self):
        state = DeepResearchState(original_query="test")
        dumped = state.model_dump()
        restored = DeepResearchState.model_validate(dumped)
        assert restored.research_profile.name == "general"

    def test_serialization_roundtrip_with_profile(self):
        state = DeepResearchState(original_query="test")
        state.extensions.research_profile = PROFILE_ACADEMIC
        dumped = state.model_dump()
        restored = DeepResearchState.model_validate(dumped)
        assert restored.research_profile.name == "academic"
        assert restored.research_profile.citation_style == "apa"
        assert restored.research_profile.source_quality_mode == ResearchMode.ACADEMIC

    def test_backward_compat_empty_extensions(self):
        """Old serialized states without research_profile should load fine."""
        state = DeepResearchState.model_validate(
            {"original_query": "test", "extensions": {}}
        )
        assert state.research_profile.name == "general"

    def test_backward_compat_null_profile(self):
        """Old serialized states with null research_profile should load fine."""
        state = DeepResearchState.model_validate(
            {"original_query": "test", "extensions": {"research_profile": None}}
        )
        assert state.research_profile.name == "general"

    def test_extensions_model_dump_excludes_none_profile(self):
        """When profile is None, model_dump should exclude it (exclude_none=True)."""
        ext = ResearchExtensions()
        dumped = ext.model_dump()
        assert "research_profile" not in dumped

    def test_extensions_model_dump_includes_profile(self):
        """When profile is set, model_dump should include it."""
        ext = ResearchExtensions(research_profile=PROFILE_ACADEMIC)
        dumped = ext.model_dump()
        assert "research_profile" in dumped
        assert dumped["research_profile"]["name"] == "academic"
