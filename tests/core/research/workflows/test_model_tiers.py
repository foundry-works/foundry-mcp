"""Unit tests for tier-based model configuration.

Tests cover:
1. ModelTierConfig dataclass parsing and methods
2. ResearchConfig._build_tier_config — string form, table form, role_assignments
3. resolve_model_for_role tier integration — priority ordering
4. get_summarization_model / get_compression_model tier awareness
5. Validation warnings for invalid tier/role names
6. Backward compatibility — no tiers = identical to existing behavior
7. from_toml_dict parsing of model_tiers key
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from foundry_mcp.config.research import ResearchConfig
from foundry_mcp.config.research_sub_configs import ModelTierConfig


# =============================================================================
# Helpers
# =============================================================================


def _make_config(**overrides: Any) -> ResearchConfig:
    """Create a ResearchConfig with optional overrides."""
    return ResearchConfig(**overrides)


# =============================================================================
# Tests: ModelTierConfig dataclass
# =============================================================================


class TestModelTierConfig:
    """Test the ModelTierConfig frozen dataclass."""

    def test_default_disabled(self) -> None:
        """Default-constructed config is disabled."""
        cfg = ModelTierConfig()
        assert cfg.enabled is False
        assert cfg.tiers == {}
        assert cfg.tier_models == {}
        assert cfg.role_assignments == {}

    def test_resolve_tier(self) -> None:
        """resolve_tier returns provider spec for known tier."""
        cfg = ModelTierConfig(
            tiers={"frontier": "[cli]gemini:pro"},
            enabled=True,
        )
        assert cfg.resolve_tier("frontier") == "[cli]gemini:pro"
        assert cfg.resolve_tier("nonexistent") is None

    def test_resolve_tier_model(self) -> None:
        """resolve_tier_model returns explicit model for known tier."""
        cfg = ModelTierConfig(
            tiers={"frontier": "[cli]gemini:pro"},
            tier_models={"frontier": "gemini-2.5-pro-preview"},
            enabled=True,
        )
        assert cfg.resolve_tier_model("frontier") == "gemini-2.5-pro-preview"
        assert cfg.resolve_tier_model("standard") is None

    def test_get_tier_for_role_custom_override(self) -> None:
        """Custom role_assignments take precedence over defaults."""
        defaults = {"research": "frontier", "summarization": "efficient"}
        cfg = ModelTierConfig(
            tiers={"frontier": "x", "standard": "y"},
            role_assignments={"research": "standard"},
            enabled=True,
        )
        assert cfg.get_tier_for_role("research", defaults) == "standard"

    def test_get_tier_for_role_default_fallback(self) -> None:
        """Falls back to default assignments when no custom override."""
        defaults = {"research": "frontier", "summarization": "efficient"}
        cfg = ModelTierConfig(
            tiers={"frontier": "x"},
            enabled=True,
        )
        assert cfg.get_tier_for_role("research", defaults) == "frontier"

    def test_get_tier_for_role_unknown(self) -> None:
        """Returns None for unknown role with no default."""
        cfg = ModelTierConfig(tiers={"frontier": "x"}, enabled=True)
        assert cfg.get_tier_for_role("nonexistent", {}) is None

    def test_frozen(self) -> None:
        """ModelTierConfig is frozen — attributes cannot be reassigned."""
        cfg = ModelTierConfig(enabled=True)
        with pytest.raises(AttributeError):
            cfg.enabled = False  # type: ignore[misc]


# =============================================================================
# Tests: _build_tier_config parsing
# =============================================================================


class TestBuildTierConfig:
    """Test ResearchConfig._build_tier_config static method."""

    def test_none_returns_disabled(self) -> None:
        """None input returns disabled config."""
        cfg = ResearchConfig._build_tier_config(None)
        assert cfg.enabled is False

    def test_empty_dict_returns_disabled(self) -> None:
        """Empty dict returns disabled config."""
        cfg = ResearchConfig._build_tier_config({})
        assert cfg.enabled is False

    def test_string_form(self) -> None:
        """Simple string form: tier_name = 'provider_spec'."""
        cfg = ResearchConfig._build_tier_config({
            "frontier": "[cli]gemini:pro",
            "standard": "[cli]gemini:flash",
        })
        assert cfg.enabled is True
        assert cfg.tiers == {
            "frontier": "[cli]gemini:pro",
            "standard": "[cli]gemini:flash",
        }
        assert cfg.tier_models == {}

    def test_table_form(self) -> None:
        """Table form with provider and model keys."""
        cfg = ResearchConfig._build_tier_config({
            "frontier": {
                "provider": "[cli]gemini:pro",
                "model": "gemini-2.5-pro-preview-0506",
            },
        })
        assert cfg.enabled is True
        assert cfg.tiers == {"frontier": "[cli]gemini:pro"}
        assert cfg.tier_models == {"frontier": "gemini-2.5-pro-preview-0506"}

    def test_table_form_provider_only(self) -> None:
        """Table form with provider but no model."""
        cfg = ResearchConfig._build_tier_config({
            "efficient": {"provider": "[cli]gemini:flash"},
        })
        assert cfg.enabled is True
        assert cfg.tiers == {"efficient": "[cli]gemini:flash"}
        assert cfg.tier_models == {}

    def test_mixed_string_and_table(self) -> None:
        """Mix of string form and table form tiers."""
        cfg = ResearchConfig._build_tier_config({
            "frontier": {"provider": "[cli]gemini:pro", "model": "my-model"},
            "standard": "[cli]gemini:flash",
        })
        assert cfg.enabled is True
        assert cfg.tiers["frontier"] == "[cli]gemini:pro"
        assert cfg.tiers["standard"] == "[cli]gemini:flash"
        assert cfg.tier_models["frontier"] == "my-model"
        assert "standard" not in cfg.tier_models

    def test_role_assignments_parsed(self) -> None:
        """role_assignments sub-dict is extracted correctly."""
        cfg = ResearchConfig._build_tier_config({
            "frontier": "[cli]gemini:pro",
            "role_assignments": {
                "summarization": "frontier",
                "compression": "standard",
            },
        })
        assert cfg.enabled is True
        assert cfg.role_assignments == {
            "summarization": "frontier",
            "compression": "standard",
        }

    def test_only_role_assignments_no_tiers_returns_disabled(self) -> None:
        """If only role_assignments but no tiers defined, config is disabled."""
        cfg = ResearchConfig._build_tier_config({
            "role_assignments": {"research": "frontier"},
        })
        assert cfg.enabled is False


# =============================================================================
# Tests: resolve_model_for_role tier integration
# =============================================================================


class TestResolveTierIntegration:
    """Test tier-based lookup within resolve_model_for_role."""

    def test_no_tiers_configured(self) -> None:
        """No tier config — model_tier_config.enabled is False, no behavior change."""
        config = _make_config(default_provider="gemini")
        assert config.model_tier_config.enabled is False
        provider, model = config.resolve_model_for_role("research")
        assert provider == "gemini"
        assert model is None

    def test_tier_provides_provider_for_role(self) -> None:
        """Tier provides provider when no per-role config is set."""
        config = _make_config(
            default_provider="gemini",
            _tier_config={
                "frontier": "[cli]claude:opus",
            },
        )
        provider, model = config.resolve_model_for_role("research")
        assert provider == "claude"
        assert model == "opus"

    def test_tier_with_table_form_model_override(self) -> None:
        """Table-form tier with explicit model override."""
        config = _make_config(
            default_provider="gemini",
            _tier_config={
                "frontier": {
                    "provider": "[cli]claude:opus",
                    "model": "claude-opus-4-6",
                },
            },
        )
        provider, model = config.resolve_model_for_role("research")
        assert provider == "claude"
        # Explicit tier model overrides embedded spec model
        assert model == "claude-opus-4-6"

    def test_per_role_override_beats_tier(self) -> None:
        """Per-role config (priority 1) takes precedence over tier (priority 2)."""
        config = _make_config(
            default_provider="gemini",
            deep_research_research_provider="openai",
            deep_research_research_model="gpt-4o",
            _tier_config={
                "frontier": "[cli]claude:opus",
            },
        )
        provider, model = config.resolve_model_for_role("research")
        assert provider == "openai"
        assert model == "gpt-4o"

    def test_tier_beats_cost_tier_defaults(self) -> None:
        """Tier (priority 2) overrides cost-tier defaults (priority 3)."""
        config = _make_config(
            default_provider="gemini",
            _tier_config={
                "efficient": "[cli]claude:haiku",
            },
        )
        # summarization is in "efficient" tier by default
        provider, model = config.resolve_model_for_role("summarization")
        assert provider == "claude"
        assert model == "haiku"

    def test_custom_role_assignment(self) -> None:
        """Custom role_assignments remap a role to a different tier."""
        config = _make_config(
            default_provider="gemini",
            _tier_config={
                "frontier": "[cli]claude:opus",
                "standard": "[cli]gemini:flash",
                "role_assignments": {
                    "summarization": "frontier",  # Promote from efficient → frontier
                },
            },
        )
        provider, model = config.resolve_model_for_role("summarization")
        assert provider == "claude"
        assert model == "opus"

    def test_all_11_roles_resolve_through_tiers(self) -> None:
        """All 11 roles resolve correctly through the tier system."""
        config = _make_config(
            default_provider="gemini",
            _tier_config={
                "frontier": "[cli]claude:opus",
                "standard": "[cli]gemini:flash",
                "efficient": "[cli]gemini:flash-lite",
            },
        )
        expected_tier_map = ResearchConfig._DEFAULT_TIER_ROLE_ASSIGNMENTS

        for role, tier_name in expected_tier_map.items():
            provider, model = config.resolve_model_for_role(role)
            tier_spec = config.model_tier_config.resolve_tier(tier_name)
            assert tier_spec is not None, f"Tier {tier_name} not found for role {role}"
            # Verify the role resolved through its tier
            assert provider is not None, f"Provider is None for role {role}"
            assert model is not None, f"Model is None for role {role}"

    def test_partial_tier_config(self) -> None:
        """Only 1-2 tiers set — uncovered roles fall through to legacy behavior."""
        config = _make_config(
            default_provider="gemini",
            _tier_config={
                "frontier": "[cli]claude:opus",
                # No standard or efficient tier
            },
        )
        # Frontier roles get the tier
        provider, model = config.resolve_model_for_role("research")
        assert provider == "claude"
        assert model == "opus"

        # Standard role (reflection) has no tier → falls through to default
        provider, model = config.resolve_model_for_role("reflection")
        assert provider == "gemini"
        assert model is None

        # Efficient role (summarization) has no tier → falls through to default
        provider, model = config.resolve_model_for_role("summarization")
        assert provider == "gemini"
        assert model is None

    def test_per_role_provider_with_tier_model(self) -> None:
        """Per-role provider is set but model is not — tier fills in model."""
        config = _make_config(
            default_provider="gemini",
            deep_research_research_provider="openai",
            # No deep_research_research_model
            _tier_config={
                "frontier": {
                    "provider": "[cli]claude:opus",
                    "model": "claude-opus-4-6",
                },
            },
        )
        provider, model = config.resolve_model_for_role("research")
        # Provider from per-role config
        assert provider == "openai"
        # Model from tier (since no per-role model set)
        assert model == "claude-opus-4-6"

    def test_backward_compat_no_tiers(self) -> None:
        """No tiers = identical to existing behavior (regression check)."""
        config_no_tiers = _make_config(
            default_provider="gemini",
            deep_research_research_provider="[cli]claude:opus",
        )
        config_empty_tiers = _make_config(
            default_provider="gemini",
            deep_research_research_provider="[cli]claude:opus",
            _tier_config=None,
        )
        for role in ResearchConfig._ROLE_RESOLUTION_CHAIN:
            assert (
                config_no_tiers.resolve_model_for_role(role)
                == config_empty_tiers.resolve_model_for_role(role)
            )


# =============================================================================
# Tests: get_summarization_model / get_compression_model
# =============================================================================


class TestTierAwareHelpers:
    """Test that get_summarization_model and get_compression_model are tier-aware."""

    def test_get_summarization_model_uses_tier(self) -> None:
        """get_summarization_model returns tier-based model."""
        config = _make_config(
            default_provider="gemini",
            _tier_config={
                "efficient": "[cli]claude:haiku",
            },
        )
        assert config.get_summarization_model() == "haiku"

    def test_get_compression_model_uses_tier(self) -> None:
        """get_compression_model returns tier-based model."""
        config = _make_config(
            default_provider="gemini",
            _tier_config={
                "efficient": {
                    "provider": "[cli]gemini:flash",
                    "model": "gemini-2.5-flash-lite",
                },
            },
        )
        assert config.get_compression_model() == "gemini-2.5-flash-lite"

    def test_get_summarization_model_explicit_override_beats_tier(self) -> None:
        """Explicit summarization model beats tier."""
        config = _make_config(
            default_provider="gemini",
            deep_research_summarization_model="custom-model",
            _tier_config={
                "efficient": "[cli]claude:haiku",
            },
        )
        assert config.get_summarization_model() == "custom-model"

    def test_get_compression_model_no_tiers(self) -> None:
        """get_compression_model without tiers returns None (no default)."""
        config = _make_config(default_provider="gemini")
        assert config.get_compression_model() is None


# =============================================================================
# Tests: Validation warnings
# =============================================================================


class TestTierValidation:
    """Test validation warnings for invalid tier/role names."""

    def test_invalid_tier_name_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Unknown tier name produces a warning, not an error."""
        with caplog.at_level(logging.WARNING):
            _make_config(
                _tier_config={
                    "ultra": "[cli]gemini:pro",  # Not a valid tier name
                },
            )
        assert "Unknown model tier 'ultra'" in caplog.text

    def test_invalid_role_in_assignments_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Unknown role in role_assignments produces a warning, not an error."""
        with caplog.at_level(logging.WARNING):
            _make_config(
                _tier_config={
                    "frontier": "[cli]gemini:pro",
                    "role_assignments": {
                        "nonexistent_role": "frontier",
                    },
                },
            )
        assert "Unknown role 'nonexistent_role'" in caplog.text

    def test_role_assigned_to_undefined_tier_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Role assigned to a tier that isn't configured produces a warning."""
        with caplog.at_level(logging.WARNING):
            _make_config(
                _tier_config={
                    "frontier": "[cli]gemini:pro",
                    "role_assignments": {
                        "research": "standard",  # standard tier is not defined
                    },
                },
            )
        assert "assigned to tier 'standard' which is not configured" in caplog.text

    def test_valid_config_no_warnings(self, caplog: pytest.LogCaptureFixture) -> None:
        """Valid tier config produces no warnings."""
        with caplog.at_level(logging.WARNING):
            _make_config(
                _tier_config={
                    "frontier": "[cli]gemini:pro",
                    "standard": "[cli]gemini:flash",
                    "efficient": "[cli]gemini:flash",
                },
            )
        # Filter to only model tier warnings
        tier_warnings = [r for r in caplog.records if "tier" in r.message.lower()]
        assert len(tier_warnings) == 0


# =============================================================================
# Tests: from_toml_dict parsing
# =============================================================================


class TestFromTomlDict:
    """Test from_toml_dict parsing of model_tiers key."""

    def test_model_tiers_key_no_unknown_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """model_tiers key doesn't trigger unknown-key warning."""
        with caplog.at_level(logging.WARNING):
            config = ResearchConfig.from_toml_dict({
                "model_tiers": {
                    "frontier": "[cli]gemini:pro",
                },
            })
        # Should NOT have "Unknown config key 'model_tiers'" warning
        unknown_warnings = [r for r in caplog.records if "Unknown config key" in r.message]
        assert len(unknown_warnings) == 0
        assert config.model_tier_config.enabled is True

    def test_model_tiers_parsed_from_toml(self) -> None:
        """model_tiers section is correctly parsed via from_toml_dict."""
        config = ResearchConfig.from_toml_dict({
            "model_tiers": {
                "frontier": "[cli]claude:opus",
                "standard": "[cli]gemini:flash",
                "efficient": "[cli]gemini:flash",
                "role_assignments": {
                    "summarization": "standard",
                },
            },
        })
        tier_cfg = config.model_tier_config
        assert tier_cfg.enabled is True
        assert tier_cfg.tiers["frontier"] == "[cli]claude:opus"
        assert tier_cfg.role_assignments == {"summarization": "standard"}

    def test_no_model_tiers_in_toml(self) -> None:
        """No model_tiers key — config defaults to disabled."""
        config = ResearchConfig.from_toml_dict({})
        assert config.model_tier_config.enabled is False
