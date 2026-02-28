"""Tests for provider alias expansion (Phase 1 of TOML config ergonomics).

Tests cover:
1. Alias expands in default_provider
2. Alias expands in list fields (consensus_providers, deep_research_planning_providers)
3. Non-alias strings pass through unchanged
4. Alias in consultation priority list
5. Alias in model_tiers tier values (string and table form)
6. Non-string alias value warns (loader-level)
7. Absent [providers] section = no change
8. No recursive expansion (alias → alias is literal)
"""

from foundry_mcp.config.parsing import _expand_alias, _expand_alias_list
from foundry_mcp.config.research import ResearchConfig
from foundry_mcp.core.llm_config.consultation import ConsultationConfig


SAMPLE_ALIASES = {
    "pro": "[cli]gemini:pro",
    "fast": "[cli]gemini:flash",
    "opus": "[cli]claude:opus",
    "codex": "[cli]codex:gpt-5.2-codex",
}


# ---------------------------------------------------------------------------
# Unit tests for _expand_alias / _expand_alias_list helpers
# ---------------------------------------------------------------------------


class TestExpandAliasHelpers:
    """Unit tests for the alias expansion helpers."""

    def test_expand_alias_match(self):
        assert _expand_alias("pro", SAMPLE_ALIASES) == "[cli]gemini:pro"

    def test_expand_alias_no_match(self):
        assert _expand_alias("[cli]claude:opus", SAMPLE_ALIASES) == "[cli]claude:opus"

    def test_expand_alias_empty_aliases(self):
        assert _expand_alias("pro", {}) == "pro"

    def test_expand_alias_list(self):
        result = _expand_alias_list(["pro", "opus", "literal"], SAMPLE_ALIASES)
        assert result == ["[cli]gemini:pro", "[cli]claude:opus", "literal"]

    def test_expand_alias_list_empty(self):
        assert _expand_alias_list([], SAMPLE_ALIASES) == []

    def test_no_recursive_expansion(self):
        """An alias whose value matches another alias key is NOT re-expanded."""
        aliases = {"a": "b", "b": "[cli]gemini:pro"}
        assert _expand_alias("a", aliases) == "b"  # not "[cli]gemini:pro"


# ---------------------------------------------------------------------------
# ResearchConfig alias expansion
# ---------------------------------------------------------------------------


class TestResearchConfigAliases:
    """Integration tests for alias expansion in ResearchConfig.from_toml_dict."""

    def test_alias_expands_default_provider(self):
        config = ResearchConfig.from_toml_dict(
            {"default_provider": "pro"},
            aliases=SAMPLE_ALIASES,
        )
        assert config.default_provider == "[cli]gemini:pro"

    def test_alias_expands_consensus_providers(self):
        config = ResearchConfig.from_toml_dict(
            {"consensus_providers": ["pro", "codex", "opus"]},
            aliases=SAMPLE_ALIASES,
        )
        assert config.consensus_providers == [
            "[cli]gemini:pro",
            "[cli]codex:gpt-5.2-codex",
            "[cli]claude:opus",
        ]

    def test_alias_expands_deep_research_planning_providers(self):
        config = ResearchConfig.from_toml_dict(
            {"deep_research_planning_providers": ["fast", "codex"]},
            aliases=SAMPLE_ALIASES,
        )
        assert config.deep_research_planning_providers == [
            "[cli]gemini:flash",
            "[cli]codex:gpt-5.2-codex",
        ]

    def test_alias_expands_planning_provider(self):
        config = ResearchConfig.from_toml_dict(
            {"deep_research_planning_provider": "fast"},
            aliases=SAMPLE_ALIASES,
        )
        assert config.deep_research_planning_provider == "[cli]gemini:flash"

    def test_alias_expands_synthesis_provider(self):
        config = ResearchConfig.from_toml_dict(
            {"deep_research_synthesis_provider": "pro"},
            aliases=SAMPLE_ALIASES,
        )
        assert config.deep_research_synthesis_provider == "[cli]gemini:pro"

    def test_non_alias_strings_pass_through(self):
        config = ResearchConfig.from_toml_dict(
            {"default_provider": "[cli]claude:opus"},
            aliases=SAMPLE_ALIASES,
        )
        assert config.default_provider == "[cli]claude:opus"

    def test_absent_aliases_no_change(self):
        """No aliases param = no expansion."""
        config = ResearchConfig.from_toml_dict(
            {"default_provider": "pro"},
        )
        assert config.default_provider == "pro"

    def test_empty_aliases_no_change(self):
        config = ResearchConfig.from_toml_dict(
            {"default_provider": "pro"},
            aliases={},
        )
        assert config.default_provider == "pro"

    def test_alias_in_model_tiers_string_form(self):
        config = ResearchConfig.from_toml_dict(
            {"model_tiers": {"frontier": "pro", "efficient": "fast"}},
            aliases=SAMPLE_ALIASES,
        )
        tier_cfg = config.model_tier_config
        assert tier_cfg.enabled is True
        assert tier_cfg.tiers["frontier"] == "[cli]gemini:pro"
        assert tier_cfg.tiers["efficient"] == "[cli]gemini:flash"

    def test_alias_in_model_tiers_table_form(self):
        config = ResearchConfig.from_toml_dict(
            {
                "model_tiers": {
                    "frontier": {"provider": "pro", "model": "gemini-2.5-pro"},
                },
            },
            aliases=SAMPLE_ALIASES,
        )
        tier_cfg = config.model_tier_config
        assert tier_cfg.tiers["frontier"] == "[cli]gemini:pro"

    def test_alias_with_nested_sub_table(self):
        """Aliases work with nested [research.deep_research] sub-tables."""
        config = ResearchConfig.from_toml_dict(
            {
                "deep_research": {
                    "planning_provider": "fast",
                    "synthesis_provider": "pro",
                },
            },
            aliases=SAMPLE_ALIASES,
        )
        assert config.deep_research_planning_provider == "[cli]gemini:flash"
        assert config.deep_research_synthesis_provider == "[cli]gemini:pro"

    def test_no_recursive_alias_expansion(self):
        """Alias whose value matches another alias key is literal."""
        aliases = {"tier1": "pro", "pro": "[cli]gemini:pro"}
        config = ResearchConfig.from_toml_dict(
            {"default_provider": "tier1"},
            aliases=aliases,
        )
        # "tier1" → "pro" (not recursively → "[cli]gemini:pro")
        assert config.default_provider == "pro"


# ---------------------------------------------------------------------------
# ConsultationConfig alias expansion
# ---------------------------------------------------------------------------


class TestConsultationConfigAliases:
    """Integration tests for alias expansion in ConsultationConfig.from_dict."""

    def test_alias_expands_priority(self):
        config = ConsultationConfig.from_dict(
            {"priority": ["codex", "pro", "opus"]},
            aliases=SAMPLE_ALIASES,
        )
        assert config.priority == [
            "[cli]codex:gpt-5.2-codex",
            "[cli]gemini:pro",
            "[cli]claude:opus",
        ]

    def test_alias_expands_overrides_keys(self):
        config = ConsultationConfig.from_dict(
            {"overrides": {"pro": {"timeout": 600}}},
            aliases=SAMPLE_ALIASES,
        )
        assert "[cli]gemini:pro" in config.overrides
        assert config.overrides["[cli]gemini:pro"]["timeout"] == 600

    def test_absent_aliases_no_change(self):
        config = ConsultationConfig.from_dict(
            {"priority": ["pro"]},
        )
        assert config.priority == ["pro"]

    def test_non_alias_strings_pass_through(self):
        config = ConsultationConfig.from_dict(
            {"priority": ["[cli]claude:opus"]},
            aliases=SAMPLE_ALIASES,
        )
        assert config.priority == ["[cli]claude:opus"]
