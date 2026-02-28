"""Tests for layered config deep-merge behaviour.

Verifies that when multiple TOML config files are loaded, sections built
via ``from_toml_dict()`` are deep-merged rather than replaced wholesale.
"""

from __future__ import annotations

import pytest

from foundry_mcp.config.loader import _deep_merge_dicts


# ---------------------------------------------------------------------------
# _deep_merge_dicts unit tests
# ---------------------------------------------------------------------------


class TestDeepMergeDicts:
    """Unit tests for the _deep_merge_dicts helper."""

    def test_flat_override(self) -> None:
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge_dicts(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self) -> None:
        base = {"top": {"a": 1, "b": 2}}
        override = {"top": {"b": 3, "c": 4}}
        result = _deep_merge_dicts(base, override)
        assert result == {"top": {"a": 1, "b": 3, "c": 4}}

    def test_list_replaces_not_merges(self) -> None:
        """Lists are replaced wholesale — no element-level merge."""
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}
        result = _deep_merge_dicts(base, override)
        assert result == {"items": [4, 5]}

    def test_base_unmodified(self) -> None:
        """Original dicts are not mutated."""
        base = {"nested": {"x": 1}}
        override = {"nested": {"y": 2}}
        _deep_merge_dicts(base, override)
        assert base == {"nested": {"x": 1}}

    def test_empty_base(self) -> None:
        result = _deep_merge_dicts({}, {"a": 1})
        assert result == {"a": 1}

    def test_empty_override(self) -> None:
        result = _deep_merge_dicts({"a": 1}, {})
        assert result == {"a": 1}

    def test_deeply_nested(self) -> None:
        base = {"l1": {"l2": {"l3": {"a": 1, "b": 2}}}}
        override = {"l1": {"l2": {"l3": {"b": 3}}}}
        result = _deep_merge_dicts(base, override)
        assert result == {"l1": {"l2": {"l3": {"a": 1, "b": 3}}}}


# ---------------------------------------------------------------------------
# Research config layered merge tests
# ---------------------------------------------------------------------------


class TestResearchConfigLayeredMerge:
    """Verify that multiple [research] TOML sections merge correctly."""

    def test_higher_priority_overrides_keys(self) -> None:
        """Higher-priority file overrides specific keys, keeps rest."""
        from foundry_mcp.config.research import ResearchConfig

        # Simulate XDG config (lower priority)
        xdg_data = {
            "default_provider": "[cli]claude:sonnet",
            "enabled": True,
            "deep_research_mode": "technical",
            "deep_research_providers": ["tavily"],
            "model_tiers": {
                "frontier": "[cli]claude:sonnet",
                "standard": "[cli]claude:sonnet",
                "efficient": "[cli]claude:haiku",
            },
        }

        # Simulate home config (higher priority) — overrides only default_provider
        home_data = {
            "default_provider": "[cli]gemini:pro",
        }

        # Deep-merge: home overrides XDG
        merged = _deep_merge_dicts(xdg_data, home_data)

        config = ResearchConfig.from_toml_dict(merged)

        # default_provider overridden by home config
        assert config.default_provider == "[cli]gemini:pro"
        # model_tiers preserved from XDG config
        tier_cfg = config.model_tier_config
        assert tier_cfg.enabled is True
        assert tier_cfg.resolve_tier("efficient") == "[cli]claude:haiku"
        # deep_research_mode preserved from XDG
        assert config.deep_research_mode == "technical"

    def test_nested_model_tiers_merge(self) -> None:
        """model_tiers sub-keys merge across files."""
        from foundry_mcp.config.research import ResearchConfig

        xdg_data = {
            "default_provider": "claude",
            "model_tiers": {
                "frontier": "[cli]claude:opus",
                "standard": "[cli]claude:sonnet",
            },
        }

        # Home adds efficient tier without touching frontier/standard
        home_data = {
            "model_tiers": {
                "efficient": "[cli]claude:haiku",
            },
        }

        merged = _deep_merge_dicts(xdg_data, home_data)
        config = ResearchConfig.from_toml_dict(merged)

        tier_cfg = config.model_tier_config
        assert tier_cfg.enabled is True
        assert tier_cfg.resolve_tier("frontier") == "[cli]claude:opus"
        assert tier_cfg.resolve_tier("standard") == "[cli]claude:sonnet"
        assert tier_cfg.resolve_tier("efficient") == "[cli]claude:haiku"
