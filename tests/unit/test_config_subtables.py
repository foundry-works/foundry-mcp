"""Tests for TOML sub-table flattening, timeout presets, and fallback chains.

Phases 2+3 of the TOML ergonomics plan: allow nested sub-tables
([research.deep_research], [research.tavily], [research.perplexity],
[research.semantic_scholar]) as an alternative to flat prefixed keys.

Phase 4: Fallback provider chains via [research.fallback_chains] and
[research.phase_fallbacks].

Phase 5: Timeout presets via [research.timeouts] sub-table.

Tests cover:
1. Nested keys parsed correctly for each prefix
2. Flat keys still work (backward compat)
3. Flat key takes priority over nested key for same field
4. Sub-table dicts inside deep_research are skipped (not flattened)
5. Empty sub-table is harmless
6. No unknown-key warnings for sub-table keys
7. Mix of nested and flat keys works
8. Timeout presets (fast, patient, explicit override, unknown, absent)
9. Fallback chains resolve to provider lists, phase fallbacks, overrides
"""

import logging

from foundry_mcp.config.research import ResearchConfig


# ---------------------------------------------------------------------------
# _flatten_sub_table unit tests (static method)
# ---------------------------------------------------------------------------


class TestFlattenSubTable:
    """Direct tests for the _flatten_sub_table static method."""

    def test_basic_flattening(self):
        data = {"deep_research": {"max_iterations": 5, "timeout": 900.0}}
        ResearchConfig._flatten_sub_table(data, "deep_research")
        assert data["deep_research_max_iterations"] == 5
        assert data["deep_research_timeout"] == 900.0
        assert "deep_research" not in data

    def test_flat_key_takes_priority(self):
        data = {
            "deep_research_max_iterations": 10,
            "deep_research": {"max_iterations": 5, "timeout": 900.0},
        }
        ResearchConfig._flatten_sub_table(data, "deep_research")
        assert data["deep_research_max_iterations"] == 10  # flat wins
        assert data["deep_research_timeout"] == 900.0  # new key added

    def test_dict_sub_values_skipped(self):
        data = {
            "deep_research": {
                "max_iterations": 5,
                "model_tiers": {"frontier": "[cli]gemini:pro"},
            },
        }
        ResearchConfig._flatten_sub_table(data, "deep_research")
        assert data["deep_research_max_iterations"] == 5
        assert "deep_research_model_tiers" not in data

    def test_empty_sub_table(self):
        data = {"deep_research": {}}
        ResearchConfig._flatten_sub_table(data, "deep_research")
        assert "deep_research" not in data

    def test_missing_sub_table(self):
        data = {"some_other_key": 42}
        ResearchConfig._flatten_sub_table(data, "deep_research")
        assert data == {"some_other_key": 42}

    def test_non_dict_sub_table_ignored(self):
        data = {"deep_research": "not-a-dict"}
        ResearchConfig._flatten_sub_table(data, "deep_research")
        # Non-dict value is left as-is (not popped, since isinstance check fails
        # after pop — the value was already popped)
        assert "deep_research" not in data

    def test_prefix_applied_correctly(self):
        data = {"tavily": {"search_depth": "advanced", "topic": "news"}}
        ResearchConfig._flatten_sub_table(data, "tavily")
        assert data["tavily_search_depth"] == "advanced"
        assert data["tavily_topic"] == "news"


# ---------------------------------------------------------------------------
# Integration: [research.deep_research] sub-table
# ---------------------------------------------------------------------------


class TestDeepResearchSubTable:
    """Integration tests for [research.deep_research] nested form."""

    def test_nested_deep_research_keys_parsed(self):
        data = {
            "deep_research": {
                "max_iterations": 7,
                "max_sub_queries": 10,
                "follow_links": False,
            },
        }
        config = ResearchConfig.from_toml_dict(data)
        assert config.deep_research_max_iterations == 7
        assert config.deep_research_max_sub_queries == 10
        assert config.deep_research_follow_links is False

    def test_flat_deep_research_still_works(self):
        data = {
            "deep_research_max_iterations": 7,
            "deep_research_max_sub_queries": 10,
        }
        config = ResearchConfig.from_toml_dict(data)
        assert config.deep_research_max_iterations == 7
        assert config.deep_research_max_sub_queries == 10

    def test_flat_beats_nested_for_same_field(self):
        data = {
            "deep_research_max_iterations": 10,
            "deep_research": {"max_iterations": 5},
        }
        config = ResearchConfig.from_toml_dict(data)
        assert config.deep_research_max_iterations == 10

    def test_nested_dict_sub_values_skipped(self):
        """model_tiers inside deep_research sub-table should not flatten."""
        data = {
            "deep_research": {
                "max_iterations": 7,
                "model_tiers": {"frontier": "[cli]gemini:pro"},
            },
        }
        config = ResearchConfig.from_toml_dict(data)
        assert config.deep_research_max_iterations == 7

    def test_empty_deep_research_sub_table(self):
        data = {"deep_research": {}}
        config = ResearchConfig.from_toml_dict(data)
        # All defaults should apply
        assert config.deep_research_max_iterations == 3

    def test_mix_nested_and_flat(self):
        data = {
            "deep_research_max_iterations": 10,
            "deep_research": {
                "max_sub_queries": 8,
                "timeout": 1200.0,
            },
        }
        config = ResearchConfig.from_toml_dict(data)
        assert config.deep_research_max_iterations == 10
        assert config.deep_research_max_sub_queries == 8
        assert config.deep_research_timeout == 1200.0

    def test_no_unknown_key_warning_for_sub_table(self, caplog):
        data = {
            "deep_research": {"max_iterations": 5},
        }
        with caplog.at_level(logging.WARNING):
            ResearchConfig.from_toml_dict(data)
        unknown_warnings = [r for r in caplog.records if "Unknown config key" in r.message]
        # "deep_research" itself should NOT appear as unknown
        assert not any("'deep_research'" in r.message for r in unknown_warnings)


# ---------------------------------------------------------------------------
# Integration: [research.tavily] sub-table
# ---------------------------------------------------------------------------


class TestTavilySubTable:
    """Integration tests for [research.tavily] nested form."""

    def test_nested_tavily_keys_parsed(self):
        data = {
            "tavily": {
                "search_depth": "advanced",
                "topic": "news",
                "include_images": True,
            },
        }
        config = ResearchConfig.from_toml_dict(data)
        assert config.tavily_search_depth == "advanced"
        assert config.tavily_topic == "news"
        assert config.tavily_include_images is True

    def test_flat_tavily_still_works(self):
        data = {"tavily_search_depth": "fast"}
        config = ResearchConfig.from_toml_dict(data)
        assert config.tavily_search_depth == "fast"

    def test_flat_beats_nested_tavily(self):
        data = {
            "tavily_search_depth": "advanced",
            "tavily": {"search_depth": "basic"},
        }
        config = ResearchConfig.from_toml_dict(data)
        assert config.tavily_search_depth == "advanced"

    def test_nested_tavily_chunks_per_source(self):
        data = {"tavily": {"chunks_per_source": 5}}
        config = ResearchConfig.from_toml_dict(data)
        assert config.tavily_chunks_per_source == 5

    def test_no_unknown_key_warning_for_tavily(self, caplog):
        data = {"tavily": {"search_depth": "basic"}}
        with caplog.at_level(logging.WARNING):
            ResearchConfig.from_toml_dict(data)
        unknown_warnings = [r for r in caplog.records if "Unknown config key" in r.message]
        assert not any("'tavily'" in r.message for r in unknown_warnings)


# ---------------------------------------------------------------------------
# Integration: [research.perplexity] sub-table
# ---------------------------------------------------------------------------


class TestPerplexitySubTable:
    """Integration tests for [research.perplexity] nested form."""

    def test_nested_perplexity_keys_parsed(self):
        data = {
            "perplexity": {
                "search_context_size": "high",
                "max_tokens": 80000,
            },
        }
        config = ResearchConfig.from_toml_dict(data)
        assert config.perplexity_search_context_size == "high"
        assert config.perplexity_max_tokens == 80000

    def test_flat_perplexity_still_works(self):
        data = {"perplexity_max_tokens": 25000}
        config = ResearchConfig.from_toml_dict(data)
        assert config.perplexity_max_tokens == 25000

    def test_flat_beats_nested_perplexity(self):
        data = {
            "perplexity_max_tokens": 25000,
            "perplexity": {"max_tokens": 80000},
        }
        config = ResearchConfig.from_toml_dict(data)
        assert config.perplexity_max_tokens == 25000

    def test_no_unknown_key_warning_for_perplexity(self, caplog):
        data = {"perplexity": {"search_context_size": "low"}}
        with caplog.at_level(logging.WARNING):
            ResearchConfig.from_toml_dict(data)
        unknown_warnings = [r for r in caplog.records if "Unknown config key" in r.message]
        assert not any("'perplexity'" in r.message for r in unknown_warnings)


# ---------------------------------------------------------------------------
# Integration: [research.semantic_scholar] sub-table
# ---------------------------------------------------------------------------


class TestSemanticScholarSubTable:
    """Integration tests for [research.semantic_scholar] nested form."""

    def test_nested_semantic_scholar_keys_parsed(self):
        data = {
            "semantic_scholar": {
                "sort_by": "citationCount",
                "sort_order": "asc",
                "use_extended_fields": False,
            },
        }
        config = ResearchConfig.from_toml_dict(data)
        assert config.semantic_scholar_sort_by == "citationCount"
        assert config.semantic_scholar_sort_order == "asc"
        assert config.semantic_scholar_use_extended_fields is False

    def test_flat_semantic_scholar_still_works(self):
        data = {"semantic_scholar_sort_by": "publicationDate"}
        config = ResearchConfig.from_toml_dict(data)
        assert config.semantic_scholar_sort_by == "publicationDate"

    def test_flat_beats_nested_semantic_scholar(self):
        data = {
            "semantic_scholar_sort_by": "publicationDate",
            "semantic_scholar": {"sort_by": "citationCount"},
        }
        config = ResearchConfig.from_toml_dict(data)
        assert config.semantic_scholar_sort_by == "publicationDate"

    def test_no_unknown_key_warning_for_semantic_scholar(self, caplog):
        data = {"semantic_scholar": {"sort_by": "citationCount"}}
        with caplog.at_level(logging.WARNING):
            ResearchConfig.from_toml_dict(data)
        unknown_warnings = [r for r in caplog.records if "Unknown config key" in r.message]
        assert not any("'semantic_scholar'" in r.message for r in unknown_warnings)


# ---------------------------------------------------------------------------
# Cross-cutting edge cases
# ---------------------------------------------------------------------------


class TestSubTableEdgeCases:
    """Edge cases spanning multiple prefixes."""

    def test_all_four_sub_tables_simultaneously(self):
        data = {
            "deep_research": {"max_iterations": 5},
            "tavily": {"search_depth": "advanced"},
            "perplexity": {"max_tokens": 80000},
            "semantic_scholar": {"sort_by": "citationCount"},
        }
        config = ResearchConfig.from_toml_dict(data)
        assert config.deep_research_max_iterations == 5
        assert config.tavily_search_depth == "advanced"
        assert config.perplexity_max_tokens == 80000
        assert config.semantic_scholar_sort_by == "citationCount"

    def test_sub_tables_with_other_flat_keys(self):
        data = {
            "enabled": True,
            "default_provider": "gemini",
            "deep_research": {"max_iterations": 5},
            "tavily": {"search_depth": "advanced"},
        }
        config = ResearchConfig.from_toml_dict(data)
        assert config.enabled is True
        assert config.default_provider == "gemini"
        assert config.deep_research_max_iterations == 5
        assert config.tavily_search_depth == "advanced"

    def test_empty_dict_for_all_sub_tables(self):
        data = {
            "deep_research": {},
            "tavily": {},
            "perplexity": {},
            "semantic_scholar": {},
        }
        config = ResearchConfig.from_toml_dict(data)
        # All defaults should apply
        assert config.deep_research_max_iterations == 3
        assert config.tavily_search_depth == "basic"
        assert config.perplexity_max_tokens == 50000
        assert config.semantic_scholar_sort_order == "desc"


# ---------------------------------------------------------------------------
# Phase 5: Timeout Presets via [research.timeouts]
# ---------------------------------------------------------------------------

# Baseline defaults for timeout fields (must match ResearchConfig defaults).
_TIMEOUT_DEFAULTS = {
    "default_timeout": 360.0,
    "deep_research_timeout": 2400.0,
    "deep_research_planning_timeout": 360.0,
    "deep_research_synthesis_timeout": 600.0,
    "deep_research_reflection_timeout": 60.0,
    "deep_research_evaluation_timeout": 360.0,
    "deep_research_supervision_wall_clock_timeout": 1800.0,
    "deep_research_summarization_timeout": 60.0,
    "deep_research_digest_timeout": 120.0,
    "summarization_timeout": 60.0,
}


class TestTimeoutPresets:
    """Tests for [research.timeouts] preset multipliers."""

    def test_fast_preset_halves_defaults(self):
        config = ResearchConfig.from_toml_dict({"timeouts": {"preset": "fast"}})
        for field, base in _TIMEOUT_DEFAULTS.items():
            actual = getattr(config, field)
            assert actual == base * 0.5, f"{field}: expected {base * 0.5}, got {actual}"

    def test_patient_preset_triples_defaults(self):
        config = ResearchConfig.from_toml_dict({"timeouts": {"preset": "patient"}})
        for field, base in _TIMEOUT_DEFAULTS.items():
            actual = getattr(config, field)
            assert actual == base * 3.0, f"{field}: expected {base * 3.0}, got {actual}"

    def test_relaxed_preset(self):
        config = ResearchConfig.from_toml_dict({"timeouts": {"preset": "relaxed"}})
        assert config.default_timeout == 360.0 * 1.5
        assert config.deep_research_synthesis_timeout == 600.0 * 1.5

    def test_default_preset_no_change(self):
        config = ResearchConfig.from_toml_dict({"timeouts": {"preset": "default"}})
        for field, base in _TIMEOUT_DEFAULTS.items():
            assert getattr(config, field) == base

    def test_explicit_flat_timeout_beats_preset(self):
        """A flat key in [research] overrides the preset multiplier."""
        config = ResearchConfig.from_toml_dict({
            "timeouts": {"preset": "patient"},
            "deep_research_synthesis_timeout": 999.0,
        })
        # Explicitly set → not multiplied
        assert config.deep_research_synthesis_timeout == 999.0
        # Other timeouts still get the 3x multiplier
        assert config.default_timeout == 360.0 * 3.0

    def test_individual_override_in_timeouts_section(self):
        """Keys inside [research.timeouts] override both preset and flat."""
        config = ResearchConfig.from_toml_dict({
            "timeouts": {
                "preset": "patient",
                "deep_research_synthesis_timeout": 777.0,
            },
        })
        assert config.deep_research_synthesis_timeout == 777.0
        # Other timeouts still get the 3x multiplier
        assert config.default_timeout == 360.0 * 3.0

    def test_unknown_preset_warns_and_uses_default(self, caplog):
        with caplog.at_level(logging.WARNING):
            config = ResearchConfig.from_toml_dict({"timeouts": {"preset": "turbo"}})
        assert any("Unknown timeout preset" in r.message for r in caplog.records)
        # Falls back to 1.0x (no change)
        for field, base in _TIMEOUT_DEFAULTS.items():
            assert getattr(config, field) == base

    def test_no_timeouts_section_no_change(self):
        config = ResearchConfig.from_toml_dict({})
        for field, base in _TIMEOUT_DEFAULTS.items():
            assert getattr(config, field) == base

    def test_empty_timeouts_section_no_change(self):
        """Empty [research.timeouts] = implicit preset 'default' (1.0x)."""
        config = ResearchConfig.from_toml_dict({"timeouts": {}})
        for field, base in _TIMEOUT_DEFAULTS.items():
            assert getattr(config, field) == base

    def test_no_unknown_key_warning_for_timeouts(self, caplog):
        with caplog.at_level(logging.WARNING):
            ResearchConfig.from_toml_dict({"timeouts": {"preset": "fast"}})
        unknown_warnings = [r for r in caplog.records if "Unknown config key" in r.message]
        assert not any("'timeouts'" in r.message for r in unknown_warnings)

    def test_timeouts_with_sub_tables(self):
        """Timeout presets work alongside sub-table flattening."""
        config = ResearchConfig.from_toml_dict({
            "timeouts": {"preset": "relaxed"},
            "deep_research": {"max_iterations": 5},
        })
        assert config.deep_research_max_iterations == 5
        assert config.default_timeout == 360.0 * 1.5


# ---------------------------------------------------------------------------
# Phase 4: Fallback Provider Chains
# ---------------------------------------------------------------------------


class TestFallbackChains:
    """Tests for [research.fallback_chains] and [research.phase_fallbacks]."""

    def test_named_chain_resolves_to_list(self):
        """A phase fallback referencing a named chain populates the providers field."""
        config = ResearchConfig.from_toml_dict({
            "fallback_chains": {
                "strong": ["[cli]gemini:pro", "[cli]claude:opus"],
            },
            "phase_fallbacks": {
                "planning": "strong",
            },
        })
        assert config.deep_research_planning_providers == [
            "[cli]gemini:pro",
            "[cli]claude:opus",
        ]

    def test_multiple_phases_use_different_chains(self):
        config = ResearchConfig.from_toml_dict({
            "fallback_chains": {
                "strong": ["[cli]gemini:pro", "[cli]claude:opus"],
                "fast": ["[cli]gemini:flash", "[cli]codex:mini"],
            },
            "phase_fallbacks": {
                "planning": "fast",
                "synthesis": "strong",
            },
        })
        assert config.deep_research_planning_providers == [
            "[cli]gemini:flash",
            "[cli]codex:mini",
        ]
        assert config.deep_research_synthesis_providers == [
            "[cli]gemini:pro",
            "[cli]claude:opus",
        ]

    def test_explicit_per_phase_list_overrides_chain(self):
        """Explicit deep_research_planning_providers beats chain assignment."""
        config = ResearchConfig.from_toml_dict({
            "fallback_chains": {
                "strong": ["[cli]gemini:pro", "[cli]claude:opus"],
            },
            "phase_fallbacks": {
                "planning": "strong",
            },
            "deep_research_planning_providers": ["[cli]codex:gpt-5"],
        })
        assert config.deep_research_planning_providers == ["[cli]codex:gpt-5"]

    def test_chain_with_aliases_expands(self):
        """Chain values are expanded through aliases."""
        aliases = {
            "pro": "[cli]gemini:pro",
            "opus": "[cli]claude:opus",
        }
        config = ResearchConfig.from_toml_dict(
            {
                "fallback_chains": {
                    "strong": ["pro", "opus"],
                },
                "phase_fallbacks": {
                    "synthesis": "strong",
                },
            },
            aliases=aliases,
        )
        assert config.deep_research_synthesis_providers == [
            "[cli]gemini:pro",
            "[cli]claude:opus",
        ]

    def test_unknown_chain_name_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            config = ResearchConfig.from_toml_dict({
                "fallback_chains": {
                    "strong": ["[cli]gemini:pro"],
                },
                "phase_fallbacks": {
                    "planning": "nonexistent",
                },
            })
        assert any("Unknown chain" in r.message for r in caplog.records)
        # Field stays at default (empty list)
        assert config.deep_research_planning_providers == []

    def test_unknown_phase_name_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            ResearchConfig.from_toml_dict({
                "fallback_chains": {
                    "strong": ["[cli]gemini:pro"],
                },
                "phase_fallbacks": {
                    "analysis": "strong",  # deprecated phase
                },
            })
        assert any("Unknown phase" in r.message for r in caplog.records)

    def test_no_chains_no_change(self):
        """Absent fallback_chains section = no behavior change."""
        config = ResearchConfig.from_toml_dict({})
        assert config.deep_research_planning_providers == []
        assert config.deep_research_synthesis_providers == []

    def test_empty_chains_section_no_change(self):
        config = ResearchConfig.from_toml_dict({
            "fallback_chains": {},
            "phase_fallbacks": {},
        })
        assert config.deep_research_planning_providers == []
        assert config.deep_research_synthesis_providers == []

    def test_non_list_chain_value_warns(self, caplog):
        """A chain value that is not a list should warn and be ignored."""
        with caplog.at_level(logging.WARNING):
            config = ResearchConfig.from_toml_dict({
                "fallback_chains": {
                    "bad": "not-a-list",
                },
                "phase_fallbacks": {
                    "planning": "bad",
                },
            })
        assert any("must be a list" in r.message for r in caplog.records)
        assert config.deep_research_planning_providers == []

    def test_phase_fallbacks_without_chains_warns(self, caplog):
        """phase_fallbacks without fallback_chains should warn."""
        with caplog.at_level(logging.WARNING):
            ResearchConfig.from_toml_dict({
                "phase_fallbacks": {
                    "planning": "strong",
                },
            })
        assert any("requires [research.fallback_chains]" in r.message for r in caplog.records)

    def test_no_unknown_key_warning_for_chain_keys(self, caplog):
        with caplog.at_level(logging.WARNING):
            ResearchConfig.from_toml_dict({
                "fallback_chains": {"strong": ["[cli]gemini:pro"]},
                "phase_fallbacks": {"planning": "strong"},
            })
        unknown_warnings = [r for r in caplog.records if "Unknown config key" in r.message]
        assert not any("'fallback_chains'" in r.message for r in unknown_warnings)
        assert not any("'phase_fallbacks'" in r.message for r in unknown_warnings)
