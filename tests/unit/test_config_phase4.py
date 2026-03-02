"""Tests for Phase 4 — Config & Data Model Bugs.

Covers:
- 4.1: per_provider_rate_limits default consistency between constructors
- 4.2: deep_research_mode validation
- 4.3: Gemini 2.0 model entries in token limits
- 4.4: Unknown TOML key warnings
- 4.5: deep_research_enable_planning_critique config field
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foundry_mcp.config.research import ResearchConfig
from foundry_mcp.core.research.workflows.deep_research._helpers import (
    estimate_token_limit_for_model,
    extract_json,
)

# ---------------------------------------------------------------------------
# 4.1 — per_provider_rate_limits default consistency
# ---------------------------------------------------------------------------


class TestPerProviderRateLimitsConsistency:
    """Both constructors must produce identical semantic_scholar rate limits."""

    def test_direct_construction_semantic_scholar_is_20(self):
        """ResearchConfig() default has semantic_scholar == 20."""
        config = ResearchConfig()
        assert config.per_provider_rate_limits["semantic_scholar"] == 20

    def test_from_toml_dict_empty_semantic_scholar_is_20(self):
        """ResearchConfig.from_toml_dict({}) default has semantic_scholar == 20."""
        config = ResearchConfig.from_toml_dict({})
        assert config.per_provider_rate_limits["semantic_scholar"] == 20

    def test_both_constructors_produce_same_rate_limits(self):
        """Direct and from_toml_dict produce identical per_provider_rate_limits."""
        direct = ResearchConfig()
        from_toml = ResearchConfig.from_toml_dict({})
        assert direct.per_provider_rate_limits == from_toml.per_provider_rate_limits

    def test_toml_override_respected(self):
        """User-supplied rate limit overrides the default."""
        config = ResearchConfig.from_toml_dict({"per_provider_rate_limits": {"semantic_scholar": 100, "tavily": 30}})
        assert config.per_provider_rate_limits["semantic_scholar"] == 100
        assert config.per_provider_rate_limits["tavily"] == 30


# ---------------------------------------------------------------------------
# 4.2 — deep_research_mode validation
# ---------------------------------------------------------------------------


class TestDeepResearchModeValidation:
    """deep_research_mode must be one of general/academic/technical."""

    @pytest.mark.parametrize("mode", ["general", "academic", "technical"])
    def test_valid_modes_accepted(self, mode: str):
        config = ResearchConfig(deep_research_mode=mode)
        assert config.deep_research_mode == mode

    def test_invalid_mode_raises_valueerror(self):
        with pytest.raises(ValueError, match="deep_research_mode must be one of"):
            ResearchConfig(deep_research_mode="invalid")

    def test_empty_string_raises_valueerror(self):
        with pytest.raises(ValueError, match="deep_research_mode must be one of"):
            ResearchConfig(deep_research_mode="")

    def test_from_toml_dict_invalid_mode_raises(self):
        """from_toml_dict also triggers mode validation via __post_init__."""
        with pytest.raises(ValueError, match="deep_research_mode must be one of"):
            ResearchConfig.from_toml_dict({"deep_research_mode": "fast"})

    def test_default_mode_is_general(self):
        config = ResearchConfig()
        assert config.deep_research_mode == "general"


# ---------------------------------------------------------------------------
# 4.3 — Gemini 2.0 model entries
# ---------------------------------------------------------------------------


class TestGemini20ModelEntries:
    """Gemini 2.0 models must resolve in both JSON and fallback registries."""

    @pytest.fixture()
    def json_limits(self) -> dict[str, int]:
        limits_path = Path(__file__).resolve().parents[2] / "src" / "foundry_mcp" / "config" / "model_token_limits.json"
        with open(limits_path) as f:
            return json.load(f)["limits"]

    @pytest.fixture()
    def fallback_limits(self) -> dict[str, int]:
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            _FALLBACK_MODEL_TOKEN_LIMITS,
        )

        return _FALLBACK_MODEL_TOKEN_LIMITS

    @pytest.mark.parametrize("model", ["gemini-2.0-pro", "gemini-2.0-flash"])
    def test_json_contains_gemini_20(self, json_limits: dict[str, int], model: str):
        assert model in json_limits
        assert json_limits[model] == 1048576

    @pytest.mark.parametrize("model", ["gemini-2.0-pro", "gemini-2.0-flash"])
    def test_fallback_contains_gemini_20(self, fallback_limits: dict[str, int], model: str):
        assert model in fallback_limits
        assert fallback_limits[model] == 1_048_576

    @pytest.mark.parametrize("model", ["gemini-2.0-pro", "gemini-2.0-flash"])
    def test_estimate_resolves_gemini_20(self, json_limits: dict[str, int], model: str):
        """estimate_token_limit_for_model finds Gemini 2.0 entries."""
        result = estimate_token_limit_for_model(model, json_limits)
        assert result == 1048576

    def test_gemini_20_flash_prefers_specific_over_general(self, json_limits: dict[str, int]):
        """gemini-2.0-flash should match before any broader 'gemini-2' pattern."""
        # Since ordering is longest-first after loading, the specific entry should win
        result = estimate_token_limit_for_model("gemini-2.0-flash-001", json_limits)
        assert result is not None
        assert result == 1048576


# ===========================================================================
# 4.4 Unknown TOML key warnings
# ===========================================================================


class TestUnknownTomlKeyWarning:
    """from_toml_dict warns on unknown keys (catches typos)."""

    def test_typo_key_logs_warning(self, caplog):
        """A typo'd key produces a warning log."""
        import logging

        with caplog.at_level(logging.WARNING, logger="foundry_mcp.config.research"):
            ResearchConfig.from_toml_dict({"deep_research_max_interations": 5})
        assert any("deep_research_max_interations" in r.message for r in caplog.records)

    def test_known_key_no_warning(self, caplog):
        """Valid keys produce no unknown-key warnings."""
        import logging

        with caplog.at_level(logging.WARNING, logger="foundry_mcp.config.research"):
            ResearchConfig.from_toml_dict({"deep_research_max_iterations": 5})
        assert not any("Unknown config key" in r.message for r in caplog.records)


# ===========================================================================
# 4.5 deep_research_enable_planning_critique config field
# ===========================================================================


class TestPlanningCritiqueConfig:
    """deep_research_enable_planning_critique is a real config field."""

    def test_default_is_true(self):
        config = ResearchConfig()
        assert config.deep_research_enable_planning_critique is True

    def test_from_toml_dict_override(self):
        config = ResearchConfig.from_toml_dict({"deep_research_enable_planning_critique": False})
        assert config.deep_research_enable_planning_critique is False


# ===========================================================================
# 6b — extract_json() backslash handling
# ===========================================================================


class TestExtractJsonBackslashHandling:
    """Backslashes outside JSON strings must not break brace matching."""

    def test_backslash_outside_string_does_not_break_parsing(self):
        """A backslash outside a JSON string should be treated as a regular char."""
        # Backslash before the opening brace
        text = r'Some text \ {"key": "value"}'
        result = extract_json(text)
        assert result == '{"key": "value"}'

    def test_backslash_between_json_objects(self):
        """Backslashes between content and JSON should not confuse the parser."""
        text = r'prefix \ text {"a": 1}'
        result = extract_json(text)
        assert result == '{"a": 1}'

    def test_backslash_in_string_still_escapes(self):
        """Backslash inside a JSON string should still escape the next char."""
        text = r'{"msg": "hello \"world\""}'
        result = extract_json(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["msg"] == 'hello "world"'

    def test_nested_braces_with_external_backslash(self):
        """Nested braces with backslash outside string."""
        text = r'\ {"outer": {"inner": 42}}'
        result = extract_json(text)
        assert result == '{"outer": {"inner": 42}}'

    def test_backslash_before_quote_outside_string(self):
        r"""Backslash before a quote outside a string boundary.

        In ``\"{"key": 1}`` the backslash is outside any JSON string,
        so it must NOT escape the quote — the quote starts a new string.
        """
        text = '\\{"key": 1}'
        result = extract_json(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["key"] == 1


# ===========================================================================
# 6c — JSON / fallback token limits sync
# ===========================================================================


class TestTokenLimitsSyncJsonFallback:
    """model_token_limits.json and _FALLBACK_MODEL_TOKEN_LIMITS must stay in sync."""

    @pytest.fixture()
    def json_limits(self) -> dict[str, int]:
        limits_path = Path(__file__).resolve().parents[2] / "src" / "foundry_mcp" / "config" / "model_token_limits.json"
        with open(limits_path) as f:
            return json.load(f)["limits"]

    @pytest.fixture()
    def fallback_limits(self) -> dict[str, int]:
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            _FALLBACK_MODEL_TOKEN_LIMITS,
        )

        return _FALLBACK_MODEL_TOKEN_LIMITS

    def test_same_keys(self, json_limits: dict[str, int], fallback_limits: dict[str, int]):
        """JSON and fallback registries must have identical key sets."""
        json_keys = set(json_limits.keys())
        fallback_keys = set(fallback_limits.keys())
        missing_from_json = fallback_keys - json_keys
        missing_from_fallback = json_keys - fallback_keys
        assert not missing_from_json, f"Keys in _FALLBACK_MODEL_TOKEN_LIMITS but not in JSON: {missing_from_json}"
        assert not missing_from_fallback, (
            f"Keys in JSON but not in _FALLBACK_MODEL_TOKEN_LIMITS: {missing_from_fallback}"
        )

    def test_same_values(self, json_limits: dict[str, int], fallback_limits: dict[str, int]):
        """JSON and fallback registries must have identical values for shared keys."""
        mismatches: list[str] = []
        for key in set(json_limits.keys()) & set(fallback_limits.keys()):
            if json_limits[key] != fallback_limits[key]:
                mismatches.append(f"  {key}: JSON={json_limits[key]} vs fallback={fallback_limits[key]}")
        assert not mismatches, "Value mismatches between JSON and fallback:\n" + "\n".join(mismatches)


# ===========================================================================
# FIX-1 Item 1.1 — Citation influence threshold ordering validation
# ===========================================================================


class TestCitationInfluenceThresholdValidation:
    """Citation influence thresholds must satisfy low < medium < high."""

    def test_default_thresholds_pass(self):
        """Default thresholds (5, 20, 100) pass validation."""
        config = ResearchConfig()
        assert config.deep_research_influence_low_citation_threshold == 5
        assert config.deep_research_influence_medium_citation_threshold == 20
        assert config.deep_research_influence_high_citation_threshold == 100

    def test_custom_valid_thresholds_pass(self):
        """Custom thresholds that satisfy low < medium < high pass."""
        config = ResearchConfig(
            deep_research_influence_low_citation_threshold=1,
            deep_research_influence_medium_citation_threshold=10,
            deep_research_influence_high_citation_threshold=50,
        )
        assert config.deep_research_influence_low_citation_threshold == 1

    def test_inverted_thresholds_raise(self):
        """Inverted thresholds (100, 20, 5) raise ValueError."""
        with pytest.raises(ValueError, match="low < medium < high"):
            ResearchConfig(
                deep_research_influence_low_citation_threshold=100,
                deep_research_influence_medium_citation_threshold=20,
                deep_research_influence_high_citation_threshold=5,
            )

    def test_equal_thresholds_raise(self):
        """Equal thresholds (20, 20, 20) raise ValueError."""
        with pytest.raises(ValueError, match="low < medium < high"):
            ResearchConfig(
                deep_research_influence_low_citation_threshold=20,
                deep_research_influence_medium_citation_threshold=20,
                deep_research_influence_high_citation_threshold=20,
            )

    def test_from_toml_dict_inverted_raises(self):
        """from_toml_dict also triggers threshold validation."""
        with pytest.raises(ValueError, match="low < medium < high"):
            ResearchConfig.from_toml_dict({
                "deep_research_influence_low_citation_threshold": 50,
                "deep_research_influence_medium_citation_threshold": 20,
                "deep_research_influence_high_citation_threshold": 10,
            })
