"""Tests for Phase 4 — Config & Data Model Bugs.

Covers:
- 4.1: per_provider_rate_limits default consistency between constructors
- 4.2: deep_research_mode validation
- 4.3: Gemini 2.0 model entries in token limits
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foundry_mcp.config.research import ResearchConfig
from foundry_mcp.core.research.workflows.deep_research._helpers import (
    estimate_token_limit_for_model,
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
        config = ResearchConfig.from_toml_dict(
            {"per_provider_rate_limits": {"semantic_scholar": 100, "tavily": 30}}
        )
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
        limits_path = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "foundry_mcp"
            / "config"
            / "model_token_limits.json"
        )
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
