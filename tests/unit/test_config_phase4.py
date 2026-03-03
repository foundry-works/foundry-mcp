"""Tests for Phase 4 — Config & Data Model Bugs.

Covers:
- 4.1: per_provider_rate_limits default consistency between constructors
- 4.2: deep_research_mode validation
- 4.3: Gemini 2.0 model entries in token limits
- 4.4: Unknown TOML key warnings
- 4.5: deep_research_enable_planning_critique config field
- P0: Claim verification config parsing, validation, and field coverage regression
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
# Citation influence threshold ordering validation
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


# ===========================================================================
# P0 — Claim verification config fields parsed from TOML
# ===========================================================================


class TestClaimVerificationConfigParsing:
    """Claim verification fields must be parsed from TOML into ResearchConfig."""

    def test_defaults_from_direct_construction(self):
        """Direct construction has correct claim verification defaults."""
        config = ResearchConfig()
        assert config.deep_research_claim_verification_enabled is False
        assert config.deep_research_claim_verification_sample_rate == 0.3
        assert config.deep_research_claim_verification_provider is None
        assert config.deep_research_claim_verification_model is None
        assert config.deep_research_claim_verification_timeout == 120.0
        assert config.deep_research_claim_verification_max_claims == 50
        assert config.deep_research_claim_verification_max_concurrent == 10
        assert config.deep_research_claim_verification_max_corrections == 5
        assert config.deep_research_claim_verification_annotate_unsupported is False
        assert config.deep_research_claim_verification_max_input_tokens == 200_000

    def test_from_toml_dict_defaults_match_direct(self):
        """from_toml_dict({}) must produce same claim verification defaults as direct construction."""
        direct = ResearchConfig()
        from_toml = ResearchConfig.from_toml_dict({})
        for suffix in (
            "enabled", "sample_rate", "provider", "model", "timeout",
            "max_claims", "max_concurrent", "max_corrections",
            "annotate_unsupported", "max_input_tokens",
        ):
            field_name = f"deep_research_claim_verification_{suffix}"
            assert getattr(from_toml, field_name) == getattr(direct, field_name), (
                f"Mismatch for {field_name}"
            )

    def test_from_toml_dict_overrides(self):
        """from_toml_dict correctly picks up claim verification overrides."""
        config = ResearchConfig.from_toml_dict({
            "deep_research_claim_verification_enabled": True,
            "deep_research_claim_verification_sample_rate": 0.5,
            "deep_research_claim_verification_provider": "openai",
            "deep_research_claim_verification_model": "gpt-4o",
            "deep_research_claim_verification_timeout": 300,
            "deep_research_claim_verification_max_claims": 100,
            "deep_research_claim_verification_max_concurrent": 20,
            "deep_research_claim_verification_max_corrections": 10,
            "deep_research_claim_verification_annotate_unsupported": True,
            "deep_research_claim_verification_max_input_tokens": 500_000,
        })
        assert config.deep_research_claim_verification_enabled is True
        assert config.deep_research_claim_verification_sample_rate == 0.5
        assert config.deep_research_claim_verification_provider == "openai"
        assert config.deep_research_claim_verification_model == "gpt-4o"
        assert config.deep_research_claim_verification_timeout == 300.0
        assert config.deep_research_claim_verification_max_claims == 100
        assert config.deep_research_claim_verification_max_concurrent == 20
        assert config.deep_research_claim_verification_max_corrections == 10
        assert config.deep_research_claim_verification_annotate_unsupported is True
        assert config.deep_research_claim_verification_max_input_tokens == 500_000

    def test_from_toml_dict_subtable_flattening(self):
        """[research.deep_research] claim_verification_enabled = true flattens correctly."""
        config = ResearchConfig.from_toml_dict({
            "deep_research": {
                "claim_verification_enabled": True,
                "claim_verification_sample_rate": 0.8,
                "claim_verification_max_claims": 25,
            }
        })
        assert config.deep_research_claim_verification_enabled is True
        assert config.deep_research_claim_verification_sample_rate == 0.8
        assert config.deep_research_claim_verification_max_claims == 25


class TestClaimVerificationConfigValidation:
    """Claim verification validation rules in __post_init__."""

    def test_sample_rate_below_zero_raises(self):
        with pytest.raises(ValueError, match="sample_rate"):
            ResearchConfig(deep_research_claim_verification_sample_rate=-0.1)

    def test_sample_rate_above_one_raises(self):
        with pytest.raises(ValueError, match="sample_rate"):
            ResearchConfig(deep_research_claim_verification_sample_rate=1.1)

    def test_sample_rate_zero_accepted(self):
        config = ResearchConfig(deep_research_claim_verification_sample_rate=0.0)
        assert config.deep_research_claim_verification_sample_rate == 0.0

    def test_sample_rate_one_accepted(self):
        config = ResearchConfig(deep_research_claim_verification_sample_rate=1.0)
        assert config.deep_research_claim_verification_sample_rate == 1.0

    def test_timeout_zero_raises(self):
        with pytest.raises(ValueError, match="timeout"):
            ResearchConfig(deep_research_claim_verification_timeout=0)

    def test_max_claims_zero_raises(self):
        with pytest.raises(ValueError, match="max_claims"):
            ResearchConfig(deep_research_claim_verification_max_claims=0)

    def test_max_concurrent_zero_raises(self):
        with pytest.raises(ValueError, match="max_concurrent"):
            ResearchConfig(deep_research_claim_verification_max_concurrent=0)

    def test_max_corrections_negative_raises(self):
        with pytest.raises(ValueError, match="max_corrections"):
            ResearchConfig(deep_research_claim_verification_max_corrections=-1)

    def test_max_corrections_zero_accepted(self):
        config = ResearchConfig(deep_research_claim_verification_max_corrections=0)
        assert config.deep_research_claim_verification_max_corrections == 0

    def test_max_input_tokens_zero_raises(self):
        with pytest.raises(ValueError, match="max_input_tokens"):
            ResearchConfig(deep_research_claim_verification_max_input_tokens=0)


class TestClaimVerificationTimeoutTypes:
    """Timeout fields must be float, not int (P2.6 / P0 checklist)."""

    def test_summarization_timeout_is_float(self):
        import dataclasses
        field_map = {f.name: f for f in dataclasses.fields(ResearchConfig)}
        assert field_map["deep_research_summarization_timeout"].type == "float"

    def test_claim_verification_timeout_is_float(self):
        import dataclasses
        field_map = {f.name: f for f in dataclasses.fields(ResearchConfig)}
        assert field_map["deep_research_claim_verification_timeout"].type == "float"

    def test_timeout_preset_multiplier_produces_float(self):
        """Timeout preset multiplier should not truncate claim_verification_timeout."""
        config = ResearchConfig.from_toml_dict({
            "timeouts": {"preset": "fast"},
        })
        # fast = 0.5x multiplier; 120.0 * 0.5 = 60.0
        assert config.deep_research_claim_verification_timeout == 60.0
        assert isinstance(config.deep_research_claim_verification_timeout, float)


class TestClaimVerificationInDeepResearchSettings:
    """Claim verification fields must be accessible via deep_research_config sub-config."""

    def test_sub_config_has_claim_verification_fields(self):
        config = ResearchConfig(
            deep_research_claim_verification_enabled=True,
            deep_research_claim_verification_sample_rate=0.7,
            deep_research_claim_verification_max_claims=30,
        )
        drc = config.deep_research_config
        assert drc.claim_verification_enabled is True
        assert drc.claim_verification_sample_rate == 0.7
        assert drc.claim_verification_max_claims == 30


# ===========================================================================
# Regression: from_toml_dict field coverage
# ===========================================================================


class TestFromTomlDictFieldCoverage:
    """Every dataclass field must appear in from_toml_dict() constructor call.

    This is a regression guard against the exact class of bug found in P0:
    a field added to the dataclass but never wired into from_toml_dict().
    """

    def test_all_fields_present_in_from_toml_dict(self):
        """Assert every init-eligible field is passed in from_toml_dict constructor."""
        import dataclasses
        import inspect

        # Fields that from_toml_dict must pass to cls(...)
        all_fields = dataclasses.fields(ResearchConfig)
        init_fields = {f.name for f in all_fields if f.init}

        # Get from_toml_dict source and look for field names in the cls(...) call
        source = inspect.getsource(ResearchConfig.from_toml_dict)

        missing = []
        for field_name in sorted(init_fields):
            # The field must appear as a keyword argument: `field_name=`
            if f"{field_name}=" not in source:
                missing.append(field_name)

        assert not missing, (
            f"Fields declared on ResearchConfig but missing from from_toml_dict() "
            f"constructor call:\n  " + "\n  ".join(missing)
        )
