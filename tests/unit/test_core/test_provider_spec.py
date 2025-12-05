"""Tests for ProviderSpec parsing and ConsultationConfig priority."""

import pytest

from foundry_mcp.core.llm_config import (
    ProviderSpec,
    ConsultationConfig,
)


# =============================================================================
# ProviderSpec Parsing Tests
# =============================================================================


class TestProviderSpecParseAPI:
    """Tests for parsing [api] provider specs."""

    def test_parse_api_openai(self):
        """Test parsing OpenAI API spec."""
        spec = ProviderSpec.parse("[api]openai/gpt-4.1")
        assert spec.type == "api"
        assert spec.provider == "openai"
        assert spec.model == "gpt-4.1"
        assert spec.backend is None

    def test_parse_api_anthropic(self):
        """Test parsing Anthropic API spec."""
        spec = ProviderSpec.parse("[api]anthropic/claude-sonnet-4")
        assert spec.type == "api"
        assert spec.provider == "anthropic"
        assert spec.model == "claude-sonnet-4"

    def test_parse_api_local(self):
        """Test parsing local API spec."""
        spec = ProviderSpec.parse("[api]local/llama3.2")
        assert spec.type == "api"
        assert spec.provider == "local"
        assert spec.model == "llama3.2"

    def test_parse_api_preserves_model_case(self):
        """Test that model names preserve case."""
        spec = ProviderSpec.parse("[api]openai/GPT-4.1-Turbo")
        assert spec.model == "GPT-4.1-Turbo"

    def test_parse_api_lowercases_provider(self):
        """Test that provider names are lowercased."""
        spec = ProviderSpec.parse("[api]OPENAI/gpt-4.1")
        assert spec.provider == "openai"


class TestProviderSpecParseCLI:
    """Tests for parsing [cli] provider specs."""

    def test_parse_cli_simple(self):
        """Test parsing simple CLI spec (transport only)."""
        spec = ProviderSpec.parse("[cli]codex")
        assert spec.type == "cli"
        assert spec.provider == "codex"
        assert spec.model is None
        assert spec.backend is None

    def test_parse_cli_with_model(self):
        """Test parsing CLI spec with model."""
        spec = ProviderSpec.parse("[cli]gemini:gemini-2.5-flash")
        assert spec.type == "cli"
        assert spec.provider == "gemini"
        assert spec.model == "gemini-2.5-flash"
        assert spec.backend is None

    def test_parse_cli_with_backend_and_model(self):
        """Test parsing CLI spec with backend routing."""
        spec = ProviderSpec.parse("[cli]opencode:openai/gpt-5.1-codex")
        assert spec.type == "cli"
        assert spec.provider == "opencode"
        assert spec.backend == "openai"
        assert spec.model == "gpt-5.1-codex"

    def test_parse_cli_opencode_gemini_backend(self):
        """Test opencode with Gemini backend routing."""
        spec = ProviderSpec.parse("[cli]opencode:gemini/gemini-2.5-pro")
        assert spec.type == "cli"
        assert spec.provider == "opencode"
        assert spec.backend == "gemini"
        assert spec.model == "gemini-2.5-pro"

    def test_parse_cli_cursor_agent(self):
        """Test parsing cursor-agent CLI spec."""
        spec = ProviderSpec.parse("[cli]cursor-agent:claude-sonnet")
        assert spec.type == "cli"
        assert spec.provider == "cursor-agent"
        assert spec.model == "claude-sonnet"

    def test_parse_cli_preserves_model_case(self):
        """Test that model names preserve case."""
        spec = ProviderSpec.parse("[cli]gemini:Gemini-2.5-Flash")
        assert spec.model == "Gemini-2.5-Flash"


class TestProviderSpecParseErrors:
    """Tests for invalid spec parsing."""

    def test_empty_spec(self):
        """Test empty spec raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ProviderSpec.parse("")

    def test_whitespace_only(self):
        """Test whitespace-only spec raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ProviderSpec.parse("   ")

    def test_missing_bracket_prefix(self):
        """Test missing bracket prefix raises ValueError."""
        with pytest.raises(ValueError, match="Expected format"):
            ProviderSpec.parse("openai/gpt-4.1")

    def test_invalid_bracket_prefix(self):
        """Test invalid bracket prefix raises ValueError."""
        with pytest.raises(ValueError, match="Expected format"):
            ProviderSpec.parse("[invalid]openai/gpt-4.1")

    def test_api_missing_model(self):
        """Test API spec without model raises ValueError."""
        with pytest.raises(ValueError, match="Expected format"):
            ProviderSpec.parse("[api]openai")


class TestProviderSpecValidation:
    """Tests for ProviderSpec validation."""

    def test_validate_known_api_provider(self):
        """Test validation passes for known API provider."""
        spec = ProviderSpec.parse("[api]openai/gpt-4.1")
        errors = spec.validate()
        assert errors == []

    def test_validate_unknown_api_provider(self):
        """Test validation warns for unknown API provider."""
        spec = ProviderSpec(type="api", provider="unknown", model="model-1")
        errors = spec.validate()
        assert len(errors) == 1
        assert "Unknown API provider" in errors[0]

    def test_validate_known_cli_provider(self):
        """Test validation passes for known CLI provider."""
        spec = ProviderSpec.parse("[cli]gemini:gemini-2.5-flash")
        errors = spec.validate()
        assert errors == []

    def test_validate_unknown_cli_provider(self):
        """Test validation warns for unknown CLI provider."""
        spec = ProviderSpec(type="cli", provider="unknown")
        errors = spec.validate()
        assert len(errors) == 1
        assert "Unknown CLI provider" in errors[0]

    def test_validate_unknown_backend(self):
        """Test validation warns for unknown backend."""
        spec = ProviderSpec(type="cli", provider="opencode", backend="unknown", model="m")
        errors = spec.validate()
        assert len(errors) == 1
        assert "Unknown backend" in errors[0]


class TestProviderSpecStr:
    """Tests for ProviderSpec string representation."""

    def test_str_api(self):
        """Test string representation for API spec."""
        spec = ProviderSpec.parse("[api]openai/gpt-4.1")
        assert str(spec) == "[api]openai/gpt-4.1"

    def test_str_cli_simple(self):
        """Test string representation for simple CLI spec."""
        spec = ProviderSpec.parse("[cli]codex")
        assert str(spec) == "[cli]codex"

    def test_str_cli_with_model(self):
        """Test string representation for CLI spec with model."""
        spec = ProviderSpec.parse("[cli]gemini:gemini-2.5-flash")
        assert str(spec) == "[cli]gemini:gemini-2.5-flash"

    def test_str_cli_with_backend(self):
        """Test string representation for CLI spec with backend."""
        spec = ProviderSpec.parse("[cli]opencode:openai/gpt-5.1")
        assert str(spec) == "[cli]opencode:openai/gpt-5.1"


# =============================================================================
# ConsultationConfig Priority Tests
# =============================================================================


class TestConsultationConfigPriority:
    """Tests for ConsultationConfig priority list."""

    def test_empty_priority_default(self):
        """Test default empty priority list."""
        config = ConsultationConfig()
        assert config.priority == []

    def test_from_dict_with_priority(self):
        """Test loading priority from dict."""
        data = {
            "priority": [
                "[cli]opencode:openai/gpt-5.1",
                "[cli]gemini:gemini-2.5-flash",
                "[api]openai/gpt-4.1",
            ]
        }
        config = ConsultationConfig.from_dict(data)
        assert len(config.priority) == 3
        assert config.priority[0] == "[cli]opencode:openai/gpt-5.1"

    def test_get_provider_specs(self):
        """Test parsing priority list into ProviderSpec objects."""
        config = ConsultationConfig(
            priority=[
                "[cli]opencode:openai/gpt-5.1",
                "[api]openai/gpt-4.1",
            ]
        )
        specs = config.get_provider_specs()
        assert len(specs) == 2
        assert specs[0].type == "cli"
        assert specs[0].provider == "opencode"
        assert specs[1].type == "api"
        assert specs[1].provider == "openai"


class TestConsultationConfigOverrides:
    """Tests for ConsultationConfig per-provider overrides."""

    def test_empty_overrides_default(self):
        """Test default empty overrides."""
        config = ConsultationConfig()
        assert config.overrides == {}

    def test_from_dict_with_overrides(self):
        """Test loading overrides from dict."""
        data = {
            "overrides": {
                "[cli]opencode:openai/gpt-5.1": {"timeout": 600},
                "[api]openai/gpt-4.1": {"temperature": 0.3},
            }
        }
        config = ConsultationConfig.from_dict(data)
        assert len(config.overrides) == 2
        assert config.overrides["[cli]opencode:openai/gpt-5.1"]["timeout"] == 600

    def test_get_override_existing(self):
        """Test getting existing override."""
        config = ConsultationConfig(
            overrides={"[api]openai/gpt-4.1": {"timeout": 120}}
        )
        override = config.get_override("[api]openai/gpt-4.1")
        assert override == {"timeout": 120}

    def test_get_override_nonexistent(self):
        """Test getting nonexistent override returns empty dict."""
        config = ConsultationConfig()
        override = config.get_override("[api]openai/gpt-4.1")
        assert override == {}


class TestConsultationConfigValidation:
    """Tests for ConsultationConfig validation."""

    def test_validate_valid_priority(self):
        """Test validation passes for valid priority list."""
        config = ConsultationConfig(
            priority=[
                "[cli]gemini:gemini-2.5-flash",
                "[api]openai/gpt-4.1",
            ]
        )
        # Should not raise
        config.validate()

    def test_validate_invalid_priority_spec(self):
        """Test validation fails for invalid spec in priority."""
        config = ConsultationConfig(
            priority=["invalid-spec"]
        )
        with pytest.raises(ValueError, match="Invalid provider specs"):
            config.validate()

    def test_validate_unknown_provider_warning(self):
        """Test validation fails for unknown provider in priority."""
        config = ConsultationConfig(
            priority=["[cli]unknown-provider:model"]
        )
        with pytest.raises(ValueError, match="Unknown CLI provider"):
            config.validate()
