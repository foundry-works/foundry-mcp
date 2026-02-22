"""Tests for ProviderSpec parsing and ConsultationConfig priority."""

import pytest

from foundry_mcp.core.llm_config.consultation import ConsultationConfig
from foundry_mcp.core.llm_config.provider_spec import ProviderSpec

# =============================================================================
# ProviderSpec Parsing Tests
# =============================================================================


class TestProviderSpecParseAPIRejected:
    """Tests that [api] provider specs are rejected with ValueError."""

    def test_parse_api_openai_rejected(self):
        """Test that [api]openai/gpt-4.1 raises ValueError."""
        with pytest.raises(ValueError, match="no longer supported"):
            ProviderSpec.parse("[api]openai/gpt-4.1")

    def test_parse_api_anthropic_rejected(self):
        """Test that [api]anthropic/claude-sonnet-4 raises ValueError."""
        with pytest.raises(ValueError, match="no longer supported"):
            ProviderSpec.parse("[api]anthropic/claude-sonnet-4")

    def test_parse_api_local_rejected(self):
        """Test that [api]local/llama3.2 raises ValueError."""
        with pytest.raises(ValueError, match="no longer supported"):
            ProviderSpec.parse("[api]local/llama3.2")


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
        spec = ProviderSpec.parse("[cli]gemini:pro")
        assert spec.type == "cli"
        assert spec.provider == "gemini"
        assert spec.model == "pro"
        assert spec.backend is None

    def test_parse_cli_with_backend_and_model(self):
        """Test parsing CLI spec with backend routing."""
        spec = ProviderSpec.parse("[cli]opencode:openai/gpt-5.2")
        assert spec.type == "cli"
        assert spec.provider == "opencode"
        assert spec.backend == "openai"
        assert spec.model == "gpt-5.2"

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


class TestProviderSpecValidation:
    """Tests for ProviderSpec validation."""

    def test_validate_known_cli_provider(self):
        """Test validation passes for known CLI provider."""
        spec = ProviderSpec.parse("[cli]gemini:pro")
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

    def test_str_cli_simple(self):
        """Test string representation for simple CLI spec."""
        spec = ProviderSpec.parse("[cli]codex")
        assert str(spec) == "[cli]codex"

    def test_str_cli_with_model(self):
        """Test string representation for CLI spec with model."""
        spec = ProviderSpec.parse("[cli]gemini:pro")
        assert str(spec) == "[cli]gemini:pro"

    def test_str_cli_with_backend(self):
        """Test string representation for CLI spec with backend."""
        spec = ProviderSpec.parse("[cli]opencode:openai/gpt-5.2")
        assert str(spec) == "[cli]opencode:openai/gpt-5.2"


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
                "[cli]gemini:pro",
                "[cli]claude:opus",
                "[cli]opencode:openai/gpt-5.2",
            ]
        }
        config = ConsultationConfig.from_dict(data)
        assert len(config.priority) == 3
        assert config.priority[0] == "[cli]gemini:pro"

    def test_get_provider_specs(self):
        """Test parsing priority list into ProviderSpec objects."""
        config = ConsultationConfig(
            priority=[
                "[cli]opencode:openai/gpt-5.2",
                "[cli]claude:opus",
            ]
        )
        specs = config.get_provider_specs()
        assert len(specs) == 2
        assert specs[0].type == "cli"
        assert specs[0].provider == "opencode"
        assert specs[1].type == "cli"
        assert specs[1].provider == "claude"


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
                "[cli]opencode:openai/gpt-5.2": {"timeout": 600},
            }
        }
        config = ConsultationConfig.from_dict(data)
        assert len(config.overrides) == 1
        assert config.overrides["[cli]opencode:openai/gpt-5.2"]["timeout"] == 600

    def test_get_override_existing(self):
        """Test getting existing override."""
        config = ConsultationConfig(overrides={"[cli]claude:opus": {"timeout": 120}})
        override = config.get_override("[cli]claude:opus")
        assert override == {"timeout": 120}

    def test_get_override_nonexistent(self):
        """Test getting nonexistent override returns empty dict."""
        config = ConsultationConfig()
        override = config.get_override("[cli]claude:opus")
        assert override == {}


class TestConsultationConfigValidation:
    """Tests for ConsultationConfig validation."""

    def test_validate_valid_priority(self):
        """Test validation passes for valid priority list."""
        config = ConsultationConfig(
            priority=[
                "[cli]gemini:pro",
                "[cli]claude:opus",
            ]
        )
        # Should not raise
        config.validate()

    def test_validate_invalid_priority_spec(self):
        """Test validation fails for invalid spec in priority."""
        config = ConsultationConfig(priority=["invalid-spec"])
        with pytest.raises(ValueError, match="Invalid provider specs"):
            config.validate()

    def test_validate_unknown_provider_warning(self):
        """Test validation fails for unknown provider in priority."""
        config = ConsultationConfig(priority=["[cli]unknown-provider:model"])
        with pytest.raises(ValueError, match="Unknown CLI provider"):
            config.validate()

    def test_validate_api_spec_in_priority_rejected(self):
        """Test validation fails for [api] spec in priority list."""
        config = ConsultationConfig(priority=["[api]openai/gpt-4.1"])
        with pytest.raises(ValueError, match="no longer supported"):
            config.validate()
