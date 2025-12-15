"""
Unit tests for foundry_mcp.core.providers.detectors module.

Tests cover:
- ProviderDetector dataclass creation and attributes
- PATH resolution via resolve_binary()
- Health probe execution via _run_probe()
- Environment variable overrides for availability
- Test mode behavior (FOUNDRY_PROVIDER_TEST_MODE)
- Public API functions: detect_provider_availability, get_provider_statuses, etc.
"""

import os
from unittest.mock import patch

import pytest

from foundry_mcp.core.providers.detectors import (
    ProviderDetector,
    register_detector,
    get_detector,
    detect_provider_availability,
    get_provider_statuses,
    get_provider_unavailability_reasons,
    list_detectors,
    reset_detectors,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_detector_registry():
    """Reset detectors before and after each test."""
    reset_detectors()
    yield
    reset_detectors()


@pytest.fixture
def custom_detector():
    """Create a custom detector for testing."""
    return ProviderDetector(
        provider_id="test-provider",
        binary_name="test-binary",
        override_env="TEST_PROVIDER_OVERRIDE",
        binary_env="TEST_PROVIDER_BINARY",
        probe_args=("--version",),
        probe_timeout=5,
    )


# =============================================================================
# ProviderDetector Dataclass Tests
# =============================================================================


class TestProviderDetectorDataclass:
    """Tests for ProviderDetector dataclass creation."""

    def test_minimal_creation(self):
        """ProviderDetector should work with required fields only."""
        detector = ProviderDetector(
            provider_id="test",
            binary_name="test-cli",
        )
        assert detector.provider_id == "test"
        assert detector.binary_name == "test-cli"
        assert detector.override_env is None
        assert detector.binary_env is None
        assert detector.probe_args == ("--version",)
        assert detector.probe_timeout == 5

    def test_full_creation(self):
        """ProviderDetector should accept all parameters."""
        detector = ProviderDetector(
            provider_id="gemini",
            binary_name="gemini",
            override_env="FOUNDRY_GEMINI_AVAILABLE_OVERRIDE",
            binary_env="FOUNDRY_GEMINI_BINARY",
            probe_args=("--help",),
            probe_timeout=10,
        )
        assert detector.provider_id == "gemini"
        assert detector.binary_name == "gemini"
        assert detector.override_env == "FOUNDRY_GEMINI_AVAILABLE_OVERRIDE"
        assert detector.binary_env == "FOUNDRY_GEMINI_BINARY"
        assert detector.probe_args == ("--help",)
        assert detector.probe_timeout == 10

    def test_immutability(self):
        """ProviderDetector should be immutable (frozen)."""
        detector = ProviderDetector(
            provider_id="test",
            binary_name="test-cli",
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            detector.provider_id = "changed"

    def test_custom_probe_args(self):
        """ProviderDetector should accept custom probe args."""
        detector = ProviderDetector(
            provider_id="test",
            binary_name="test-cli",
            probe_args=("--version", "--json"),
        )
        assert detector.probe_args == ("--version", "--json")

    def test_empty_probe_args(self):
        """ProviderDetector should accept empty probe args."""
        detector = ProviderDetector(
            provider_id="test",
            binary_name="test-cli",
            probe_args=(),
        )
        assert detector.probe_args == ()


# =============================================================================
# PATH Resolution Tests
# =============================================================================


class TestResolveBinary:
    """Tests for ProviderDetector.resolve_binary() method."""

    def test_resolve_binary_found_in_path(self):
        """resolve_binary should find binary in PATH."""
        detector = ProviderDetector(
            provider_id="test",
            binary_name="python",  # Python should be available
        )
        result = detector.resolve_binary()
        assert result is not None
        assert "python" in result.lower()

    def test_resolve_binary_not_found(self):
        """resolve_binary should return None for missing binary."""
        detector = ProviderDetector(
            provider_id="test",
            binary_name="nonexistent-binary-xyz-12345",
        )
        result = detector.resolve_binary()
        assert result is None

    def test_resolve_binary_from_env_override(self):
        """resolve_binary should use binary_env if set."""
        detector = ProviderDetector(
            provider_id="test",
            binary_name="nonexistent",
            binary_env="TEST_BINARY_PATH",
        )
        with patch.dict(os.environ, {"TEST_BINARY_PATH": "python"}):
            result = detector.resolve_binary()
            assert result is not None
            assert "python" in result.lower()

    def test_resolve_binary_custom_tool_path(self):
        """resolve_binary should check FOUNDRY_TOOL_PATH first."""
        detector = ProviderDetector(
            provider_id="test",
            binary_name="python",
        )
        # With empty FOUNDRY_TOOL_PATH, should not find python
        with patch.dict(os.environ, {"FOUNDRY_TOOL_PATH": "/nonexistent/path"}):
            result = detector.resolve_binary()
            assert result is None


# =============================================================================
# Environment Override Tests
# =============================================================================


class TestEnvironmentOverrides:
    """Tests for environment variable override behavior."""

    def test_override_true_returns_available(self, custom_detector):
        """Override set to true should return available."""
        with patch.dict(os.environ, {"TEST_PROVIDER_OVERRIDE": "true"}):
            assert custom_detector.is_available() is True

    def test_override_one_returns_available(self, custom_detector):
        """Override set to '1' should return available."""
        with patch.dict(os.environ, {"TEST_PROVIDER_OVERRIDE": "1"}):
            assert custom_detector.is_available() is True

    def test_override_yes_returns_available(self, custom_detector):
        """Override set to 'yes' should return available."""
        with patch.dict(os.environ, {"TEST_PROVIDER_OVERRIDE": "yes"}):
            assert custom_detector.is_available() is True

    def test_override_false_returns_unavailable(self, custom_detector):
        """Override set to false should return unavailable."""
        with patch.dict(os.environ, {"TEST_PROVIDER_OVERRIDE": "false"}):
            assert custom_detector.is_available() is False

    def test_override_zero_returns_unavailable(self, custom_detector):
        """Override set to '0' should return unavailable."""
        with patch.dict(os.environ, {"TEST_PROVIDER_OVERRIDE": "0"}):
            assert custom_detector.is_available() is False

    def test_override_no_returns_unavailable(self, custom_detector):
        """Override set to 'no' should return unavailable."""
        with patch.dict(os.environ, {"TEST_PROVIDER_OVERRIDE": "no"}):
            assert custom_detector.is_available() is False

    def test_override_case_insensitive(self, custom_detector):
        """Override should be case insensitive."""
        with patch.dict(os.environ, {"TEST_PROVIDER_OVERRIDE": "TRUE"}):
            assert custom_detector.is_available() is True
        with patch.dict(os.environ, {"TEST_PROVIDER_OVERRIDE": "False"}):
            assert custom_detector.is_available() is False

    def test_no_override_falls_through(self, custom_detector):
        """Without override, should check actual binary availability."""
        # Ensure override env is not set
        env = {k: v for k, v in os.environ.items() if k != "TEST_PROVIDER_OVERRIDE"}
        with patch.dict(os.environ, env, clear=True):
            # Without binary, should return False (binary not found)
            assert custom_detector.is_available() is False


# =============================================================================
# Test Mode Tests
# =============================================================================


class TestTestMode:
    """Tests for FOUNDRY_PROVIDER_TEST_MODE behavior."""

    def test_test_mode_returns_false_without_override(self, custom_detector):
        """In test mode without override, should return False."""
        with patch.dict(os.environ, {"FOUNDRY_PROVIDER_TEST_MODE": "1"}, clear=True):
            assert custom_detector.is_available() is False

    def test_test_mode_respects_override(self, custom_detector):
        """In test mode with override, should respect override."""
        with patch.dict(os.environ, {
            "FOUNDRY_PROVIDER_TEST_MODE": "1",
            "TEST_PROVIDER_OVERRIDE": "true",
        }):
            assert custom_detector.is_available() is True

    def test_test_mode_off_checks_binary(self):
        """With test mode off, should check actual binary."""
        detector = ProviderDetector(
            provider_id="python",
            binary_name="python",  # Should exist
        )
        with patch.dict(os.environ, {"FOUNDRY_PROVIDER_TEST_MODE": "0"}):
            # Python should be available
            result = detector.is_available(use_probe=False)
            assert result is True


# =============================================================================
# Probe Execution Tests
# =============================================================================

# Check if test mode is enabled (real probes disabled)
_TEST_MODE_ENABLED = os.environ.get("FOUNDRY_PROVIDER_TEST_MODE", "").lower() in (
    "1", "true", "yes", "on"
)


class TestProbeExecution:
    """Tests for health probe execution.

    Note: These tests require real binary execution and are skipped in
    FOUNDRY_PROVIDER_TEST_MODE since test mode disables real probes.
    """

    @pytest.mark.skipif(
        _TEST_MODE_ENABLED,
        reason="Test mode disables real probe execution"
    )
    def test_probe_success(self):
        """Successful probe should return True."""
        detector = ProviderDetector(
            provider_id="python",
            binary_name="python",
            probe_args=("--version",),
        )
        # Use PATH resolution to find python
        with patch.dict(os.environ, {}, clear=False):
            result = detector.is_available(use_probe=True)
            assert result is True

    def test_probe_failure_returns_false(self):
        """Failed probe should return False."""
        detector = ProviderDetector(
            provider_id="test",
            binary_name="python",
            probe_args=("--nonexistent-flag-xyz",),
        )
        # The probe will fail due to bad argument
        result = detector.is_available(use_probe=True)
        assert result is False

    @pytest.mark.skipif(
        _TEST_MODE_ENABLED,
        reason="Test mode disables real probe execution"
    )
    def test_skip_probe_only_checks_path(self):
        """use_probe=False should only check PATH resolution."""
        detector = ProviderDetector(
            provider_id="python",
            binary_name="python",
            probe_args=("--nonexistent-flag-xyz",),  # Bad probe args
        )
        # With use_probe=False, should succeed (python exists in PATH)
        result = detector.is_available(use_probe=False)
        assert result is True

    @pytest.mark.skipif(
        _TEST_MODE_ENABLED,
        reason="Test mode disables real probe execution"
    )
    def test_empty_probe_args_skips_probe(self):
        """Empty probe_args should skip probe execution."""
        detector = ProviderDetector(
            provider_id="python",
            binary_name="python",
            probe_args=(),
        )
        result = detector.is_available(use_probe=True)
        assert result is True


# =============================================================================
# Public API Tests
# =============================================================================


class TestRegisterDetector:
    """Tests for register_detector function."""

    def test_register_new_detector(self, custom_detector):
        """Should register a new detector."""
        register_detector(custom_detector)
        assert get_detector("test-provider") is custom_detector

    def test_register_duplicate_raises(self, custom_detector):
        """Should raise ValueError for duplicate registration."""
        register_detector(custom_detector)
        with pytest.raises(ValueError, match="already exists"):
            register_detector(custom_detector)

    def test_register_with_replace(self, custom_detector):
        """Should allow replacement with replace=True."""
        register_detector(custom_detector)
        new_detector = ProviderDetector(
            provider_id="test-provider",
            binary_name="different-binary",
        )
        register_detector(new_detector, replace=True)
        assert get_detector("test-provider") is new_detector


class TestGetDetector:
    """Tests for get_detector function."""

    def test_get_existing_detector(self):
        """Should return registered detector."""
        # Default detectors are registered
        detector = get_detector("gemini")
        assert detector is not None
        assert detector.provider_id == "gemini"

    def test_get_nonexistent_detector(self):
        """Should return None for unregistered detector."""
        detector = get_detector("nonexistent-provider")
        assert detector is None


class TestDetectProviderAvailability:
    """Tests for detect_provider_availability function."""

    def test_detect_registered_provider(self):
        """Should check availability for registered provider."""
        # Force gemini to be available via override
        with patch.dict(os.environ, {"FOUNDRY_GEMINI_AVAILABLE_OVERRIDE": "true"}):
            result = detect_provider_availability("gemini")
            assert result is True

    def test_detect_unregistered_provider_raises(self):
        """Should raise KeyError for unregistered provider."""
        with pytest.raises(KeyError, match="No detector registered"):
            detect_provider_availability("nonexistent-provider-xyz")


class TestGetProviderStatuses:
    """Tests for get_provider_statuses function."""

    def test_returns_dict_of_all_providers(self):
        """Should return status dict for all registered providers."""
        # Force all providers to be unavailable via test mode
        with patch.dict(os.environ, {"FOUNDRY_PROVIDER_TEST_MODE": "1"}):
            statuses = get_provider_statuses()
            assert isinstance(statuses, dict)
            assert "gemini" in statuses
            assert "codex" in statuses
            assert "cursor-agent" in statuses
            assert "claude" in statuses
            assert "opencode" in statuses

    def test_returns_bool_values(self):
        """Should return boolean availability values."""
        with patch.dict(os.environ, {"FOUNDRY_PROVIDER_TEST_MODE": "1"}):
            statuses = get_provider_statuses()
            for provider_id, available in statuses.items():
                assert isinstance(available, bool)

    def test_respects_overrides(self):
        """Should respect per-provider overrides."""
        with patch.dict(os.environ, {
            "FOUNDRY_PROVIDER_TEST_MODE": "1",
            "FOUNDRY_GEMINI_AVAILABLE_OVERRIDE": "true",
        }):
            statuses = get_provider_statuses()
            assert statuses["gemini"] is True
            assert statuses["codex"] is False  # No override


class TestListDetectors:
    """Tests for list_detectors function."""

    def test_returns_all_detectors(self):
        """Should return all registered detectors."""
        detectors = list(list_detectors())
        assert len(detectors) >= 5  # Default detectors
        provider_ids = [d.provider_id for d in detectors]
        assert "gemini" in provider_ids
        assert "codex" in provider_ids
        assert "cursor-agent" in provider_ids
        assert "claude" in provider_ids
        assert "opencode" in provider_ids


class TestResetDetectors:
    """Tests for reset_detectors function."""

    def test_reset_removes_custom_detectors(self, custom_detector):
        """reset should remove custom detectors."""
        register_detector(custom_detector)
        assert get_detector("test-provider") is not None
        reset_detectors()
        assert get_detector("test-provider") is None

    def test_reset_restores_defaults(self):
        """reset should restore default detectors."""
        # Clear all detectors
        from foundry_mcp.core.providers.detectors import _DETECTORS
        _DETECTORS.clear()
        assert get_detector("gemini") is None

        reset_detectors()
        assert get_detector("gemini") is not None


# =============================================================================
# Default Detector Configuration Tests
# =============================================================================


class TestDefaultDetectors:
    """Tests for default detector configurations."""

    def test_gemini_detector_config(self):
        """Gemini detector should have correct configuration."""
        detector = get_detector("gemini")
        assert detector is not None
        assert detector.binary_name == "gemini"
        assert detector.override_env == "FOUNDRY_GEMINI_AVAILABLE_OVERRIDE"
        assert detector.binary_env == "FOUNDRY_GEMINI_BINARY"
        assert detector.probe_args == ("--help",)

    def test_codex_detector_config(self):
        """Codex detector should have correct configuration."""
        detector = get_detector("codex")
        assert detector is not None
        assert detector.binary_name == "codex"
        assert detector.override_env == "FOUNDRY_CODEX_AVAILABLE_OVERRIDE"
        assert detector.binary_env == "FOUNDRY_CODEX_BINARY"
        assert detector.probe_args == ("--version",)

    def test_cursor_agent_detector_config(self):
        """Cursor-agent detector should have correct configuration."""
        detector = get_detector("cursor-agent")
        assert detector is not None
        assert detector.binary_name == "cursor-agent"
        assert detector.override_env == "FOUNDRY_CURSOR_AGENT_AVAILABLE_OVERRIDE"
        assert detector.binary_env == "FOUNDRY_CURSOR_AGENT_BINARY"

    def test_claude_detector_config(self):
        """Claude detector should have correct configuration."""
        detector = get_detector("claude")
        assert detector is not None
        assert detector.binary_name == "claude"
        assert detector.override_env == "FOUNDRY_CLAUDE_AVAILABLE_OVERRIDE"
        assert detector.binary_env == "FOUNDRY_CLAUDE_BINARY"

    def test_opencode_detector_config(self):
        """Opencode detector should have correct configuration."""
        detector = get_detector("opencode")
        assert detector is not None
        assert detector.binary_name == "opencode"
        assert detector.override_env == "FOUNDRY_OPENCODE_AVAILABLE_OVERRIDE"
        assert detector.binary_env == "FOUNDRY_OPENCODE_BINARY"


# =============================================================================
# Diagnostic Function Tests
# =============================================================================


class TestGetUnavailabilityReason:
    """Tests for ProviderDetector.get_unavailability_reason() method."""

    def test_returns_none_when_available_via_override(self, custom_detector):
        """Should return None when provider is available via override."""
        with patch.dict(os.environ, {"TEST_PROVIDER_OVERRIDE": "true"}):
            reason = custom_detector.get_unavailability_reason()
            assert reason is None

    def test_returns_reason_when_disabled_via_override(self, custom_detector):
        """Should return reason when explicitly disabled via override."""
        with patch.dict(os.environ, {"TEST_PROVIDER_OVERRIDE": "0"}):
            reason = custom_detector.get_unavailability_reason()
            assert reason is not None
            assert "Explicitly disabled" in reason
            assert "TEST_PROVIDER_OVERRIDE" in reason

    def test_returns_reason_for_test_mode(self, custom_detector):
        """Should return reason when in test mode without override."""
        with patch.dict(os.environ, {"FOUNDRY_PROVIDER_TEST_MODE": "1"}, clear=True):
            reason = custom_detector.get_unavailability_reason()
            assert reason is not None
            assert "Test mode enabled" in reason

    def test_returns_reason_for_missing_binary(self):
        """Should return reason when binary is not found."""
        detector = ProviderDetector(
            provider_id="missing",
            binary_name="nonexistent-binary-xyz-12345",
        )
        reason = detector.get_unavailability_reason()
        assert reason is not None
        assert "not found in PATH" in reason
        assert "nonexistent-binary-xyz-12345" in reason

    @pytest.mark.skipif(
        _TEST_MODE_ENABLED,
        reason="Test mode disables real probe execution"
    )
    def test_returns_reason_for_probe_failure(self):
        """Should return reason when health probe fails."""
        detector = ProviderDetector(
            provider_id="test",
            binary_name="python",  # Exists
            probe_args=("--nonexistent-flag-xyz",),  # Will fail
        )
        reason = detector.get_unavailability_reason(use_probe=True)
        assert reason is not None
        assert "Health probe failed" in reason

    @pytest.mark.skipif(
        _TEST_MODE_ENABLED,
        reason="Test mode disables real probe execution"
    )
    def test_returns_none_for_available_provider(self):
        """Should return None when provider is available."""
        detector = ProviderDetector(
            provider_id="python",
            binary_name="python",
            probe_args=("--version",),
        )
        reason = detector.get_unavailability_reason(use_probe=True)
        assert reason is None


class TestGetProviderUnavailabilityReasons:
    """Tests for get_provider_unavailability_reasons function."""

    def test_returns_dict_of_all_providers(self):
        """Should return reasons dict for all registered providers."""
        with patch.dict(os.environ, {"FOUNDRY_PROVIDER_TEST_MODE": "1"}):
            reasons = get_provider_unavailability_reasons()
            assert isinstance(reasons, dict)
            assert "gemini" in reasons
            assert "codex" in reasons
            assert "cursor-agent" in reasons
            assert "claude" in reasons
            assert "opencode" in reasons

    def test_returns_optional_str_values(self):
        """Should return Optional[str] values (None or string)."""
        with patch.dict(os.environ, {"FOUNDRY_PROVIDER_TEST_MODE": "1"}):
            reasons = get_provider_unavailability_reasons()
            for provider_id, reason in reasons.items():
                assert reason is None or isinstance(reason, str)

    def test_respects_overrides(self):
        """Should return None for providers with available override."""
        with patch.dict(os.environ, {
            "FOUNDRY_PROVIDER_TEST_MODE": "1",
            "FOUNDRY_GEMINI_AVAILABLE_OVERRIDE": "true",
        }):
            reasons = get_provider_unavailability_reasons()
            assert reasons["gemini"] is None  # Available via override
            assert reasons["codex"] is not None  # No override, test mode

    def test_unavailable_reasons_contain_details(self):
        """Unavailable reasons should contain diagnostic details."""
        with patch.dict(os.environ, {"FOUNDRY_PROVIDER_TEST_MODE": "1"}):
            reasons = get_provider_unavailability_reasons()
            # In test mode, should mention test mode
            for provider_id, reason in reasons.items():
                if reason is not None:
                    assert "Test mode" in reason or "not found" in reason or "probe failed" in reason
