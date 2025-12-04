"""
Provider availability detection utilities.

Centralizes CLI discovery strategies (PATH resolution, environment overrides,
and health probes) so provider modules and higher-level tooling can share
consistent logic.

Environment Variables:
    FOUNDRY_PROVIDER_TEST_MODE: When set to "1"/"true"/"yes", bypasses real CLI
        probes and returns availability based on environment override only.
        Useful for CI/CD environments where CLIs may not be installed.

    FOUNDRY_TOOL_PATH: Additional PATH directories for binary resolution.

    Per-provider overrides (e.g., FOUNDRY_GEMINI_AVAILABLE_OVERRIDE):
        Set to "1"/"true"/"yes" to force availability, "0"/"false"/"no" to force
        unavailability. Takes precedence over actual CLI detection.

Example:
    >>> from foundry_mcp.core.providers.detectors import detect_provider_availability
    >>> detect_provider_availability("gemini")
    True
    >>> from foundry_mcp.core.providers.detectors import get_provider_statuses
    >>> get_provider_statuses()
    {'gemini': True, 'codex': False, 'cursor-agent': True, 'claude': True, 'opencode': False}
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

# Environment variable for test mode (bypasses real CLI probes)
_TEST_MODE_ENV = "FOUNDRY_PROVIDER_TEST_MODE"

# Environment variable for additional tool PATH directories
_TOOL_PATH_ENV = "FOUNDRY_TOOL_PATH"


def _coerce_bool(value: Optional[str]) -> Optional[bool]:
    """
    Convert an environment variable string to a boolean.

    Args:
        value: String value from environment variable

    Returns:
        True for "1", "true", "yes", "on" (case-insensitive)
        False for "0", "false", "no", "off" (case-insensitive)
        None for any other value or if value is None
    """
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return None


def _is_test_mode() -> bool:
    """Check if test mode is enabled (bypasses real CLI probes)."""
    return _coerce_bool(os.environ.get(_TEST_MODE_ENV)) is True


def _resolve_executable(binary: str) -> Optional[str]:
    """
    Resolve a binary name to its full path.

    Checks FOUNDRY_TOOL_PATH first (if set), then falls back to system PATH.

    Args:
        binary: Binary name to resolve (e.g., "gemini", "codex")

    Returns:
        Full path to the binary if found, None otherwise
    """
    configured_path = os.environ.get(_TOOL_PATH_ENV)
    if configured_path:
        return shutil.which(binary, path=configured_path)
    return shutil.which(binary)


@dataclass(frozen=True)
class ProviderDetector:
    """
    Configuration describing how to detect a provider CLI.

    This dataclass encapsulates all information needed to detect whether
    a provider's CLI tool is available and functional.

    Attributes:
        provider_id: Canonical provider identifier (e.g., "gemini", "codex")
        binary_name: Default binary name to search for in PATH
        override_env: Environment variable to force availability (True/False)
        binary_env: Environment variable to override the binary path/name
        probe_args: Arguments for health probe command (default: ("--version",))
        probe_timeout: Timeout in seconds for health probe (default: 5)

    Example:
        >>> detector = ProviderDetector(
        ...     provider_id="gemini",
        ...     binary_name="gemini",
        ...     override_env="FOUNDRY_GEMINI_AVAILABLE_OVERRIDE",
        ...     binary_env="FOUNDRY_GEMINI_BINARY",
        ...     probe_args=("--help",),
        ... )
        >>> detector.is_available()
        True
    """

    provider_id: str
    binary_name: str
    override_env: Optional[str] = None
    binary_env: Optional[str] = None
    probe_args: Sequence[str] = field(default_factory=lambda: ("--version",))
    probe_timeout: int = 5

    def resolve_binary(self) -> Optional[str]:
        """
        Resolve the binary path for this provider.

        Checks the binary_env environment variable first, then falls back
        to resolving binary_name via PATH.

        Returns:
            Full path to the binary if found, None otherwise
        """
        binary = self.binary_name
        if self.binary_env:
            env_binary = os.environ.get(self.binary_env)
            if env_binary:
                binary = env_binary
        return _resolve_executable(binary)

    def _run_probe(self, executable: str) -> bool:
        """
        Run a health probe against the CLI binary.

        Args:
            executable: Full path to the binary

        Returns:
            True if probe succeeds, False otherwise
        """
        if not self.probe_args:
            return True

        try:
            subprocess.run(
                [executable, *self.probe_args],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=self.probe_timeout,
                check=True,
            )
            return True
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug(
                "Probe for provider '%s' failed via %s: %s",
                self.provider_id,
                executable,
                exc,
            )
            return False

    def is_available(self, *, use_probe: bool = True) -> bool:
        """
        Check whether this provider is available.

        Resolution order:
        1. Check override_env (if set, returns its boolean value)
        2. In test mode, return False (no real CLI available)
        3. Resolve binary via PATH
        4. Optionally run health probe

        Args:
            use_probe: When True, run health probe after finding binary.
                When False, only check PATH resolution.

        Returns:
            True if provider is available, False otherwise
        """
        # Check environment override first
        if self.override_env:
            override = _coerce_bool(os.environ.get(self.override_env))
            if override is not None:
                logger.debug(
                    "Provider '%s' availability override: %s",
                    self.provider_id,
                    override,
                )
                return override

        # In test mode, return False unless overridden above
        if _is_test_mode():
            logger.debug(
                "Provider '%s' unavailable (test mode enabled)",
                self.provider_id,
            )
            return False

        # Resolve binary path
        executable = self.resolve_binary()
        if not executable:
            logger.debug(
                "Provider '%s' unavailable (binary not found in PATH)",
                self.provider_id,
            )
            return False

        # Skip probe if not requested
        if not use_probe:
            return True

        # Run health probe
        return self._run_probe(executable)

    def get_unavailability_reason(self, *, use_probe: bool = True) -> Optional[str]:
        """
        Return a human-readable reason why this provider is unavailable.

        Useful for diagnostic messages when a provider cannot be used.
        Returns None if the provider is available.

        Args:
            use_probe: When True, include probe failures in reasons.
                When False, only check PATH/override status.

        Returns:
            String describing why unavailable, or None if available

        Example:
            >>> detector = ProviderDetector(
            ...     provider_id="missing",
            ...     binary_name="nonexistent-binary",
            ... )
            >>> detector.get_unavailability_reason()
            "Binary 'nonexistent-binary' not found in PATH"
        """
        # Check environment override
        if self.override_env:
            override = _coerce_bool(os.environ.get(self.override_env))
            if override is False:
                return f"Explicitly disabled via {self.override_env}=0"
            if override is True:
                return None  # Available via override

        # Check test mode
        if _is_test_mode():
            return f"Test mode enabled ({_TEST_MODE_ENV}=1) and no override set"

        # Check binary resolution
        executable = self.resolve_binary()
        if not executable:
            return f"Binary '{self.binary_name}' not found in PATH"

        # Check probe if requested
        if use_probe and not self._run_probe(executable):
            probe_cmd = f"{executable} {' '.join(self.probe_args)}"
            return f"Health probe failed: {probe_cmd}"

        return None  # Available


# =============================================================================
# Default Provider Detectors
# =============================================================================

_DEFAULT_DETECTORS: tuple[ProviderDetector, ...] = (
    ProviderDetector(
        provider_id="gemini",
        binary_name="gemini",
        override_env="FOUNDRY_GEMINI_AVAILABLE_OVERRIDE",
        binary_env="FOUNDRY_GEMINI_BINARY",
        probe_args=("--help",),
    ),
    ProviderDetector(
        provider_id="codex",
        binary_name="codex",
        override_env="FOUNDRY_CODEX_AVAILABLE_OVERRIDE",
        binary_env="FOUNDRY_CODEX_BINARY",
        probe_args=("--version",),
    ),
    ProviderDetector(
        provider_id="cursor-agent",
        binary_name="cursor-agent",
        override_env="FOUNDRY_CURSOR_AGENT_AVAILABLE_OVERRIDE",
        binary_env="FOUNDRY_CURSOR_AGENT_BINARY",
        probe_args=("--version",),
    ),
    ProviderDetector(
        provider_id="claude",
        binary_name="claude",
        override_env="FOUNDRY_CLAUDE_AVAILABLE_OVERRIDE",
        binary_env="FOUNDRY_CLAUDE_BINARY",
        probe_args=("--version",),
    ),
    ProviderDetector(
        provider_id="opencode",
        binary_name="opencode",
        override_env="FOUNDRY_OPENCODE_AVAILABLE_OVERRIDE",
        binary_env="FOUNDRY_OPENCODE_BINARY",
        probe_args=("--version",),
    ),
)

# Global detector registry
_DETECTORS: Dict[str, ProviderDetector] = {}


def _reset_default_detectors() -> None:
    """Reset the detector registry to default detectors."""
    _DETECTORS.clear()
    for detector in _DEFAULT_DETECTORS:
        _DETECTORS[detector.provider_id] = detector


# =============================================================================
# Public API
# =============================================================================


def register_detector(detector: ProviderDetector, *, replace: bool = False) -> None:
    """
    Register a detector configuration.

    Args:
        detector: ProviderDetector instance to register
        replace: If True, overwrite existing registration. If False (default),
            raise ValueError if provider_id already exists.

    Raises:
        ValueError: If provider_id already registered and replace=False

    Example:
        >>> custom_detector = ProviderDetector(
        ...     provider_id="my-provider",
        ...     binary_name="my-cli",
        ... )
        >>> register_detector(custom_detector)
    """
    if detector.provider_id in _DETECTORS and not replace:
        raise ValueError(f"Detector for '{detector.provider_id}' already exists")
    _DETECTORS[detector.provider_id] = detector
    logger.debug("Registered detector for provider '%s'", detector.provider_id)


def get_detector(provider_id: str) -> Optional[ProviderDetector]:
    """
    Return the detector for a provider ID.

    Args:
        provider_id: Provider identifier (e.g., "gemini", "codex")

    Returns:
        ProviderDetector if registered, None otherwise
    """
    return _DETECTORS.get(provider_id)


def detect_provider_availability(provider_id: str, *, use_probe: bool = True) -> bool:
    """
    Check whether a provider is available.

    Args:
        provider_id: Provider identifier (e.g., "gemini", "codex", "cursor-agent")
        use_probe: When True, run health probe. When False, only check PATH.

    Returns:
        True if provider is available, False otherwise

    Raises:
        KeyError: If no detector registered for provider_id

    Example:
        >>> detect_provider_availability("gemini")
        True
        >>> detect_provider_availability("nonexistent")
        KeyError: "No detector registered for provider 'nonexistent'"
    """
    detector = get_detector(provider_id)
    if detector is None:
        raise KeyError(f"No detector registered for provider '{provider_id}'")
    return detector.is_available(use_probe=use_probe)


def get_provider_statuses(*, use_probe: bool = True) -> Dict[str, bool]:
    """
    Return availability map for all registered detectors.

    Args:
        use_probe: When True, run health probes. When False, only check PATH.

    Returns:
        Dict mapping provider_id to availability boolean

    Example:
        >>> get_provider_statuses()
        {'gemini': True, 'codex': False, 'cursor-agent': True, 'claude': True, 'opencode': False}
    """
    return {
        provider_id: detector.is_available(use_probe=use_probe)
        for provider_id, detector in _DETECTORS.items()
    }


def get_provider_unavailability_reasons(
    *, use_probe: bool = True
) -> Dict[str, Optional[str]]:
    """
    Return unavailability reasons for all registered detectors.

    Useful for diagnostic messages when providers cannot be used.
    Available providers have None as their reason.

    Args:
        use_probe: When True, include probe failures in reasons.
            When False, only check PATH/override status.

    Returns:
        Dict mapping provider_id to reason string (None if available)

    Example:
        >>> get_provider_unavailability_reasons()
        {
            'gemini': None,  # available
            'codex': "Binary 'codex' not found in PATH",
            'cursor-agent': None,  # available
            'claude': None,  # available
            'opencode': "Health probe failed: /usr/bin/opencode --version"
        }
    """
    return {
        provider_id: detector.get_unavailability_reason(use_probe=use_probe)
        for provider_id, detector in _DETECTORS.items()
    }


def list_detectors() -> Iterable[ProviderDetector]:
    """
    Return all registered detector configurations.

    Returns:
        Tuple of registered ProviderDetector instances

    Example:
        >>> for detector in list_detectors():
        ...     print(detector.provider_id)
        gemini
        codex
        cursor-agent
        claude
        opencode
    """
    return tuple(_DETECTORS.values())


def reset_detectors() -> None:
    """
    Reset detectors to the default set.

    Primarily used by tests to restore a clean state.
    """
    _reset_default_detectors()


# Initialize with default detectors
_reset_default_detectors()


__all__ = [
    "ProviderDetector",
    "register_detector",
    "get_detector",
    "detect_provider_availability",
    "get_provider_statuses",
    "get_provider_unavailability_reasons",
    "list_detectors",
    "reset_detectors",
]
