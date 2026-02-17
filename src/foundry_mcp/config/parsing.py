"""Parsing and normalization helpers for configuration values.

Provides boolean parsing, provider spec parsing, feature flag parsing,
and commit cadence normalization used by other config sub-modules.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_VALID_COMMIT_CADENCE = {"manual", "task", "phase"}
_FEATURE_FLAG_ENV_VAR = "FOUNDRY_MCP_FEATURE_FLAGS"
_FEATURE_FLAG_ENV_PREFIX = "FOUNDRY_MCP_FEATURE_FLAG_"
_FEATURE_FLAG_DEPENDENCIES: Dict[str, Tuple[str, ...]] = {
    "autonomy_fidelity_gates": ("autonomy_sessions",),
}


def _normalize_commit_cadence(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in _VALID_COMMIT_CADENCE:
        logger.warning(
            "Invalid commit cadence '%s'. Falling back to 'manual'. Valid options: %s",
            value,
            ", ".join(sorted(_VALID_COMMIT_CADENCE)),
        )
        return "manual"
    return normalized


def _parse_provider_spec(spec: str) -> Tuple[str, Optional[str]]:
    """Parse a provider specification into (provider_id, model).

    Supports both simple names and ProviderSpec bracket notation:
    - "gemini" -> ("gemini", None)
    - "[cli]gemini:pro" -> ("gemini", "pro")
    - "[cli]opencode:openai/gpt-5.2" -> ("opencode", "openai/gpt-5.2")
    - "[api]openai/gpt-4.1" -> ("openai", "gpt-4.1")

    Args:
        spec: Provider specification string

    Returns:
        Tuple of (provider_id, model) where model may be None
    """
    spec = spec.strip()

    # Simple name (no brackets) - backward compatible
    if not spec.startswith("["):
        return (spec, None)

    # Try to parse with ProviderSpec
    try:
        from foundry_mcp.core.llm_config.provider_spec import ProviderSpec

        parsed = ProviderSpec.parse(spec)
        # Build model string with backend routing if present
        model = None
        if parsed.backend and parsed.model:
            model = f"{parsed.backend}/{parsed.model}"
        elif parsed.model:
            model = parsed.model
        return (parsed.provider, model)
    except (ValueError, ImportError) as e:
        logger.warning("Failed to parse provider spec '%s': %s", spec, e)
        # Fall back to treating as simple name (strip brackets)
        return (spec.split("]")[-1].split(":")[0], None)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def _try_parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    return None


def _normalize_feature_flag_name(raw_name: str) -> str:
    """Normalize feature flag names across TOML and env inputs."""
    return raw_name.strip().lower().replace("-", "_")


def _parse_feature_flags_mapping(
    mapping: Dict[Any, Any], *, source: str
) -> Tuple[Dict[str, bool], List[str]]:
    """Parse a feature-flag mapping and return (flags, warnings)."""
    flags: Dict[str, bool] = {}
    warnings: List[str] = []

    for raw_name, raw_value in mapping.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            warnings.append(
                f"Ignoring feature flag with invalid name from {source}: {raw_name!r}"
            )
            continue

        name = _normalize_feature_flag_name(raw_name)
        parsed = _try_parse_bool(raw_value)
        if parsed is None:
            warnings.append(
                f"Ignoring feature flag '{name}' from {source}: value must be boolean-compatible, got {raw_value!r}"
            )
            continue

        flags[name] = parsed

    return flags, warnings


def _parse_feature_flags_env(
    raw_env_value: str,
) -> Tuple[Dict[str, bool], List[str]]:
    """Parse FOUNDRY_MCP_FEATURE_FLAGS from env.

    Accepted formats:
    - "autonomy_sessions" (enables flag)
    - "autonomy_sessions=true,autonomy_fidelity_gates=false"
    """
    flags: Dict[str, bool] = {}
    warnings: List[str] = []

    for token in raw_env_value.split(","):
        entry = token.strip()
        if not entry:
            continue

        if "=" in entry:
            raw_name, raw_value = entry.split("=", 1)
            name = _normalize_feature_flag_name(raw_name)
            if not name:
                warnings.append(
                    f"Ignoring malformed feature flag entry in {_FEATURE_FLAG_ENV_VAR}: {entry!r}"
                )
                continue
            parsed = _try_parse_bool(raw_value)
            if parsed is None:
                warnings.append(
                    f"Ignoring feature flag '{name}' in {_FEATURE_FLAG_ENV_VAR}: expected true/false value, got {raw_value!r}"
                )
                continue
            flags[name] = parsed
            continue

        name = _normalize_feature_flag_name(entry)
        if not name:
            warnings.append(
                f"Ignoring malformed feature flag entry in {_FEATURE_FLAG_ENV_VAR}: {entry!r}"
            )
            continue
        flags[name] = True

    return flags, warnings
