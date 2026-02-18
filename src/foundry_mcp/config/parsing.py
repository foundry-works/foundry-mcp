"""Parsing and normalization helpers for configuration values.

Provides boolean parsing, provider spec parsing,
and commit cadence normalization used by other config sub-modules.
"""

import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

_VALID_COMMIT_CADENCE = {"manual", "task", "phase"}


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


