"""Model limit resolution and effective context calculation.

Provides:
    - get_model_limits(): Resolve limits with config override support
    - get_effective_context(): Calculate available context after reservations
"""

import logging
from typing import Any, Optional

from .models import BudgetingMode, ModelContextLimits
from .registry import _DEFAULT_FALLBACK, DEFAULT_MODEL_LIMITS

logger = logging.getLogger(__name__)


def get_model_limits(
    provider: str,
    model: Optional[str] = None,
    *,
    config_overrides: Optional[dict[str, Any]] = None,
) -> ModelContextLimits:
    """Get token limits for a specific provider/model combination.

    Resolution order:
    1. Config overrides (if provided)
    2. Exact model match in DEFAULT_MODEL_LIMITS
    3. Provider's _default entry
    4. Global _DEFAULT_FALLBACK

    Args:
        provider: Provider identifier (e.g., "claude", "gemini", "codex")
        model: Optional model identifier (e.g., "opus", "flash", "gpt-4.1")
        config_overrides: Optional dict with context_window, max_output_tokens,
            budgeting_mode, output_reserved overrides

    Returns:
        ModelContextLimits for the specified provider/model

    Example:
        # Get Claude Opus limits
        limits = get_model_limits("claude", "opus")

        # Get Gemini limits with config override
        limits = get_model_limits(
            "gemini",
            "flash",
            config_overrides={"max_output_tokens": 4096}
        )
    """
    provider_lower = provider.lower()
    model_lower = model.lower() if model else None

    # Start with fallback
    base_limits = _DEFAULT_FALLBACK

    # Try to find provider in registry
    if provider_lower in DEFAULT_MODEL_LIMITS:
        provider_limits = DEFAULT_MODEL_LIMITS[provider_lower]

        # Try exact model match
        if model_lower and model_lower in provider_limits:
            base_limits = provider_limits[model_lower]
        # Fall back to provider default
        elif "_default" in provider_limits:
            base_limits = provider_limits["_default"]
        else:
            logger.debug(f"No limits found for {provider}:{model}, using global fallback")
    else:
        logger.debug(f"Unknown provider '{provider}', using global fallback")

    # Apply config overrides if provided
    if config_overrides:
        return _apply_overrides(base_limits, config_overrides)

    return base_limits


def _apply_overrides(
    base: ModelContextLimits,
    overrides: dict[str, Any],
) -> ModelContextLimits:
    """Apply configuration overrides to base limits.

    Args:
        base: Base ModelContextLimits to override
        overrides: Dict with optional keys: context_window, max_output_tokens,
            budgeting_mode, output_reserved

    Returns:
        New ModelContextLimits with overrides applied
    """
    context_window = overrides.get("context_window", base.context_window)
    max_output_tokens = overrides.get("max_output_tokens", base.max_output_tokens)

    # Handle budgeting_mode as string or enum
    budgeting_mode_value = overrides.get("budgeting_mode", base.budgeting_mode)
    if isinstance(budgeting_mode_value, str):
        budgeting_mode = BudgetingMode(budgeting_mode_value)
    else:
        budgeting_mode = budgeting_mode_value

    output_reserved = overrides.get("output_reserved", base.output_reserved)

    return ModelContextLimits(
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        budgeting_mode=budgeting_mode,
        output_reserved=output_reserved,
    )


def get_effective_context(
    limits: ModelContextLimits,
    output_budget: Optional[int] = None,
) -> int:
    """Calculate effective input context after output reservation.

    For INPUT_ONLY mode: Returns full context_window (output is separate)
    For COMBINED mode: Returns context_window minus output reservation

    Args:
        limits: Model limits to calculate from
        output_budget: Specific output budget to reserve (COMBINED mode only).
            If not provided, uses limits.output_reserved or limits.max_output_tokens.

    Returns:
        Effective input context in tokens

    Example:
        limits = get_model_limits("claude", "opus")
        effective = get_effective_context(limits)  # 200,000 for INPUT_ONLY

        # COMBINED mode example
        combined_limits = ModelContextLimits(
            context_window=100_000,
            max_output_tokens=8_000,
            budgeting_mode=BudgetingMode.COMBINED,
            output_reserved=8_000,
        )
        effective = get_effective_context(combined_limits)  # 92,000
    """
    if limits.budgeting_mode == BudgetingMode.INPUT_ONLY:
        # Input and output are separate pools
        return limits.context_window

    # COMBINED mode: must reserve space for output
    if output_budget is not None:
        reserved = min(output_budget, limits.context_window - 1)
    elif limits.output_reserved > 0:
        reserved = limits.output_reserved
    else:
        # Default to max_output_tokens if no explicit reservation
        reserved = min(limits.max_output_tokens, limits.context_window // 2)

    effective = limits.context_window - reserved
    return max(effective, 1)  # Ensure at least 1 token for input
