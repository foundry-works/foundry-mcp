"""Preflight token validation before provider dispatch.

Provides:
    - PreflightResult: Validation result dataclass
    - preflight_count(): Validate single payload against budget
    - preflight_count_multiple(): Validate multiple payloads against budget
    - get_provider_model_from_spec(): Parse provider specification strings
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

from .budget import TokenBudget
from .estimation import estimate_tokens

logger = logging.getLogger(__name__)


@dataclass
class PreflightResult:
    """Result of preflight token validation.

    Contains validation status and detailed token counts for debugging
    and adjustment decisions.

    Attributes:
        valid: Whether the payload fits within the budget
        estimated_tokens: Estimated token count for the payload
        effective_budget: Effective budget after reservations and safety margin
        remaining_tokens: Tokens remaining after this payload (if valid)
        overflow_tokens: Tokens over budget (if invalid), 0 otherwise
        is_final_fit: Whether this was a final-fit revalidation

    Example:
        result = preflight_count(payload, budget)
        if not result.valid:
            print(f"Payload exceeds budget by {result.overflow_tokens} tokens")
            # Try reducing payload size
    """

    valid: bool
    estimated_tokens: int
    effective_budget: int
    remaining_tokens: int
    overflow_tokens: int
    is_final_fit: bool = False

    def __post_init__(self) -> None:
        """Validate result consistency."""
        if self.estimated_tokens < 0:
            raise ValueError(f"estimated_tokens must be non-negative, got {self.estimated_tokens}")
        if self.effective_budget < 0:
            raise ValueError(f"effective_budget must be non-negative, got {self.effective_budget}")

    @property
    def usage_fraction(self) -> float:
        """Calculate what fraction of budget this payload would use.

        Returns:
            Fraction of effective budget used (0.0 to 1.0+)
        """
        if self.effective_budget <= 0:
            return 1.0 if self.estimated_tokens > 0 else 0.0
        return self.estimated_tokens / self.effective_budget

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dict representation of the result
        """
        return {
            "valid": self.valid,
            "estimated_tokens": self.estimated_tokens,
            "effective_budget": self.effective_budget,
            "remaining_tokens": self.remaining_tokens,
            "overflow_tokens": self.overflow_tokens,
            "is_final_fit": self.is_final_fit,
            "usage_fraction": self.usage_fraction,
        }


def preflight_count(
    content: str,
    budget: TokenBudget,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    is_final_fit: bool = False,
    warn_on_heuristic: bool = True,
) -> PreflightResult:
    """Validate payload size against token budget before provider dispatch.

    Estimates tokens in the content and checks if it fits within the
    budget's remaining capacity. Use is_final_fit=True for revalidation
    after content adjustments.

    Args:
        content: Text content to validate
        budget: TokenBudget to validate against
        provider: Optional provider for estimation accuracy
        model: Optional model for estimation accuracy
        is_final_fit: True if this is a final revalidation after adjustments
        warn_on_heuristic: Emit warning when using heuristic estimation

    Returns:
        PreflightResult with validation status and token counts

    Example:
        budget = TokenBudget(total_budget=100_000, reserved_output=8_000)

        # Initial preflight check
        result = preflight_count(payload, budget, provider="claude")
        if not result.valid:
            # Adjust payload...
            adjusted_payload = truncate(payload, result.effective_budget)

            # Final-fit revalidation
            result = preflight_count(
                adjusted_payload, budget,
                provider="claude",
                is_final_fit=True
            )
            if not result.valid:
                raise TokenBudgetExceeded(result.overflow_tokens)

        # Proceed with dispatch
        budget.allocate(result.estimated_tokens)
    """
    # Estimate tokens in content
    estimated = estimate_tokens(
        content,
        provider=provider,
        model=model,
        warn_on_heuristic=warn_on_heuristic,
    )

    # Get remaining budget capacity
    remaining = budget.remaining()
    effective = budget.effective_budget()

    # Check if content fits
    valid = estimated <= remaining
    overflow = max(0, estimated - remaining) if not valid else 0
    remaining_after = max(0, remaining - estimated) if valid else 0

    result = PreflightResult(
        valid=valid,
        estimated_tokens=estimated,
        effective_budget=effective,
        remaining_tokens=remaining_after,
        overflow_tokens=overflow,
        is_final_fit=is_final_fit,
    )

    # Log validation result
    if is_final_fit:
        if valid:
            logger.debug(f"Final-fit validation passed: {estimated} tokens ({result.usage_fraction:.1%} of budget)")
        else:
            logger.warning(
                f"Final-fit validation FAILED: {estimated} tokens exceeds remaining {remaining} by {overflow}"
            )
    else:
        logger.debug(f"Preflight {'passed' if valid else 'failed'}: {estimated}/{remaining} tokens")

    return result


def preflight_count_multiple(
    contents: list[str],
    budget: TokenBudget,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    warn_on_heuristic: bool = True,
) -> tuple[bool, list[int], int]:
    """Validate multiple payloads against token budget.

    Estimates tokens for each content item and checks if the total
    fits within the budget. Useful for batching multiple items.

    Args:
        contents: List of text content to validate
        budget: TokenBudget to validate against
        provider: Optional provider for estimation accuracy
        model: Optional model for estimation accuracy
        warn_on_heuristic: Emit warning when using heuristic estimation

    Returns:
        Tuple of (valid, token_counts, total_tokens) where:
        - valid: Whether all content fits within remaining budget
        - token_counts: List of estimated tokens per content item
        - total_tokens: Sum of all token estimates

    Example:
        items = ["first item", "second item", "third item"]
        valid, counts, total = preflight_count_multiple(items, budget)
        if valid:
            for item, count in zip(items, counts):
                budget.allocate(count)
    """
    if not contents:
        return True, [], 0

    # Estimate each content item (only warn once for first heuristic use)
    token_counts = []
    for i, content in enumerate(contents):
        count = estimate_tokens(
            content,
            provider=provider,
            model=model,
            warn_on_heuristic=warn_on_heuristic and i == 0,
        )
        token_counts.append(count)

    total = sum(token_counts)
    valid = total <= budget.remaining()

    logger.debug(
        f"Preflight batch {'passed' if valid else 'failed'}: "
        f"{total}/{budget.remaining()} tokens across {len(contents)} items"
    )

    return valid, token_counts, total


def get_provider_model_from_spec(provider_spec: str) -> tuple[str, Optional[str]]:
    """Parse a provider specification into provider and model components.

    Supports formats:
    - "provider" -> ("provider", None)
    - "provider:model" -> ("provider", "model")
    - "[cli]provider:model" -> ("provider", "model")

    Args:
        provider_spec: Provider specification string

    Returns:
        Tuple of (provider, model) where model may be None

    Example:
        >>> get_provider_model_from_spec("claude")
        ("claude", None)
        >>> get_provider_model_from_spec("gemini:flash")
        ("gemini", "flash")
        >>> get_provider_model_from_spec("[cli]claude:opus")
        ("claude", "opus")
    """
    # Strip CLI prefix if present
    spec = provider_spec
    if spec.startswith("[") and "]" in spec:
        spec = spec.split("]", 1)[1]

    # Split on colon for model
    if ":" in spec:
        provider, model = spec.split(":", 1)
        return provider.strip(), model.strip() if model else None

    return spec.strip(), None
