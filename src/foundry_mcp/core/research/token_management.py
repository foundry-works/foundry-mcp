"""Token management utilities for deep research workflows.

Provides centralized token budget calculations, model context limits,
and token estimation for managing content fidelity in token-constrained
environments.

Key Components:
    - ModelContextLimits: Dataclass defining model token constraints
    - BudgetingMode: Enum for input-only vs combined budgeting strategies
    - TokenBudget: Mutable budget tracker with allocation and safety margins
    - DEFAULT_MODEL_LIMITS: Pre-configured limits for common providers/models
    - get_model_limits(): Resolve limits with config override support
    - get_effective_context(): Calculate available context after reservations
    - estimate_tokens(): Token estimation with fallback chain and caching
    - preflight_count(): Validate payload size before provider dispatch

Usage:
    from foundry_mcp.core.research.token_management import (
        get_model_limits,
        get_effective_context,
        estimate_tokens,
        BudgetingMode,
        TokenBudget,
    )

    # Get limits for a specific model
    limits = get_model_limits("claude", "opus")

    # Calculate effective context for input
    effective = get_effective_context(limits, output_budget=4000)

    # Track token usage with safety margin
    budget = TokenBudget(total_budget=100_000, reserved_output=8_000)
    if budget.can_fit(5_000):
        budget.allocate(5_000)

    # Estimate tokens in content (with caching)
    tokens = estimate_tokens("Hello, world!", provider="claude")
"""

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Optional
import hashlib
import logging
import threading
import warnings

logger = logging.getLogger(__name__)

# Optional tiktoken import for accurate token counting
try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None  # type: ignore
    _TIKTOKEN_AVAILABLE = False


class BudgetingMode(str, Enum):
    """Token budgeting strategies for different model architectures.

    Different models handle input/output token budgets differently:
    - INPUT_ONLY: Context window is for input only; output is separate
      (e.g., Claude, GPT-4). Use full context_window for input.
    - COMBINED: Context window includes both input and output
      (e.g., some Gemini modes). Must reserve space for output.

    The budgeting mode affects how get_effective_context() calculates
    available input space.
    """

    INPUT_ONLY = "input_only"
    COMBINED = "combined"


@dataclass(frozen=True)
class ModelContextLimits:
    """Token limits for a specific model.

    Defines the token constraints for a model including context window size,
    maximum output tokens, and how to budget between input and output.

    Attributes:
        context_window: Maximum input context tokens the model accepts
        max_output_tokens: Maximum tokens the model can generate in output
        budgeting_mode: How to allocate tokens between input and output
        output_reserved: Tokens to reserve for output when mode is COMBINED

    Example:
        # Claude Opus limits
        limits = ModelContextLimits(
            context_window=200_000,
            max_output_tokens=32_000,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        )
    """

    context_window: int
    max_output_tokens: int
    budgeting_mode: BudgetingMode = BudgetingMode.INPUT_ONLY
    output_reserved: int = 0

    def __post_init__(self) -> None:
        """Validate limits after initialization."""
        if self.context_window <= 0:
            raise ValueError(f"context_window must be positive, got {self.context_window}")
        if self.max_output_tokens <= 0:
            raise ValueError(f"max_output_tokens must be positive, got {self.max_output_tokens}")
        if self.output_reserved < 0:
            raise ValueError(f"output_reserved must be non-negative, got {self.output_reserved}")
        if self.budgeting_mode == BudgetingMode.COMBINED:
            if self.output_reserved > self.context_window:
                raise ValueError(
                    f"output_reserved ({self.output_reserved}) cannot exceed "
                    f"context_window ({self.context_window})"
                )


# =============================================================================
# Default Model Limits Registry
# =============================================================================

# Conservative fallback for unknown models
_DEFAULT_FALLBACK = ModelContextLimits(
    context_window=128_000,
    max_output_tokens=8_000,
    budgeting_mode=BudgetingMode.INPUT_ONLY,
)

# Default limits by provider and model
# Format: {provider: {model: ModelContextLimits}}
DEFAULT_MODEL_LIMITS: dict[str, dict[str, ModelContextLimits]] = {
    # Anthropic Claude models
    "claude": {
        "opus": ModelContextLimits(
            context_window=200_000,
            max_output_tokens=32_000,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        ),
        "sonnet": ModelContextLimits(
            context_window=200_000,
            max_output_tokens=16_000,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        ),
        "haiku": ModelContextLimits(
            context_window=200_000,
            max_output_tokens=8_000,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        ),
        # Default for claude provider without specific model
        "_default": ModelContextLimits(
            context_window=200_000,
            max_output_tokens=16_000,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        ),
    },
    # Google Gemini models
    "gemini": {
        "flash": ModelContextLimits(
            context_window=1_000_000,
            max_output_tokens=8_192,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        ),
        "pro": ModelContextLimits(
            context_window=2_000_000,
            max_output_tokens=8_192,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        ),
        # Gemini 2.0 variants
        "2.0-flash": ModelContextLimits(
            context_window=1_000_000,
            max_output_tokens=8_192,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        ),
        "_default": ModelContextLimits(
            context_window=1_000_000,
            max_output_tokens=8_192,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        ),
    },
    # OpenAI Codex models (hypothetical future models)
    "codex": {
        "gpt-5.2-codex": ModelContextLimits(
            context_window=256_000,
            max_output_tokens=32_000,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        ),
        "gpt-4.1": ModelContextLimits(
            context_window=128_000,
            max_output_tokens=16_000,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        ),
        "o3": ModelContextLimits(
            context_window=200_000,
            max_output_tokens=100_000,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        ),
        "o4-mini": ModelContextLimits(
            context_window=128_000,
            max_output_tokens=65_536,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        ),
        "_default": ModelContextLimits(
            context_window=128_000,
            max_output_tokens=16_000,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        ),
    },
    # Cursor Agent (IDE integration)
    "cursor-agent": {
        "_default": ModelContextLimits(
            context_window=128_000,
            max_output_tokens=16_000,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        ),
    },
    # OpenCode provider
    "opencode": {
        "_default": ModelContextLimits(
            context_window=128_000,
            max_output_tokens=16_000,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        ),
    },
}


# =============================================================================
# Limit Resolution Functions
# =============================================================================


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
            logger.debug(
                f"No limits found for {provider}:{model}, using global fallback"
            )
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


@dataclass
class TokenBudget:
    """Tracks token budget allocation and usage for a workflow.

    Provides methods to check available budget, allocate tokens, and
    track usage with a configurable safety margin.

    Attributes:
        total_budget: Total tokens available for the workflow
        reserved_output: Tokens reserved for output generation
        safety_margin: Fraction of budget to keep as buffer (0.0-1.0)
        used_tokens: Tokens already consumed (mutable, updated by allocate())

    Example:
        budget = TokenBudget(
            total_budget=100_000,
            reserved_output=8_000,
            safety_margin=0.1,
        )
        # Effective budget: (100_000 - 8_000) * (1 - 0.1) = 82_800

        if budget.can_fit(5_000):
            budget.allocate(5_000)
    """

    total_budget: int
    reserved_output: int = 0
    safety_margin: float = 0.1
    used_tokens: int = 0

    def __post_init__(self) -> None:
        """Validate budget parameters after initialization."""
        if self.total_budget <= 0:
            raise ValueError(f"total_budget must be positive, got {self.total_budget}")
        if self.reserved_output < 0:
            raise ValueError(f"reserved_output must be non-negative, got {self.reserved_output}")
        if self.reserved_output >= self.total_budget:
            raise ValueError(
                f"reserved_output ({self.reserved_output}) must be less than "
                f"total_budget ({self.total_budget})"
            )
        if not 0.0 <= self.safety_margin < 1.0:
            raise ValueError(
                f"safety_margin must be in [0.0, 1.0), got {self.safety_margin}"
            )
        if self.used_tokens < 0:
            raise ValueError(f"used_tokens must be non-negative, got {self.used_tokens}")

    def effective_budget(self) -> int:
        """Calculate the effective budget after reservations and safety margin.

        The effective budget is:
            (total_budget - reserved_output) * (1 - safety_margin)

        Returns:
            Effective token budget available for allocation
        """
        available = self.total_budget - self.reserved_output
        return int(available * (1.0 - self.safety_margin))

    def remaining(self) -> int:
        """Calculate remaining tokens available for allocation.

        Returns:
            Tokens remaining (effective_budget - used_tokens), minimum 0
        """
        return max(0, self.effective_budget() - self.used_tokens)

    def can_fit(self, tokens: int) -> bool:
        """Check if a given number of tokens can fit in the remaining budget.

        Args:
            tokens: Number of tokens to check

        Returns:
            True if tokens fit within remaining budget, False otherwise
        """
        if tokens < 0:
            raise ValueError(f"tokens must be non-negative, got {tokens}")
        return tokens <= self.remaining()

    def allocate(self, tokens: int) -> bool:
        """Allocate tokens from the budget.

        Attempts to allocate the specified tokens. If successful, updates
        used_tokens and returns True. If insufficient budget, returns False
        without modifying state.

        Args:
            tokens: Number of tokens to allocate

        Returns:
            True if allocation succeeded, False if insufficient budget

        Raises:
            ValueError: If tokens is negative
        """
        if tokens < 0:
            raise ValueError(f"tokens must be non-negative, got {tokens}")
        if not self.can_fit(tokens):
            logger.debug(
                f"Token allocation failed: requested {tokens}, "
                f"remaining {self.remaining()}"
            )
            return False
        self.used_tokens += tokens
        return True

    def usage_fraction(self) -> float:
        """Calculate the fraction of effective budget used.

        Returns:
            Fraction of budget used (0.0 to 1.0+)
        """
        effective = self.effective_budget()
        if effective <= 0:
            return 1.0 if self.used_tokens > 0 else 0.0
        return self.used_tokens / effective


# =============================================================================
# Token Estimation
# =============================================================================

# Cache for token estimates: maps (content_hash, provider) -> token_count
_TOKEN_ESTIMATE_CACHE: dict[tuple[str, str], int] = {}
_TOKEN_CACHE_LOCK = threading.Lock()

# Maximum cache size to prevent unbounded memory growth
_MAX_CACHE_SIZE = 10_000

# Warning category for heuristic fallback
class TokenCountEstimateWarning(UserWarning):
    """Warning emitted when using character-based heuristic for token estimation."""
    pass


# Provider-specific tokenizer factories (for future extension)
_PROVIDER_TOKENIZERS: dict[str, Callable[[str], int]] = {}


def register_provider_tokenizer(provider: str, tokenizer: Callable[[str], int]) -> None:
    """Register a provider-specific tokenizer function.

    Args:
        provider: Provider identifier (e.g., "claude", "gemini")
        tokenizer: Function that takes content string and returns token count

    Example:
        def my_tokenizer(content: str) -> int:
            return len(my_api.count_tokens(content))
        register_provider_tokenizer("my_provider", my_tokenizer)
    """
    _PROVIDER_TOKENIZERS[provider.lower()] = tokenizer


def _content_hash(content: str) -> str:
    """Generate a hash of content for cache keying.

    Uses SHA-256 truncated to 16 characters for reasonable uniqueness
    while keeping cache keys compact.

    Args:
        content: Text content to hash

    Returns:
        Hex string hash of the content
    """
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:16]


@lru_cache(maxsize=32)
def _get_cached_encoding(model_name: str) -> Any:
    """Get a cached tiktoken encoding for the given model name.

    Uses lru_cache to avoid repeated encoding lookups, which can be
    expensive as tiktoken loads encoding data from disk.

    Args:
        model_name: Model name to get encoding for, or "" for default cl100k_base

    Returns:
        tiktoken Encoding object

    Raises:
        RuntimeError: If tiktoken is not available
    """
    if not _TIKTOKEN_AVAILABLE or tiktoken is None:
        raise RuntimeError("tiktoken is not available")

    if model_name:
        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Model not found, fall back to cl100k_base (GPT-4/Claude-like)
            return tiktoken.get_encoding("cl100k_base")
    else:
        # Default to cl100k_base for modern models
        return tiktoken.get_encoding("cl100k_base")


def _estimate_with_tiktoken(content: str, model: Optional[str] = None) -> Optional[int]:
    """Attempt to estimate tokens using tiktoken.

    Args:
        content: Text to estimate
        model: Optional model name for encoding selection

    Returns:
        Token count if tiktoken available and successful, None otherwise
    """
    if not _TIKTOKEN_AVAILABLE or tiktoken is None:
        return None

    try:
        encoding = _get_cached_encoding(model or "")
        return len(encoding.encode(content))
    except Exception as e:
        logger.debug(f"tiktoken estimation failed: {e}")
        return None


def _estimate_heuristic(content: str) -> int:
    """Estimate tokens using character-based heuristic.

    Uses the common approximation of ~4 characters per token for
    English text. This is a rough estimate and may be inaccurate
    for non-English text, code, or special characters.

    Args:
        content: Text to estimate

    Returns:
        Estimated token count (minimum 1)
    """
    # ~4 characters per token is a common approximation
    # Add 1 to handle empty strings and ensure minimum of 1
    return max(1, len(content) // 4)


def estimate_tokens(
    content: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    *,
    use_cache: bool = True,
    warn_on_heuristic: bool = True,
) -> int:
    """Estimate the token count for content.

    Uses a fallback chain for estimation:
    1. Provider-native tokenizer (if registered)
    2. tiktoken (if available)
    3. Character/4 heuristic (always available)

    Results are cached by content hash and provider for efficiency.

    Args:
        content: Text content to estimate tokens for
        provider: Optional provider for provider-specific estimation
        model: Optional model for model-specific estimation
        use_cache: Whether to use/update the cache (default True)
        warn_on_heuristic: Emit warning when falling back to heuristic (default True)

    Returns:
        Estimated token count (minimum 1)

    Warns:
        TokenCountEstimateWarning: When using character-based heuristic fallback

    Example:
        # Basic usage
        tokens = estimate_tokens("Hello, world!")

        # With provider context
        tokens = estimate_tokens(long_content, provider="claude", model="opus")

        # Disable caching for one-off estimates
        tokens = estimate_tokens(content, use_cache=False)
    """
    if not content:
        return 0

    provider_key = (provider or "").lower()
    cache_key = (_content_hash(content), provider_key)

    # Check cache first
    if use_cache:
        with _TOKEN_CACHE_LOCK:
            if cache_key in _TOKEN_ESTIMATE_CACHE:
                return _TOKEN_ESTIMATE_CACHE[cache_key]

    estimate: Optional[int] = None

    # Try provider-native tokenizer first
    if provider_key and provider_key in _PROVIDER_TOKENIZERS:
        try:
            estimate = _PROVIDER_TOKENIZERS[provider_key](content)
            logger.debug(f"Used provider-native tokenizer for {provider_key}")
        except Exception as e:
            logger.debug(f"Provider tokenizer failed for {provider_key}: {e}")

    # Try tiktoken if provider-native didn't work
    if estimate is None:
        estimate = _estimate_with_tiktoken(content, model)
        if estimate is not None:
            logger.debug("Used tiktoken for token estimation")

    # Fall back to heuristic
    if estimate is None:
        estimate = _estimate_heuristic(content)
        logger.debug("Used character heuristic for token estimation")

        if warn_on_heuristic:
            warnings.warn(
                "TOKEN_COUNT_ESTIMATE_USED: Using character-based heuristic for token "
                f"estimation (provider={provider or 'unknown'}). Install tiktoken for "
                "more accurate counts.",
                TokenCountEstimateWarning,
                stacklevel=2,
            )

    # Update cache (with size limit)
    if use_cache:
        with _TOKEN_CACHE_LOCK:
            if len(_TOKEN_ESTIMATE_CACHE) >= _MAX_CACHE_SIZE:
                # Simple eviction: clear half the cache
                keys_to_remove = list(_TOKEN_ESTIMATE_CACHE.keys())[: _MAX_CACHE_SIZE // 2]
                for key in keys_to_remove:
                    del _TOKEN_ESTIMATE_CACHE[key]
            _TOKEN_ESTIMATE_CACHE[cache_key] = estimate

    return estimate


def clear_token_cache() -> int:
    """Clear the token estimation cache.

    Returns:
        Number of entries cleared

    Example:
        cleared = clear_token_cache()
        print(f"Cleared {cleared} cached estimates")
    """
    with _TOKEN_CACHE_LOCK:
        count = len(_TOKEN_ESTIMATE_CACHE)
        _TOKEN_ESTIMATE_CACHE.clear()
    return count


def get_cache_stats() -> dict[str, int]:
    """Get statistics about the token estimation cache.

    Returns:
        Dict with 'size' and 'max_size' keys

    Example:
        stats = get_cache_stats()
        print(f"Cache: {stats['size']}/{stats['max_size']} entries")
    """
    with _TOKEN_CACHE_LOCK:
        size = len(_TOKEN_ESTIMATE_CACHE)
    return {
        "size": size,
        "max_size": _MAX_CACHE_SIZE,
    }


# =============================================================================
# Preflight Validation
# =============================================================================


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
            logger.debug(
                f"Final-fit validation passed: {estimated} tokens "
                f"({result.usage_fraction:.1%} of budget)"
            )
        else:
            logger.warning(
                f"Final-fit validation FAILED: {estimated} tokens exceeds "
                f"remaining {remaining} by {overflow}"
            )
    else:
        logger.debug(
            f"Preflight {'passed' if valid else 'failed'}: "
            f"{estimated}/{remaining} tokens"
        )

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
