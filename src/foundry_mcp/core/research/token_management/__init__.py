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

from .budget import TokenBudget
from .estimation import (
    TokenCountEstimateWarning,
    _get_cached_encoding,
    _PROVIDER_TOKENIZERS,
    _TIKTOKEN_AVAILABLE,
    clear_token_cache,
    estimate_tokens,
    get_cache_stats,
    register_provider_tokenizer,
)
from .limits import get_effective_context, get_model_limits
from .models import BudgetingMode, ModelContextLimits
from .preflight import (
    PreflightResult,
    get_provider_model_from_spec,
    preflight_count,
    preflight_count_multiple,
)
from .registry import DEFAULT_MODEL_LIMITS

__all__ = [
    # Models
    "BudgetingMode",
    "ModelContextLimits",
    # Registry
    "DEFAULT_MODEL_LIMITS",
    # Limits
    "get_model_limits",
    "get_effective_context",
    # Budget
    "TokenBudget",
    # Estimation
    "TokenCountEstimateWarning",
    "estimate_tokens",
    "clear_token_cache",
    "get_cache_stats",
    "register_provider_tokenizer",
    "_get_cached_encoding",
    "_PROVIDER_TOKENIZERS",
    "_TIKTOKEN_AVAILABLE",
    # Preflight
    "PreflightResult",
    "preflight_count",
    "preflight_count_multiple",
    "get_provider_model_from_spec",
]
