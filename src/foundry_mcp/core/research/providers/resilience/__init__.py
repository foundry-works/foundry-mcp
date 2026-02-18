"""Provider resilience configuration and error classification.

Centralized resilience utilities for search providers including:
- Per-provider configuration for rate limiting, retries, and circuit breakers
- Error classification for unified retry/circuit-breaker decisions
- ProviderResilienceManager singleton for state management
"""

from foundry_mcp.core.errors.resilience import (
    RateLimitWaitError,
    TimeBudgetExceededError,
)
from foundry_mcp.core.research.providers.resilience.config import (
    PROVIDER_CONFIGS,
    get_provider_config,
)
from foundry_mcp.core.research.providers.resilience.execution import (
    _default_classify_error,
    execute_with_resilience,
)
from foundry_mcp.core.research.providers.resilience.manager import (
    ProviderResilienceManager,
    get_resilience_manager,
    reset_resilience_manager_for_testing,
)
from foundry_mcp.core.research.providers.resilience.models import (
    ErrorClassification,
    ErrorType,
    ProviderResilienceConfig,
    ProviderStatus,
    SleepFunc,
)
from foundry_mcp.core.research.providers.resilience.retry import (
    async_retry_with_backoff,
)

__all__ = [
    # Models & enums
    "ErrorType",
    "ProviderResilienceConfig",
    "ErrorClassification",
    "ProviderStatus",
    "SleepFunc",
    # Config
    "PROVIDER_CONFIGS",
    "get_provider_config",
    # Manager
    "ProviderResilienceManager",
    "get_resilience_manager",
    "reset_resilience_manager_for_testing",
    # Retry
    "async_retry_with_backoff",
    # Execution
    "execute_with_resilience",
    "_default_classify_error",
    # Error re-exports
    "RateLimitWaitError",
    "TimeBudgetExceededError",
]
