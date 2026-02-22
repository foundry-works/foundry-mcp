"""Resilience data models, enums, and protocols.

Defines the core types used across the resilience sub-package:
- ErrorType enum for error classification
- ProviderResilienceConfig for per-provider tuning
- ErrorClassification for retry/circuit-breaker decisions
- ProviderStatus for observability
- SleepFunc protocol for injectable async sleep
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol


class ErrorType(str, Enum):
    """Classification of error types for resilience decisions."""

    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    QUOTA_EXCEEDED = "quota_exceeded"
    NETWORK = "network"
    INVALID_REQUEST = "invalid_request"
    UNKNOWN = "unknown"


@dataclass
class ProviderResilienceConfig:
    """Per-provider resilience configuration.

    Configures rate limiting, retry behavior, and circuit breaker thresholds.
    """

    # Rate limiting
    requests_per_second: float = 1.0
    burst_limit: int = 3

    # Retry behavior
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: float = 0.5  # Fractional jitter range around base delay (0.0-1.0, 0.5 => 50-150%)

    # Circuit breaker
    circuit_failure_threshold: int = 5
    circuit_recovery_timeout: float = 30.0


@dataclass
class ErrorClassification:
    """Classification result for an error.

    Determines how the resilience layer should handle a specific error.
    """

    retryable: bool
    trips_breaker: bool
    backoff_seconds: Optional[float] = None
    error_type: ErrorType = ErrorType.UNKNOWN


@dataclass
class ProviderStatus:
    """Status of a provider's resilience components."""

    provider_name: str
    is_available: bool
    circuit_state: str
    circuit_failure_count: int
    rate_limit_remaining: int
    rate_limit_reset_in: float


class SleepFunc(Protocol):
    """Protocol for injectable sleep function."""

    async def __call__(self, seconds: float) -> None: ...
