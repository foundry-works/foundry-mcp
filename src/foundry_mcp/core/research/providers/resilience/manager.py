"""ProviderResilienceManager singleton for state management.

Manages per-provider rate limiters and circuit breakers with lazy
initialization. Provides state inspection API for observability.
"""

import asyncio
import threading
from typing import Optional

from foundry_mcp.core.rate_limit import RateLimitConfig, TokenBucketLimiter
from foundry_mcp.core.resilience import CircuitBreaker, CircuitState

from foundry_mcp.core.research.providers.resilience.config import (
    PROVIDER_CONFIGS,
    get_provider_config,
)
from foundry_mcp.core.research.providers.resilience.models import (
    ProviderResilienceConfig,
    ProviderStatus,
)


class ProviderResilienceManager:
    """Singleton manager for provider resilience state.

    Manages per-provider rate limiters and circuit breakers with lazy
    initialization. Provides state inspection API for observability.

    Thread-safe via threading.Lock for sync operations and asyncio.Lock for async.
    """

    _instance: Optional["ProviderResilienceManager"] = None
    _async_lock: Optional[asyncio.Lock]
    _sync_lock: threading.Lock

    def __init__(self) -> None:
        """Initialize manager with empty state.

        Use get_resilience_manager() to get the singleton instance.
        """
        self._rate_limiters: dict[str, TokenBucketLimiter] = {}
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._async_lock = None  # Lazy-init: created on first async use
        self._sync_lock = threading.Lock()

    def _get_async_lock(self) -> asyncio.Lock:
        """Get or lazily create the async lock.

        Must be called from within a running event loop.
        """
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        return self._async_lock

    def _get_or_create_rate_limiter(
        self,
        provider_name: str,
        config: Optional[ProviderResilienceConfig] = None,
    ) -> TokenBucketLimiter:
        """Get or lazily create a rate limiter for a provider."""
        if provider_name not in self._rate_limiters:
            config = config or get_provider_config(provider_name)
            # Convert RPS to RPM for TokenBucketLimiter
            rpm = int(config.requests_per_second * 60)
            rate_config = RateLimitConfig(
                requests_per_minute=rpm,
                burst_limit=config.burst_limit,
                enabled=True,
                reason=f"{provider_name} rate limit",
            )
            self._rate_limiters[provider_name] = TokenBucketLimiter(rate_config)
        elif config is not None:
            # Update existing limiter config to honor overrides
            limiter = self._rate_limiters[provider_name]
            rpm = int(config.requests_per_second * 60)
            limiter.config.requests_per_minute = rpm
            limiter.config.burst_limit = config.burst_limit
            limiter.config.enabled = True
            limiter.config.reason = f"{provider_name} rate limit"
            limiter.state.tokens = min(
                limiter.state.tokens,
                float(limiter.config.burst_limit),
            )
        return self._rate_limiters[provider_name]

    def _get_or_create_circuit_breaker(
        self,
        provider_name: str,
        config: Optional[ProviderResilienceConfig] = None,
    ) -> CircuitBreaker:
        """Get or lazily create a circuit breaker for a provider."""
        if provider_name not in self._circuit_breakers:
            config = config or get_provider_config(provider_name)
            self._circuit_breakers[provider_name] = CircuitBreaker(
                name=provider_name,
                failure_threshold=config.circuit_failure_threshold,
                recovery_timeout=config.circuit_recovery_timeout,
            )
        elif config is not None:
            breaker = self._circuit_breakers[provider_name]
            breaker.failure_threshold = config.circuit_failure_threshold
            breaker.recovery_timeout = config.circuit_recovery_timeout
        return self._circuit_breakers[provider_name]

    def _is_breaker_available(self, breaker: CircuitBreaker) -> bool:
        """Check breaker availability without mutating probe counters."""
        return breaker.is_available()

    async def get_rate_limiter(
        self,
        provider_name: str,
        config: Optional[ProviderResilienceConfig] = None,
    ) -> TokenBucketLimiter:
        """Get rate limiter for a provider (thread-safe).

        Args:
            provider_name: Name of the provider

        Returns:
            TokenBucketLimiter for the provider
        """
        async with self._get_async_lock():
            return self._get_or_create_rate_limiter(provider_name, config=config)

    async def get_circuit_breaker(
        self,
        provider_name: str,
        config: Optional[ProviderResilienceConfig] = None,
    ) -> CircuitBreaker:
        """Get circuit breaker for a provider (thread-safe).

        Args:
            provider_name: Name of the provider

        Returns:
            CircuitBreaker for the provider
        """
        async with self._get_async_lock():
            return self._get_or_create_circuit_breaker(provider_name, config=config)

    def get_breaker_state(self, provider_name: str) -> CircuitState:
        """Get current circuit breaker state for a provider (thread-safe).

        Args:
            provider_name: Name of the provider

        Returns:
            CircuitState (CLOSED, OPEN, or HALF_OPEN)
        """
        with self._sync_lock:
            breaker = self._get_or_create_circuit_breaker(provider_name)
            return breaker.state

    def is_provider_available(self, provider_name: str) -> bool:
        """Check if a provider is available (circuit not open, thread-safe).

        Args:
            provider_name: Name of the provider

        Returns:
            True if provider can accept requests
        """
        with self._sync_lock:
            breaker = self._get_or_create_circuit_breaker(provider_name)
            return self._is_breaker_available(breaker)

    def get_provider_status(self, provider_name: str) -> ProviderStatus:
        """Get comprehensive status for a provider (thread-safe).

        Args:
            provider_name: Name of the provider

        Returns:
            ProviderStatus with circuit and rate limit state
        """
        with self._sync_lock:
            breaker = self._get_or_create_circuit_breaker(provider_name)
            limiter = self._get_or_create_rate_limiter(provider_name)
            rate_check = limiter.check()
            is_available = self._is_breaker_available(breaker)

            return ProviderStatus(
                provider_name=provider_name,
                is_available=is_available,
                circuit_state=breaker.state.value,
                circuit_failure_count=breaker.failure_count,
                rate_limit_remaining=rate_check.remaining,
                rate_limit_reset_in=rate_check.reset_in,
            )

    def get_all_provider_statuses(self) -> dict[str, ProviderStatus]:
        """Get status for all known providers.

        Returns:
            Dict mapping provider names to their status
        """
        statuses = {}
        for provider_name in PROVIDER_CONFIGS:
            statuses[provider_name] = self.get_provider_status(provider_name)
        return statuses

    def reset(self) -> None:
        """Reset all resilience state (for testing)."""
        self._rate_limiters.clear()
        self._circuit_breakers.clear()


# Module-level singleton
_resilience_manager: Optional[ProviderResilienceManager] = None
_resilience_manager_lock = threading.Lock()


def get_resilience_manager() -> ProviderResilienceManager:
    """Get the singleton ProviderResilienceManager instance.

    Thread-safe via double-checked locking.

    Returns:
        The global ProviderResilienceManager instance
    """
    global _resilience_manager
    if _resilience_manager is None:
        with _resilience_manager_lock:
            if _resilience_manager is None:
                _resilience_manager = ProviderResilienceManager()
    return _resilience_manager


def reset_resilience_manager_for_testing() -> None:
    """Reset the singleton manager for test isolation.

    Creates a fresh manager instance with no state.
    """
    global _resilience_manager
    with _resilience_manager_lock:
        _resilience_manager = ProviderResilienceManager()
