"""Unit tests for provider resilience module.

Tests cover:
- ProviderResilienceConfig and ErrorClassification dataclasses
- ProviderResilienceManager singleton behavior
- async_retry_with_backoff with deterministic jitter
- execute_with_resilience unified executor
- Circuit breaker state transitions
- Rate limiter token acquisition
- Time budget enforcement
"""

import asyncio
import random
from unittest.mock import patch

import pytest

from foundry_mcp.core.research.providers.resilience import (
    PROVIDER_CONFIGS,
    ErrorClassification,
    ErrorType,
    ProviderResilienceConfig,
    ProviderStatus,
    RateLimitWaitError,
    TimeBudgetExceededError,
    _default_classify_error,
    async_retry_with_backoff,
    execute_with_resilience,
    get_provider_config,
    get_resilience_manager,
    reset_resilience_manager_for_testing,
)
from foundry_mcp.core.resilience import CircuitBreakerError, CircuitState


class TestProviderResilienceConfig:
    """Tests for ProviderResilienceConfig dataclass."""

    def test_default_values(self):
        """Default config has expected values."""
        config = ProviderResilienceConfig()
        assert config.requests_per_second == 1.0
        assert config.burst_limit == 3
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.jitter == 0.5
        assert config.circuit_failure_threshold == 5
        assert config.circuit_recovery_timeout == 30.0

    def test_custom_values(self):
        """Custom config preserves values."""
        config = ProviderResilienceConfig(
            requests_per_second=0.5,
            burst_limit=2,
            max_retries=5,
        )
        assert config.requests_per_second == 0.5
        assert config.burst_limit == 2
        assert config.max_retries == 5

    def test_semantic_scholar_rate(self):
        """Semantic Scholar has 0.9 RPS as per spec."""
        config = PROVIDER_CONFIGS["semantic_scholar"]
        assert config.requests_per_second == 0.9


class TestErrorClassification:
    """Tests for ErrorClassification dataclass."""

    def test_default_values(self):
        """Default classification has expected values."""
        classification = ErrorClassification(retryable=True, trips_breaker=False)
        assert classification.retryable is True
        assert classification.trips_breaker is False
        assert classification.backoff_seconds is None
        assert classification.error_type == ErrorType.UNKNOWN

    def test_with_all_fields(self):
        """Classification with all fields set."""
        classification = ErrorClassification(
            retryable=False,
            trips_breaker=True,
            backoff_seconds=5.0,
            error_type=ErrorType.RATE_LIMIT,
        )
        assert classification.retryable is False
        assert classification.trips_breaker is True
        assert classification.backoff_seconds == 5.0
        assert classification.error_type == ErrorType.RATE_LIMIT


class TestDefaultClassifyError:
    """Tests for _default_classify_error function."""

    def test_rate_limit_429(self):
        """429 errors are retryable, don't trip breaker."""
        error = Exception("HTTP 429 Too Many Requests")
        classification = _default_classify_error(error)
        assert classification.retryable is True
        assert classification.trips_breaker is False
        assert classification.error_type == ErrorType.RATE_LIMIT

    def test_server_error_500(self):
        """500 errors are retryable, trip breaker."""
        error = Exception("HTTP 500 Internal Server Error")
        classification = _default_classify_error(error)
        assert classification.retryable is True
        assert classification.trips_breaker is True
        assert classification.error_type == ErrorType.SERVER_ERROR

    def test_auth_error_401(self):
        """401 errors are not retryable, trip breaker."""
        error = Exception("HTTP 401 Unauthorized")
        classification = _default_classify_error(error)
        assert classification.retryable is False
        assert classification.trips_breaker is True
        assert classification.error_type == ErrorType.AUTHENTICATION

    def test_timeout_error(self):
        """Timeout errors are retryable, trip breaker."""
        error = Exception("Connection timed out")
        classification = _default_classify_error(error)
        assert classification.retryable is True
        assert classification.trips_breaker is True
        assert classification.error_type == ErrorType.TIMEOUT

    def test_network_error(self):
        """Network errors are retryable, trip breaker."""
        error = Exception("Connection refused")
        classification = _default_classify_error(error)
        assert classification.retryable is True
        assert classification.trips_breaker is True
        assert classification.error_type == ErrorType.NETWORK

    def test_unknown_error(self):
        """Unknown errors are not retryable, trip breaker."""
        error = Exception("Something went wrong")
        classification = _default_classify_error(error)
        assert classification.retryable is False
        assert classification.trips_breaker is True
        assert classification.error_type == ErrorType.UNKNOWN


class TestProviderConfigs:
    """Tests for PROVIDER_CONFIGS dict."""

    def test_all_providers_present(self):
        """All expected providers are configured."""
        expected = {"tavily", "google", "perplexity", "semantic_scholar", "tavily_extract", "openalex", "crossref"}
        assert set(PROVIDER_CONFIGS.keys()) == expected

    def test_get_provider_config_existing(self):
        """get_provider_config returns config for known provider."""
        config = get_provider_config("tavily")
        assert config == PROVIDER_CONFIGS["tavily"]

    def test_get_provider_config_unknown(self):
        """get_provider_config returns default for unknown provider."""
        config = get_provider_config("unknown_provider")
        assert config.requests_per_second == 1.0  # Default


class TestProviderResilienceManager:
    """Tests for ProviderResilienceManager singleton."""

    def setup_method(self):
        """Reset manager before each test."""
        reset_resilience_manager_for_testing()

    def test_singleton_behavior(self):
        """get_resilience_manager returns same instance."""
        mgr1 = get_resilience_manager()
        mgr2 = get_resilience_manager()
        assert mgr1 is mgr2

    def test_reset_creates_new_instance(self):
        """reset_resilience_manager_for_testing creates new instance."""
        mgr1 = get_resilience_manager()
        reset_resilience_manager_for_testing()
        mgr2 = get_resilience_manager()
        assert mgr1 is not mgr2

    def test_isolated_limiters_per_provider(self):
        """Manager creates isolated rate limiters per provider."""
        mgr = get_resilience_manager()
        limiter1 = mgr._get_or_create_rate_limiter("tavily")
        limiter2 = mgr._get_or_create_rate_limiter("google")
        assert limiter1 is not limiter2

    def test_isolated_breakers_per_provider(self):
        """Manager creates isolated circuit breakers per provider."""
        mgr = get_resilience_manager()
        breaker1 = mgr._get_or_create_circuit_breaker("tavily")
        breaker2 = mgr._get_or_create_circuit_breaker("google")
        assert breaker1 is not breaker2

    def test_same_limiter_for_same_provider(self):
        """Manager returns same limiter for same provider."""
        mgr = get_resilience_manager()
        limiter1 = mgr._get_or_create_rate_limiter("tavily")
        limiter2 = mgr._get_or_create_rate_limiter("tavily")
        assert limiter1 is limiter2

    def test_reset_clears_all_state(self):
        """reset() clears all limiters and breakers."""
        mgr = get_resilience_manager()
        mgr._get_or_create_rate_limiter("tavily")
        mgr._get_or_create_circuit_breaker("tavily")
        assert len(mgr._rate_limiters) == 1
        assert len(mgr._circuit_breakers) == 1

        mgr.reset()
        assert len(mgr._rate_limiters) == 0
        assert len(mgr._circuit_breakers) == 0

    def test_get_breaker_state(self):
        """get_breaker_state returns circuit state."""
        mgr = get_resilience_manager()
        state = mgr.get_breaker_state("tavily")
        assert state == CircuitState.CLOSED

    def test_is_provider_available(self):
        """is_provider_available returns availability."""
        mgr = get_resilience_manager()
        assert mgr.is_provider_available("tavily") is True

        # Trip the breaker
        breaker = mgr._get_or_create_circuit_breaker("tavily")
        for _ in range(10):
            breaker.record_failure()
        assert mgr.is_provider_available("tavily") is False

    def test_is_provider_available_does_not_consume_half_open(self):
        """Availability checks should not consume half-open probe slots."""
        mgr = get_resilience_manager()
        breaker = mgr._get_or_create_circuit_breaker("tavily")
        breaker.recovery_timeout = 0.0

        for _ in range(10):
            breaker.record_failure()

        before_calls = breaker.half_open_calls
        assert mgr.is_provider_available("tavily") is True
        assert breaker.half_open_calls == before_calls

    def test_get_provider_status(self):
        """get_provider_status returns ProviderStatus."""
        mgr = get_resilience_manager()
        status = mgr.get_provider_status("tavily")
        assert isinstance(status, ProviderStatus)
        assert status.provider_name == "tavily"
        assert status.is_available is True
        assert status.circuit_state == "closed"

    def test_get_all_provider_statuses(self):
        """get_all_provider_statuses returns all providers."""
        mgr = get_resilience_manager()
        statuses = mgr.get_all_provider_statuses()
        assert len(statuses) == 7
        assert "tavily" in statuses
        assert "semantic_scholar" in statuses
        assert "openalex" in statuses
        assert "crossref" in statuses


class TestAsyncRetryWithBackoff:
    """Tests for async_retry_with_backoff function."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self):
        """Successful call returns immediately."""

        async def success():
            return "result"

        result = await async_retry_with_backoff(success)
        assert result == "result"

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Failed calls are retried."""
        call_count = [0]

        async def fail_twice():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("temporary error")
            return "success"

        result = await async_retry_with_backoff(
            fail_twice,
            max_retries=3,
            base_delay=0.001,
            retryable_exceptions=[ValueError],
        )
        assert result == "success"
        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self):
        """Exception raised when max retries exhausted."""

        async def always_fail():
            raise ValueError("permanent error")

        with pytest.raises(ValueError, match="permanent error"):
            await async_retry_with_backoff(
                always_fail,
                max_retries=2,
                base_delay=0.001,
            )

    @pytest.mark.asyncio
    async def test_non_retryable_exception(self):
        """Non-retryable exceptions raised immediately."""
        call_count = [0]

        async def raise_type_error():
            call_count[0] += 1
            raise TypeError("not retryable")

        with pytest.raises(TypeError):
            await async_retry_with_backoff(
                raise_type_error,
                max_retries=3,
                retryable_exceptions=[ValueError],  # TypeError not in list
            )
        assert call_count[0] == 1  # No retries

    @pytest.mark.asyncio
    async def test_jitter_deterministic_with_seeded_rng(self):
        """Jitter is deterministic with seeded RNG."""
        sleep_times: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleep_times.append(seconds)

        call_count = [0]

        async def fail_twice():
            call_count[0] += 1
            if call_count[0] < 3:
                raise RuntimeError("fail")
            return "done"

        # Run twice with same seed - should get same delays
        seeded_rng = random.Random(42)
        sleep_times.clear()
        call_count[0] = 0

        await async_retry_with_backoff(
            fail_twice,
            max_retries=3,
            base_delay=1.0,
            rng=seeded_rng,
            sleep_func=fake_sleep,
        )
        delays_run1 = sleep_times.copy()

        # Second run with fresh seeded RNG
        seeded_rng2 = random.Random(42)
        sleep_times.clear()
        call_count[0] = 0

        await async_retry_with_backoff(
            fail_twice,
            max_retries=3,
            base_delay=1.0,
            rng=seeded_rng2,
            sleep_func=fake_sleep,
        )
        delays_run2 = sleep_times.copy()

        # Delays should be identical with same seed
        assert delays_run1 == delays_run2

    @pytest.mark.asyncio
    async def test_jitter_range(self):
        """Jitter is within 50-150% range."""
        sleep_times: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleep_times.append(seconds)

        call_count = [0]

        async def fail_many():
            call_count[0] += 1
            if call_count[0] < 10:
                raise RuntimeError("fail")
            return "done"

        await async_retry_with_backoff(
            fail_many,
            max_retries=10,
            base_delay=1.0,
            exponential_base=1.0,  # No exponential growth
            rng=random.Random(),
            sleep_func=fake_sleep,
        )

        # All delays should be in range [0.5, 1.5] for base_delay=1.0
        for delay in sleep_times:
            assert 0.5 <= delay <= 1.5, f"Delay {delay} outside jitter range"


class TestCircuitBreakerStateTransitions:
    """Tests for circuit breaker state transitions via manager."""

    def setup_method(self):
        reset_resilience_manager_for_testing()

    def test_closed_to_open(self):
        """Circuit transitions from CLOSED to OPEN on failures."""
        mgr = get_resilience_manager()
        breaker = mgr._get_or_create_circuit_breaker("tavily")

        assert breaker.state == CircuitState.CLOSED

        # Record failures up to threshold
        for _ in range(5):
            breaker.record_failure()

        assert breaker.state == CircuitState.OPEN

    def test_open_blocks_requests(self):
        """OPEN circuit blocks can_execute."""
        mgr = get_resilience_manager()
        breaker = mgr._get_or_create_circuit_breaker("tavily")

        # Trip the breaker
        for _ in range(5):
            breaker.record_failure()

        assert breaker.can_execute() is False

    def test_half_open_allows_limited_requests(self):
        """HALF_OPEN allows limited requests after recovery timeout."""
        mgr = get_resilience_manager()
        # Create breaker and set short recovery timeout for testing
        breaker = mgr._get_or_create_circuit_breaker("tavily")
        breaker.recovery_timeout = 0.01

        # Trip the breaker
        for _ in range(5):
            breaker.record_failure()

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        import time

        time.sleep(0.02)

        # First request transitions to HALF_OPEN
        assert breaker.can_execute() is True
        assert breaker.state == CircuitState.HALF_OPEN

    def test_half_open_to_closed_on_success(self):
        """HALF_OPEN transitions to CLOSED on enough successes."""
        mgr = get_resilience_manager()
        breaker = mgr._get_or_create_circuit_breaker("tavily")
        breaker.recovery_timeout = 0.01
        breaker.half_open_max_calls = 2

        # Trip the breaker
        for _ in range(5):
            breaker.record_failure()

        # Wait for recovery timeout
        import time

        time.sleep(0.02)

        # First call: transitions to HALF_OPEN, counter stays 0
        assert breaker.can_execute() is True
        assert breaker.state == CircuitState.HALF_OPEN
        assert breaker.half_open_calls == 0
        breaker.record_success()

        # Second call: counter becomes 1
        assert breaker.can_execute() is True
        assert breaker.half_open_calls == 1
        breaker.record_success()

        # Third call: counter becomes 2, which meets half_open_max_calls
        assert breaker.can_execute() is True
        assert breaker.half_open_calls == 2
        breaker.record_success()  # This should close the circuit

        assert breaker.state == CircuitState.CLOSED


class TestRateLimiterTokenAcquisition:
    """Tests for rate limiter token acquisition."""

    def setup_method(self):
        reset_resilience_manager_for_testing()

    def test_acquire_token_success(self):
        """Token acquisition succeeds when tokens available."""
        mgr = get_resilience_manager()
        limiter = mgr._get_or_create_rate_limiter("tavily")

        result = limiter.acquire()
        assert result.allowed is True

    def test_acquire_exhausts_tokens(self):
        """Token acquisition exhausts burst limit."""
        mgr = get_resilience_manager()
        limiter = mgr._get_or_create_rate_limiter("tavily")

        # Acquire all burst tokens
        for _ in range(3):  # burst_limit = 3
            result = limiter.acquire()
            assert result.allowed is True

        # Next acquisition should be throttled
        result = limiter.acquire()
        assert result.allowed is False

    def test_tokens_refill_over_time(self):
        """Tokens refill based on RPS config."""
        mgr = get_resilience_manager()
        limiter = mgr._get_or_create_rate_limiter("tavily")

        # Exhaust tokens
        for _ in range(10):
            limiter.acquire()

        result = limiter.check()
        assert result.allowed is False

        # Wait for refill (1 RPS = 1 token/second)
        import time

        time.sleep(1.1)

        result = limiter.check()
        assert result.allowed is True


class TestExecuteWithResilience:
    """Tests for execute_with_resilience unified executor."""

    def setup_method(self):
        reset_resilience_manager_for_testing()

    @pytest.mark.asyncio
    async def test_success_execution(self):
        """Successful execution returns result."""

        async def success():
            return "result"

        result = await execute_with_resilience(success, "tavily")
        assert result == "result"

    @pytest.mark.asyncio
    async def test_circuit_breaker_rejects(self):
        """Open circuit breaker raises CircuitBreakerError."""
        mgr = get_resilience_manager()
        breaker = mgr._get_or_create_circuit_breaker("tavily")

        # Trip the breaker
        for _ in range(10):
            breaker.record_failure()

        async def should_not_run():
            raise RuntimeError("Should not execute")

        with pytest.raises(CircuitBreakerError) as exc_info:
            await execute_with_resilience(should_not_run, "tavily", manager=mgr)

        assert exc_info.value.breaker_name == "tavily"

    @pytest.mark.asyncio
    async def test_time_budget_exceeded(self):
        """Time budget exceeded raises TimeBudgetExceededError."""

        async def slow_func():
            await asyncio.sleep(1.0)
            return "done"

        with pytest.raises(TimeBudgetExceededError) as exc_info:
            await execute_with_resilience(slow_func, "tavily", time_budget=0.1)

        assert exc_info.value.budget_seconds == 0.1

    @pytest.mark.asyncio
    async def test_rate_limit_wait_error(self):
        """Rate limit wait exceeding max raises RateLimitWaitError."""
        mgr = get_resilience_manager()
        limiter = mgr._get_or_create_rate_limiter("tavily")

        # Exhaust tokens
        for _ in range(10):
            limiter.acquire()

        async def func():
            return "done"

        with pytest.raises(RateLimitWaitError) as exc_info:
            await execute_with_resilience(
                func,
                "tavily",
                max_wait_seconds=0.001,  # Very low to trigger error
                manager=mgr,
            )

        assert exc_info.value.provider == "tavily"

    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self):
        """Transient errors trigger retry."""
        call_count = [0]

        async def fail_then_succeed():
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("500 Internal Server Error")
            return "success"

        result = await execute_with_resilience(
            fail_then_succeed,
            "tavily",
            time_budget=30.0,
        )
        assert result == "success"
        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_records_success_to_breaker(self):
        """Successful execution records to circuit breaker."""
        mgr = get_resilience_manager()
        breaker = mgr._get_or_create_circuit_breaker("tavily")

        # Add some failures
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.failure_count == 2

        async def success():
            return "done"

        await execute_with_resilience(success, "tavily", manager=mgr)

        # Success should reset failure count
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_records_failure_to_breaker(self):
        """Failed execution records to circuit breaker."""
        mgr = get_resilience_manager()
        breaker = mgr._get_or_create_circuit_breaker("tavily")
        initial_failures = breaker.failure_count

        async def always_fail():
            raise Exception("500 Server Error")

        with pytest.raises(Exception):
            await execute_with_resilience(always_fail, "tavily", manager=mgr)

        # Failures should be recorded (max_retries + 1 attempts)
        assert breaker.failure_count > initial_failures

    @pytest.mark.asyncio
    async def test_custom_error_classifier(self):
        """Custom error classifier is used."""
        call_count = [0]

        def custom_classifier(error: Exception) -> ErrorClassification:
            # Mark all errors as non-retryable (error param required by signature)
            _ = error  # Satisfy linter - param required by classify_error signature
            return ErrorClassification(
                retryable=False,
                trips_breaker=False,
                error_type=ErrorType.UNKNOWN,
            )

        async def fail_once():
            call_count[0] += 1
            raise ValueError("error")

        with pytest.raises(ValueError):
            await execute_with_resilience(
                fail_once,
                "tavily",
                classify_error=custom_classifier,
            )

        # Should not retry because custom classifier says not retryable
        assert call_count[0] == 1

    @pytest.mark.asyncio
    async def test_rate_limit_token_acquired_per_attempt(self):
        """Each attempt should consume a rate limit token."""
        mgr = get_resilience_manager()
        config = ProviderResilienceConfig(
            requests_per_second=1000.0,
            burst_limit=10,
            max_retries=2,
            jitter=0.0,
        )
        call_count = [0]

        async def fail_twice_then_succeed():
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("500 Internal Server Error")
            return "ok"

        result = await execute_with_resilience(
            fail_twice_then_succeed,
            "tavily",
            manager=mgr,
            time_budget=5.0,
            resilience_config=config,
        )
        assert result == "ok"
        limiter = mgr._get_or_create_rate_limiter("tavily", config=config)
        assert limiter.state.request_count == 3

    @pytest.mark.asyncio
    async def test_backoff_seconds_overrides_delay(self):
        """backoff_seconds should override computed delay."""
        mgr = get_resilience_manager()
        config = ProviderResilienceConfig(
            requests_per_second=1000.0,
            burst_limit=10,
            max_retries=1,
            base_delay=1.0,
            jitter=0.0,
        )
        sleep_times: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleep_times.append(seconds)

        def classifier(_: Exception) -> ErrorClassification:
            return ErrorClassification(
                retryable=True,
                trips_breaker=False,
                backoff_seconds=5.0,
                error_type=ErrorType.RATE_LIMIT,
            )

        async def always_fail():
            raise Exception("429 Too Many Requests")

        with patch("foundry_mcp.core.research.providers.resilience.execution.asyncio.sleep", fake_sleep):
            with pytest.raises(Exception):
                await execute_with_resilience(
                    always_fail,
                    "tavily",
                    manager=mgr,
                    classify_error=classifier,
                    resilience_config=config,
                )

        assert sleep_times == [5.0]


class TestDeterministicJitterIntegration:
    """Integration tests for deterministic jitter with seeded RNG.

    Verifies that retry delays are reproducible when using seeded RNG,
    enabling reliable testing of backoff behavior.
    """

    def setup_method(self):
        reset_resilience_manager_for_testing()

    @pytest.mark.asyncio
    async def test_execute_with_resilience_jitter_is_bounded(self):
        """Verify jitter stays within 50-150% of calculated delay."""
        sleep_times: list[float] = []
        original_sleep = asyncio.sleep

        async def tracking_sleep(seconds: float) -> None:
            sleep_times.append(seconds)
            # Use very short actual sleep for test speed
            await original_sleep(0.001)

        call_count = [0]

        async def fail_twice():
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("500 Internal Server Error")
            return "success"

        # Patch asyncio.sleep at module level
        with patch("foundry_mcp.core.research.providers.resilience.execution.asyncio.sleep", tracking_sleep):
            result = await execute_with_resilience(
                fail_twice,
                "tavily",
                time_budget=30.0,
            )

        assert result == "success"
        assert len(sleep_times) == 2  # Two retries before success

        # Verify jitter range: base_delay=1.0, so first retry ~1.0 * [0.5, 1.5]
        # Second retry: 2.0 * [0.5, 1.5]
        for i, delay in enumerate(sleep_times):
            base = 1.0 * (2.0**i)  # base_delay * exponential_base^attempt
            min_delay = base * 0.5
            max_delay = base * 1.5
            assert min_delay <= delay <= max_delay, f"Delay {delay} outside expected range [{min_delay}, {max_delay}]"

    @pytest.mark.asyncio
    async def test_async_retry_multiple_runs_same_seed_same_delays(self):
        """Multiple executions with same seed produce identical delays."""

        async def fail_many():
            raise RuntimeError("fail")

        # Collect delays from multiple runs with same seed
        all_delays: list[list[float]] = []

        for _ in range(3):
            sleep_times: list[float] = []

            async def tracking_sleep(seconds: float) -> None:
                sleep_times.append(seconds)

            seeded_rng = random.Random(12345)

            with pytest.raises(RuntimeError):
                await async_retry_with_backoff(
                    fail_many,
                    max_retries=3,
                    base_delay=1.0,
                    rng=seeded_rng,
                    sleep_func=tracking_sleep,
                )
            all_delays.append(sleep_times.copy())

        # All runs should have identical delays
        assert all_delays[0] == all_delays[1] == all_delays[2]
        assert len(all_delays[0]) == 3  # 3 retries


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker behavior.

    Tests circuit breaker state transitions through the full
    execute_with_resilience stack with realistic error scenarios.
    """

    def setup_method(self):
        reset_resilience_manager_for_testing()

    @pytest.mark.asyncio
    async def test_circuit_opens_after_consecutive_failures(self):
        """Circuit opens after failure threshold reached."""
        mgr = get_resilience_manager()
        call_count = [0]

        async def always_fail():
            call_count[0] += 1
            raise Exception("500 Server Error")

        # Execute multiple times to trip the circuit
        # Default threshold is 5, and we retry 3 times per call
        for _ in range(2):  # 2 calls * 4 attempts = 8 failures > threshold
            with pytest.raises(Exception):
                await execute_with_resilience(
                    always_fail,
                    "tavily",
                    manager=mgr,
                    time_budget=30.0,
                )

        # Circuit should now be open
        breaker = mgr._get_or_create_circuit_breaker("tavily")
        assert breaker.state == CircuitState.OPEN

        # Further calls should fail fast with CircuitBreakerError
        with pytest.raises(CircuitBreakerError) as exc_info:
            await execute_with_resilience(
                always_fail,
                "tavily",
                manager=mgr,
            )
        assert exc_info.value.breaker_name == "tavily"

    @pytest.mark.asyncio
    async def test_circuit_allows_probe_after_recovery_timeout(self):
        """Circuit transitions to HALF_OPEN after recovery timeout."""
        mgr = get_resilience_manager()
        breaker = mgr._get_or_create_circuit_breaker("tavily")
        breaker.recovery_timeout = 0.01  # Very short for testing

        # Trip the circuit
        for _ in range(10):
            breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.02)

        # Next check should transition to HALF_OPEN
        assert breaker.can_execute() is True
        assert breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_circuit_closes_after_half_open_successes(self):
        """Circuit returns to CLOSED after successful HALF_OPEN calls."""
        mgr = get_resilience_manager()
        breaker = mgr._get_or_create_circuit_breaker("tavily")
        breaker.recovery_timeout = 0.01
        breaker.half_open_max_calls = 2

        # Trip the circuit
        for _ in range(10):
            breaker.record_failure()

        # Wait for recovery
        await asyncio.sleep(0.02)
        breaker.can_execute()  # Transition to HALF_OPEN

        call_count = [0]

        async def succeed():
            call_count[0] += 1
            return "ok"

        # Make successful calls to close the circuit
        for _ in range(3):
            result = await execute_with_resilience(
                succeed,
                "tavily",
                manager=mgr,
            )
            assert result == "ok"

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_reopens_on_half_open_failure(self):
        """Circuit returns to OPEN on failure during HALF_OPEN."""
        mgr = get_resilience_manager()
        breaker = mgr._get_or_create_circuit_breaker("tavily")
        breaker.recovery_timeout = 0.01

        # Trip the circuit
        for _ in range(10):
            breaker.record_failure()

        # Wait for recovery
        await asyncio.sleep(0.02)
        breaker.can_execute()  # Transition to HALF_OPEN
        assert breaker.state == CircuitState.HALF_OPEN

        async def fail_once():
            raise Exception("500 Server Error")

        with pytest.raises(Exception):
            await execute_with_resilience(
                fail_once,
                "tavily",
                manager=mgr,
            )

        # Should be back to OPEN
        assert breaker.state == CircuitState.OPEN


class TestRateLimiterIntegration:
    """Integration tests for rate limiter behavior.

    Tests rate limiting through execute_with_resilience including
    wait behavior and token acquisition.
    """

    def setup_method(self):
        reset_resilience_manager_for_testing()

    @pytest.mark.asyncio
    async def test_rate_limiter_allows_burst(self):
        """Rate limiter allows burst of requests up to burst_limit."""
        mgr = get_resilience_manager()
        limiter = mgr._get_or_create_rate_limiter("tavily")

        # Should allow burst_limit (3) immediate acquisitions
        for i in range(3):
            result = limiter.acquire()
            assert result.allowed is True, f"Burst request {i + 1} should be allowed"

        # Fourth should be throttled
        result = limiter.acquire()
        assert result.allowed is False

    @pytest.mark.asyncio
    async def test_rate_limiter_wait_succeeds_within_timeout(self):
        """Rate limiter wait succeeds when wait time < max_wait_seconds."""
        mgr = get_resilience_manager()
        limiter = mgr._get_or_create_rate_limiter("tavily")

        # Exhaust burst
        for _ in range(3):
            limiter.acquire()

        call_count = [0]

        async def succeed():
            call_count[0] += 1
            return "result"

        # With max_wait_seconds=5.0 (default), should succeed after wait
        result = await execute_with_resilience(
            succeed,
            "tavily",
            manager=mgr,
            max_wait_seconds=5.0,
        )
        assert result == "result"

    @pytest.mark.asyncio
    async def test_rate_limiter_rejects_when_wait_exceeds_max(self):
        """RateLimitWaitError raised when wait would exceed max_wait_seconds."""
        mgr = get_resilience_manager()
        limiter = mgr._get_or_create_rate_limiter("tavily")

        # Exhaust burst
        for _ in range(10):
            limiter.acquire()

        async def succeed():
            return "result"

        # With very low max_wait_seconds, should raise
        with pytest.raises(RateLimitWaitError) as exc_info:
            await execute_with_resilience(
                succeed,
                "tavily",
                manager=mgr,
                max_wait_seconds=0.001,
            )
        assert exc_info.value.provider == "tavily"

    def test_rate_limiter_isolated_per_provider(self):
        """Each provider has isolated rate limiter state."""
        mgr = get_resilience_manager()

        # Exhaust tavily
        tavily_limiter = mgr._get_or_create_rate_limiter("tavily")
        for _ in range(10):
            tavily_limiter.acquire()

        # Google should still have tokens
        google_limiter = mgr._get_or_create_rate_limiter("google")
        result = google_limiter.acquire()
        assert result.allowed is True


class TestTimeBudgetIntegration:
    """Integration tests for time budget enforcement.

    Tests timeout and cancellation behavior through execute_with_resilience.
    """

    def setup_method(self):
        reset_resilience_manager_for_testing()

    @pytest.mark.asyncio
    async def test_time_budget_cancels_slow_operation(self):
        """Operation cancelled when exceeding time budget."""

        async def slow_func():
            await asyncio.sleep(10.0)  # Much longer than budget
            return "done"

        with pytest.raises(TimeBudgetExceededError) as exc_info:
            await execute_with_resilience(
                slow_func,
                "tavily",
                time_budget=0.1,
            )

        assert exc_info.value.budget_seconds == 0.1
        assert exc_info.value.operation == "tavily"

    @pytest.mark.asyncio
    async def test_time_budget_accounts_for_retries(self):
        """Time budget is checked before each retry attempt."""
        call_count = [0]
        call_times: list[float] = []

        async def slow_fail():
            import time

            call_count[0] += 1
            call_times.append(time.monotonic())
            # Each call takes 0.05s before failing
            await asyncio.sleep(0.05)
            raise Exception("500 Server Error")

        with pytest.raises((TimeBudgetExceededError, Exception)):
            await execute_with_resilience(
                slow_fail,
                "tavily",
                time_budget=0.2,  # Allow ~2-3 attempts
            )

        # Should have made some attempts before budget exhausted
        assert call_count[0] >= 1

    @pytest.mark.asyncio
    async def test_time_budget_none_allows_unlimited(self):
        """No time budget (None) allows operation to complete."""
        call_count = [0]

        async def slow_but_succeed():
            call_count[0] += 1
            await asyncio.sleep(0.05)
            return "done"

        result = await execute_with_resilience(
            slow_but_succeed,
            "tavily",
            time_budget=None,  # No budget
        )

        assert result == "done"
        assert call_count[0] == 1

    @pytest.mark.asyncio
    async def test_time_budget_exceeded_before_execution(self):
        """TimeBudgetExceededError if budget exhausted before execution."""
        mgr = get_resilience_manager()
        limiter = mgr._get_or_create_rate_limiter("tavily")

        # Exhaust tokens so we need to wait
        for _ in range(10):
            limiter.acquire()

        async def should_not_run():
            raise RuntimeError("Should not execute")

        # Very short budget that will expire during rate limit wait
        with pytest.raises((RateLimitWaitError, TimeBudgetExceededError)):
            await execute_with_resilience(
                should_not_run,
                "tavily",
                time_budget=0.001,
                max_wait_seconds=10.0,  # Long max wait but short budget
                manager=mgr,
            )


class TestFullStackIntegration:
    """Full stack integration tests with mocked HTTP responses.

    Tests the complete resilience stack including error classification,
    retry, circuit breaker, and rate limiting working together.
    """

    def setup_method(self):
        reset_resilience_manager_for_testing()

    @pytest.mark.asyncio
    async def test_http_429_triggers_retry_without_tripping_breaker(self):
        """429 rate limit errors are retried but don't trip circuit breaker."""
        mgr = get_resilience_manager()
        breaker = mgr._get_or_create_circuit_breaker("tavily")
        initial_failures = breaker.failure_count
        call_count = [0]

        async def rate_limited_then_success():
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("HTTP 429 Too Many Requests")
            return "success"

        result = await execute_with_resilience(
            rate_limited_then_success,
            "tavily",
            manager=mgr,
            time_budget=30.0,
        )

        assert result == "success"
        assert call_count[0] == 3
        # 429s should not increment failure count
        assert breaker.failure_count == initial_failures

    @pytest.mark.asyncio
    async def test_http_500_triggers_retry_and_trips_breaker(self):
        """500 server errors are retried and trip circuit breaker."""
        mgr = get_resilience_manager()
        breaker = mgr._get_or_create_circuit_breaker("tavily")
        call_count = [0]

        async def server_error_then_success():
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("HTTP 500 Internal Server Error")
            return "success"

        result = await execute_with_resilience(
            server_error_then_success,
            "tavily",
            manager=mgr,
            time_budget=30.0,
        )

        assert result == "success"
        assert call_count[0] == 3
        # 500s should increment then reset failure count on success
        assert breaker.failure_count == 0  # Reset by success

    @pytest.mark.asyncio
    async def test_http_401_not_retried_trips_breaker(self):
        """401 auth errors are not retried and trip circuit breaker."""
        mgr = get_resilience_manager()
        call_count = [0]

        async def auth_error():
            call_count[0] += 1
            raise Exception("HTTP 401 Unauthorized")

        with pytest.raises(Exception, match="401 Unauthorized"):
            await execute_with_resilience(
                auth_error,
                "tavily",
                manager=mgr,
            )

        # Should only be called once (no retry)
        assert call_count[0] == 1

    @pytest.mark.asyncio
    async def test_combined_rate_limit_circuit_breaker_timeout(self):
        """Test combined resilience behaviors in sequence."""
        mgr = get_resilience_manager()
        call_count = [0]

        async def intermittent_failure():
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("HTTP 503 Service Unavailable")
            if call_count[0] == 2:
                raise Exception("HTTP 429 Too Many Requests")
            return {"status": "ok"}

        result = await execute_with_resilience(
            intermittent_failure,
            "tavily",
            manager=mgr,
            time_budget=30.0,
        )

        assert result == {"status": "ok"}
        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_failover_simulation_multiple_providers(self):
        """Simulate failover between providers when one is unavailable."""
        mgr = get_resilience_manager()

        # Trip circuit for tavily
        tavily_breaker = mgr._get_or_create_circuit_breaker("tavily")
        for _ in range(10):
            tavily_breaker.record_failure()
        assert mgr.is_provider_available("tavily") is False

        # Google should still be available
        assert mgr.is_provider_available("google") is True

        async def succeed():
            return "from_google"

        # Tavily call should fail fast
        with pytest.raises(CircuitBreakerError):
            await execute_with_resilience(
                succeed,
                "tavily",
                manager=mgr,
            )

        # Google call should succeed
        result = await execute_with_resilience(
            succeed,
            "google",
            manager=mgr,
        )
        assert result == "from_google"

    @pytest.mark.asyncio
    async def test_provider_status_reflects_all_state(self):
        """ProviderStatus accurately reflects combined state."""
        mgr = get_resilience_manager()

        # Fresh provider should be available
        status = mgr.get_provider_status("tavily")
        assert status.is_available is True
        assert status.circuit_state == "closed"
        assert status.circuit_failure_count == 0

        # Add some failures
        breaker = mgr._get_or_create_circuit_breaker("tavily")
        for _ in range(3):
            breaker.record_failure()

        status = mgr.get_provider_status("tavily")
        assert status.is_available is True  # Not yet at threshold
        assert status.circuit_failure_count == 3

        # Trip the breaker
        for _ in range(5):
            breaker.record_failure()

        status = mgr.get_provider_status("tavily")
        assert status.is_available is False
        assert status.circuit_state == "open"

    @pytest.mark.asyncio
    async def test_all_providers_status_report(self):
        """get_all_provider_statuses returns status for all configured providers."""
        mgr = get_resilience_manager()
        statuses = mgr.get_all_provider_statuses()

        expected_providers = {"tavily", "google", "perplexity", "semantic_scholar", "tavily_extract", "openalex", "crossref"}
        assert set(statuses.keys()) == expected_providers

        for provider_name, status in statuses.items():
            assert status.provider_name == provider_name
            assert isinstance(status.is_available, bool)
            assert status.circuit_state in ("closed", "open", "half_open")

    @pytest.mark.asyncio
    async def test_semantic_scholar_respects_lower_rate_limit(self):
        """Semantic Scholar provider uses 0.9 RPS config."""
        mgr = get_resilience_manager()
        limiter = mgr._get_or_create_rate_limiter("semantic_scholar")

        # Should only allow 2 burst (lower than default 3)
        config = get_provider_config("semantic_scholar")
        assert config.burst_limit == 2

        # Verify limiter uses this config
        result1 = limiter.acquire()
        result2 = limiter.acquire()
        result3 = limiter.acquire()

        assert result1.allowed is True
        assert result2.allowed is True
        assert result3.allowed is False  # Exceeded burst_limit of 2
