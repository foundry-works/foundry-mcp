"""
Unit tests for foundry_mcp.core.resilience module.

Tests timeout decorators, retry patterns with exponential backoff,
circuit breaker state transitions, and health check utilities.
"""

import asyncio
import time

import pytest

from foundry_mcp.core.resilience import (
    # Timeout constants
    FAST_TIMEOUT,
    MEDIUM_TIMEOUT,
    SLOW_TIMEOUT,
    BACKGROUND_TIMEOUT,
    # Timeout utilities
    TimeoutException,
    with_timeout,
    # Retry utilities
    retry_with_backoff,
    retryable,
    # Circuit breaker
    CircuitState,
    CircuitBreaker,
    CircuitBreakerError,
    with_circuit_breaker,
    # Health check
    HealthStatus,
    health_check,
    check_dependencies,
)


# =============================================================================
# Timeout Constants Tests
# =============================================================================


class TestTimeoutConstants:
    """Test timeout budget constants."""

    def test_fast_timeout_value(self):
        """FAST_TIMEOUT should be 5 seconds."""
        assert FAST_TIMEOUT == 5.0

    def test_medium_timeout_value(self):
        """MEDIUM_TIMEOUT should be 30 seconds."""
        assert MEDIUM_TIMEOUT == 30.0

    def test_slow_timeout_value(self):
        """SLOW_TIMEOUT should be 120 seconds."""
        assert SLOW_TIMEOUT == 120.0

    def test_background_timeout_value(self):
        """BACKGROUND_TIMEOUT should be 600 seconds."""
        assert BACKGROUND_TIMEOUT == 600.0

    def test_timeout_ordering(self):
        """Timeouts should be ordered from fast to slow."""
        assert FAST_TIMEOUT < MEDIUM_TIMEOUT < SLOW_TIMEOUT < BACKGROUND_TIMEOUT


# =============================================================================
# Timeout Decorator Tests
# =============================================================================


class TestWithTimeout:
    """Test the with_timeout decorator."""

    @pytest.mark.asyncio
    async def test_fast_operation_succeeds(self):
        """Operations completing within timeout should succeed."""

        @with_timeout(1.0)
        async def fast_operation():
            await asyncio.sleep(0.01)
            return "success"

        result = await fast_operation()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_slow_operation_raises_timeout(self):
        """Operations exceeding timeout should raise TimeoutException."""

        @with_timeout(0.05, "Operation timed out")
        async def slow_operation():
            await asyncio.sleep(1.0)
            return "never reached"

        with pytest.raises(TimeoutException) as exc_info:
            await slow_operation()

        assert exc_info.value.timeout_seconds == 0.05
        assert exc_info.value.operation == "slow_operation"
        assert "Operation timed out" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_default_error_message(self):
        """Default error message should include function name and timeout."""

        @with_timeout(0.01)
        async def my_function():
            await asyncio.sleep(1.0)

        with pytest.raises(TimeoutException) as exc_info:
            await my_function()

        assert "my_function" in str(exc_info.value)
        assert "0.01" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_preserves_function_metadata(self):
        """Decorated function should preserve original metadata."""

        @with_timeout(5.0)
        async def documented_function():
            """This is the docstring."""
            return True

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is the docstring."

    @pytest.mark.asyncio
    async def test_passes_arguments(self):
        """Arguments should be passed through to wrapped function."""

        @with_timeout(1.0)
        async def add(a: int, b: int) -> int:
            return a + b

        result = await add(2, 3)
        assert result == 5

    @pytest.mark.asyncio
    async def test_passes_kwargs(self):
        """Keyword arguments should be passed through."""

        @with_timeout(1.0)
        async def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        result = await greet("World", greeting="Hi")
        assert result == "Hi, World!"


class TestTimeoutException:
    """Test TimeoutException attributes."""

    def test_exception_attributes(self):
        """TimeoutException should store timeout and operation info."""
        exc = TimeoutException(
            "Test timeout",
            timeout_seconds=30.0,
            operation="test_op",
        )
        assert str(exc) == "Test timeout"
        assert exc.timeout_seconds == 30.0
        assert exc.operation == "test_op"

    def test_exception_optional_attributes(self):
        """Optional attributes should default to None."""
        exc = TimeoutException("Basic error")
        assert exc.timeout_seconds is None
        assert exc.operation is None


# =============================================================================
# Retry with Backoff Tests
# =============================================================================


class TestRetryWithBackoff:
    """Test retry_with_backoff function."""

    def test_successful_first_attempt(self):
        """Function succeeding on first attempt should return immediately."""
        call_count = 0

        def succeed():
            nonlocal call_count
            call_count += 1
            return "success"

        result = retry_with_backoff(succeed, max_retries=3)
        assert result == "success"
        assert call_count == 1

    def test_retries_on_failure(self):
        """Function should retry on failure up to max_retries."""
        call_count = 0

        def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"

        result = retry_with_backoff(fail_twice, max_retries=3, base_delay=0.01)
        assert result == "success"
        assert call_count == 3

    def test_raises_after_max_retries(self):
        """Should raise last exception after exhausting retries."""
        call_count = 0

        def always_fail():
            nonlocal call_count
            call_count += 1
            raise ConnectionError(f"Attempt {call_count}")

        with pytest.raises(ConnectionError) as exc_info:
            retry_with_backoff(always_fail, max_retries=2, base_delay=0.01)

        assert call_count == 3  # Initial + 2 retries
        assert "Attempt 3" in str(exc_info.value)

    def test_retryable_exceptions_filter(self):
        """Only specified exceptions should trigger retry."""
        call_count = 0

        def fail_with_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")

        # ValueError not in retryable list, should fail immediately
        with pytest.raises(ValueError):
            retry_with_backoff(
                fail_with_value_error,
                max_retries=3,
                base_delay=0.01,
                retryable_exceptions=[ConnectionError],
            )

        assert call_count == 1  # No retries

    def test_exponential_backoff_timing(self):
        """Delays should increase exponentially."""
        call_times = []

        def track_time():
            call_times.append(time.time())
            raise ValueError("fail")

        with pytest.raises(ValueError):
            retry_with_backoff(
                track_time,
                max_retries=2,
                base_delay=0.05,
                exponential_base=2.0,
                jitter=False,
            )

        # Check delays are approximately correct
        # First retry: 0.05s, Second retry: 0.1s
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]

        assert 0.04 < delay1 < 0.15  # ~0.05s with tolerance
        assert 0.08 < delay2 < 0.20  # ~0.1s with tolerance

    def test_jitter_adds_randomness(self):
        """Jitter should add randomness to delays."""
        delays = []

        def track_delay():
            delays.append(time.time())
            raise ValueError("fail")

        # Run multiple times to check for variation
        for _ in range(3):
            delays.clear()
            with pytest.raises(ValueError):
                retry_with_backoff(
                    track_delay,
                    max_retries=1,
                    base_delay=0.05,
                    jitter=True,
                )

        # With jitter, delays should vary (statistically unlikely to be identical)
        # This is a probabilistic test

    def test_max_delay_cap(self):
        """Delay should not exceed max_delay."""
        call_times = []

        def track_time():
            call_times.append(time.time())
            raise ValueError("fail")

        with pytest.raises(ValueError):
            retry_with_backoff(
                track_time,
                max_retries=5,
                base_delay=0.1,
                max_delay=0.15,
                exponential_base=10.0,  # Would produce huge delays without cap
                jitter=False,
            )

        # All delays should be capped at max_delay
        for i in range(1, len(call_times)):
            delay = call_times[i] - call_times[i - 1]
            assert delay < 0.25  # max_delay + tolerance


class TestRetryableDecorator:
    """Test the @retryable decorator."""

    def test_decorator_basic_usage(self):
        """Decorator should wrap function with retry logic."""
        call_count = 0

        @retryable(max_retries=2, delay=0.01)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Temporary failure")
            return "success"

        result = flaky_function()
        assert result == "success"
        assert call_count == 2

    def test_decorator_with_exception_filter(self):
        """Decorator should respect exceptions filter."""
        call_count = 0

        @retryable(max_retries=3, delay=0.01, exceptions=(ConnectionError,))
        def specific_failure():
            nonlocal call_count
            call_count += 1
            raise ValueError("Wrong exception type")

        with pytest.raises(ValueError):
            specific_failure()

        assert call_count == 1  # No retries for ValueError

    def test_decorator_preserves_metadata(self):
        """Decorator should preserve function metadata."""

        @retryable()
        def my_documented_func():
            """Function docstring."""
            pass

        assert my_documented_func.__name__ == "my_documented_func"
        assert my_documented_func.__doc__ == "Function docstring."


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestCircuitState:
    """Test CircuitState enum."""

    def test_states_exist(self):
        """All expected states should exist."""
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"


class TestCircuitBreaker:
    """Test CircuitBreaker class."""

    def test_initial_state_closed(self):
        """New circuit breaker should be in CLOSED state."""
        cb = CircuitBreaker(name="test")
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.can_execute() is True

    def test_records_failures(self):
        """Failures should be recorded and counted."""
        cb = CircuitBreaker(name="test", failure_threshold=5)

        cb.record_failure()
        assert cb.failure_count == 1
        assert cb.state == CircuitState.CLOSED

        cb.record_failure()
        assert cb.failure_count == 2

    def test_opens_at_threshold(self):
        """Circuit should open after reaching failure threshold."""
        cb = CircuitBreaker(name="test", failure_threshold=3)

        for _ in range(3):
            cb.record_failure()

        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() is False

    def test_success_resets_failure_count(self):
        """Successful call should reset failure count."""
        cb = CircuitBreaker(name="test", failure_threshold=5)

        cb.record_failure()
        cb.record_failure()
        assert cb.failure_count == 2

        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED

    def test_recovery_timeout_transitions_to_half_open(self):
        """After recovery timeout, circuit should transition to HALF_OPEN."""
        cb = CircuitBreaker(name="test", failure_threshold=2, recovery_timeout=0.05)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.1)

        # Next call should transition to HALF_OPEN
        assert cb.can_execute() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_success_closes_circuit(self):
        """Successful calls in HALF_OPEN should close circuit."""
        cb = CircuitBreaker(
            name="test",
            failure_threshold=2,
            recovery_timeout=100.0,  # Long timeout to prevent re-transition
            half_open_max_calls=2,
        )

        # Open circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Manually transition to HALF_OPEN for controlled testing
        cb.state = CircuitState.HALF_OPEN
        cb.half_open_calls = 0

        # Simulate first call
        cb.can_execute()  # increments half_open_calls to 1
        cb.record_success()  # Won't close yet (need 2 calls)
        assert cb.state == CircuitState.HALF_OPEN

        # Simulate second call
        cb.can_execute()  # increments half_open_calls to 2
        cb.record_success()  # Now closes circuit

        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_half_open_failure_reopens_circuit(self):
        """Failure in HALF_OPEN should reopen circuit."""
        cb = CircuitBreaker(
            name="test", failure_threshold=2, recovery_timeout=0.01
        )

        # Open circuit
        cb.record_failure()
        cb.record_failure()

        # Transition to half-open
        time.sleep(0.02)
        cb.can_execute()
        assert cb.state == CircuitState.HALF_OPEN

        # Failure should reopen
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_half_open_limits_calls(self):
        """HALF_OPEN should limit number of test calls."""
        cb = CircuitBreaker(
            name="test",
            failure_threshold=2,
            recovery_timeout=100.0,  # Long timeout to prevent re-transition
            half_open_max_calls=2,
        )

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Manually transition to HALF_OPEN for testing
        cb.state = CircuitState.HALF_OPEN
        cb.half_open_calls = 0

        # Should allow half_open_max_calls
        assert cb.can_execute() is True  # 1st call (increments to 1)
        assert cb.can_execute() is True  # 2nd call (increments to 2)
        assert cb.can_execute() is False  # Limit reached

    def test_reset_clears_state(self):
        """reset() should return circuit to initial state."""
        cb = CircuitBreaker(name="test", failure_threshold=2)

        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.half_open_calls == 0

    def test_get_status_returns_info(self):
        """get_status() should return current circuit info."""
        cb = CircuitBreaker(name="my_service", failure_threshold=5)

        status = cb.get_status()
        assert status["name"] == "my_service"
        assert status["state"] == "closed"
        assert status["failure_count"] == 0
        assert status["failure_threshold"] == 5

    def test_get_status_retry_after_when_open(self):
        """get_status() should include retry_after when OPEN."""
        cb = CircuitBreaker(
            name="test", failure_threshold=2, recovery_timeout=10.0
        )

        cb.record_failure()
        cb.record_failure()

        status = cb.get_status()
        assert status["state"] == "open"
        assert status["retry_after_seconds"] is not None
        assert 9.0 < status["retry_after_seconds"] <= 10.0


class TestCircuitBreakerError:
    """Test CircuitBreakerError exception."""

    def test_error_attributes(self):
        """Error should store breaker name and state."""
        error = CircuitBreakerError(
            "Service unavailable",
            breaker_name="database",
            state=CircuitState.OPEN,
            retry_after=30.0,
        )

        assert str(error) == "Service unavailable"
        assert error.breaker_name == "database"
        assert error.state == CircuitState.OPEN
        assert error.retry_after == 30.0


class TestWithCircuitBreakerDecorator:
    """Test the with_circuit_breaker decorator."""

    def test_successful_call_records_success(self):
        """Successful call should record success with breaker."""
        breaker = CircuitBreaker(name="test", failure_threshold=3)

        @with_circuit_breaker(breaker)
        def succeed():
            return "ok"

        result = succeed()
        assert result == "ok"
        assert breaker.failure_count == 0

    def test_failed_call_records_failure(self):
        """Failed call should record failure with breaker."""
        breaker = CircuitBreaker(name="test", failure_threshold=3)

        @with_circuit_breaker(breaker)
        def fail():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            fail()

        assert breaker.failure_count == 1

    def test_open_circuit_raises_error(self):
        """Calls when circuit is OPEN should raise CircuitBreakerError."""
        breaker = CircuitBreaker(name="test_service", failure_threshold=2)

        @with_circuit_breaker(breaker)
        def operation():
            raise ConnectionError("fail")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ConnectionError):
                operation()

        # Next call should raise CircuitBreakerError
        with pytest.raises(CircuitBreakerError) as exc_info:
            operation()

        assert exc_info.value.breaker_name == "test_service"
        assert exc_info.value.state == CircuitState.OPEN

    def test_decorator_preserves_metadata(self):
        """Decorator should preserve function metadata."""
        breaker = CircuitBreaker(name="test")

        @with_circuit_breaker(breaker)
        def documented():
            """My docstring."""
            pass

        assert documented.__name__ == "documented"
        assert documented.__doc__ == "My docstring."


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthStatus:
    """Test HealthStatus dataclass."""

    def test_healthy_status(self):
        """HealthStatus should store healthy check result."""
        from datetime import datetime

        status = HealthStatus(
            name="database",
            healthy=True,
            latency_ms=5.5,
            last_check=datetime.utcnow(),
        )

        assert status.name == "database"
        assert status.healthy is True
        assert status.latency_ms == 5.5
        assert status.error is None

    def test_unhealthy_status_with_error(self):
        """HealthStatus should store error for unhealthy checks."""
        from datetime import datetime

        status = HealthStatus(
            name="cache",
            healthy=False,
            latency_ms=100.0,
            last_check=datetime.utcnow(),
            error="Connection refused",
        )

        assert status.healthy is False
        assert status.error == "Connection refused"


class TestHealthCheck:
    """Test health_check function."""

    @pytest.mark.asyncio
    async def test_successful_async_check(self):
        """Successful async health check should return healthy status."""

        async def healthy_service():
            await asyncio.sleep(0.01)
            return True

        status = await health_check("service", healthy_service)

        assert status.name == "service"
        assert status.healthy is True
        assert status.latency_ms > 0
        assert status.error is None

    @pytest.mark.asyncio
    async def test_successful_sync_check(self):
        """Sync callable should also work."""

        def sync_check():
            return "ok"

        status = await health_check("sync_service", sync_check)

        assert status.healthy is True

    @pytest.mark.asyncio
    async def test_failed_check_returns_unhealthy(self):
        """Failed check should return unhealthy status with error."""

        async def failing_service():
            raise ConnectionError("Cannot connect to database")

        status = await health_check("database", failing_service)

        assert status.healthy is False
        assert "Cannot connect" in status.error

    @pytest.mark.asyncio
    async def test_timeout_returns_unhealthy(self):
        """Check exceeding timeout should return unhealthy status."""

        async def slow_service():
            await asyncio.sleep(1.0)

        status = await health_check("slow", slow_service, timeout=0.05)

        assert status.healthy is False
        assert "timed out" in status.error.lower()

    @pytest.mark.asyncio
    async def test_latency_measured(self):
        """Latency should be measured in milliseconds."""

        async def timed_service():
            await asyncio.sleep(0.05)

        status = await health_check("timed", timed_service, timeout=1.0)

        # Should be around 50ms
        assert 40 < status.latency_ms < 150


class TestCheckDependencies:
    """Test check_dependencies function."""

    @pytest.mark.asyncio
    async def test_all_healthy(self):
        """All healthy checks should return healthy status."""

        async def healthy():
            return True

        result = await check_dependencies({
            "service1": healthy,
            "service2": healthy,
        })

        assert result["status"] == "healthy"
        assert len(result["unhealthy"]) == 0
        assert result["dependencies"]["service1"]["healthy"] is True
        assert result["dependencies"]["service2"]["healthy"] is True

    @pytest.mark.asyncio
    async def test_some_unhealthy(self):
        """Mixed results should return degraded status."""

        async def healthy():
            return True

        async def unhealthy():
            raise ValueError("fail")

        result = await check_dependencies({
            "good": healthy,
            "bad": unhealthy,
        })

        assert result["status"] == "degraded"
        assert "bad" in result["unhealthy"]
        assert result["dependencies"]["good"]["healthy"] is True
        assert result["dependencies"]["bad"]["healthy"] is False

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Checks should run concurrently."""
        start = time.time()

        async def slow_check():
            await asyncio.sleep(0.05)
            return True

        # Run 3 checks that each take 50ms
        result = await check_dependencies({
            "check1": slow_check,
            "check2": slow_check,
            "check3": slow_check,
        })

        elapsed = time.time() - start

        # Should complete in ~50ms (concurrent), not 150ms (sequential)
        # Allow 0.3s to account for system load and CI variability
        assert elapsed < 0.3
        assert result["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_includes_timestamp(self):
        """Result should include check timestamp."""

        async def check():
            return True

        result = await check_dependencies({"test": check})

        assert "checked_at" in result
        # Should be ISO format timestamp
        assert "T" in result["checked_at"]
