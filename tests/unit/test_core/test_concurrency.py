"""Tests for foundry_mcp.core.concurrency module."""

import asyncio
import pytest
import time

from foundry_mcp.core.concurrency import (
    # Concurrency limiting
    ConcurrencyLimiter,
    get_tool_limiter,
    configure_tool_limiter,
    get_all_limiter_stats,
    # Cancellation handling
    CancellationToken,
    cancellable_scope,
    with_cancellation,
    run_with_cancellation_checkpoints,
    cancel_tasks_gracefully,
    # Request context
    RequestContext,
    request_context,
    get_current_context,
    get_current_context_or_none,
    get_request_id,
    get_client_id,
    get_elapsed_time,
)


# =============================================================================
# ConcurrencyLimiter Tests
# =============================================================================


class TestConcurrencyLimiter:
    """Tests for ConcurrencyLimiter class."""

    @pytest.mark.asyncio
    async def test_basic_limiting(self):
        """Test that limiter restricts concurrent operations."""
        limiter = ConcurrencyLimiter(max_concurrent=2)
        active_count = 0
        max_observed = 0

        async def track_concurrency():
            nonlocal active_count, max_observed
            active_count += 1
            max_observed = max(max_observed, active_count)
            await asyncio.sleep(0.01)
            active_count -= 1
            return True

        # Run 5 operations with max 2 concurrent
        coros = [track_concurrency() for _ in range(5)]
        result = await limiter.gather(coros)

        assert result.stats.total == 5
        assert result.stats.succeeded == 5
        assert max_observed <= 2

    @pytest.mark.asyncio
    async def test_gather_returns_results(self):
        """Test that gather returns results in order."""
        limiter = ConcurrencyLimiter(max_concurrent=3)

        async def process(n: int) -> int:
            await asyncio.sleep(0.001)
            return n * 2

        coros = [process(i) for i in range(5)]
        result = await limiter.gather(coros)

        assert result.all_succeeded
        assert result.results == [0, 2, 4, 6, 8]

    @pytest.mark.asyncio
    async def test_gather_handles_exceptions(self):
        """Test that gather captures exceptions when return_exceptions=True."""
        limiter = ConcurrencyLimiter(max_concurrent=2)

        async def maybe_fail(n: int) -> int:
            if n == 2:
                raise ValueError("Failed!")
            return n

        coros = [maybe_fail(i) for i in range(4)]
        result = await limiter.gather(coros, return_exceptions=True)

        assert result.stats.succeeded == 3
        assert result.stats.failed == 1
        assert not result.all_succeeded

        # Check that failed result has error
        failed = result.failed_results()
        assert len(failed) == 1
        assert failed[0][0] == 2  # Index 2 failed
        assert isinstance(failed[0][1], ValueError)

    @pytest.mark.asyncio
    async def test_acquire_context_manager(self):
        """Test acquire() as context manager."""
        limiter = ConcurrencyLimiter(max_concurrent=1)

        async with limiter.acquire():
            assert limiter.active_count == 1

        assert limiter.active_count == 0

    @pytest.mark.asyncio
    async def test_run_with_timeout(self):
        """Test run() with timeout."""
        limiter = ConcurrencyLimiter(max_concurrent=1)

        async def slow_op():
            await asyncio.sleep(10)
            return "done"

        with pytest.raises(asyncio.TimeoutError):
            await limiter.run(slow_op(), timeout=0.01)

    @pytest.mark.asyncio
    async def test_map_function(self):
        """Test map() convenience method."""
        limiter = ConcurrencyLimiter(max_concurrent=2)

        async def double(x: int) -> int:
            return x * 2

        result = await limiter.map(double, [1, 2, 3, 4])

        assert result.all_succeeded
        assert result.successful_results() == [2, 4, 6, 8]

    def test_get_stats(self):
        """Test get_stats() returns correct info."""
        limiter = ConcurrencyLimiter(max_concurrent=5, name="test_limiter")
        stats = limiter.get_stats()

        assert stats["max_concurrent"] == 5
        assert stats["name"] == "test_limiter"
        assert stats["active_count"] == 0


class TestToolLimiterRegistry:
    """Tests for per-tool limiter registry."""

    def test_get_tool_limiter_creates_new(self):
        """Test get_tool_limiter creates limiter if not exists."""
        limiter = get_tool_limiter("unique_test_tool", default_limit=7)
        assert limiter.max_concurrent == 7

    def test_get_tool_limiter_returns_existing(self):
        """Test get_tool_limiter returns same instance."""
        limiter1 = get_tool_limiter("reuse_test_tool")
        limiter2 = get_tool_limiter("reuse_test_tool")
        assert limiter1 is limiter2

    def test_configure_tool_limiter(self):
        """Test configure_tool_limiter sets custom config."""
        limiter = configure_tool_limiter(
            "configured_tool",
            max_concurrent=15,
            timeout=30.0,
        )
        assert limiter.max_concurrent == 15
        assert limiter.config.timeout == 30.0

    def test_get_all_limiter_stats(self):
        """Test get_all_limiter_stats returns all limiters."""
        configure_tool_limiter("stats_test_1", max_concurrent=3)
        configure_tool_limiter("stats_test_2", max_concurrent=5)

        stats = get_all_limiter_stats()
        assert "stats_test_1" in stats
        assert "stats_test_2" in stats


# =============================================================================
# Cancellation Handling Tests
# =============================================================================


class TestCancellableScope:
    """Tests for cancellable_scope context manager."""

    @pytest.mark.asyncio
    async def test_normal_execution(self):
        """Test scope passes through normal execution."""
        result = None

        async with cancellable_scope():
            result = "completed"

        assert result == "completed"

    @pytest.mark.asyncio
    async def test_cleanup_on_cancel(self):
        """Test cleanup function is called on cancellation."""
        cleanup_called = False

        async def cleanup():
            nonlocal cleanup_called
            cleanup_called = True

        async def cancellable_op():
            async with cancellable_scope(cleanup_func=cleanup):
                await asyncio.sleep(10)

        task = asyncio.create_task(cancellable_op())
        await asyncio.sleep(0.01)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert cleanup_called


class TestWithCancellationDecorator:
    """Tests for @with_cancellation decorator."""

    @pytest.mark.asyncio
    async def test_decorated_function_runs(self):
        """Test decorated function executes normally."""
        @with_cancellation()
        async def my_func():
            return "success"

        result = await my_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_cleanup_called_on_cancel(self):
        """Test cleanup is called when decorated function is cancelled."""
        cleanup_called = False

        async def cleanup():
            nonlocal cleanup_called
            cleanup_called = True

        @with_cancellation(cleanup_func=cleanup)
        async def slow_func():
            await asyncio.sleep(10)

        task = asyncio.create_task(slow_func())
        await asyncio.sleep(0.01)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert cleanup_called


class TestRunWithCancellationCheckpoints:
    """Tests for run_with_cancellation_checkpoints function."""

    @pytest.mark.asyncio
    async def test_completes_all_items(self):
        """Test all items are processed when not cancelled."""
        async def process(item):
            return item * 2

        result = await run_with_cancellation_checkpoints(
            items=[1, 2, 3],
            process_func=process,
        )

        assert result.completed
        assert not result.was_cancelled
        assert result.partial_results == [2, 4, 6]
        assert result.processed_count == 3

    @pytest.mark.asyncio
    async def test_returns_partial_on_cancel(self):
        """Test partial results returned on cancellation."""
        processed = []

        async def slow_process(item):
            processed.append(item)
            await asyncio.sleep(0.1)
            return item

        async def run_and_cancel():
            task = asyncio.create_task(
                run_with_cancellation_checkpoints(
                    items=list(range(100)),
                    process_func=slow_process,
                    checkpoint_interval=1,
                )
            )
            await asyncio.sleep(0.05)
            task.cancel()
            return await task

        result = await run_and_cancel()

        assert result.was_cancelled
        assert not result.completed
        assert len(result.partial_results) < 100


class TestCancellationToken:
    """Tests for CancellationToken class."""

    def test_initial_state(self):
        """Test token starts not cancelled."""
        token = CancellationToken()
        assert not token.is_cancelled

    def test_cancel_sets_flag(self):
        """Test cancel() sets is_cancelled."""
        token = CancellationToken()
        token.cancel()
        assert token.is_cancelled

    @pytest.mark.asyncio
    async def test_check_raises_when_cancelled(self):
        """Test check() raises CancelledError when cancelled."""
        token = CancellationToken()
        token.cancel()

        with pytest.raises(asyncio.CancelledError):
            await token.check()

    @pytest.mark.asyncio
    async def test_wait_for_cancel_returns_true(self):
        """Test wait_for_cancel returns True when cancelled."""
        token = CancellationToken()

        async def cancel_later():
            await asyncio.sleep(0.01)
            token.cancel()

        asyncio.create_task(cancel_later())
        result = await token.wait_for_cancel(timeout=1.0)

        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_cancel_timeout(self):
        """Test wait_for_cancel returns False on timeout."""
        token = CancellationToken()
        result = await token.wait_for_cancel(timeout=0.01)
        assert result is False


class TestCancelTasksGracefully:
    """Tests for cancel_tasks_gracefully function."""

    @pytest.mark.asyncio
    async def test_cancels_all_tasks(self):
        """Test all tasks are cancelled."""
        async def slow_op():
            await asyncio.sleep(10)

        tasks = [asyncio.create_task(slow_op()) for _ in range(3)]
        errors = await cancel_tasks_gracefully(tasks, timeout=1.0)

        assert len(errors) == 3
        assert all(e is None for e in errors)  # Clean cancellation

    @pytest.mark.asyncio
    async def test_handles_empty_list(self):
        """Test handles empty task list."""
        errors = await cancel_tasks_gracefully([])
        assert errors == []


# =============================================================================
# Request Context Tests
# =============================================================================


class TestRequestContext:
    """Tests for RequestContext dataclass."""

    def test_elapsed_time(self):
        """Test elapsed time calculation."""
        start = time.monotonic()
        ctx = RequestContext(
            request_id="test",
            client_id="client",
            start_time=start,
        )

        # Small delay
        time.sleep(0.01)
        assert ctx.elapsed_seconds > 0
        assert ctx.elapsed_ms > 0

    def test_to_dict(self):
        """Test to_dict serialization."""
        ctx = RequestContext(
            request_id="req-123",
            client_id="client-456",
            start_time=time.monotonic(),
        )
        d = ctx.to_dict()

        assert d["request_id"] == "req-123"
        assert d["client_id"] == "client-456"
        assert "elapsed_ms" in d
        assert "start_timestamp" in d


class TestRequestContextManager:
    """Tests for request_context async context manager."""

    @pytest.mark.asyncio
    async def test_sets_context_variables(self):
        """Test context variables are set."""
        async with request_context(req_id="test-req", cli_id="test-client"):
            assert get_request_id() == "test-req"
            assert get_client_id() == "test-client"

    @pytest.mark.asyncio
    async def test_auto_generates_request_id(self):
        """Test request ID is auto-generated if not provided."""
        async with request_context(cli_id="test"):
            req_id = get_request_id()
            assert req_id  # Not empty
            assert len(req_id) == 36  # UUID format

    @pytest.mark.asyncio
    async def test_resets_on_exit(self):
        """Test context is reset after exiting."""
        async with request_context(req_id="inside"):
            pass

        assert get_request_id() == ""

    @pytest.mark.asyncio
    async def test_yields_context_object(self):
        """Test context manager yields RequestContext."""
        async with request_context(req_id="test", cli_id="client") as ctx:
            assert isinstance(ctx, RequestContext)
            assert ctx.request_id == "test"
            assert ctx.client_id == "client"


class TestContextHelpers:
    """Tests for context helper functions."""

    @pytest.mark.asyncio
    async def test_get_current_context(self):
        """Test get_current_context returns context."""
        async with request_context(req_id="ctx-test"):
            ctx = get_current_context()
            assert ctx.request_id == "ctx-test"

    @pytest.mark.asyncio
    async def test_get_current_context_raises_outside(self):
        """Test get_current_context raises outside context."""
        with pytest.raises(RuntimeError):
            get_current_context()

    @pytest.mark.asyncio
    async def test_get_current_context_or_none(self):
        """Test get_current_context_or_none returns None outside."""
        assert get_current_context_or_none() is None

        async with request_context(req_id="test"):
            ctx = get_current_context_or_none()
            assert ctx is not None

    @pytest.mark.asyncio
    async def test_get_elapsed_time(self):
        """Test get_elapsed_time returns elapsed time."""
        async with request_context():
            await asyncio.sleep(0.01)
            elapsed = get_elapsed_time()
            assert elapsed > 0

    @pytest.mark.asyncio
    async def test_get_elapsed_time_outside_context(self):
        """Test get_elapsed_time returns 0 outside context."""
        assert get_elapsed_time() == 0.0
